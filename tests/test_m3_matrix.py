"""Tests for M3: readers trained through an aggregation over prompt cells."""

from collections import Counter
import dataclasses
from pathlib import Path
from typing import Any
from typing import cast
import warnings
import weakref

import numpy as np
from numpy.typing import NDArray
import pytest
from sklearn.exceptions import ConvergenceWarning
import torch

from tomography.activations import Activations
from tomography.activations import Span
from tomography.activations import collect_activations
from tomography.cli import _parse_override
from tomography.cli import load_experiment
from tomography.datasets.records import Example
from tomography.datasets.stimulus import StimulusSet
from tomography.experiments import m2_readers
from tomography.experiments import m3_matrix
from tomography.experiments import p2_positions
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.m2_readers import read_positions
from tomography.experiments.m3_matrix import CELL_SETS
from tomography.experiments.m3_matrix import SEGMENTS
from tomography.experiments.m3_matrix import SET_SEGMENTS
from tomography.experiments.m3_matrix import CellSet
from tomography.experiments.m3_matrix import prompt_cells
from tomography.experiments.m3_matrix import prompt_max_set
from tomography.experiments.m3_matrix import select_cells
from tomography.experiments.m3_matrix import stimulus_token_set
from tomography.experiments.p2_positions import misleading_template
from tomography.experiments.p2_positions import with_decoys
from tomography.experiments.p3_size import spearman
from tomography.experiments.sweep import chat_format_for
from tomography.experiments.sweep import shuffle_within_groups
from tomography.experiments.variants import READING_TASKS
from tomography.experiments.variants import task_set
from tomography.experiments.variants import template_variants
from tomography.model import BosMode
from tomography.model import LoadedModel
from tomography.reading.metrics import auroc
from tomography.reading.metrics import group_accuracy
from tomography.reading.probes import AttentionReader
from tomography.reading.probes import CellMaxReader
from tomography.reading.probes import fit_attention
from tomography.reading.probes import fit_cell_max
from tomography.reading.stats import delong_test
from tomography.reading.stats import holm_correction
from tomography.reading.stats import paired_auroc_diff

DEFAULT_HASH = "fdb1d8d7ccf3"
M2_DEFAULT_HASH = "3c82fe936345"
LAYERS = (1, 2, 3)
PAPER_DECOY_LINE = p2_positions.DECOY_LINE
# The paper's decoy line is too long for the 128 positions of the tiny models
# once it sits inside a paper template.
SHORT_DECOY_LINE = "Hint: {noun} is {{question}}.\n"
PREFIX, DECOY, STIMULUS, FINAL = (
    SEGMENTS.index(s) for s in ("prefix", "decoy", "stimulus", "final")
)


def test_segments_and_cell_sets_are_named_in_prompt_order() -> None:
    assert SEGMENTS == ("prefix", "decoy", "stimulus", "final")
    assert CELL_SETS == ("prompt", "stimulus", "final", "upper")
    assert SET_SEGMENTS == {
        "prompt": SEGMENTS,
        "stimulus": ("stimulus",),
        "final": ("final",),
        "upper": SEGMENTS,
    }


@pytest.fixture(autouse=True)
def _short_decoy_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(p2_positions, "DECOY_LINE", SHORT_DECOY_LINE)


def _examples(task: str) -> list[Example]:
    """Three pairs of stimuli of unequal length."""
    if task == "sentiment":
        words = {1: ("good", "fine", "great"), 0: ("bad", "poor", "dull")}
        return [
            Example(
                f"sst2-val-{label}-{i}",
                f"a {words[label][i]} one{' more' * i}",
                label,
                "val",
            )
            for label in (0, 1)
            for i in range(3)
        ]
    words = {1: ("sun", "gift", "joy"), 0: ("rain", "loss", "pain")}
    return [
        Example(
            f"ethics-test-{i}:{suffix}",
            f"a {words[label][i]} day{' more' * i}",
            label,
            "test",
            {"pair": f"ethics-test-{i}"},
        )
        for i in range(3)
        for suffix, label in (("a", 1), ("b", 0))
    ]


def _stimuli(
    loaded: LoadedModel, task_name: str, *, decoy: bool
) -> StimulusSet:
    """The report items of a task, on the paper or the misleading template."""
    task = READING_TASKS[task_name]
    variant = template_variants(task, chat_format_for(loaded), kinds=("paper",))
    paper = variant[0][0]
    rows = _examples(task_name)
    if decoy:
        paper = dataclasses.replace(
            paper,
            name="misleading",
            template=misleading_template(task.template, task.noun),
        )
        rows = with_decoys(rows, "against", seed=0)
    return task_set(task, paper, rows, n_pairs=3, seed=0)


def _cells(
    loaded: LoadedModel,
    stimuli: StimulusSet,
    *,
    decoy: bool,
    layers: tuple[int, ...] = LAYERS,
) -> tuple[Activations, NDArray[np.int8]]:
    return prompt_cells(
        loaded, stimuli, layers, bos=None, batch_size=4, decoy=decoy
    )


@pytest.mark.parametrize("bos", ["prepend", "none"])
def test_prompt_cells_read_every_token_but_bos(
    tiny_arch: LoadedModel, bos: BosMode
) -> None:
    stimuli = _stimuli(tiny_arch, "sentiment", decoy=True)
    encoded = tiny_arch.encode(stimuli.texts, bos=bos)
    assert len(set(encoded.lengths)) > 1

    cells, segments = prompt_cells(
        tiny_arch, stimuli, LAYERS, bos=bos, batch_size=4, decoy=True
    )

    first = int(encoded.has_bos)
    lengths = np.array(encoded.lengths)
    assert cells.position == "response"
    assert cells.layers == LAYERS
    assert cells.n_stimuli == len(stimuli.texts)
    np.testing.assert_array_equal(
        np.bincount(cells.row_stimulus), lengths - first
    )
    np.testing.assert_array_equal(
        cells.row_token,
        np.concatenate([np.arange(first, n) for n in lengths]),
    )
    np.testing.assert_array_equal(
        cells.row_token_id,
        np.concatenate(
            [
                encoded.input_ids[i, first:n].numpy()
                for i, n in enumerate(lengths)
            ]
        ),
    )
    assert cells.data.shape == (
        len(LAYERS),
        int(lengths.sum()) - first * lengths.size,
        tiny_arch.d_model,
    )
    assert segments.dtype == np.int8
    assert segments.shape == cells.row_stimulus.shape


def _assert_same_rows(got: Activations, expected: Activations) -> None:
    assert got.layers == expected.layers
    assert got.n_stimuli == expected.n_stimuli
    for field in ("row_stimulus", "row_rank", "row_token", "row_token_id"):
        np.testing.assert_array_equal(
            getattr(got, field), getattr(expected, field)
        )
    np.testing.assert_array_equal(got.data, expected.data)


def _span(stimuli: StimulusSet, index: int) -> Span:
    span = stimuli.response_spans[index]
    assert span is not None
    return span


def _end_inside_a_token(stimuli: StimulusSet) -> StimulusSet:
    """Ends each stimulus span one character early, inside its last token.

    That token then overlaps both the stimulus and the final line.
    """
    spans = [_span(stimuli, i) for i in range(len(stimuli.texts))]
    return dataclasses.replace(
        stimuli, response_spans=tuple((a, b - 1) for a, b in spans)
    )


@pytest.mark.parametrize("task", ["sentiment", "utility"])
@pytest.mark.parametrize("decoy", [False, True])
@pytest.mark.parametrize("shared_token", [False, True])
def test_stimulus_cells_are_the_response_rows(
    tiny_arch: LoadedModel, task: str, decoy: bool, shared_token: bool
) -> None:
    stimuli = _stimuli(tiny_arch, task, decoy=decoy)
    if shared_token:
        stimuli = _end_inside_a_token(stimuli)
    cells, segments = _cells(tiny_arch, stimuli, decoy=decoy)

    got = select_cells(cells, segments, "stimulus", LAYERS)

    expected = collect_activations(
        tiny_arch,
        stimuli.texts,
        layers=LAYERS,
        position="response",
        response_spans=stimuli.response_spans,
        batch_size=4,
    )
    _assert_same_rows(got, expected)


@pytest.mark.parametrize("task", ["sentiment", "utility"])
def test_prefix_cells_are_equal_across_texts_of_one_template(
    tiny_arch: LoadedModel, task: str
) -> None:
    stimuli = _stimuli(tiny_arch, task, decoy=False)
    cells, segments = _cells(tiny_arch, stimuli, decoy=False)

    prefix = segments == PREFIX
    rows = [
        prefix & (cells.row_stimulus == i) for i in range(len(stimuli.texts))
    ]
    assert rows[0].sum() > 0
    for row in rows[1:]:
        np.testing.assert_array_equal(
            cells.row_token_id[row], cells.row_token_id[rows[0]]
        )
        np.testing.assert_array_equal(
            cells.data[:, row], cells.data[:, rows[0]]
        )


@pytest.mark.parametrize("task", ["sentiment", "utility"])
@pytest.mark.parametrize("decoy", [False, True])
def test_segments_follow_prompt_order(
    tiny_arch: LoadedModel, task: str, decoy: bool
) -> None:
    stimuli = _stimuli(tiny_arch, task, decoy=decoy)
    cells, segments = _cells(tiny_arch, stimuli, decoy=decoy)

    noun = READING_TASKS[task].noun
    for i, label in enumerate(stimuli.labels):
        codes = segments[cells.row_stimulus == i]
        ids = cells.row_token_id[cells.row_stimulus == i]
        start = int(np.argmax(codes == STIMULUS))
        assert codes[0] == PREFIX
        assert set(codes[:start].tolist()) <= {PREFIX, DECOY}
        assert set(codes[start:].tolist()) == {STIMULUS, FINAL}
        assert np.all(np.diff(codes[start:]) >= 0)
        line = np.flatnonzero(codes == DECOY)
        if not decoy:
            assert line.size == 0
            continue
        np.testing.assert_array_equal(np.diff(line), 1)
        level = "low" if label == 1 else "high"
        assert tiny_arch.tokenizer.decode(ids[line].tolist()) == (
            f"Hint: {noun} is {level}.\n"
        )


@pytest.mark.parametrize(
    "head",
    ["USER: ", "<|im_start|>user\nRate it.\n"],
    ids=["after_a_tag_on_its_line", "on_its_own_line"],
)
def test_the_paper_decoy_line_is_the_decoy_segment(
    tiny_arch: LoadedModel, monkeypatch: pytest.MonkeyPatch, head: str
) -> None:
    monkeypatch.setattr(p2_positions, "DECOY_LINE", PAPER_DECOY_LINE)
    line = PAPER_DECOY_LINE.format(noun="sentiment").replace(
        "{question}", "high"
    )
    words = ("a bad one", "a good one more")
    start = len(head + line)
    stimuli = StimulusSet(
        texts=tuple(f"{head}{line}{w}\nASSISTANT:" for w in words),
        response_spans=tuple((start, start + len(w)) for w in words),
        concept_spans=(None, None),
        labels=np.array([0, 1]),
        pairs=np.array([[0, 1]]),
        groups=np.array([[0, 1]]),
        uids=("a", "b"),
    )

    cells, segments = _cells(tiny_arch, stimuli, decoy=True)

    for i in range(len(words)):
        ids = cells.row_token_id[
            (cells.row_stimulus == i) & (segments == DECOY)
        ]
        # A tokenizer whose tokens keep their leading space puts the space
        # after the tag in the first decoy token.
        assert tiny_arch.tokenizer.decode(ids.tolist()) in (line, " " + line)


@pytest.mark.parametrize("decoy", [False, True])
def test_the_last_final_cell_is_the_last_row(
    tiny_arch: LoadedModel, decoy: bool
) -> None:
    stimuli = _stimuli(tiny_arch, "utility", decoy=decoy)
    cells, segments = _cells(tiny_arch, stimuli, decoy=decoy)

    ends = np.cumsum(np.bincount(cells.row_stimulus)) - 1
    np.testing.assert_array_equal(segments[ends], FINAL)
    last = collect_activations(
        tiny_arch, stimuli.texts, layers=LAYERS, position="last", batch_size=4
    )
    np.testing.assert_array_equal(cells.row_token[ends], last.row_token)
    np.testing.assert_array_equal(cells.data[:, ends], last.data)


def _m2_config() -> m2_readers.Config:
    return m2_readers.Config(model="tiny", layers=LAYERS, batch_size=4)


@pytest.mark.parametrize("task", ["sentiment", "utility"])
def test_stimulus_token_reads_the_last_stimulus_cell(
    tiny_arch: LoadedModel, task: str
) -> None:
    stimuli = _stimuli(tiny_arch, task, decoy=True)
    cells, segments = _cells(tiny_arch, stimuli, decoy=True)
    stimulus = select_cells(cells, segments, "stimulus", LAYERS)

    read = read_positions(
        tiny_arch, stimulus_token_set(stimuli), ("concept",), _m2_config()
    )["concept"]

    ends = np.cumsum(np.bincount(stimulus.row_stimulus)) - 1
    np.testing.assert_array_equal(read.row_token, stimulus.row_token[ends])
    np.testing.assert_array_equal(read.data, stimulus.data[:, ends])


def test_prompt_max_pools_every_cell_of_a_text(tiny_arch: LoadedModel) -> None:
    stimuli = _stimuli(tiny_arch, "sentiment", decoy=True)
    cells, _ = _cells(tiny_arch, stimuli, decoy=True)

    read = read_positions(
        tiny_arch, prompt_max_set(stimuli), ("max",), _m2_config()
    )["max"]

    counts = np.bincount(cells.row_stimulus)
    np.testing.assert_array_equal(
        read.data,
        np.maximum.reduceat(cells.data, np.cumsum(counts) - counts, axis=1),
    )


@pytest.mark.parametrize("cell_set", CELL_SETS)
def test_select_cells_keeps_the_rows_and_layers_of_a_set(
    tiny_gpt2: LoadedModel, cell_set: CellSet
) -> None:
    stimuli = _stimuli(tiny_gpt2, "utility", decoy=True)
    cells, segments = _cells(tiny_gpt2, stimuli, decoy=True)

    got = select_cells(cells, segments, cell_set, (3, 1))

    codes = [SEGMENTS.index(s) for s in SET_SEGMENTS[cell_set]]
    keep = np.isin(segments, codes)
    assert got.layers == (1, 3)
    assert got.position == "response"
    assert got.n_stimuli == cells.n_stimuli
    np.testing.assert_array_equal(got.data, cells.data[[0, 2]][:, keep])
    for field in ("row_stimulus", "row_token", "row_token_id"):
        np.testing.assert_array_equal(
            getattr(got, field), getattr(cells, field)[keep]
        )
    counts = np.bincount(got.row_stimulus)
    np.testing.assert_array_equal(
        got.row_rank, np.concatenate([np.arange(n) for n in counts])
    )


def test_select_cells_refuses_an_unknown_set_and_layers_not_read(
    tiny_gpt2: LoadedModel,
) -> None:
    stimuli = _stimuli(tiny_gpt2, "sentiment", decoy=False)
    cells, segments = _cells(tiny_gpt2, stimuli, decoy=False, layers=(1, 3))

    with pytest.raises(ValueError, match=r"^unknown cell set 'nope'"):
        select_cells(cells, segments, cast(CellSet, "nope"), (1, 3))
    with pytest.raises(
        ValueError, match=r"^layers \[2\] were not read, only \(1, 3\)$"
    ):
        select_cells(cells, segments, "prompt", (1, 2))


def _replace(
    stimuli: StimulusSet, index: int, text: str, span: Span | None
) -> StimulusSet:
    texts, spans = list(stimuli.texts), list(stimuli.response_spans)
    texts[index], spans[index] = text, span
    return dataclasses.replace(
        stimuli, texts=tuple(texts), response_spans=tuple(spans)
    )


def test_select_cells_names_a_text_left_without_rows(
    tiny_gpt2: LoadedModel,
) -> None:
    stimuli = _stimuli(tiny_gpt2, "utility", decoy=False)
    text, (start, _) = stimuli.texts[1], _span(stimuli, 1)
    to_the_end = _replace(stimuli, 1, text, (start, len(text)))
    cells, segments = _cells(tiny_gpt2, to_the_end, decoy=False)

    with pytest.raises(
        ValueError, match=r"^stimulus 1 has no cells in the set 'final'$"
    ):
        select_cells(cells, segments, "final", LAYERS)


def _no_model_call(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the model was called")

    monkeypatch.setattr(m3_matrix, "collect_activations", fail)


def test_prompt_cells_name_a_text_without_a_stimulus_span(
    tiny_gpt2: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    stimuli = _stimuli(tiny_gpt2, "sentiment", decoy=False)
    broken = _replace(stimuli, 2, stimuli.texts[2], None)
    _no_model_call(monkeypatch)

    with pytest.raises(ValueError, match=r"^stimulus 2 has no stimulus span$"):
        _cells(tiny_gpt2, broken, decoy=False)


def _replace_decoy_line(
    stimuli: StimulusSet, index: int, replacement: str
) -> StimulusSet:
    """Replaces the line before a stimulus line; `{line}` keeps the old one."""
    text, (start, end) = stimuli.texts[index], _span(stimuli, index)
    line_end = text.rfind("\n", 0, start) + 1
    line_start = text.rfind("\n", 0, line_end - 1) + 1
    new = replacement.format(line=text[line_start:line_end])
    shift = len(new) - (line_end - line_start)
    return _replace(
        stimuli,
        index,
        text[:line_start] + new + text[line_end:],
        (start + shift, end + shift),
    )


@pytest.mark.parametrize(
    ("index", "replacement"),
    [(1, ""), (3, "{line}it is so.\n"), (4, "Hint: none\n")],
    ids=["missing", "not_before_the_stimulus", "altered"],
)
def test_prompt_cells_name_a_misleading_text_without_its_decoy_line(
    tiny_gpt2: LoadedModel,
    monkeypatch: pytest.MonkeyPatch,
    index: int,
    replacement: str,
) -> None:
    misleading = _stimuli(tiny_gpt2, "utility", decoy=True)
    broken = _replace_decoy_line(misleading, index, replacement)
    _no_model_call(monkeypatch)

    with pytest.raises(
        ValueError,
        match=rf"^stimulus {index} has no decoy line right before the line "
        r"of its stimulus$",
    ):
        _cells(tiny_gpt2, broken, decoy=True)


ONE_ROW_READOUTS = ("last", "concept", "mean", "max")
RELABELLED_READOUTS = ("stimulus_token", "prompt_max")
SENTIMENT_CONDITIONS = [
    "original",
    "random",
    "against",
    "paraphrase_0",
    "paraphrase_1",
    "paraphrase_2",
]
UTILITY_CONDITIONS = ["original", "random", "against"]


def test_config_defaults_and_round_trip() -> None:
    config = m3_matrix.Config()
    m2 = m2_readers.Config()

    assert m3_matrix.POOLINGS == ("cellattn", "cellmax")
    assert config.experiment == "m3"
    assert config.scorers == ("logistic", "mean_diff")
    assert config.readouts == ONE_ROW_READOUTS
    assert config.cell_layers == (4, 8, 12, 16, 20, 24, 28)
    assert config.upper_layers == (16, 20, 24, 28)
    assert config.cell_sets == ("prompt",)
    assert config.poolings == ("cellattn", "cellmax")
    assert config.baselines == (
        "logistic/last",
        "mean_diff/last",
        "logistic/concept",
        "logistic/stimulus_token",
    )
    changed = {"experiment", "scorers", "readouts", "baselines"}
    for field in dataclasses.fields(m2_readers.Config):
        if field.name not in changed:
            assert getattr(config, field.name) == getattr(m2, field.name)
    assert m3_matrix.Config.from_dict(config.to_dict()) == config


def test_the_default_config_hash_is_stable() -> None:
    assert m3_matrix.Config().config_hash() == DEFAULT_HASH
    assert m2_readers.Config().config_hash() == M2_DEFAULT_HASH


def test_m3_is_discovered_by_the_cli() -> None:
    assert load_experiment("m3").config_cls is m3_matrix.Config


def test_set_reads_the_cell_settings_as_tuples() -> None:
    config_cls = m3_matrix.Config
    data = dict(
        [
            _parse_override(config_cls, 'cell_sets=["stimulus", "upper"]'),
            _parse_override(config_cls, 'poolings=["cellmax"]'),
            _parse_override(config_cls, "cell_layers=[2, 4]"),
            _parse_override(config_cls, "upper_layers=[4]"),
        ]
    )
    config = config_cls.from_dict(data)
    assert config.cell_sets == ("stimulus", "upper")
    assert config.poolings == ("cellmax",)
    assert config.cell_layers == (2, 4)
    assert config.upper_layers == (4,)


def _sentiment_rows(per_class: int = 12) -> list[Example]:
    words = {1: ("good", "fine", "great"), 0: ("bad", "poor", "dull")}
    return [
        Example(
            f"sst2-{split}-{label}-{i}",
            f"a {words[label][i % 3]} one {i}",
            label,
            split,
        )
        for split in ("train", "val")
        for label in (0, 1)
        for i in range(per_class)
    ]


def _utility_rows(pairs: int = 8) -> list[Example]:
    words = {
        1: ("sun", "gift", "joy", "fun"),
        0: ("rain", "loss", "pain", "fear"),
    }
    return [
        Example(
            f"ethics-{split}-{i}:{suffix}",
            f"a {words[label][i % 4]} day {i}",
            label,
            split,
            {"pair": f"ethics-{split}-{i}"},
        )
        for split in ("train", "val", "test")
        for i in range(pairs)
        for suffix, label in (("a", 1), ("b", 0))
    ]


def _sources() -> dict[str, list[Example]]:
    return {"sst2": _sentiment_rows(), "ethics_utility": _utility_rows()}


def _config(**overrides: Any) -> m3_matrix.Config:
    base = m3_matrix.Config(
        model="tiny",
        tasks=("sentiment", "utility"),
        n_seeds=2,
        n_pairs=4,
        n_eval_pairs=3,
        top_k=2,
        layers=LAYERS,
        cell_layers=(1, 3),
        upper_layers=(3,),
        cell_sets=CELL_SETS,
        logistic_c=(0.1, 10.0),
        n_boot=50,
    )
    return dataclasses.replace(base, **overrides)


def _refused(config: m3_matrix.Config, tmp_path: Path) -> None:
    m3_matrix.execute(
        config,
        RunDir.create(config, out_root=tmp_path),
        cast(Any, None),
        _sources(),
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"poolings": ("cellsum",)},
            r"^unknown pooling 'cellsum', expected one of "
            r"\('cellattn', 'cellmax'\)$",
        ),
        ({"poolings": ()}, r"^m3 needs at least one entry in poolings$"),
        (
            {"cell_sets": ("prompt", "nope")},
            r"^unknown cell set 'nope', expected one of "
            r"\('prompt', 'stimulus', 'final', 'upper'\)$",
        ),
        ({"cell_sets": ()}, r"^m3 needs at least one entry in cell_sets$"),
        ({"cell_layers": ()}, r"^m3 needs at least one entry in cell_layers$"),
        (
            {"cell_layers": (1, 4), "upper_layers": (4,)},
            r"^cell_layers \(1, 4\) must be among the layers read "
            r"\(1, 2, 3\)$",
        ),
        (
            {"upper_layers": (2,)},
            r"^upper_layers must name layers of cell_layers \(1, 3\), "
            r"got \(2,\)$",
        ),
        (
            {"upper_layers": ()},
            r"^upper_layers must name layers of cell_layers \(1, 3\), "
            r"got \(\)$",
        ),
        (
            {"baselines": ("lda/last",)},
            r"^baseline 'lda/last' is not a cell of the run",
        ),
        (
            {"baselines": ("logistic/cellmax_prompt",)},
            r"^baseline 'logistic/cellmax_prompt' is not a cell of the run",
        ),
        (
            {"readouts": ("last", "token_score_max"), "baselines": ()},
            r"^m3 reads no token readouts, the cell readers replace them: "
            r"got \('token_score_max',\)$",
        ),
        (
            {"readouts": ("attention", "last"), "baselines": ()},
            r"^m3 reads no token readouts, the cell readers replace them: "
            r"got \('attention',\)$",
        ),
        (
            {"reader_series": False},
            r"^m3 fits the reader series, so reader_series must be True$",
        ),
        ({"top_k": 0}, r"^top_k must be at least 1"),
    ],
)
def test_a_bad_config_is_refused_before_any_model_call(
    overrides: dict[str, Any], message: str, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match=message):
        _refused(_config(**overrides), tmp_path)


def test_the_relabelled_readouts_can_be_baselines(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class LoadedError(Exception):
        pass

    def loaded(*args: Any, **kwargs: Any) -> None:
        raise LoadedError

    monkeypatch.setattr(m3_matrix, "load_sources", lambda sources: {})
    monkeypatch.setattr(m3_matrix, "load_model", loaded)
    config = _config(
        baselines=("logistic/stimulus_token", "mean_diff/prompt_max")
    )
    with pytest.raises(LoadedError):
        m3_matrix.run(config, RunDir.create(config, out_root=tmp_path))


def test_run_refuses_a_bad_config_before_loading_the_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def no_load(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the model was loaded")

    monkeypatch.setattr(m3_matrix, "load_model", no_load)
    config = _config(poolings=("cellsum",))
    with pytest.raises(ValueError, match="unknown pooling"):
        m3_matrix.run(config, RunDir.create(config, out_root=tmp_path))


def test_run_loads_the_model_in_float32(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[Any, ...]] = []

    class LoadedError(Exception):
        pass

    def record(key: str, **kwargs: Any) -> None:
        calls.append((key, kwargs))
        raise LoadedError

    monkeypatch.setattr(m3_matrix, "load_sources", lambda sources: {})
    monkeypatch.setattr(m3_matrix, "load_model", record)
    config = _config()
    with pytest.raises(LoadedError):
        m3_matrix.run(config, RunDir.create(config, out_root=tmp_path))
    assert calls == [("tiny", {"dtype": torch.float32})]


def test_run_loads_the_sources_of_every_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    class LoadedError(Exception):
        pass

    def record(sources: list[str]) -> None:
        calls.append(list(sources))
        raise LoadedError

    monkeypatch.setattr(m3_matrix, "load_model", lambda *a, **k: None)
    monkeypatch.setattr(m3_matrix, "load_sources", record)
    config = _config()
    with pytest.raises(LoadedError):
        m3_matrix.run(config, RunDir.create(config, out_root=tmp_path))
    assert calls == [["sst2", "ethics_utility"]]


def _quiet_execute(
    config: m3_matrix.Config, run_dir: RunDir, loaded: LoadedModel
) -> None:
    # Fits on the tiny model's handful of rows can stop on an L-BFGS line
    # search; the warning is expected there and not under test.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m3_matrix.execute(config, run_dir, loaded, _sources())


@pytest.fixture(scope="module")
def runs() -> dict[str, RunDir]:
    return {}


@pytest.fixture
def finished(
    runs: dict[str, RunDir],
    tiny_gpt2: LoadedModel,
    tmp_path_factory: pytest.TempPathFactory,
) -> RunDir:
    if "run" not in runs:
        config = _config()
        run_dir = RunDir.create(config, out_root=tmp_path_factory.mktemp("m3"))
        _quiet_execute(config, run_dir, tiny_gpt2)
        runs["run"] = run_dir
    return runs["run"]


def _table(run_dir: RunDir, name: str = "table") -> list[dict[str, Any]]:
    payload = cast(dict[str, Any], run_dir.read_json(name))
    return [
        dict(zip(payload["columns"], row, strict=True))
        for row in payload["rows"]
    ]


def _entries(run_dir: RunDir, task: str) -> list[str]:
    sidecar = cast(dict[str, Any], run_dir.read_json(f"scores_{task}"))
    return cast(list[str], sidecar["entries"])


def test_a_run_writes_the_table_the_comparisons_and_the_scores(
    finished: RunDir,
) -> None:
    names = {p.name for p in finished.path.iterdir()}
    for stem in ("table", "comparisons", "summary"):
        assert f"{stem}.json" in names
    for task in ("sentiment", "utility"):
        assert {f"scores_{task}.npz", f"auroc_{task}.npz"} <= names
    table = cast(dict[str, Any], finished.read_json("table"))
    assert table["columns"] == list(m2_readers.COLUMNS)
    comparisons = cast(dict[str, Any], finished.read_json("comparisons"))
    assert comparisons["columns"] == list(m2_readers.COMPARISON_COLUMNS)
    summary = cast(dict[str, Any], finished.read_json("summary"))
    assert summary["tasks"] == ["sentiment", "utility"]
    assert summary["conditions"] == {
        "sentiment": SENTIMENT_CONDITIONS,
        "utility": UTILITY_CONDITIONS,
    }
    assert summary["n_eval_pairs"]["utility"] == {"select": 3, "report": 3}
    assert "best rule" in summary["families"]


def test_every_cell_has_a_row_per_condition_and_series(
    finished: RunDir,
) -> None:
    readouts = (*ONE_ROW_READOUTS, *RELABELLED_READOUTS)
    fitted = [
        (scorer, readout, rule)
        for scorer in ("logistic", "mean_diff")
        for readout in readouts
        for rule in ("best", "top_k")
    ]
    fitted += [
        ("logistic", f"{pooling}_{cell_set}", "joint")
        for cell_set in CELL_SETS
        for pooling in ("cellattn", "cellmax")
    ]
    controls = [
        ("random", readout, rule)
        for readout in readouts
        for rule in ("best", "top_k")
    ]
    controls += [("random", f"cellmax_{s}", "joint") for s in CELL_SETS]
    controls.append(("text", "text", "none"))
    for task, conditions in (
        ("sentiment", SENTIMENT_CONDITIONS),
        ("utility", UTILITY_CONDITIONS),
    ):
        expected = {
            (*cell, series, condition)
            for cell in fitted
            for series in ("reader", "shuffled")
            for condition in conditions
        } | {
            (*cell, "reader", condition)
            for cell in controls
            for condition in conditions
        }
        got = [
            (
                r["scorer"],
                r["readout"],
                r["layer_rule"],
                r["series"],
                r["condition"],
            )
            for r in _table(finished)
            if r["task"] == task
        ]
        assert len(got) == len(set(got))
        assert set(got) == expected


def test_every_cell_scores_the_same_report_items(finished: RunDir) -> None:
    arrays = finished.read_arrays("scores_sentiment")
    sidecar = cast(dict[str, Any], finished.read_json("scores_sentiment"))
    n_entries = len(sidecar["entries"])

    assert "logistic/cellmax_upper/joint/shuffled" in sidecar["entries"]
    assert arrays["report"].shape == (n_entries, 2, 6, 6)
    assert arrays["select"].shape == (n_entries, 2, 6)
    assert np.isfinite(arrays["report"]).all()
    layers = sidecar["layers"]
    assert layers["logistic/cellattn_prompt/joint/reader"] == [[1, 3]] * 2
    assert layers["logistic/cellmax_upper/joint/shuffled"] == [[3]] * 2
    assert layers["random/cellmax_final/joint/reader"] == [[1, 3]] * 2


def _capture_items(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Records the items `_prepare_task` renders for each task."""
    captured: list[Any] = []
    prepare = m2_readers._prepare_task

    def capturing(*args: Any, **kwargs: Any) -> Any:
        splits, skipped = prepare(*args, **kwargs)
        captured.append(splits)
        return splits, skipped

    monkeypatch.setattr(m2_readers, "_prepare_task", capturing)
    return captured


def _recording(
    monkeypatch: pytest.MonkeyPatch, name: str, calls: list[dict[str, Any]]
) -> None:
    """Records the keyword arguments of every call of an m3 fit."""
    fit = getattr(m3_matrix, name)

    def recording(acts: Activations, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return fit(acts, **kwargs)

    monkeypatch.setattr(m3_matrix, name, recording)


def test_shuffled_cell_readers_tune_and_threshold_on_shuffled_labels(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tuned: list[NDArray[np.int64]] = []
    fits: list[dict[str, Any]] = []
    tuning = group_accuracy

    def recording(scores: Any, groups: Any, labels: Any) -> Any:
        tuned.append(np.array(labels))
        return tuning(scores, groups, labels)

    monkeypatch.setattr(m3_matrix, "group_accuracy", recording)
    _recording(monkeypatch, "fit_attention", fits)
    captured = _capture_items(monkeypatch)
    config = _config(
        tasks=("sentiment",),
        scorers=("mean_diff",),
        readouts=("last",),
        cell_sets=("prompt",),
        poolings=("cellattn",),
        controls=("shuffled",),
        decoys=(),
        shifts=(),
        baselines=("mean_diff/last",),
        logistic_c=(1.0,),
    )
    run_dir = RunDir.create(config, out_root=tmp_path)
    _quiet_execute(config, run_dir, tiny_gpt2)

    train, select = captured[0].train_items, captured[0].select_items
    shuffled = [
        (
            shuffle_within_groups(
                train[i].labels,
                train[i].groups,
                seed=derive_seed(config.seed, "shuffled", "sentiment", i),
            ),
            shuffle_within_groups(
                select.labels,
                select.groups,
                seed=derive_seed(
                    config.seed, "shuffled-select", "sentiment", i
                ),
            ),
        )
        for i in range(config.n_seeds)
    ]
    # Each draw fits the reader series, then the shuffled one.
    expected = [
        pair
        for i, (labels, select_labels) in enumerate(shuffled)
        for pair in (
            (train[i].labels, select.labels),
            (labels, select_labels),
        )
    ]
    assert len(fits) == len(tuned) == len(expected)
    for fit, tune, (labels, select_labels) in zip(
        fits, tuned, expected, strict=True
    ):
        assert fit["joint"] is True
        assert np.array_equal(fit["labels"], labels)
        assert np.array_equal(tune, select_labels)
    for _, select_labels in shuffled:
        assert (select_labels[select.groups].sum(axis=1) == 1).all()
    arrays = run_dir.read_arrays("scores_sentiment")
    e = _entries(run_dir, "sentiment").index(
        "logistic/cellattn_prompt/joint/shuffled"
    )
    for i, (_, select_labels) in enumerate(shuffled):
        assert np.array_equal(
            arrays["entry_select_labels"][e, i], select_labels
        )


def test_the_replica_rows_equal_the_rows_of_m2(
    finished: RunDir, tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    config = _config()
    shared = {
        field.name: getattr(config, field.name)
        for field in dataclasses.fields(m2_readers.Config)
    }
    m2_config = m2_readers.Config(
        **{
            **shared,
            "experiment": "m2",
            "baselines": ("logistic/last", "mean_diff/last"),
        }
    )
    run_dir = RunDir.create(m2_config, out_root=tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m2_readers.execute(m2_config, run_dir, tiny_gpt2, _sources())

    replicas = [
        r
        for r in _table(finished)
        if r["readout"] in (*ONE_ROW_READOUTS, "text")
    ]
    assert replicas == _table(run_dir)


def _set_layers(config: m3_matrix.Config, cell_set: CellSet) -> tuple[int, ...]:
    return config.upper_layers if cell_set == "upper" else config.cell_layers


def _fitted_readers(
    loaded: LoadedModel, config: m3_matrix.Config
) -> list[tuple[CellSet, Any]]:
    train = _stimuli(loaded, "utility", decoy=False)
    cells, segments = prompt_cells(
        loaded, train, config.cell_layers, bos=None, batch_size=4, decoy=False
    )
    readers: list[tuple[CellSet, Any]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for cell_set in ("prompt", "upper", "final"):
            acts = select_cells(
                cells, segments, cell_set, _set_layers(config, cell_set)
            )
            readers += [
                (
                    cell_set,
                    fit_attention(acts, labels=train.labels, joint=True),
                ),
                (cell_set, fit_cell_max(acts, labels=train.labels)),
            ]
    return readers


@pytest.mark.parametrize("batch_size", [1, 3])
def test_streamed_scores_equal_whole_set_scores(
    tiny_arch: LoadedModel, batch_size: int
) -> None:
    config = _config(batch_size=batch_size)
    readers = _fitted_readers(tiny_arch, config)
    report = _stimuli(tiny_arch, "utility", decoy=True)

    streamed = m3_matrix.stream_scores(
        tiny_arch, report, readers, config, decoy=True
    )

    cells, segments = prompt_cells(
        tiny_arch,
        report,
        config.cell_layers,
        bos=None,
        batch_size=batch_size,
        decoy=True,
    )
    for k, (cell_set, reader) in enumerate(readers):
        acts = select_cells(
            cells, segments, cell_set, _set_layers(config, cell_set)
        )
        np.testing.assert_array_equal(streamed.scores[k], reader.score(acts)[0])
        np.testing.assert_allclose(
            streamed.shares[k], _shares(reader, acts, segments, cell_set)
        )
        np.testing.assert_array_equal(
            streamed.cells[k],
            np.bincount(acts.row_stimulus) * len(acts.layers),
        )
    assert streamed.scores.dtype == np.float64
    assert streamed.shares.shape == (len(readers), len(report.texts), 4)


def test_stream_scores_name_the_batch_of_a_text_without_its_decoy_line(
    tiny_gpt2: LoadedModel,
) -> None:
    config = _config(batch_size=3)
    readers = _fitted_readers(tiny_gpt2, config)[:1]
    report = _stimuli(tiny_gpt2, "utility", decoy=True)
    broken = _replace_decoy_line(report, 4, "")

    with pytest.raises(
        ValueError,
        match=r"^in the batch that starts at stimulus 3: stimulus 1 has no "
        r"decoy line right before the line of its stimulus$",
    ):
        m3_matrix.stream_scores(tiny_gpt2, broken, readers, config, decoy=True)


@dataclasses.dataclass(frozen=True, eq=False)
class _Recorded(CellMaxReader):
    """A cell max reader that records the scores it gives."""

    outputs: list[NDArray[np.float32]] = dataclasses.field(default_factory=list)

    def score(self, acts: Activations) -> NDArray[np.float32]:
        scores = super().score(acts)
        self.outputs.append(scores)
        return scores


def test_the_grid_value_is_the_first_best_on_select(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fitted: list[tuple[float, _Recorded]] = []
    tuned: list[tuple[int, ...]] = []
    fit = fit_cell_max

    def recording(acts: Activations, **kwargs: Any) -> _Recorded:
        reader = fit(acts, **kwargs)
        fields = {
            f.name: getattr(reader, f.name) for f in dataclasses.fields(reader)
        }
        fitted.append((kwargs["c"], _Recorded(**fields)))
        return fitted[-1][1]

    def tuning(scores: Any, groups: Any, labels: Any) -> NDArray[np.float64]:
        tuned.append(np.shape(scores))
        # The second and third grid values tie; the second is kept.
        return np.array([[0.4], [0.9], [0.9]])

    monkeypatch.setattr(m3_matrix, "fit_cell_max", recording)
    monkeypatch.setattr(m3_matrix, "group_accuracy", tuning)
    config = _config(
        tasks=("utility",),
        scorers=("logistic",),
        readouts=("last",),
        cell_sets=("prompt",),
        poolings=("cellmax",),
        controls=(),
        decoys=(),
        baselines=("logistic/last",),
        logistic_c=(0.1, 1.0, 10.0),
    )
    run_dir = RunDir.create(config, out_root=tmp_path)
    _quiet_execute(config, run_dir, tiny_gpt2)

    arrays = run_dir.read_arrays("scores_utility")
    e = _entries(run_dir, "utility").index(
        "logistic/cellmax_prompt/joint/reader"
    )
    assert [c for c, _ in fitted] == [0.1, 1.0, 10.0] * config.n_seeds
    assert tuned == [(3, 1, 6)] * config.n_seeds
    for i in range(config.n_seeds):
        readers = [reader for _, reader in fitted[3 * i : 3 * i + 3]]
        # Only the kept reader scores the report items.
        assert [len(r.outputs) for r in readers] == [1, 2, 1]
        kept = readers[1].outputs
        assert np.array_equal(arrays["select"][e, i], kept[0][0])
        assert np.array_equal(arrays["report"][e, i, 0], kept[1][0])
    heads = run_dir.read_arrays("cellheads_utility")
    assert heads["logistic_cellmax_prompt_reader_c"].tolist() == [1.0] * 2


def _ensemble(select: NDArray[Any], report: NDArray[Any]) -> NDArray[Any]:
    mean = select.mean(axis=-1, keepdims=True)
    std = select.std(axis=-1, keepdims=True)
    return ((report - mean[:, None]) / std[:, None]).mean(axis=0)


def test_joint_cells_are_compared_with_each_baseline_at_top_k_and_best(
    finished: RunDir,
) -> None:
    config = _config()
    rows = [
        r for r in _table(finished, "comparisons") if r["layer_rule"] == "joint"
    ]
    joint = [
        ("logistic", f"{pooling}_{cell_set}")
        for cell_set in CELL_SETS
        for pooling in m3_matrix.POOLINGS
    ]
    for task, conditions in (
        ("sentiment", SENTIMENT_CONDITIONS),
        ("utility", UTILITY_CONDITIONS),
    ):
        got = [
            (
                r["condition"],
                r["baseline"],
                r["baseline_rule"],
                r["scorer"],
                r["readout"],
            )
            for r in rows
            if r["task"] == task
        ]
        assert sorted(got) == sorted(
            (condition, baseline, rule, *cell)
            for condition in conditions
            for baseline in config.baselines
            for rule in ("top_k", "best")
            for cell in joint
        )

    arrays = finished.read_arrays("scores_sentiment")
    entries = _entries(finished, "sentiment")
    labels = arrays["report_labels"]
    clusters = np.empty(labels.shape[0], dtype=np.int64)
    for g, members in enumerate(arrays["report_groups"]):
        clusters[members] = g

    def ensemble(name: str) -> NDArray[Any]:
        e = entries.index(name)
        against = SENTIMENT_CONDITIONS.index("against")
        ensembled = _ensemble(arrays["select"][e], arrays["report"][e])
        return cast(NDArray[Any], ensembled[against])

    family = [
        r
        for r in rows
        if r["task"] == "sentiment"
        and r["condition"] == "against"
        and r["baseline"] == "logistic/stimulus_token"
        and r["baseline_rule"] == "best"
    ]
    base = ensemble("logistic/stimulus_token/best/reader")
    for row in family:
        cell = ensemble(f"logistic/{row['readout']}/joint/reader")
        diff = paired_auroc_diff(
            cell, base, labels, clusters, n_boot=50, seed=config.seed
        )
        assert row["auroc_diff"] == pytest.approx(diff.difference)
        assert row["auroc_lo"] == pytest.approx(diff.lower)
        assert row["delong_p"] == pytest.approx(
            delong_test(cell, base, labels).p_value
        )
    for column in ("delong", "mcnemar"):
        holm = holm_correction(np.array([r[f"{column}_p"] for r in family]))
        np.testing.assert_allclose(
            [r[f"{column}_holm"] for r in family], holm.adjusted
        )


def _counting(
    monkeypatch: pytest.MonkeyPatch, module: Any, read: Counter[str]
) -> None:
    """Counts every text a module sends through the model."""

    def counting(loaded: LoadedModel, texts: Any, **kw: Any) -> Activations:
        read.update(texts)
        return collect_activations(loaded, texts, **kw)

    monkeypatch.setattr(module, "collect_activations", counting)


def test_each_text_is_read_once_per_pass(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    one_row: Counter[str] = Counter()
    cells: Counter[str] = Counter()
    _counting(monkeypatch, m2_readers, one_row)
    _counting(monkeypatch, m3_matrix, cells)
    captured = _capture_items(monkeypatch)
    config = _config(tasks=("sentiment",), cell_sets=("prompt", "final"))
    _quiet_execute(config, RunDir.create(config, out_root=tmp_path), tiny_gpt2)

    splits = captured[0]
    texts = Counter(
        text
        for items in (
            *splits.train_items,
            splits.select_items,
            *splits.report_items.values(),
        )
        for text in items.texts
    )
    assert cells == texts
    # The last token, the stimulus tokens, the concept tokens, then the
    # stimulus token and the prompt maximum.
    assert one_row == Counter({text: 5 * n for text, n in texts.items()})


def test_cells_are_released_before_the_next_are_read(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    one_row: list[weakref.ref[Activations]] = []
    held: list[weakref.ref[Activations]] = []
    read = m2_readers.read_positions
    extract = m3_matrix.prompt_cells

    def reading(*args: Any, **kwargs: Any) -> Any:
        positions = read(*args, **kwargs)
        one_row.extend(weakref.ref(acts) for acts in positions.values())
        return positions

    def extracting(*args: Any, **kwargs: Any) -> Any:
        assert all(ref() is None for ref in one_row)
        assert all(ref() is None for ref in held)
        cells, segments = extract(*args, **kwargs)
        held.append(weakref.ref(cells))
        return cells, segments

    monkeypatch.setattr(m2_readers, "read_positions", reading)
    monkeypatch.setattr(m3_matrix, "prompt_cells", extracting)
    config = _config(tasks=("utility",), cell_sets=("prompt", "upper"))
    _quiet_execute(config, RunDir.create(config, out_root=tmp_path), tiny_gpt2)

    # Two draws, the select items in one batch, then one batch per condition.
    assert len(held) == 2 + 1 + 3


def test_a_rerun_writes_the_same_table(
    finished: RunDir, tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    config = _config(tasks=("utility",))
    run_dir = RunDir.create(config, out_root=tmp_path)
    _quiet_execute(config, run_dir, tiny_gpt2)

    first = [r for r in _table(finished) if r["task"] == "utility"]
    assert _table(run_dir) == first


def test_plot_draws_the_auroc_matrix_of_every_task(
    finished: RunDir, tmp_path: Path
) -> None:
    names = {p.name for p in m3_matrix.plot(finished, tmp_path)}
    assert {"auroc_sentiment.png", "auroc_utility.pdf"} <= names


def _random_cells(seed: int = 0) -> tuple[Activations, NDArray[np.int64]]:
    """Ragged cells of six stimuli at two layers, and their labels."""
    counts = np.array([2, 3, 1, 4, 2, 3])
    n_rows = int(counts.sum())
    data = np.random.default_rng(seed).normal(2.0, 3.0, (2, n_rows, 5))
    acts = Activations(
        data=data.astype(np.float32),
        layers=(2, 5),
        position="response",
        row_stimulus=np.repeat(np.arange(counts.size), counts),
        row_rank=np.concatenate([np.arange(n) for n in counts]),
        row_token=np.arange(n_rows),
        row_token_id=np.zeros(n_rows, dtype=np.int64),
        n_stimuli=counts.size,
    )
    return acts, np.array([0, 1, 0, 1, 1, 0])


def test_the_random_cell_reader_has_gaussian_heads_on_scaled_cells() -> None:
    acts, labels = _random_cells()

    reader = m3_matrix.random_cell_max(acts, labels, seed=7)

    rows = acts.data.astype(np.float64)
    centers = rows.mean(axis=1)
    centered = rows - centers[:, None]
    scales = np.sqrt(np.square(centered).sum(axis=(1, 2)) / rows.shape[1])
    heads = np.random.default_rng(7).standard_normal((2, 5))
    sign = np.sign(reader.values[0, 0] * heads[0, 0])
    assert reader.layers == (2, 5)
    np.testing.assert_allclose(reader.values * scales[:, None], sign * heads)
    np.testing.assert_allclose(
        reader.biases, -(reader.values * centers).sum(axis=1)
    )
    np.testing.assert_array_equal(reader.signs, [1.0, 1.0])
    again = m3_matrix.random_cell_max(acts, labels, seed=7)
    np.testing.assert_array_equal(again.values, reader.values)
    np.testing.assert_array_equal(again.biases, reader.biases)


def _flipped(reader: CellMaxReader) -> CellMaxReader:
    return dataclasses.replace(
        reader, values=-reader.values, biases=-reader.biases
    )


def test_the_random_cell_reader_takes_the_sign_that_ranks_labels_best() -> None:
    acts, labels = _random_cells()

    reader = m3_matrix.random_cell_max(acts, labels, seed=3)
    inverted = m3_matrix.random_cell_max(acts, 1 - labels, seed=3)

    area = float(auroc(reader.score(acts)[0], labels))
    assert area > float(auroc(_flipped(reader).score(acts)[0], labels))
    np.testing.assert_array_equal(inverted.values, -reader.values)


def _shares(
    reader: Any,
    acts: Activations,
    segments: NDArray[np.int8],
    cell_set: CellSet,
) -> NDArray[np.float64]:
    """Each stimulus's reader weight per segment, from `cell_weights`."""
    codes = [SEGMENTS.index(s) for s in SET_SEGMENTS[cell_set]]
    kept = segments[np.isin(segments, codes)]
    weights = m3_matrix.cell_weights(reader, acts).sum(axis=0)
    shares = np.zeros((acts.n_stimuli, len(SEGMENTS)))
    np.add.at(shares, (acts.row_stimulus, kept), weights)
    return shares


def _cell_values(
    acts: Activations, heads: NDArray[np.float64], biases: NDArray[np.float64]
) -> NDArray[Any]:
    rows = acts.data.astype(np.float64)
    return cast(
        NDArray[Any], np.einsum("krd,kd->kr", rows, heads) + biases[:, None]
    )


def _per_stimulus(
    acts: Activations, values: NDArray[np.float64]
) -> NDArray[np.float64]:
    totals = np.zeros(acts.n_stimuli)
    np.add.at(
        totals, np.tile(acts.row_stimulus, values.shape[0]), values.ravel()
    )
    return totals


def test_cell_weights_put_a_cell_max_reader_on_its_highest_cell() -> None:
    acts, labels = _random_cells()
    # Stimulus 0's two rows tie at every layer, so they share its weight.
    acts.data[:, 1] = acts.data[:, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        reader = fit_cell_max(acts, labels=labels)

    weights = m3_matrix.cell_weights(reader, acts)

    values = _cell_values(acts, reader.values, reader.biases)
    assert weights.shape == values.shape
    for s in range(acts.n_stimuli):
        rows = acts.row_stimulus == s
        w, v = weights[:, rows], values[:, rows]
        top = v == v.max()
        np.testing.assert_allclose(w[top], 1.0 / top.sum())
        assert (w[~top] == 0.0).all()
    assert (weights[:, :2].max(axis=1) == [0.5, 0.5]).any()
    np.testing.assert_allclose(
        _per_stimulus(acts, weights * values),
        reader.score(acts)[0],
        rtol=1e-6,
    )


def test_cell_weights_of_a_joint_attention_reader_are_its_softmax() -> None:
    acts, labels = _random_cells()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        reader = fit_attention(acts, labels=labels, joint=True)

    weights = m3_matrix.cell_weights(reader, acts)

    logits = _cell_values(acts, reader.queries, reader.offsets)
    for s in range(acts.n_stimuli):
        rows = acts.row_stimulus == s
        exp = np.exp(logits[:, rows] - logits[:, rows].max())
        np.testing.assert_allclose(weights[:, rows], exp / exp.sum())
    heads = _cell_values(acts, reader.values, reader.biases)
    np.testing.assert_allclose(
        _per_stimulus(acts, weights * heads),
        reader.score(acts)[0],
        rtol=1e-5,
    )


def test_the_diagnostics_share_each_reader_among_the_segments(
    finished: RunDir,
) -> None:
    names = {p.name for p in finished.path.iterdir()}
    assert {"diagnostics.json", "cellheads_sentiment.npz"} <= names
    payload = cast(dict[str, Any], finished.read_json("diagnostics"))
    assert payload["columns"] == list(m3_matrix.DIAGNOSTIC_COLUMNS)
    summary = cast(dict[str, Any], finished.read_json("summary"))
    assert "prefix" in summary["diagnostics"]
    rows = [
        r for r in _table(finished, "diagnostics") if r["task"] == "sentiment"
    ]
    cells = [
        (scorer, f"{pooling}_{cell_set}", series)
        for cell_set in CELL_SETS
        for pooling in m3_matrix.POOLINGS
        for scorer, series in (
            ("logistic", "reader"),
            ("logistic", "shuffled"),
            *((("random", "reader"),) if pooling == "cellmax" else ()),
        )
    ]
    assert sorted(
        (r["scorer"], r["readout"], r["series"], r["condition"]) for r in rows
    ) == sorted(
        (*cell, condition)
        for cell in cells
        for condition in SENTIMENT_CONDITIONS
    )
    by_cell = {}
    for row in rows:
        shares = [row[segment] for segment in SEGMENTS]
        assert sum(shares) == pytest.approx(1.0)
        assert min(shares) >= 0.0
        cell_set = row["readout"].split("_")[1]
        if cell_set in ("stimulus", "final"):
            assert row[cell_set] == pytest.approx(1.0)
        if row["condition"] not in ("random", "against"):
            assert row["decoy"] == 0.0
        by_cell[row["readout"], row["series"], row["condition"]] = row
    for (readout, series, condition), row in by_cell.items():
        if readout.endswith("_upper"):
            prompt = by_cell[
                readout.replace("_upper", "_prompt"), series, condition
            ]
            # The prompt set reads layers 1 and 3, the upper set layer 3.
            assert 2 * row["cells"] == pytest.approx(prompt["cells"])


@pytest.fixture
def captured(
    runs: dict[str, Any],
    tiny_gpt2: LoadedModel,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[RunDir, Any]:
    if "captured" not in runs:
        items = _capture_items(monkeypatch)
        config = _captured_config()
        run_dir = RunDir.create(config, out_root=tmp_path_factory.mktemp("m3"))
        _quiet_execute(config, run_dir, tiny_gpt2)
        runs["captured"] = (run_dir, items[0])
    return cast(tuple[RunDir, Any], runs["captured"])


def _captured_config() -> m3_matrix.Config:
    return _config(tasks=("utility",), cell_sets=("prompt", "upper"))


def test_the_cell_count_correlation_is_taken_within_each_label(
    captured: tuple[RunDir, Any], tiny_gpt2: LoadedModel
) -> None:
    run_dir, splits = captured
    config = _captured_config()
    arrays = run_dir.read_arrays("scores_utility")
    e = _entries(run_dir, "utility").index(
        "logistic/cellmax_prompt/joint/reader"
    )
    rows = {
        r["condition"]: r
        for r in _table(run_dir, "diagnostics")
        if (r["scorer"], r["readout"], r["series"])
        == ("logistic", "cellmax_prompt", "reader")
    }
    defined = 0
    for m, condition in enumerate(splits.conditions):
        cells, _ = prompt_cells(
            tiny_gpt2,
            splits.report_items[condition],
            config.cell_layers,
            bos=None,
            batch_size=config.batch_size,
            decoy=condition in config.decoys,
        )
        counts = np.bincount(cells.row_stimulus) * len(config.cell_layers)
        labels = splits.report_items[condition].labels
        rhos = [
            spearman(
                counts[labels == label].astype(np.float64),
                arrays["report"][e, i, m][labels == label],
            )
            for i in range(config.n_seeds)
            for label in (0, 1)
        ]
        finite = [rho for rho in rhos if np.isfinite(rho)]
        assert rows[condition]["cells"] == pytest.approx(counts.mean())
        if finite:
            defined += 1
            assert rows[condition]["cells_rho"] == pytest.approx(
                np.mean(finite)
            )
        else:
            assert rows[condition]["cells_rho"] is None
    assert defined > 0


def _kept_reader(heads: Any, key: str, i: int, layers: tuple[int, ...]) -> Any:
    """Draw i's kept reader of a key of `cellheads_<task>`."""
    fields = {
        name: heads[f"{key}_{name}"][i]
        for name in ("values", "biases", "signs")
    }
    if f"{key}_queries" in heads:
        return AttentionReader(
            layers=layers,
            joint=True,
            queries=heads[f"{key}_queries"][i],
            offsets=heads[f"{key}_offsets"][i],
            **fields,
        )
    return CellMaxReader(layers=layers, **fields)


def test_the_kept_heads_reproduce_the_select_scores(
    captured: tuple[RunDir, Any], tiny_gpt2: LoadedModel
) -> None:
    run_dir, splits = captured
    config = _captured_config()
    heads = run_dir.read_arrays("cellheads_utility")
    sidecar = cast(dict[str, Any], run_dir.read_json("cellheads_utility"))
    arrays = run_dir.read_arrays("scores_utility")
    entries = _entries(run_dir, "utility")
    cells, segments = prompt_cells(
        tiny_gpt2,
        splits.select_items,
        config.cell_layers,
        bos=None,
        batch_size=config.batch_size,
        decoy=False,
    )
    listed = []
    for key, entry in sidecar["entries"].items():
        listed.append(entry["entry"])
        cell_set = cast(CellSet, entry["entry"].split("/")[1].split("_")[1])
        acts = select_cells(
            cells, segments, cell_set, _set_layers(config, cell_set)
        )
        assert entry["layers"] == list(acts.layers)
        e = entries.index(entry["entry"])
        for i in range(config.n_seeds):
            reader = _kept_reader(heads, key, i, acts.layers)
            np.testing.assert_array_equal(
                reader.score(acts)[0], arrays["select"][e, i]
            )
        if entry["entry"].startswith("random/"):
            assert f"{key}_c" not in heads
        else:
            assert set(heads[f"{key}_c"]) <= set(config.logistic_c)
    assert sorted(listed) == sorted(e for e in entries if "/joint/" in e)


def test_the_segment_shares_are_the_kept_readers_mean_weights(
    captured: tuple[RunDir, Any], tiny_gpt2: LoadedModel
) -> None:
    run_dir, splits = captured
    config = _captured_config()
    heads = run_dir.read_arrays("cellheads_utility")
    sidecar = cast(dict[str, Any], run_dir.read_json("cellheads_utility"))
    rows = {
        (r["scorer"], r["readout"], r["series"], r["condition"]): r
        for r in _table(run_dir, "diagnostics")
    }
    for condition in splits.conditions:
        cells, segments = prompt_cells(
            tiny_gpt2,
            splits.report_items[condition],
            config.cell_layers,
            bos=None,
            batch_size=config.batch_size,
            decoy=condition in config.decoys,
        )
        for key, entry in sidecar["entries"].items():
            scorer, readout, _, series = entry["entry"].split("/")
            cell_set = cast(CellSet, readout.split("_")[1])
            acts = select_cells(
                cells, segments, cell_set, _set_layers(config, cell_set)
            )
            shares = np.mean(
                [
                    _shares(
                        _kept_reader(heads, key, i, acts.layers),
                        acts,
                        segments,
                        cell_set,
                    )
                    for i in range(config.n_seeds)
                ],
                axis=(0, 1),
            )
            row = rows[scorer, readout, series, condition]
            np.testing.assert_allclose(
                [row[segment] for segment in SEGMENTS], shares, atol=1e-12
            )


def test_the_summary_counts_fits_seconds_and_convergence_warnings(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def planting(fit: Any, *, planted: bool) -> Any:
        def fitting(acts: Activations, **kwargs: Any) -> Any:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                reader = fit(acts, **kwargs)
            if planted:
                warnings.warn("planted", ConvergenceWarning, stacklevel=2)
            return reader

        return fitting

    monkeypatch.setattr(
        m3_matrix, "fit_cell_max", planting(fit_cell_max, planted=True)
    )
    monkeypatch.setattr(
        m3_matrix, "fit_attention", planting(fit_attention, planted=False)
    )
    config = _config(tasks=("utility",), cell_sets=("prompt",), decoys=())
    run_dir = RunDir.create(config, out_root=tmp_path)
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter("always")
        m3_matrix.execute(config, run_dir, tiny_gpt2, _sources())

    summary = cast(dict[str, Any], run_dir.read_json("summary"))
    fits = summary["fits"]["utility"]
    n = config.n_seeds * len(config.logistic_c)
    assert set(fits) == {
        f"logistic/{pooling}_prompt/joint/{series}"
        for pooling in m3_matrix.POOLINGS
        for series in ("reader", "shuffled")
    }
    for name, stats in fits.items():
        assert stats["fits"] == n
        assert stats["seconds"] > 0.0
        expected = n if "cellmax" in name else 0
        assert stats["convergence_warnings"] == expected
    # The counted warnings still reach the caller.
    planted = [w for w in shown if str(w.message) == "planted"]
    assert len(planted) == 2 * n
