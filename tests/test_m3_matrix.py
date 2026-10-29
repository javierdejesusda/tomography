"""Tests for M3: readers trained through an aggregation over prompt cells."""

import dataclasses
from typing import Any
from typing import cast
import warnings

import numpy as np
from numpy.typing import NDArray
import pytest
from sklearn.exceptions import ConvergenceWarning

from tomography.activations import Activations
from tomography.activations import Span
from tomography.activations import collect_activations
from tomography.cli import _parse_override
from tomography.datasets.records import Example
from tomography.datasets.stimulus import StimulusSet
from tomography.experiments import m2_readers
from tomography.experiments import m3_matrix
from tomography.experiments import p2_positions
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
from tomography.experiments.sweep import chat_format_for
from tomography.experiments.variants import READING_TASKS
from tomography.experiments.variants import task_set
from tomography.experiments.variants import template_variants
from tomography.model import BosMode
from tomography.model import LoadedModel
from tomography.reading.metrics import auroc
from tomography.reading.probes import CellMaxReader
from tomography.reading.probes import fit_attention
from tomography.reading.probes import fit_cell_max

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
    expected = [
        reader.score(
            select_cells(
                cells, segments, cell_set, _set_layers(config, cell_set)
            )
        )[0]
        for cell_set, reader in readers
    ]
    assert streamed.dtype == np.float64
    np.testing.assert_array_equal(streamed, np.array(expected))


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
