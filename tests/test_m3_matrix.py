"""Tests for M3's cell extraction: every token of the prompt at many layers."""

import dataclasses
from typing import Any
from typing import cast

import numpy as np
import pytest

from tomography.activations import Activations
from tomography.activations import Span
from tomography.activations import collect_activations
from tomography.datasets.records import Example
from tomography.datasets.stimulus import StimulusSet
from tomography.experiments import m2_readers
from tomography.experiments import m3_matrix
from tomography.experiments import p2_positions
from tomography.experiments.m2_readers import read_positions
from tomography.experiments.m3_matrix import CELL_SETS
from tomography.experiments.m3_matrix import SEGMENTS
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

LAYERS = (1, 2, 3)
# The paper's decoy line is too long for the 128 positions of the tiny models.
SHORT_DECOY_LINE = "Hint: {noun} is {{question}}.\n"
PREFIX, DECOY, STIMULUS, FINAL = (
    SEGMENTS.index(s) for s in ("prefix", "decoy", "stimulus", "final")
)
EXPECTED_SEGMENTS = {
    "prompt": SEGMENTS,
    "stimulus": ("stimulus",),
    "final": ("final",),
    "upper": SEGMENTS,
}


def test_segments_and_cell_sets_are_named_in_prompt_order() -> None:
    assert SEGMENTS == ("prefix", "decoy", "stimulus", "final")
    assert CELL_SETS == ("prompt", "stimulus", "final", "upper")


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


ROW_FIELDS = ("row_stimulus", "row_rank", "row_token", "row_token_id")


def _assert_same_rows(got: Activations, expected: Activations) -> None:
    assert got.layers == expected.layers
    assert got.n_stimuli == expected.n_stimuli
    for field in ROW_FIELDS:
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
    cells, segments = prompt_cells(
        tiny_arch, stimuli, LAYERS, bos=None, batch_size=4, decoy=decoy
    )

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
    cells, segments = prompt_cells(
        tiny_arch, stimuli, LAYERS, bos=None, batch_size=4, decoy=False
    )

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
    cells, segments = prompt_cells(
        tiny_arch, stimuli, LAYERS, bos=None, batch_size=4, decoy=decoy
    )

    noun = READING_TASKS[task].noun
    for i, label in enumerate(stimuli.labels):
        codes = segments[cells.row_stimulus == i]
        ids = cells.row_token_id[cells.row_stimulus == i]
        start = int(np.argmax(codes == STIMULUS))
        assert codes[0] == PREFIX
        assert set(codes[:start].tolist()) <= {PREFIX, DECOY}
        assert set(codes[start:].tolist()) == {STIMULUS, FINAL}
        assert np.all(np.diff(codes[start:]) >= 0)
        # Utility's "Scenario:" tag is prefix between the decoy and the
        # stimulus, so only the concept template is in order throughout.
        in_order = bool(np.all(np.diff(codes) >= 0))
        assert in_order == (task == "sentiment" or not decoy)
        line = np.flatnonzero(codes == DECOY)
        if not decoy:
            assert line.size == 0
            continue
        np.testing.assert_array_equal(np.diff(line), 1)
        level = "low" if label == 1 else "high"
        assert tiny_arch.tokenizer.decode(ids[line].tolist()) == (
            f"Hint: {noun} is {level}.\n"
        )


@pytest.mark.parametrize("decoy", [False, True])
def test_the_last_final_cell_is_the_last_row(
    tiny_arch: LoadedModel, decoy: bool
) -> None:
    stimuli = _stimuli(tiny_arch, "utility", decoy=decoy)
    cells, segments = prompt_cells(
        tiny_arch, stimuli, LAYERS, bos=None, batch_size=4, decoy=decoy
    )

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
    cells, segments = prompt_cells(
        tiny_arch, stimuli, LAYERS, bos=None, batch_size=4, decoy=True
    )
    stimulus = select_cells(cells, segments, "stimulus", LAYERS)

    read = read_positions(
        tiny_arch, stimulus_token_set(stimuli), ("concept",), _m2_config()
    )["concept"]

    ends = np.cumsum(np.bincount(stimulus.row_stimulus)) - 1
    np.testing.assert_array_equal(read.row_token, stimulus.row_token[ends])
    np.testing.assert_array_equal(read.data, stimulus.data[:, ends])


def test_prompt_max_pools_every_cell_of_a_text(tiny_arch: LoadedModel) -> None:
    stimuli = _stimuli(tiny_arch, "sentiment", decoy=True)
    cells, _ = prompt_cells(
        tiny_arch, stimuli, LAYERS, bos=None, batch_size=4, decoy=True
    )

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
    cells, segments = prompt_cells(
        tiny_gpt2, stimuli, LAYERS, bos=None, batch_size=4, decoy=True
    )

    got = select_cells(cells, segments, cell_set, (3, 1))

    codes = [SEGMENTS.index(s) for s in EXPECTED_SEGMENTS[cell_set]]
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
    cells, segments = prompt_cells(
        tiny_gpt2, stimuli, (1, 3), bos=None, batch_size=4, decoy=False
    )

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
    cells, segments = prompt_cells(
        tiny_gpt2, to_the_end, LAYERS, bos=None, batch_size=4, decoy=False
    )

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
        prompt_cells(
            tiny_gpt2, broken, LAYERS, bos=None, batch_size=4, decoy=False
        )


@pytest.mark.parametrize("case", ["missing", "not_before_the_stimulus"])
def test_prompt_cells_name_a_misleading_text_without_its_decoy_line(
    tiny_gpt2: LoadedModel, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    misleading = _stimuli(tiny_gpt2, "utility", decoy=True)
    if case == "missing":
        original = _stimuli(tiny_gpt2, "utility", decoy=False)
        index = 1
        broken = _replace(
            misleading, 1, original.texts[1], original.response_spans[1]
        )
    else:
        index = 3
        text, (start, end) = misleading.texts[3], _span(misleading, 3)
        line = text.rfind("\n", 0, start) + 1
        moved = text[:line] + "x\n" + text[line:]
        broken = _replace(misleading, 3, moved, (start + 2, end + 2))
    _no_model_call(monkeypatch)

    with pytest.raises(
        ValueError,
        match=rf"^stimulus {index} has no decoy line right before the line "
        r"of its stimulus$",
    ):
        prompt_cells(
            tiny_gpt2, broken, LAYERS, bos=None, batch_size=4, decoy=True
        )
