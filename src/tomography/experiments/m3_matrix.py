"""M3: readers trained through an aggregation over the cells of the prompt.

A cell is one (token, layer) of a prompt. `prompt_cells` reads the cells of
every token but BOS at chosen layers and labels each token with its segment,
listed in prompt order in `SEGMENTS`: `prefix` (every token before the
stimulus that is not in the decoy line: the chat tag, the instruction, and
a tag such as utility's `Scenario:` after the decoy line), `decoy` (P2's
misleading line, in the misleading conditions only), `stimulus` (the
`{stimulus}` slot) and `final` (from the stimulus end to the text end, where
`last` and `concept` read). `select_cells` keeps the cells of a cell set:
`prompt` and `upper` hold every segment (`upper` at the upper layers),
`stimulus` and `final` one each. `stimulus_token_set` and `prompt_max_set`
give the spans through which M2's `concept` and `max` positions read the
last stimulus token and the feature-wise maximum over the prompt.

The experiment runs M2's protocol on M2's items. Its one-row phase fits
M2's scorers on M2's one-row readouts and on `stimulus_token` and
`prompt_max`, with M2's helpers. Its cell phase fits, per draw, a logistic
head per layer trained through the pooling of a cell set's cells: one
softmax over all of them (`cellattn`) or their maximum (`cellmax`), as a
joint reader. Every reader scores the report items of every condition and
is compared with the one-row baselines, next to M2's controls and a random
cell max reader. Per condition, the diagnostics share each cell reader's
weight (`cell_weights`) among the segments and set its score against the
number of cells of a text, since the cell max rises with that number.
"""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
from dataclasses import dataclass
from pathlib import Path
import re
import time
from types import MappingProxyType
from typing import Any
from typing import Literal
from typing import NamedTuple
from typing import get_args
import warnings

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
import torch

from tomography.activations import Activations
from tomography.activations import PositionStrategy
from tomography.activations import Span
from tomography.activations import _overlapping_tokens
from tomography.activations import collect_activations
from tomography.datasets.records import Example
from tomography.datasets.stimulus import StimulusSet
from tomography.experiments import m2_readers
from tomography.experiments import p2_positions
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.p3_size import spearman
from tomography.experiments.sweep import shuffle_within_groups
from tomography.experiments.variants import ReadingTask
from tomography.experiments.variants import TaskSource
from tomography.experiments.variants import load_sources
from tomography.experiments.variants import target_of
from tomography.experiments.variants import task_examples
from tomography.model import BosMode
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.reading.metrics import auroc
from tomography.reading.metrics import group_accuracy
from tomography.reading.probes import AttentionReader
from tomography.reading.probes import CellMaxReader
from tomography.reading.probes import _scaled_cells
from tomography.reading.probes import _unscaled
from tomography.reading.probes import fit_attention
from tomography.reading.probes import fit_cell_max

Segment = Literal["prefix", "decoy", "stimulus", "final"]
CellSet = Literal["prompt", "stimulus", "final", "upper"]
Pooling = Literal["cellattn", "cellmax"]
SEGMENTS: tuple[Segment, ...] = get_args(Segment)
CELL_SETS: tuple[CellSet, ...] = get_args(CellSet)
POOLINGS: tuple[Pooling, ...] = get_args(Pooling)
SET_SEGMENTS: Mapping[CellSet, tuple[Segment, ...]] = MappingProxyType(
    {
        "prompt": SEGMENTS,
        "stimulus": ("stimulus",),
        "final": ("final",),
        "upper": SEGMENTS,
    }
)
TOKEN_READOUTS: tuple[m2_readers.Readout, ...] = (
    "token_score_max",
    "attention",
)
FAMILIES = (
    "one per task, condition, layer rule and baseline; Holm over the DeLong "
    "and, separately, the McNemar p values of its cells; the joint cells are "
    "compared with each baseline's top_k rule and, in a family of their own, "
    "its best rule"
)
DIAGNOSTIC_COLUMNS = (
    "task",
    "condition",
    "scorer",
    "readout",
    "series",
    *SEGMENTS,
    "cells",
    "cells_rho",
)
DIAGNOSTICS = (
    "one row per task, condition and cell reader; each segment column is the "
    "share of a report text's reader weight on the cells of that segment "
    "(cellmax: its highest cell, ties split; cellattn: its attention mass), "
    "averaged over the texts and draws; cells is a text's mean number of "
    "cells in the reader's set, and cells_rho the Spearman correlation of "
    "that number with the score within each label, averaged over the labels "
    "and draws where it is defined (null if none is), since the cell max "
    "rises with the number of cells; a prefix share can still be "
    "decoy-driven, as utility's Scenario: tag follows the misleading line"
)
CellReader = AttentionReader | CellMaxReader


@dataclass(frozen=True, kw_only=True)
class Config(m2_readers.Config):
    """M3 settings: M2's protocol with readers trained through the cells.

    The fields shared with M2 keep M2's defaults, so a run reads M2's
    items. `token_layers` is not read: the cell readers replace M2's token
    readouts and read `cell_layers`.

    Attributes:
        scorers: The scorers of the one-row readouts, from M2's `SCORERS`.
        readouts: M2's one-row readouts to replicate; every scorer also
            reads `stimulus_token` and `prompt_max`.
        cell_layers: The layers whose cells the cell readers read, among
            the layers read.
        upper_layers: The layers of the `upper` set, among `cell_layers`.
        cell_sets: The cell sets to read, from `CELL_SETS`.
        poolings: How the cell readers pool their cells, from `POOLINGS`:
            `cellattn` one softmax over the cells of all layers, `cellmax`
            the highest cell.
        baselines: The cells every other cell is compared with, as
            `<scorer>/<readout>` of a one-row readout.
    """

    experiment: str = "m3"
    scorers: tuple[m2_readers.Scorer, ...] = ("logistic", "mean_diff")
    readouts: tuple[m2_readers.Readout, ...] = (
        "last",
        "concept",
        "mean",
        "max",
    )
    cell_layers: tuple[int, ...] = (4, 8, 12, 16, 20, 24, 28)
    upper_layers: tuple[int, ...] = (16, 20, 24, 28)
    cell_sets: tuple[CellSet, ...] = ("prompt",)
    poolings: tuple[Pooling, ...] = POOLINGS
    baselines: tuple[str, ...] = (
        "logistic/last",
        "mean_diff/last",
        "logistic/concept",
        "logistic/stimulus_token",
    )


def prompt_cells(
    loaded: LoadedModel,
    stimuli: StimulusSet,
    layers: Sequence[int],
    *,
    bos: BosMode | None,
    batch_size: int,
    decoy: bool,
) -> tuple[Activations, NDArray[np.int8]]:
    """Reads the cells of every token but BOS and labels their segments.

    The cells are the `response` rows of `collect_activations` over
    whole-text spans, so BOS, whose offsets are empty, and padding are never
    read, and the rows stay ragged and stimulus-major.

    Args:
        loaded: The wrapped model.
        stimuli: The stimuli, with the stimulus span of each text.
        layers: The layers to read.
        bos: BOS policy for `LoadedModel.encode`; None uses the model's
            default.
        batch_size: Texts per forward pass.
        decoy: Whether the texts carry P2's misleading line.

    Returns:
        The `response` rows of every token but BOS, and the segment code of
        each row, its index in `SEGMENTS`.

    Raises:
        ValueError: If a text has no stimulus span, or `decoy` is set and
            the line before a stimulus line is not the decoy line (the
            error names the stimulus); both before any forward pass.
    """
    encoded = loaded.encode(stimuli.texts, bos=bos)
    token_segments = [
        _token_segments(
            index,
            text,
            span,
            encoded.offsets[index][: encoded.lengths[index]],
            decoy=decoy,
        )
        for index, (text, span) in enumerate(
            zip(stimuli.texts, stimuli.response_spans, strict=True)
        )
    ]
    cells = collect_activations(
        loaded,
        stimuli.texts,
        layers=layers,
        position="response",
        response_spans=_whole_texts(stimuli.texts),
        bos=bos,
        batch_size=batch_size,
    )
    lengths = np.array([codes.size for codes in token_segments])
    starts = np.cumsum(lengths) - lengths
    segments = np.concatenate(token_segments)[
        starts[cells.row_stimulus] + cells.row_token
    ]
    return cells, segments


def stimulus_token_set(stimuli: StimulusSet) -> StimulusSet:
    """Marks the stimulus as the concept, for the `stimulus_token` readout.

    M2's `concept` position then reads the last stimulus token, the
    stimulus analogue of `last`.

    Args:
        stimuli: The stimuli.

    Returns:
        The stimuli with their stimulus spans as concept spans.
    """
    return dataclasses.replace(stimuli, concept_spans=stimuli.response_spans)


def prompt_max_set(stimuli: StimulusSet) -> StimulusSet:
    """Marks the whole text as the stimulus, for the `prompt_max` readout.

    M2's `max` position then takes the feature-wise maximum over every
    token but BOS, so over the rows of `prompt_cells`.

    Args:
        stimuli: The stimuli.

    Returns:
        The stimuli with whole-text stimulus spans.
    """
    return dataclasses.replace(
        stimuli, response_spans=_whole_texts(stimuli.texts)
    )


def _whole_texts(texts: Sequence[str]) -> tuple[Span, ...]:
    """Spans that cover each text, so they overlap every token but BOS."""
    return tuple((0, len(text)) for text in texts)


def _token_segments(
    index: int,
    text: str,
    span: Span | None,
    offsets: tuple[tuple[int, int], ...],
    *,
    decoy: bool,
) -> NDArray[np.int8]:
    """Labels each token of a text with the code of its segment.

    A token takes a segment when it overlaps the segment's characters, by
    the rule of `collect_activations`, and the stimulus wins a token it
    shares with the final line or the decoy line.

    Args:
        index: The index of the text, to name it in errors.
        text: The text.
        span: The character span of its stimulus.
        offsets: The character span of each of its tokens.
        decoy: Whether the text carries the decoy line.

    Returns:
        The segment code of each token, int8.

    Raises:
        ValueError: If `span` is None, or `decoy` is set and the line before
            the stimulus line is not the decoy line.
    """
    if span is None:
        raise ValueError(f"stimulus {index} has no stimulus span")
    parts: list[tuple[Segment, Span]] = []
    if decoy:
        line = _decoy_line(text, span[0])
        if line is None:
            raise ValueError(
                f"stimulus {index} has no decoy line right before the line "
                "of its stimulus"
            )
        parts.append(("decoy", line))
    parts += [("final", (span[1], len(text))), ("stimulus", span)]
    codes = np.full(len(offsets), SEGMENTS.index("prefix"), dtype=np.int8)
    for segment, part in parts:
        codes[_overlapping_tokens(offsets, len(offsets), part)] = (
            SEGMENTS.index(segment)
        )
    return codes


def _decoy_line(text: str, stimulus_start: int) -> Span | None:
    """Finds the line `misleading_template` puts right before the stimulus.

    The line is `p2_positions.DECOY_LINE` with any text on one line in its
    slots, and it ends where the line that holds the stimulus starts; a
    chat tag may precede it on its line.

    Returns:
        Its character span, or None when the text right before the stimulus
        line is not that line.
    """
    end = text.rfind("\n", 0, stimulus_start) + 1
    pattern = "".join(
        ".+?" if part.startswith("{") else re.escape(part)
        for part in re.split(r"(\{+\w+\}+)", p2_positions.DECOY_LINE)
    )
    found = re.search(rf"{pattern}\Z", text[:end])
    return None if found is None else (found.start(), end)


def select_cells(
    cells: Activations,
    segments: NDArray[np.int8],
    cell_set: CellSet,
    layers: Sequence[int],
) -> Activations:
    """Keeps the cells of a cell set at some layers.

    Args:
        cells: The cells of `prompt_cells`.
        segments: The segment code of each row of `cells`.
        cell_set: The set whose segments to keep, from `CELL_SETS`.
        layers: The layers to keep, in the order `cells` reads them.

    Returns:
        The rows of the set's segments at those layers, stimulus-major,
        ranked again within each stimulus.

    Raises:
        ValueError: If the set is unknown, a layer was not read, or a
            stimulus has no cells in the set (the error names it).
    """
    if cell_set not in SET_SEGMENTS:
        raise ValueError(
            f"unknown cell set {cell_set!r}, expected one of {CELL_SETS}"
        )
    missing = [layer for layer in layers if layer not in cells.layers]
    if missing:
        raise ValueError(f"layers {missing} were not read, only {cells.layers}")
    keep = np.flatnonzero(_in_set(segments, cell_set))
    at = [k for k, layer in enumerate(cells.layers) if layer in layers]
    stimulus = cells.row_stimulus[keep]
    counts = np.bincount(stimulus, minlength=cells.n_stimuli)
    if not counts.all():
        raise ValueError(
            f"stimulus {int(np.argmin(counts))} has no cells in the set "
            f"{cell_set!r}"
        )
    starts = np.cumsum(counts) - counts
    return Activations(
        data=cells.data[np.ix_(at, keep)],
        layers=tuple(cells.layers[k] for k in at),
        position=cells.position,
        row_stimulus=stimulus,
        row_rank=np.arange(keep.size) - starts[stimulus],
        row_token=cells.row_token[keep],
        row_token_id=cells.row_token_id[keep],
        n_stimuli=cells.n_stimuli,
    )


def _in_set(segments: NDArray[np.int8], cell_set: CellSet) -> NDArray[np.bool_]:
    """Whether each row's segment is one of the cell set's."""
    return np.isin(
        segments,
        [SEGMENTS.index(segment) for segment in SET_SEGMENTS[cell_set]],
    )


RELABELLED: Mapping[
    str, tuple[Callable[[StimulusSet], StimulusSet], PositionStrategy]
] = MappingProxyType(
    {
        "stimulus_token": (stimulus_token_set, "concept"),
        "prompt_max": (prompt_max_set, "max"),
    }
)


def random_cell_max(
    acts: Activations, labels: NDArray[np.int64], *, seed: int
) -> CellMaxReader:
    """Draws the random-head control of the cell max reader; nothing is fit.

    Each layer gets a standard Gaussian head, with no bias, on its cells as
    `fit_cell_max` scales them: centered on the layer's mean over `acts` and
    divided by its root mean square norm. One sign for every head is set on
    the labels: the one whose stimulus scores over `acts` have the higher
    AUROC, plus on a tie.

    Args:
        acts: The training cells.
        labels: Their binary labels, one per stimulus.
        seed: Seed of the heads.

    Returns:
        The reader, in activation units, with signs +1.

    Raises:
        ValueError: If `acts` is not finite, a layer's rows are all equal,
            or the labels lack a class.
    """
    centers, scales = _scaled_cells(acts.data, acts.layers)[1:]
    heads = np.random.default_rng(seed).standard_normal(
        (len(acts.layers), acts.data.shape[2])
    )
    values, shifts = _unscaled(heads, centers, scales)
    readers = [
        CellMaxReader(
            layers=tuple(int(layer) for layer in acts.layers),
            values=sign * values,
            biases=-sign * shifts,
            signs=np.ones(len(acts.layers), dtype=np.float32),
        )
        for sign in (1.0, -1.0)
    ]
    areas = [float(auroc(reader.score(acts)[0], labels)) for reader in readers]
    return readers[int(np.argmax(areas))]


def cell_weights(reader: CellReader, acts: Activations) -> NDArray[np.float64]:
    """The weight of each cell in its stimulus's score.

    A cell max reader puts all of a stimulus's weight on its highest cell,
    split evenly among tied cells; a joint attention reader spreads it by
    the softmax of the cells' logits. Either way a stimulus scores the sum
    of its cells' values times their weights, and its weights sum to 1.

    Args:
        reader: A cell max reader or a joint attention reader.
        acts: Cells of the reader's layers.

    Returns:
        The weights, shape (K, n_rows), in the layout of `acts.data`.

    Raises:
        ValueError: If the reader is a per-layer attention reader, or a
            stimulus has no rows.
    """
    if isinstance(reader, CellMaxReader):
        heads, offsets = reader.values, reader.biases
    elif reader.joint:
        heads, offsets = reader.queries, reader.offsets
    else:
        raise ValueError(
            "cell_weights needs a cell max or a joint attention reader, got "
            "a per-layer attention reader"
        )
    # The reader's own float64 arithmetic, so the highest cell is its own.
    logits = np.stack(
        [
            rows.astype(np.float64) @ heads[k] + offsets[k]
            for k, rows in enumerate(acts.data)
        ]
    )
    stimulus = acts.row_stimulus
    top = m2_readers.token_score_max(logits, acts).max(axis=0)[stimulus]
    if isinstance(reader, CellMaxReader):
        weights = (logits == top).astype(np.float64)
    else:
        weights = np.exp(logits - top)
    totals = np.bincount(
        stimulus, weights.sum(axis=0), minlength=acts.n_stimuli
    )
    return np.asarray(weights / totals[stimulus], dtype=np.float64)


class Streamed(NamedTuple):
    """What `stream_scores` keeps of each reader on each stimulus.

    Attributes:
        scores: The scores, shape (n_readers, n_stimuli).
        shares: The share of each stimulus's weight (see `cell_weights`) on
            the cells of each segment of `SEGMENTS`, shape (n_readers,
            n_stimuli, len(SEGMENTS)).
        cells: The number of cells each reader reads of each stimulus,
            shape (n_readers, n_stimuli).
    """

    scores: NDArray[np.float64]
    shares: NDArray[np.float64]
    cells: NDArray[np.int64]


def stream_scores(
    loaded: LoadedModel,
    stimuli: StimulusSet,
    readers: Sequence[tuple[CellSet, CellReader]],
    config: Config,
    *,
    decoy: bool,
) -> Streamed:
    """Scores stimuli with cell readers, reading their cells batch by batch.

    Each batch of `config.batch_size` texts is read once by `prompt_cells`
    at `cell_layers`, every reader scores the cells of its set, and the
    cells are released before the next batch is read, so only the scores,
    the segment shares and the cell counts are held.

    Args:
        loaded: The wrapped model.
        stimuli: The stimuli to score.
        readers: Each reader, with the cell set it reads.
        config: Supplies `cell_layers`, `upper_layers`, the BOS policy and
            the batch size.
        decoy: Whether the texts carry P2's misleading line.

    Returns:
        The scores, segment shares and cell counts of each reader and
        stimulus.

    Raises:
        ValueError: If a batch is refused by `prompt_cells`, `select_cells`
            or a reader; the error names the stimulus the batch starts at.
    """
    batches = []
    for first in range(0, len(stimuli.texts), config.batch_size):
        rows = slice(first, first + config.batch_size)
        try:
            batches.append(
                _batch_scores(
                    loaded, _rows(stimuli, rows), readers, config, decoy=decoy
                )
            )
        except ValueError as error:
            raise ValueError(
                f"in the batch that starts at stimulus {first}: {error}"
            ) from error
    return Streamed(
        *(np.concatenate(parts, axis=1) for parts in zip(*batches, strict=True))
    )


def _batch_scores(
    loaded: LoadedModel,
    stimuli: StimulusSet,
    readers: Sequence[tuple[CellSet, CellReader]],
    config: Config,
    *,
    decoy: bool,
) -> Streamed:
    """Scores one batch; its cells are released on return."""
    cells, segments = prompt_cells(
        loaded,
        stimuli,
        config.cell_layers,
        bos=config.bos,
        batch_size=config.batch_size,
        decoy=decoy,
    )
    kept = {
        cell_set: (
            select_cells(
                cells, segments, cell_set, _set_layers(config, cell_set)
            ),
            segments[_in_set(segments, cell_set)],
        )
        for cell_set in dict.fromkeys(cell_set for cell_set, _ in readers)
    }
    n_stimuli, n_segments = len(stimuli.texts), len(SEGMENTS)
    scores, shares, counts = [], [], []
    for cell_set, reader in readers:
        acts, kept_segments = kept[cell_set]
        scores.append(reader.score(acts)[0])
        shares.append(
            np.bincount(
                acts.row_stimulus * n_segments + kept_segments,
                cell_weights(reader, acts).sum(axis=0),
                minlength=n_stimuli * n_segments,
            ).reshape(n_stimuli, n_segments)
        )
        counts.append(np.bincount(acts.row_stimulus) * len(acts.layers))
    return Streamed(
        np.array(scores, dtype=np.float64), np.array(shares), np.array(counts)
    )


def _rows(stimuli: StimulusSet, rows: slice) -> StimulusSet:
    """The stimuli of some rows, without the pairs and groups."""
    return StimulusSet(
        texts=stimuli.texts[rows],
        response_spans=stimuli.response_spans[rows],
        concept_spans=stimuli.concept_spans[rows],
        labels=stimuli.labels[rows],
        pairs=stimuli.pairs[:0],
        groups=stimuli.groups[:0],
        uids=stimuli.uids[rows],
    )


def _set_layers(config: Config, cell_set: CellSet) -> tuple[int, ...]:
    """The layers a cell set reads."""
    return config.upper_layers if cell_set == "upper" else config.cell_layers


class _Fit(NamedTuple):
    """The readers one draw fits for one cell and series.

    Attributes:
        cell: The cell, `<scorer>/<pooling>_<set>/joint`.
        series: `reader`, or `shuffled` for a refit on shuffled labels.
        cell_set: The set the readers read.
        readers: One reader per grid value.
        select_labels: The select labels that tune and threshold them.
        seconds: The time the fits took; 0 for the random control.
        convergence_warnings: The `ConvergenceWarning`s the fits raised.
    """

    cell: m2_readers.Cell
    series: str
    cell_set: CellSet
    readers: list[CellReader]
    select_labels: NDArray[np.int64]
    seconds: float = 0.0
    convergence_warnings: int = 0


def _fit_cells(
    pooling: Pooling, train: Activations, labels: NDArray[np.int64], c: float
) -> CellReader:
    """Fits a joint reader through the pooling of the training cells."""
    if pooling == "cellattn":
        return fit_attention(train, labels=labels, c=c, joint=True)
    return fit_cell_max(train, labels=labels, c=c)


def _grid_fits(
    pooling: Pooling,
    train: Activations,
    labels: NDArray[np.int64],
    grid: Sequence[float],
) -> tuple[list[CellReader], float, int]:
    """Fits a reader per grid value, timing the fits and counting warnings.

    Every `ConvergenceWarning` is counted, then shown again under the
    caller's filters; under the `default` action a warning repeated over
    the grid shows once.

    Returns:
        The readers, the seconds the fits took, and the number of
        `ConvergenceWarning`s they raised.
    """
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        readers = [_fit_cells(pooling, train, labels, c) for c in grid]
    seconds = time.perf_counter() - start
    # One registry per grid: every `catch_warnings` clears the registries,
    # so a longer-lived one would not dedupe across grids either.
    registry: dict[Any, Any] = {}
    for shown in caught:
        warnings.warn_explicit(
            shown.message,
            shown.category,
            shown.filename,
            shown.lineno,
            registry=registry,
        )
    counted = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    return readers, seconds, counted


def _shuffled(
    config: Config,
    task: ReadingTask,
    items: m2_readers._Splits,
    i: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """M2's shuffles of draw i's training and select labels.

    They are the shuffles of M2's `_cell_entries`, within the training and
    the select groups, so every cell of a draw shares them.
    """
    stimuli, select = items.train_items[i], items.select_items
    return (
        shuffle_within_groups(
            stimuli.labels,
            stimuli.groups,
            seed=derive_seed(config.seed, "shuffled", task.name, i),
        ),
        shuffle_within_groups(
            select.labels,
            select.groups,
            seed=derive_seed(config.seed, "shuffled-select", task.name, i),
        ),
    )


def _draw_fits(
    config: Config,
    loaded: LoadedModel,
    task: ReadingTask,
    items: m2_readers._Splits,
    i: int,
) -> list[_Fit]:
    """Fits every cell reader of draw i; its cells are released on return."""
    stimuli = items.train_items[i]
    cells, segments = prompt_cells(
        loaded,
        stimuli,
        config.cell_layers,
        bos=config.bos,
        batch_size=config.batch_size,
        decoy=False,
    )
    series = [("reader", stimuli.labels, items.select_items.labels)]
    if "shuffled" in config.controls:
        series.append(("shuffled", *_shuffled(config, task, items, i)))
    fits: list[_Fit] = []
    # `select_cells` copies, so each set's copy is freed before the next,
    # and the cells once the last set is copied.
    *sets, last = config.cell_sets
    for cell_set in sets:
        train = select_cells(
            cells, segments, cell_set, _set_layers(config, cell_set)
        )
        fits += _set_fits(config, task, items, i, train, cell_set, series)
        del train
    train = select_cells(cells, segments, last, _set_layers(config, last))
    del cells, segments
    return fits + _set_fits(config, task, items, i, train, last, series)


def _set_fits(
    config: Config,
    task: ReadingTask,
    items: m2_readers._Splits,
    i: int,
    train: Activations,
    cell_set: CellSet,
    series: Sequence[tuple[str, NDArray[np.int64], NDArray[np.int64]]],
) -> list[_Fit]:
    """Fits the readers of one cell set at every grid value, and the control.

    The random control draws its heads with the seed M2 would give a
    `random` scorer on the readout `cellmax_<set>`.
    """
    fits = []
    for pooling in config.poolings:
        cell = m2_readers.Cell("logistic", f"{pooling}_{cell_set}", "joint")
        for name, labels, select_labels in series:
            readers, seconds, caught = _grid_fits(
                pooling, train, labels, config.logistic_c
            )
            fits.append(
                _Fit(
                    cell,
                    name,
                    cell_set,
                    readers,
                    select_labels,
                    seconds,
                    caught,
                )
            )
    if "random" in config.controls and "cellmax" in config.poolings:
        readout = f"cellmax_{cell_set}"
        seed = derive_seed(config.seed, "fit", task.name, "random", readout, i)
        fits.append(
            _Fit(
                m2_readers.Cell("random", readout, "joint"),
                "reader",
                cell_set,
                [
                    random_cell_max(
                        train, items.train_items[i].labels, seed=seed
                    )
                ],
                items.select_items.labels,
            )
        )
    return fits


class _CellPhase(NamedTuple):
    """What the cell phase gives of one task.

    Attributes:
        entries: The entries of the cell readers and the random control.
        diagnostics: The task's rows of `diagnostics.json`.
        fits: The task's `fits` entry of the summary.
        heads: The arrays of `cellheads_<task>.npz`.
        sidecar: The JSON sidecar of those arrays.
    """

    entries: list[m2_readers._Entry]
    diagnostics: list[list[Any]]
    fits: dict[str, dict[str, float]]
    heads: dict[str, NDArray[Any]]
    sidecar: dict[str, Any]


def _cell_phase(
    config: Config,
    loaded: LoadedModel,
    task: ReadingTask,
    items: m2_readers._Splits,
) -> _CellPhase:
    """Fits, tunes and scores the cell readers and their random control.

    Every draw is fitted first. The select items are then streamed once
    through every reader; each draw keeps the first best grid value on the
    select pair accuracy, as M2's `_tuned` does with one layer; and the
    report items of each condition are streamed once through the kept
    readers.
    """
    fits = [
        fit
        for i in range(len(items.train_items))
        for fit in _draw_fits(config, loaded, task, items, i)
    ]
    select = items.select_items
    scored = stream_scores(
        loaded,
        select,
        [(fit.cell_set, reader) for fit in fits for reader in fit.readers],
        config,
        decoy=False,
    ).scores
    sizes = np.cumsum([len(fit.readers) for fit in fits])[:-1]
    best: list[int] = []
    vals = []
    kept: list[tuple[CellSet, CellReader]] = []
    for fit, val in zip(fits, np.split(scored, sizes), strict=True):
        metric = group_accuracy(val[:, None], select.groups, fit.select_labels)
        b = int(m2_readers.tune_per_layer(metric)[0])
        best.append(b)
        vals.append(val[b])
        kept.append((fit.cell_set, fit.readers[b]))
    streamed = [
        stream_scores(
            loaded,
            items.report_items[condition],
            kept,
            config,
            decoy=condition in config.decoys,
        )
        for condition in items.conditions
    ]
    report = Streamed(
        *(np.stack(parts, axis=1) for parts in zip(*streamed, strict=True))
    )
    groups: dict[tuple[m2_readers.Cell, str], list[int]] = {}
    for j, fit in enumerate(fits):
        groups.setdefault((fit.cell, fit.series), []).append(j)
    entries = [
        m2_readers._Entry(
            cell,
            series,
            [
                m2_readers._Choice(
                    _set_layers(config, fits[j].cell_set),
                    vals[j],
                    report.scores[j],
                    fits[j].select_labels,
                )
                for j in draws
            ],
        )
        for (cell, series), draws in groups.items()
    ]
    fit_counts = {
        f"{cell.name}/{series}": {
            "fits": sum(len(fits[j].readers) for j in draws),
            "seconds": sum(fits[j].seconds for j in draws),
            "convergence_warnings": sum(
                fits[j].convergence_warnings for j in draws
            ),
        }
        for (cell, series), draws in groups.items()
        if cell.scorer == "logistic"
    }
    return _CellPhase(
        entries,
        _diagnostic_rows(task, items, groups, report),
        fit_counts,
        *_heads(config, fits, best, groups),
    )


def _diagnostic_rows(
    task: ReadingTask,
    items: m2_readers._Splits,
    groups: Mapping[tuple[m2_readers.Cell, str], Sequence[int]],
    report: Streamed,
) -> list[list[Any]]:
    """The segment shares and cell counts of each cell reader and condition.

    Args:
        task: The task.
        items: Its items.
        groups: The fits of each cell and series, one per draw.
        report: The report items of each fit, stacked by condition on the
            second axis.

    Returns:
        Rows of `DIAGNOSTIC_COLUMNS`.
    """
    rows: list[list[Any]] = []
    for (cell, series), draws in groups.items():
        for m, condition in enumerate(items.conditions):
            labels = items.report_items[condition].labels
            rhos = [
                spearman(
                    report.cells[j, m][labels == label].astype(np.float64),
                    report.scores[j, m][labels == label],
                )
                for j in draws
                for label in (0, 1)
            ]
            finite = [rho for rho in rhos if np.isfinite(rho)]
            rows.append(
                [
                    task.name,
                    condition,
                    cell.scorer,
                    cell.readout,
                    series,
                    *report.shares[list(draws), m].mean(axis=(0, 1)).tolist(),
                    float(report.cells[draws[0], m].mean()),
                    float(np.mean(finite)) if finite else None,
                ]
            )
    return rows


def _heads(
    config: Config,
    fits: Sequence[_Fit],
    best: Sequence[int],
    groups: Mapping[tuple[m2_readers.Cell, str], Sequence[int]],
) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    """The kept readers' parameters, stacked over the draws.

    Args:
        config: Supplies the grid of `logistic_c`.
        fits: Every fit of the task.
        best: The grid value each fit kept.
        groups: The fits of each cell and series, one per draw.

    Returns:
        The arrays, keyed `<scorer>_<readout>_<series>_<field>`, and their
        sidecar, which names the entry and the layers of each key.
    """
    arrays: dict[str, NDArray[Any]] = {}
    listed: dict[str, Any] = {}
    for (cell, series), draws in groups.items():
        key = f"{cell.scorer}_{cell.readout}_{series}"
        readers = [fits[j].readers[best[j]] for j in draws]
        fields = ["values", "biases", "signs"]
        if isinstance(readers[0], AttentionReader):
            fields += ["queries", "offsets"]
        for field in fields:
            arrays[f"{key}_{field}"] = np.stack(
                [getattr(reader, field) for reader in readers]
            )
        if cell.scorer == "logistic":
            arrays[f"{key}_c"] = np.array(
                [config.logistic_c[best[j]] for j in draws]
            )
        listed[key] = {
            "entry": f"{cell.name}/{series}",
            "layers": list(readers[0].layers),
        }
    sidecar = {
        "entries": listed,
        "axes": {
            "values": ["seed", "layer", "feature"],
            "queries": ["seed", "layer", "feature"],
            "biases": ["seed", "layer"],
            "offsets": ["seed", "layer"],
            "signs": ["seed", "layer"],
            "c": ["seed"],
        },
    }
    return arrays, sidecar


def _row_entries(
    config: Config,
    task: ReadingTask,
    splits: m2_readers._Splits,
    listed: Sequence[m2_readers.Cell],
) -> list[m2_readers._Entry]:
    """Fits one-row cells and controls as M2's `_run_task` does."""
    rules: dict[tuple[str, str], list[str]] = {}
    for cell in listed:
        rules.setdefault((cell.scorer, cell.readout), []).append(
            cell.layer_rule
        )
    entries: list[m2_readers._Entry] = []
    for (scorer, readout), cell_rules in rules.items():
        if scorer == "text":
            text = m2_readers.Cell(scorer, readout, "none")
            choices = m2_readers._text_choices(splits)
            entries.append(m2_readers._Entry(text, "reader", choices))
        else:
            entries += m2_readers._cell_entries(
                config, task, splits, scorer, readout, cell_rules
            )
    return entries


def _replicas(
    config: Config,
    loaded: LoadedModel,
    task: ReadingTask,
    scorers: Sequence[str],
    readouts: Sequence[m2_readers.Readout],
    examples: Mapping[str, Sequence[Example]],
) -> tuple[list[m2_readers._Entry], m2_readers._Splits, dict[str, str]]:
    """Fits M2's one-row cells on M2's items and releases their activations.

    Returns:
        The entries, the items without their activations, and the reasons
        of the skipped template shifts.
    """
    splits, skipped = m2_readers._prepare_task(
        config, loaded, task, readouts, examples
    )
    listed = m2_readers.cells(config, readouts, scorers=scorers)
    entries = _row_entries(config, task, splits, listed)
    items = dataclasses.replace(splits, train={}, select={}, report={})
    return entries, items, skipped


def _relabelled(
    config: Config,
    loaded: LoadedModel,
    task: ReadingTask,
    items: m2_readers._Splits,
    scorers: Sequence[str],
    readout: str,
) -> list[m2_readers._Entry]:
    """Fits a relabelled readout and its random control on its own pass.

    The readout is M2's `concept` or `max` position over other spans, so
    M2's helpers fit it under that readout's key, with its seeds, and the
    entries are relabelled. Its activations are released on return.
    """
    mark, position = RELABELLED[readout]

    def read(stimuli: StimulusSet) -> Activations:
        return m2_readers.read_positions(
            loaded, mark(stimuli), (position,), config
        )[position]

    splits = dataclasses.replace(
        items,
        train={position: [read(s) for s in items.train_items]},
        select={position: read(items.select_items)},
        report={position: [read(s) for s in items.report_items.values()]},
    )
    listed = [
        cell
        for cell in m2_readers.cells(config, [position], scorers=scorers)
        if cell.scorer != "text"
    ]
    return [
        entry._replace(cell=entry.cell._replace(readout=readout))
        for entry in _row_entries(config, task, splits, listed)
    ]


def _comparisons(
    config: Config,
    task: ReadingTask,
    items: m2_readers._Splits,
    entries: Sequence[m2_readers._Entry],
) -> list[list[Any]]:
    """M2's comparisons, plus the joint cells against each baseline's best.

    M2 compares the joint cells with each baseline's `top_k` rule. Each
    baseline's `best` cell is also compared with the joint cells, in a Holm
    family of their own, by handing M2's `_comparison_rows` the joint cells
    relabelled `best`; their rows keep the rule `joint`.
    """
    rows = m2_readers._comparison_rows(config, task, items, entries)
    rule = m2_readers.COMPARISON_COLUMNS.index("layer_rule")
    joint = [
        entry._replace(cell=entry.cell._replace(layer_rule="best"))
        for entry in entries
        if entry.cell.layer_rule == "joint"
    ]
    for baseline in config.baselines:
        scorer, readout = baseline.split("/")
        base = m2_readers.Cell(scorer, readout, "best")
        family = [entry for entry in entries if entry.cell == base] + joint
        best = dataclasses.replace(
            config, baselines=(baseline,), layer_rules=("best",)
        )
        for row in m2_readers._comparison_rows(best, task, items, family):
            row[rule] = "joint"
            rows.append(row)
    return rows


def _check(config: Config) -> tuple[str, ...]:
    """Checks what the config alone settles and expands its tasks.

    M2's checks run first, with M2's messages, all but its baseline check,
    which m3 widens to the relabelled readouts.
    """
    tasks = m2_readers._check(dataclasses.replace(config, baselines=()))
    if not config.reader_series:
        raise ValueError(
            "m3 fits the reader series, so reader_series must be True"
        )
    token = tuple(r for r in config.readouts if r in TOKEN_READOUTS)
    if token:
        raise ValueError(
            "m3 reads no token readouts, the cell readers replace them: "
            f"got {token}"
        )
    for field, values, known, noun in (
        ("cell_sets", config.cell_sets, CELL_SETS, "cell set"),
        ("poolings", config.poolings, POOLINGS, "pooling"),
    ):
        if not values:
            raise ValueError(f"m3 needs at least one entry in {field}")
        for value in values:
            if value not in known:
                raise ValueError(
                    f"unknown {noun} {value!r}, expected one of {known}"
                )
    if not config.cell_layers:
        raise ValueError("m3 needs at least one entry in cell_layers")
    if not config.upper_layers or not set(config.upper_layers) <= set(
        config.cell_layers
    ):
        raise ValueError(
            f"upper_layers must name layers of cell_layers "
            f"{config.cell_layers}, got {config.upper_layers}"
        )
    readouts = (*config.readouts, *RELABELLED)
    for baseline in config.baselines:
        scorer, _, readout = baseline.partition("/")
        if scorer not in config.scorers or readout not in readouts:
            raise ValueError(
                f"baseline {baseline!r} is not a cell of the run: give "
                "<scorer>/<readout> from the configured scorers and the "
                f"one-row readouts {readouts}"
            )
    return tasks


def _run_task(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    name: str,
    examples: Mapping[str, Sequence[Example]],
) -> tuple[list[list[Any]], list[list[Any]], list[list[Any]], dict[str, Any]]:
    """Runs both phases on one task and writes its files.

    The one-row activations are released before the cell phase reads.

    Returns:
        The table rows, the comparison rows, the diagnostic rows, and the
        task's `conditions`, `skipped`, `n_eval_pairs` and `fits` entries
        of the summary.
    """
    task = m2_readers._task(name)
    scorers, readouts, skipped = m2_readers._readable(task, config)
    entries, items, shift_skips = _replicas(
        config, loaded, task, scorers, readouts, examples
    )
    for readout in RELABELLED:
        entries += _relabelled(config, loaded, task, items, scorers, readout)
    cells = _cell_phase(config, loaded, task, items)
    entries += cells.entries
    entry_rows = [
        m2_readers._metric_rows(config, task, items, e) for e in entries
    ]
    m2_readers._write_task(run_dir, task, items, entries, entry_rows)
    run_dir.write_arrays(f"cellheads_{task.name}", **cells.heads)
    run_dir.write_json(f"cellheads_{task.name}", cells.sidecar)
    return (
        [row for block in entry_rows for row in block],
        _comparisons(config, task, items, entries),
        cells.diagnostics,
        {
            "conditions": items.conditions,
            "skipped": {**skipped, **shift_skips},
            "n_eval_pairs": {
                "select": int(items.select_items.groups.shape[0]),
                "report": int(items.report_items["original"].groups.shape[0]),
            },
            "fits": cells.fits,
        },
    )


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    examples: Mapping[str, Sequence[Example]],
) -> None:
    """Runs the one-row and the cell phase on every task.

    The one-row phase is M2's on M2's items: per task and seed, every
    one-row cell is fitted on the training items of the paper template,
    tuned and given its layer rule and thresholds on the select items of
    the original prompts. The cell phase then reads each draw's training
    cells once, fits every cell reader at every grid value and releases
    them; streams the select items once through every reader and keeps, per
    draw, the first best grid value; and streams each condition's report
    items once through the kept readers. Writes M2's files: per task
    `scores_<task>` and `auroc_<task>`, and `table.json`,
    `comparisons.json` and `summary.json`; and per task the kept readers'
    parameters in `cellheads_<task>`, and `diagnostics.json` (see
    `DIAGNOSTICS`). The summary also counts, per task and fitted cell
    reader, the fits, their seconds and their `ConvergenceWarning`s.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model, in float32 for a real run.
        examples: The examples of each source, by source name.

    Raises:
        ValueError: If the config is refused (see `run`), a source is
            missing, or `cell_layers` holds a layer that is not read (all
            checked before any model call).
    """
    tasks = _check(config)
    for name in tasks:
        task = m2_readers._task(name)
        task_examples(task, examples)
        task_examples(target_of(task), examples)
    layers = m2_readers._read_layers(config, loaded)
    if not set(config.cell_layers) <= set(layers):
        raise ValueError(
            f"cell_layers {config.cell_layers} must be among the layers "
            f"read {layers}"
        )
    rows: list[list[Any]] = []
    comparisons: list[list[Any]] = []
    diagnostics: list[list[Any]] = []
    summary: dict[str, Any] = {
        "tasks": list(tasks),
        "conditions": {},
        "skipped": {},
        "n_eval_pairs": {},
        "fits": {},
        "families": FAMILIES,
        "diagnostics": DIAGNOSTICS,
    }
    for name in tasks:
        task_rows, task_comparisons, task_diagnostics, parts = _run_task(
            config,
            run_dir,
            loaded,
            name,
            m2_readers._task_examples(name, examples),
        )
        rows += task_rows
        comparisons += task_comparisons
        diagnostics += task_diagnostics
        for key, value in parts.items():
            summary[key][name] = value
    run_dir.write_json(
        "table", {"columns": list(m2_readers.COLUMNS), "rows": rows}
    )
    run_dir.write_json(
        "comparisons",
        {"columns": list(m2_readers.COMPARISON_COLUMNS), "rows": comparisons},
    )
    run_dir.write_json(
        "diagnostics",
        {"columns": list(DIAGNOSTIC_COLUMNS), "rows": diagnostics},
    )
    run_dir.write_json("summary", summary)


def run(config: Config, run_dir: RunDir) -> None:
    """Loads the model in float32 and the datasets, then runs `execute`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.

    Raises:
        ValueError: If M2 refuses the config, `reader_series` is False, a
            readout is a token readout, a cell set or pooling is unknown or
            none is given, `cell_layers` is empty, `upper_layers` is empty
            or not among `cell_layers`, or a baseline is not a one-row cell
            of the run (all checked before the model is loaded).
    """
    tasks = map(m2_readers._task, _check(config))
    sources: list[TaskSource] = list(
        dict.fromkeys(
            source
            for task in tasks
            for source in (task.source, target_of(task).source)
        )
    )
    execute(
        config,
        run_dir,
        load_model(config.model, dtype=torch.float32),
        load_sources(sources),
    )


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws the AUROC matrix of every task, as M2's `plot` does.

    Args:
        run_dir: A finished M3 run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths.
    """
    return m2_readers.plot(run_dir, out_dir)
