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
cell max reader.
"""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
from dataclasses import dataclass
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any
from typing import Literal
from typing import NamedTuple
from typing import get_args

import numpy as np
from numpy.typing import NDArray
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
    codes = [SEGMENTS.index(segment) for segment in SET_SEGMENTS[cell_set]]
    keep = np.flatnonzero(np.isin(segments, codes))
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
    _, centers, scales = _scaled_cells(acts.data, acts.layers)
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


def stream_scores(
    loaded: LoadedModel,
    stimuli: StimulusSet,
    readers: Sequence[tuple[CellSet, CellReader]],
    config: Config,
    *,
    decoy: bool,
) -> NDArray[np.float64]:
    """Scores stimuli with cell readers, reading their cells batch by batch.

    Each batch of `config.batch_size` texts is read once by `prompt_cells`
    at `cell_layers`, every reader scores the cells of its set, and the
    cells are released before the next batch is read, so only the scores
    are held.

    Args:
        loaded: The wrapped model.
        stimuli: The stimuli to score.
        readers: Each reader, with the cell set it reads.
        config: Supplies `cell_layers`, `upper_layers`, the BOS policy and
            the batch size.
        decoy: Whether the texts carry P2's misleading line.

    Returns:
        The score of each reader and stimulus, shape (n_readers, n_stimuli).

    Raises:
        ValueError: If a batch is refused by `prompt_cells`, `select_cells`
            or a reader; the error names the stimulus the batch starts at.
    """
    scores = np.empty((len(readers), len(stimuli.texts)))
    for first in range(0, len(stimuli.texts), config.batch_size):
        rows = slice(first, first + config.batch_size)
        try:
            scores[:, rows] = _batch_scores(
                loaded, _rows(stimuli, rows), readers, config, decoy=decoy
            )
        except ValueError as error:
            raise ValueError(
                f"in the batch that starts at stimulus {first}: {error}"
            ) from error
    return scores


def _batch_scores(
    loaded: LoadedModel,
    stimuli: StimulusSet,
    readers: Sequence[tuple[CellSet, CellReader]],
    config: Config,
    *,
    decoy: bool,
) -> NDArray[np.float64]:
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
        cell_set: select_cells(
            cells, segments, cell_set, _set_layers(config, cell_set)
        )
        for cell_set in dict.fromkeys(cell_set for cell_set, _ in readers)
    }
    return np.array(
        [reader.score(kept[cell_set])[0] for cell_set, reader in readers],
        dtype=np.float64,
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
    """

    cell: m2_readers.Cell
    series: str
    cell_set: CellSet
    readers: list[CellReader]
    select_labels: NDArray[np.int64]


def _fit_cells(
    pooling: Pooling, train: Activations, labels: NDArray[np.int64], c: float
) -> CellReader:
    """Fits a joint reader through the pooling of the training cells."""
    if pooling == "cellattn":
        return fit_attention(train, labels=labels, c=c, joint=True)
    return fit_cell_max(train, labels=labels, c=c)


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
    for cell_set in config.cell_sets:
        train = select_cells(
            cells, segments, cell_set, _set_layers(config, cell_set)
        )
        fits += _set_fits(config, task, items, i, train, cell_set, series)
        # `select_cells` copies, so one set's copy is freed before the next.
        del train
    return fits


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
    fits = [
        _Fit(
            m2_readers.Cell("logistic", f"{pooling}_{cell_set}", "joint"),
            name,
            cell_set,
            [_fit_cells(pooling, train, labels, c) for c in config.logistic_c],
            select_labels,
        )
        for pooling in config.poolings
        for name, labels, select_labels in series
    ]
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


def _matrix_entries(
    config: Config,
    loaded: LoadedModel,
    task: ReadingTask,
    items: m2_readers._Splits,
) -> list[m2_readers._Entry]:
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
    )
    sizes = np.cumsum([len(fit.readers) for fit in fits])[:-1]
    kept: list[tuple[CellSet, CellReader]] = []
    vals = []
    for fit, val in zip(fits, np.split(scored, sizes), strict=True):
        metric = group_accuracy(val[:, None], select.groups, fit.select_labels)
        best = int(m2_readers.tune_per_layer(metric)[0])
        kept.append((fit.cell_set, fit.readers[best]))
        vals.append(val[best])
    report = np.stack(
        [
            stream_scores(
                loaded,
                items.report_items[condition],
                kept,
                config,
                decoy=condition in config.decoys,
            )
            for condition in items.conditions
        ],
        axis=1,
    )
    choices: dict[tuple[m2_readers.Cell, str], list[m2_readers._Choice]] = {}
    for fit, val, scores in zip(fits, vals, report, strict=True):
        choices.setdefault((fit.cell, fit.series), []).append(
            m2_readers._Choice(
                _set_layers(config, fit.cell_set),
                val,
                scores,
                fit.select_labels,
            )
        )
    return [
        m2_readers._Entry(cell, series, per_seed)
        for (cell, series), per_seed in choices.items()
    ]


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
        rows += [
            [*row[:2], "joint", *row[3:]]
            for row in m2_readers._comparison_rows(best, task, items, family)
        ]
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
) -> tuple[list[list[Any]], list[list[Any]], dict[str, Any]]:
    """Runs both phases on one task and writes its files.

    The one-row activations are released before the cell phase reads.

    Returns:
        The table rows, the comparison rows, and the task's `conditions`,
        `skipped` and `n_eval_pairs` entries of the summary.
    """
    task = m2_readers._task(name)
    scorers, readouts, skipped = m2_readers._readable(task, config)
    entries, items, shift_skips = _replicas(
        config, loaded, task, scorers, readouts, examples
    )
    for readout in RELABELLED:
        entries += _relabelled(config, loaded, task, items, scorers, readout)
    entries += _matrix_entries(config, loaded, task, items)
    entry_rows = [
        m2_readers._metric_rows(config, task, items, e) for e in entries
    ]
    m2_readers._write_task(run_dir, task, items, entries, entry_rows)
    return (
        [row for block in entry_rows for row in block],
        _comparisons(config, task, items, entries),
        {
            "conditions": items.conditions,
            "skipped": {**skipped, **shift_skips},
            "n_eval_pairs": {
                "select": int(items.select_items.groups.shape[0]),
                "report": int(items.report_items["original"].groups.shape[0]),
            },
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
    `comparisons.json` and `summary.json`.

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
    summary: dict[str, Any] = {
        "tasks": list(tasks),
        "conditions": {},
        "skipped": {},
        "n_eval_pairs": {},
        "families": FAMILIES,
    }
    for name in tasks:
        task_rows, task_comparisons, parts = _run_task(
            config,
            run_dir,
            loaded,
            name,
            m2_readers._task_examples(name, examples),
        )
        rows += task_rows
        comparisons += task_comparisons
        for key, value in parts.items():
            summary[key][name] = value
    run_dir.write_json(
        "table", {"columns": list(m2_readers.COLUMNS), "rows": rows}
    )
    run_dir.write_json(
        "comparisons",
        {"columns": list(m2_readers.COMPARISON_COLUMNS), "rows": comparisons},
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
