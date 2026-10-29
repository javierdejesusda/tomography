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
`stream_scores` scores stimuli with cell readers batch by batch, and
`random_cell_max` draws the random control of the cell max reader.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Literal
from typing import get_args

import numpy as np
from numpy.typing import NDArray

from tomography.activations import Activations
from tomography.activations import Span
from tomography.activations import _overlapping_tokens
from tomography.activations import collect_activations
from tomography.datasets.stimulus import StimulusSet
from tomography.experiments import m2_readers
from tomography.experiments import p2_positions
from tomography.model import BosMode
from tomography.model import LoadedModel
from tomography.reading.metrics import auroc
from tomography.reading.probes import AttentionReader
from tomography.reading.probes import CellMaxReader
from tomography.reading.probes import _scaled_cells
from tomography.reading.probes import _unscaled

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
