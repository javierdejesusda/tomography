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
`stimulus` and `final` one each.
"""

from collections.abc import Mapping
from collections.abc import Sequence
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
from tomography.experiments import p2_positions
from tomography.model import BosMode
from tomography.model import LoadedModel

Segment = Literal["prefix", "decoy", "stimulus", "final"]
CellSet = Literal["prompt", "stimulus", "final", "upper"]
SEGMENTS: tuple[Segment, ...] = get_args(Segment)
CELL_SETS: tuple[CellSet, ...] = get_args(CellSet)
SET_SEGMENTS: Mapping[CellSet, tuple[Segment, ...]] = MappingProxyType(
    {
        "prompt": SEGMENTS,
        "stimulus": ("stimulus",),
        "final": ("final",),
        "upper": SEGMENTS,
    }
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

    The line runs from the text of `p2_positions.DECOY_LINE` before its
    first slot to the start of the line that holds the stimulus.

    Returns:
        Its character span, or None when the line before the stimulus line
        does not hold it.
    """
    end = text.rfind("\n", 0, stimulus_start) + 1
    start = text.rfind(p2_positions.DECOY_LINE.partition("{")[0], 0, end)
    if start < 0 or "\n" in text[start : end - 1]:
        return None
    return start, end


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
