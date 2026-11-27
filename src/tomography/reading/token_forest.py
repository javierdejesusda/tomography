"""Literal token forests and a matched token-count control."""

from collections import defaultdict
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from tomography import activations
from tomography import model
from tomography.datasets import stimulus
from tomography.experiments import m3_matrix
from tomography.reading import ensembles
from tomography.reading import reader

Alignment = Literal["right", "left"]
Rows = NDArray[np.float32]
Indices = NDArray[np.int64]
Scores = NDArray[np.float64]
LITERAL_LAYERS = (4, 8, 12, 16, 20, 24, 28)


def literal_layers(layers: Sequence[int]) -> tuple[int, ...]:
    """Returns the bridge depths available in the configured layer set."""
    return tuple(layer for layer in LITERAL_LAYERS if layer in layers)


@dataclass(frozen=True)
class TokenLayout:
    """A flattened layer-major layout of the training token blocks.

    Attributes:
        layers: Retained layers, in feature column order.
        width: Longest stimulus-to-end token span in this training draw.
        depth: Residual stream width.
        alignment: Side where the surviving tokens are placed.
    """

    layers: tuple[int, ...]
    width: int
    depth: int
    alignment: Alignment

    @property
    def groups(self) -> tuple[range, ...]:
        """Returns one contiguous column group per retained layer."""
        size = self.width * self.depth
        return tuple(
            range(offset, offset + size)
            for offset in range(0, self.n_columns, size)
        )

    @property
    def n_columns(self) -> int:
        """Returns the flattened width across all retained layers."""
        return len(self.layers) * self.width * self.depth

    def pack(
        self, rows: Rows, cells: Rows, counts: Indices, lengths: Indices
    ) -> None:
        """Writes one flat row per prompt, keeping its last width tokens.

        Args:
            rows: Caller-owned float32 output matrix.
            cells: Layer-major, stimulus-major ragged whole-prompt cells.
            counts: Total number of cells per prompt.
            lengths: Number of cells from the stimulus start to the end.
        """
        blocks = rows.reshape(-1, len(self.layers), self.width, self.depth)
        blocks.fill(0)
        end = 0
        for block, count, length in zip(blocks, counts, lengths, strict=True):
            end += int(count)
            take = min(int(length), self.width)
            start = self.width - take if self.alignment == "right" else 0
            block[:, start : start + take] = cells[:, end - take : end]


def span_lengths(
    loaded: model.LoadedModel,
    items: stimulus.StimulusSet,
    *,
    bos: model.BosMode | None,
    batch_size: int,
) -> Indices:
    """Counts cells from the first stimulus-overlapping token to the end.

    Args:
        loaded: Model supplying the same tokenizer as M3.
        items: Prompts with marked stimulus spans.
        bos: The forward pass's BOS policy.
        batch_size: Maximum number of prompts tokenized together.

    Returns:
        Untruncated lengths in original item order, excluding BOS and padding.

    Raises:
        ValueError: If a stimulus span is missing, invalid or overlaps no token.
    """
    lengths = np.empty(len(items.texts), dtype=np.int64)
    for start in range(0, len(items.texts), batch_size):
        texts = items.texts[start : start + batch_size]
        encoded = loaded.encode(texts, bos=bos)
        for row, text in enumerate(texts):
            index = start + row
            span = items.response_spans[index]
            offsets, size = encoded.offsets[row], encoded.lengths[row]
            hit = (
                activations._overlapping_tokens(offsets, size, span)
                if span is not None and 0 <= span[0] < span[1] <= len(text)
                else []
            )
            if not hit:
                raise ValueError(
                    f"stimulus {index} has no token at its stimulus start"
                )
            lengths[index] = sum(lo < hi for lo, hi in offsets[hit[0] : size])
    return lengths


def fit_length_control(
    items: stimulus.StimulusSet, lengths: Indices
) -> Callable[[Indices], Scores]:
    """Orients raw span token counts on training groups, keeping ties positive.

    Args:
        items: Training stimuli with labels and evaluation groups.
        lengths: Untruncated token counts in training item order.

    Returns:
        A scorer that applies the fixed sign to untruncated token counts.
    """
    groups = reader._orientation_groups(items.pairs, items.groups, items.labels)
    sign = np.float64(
        reader._orientation(lengths[None].astype(float), items.labels, groups)[
            0
        ]
    )
    return lambda counts: counts.astype(np.float64) * sign


def fit_token_forest(
    layout: TokenLayout,
    rows: Rows,
    items: stimulus.StimulusSet,
    n_trees: int,
    seed: int,
    *,
    retain_columns: bool = True,
) -> ensembles.GroupedForest:
    """Fits the shared forest over three whole token blocks per tree.

    Args:
        layout: Feature groups of the packed rows.
        rows: Packed training rows in item order.
        items: Training labels and pair indices.
        n_trees: Number of trees.
        seed: Pair bootstrap, layer draw and split seed.
        retain_columns: Keep local column mappings for sklearn replay.

    Returns:
        A fitted global-node scorer.
    """
    return ensembles.fit_grouped_forest(
        rows,
        items.labels,
        items.pairs,
        layout.groups,
        m=min(3, len(layout.layers)),
        n_trees=n_trees,
        seed=seed,
        retain_columns=retain_columns,
    )


def _cell_batches(
    loaded: model.LoadedModel,
    items: stimulus.StimulusSet,
    layers: tuple[int, ...],
    bos: model.BosMode | None,
    batch_size: int,
) -> Iterator[tuple[slice, Rows, Indices]]:
    """Reads each batch once, in the original order and at retained depths."""
    for start in range(0, len(items.texts), batch_size):
        rows = slice(start, min(start + batch_size, len(items.texts)))
        batch = m3_matrix._rows(items, rows)
        # The decoy line is before the start, not part of the retained span.
        cells, _ = m3_matrix.prompt_cells(
            loaded, batch, layers, bos=bos, batch_size=batch_size, decoy=False
        )
        counts = np.bincount(cells.row_stimulus, minlength=cells.n_stimuli)
        yield rows, cells.data, counts
        del cells


@dataclass(frozen=True)
class LiteralDraw:
    """The two readers' scores for one matched draw.

    Attributes:
        width: Training-derived token columns per layer.
        forest_select: Forest probabilities in select item order.
        forest_report: Forest probabilities in condition and report item order.
        length_select: Oriented token counts in select item order.
        length_report: Oriented counts in condition and report item order.
    """

    width: int
    forest_select: Scores
    forest_report: Scores
    length_select: Scores
    length_report: Scores


def score_splits(
    loaded: model.LoadedModel,
    train: Sequence[stimulus.StimulusSet],
    select: stimulus.StimulusSet,
    report: Sequence[stimulus.StimulusSet],
    layers: tuple[int, ...],
    *,
    bos: model.BosMode | None,
    batch_size: int,
    alignment: Alignment,
    n_trees: int,
    seeds: Sequence[int],
) -> list[LiteralDraw]:
    """Fits draw-specific layouts and scores each evaluation batch once.

    Args:
        loaded: Causal model for tokenization and forward passes.
        train: Training stimuli for each draw.
        select: Original-template selection stimuli.
        report: Reporting stimuli in condition order.
        layers: Retained layer indices, shared by all draws.
        bos: The run's BOS policy.
        batch_size: Maximum number of prompts forwarded together.
        alignment: Side where the surviving tokens are placed.
        n_trees: Number of trees per draw.
        seeds: Pair bootstrap and split seed for each draw.

    Returns:
        Matched forest and length scores in draw order.
    """
    # Check all splits before the first literal forward, even the last batch.
    counts = [
        span_lengths(loaded, items, bos=bos, batch_size=batch_size)
        for items in (*train, select, *report)
    ]
    select_counts, *report_counts = counts[len(train) :]
    by_layout: dict[TokenLayout, list[tuple[int, ensembles.GroupedForest]]] = (
        defaultdict(list)
    )
    widths = []
    controls = []
    for index, (items, lengths, seed) in enumerate(
        zip(train, counts[: len(train)], seeds, strict=True)
    ):
        layout = TokenLayout(
            layers, int(lengths.max()), loaded.d_model, alignment
        )
        rows = np.empty((len(items.texts), layout.n_columns), dtype=np.float32)
        for span, cells, totals in _cell_batches(
            loaded, items, layers, bos, batch_size
        ):
            layout.pack(rows[span], cells, totals, lengths[span])
            del cells
        forest = fit_token_forest(
            layout, rows, items, n_trees, seed, retain_columns=False
        )
        by_layout[layout].append((index, forest))
        widths.append(layout.width)
        controls.append(fit_length_control(items, lengths))
        del rows

    def evaluate(items: stimulus.StimulusSet, lengths: Indices) -> Scores:
        scores = np.empty((len(train), len(items.texts)), dtype=np.float64)
        for span, cells, totals in _cell_batches(
            loaded, items, layers, bos, batch_size
        ):
            for layout, forests in by_layout.items():
                rows = np.empty(
                    (len(totals), layout.n_columns), dtype=np.float32
                )
                layout.pack(rows, cells, totals, lengths[span])
                for index, forest in forests:
                    scores[index, span] = forest.score(rows)
                del rows
            del cells
        return scores

    forest_select = evaluate(select, select_counts)
    forest_report = np.stack(
        [
            evaluate(items, lengths)
            for items, lengths in zip(report, report_counts, strict=True)
        ],
        axis=1,
    )
    return [
        LiteralDraw(
            widths[index],
            forest_select[index],
            forest_report[index],
            control(select_counts),
            np.stack([control(lengths) for lengths in report_counts]),
        )
        for index, control in enumerate(controls)
    ]
