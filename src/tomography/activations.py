"""Residual-stream activations at chosen token positions, as numpy arrays."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from typing import Literal

import numpy as np
from numpy.typing import NDArray
import torch

from tomography.hooks import capture_residuals
from tomography.model import BosMode
from tomography.model import EncodedBatch
from tomography.model import LoadedModel

PositionStrategy = Literal["last", "response", "concept", "mean"]
Span = tuple[int, int]


@dataclass(frozen=True, eq=False)
class Activations:
    """Activations of a set of stimuli, one row per read token position.

    Rows are ordered stimulus-major, then by token. `last`, `concept` and
    `mean` give one row per stimulus; `response` gives one row per token of
    the marked span.

    Attributes:
        data: `(n_layers_sel, n_rows, d_model)` float32, layer-major.
        layers: Layer indices in the order of `data` axis 0.
        position: The strategy that produced the rows.
        row_stimulus: `(n_rows,)` index into the input texts.
        row_rank: `(n_rows,)` 0-based rank among the rows of the same
            stimulus.
        row_token: `(n_rows,)` token index in the model input (BOS is 0
            when prepended); -1 for `mean`.
        row_token_id: `(n_rows,)` vocabulary id of that token; -1 for
            `mean`.
        n_stimuli: Number of input texts.
    """

    data: NDArray[np.float32]
    layers: tuple[int, ...]
    position: PositionStrategy
    row_stimulus: NDArray[np.int64]
    row_rank: NDArray[np.int64]
    row_token: NDArray[np.int64]
    row_token_id: NDArray[np.int64]
    n_stimuli: int

    def row_values(self, per_stimulus: NDArray[Any]) -> NDArray[Any]:
        """Repeats per-stimulus values for each row of the stimulus.

        Args:
            per_stimulus: An array whose first axis has `n_stimuli` entries,
                for example the labels of a `StimulusSet`.

        Returns:
            An array with `n_rows` entries on the first axis.

        Raises:
            ValueError: If the first axis is not `n_stimuli` long.
        """
        values = np.asarray(per_stimulus)
        if values.shape[0] != self.n_stimuli:
            raise ValueError(
                f"expected {self.n_stimuli} per-stimulus values (n_stimuli), "
                f"got {values.shape[0]}"
            )
        return values[self.row_stimulus]

    def row_groups(
        self,
        groups: NDArray[np.int64],
        *,
        ragged: Literal["error", "truncate"] = "error",
    ) -> NDArray[np.int64]:
        """Maps stimulus groups to row groups by aligning members by rank.

        Pairs are groups with k = 2. For one-row positions the result equals
        `groups`. For `response` rows the rank-r token of each member forms
        one row group, so function-type response tokens reach the reader as
        aligned groups.

        Args:
            groups: `(g, k)` stimulus indices, padded with -1 (ignored).
            ragged: What to do when members have unequal row counts:
                `error` raises, `truncate` keeps the minimum count.

        Returns:
            `(g_rows, k)` int64 row indices, group-major then by rank, with
            the -1 padding kept.

        Raises:
            ValueError: If `groups` is not 2D, a group has no members or an
                index outside `n_stimuli`, `ragged` is unknown, or row counts
                are unequal under `error`.
        """
        if ragged not in ("error", "truncate"):
            raise ValueError(
                f"ragged must be 'error' or 'truncate', got {ragged!r}"
            )
        members = np.asarray(groups)
        if members.ndim != 2:
            raise ValueError(f"groups must be 2D, got {members.ndim}D")
        counts = np.bincount(self.row_stimulus, minlength=self.n_stimuli)
        starts = np.cumsum(counts) - counts
        blocks = [np.empty((0, members.shape[1]), dtype=np.int64)]
        for index, group in enumerate(members):
            valid = group >= 0
            ids = group[valid]
            if ids.size == 0:
                raise ValueError(f"group {index} has no members")
            if ids.max() >= self.n_stimuli:
                raise ValueError(
                    f"group {index} has an index outside n_stimuli="
                    f"{self.n_stimuli}"
                )
            sizes = counts[ids]
            if ragged == "error" and np.unique(sizes).size > 1:
                raise ValueError(
                    f"group {index} has members with unequal row counts "
                    f"{sizes.tolist()}"
                )
            n_rows = int(sizes.min())
            block = np.full((n_rows, members.shape[1]), -1, dtype=np.int64)
            block[:, valid] = starts[ids] + np.arange(n_rows)[:, None]
            blocks.append(block)
        return np.concatenate(blocks, axis=0)


def _overlapping_tokens(
    offsets: tuple[tuple[int, int], ...], length: int, span: Span
) -> list[int]:
    """Returns the tokens whose character offsets overlap `span`.

    A token is in the span iff `tok_start < end and tok_end > start and
    tok_end > tok_start`. Containment would be wrong: GPT-2 and Qwen offsets
    include the leading space, and BOS and padding have empty offsets.
    """
    start, end = span
    return [
        token
        for token in range(length)
        if offsets[token][0] < end
        and offsets[token][1] > start
        and offsets[token][1] > offsets[token][0]
    ]


def _select_tokens(
    position: PositionStrategy,
    index: int,
    batch: EncodedBatch,
    row: int,
    spans: Sequence[Span | None] | None,
) -> list[int]:
    """Returns the token indices read for stimulus `index` (batch `row`)."""
    length = batch.lengths[row]
    if position == "last":
        return [length - 1]
    assert spans is not None
    span = spans[index]
    if span is None:
        raise ValueError(f"stimulus {index} has no span for {position!r}")
    tokens = _overlapping_tokens(batch.offsets[row], length, span)
    if not tokens:
        raise ValueError(f"stimulus {index}: the span {span} overlaps no token")
    return tokens[-1:] if position == "concept" else tokens


def _check_arguments(
    n_texts: int,
    position: PositionStrategy,
    spans: Sequence[Span | None] | None,
    batch_size: int,
) -> None:
    if n_texts == 0:
        raise ValueError("texts is empty")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if position not in ("last", "response", "concept", "mean"):
        raise ValueError(f"unknown position {position!r}")
    if position == "last":
        return
    name = "concept_spans" if position == "concept" else "response_spans"
    if spans is None:
        raise ValueError(f"position {position!r} needs {name}")
    if len(spans) != n_texts:
        raise ValueError(f"{name} has {len(spans)} entries for {n_texts} texts")


def _collect_batch(
    loaded: LoadedModel,
    texts: Sequence[str],
    first: int,
    layers: Sequence[int],
    position: PositionStrategy,
    spans: Sequence[Span | None] | None,
    bos: BosMode | None,
) -> tuple[torch.Tensor, list[tuple[int, int, int, int]]]:
    """Runs one batch; returns `(K, rows, d)` data and per-row metadata.

    The metadata tuples are `(stimulus, rank, token, token_id)`.
    """
    batch = loaded.encode(texts, bos=bos, padding_side="right")
    selections = [
        _select_tokens(position, first + row, batch, row, spans)
        for row in range(len(texts))
    ]
    rows = torch.tensor(
        [row for row, tokens in enumerate(selections) for _ in tokens],
        device=loaded.device,
    )
    cols = torch.tensor(
        [token for tokens in selections for token in tokens],
        device=loaded.device,
    )
    with torch.inference_mode(), capture_residuals(loaded, layers) as captured:
        loaded.decoder(
            input_ids=batch.input_ids.to(loaded.device),
            attention_mask=batch.attention_mask.to(loaded.device),
            position_ids=batch.position_ids.to(loaded.device),
            use_cache=False,
        )
        gathered = torch.stack(
            [captured[layer][rows, cols].float() for layer in layers]
        )
        if position == "mean":
            sums = torch.zeros(
                len(layers),
                len(texts),
                gathered.shape[-1],
                device=gathered.device,
            )
            sums.index_add_(1, rows, gathered)
            counts = torch.bincount(rows, minlength=len(texts))
            gathered = sums / counts.view(1, -1, 1)
    metadata: list[tuple[int, int, int, int]] = []
    for row, tokens in enumerate(selections):
        if position == "mean":
            metadata.append((first + row, 0, -1, -1))
            continue
        ids = batch.input_ids[row, tokens].tolist()
        for rank, token in enumerate(tokens):
            metadata.append((first + row, rank, token, ids[rank]))
    return gathered.cpu(), metadata


def collect_activations(
    loaded: LoadedModel,
    texts: Sequence[str],
    *,
    layers: Sequence[int] | None = None,
    position: PositionStrategy = "last",
    response_spans: Sequence[Span | None] | None = None,
    concept_spans: Sequence[Span | None] | None = None,
    bos: BosMode | None = None,
    batch_size: int = 8,
) -> Activations:
    """Collects residual-stream activations for a list of texts.

    Texts are right padded and run through the decoder without the LM head
    under `torch.inference_mode`; batched results equal per-example results
    within rtol 1e-4 and atol 1e-3 (SDPA masking noise).

    Args:
        loaded: The wrapped model.
        texts: The stimuli.
        layers: Layer indices in 0..L; None means all of them.
        position: `last` reads the last real token, `response` every token
            overlapping `response_spans[i]`, `concept` the last token
            overlapping `concept_spans[i]`, `mean` one row averaging the
            response tokens.
        response_spans: Half-open character ranges, required for `response`
            and `mean`.
        concept_spans: Half-open character ranges, required for `concept`.
        bos: BOS policy for `LoadedModel.encode`; None uses the model's
            default.
        batch_size: Texts per forward pass.

    Returns:
        The activations, rows stimulus-major then in token order.

    Raises:
        ValueError: If the arguments are inconsistent, a layer is outside
            0..L, a span is missing, or a span overlaps no token (the error
            names the stimulus).
    """
    spans = concept_spans if position == "concept" else response_spans
    _check_arguments(len(texts), position, spans, batch_size)
    selected = list(range(loaded.n_layers + 1) if layers is None else layers)
    if not selected:
        raise ValueError("layers is empty")
    chunks: list[torch.Tensor] = []
    metadata: list[tuple[int, int, int, int]] = []
    for first in range(0, len(texts), batch_size):
        chunk, rows = _collect_batch(
            loaded,
            texts[first : first + batch_size],
            first,
            selected,
            position,
            spans,
            bos,
        )
        chunks.append(chunk)
        metadata += rows
    columns = np.array(metadata, dtype=np.int64).T
    return Activations(
        data=torch.cat(chunks, dim=1).numpy(),
        layers=tuple(selected),
        position=position,
        row_stimulus=columns[0],
        row_rank=columns[1],
        row_token=columns[2],
        row_token_id=columns[3],
        n_stimuli=len(texts),
    )
