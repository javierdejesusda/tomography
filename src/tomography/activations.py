"""Residual-stream activations at chosen token positions, as numpy arrays."""

from dataclasses import dataclass
from typing import Any
from typing import Literal

import numpy as np
from numpy.typing import NDArray

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
