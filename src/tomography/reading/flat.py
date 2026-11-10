"""A linear reader over the flattened, position-aligned cell matrix.

The cell readers of `probes` share one head per layer across the tokens, so
they need no alignment. The flat reader instead gives every (layer, slot) of
a text its own weights: a text's input is the concatenation of its
standardized cells z(t, l) at fixed slots, and the reader is an
L2-regularized logistic regression on that vector. The cells are scaled as
in `probes._fit_block`, with `probes._scaled_cells` on the training cells
of each layer.

Slots. A text's cells come in segments, each with an integer code; the
segments are laid out in code order, and segment s gets `widths[s]` slots,
the most tokens any training text has in it. Within a segment the tokens
are aligned from its end: its last token takes the segment's last slot, the
token before it the slot before, and so on. A shorter segment leaves its
first slots empty, and an empty slot holds zero, the training mean once
standardized, so it adds nothing to the score. A longer segment, which only
evaluation texts can have, loses its first tokens. A segment that no
training text has gets no slots, so the reader drops all of its cells and
has no weights for them.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import torch

from tomography.activations import Activations
from tomography.reading.probes import _fit_inputs
from tomography.reading.probes import _minimize
from tomography.reading.probes import _penalized_loss
from tomography.reading.probes import _scaled_cells
from tomography.reading.probes import _scored_stimulus


@dataclass(frozen=True)
class FlatReader:
    """A linear reader of the aligned cells of a text at fixed slots.

    Slot s of layer k reads a cell x as `values[k, s] @ x - shifts[k, s]`,
    and a text scores the sum of its cells' readings plus `bias`; an empty
    slot reads nothing. The slots and the alignment rule are those of the
    module docstring. The parameters are in activation units. The reader
    needs each row's segment code, so `score` takes them next to the rows,
    unlike `probes.Reader`.

    Attributes:
        layers: Layer indices, in the order of the first axis of the input.
        widths: The number of slots of each segment, shape (n_segments,),
            indexed by segment code; 0 for a segment no training text has.
        values: Weights of shape (K, W, d), W the sum of `widths`, the
            slots of the segments in code order.
        shifts: Shifts of shape (K, W).
        bias: The bias.
    """

    layers: tuple[int, ...]
    widths: NDArray[np.int64]
    values: NDArray[np.float64]
    shifts: NDArray[np.float64]
    bias: float

    @property
    def method(self) -> str:
        """`flat`."""
        return "flat"

    def score(
        self, acts: Activations, segments: NDArray[np.int8]
    ) -> NDArray[np.float32]:
        """Scores every stimulus from its cells at their slots.

        Args:
            acts: Token rows of the layers in `layers`, stimulus-major and
                in token order within a stimulus.
            segments: The segment code of each row.

        Returns:
            Scores of shape (1, n_stimuli).

        Raises:
            ValueError: If `acts` holds other layers or another width, a
                stimulus has no rows, or `segments` does not give one code in
                [0, n_segments) per row.
        """
        _scored_stimulus(acts, self.layers, self.values.shape[2])
        slots = _slots(acts, segments, self.widths)
        kept = slots >= 0
        stimulus = acts.row_stimulus[kept]
        slot = slots[kept]
        scores = np.full(acts.n_stimuli, self.bias)
        for k, layer_rows in enumerate(acts.data):
            rows = layer_rows[kept].astype(np.float64)
            cells = np.einsum("rd,rd->r", rows, self.values[k, slot])
            cells -= self.shifts[k, slot]
            scores += np.bincount(stimulus, cells, minlength=acts.n_stimuli)
        return scores[None].astype(np.float32)

    def dropped(
        self, acts: Activations, segments: NDArray[np.int8]
    ) -> NDArray[np.int64]:
        """Counts the cells of each stimulus and segment that have no slot.

        Args:
            acts: Token rows, as for `score`.
            segments: The segment code of each row.

        Returns:
            The number of dropped cells (tokens times layers), shape
            (n_stimuli, n_segments): the first tokens of a segment longer
            than its width, and every token of a segment with no slots.

        Raises:
            ValueError: If `segments` does not give one code in
                [0, n_segments) per row.
        """
        n_segments = self.widths.shape[0]
        lost = _slots(acts, segments, self.widths) < 0
        key = acts.row_stimulus * n_segments + np.asarray(segments, np.int64)
        counts = np.bincount(key[lost], minlength=acts.n_stimuli * n_segments)
        return counts.reshape(acts.n_stimuli, n_segments) * len(acts.layers)


def fit_flat(
    acts: Activations,
    *,
    segments: NDArray[np.int8],
    labels: NDArray[np.int64],
    n_segments: int,
    c: float = 1.0,
    tol: float = 1e-4,
    max_iter: int = 1000,
) -> FlatReader:
    """Fits an L2-regularized logistic regression on the flat cell matrix.

    The slots are set on the training texts (see the module docstring), so
    none of their cells is dropped. The loss is that of the cell readers of
    `probes`: the mean logistic loss of the scores plus (1 / (2 c n)) times
    the squared norm of the weights on the standardized cells, with the
    bias not penalized, so `c` means scikit-learn's `C` and the fit is
    `LogisticRegression(C=c)` on the flat matrix.

    The fit runs in dual form. With n texts and p = K W d inputs, p in the
    millions and n about 128, the optimal weights lie in the span of the n
    flat rows (the gradient of the loss is a combination of the rows, and
    the bias, which is not penalized, does not change that). The Gram
    matrix G of the rows is accumulated one layer at a time, so the n by p
    matrix is never held: one layer's rows take n W d floats. With
    G = U S U^T, the rows have the coordinates F = U S^(1/2) in an
    orthonormal basis of their span, in which the problem is the same
    logistic regression on n inputs with the same penalty, as the squared
    norm of weights in the span equals that of their coordinates. It is
    minimized with the L-BFGS-B of the cell fits (`probes._minimize`):
    float64, from zeros, `tol` as the gradient tolerance on the
    coordinates, a `ConvergenceWarning` when it stops before converging and
    a `StalledFitWarning` when its loss never leaves its zero start's.
    Eigenvalues of G at most `n` machine epsilons of the largest are
    rounding (G loses the square of the rows' precision), so their
    directions are left out; the weights are then the rows times
    U S^(-1/2) times the fitted coordinates, a second pass over the layers.
    A primal fit would need the n by p matrix, gigabytes at the real
    shapes, and p-dimensional L-BFGS steps; the dual one costs one n by n
    Gram matrix (n^2 p multiply-adds) and then works on n + 1 parameters.
    Nothing is random, so a refit on one machine is bitwise identical.

    Args:
        acts: Token rows of the layers to read, stimulus-major and in token
            order within a stimulus.
        segments: The segment code of each row, in [0, n_segments).
        labels: Binary labels of shape (n_stimuli,), 1 for the positive
            pole.
        n_segments: The number of segment codes, so a segment no training
            text has still gets its width, 0.
        c: The inverse L2 strength, scikit-learn's `C`: smaller values
            regularize more.
        tol: Gradient tolerance of L-BFGS.
        max_iter: Iteration cap of L-BFGS.

    Returns:
        The reader.

    Raises:
        ValueError: If the labels do not have one binary entry per stimulus
            with both classes, `c` is not positive, a stimulus has no rows,
            `segments` does not give one code in [0, n_segments) per row,
            `acts` is not finite, or a layer's rows are all equal.
    """
    stimulus, targets = _fit_inputs(acts, labels, c)
    key = acts.row_stimulus * n_segments + _codes(acts, segments, n_segments)
    runs = np.bincount(key, minlength=acts.n_stimuli * n_segments)
    widths = runs.reshape(acts.n_stimuli, n_segments).max(axis=0)
    values, shifts, bias = _fit_weights(
        acts,
        stimulus.numpy(),
        _slots(acts, segments, widths),
        widths,
        targets,
        c,
        tol,
        max_iter,
    )
    return FlatReader(
        layers=tuple(int(layer) for layer in acts.layers),
        widths=widths,
        values=values,
        shifts=shifts,
        bias=bias,
    )


def _fit_weights(
    acts: Activations,
    stimulus: NDArray[np.int64],
    slots: NDArray[np.int64],
    widths: NDArray[np.int64],
    targets: torch.Tensor,
    c: float,
    tol: float,
    max_iter: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    # Fits the dual form of `fit_flat` and returns the weights and shifts
    # in activation units and the bias. Kept apart from `fit_flat` so the
    # warnings of `_minimize` reach the caller of `fit_flat`.
    n = acts.n_stimuli
    size = int(widths.sum())

    def flat(k: int) -> tuple[NDArray[np.float64], float, NDArray[np.float64]]:
        # Layer k's flat rows, (n, W d), with its center and scale.
        scaled, centers, scales = _scaled_cells(
            acts.data[k : k + 1], acts.layers[k : k + 1]
        )
        rows = np.zeros((n, size, acts.data.shape[2]))
        rows[stimulus, slots] = scaled[0].numpy()
        return rows.reshape(n, -1), float(scales[0]), centers[0]

    gram = np.zeros((n, n))
    for k in range(len(acts.layers)):
        rows = flat(k)[0]
        gram += rows @ rows.T
        del rows
    eigenvalues, basis = np.linalg.eigh(gram)
    kept = eigenvalues > eigenvalues.max() * n * np.finfo(np.float64).eps
    roots = np.sqrt(eigenvalues[kept])
    features = torch.from_numpy(basis[:, kept] * roots)

    def loss(params: torch.Tensor) -> torch.Tensor:
        scores = features @ params[:-1] + params[-1]
        return _penalized_loss(scores, targets, params[:-1].square().sum(), c)

    solution = _minimize(loss, features.shape[1] + 1, tol, max_iter, "flat")
    duals = basis[:, kept] @ (solution[:-1] / roots)
    values = np.empty((len(acts.layers), size, acts.data.shape[2]))
    shifts = np.empty((len(acts.layers), size))
    for k in range(len(acts.layers)):
        rows, scale, center = flat(k)
        values[k] = (duals @ rows).reshape(size, -1) / scale
        shifts[k] = values[k] @ center
    return values, shifts, float(solution[-1])


def _codes(
    acts: Activations, segments: NDArray[np.int8], n_segments: int
) -> NDArray[np.int64]:
    # Checks that `segments` gives each row a code in [0, n_segments).
    codes = np.asarray(segments, dtype=np.int64)
    n_rows = acts.row_stimulus.shape[0]
    if codes.shape != (n_rows,):
        raise ValueError(
            f"segments must have one code per row ({n_rows}), got "
            f"{codes.shape[0] if codes.ndim == 1 else codes.shape}"
        )
    outside = (codes < 0) | (codes >= n_segments)
    if outside.any():
        raise ValueError(
            f"segment codes must be in [0, {n_segments}), got "
            f"{int(codes[outside][0])}"
        )
    return codes


def _slots(
    acts: Activations, segments: NDArray[np.int8], widths: NDArray[np.int64]
) -> NDArray[np.int64]:
    # The slot of each row by the rule of the module docstring, or -1 for a
    # row that has none.
    n_segments = widths.shape[0]
    codes = _codes(acts, segments, n_segments)
    key = acts.row_stimulus * n_segments + codes
    # Rows are in token order within a stimulus, and a stable sort keeps it.
    order = np.argsort(key, kind="stable")
    runs = np.bincount(key, minlength=acts.n_stimuli * n_segments)
    starts = np.cumsum(runs) - runs
    rank = np.empty_like(key)
    rank[order] = np.arange(key.shape[0]) - starts[key[order]]
    from_end = runs[key] - 1 - rank
    ends = np.cumsum(widths) - 1
    return np.where(from_end < widths[codes], ends[codes] - from_end, -1)
