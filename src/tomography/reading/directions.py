"""Direction finders for linear reading of a concept or function.

Each finder turns the activations of one layer into one unit direction.
Orientation is not decided here: every direction gets a deterministic sign,
and `RepReader` orients each layer with the labels afterwards.
"""

from dataclasses import dataclass
from typing import Any
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]
Method = Literal[
    "pca_diff", "pca_raw", "mean_diff", "logistic", "kmeans", "random"
]
Preprocess = Literal["center", "unit", "none"]


@dataclass(frozen=True, eq=False)
class FitResult:
    """A fitted direction for one layer.

    Attributes:
        direction: Unit-norm vector of shape (d,). Its sign is fixed so that
            the coordinate of largest magnitude is positive.
        mean: Vector of shape (d,) subtracted before projecting a row onto
            the direction. It is the mean of the vectors the estimator saw.
        raw_norm: Norm of the estimated vector before normalization.
    """

    direction: NDArray[np.float32]
    mean: NDArray[np.float32]
    raw_norm: float


def fit_direction(
    method: Method,
    h: FloatArray,
    *,
    labels: NDArray[np.int64] | None = None,
    pairs: NDArray[np.int64] | None = None,
    preprocess: Preprocess = "center",
    seed: int = 0,
) -> FitResult:
    """Fits one direction to the activations of one layer.

    Principal components come from a full SVD, never a randomized solver, so
    the result is deterministic.

    Args:
        method: The direction finder. `pca_diff` takes the first principal
            component of the within-pair differences and needs `pairs`.
        h: Activations of shape (n, d).
        labels: Binary labels of shape (n,), 1 for the positive pole.
        pairs: Row indices of shape (m, 2); difference i is
            `h[pairs[i, 0]] - h[pairs[i, 1]]`.
        preprocess: How differences are prepared before the SVD. `center`
            subtracts their mean, as the reference implementation does.
            `unit` first scales each difference to unit length, as the paper
            describes, then centers. `none` does neither. The returned mean
            is the mean of the differences after any unit scaling.
        seed: Seed for the stochastic finders.

    Returns:
        The unit direction, the mean it was centered on and the raw norm.

    Raises:
        ValueError: If the inputs do not fit the method.
    """
    if method != "pca_diff":
        raise ValueError(f"unsupported method {method!r}")
    if preprocess not in ("center", "unit", "none"):
        raise ValueError(f"unsupported preprocess {preprocess!r}")
    if pairs is None:
        raise ValueError("pca_diff needs pairs")
    rows = np.asarray(h, dtype=np.float64)
    vectors = rows[pairs[:, 0]] - rows[pairs[:, 1]]
    if preprocess == "unit":
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.where(norms > 0, norms, 1.0)
    mean = vectors.mean(axis=0)
    fitted = vectors if preprocess == "none" else vectors - mean
    direction = np.linalg.svd(fitted, full_matrices=False)[2][0]
    return _result(direction, mean)


def _result(
    vector: NDArray[np.float64], mean: NDArray[np.float64]
) -> FitResult:
    raw_norm = float(np.linalg.norm(vector))
    unit = vector / raw_norm
    if unit[np.argmax(np.abs(unit))] < 0:
        unit = -unit
    return FitResult(
        direction=unit.astype(np.float32),
        mean=mean.astype(np.float32),
        raw_norm=raw_norm,
    )
