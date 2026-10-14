"""Direction finders for linear reading of a concept or function.

Each finder turns the activations of one layer into one unit direction.
Orientation is not decided here: every direction gets a deterministic sign,
and `RepReader` orients each layer with the labels afterwards.
"""

from dataclasses import dataclass
from typing import Any
from typing import Literal
from typing import get_args

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

FloatArray = NDArray[np.floating[Any]]
Method = Literal[
    "pca_diff", "pca_raw", "mean_diff", "logistic", "kmeans", "random"
]
Preprocess = Literal["center", "unit", "none"]

_METHODS = get_args(Method)
_PREPROCESSES = get_args(Preprocess)


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
            `pca_raw` does the same on the rows themselves. `mean_diff` is
            the difference of the two class means and `logistic` the
            coefficients of an L2-regularized logistic regression; both need
            `labels`. The regression runs on rows divided by one global
            scalar, so the penalty does not depend on the activation scale.
            `kmeans` is the centroid difference of a two-cluster K-means and
            `random` a Gaussian draw from `seed`; neither reads labels.
        h: Activations of shape (n, d).
        labels: Binary labels of shape (n,), 1 for the positive pole.
        pairs: Row indices of shape (m, 2); difference i is
            `h[pairs[i, 0]] - h[pairs[i, 1]]`.
        preprocess: How the vectors are prepared before fitting; they are
            the differences for `pca_diff` and the rows otherwise. `center`
            subtracts their mean, as the reference implementation does.
            `unit` first scales each vector to unit length, as the paper
            describes, then centers. `none` does neither. The returned mean
            is the mean of the vectors after any unit scaling.
        seed: Seed for the stochastic finders.

    Returns:
        The unit direction, the mean it was centered on and the raw norm.

    Raises:
        ValueError: If the inputs do not fit the method.
    """
    if method not in _METHODS:
        raise ValueError(f"unsupported method {method!r}")
    if preprocess not in _PREPROCESSES:
        raise ValueError(f"unsupported preprocess {preprocess!r}")
    rows = _as_matrix(h)
    if labels is not None:
        _check_labels(labels, rows.shape[0])
    vectors = _pair_differences(rows, pairs) if method == "pca_diff" else rows
    if preprocess == "unit":
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.where(norms > 0, norms, 1.0)
    mean = vectors.mean(axis=0)
    fitted = vectors if preprocess == "none" else vectors - mean
    if method == "mean_diff":
        direction = _class_difference(fitted, labels)
    elif method == "logistic":
        direction = _logistic_coefficients(fitted, labels)
    elif method == "kmeans":
        direction = _centroid_difference(fitted, seed)
    elif method == "random":
        rng = np.random.default_rng(seed)
        direction = rng.standard_normal(size=(rows.shape[1],))
    else:
        direction = _first_component(fitted)
    return _result(direction, mean)


def _first_component(rows: NDArray[np.float64]) -> NDArray[np.float64]:
    components = np.linalg.svd(rows, full_matrices=False)[2]
    return np.asarray(components[0], dtype=np.float64)


def _as_matrix(h: FloatArray) -> NDArray[np.float64]:
    rows = np.asarray(h, dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError(f"h must have shape (n, d), got {rows.shape}")
    if rows.shape[0] < 2:
        raise ValueError("h needs at least 2 rows")
    if not np.isfinite(rows).all():
        raise ValueError("h must be finite")
    return rows


def _check_labels(labels: NDArray[np.int64], n_rows: int) -> None:
    if labels.shape != (n_rows,):
        raise ValueError(
            f"labels must have shape ({n_rows},), got {labels.shape}"
        )
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be 0 or 1")


def _pair_differences(
    rows: NDArray[np.float64], pairs: NDArray[np.int64] | None
) -> NDArray[np.float64]:
    if pairs is None:
        raise ValueError("pca_diff needs pairs")
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"pairs must have shape (m, 2), got {pairs.shape}")
    if pairs.shape[0] < 2:
        raise ValueError("pca_diff needs at least 2 pairs")
    if pairs.min() < 0 or pairs.max() >= rows.shape[0]:
        raise ValueError("pairs must index the rows of h")
    return np.asarray(rows[pairs[:, 0]] - rows[pairs[:, 1]], dtype=np.float64)


def _positive_mask(labels: NDArray[np.int64] | None) -> NDArray[np.bool_]:
    if labels is None:
        raise ValueError("this method needs labels")
    positive = np.asarray(labels == 1)
    if positive.all() or not positive.any():
        raise ValueError("labels must contain both classes")
    return positive


def _class_difference(
    rows: NDArray[np.float64], labels: NDArray[np.int64] | None
) -> NDArray[np.float64]:
    positive = _positive_mask(labels)
    positive_mean = rows[positive].mean(axis=0)
    negative_mean = rows[~positive].mean(axis=0)
    return np.asarray(positive_mean - negative_mean, dtype=np.float64)


def _logistic_coefficients(
    rows: NDArray[np.float64], labels: NDArray[np.int64] | None
) -> NDArray[np.float64]:
    positive = _positive_mask(labels)
    scale = float(np.sqrt((rows**2).sum(axis=1).mean()))
    if scale == 0.0:
        raise ValueError("the fitted direction has zero norm")
    model = LogisticRegression(max_iter=1000).fit(rows / scale, positive)
    return np.asarray(model.coef_[0], dtype=np.float64) / scale


def _centroid_difference(
    rows: NDArray[np.float64], seed: int
) -> NDArray[np.float64]:
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(rows)
    first, second = kmeans.cluster_centers_
    return np.asarray(second - first, dtype=np.float64)


def _result(
    vector: NDArray[np.float64], mean: NDArray[np.float64]
) -> FitResult:
    raw_norm = float(np.linalg.norm(vector))
    if raw_norm == 0.0:
        raise ValueError("the fitted direction has zero norm")
    unit = vector / raw_norm
    if unit[np.argmax(np.abs(unit))] < 0:
        unit = -unit
    return FitResult(
        direction=unit.astype(np.float32),
        mean=mean.astype(np.float32),
        raw_norm=raw_norm,
    )
