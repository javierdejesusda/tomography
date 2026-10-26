"""Direction finders for linear reading of a concept or function.

Each finder turns the activations of one layer into one unit direction.
Orientation is not decided here: every direction gets a deterministic sign,
and `RepReader` orients each layer with the labels afterwards.
"""

from dataclasses import dataclass
from typing import Any
from typing import Literal
from typing import assert_never
from typing import get_args

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

FloatArray = NDArray[np.floating[Any]]
Method = Literal[
    "pca_diff",
    "pca_raw",
    "mean_diff",
    "logistic",
    "kmeans",
    "random",
    "whitened_mean_diff",
]
Preprocess = Literal["center", "unit", "none"]

_METHODS = get_args(Method)
_PREPROCESSES = get_args(Preprocess)
_FAINT_TOL = 1e-10


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
    refit_at_zero: bool = False,
    c: float = 1.0,
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
            `whitened_mean_diff` is the whitened (IID) mass-mean probe of
            Marks and Tegmark (2024), `S^-1 (m1 - m0)`, with S the pooled
            within-class covariance shrunk toward a scaled identity by the
            Ledoit-Wolf weight, as the LDA reader computes it; it needs
            `labels`.
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
        refit_at_zero: Only for `logistic`. When the solver stops at its
            start with all-zero coefficients because the gradient there is
            under its tolerance, repeat the fit with a tighter one instead
            of failing with a zero norm. Off by default, so a fit that
            finds no signal is an error.
        c: Only for `logistic`. The inverse L2 strength of the regression
            on the scaled rows, scikit-learn's `C`: smaller values
            regularize more.

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
    max_abs = float(np.abs(vectors).max())
    mean = vectors.mean(axis=0)
    fitted = vectors if preprocess == "none" else vectors - mean
    if (
        method == "mean_diff"
        or method == "logistic"
        or method == "whitened_mean_diff"
    ):
        if labels is None:
            raise ValueError(f"{method} needs labels")
        positive = labels == 1
        if positive.all() or not positive.any():
            raise ValueError("labels must contain both classes")
        if method == "mean_diff":
            direction = _class_difference(fitted, positive)
        elif method == "whitened_mean_diff":
            direction = _whitened_class_difference(fitted, positive)
        else:
            direction = _logistic_coefficients(
                fitted, positive, refit_at_zero, c
            )
    elif method == "kmeans":
        direction = _centroid_difference(fitted, seed)
    elif method == "random":
        direction = _random_direction(rows.shape[1], seed)
    elif method == "pca_diff" or method == "pca_raw":
        direction = _first_component(fitted, max_abs)
    else:
        assert_never(method)
    return _result(direction, mean)


def _first_component(
    rows: NDArray[np.float64], max_abs: float
) -> NDArray[np.float64]:
    _, singular_values, components = np.linalg.svd(rows, full_matrices=False)
    bound = np.finfo(np.float64).eps * max(rows.shape) * max_abs
    if singular_values[0] <= bound:
        raise ValueError("the fitted direction has zero norm")
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
    return rows[pairs[:, 0]] - rows[pairs[:, 1]]


def _class_difference(
    rows: NDArray[np.float64], positive: NDArray[np.bool_]
) -> NDArray[np.float64]:
    return rows[positive].mean(axis=0) - rows[~positive].mean(axis=0)


def _whitened_class_difference(
    rows: NDArray[np.float64], positive: NDArray[np.bool_]
) -> NDArray[np.float64]:
    # Imported here because `probes` imports this module through `reader`.
    from tomography.reading.probes import _shrunk_mass_mean

    return _shrunk_mass_mean(rows, positive, None)


def _logistic_coefficients(
    rows: NDArray[np.float64],
    positive: NDArray[np.bool_],
    refit_at_zero: bool,
    c: float,
) -> NDArray[np.float64]:
    scale = float(np.linalg.norm(rows) / np.sqrt(rows.shape[0]))
    if scale == 0.0:
        raise ValueError("the fitted direction has zero norm")
    scaled = rows / scale
    model = LogisticRegression(C=c, max_iter=1000).fit(scaled, positive)
    if refit_at_zero and not model.coef_.any():
        model = LogisticRegression(C=c, max_iter=1000, tol=_FAINT_TOL).fit(
            scaled, positive
        )
    return np.asarray(model.coef_[0], dtype=np.float64) / scale


def _centroid_difference(
    rows: NDArray[np.float64], seed: int
) -> NDArray[np.float64]:
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(rows)
    first, second = kmeans.cluster_centers_
    return np.asarray(second - first, dtype=np.float64)


def _random_direction(d: int, seed: int) -> NDArray[np.float64]:
    return np.random.default_rng(seed).standard_normal(size=(d,))


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
