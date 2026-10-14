"""Layer-wise analyses of activations: PCA variance and t-SNE maps."""

from typing import get_args

import numpy as np
from numpy.typing import NDArray

from tomography.reading.directions import FloatArray
from tomography.reading.directions import Preprocess

_PREPROCESSES = get_args(Preprocess)


def pca_explained_variance(
    h: FloatArray,
    *,
    pairs: NDArray[np.int64] | None = None,
    n_components: int = 10,
    preprocess: Preprocess = "center",
) -> NDArray[np.float64]:
    """Computes the explained-variance ratios of the leading components.

    Each layer is decomposed with a full SVD, so the result is deterministic.

    Args:
        h: Activations of shape (K, n, d): layer, row, feature.
        pairs: Row indices of shape (m, 2). If given, the decomposed rows are
            the differences `h[:, pairs[:, 0]] - h[:, pairs[:, 1]]`;
            otherwise they are the activations.
        n_components: Number of leading components to report.
        preprocess: How the rows are prepared, as in `fit_direction`:
            `center` subtracts their mean, `unit` first scales each row to
            unit length, `none` does neither.

    Returns:
        Array of shape (K, n_components): squared singular values over their
        total, non-increasing along the last axis.

    Raises:
        ValueError: If `h` is not 3-D or not finite, `pairs` is malformed or
            out of range, `preprocess` is unknown, `n_components` is below 1
            or above `min(rows, d)`, or the rows have no variance.
    """
    if preprocess not in _PREPROCESSES:
        raise ValueError(f"unsupported preprocess {preprocess!r}")
    layers = np.asarray(h, dtype=np.float64)
    if layers.ndim != 3:
        raise ValueError(f"h must have shape (K, n, d), got {layers.shape}")
    if not np.isfinite(layers).all():
        raise ValueError("h must be finite")
    if pairs is not None:
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError(f"pairs must have shape (m, 2), got {pairs.shape}")
        if pairs.size and (pairs.min() < 0 or pairs.max() >= layers.shape[1]):
            raise ValueError("pairs must index the rows of h")
        layers = layers[:, pairs[:, 0]] - layers[:, pairs[:, 1]]
    limit = min(layers.shape[1:])
    if not 1 <= n_components <= limit:
        raise ValueError(
            f"n_components must be in 1..{limit}, got {n_components}"
        )
    ratios = np.empty((layers.shape[0], n_components))
    for k, rows in enumerate(layers):
        if preprocess == "unit":
            norms = np.linalg.norm(rows, axis=1, keepdims=True)
            rows = rows / np.where(norms > 0, norms, 1.0)
        if preprocess != "none":
            rows = rows - rows.mean(axis=0)
        squared = np.linalg.svd(rows, compute_uv=False) ** 2
        total = float(squared.sum())
        if total <= 0.0:
            raise ValueError(f"layer {k} has no variance")
        ratios[k] = squared[:n_components] / total
    return ratios
