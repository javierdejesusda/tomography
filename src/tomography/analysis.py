"""Layer-wise analyses: PCA variance, t-SNE maps, direction geometry, lens."""

from collections.abc import Sequence
from typing import get_args

import numpy as np
from numpy.typing import NDArray
from scipy.stats import t as student_t
from sklearn.manifold import TSNE

from tomography.reading.directions import FloatArray
from tomography.reading.directions import Preprocess

_PREPROCESSES = get_args(Preprocess)


def _layers(h: FloatArray) -> NDArray[np.float64]:
    layers = np.asarray(h, dtype=np.float64)
    if layers.ndim != 3:
        raise ValueError(f"h must have shape (K, n, d), got {layers.shape}")
    if not np.isfinite(layers).all():
        raise ValueError("h must be finite")
    return layers


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
    layers = _layers(h)
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


def tsne_embed(
    h: FloatArray, *, perplexity: float = 30.0, seed: int = 0
) -> NDArray[np.float32]:
    """Embeds the rows of every layer in 2-D with t-SNE.

    Layers are embedded independently. With the PCA initialization the
    embedding is deterministic for a given input, and probes on clustered
    data gave the same embedding for different seeds, so do not rely on
    `seed` to produce a different layout.

    Args:
        h: Activations of shape (K, n, d): layer, row, feature.
        perplexity: The t-SNE perplexity; it must be below n.
        seed: The `random_state` passed to scikit-learn.

    Returns:
        Array of shape (K, n, 2).

    Raises:
        ValueError: If `h` is not 3-D or not finite, or `perplexity` is not
            below the number of rows.
    """
    layers = _layers(h)
    if perplexity >= layers.shape[1]:
        raise ValueError(
            f"perplexity must be below the {layers.shape[1]} rows, "
            f"got {perplexity}"
        )
    tsne = TSNE(
        n_components=2, init="pca", perplexity=perplexity, random_state=seed
    )
    return np.stack([tsne.fit_transform(rows) for rows in layers]).astype(
        np.float32
    )


def _unit_runs(vectors: FloatArray) -> NDArray[np.float64]:
    runs = np.asarray(vectors, dtype=np.float64)
    if runs.ndim != 3:
        raise ValueError(f"vectors must have shape (N, K, d), got {runs.shape}")
    if not np.isfinite(runs).all():
        raise ValueError("vectors must be finite")
    norms = np.linalg.norm(runs, axis=-1, keepdims=True)
    if (norms == 0.0).any():
        raise ValueError("vectors must not contain a zero vector")
    return np.asarray(runs / norms)


def layer_cosines(vectors: FloatArray) -> NDArray[np.float64]:
    """Computes the cosine between the directions of every pair of layers.

    Args:
        vectors: Directions of shape (N, K, d): run, layer, width. Each one
            is normalized first.

    Returns:
        Array of shape (K, K): the cosine between layers i and j within each
        run, averaged over the N runs.

    Raises:
        ValueError: If `vectors` is not 3-D, not finite or has a zero vector.
    """
    unit = _unit_runs(vectors)
    return np.asarray(np.einsum("nid,njd->ij", unit, unit) / unit.shape[0])


def seed_agreement(vectors: FloatArray) -> NDArray[np.float64]:
    """Computes how well each run agrees with the others at every layer.

    The mean over runs equals the mean pairwise cosine. The rows are not
    independent, because every pair of runs enters two of them, so a band for
    that mean resamples `agreement_pseudo_values` instead.

    Args:
        vectors: Directions of shape (N, K, d): run, layer, width. Each one
            is normalized first.

    Returns:
        Array of shape (N, K): for each run and layer, the mean cosine to
        the other N - 1 runs.

    Raises:
        ValueError: If `vectors` is not 3-D, not finite or has a zero vector,
            or there are fewer than two runs.
    """
    unit = _unit_runs(vectors)
    n_runs = unit.shape[0]
    if n_runs < 2:
        raise ValueError("seed agreement needs at least two runs")
    to_total = np.einsum("nkd,kd->nk", unit, unit.sum(axis=0))
    to_self = np.einsum("nkd,nkd->nk", unit, unit)
    return np.asarray((to_total - to_self) / (n_runs - 1))


def agreement_pseudo_values(vectors: FloatArray) -> NDArray[np.float64]:
    """Computes the jackknife pseudo-values of the mean pairwise cosine.

    The mean pairwise cosine is a U-statistic of degree two. The rows of
    `seed_agreement` share pairs, so resampling them gives a band that is too
    narrow. The pseudo-value of run i is `N * U - (N - 1) * U_without_i`; the
    pseudo-values average to the mean pairwise cosine and are the units to
    resample for its band.

    Args:
        vectors: Directions of shape (N, K, d): run, layer, width. Each one
            is normalized first.

    Returns:
        Array of shape (N, K): for each run and layer, the pseudo-value.

    Raises:
        ValueError: If `vectors` is not 3-D, not finite or has a zero vector,
            or there are fewer than three runs.
    """
    unit = _unit_runs(vectors)
    n_runs = unit.shape[0]
    if n_runs < 3:
        raise ValueError("pseudo-values need at least three runs")
    rows = seed_agreement(unit)
    mean = rows.mean(axis=0)
    return np.asarray(mean + 2.0 * (n_runs - 1) / (n_runs - 2) * (rows - mean))


def agreement_band(
    vectors: FloatArray, *, alpha: float = 0.05
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Computes a t-interval of the mean pairwise cosine at every layer.

    The interval is `U +- t * SE` over the jackknife pseudo-values: U is
    their mean, SE their standard deviation over the square root of the
    number of runs, and t the `1 - alpha / 2` quantile of Student's t with
    `N - 1` degrees of freedom. A bootstrap of so few runs covers less than
    its level. The bounds are clipped to [-1, 1], the range of a cosine.

    Args:
        vectors: Directions of shape (N, K, d): run, layer, width. Each one
            is normalized first.
        alpha: One minus the coverage; 0.05 gives a 95% band.

    Returns:
        The lower and upper bounds, each of shape (K,).

    Raises:
        ValueError: If `vectors` is not 3-D, not finite or has a zero vector,
            there are fewer than three runs, or `alpha` is not in (0, 1).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    pseudo = agreement_pseudo_values(vectors)
    n_runs = pseudo.shape[0]
    mean = pseudo.mean(axis=0)
    half = (
        student_t.ppf(1.0 - alpha / 2.0, n_runs - 1)
        * pseudo.std(axis=0, ddof=1)
        / np.sqrt(n_runs)
    )
    return (
        np.asarray(np.clip(mean - half, -1.0, 1.0)),
        np.asarray(np.clip(mean + half, -1.0, 1.0)),
    )


def mean_direction(vectors: FloatArray) -> NDArray[np.float64]:
    """Averages the unit directions of the runs at every layer.

    Args:
        vectors: Directions of shape (N, K, d): run, layer, width. Each one
            is normalized first.

    Returns:
        Array of shape (K, d): the normalized mean of the unit directions.

    Raises:
        ValueError: If `vectors` is not 3-D, not finite or has a zero vector,
            or the runs cancel at a layer (mean norm below 1e-12).
    """
    mean = _unit_runs(vectors).mean(axis=0)
    norms = np.linalg.norm(mean, axis=-1, keepdims=True)
    if (norms < 1e-12).any():
        raise ValueError("the runs cancel: their mean direction has no norm")
    return np.asarray(mean / norms)


def lens_logits(
    vectors: FloatArray,
    unembedding: FloatArray,
    *,
    scale: FloatArray | None = None,
    center: bool = False,
) -> NDArray[np.float32]:
    """Reads directions through the final norm and the unembedding.

    The direction is centered over its width when `center` (LayerNorm),
    multiplied by `scale` (the norm weight) when given, and multiplied by the
    unembedding, in float32. The bias of the norm and its division by the
    scale of the residual are left out: the bias does not depend on the
    direction and the division is one positive scalar, which `standardize`
    removes.

    Args:
        vectors: Directions of shape (..., d).
        unembedding: Output embedding of shape (V, d).
        scale: The norm weight of shape (d,), or None for none.
        center: Whether to subtract the mean over the width first.

    Returns:
        Array of shape (..., V) in float32.

    Raises:
        ValueError: If the width of `vectors` or `scale` differs from the
            width of `unembedding`.
    """
    direction = np.asarray(vectors, dtype=np.float32)
    matrix = np.asarray(unembedding, dtype=np.float32)
    if direction.shape[-1] != matrix.shape[1]:
        raise ValueError(
            f"vectors have width {direction.shape[-1]} but the unembedding "
            f"has width {matrix.shape[1]}"
        )
    if center:
        direction = direction - direction.mean(axis=-1, keepdims=True)
    if scale is not None:
        weight = np.asarray(scale, dtype=np.float32)
        if weight.shape != (matrix.shape[1],):
            raise ValueError(
                f"scale must have shape ({matrix.shape[1]},), "
                f"got {weight.shape}"
            )
        direction = direction * weight
    return np.asarray(direction @ matrix.T, dtype=np.float32)


def standardize(logits: FloatArray) -> NDArray[np.float64]:
    """Computes the z-score of each row over the vocabulary.

    Args:
        logits: Array of shape (..., V).

    Returns:
        Array of the same shape in float64 with mean 0 and standard
        deviation 1 along the last axis.

    Raises:
        ValueError: If a row is constant.
    """
    values = np.asarray(logits, dtype=np.float64)
    if (values.max(axis=-1) == values.min(axis=-1)).any():
        raise ValueError("a constant row has no z-score")
    centered = values - values.mean(axis=-1, keepdims=True)
    return np.asarray(centered / values.std(axis=-1, keepdims=True))


def pole_separation(
    logits: FloatArray, positive: Sequence[int], negative: Sequence[int]
) -> NDArray[np.float64]:
    """Measures how far apart two sets of tokens sit in a lens reading.

    Args:
        logits: Array of shape (..., V).
        positive: Token ids of the positive pole.
        negative: Token ids of the negative pole.

    Returns:
        Array of shape (...): the mean z-score of the positive ids minus the
        mean z-score of the negative ids, in vocabulary standard deviations.
        It is 0 for a direction unrelated to the poles.

    Raises:
        ValueError: If a pole is empty, an id is outside the vocabulary or a
            row is constant.
    """
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("each pole needs at least one token id")
    z = standardize(logits)
    size = z.shape[-1]
    ids = np.asarray([*positive, *negative])
    if ids.min() < 0 or ids.max() >= size:
        raise ValueError(f"pole ids must be in 0..{size - 1}")
    return np.asarray(
        z[..., list(positive)].mean(axis=-1)
        - z[..., list(negative)].mean(axis=-1)
    )


def top_tokens(
    logits: FloatArray, k: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Finds the highest and the lowest tokens of one reading.

    Args:
        logits: Array of shape (V,).
        k: How many ids to return at each end, in 1..V.

    Returns:
        The `k` highest ids, best first, and the `k` lowest ids, lowest
        first. Ties keep the lower id first.

    Raises:
        ValueError: If `logits` is not 1-D or `k` is outside 1..V.
    """
    row = np.asarray(logits, dtype=np.float64)
    if row.ndim != 1:
        raise ValueError(f"logits must have shape (V,), got {row.shape}")
    if not 1 <= k <= row.shape[0]:
        raise ValueError(f"k must be in 1..{row.shape[0]}, got {k}")
    top = np.argsort(-row, kind="stable")[:k]
    bottom = np.argsort(row, kind="stable")[:k]
    return top.astype(np.int64), bottom.astype(np.int64)
