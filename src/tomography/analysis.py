"""Layer-wise analyses: PCA variance, t-SNE maps, geometry, lens, SAE fits."""

from collections.abc import Sequence
import itertools
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


def _unit_vectors(
    vectors: FloatArray, ndim: int, shape: str
) -> NDArray[np.float64]:
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != ndim:
        raise ValueError(f"vectors must have shape {shape}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("vectors must be finite")
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if (norms == 0.0).any():
        raise ValueError("vectors must not contain a zero vector")
    return np.asarray(values / norms)


def _unit_runs(vectors: FloatArray) -> NDArray[np.float64]:
    return _unit_vectors(vectors, 3, "(N, K, d)")


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
    that mean is built from `agreement_pseudo_values` instead.

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
    `seed_agreement` share pairs, so a band that treats them as independent is
    too narrow. The pseudo-value of run i is `N * U - (N - 1) * U_without_i`;
    the pseudo-values average to the mean pairwise cosine and are the units
    `agreement_band` takes its standard error from.

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


_FIT_BLOCK = 128
_FIT_TIE = 1e-6
_BASIS_FLOOR = 1e-10


def _decoder32(decoder: FloatArray) -> NDArray[np.float32]:
    matrix = np.asarray(decoder, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"decoder must have shape (d, F), got {matrix.shape}")
    return matrix


def column_norms(decoder: FloatArray) -> NDArray[np.float64]:
    """Computes the Euclidean norm of each decoder column.

    The squares are accumulated in float64 without copying the decoder.

    Args:
        decoder: Feature directions of shape (d, F): one column per feature.

    Returns:
        Array of shape (F,) in float64.

    Raises:
        ValueError: If `decoder` is not 2-D, has a zero column or holds a
            value that is not finite.
    """
    matrix = _decoder32(decoder)
    squares = np.einsum("df,df->f", matrix, matrix, dtype=np.float64)
    norms = np.sqrt(squares)
    if not np.isfinite(norms).all():
        raise ValueError("decoder must be finite")
    if (norms == 0.0).any():
        raise ValueError("decoder must not contain a zero column")
    return np.asarray(norms)


def _decoder_inputs(
    vectors: FloatArray, decoder: FloatArray
) -> tuple[NDArray[np.float64], NDArray[np.float32], NDArray[np.float32]]:
    matrix = _decoder32(decoder)
    unit = _unit_vectors(vectors, 2, "(n, d)")
    if unit.shape[1] != matrix.shape[0]:
        raise ValueError(
            f"vectors have width {unit.shape[1]} but the decoder has "
            f"{matrix.shape[0]} rows"
        )
    inverse = (1.0 / column_norms(matrix)).astype(np.float32)
    return unit, matrix, inverse


def decoder_cosines(
    vectors: FloatArray, decoder: FloatArray
) -> NDArray[np.float32]:
    """Computes the cosine of each vector with each decoder column.

    The vectors are normalized and the products are divided by the column
    norms, so neither the vectors nor the decoder is rescaled in memory.
    The products are taken in float32.

    Args:
        vectors: Directions of shape (n, d).
        decoder: Feature directions of shape (d, F): one column per feature.

    Returns:
        Array of shape (n, F) in float32.

    Raises:
        ValueError: If `vectors` or `decoder` is not 2-D, the widths differ,
            a vector is zero, a decoder column is zero or a value is not
            finite.
    """
    unit, matrix, inverse = _decoder_inputs(vectors, decoder)
    cosines = unit.astype(np.float32) @ matrix
    cosines *= inverse
    return np.asarray(cosines)


def top_features(cosines: FloatArray, k: int) -> NDArray[np.int64]:
    """Finds the features with the largest absolute cosine.

    Args:
        cosines: Array of shape (n, F), one row per direction.
        k: How many ids to return per row, in 1..F.

    Returns:
        Array of shape (n, k): the ids by decreasing `|cosine|`, the lower id
        first on ties.

    Raises:
        ValueError: If `cosines` is not 2-D or not finite, or `k` is outside
            1..F.
    """
    values = np.asarray(cosines)
    if values.ndim != 2:
        raise ValueError(f"cosines must have shape (n, F), got {values.shape}")
    if not 1 <= k <= values.shape[1]:
        raise ValueError(f"k must be in 1..{values.shape[1]}, got {k}")
    if not np.isfinite(values).all():
        raise ValueError("cosines must be finite")
    top = np.empty((values.shape[0], k), dtype=np.int64)
    for i, row in enumerate(values):
        top[i] = np.argsort(-np.abs(row), kind="stable")[:k]
    return top


def _match(
    unit: NDArray[np.float64],
    decoder: NDArray[np.float32],
    inverse: NDArray[np.float32],
    n_atoms: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    n, d = unit.shape
    rows = np.arange(n)
    residual = unit.copy()
    basis = np.zeros((n, n_atoms, d))
    taken = np.zeros((n, decoder.shape[1]), dtype=bool)
    atoms = np.empty((n, n_atoms), dtype=np.int64)
    gains = np.empty((n, n_atoms))
    for t in range(n_atoms):
        score = np.abs(residual.astype(np.float32) @ decoder * inverse)
        score[taken] = -1.0
        near_best = score >= score.max(axis=1, keepdims=True) - _FIT_TIE
        chosen = near_best.argmax(axis=1)
        atoms[:, t] = chosen
        taken[rows, chosen] = True
        atom = decoder[:, chosen].T.astype(np.float64)
        atom *= inverse[chosen].astype(np.float64)[:, None]
        for _ in range(2):
            atom -= np.einsum(
                "bt,btd->bd",
                np.einsum("btd,bd->bt", basis[:, :t], atom),
                basis[:, :t],
            )
        length = np.linalg.norm(atom, axis=1)
        usable = length > _BASIS_FLOOR
        basis[usable, t] = atom[usable] / length[usable, None]
        along = np.einsum("bd,bd->b", basis[:, t], residual)
        residual -= along[:, None] * basis[:, t]
        gains[:, t] = along**2
    return np.minimum(np.cumsum(gains, axis=1), 1.0), atoms


def sparse_fit(
    vectors: FloatArray, decoder: FloatArray, n_atoms: int
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Fits each vector with a few decoder columns by orthogonal matching.

    Each step picks the unselected column most correlated with the residual
    (ties within 1e-6 go to the lower id), then refits all selected columns
    to the vector by least squares. The vectors are normalized and the
    correlations divided by the column norms, so the decoder is read in
    place. Rows are fitted in blocks, so memory does not grow with their
    number.

    Args:
        vectors: Directions of shape (n, d).
        decoder: Feature directions of shape (d, F): one column per feature.
        n_atoms: Number of columns to select, in 1..min(d, F).

    Returns:
        The explained fraction after 1..`n_atoms` atoms, `1 - |v - A c|^2`
        for the unit vector `v`, shape (n, n_atoms), non-decreasing and in
        [0, 1]; and the selected ids in order, shape (n, n_atoms), distinct
        within a row.

    Raises:
        ValueError: If the inputs are unusable (see `decoder_cosines`) or
            `n_atoms` is outside 1..min(d, F).
    """
    unit, matrix, inverse = _decoder_inputs(vectors, decoder)
    limit = min(matrix.shape)
    if not 1 <= n_atoms <= limit:
        raise ValueError(f"n_atoms must be in 1..{limit}, got {n_atoms}")
    explained = np.empty((unit.shape[0], n_atoms))
    atoms = np.empty((unit.shape[0], n_atoms), dtype=np.int64)
    for start in range(0, unit.shape[0], _FIT_BLOCK):
        block = slice(start, start + _FIT_BLOCK)
        explained[block], atoms[block] = _match(
            unit[block], matrix, inverse, n_atoms
        )
    return explained, atoms


def dictionary_directions(
    decoder: FloatArray, n: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Draws unit directions with the covariance of the decoder columns.

    Each direction is `normalize(g @ decoder.T)` for a standard normal `g`
    over the F features, so it follows the anisotropy of the dictionary that
    an isotropic Gaussian ignores. `g` is drawn as
    `rng.standard_normal((n, F))` and cast to float32.

    Args:
        decoder: Feature directions of shape (d, F): one column per feature.
        n: Number of directions, at least 1.
        rng: The source of the draws.

    Returns:
        Array of shape (n, d) in float64 with unit rows.

    Raises:
        ValueError: If `decoder` is not 2-D, `n` is below 1, or a draw is
            not finite or is zero, as when the decoder holds a non-finite
            value or only zeros.
    """
    matrix = _decoder32(decoder)
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    draws = rng.standard_normal((n, matrix.shape[1])).astype(np.float32)
    return _unit_vectors(draws @ matrix.T, 2, "(n, d)")


def top_overlap(ids: NDArray[np.int64]) -> float:
    """Averages the Jaccard index of the id sets over every pair of rows.

    Args:
        ids: Array of shape (n, k), n >= 2: the top ids of each run.

    Returns:
        The mean over the n (n - 1) / 2 row pairs of
        `|A and B| / |A or B|`: 1 when every row holds the same ids, 0 when
        no two rows share one.

    Raises:
        ValueError: If `ids` is not of shape (n, k) with k >= 1, or has fewer
            than 2 rows.
    """
    rows = np.asarray(ids)
    if rows.ndim != 2 or rows.shape[1] < 1:
        raise ValueError(f"ids must have shape (n, k), got {rows.shape}")
    if rows.shape[0] < 2:
        raise ValueError(f"ids needs at least 2 rows, got {rows.shape[0]}")
    sets = [set(row.tolist()) for row in rows]
    pairs = itertools.combinations(sets, 2)
    return float(np.mean([len(a & b) / len(a | b) for a, b in pairs]))
