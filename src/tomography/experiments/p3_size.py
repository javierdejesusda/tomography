"""P3: how many pairs a reading direction needs, and how much one draw misleads.

For each task and each training-set size, many resampled training sets are
read at once, each compared with a reader fitted on all the training data. The
spread of the report accuracy, the share of failed draws, the agreement of the
pair differences as a warning sign and the size where returns diminish come
from the same sweeps.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from tomography.experiments.sweep import Prepared


def oriented_differences(prepared: Prepared) -> NDArray[np.float64]:
    """Takes the label-1 member minus the label-0 member of every pair.

    The order inside a stored pair is random, so a plain difference would
    flip sign from pair to pair; the oriented one does not.

    Args:
        prepared: The stimuli and activations of a training set, read at one
            row per stimulus.

    Returns:
        Array of shape (K, m, d): layer, pair and feature.

    Raises:
        ValueError: If a pair holds two members of the same label.
    """
    stimuli, act = prepared
    pairs = act.row_groups(stimuli.pairs)
    labels = act.row_values(stimuli.labels)
    first = labels[pairs[:, 0]]
    equal = np.flatnonzero(first == labels[pairs[:, 1]])
    if equal.size:
        raise ValueError(f"pair {int(equal[0])} has equal labels")
    swap = first == 0
    positive = np.where(swap, pairs[:, 1], pairs[:, 0])
    negative = np.where(swap, pairs[:, 0], pairs[:, 1])
    return act.data[:, positive].astype(np.float64) - act.data[
        :, negative
    ].astype(np.float64)


def directional_agreement(diffs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Averages the cosine between each difference and the mean difference.

    It is one when every pair points the same way and falls toward zero as
    the pairs disagree, so a training set can be judged without labels from
    the evaluation set.

    Args:
        diffs: Pair differences of shape (K, m, d), with m at least 2.

    Returns:
        Array of shape (K,).

    Raises:
        ValueError: If `diffs` is not 3-D, has fewer than two pairs, or a
            difference or the mean difference is the zero vector.
    """
    vectors = np.asarray(diffs, dtype=np.float64)
    if vectors.ndim != 3 or vectors.shape[1] < 2:
        raise ValueError(
            "directional_agreement needs differences of shape (K, m, d) with "
            f"m >= 2, got {vectors.shape}"
        )
    mean = vectors.mean(axis=1)
    norms = np.linalg.norm(vectors, axis=2)
    mean_norms = np.linalg.norm(mean, axis=1)
    if (norms == 0.0).any() or (mean_norms == 0.0).any():
        raise ValueError("directional_agreement is undefined for a zero vector")
    cosines = np.einsum("kmd,kd->km", vectors, mean) / (
        norms * mean_norms[:, None]
    )
    return np.asarray(cosines.mean(axis=1), dtype=np.float64)


def saturation_size(
    sizes: Sequence[int], accuracy: NDArray[np.float64], *, delta: float
) -> int:
    """Finds the smallest size that is within `delta` of the largest.

    Args:
        sizes: The training-set sizes, Z of them.
        accuracy: Report accuracy of shape (Z, R): size and resample.
        delta: How far below the mean accuracy of the largest size a size's
            mean may fall and still count as saturated.

    Returns:
        The smallest size whose mean accuracy is at least the largest
        size's mean minus `delta`.

    Raises:
        ValueError: If `accuracy` does not have shape (Z, R) or `delta` is
            negative.
    """
    grid, values = _checked_saturation_inputs(sizes, accuracy, delta)
    return int(_saturated(grid, values.mean(axis=1), delta))


def saturation_ci(
    sizes: Sequence[int],
    accuracy: NDArray[np.float64],
    *,
    delta: float,
    n_boot: int,
    seed: int,
) -> tuple[int, int, int]:
    """Bounds the saturation size by a percentile bootstrap over resamples.

    Each bootstrap draw resamples the R resamples of every size with
    replacement, independently per size, and finds the saturation size of
    the resampled means.

    Args:
        sizes: The training-set sizes, Z of them.
        accuracy: Report accuracy of shape (Z, R): size and resample.
        delta: As in `saturation_size`.
        n_boot: Number of bootstrap draws.
        seed: Seed of the draws.

    Returns:
        The saturation size and the 2.5 and 97.5 percent bounds of its
        bootstrap distribution, each one of `sizes`.

    Raises:
        ValueError: If `accuracy` does not have shape (Z, R), `delta` is
            negative or `n_boot` is below 1.
    """
    grid, values = _checked_saturation_inputs(sizes, accuracy, delta)
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot}")
    n_sizes, n_resamples = values.shape
    draws = np.random.default_rng(seed).integers(
        0, n_resamples, size=(n_boot, n_sizes, n_resamples)
    )
    means = np.take_along_axis(values[None], draws, axis=2).mean(axis=2)
    boot = _saturated(grid, means, delta)
    return (
        saturation_size(sizes, accuracy, delta=delta),
        int(np.percentile(boot, 2.5, method="lower")),
        int(np.percentile(boot, 97.5, method="higher")),
    )


def _checked_saturation_inputs(
    sizes: Sequence[int], accuracy: NDArray[np.float64], delta: float
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    grid = np.asarray(sizes, dtype=np.int64)
    values = np.asarray(accuracy, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != grid.shape[0]:
        raise ValueError(
            f"saturation needs accuracy of shape ({grid.shape[0]}, R), one "
            f"row per size, got {values.shape}"
        )
    if delta < 0.0:
        raise ValueError(f"saturation needs a delta of at least 0, got {delta}")
    return grid, values


def _saturated(
    sizes: NDArray[np.int64], means: NDArray[np.float64], delta: float
) -> NDArray[np.int64]:
    """Returns the smallest size reaching the largest size's mean minus delta.

    Args:
        sizes: The sizes, shape (Z,).
        means: Mean accuracy of shape (..., Z).
        delta: The tolerance, at least 0, so the largest size always counts.

    Returns:
        Array of shape (...,).
    """
    floor = means[..., np.argmax(sizes)][..., None] - delta
    return np.where(means >= floor, sizes, np.iinfo(np.int64).max).min(axis=-1)


def spearman(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Computes the Spearman rank correlation, averaging the ranks of ties.

    Args:
        x: Values of shape (n,).
        y: Values of shape (n,).

    Returns:
        The Pearson correlation of the ranks; NaN when one side is constant.

    Raises:
        ValueError: If `x` and `y` are not vectors of the same length, at
            least 2.
    """
    first = np.asarray(x, dtype=np.float64)
    second = np.asarray(y, dtype=np.float64)
    if first.ndim != 1 or first.shape != second.shape or first.shape[0] < 2:
        raise ValueError(
            "spearman needs two vectors of the same length, at least 2, got "
            f"shapes {first.shape} and {second.shape}"
        )
    ranks_x = _average_ranks(first)
    ranks_y = _average_ranks(second)
    centered_x = ranks_x - ranks_x.mean()
    centered_y = ranks_y - ranks_y.mean()
    norm = np.sqrt((centered_x**2).sum() * (centered_y**2).sum())
    if norm == 0.0:
        return float("nan")
    return float((centered_x * centered_y).sum() / norm)


def _average_ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    ends = np.r_[starts[1:], values.shape[0]]
    ranks = np.empty(values.shape[0], dtype=np.float64)
    ranks[order] = np.repeat((starts + ends + 1) / 2, ends - starts)
    return ranks
