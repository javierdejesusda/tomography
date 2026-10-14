"""Metrics for scoring readers: accuracy, AUROC and interval estimates."""

from collections.abc import Sequence
import statistics
from typing import Any
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]


class GroupIndex(NamedTuple):
    """Row groups prepared for gathering scores.

    Attributes:
        members: Row indices of shape (g, k), with padding replaced by 0.
        valid: Boolean mask of shape (g, k), False on padding.
        positive: Column of the label-1 member of each group, shape (g,).
    """

    members: NDArray[np.int64]
    valid: NDArray[np.bool_]
    positive: NDArray[np.int64]


def parse_groups(
    groups: NDArray[np.int64], labels: NDArray[np.int64]
) -> GroupIndex:
    """Validates row groups and locates the label-1 member of each.

    Args:
        groups: Row indices of shape (g, k), padded with -1.
        labels: Binary labels of shape (n,).

    Returns:
        The groups ready for gathering scores.

    Raises:
        ValueError: If `groups` is empty, malformed or out of range, or a
            group does not have exactly one label-1 member.
    """
    groups = np.asarray(groups)
    if groups.ndim != 2 or 0 in groups.shape:
        raise ValueError(f"groups must have shape (g, k), got {groups.shape}")
    if groups.max() >= labels.shape[0]:
        raise ValueError("groups must index the rows of labels")
    valid = groups >= 0
    members = np.where(valid, groups, 0)
    is_positive = valid & (labels[members] == 1)
    if not (is_positive.sum(axis=1) == 1).all():
        raise ValueError("each group needs exactly one label-1 member")
    return GroupIndex(members, valid, is_positive.argmax(axis=1))


def group_accuracy(
    scores: FloatArray,
    groups: NDArray[np.int64],
    labels: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Measures how often the label-1 member of a group scores highest.

    Ties among the maxima share the credit, so a constant scorer sits at
    chance: 1/2 for pairs and 1/k for groups of k. Covers paired readings
    (k = 2) and multiple choice (k > 2).

    Args:
        scores: Scores of shape (..., n); leading axes are kept, for
            example (seeds, layers, n).
        groups: Row indices of shape (g, k) with exactly one label-1 member
            per group. Ragged groups are padded with -1, which is ignored.
        labels: Binary labels of shape (n,).

    Returns:
        Accuracy of shape `scores.shape[:-1]`.

    Raises:
        ValueError: If the shapes disagree, a group does not have exactly
            one label-1 member, or a score is not finite.
    """
    values = _checked_scores(scores, labels)
    index = parse_groups(groups, labels)
    members = np.where(index.valid, values[..., index.members], -np.inf)
    tied = index.valid & (members == members.max(axis=-1, keepdims=True))
    wins = tied[..., np.arange(index.members.shape[0]), index.positive]
    credit = wins / tied.sum(axis=-1)
    return np.asarray(credit.mean(axis=-1), dtype=np.float64)


def auroc(scores: FloatArray, labels: NDArray[np.int64]) -> NDArray[np.float64]:
    """Computes the area under the ROC curve, ranking ties by average rank.

    Args:
        scores: Scores of shape (..., n); higher should mean label 1.
        labels: Binary labels of shape (n,), both classes present.

    Returns:
        AUROC of shape `scores.shape[:-1]`.

    Raises:
        ValueError: If the shapes disagree, a class is missing, or a score
            is not finite.
    """
    values = _checked_scores(scores, labels)
    positive = labels == 1
    n_positive = int(positive.sum())
    n_negative = labels.shape[0] - n_positive
    if n_positive == 0 or n_negative == 0:
        raise ValueError("labels must contain both classes")
    flat = values.reshape(-1, values.shape[-1])
    positive_rank_sums = np.array(
        [_average_ranks(row)[positive].sum() for row in flat]
    )
    rank_sum_floor = n_positive * (n_positive + 1) / 2
    result = (positive_rank_sums - rank_sum_floor) / (n_positive * n_negative)
    return np.asarray(result.reshape(values.shape[:-1]), dtype=np.float64)


def bootstrap_ci(
    values: FloatArray,
    *,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Computes a percentile bootstrap interval of the mean over axis 0.

    Axis 0 holds the repeated runs, for example seeds; resampling draws
    whole runs with replacement. A single run gives a zero-width interval.

    Args:
        values: Array of shape (S, ...) with S >= 1.
        alpha: One minus the coverage; 0.05 gives a 95% interval.
        n_boot: Number of bootstrap resamples.
        seed: Seed of the resampling.

    Returns:
        The lower and upper bounds, each of shape `values.shape[1:]`.

    Raises:
        ValueError: If `values` is empty or `alpha` or `n_boot` is invalid.
    """
    runs = np.asarray(values, dtype=np.float64)
    if runs.ndim == 0 or runs.shape[0] == 0:
        raise ValueError("values needs at least one run along axis 0")
    _check_alpha(alpha)
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    n_runs = runs.shape[0]
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n_runs, np.full(n_runs, 1.0 / n_runs), size=n_boot)
    means = (draws @ runs.reshape(n_runs, -1)) / n_runs
    lo, hi = np.percentile(means, [50 * alpha, 100 - 50 * alpha], axis=0)
    shape = runs.shape[1:]
    return lo.reshape(shape), hi.reshape(shape)


def wilson_ci(
    successes: int, trials: int, *, alpha: float = 0.05
) -> tuple[float, float]:
    """Computes the Wilson score interval of a success rate.

    Args:
        successes: Number of successes.
        trials: Number of trials, at least 1.
        alpha: One minus the coverage; 0.05 gives a 95% interval.

    Returns:
        The lower and upper bounds, inside [0, 1].

    Raises:
        ValueError: If the counts or `alpha` are out of range.
    """
    if trials < 1:
        raise ValueError("trials must be at least 1")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between 0 and trials")
    _check_alpha(alpha)
    z = statistics.NormalDist().inv_cdf(1.0 - alpha / 2.0)
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * np.sqrt(rate * (1.0 - rate) / trials + z * z / (4.0 * trials**2))
        / denominator
    )
    lower = 0.0 if successes == 0 else float(center - half_width)
    upper = 1.0 if successes == trials else float(center + half_width)
    return lower, upper


def best_layer(acc: FloatArray, layers: Sequence[int]) -> int:
    """Picks the layer with the highest accuracy, the first one on ties.

    Args:
        acc: Accuracy per layer, shape (K,).
        layers: The layer index of each entry of `acc`.

    Returns:
        The layer index (an element of `layers`, not a position).

    Raises:
        ValueError: If `acc` is not a non-empty vector without NaN, or its
            length differs from `layers`.
    """
    values = np.asarray(acc, dtype=np.float64)
    if values.ndim != 1 or values.shape[0] == 0:
        raise ValueError(f"acc must be a non-empty vector, got {values.shape}")
    if values.shape[0] != len(layers):
        raise ValueError("acc and layers must have the same length")
    if np.isnan(values).any():
        raise ValueError("acc must not contain NaN")
    return int(layers[int(np.argmax(values))])


def cosine_matrix(vectors: FloatArray) -> NDArray[np.float64]:
    """Computes all pairwise cosine similarities between row vectors.

    Args:
        vectors: Array of shape (m, d) without zero rows.

    Returns:
        Symmetric matrix of shape (m, m) with entries in [-1, 1].

    Raises:
        ValueError: If `vectors` is not a matrix or has a zero row.
    """
    rows = np.asarray(vectors, dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError(f"vectors must have shape (m, d), got {rows.shape}")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if (norms == 0.0).any():
        raise ValueError("vectors must not contain a zero vector")
    unit = rows / norms
    return np.clip(unit @ unit.T, -1.0, 1.0)


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def _checked_scores(
    scores: FloatArray, labels: NDArray[np.int64]
) -> NDArray[np.float64]:
    if labels.ndim != 1 or not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be a vector of 0 and 1")
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] != labels.shape[0]:
        raise ValueError(
            f"scores {values.shape} do not match labels {labels.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    return values


def _average_ranks(row: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(row, kind="stable")
    ordered = row[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    ends = np.r_[starts[1:], row.shape[0]]
    run_ranks = (starts + ends + 1) / 2
    ranks = np.empty(row.shape[0], dtype=np.float64)
    ranks[order] = np.repeat(run_ranks, ends - starts)
    return ranks
