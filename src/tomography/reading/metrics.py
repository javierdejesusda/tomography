"""Metrics for scoring readers: accuracy, AUROC and interval estimates."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]


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
    groups = np.asarray(groups)
    if groups.ndim != 2 or 0 in groups.shape:
        raise ValueError(f"groups must have shape (g, k), got {groups.shape}")
    if groups.max() >= values.shape[-1]:
        raise ValueError("groups must index the scores")
    valid = groups >= 0
    safe = np.where(valid, groups, 0)
    is_positive = valid & (labels[safe] == 1)
    if not (is_positive.sum(axis=1) == 1).all():
        raise ValueError("each group needs exactly one label-1 member")
    members = np.where(valid, values[..., safe], -np.inf)
    tied = valid & (members == members.max(axis=-1, keepdims=True))
    positive_column = is_positive.argmax(axis=1)
    wins = tied[..., np.arange(groups.shape[0]), positive_column]
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
