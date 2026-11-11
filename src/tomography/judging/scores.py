"""Scores computed from stored judgments."""

from collections.abc import Sequence
from fractions import Fraction
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tomography.judging.rubric import SCORES

SELECTION_RULE = (
    "the coefficient with the highest mean overall score on the tuning "
    "half; a tie goes to the smallest magnitude, then to the lower value"
)


def _checked(scores: NDArray[np.integer[Any]]) -> NDArray[np.integer[Any]]:
    """Returns the scores as an array, or raises on a shape or a value."""
    values = np.asarray(scores)
    if values.ndim != 2:
        raise ValueError(
            f"scores must have shape (items, criteria), got {values.shape}"
        )
    if not np.isin(values, SCORES).all():
        raise ValueError(f"scores must be one of {SCORES}")
    return values


def harmonic_mean(scores: NDArray[np.integer[Any]]) -> NDArray[np.float64]:
    """Combines the criteria of each item as AxBench does.

    Args:
        scores: Integer scores of shape `(items, criteria)`, each one of
            `SCORES`.

    Returns:
        The harmonic mean of the criteria of each item, shape `(items,)`,
        defined as 0 when any criterion is 0.

    Raises:
        ValueError: If `scores` is not 2-D or holds a value outside `SCORES`.
    """
    floats = _checked(scores).astype(np.float64)
    positive = (floats > 0).all(axis=1)
    overall = np.zeros(floats.shape[0], dtype=np.float64)
    overall[positive] = floats.shape[1] / (1.0 / floats[positive]).sum(axis=1)
    return overall


def overall_mean(scores: NDArray[np.integer[Any]]) -> Fraction:
    """Averages the overall score of the items exactly.

    Args:
        scores: Integer scores of shape `(items, criteria)`, each one of
            `SCORES`, with at least one item.

    Returns:
        The mean over the items of the harmonic mean of their criteria (0
        when any criterion is 0), as an exact fraction. The same scores in
        another order give the same value, which a sum of floats does not
        guarantee.

    Raises:
        ValueError: If `scores` is not 2-D, has no item or holds a value
            outside `SCORES`.
    """
    rows = _checked(scores).tolist()
    if not rows:
        raise ValueError("scores must hold at least one item")
    total = Fraction(0)
    for row in rows:
        if all(row):
            total += Fraction(len(row)) / sum(
                (Fraction(1, score) for score in row), Fraction(0)
            )
    return total / len(rows)


def select_coefficient(
    coeffs: Sequence[float], means: Sequence[Fraction | float]
) -> float:
    """Picks the coefficient with the highest mean score.

    Args:
        coeffs: The candidate coefficients.
        means: The mean score of each coefficient on the tuning prompts.
            Exact means (`overall_mean`) make equal sets of scores tie;
            float means can differ in the last bit by the order of a sum.

    Returns:
        The coefficient with the highest mean. A tie goes to the smallest
        magnitude, the weakest edit that reaches the score, and then to the
        lower value.

    Raises:
        ValueError: If there is no coefficient, the lengths differ or a mean
            is not finite.
    """
    if not coeffs:
        raise ValueError("no coefficient to select from")
    if len(coeffs) != len(means):
        raise ValueError(
            f"got {len(means)} means for {len(coeffs)} coefficient values"
        )
    if not np.isfinite(np.asarray(means, dtype=np.float64)).all():
        raise ValueError("the mean score of every coefficient must be finite")
    best = max(means)
    tied = [float(c) for c, m in zip(coeffs, means, strict=True) if m == best]
    return min(tied, key=lambda c: (abs(c), c))


def weighted_kappa(first: Sequence[int], second: Sequence[int]) -> float:
    """Measures the agreement of two judges with quadratic weights.

    Args:
        first: The scores of the first judge, each one of `SCORES`.
        second: The scores of the second judge for the same items.

    Returns:
        Cohen's kappa with quadratic weights: one minus the mean squared
        distance of the paired scores over the squared distance expected from
        the margins of the two judges. 1 when both judges give one and the
        same score to every item.

    Raises:
        ValueError: If the scores are not two 1-D sequences of one length
            with at least one item, or a score is outside `SCORES`.
    """
    a = np.asarray(first)
    b = np.asarray(second)
    if a.ndim != 1 or a.shape != b.shape or a.size == 0:
        raise ValueError(
            "scores must be two 1-D sequences of one length with at least "
            f"one item, got shapes {a.shape} and {b.shape}"
        )
    if not (np.isin(a, SCORES).all() and np.isin(b, SCORES).all()):
        raise ValueError(f"scores must be one of {SCORES}")
    x = a.astype(np.float64)
    y = b.astype(np.float64)
    expected = float(((x[:, None] - y[None, :]) ** 2).mean())
    if expected == 0.0:
        return 1.0
    return 1.0 - float(((x - y) ** 2).mean()) / expected
