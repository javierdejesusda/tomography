"""Scores computed from stored judgments."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tomography.judging.rubric import SCORES

SELECTION_RULE = (
    "the coefficient with the highest mean overall score on the tuning "
    "half; a tie goes to the smallest magnitude, then to the lower value"
)


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
    values = np.asarray(scores)
    if values.ndim != 2:
        raise ValueError(
            f"scores must have shape (items, criteria), got {values.shape}"
        )
    if not np.isin(values, SCORES).all():
        raise ValueError(f"scores must be one of {SCORES}")
    floats = values.astype(np.float64)
    positive = (floats > 0).all(axis=1)
    overall = np.zeros(floats.shape[0], dtype=np.float64)
    overall[positive] = floats.shape[1] / (1.0 / floats[positive]).sum(axis=1)
    return overall


def select_coefficient(
    coeffs: Sequence[float], means: Sequence[float]
) -> float:
    """Picks the coefficient with the highest mean score.

    Args:
        coeffs: The candidate coefficients.
        means: The mean score of each coefficient on the tuning prompts.

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
