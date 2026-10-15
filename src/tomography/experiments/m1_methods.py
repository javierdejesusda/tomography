"""M1: six direction finders compared on correlation, steering and erasure.

Each finder reads one concept (utility, from ETHICS) and is then compared on
three tasks: correlation (does the direction read held-out pairs),
manipulation (does steering along it change the sentiment of the
continuation of a truncated scenario) and termination (does projecting it
out remove the concept). The `random` finder is the null of the comparison.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tomography.reading.directions import Method
from tomography.reading.metrics import group_accuracy
from tomography.reading.reader import RepReader

METHODS: tuple[Method, ...] = (
    "pca_diff",
    "pca_raw",
    "mean_diff",
    "logistic",
    "kmeans",
    "random",
)

_CONTROL_DEPTHS = (0.4, 0.55, 0.7)
_ZERO_NORM = "zero norm"


@dataclass(frozen=True)
class Tally:
    """Wins, losses and ties of paired comparisons.

    Attributes:
        wins: Pairs whose first side is strictly higher.
        losses: Pairs whose first side is strictly lower.
        ties: Pairs with equal sides.
    """

    wins: int
    losses: int
    ties: int

    @property
    def decisive(self) -> int:
        """Pairs that are a win or a loss."""
        return self.wins + self.losses

    @property
    def rate(self) -> float:
        """Wins over decisive pairs; NaN when no pair is decisive."""
        return self.wins / self.decisive if self.decisive else float("nan")


def truncate_scenario(text: str, words: int) -> str:
    """Cuts a scenario to its first words.

    Args:
        text: The scenario.
        words: Number of whitespace-separated words to keep.

    Returns:
        The first `words` words joined by single spaces; a shorter text
        keeps all its words.

    Raises:
        ValueError: If `words` is below 1.
    """
    if words < 1:
        raise ValueError(f"words must be at least 1, got {words}")
    return " ".join(text.split()[:words])


def tally(
    positive: NDArray[np.float64], negative: NDArray[np.float64]
) -> Tally:
    """Counts, per prompt, whether the first score is above the second.

    Args:
        positive: One score per prompt under the positive coefficient.
        negative: One score per prompt under the negative coefficient.

    Returns:
        The wins, losses and ties. A prompt with a NaN score on either
        side (an empty completion) counts as none of them.

    Raises:
        ValueError: If the inputs are not two vectors of one length.
    """
    if positive.shape != negative.shape or positive.ndim != 1:
        raise ValueError(
            "need two vectors of one length, got shapes "
            f"{positive.shape} and {negative.shape}"
        )
    return Tally(
        wins=int((positive > negative).sum()),
        losses=int((positive < negative).sum()),
        ties=int((positive == negative).sum()),
    )


def paired_strengths(coeffs: Sequence[float]) -> tuple[float, ...]:
    """Returns the positive half of a symmetric coefficient grid.

    Args:
        coeffs: The coefficients, in any order.

    Returns:
        The positive values, sorted.

    Raises:
        ValueError: If `coeffs` lacks 0.0, has no positive value, or has a
            positive value without its negative or a negative value without
            its positive.
    """
    if 0.0 not in coeffs:
        raise ValueError("coeffs must contain 0.0")
    strengths = tuple(sorted(c for c in coeffs if c > 0))
    if not strengths:
        raise ValueError("coeffs must hold at least one positive strength")
    positive_side = {c for c in coeffs if c > 0}
    negative_side = {-c for c in coeffs if c < 0}
    if positive_side != negative_side:
        raise ValueError(
            "coeffs must hold every strength with its negative, got "
            f"{tuple(sorted(coeffs))}"
        )
    return strengths


def project_out(
    h: NDArray[np.float32], direction: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Removes the component of every vector along a direction.

    Args:
        h: Vectors of shape (..., d).
        direction: A vector of shape (d,); it need not be a unit vector.

    Returns:
        The vectors with no component along the direction, in float32. The
        arithmetic runs in float64.

    Raises:
        ValueError: If `direction` is not a nonzero vector of the width of
            `h`.
    """
    if direction.ndim != 1 or direction.shape[0] != h.shape[-1]:
        raise ValueError(
            f"direction must have shape ({h.shape[-1]},), got {direction.shape}"
        )
    unit = direction.astype(np.float64)
    norm = np.linalg.norm(unit)
    if not (np.isfinite(norm) and norm > 0):
        raise ValueError("direction must be finite and nonzero")
    unit /= norm
    wide = h.astype(np.float64)
    return (wide - (wide @ unit)[..., None] * unit).astype(np.float32)


def _probe_accuracy(
    train_h: NDArray[np.float32],
    train_labels: NDArray[np.int64],
    test_h: NDArray[np.float32],
    test_groups: NDArray[np.int64],
    test_labels: NDArray[np.int64],
    seed: int,
) -> float:
    try:
        probe = RepReader.fit(
            "logistic", train_h[None], [0], labels=train_labels, seed=seed
        )
    except ValueError as error:
        if _ZERO_NORM not in str(error):
            raise
        return 0.5
    scores = probe.score(test_h[None])
    return float(group_accuracy(scores, test_groups, test_labels)[0])


def erased_probe_accuracy(
    train_h: NDArray[np.float32],
    train_labels: NDArray[np.int64],
    test_h: NDArray[np.float32],
    test_groups: NDArray[np.int64],
    test_labels: NDArray[np.int64],
    direction: NDArray[np.float32],
    *,
    seed: int,
) -> tuple[float, float]:
    """Measures a fresh logistic probe before and after an erasure.

    The probe is fit on `train_h` and scores the pairs of `test_h`. The
    second probe is fit and scored after `direction` is projected out of
    both sets. A logistic fit that `RepReader.fit` rejects for a zero-norm
    direction (what projecting out the class-mean difference leaves) is a
    constant scorer, which counts as chance, 0.5.

    Args:
        train_h: Training activations of one layer, shape (n, d).
        train_labels: Binary labels of the training rows.
        test_h: Test activations of the same layer, shape (m, d).
        test_groups: Row groups of `test_h`, shape (g, k), each with one
            label-1 member.
        test_labels: Binary labels of the test rows.
        direction: The direction to project out, shape (d,).
        seed: Seed of the probe.

    Returns:
        The group accuracy of the probe before and after the erasure.

    Raises:
        ValueError: If the probe cannot be fit or scored for a reason other
            than a zero-norm direction, or `direction` is unusable.
    """
    before = _probe_accuracy(
        train_h, train_labels, test_h, test_groups, test_labels, seed
    )
    after = _probe_accuracy(
        project_out(train_h, direction),
        train_labels,
        project_out(test_h, direction),
        test_groups,
        test_labels,
        seed,
    )
    return before, after


def default_control_layers(n_layers: int) -> tuple[int, ...]:
    """Returns the layers every method is steered at.

    They sit at 40, 55 and 70 percent of the depth, so the layer is not a
    confound between methods.

    Args:
        n_layers: Number of transformer layers.

    Returns:
        The rounded layers, at least 1, without duplicates, sorted.
    """
    return tuple(sorted({max(round(n_layers * f), 1) for f in _CONTROL_DEPTHS}))
