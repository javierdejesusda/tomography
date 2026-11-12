"""Tests for the scores computed from stored judgments."""

from fractions import Fraction

import numpy as np
import pytest

from tomography import judges
from tomography.judging.scores import harmonic_mean
from tomography.judging.scores import overall_mean
from tomography.judging.scores import select_coefficient
from tomography.judging.scores import weighted_kappa


def test_the_harmonic_mean_is_hand_computed_for_three_criteria() -> None:
    scores = np.array([[2, 2, 2], [1, 2, 2], [1, 1, 2], [1, 1, 1]])

    overall = harmonic_mean(scores)

    assert overall.dtype == np.float64
    assert overall.tolist() == pytest.approx([2.0, 1.5, 1.2, 1.0])


def test_the_harmonic_mean_is_hand_computed_for_two_criteria() -> None:
    overall = harmonic_mean(np.array([[1, 2], [2, 2], [2, 1]]))

    assert overall.tolist() == pytest.approx([4 / 3, 2.0, 4 / 3])


def test_any_zero_makes_the_overall_score_zero() -> None:
    scores = np.array([[0, 2, 2], [2, 0, 2], [2, 2, 0], [0, 0, 0]])

    assert harmonic_mean(scores).tolist() == [0.0, 0.0, 0.0, 0.0]


def test_no_items_give_no_scores() -> None:
    assert harmonic_mean(np.zeros((0, 3), dtype=np.int64)).shape == (0,)


@pytest.mark.parametrize(
    "scores",
    [np.array([1, 2]), np.array([[3, 1]]), np.array([[-1, 1]])],
    ids=["one_dimension", "above_range", "below_range"],
)
def test_the_harmonic_mean_rejects_other_inputs(
    scores: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="scores"):
        harmonic_mean(scores)


def test_the_selected_coefficient_has_the_highest_mean() -> None:
    assert select_coefficient([0.1, 0.2, 0.4], [0.5, 1.25, 1.0]) == 0.2


def test_a_tie_goes_to_the_smallest_edit() -> None:
    assert select_coefficient([0.4, 0.1, 0.2], [1.0, 0.25, 1.0]) == 0.2
    assert select_coefficient([0.2, -0.2, 0.4], [1.0, 1.0, 1.0]) == -0.2


@pytest.mark.parametrize(
    ("coeffs", "means"),
    [([], []), ([0.1, 0.2], [1.0]), ([0.1, 0.2], [1.0, float("nan")])],
    ids=["empty", "lengths", "not_finite"],
)
def test_selection_rejects_what_it_cannot_rank(
    coeffs: list[float], means: list[float]
) -> None:
    with pytest.raises(ValueError, match="coefficient"):
        select_coefficient(coeffs, means)


def test_label_agreement_is_cohens_kappa_of_three_score_levels() -> None:
    first = [0, 0, 1, 1, 2, 2, 2, 0, 1, 2]
    second = [0, 1, 1, 1, 2, 0, 2, 0, 2, 2]

    agreement = judges.label_agreement(first, second)

    observed = 7 / 10
    chance = (3 * 3 + 3 * 3 + 4 * 4) / 100
    assert agreement.rate == pytest.approx(observed)
    assert agreement.kappa == pytest.approx((observed - chance) / (1 - chance))


def test_label_agreement_is_one_when_both_use_a_single_level() -> None:
    assert judges.label_agreement([2, 2], [2, 2]) == judges.Agreement(1.0, 1.0)


@pytest.mark.parametrize(
    ("first", "second"),
    [([], []), ([0, 1], [0]), ([0, -1], [0, 1]), ([[0, 1]], [[0, 1]])],
    ids=["empty", "lengths", "negative", "two_dimensions"],
)
def test_label_agreement_rejects_invalid_labels(
    first: list[object], second: list[object]
) -> None:
    with pytest.raises(ValueError, match="labels"):
        judges.label_agreement(first, second)


def test_judge_agreement_still_thresholds_probabilities_first() -> None:
    first = [0.9, 0.8, 0.2, 0.6, 0.4, 0.1]
    second = [0.7, 0.3, 0.1, 0.9, 0.6, 0.2]

    agreement = judges.judge_agreement(first, second)

    observed = 4 / 6
    chance = (3 * 3 + 3 * 3) / 36
    assert agreement.rate == pytest.approx(observed)
    assert agreement.kappa == pytest.approx((observed - chance) / (1 - chance))
    assert agreement == judges.label_agreement(
        [1, 1, 0, 1, 0, 0], [1, 0, 0, 1, 1, 0]
    )


# One (0, 2), two (1, 2) and thirty-seven (2, 2): 0 + 2 * 4/3 + 37 * 2 over 40.
# Summed as floats, this order and its reverse differ in the last bit.
TIED = np.array([(0, 2)] + [(1, 2)] * 2 + [(2, 2)] * 37)


def test_the_exact_mean_does_not_depend_on_the_order_of_the_items() -> None:
    assert overall_mean(TIED) == Fraction(23, 12)
    assert overall_mean(TIED[::-1]) == Fraction(23, 12)
    assert overall_mean(np.array([[1, 1, 2], [0, 2, 2]])) == Fraction(3, 5)
    assert float(overall_mean(TIED)) == pytest.approx(
        float(harmonic_mean(TIED).mean())
    )


def test_equal_sets_of_scores_tie_and_the_smallest_magnitude_wins() -> None:
    means = [overall_mean(TIED), overall_mean(TIED[::-1])]

    assert select_coefficient([0.1, 0.4], means) == 0.1
    assert select_coefficient([0.4, 0.1], means) == 0.1
    assert select_coefficient([0.1, 0.4], means[::-1]) == 0.1


@pytest.mark.parametrize(
    "scores",
    [np.zeros((0, 2), dtype=np.int64), np.array([1, 2]), np.array([[1, 3]])],
    ids=["no-item", "one-dimension", "outside-scale"],
)
def test_the_exact_mean_rejects_other_inputs(scores: np.ndarray) -> None:
    with pytest.raises(ValueError, match="scores"):
        overall_mean(scores)


def test_the_weighted_kappa_is_hand_computed_with_quadratic_weights() -> None:
    first = [0, 0, 1, 2, 2, 2]
    second = [0, 1, 1, 2, 0, 2]

    # Observed squared distance: (0 + 1 + 0 + 0 + 4 + 0) / 6 = 5/6. Expected
    # from the margins (2, 1, 3) / 6 and (2, 2, 2) / 6: 54/36, so the
    # kappa is 1 - (5/6) / (3/2).
    assert weighted_kappa(first, second) == pytest.approx(4 / 9)
    assert weighted_kappa(second, first) == pytest.approx(4 / 9)
    assert weighted_kappa(first, first) == 1.0
    assert weighted_kappa([2, 2], [2, 2]) == 1.0
    # A miss by two levels costs more than a miss by one.
    assert weighted_kappa([0, 1, 2, 0], [0, 1, 2, 2]) < weighted_kappa(
        [0, 1, 2, 0], [0, 1, 2, 1]
    )


@pytest.mark.parametrize(
    ("first", "second"),
    [([], []), ([0, 1], [0]), ([0, 3], [0, 1]), ([[0, 1]], [[0, 1]])],
    ids=["empty", "lengths", "outside-scale", "two-dimensions"],
)
def test_the_weighted_kappa_rejects_invalid_scores(
    first: list[int], second: list[int]
) -> None:
    with pytest.raises(ValueError, match="scores"):
        weighted_kappa(first, second)
