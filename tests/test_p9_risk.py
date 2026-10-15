"""Tests for the interval and selective-prediction tools of the P9 study."""

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.p9.risk import aurc
from tomography.p9.risk import auroc_ci
from tomography.p9.risk import auroc_diff_ci
from tomography.p9.risk import auroc_replicates
from tomography.p9.risk import group_rate
from tomography.p9.risk import orient_on_train
from tomography.p9.risk import risk_at
from tomography.p9.risk import risk_band
from tomography.p9.risk import risk_coverage
from tomography.p9.risk import stack_scores
from tomography.reading.metrics import auroc

GRID = tuple(range(5, 101, 5))
NO_RESAMPLE = "no bootstrap resample holds both classes"


def _planted(
    n: int = 600, error_rate: float = 0.425, shift: float = 1.5
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    rng = np.random.default_rng(0)
    correct = (rng.random(n) >= error_rate).astype(np.int64)
    informative = shift * correct + rng.standard_normal(n)
    noise = rng.standard_normal(n)
    return informative, noise, correct


def test_auroc_ci_brackets_the_point() -> None:
    informative, _, correct = _planted()

    point, lo, hi = auroc_ci(informative, correct)

    assert point == pytest.approx(float(auroc(informative, correct)))
    assert 0.80 < point < 0.90
    assert lo < point < hi
    assert hi - lo < 0.10


def test_auroc_ci_is_deterministic_under_the_seed() -> None:
    informative, _, correct = _planted()

    first = auroc_ci(informative, correct, seed=3)

    assert auroc_ci(informative, correct, seed=3) == first
    assert auroc_ci(informative, correct, seed=4) != first


def test_auroc_ci_widens_as_alpha_shrinks() -> None:
    informative, _, correct = _planted()

    _, lo95, hi95 = auroc_ci(informative, correct, alpha=0.05)
    _, lo99, hi99 = auroc_ci(informative, correct, alpha=0.01)

    assert lo99 < lo95
    assert hi99 > hi95


def test_replicates_skip_single_class_resamples() -> None:
    scores = np.array([0.1, 0.5, 0.9])
    labels = np.array([0, 0, 1], dtype=np.int64)

    reps = auroc_replicates(scores, labels, n_boot=200, seed=0)

    assert 0 < reps.shape[0] < 200
    assert np.all(np.isfinite(reps))


def test_replicates_hold_every_resample_when_both_classes_abound() -> None:
    informative, _, correct = _planted()

    reps = auroc_replicates(informative, correct, n_boot=50, seed=0)

    assert reps.shape == (50,)


def test_all_three_raise_when_no_resample_holds_both_classes() -> None:
    scores = np.array([0.1, 0.5, 0.9])
    labels = np.zeros(3, dtype=np.int64)

    with pytest.raises(ValueError, match=NO_RESAMPLE):
        auroc_replicates(scores, labels, n_boot=20, seed=0)
    with pytest.raises(ValueError, match=NO_RESAMPLE):
        auroc_ci(scores, labels, n_boot=20)
    with pytest.raises(ValueError, match=NO_RESAMPLE):
        auroc_diff_ci(scores, scores, labels, n_boot=20)


def test_diff_ci_excludes_zero_against_noise() -> None:
    informative, noise, correct = _planted()

    diff, lo, hi = auroc_diff_ci(informative, noise, correct)

    assert diff == pytest.approx(
        float(auroc(informative, correct)) - float(auroc(noise, correct))
    )
    assert 0.2 < diff < 0.6
    assert lo > 0.0
    assert lo < diff < hi


def test_diff_ci_resamples_both_scores_on_the_same_items() -> None:
    informative, _, correct = _planted()

    assert auroc_diff_ci(informative, informative, correct) == (0.0, 0.0, 0.0)


def test_diff_ci_flips_when_the_scores_swap() -> None:
    informative, noise, correct = _planted()

    diff, lo, hi = auroc_diff_ci(informative, noise, correct)
    flipped, flipped_lo, flipped_hi = auroc_diff_ci(noise, informative, correct)

    assert flipped == pytest.approx(-diff)
    assert flipped_lo == pytest.approx(-hi)
    assert flipped_hi == pytest.approx(-lo)


def test_oracle_confidence_has_no_risk_up_to_the_accuracy() -> None:
    _, _, correct = _planted()

    coverage, risk = risk_coverage(correct.astype(np.float64), correct)

    n_correct = int(correct.sum())
    assert np.all(risk[:n_correct] == 0.0)
    assert risk[n_correct] > 0.0
    assert coverage[n_correct - 1] == pytest.approx(n_correct / correct.size)


def test_constant_confidence_is_never_in_the_oracle_order() -> None:
    _, _, correct = _planted()
    sorted_correct = np.sort(correct)[::-1].copy()
    error_rate = 1.0 - float(sorted_correct.mean())
    constant = np.zeros(sorted_correct.size)

    _, risk = risk_coverage(constant, sorted_correct)

    assert abs(float(risk.mean()) - error_rate) < 0.05
    assert risk[: int(sorted_correct.sum())].max() > 0.0


def test_risk_coverage_follows_the_confidence_order() -> None:
    confidence = np.array([0.9, 0.8, 0.7, 0.1])
    correct = np.array([1, 1, 0, 1], dtype=np.int64)

    coverage, risk = risk_coverage(confidence, correct)

    np.testing.assert_allclose(coverage, [0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(risk, [0.0, 0.0, 1 / 3, 0.25])


def test_risk_coverage_ends_at_the_error_rate_and_full_coverage() -> None:
    informative, _, correct = _planted()

    coverage, risk = risk_coverage(informative, correct)

    assert coverage.shape == risk.shape == (correct.size,)
    assert coverage[0] == pytest.approx(1 / correct.size)
    assert coverage[-1] == 1.0
    assert risk[-1] == pytest.approx(1.0 - float(correct.mean()))


def test_risk_coverage_is_deterministic_under_the_seed() -> None:
    constant = np.zeros(200)
    correct = np.tile([1, 0, 0, 1, 1], 40).astype(np.int64)

    first = risk_coverage(constant, correct, seed=1)[1]

    np.testing.assert_array_equal(
        risk_coverage(constant, correct, seed=1)[1], first
    )
    assert not np.array_equal(
        risk_coverage(constant, correct, seed=2)[1], first
    )


def test_aurc_is_the_mean_risk() -> None:
    confidence = np.array([0.9, 0.8, 0.7, 0.1])
    correct = np.array([1, 1, 0, 1], dtype=np.int64)

    assert aurc(confidence, correct) == pytest.approx(
        (0 + 0 + 1 / 3 + 0.25) / 4
    )


def test_aurc_is_lower_for_an_informative_score() -> None:
    informative, noise, correct = _planted()

    assert aurc(informative, correct) < aurc(noise, correct) - 0.1


def test_aurc_of_a_constant_score_is_the_error_rate() -> None:
    _, _, correct = _planted()
    error_rate = 1.0 - float(correct.mean())

    assert abs(aurc(np.zeros(correct.size), correct) - error_rate) < 0.05


def test_risk_at_reads_the_first_item_past_each_coverage() -> None:
    confidence = np.array([0.9, 0.8, 0.7, 0.1])
    correct = np.array([1, 1, 0, 1], dtype=np.int64)
    coverage, risk = risk_coverage(confidence, correct)

    at = risk_at(coverage, risk, (1, 25, 50, 75, 100))

    np.testing.assert_allclose(at, [0.0, 0.0, 0.0, 1 / 3, 0.25])


def test_risk_at_is_exact_where_float_division_overshoots() -> None:
    risk = np.arange(100, dtype=np.float64)
    coverage = (np.arange(100) + 1) / 100

    assert risk_at(coverage, risk, (55,))[0] == 54.0


def test_risk_band_brackets_the_point() -> None:
    informative, _, correct = _planted()

    point, lo, hi = risk_band(informative, correct, GRID, n_boot=100)

    assert point.shape == lo.shape == hi.shape == (len(GRID),)
    assert np.all(lo <= point)
    assert np.all(point <= hi)
    assert hi[-1] > lo[-1]


def test_risk_band_is_deterministic_under_the_seed() -> None:
    informative, _, correct = _planted(n=200)

    first = risk_band(informative, correct, GRID, n_boot=40, seed=2)
    second = risk_band(informative, correct, GRID, n_boot=40, seed=2)

    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


def test_stacking_noise_is_train_fitted_and_test_scored() -> None:
    informative, _, correct = _planted()
    rng = np.random.default_rng(1)
    noise_cols = rng.standard_normal((correct.size, 2))
    with_signal = np.column_stack([noise_cols, informative])
    train, test = slice(0, 400), slice(400, None)

    noise_only = stack_scores(
        noise_cols[train], correct[train], noise_cols[test]
    )
    lifted = stack_scores(with_signal[train], correct[train], with_signal[test])

    assert noise_only.shape == lifted.shape == (200,)
    assert float(auroc(noise_only, correct[test])) < 0.60
    assert float(auroc(lifted, correct[test])) > 0.80
    diff, lo, _ = auroc_diff_ci(lifted, noise_only, correct[test])
    assert diff > 0.0
    assert lo > 0.0


def test_stack_scores_does_not_use_the_other_test_rows() -> None:
    informative, _, correct = _planted()
    features = np.column_stack([informative, informative**2])

    full = stack_scores(features[:400], correct[:400], features[400:])
    head = stack_scores(features[:400], correct[:400], features[400:450])

    np.testing.assert_allclose(head, full[:50])


def test_stack_scores_is_deterministic() -> None:
    informative, _, correct = _planted()
    features = informative[:, None]

    first = stack_scores(features[:400], correct[:400], features[400:])

    np.testing.assert_array_equal(
        stack_scores(features[:400], correct[:400], features[400:]), first
    )


def test_orient_keeps_a_score_that_ranks_by_the_labels() -> None:
    train_scores = np.array([0.1, 0.2, 0.8, 0.9])
    train_labels = np.array([0, 0, 1, 1], dtype=np.int64)
    test_scores = np.array([0.3, 0.7])

    np.testing.assert_array_equal(
        orient_on_train(train_scores, train_labels, test_scores), test_scores
    )


def test_orient_negates_a_score_that_ranks_against_the_labels() -> None:
    train_scores = np.array([0.9, 0.8, 0.2, 0.1])
    train_labels = np.array([0, 0, 1, 1], dtype=np.int64)
    test_scores = np.array([0.3, 0.7])

    np.testing.assert_array_equal(
        orient_on_train(train_scores, train_labels, test_scores), -test_scores
    )


def test_orient_keeps_a_score_at_chance() -> None:
    train_scores = np.array([0.5, 0.5, 0.5, 0.5])
    train_labels = np.array([0, 1, 0, 1], dtype=np.int64)
    test_scores = np.array([0.3, 0.7])

    np.testing.assert_array_equal(
        orient_on_train(train_scores, train_labels, test_scores), test_scores
    )


def test_group_rate_is_the_smoothed_rate_of_the_key() -> None:
    train_keys = ["a", "a", "b"]
    train_correct = np.array([1, 0, 1], dtype=np.int64)
    overall = 2 / 3

    rates = group_rate(train_keys, train_correct, ["b", "a"])

    np.testing.assert_allclose(rates, [(1 + overall) / 2, (1 + overall) / 3])


def test_group_rate_gives_an_unseen_key_the_overall_rate() -> None:
    train_keys = ["a", "a", "b"]
    train_correct = np.array([1, 0, 1], dtype=np.int64)

    rates = group_rate(train_keys, train_correct, ["c"])

    np.testing.assert_allclose(rates, [2 / 3])
