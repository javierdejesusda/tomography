from collections.abc import Callable
import functools
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from tomography.p9.risk import auroc_diff_ci
from tomography.reading.metrics import best_layer
from tomography.reading.stats import LayerChoice
from tomography.reading.stats import best_validation_layer
from tomography.reading.stats import delong_test
from tomography.reading.stats import holm_correction
from tomography.reading.stats import mcnemar_test
from tomography.reading.stats import paired_accuracy_diff
from tomography.reading.stats import paired_auroc_diff
from tomography.reading.stats import threshold_metrics
from tomography.reading.stats import top_k_layers
from tomography.reading.stats import tpr_at_fpr


def test_paired_auroc_diff_with_one_row_per_cluster_is_item_bootstrap() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=40)
    a = labels + rng.standard_normal(40)
    b = 0.5 * labels + rng.standard_normal(40)

    result = paired_auroc_diff(a, b, labels, np.arange(40), n_boot=500, seed=3)

    expected = auroc_diff_ci(a, b, labels, n_boot=500, seed=3)
    assert (result.difference, result.lower, result.upper) == expected
    assert result.n_resamples == 500


def test_paired_auroc_diff_resamples_the_rows_of_a_cluster_together() -> None:
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, size=30)
    a = labels + rng.standard_normal(30)
    b = rng.standard_normal(30)
    copies = rng.permutation(np.repeat(np.arange(30), 2))

    single = paired_auroc_diff(a, b, labels, np.arange(30), n_boot=400, seed=5)
    clustered = paired_auroc_diff(
        a[copies], b[copies], labels[copies], 7 * copies + 3, n_boot=400, seed=5
    )
    independent = paired_auroc_diff(
        a[copies], b[copies], labels[copies], np.arange(60), n_boot=400, seed=5
    )

    assert clustered == single
    assert independent.upper - independent.lower < single.upper - single.lower


def test_paired_auroc_diff_skips_resamples_that_hold_one_class() -> None:
    labels = np.array([1, 1, 0, 0])
    a = np.array([0.9, 0.8, 0.1, 0.2])
    b = np.array([0.3, 0.6, 0.5, 0.4])
    clusters = np.array([0, 0, 1, 1])

    result = paired_auroc_diff(a, b, labels, clusters, n_boot=200, seed=0)

    draws = np.random.default_rng(0).integers(2, size=(200, 2))
    assert result == (0.5, 0.5, 0.5, int((draws[:, 0] != draws[:, 1]).sum()))


def test_paired_auroc_diff_needs_a_resample_with_both_classes() -> None:
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.1, 0.2])

    with pytest.raises(ValueError, match="no resample holds both classes"):
        # Seed 0 draws cluster 1 twice, a resample without positives.
        paired_auroc_diff(
            scores, scores, labels, np.array([0, 0, 1, 1]), n_boot=1, seed=0
        )


@pytest.mark.parametrize(
    ("b", "labels", "clusters", "kwargs", "message"),
    [
        (np.zeros(3), [0, 1, 0, 1], np.arange(4), {}, "same length"),
        (np.zeros(4), [0, 1, 0], np.arange(4), {}, "labels"),
        (np.zeros(4), [1, 1, 1, 1], np.arange(4), {}, "both classes"),
        (np.zeros(4), [0, 1, 0, 1], np.arange(3), {}, "clusters"),
        (np.zeros(4), [0, 1, 0, 1], np.zeros(4), {}, "integers"),
        (np.array([0, 0, np.nan, 0]), [0, 1, 0, 1], np.arange(4), {}, "finite"),
        (np.zeros(4), [0, 1, 0, 1], np.arange(4), {"n_boot": 0}, "n_boot"),
        (np.zeros(4), [0, 1, 0, 1], np.arange(4), {"alpha": 0.0}, "alpha"),
        (np.zeros(4), [0, 1, 0, 1], np.arange(4), {"alpha": 1.0}, "alpha"),
    ],
    ids=[
        "scorer-length",
        "labels-length",
        "one-class",
        "clusters-length",
        "float-clusters",
        "nan-score",
        "no-resamples",
        "alpha-zero",
        "alpha-one",
    ],
)
def test_paired_auroc_diff_rejects_malformed_input(
    b: NDArray[np.float64],
    labels: list[int],
    clusters: NDArray[np.generic],
    kwargs: dict[str, float],
    message: str,
) -> None:
    a = np.array([0.1, 0.9, 0.2, 0.8])

    with pytest.raises(ValueError, match=message):
        paired_auroc_diff(
            a,
            b,
            np.array(labels, dtype=np.int64),
            clusters,  # type: ignore[arg-type]
            **kwargs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("correct_a", "correct_b", "clusters", "kwargs", "message"),
    [
        (np.ones(4), np.ones(3), np.arange(4), {}, "same length"),
        (np.ones(0), np.ones(0), np.arange(0), {}, "at least one row"),
        (np.ones(4), np.ones(4), np.arange(3), {}, "clusters"),
        (np.ones(4), np.ones(4), np.zeros(4), {}, "integers"),
        (np.ones(4), np.array([1.0, 0.0, 2.0, 1.0]), np.arange(4), {}, "0, 1"),
        (
            np.array([1.0, np.nan, 0.0, 1.0]),
            np.ones(4),
            np.arange(4),
            {},
            "0, 1",
        ),
        (np.ones(4), np.ones(4), np.arange(4), {"n_boot": 0}, "n_boot"),
        (np.ones(4), np.ones(4), np.arange(4), {"alpha": 1.0}, "alpha"),
    ],
    ids=[
        "scorer-length",
        "empty",
        "clusters-length",
        "float-clusters",
        "credit-above-one",
        "nan-credit",
        "no-resamples",
        "alpha-one",
    ],
)
def test_paired_accuracy_diff_rejects_malformed_input(
    correct_a: NDArray[np.float64],
    correct_b: NDArray[np.float64],
    clusters: NDArray[np.generic],
    kwargs: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        paired_accuracy_diff(
            correct_a,
            correct_b,
            clusters,  # type: ignore[arg-type]
            **kwargs,  # type: ignore[arg-type]
        )


def test_paired_accuracy_diff_with_one_row_per_cluster_is_item_bootstrap() -> (
    None
):
    rng = np.random.default_rng(2)
    correct_a = rng.random(50) < 0.8
    correct_b = rng.random(50) < 0.6

    result = paired_accuracy_diff(
        correct_a, correct_b, np.arange(50), n_boot=300, seed=4
    )

    rows = np.random.default_rng(4).integers(50, size=(300, 50))
    diffs = correct_a[rows].mean(axis=1) - correct_b[rows].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    assert result.difference == pytest.approx(
        correct_a.mean() - correct_b.mean()
    )
    assert result.lower == pytest.approx(lo)
    assert result.upper == pytest.approx(hi)
    assert result.n_resamples == 300


def test_paired_accuracy_diff_pools_the_rows_of_unequal_clusters() -> None:
    credit_a = np.array([1.0, 0.5, 1.0, 0.0, 1.0, 1.0, 0.0])
    credit_b = np.array([0.0, 0.5, 1.0, 1.0, 0.0, 0.0, 1.0])
    clusters = np.array([4, 4, 9, 1, 1, 1, 9])

    result = paired_accuracy_diff(
        credit_a, credit_b, clusters, n_boot=200, alpha=0.2, seed=8
    )

    members = [np.flatnonzero(clusters == c) for c in (1, 4, 9)]
    draws = np.random.default_rng(8).integers(3, size=(200, 3))
    diffs = []
    for drawn in draws:
        rows = np.concatenate([members[c] for c in drawn])
        diffs.append(credit_a[rows].mean() - credit_b[rows].mean())
    lo, hi = np.percentile(diffs, [10, 90])
    assert result.difference == pytest.approx(4.5 / 7 - 3.5 / 7)
    assert result.lower == pytest.approx(lo)
    assert result.upper == pytest.approx(hi)
    assert result.n_resamples == 200


def _delong_by_definition(
    a: NDArray[np.float64], b: NDArray[np.float64], labels: NDArray[np.int64]
) -> tuple[float, float]:
    positive = labels == 1
    over_positives = []
    over_negatives = []
    for scores in (a, b):
        x = scores[positive][:, None]
        y = scores[~positive][None, :]
        kernel = (x > y) + 0.5 * (x == y)
        over_positives.append(kernel.mean(axis=1))
        over_negatives.append(kernel.mean(axis=0))
    covariance = (
        np.cov(over_positives) / positive.sum()
        + np.cov(over_negatives) / (~positive).sum()
    )
    contrast = np.array([1.0, -1.0])
    difference = over_positives[0].mean() - over_positives[1].mean()
    return float(difference), float(np.sqrt(contrast @ covariance @ contrast))


def test_delong_test_matches_the_hand_computed_example() -> None:
    labels = np.array([1, 0, 1, 0, 1])
    a = np.array([3.0, 1.0, 4.0, 2.0, 5.0])
    b = np.array([0.2, 0.5, 0.6, 0.1, 0.9])

    result = delong_test(a, b, labels)

    # The components of a are all 1; those of b are (1/2, 1, 1) over the
    # positives and (2/3, 1) over the negatives, so the variance of the
    # difference is (1/12) / 3 + (1/18) / 2 = 1/18.
    assert result.difference == pytest.approx(1 / 6)
    assert result.standard_error == pytest.approx(np.sqrt(1 / 18))
    assert result.z == pytest.approx(np.sqrt(2) / 2)
    assert result.p_value == pytest.approx(math.erfc(0.5))


def test_delong_test_matches_the_structural_components_by_definition() -> None:
    rng = np.random.default_rng(6)
    labels = rng.permutation(np.repeat([1, 0], [23, 31]))
    a = (rng.integers(0, 6, size=54) + 2 * labels).astype(np.float64)
    b = rng.standard_normal(54) + labels

    result = delong_test(a, b, labels)

    difference, standard_error = _delong_by_definition(a, b, labels)
    z = difference / standard_error
    assert result.difference == pytest.approx(difference)
    assert result.standard_error == pytest.approx(standard_error)
    assert result.z == pytest.approx(z)
    assert result.p_value == pytest.approx(math.erfc(abs(z) / math.sqrt(2)))


def test_delong_test_without_variance_rejects_only_a_real_difference() -> None:
    labels = np.array([1, 1, 0, 0])
    perfect = np.array([2.0, 3.0, 0.0, 1.0])

    same = delong_test(perfect, perfect, labels)
    apart = delong_test(perfect, np.full(4, 0.5), labels)

    assert same == (0.0, 0.0, 0.0, 1.0)
    assert apart == (0.5, 0.0, math.inf, 0.0)


@pytest.mark.parametrize(
    ("b", "labels", "message"),
    [
        (np.zeros(3), [1, 1, 0, 0], "same length"),
        (np.zeros(4), [1, 1, 0], "labels"),
        (np.zeros(4), [1, 0, 0, 0], "two positives"),
        (np.zeros(4), [1, 1, 1, 0], "two negatives"),
        (np.array([0, 0, np.inf, 0]), [1, 1, 0, 0], "finite"),
    ],
    ids=[
        "scorer-length",
        "labels-length",
        "one-positive",
        "one-negative",
        "infinite-score",
    ],
)
def test_delong_test_rejects_malformed_input(
    b: NDArray[np.float64], labels: list[int], message: str
) -> None:
    a = np.array([0.4, 0.9, 0.1, 0.3])

    with pytest.raises(ValueError, match=message):
        delong_test(a, b, np.array(labels, dtype=np.int64))


def _outcomes(
    a_only: int, b_only: int, both: int, neither: int
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    counts = [a_only, b_only, both, neither]
    correct_a = np.repeat([True, False, True, False], counts)
    correct_b = np.repeat([False, True, True, False], counts)
    order = np.random.default_rng(0).permutation(sum(counts))
    return correct_a[order], correct_b[order]


@pytest.mark.parametrize(
    ("a_only", "b_only", "p_value"),
    [
        (1, 9, 2 * (1 + 10) / 2**10),
        (9, 1, 2 * (1 + 10) / 2**10),
        (3, 5, 2 * (1 + 8 + 28 + 56) / 2**8),
        (0, 4, 2 * 1 / 2**4),
        (5, 5, 1.0),
    ],
    ids=["mostly-b", "mostly-a", "close", "all-b", "even"],
)
def test_mcnemar_test_runs_the_exact_binomial_on_discordant_items(
    a_only: int, b_only: int, p_value: float
) -> None:
    correct_a, correct_b = _outcomes(a_only, b_only, both=6, neither=4)

    result = mcnemar_test(correct_a, correct_b)

    assert result.a_only == a_only
    assert result.b_only == b_only
    assert result.p_value == pytest.approx(p_value)


def test_mcnemar_test_without_discordant_items_finds_no_difference() -> None:
    correct_a, correct_b = _outcomes(0, 0, both=3, neither=2)

    assert mcnemar_test(correct_a, correct_b) == (0, 0, 1.0)


@pytest.mark.parametrize(
    ("correct_a", "correct_b", "message"),
    [
        (np.array([True, False]), np.array([True]), "same length"),
        (
            np.ones((2, 2), dtype=bool),
            np.ones((2, 2), dtype=bool),
            "same length",
        ),
        (
            np.array([], dtype=bool),
            np.array([], dtype=bool),
            "at least one row",
        ),
        (np.array([1.0, 0.5]), np.array([1.0, 0.0]), "0 or 1"),
    ],
    ids=["length", "matrix", "empty", "tie-credit"],
)
def test_mcnemar_test_rejects_malformed_input(
    correct_a: NDArray[np.generic], correct_b: NDArray[np.generic], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        mcnemar_test(correct_a, correct_b)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("p_values", "alpha", "adjusted", "reject"),
    [
        # Sorted, 0.005, 0.01, 0.03, 0.04 times 4, 3, 2, 1 give 0.02, 0.03,
        # 0.06, 0.04, and the running maximum gives 0.02, 0.03, 0.06, 0.06.
        (
            [0.01, 0.04, 0.03, 0.005],
            0.05,
            [0.03, 0.06, 0.06, 0.02],
            [True, False, False, True],
        ),
        (
            [0.5, 0.02, 0.02, 0.9],
            0.1,
            [1.0, 0.08, 0.08, 1.0],
            [False, True, True, False],
        ),
        ([0.03], 0.05, [0.03], [True]),
        ([0.2, 0.3], 0.05, [0.4, 0.4], [False, False]),
    ],
    ids=["hand-example", "ties-and-cap", "single", "none-rejected"],
)
def test_holm_correction_matches_the_hand_computed_step_down(
    p_values: list[float],
    alpha: float,
    adjusted: list[float],
    reject: list[bool],
) -> None:
    result = holm_correction(np.array(p_values), alpha=alpha)

    np.testing.assert_allclose(result.adjusted, adjusted)
    np.testing.assert_array_equal(result.reject, reject)


@pytest.mark.parametrize(
    ("p_values", "alpha", "message"),
    [
        (np.zeros(0), 0.05, "non-empty vector"),
        (np.full((2, 2), 0.1), 0.05, "non-empty vector"),
        (np.array([0.2, 1.2]), 0.05, r"\[0, 1\]"),
        (np.array([-0.1, 0.2]), 0.05, r"\[0, 1\]"),
        (np.array([0.2, np.nan]), 0.05, r"\[0, 1\]"),
        (np.array([0.2]), 0.0, "alpha"),
        (np.array([0.2]), 1.0, "alpha"),
    ],
    ids=[
        "empty-family",
        "matrix",
        "above-one",
        "negative",
        "nan",
        "alpha-zero",
        "alpha-one",
    ],
)
def test_holm_correction_rejects_malformed_input(
    p_values: NDArray[np.float64], alpha: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        holm_correction(p_values, alpha=alpha)


def test_threshold_metrics_match_the_hand_computed_example() -> None:
    val_scores = np.array([0.1, 0.4, 0.35, 0.8])
    val_labels = np.array([0, 0, 1, 1])
    test_scores = np.array(
        [0.9, 0.5, 0.3, 0.6, 0.2, 0.1, 0.05, 0.15, 0.22, 0.0]
    )
    test_labels = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0])

    result = threshold_metrics(val_scores, val_labels, test_scores, test_labels)

    # On validation, the splits 0.1 | 0.35 and 0.4 | 0.8 both reach a
    # balanced accuracy of 3/4; the first is kept, at its midpoint. On test
    # the rule gives 3 true positives, 1 false positive, 2 false negatives
    # and 4 true negatives.
    assert result.threshold == pytest.approx(0.225)
    assert result.accuracy == pytest.approx(7 / 10)
    assert result.precision == pytest.approx(3 / 4)
    assert result.recall == pytest.approx(3 / 5)
    assert result.f1 == pytest.approx(2 / 3)


def test_threshold_metrics_maximize_validation_balanced_accuracy() -> None:
    rng = np.random.default_rng(7)
    val_labels = rng.integers(0, 2, size=80)
    val_scores = np.round(val_labels + rng.standard_normal(80), 1)
    test_labels = rng.integers(0, 2, size=60)
    test_scores = test_labels + rng.standard_normal(60)

    result = threshold_metrics(val_scores, val_labels, test_scores, test_labels)

    def balanced(threshold: float) -> float:
        called = (val_scores > threshold).astype(np.int64)
        return float(balanced_accuracy_score(val_labels, called))

    best = max(balanced(t) for t in np.r_[-np.inf, np.unique(val_scores)])
    below = val_scores[val_scores < result.threshold].max()
    above = val_scores[val_scores > result.threshold].min()
    called = (test_scores > result.threshold).astype(np.int64)
    assert balanced(result.threshold) == pytest.approx(best)
    assert result.threshold == pytest.approx((below + above) / 2)
    assert result.accuracy == pytest.approx(accuracy_score(test_labels, called))
    assert result.precision == pytest.approx(
        precision_score(test_labels, called)
    )
    assert result.recall == pytest.approx(recall_score(test_labels, called))
    assert result.f1 == pytest.approx(f1_score(test_labels, called))


def test_threshold_metrics_without_positive_calls_have_zero_precision() -> None:
    result = threshold_metrics(
        np.array([0.0, 1.0]),
        np.array([0, 1]),
        np.array([0.1, 0.2, 0.3]),
        np.array([0, 1, 1]),
    )

    assert result == (0.5, 1 / 3, 0.0, 0.0, 0.0)


def test_threshold_metrics_call_every_row_positive_for_an_inverted_scorer() -> (
    None
):
    result = threshold_metrics(
        np.array([0.9, 0.8, 0.2, 0.1]),
        np.array([0, 0, 1, 1]),
        np.array([0.5, 0.1, 0.7]),
        np.array([1, 0, 1]),
    )

    assert result == (-math.inf, 2 / 3, 2 / 3, 1.0, 0.8)


SPLITS = {
    "val_scores": [0.1, 0.9, 0.2, 0.8],
    "val_labels": [0, 1, 0, 1],
    "test_scores": [0.3, 0.7, 0.6],
    "test_labels": [0, 1, 1],
}
MALFORMED_SPLITS = [
    ({"val_labels": [0, 1, 0]}, "same length"),
    ({"test_scores": [0.3, 0.7]}, "same length"),
    ({"test_labels": [1, 1, 1]}, "both classes"),
    ({"val_labels": [0, 2, 0, 1]}, "0 or 1"),
    ({"val_scores": [0.1, np.nan, 0.2, 0.8]}, "finite"),
    ({"test_scores": [0.3, np.inf, 0.6]}, "finite"),
]
MALFORMED_IDS = [
    "val-length",
    "test-length",
    "test-one-class",
    "not-binary",
    "nan-val-score",
    "infinite-test-score",
]


def _splits(override: dict[str, list[float]]) -> list[NDArray[np.generic]]:
    arguments = {**SPLITS, **override}
    return [np.array(arguments[name]) for name in SPLITS]


@pytest.mark.parametrize(
    ("override", "message"),
    [*MALFORMED_SPLITS, ({"val_labels": [1, 1, 1, 1]}, "both classes")],
    ids=[*MALFORMED_IDS, "val-one-class"],
)
def test_threshold_metrics_rejects_malformed_input(
    override: dict[str, list[float]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        threshold_metrics(*_splits(override))  # type: ignore[arg-type]


def test_tpr_at_fpr_sets_the_threshold_on_the_validation_negatives() -> None:
    val_scores = np.r_[np.arange(256.0), 300.0, 100.0]
    val_labels = np.r_[np.zeros(256, dtype=np.int64), 1, 1]
    test_scores = np.array([254.0, 260.0, 253.0, 100.0, 255.0, 10.0, 20.0])
    test_labels = np.array([1, 1, 1, 1, 0, 0, 0])

    result = tpr_at_fpr(val_scores, val_labels, test_scores, test_labels)

    # 1% of 256 negatives lets 2 of them above the threshold, which is the
    # third highest negative, 253; a score equal to it is called negative.
    assert result == (2 / 4, 1 / 3, 253.0, 2)


@pytest.mark.parametrize(
    ("negatives", "max_fpr", "threshold", "negatives_above"),
    [
        ([5.0, 5.0, 5.0, 1.0, 2.0], 0.4, 5.0, 0),
        (np.arange(50.0), 0.01, 49.0, 0),
        (np.arange(100.0), 0.29, 70.0, 29),
    ],
    ids=["tie-at-threshold", "too-few-negatives", "decimal-rate"],
)
def test_tpr_at_fpr_lets_at_most_the_allowed_negatives_through(
    negatives: list[float],
    max_fpr: float,
    threshold: float,
    negatives_above: int,
) -> None:
    val_scores = np.array(negatives)

    result = tpr_at_fpr(
        val_scores,
        np.zeros(val_scores.shape[0], dtype=np.int64),
        np.array([-1.0, 1000.0]),
        np.array([0, 1]),
        max_fpr=max_fpr,
    )

    assert result == (1.0, 0.0, threshold, negatives_above)


@pytest.mark.parametrize(
    ("override", "kwargs", "message"),
    [
        *[(override, {}, message) for override, message in MALFORMED_SPLITS],
        ({"val_labels": [1, 1, 1, 1]}, {}, "negative"),
        ({}, {"max_fpr": 0.0}, "max_fpr"),
        ({}, {"max_fpr": 1.0}, "max_fpr"),
    ],
    ids=[*MALFORMED_IDS, "no-val-negative", "fpr-zero", "fpr-one"],
)
def test_tpr_at_fpr_rejects_malformed_input(
    override: dict[str, list[float]],
    kwargs: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        tpr_at_fpr(*_splits(override), **kwargs)  # type: ignore[arg-type]


def test_best_validation_layer_takes_the_first_maximum() -> None:
    val_metric = np.array([0.6, 0.9, 0.9, 0.7])
    layers = [3, 6, 9, 12]
    val_scores = np.arange(12.0).reshape(4, 3)
    test_scores = -np.arange(8.0).reshape(4, 2)

    choice = best_validation_layer(val_metric, layers, val_scores, test_scores)

    assert choice.layers == (best_layer(val_metric, layers),)
    assert choice.layers == (6,)
    np.testing.assert_array_equal(choice.val_scores, [3.0, 4.0, 5.0])
    np.testing.assert_array_equal(choice.test_scores, [-2.0, -3.0])


def test_top_k_layers_average_scores_standardized_on_validation() -> None:
    val_metric = np.array([0.6, 0.9, 0.8])
    val_scores = np.array([[5.0, 7.0], [1.0, 3.0], [10.0, 30.0]])
    test_scores = np.array(
        [[0.0, 0.0, 0.0], [2.0, 5.0, 0.0], [40.0, 0.0, 20.0]]
    )

    choice = top_k_layers(val_metric, [4, 8, 12], val_scores, test_scores, k=2)

    # Layer 8 is standardized with mean 2 and deviation 1, layer 12 with
    # mean 20 and deviation 10.
    assert choice.layers == (8, 12)
    np.testing.assert_allclose(choice.val_scores, [-1.0, 1.0])
    np.testing.assert_allclose(choice.test_scores, [1.0, 0.5, -1.0])


def test_top_k_layers_ignore_the_scale_of_each_layer() -> None:
    rng = np.random.default_rng(9)
    val_scores = rng.standard_normal((6, 20))
    test_scores = rng.standard_normal((6, 15))
    val_metric = np.array([0.5, 0.8, 0.7, 0.8, 0.6, 0.75])
    scale = np.array([[1.0], [1000.0], [0.01], [1.0], [5.0], [3.0]])

    plain = top_k_layers(val_metric, range(6), val_scores, test_scores, k=3)
    scaled = top_k_layers(
        val_metric,
        range(6),
        scale * val_scores + 7.0,
        scale * test_scores + 7.0,
        k=3,
    )

    assert plain.layers == scaled.layers == (1, 3, 5)
    np.testing.assert_allclose(scaled.val_scores, plain.val_scores)
    np.testing.assert_allclose(scaled.test_scores, plain.test_scores)


LAYER_INPUTS: dict[str, list[Any]] = {
    "val_metric": [0.6, 0.9, 0.8],
    "layers": [4, 8, 12],
    "val_scores": [[5.0, 7.0], [1.0, 3.0], [10.0, 30.0]],
    "test_scores": [[0.0], [2.0], [40.0]],
}


def _layer_inputs(override: dict[str, list[Any]]) -> list[Any]:
    arguments = {**LAYER_INPUTS, **override}
    return [
        np.array(arguments["val_metric"]),
        arguments["layers"],
        np.array(arguments["val_scores"]),
        np.array(arguments["test_scores"]),
    ]


@pytest.mark.parametrize(
    "choose",
    [best_validation_layer, functools.partial(top_k_layers, k=2)],
    ids=["best", "top-k"],
)
@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"layers": [4, 8]}, "one entry per layer"),
        ({"val_metric": [0.6, np.nan, 0.8]}, "NaN"),
        ({"layers": [4, 4, 12]}, "distinct"),
        ({"val_scores": [[5.0, 7.0], [1.0, 3.0]]}, "val_scores"),
        ({"test_scores": [0.0, 2.0, 40.0]}, "test_scores"),
        ({"test_scores": [[0.0], [np.inf], [40.0]]}, "finite"),
    ],
    ids=[
        "layers-length",
        "nan-metric",
        "repeated-layer",
        "val-rows",
        "test-vector",
        "infinite-score",
    ],
)
def test_layer_choices_reject_malformed_input(
    choose: Callable[..., LayerChoice],
    override: dict[str, list[Any]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        choose(*_layer_inputs(override))


@pytest.mark.parametrize(
    ("k", "val_scores", "message"),
    [
        (0, LAYER_INPUTS["val_scores"], "k must be"),
        (4, LAYER_INPUTS["val_scores"], "k must be"),
        (2, [[5.0, 7.0], [3.0, 3.0], [10.0, 30.0]], "constant"),
    ],
    ids=["no-layer", "more-than-layers", "constant-layer"],
)
def test_top_k_layers_rejects_bad_k_and_constant_layers(
    k: int, val_scores: list[list[float]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        top_k_layers(*_layer_inputs({"val_scores": val_scores}), k=k)
