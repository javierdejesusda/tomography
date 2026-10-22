import math

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.p9.risk import auroc_diff_ci
from tomography.reading.stats import delong_test
from tomography.reading.stats import holm_correction
from tomography.reading.stats import mcnemar_test
from tomography.reading.stats import paired_accuracy_diff
from tomography.reading.stats import paired_auroc_diff


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
