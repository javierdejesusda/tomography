import numpy as np
from numpy.typing import NDArray
import pytest
from sklearn.metrics import roc_auc_score

from tomography.reading.metrics import auroc
from tomography.reading.metrics import best_layer
from tomography.reading.metrics import bootstrap_ci
from tomography.reading.metrics import cosine_matrix
from tomography.reading.metrics import group_accuracy
from tomography.reading.metrics import wilson_ci


def i64(*rows: list[int]) -> NDArray[np.int64]:
    return np.array(rows, dtype=np.int64)


def test_group_accuracy_counts_groups_where_the_positive_member_wins() -> None:
    scores = np.array([0.9, 0.1, 0.5, 0.7, 0.2, 0.8])
    groups = i64([0, 1], [2, 3], [4, 5])
    labels = np.array([1, 0, 0, 1, 1, 0], dtype=np.int64)

    acc = group_accuracy(scores, groups, labels)

    assert float(acc) == pytest.approx(2 / 3)
    assert acc.shape == ()


def test_group_accuracy_splits_credit_among_tied_maxima() -> None:
    scores = np.array([1.0, 1.0, 1.0, 0.0])
    groups = i64([0, 1, 2, 3])
    labels = np.array([0, 0, 1, 0], dtype=np.int64)

    assert float(group_accuracy(scores, groups, labels)) == pytest.approx(1 / 3)


def test_tied_maxima_without_the_positive_earn_no_credit() -> None:
    scores = np.array([1.0, 1.0, 0.0, 0.5])
    groups = i64([0, 1, 2, 3])
    labels = np.array([0, 0, 1, 0], dtype=np.int64)

    assert float(group_accuracy(scores, groups, labels)) == 0.0


@pytest.mark.parametrize("k", [2, 3, 4], ids=["pairs", "triples", "quads"])
def test_constant_scorer_sits_at_chance(k: int) -> None:
    n_groups = 12
    groups = np.arange(n_groups * k, dtype=np.int64).reshape(n_groups, k)
    labels = np.zeros(n_groups * k, dtype=np.int64)
    labels[::k] = 1

    acc = group_accuracy(np.full(n_groups * k, 3.5), groups, labels)

    assert float(acc) == pytest.approx(1 / k)


def test_group_accuracy_ignores_padding_in_ragged_groups() -> None:
    scores = np.array([2.0, 1.0, 0.0, 5.0, 5.0])
    groups = i64([0, 1, 2, -1], [3, 4, -1, -1])
    labels = np.array([1, 0, 0, 1, 0], dtype=np.int64)

    acc = group_accuracy(scores, groups, labels)

    assert float(acc) == pytest.approx((1.0 + 0.5) / 2)


def test_group_accuracy_keeps_leading_axes_of_the_scores() -> None:
    rng = np.random.default_rng(0)
    scores = rng.standard_normal((3, 4, 10))
    groups = i64([0, 1], [2, 3], [4, 5], [6, 7], [8, 9])
    labels = np.array([1, 0, 0, 1, 1, 0, 0, 1, 1, 0], dtype=np.int64)

    acc = group_accuracy(scores, groups, labels)

    assert acc.shape == (3, 4)
    for seed_index in range(3):
        for layer_index in range(4):
            single = group_accuracy(
                scores[seed_index, layer_index], groups, labels
            )
            assert acc[seed_index, layer_index] == pytest.approx(float(single))


@pytest.mark.parametrize(
    ("groups", "labels", "scores", "message"),
    [
        (i64([0, 1]), [0, 0], [1.0, 2.0], "exactly one"),
        (i64([0, 1]), [1, 1], [1.0, 2.0], "exactly one"),
        (i64([0, 5]), [1, 0], [1.0, 2.0], "groups"),
        (i64([0, 1]), [1, 0, 1], [1.0, 2.0], "labels"),
        (i64([0, 1]), [1, 0], [1.0, np.nan], "finite"),
        (np.zeros((0, 2), dtype=np.int64), [1, 0], [1.0, 2.0], "groups"),
        (np.zeros((2, 0), dtype=np.int64), [1, 0], [1.0, 2.0], "groups"),
        (np.array([[0.0, 1.0]]), [1, 0], [1.0, 2.0], "groups"),
        (i64([0, -2, 2]), [1, 0, 0], [1.0, 5.0, 0.0], "groups"),
    ],
    ids=[
        "no-positive",
        "two-positives",
        "index-out-of-range",
        "labels-length",
        "nan-score",
        "no-groups",
        "no-columns",
        "float-dtype",
        "index-below-padding",
    ],
)
def test_group_accuracy_rejects_malformed_input(
    groups: NDArray[np.generic],
    labels: list[int],
    scores: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        group_accuracy(
            np.array(scores), groups, np.array(labels, dtype=np.int64)
        )


def test_auroc_matches_the_textbook_example() -> None:
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    labels = np.array([0, 0, 1, 1], dtype=np.int64)

    assert float(auroc(scores, labels)) == pytest.approx(0.75)


def test_auroc_ranks_ties_by_average_rank() -> None:
    labels = np.array([0, 1, 1], dtype=np.int64)

    assert float(auroc(np.array([1.0, 1.0, 2.0]), labels)) == pytest.approx(
        0.75
    )
    assert float(auroc(np.full(3, 7.0), labels)) == pytest.approx(0.5)


def test_auroc_is_one_for_perfect_and_zero_for_inverted_scores() -> None:
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    scores = np.array([-1.0, 0.0, 1.0, 2.0])

    assert float(auroc(scores, labels)) == 1.0
    assert float(auroc(-scores, labels)) == 0.0


def test_auroc_agrees_with_scikit_learn_over_leading_axes() -> None:
    rng = np.random.default_rng(1)
    labels = (rng.random(60) < 0.4).astype(np.int64)
    scores = np.round(rng.standard_normal((3, 5, 60)) + labels, 1)

    result = auroc(scores, labels)

    assert result.shape == (3, 5)
    assert result.dtype == np.float64
    for i in range(3):
        for j in range(5):
            assert result[i, j] == pytest.approx(
                roc_auc_score(labels, scores[i, j])
            )


@pytest.mark.parametrize(
    ("labels", "scores", "message"),
    [
        ([1, 1, 1], [0.1, 0.2, 0.3], "both classes"),
        ([0, 1], [0.1, 0.2, 0.3], "labels"),
        ([0, 1, 2], [0.1, 0.2, 0.3], "labels"),
        ([0, 1], [0.1, np.inf], "finite"),
    ],
    ids=["one-class", "length", "not-binary", "infinite-score"],
)
def test_auroc_rejects_malformed_input(
    labels: list[int], scores: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        auroc(np.array(scores), np.array(labels, dtype=np.int64))


def test_bootstrap_ci_is_exact_for_two_values() -> None:
    values = np.array([0.0, 1.0])

    lo, hi = bootstrap_ci(values, seed=0)

    assert float(lo) == 0.0
    assert float(hi) == 1.0


def test_bootstrap_ci_of_constant_values_collapses_to_the_value() -> None:
    values = np.full((6, 3), 0.7)

    lo, hi = bootstrap_ci(values, seed=0)

    np.testing.assert_allclose(lo, 0.7)
    np.testing.assert_allclose(hi, 0.7)


def test_bootstrap_ci_has_the_width_of_the_normal_interval() -> None:
    rng = np.random.default_rng(2)
    values = rng.standard_normal(200)

    lo, hi = bootstrap_ci(values, seed=1)

    expected_half_width = 1.96 * values.std(ddof=1) / np.sqrt(200)
    assert float(lo) < values.mean() < float(hi)
    assert float(hi - lo) / 2 == pytest.approx(expected_half_width, rel=0.15)


def test_bootstrap_ci_reduces_the_seed_axis_only() -> None:
    values = np.random.default_rng(3).random((10, 4, 5))

    lo, hi = bootstrap_ci(values, seed=0)

    assert lo.shape == hi.shape == (4, 5)
    assert (lo <= values.mean(axis=0)).all()
    assert (values.mean(axis=0) <= hi).all()


def test_bootstrap_ci_is_reproducible_and_depends_on_the_seed() -> None:
    values = np.random.default_rng(4).random((10, 3))

    first = bootstrap_ci(values, seed=5)
    second = bootstrap_ci(values, seed=5)
    other = bootstrap_ci(values, seed=6)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert not np.array_equal(first[0], other[0])


def test_bootstrap_ci_narrows_as_alpha_grows() -> None:
    values = np.random.default_rng(5).random(30)

    narrow = bootstrap_ci(values, alpha=0.5, seed=0)
    wide = bootstrap_ci(values, alpha=0.05, seed=0)

    assert float(narrow[1] - narrow[0]) < float(wide[1] - wide[0])


@pytest.mark.parametrize(
    ("values", "kwargs", "message"),
    [
        (np.zeros(0), {}, "values"),
        (np.zeros(3), {"alpha": 0.0}, "alpha"),
        (np.zeros(3), {"alpha": 1.0}, "alpha"),
        (np.zeros(3), {"n_boot": 0}, "n_boot"),
    ],
    ids=["empty", "alpha-zero", "alpha-one", "no-resamples"],
)
def test_bootstrap_ci_rejects_bad_arguments(
    values: NDArray[np.float64], kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        bootstrap_ci(values, **kwargs)  # type: ignore[arg-type]


def test_wilson_ci_matches_the_textbook_interval() -> None:
    lo, hi = wilson_ci(8, 10)

    assert lo == pytest.approx(0.4902, abs=1e-3)
    assert hi == pytest.approx(0.9433, abs=1e-3)


def test_wilson_ci_takes_fractional_successes_between_its_neighbours() -> None:
    low, mid, high = wilson_ci(7, 10), wilson_ci(7.5, 10), wilson_ci(8, 10)

    assert low[0] < mid[0] < high[0]
    assert low[1] < mid[1] < high[1]


def test_wilson_ci_stays_inside_the_unit_interval_at_the_extremes() -> None:
    assert wilson_ci(0, 20)[0] == 0.0
    assert wilson_ci(20, 20)[1] == 1.0
    assert 0.0 < wilson_ci(0, 20)[1] < 0.2


def test_wilson_ci_is_symmetric_and_shrinks_with_more_trials() -> None:
    lo, hi = wilson_ci(3, 10)
    mirrored_lo, mirrored_hi = wilson_ci(7, 10)
    big_lo, big_hi = wilson_ci(30, 100)

    assert (1 - hi, 1 - lo) == pytest.approx((mirrored_lo, mirrored_hi))
    assert big_hi - big_lo < hi - lo


def test_wilson_ci_widens_with_a_smaller_alpha() -> None:
    narrow = wilson_ci(8, 10, alpha=0.2)
    wide = wilson_ci(8, 10, alpha=0.01)

    assert wide[0] < narrow[0] < narrow[1] < wide[1]


@pytest.mark.parametrize(
    ("successes", "trials", "alpha", "message"),
    [
        (1, 0, 0.05, "trials"),
        (-1, 5, 0.05, "successes"),
        (6, 5, 0.05, "successes"),
        (2, 5, 0.0, "alpha"),
        (2, 5, 1.0, "alpha"),
    ],
    ids=["no-trials", "negative", "too-many", "alpha-zero", "alpha-one"],
)
def test_wilson_ci_rejects_bad_arguments(
    successes: int, trials: int, alpha: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        wilson_ci(successes, trials, alpha=alpha)


def test_best_layer_returns_the_first_maximum_as_a_layer_index() -> None:
    acc = np.array([0.5, 0.9, 0.9, 0.7])

    layer = best_layer(acc, [0, 2, 4, 6])

    assert layer == 2
    assert isinstance(layer, int)


def test_best_layer_accepts_a_single_layer() -> None:
    assert best_layer(np.array([0.3]), (12,)) == 12


@pytest.mark.parametrize(
    ("acc", "layers", "message"),
    [
        (np.array([0.1, 0.2]), [0], "layers"),
        (np.zeros((2, 2)), [0, 1], "acc"),
        (np.zeros(0), [], "acc"),
        (np.array([0.1, np.nan]), [0, 1], "NaN"),
    ],
    ids=["length", "two-dimensional", "empty", "nan"],
)
def test_best_layer_rejects_malformed_input(
    acc: NDArray[np.float64], layers: list[int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        best_layer(acc, layers)


def test_cosine_matrix_of_known_vectors() -> None:
    vectors = np.array([[1.0, 0.0], [0.0, 2.0], [-3.0, 0.0], [1.0, 1.0]])

    cosines = cosine_matrix(vectors)

    assert cosines.shape == (4, 4)
    assert cosines.dtype == np.float64
    np.testing.assert_allclose(np.diag(cosines), 1.0)
    assert cosines[0, 1] == pytest.approx(0.0)
    assert cosines[0, 2] == pytest.approx(-1.0)
    assert cosines[0, 3] == pytest.approx(1 / np.sqrt(2))
    np.testing.assert_allclose(cosines, cosines.T)


def test_cosine_matrix_ignores_vector_scale() -> None:
    vectors = np.random.default_rng(6).standard_normal((5, 16))
    scales = np.array([1.0, 10.0, 0.1, 100.0, 3.0])[:, None]

    np.testing.assert_allclose(
        cosine_matrix(vectors * scales), cosine_matrix(vectors), atol=1e-12
    )


def test_cosine_matrix_stays_inside_the_unit_interval() -> None:
    vectors = np.tile(np.random.default_rng(3).standard_normal(64), (3, 1))

    cosines = cosine_matrix(vectors.astype(np.float32))

    assert cosines.max() <= 1.0
    assert cosines.min() >= -1.0


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        (np.ones(3), "shape"),
        (np.array([[1.0, 0.0], [0.0, 0.0]]), "zero"),
    ],
    ids=["one-dimensional", "zero-vector"],
)
def test_cosine_matrix_rejects_malformed_input(
    vectors: NDArray[np.float64], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        cosine_matrix(vectors)
