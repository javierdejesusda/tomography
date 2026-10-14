import numpy as np
from numpy.typing import NDArray
import pytest
from sklearn.metrics import roc_auc_score

from tomography.reading.metrics import auroc
from tomography.reading.metrics import group_accuracy


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
    ],
    ids=[
        "no-positive",
        "two-positives",
        "index-out-of-range",
        "labels-length",
        "nan-score",
        "no-groups",
        "no-columns",
    ],
)
def test_group_accuracy_rejects_malformed_input(
    groups: NDArray[np.int64],
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
