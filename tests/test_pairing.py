import numpy as np
import pytest

from tomography.pairing import label_contrastive_pairs
from tomography.pairing import one_vs_rest_pairs
from tomography.pairing import persona_pairs
from tomography.pairing import random_pairs


def test_random_pairs_is_a_perfect_matching() -> None:
    pairs = random_pairs(10, seed=0)

    assert pairs.dtype == np.int64
    assert pairs.shape == (5, 2)
    assert sorted(pairs.ravel().tolist()) == list(range(10))


def test_random_pairs_is_deterministic_per_seed() -> None:
    first = random_pairs(20, seed=3)
    second = random_pairs(20, seed=3)
    other = random_pairs(20, seed=4)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)


def test_random_pairs_randomizes_the_order_inside_pairs() -> None:
    pairs = random_pairs(800, seed=1)

    ascending = float(np.mean(pairs[:, 0] < pairs[:, 1]))
    assert 0.4 < ascending < 0.6


def test_random_pairs_rejects_an_odd_count() -> None:
    with pytest.raises(ValueError, match="even"):
        random_pairs(5, seed=0)


def test_random_pairs_accepts_zero() -> None:
    assert random_pairs(0, seed=0).shape == (0, 2)


def test_label_contrastive_pairs_match_one_of_each_label() -> None:
    labels = np.array([1, 0, 0, 1, 1, 0, 0, 0], dtype=np.int64)

    pairs = label_contrastive_pairs(labels, seed=0)

    assert pairs.dtype == np.int64
    assert pairs.shape == (3, 2)
    assert sorted(labels[pairs].sum(axis=1).tolist()) == [1, 1, 1]
    assert len(set(pairs.ravel().tolist())) == 6


def test_label_contrastive_pairs_randomize_the_order_inside_pairs() -> None:
    labels = np.array([1, 0] * 400, dtype=np.int64)

    pairs = label_contrastive_pairs(labels, seed=2)

    positive_first = float(np.mean(labels[pairs[:, 0]] == 1))
    assert 0.4 < positive_first < 0.6


def test_label_contrastive_pairs_cap_with_n_pairs() -> None:
    labels = np.array([1, 0] * 10, dtype=np.int64)

    pairs = label_contrastive_pairs(labels, seed=0, n_pairs=4)

    assert pairs.shape == (4, 2)
    assert (labels[pairs].sum(axis=1) == 1).all()


def test_label_contrastive_pairs_reject_too_many_pairs() -> None:
    labels = np.array([1, 0, 0, 0], dtype=np.int64)

    with pytest.raises(ValueError, match="n_pairs"):
        label_contrastive_pairs(labels, seed=0, n_pairs=2)


def test_label_contrastive_pairs_need_both_labels() -> None:
    with pytest.raises(ValueError, match="label"):
        label_contrastive_pairs(np.ones(4, dtype=np.int64), seed=0)


def test_label_contrastive_pairs_reject_non_binary_labels() -> None:
    with pytest.raises(ValueError, match="binary"):
        label_contrastive_pairs(np.array([0, 1, 2], dtype=np.int64), seed=0)


def test_label_contrastive_pairs_stay_inside_strata() -> None:
    labels = np.array([1, 0, 1, 0, 1, 0, 0, 0], dtype=np.int64)
    strata = np.array([0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int64)

    pairs = label_contrastive_pairs(labels, seed=0, strata=strata)

    assert pairs.shape == (3, 2)
    np.testing.assert_array_equal(strata[pairs[:, 0]], strata[pairs[:, 1]])
    assert sorted(strata[pairs[:, 0]].tolist()) == [0, 1, 2]
    assert (labels[pairs].sum(axis=1) == 1).all()


def test_label_contrastive_pairs_cap_the_total_across_strata() -> None:
    labels = np.array([1, 0] * 6, dtype=np.int64)
    strata = np.repeat(np.arange(3, dtype=np.int64), 4)

    pairs = label_contrastive_pairs(labels, seed=0, strata=strata, n_pairs=4)

    assert pairs.shape == (4, 2)
    np.testing.assert_array_equal(strata[pairs[:, 0]], strata[pairs[:, 1]])


def test_label_contrastive_pairs_reject_mismatched_strata() -> None:
    labels = np.array([1, 0], dtype=np.int64)

    with pytest.raises(ValueError, match="strata"):
        label_contrastive_pairs(
            labels, seed=0, strata=np.zeros(3, dtype=np.int64)
        )


def test_one_vs_rest_pairs_match_the_target_with_other_classes() -> None:
    classes = np.array([0, 1, 2, 0, 1, 2, 3, 3, 1, 0], dtype=np.int64)

    pairs = one_vs_rest_pairs(classes, target=1, seed=0)

    assert pairs.dtype == np.int64
    assert pairs.shape == (3, 2)
    members = pairs.ravel().tolist()
    assert len(set(members)) == len(members)
    in_target = classes[pairs] == 1
    assert (in_target.sum(axis=1) == 1).all()


def test_one_vs_rest_pairs_randomize_the_order_inside_pairs() -> None:
    classes = np.array([0, 1] * 400, dtype=np.int64)

    pairs = one_vs_rest_pairs(classes, target=1, seed=5)

    target_first = float(np.mean(classes[pairs[:, 0]] == 1))
    assert 0.4 < target_first < 0.6


def test_one_vs_rest_pairs_cap_with_n_pairs() -> None:
    classes = np.array([0, 1, 2] * 10, dtype=np.int64)

    pairs = one_vs_rest_pairs(classes, target=2, seed=0, n_pairs=5)

    assert pairs.shape == (5, 2)


def test_one_vs_rest_pairs_reject_too_many_pairs() -> None:
    classes = np.array([0, 0, 0, 1], dtype=np.int64)

    with pytest.raises(ValueError, match="n_pairs"):
        one_vs_rest_pairs(classes, target=1, seed=0, n_pairs=2)


def test_one_vs_rest_pairs_need_the_target_class() -> None:
    with pytest.raises(ValueError, match="target"):
        one_vs_rest_pairs(np.array([0, 1], dtype=np.int64), target=7, seed=0)


def test_persona_pairs_alternate_the_difference_sign() -> None:
    pairs = persona_pairs(8)

    assert pairs.dtype == np.int64
    np.testing.assert_array_equal(pairs, [[0, 1], [3, 2], [4, 5], [7, 6]])


def test_persona_pairs_reject_an_odd_count() -> None:
    with pytest.raises(ValueError, match="even"):
        persona_pairs(3)
