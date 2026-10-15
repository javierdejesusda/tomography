"""Tests for the P3 stimulus-set size study."""

import math

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.activations import Activations
from tomography.datasets.stimulus import StimulusSet
from tomography.experiments.p3_size import directional_agreement
from tomography.experiments.p3_size import oriented_differences
from tomography.experiments.p3_size import saturation_ci
from tomography.experiments.p3_size import saturation_size
from tomography.experiments.p3_size import spearman
from tomography.experiments.sweep import Prepared


def _prepared(
    data: NDArray[np.float64],
    labels: list[int],
    pairs: list[tuple[int, int]],
) -> Prepared:
    """Wraps activations of shape (K, n, d), one row per stimulus."""
    n = len(labels)
    index = np.arange(n, dtype=np.int64)
    pair_array = np.array(pairs, dtype=np.int64)
    stimuli = StimulusSet(
        texts=tuple(f"text {i}" for i in range(n)),
        response_spans=(None,) * n,
        concept_spans=(None,) * n,
        labels=np.array(labels, dtype=np.int64),
        pairs=pair_array,
        groups=pair_array.copy(),
        uids=tuple(f"u{i}" for i in range(n)),
    )
    act = Activations(
        data=data.astype(np.float32),
        layers=tuple(range(1, data.shape[0] + 1)),
        position="last",
        row_stimulus=index,
        row_rank=np.zeros(n, dtype=np.int64),
        row_token=np.zeros(n, dtype=np.int64),
        row_token_id=np.zeros(n, dtype=np.int64),
        n_stimuli=n,
    )
    return stimuli, act


def test_spearman_of_a_swapped_pair_is_point_eight() -> None:
    got = spearman(
        np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 3.0, 2.0, 4.0])
    )
    assert got == pytest.approx(0.8)
    assert isinstance(got, float)


def test_spearman_averages_the_ranks_of_ties() -> None:
    got = spearman(
        np.array([1.0, 2.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0, 4.0])
    )
    assert got == pytest.approx(3 / math.sqrt(10))


def test_spearman_follows_ranks_not_values() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(x, np.array([1.0, 10.0, 100.0, 1000.0])) == pytest.approx(
        1.0
    )
    assert spearman(x, x[::-1]) == pytest.approx(-1.0)


@pytest.mark.parametrize(
    ("x", "y"),
    [
        ([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]),
        ([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]),
    ],
    ids=["constant_x", "constant_y"],
)
def test_spearman_is_nan_when_one_side_is_constant(
    x: list[float], y: list[float]
) -> None:
    assert math.isnan(spearman(np.array(x), np.array(y)))


@pytest.mark.parametrize(
    ("x", "y"),
    [
        ([1.0, 2.0, 3.0], [1.0, 2.0]),
        ([1.0], [1.0]),
        ([[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]),
    ],
    ids=["different_lengths", "one_value", "two_dimensional"],
)
def test_spearman_rejects_bad_shapes(x: list[object], y: list[object]) -> None:
    with pytest.raises(ValueError, match="spearman"):
        spearman(np.array(x), np.array(y))


def test_oriented_differences_subtract_label_0_from_label_1() -> None:
    data = np.arange(2 * 4 * 3, dtype=np.float64).reshape(2, 4, 3) ** 2
    got = oriented_differences(_prepared(data, [1, 0, 0, 1], [(0, 1), (2, 3)]))
    assert got.shape == (2, 2, 3)
    np.testing.assert_allclose(got[:, 0], data[:, 0] - data[:, 1])
    np.testing.assert_allclose(got[:, 1], data[:, 3] - data[:, 2])


def test_a_flipped_pair_gives_the_same_difference() -> None:
    data = np.random.default_rng(0).normal(size=(2, 4, 3))
    labels = [1, 0, 1, 0]
    straight = oriented_differences(_prepared(data, labels, [(0, 1), (2, 3)]))
    flipped = oriented_differences(_prepared(data, labels, [(1, 0), (2, 3)]))
    np.testing.assert_array_equal(straight, flipped)


def test_a_pair_with_equal_labels_raises() -> None:
    data = np.zeros((1, 4, 2))
    with pytest.raises(ValueError, match="pair 1 has equal labels"):
        oriented_differences(_prepared(data, [1, 0, 1, 1], [(0, 1), (2, 3)]))


def test_the_agreement_of_two_differences_matches_the_hand_value() -> None:
    diffs = np.array([[[1.0, 0.0], [1.0, 1.0]]])
    expected = (1 / math.sqrt(1.25) + 1.5 / math.sqrt(2.5)) / 2
    np.testing.assert_allclose(directional_agreement(diffs), [expected])


def test_identical_differences_agree_completely() -> None:
    diffs = np.tile(np.array([2.0, -1.0, 0.5]), (1, 5, 1))
    np.testing.assert_allclose(directional_agreement(diffs), [1.0])


def test_each_layer_has_its_own_agreement() -> None:
    first = np.array([[1.0, 0.0], [1.0, 1.0]])
    second = np.array([[1.0, 0.0], [1.0, 0.0]])
    got = directional_agreement(np.stack([first, second]))
    assert got.shape == (2,)
    np.testing.assert_allclose(got[1], 1.0)
    assert got[0] < 1.0


@pytest.mark.parametrize(
    "diffs",
    [
        [[[0.0, 0.0], [1.0, 1.0]]],
        [[[1.0, 0.0], [-1.0, 0.0]]],
    ],
    ids=["zero_difference", "zero_mean"],
)
def test_agreement_raises_on_a_zero_vector(diffs: list[object]) -> None:
    with pytest.raises(ValueError, match="zero"):
        directional_agreement(np.array(diffs))


@pytest.mark.parametrize(
    "diffs",
    [np.ones((1, 1, 3)), np.ones((4, 3))],
    ids=["one_pair", "two_dimensional"],
)
def test_agreement_needs_two_pairs_in_three_dimensions(
    diffs: NDArray[np.float64],
) -> None:
    with pytest.raises(ValueError, match="directional_agreement"):
        directional_agreement(diffs)


SIZES = (2, 4, 8, 16, 32)


def _planted(
    means: list[float],
    *,
    resamples: int = 10,
    noise: float = 0.0,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Accuracy of shape (Z, R) around a planted curve."""
    jitter = np.random.default_rng(seed).normal(
        0.0, noise, size=(len(means), resamples)
    )
    return np.asarray(means)[:, None] + jitter


def test_saturation_is_the_smallest_size_within_delta_of_the_largest() -> None:
    accuracy = _planted([0.55, 0.70, 0.895, 0.90, 0.90])
    assert saturation_size(SIZES, accuracy, delta=0.01) == 8
    assert saturation_size(SIZES, accuracy, delta=0.0) == 16
    assert saturation_size(SIZES, accuracy, delta=0.5) == 2


def test_saturation_accepts_a_size_exactly_delta_below() -> None:
    accuracy = _planted([0.5, 0.75, 1.0])
    assert saturation_size((2, 4, 8), accuracy, delta=0.25) == 4


def test_saturation_takes_the_first_size_even_if_the_curve_dips_later() -> None:
    accuracy = _planted([0.5, 0.9, 0.7, 0.9])
    assert saturation_size((2, 4, 8, 16), accuracy, delta=0.01) == 4


def test_saturation_averages_the_resamples_of_a_size() -> None:
    accuracy = np.array([[0.25, 0.75], [1.0, 0.5], [1.0, 1.0]])
    assert saturation_size((2, 4, 8), accuracy, delta=0.125) == 8


def test_saturation_returns_a_plain_int() -> None:
    got = saturation_size(SIZES, _planted([0.5, 0.6, 0.7, 0.8, 0.9]), delta=0.0)
    assert type(got) is int


@pytest.mark.parametrize(
    ("sizes", "shape", "delta"),
    [
        (SIZES, (4, 3), 0.01),
        (SIZES, (5,), 0.01),
        (SIZES, (5, 3), -0.01),
    ],
    ids=["wrong_size_count", "one_dimensional", "negative_delta"],
)
def test_saturation_rejects_bad_inputs(
    sizes: tuple[int, ...], shape: tuple[int, ...], delta: float
) -> None:
    with pytest.raises(ValueError, match="saturation"):
        saturation_size(sizes, np.ones(shape), delta=delta)


def test_the_saturation_interval_contains_the_point_of_a_planted_curve() -> (
    None
):
    accuracy = _planted([0.55, 0.70, 0.90, 0.91, 0.91], noise=0.01)
    point, lower, upper = saturation_ci(
        SIZES, accuracy, delta=0.03, n_boot=500, seed=0
    )
    assert point == 8
    assert lower <= point <= upper
    assert lower >= SIZES[0]
    assert upper <= SIZES[-1]


def test_the_saturation_interval_is_deterministic_under_the_seed() -> None:
    accuracy = _planted([0.55, 0.70, 0.90, 0.91, 0.91], noise=0.03)
    first = saturation_ci(SIZES, accuracy, delta=0.01, n_boot=300, seed=3)
    second = saturation_ci(SIZES, accuracy, delta=0.01, n_boot=300, seed=3)
    assert first == second
    assert all(type(value) is int for value in first)


def test_a_noisy_curve_has_a_wider_interval_than_a_clean_one() -> None:
    means = [0.55, 0.70, 0.90, 0.91, 0.91]
    clean = saturation_ci(
        SIZES, _planted(means), delta=0.01, n_boot=300, seed=0
    )
    noisy = saturation_ci(
        SIZES, _planted(means, noise=0.05), delta=0.01, n_boot=300, seed=0
    )
    assert clean[1] == clean[0] == clean[2]
    assert noisy[2] - noisy[1] > 0


def test_one_resample_per_size_gives_a_degenerate_interval() -> None:
    accuracy = _planted([0.55, 0.70, 0.90, 0.91, 0.91], resamples=1)
    point, lower, upper = saturation_ci(
        SIZES, accuracy, delta=0.02, n_boot=50, seed=0
    )
    assert lower == point == upper == 8


def test_the_interval_of_a_constant_curve_sits_at_the_first_size() -> None:
    got = saturation_ci(
        SIZES, np.ones((len(SIZES), 4)), delta=0.01, n_boot=10, seed=0
    )
    assert got == (2, 2, 2)


def test_the_interval_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="saturation"):
        saturation_ci(SIZES, np.ones((4, 3)), delta=0.01, n_boot=10, seed=0)
    with pytest.raises(ValueError, match="n_boot"):
        saturation_ci(SIZES, np.ones((5, 3)), delta=0.01, n_boot=0, seed=0)
