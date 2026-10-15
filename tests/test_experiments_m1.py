"""Tests for the M1 method comparison."""

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.experiments.m1_methods import METHODS
from tomography.experiments.m1_methods import Tally
from tomography.experiments.m1_methods import default_control_layers
from tomography.experiments.m1_methods import erased_probe_accuracy
from tomography.experiments.m1_methods import paired_strengths
from tomography.experiments.m1_methods import project_out
from tomography.experiments.m1_methods import tally
from tomography.experiments.m1_methods import truncate_scenario

NAN = float("nan")


def test_the_methods_are_the_six_direction_finders_with_random_last() -> None:
    assert METHODS == (
        "pca_diff",
        "pca_raw",
        "mean_diff",
        "logistic",
        "kmeans",
        "random",
    )


def test_truncate_scenario_keeps_the_first_words() -> None:
    text = "I took the dog out for a long walk"

    assert truncate_scenario(text, 4) == "I took the dog"


def test_truncate_scenario_leaves_a_shorter_text_unchanged() -> None:
    assert truncate_scenario("a short one", 6) == "a short one"


def test_truncate_scenario_cuts_on_any_whitespace() -> None:
    assert truncate_scenario("a  b\tc\nd e", 3) == "a b c"


@pytest.mark.parametrize("words", [0, -2])
def test_truncate_scenario_needs_at_least_one_word(words: int) -> None:
    with pytest.raises(ValueError, match="words"):
        truncate_scenario("a b c", words)


def test_tally_counts_wins_losses_and_ties() -> None:
    positive = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
    negative = np.array([1.0, 2.0, 2.0, 4.0, 4.0])

    result = tally(positive, negative)

    assert result == Tally(wins=2, losses=1, ties=2)
    assert result.decisive == 3
    assert result.rate == pytest.approx(2 / 3)


def test_tally_drops_a_pair_with_a_nan_side() -> None:
    positive = np.array([1.0, NAN, 3.0, 4.0])
    negative = np.array([0.0, 5.0, NAN, 4.0])

    result = tally(positive, negative)

    assert result == Tally(wins=1, losses=0, ties=1)
    assert result.rate == 1.0


@pytest.mark.parametrize(
    ("positive", "negative"),
    [
        ([1.0, 2.0], [1.0, 2.0]),
        ([NAN, 1.0], [2.0, NAN]),
        ([], []),
    ],
    ids=["all-ties", "all-nan-pairs", "empty"],
)
def test_the_rate_is_nan_with_nothing_decisive(
    positive: list[float], negative: list[float]
) -> None:
    result = tally(np.array(positive), np.array(negative))

    assert result.decisive == 0
    assert np.isnan(result.rate)


@pytest.mark.parametrize(
    ("positive", "negative"),
    [
        (np.zeros(3), np.zeros(4)),
        (np.zeros((2, 3)), np.zeros((2, 3))),
    ],
    ids=["length", "not-1d"],
)
def test_tally_needs_two_vectors_of_one_length(
    positive: NDArray[np.float64], negative: NDArray[np.float64]
) -> None:
    with pytest.raises(ValueError, match="two vectors"):
        tally(positive, negative)


def test_paired_strengths_returns_the_sorted_positive_half() -> None:
    grid = (0.2, -0.1, 0.0, 0.1, -0.2, 0.3, -0.3)

    assert paired_strengths(grid) == (0.1, 0.2, 0.3)


@pytest.mark.parametrize(
    ("grid", "message"),
    [
        ((-0.1, 0.1), "contain 0.0"),
        ((-0.1, 0.0, 0.1, 0.2), "its negative"),
        ((-0.2, -0.1, 0.0, 0.1), "its negative"),
        ((0.0,), "positive strength"),
    ],
    ids=["no-zero", "missing-negative", "unpaired-negative", "no-strength"],
)
def test_paired_strengths_refuses_a_grid_that_is_not_symmetric(
    grid: tuple[float, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        paired_strengths(grid)


def _unit(width: int, axis: int) -> NDArray[np.float32]:
    return np.eye(width, dtype=np.float32)[axis]


def test_project_out_removes_the_component_along_the_direction() -> None:
    h = np.random.default_rng(0).normal(size=(10, 6)).astype(np.float32)
    direction = _unit(6, 0) + _unit(6, 3)

    projected = project_out(h, direction)

    unit = direction / np.linalg.norm(direction)
    assert np.abs(projected @ unit).max() < 1e-6
    assert projected.dtype == np.float32


def test_project_out_leaves_an_orthogonal_vector_unchanged() -> None:
    h = np.tile(_unit(6, 1) * 2.0 + _unit(6, 4), (3, 1))

    np.testing.assert_allclose(project_out(h, _unit(6, 0)), h, atol=1e-7)


def test_project_out_normalizes_the_direction() -> None:
    h = np.random.default_rng(1).normal(size=(5, 6)).astype(np.float32)
    direction = np.array([1, 2, 0, 0, 3, 0], dtype=np.float32)

    np.testing.assert_allclose(
        project_out(h, 7.5 * direction), project_out(h, direction), atol=1e-6
    )


@pytest.mark.parametrize(
    "direction",
    [np.zeros(6, dtype=np.float32), np.ones(4, dtype=np.float32)],
    ids=["zero", "width"],
)
def test_project_out_refuses_a_direction_it_cannot_use(
    direction: NDArray[np.float32],
) -> None:
    h = np.ones((3, 6), dtype=np.float32)

    with pytest.raises(ValueError, match="direction"):
        project_out(h, direction)


class Planted:
    """Activations whose class is carried by axis 0 of 8.

    Attributes:
        train_h: Training activations, 40 rows.
        train_labels: Their labels.
        test_h: Test activations, 40 rows forming 20 pairs.
        test_groups: The pairs, as row indices.
        test_labels: The test labels.
    """

    def __init__(self) -> None:
        rng = np.random.default_rng(5)
        labels = np.tile(np.array([1, 0]), 40).astype(np.int64)
        h = rng.normal(size=(80, 8)).astype(np.float32)
        h[:, 0] = np.where(labels == 1, 4.0, -4.0) + 0.1 * rng.normal(size=80)
        self.train_h = h[:40]
        self.train_labels = labels[:40]
        self.test_h = h[40:]
        self.test_groups = np.arange(40).reshape(20, 2)
        self.test_labels = labels[40:]

    def erase(
        self,
        direction: NDArray[np.float32],
        *,
        train_h: NDArray[np.float32] | None = None,
        test_h: NDArray[np.float32] | None = None,
    ) -> tuple[float, float]:
        return erased_probe_accuracy(
            self.train_h if train_h is None else train_h,
            self.train_labels,
            self.test_h if test_h is None else test_h,
            self.test_groups,
            self.test_labels,
            direction,
            seed=0,
        )


def test_erasing_the_planted_axis_takes_the_probe_to_chance() -> None:
    before, after = Planted().erase(_unit(8, 0))

    assert before > 0.95
    assert after < 0.8


def test_erasing_an_orthogonal_axis_leaves_the_probe_alone() -> None:
    before, after = Planted().erase(_unit(8, 1))

    assert before > 0.95
    assert after == pytest.approx(before, abs=0.05)


def test_erasing_the_class_mean_difference_gives_exactly_chance() -> None:
    planted = Planted()
    positive = planted.train_labels == 1
    difference = planted.train_h[positive].mean(axis=0) - planted.train_h[
        ~positive
    ].mean(axis=0)

    before, after = planted.erase(difference)

    assert before > 0.95
    assert after == 0.5


def test_all_zero_activations_give_exactly_chance() -> None:
    planted = Planted()

    result = planted.erase(
        _unit(8, 0),
        train_h=np.zeros_like(planted.train_h),
        test_h=np.zeros_like(planted.test_h),
    )

    assert result == (0.5, 0.5)


def test_a_probe_error_other_than_a_zero_fit_propagates() -> None:
    planted = Planted()

    with pytest.raises(ValueError, match="both classes"):
        erased_probe_accuracy(
            planted.train_h,
            np.ones_like(planted.train_labels),
            planted.test_h,
            planted.test_groups,
            planted.test_labels,
            _unit(8, 0),
            seed=0,
        )


@pytest.mark.parametrize(
    ("n_layers", "expected"),
    [(12, (5, 7, 8)), (28, (11, 15, 20)), (3, (1, 2)), (1, (1,))],
    ids=["gpt2", "qwen3-1.7b", "tiny", "single-layer"],
)
def test_default_control_layers_sit_at_40_55_and_70_percent_of_the_depth(
    n_layers: int, expected: tuple[int, ...]
) -> None:
    assert default_control_layers(n_layers) == expected
