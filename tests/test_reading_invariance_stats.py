from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray
import pytest
from sklearn.metrics import roc_auc_score

from tomography.reading.invariance_stats import TaskScores
from tomography.reading.invariance_stats import resample_pairs
from tomography.reading.invariance_stats import resampled_auroc
from tomography.reading.invariance_stats import resampled_pair_accuracy
from tomography.reading.invariance_stats import stage1_gate
from tomography.reading.invariance_stats import stage1_selection
from tomography.reading.metrics import group_accuracy

ARMS = ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8")
WRAPPED = "S-D1-against"


def flat(
    auroc: float, n_pairs: int = 4, n_draws: int = 200
) -> NDArray[np.float64]:
    """Scores whose draw-averaged AUROC is `auroc` in every resample.

    Every pair of a draw is identical, so a resample cannot change the
    AUROC of a draw; the draws set the average in steps of 1 / (2 n_draws).
    """
    halves = round(auroc * 2 * n_draws)
    ones, tie = divmod(halves, 2)
    scores = np.zeros((n_draws, n_pairs, 2))
    scores[:ones, :, 0] = 1.0
    scores[ones + tie :, :, 1] = 1.0
    return scores


def stage1_task(
    name: str, arms: Mapping[str, tuple[float, float]]
) -> TaskScores:
    """One stage 1 task; each arm gets (clean AUROC, wrapped AUROC)."""
    return TaskScores(
        name,
        {
            arm: {"clean": flat(clean), WRAPPED: flat(wrapped)}
            for arm, (clean, wrapped) in arms.items()
        },
    )


def base_arms(
    **overrides: tuple[float, float],
) -> dict[str, tuple[float, float]]:
    arms = {arm: (0.9, 0.7) for arm in ARMS}
    arms.update(overrides)
    return arms


def random_scores(
    rng: np.random.Generator, n_draws: int, n_pairs: int
) -> NDArray[np.float64]:
    return rng.integers(0, 4, size=(n_draws, n_pairs, 2)).astype(float)


def materialized_auroc(
    scores: NDArray[np.float64], counts: NDArray[np.int64]
) -> float:
    rows = np.repeat(np.arange(scores.shape[1]), counts)
    labels = np.r_[np.ones(rows.size), np.zeros(rows.size)]
    return float(
        np.mean(
            [
                roc_auc_score(labels, np.r_[draw[rows, 0], draw[rows, 1]])
                for draw in scores
            ]
        )
    )


def test_task_scores_rejects_arrays_with_different_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        TaskScores(
            "t",
            {
                "A1": {"clean": np.zeros((2, 3, 2))},
                "A2": {"clean": np.zeros((2, 4, 2))},
            },
        )


def test_task_scores_rejects_a_reader_missing_a_condition() -> None:
    with pytest.raises(ValueError, match="conditions"):
        TaskScores(
            "t",
            {
                "A1": {"clean": np.zeros((2, 3, 2)), "x": np.zeros((2, 3, 2))},
                "A2": {"clean": np.zeros((2, 3, 2))},
            },
        )


def test_task_scores_requires_a_clean_condition() -> None:
    with pytest.raises(ValueError, match="clean"):
        TaskScores("t", {"A1": {"x": np.zeros((2, 3, 2))}})


def test_task_scores_rejects_non_finite_scores() -> None:
    scores = np.zeros((2, 3, 2))
    scores[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        TaskScores("t", {"A1": {"clean": scores}})


def test_resample_pairs_draws_n_pairs_with_replacement() -> None:
    counts = resample_pairs(7, 50, np.random.default_rng(3))

    assert counts.shape == (50, 7)
    assert (counts.sum(axis=1) == 7).all()
    assert (counts >= 0).all()
    assert (counts == 0).any()
    np.testing.assert_array_equal(
        counts, resample_pairs(7, 50, np.random.default_rng(3))
    )


def test_resampled_auroc_matches_sklearn_on_materialized_resamples() -> None:
    rng = np.random.default_rng(0)
    scores = random_scores(rng, n_draws=3, n_pairs=9)
    counts = resample_pairs(9, 40, rng)

    fast = resampled_auroc(scores, counts)

    expected = [materialized_auroc(scores, row) for row in counts]
    np.testing.assert_allclose(fast, expected, rtol=0, atol=1e-12)


def test_resampled_pair_accuracy_matches_group_accuracy() -> None:
    rng = np.random.default_rng(1)
    scores = random_scores(rng, n_draws=3, n_pairs=9)
    counts = resample_pairs(9, 20, rng)

    fast = resampled_pair_accuracy(scores, counts)

    for value, row in zip(fast, counts, strict=True):
        rows = np.repeat(np.arange(9), row)
        texts = scores[:, rows, :].reshape(3, -1)
        groups = np.arange(texts.shape[1]).reshape(-1, 2)
        labels = np.tile(np.array([1, 0], dtype=np.int64), rows.size)
        accuracy = group_accuracy(texts, groups, labels).mean()
        assert value == pytest.approx(accuracy, abs=1e-12)


def test_stage1_endpoint_is_the_minimum_of_draw_averaged_aurocs() -> None:
    task = stage1_task("t", base_arms(A2=(0.95, 0.75), A6=(0.8, 0.85)))

    selection = stage1_selection([task])

    assert selection.auroc["t"]["A6"]["clean"] == pytest.approx(0.8)
    assert selection.endpoint["t"]["A6"] == pytest.approx(0.8)
    assert selection.endpoint["t"]["A1"] == pytest.approx(0.7)
    assert selection.best_baseline["t"] == "A2"
    assert selection.gain["t"]["A6"] == pytest.approx(0.05)
    assert selection.gain["t"]["A1"] == pytest.approx(-0.05)


def test_stage1_eligibility_allows_clean_auroc_exactly_a1_minus_001() -> None:
    task = stage1_task(
        "t",
        base_arms(A5=(0.89, 0.8), A6=(0.8875, 0.9)),
    )

    selection = stage1_selection([task])

    assert "A5" in selection.eligible
    assert "A6" not in selection.eligible
    assert selection.winner == "A5"


def test_stage1_eligibility_must_hold_in_every_task() -> None:
    good = stage1_task("a", base_arms(A6=(0.9, 0.9)))
    bad = stage1_task("b", base_arms(A6=(0.8875, 0.9)))

    selection = stage1_selection([good, bad])

    assert "A6" not in selection.eligible


def test_stage1_winner_has_the_highest_mean_gain_over_tasks() -> None:
    first = stage1_task("a", base_arms(A5=(0.9, 0.8), A6=(0.9, 0.76)))
    second = stage1_task("b", base_arms(A5=(0.9, 0.7), A6=(0.9, 0.76)))

    selection = stage1_selection([first, second])

    assert selection.mean_gain["A5"] == pytest.approx(0.05)
    assert selection.mean_gain["A6"] == pytest.approx(0.06)
    assert selection.winner == "A6"


def test_stage1_tie_within_0005_goes_to_the_simpler_arm() -> None:
    tied = stage1_task("t", base_arms(A6=(0.9, 0.8), A7=(0.9, 0.795)))
    apart = stage1_task("t", base_arms(A6=(0.9, 0.8), A7=(0.9, 0.7925)))

    assert stage1_selection([tied]).winner == "A7"
    assert stage1_selection([apart]).winner == "A6"


@pytest.mark.parametrize(
    ("first", "second"),
    [("A1", "A2"), ("A2", "A3"), ("A4", "A5"), ("A5", "A7"), ("A8", "A6")],
)
def test_stage1_ties_follow_the_preregistered_order(
    first: str, second: str
) -> None:
    arms = {arm: (0.9, 0.7) for arm in ARMS}
    arms[first] = arms[second] = (0.9, 0.8)
    if first != "A1":
        arms["A1"] = (0.9, 0.6)

    assert stage1_selection([stage1_task("t", arms)]).winner == first


def test_stage1_rejects_unknown_arms_and_missing_baselines() -> None:
    with pytest.raises(ValueError, match="arm"):
        stage1_selection([stage1_task("t", base_arms(B3=(0.9, 0.7)))])
    arms = base_arms()
    del arms["A3"]
    with pytest.raises(ValueError, match="A3"):
        stage1_selection([stage1_task("t", arms)])


def test_stage1_gate_passes_at_exactly_002_above_the_comparator() -> None:
    task = stage1_task("t", base_arms(A4=(0.9, 0.75), A6=(0.9, 0.77)))

    gate = stage1_gate([task], rng=0, n_resamples=200)

    assert gate.selection.winner == "A6"
    assert gate.comparator == "A4"
    assert gate.statistic == pytest.approx(0.02)
    assert gate.interval == pytest.approx((0.02, 0.02))
    assert gate.passed
    assert gate.reasons == ()


def test_stage1_gate_fails_below_002() -> None:
    task = stage1_task("t", base_arms(A4=(0.9, 0.75), A6=(0.9, 0.7675)))

    gate = stage1_gate([task], rng=0, n_resamples=200)

    assert gate.statistic == pytest.approx(0.0175)
    assert not gate.passed
    assert gate.reasons == ("the statistic is below 0.02",)


def test_stage1_gate_fails_when_the_interval_reaches_0() -> None:
    weak = np.array([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    strong = np.array([[[1.0, 0.0]] * 4])
    arms = {arm: {"clean": weak} for arm in ARMS}
    arms["A6"] = {"clean": strong}

    gate = stage1_gate([TaskScores("t", arms)], rng=0, n_resamples=2000)

    assert gate.selection.winner == "A6"
    assert gate.statistic == pytest.approx(0.25)
    assert gate.interval[0] == pytest.approx(0.0)
    assert not gate.passed
    assert gate.reasons == ("the interval is not above 0",)


@pytest.mark.parametrize("winner", ["A1", "A2", "A3", "A4"])
def test_stage1_gate_fails_automatically_for_a_baseline_winner(
    winner: str,
) -> None:
    arms = {arm: (0.9, 0.6) for arm in ARMS}
    arms[winner] = (0.9, 0.9)
    if winner != "A3":
        arms["A3"] = (0.9, 0.62)

    gate = stage1_gate([stage1_task("t", arms)], rng=0, n_resamples=50)

    assert gate.selection.winner == winner
    assert not gate.passed
    assert gate.reasons[0] == (
        f"the winner {winner} is a single-position or plain logistic regression"
    )


def test_stage1_gate_comparator_tie_goes_to_a3() -> None:
    task = stage1_task("t", base_arms(A6=(0.9, 0.8)))

    gate = stage1_gate([task], rng=0, n_resamples=50)

    assert gate.comparator == "A3"


def test_stage1_gate_bootstrap_matches_a_materialized_reference() -> None:
    rng = np.random.default_rng(5)
    tasks = [
        TaskScores(
            name,
            {
                arm: {
                    "clean": random_scores(rng, 2, 6),
                    WRAPPED: random_scores(rng, 2, 6),
                }
                for arm in ARMS
            },
        )
        for name in ("b", "a")
    ]

    gate = stage1_gate(tasks, rng=11, n_resamples=30)

    reference = np.random.default_rng(11)
    by_name = {task.task: task for task in tasks}
    counts = {
        name: resample_pairs(6, 30, reference) for name in sorted(by_name)
    }
    expected = []
    for b in range(30):
        worst = {
            name: {
                arm: min(
                    materialized_auroc(scores, counts[name][b])
                    for scores in by_name[name].scores[arm].values()
                )
                for arm in ("A3", "A4", gate.selection.winner)
            }
            for name in by_name
        }
        a3 = np.mean([worst[name]["A3"] for name in by_name])
        a4 = np.mean([worst[name]["A4"] for name in by_name])
        comparator = "A3" if a3 >= a4 else "A4"
        expected.append(
            np.mean(
                [
                    worst[name][gate.selection.winner] - worst[name][comparator]
                    for name in by_name
                ]
            )
        )
    np.testing.assert_allclose(gate.bootstrap, expected, rtol=0, atol=1e-12)
    low, high = np.percentile(expected, [2.5, 97.5])
    assert gate.interval == pytest.approx((low, high))


def test_stage1_gate_is_reproducible_from_its_seed() -> None:
    rng = np.random.default_rng(2)
    arms = {arm: {"clean": random_scores(rng, 2, 8)} for arm in ARMS}
    task = TaskScores("t", arms)

    first = stage1_gate([task], rng=4, n_resamples=40)
    again = stage1_gate([task], rng=np.random.default_rng(4), n_resamples=40)
    other = stage1_gate([task], rng=5, n_resamples=40)

    np.testing.assert_array_equal(first.bootstrap, again.bootstrap)
    assert not np.array_equal(first.bootstrap, other.bootstrap)
