from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray
import pytest
from sklearn.metrics import roc_auc_score

from tomography.reading.invariance_stats import Combination
from tomography.reading.invariance_stats import TaskScores
from tomography.reading.invariance_stats import bootstrap_combination
from tomography.reading.invariance_stats import holm_adjust
from tomography.reading.invariance_stats import resample_pairs
from tomography.reading.invariance_stats import resampled_auroc
from tomography.reading.invariance_stats import resampled_pair_accuracy
from tomography.reading.invariance_stats import stage1_gate
from tomography.reading.invariance_stats import stage1_selection
from tomography.reading.invariance_stats import stage2
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


BASELINES = tuple(f"B{i}" for i in range(1, 10))


def combination(
    reader: tuple[float, float],
    baselines: tuple[float, float] = (0.9, 0.7),
    *,
    family: str = "qwen",
    checkpoint: str = "c0",
    task: str = "t",
    name: str = "A7",
    n_pairs: int = 4,
    **others: tuple[float, float],
) -> Combination:
    """One stage 2 combination; readers get (clean AUROC, wrapped AUROC)."""
    specs = {name: reader, **dict.fromkeys(BASELINES, baselines), **others}
    scores = {
        key: {
            "clean": flat(clean, n_pairs),
            WRAPPED: flat(wrapped, n_pairs),
        }
        for key, (clean, wrapped) in specs.items()
    }
    return Combination(family, checkpoint, name, TaskScores(task, scores))


def families(
    qwen_wins: int, llama_wins: int, win: float = 0.8, loss: float = 0.65
) -> list[Combination]:
    """Every combination of both families, wins first in each family."""
    out = []
    for family, size, wins in (
        ("qwen", 16, qwen_wins),
        ("llama", 8, llama_wins),
    ):
        for i in range(size):
            out.append(
                combination(
                    (0.9, win if i < wins else loss),
                    family=family,
                    checkpoint=f"{family}{i // 4}",
                    task=f"t{i % 4}",
                )
            )
    return out


def test_combination_delta_takes_the_maximum_over_b1_to_b9() -> None:
    result = bootstrap_combination(
        combination((0.9, 0.8), B5=(0.9, 0.75)), rng=0, n_resamples=50
    )

    assert result.auroc["A7"][WRAPPED] == pytest.approx(0.8)
    assert result.worst["A7"] == pytest.approx(0.8)
    assert result.worst["B5"] == pytest.approx(0.75)
    assert result.best_baseline == "B5"
    assert result.delta == pytest.approx(0.05)
    assert result.delta_interval == pytest.approx((0.05, 0.05))


def test_controls_never_enter_the_best_baseline() -> None:
    result = bootstrap_combination(
        combination((0.9, 0.8), random_direction=(0.95, 0.95), A5=(0.95, 0.9)),
        rng=0,
        n_resamples=50,
    )

    assert result.best_baseline in BASELINES
    assert result.delta == pytest.approx(0.1)


def test_combination_requires_b1_to_b9_and_a_reader_apart_from_them() -> None:
    whole = combination((0.9, 0.8))
    scores = dict(whole.scores.scores)
    del scores["B9"]
    with pytest.raises(ValueError, match="B9"):
        Combination("qwen", "c0", "A7", TaskScores("t", scores))
    with pytest.raises(ValueError, match="reader"):
        Combination("qwen", "c0", "B3", whole.scores)
    with pytest.raises(ValueError, match="reader"):
        Combination("qwen", "c0", "A1", whole.scores)


@pytest.mark.parametrize(
    ("wrapped", "expected"), [(0.8, 1 / 100), (0.7, 1.0), (0.6, 1.0)]
)
def test_p_value_counts_resamples_with_delta_at_most_0(
    wrapped: float, expected: float
) -> None:
    result = bootstrap_combination(
        combination((0.9, wrapped)), rng=0, n_resamples=99
    )

    assert result.p_value == pytest.approx(expected)


def test_combination_bootstrap_matches_a_materialized_reference() -> None:
    rng = np.random.default_rng(8)
    names = ("A7", *BASELINES, "random_direction")
    scores = {
        name: {
            "clean": random_scores(rng, 2, 5),
            WRAPPED: random_scores(rng, 2, 5),
        }
        for name in names
    }
    result = bootstrap_combination(
        Combination("qwen", "c0", "A7", TaskScores("t", scores)),
        rng=3,
        n_resamples=40,
    )

    counts = resample_pairs(5, 40, np.random.default_rng(3))
    deltas, clean = [], []
    for row in counts:
        worst = {
            name: min(materialized_auroc(s, row) for s in scores[name].values())
            for name in names
        }
        deltas.append(worst["A7"] - max(worst[b] for b in BASELINES))
        reader = resampled_pair_accuracy(scores["A7"]["clean"], row[None])
        last = resampled_pair_accuracy(scores["B3"]["clean"], row[None])
        clean.append(float(reader[0] - last[0]))
    np.testing.assert_allclose(result.delta_bootstrap, deltas, atol=1e-12)
    assert result.p_value == pytest.approx(
        (1 + np.sum(np.array(deltas) <= 0)) / 41
    )
    assert result.clean_lower == pytest.approx(np.percentile(clean, 2.5))


@pytest.mark.parametrize(("clean", "holds"), [(0.8, False), (0.8025, True)])
def test_clean_non_inferiority_needs_a_lower_bound_above_minus_002(
    clean: float, holds: bool
) -> None:
    result = bootstrap_combination(
        combination((clean, 0.79), (0.82, 0.7)), rng=0, n_resamples=50
    )

    assert result.clean_accuracy["B3"] == pytest.approx(0.82)
    assert result.clean_difference == pytest.approx(clean - 0.82)
    assert result.clean_lower == pytest.approx(clean - 0.82)
    assert result.non_inferior is holds


def test_holm_adjust_steps_down_and_stays_monotone() -> None:
    adjusted = holm_adjust(np.array([0.01, 0.04, 0.03, 0.005]))

    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06, 0.02])
    np.testing.assert_allclose(holm_adjust(np.array([0.5, 0.9])), [1.0, 1.0])


def test_holm_runs_within_each_family() -> None:
    result = stage2(families(16, 8), rng=0, n_resamples=199)

    assert result.holm_p_value[0] == pytest.approx(16 / 200)
    assert result.holm_p_value[-1] == pytest.approx(8 / 200)
    assert result.wins == {"qwen": 0, "llama": 8}
    assert not result.h2


@pytest.mark.parametrize(
    ("qwen", "llama", "h2"),
    [(9, 5, True), (8, 5, False), (9, 4, False), (16, 8, True)],
)
def test_h2_needs_9_of_16_qwen_and_5_of_8_llama_wins(
    qwen: int, llama: int, h2: bool
) -> None:
    result = stage2(families(qwen, llama), rng=0, n_resamples=999)

    assert result.wins == {"qwen": qwen, "llama": llama}
    assert sum(result.win) == qwen + llama
    assert result.pooled.above_zero
    assert result.h2 is h2


def test_a_win_needs_clean_non_inferiority() -> None:
    combos = families(9, 5)
    combos[0] = combination(
        (0.85, 0.8), checkpoint=combos[0].checkpoint, task="t0"
    )

    result = stage2(combos, rng=0, n_resamples=999)

    assert result.holm_p_value[0] < 0.05
    assert not result.combinations[0].non_inferior
    assert not result.win[0]
    assert result.wins["qwen"] == 8
    assert not result.h2


def test_h2_needs_a_pooled_effect_above_0() -> None:
    result = stage2(families(9, 5, win=0.71, loss=0.5), rng=0, n_resamples=999)

    assert result.wins == {"qwen": 9, "llama": 5}
    assert result.pooled.estimate == pytest.approx((14 * 0.01 - 10 * 0.2) / 24)
    assert not result.pooled.above_zero
    assert not result.h2


def test_pooled_effect_reuses_the_resampled_pairs_of_each_task() -> None:
    rng = np.random.default_rng(4)
    one = random_scores(rng, 2, 6)
    two = random_scores(rng, 2, 6)
    combos = [
        combination(
            (0.9, 0.8),
            family=c.family,
            checkpoint=c.checkpoint,
            task=c.scores.task,
            n_pairs=6,
        )
        for c in families(16, 8)
    ]

    def mirrored(c: Combination, swap: bool) -> Combination:
        reader, other = (two, one) if swap else (one, two)
        scores = {
            name: dict.fromkeys(
                ("clean", WRAPPED), reader if name == "A7" else other
            )
            for name in ("A7", *BASELINES)
        }
        task = TaskScores(c.scores.task, scores)
        return Combination(c.family, c.checkpoint, "A7", task)

    combos[0] = mirrored(combos[0], swap=False)
    combos[4] = mirrored(combos[4], swap=True)
    combos[1] = mirrored(combos[1], swap=False)
    combos[17] = mirrored(combos[17], swap=True)

    result = stage2(combos, rng=0, n_resamples=200)

    assert combos[0].scores.task == combos[4].scores.task
    assert combos[1].scores.task == combos[17].scores.task
    assert np.ptp(result.combinations[0].delta_bootstrap) > 0
    assert np.ptp(result.pooled.bootstrap) == pytest.approx(0, abs=1e-12)
    assert np.ptp(result.pooled_by_family["llama"].bootstrap) > 0


def test_utility_contrast_subtracts_the_random_direction() -> None:
    result = bootstrap_combination(
        combination((0.9, 0.8), random_direction=(0.85, 0.75)),
        rng=0,
        n_resamples=50,
    )

    contrast = result.random_contrast
    assert contrast is not None
    assert contrast["A7"]["clean"].estimate == pytest.approx(0.05)
    assert contrast["A7"][WRAPPED].interval == pytest.approx((0.05, 0.05))
    assert contrast["B3"][WRAPPED].estimate == pytest.approx(-0.05)
    assert "random_direction" not in contrast
    plain = bootstrap_combination(combination((0.9, 0.8)), rng=0, n_resamples=5)
    assert plain.random_contrast is None


def test_secondary_contrast_is_a6_minus_a5_only_for_an_a6_reader() -> None:
    a6 = bootstrap_combination(
        combination((0.9, 0.8), name="A6", A5=(0.9, 0.78)),
        rng=0,
        n_resamples=50,
    )
    a7 = bootstrap_combination(
        combination((0.9, 0.8), A5=(0.9, 0.78)), rng=0, n_resamples=50
    )

    assert a6.secondary is not None
    assert a6.secondary.estimate == pytest.approx(0.02)
    assert a6.secondary.interval == pytest.approx((0.02, 0.02))
    assert a7.secondary is None
    with pytest.raises(ValueError, match="A5"):
        combination((0.9, 0.8), name="A6")


def test_stage2_pools_the_secondary_contrast() -> None:
    combos = [
        combination(
            (0.9, 0.8),
            name="A6",
            family=c.family,
            checkpoint=c.checkpoint,
            task=c.scores.task,
            A5=(0.9, 0.77),
        )
        for c in families(16, 8)
    ]

    result = stage2(combos, rng=0, n_resamples=50)

    assert result.pooled_secondary is not None
    assert result.pooled_secondary.estimate == pytest.approx(0.03)
    assert result.pooled_secondary.above_zero
    assert (
        stage2(families(16, 8), rng=0, n_resamples=5).pooled_secondary is None
    )


def test_stage2_rejects_wrong_families_and_mixed_inputs() -> None:
    combos = families(16, 8)
    with pytest.raises(ValueError, match="family"):
        stage2(combos[:-1], rng=0, n_resamples=5)
    with pytest.raises(ValueError, match="once"):
        stage2([*combos[:-1], combos[0]], rng=0, n_resamples=5)
    mixed = combination(
        (0.9, 0.8), name="A8", family="llama", checkpoint="llama1", task="t3"
    )
    with pytest.raises(ValueError, match="reader"):
        stage2([*combos[:-1], mixed], rng=0, n_resamples=5)
    wide = combination(
        (0.9, 0.8), n_pairs=5, family="llama", checkpoint="llama1", task="t3"
    )
    with pytest.raises(ValueError, match="pairs"):
        stage2([*combos[:-1], wide], rng=0, n_resamples=5)


def test_stage2_is_reproducible_from_its_seed() -> None:
    combos = families(9, 5)

    first = stage2(combos, rng=7, n_resamples=20)
    again = stage2(combos, rng=np.random.default_rng(7), n_resamples=20)

    np.testing.assert_array_equal(
        first.pooled.bootstrap, again.pooled.bootstrap
    )
