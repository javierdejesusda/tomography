"""Tests for the M1 method comparison."""

from collections.abc import Sequence
import dataclasses
from pathlib import Path
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.cli import load_experiment
from tomography.datasets.ethics import load_ethics_utility
from tomography.datasets.records import Example
from tomography.datasets.records import Split
from tomography.datasets.sources import OfflineCacheMissError
from tomography.datasets.wikitext import load_wikitext
from tomography.experiments import m1_methods
from tomography.experiments.base import RunDir
from tomography.experiments.effects import SideEffects
from tomography.experiments.effects import read_trials
from tomography.experiments.effects import split_held_out
from tomography.experiments.m1_methods import METHODS
from tomography.experiments.m1_methods import Config
from tomography.experiments.m1_methods import Tally
from tomography.experiments.m1_methods import default_control_layers
from tomography.experiments.m1_methods import erased_probe_accuracy
from tomography.experiments.m1_methods import execute
from tomography.experiments.m1_methods import paired_strengths
from tomography.experiments.m1_methods import plot
from tomography.experiments.m1_methods import project_out
from tomography.experiments.m1_methods import run
from tomography.experiments.m1_methods import tally
from tomography.experiments.m1_methods import truncate_scenario
from tomography.experiments.sweep import SERIES
from tomography.judges import load_judge
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.reading.metrics import wilson_ci

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


WORDS = (
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "omicron",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
)
STRENGTHS = (0.1, 0.3)
STEERED = (-0.3, -0.1, 0.1, 0.3)
N_STEER = 2
N_PROMPTS = 10
N_GAP = 6
N_TEST_PAIRS = 8
FILES = sorted(
    [
        f"{stem}.{extension}"
        for stem in (
            "curves",
            "curves_val",
            "curves_success",
            "curves_success_check",
            "trials",
            "trials_termination",
        )
        for extension in ("npz", "json")
    ]
    + ["table.json", "table_seeds.json", "table_effects.json", "summary.json"]
    + ["config.json", "meta.json"]
)


FILES_WITHOUT_EFFECTS = [f for f in FILES if f != "table_effects.json"]


def _utility(
    n_pairs: int = 60, *, shared_openings: bool = False
) -> list[Example]:
    """ETHICS-like pairs: label 1 is the more pleasant scenario.

    With `shared_openings` the two scenarios of a test pair open with the
    same six words, as the real pairs do, and differ only after them.
    """
    rng = np.random.default_rng(1)
    examples = []
    for i in range(n_pairs):
        split: Split = (
            "train" if i % 5 < 3 else ("val" if i % 5 == 3 else "test")
        )
        for suffix, label in (("a", 1), ("b", 0)):
            text = " ".join(rng.choice(WORDS, 9))
            if shared_openings and split == "test":
                text = f"case {i} opens the same way {suffix} then differs"
            examples.append(
                Example(
                    uid=f"u{i}:{suffix}",
                    text=text,
                    label=label,
                    split=split,
                    meta={"pair": f"p{i}"},
                )
            )
    return examples


def _texts() -> list[str]:
    return [
        " ".join(np.random.default_rng(i).choice(WORDS, 12)) for i in range(60)
    ]


def stub_score(texts: Sequence[str]) -> NDArray[np.float32]:
    """Two judges: the vowel fraction and the space fraction of a text."""
    return np.array(
        [
            [
                sum(c in "aeiou" for c in text) / max(len(text), 1),
                text.count(" ") / max(len(text), 1),
            ]
            for text in texts
        ],
        dtype=np.float32,
    )


def _twin_score(texts: Sequence[str]) -> NDArray[np.float32]:
    first = stub_score(texts)[:, 0]
    return np.column_stack([first, first])


def _negated_score(texts: Sequence[str]) -> NDArray[np.float32]:
    first = stub_score(texts)[:, 0]
    return np.column_stack([first, -first])


SMALL = {
    "methods": ("pca_diff", "logistic"),
    "n_prompts": 6,
    "n_termination": 4,
}


def _config(**overrides: Any) -> Config:
    base = Config(
        model="tiny",
        n_seeds=3,
        n_steer_seeds=N_STEER,
        n_pairs=12,
        n_eval_pairs=N_TEST_PAIRS,
        n_prompts=N_PROMPTS,
        n_termination=N_GAP,
        n_calibration=8,
        n_neutral=8,
        max_new_tokens=6,
        coeffs=(-0.3, -0.1, 0.0, 0.1, 0.3),
        batch_size=8,
    )
    return dataclasses.replace(base, **overrides)


def _execute(
    loaded: LoadedModel,
    out_root: Path,
    *,
    score: Any = stub_score,
    utility: Sequence[Example] | None = None,
    **overrides: Any,
) -> RunDir:
    config = _config(**overrides)
    run_dir = RunDir.create(config, out_root=out_root)
    execute(
        config,
        run_dir,
        loaded,
        _utility() if utility is None else utility,
        _texts(),
        score,
    )
    return run_dir


_RUNS: dict[str, RunDir] = {}


@pytest.fixture
def ran(
    tiny_arch: LoadedModel, tmp_path_factory: pytest.TempPathFactory
) -> RunDir:
    """The finished default run of the architecture, built once."""
    key = type(tiny_arch.model).__name__
    if key not in _RUNS:
        _RUNS[key] = _execute(tiny_arch, tmp_path_factory.mktemp(key))
    return _RUNS[key]


def _nan(value: float | None) -> float:
    return NAN if value is None else value


def _table(run_dir: RunDir) -> list[list[Any]]:
    payload = run_dir.read_json("table")
    assert isinstance(payload, dict)
    assert payload["columns"] == [
        "method",
        "task",
        "setting",
        "value",
        "lo",
        "hi",
        "n",
    ]
    rows: list[list[Any]] = payload["rows"]
    return rows


def _seed_rows(run_dir: RunDir) -> list[list[Any]]:
    payload = run_dir.read_json("table_seeds")
    assert isinstance(payload, dict)
    assert payload["columns"] == [
        "method",
        "seed",
        "strength",
        "judge",
        "wins",
        "losses",
        "ties",
        "rate",
        "lo",
        "hi",
    ]
    rows: list[list[Any]] = payload["rows"]
    return rows


def _summary(run_dir: RunDir) -> dict[str, Any]:
    payload = run_dir.read_json("summary")
    assert isinstance(payload, dict)
    return payload


def test_config_defaults_and_round_trip() -> None:
    config = Config()

    assert config.experiment == "m1"
    assert config.model == "gpt2"
    assert config.methods == METHODS
    assert config.judge == "sentiment-primary"
    assert config.judge_check == "sentiment-secondary"
    assert config.n_seeds == 10
    assert config.n_steer_seeds == 3
    assert config.n_pairs == 64
    assert config.n_eval_pairs is None
    assert config.preprocess == "center"
    assert config.position == "last"
    assert config.truncate_words == 6
    assert config.n_termination == 100
    assert config.n_prompts == 250
    assert config.num_samples == 1
    assert config.fluency is True
    assert Config.from_dict(config.to_dict()) == config


def test_config_overrides_survive_a_round_trip() -> None:
    config = Config(
        methods=("logistic", "random"),
        judge="sentiment-secondary",
        judge_check="sentiment-primary",
        n_steer_seeds=1,
    )

    assert Config.from_dict(config.to_dict()) == config
    assert config.methods == ("logistic", "random")


def test_the_cli_discovers_the_experiment() -> None:
    experiment = load_experiment("m1")

    assert experiment.config_cls is Config
    assert experiment.run is run
    assert experiment.plot is plot


def test_execute_writes_the_files_and_the_table(ran: RunDir) -> None:
    assert sorted(p.name for p in ran.path.iterdir() if p.is_file()) == FILES

    rows = _table(ran)

    assert len(rows) == 56
    per_task: dict[str, int] = {}
    for row in rows:
        per_task[row[1]] = per_task.get(row[1], 0) + 1
    assert per_task == {
        "correlation": 6,
        "manipulation": 12,
        "manipulation_check": 12,
        "termination_gap": 7,
        "termination_gap_check": 7,
        "termination_probe": 12,
    }
    assert {row[0] for row in rows} == {*METHODS, "none"}


def test_the_seed_rows_tally_the_paired_trials(ran: RunDir) -> None:
    rows = _seed_rows(ran)
    trials = {t.name: t for t in read_trials(ran, "trials")}

    assert len(rows) == 48
    assert {(r[0], r[1], r[2], r[3]) for r in rows} == {
        (method, seed, strength, judge)
        for method in METHODS
        for seed in range(N_STEER)
        for strength in STRENGTHS
        for judge in ("primary", "check")
    }
    assert all(r[4] + r[5] + r[6] == N_PROMPTS for r in rows)
    for method, seed, strength, judge, wins, losses, ties, rate, lo, hi in rows:
        column = ("primary", "check").index(judge)
        up = trials[f"{method}/{seed}/{strength:g}"].per_prompt()[:, column]
        down = trials[f"{method}/{seed}/{-strength:g}"].per_prompt()[:, column]
        expected = tally(up, down)
        assert [wins, losses, ties] == [
            expected.wins,
            expected.losses,
            expected.ties,
        ]
        assert _nan(rate) == pytest.approx(expected.rate, nan_ok=True)
        if expected.decisive:
            assert [lo, hi] == pytest.approx(
                wilson_ci(expected.wins, expected.decisive)
            )
        else:
            assert lo is None
            assert hi is None


def test_the_pooled_rows_and_the_curves_follow_the_seed_rows(
    ran: RunDir,
) -> None:
    seeds = _seed_rows(ran)
    table = {(r[0], r[1], r[2]): r for r in _table(ran)}
    curves = {
        "primary": ran.read_arrays("curves_success"),
        "check": ran.read_arrays("curves_success_check"),
    }
    tasks = {"primary": "manipulation", "check": "manipulation_check"}

    for m, method in enumerate(METHODS):
        for k, strength in enumerate(STRENGTHS):
            for judge, task in tasks.items():
                cells = [
                    r
                    for r in seeds
                    if (r[0], r[2], r[3]) == (method, strength, judge)
                ]
                wins = sum(r[4] for r in cells)
                decisive = wins + sum(r[5] for r in cells)
                row = table[(method, task, f"{strength:g}")]
                assert row[6] == decisive
                if decisive:
                    assert row[3] == pytest.approx(wins / decisive)
                    assert row[4:6] == pytest.approx(wilson_ci(wins, decisive))
                else:
                    assert row[3] is None
                np.testing.assert_array_equal(
                    curves[judge]["per_seed"][m, :, k],
                    [_nan(r[7]) for r in cells],
                )


@pytest.mark.parametrize("stem", ["curves_success", "curves_success_check"])
def test_the_success_curves_have_one_series_per_method(
    ran: RunDir, stem: str
) -> None:
    arrays = ran.read_arrays(stem)
    sidecar = ran.read_json(stem)

    assert arrays["x"].tolist() == list(STRENGTHS)
    assert arrays["mean"].shape == (6, 2)
    assert arrays["per_seed"].shape == (6, N_STEER, 2)
    assert isinstance(sidecar, dict)
    assert sidecar["series"] == list(METHODS)
    assert sidecar["chance"] == 0.5


def test_the_correlation_curves_and_rows_pick_the_layer_on_val(
    ran: RunDir, tiny_arch: LoadedModel
) -> None:
    n_layers = tiny_arch.n_layers
    for stem in ("curves", "curves_val"):
        arrays = ran.read_arrays(stem)
        sidecar = ran.read_json(stem)
        assert arrays["x"].tolist() == list(range(1, n_layers + 1))
        assert arrays["mean"].shape == (6, n_layers)
        assert arrays["per_seed"].shape == (6, 3, n_layers)
        assert isinstance(sidecar, dict)
        assert sidecar["series"] == list(METHODS)
        assert sidecar["chance"] == 0.5
        assert sidecar["n_layers"] == n_layers
    val = ran.read_arrays("curves_val")["mean"]
    test = ran.read_arrays("curves")["per_seed"]
    rows = {r[0]: r for r in _table(ran) if r[1] == "correlation"}

    for m, method in enumerate(METHODS):
        best = int(np.argmax(val[m]))
        row = rows[method]
        assert row[2] == f"layer{best + 1}"
        assert row[3] == pytest.approx(test[m, :, best].mean())
        assert row[4] <= row[3] <= row[5]
        assert row[6] == N_TEST_PAIRS


def test_the_trials_are_named_by_what_made_them(ran: RunDir) -> None:
    trials = read_trials(ran, "trials")
    termination = read_trials(ran, "trials_termination")

    names = [t.name for t in trials]
    assert names == [
        "none",
        *(
            f"{method}/{seed}/{coeff:g}"
            for method in METHODS
            for seed in range(N_STEER)
            for coeff in STEERED
        ),
    ]
    assert len(set(names)) == len(names) == 49
    assert np.stack([t.scores for t in trials]).shape == (
        49,
        N_PROMPTS,
        1,
        2,
    )
    assert [t.name for t in termination] == [
        "none",
        *(
            f"{method}/{seed}/projection"
            for method in METHODS
            for seed in range(N_STEER)
        ),
    ]
    assert len(termination) == 13
    assert termination[0].scores.shape == (2 * N_GAP, 1, 2)


def test_the_prompts_are_raw_truncated_and_full_scenarios(
    ran: RunDir,
) -> None:
    utility = _utility()
    test_rows = [e for e in utility if e.split == "test"]
    cut = {" ".join(e.text.split()[:6]) for e in test_rows}

    steering = ran.read_json("trials")
    termination = ran.read_json("trials_termination")

    assert isinstance(steering, dict)
    assert isinstance(termination, dict)
    assert len(steering["prompts"]) == N_PROMPTS
    assert set(steering["prompts"]) <= cut
    assert termination["prompts"] == [
        *[e.text for e in test_rows if e.label == 1][:N_GAP],
        *[e.text for e in test_rows if e.label == 0][:N_GAP],
    ]


def test_the_summary_describes_the_run(ran: RunDir) -> None:
    summary = _summary(ran)
    none = read_trials(ran, "trials")[0]

    assert set(summary) == {
        "control_layers",
        "strengths",
        "methods",
        "n_prompts",
        "n_gap_pairs",
        "none_score",
        "judge_agreement",
        "empty_rate",
    }
    assert summary["control_layers"] == [1, 2]
    assert summary["strengths"] == list(STRENGTHS)
    assert summary["methods"] == list(METHODS)
    assert summary["n_prompts"] == N_PROMPTS
    assert summary["n_gap_pairs"] == N_GAP
    assert len(summary["none_score"]) == 2
    assert summary["none_score"] == pytest.approx(
        none.mean_score().tolist(), nan_ok=True
    )
    assert 0.0 <= summary["judge_agreement"] <= 1.0
    assert summary["empty_rate"] == pytest.approx(
        np.mean(
            [
                t.empty_rate
                for stem in ("trials", "trials_termination")
                for t in read_trials(ran, stem)
            ]
        )
    )


def test_the_layer_is_chosen_on_val_and_reported_on_test(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planted = np.full((2, len(SERIES), 3, tiny_arch.n_layers), 0.5)
    planted[0, SERIES.index("reader")] = [0.9, 0.6, 0.5]
    planted[1, SERIES.index("reader")] = [0.55, 0.6, 0.8]
    real = m1_methods.sweep_from_activations

    def spy(train: Any, evals: Any, **kwargs: Any) -> Any:
        result = real(train, evals, **kwargs)
        assert result.eval_names == ("val", "test")
        return dataclasses.replace(result, accuracy=planted)

    monkeypatch.setattr(m1_methods, "sweep_from_activations", spy)

    run_dir = _execute(tiny_arch, tmp_path, **SMALL)

    rows = [r for r in _table(run_dir) if r[1] == "correlation"]
    assert [r[0] for r in rows] == list(SMALL["methods"])
    for row in rows:
        assert row[2] == "layer1"
        assert row[3] == pytest.approx(0.55)
        assert row[4:6] == pytest.approx([0.55, 0.55])
    np.testing.assert_allclose(
        run_dir.read_arrays("curves_val")["mean"][0], [0.9, 0.6, 0.5]
    )


def test_the_termination_gap_is_pleasant_minus_unpleasant(
    ran: RunDir,
) -> None:
    termination = {t.name: t for t in read_trials(ran, "trials_termination")}
    rows = {
        (r[0], r[1]): r
        for r in _table(ran)
        if r[1] in ("termination_gap", "termination_gap_check")
    }

    def gap(name: str, column: int) -> float:
        per = termination[name].per_prompt()[:, column]
        return float(np.nanmean(per[:N_GAP] - per[N_GAP:]))

    for column, task in enumerate(("termination_gap", "termination_gap_check")):
        none = rows[("none", task)]
        assert none[2] == "none"
        assert none[3] == pytest.approx(gap("none", column))
        assert none[4] <= none[3] <= none[5]
        assert none[6] == N_GAP
        for method in METHODS:
            seeds = [
                gap(f"{method}/{s}/projection", column) for s in range(N_STEER)
            ]
            row = rows[(method, task)]
            assert row[2] == "projection"
            assert row[3] == pytest.approx(np.mean(seeds))
            assert row[4] == pytest.approx(min(seeds))
            assert row[5] == pytest.approx(max(seeds))
            assert row[6] == N_GAP


def test_the_termination_probe_rows_report_before_and_after(
    ran: RunDir,
) -> None:
    rows = {(r[0], r[2]): r for r in _table(ran) if r[1] == "termination_probe"}

    assert set(rows) == {(m, s) for m in METHODS for s in ("before", "after")}
    for row in rows.values():
        assert 0.0 <= row[4] <= row[3] <= row[5] <= 1.0
        assert row[6] == N_STEER * 2
    assert rows[("mean_diff", "after")][3:6] == [0.5, 0.5, 0.5]


def test_plot_writes_four_figures_in_three_formats(ran: RunDir) -> None:
    paths = plot(ran, ran.path / "figs")

    assert sorted(p.name for p in paths) == sorted(
        f"{stem}.{extension}"
        for stem in (
            "curves",
            "curves_val",
            "curves_success",
            "curves_success_check",
        )
        for extension in ("pdf", "png", "svg")
    )
    assert all(p.stat().st_size > 0 for p in paths)


def test_every_method_steers_at_the_same_layers_with_calibrated_norms(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arms: list[dict[str, Any]] = []
    norm_calls: list[tuple[tuple[str, ...], tuple[int, ...]]] = []
    sweeps: dict[str, Any] = {}
    real_arm = m1_methods.reader_arm
    real_norms = m1_methods.mean_residual_norms
    real_sweep = m1_methods.sweep_from_activations

    def arm_spy(loaded: Any, settings: Any, reader: Any, **kwargs: Any) -> Any:
        arms.append({"reader": reader, **kwargs})
        return real_arm(loaded, settings, reader, **kwargs)

    def norms_spy(
        loaded: Any, texts: Sequence[str], layers: Sequence[int], **kwargs: Any
    ) -> Any:
        norm_calls.append((tuple(texts), tuple(layers)))
        return real_norms(loaded, texts, layers, **kwargs)

    def sweep_spy(train: Any, evals: Any, **kwargs: Any) -> Any:
        sweeps[kwargs["method"]] = real_sweep(train, evals, **kwargs)
        return sweeps[kwargs["method"]]

    monkeypatch.setattr(m1_methods, "reader_arm", arm_spy)
    monkeypatch.setattr(m1_methods, "mean_residual_norms", norms_spy)
    monkeypatch.setattr(m1_methods, "sweep_from_activations", sweep_spy)
    config = _config(**SMALL)
    run_dir = _execute(tiny_arch, tmp_path, **SMALL)

    held = split_held_out(
        _texts(), n_calibration=8, n_neutral=8, seed=config.seed
    )
    assert norm_calls == [(held.calibration, (1, 2))]
    assert len(arms) == len(config.methods) * N_STEER * (len(STEERED) + 1)
    for method in config.methods:
        readers = sweeps[method].readers[:N_STEER]
        used = [a for a in arms if a["reader"].method == method]
        assert {id(a["reader"]) for a in used} == {id(r) for r in readers}
        steered = [a for a in used if a["op"] == "linear"]
        projected = [a for a in used if a["op"] == "projection"]
        assert sorted({a["coeff"] for a in steered}) == sorted(STEERED)
        assert all(a["layers"] == (1, 2) for a in used)
        assert all(sorted(a["norms"]) == [1, 2] for a in steered)
        assert len(projected) == N_STEER
        assert all(
            (a["coeff"], a["mode"], a["mask"]) == (1.0, "absolute", "all")
            for a in projected
        )
    steered_arms = [a for a in arms if a["op"] == "linear"]
    projection_arms = [a for a in arms if a["op"] == "projection"]
    steered_names = [t.name for t in read_trials(run_dir, "trials")][1:]
    projection_names = [
        t.name for t in read_trials(run_dir, "trials_termination")
    ][1:]
    for arm, name in zip(steered_arms, steered_names, strict=True):
        method, seed, coeff = name.split("/")
        assert arm["reader"] is sweeps[method].readers[int(seed)]
        assert arm["coeff"] == float(coeff)
    for arm, name in zip(projection_arms, projection_names, strict=True):
        method, seed, kind = name.split("/")
        assert arm["reader"] is sweeps[method].readers[int(seed)]
        assert kind == "projection"


def test_the_readers_train_on_train_and_score_on_test_pairs_only(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    splits: list[set[str]] = []
    origin: dict[tuple[str, ...], str] = {}
    collected: list[tuple[str, Any]] = []
    probes: list[tuple[int, int]] = []
    sources: list[tuple[set[str], set[str]]] = []
    real_stimuli = m1_methods.build_stimulus_set
    real_activations = m1_methods.stimulus_activations
    real_probe = m1_methods.erased_probe_accuracy

    def stimuli_spy(examples: Any, *args: Any, **kwargs: Any) -> Any:
        splits.append({e.split for e in examples})
        built = real_stimuli(examples, *args, **kwargs)
        origin[built.texts] = ",".join(sorted(splits[-1]))
        return built

    def activations_spy(loaded: Any, stimuli: Any, **kwargs: Any) -> Any:
        act = real_activations(loaded, stimuli, **kwargs)
        collected.append((origin[stimuli.texts], act))
        return act

    def came_from(array: Any) -> set[str]:
        return {
            split
            for split, act in collected
            if np.shares_memory(array, act.data)
        }

    def probe_spy(
        train_h: Any, train_labels: Any, test_h: Any, *args: Any, **kwargs: Any
    ) -> Any:
        probes.append((train_h.shape[0], test_h.shape[0]))
        sources.append((came_from(train_h), came_from(test_h)))
        return real_probe(train_h, train_labels, test_h, *args, **kwargs)

    monkeypatch.setattr(m1_methods, "build_stimulus_set", stimuli_spy)
    monkeypatch.setattr(m1_methods, "stimulus_activations", activations_spy)
    monkeypatch.setattr(m1_methods, "erased_probe_accuracy", probe_spy)
    config = _config(**SMALL)

    _execute(tiny_arch, tmp_path, **SMALL)

    assert splits == [{"train"}] * config.n_seeds + [{"val"}, {"test"}]
    n_probes = len(config.methods) * N_STEER * 2
    assert probes == [(2 * config.n_pairs, 2 * N_TEST_PAIRS)] * n_probes
    assert sources == [({"train"}, {"test"})] * n_probes


def test_the_prompts_are_distinct_even_when_pairs_open_alike(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    utility = _utility(shared_openings=True)

    run_dir = _execute(
        tiny_arch,
        tmp_path,
        utility=utility,
        **{**SMALL, "n_prompts": N_PROMPTS},
    )

    payload = run_dir.read_json("trials")
    assert isinstance(payload, dict)
    assert len(set(payload["prompts"])) == len(payload["prompts"]) == N_PROMPTS


def test_the_shortage_of_prompts_counts_the_distinct_openings(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("nothing may run before the checks")

    monkeypatch.setattr(m1_methods, "stimulus_activations", forbidden)
    utility = _utility(shared_openings=True)

    with pytest.raises(ValueError, match="n_prompts"):
        _execute(
            tiny_arch, tmp_path, utility=utility, **{**SMALL, "n_prompts": 13}
        )


def test_the_judge_agreement_is_one_when_both_columns_agree(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(tiny_arch, tmp_path, score=_twin_score, **SMALL)

    summary = _summary(run_dir)
    assert summary["judge_agreement"] == 1.0
    assert summary["none_score"][0] == summary["none_score"][1]


def test_the_judge_agreement_counts_the_cells_on_the_same_side_of_half(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(tiny_arch, tmp_path, score=_negated_score, **SMALL)

    rows = _table(run_dir)
    primary = [
        r[3] for r in rows if r[1] == "manipulation" and r[3] is not None
    ]
    seeds = _seed_rows(run_dir)
    first = [r for r in seeds if r[3] == "primary"]
    second = [r for r in seeds if r[3] == "check"]
    assert [(r[0], r[1], r[2]) for r in first] == [
        (r[0], r[1], r[2]) for r in second
    ]
    assert all(
        (c[4], c[5], c[6]) == (r[5], r[4], r[6])
        for r, c in zip(first, second, strict=True)
    )
    expected = np.mean([rate == 0.5 for rate in primary])
    assert _summary(run_dir)["judge_agreement"] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"coeffs": (-0.1, 0.1)}, "coeffs"),
        ({"coeffs": (-0.1, 0.0, 0.1, 0.3)}, "coeffs"),
        ({"coeffs": (-0.2, -0.1, 0.0, 0.1)}, "coeffs"),
        ({"n_steer_seeds": 0}, "n_steer_seeds"),
        ({"n_steer_seeds": 4}, "n_steer_seeds"),
        ({"layers": (0,)}, "layers"),
        ({"layers": (1, 4)}, "layers"),
        ({"layers": (1, 1)}, "layers"),
        ({"n_prompts": 30}, "n_prompts"),
        ({"methods": ("logistic", "logistic")}, "methods"),
        ({"coeffs": (-0.1, 0.0, 0.1, 0.1)}, "coeffs"),
        ({"coeffs": (-0.1, -0.10000001, 0.0, 0.1, 0.10000001)}, "coeffs"),
    ],
    ids=[
        "no-zero",
        "missing-negative",
        "unpaired-negative",
        "no-steering-seed",
        "too-many-steering-seeds",
        "layer-zero",
        "layer-above-depth",
        "repeated-layer",
        "more-prompts-than-scenarios",
        "repeated-method",
        "repeated-strength",
        "strengths-with-one-label",
    ],
)
def test_bad_inputs_fail_before_any_forward(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
    message: str,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("nothing may run before the checks")

    for name in (
        "stimulus_activations",
        "mean_residual_norms",
        "sweep_from_activations",
        "run_trial",
    ):
        monkeypatch.setattr(m1_methods, name, forbidden)

    with pytest.raises(ValueError, match=message):
        _execute(tiny_arch, tmp_path, **overrides)


def test_a_nan_coefficient_fails_before_any_forward(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("nothing may run before the checks")

    monkeypatch.setattr(m1_methods, "stimulus_activations", forbidden)
    run_dir = RunDir.create(_config(), out_root=tmp_path)
    config = _config(coeffs=(-0.1, NAN, 0.0, 0.1))

    with pytest.raises(ValueError, match="finite"):
        execute(config, run_dir, tiny_arch, _utility(), _texts(), stub_score)


def test_a_test_split_without_a_complete_pair_fails_before_any_forward(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("nothing may run before the checks")

    monkeypatch.setattr(m1_methods, "stimulus_activations", forbidden)
    utility = [
        e for e in _utility() if not (e.split == "test" and e.label == 0)
    ]

    with pytest.raises(ValueError, match="both scenarios"):
        _execute(tiny_arch, tmp_path, utility=utility)


class _FakeJudge:
    def __init__(self, value: float) -> None:
        self.value = value

    def score(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.full(len(texts), self.value, dtype=np.float32)


def test_run_loads_the_model_the_data_and_both_judges(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models: list[str] = []
    judges: list[str] = []
    config = _config(
        judge="sentiment-primary", judge_check="sentiment-secondary", **SMALL
    )

    def fake_load_model(key: str) -> LoadedModel:
        models.append(key)
        return tiny_arch

    def fake_load_judge(key: str) -> _FakeJudge:
        judges.append(key)
        return _FakeJudge(0.2 if key == config.judge else 0.8)

    def fake_wikitext() -> list[Example]:
        return [
            Example(uid=f"w{i}", text=text, label=-1, split="test")
            for i, text in enumerate(_texts())
        ]

    monkeypatch.setattr(m1_methods, "load_model", fake_load_model)
    monkeypatch.setattr(m1_methods, "load_judge", fake_load_judge)
    monkeypatch.setattr(m1_methods, "load_ethics_utility", _utility)
    monkeypatch.setattr(m1_methods, "load_wikitext", fake_wikitext)
    run_dir = RunDir.create(config, out_root=tmp_path)

    run(config, run_dir)

    assert models == ["tiny"]
    assert judges == ["sentiment-primary", "sentiment-secondary"]
    assert _summary(run_dir)["none_score"] == pytest.approx([0.2, 0.8])
    assert run_dir.meta["finished_at"] is None


@pytest.mark.slow
def test_m1_on_gpt2_reports_success_and_gaps_under_both_judges(
    tmp_path: Path,
) -> None:
    try:
        utility = load_ethics_utility(offline=True)
        texts = [e.text for e in load_wikitext(offline=True)]
        primary = load_judge("sentiment-primary")
        check = load_judge("sentiment-secondary")
    except (OfflineCacheMissError, OSError) as error:
        pytest.skip(f"a cache is missing: {error}")

    def score(batch: Sequence[str]) -> NDArray[np.float32]:
        return np.column_stack([primary.score(batch), check.score(batch)])

    config = Config(
        methods=("logistic", "random"),
        n_seeds=2,
        n_steer_seeds=1,
        n_eval_pairs=128,
        coeffs=(-0.2, 0.0, 0.2),
        n_prompts=32,
        n_termination=16,
    )
    run_dir = RunDir.create(config, out_root=tmp_path)

    started = time.perf_counter()
    execute(config, run_dir, load_model("gpt2"), utility, texts, score)
    elapsed = time.perf_counter() - started

    generated = sum(
        int(np.stack([t.lengths for t in read_trials(run_dir, stem)]).sum())
        for stem in ("trials", "trials_termination")
    )
    for row in _table(run_dir):
        if row[1] != "correlation":
            print(row)
    print(_summary(run_dir))
    print(f"{elapsed / generated:.4f} seconds per generated token")
    assert generated > 0


def test_the_agreement_gives_exactly_one_half_its_own_side() -> None:
    primary = np.array([0.3, 0.5, 0.5, 0.8, NAN])
    check = np.array([0.4, 0.5, 0.3, 0.2, 0.9])

    assert m1_methods._agreement(primary, check) == pytest.approx(0.5)
    assert np.isnan(m1_methods._agreement(primary[-1:], check[:1] * NAN))


def test_the_spread_ignores_non_finite_values() -> None:
    assert m1_methods._spread([1.0, NAN, 3.0]) == (2.0, 1.0, 3.0)
    assert all(np.isnan(m1_methods._spread([NAN, NAN])))


def test_the_effects_table_has_one_row_per_arm_with_planted_side_effects(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arms: dict[int, dict[str, Any]] = {}
    calls: list[dict[str, Any]] = []
    real_arm = m1_methods.reader_arm
    real_plain = m1_methods.plain_arm

    def arm_spy(loaded: Any, settings: Any, reader: Any, **kwargs: Any) -> Any:
        arm = real_arm(loaded, settings, reader, **kwargs)
        arms[id(arm)] = {"reader": reader, **kwargs}
        return arm

    def plain_spy(*args: Any, **kwargs: Any) -> Any:
        arm = real_plain(*args, **kwargs)
        arms[id(arm)] = {"plain": True}
        return arm

    def effects_spy(
        loaded: Any,
        arm: Any,
        trial: Any,
        neutral: Sequence[str],
        *,
        batch_size: int,
    ) -> SideEffects:
        calls.append(
            {
                "arm": arms[id(arm)],
                "name": trial.name,
                "neutral": tuple(neutral),
                "batch_size": batch_size,
            }
        )
        k = float(len(calls))
        return SideEffects(
            perplexity=None if k == 1.0 else 10.0 * k,
            fluency=-k,
            distinct_1=k / 100,
            distinct_2=k / 50,
            mean_length=k + 0.5,
            empty_rate=k / 1000,
        )

    monkeypatch.setattr(m1_methods, "reader_arm", arm_spy)
    monkeypatch.setattr(m1_methods, "plain_arm", plain_spy)
    monkeypatch.setattr(m1_methods, "side_effects", effects_spy)
    config = _config(**SMALL)
    run_dir = _execute(tiny_arch, tmp_path, **SMALL)

    payload = run_dir.read_json("table_effects")
    assert isinstance(payload, dict)
    assert payload["columns"] == [
        "method",
        "seed",
        "coeff",
        "perplexity",
        "fluency",
        "distinct_1",
        "distinct_2",
        "mean_length",
        "empty_rate",
    ]
    trials = read_trials(run_dir, "trials")
    assert len(payload["rows"]) == len(trials)
    assert len(calls) == len(trials)
    held = split_held_out(
        _texts(), n_calibration=8, n_neutral=8, seed=config.seed
    )
    assert calls[0]["arm"] == {"plain": True}
    assert payload["rows"][0] == [
        "none",
        None,
        0.0,
        None,
        -1.0,
        0.01,
        0.02,
        1.5,
        0.001,
    ]
    for k, (row, call, trial) in enumerate(
        zip(payload["rows"], calls, trials, strict=True), start=1
    ):
        assert call["name"] == trial.name.rsplit("/", 1)[-1]
        assert call["neutral"] == held.neutral
        assert call["batch_size"] == config.batch_size
        if k > 1:
            method, seed, coeff = trial.name.split("/")
            assert row[:3] == [method, int(seed), float(coeff)]
            assert call["arm"]["coeff"] == float(coeff)
            assert call["arm"]["op"] == "linear"
            assert call["arm"]["reader"].method == method
            assert row[3:] == pytest.approx(
                [10.0 * k, -k, k / 100, k / 50, k + 0.5, k / 1000]
            )


def test_without_fluency_no_effects_are_measured_or_written(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("side_effects must not run")

    monkeypatch.setattr(m1_methods, "side_effects", forbidden)
    run_dir = _execute(tiny_arch, tmp_path, fluency=False, **SMALL)

    names = sorted(p.name for p in run_dir.path.iterdir() if p.is_file())
    assert names == FILES_WITHOUT_EFFECTS


def test_the_fluency_switch_changes_the_config_hash() -> None:
    assert _config().config_hash() != _config(fluency=False).config_hash()
