"""Tests for the C1 sentiment steering experiment."""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any
from typing import NamedTuple

from huggingface_hub import snapshot_download
import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.cli import load_experiment
from tomography.control.controllers import mean_residual_norms
from tomography.datasets.records import Example
from tomography.datasets.sst2 import load_sst2
from tomography.datasets.wikitext import load_wikitext
from tomography.experiments import c1_sentiment_steering
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.c1_sentiment_steering import Config
from tomography.experiments.c1_sentiment_steering import default_sweep_layers
from tomography.experiments.c1_sentiment_steering import dose_with_none
from tomography.experiments.c1_sentiment_steering import draw_prompts
from tomography.experiments.c1_sentiment_steering import execute
from tomography.experiments.c1_sentiment_steering import judge_agreement
from tomography.experiments.c1_sentiment_steering import plot
from tomography.experiments.c1_sentiment_steering import prompt_fragments
from tomography.experiments.c1_sentiment_steering import run
from tomography.experiments.c1_sentiment_steering import slope_layer
from tomography.experiments.c1_sentiment_steering import window_around
from tomography.experiments.effects import read_trials
from tomography.experiments.effects import split_held_out
from tomography.experiments.r1_sentiment import select_examples
from tomography.experiments.steering import Arm
from tomography.experiments.steering import GenSettings
from tomography.experiments.steering import dose_response
from tomography.experiments.steering import dose_slope
from tomography.experiments.steering import plain_arm
from tomography.experiments.steering import reader_arm
from tomography.experiments.steering import run_trial
from tomography.judges import JUDGES
from tomography.judges import load_judge
from tomography.model import LoadedModel
from tomography.model import load_model

PROMPTS = ["The quick brown fox", "The lazy dog", "Honest answers are"]
SETTINGS = GenSettings(
    max_new_tokens=4,
    temperature=0.8,
    top_p=0.95,
    num_samples=2,
    seed=3,
    bos="none",
    batch_size=2,
)


@dataclasses.dataclass(frozen=True)
class StubSource:
    """A fitted-reader stand-in: unit directions and positive signs."""

    layers: tuple[int, ...]
    directions: NDArray[np.float32]
    signs: NDArray[np.float32]


def _source(loaded: LoadedModel) -> StubSource:
    rng = np.random.default_rng(0)
    directions = rng.standard_normal((2, loaded.d_model)).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return StubSource(
        layers=(1, 2), directions=directions, signs=np.ones(2, np.float32)
    )


def _score(texts: Sequence[str]) -> NDArray[np.float32]:
    vowels = [sum(ch in "aeiou" for ch in t) / max(len(t), 1) for t in texts]
    spaces = [t.count(" ") / max(len(t), 1) for t in texts]
    return np.stack([vowels, spaces], axis=1).astype(np.float32)


@pytest.fixture
def full_length(
    tiny_arch: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> LoadedModel:
    """A tiny model whose rows never end early on the shared pad/eos id."""
    monkeypatch.setattr(tiny_arch.model.generation_config, "eos_token_id", None)
    return tiny_arch


@pytest.mark.parametrize(
    ("n_layers", "count", "expected"),
    [
        (12, 6, (1, 3, 5, 8, 10, 12)),
        (28, 6, (1, 6, 12, 17, 23, 28)),
        (3, 6, (1, 2, 3)),
        (12, 4, (1, 5, 8, 12)),
        (12, 1, (1,)),
    ],
    ids=["gpt2", "qwen3", "tiny", "four", "one"],
)
def test_default_sweep_layers_spaces_the_depth(
    n_layers: int, count: int, expected: tuple[int, ...]
) -> None:
    assert default_sweep_layers(n_layers, count) == expected


def test_default_sweep_layers_defaults_to_six() -> None:
    assert default_sweep_layers(12) == default_sweep_layers(12, 6)


def test_default_sweep_layers_rejects_a_count_below_one() -> None:
    with pytest.raises(ValueError, match="count"):
        default_sweep_layers(12, 0)


@pytest.mark.parametrize(
    ("layer", "width", "expected"),
    [
        (6, 3, (5, 6, 7)),
        (1, 3, (1, 2, 3)),
        (12, 3, (10, 11, 12)),
        (2, 3, (1, 2, 3)),
        (6, 4, (4, 5, 6, 7)),
        (6, 2, (5, 6)),
        (1, 2, (1, 2)),
        (12, 2, (11, 12)),
        (6, 1, (6,)),
        (6, 12, tuple(range(1, 13))),
        (1, 12, tuple(range(1, 13))),
    ],
    ids=[
        "centered",
        "low-edge",
        "high-edge",
        "near-low-edge",
        "even-four",
        "even-two",
        "even-low-edge",
        "even-high-edge",
        "single",
        "whole-depth",
        "whole-depth-at-edge",
    ],
)
def test_window_around_is_centered_and_stays_inside_the_depth(
    layer: int, width: int, expected: tuple[int, ...]
) -> None:
    assert window_around(layer, width, 12) == expected


@pytest.mark.parametrize(
    ("layer", "width"),
    [(6, 0), (6, 13), (0, 3), (13, 3)],
    ids=["width-zero", "width-too-wide", "layer-zero", "layer-too-deep"],
)
def test_window_around_rejects_values_outside_the_depth(
    layer: int, width: int
) -> None:
    with pytest.raises(ValueError, match=r"width|layer"):
        window_around(layer, width, 12)


COEFFS = (-0.2, 0.0, 0.2)


def _ramp(slope: float, readers: int = 2) -> NDArray[np.float64]:
    return np.tile(np.array(COEFFS) * slope + 0.5, (readers, 1))


def test_slope_layer_picks_the_layer_with_the_largest_mean_slope() -> None:
    per_seed = np.stack([_ramp(0.1), _ramp(2.0), _ramp(-3.0)])
    assert slope_layer(per_seed, COEFFS, (3, 6, 9)) == 6


def test_slope_layer_averages_the_readers() -> None:
    steep_and_flat = np.stack([_ramp(4.0, 1)[0], _ramp(0.0, 1)[0]])
    steady = _ramp(1.5)
    assert slope_layer(np.stack([steep_and_flat, steady]), COEFFS, (3, 6)) == 3


def test_slope_layer_skips_a_reader_with_fewer_than_two_finite_points() -> None:
    lonely = np.array([0.5, np.nan, np.nan])
    first = np.stack([_ramp(2.0, 1)[0], lonely])
    assert slope_layer(np.stack([first, _ramp(1.5)]), COEFFS, (3, 6)) == 3


def test_slope_layer_never_picks_a_layer_without_a_finite_slope() -> None:
    blank = np.full((2, 3), np.nan)
    per_seed = np.stack([blank, _ramp(-1.0)])
    assert slope_layer(per_seed, COEFFS, (3, 6)) == 6


def test_slope_layer_raises_when_no_layer_has_a_slope() -> None:
    with pytest.raises(ValueError, match="slope"):
        slope_layer(np.full((2, 2, 3), np.nan), COEFFS, (3, 6))


def test_slope_layer_breaks_ties_toward_the_first_layer() -> None:
    per_seed = np.stack([_ramp(1.0), _ramp(1.0), _ramp(0.5)])
    assert slope_layer(per_seed, COEFFS, (3, 6, 9)) == 3


def test_prompt_fragments_cut_the_first_words_and_drop_repeats() -> None:
    texts = [
        "The film is great fun",
        "The film is a bore",
        "  Spaced   out   words here ",
        "A tiny one",
    ]
    assert prompt_fragments(texts, 3) == [
        "The film is",
        "Spaced out words",
        "A tiny one",
    ]


def test_prompt_fragments_drop_empty_texts_and_keep_short_ones() -> None:
    assert prompt_fragments(["", "   ", "Hi", "Hi there you all"], 2) == [
        "Hi",
        "Hi there",
    ]


def test_prompt_fragments_reject_fewer_than_one_word() -> None:
    with pytest.raises(ValueError, match="words"):
        prompt_fragments(["a b c"], 0)


POOL = [f"prompt {i}" for i in range(20)]


def test_draw_prompts_returns_disjoint_lists_from_one_permutation() -> None:
    first, second = draw_prompts(POOL, 5, sets=2, seed=4)
    order = np.random.default_rng(4).permutation(len(POOL))
    assert first == [POOL[i] for i in order[:5]]
    assert second == [POOL[i] for i in order[5:10]]
    assert not set(first) & set(second)


def test_draw_prompts_is_deterministic_and_follows_the_seed() -> None:
    assert draw_prompts(POOL, 5, sets=2, seed=4) == draw_prompts(
        POOL, 5, sets=2, seed=4
    )
    assert draw_prompts(POOL, 5, sets=2, seed=4) != draw_prompts(
        POOL, 5, sets=2, seed=5
    )


def test_draw_prompts_can_use_the_whole_pool() -> None:
    lists = draw_prompts(POOL, 5, sets=4, seed=0)
    assert sorted(p for chosen in lists for p in chosen) == sorted(POOL)


@pytest.mark.parametrize(
    ("n", "sets"),
    [(0, 2), (5, 0), (11, 2)],
    ids=["no-prompts", "no-sets", "short-pool"],
)
def test_draw_prompts_rejects_a_bad_size_or_a_short_pool(
    n: int, sets: int
) -> None:
    with pytest.raises(ValueError, match=r"n |sets|pool"):
        draw_prompts(POOL, n, sets=sets, seed=0)


def _arms(
    loaded: LoadedModel, calls: list[float]
) -> tuple[Callable[[float], Arm], Arm]:
    source = _source(loaded)

    def make(coeff: float) -> Arm:
        calls.append(coeff)
        return reader_arm(
            loaded,
            SETTINGS,
            source,
            layers=[1, 2],
            coeff=coeff,
            norms={1: 5.0, 2: 5.0},
        )

    return make, plain_arm(loaded, SETTINGS)


def test_dose_with_none_equals_dose_response_bit_for_bit(
    full_length: LoadedModel,
) -> None:
    calls: list[float] = []
    make, plain = _arms(full_length, calls)
    coeffs = (-0.3, 0.0, 0.3)
    none = run_trial(plain, PROMPTS, _score, columns=2)

    shared = dose_with_none(make, coeffs, PROMPTS, _score, none, columns=2)
    full = dose_response(make, coeffs, PROMPTS, _score, columns=2)

    assert shared.coeffs == full.coeffs == coeffs
    assert shared.trials[1] is none
    for got, expected in zip(shared.trials, full.trials, strict=True):
        assert got.completions == expected.completions
        np.testing.assert_array_equal(got.scores, expected.scores)
        np.testing.assert_array_equal(got.lengths, expected.lengths)
    assert calls[:2] == [-0.3, 0.3]


def test_dose_with_none_never_builds_the_arm_of_coefficient_zero(
    full_length: LoadedModel,
) -> None:
    calls: list[float] = []
    make, plain = _arms(full_length, calls)
    none = run_trial(plain, PROMPTS, _score, columns=2)

    dose = dose_with_none(
        make, (0.0, 0.1, 0.2), PROMPTS, _score, none, columns=2
    )

    assert calls == [0.1, 0.2]
    assert dose.coeffs == (0.0, 0.1, 0.2)
    assert dose.trials[0] is none
    assert [t.name for t in dose.trials[1:]] == ["0.1", "0.2"]


@pytest.mark.parametrize(
    ("coeffs", "prompts", "message"),
    [
        ((-0.1, 0.1), PROMPTS, "0.0"),
        ((0.0,), PROMPTS, "nonzero"),
        ((-0.1, 0.0, 0.1), PROMPTS[:2], "prompts"),
    ],
    ids=["no-zero", "only-zero", "other-prompts"],
)
def test_dose_with_none_rejects_a_grid_or_prompts_that_do_not_fit(
    full_length: LoadedModel,
    coeffs: tuple[float, ...],
    prompts: list[str],
    message: str,
) -> None:
    calls: list[float] = []
    make, plain = _arms(full_length, calls)
    none = run_trial(plain, PROMPTS, _score, columns=2)

    with pytest.raises(ValueError, match=message):
        dose_with_none(make, coeffs, prompts, _score, none, columns=2)
    assert calls == []


def test_config_defaults() -> None:
    config = Config()
    assert config.experiment == "c1"
    assert config.model == "gpt2"
    assert config.method == "logistic"
    assert config.judge == "sentiment-primary"
    assert config.judge_check == "sentiment-secondary"
    assert config.n_seeds == 10
    assert config.n_steer_seeds == 3
    assert config.n_pairs == 64
    assert config.n_eval_pairs == 256
    assert config.min_words == 3
    assert config.sweep_layers is None
    assert config.window == 3
    assert config.ops == ("linear", "piecewise")
    assert config.truncate_words == 3
    assert len(config.prompt_positive) == len(config.prompt_negative) == 2
    assert config.n_prompts == 100
    assert config.mode == "relative"


def test_config_round_trips_through_a_dict_with_its_tuple_fields() -> None:
    config = Config(
        sweep_layers=(2, 4),
        ops=("piecewise",),
        coeffs=(-0.1, 0.0, 0.1),
        prompt_positive=("Be happy. ",),
        prompt_negative=("Be sad. ", "Be angry. "),
    )
    assert Config.from_dict(config.to_dict()) == config


def test_the_cli_discovers_the_experiment() -> None:
    experiment = load_experiment("c1")
    assert experiment.config_cls is Config
    assert experiment.run is run
    assert experiment.plot is plot


def _ramp_stack(*slopes: float) -> NDArray[np.float64]:
    return np.stack([_ramp(slope) for slope in slopes])


def test_judge_agreement_counts_the_layers_with_one_sign_of_slope() -> None:
    primary = _ramp_stack(1.0, 2.0, -1.0, -2.0)
    check = _ramp_stack(0.5, -0.5, -0.5, 0.5)
    assert judge_agreement(primary, check, COEFFS) == pytest.approx(0.5)
    assert judge_agreement(primary, primary * 3, COEFFS) == pytest.approx(1.0)
    assert judge_agreement(primary, -primary, COEFFS) == pytest.approx(0.0)


def test_judge_agreement_leaves_out_layers_without_a_slope_under_a_judge() -> (
    None
):
    blank = np.full((1, 2, 3), np.nan)
    primary = np.concatenate([_ramp_stack(1.0, 1.0), blank])
    check = np.concatenate([_ramp_stack(1.0, -1.0), _ramp_stack(1.0)])
    assert judge_agreement(primary, check, COEFFS) == pytest.approx(0.5)


def test_judge_agreement_is_nan_when_no_layer_has_both_slopes() -> None:
    blank = np.full((2, 2, 3), np.nan)
    assert np.isnan(judge_agreement(blank, _ramp_stack(1.0, 2.0), COEFFS))


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
SWEEP = (1, 2)
NONZERO = (-0.2, 0.2)
READERS = (0, 1)
OPS = ("linear", "piecewise")
PROMPTING = ("prompting+1", "prompting+2", "prompting-1", "prompting-2")


def _sst2(n: int = 80) -> list[Example]:
    rng = np.random.default_rng(2)
    return [
        Example(
            uid=f"s{i}",
            text=" ".join(rng.choice(WORDS, 6)),
            label=i % 2,
            split="train" if (i // 2) % 4 else "val",
        )
        for i in range(n)
    ]


def _texts(n: int = 60) -> list[str]:
    return [
        " ".join(np.random.default_rng(i).choice(WORDS, 12)) for i in range(n)
    ]


def _config(**overrides: Any) -> Config:
    base = Config(
        model="tiny",
        n_seeds=3,
        n_steer_seeds=2,
        n_pairs=8,
        n_eval_pairs=8,
        n_prompts=8,
        n_calibration=10,
        n_neutral=8,
        max_new_tokens=5,
        coeffs=(-0.2, 0.0, 0.2),
        batch_size=8,
        sweep_layers=SWEEP,
        window=2,
    )
    return dataclasses.replace(base, **overrides)


class Finished(NamedTuple):
    config: Config
    run_dir: RunDir
    loaded: LoadedModel


_FINISHED: dict[str, Finished] = {}


@pytest.fixture
def finished(
    tiny_arch: LoadedModel, tmp_path_factory: pytest.TempPathFactory
) -> Finished:
    """One default tiny run per architecture, shared by the read-only tests."""
    key = type(tiny_arch.model).__name__
    if key not in _FINISHED:
        config = _config()
        run_dir = RunDir.create(config, out_root=tmp_path_factory.mktemp("c1"))
        execute(config, run_dir, tiny_arch, _sst2(), _texts(), _score)
        _FINISHED[key] = Finished(config, run_dir, tiny_arch)
    return _FINISHED[key]


def test_execute_writes_sixteen_files(finished: Finished) -> None:
    names = sorted(p.name for p in finished.run_dir.path.iterdir())
    stems = (
        "curves_layers",
        "curves_layers_check",
        "curves_operators",
        "curves_perplexity",
        "trials",
        "trials_sweep",
    )
    assert names == sorted(
        [
            "config.json",
            "meta.json",
            "summary.json",
            "table.json",
            *(f"{stem}.{ext}" for stem in stems for ext in ("npz", "json")),
        ]
    )


def test_the_trials_are_named_after_what_made_them(
    finished: Finished,
) -> None:
    sweep = read_trials(finished.run_dir, "trials_sweep")
    assert [t.name for t in sweep] == [
        "none",
        *(
            f"layer{layer}/{seed}/{c:g}"
            for layer in SWEEP
            for seed in READERS
            for c in NONZERO
        ),
    ]
    trials = read_trials(finished.run_dir, "trials")
    assert [t.name for t in trials] == [
        "none",
        *(
            f"{op}/{seed}/{c:g}"
            for op in OPS
            for seed in READERS
            for c in NONZERO
        ),
        *PROMPTING,
    ]
    assert len(sweep) == 9
    assert len(trials) == 13
    assert trials[0].scores.shape == (8, 2, 2)


def test_selection_and_report_prompts_are_disjoint_draws(
    finished: Finished,
) -> None:
    val = select_examples(_sst2(), split="val", min_words=3)
    pool = prompt_fragments([e.text for e in val], 3)
    select, report = draw_prompts(
        pool, 8, sets=2, seed=derive_seed(finished.config.seed, "prompts")
    )
    sweep = read_trials(finished.run_dir, "trials_sweep")
    trials = read_trials(finished.run_dir, "trials")
    assert {t.prompts for t in sweep} == {tuple(select)}
    assert {t.prompts for t in trials} == {tuple(report)}
    assert not set(select) & set(report)


def test_coefficient_zero_is_one_trial_shared_by_every_series(
    finished: Finished,
) -> None:
    sweep_none = read_trials(finished.run_dir, "trials_sweep")[0]
    report_none = read_trials(finished.run_dir, "trials")[0]
    for stem, none, column in (
        ("curves_layers", sweep_none, 0),
        ("curves_layers_check", sweep_none, 1),
        ("curves_operators", report_none, 0),
    ):
        curves = finished.run_dir.read_arrays(stem)
        at = curves["x"].tolist().index(0.0)
        np.testing.assert_allclose(
            curves["per_seed"][:, :, at], none.mean_score()[column]
        )


def test_the_curves_follow_the_layers_the_operators_and_the_judges(
    finished: Finished,
) -> None:
    layers = finished.run_dir.read_arrays("curves_layers")
    assert layers["x"].tolist() == [-0.2, 0.0, 0.2]
    assert layers["per_seed"].shape == (2, 2, 3)
    sidecar = finished.run_dir.read_json("curves_layers")
    assert isinstance(sidecar, dict)
    assert sidecar["series"] == ["layer 1", "layer 2"]
    assert sidecar["xlabel"] == "Coefficient (relative)"
    assert "primary" in sidecar["ylabel"]
    check = finished.run_dir.read_json("curves_layers_check")
    assert isinstance(check, dict)
    assert check["series"] == ["layer 1", "layer 2"]
    assert "check" in check["ylabel"]
    operators = finished.run_dir.read_json("curves_operators")
    assert isinstance(operators, dict)
    assert operators["series"] == list(OPS)
    per_seed = finished.run_dir.read_arrays("curves_operators")["per_seed"]
    assert per_seed.shape == (2, 2, 3)


def test_the_perplexity_curve_is_finite_and_starts_from_the_plain_model(
    finished: Finished,
) -> None:
    curves = finished.run_dir.read_arrays("curves_perplexity")
    assert curves["per_seed"].shape == (1, 2, 3)
    assert np.isfinite(curves["per_seed"]).all()
    sidecar = finished.run_dir.read_json("curves_perplexity")
    assert isinstance(sidecar, dict)
    assert sidecar["series"] == ["neutral perplexity"]
    held = split_held_out(_texts(), n_calibration=10, n_neutral=8, seed=0)
    plain = plain_arm(finished.loaded, GenSettings.from_config(finished.config))
    assert plain.perplexity is not None
    np.testing.assert_allclose(
        curves["per_seed"][0, :, 1], plain.perplexity(held.neutral)
    )


def _table(finished: Finished) -> list[dict[str, Any]]:
    table = finished.run_dir.read_json("table")
    assert isinstance(table, dict)
    return [
        dict(zip(table["columns"], row, strict=True)) for row in table["rows"]
    ]


def test_the_table_lists_none_linear_and_the_prompting_arms(
    finished: Finished,
) -> None:
    table = finished.run_dir.read_json("table")
    assert isinstance(table, dict)
    assert table["columns"] == [
        "arm",
        "coeff",
        "seed",
        "score",
        "score_check",
        "perplexity",
        "fluency",
        "distinct_1",
        "distinct_2",
        "mean_length",
        "empty_rate",
    ]
    rows = _table(finished)
    assert [r["arm"] for r in rows] == ["none", *["linear"] * 4, *PROMPTING]
    assert (rows[0]["coeff"], rows[0]["seed"]) == (0.0, None)
    assert {(r["coeff"], r["seed"]) for r in rows[1:5]} == {
        (c, seed) for c in NONZERO for seed in READERS
    }
    assert all(r["coeff"] is None and r["seed"] is None for r in rows[5:])
    assert all(np.isfinite(r["perplexity"]) for r in rows[:5])
    assert all(r["perplexity"] is None for r in rows[5:])
    for row in rows:
        assert np.isfinite(row["score"])
        assert np.isfinite(row["score_check"])
        assert np.isfinite(row["mean_length"])


def test_the_table_scores_are_the_trial_means(finished: Finished) -> None:
    rows = _table(finished)
    trials = {t.name: t for t in read_trials(finished.run_dir, "trials")}
    named = [
        (trials["none"], rows[0]),
        *((trials[f"linear/{r['seed']}/{r['coeff']:g}"], r) for r in rows[1:5]),
        *((trials[r["arm"]], r) for r in rows[5:]),
    ]
    for trial, row in named:
        assert row["score"] == pytest.approx(trial.mean_score()[0])
        assert row["score_check"] == pytest.approx(trial.mean_score()[1])
        assert row["empty_rate"] == trial.empty_rate


def test_the_none_row_perplexity_is_the_plain_model_on_the_neutral_slice(
    finished: Finished,
) -> None:
    held = split_held_out(_texts(), n_calibration=10, n_neutral=8, seed=0)
    plain = plain_arm(finished.loaded, GenSettings.from_config(finished.config))
    assert plain.perplexity is not None
    assert _table(finished)[0]["perplexity"] == pytest.approx(
        plain.perplexity(held.neutral)
    )


def test_the_prompting_arms_prepend_their_instruction_on_the_report_prompts(
    finished: Finished,
) -> None:
    config = finished.config
    trials = {t.name: t for t in read_trials(finished.run_dir, "trials")}
    settings = GenSettings.from_config(config)
    prefixes = {
        "prompting+1": config.prompt_positive[0],
        "prompting+2": config.prompt_positive[1],
        "prompting-1": config.prompt_negative[0],
        "prompting-2": config.prompt_negative[1],
    }
    for name, prefix in prefixes.items():
        expected = run_trial(
            plain_arm(finished.loaded, settings, prefix=prefix),
            trials[name].prompts,
            _score,
            columns=2,
        )
        assert expected.completions == trials[name].completions


def test_the_summary_reports_the_layers_the_window_and_the_agreement(
    finished: Finished,
) -> None:
    summary = finished.run_dir.read_json("summary")
    assert isinstance(summary, dict)
    assert set(summary) == {
        "reading_layer",
        "steer_layer",
        "window",
        "sweep_layers",
        "judge_agreement",
        "n_prompts",
    }
    n_layers = finished.loaded.n_layers
    coeffs = (-0.2, 0.0, 0.2)
    primary = finished.run_dir.read_arrays("curves_layers")["per_seed"]
    check = finished.run_dir.read_arrays("curves_layers_check")["per_seed"]
    assert summary["reading_layer"] in range(1, n_layers + 1)
    assert summary["sweep_layers"] == list(SWEEP)
    assert summary["steer_layer"] == slope_layer(primary, coeffs, SWEEP)
    assert summary["window"] == list(
        window_around(summary["steer_layer"], 2, n_layers)
    )
    expected = judge_agreement(primary, check, coeffs)
    assert summary["judge_agreement"] == pytest.approx(expected, nan_ok=True)
    assert summary["n_prompts"] == 8


def _spy_reading_sweep(
    monkeypatch: pytest.MonkeyPatch, planted: NDArray[np.float64] | None = None
) -> list[Any]:
    results: list[Any] = []
    real = c1_sentiment_steering.reading_sweep

    def spy(*args: Any, **kwargs: Any) -> Any:
        result = real(*args, **kwargs)
        results.append(result)
        if planted is None:
            return result
        accuracy = np.full(result.accuracy.shape, 0.5)
        accuracy[0, 0] = planted
        return dataclasses.replace(result, accuracy=accuracy)

    monkeypatch.setattr(c1_sentiment_steering, "reading_sweep", spy)
    return results


def _spy_reader_arm(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    real = c1_sentiment_steering.reader_arm

    def spy(
        loaded: LoadedModel, settings: Any, reader: Any, **kwargs: Any
    ) -> Any:
        calls.append({"reader": reader, **kwargs})
        return real(loaded, settings, reader, **kwargs)

    monkeypatch.setattr(c1_sentiment_steering, "reader_arm", spy)
    return calls


def test_the_arms_use_the_sweep_layers_the_window_and_the_first_readers(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _spy_reader_arm(monkeypatch)
    results = _spy_reading_sweep(monkeypatch)
    config = _config()
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _sst2(), _texts(), _score)
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)

    single = [c for c in calls if len(c["layers"]) == 1]
    headline = [c for c in calls if len(c["layers"]) > 1]
    assert {tuple(c["layers"]) for c in single} == {(1,), (2,)}
    assert {tuple(c["layers"]) for c in headline} == {tuple(summary["window"])}
    assert {c.get("op", "linear") for c in single} == {"linear"}
    assert {c["op"] for c in headline} == set(OPS)
    steering = {id(r) for r in results[0].readers[:2]}
    assert {id(c["reader"]) for c in calls} == steering
    assert all(c["coeff"] != 0.0 for c in calls)
    assert {c["mode"] for c in calls} == {"relative"}
    assert {c["mask"] for c in calls} == {"response"}
    assert {c["preserve_norm"] for c in calls} == {False}
    held = split_held_out(_texts(), n_calibration=10, n_neutral=8, seed=0)
    norms = mean_residual_norms(
        tiny_arch, held.calibration, [1, 2, 3], batch_size=8
    )
    assert all(c["norms"] == norms for c in calls)


def test_the_reading_layer_is_the_best_mean_validation_layer_and_does_not_steer(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planted = np.array(
        [[0.9, 0.1, 0.1], [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]], dtype=np.float64
    )
    _spy_reading_sweep(monkeypatch, planted)
    monkeypatch.setattr(
        c1_sentiment_steering, "slope_layer", lambda *args, **kwargs: 1
    )
    config = _config()
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _sst2(), _texts(), _score)
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    assert summary["reading_layer"] == 3
    assert summary["steer_layer"] == 1
    assert summary["window"] == [1, 2]


def test_the_side_effect_rows_follow_the_first_operator_and_a_single_reader(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config(ops=("piecewise", "linear"), n_steer_seeds=1)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _sst2(), _texts(), _score)
    table = run_dir.read_json("table")
    assert isinstance(table, dict)
    arms = [row[0] for row in table["rows"]]
    assert arms == ["none", "piecewise", "piecewise", *PROMPTING]
    per_seed = run_dir.read_arrays("curves_operators")["per_seed"]
    assert per_seed.shape == (2, 1, 3)
    sidecar = run_dir.read_json("curves_operators")
    assert isinstance(sidecar, dict)
    assert sidecar["series"] == ["piecewise", "linear"]


def test_the_steering_mode_mask_and_norm_follow_the_config(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _spy_reader_arm(monkeypatch)
    config = _config(mode="absolute", mask="all", preserve_norm=True)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _sst2(), _texts(), _score)
    assert {c["mode"] for c in calls} == {"absolute"}
    assert {c["mask"] for c in calls} == {"all"}
    assert {c["preserve_norm"] for c in calls} == {True}
    sidecar = run_dir.read_json("curves_layers")
    assert isinstance(sidecar, dict)
    assert sidecar["xlabel"] == "Coefficient"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"coeffs": (-0.2, 0.2)}, "coeffs"),
        ({"coeffs": (0.0,)}, "coeffs"),
        ({"n_steer_seeds": 4}, "n_steer_seeds"),
        ({"n_steer_seeds": 0}, "n_steer_seeds"),
        ({"sweep_layers": ()}, "sweep_layers"),
        ({"sweep_layers": (0, 2)}, "sweep_layers"),
        ({"sweep_layers": (1, 4)}, "sweep_layers"),
        ({"sweep_layers": (1, 1)}, "sweep_layers"),
        ({"sweep_layers": (1,) * 9}, "sweep_layers"),
        ({"window": 0}, "window"),
        ({"window": 4}, "window"),
        ({"ops": ()}, "ops"),
        ({"ops": ("linear", "linear")}, "ops"),
        ({"n_prompts": 100}, "pool"),
    ],
    ids=[
        "no-zero",
        "only-zero",
        "steer-seeds-above-seeds",
        "steer-seeds-zero",
        "no-sweep-layer",
        "sweep-layer-zero",
        "sweep-layer-too-deep",
        "sweep-layer-repeated",
        "nine-sweep-layers",
        "window-zero",
        "window-too-wide",
        "no-operator",
        "operator-repeated",
        "prompt-pool-too-small",
    ],
)
def test_an_invalid_config_raises_before_the_model_runs(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
    message: str,
) -> None:
    def reached(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the model was reached")

    monkeypatch.setattr(c1_sentiment_steering, "reading_sweep", reached)
    config = _config(**overrides)
    run_dir = RunDir.create(config, out_root=tmp_path)
    with pytest.raises(ValueError, match=message):
        execute(config, run_dir, tiny_arch, _sst2(), _texts(), _score)


class StubJudge:
    def __init__(self, column: int) -> None:
        self.column = column

    def score(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.ascontiguousarray(_score(texts)[:, self.column])


def test_run_scores_with_both_judges_in_two_columns(
    finished: Finished, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: dict[str, list[str]] = {"model": [], "judge": []}

    def load_model(key: str) -> LoadedModel:
        asked["model"].append(key)
        return finished.loaded

    def load_judge(key: str) -> StubJudge:
        asked["judge"].append(key)
        return StubJudge(0 if key == "sentiment-primary" else 1)

    monkeypatch.setattr(c1_sentiment_steering, "load_model", load_model)
    monkeypatch.setattr(c1_sentiment_steering, "load_judge", load_judge)
    monkeypatch.setattr(c1_sentiment_steering, "load_sst2", _sst2)
    monkeypatch.setattr(
        c1_sentiment_steering,
        "load_wikitext",
        lambda: [SimpleNamespace(text=t) for t in _texts()],
    )
    run_dir = RunDir.create(finished.config, out_root=tmp_path)
    run(finished.config, run_dir)

    assert asked == {
        "model": ["tiny"],
        "judge": ["sentiment-primary", "sentiment-secondary"],
    }
    for stem in ("trials", "trials_sweep"):
        np.testing.assert_array_equal(
            run_dir.read_arrays(stem)["scores"],
            finished.run_dir.read_arrays(stem)["scores"],
        )


def test_plot_draws_the_four_curve_files(
    finished: Finished, tmp_path: Path
) -> None:
    paths = plot(finished.run_dir, tmp_path / "figs")
    stems = (
        "curves_layers",
        "curves_layers_check",
        "curves_operators",
        "curves_perplexity",
    )
    assert [p.name for p in paths] == [
        f"{stem}.{ext}" for stem in stems for ext in ("pdf", "png", "svg")
    ]
    assert all(p.stat().st_size > 0 for p in paths)


def _print_report(run_dir: RunDir) -> None:
    summary = run_dir.read_json("summary")
    print(summary)
    for stem in ("curves_layers", "curves_layers_check"):
        arrays = run_dir.read_arrays(stem)
        sidecar = run_dir.read_json(stem)
        assert isinstance(sidecar, dict)
        slopes = [
            float(np.mean([dose_slope(arrays["x"], seed) for seed in row]))
            for row in arrays["per_seed"]
        ]
        print(
            stem, dict(zip(sidecar["series"], np.round(slopes, 3), strict=True))
        )
    table = run_dir.read_json("table")
    assert isinstance(table, dict)
    print(table["columns"])
    for row in table["rows"]:
        print(row)


def test_the_report_prints_the_slopes_and_the_side_effect_rows(
    finished: Finished, capsys: pytest.CaptureFixture[str]
) -> None:
    _print_report(finished.run_dir)
    printed = capsys.readouterr().out
    assert "curves_layers_check" in printed
    assert "prompting+1" in printed


def _cached[T](load: Callable[[], T]) -> T:
    try:
        return load()
    except FileNotFoundError as error:
        pytest.skip(f"cache missing: {error}")


def _cached_judge(key: str) -> Any:
    spec = JUDGES[key]
    try:
        snapshot_download(
            spec.repo_id, revision=spec.revision, local_files_only=True
        )
    except FileNotFoundError as error:
        pytest.skip(f"cache missing: {error}")
    return load_judge(key)


@pytest.mark.slow
def test_c1_on_gpt2_prints_the_slopes_under_both_judges(tmp_path: Path) -> None:
    sst2 = _cached(lambda: load_sst2(offline=True))
    texts = [e.text for e in _cached(lambda: load_wikitext(offline=True))]
    first = _cached_judge("sentiment-primary")
    second = _cached_judge("sentiment-secondary")

    def score(batch: Sequence[str]) -> NDArray[np.float32]:
        return np.column_stack([first.score(batch), second.score(batch)])

    config = Config(
        n_seeds=3,
        n_steer_seeds=1,
        n_pairs=32,
        n_eval_pairs=128,
        sweep_layers=(4, 8),
        coeffs=(-0.2, 0.0, 0.2),
        n_prompts=16,
        num_samples=1,
    )
    run_dir = RunDir.create(config, out_root=tmp_path)
    started = time.perf_counter()
    execute(config, run_dir, load_model("gpt2"), sst2, texts, score)
    seconds = time.perf_counter() - started

    _print_report(run_dir)
    tokens = sum(
        int(trial.lengths.sum())
        for stem in ("trials_sweep", "trials")
        for trial in read_trials(run_dir, stem)
    )
    print(f"{seconds / tokens:.4f} s per generated token, whole run")
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    assert summary["steer_layer"] in (4, 8)
