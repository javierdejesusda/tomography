"""Tests for the P8 instruction-tuning experiment."""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.cli import load_experiment
from tomography.experiments import p6_scale
from tomography.experiments import p8_tuning
from tomography.experiments.aggregate import read_curves
from tomography.experiments.base import ExperimentConfig
from tomography.experiments.base import RunDir
from tomography.experiments.effects import write_dose_curves
from tomography.experiments.p8_tuning import Config
from tomography.experiments.p8_tuning import execute
from tomography.experiments.p8_tuning import gap_interval
from tomography.experiments.p8_tuning import paired_difference
from tomography.experiments.p8_tuning import plot
from tomography.experiments.p8_tuning import run
from tomography.experiments.p8_tuning import template_scores
from tomography.experiments.sweep import SERIES
from tomography.experiments.sweep import write_curves
from tomography.model import MODELS

OFFSETS = np.array([-0.02, 0.02, -0.01, 0.01])


@dataclasses.dataclass(frozen=True, kw_only=True)
class _RunConfig(ExperimentConfig):
    """Test-only config of a finished source run."""


def _new_run(out_root: Path, experiment: str, model: str) -> RunDir:
    return RunDir.create(
        _RunConfig(experiment=experiment, model=model), out_root=out_root
    )


def _array(*values: float) -> NDArray[np.float64]:
    return np.asarray(values, dtype=np.float64)


def test_the_config_defaults() -> None:
    config = Config()

    assert config.experiment == "p8"
    assert config.model == "qwen3-1.7b"
    assert config.base_models == ("qwen3-1.7b-base", "qwen3-0.6b-base")
    assert config.tuned_models == ("qwen3-1.7b", "qwen3-0.6b")
    assert config.reference == "gpt2"
    assert config.concept_sources == ("r1", "r2", "r3", "r7")
    assert config.function_sources == ("r5",)
    assert config.template_source == "p1"
    assert config.template_task == "sentiment"
    assert config.steering_source == "c1"
    assert config.extra_runs == ()
    assert config.n_boot == 2000


def test_the_default_models_are_registered_checkpoints() -> None:
    config = Config()

    models = [*config.base_models, *config.tuned_models, config.reference]
    assert all(model in MODELS for model in models)
    assert config.model in config.tuned_models


def test_the_config_survives_a_round_trip_through_a_dict() -> None:
    config = Config(reference=None, template_source=None, n_boot=7)

    assert Config.from_dict(config.to_dict()) == config


def test_paired_difference_is_tuned_minus_base_per_seed() -> None:
    diff = paired_difference(_array(0.5, 0.6, 0.7), _array(0.6, 0.6, 0.9))

    np.testing.assert_allclose(diff, [0.1, 0.0, 0.2])


@pytest.mark.parametrize(
    ("base", "tuned"),
    [
        (_array(0.1, 0.2, 0.3, 0.4, 0.5), _array(0.2, 0.4, 0.6)),
        (_array(0.2, 0.4, 0.6), _array(0.1, 0.2, 0.3, 0.4, 0.5)),
    ],
    ids=["more-base-seeds", "more-tuned-seeds"],
)
def test_paired_difference_keeps_the_first_seeds_of_the_longer_run(
    base: NDArray[np.float64], tuned: NDArray[np.float64]
) -> None:
    diff = paired_difference(base, tuned)

    n = min(base.size, tuned.size)
    assert diff.shape == (n,)
    np.testing.assert_allclose(diff, tuned[:n] - base[:n])


@pytest.mark.parametrize(
    ("base", "tuned"),
    [
        (_array(), _array(0.1)),
        (_array(0.1), _array()),
        (np.zeros((2, 2)), np.zeros(2)),
    ],
    ids=["empty-base", "empty-tuned", "not-1d"],
)
def test_paired_difference_needs_one_seed_axis_with_a_seed(
    base: NDArray[np.float64], tuned: NDArray[np.float64]
) -> None:
    with pytest.raises(ValueError, match="seed"):
        paired_difference(base, tuned)


def _diffs() -> dict[str, NDArray[np.float64]]:
    return {
        "f": _array(0.1, 0.3, 0.2, 0.2),
        "c1": _array(0.0, 0.0, 0.0, 0.0),
        "c2": _array(0.1, 0.1, 0.1, 0.1),
    }


def test_the_gap_point_is_the_function_mean_minus_the_concept_mean() -> None:
    point, lo, hi = gap_interval(
        _diffs(), ["f"], ["c1", "c2"], n_boot=200, seed=0
    )

    assert point == pytest.approx(0.2 - 0.05)
    assert lo <= point <= hi
    assert lo < hi


def test_the_gap_averages_the_sources_of_a_kind_not_their_seeds() -> None:
    diffs = {
        "f1": _array(0.4, 0.4, 0.4, 0.4),
        "f2": _array(0.0, 0.0),
        "c": _array(0.1, 0.1, 0.1),
    }

    point, lo, hi = gap_interval(diffs, ["f1", "f2"], ["c"], n_boot=50, seed=0)

    assert point == pytest.approx(0.2 - 0.1)
    assert (lo, hi) == (pytest.approx(point), pytest.approx(point))


def test_a_source_outside_both_lists_does_not_move_the_gap() -> None:
    with_extra = {**_diffs(), "other": _array(5.0, 5.0, 5.0, 5.0)}

    plain = gap_interval(_diffs(), ["f"], ["c1", "c2"], n_boot=100, seed=3)
    extra = gap_interval(with_extra, ["f"], ["c1", "c2"], n_boot=100, seed=3)

    assert extra == plain


def test_the_gap_interval_is_reproducible_by_seed() -> None:
    first = gap_interval(_diffs(), ["f"], ["c2"], n_boot=100, seed=1)
    again = gap_interval(_diffs(), ["f"], ["c2"], n_boot=100, seed=1)
    other = gap_interval(_diffs(), ["f"], ["c2"], n_boot=100, seed=2)

    assert first == again
    assert first[0] == other[0]
    assert first[1:] != other[1:]


def test_the_sources_of_one_kind_are_resampled_independently() -> None:
    seeds = _array(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    zero = np.zeros(8)

    _, lo_one, hi_one = gap_interval(
        {"a": seeds, "c": zero}, ["a"], ["c"], n_boot=2000, seed=0
    )
    _, lo_two, hi_two = gap_interval(
        {"a": seeds, "b": seeds, "c": zero},
        ["a", "b"],
        ["c"],
        n_boot=2000,
        seed=0,
    )

    assert (hi_two - lo_two) < 0.9 * (hi_one - lo_one)


@pytest.mark.parametrize(
    ("functions", "concepts", "message"),
    [
        ([], ["c1"], "function"),
        (["f"], [], "concept"),
        (["f", "missing"], ["c1"], "missing"),
        (["f"], ["c1", "missing"], "missing"),
    ],
    ids=["no-function", "no-concept", "unknown-function", "unknown-concept"],
)
def test_the_gap_needs_both_kinds_and_known_sources(
    functions: list[str], concepts: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        gap_interval(_diffs(), functions, concepts, n_boot=10, seed=0)


def _write_variants(
    run_dir: RunDir, stem: str, curves: Mapping[str, Sequence[float]]
) -> None:
    """Writes one series per template variant, the seeds shifted by OFFSETS."""
    per_seed = np.stack(
        [
            np.asarray(curve)[None, :] + OFFSETS[:, None]
            for curve in curves.values()
        ]
    )
    write_curves(
        run_dir,
        per_seed,
        list(range(1, per_seed.shape[2] + 1)),
        series=list(curves),
        stem=stem,
        chance=0.5,
        n_layers=per_seed.shape[2],
    )


def _p1_run(
    out_root: Path,
    model: str,
    test: Mapping[str, Sequence[float]],
    *,
    val: Mapping[str, Sequence[float]] | None = None,
    task: str = "sentiment",
) -> RunDir:
    """A finished P1-like run of one task, one series per variant."""
    run_dir = _new_run(out_root, "p1", model)
    _write_variants(run_dir, f"curves_{task}", test)
    if val is not None:
        _write_variants(run_dir, f"curves_{task}_val", val)
    run_dir.finalize()
    return run_dir


def test_template_scores_gives_one_seed_vector_per_variant(
    tmp_path: Path,
) -> None:
    run = _p1_run(
        tmp_path,
        "m",
        {"plain": [0.5, 0.7, 0.5], "tagged": [0.5, 0.5, 0.6]},
        val={"plain": [0.5, 0.9, 0.5], "tagged": [0.5, 0.5, 0.9]},
    )

    scores = template_scores(run)

    assert list(scores) == ["plain", "tagged"]
    np.testing.assert_allclose(scores["plain"], 0.7 + OFFSETS)
    np.testing.assert_allclose(scores["tagged"], 0.6 + OFFSETS)


def test_each_variant_takes_the_layer_its_validation_curve_picks(
    tmp_path: Path,
) -> None:
    run = _p1_run(
        tmp_path,
        "m",
        {"a": [0.6, 0.5, 0.9], "b": [0.5, 0.8, 0.9]},
        val={"a": [0.9, 0.5, 0.5], "b": [0.5, 0.9, 0.5]},
    )

    scores = template_scores(run)

    np.testing.assert_allclose(scores["a"], 0.6 + OFFSETS)
    np.testing.assert_allclose(scores["b"], 0.8 + OFFSETS)


def test_without_validation_curves_the_report_curves_choose_the_layer(
    tmp_path: Path,
) -> None:
    run = _p1_run(tmp_path, "m", {"a": [0.6, 0.9, 0.5], "b": [0.5, 0.5, 0.7]})

    scores = template_scores(run)

    np.testing.assert_allclose(scores["a"], 0.9 + OFFSETS)
    np.testing.assert_allclose(scores["b"], 0.7 + OFFSETS)


def test_the_task_picks_the_pair_of_curves_that_is_read(tmp_path: Path) -> None:
    run_dir = _new_run(tmp_path, "p1", "m")
    _write_variants(
        run_dir, "curves_sentiment", {"a": [0.9, 0.5], "b": [0.8, 0.5]}
    )
    _write_variants(
        run_dir, "curves_morality", {"a": [0.5, 0.6], "b": [0.5, 0.7]}
    )
    _write_variants(
        run_dir, "curves_morality_val", {"a": [0.5, 0.9], "b": [0.9, 0.5]}
    )
    run_dir.finalize()

    default = template_scores(run_dir)
    morality = template_scores(run_dir, "morality")

    np.testing.assert_allclose(default["a"], 0.9 + OFFSETS)
    np.testing.assert_allclose(morality["a"], 0.6 + OFFSETS)
    np.testing.assert_allclose(morality["b"], 0.5 + OFFSETS)


def test_a_run_without_the_task_curves_is_rejected(tmp_path: Path) -> None:
    run = _p1_run(tmp_path, "m", {"a": [0.6, 0.7], "b": [0.5, 0.6]})

    with pytest.raises(ValueError, match="curves_morality"):
        template_scores(run, "morality")


def test_one_variant_is_not_enough_to_compare_templates(tmp_path: Path) -> None:
    run = _p1_run(tmp_path, "m", {"only": [0.6, 0.7]})

    with pytest.raises(ValueError, match="two variants"):
        template_scores(run)


BASE_MODELS = ("b3", "b6")
TUNED_MODELS = ("t3", "t6")
REFERENCE = "ref"
CONCEPTS = ("r1", "r2")
FUNCTIONS = ("r5",)
MODELS_ORDER = ("b3", "t3", "b6", "t6", "ref")
DEPTH = {"b3": 3, "t3": 3, "b6": 6, "t6": 6, "ref": 4}
LAYER = {"b3": 2, "t3": 2, "b6": 3, "t6": 4, "ref": 2}
BASE_PEAK = {"b3": 0.60, "b6": 0.70, "ref": 0.58}
SOURCE_OFFSET = {"r1": 0.0, "r2": 0.05, "r5": 0.0}
SHIFT = {
    "t3": {"r1": 0.05, "r2": -0.05, "r5": 0.20},
    "t6": {"r1": 0.0, "r2": 0.0, "r5": 0.10},
}
SEED_DELTA = np.array([-0.01, 0.01, 0.0, 0.0])
PAIR_OF = {"t3": "b3", "t6": "b6"}


def _per_seed(
    reader: Sequence[float], offsets: NDArray[np.float64] = OFFSETS
) -> NDArray[np.float64]:
    """Accuracy of shape (3, N, K): the reader shifted per seed, two nulls."""
    rows = np.asarray(reader)[None, :] + offsets[:, None]
    null = np.full_like(rows, 0.5)
    return np.stack([rows, null, null])


def _reading_run(
    out_root: Path,
    experiment: str,
    model: str,
    reader: Sequence[float],
    *,
    val: Sequence[float] | None = None,
    offsets: NDArray[np.float64] = OFFSETS,
) -> RunDir:
    """A finished reading run with `curves` and, if given, `curves_val`."""
    run_dir = _new_run(out_root, experiment, model)
    layers = list(range(1, len(reader) + 1))
    for stem, values in (("curves", reader), ("curves_val", val)):
        if values is not None:
            write_curves(
                run_dir,
                _per_seed(values, offsets),
                layers,
                series=SERIES,
                stem=stem,
                chance=0.5,
                n_layers=len(reader),
            )
    run_dir.finalize()
    return run_dir


def _peaked(n_layers: int, layer: int, value: float) -> list[float]:
    """A flat curve with one point at `layer` raised to `value`."""
    curve = [0.5] * n_layers
    curve[layer - 1] = value
    return curve


def _peak(source: str, model: str) -> float:
    base = BASE_PEAK[PAIR_OF.get(model, model)] + SOURCE_OFFSET[source]
    return base + SHIFT.get(model, {}).get(source, 0.0)


def _reading_runs(out_root: Path) -> dict[tuple[str, str], RunDir]:
    """Finished R-like runs of three sources on the five models.

    R1-like runs have no validation curves, R2-like runs have them on the base
    models only and R5-like runs have them on every model.
    """
    runs: dict[tuple[str, str], RunDir] = {}
    for model in MODELS_ORDER:
        tuned = model in PAIR_OF
        offsets = OFFSETS + SEED_DELTA if tuned else OFFSETS
        for source in (*CONCEPTS, *FUNCTIONS):
            curve = _peaked(DEPTH[model], LAYER[model], _peak(source, model))
            has_val = source == "r5" or (source == "r2" and not tuned)
            val = _peaked(DEPTH[model], LAYER[model], 0.9) if has_val else None
            runs[(source, model)] = _reading_run(
                out_root, source, model, curve, val=val, offsets=offsets
            )
    return runs


def _config(**overrides: Any) -> Config:
    base = Config(
        model="t3",
        base_models=BASE_MODELS,
        tuned_models=TUNED_MODELS,
        reference=REFERENCE,
        concept_sources=CONCEPTS,
        function_sources=FUNCTIONS,
        template_source=None,
        steering_source=None,
        n_boot=200,
    )
    return dataclasses.replace(base, **overrides)


def _execute(tmp_path: Path, **overrides: Any) -> RunDir:
    config = _config(**overrides)
    runs = _reading_runs(tmp_path / "in")
    run_dir = RunDir.create(config, out_root=tmp_path / "out")
    execute(config, run_dir, runs)
    return run_dir


def _rows(run_dir: RunDir, stem: str = "table") -> list[dict[str, Any]]:
    payload = run_dir.read_json(stem)
    assert isinstance(payload, dict)
    return [
        dict(zip(payload["columns"], row, strict=True))
        for row in payload["rows"]
    ]


def _summary(run_dir: RunDir) -> dict[str, Any]:
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    return summary


def test_execute_copies_the_reader_curves_of_every_source_and_model(
    tmp_path: Path,
) -> None:
    run_dir = _execute(tmp_path)

    for source in (*CONCEPTS, *FUNCTIONS):
        for model in MODELS_ORDER:
            copied = read_curves(run_dir, f"curves_{source}_{model}")
            assert copied.series == (model,)
            assert copied.n_layers == DEPTH[model]


def test_the_table_has_one_row_per_pair_and_source(tmp_path: Path) -> None:
    run_dir = _execute(tmp_path)

    payload = run_dir.read_json("table")
    assert isinstance(payload, dict)
    assert payload["columns"] == [
        "source",
        "kind",
        "base",
        "tuned",
        "base_accuracy",
        "tuned_accuracy",
        "delta",
        "lo",
        "hi",
        "base_depth",
        "tuned_depth",
        "selected_on",
    ]
    rows = _rows(run_dir)
    assert [(r["base"], r["tuned"], r["source"]) for r in rows] == [
        (base, tuned, source)
        for base, tuned in (("b3", "t3"), ("b6", "t6"))
        for source in ("r1", "r2", "r5")
    ]
    assert [r["kind"] for r in rows[:3]] == ["concept", "concept", "function"]


def test_the_table_reports_the_planted_accuracies_deltas_and_depths(
    tmp_path: Path,
) -> None:
    run_dir = _execute(tmp_path)

    for row in _rows(run_dir):
        source, base, tuned = row["source"], row["base"], row["tuned"]
        assert row["base_accuracy"] == pytest.approx(_peak(source, base))
        assert row["tuned_accuracy"] == pytest.approx(_peak(source, tuned))
        assert row["delta"] == pytest.approx(SHIFT[tuned][source])
        assert row["lo"] <= row["delta"] <= row["hi"]
        assert row["hi"] - row["lo"] <= 0.02 + 1e-9
        assert row["base_depth"] == pytest.approx(LAYER[base] / DEPTH[base])
        assert row["tuned_depth"] == pytest.approx(LAYER[tuned] / DEPTH[tuned])


def test_selected_on_is_the_shared_curves_or_both_joined(
    tmp_path: Path,
) -> None:
    run_dir = _execute(tmp_path)

    chosen = {r["source"]: r["selected_on"] for r in _rows(run_dir)}
    assert chosen == {
        "r1": "curves",
        "r2": "curves_val/curves",
        "r5": "curves_val",
    }


def test_the_delta_pairs_the_first_seeds_of_the_two_runs(
    tmp_path: Path,
) -> None:
    config = _config()
    runs = _reading_runs(tmp_path / "in")
    runs[("r1", "t3")] = _reading_run(
        tmp_path / "replaced",
        "r1",
        "t3",
        _peaked(3, 2, 0.6),
        offsets=np.array([0.0, 0.1, 0.2]),
    )
    run_dir = RunDir.create(config, out_root=tmp_path / "out")

    execute(config, run_dir, runs)

    row = next(
        r for r in _rows(run_dir) if (r["source"], r["tuned"]) == ("r1", "t3")
    )
    assert row["delta"] == pytest.approx(np.mean([0.02, 0.08, 0.21]))


def test_the_reference_has_its_own_table(tmp_path: Path) -> None:
    run_dir = _execute(tmp_path)

    payload = run_dir.read_json("table_reference")
    assert isinstance(payload, dict)
    assert payload["columns"] == [
        "source",
        "model",
        "accuracy",
        "lo",
        "hi",
        "depth",
    ]
    rows = _rows(run_dir, "table_reference")
    assert [(r["source"], r["model"]) for r in rows] == [
        (source, "ref") for source in ("r1", "r2", "r5")
    ]
    assert [r["accuracy"] for r in rows] == pytest.approx(
        [_peak(s, "ref") for s in ("r1", "r2", "r5")]
    )
    assert all(r["lo"] <= r["accuracy"] <= r["hi"] for r in rows)
    assert [r["depth"] for r in rows] == pytest.approx([0.5] * 3)


def test_without_a_reference_there_is_no_reference_table(
    tmp_path: Path,
) -> None:
    run_dir = _execute(tmp_path, reference=None)

    assert not (run_dir.path / "table_reference.json").exists()
    assert not (run_dir.path / "curves_r1_ref.npz").exists()
    assert _summary(run_dir)["models"] == ["b3", "t3", "b6", "t6"]


def test_the_function_gap_is_the_planted_gain_of_the_function_source(
    tmp_path: Path,
) -> None:
    run_dir = _execute(tmp_path)

    gap = _summary(run_dir)["function_gap"]
    assert set(gap) == {"t3", "t6"}
    assert gap["t3"]["point"] == pytest.approx(0.20)
    assert gap["t6"]["point"] == pytest.approx(0.10)
    for tuned in ("t3", "t6"):
        entry = gap[tuned]
        assert entry["lo"] <= entry["point"] <= entry["hi"]
        assert entry["hi"] - entry["lo"] < 0.04


def test_the_summary_names_the_pairs_sources_and_models(
    tmp_path: Path,
) -> None:
    run_dir = _execute(tmp_path)

    summary = _summary(run_dir)
    assert set(summary) == {"pairs", "sources", "models", "function_gap"}
    assert summary["pairs"] == [["b3", "t3"], ["b6", "t6"]]
    assert summary["sources"] == {
        "concept": ["r1", "r2"],
        "function": ["r5"],
    }
    assert summary["models"] == ["b3", "t3", "b6", "t6", "ref"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model": "ref"}, "must be one of tuned_models"),
        ({"base_models": ("b3",)}, "same length"),
        ({"base_models": (), "tuned_models": ()}, "at least one pair"),
        ({"concept_sources": ()}, "concept_sources"),
        ({"function_sources": ()}, "function_sources"),
        (
            {
                "concept_sources": ("r1", "r2", "r3", "r4", "r6", "r7"),
                "function_sources": ("r5",),
            },
            "at most 6",
        ),
        ({"function_sources": ("r1",)}, "only once"),
        ({"base_models": ("b3", "b3")}, "only once"),
        ({"reference": "b3"}, "only once"),
        (
            {
                "base_models": tuple(f"b{i}" for i in range(4)),
                "tuned_models": tuple(f"t{i}" for i in range(4)),
                "model": "t0",
            },
            "at most 8",
        ),
    ],
    ids=[
        "model-not-tuned",
        "unpaired",
        "no-pair",
        "no-concept",
        "no-function",
        "seven-sources",
        "source-of-both-kinds",
        "repeated-base",
        "reference-is-a-base",
        "nine-models",
    ],
)
def test_execute_rejects_a_bad_config_before_reading_anything(
    tmp_path: Path, overrides: dict[str, Any], message: str
) -> None:
    config = _config(**overrides)
    run_dir = RunDir.create(config, out_root=tmp_path)
    runs: Mapping[tuple[str, str], RunDir] = {}

    with pytest.raises(ValueError, match=message):
        execute(config, run_dir, runs)

    assert not list(run_dir.path.glob("*.npz"))


def test_execute_names_a_missing_run(tmp_path: Path) -> None:
    config = _config()
    runs = _reading_runs(tmp_path / "in")
    del runs[("r5", "t6")]
    run_dir = RunDir.create(config, out_root=tmp_path / "out")

    with pytest.raises(ValueError, match=r"runs.*r5.*t6"):
        execute(config, run_dir, runs)

    assert not list(run_dir.path.glob("*.npz"))


LEVELS = {
    "b3": (0.5, 0.7, 0.9),
    "t3": (0.68, 0.70, 0.72),
    "b6": (0.55, 0.70, 0.85),
    "t6": (0.60, 0.70, 0.80),
    "ref": (0.6, 0.7, 0.8),
}
VARIANTS = ("v1", "v2", "v3")
SPREAD_SCALE = np.array([0.8, 1.2, 1.0, 1.0])


def _spread_run(
    out_root: Path,
    model: str,
    variants: Mapping[str, float],
    *,
    task: str,
    scale: NDArray[np.float64] = SPREAD_SCALE,
) -> RunDir:
    """A finished P1-like run: one flat curve with a peak per variant.

    The variants sit around their mean by a per-seed multiple, so every seed
    has its own spread over the variants. With the default `scale` the mean
    multiple is one.
    """
    run_dir = _new_run(out_root, "p1", model)
    levels = np.array(list(variants.values()))
    per_seed = np.stack(
        [
            [
                _peaked_floor(
                    4,
                    2,
                    levels.mean() + (level - levels.mean()) * multiple,
                )
                for multiple in scale
            ]
            for level in levels
        ]
    )
    write_curves(
        run_dir,
        per_seed,
        [1, 2, 3, 4],
        series=list(variants),
        stem=f"curves_{task}",
        chance=0.5,
        n_layers=4,
    )
    run_dir.finalize()
    return run_dir


def _peaked_floor(n_layers: int, layer: int, value: float) -> list[float]:
    curve = [0.4] * n_layers
    curve[layer - 1] = value
    return curve


def _template_runs(
    out_root: Path,
    *,
    task: str = "sentiment",
    first_order: Sequence[str] = VARIANTS,
    scales: Mapping[str, NDArray[np.float64]] | None = None,
) -> dict[tuple[str, str], RunDir]:
    """P1-like runs on the five models.

    Base models and the reference also have `no_trailing_space`, which exists
    only under the plain format. The first model lists the variants in
    `first_order`. `scales` gives the per-seed multiple of a model, by key,
    for the models that do not take the default.
    """
    runs: dict[tuple[str, str], RunDir] = {}
    for model in MODELS_ORDER:
        names = list(VARIANTS)
        if model == MODELS_ORDER[0]:
            names = list(first_order)
        levels = dict(zip(VARIANTS, LEVELS[model], strict=True))
        variants = {name: levels[name] for name in names}
        if model not in PAIR_OF:
            variants["no_trailing_space"] = 0.6
        runs[("p1", model)] = _spread_run(
            out_root,
            model,
            variants,
            task=task,
            scale=(scales or {}).get(model, SPREAD_SCALE),
        )
    return runs


def _execute_templates(
    tmp_path: Path,
    *,
    task: str = "sentiment",
    first_order: Sequence[str] = VARIANTS,
    **overrides: Any,
) -> RunDir:
    config = _config(template_source="p1", template_task=task, **overrides)
    runs = _reading_runs(tmp_path / "in")
    runs.update(
        _template_runs(tmp_path / "in", task=task, first_order=first_order)
    )
    run_dir = RunDir.create(config, out_root=tmp_path / "out")
    execute(config, run_dir, runs)
    return run_dir


def _expected_sd(model: str) -> float:
    return float(np.std(LEVELS[model]))


def test_the_template_matrix_has_the_mean_accuracy_of_every_variant(
    tmp_path: Path,
) -> None:
    run_dir = _execute_templates(tmp_path)

    sidecar = run_dir.read_json("matrix_templates")
    assert isinstance(sidecar, dict)
    assert sidecar["rows"] == ["b3", "t3", "b6", "t6", "ref"]
    assert sidecar["cols"] == ["v1", "v2", "v3"]
    assert "sentiment" in sidecar["title"]
    values = run_dir.read_arrays("matrix_templates")["values"]
    np.testing.assert_allclose(
        values, [LEVELS[m] for m in MODELS_ORDER], atol=1e-9
    )


def test_the_variants_are_those_every_model_has_in_the_first_models_order(
    tmp_path: Path,
) -> None:
    run_dir = _execute_templates(tmp_path, first_order=("v3", "v1", "v2"))

    summary = _summary(run_dir)
    assert summary["template_variants"] == ["v3", "v1", "v2"]
    sidecar = run_dir.read_json("matrix_templates")
    assert isinstance(sidecar, dict)
    assert sidecar["cols"] == ["v3", "v1", "v2"]
    values = run_dir.read_arrays("matrix_templates")["values"]
    np.testing.assert_allclose(values[0], [0.9, 0.5, 0.7], atol=1e-9)


def test_the_dropped_variants_are_listed_per_model(tmp_path: Path) -> None:
    run_dir = _execute_templates(tmp_path)

    assert _summary(run_dir)["template_dropped"] == {
        "b3": ["no_trailing_space"],
        "t3": [],
        "b6": ["no_trailing_space"],
        "t6": [],
        "ref": ["no_trailing_space"],
    }


def test_the_template_spread_is_the_per_seed_deviation_over_variants(
    tmp_path: Path,
) -> None:
    run_dir = _execute_templates(tmp_path)

    spread = _summary(run_dir)["template_sd"]
    assert set(spread) == set(MODELS_ORDER)
    for model in MODELS_ORDER:
        entry = spread[model]
        assert entry["mean"] == pytest.approx(_expected_sd(model))
        assert entry["lo"] <= entry["mean"] <= entry["hi"]
        assert 0.8 * entry["mean"] - 1e-9 <= entry["lo"]
        assert entry["hi"] <= 1.2 * entry["mean"] + 1e-9


def test_tuning_that_narrows_the_spread_gives_a_negative_sd_delta(
    tmp_path: Path,
) -> None:
    run_dir = _execute_templates(tmp_path)

    delta = _summary(run_dir)["template_sd_delta"]
    assert set(delta) == {"t3", "t6"}
    for base, tuned in (("b3", "t3"), ("b6", "t6")):
        entry = delta[tuned]
        expected = _expected_sd(tuned) - _expected_sd(base)
        assert entry["mean"] == pytest.approx(expected)
        assert entry["lo"] <= entry["mean"] <= entry["hi"] < 0.0
        assert entry["lo"] < entry["hi"]


def test_the_sd_delta_pairs_every_seed_with_its_own_seed(
    tmp_path: Path,
) -> None:
    shift = 0.01
    scales = {
        tuned: (SPREAD_SCALE * _expected_sd(base) + shift) / _expected_sd(tuned)
        for tuned, base in PAIR_OF.items()
    }
    config = _config(template_source="p1")
    runs = _reading_runs(tmp_path / "in")
    runs.update(_template_runs(tmp_path / "in", scales=scales))
    run_dir = RunDir.create(config, out_root=tmp_path / "out")

    execute(config, run_dir, runs)

    delta = _summary(run_dir)["template_sd_delta"]
    for tuned in PAIR_OF:
        assert delta[tuned]["mean"] == pytest.approx(shift)
        assert delta[tuned]["hi"] - delta[tuned]["lo"] == pytest.approx(
            0.0, abs=1e-12
        )


def test_the_task_picks_the_template_curves(tmp_path: Path) -> None:
    run_dir = _execute_templates(tmp_path, task="morality")

    sidecar = run_dir.read_json("matrix_templates")
    assert isinstance(sidecar, dict)
    assert "morality" in sidecar["title"]
    assert _summary(run_dir)["template_variants"] == ["v1", "v2", "v3"]


def test_the_summary_with_templates_has_the_template_keys(
    tmp_path: Path,
) -> None:
    run_dir = _execute_templates(tmp_path)

    assert set(_summary(run_dir)) == {
        "pairs",
        "sources",
        "models",
        "function_gap",
        "template_variants",
        "template_dropped",
        "template_sd",
        "template_sd_delta",
    }


def test_fewer_than_two_shared_variants_are_rejected(tmp_path: Path) -> None:
    config = _config(template_source="p1")
    runs = _reading_runs(tmp_path / "in")
    runs.update(_template_runs(tmp_path / "in"))
    runs[("p1", "t6")] = _spread_run(
        tmp_path / "other", "t6", {"v1": 0.7, "extra": 0.6}, task="sentiment"
    )
    run_dir = RunDir.create(config, out_root=tmp_path / "out")

    with pytest.raises(ValueError, match="shared by every model"):
        execute(config, run_dir, runs)


def test_a_model_without_a_template_run_is_named_before_any_read(
    tmp_path: Path,
) -> None:
    config = _config(template_source="p1")
    runs = _reading_runs(tmp_path / "in")
    runs.update(_template_runs(tmp_path / "in"))
    del runs[("p1", "ref")]
    run_dir = RunDir.create(config, out_root=tmp_path / "out")

    with pytest.raises(ValueError, match=r"runs.*p1.*ref"):
        execute(config, run_dir, runs)

    assert not list(run_dir.path.glob("*.npz"))


COEFFS = (-0.3, -0.1, 0.0, 0.1, 0.3)
BASE_SLOPES = {"b3": [0.2, 0.4, 0.6], "b6": [0.1, 0.5, 0.9]}
TUNED_SLOPES = {
    "t3": [1.2, 1.4, 1.6, 1.8],
    "t6": [0.6, 1.0, 1.4],
}


def _c1_run(
    out_root: Path,
    model: str,
    slopes: Sequence[float],
    *,
    coeffs: Sequence[float] = COEFFS,
    series: Sequence[str] = ("linear", "piecewise"),
) -> RunDir:
    """A finished C1-like run whose linear readers have the given slopes."""
    run_dir = _new_run(out_root, "c1", model)
    x = np.asarray(coeffs)
    curves = {
        "linear": 0.5 + np.asarray(slopes)[:, None] * x[None, :],
        "piecewise": np.full((len(slopes), len(x)), 0.5),
    }
    write_dose_curves(
        run_dir,
        np.stack([curves[name] for name in series]),
        coeffs,
        series=series,
        stem="curves_operators",
        xlabel="Relative coefficient",
        ylabel="Score",
    )
    run_dir.finalize()
    return run_dir


def _steering_runs(out_root: Path) -> dict[tuple[str, str], RunDir]:
    """C1-like runs on the paired models only."""
    slopes = {**BASE_SLOPES, **TUNED_SLOPES}
    return {
        ("c1", model): _c1_run(out_root, model, s)
        for model, s in slopes.items()
    }


def _execute_doses(
    tmp_path: Path, runs: Mapping[tuple[str, str], RunDir] | None = None
) -> RunDir:
    config = _config(steering_source="c1")
    all_runs = _reading_runs(tmp_path / "in")
    all_runs.update(_steering_runs(tmp_path / "in") if runs is None else runs)
    run_dir = RunDir.create(config, out_root=tmp_path / "out")
    execute(config, run_dir, all_runs)
    return run_dir


def test_the_dose_curves_pair_the_base_and_tuned_readers(
    tmp_path: Path,
) -> None:
    run_dir = _execute_doses(tmp_path)

    curves = read_curves(run_dir, "curves_dose_t3")
    assert curves.series == ("b3", "t3")
    np.testing.assert_allclose(curves.x, COEFFS)
    assert curves.per_seed.shape == (2, 3, 5)
    np.testing.assert_allclose(
        curves.mean[0], 0.5 + 0.4 * np.asarray(COEFFS), atol=1e-12
    )
    np.testing.assert_allclose(
        curves.mean[1], 0.5 + 1.4 * np.asarray(COEFFS), atol=1e-12
    )
    assert curves.chance is None


def test_the_slopes_hold_both_models_readers_and_their_paired_difference(
    tmp_path: Path,
) -> None:
    run_dir = _execute_doses(tmp_path)

    slopes = _summary(run_dir)["slopes"]
    assert set(slopes) == {"t3", "t6"}
    assert slopes["t3"]["base"] == pytest.approx([0.2, 0.4, 0.6])
    assert slopes["t3"]["tuned"] == pytest.approx([1.2, 1.4, 1.6, 1.8])
    assert slopes["t3"]["delta"] == pytest.approx(1.0)
    assert slopes["t6"]["delta"] == pytest.approx(0.5)


def test_the_reference_is_not_steered(tmp_path: Path) -> None:
    run_dir = _execute_doses(tmp_path)

    assert not (run_dir.path / "curves_dose_ref.npz").exists()
    assert "ref" not in _summary(run_dir)["slopes"]


def test_the_summary_with_doses_has_the_slopes(tmp_path: Path) -> None:
    run_dir = _execute_doses(tmp_path)

    assert set(_summary(run_dir)) == {
        "pairs",
        "sources",
        "models",
        "function_gap",
        "slopes",
    }


def test_pairs_with_different_coefficient_grids_are_rejected(
    tmp_path: Path,
) -> None:
    runs = _steering_runs(tmp_path / "in")
    runs[("c1", "t6")] = _c1_run(
        tmp_path / "other",
        "t6",
        TUNED_SLOPES["t6"],
        coeffs=(-0.3, -0.1, 0.0, 0.1, 0.2),
    )

    with pytest.raises(ValueError, match="coefficient grid"):
        _execute_doses(tmp_path, runs)


def test_a_steering_run_without_the_linear_operator_is_rejected(
    tmp_path: Path,
) -> None:
    runs = _steering_runs(tmp_path / "in")
    runs[("c1", "t3")] = _c1_run(
        tmp_path / "other", "t3", TUNED_SLOPES["t3"], series=("piecewise",)
    )

    with pytest.raises(ValueError, match="linear"):
        _execute_doses(tmp_path, runs)


def test_a_paired_model_without_a_steering_run_is_named_before_any_read(
    tmp_path: Path,
) -> None:
    config = _config(steering_source="c1")
    runs = _reading_runs(tmp_path / "in")
    runs.update(_steering_runs(tmp_path / "in"))
    del runs[("c1", "t6")]
    run_dir = RunDir.create(config, out_root=tmp_path / "out")

    with pytest.raises(ValueError, match=r"runs.*c1.*t6"):
        execute(config, run_dir, runs)

    assert not list(run_dir.path.glob("*.npz"))


@pytest.fixture
def figures(monkeypatch: pytest.MonkeyPatch) -> list[Figure]:
    """The figures that P6 and P8 save, in order."""
    saved: list[Figure] = []
    real = p6_scale.save_figure

    def spy(fig: Figure, out_dir: Path, name: str, **kwargs: Any) -> list[Path]:
        saved.append(fig)
        return real(fig, out_dir, name, **kwargs)

    monkeypatch.setattr(p6_scale, "save_figure", spy)
    monkeypatch.setattr(p8_tuning, "save_figure", spy)
    return saved


def _full_runs(tmp_path: Path) -> dict[tuple[str, str], RunDir]:
    runs = _reading_runs(tmp_path / "in")
    runs.update(_template_runs(tmp_path / "in"))
    runs.update(_steering_runs(tmp_path / "in"))
    return runs


def _full_run(tmp_path: Path, **overrides: Any) -> RunDir:
    config = _config(template_source="p1", steering_source="c1", **overrides)
    runs = _full_runs(tmp_path)
    run_dir = RunDir.create(config, out_root=tmp_path / "out")
    execute(config, run_dir, runs)
    return run_dir


def _labeled(ax: Any) -> dict[str, Any]:
    return {
        line.get_label(): line
        for line in ax.get_lines()
        if not line.get_label().startswith("_")
    }


def test_the_cli_discovers_the_experiment() -> None:
    experiment = load_experiment("p8")

    assert experiment.config_cls is Config
    assert experiment.run is run
    assert experiment.plot is plot


@pytest.mark.parametrize(
    ("overrides", "experiments", "models"),
    [
        ({}, ["r1", "r2", "r5", "p1", "c1"], ["b3", "t3", "b6", "t6", "ref"]),
        (
            {
                "template_source": None,
                "steering_source": None,
                "reference": None,
            },
            ["r1", "r2", "r5"],
            ["b3", "t3", "b6", "t6"],
        ),
    ],
    ids=["all-parts", "reading-only"],
)
def test_run_resolves_every_source_and_model_then_executes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
    experiments: list[str],
    models: list[str],
) -> None:
    settings = {
        "template_source": "p1",
        "steering_source": "c1",
        "extra_runs": ("elsewhere",),
        **overrides,
    }
    config = _config(**settings)
    runs = _full_runs(tmp_path)
    asked: list[tuple[Path, list[str], list[str], tuple[str, ...]]] = []

    def resolve(
        out_root: Path,
        wanted: Sequence[str],
        names: Sequence[str],
        extra_runs: Sequence[str] = (),
    ) -> dict[tuple[str, str], RunDir]:
        asked.append((out_root, list(wanted), list(names), tuple(extra_runs)))
        return runs

    monkeypatch.setattr(p8_tuning, "resolve_runs", resolve)
    run_dir = RunDir.create(config, out_root=tmp_path / "out")

    run(config, run_dir)

    assert asked == [(tmp_path / "out", experiments, models, ("elsewhere",))]
    assert _summary(run_dir)["models"] == models


def test_run_rejects_unpaired_models_before_it_looks_for_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def resolve(*args: Any, **kwargs: Any) -> dict[tuple[str, str], RunDir]:
        raise AssertionError("no run should be looked up")

    monkeypatch.setattr(p8_tuning, "resolve_runs", resolve)
    config = _config(base_models=("b3",))
    run_dir = RunDir.create(config, out_root=tmp_path)

    with pytest.raises(ValueError, match="same length"):
        run(config, run_dir)


def test_plot_writes_fifteen_files_with_every_part(
    tmp_path: Path,
) -> None:
    run_dir = _full_run(tmp_path)

    paths = plot(run_dir, tmp_path / "figs")

    names = ("tuning_depth", "tuning_pairs", "matrix_templates")
    names += ("curves_dose_t3", "curves_dose_t6")
    assert [p.name for p in paths] == [
        f"{name}.{extension}"
        for name in names
        for extension in ("pdf", "png", "svg")
    ]
    assert all(p.stat().st_size > 0 for p in paths)


def test_plot_without_the_optional_parts_writes_two_figures(
    tmp_path: Path,
) -> None:
    config = _config()
    runs = _reading_runs(tmp_path / "in")
    run_dir = RunDir.create(config, out_root=tmp_path / "out")
    execute(config, run_dir, runs)

    paths = plot(run_dir, tmp_path / "figs")

    assert [p.name for p in paths] == [
        f"{name}.{extension}"
        for name in ("tuning_depth", "tuning_pairs")
        for extension in ("pdf", "png", "svg")
    ]


def test_the_depth_figure_has_a_panel_per_source_and_a_curve_per_model(
    tmp_path: Path, figures: list[Figure]
) -> None:
    run_dir = _full_run(tmp_path)

    plot(run_dir, tmp_path / "figs")

    fig = figures[0]
    assert [ax.get_title() for ax in fig.axes] == ["r1", "r2", "r5"]
    for ax in fig.axes:
        assert list(_labeled(ax)) == ["b3", "t3", "b6", "t6", "ref"]


def test_the_pairs_figure_joins_the_base_and_tuned_accuracy_of_every_source(
    tmp_path: Path, figures: list[Figure]
) -> None:
    run_dir = _full_run(tmp_path)

    plot(run_dir, tmp_path / "figs")

    fig = figures[1]
    assert [ax.get_title() for ax in fig.axes] == ["b3 vs t3", "b6 vs t6"]
    rows = _rows(run_dir)
    for ax, tuned in zip(fig.axes, ("t3", "t6"), strict=True):
        mine = [r for r in rows if r["tuned"] == tuned]
        lines = _labeled(ax)
        assert list(lines) == ["base", "tuned"]
        np.testing.assert_allclose(
            lines["base"].get_ydata(), [r["base_accuracy"] for r in mine]
        )
        np.testing.assert_allclose(
            lines["tuned"].get_ydata(), [r["tuned_accuracy"] for r in mine]
        )
        assert [t.get_text() for t in ax.get_xticklabels()] == [
            "r1",
            "r2",
            "r5\n(function)",
        ]
        links = [line for line in ax.get_lines() if line.get_label() == "_link"]
        assert len(links) == len(mine)
        for link, row in zip(links, mine, strict=True):
            np.testing.assert_allclose(
                link.get_ydata(), [row["base_accuracy"], row["tuned_accuracy"]]
            )


def test_the_pairs_figure_has_one_legend_for_all_panels(
    tmp_path: Path, figures: list[Figure]
) -> None:
    run_dir = _full_run(tmp_path)

    plot(run_dir, tmp_path / "figs")

    fig = figures[1]
    assert all(ax.get_legend() is None for ax in fig.axes)
    (legend,) = fig.legends
    assert [t.get_text() for t in legend.get_texts()] == ["base", "tuned"]


@pytest.mark.parametrize(
    ("n_pairs", "rows", "cols"),
    [(1, 1, 1), (2, 1, 2), (3, 2, 2)],
    ids=["one", "two", "three"],
)
def test_the_pairs_figure_takes_two_columns_at_most(
    tmp_path: Path,
    figures: list[Figure],
    n_pairs: int,
    rows: int,
    cols: int,
) -> None:
    pairs = [(f"b{i}", f"t{i}") for i in range(n_pairs)]
    config = _config(
        base_models=tuple(b for b, _ in pairs),
        tuned_models=tuple(t for _, t in pairs),
        model="t0",
        reference=None,
    )
    run_dir = RunDir.create(config, out_root=tmp_path / "out")
    execute(config, run_dir, _many_pairs(tmp_path / "in", pairs))

    plot(run_dir, tmp_path / "figs")

    fig = figures[1]
    grid = fig.axes[0].get_subplotspec().get_gridspec()
    assert len(fig.axes) == n_pairs
    assert (grid.nrows, grid.ncols) == (rows, cols)


def _many_pairs(
    out_root: Path, pairs: Sequence[tuple[str, str]]
) -> dict[tuple[str, str], RunDir]:
    """Reading runs of the three sources on each pair, tuned 0.1 above base."""
    runs: dict[tuple[str, str], RunDir] = {}
    for base, tuned in pairs:
        for model, peak in ((base, 0.6), (tuned, 0.7)):
            for source in (*CONCEPTS, *FUNCTIONS):
                runs[(source, model)] = _reading_run(
                    out_root, source, model, _peaked(3, 2, peak)
                )
    return runs
