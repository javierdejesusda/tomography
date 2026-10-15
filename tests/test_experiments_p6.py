"""Tests for the P6 scale experiment."""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray
import pytest
import torch
from transformers import AutoModelForCausalLM
from transformers import GPT2Config
from transformers import PretrainedConfig

from tomography.cli import load_experiment
from tomography.experiments import p6_scale
from tomography.experiments.aggregate import read_curves
from tomography.experiments.base import ExperimentConfig
from tomography.experiments.base import RunDir
from tomography.experiments.effects import write_dose_curves
from tomography.experiments.p6_scale import GPT2_SIZES
from tomography.experiments.p6_scale import Config
from tomography.experiments.p6_scale import copy_reader_curves
from tomography.experiments.p6_scale import count_parameters
from tomography.experiments.p6_scale import execute
from tomography.experiments.p6_scale import parameter_count
from tomography.experiments.p6_scale import plot
from tomography.experiments.p6_scale import plot_depth_panels
from tomography.experiments.p6_scale import run
from tomography.experiments.p6_scale import steering_slopes
from tomography.experiments.sweep import SERIES
from tomography.experiments.sweep import write_curves
from tomography.model import MODELS
from tomography.model import LoadedModel
from tomography.model import UnknownCheckpointError
from tomography.reading.metrics import bootstrap_ci

OFFSETS = np.array([-0.02, 0.02, -0.01, 0.01])
COEFFS = (-0.3, -0.1, 0.0, 0.1, 0.3)
SIZES = ("s3", "s6", "s12")
DEPTH = {"s3": 3, "s6": 6, "s12": 12}
PARAMS = {"s3": 1_000, "s6": 10_000, "s12": 100_000}
R1_PEAK = {"s3": (2, 0.60), "s6": (3, 0.70), "s12": (9, 0.80)}
R2_VAL = {"s3": (1, 0.55), "s6": (2, 0.65), "s12": (4, 0.75)}
SLOPE = {"s3": 0.5, "s6": 1.5, "s12": 2.5}
READING_LAYER = {"s3": 2, "s6": 3, "s12": 8}
STEER_LAYER = {"s3": 3, "s6": 4, "s12": 6}


@dataclasses.dataclass(frozen=True, kw_only=True)
class _RunConfig(ExperimentConfig):
    """Test-only config of a finished source run."""


def _new_run(out_root: Path, experiment: str, model: str) -> RunDir:
    return RunDir.create(
        _RunConfig(experiment=experiment, model=model), out_root=out_root
    )


def _per_seed(reader: Sequence[float]) -> NDArray[np.float64]:
    """Accuracy of shape (3, N, K): the reader shifted per seed, two nulls."""
    rows = np.asarray(reader)[None, :] + OFFSETS[:, None]
    null = np.full_like(rows, 0.5)
    return np.stack([rows, null, null])


def _reading_run(
    out_root: Path,
    experiment: str,
    model: str,
    reader: Sequence[float],
    *,
    val: Sequence[float] | None = None,
) -> RunDir:
    """A finished reading run with `curves` and, if given, `curves_val`."""
    run_dir = _new_run(out_root, experiment, model)
    layers = list(range(1, len(reader) + 1))
    for stem, values in (("curves", reader), ("curves_val", val)):
        if values is not None:
            write_curves(
                run_dir,
                _per_seed(values),
                layers,
                series=SERIES,
                stem=stem,
                chance=0.5,
                n_layers=len(reader),
            )
    run_dir.finalize()
    return run_dir


def _c1_run(
    out_root: Path,
    model: str,
    slopes: Sequence[float],
    *,
    series: Sequence[str] = ("linear", "piecewise"),
    reading_layer: int = 2,
    steer_layer: int = 3,
) -> RunDir:
    """A finished C1-like run whose linear readers have the given slopes."""
    run_dir = _new_run(out_root, "c1", model)
    x = np.asarray(COEFFS)
    curves = {
        "linear": 0.5 + np.asarray(slopes)[:, None] * x[None, :],
        "piecewise": np.full((len(slopes), len(x)), 0.5),
    }
    write_dose_curves(
        run_dir,
        np.stack([curves[name] for name in series]),
        COEFFS,
        series=series,
        stem="curves_operators",
        xlabel="Relative coefficient",
        ylabel="Score",
    )
    run_dir.write_json(
        "summary",
        {"reading_layer": reading_layer, "steer_layer": steer_layer},
    )
    run_dir.finalize()
    return run_dir


def _peaked(
    n_layers: int, layer: int, value: float, floor: float = 0.5
) -> list[float]:
    """A flat curve with one point at `layer` raised to `value`."""
    curve = [floor] * n_layers
    curve[layer - 1] = value
    return curve


def _world(
    out_root: Path, *, steering: bool = True
) -> dict[tuple[str, str], RunDir]:
    """Finished R1-like, R2-like and C1-like runs on three made-up sizes.

    The R1-like runs have no validation curves. The R2-like runs have them,
    and their test curves peak higher at the last layer than at the layer the
    validation curves choose.
    """
    runs: dict[tuple[str, str], RunDir] = {}
    for model in SIZES:
        depth = DEPTH[model]
        layer, accuracy = R1_PEAK[model]
        runs[("r1", model)] = _reading_run(
            out_root, "r1", model, _peaked(depth, layer, accuracy)
        )
        chosen, at_chosen = R2_VAL[model]
        test = _peaked(depth, depth, at_chosen + 0.1)
        test[chosen - 1] = at_chosen
        runs[("r2", model)] = _reading_run(
            out_root, "r2", model, test, val=_peaked(depth, chosen, 0.9)
        )
        if steering:
            slope = SLOPE[model]
            runs[("c1", model)] = _c1_run(
                out_root,
                model,
                [slope - 0.1, slope, slope + 0.1],
                reading_layer=READING_LAYER[model],
                steer_layer=STEER_LAYER[model],
            )
    return runs


def _config(**overrides: Any) -> Config:
    base = Config(model="s3", models=SIZES, sources=("r1", "r2"), n_boot=50)
    return dataclasses.replace(base, **overrides)


def _execute(tmp_path: Path, **overrides: Any) -> RunDir:
    config = _config(**overrides)
    runs = _world(tmp_path / "in", steering=config.steering_source is not None)
    run_dir = RunDir.create(config, out_root=tmp_path / "out")
    execute(config, run_dir, runs, PARAMS)
    return run_dir


def _rows(run_dir: RunDir, stem: str = "table") -> list[dict[str, Any]]:
    payload = run_dir.read_json(stem)
    assert isinstance(payload, dict)
    return [
        dict(zip(payload["columns"], row, strict=True))
        for row in payload["rows"]
    ]


def test_the_config_defaults_cover_the_four_gpt2_sizes() -> None:
    config = Config()

    assert config.experiment == "p6"
    assert GPT2_SIZES == ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")
    assert config.models == GPT2_SIZES
    assert config.model in config.models
    assert config.sources == ("r1", "r2", "r3", "r5", "r7")
    assert config.steering_source == "c1"
    assert config.extra_runs == ()
    assert config.n_boot == 2000


def test_count_parameters_matches_the_built_model(
    tiny_arch: LoadedModel,
) -> None:
    built = sum(p.numel() for p in tiny_arch.model.parameters())

    assert count_parameters(tiny_arch.model.config) == built


def test_count_parameters_counts_a_tied_embedding_once(
    tiny_gpt2: LoadedModel,
) -> None:
    model = tiny_gpt2.model
    with_copies = sum(
        p.numel() for _, p in model.named_parameters(remove_duplicate=False)
    )

    count = count_parameters(model.config)

    assert count == with_copies - model.get_input_embeddings().weight.numel()


def test_count_parameters_builds_the_model_on_the_meta_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    devices: list[str] = []
    real = AutoModelForCausalLM.from_config

    def spy(config: PretrainedConfig) -> object:
        devices.append(torch.get_default_device().type)
        return real(config)

    monkeypatch.setattr(AutoModelForCausalLM, "from_config", spy)
    config = GPT2Config(vocab_size=50, n_layer=2, n_embd=16, n_head=2)

    assert count_parameters(config) > 0
    assert devices == ["meta"]


def test_parameter_count_loads_the_pinned_config_of_the_key(
    tiny_gpt2: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str | None]] = []

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(
            repo_id: str, revision: str | None = None
        ) -> object:
            calls.append((repo_id, revision))
            return tiny_gpt2.model.config

    monkeypatch.setattr(p6_scale, "AutoConfig", FakeAutoConfig)

    count = parameter_count("gpt2-medium")

    spec = MODELS["gpt2-medium"]
    assert calls == [(spec.repo_id, spec.revision)]
    assert count == sum(p.numel() for p in tiny_gpt2.model.parameters())


def test_parameter_count_rejects_an_unknown_key() -> None:
    with pytest.raises(UnknownCheckpointError, match="not-a-model"):
        parameter_count("not-a-model")


@pytest.mark.parametrize(
    "model", ["tiny", "qwen3-1.7b"], ids=["plain", "dotted"]
)
def test_copy_reader_curves_keeps_only_the_reader_series(
    tmp_path: Path, model: str
) -> None:
    source = _reading_run(tmp_path / "in", "r1", model, [0.5, 0.7, 0.9, 0.6])
    run_dir = _new_run(tmp_path / "out", "p6", "gpt2")

    copy_reader_curves(run_dir, "r1", model, source, n_boot=40, seed=3)

    copied = read_curves(run_dir, f"curves_r1_{model}")
    original = read_curves(source)
    assert copied.series == (model,)
    assert copied.x.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert copied.n_layers == 4
    assert copied.chance == 0.5
    np.testing.assert_array_equal(copied.per_seed[0], original.per_seed[0])
    np.testing.assert_allclose(copied.mean[0], original.mean[0])


def test_copy_reader_curves_draws_a_fresh_band_from_the_run_settings(
    tmp_path: Path,
) -> None:
    source = _reading_run(tmp_path / "in", "r1", "tiny", [0.5, 0.7, 0.9, 0.6])
    run_dir = _new_run(tmp_path / "out", "p6", "gpt2")

    copy_reader_curves(run_dir, "r1", "tiny", source, n_boot=40, seed=3)

    copied = read_curves(run_dir, "curves_r1_tiny")
    lo, hi = bootstrap_ci(read_curves(source).per_seed[0], n_boot=40, seed=3)
    np.testing.assert_allclose(copied.lo[0], lo)
    np.testing.assert_allclose(copied.hi[0], hi)


def test_a_reading_run_without_a_reader_series_is_rejected(
    tmp_path: Path,
) -> None:
    source = _new_run(tmp_path / "in", "r1", "tiny")
    write_curves(
        source,
        _per_seed([0.5, 0.7])[1:2],
        [1, 2],
        series=("random",),
        chance=0.5,
        n_layers=2,
    )
    run_dir = _new_run(tmp_path / "out", "p6", "gpt2")

    with pytest.raises(ValueError, match="no reader series"):
        copy_reader_curves(run_dir, "r1", "tiny", source, n_boot=10, seed=0)


def test_steering_slopes_returns_one_linear_slope_per_reader(
    tmp_path: Path,
) -> None:
    run = _c1_run(
        tmp_path, "tiny", [0.2, 0.6, 1.0], series=("piecewise", "linear")
    )

    slopes = steering_slopes(run)

    assert slopes.shape == (3,)
    np.testing.assert_allclose(slopes, [0.2, 0.6, 1.0])


def test_steering_slopes_needs_the_linear_operator(tmp_path: Path) -> None:
    run = _c1_run(tmp_path, "tiny", [0.2, 0.6], series=("piecewise",))

    with pytest.raises(ValueError, match="linear"):
        steering_slopes(run)


def test_execute_copies_the_reader_curves_of_every_source_and_model(
    tmp_path: Path,
) -> None:
    run_dir = _execute(tmp_path)

    for source in ("r1", "r2"):
        for model in SIZES:
            copied = read_curves(run_dir, f"curves_{source}_{model}")
            assert copied.series == (model,)
            assert copied.n_layers == DEPTH[model]


def test_execute_tabulates_every_source_and_model_in_order(
    tmp_path: Path,
) -> None:
    run_dir = _execute(tmp_path)

    payload = run_dir.read_json("table")
    assert isinstance(payload, dict)
    assert payload["columns"] == [
        "source",
        "model",
        "params",
        "n_layers",
        "layer",
        "depth",
        "selected_on",
        "accuracy",
        "lo",
        "hi",
    ]
    rows = _rows(run_dir)
    assert [(r["source"], r["model"]) for r in rows] == [
        (source, model) for source in ("r1", "r2") for model in SIZES
    ]
    assert [r["params"] for r in rows[:3]] == [PARAMS[m] for m in SIZES]
    assert [r["n_layers"] for r in rows[:3]] == [DEPTH[m] for m in SIZES]


def test_the_best_accuracy_follows_the_planted_order_of_the_sizes(
    tmp_path: Path,
) -> None:
    run_dir = _execute(tmp_path)

    first = [r for r in _rows(run_dir) if r["source"] == "r1"]
    assert [r["layer"] for r in first] == [R1_PEAK[m][0] for m in SIZES]
    assert [r["depth"] for r in first] == pytest.approx(
        [R1_PEAK[m][0] / DEPTH[m] for m in SIZES]
    )
    assert [r["accuracy"] for r in first] == pytest.approx(
        [R1_PEAK[m][1] for m in SIZES]
    )
    assert [r["accuracy"] for r in first] == sorted(
        r["accuracy"] for r in first
    )
    assert all(r["lo"] <= r["accuracy"] <= r["hi"] for r in first)
    assert all(r["lo"] >= 0.58 and r["hi"] <= 0.82 for r in first)


def test_the_layer_is_chosen_on_validation_curves_when_the_run_has_them(
    tmp_path: Path,
) -> None:
    run_dir = _execute(tmp_path)

    rows = _rows(run_dir)
    first = [r for r in rows if r["source"] == "r1"]
    second = [r for r in rows if r["source"] == "r2"]
    assert {r["selected_on"] for r in first} == {"curves"}
    assert {r["selected_on"] for r in second} == {"curves_val"}
    assert [r["layer"] for r in second] == [R2_VAL[m][0] for m in SIZES]
    assert [r["accuracy"] for r in second] == pytest.approx(
        [R2_VAL[m][1] for m in SIZES]
    )


def test_the_steering_table_gives_the_slopes_and_the_depths(
    tmp_path: Path,
) -> None:
    run_dir = _execute(tmp_path)

    payload = run_dir.read_json("table_steering")
    assert isinstance(payload, dict)
    assert payload["columns"] == [
        "model",
        "params",
        "steer_layer",
        "steer_depth",
        "reading_layer",
        "reading_depth",
        "slope",
        "slope_min",
        "slope_max",
        "n_readers",
    ]
    rows = _rows(run_dir, "table_steering")
    assert [r["model"] for r in rows] == list(SIZES)
    assert [r["params"] for r in rows] == [PARAMS[m] for m in SIZES]
    assert [r["steer_layer"] for r in rows] == [STEER_LAYER[m] for m in SIZES]
    assert [r["steer_depth"] for r in rows] == pytest.approx(
        [STEER_LAYER[m] / DEPTH[m] for m in SIZES]
    )
    assert [r["reading_depth"] for r in rows] == pytest.approx(
        [READING_LAYER[m] / DEPTH[m] for m in SIZES]
    )
    assert [r["slope"] for r in rows] == pytest.approx(
        [SLOPE[m] for m in SIZES], abs=0.05
    )
    assert [r["slope_min"] for r in rows] == pytest.approx(
        [SLOPE[m] - 0.1 for m in SIZES], abs=0.05
    )
    assert [r["slope_max"] for r in rows] == pytest.approx(
        [SLOPE[m] + 0.1 for m in SIZES], abs=0.05
    )
    assert {r["n_readers"] for r in rows} == {3}


def test_without_a_steering_source_there_is_no_steering_table(
    tmp_path: Path,
) -> None:
    run_dir = _execute(tmp_path, steering_source=None)

    assert not (run_dir.path / "table_steering.json").exists()
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    assert summary["steering"] is None


def test_the_summary_names_the_run_and_the_accuracy_per_decade(
    tmp_path: Path,
) -> None:
    run_dir = _execute(tmp_path)

    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    assert set(summary) == {
        "models",
        "params",
        "sources",
        "steering",
        "accuracy_per_decade",
    }
    assert summary["models"] == list(SIZES)
    assert summary["params"] == PARAMS
    assert summary["sources"] == ["r1", "r2"]
    assert summary["steering"] == "c1"
    per_decade = summary["accuracy_per_decade"]
    assert set(per_decade) == {"r1", "r2"}
    assert per_decade["r1"] == pytest.approx(0.10)
    assert per_decade["r2"] == pytest.approx(0.10)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model": "other"}, "model"),
        ({"models": ("s3",)}, "models"),
        ({"models": tuple(f"s{i}" for i in range(9))}, "models"),
        ({"models": ("s3", "s3", "s6")}, "models"),
        ({"sources": ()}, "sources"),
        (
            {"sources": ("r1", "r2", "r3", "r4", "r5", "r6", "r7")},
            "sources",
        ),
        ({"sources": ("r1", "r1")}, "sources"),
    ],
    ids=[
        "model-outside",
        "one-model",
        "nine-models",
        "repeated-model",
        "no-source",
        "seven-sources",
        "repeated-source",
    ],
)
def test_execute_rejects_a_bad_config_before_reading_anything(
    tmp_path: Path, overrides: dict[str, Any], message: str
) -> None:
    config = _config(**overrides)
    run_dir = RunDir.create(config, out_root=tmp_path)
    runs: Mapping[tuple[str, str], RunDir] = {}

    with pytest.raises(ValueError, match=message):
        execute(config, run_dir, runs, PARAMS)

    assert not list(run_dir.path.glob("*.npz"))


def test_execute_names_a_missing_run(tmp_path: Path) -> None:
    config = _config()
    runs = _world(tmp_path / "in")
    del runs[("r2", "s12")]
    run_dir = RunDir.create(config, out_root=tmp_path / "out")

    with pytest.raises(ValueError, match=r"runs.*r2.*s12"):
        execute(config, run_dir, runs, PARAMS)

    assert not list(run_dir.path.glob("*.npz"))


def test_execute_names_a_model_without_a_parameter_count(
    tmp_path: Path,
) -> None:
    config = _config()
    runs = _world(tmp_path / "in")
    run_dir = RunDir.create(config, out_root=tmp_path / "out")

    with pytest.raises(ValueError, match=r"n_params.*s12"):
        execute(config, run_dir, runs, {"s3": 1, "s6": 2})


@pytest.fixture
def figures(monkeypatch: pytest.MonkeyPatch) -> list[Figure]:
    """The figures that `p6_scale` saves, in order."""
    saved: list[Figure] = []
    real = p6_scale.save_figure

    def spy(fig: Figure, out_dir: Path, name: str, **kwargs: Any) -> list[Path]:
        saved.append(fig)
        return real(fig, out_dir, name, **kwargs)

    monkeypatch.setattr(p6_scale, "save_figure", spy)
    return saved


def _panel_run(
    tmp_path: Path, sources: Sequence[str], models: Sequence[str] = SIZES
) -> RunDir:
    """A run holding only the reader copies the depth panels read."""
    run_dir = _new_run(tmp_path, "p6", "s3")
    for source in sources:
        for model in models:
            depth = DEPTH[model]
            write_curves(
                run_dir,
                _per_seed(_peaked(depth, 2, 0.8))[:1],
                list(range(1, depth + 1)),
                series=(model,),
                stem=f"curves_{source}_{model}",
                chance=0.5,
                n_layers=depth,
            )
    return run_dir


def _labeled(ax: Any) -> dict[str, Any]:
    return {
        line.get_label(): line
        for line in ax.get_lines()
        if not line.get_label().startswith("_")
    }


def _floats(values: Any) -> NDArray[np.float64]:
    """Error bar lines keep their data as objects."""
    return np.asarray(values, dtype=np.float64)


def _bar_ends(container: Any) -> NDArray[np.float64]:
    """The (low, high) end of every vertical bar of an error bar container."""
    (bars,) = container.lines[2]
    return np.array([[seg[0][1], seg[1][1]] for seg in bars.get_segments()])


def test_plot_writes_the_depth_and_best_figures_in_three_formats(
    tmp_path: Path,
) -> None:
    run_dir = _execute(tmp_path)

    paths = plot(run_dir, tmp_path / "figs")

    assert [p.name for p in paths] == [
        f"{name}.{extension}"
        for name in ("scale_depth", "scale_best")
        for extension in ("pdf", "png", "svg")
    ]
    assert all(p.stat().st_size > 0 for p in paths)


def test_the_depth_panels_draw_each_model_on_relative_depth(
    tmp_path: Path, figures: list[Figure]
) -> None:
    run_dir = _execute(tmp_path)

    paths = plot_depth_panels(
        run_dir, tmp_path / "figs", ["r1", "r2"], SIZES, "depth"
    )

    assert [p.name for p in paths] == ["depth.pdf", "depth.png", "depth.svg"]
    fig = figures[0]
    assert [ax.get_title() for ax in fig.axes] == ["r1", "r2"]
    for ax in fig.axes:
        lines = _labeled(ax)
        assert list(lines) == list(SIZES)
        assert ax.get_xlim() == (0.0, 1.0)
        assert "_chance" in [line.get_label() for line in ax.get_lines()]
        for model in SIZES:
            layers = np.arange(1, DEPTH[model] + 1)
            np.testing.assert_allclose(
                lines[model].get_xdata(), layers / DEPTH[model]
            )


def test_only_the_first_depth_panel_has_a_legend(
    tmp_path: Path, figures: list[Figure]
) -> None:
    run_dir = _execute(tmp_path)

    plot_depth_panels(run_dir, tmp_path / "figs", ["r1", "r2"], SIZES, "depth")

    first, second = figures[0].axes
    assert first.get_legend() is not None
    assert second.get_legend() is None


@pytest.mark.parametrize(
    ("n_sources", "rows", "cols"),
    [(1, 1, 1), (2, 1, 2), (3, 1, 3), (4, 2, 3), (5, 2, 3), (6, 2, 3)],
    ids=["one", "two", "three", "four", "five", "six"],
)
def test_the_depth_panels_take_three_columns_at_most(
    tmp_path: Path,
    figures: list[Figure],
    n_sources: int,
    rows: int,
    cols: int,
) -> None:
    sources = [f"r{i}" for i in range(1, n_sources + 1)]
    run_dir = _panel_run(tmp_path, sources)

    plot_depth_panels(run_dir, tmp_path / "figs", sources, SIZES, "depth")

    fig = figures[0]
    grid = fig.axes[0].get_subplotspec().get_gridspec()
    assert len(fig.axes) == n_sources
    assert (grid.nrows, grid.ncols) == (rows, cols)


@pytest.mark.parametrize(
    ("sources", "models"),
    [
        ([], SIZES),
        ([f"r{i}" for i in range(1, 8)], SIZES),
        (["r1"], ()),
    ],
    ids=["no-source", "seven-sources", "no-model"],
)
def test_the_depth_panels_reject_a_bad_count(
    tmp_path: Path, sources: list[str], models: Sequence[str]
) -> None:
    run_dir = _new_run(tmp_path, "p6", "s3")

    with pytest.raises(ValueError, match=r"sources|models"):
        plot_depth_panels(run_dir, tmp_path / "figs", sources, models, "depth")


def test_the_best_figure_plots_accuracy_depth_and_slope_against_parameters(
    tmp_path: Path, figures: list[Figure]
) -> None:
    run_dir = _execute(tmp_path)

    plot(run_dir, tmp_path / "figs")

    accuracy, depth, slope = figures[1].axes
    assert {ax.get_xscale() for ax in (accuracy, depth, slope)} == {"log"}
    params = [PARAMS[m] for m in SIZES]
    rows = _rows(run_dir)
    bars = {c.get_label(): c for c in accuracy.containers}
    assert list(bars) == ["r1", "r2"]
    for source in ("r1", "r2"):
        mine = [r for r in rows if r["source"] == source]
        line = bars[source].lines[0]
        np.testing.assert_allclose(_floats(line.get_xdata()), params)
        np.testing.assert_allclose(
            _floats(line.get_ydata()), [r["accuracy"] for r in mine]
        )
        np.testing.assert_allclose(
            _bar_ends(bars[source]),
            [[r["lo"], r["hi"]] for r in mine],
        )
        np.testing.assert_allclose(
            _labeled(depth)[source].get_ydata(), [r["depth"] for r in mine]
        )
    steering = _rows(run_dir, "table_steering")
    (container,) = slope.containers
    np.testing.assert_allclose(_floats(container.lines[0].get_xdata()), params)
    np.testing.assert_allclose(
        _floats(container.lines[0].get_ydata()),
        [r["slope"] for r in steering],
    )
    np.testing.assert_allclose(
        _bar_ends(container),
        [[r["slope_min"], r["slope_max"]] for r in steering],
    )


def test_the_best_figure_has_no_slope_panel_without_steering(
    tmp_path: Path, figures: list[Figure]
) -> None:
    run_dir = _execute(tmp_path, steering_source=None)

    paths = plot(run_dir, tmp_path / "figs")

    assert len(figures[1].axes) == 2
    assert len(paths) == 6


def test_the_cli_discovers_the_experiment() -> None:
    experiment = load_experiment("p6")

    assert experiment.config_cls is Config
    assert experiment.run is run
    assert experiment.plot is plot


@pytest.mark.parametrize(
    ("overrides", "experiments"),
    [
        ({}, ["r1", "r2", "c1"]),
        ({"steering_source": None}, ["r1", "r2"]),
    ],
    ids=["with-steering", "without-steering"],
)
def test_run_resolves_the_runs_counts_the_parameters_and_executes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
    experiments: list[str],
) -> None:
    config = _config(extra_runs=("elsewhere",), **overrides)
    runs = _world(tmp_path / "in", steering=config.steering_source is not None)
    asked: list[tuple[Path, list[str], list[str], tuple[str, ...]]] = []
    counted: list[str] = []

    def resolve(
        out_root: Path,
        wanted: Sequence[str],
        models: Sequence[str],
        extra_runs: Sequence[str] = (),
    ) -> dict[tuple[str, str], RunDir]:
        asked.append((out_root, list(wanted), list(models), tuple(extra_runs)))
        return runs

    def count(key: str) -> int:
        counted.append(key)
        return PARAMS[key]

    monkeypatch.setattr(p6_scale, "resolve_runs", resolve)
    monkeypatch.setattr(p6_scale, "parameter_count", count)
    run_dir = RunDir.create(config, out_root=tmp_path / "out")

    run(config, run_dir)

    assert asked == [
        (tmp_path / "out", experiments, list(SIZES), ("elsewhere",))
    ]
    assert counted == list(SIZES)
    assert [r["params"] for r in _rows(run_dir)[:3]] == [
        PARAMS[m] for m in SIZES
    ]


def test_an_interval_that_misses_the_mean_still_plots(tmp_path: Path) -> None:
    run_dir = _execute(tmp_path, steering_source=None)
    payload = run_dir.read_json("table")
    assert isinstance(payload, dict)
    columns = payload["columns"]
    for row in payload["rows"]:
        row[columns.index("lo")] = row[columns.index("accuracy")] + 0.01
        row[columns.index("hi")] = row[columns.index("accuracy")] - 0.01
    run_dir.write_json("table", payload)

    paths = plot(run_dir, tmp_path / "figs")

    assert len(paths) == 6


def test_the_panel_titles_sit_above_the_legend(
    tmp_path: Path, figures: list[Figure]
) -> None:
    run_dir = _execute(tmp_path)

    plot_depth_panels(run_dir, tmp_path / "figs", ["r1", "r2"], SIZES, "depth")

    first, second = figures[0].axes
    legend = first.get_legend()
    assert legend is not None
    legend_top = legend.get_window_extent().y1
    assert first.title.get_window_extent().y0 >= legend_top
    assert second.title.get_window_extent().y0 >= legend_top


@pytest.mark.slow
def test_the_registered_checkpoints_have_their_published_sizes() -> None:
    keys = (*GPT2_SIZES, "qwen3-1.7b-base", "qwen3-1.7b")
    keys += ("qwen3-0.6b-base", "qwen3-0.6b")
    counts: dict[str, int] = {}
    for key in keys:
        try:
            counts[key] = parameter_count(key)
        except OSError:
            pytest.skip(f"the config of {key} is not in the cache")

    print(f"parameter counts: {counts}")
    expected = (124e6, 355e6, 774e6, 1558e6)
    for key, size in zip(GPT2_SIZES, expected, strict=True):
        assert counts[key] == pytest.approx(size, rel=0.01)
    assert counts["qwen3-1.7b-base"] == counts["qwen3-1.7b"]
    assert counts["qwen3-0.6b-base"] == counts["qwen3-0.6b"]
    assert counts["qwen3-0.6b"] < counts["qwen3-1.7b"]
    assert 1.5e9 < counts["qwen3-1.7b"] < 2.0e9
    assert 0.5e9 < counts["qwen3-0.6b"] < 0.7e9
