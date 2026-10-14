"""Tests for the R2 utility reading experiment."""

from collections.abc import Sequence
import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tomography.analysis import pca_explained_variance
from tomography.cli import load_experiment
from tomography.datasets.ethics import load_ethics_utility
from tomography.datasets.records import Example
from tomography.datasets.records import Split
from tomography.experiments import r2_utility
from tomography.experiments.base import RunDir
from tomography.experiments.r2_utility import Config
from tomography.experiments.r2_utility import execute
from tomography.experiments.r2_utility import plot
from tomography.experiments.r2_utility import run
from tomography.experiments.sweep import SERIES
from tomography.experiments.sweep import Prepared
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.templates import PAPER_TEMPLATES

PLEASANT = ("joy", "party", "gift", "sun")
UNPLEASANT = ("loss", "storm", "fine", "debt")
SAME_LENGTH = (("A",), ("B",))


def _examples(
    rows: dict[Split, int] | None = None,
    words: tuple[tuple[str, ...], tuple[str, ...]] = (PLEASANT, UNPLEASANT),
) -> list[Example]:
    rows = rows or {"train": 12, "val": 12, "test": 12}
    pleasant, unpleasant = words
    examples = []
    for split, count in rows.items():
        for r in range(count):
            row = f"{split}-{r}"
            good = pleasant[r % len(pleasant)]
            bad = unpleasant[(r + 1) % len(unpleasant)]
            for suffix, label, word in (("a", 1, good), ("b", 0, bad)):
                examples.append(
                    Example(
                        uid=f"{row}:{suffix}",
                        text=f"I got a {word} today, number {r}",
                        label=label,
                        split=split,
                        meta={"pair": row},
                    )
                )
    return examples


def _config(**overrides: Any) -> Config:
    base = Config(
        model="tiny",
        n_seeds=3,
        n_pairs=8,
        n_eval_pairs=6,
        n_components=3,
    )
    return dataclasses.replace(base, **overrides)


def test_config_defaults_and_round_trip() -> None:
    config = Config()
    assert config.experiment == "r2"
    assert config.model == "gpt2"
    assert config.method == "logistic"
    assert config.n_components == 10
    assert config.n_eval_pairs is None
    assert config.n_seeds == 10
    assert config.n_pairs == 64
    assert Config.from_dict(config.to_dict()) == config


def test_the_cli_discovers_the_experiment() -> None:
    experiment = load_experiment("r2")
    assert experiment.config_cls is Config
    assert experiment.run is run
    assert experiment.plot is plot


def test_execute_writes_curves_variance_readers_and_summary(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config()
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _examples())
    n_layers = tiny_arch.n_layers
    layers = np.arange(1, n_layers + 1)

    for stem in ("curves", "curves_val"):
        curves = run_dir.read_arrays(stem)
        assert curves["x"].tolist() == layers.tolist()
        assert curves["mean"].shape == (3, n_layers)
        assert curves["per_seed"].shape == (3, 3, n_layers)
        sidecar = run_dir.read_json(stem)
        assert isinstance(sidecar, dict)
        assert sidecar["series"] == list(SERIES)
        assert sidecar["chance"] == 0.5

    variance = run_dir.read_arrays("variance")
    assert variance["mean"].shape == (3, n_layers)
    assert variance["per_seed"].shape == (3, 3, n_layers)
    assert variance["per_seed"].min() >= 0
    assert variance["per_seed"].max() <= 1
    assert (variance["mean"][0] >= variance["mean"][1]).all()
    assert (variance["mean"][1] >= variance["mean"][2]).all()
    sidecar = run_dir.read_json("variance")
    assert isinstance(sidecar, dict)
    assert sidecar["series"] == ["PC1", "PC2", "PC3"]
    assert sidecar["ylabel"] == "Explained variance"
    assert sidecar["chance"] is None

    readers = run_dir.read_arrays("readers")
    assert readers["layers"].dtype == np.int64
    assert readers["layers"].tolist() == layers.tolist()
    assert readers["directions"].shape == (3, n_layers, tiny_arch.d_model)
    assert readers["directions"].dtype == np.float32
    assert readers["signs"].shape == (3, n_layers)
    assert readers["signs"].dtype == np.float32

    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    assert set(summary) == {
        "best_layer",
        "val_accuracy",
        "test_accuracy",
        "ci",
        "n_seeds",
        "n_pairs",
        "n_eval_pairs",
    }
    val = run_dir.read_arrays("curves_val")
    test = run_dir.read_arrays("curves")
    best = int(np.argmax(val["mean"][0]))
    assert summary["best_layer"] == layers[best]
    assert summary["val_accuracy"] == pytest.approx(val["mean"][0, best])
    assert summary["test_accuracy"] == pytest.approx(test["mean"][0, best])
    assert summary["ci"] == pytest.approx(
        [test["lo"][0, best], test["hi"][0, best]]
    )
    assert summary["n_seeds"] == 3
    assert summary["n_pairs"] == 8
    assert summary["n_eval_pairs"] == {"val": 6, "test": 6}


def test_stimuli_are_grouped_by_row_and_split_before_pairing(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[Example], dict[str, Any]]] = []
    real = r2_utility.build_stimulus_set

    def spy(examples: Any, template: Any, chat: Any, **kwargs: Any) -> Any:
        calls.append((list(examples), {"template": template, **kwargs}))
        return real(examples, template, chat, **kwargs)

    monkeypatch.setattr(r2_utility, "build_stimulus_set", spy)
    config = _config()
    execute(
        config,
        RunDir.create(config, out_root=tmp_path),
        tiny_arch,
        _examples(),
    )
    assert len(calls) == config.n_seeds + 2
    for examples, kwargs in calls:
        assert kwargs["pairing"] == "by_group"
        assert kwargs["template"] is PAPER_TEMPLATES["utility"]
        assert kwargs["group_ids"] == [str(e.meta["pair"]) for e in examples]
    splits = [{e.split for e in examples} for examples, _ in calls]
    assert splits[: config.n_seeds] == [{"train"}] * config.n_seeds
    assert splits[config.n_seeds :] == [{"val"}, {"test"}]
    ids = [
        {str(e.meta["pair"]) for e in calls[i][0]}
        for i in (0, config.n_seeds, config.n_seeds + 1)
    ]
    assert len(ids[0] | ids[1] | ids[2]) == sum(len(i) for i in ids)


def test_variance_reads_the_token_rows_of_the_response_pairs(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trained: list[Prepared] = []
    real = r2_utility.sweep_from_activations

    def spy(train: Sequence[Prepared], evals: Any, **kwargs: Any) -> Any:
        trained.extend(train)
        return real(train, evals, **kwargs)

    monkeypatch.setattr(r2_utility, "sweep_from_activations", spy)
    config = _config(n_seeds=2, position="response", preprocess="unit")
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _examples(words=SAME_LENGTH))
    per_seed = run_dir.read_arrays("variance")["per_seed"]
    assert len(trained) == config.n_seeds
    for i, (stimuli, act) in enumerate(trained):
        expected = pca_explained_variance(
            act.data,
            pairs=act.row_groups(stimuli.pairs),
            n_components=3,
            preprocess="unit",
        )[:, :3]
        np.testing.assert_allclose(per_seed[:, i], expected.T)


def test_execute_reports_the_rows_of_each_eval_split(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config(n_eval_pairs=None, n_seeds=2)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(
        config,
        run_dir,
        tiny_arch,
        _examples({"train": 12, "val": 5, "test": 7}),
    )
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    assert summary["n_eval_pairs"] == {"val": 5, "test": 7}


def test_too_few_components_fail_before_the_model_runs(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the model must not run")

    monkeypatch.setattr(r2_utility, "stimulus_activations", forbidden)
    config = _config(n_components=2)
    run_dir = RunDir.create(config, out_root=tmp_path)
    with pytest.raises(ValueError, match="n_components"):
        execute(config, run_dir, tiny_arch, _examples())


def test_execute_is_deterministic(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config(n_seeds=2)
    first = RunDir.create(config, out_root=tmp_path / "a")
    second = RunDir.create(config, out_root=tmp_path / "b")
    execute(config, first, tiny_arch, _examples())
    execute(config, second, tiny_arch, _examples())
    for stem in ("curves", "curves_val", "variance", "readers"):
        a, b = first.read_arrays(stem), second.read_arrays(stem)
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])


def test_the_default_direction_is_logistic_and_pca_diff_stays_selectable(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config(n_seeds=2)
    pca = dataclasses.replace(config, method="pca_diff")
    default_dir = RunDir.create(config, out_root=tmp_path / "default")
    pca_dir = RunDir.create(pca, out_root=tmp_path / "pca")
    execute(config, default_dir, tiny_arch, _examples())
    execute(pca, pca_dir, tiny_arch, _examples())
    default = default_dir.read_arrays("readers")["directions"]
    chosen = pca_dir.read_arrays("readers")["directions"]
    assert default.shape == chosen.shape
    assert not np.allclose(default, chosen)


def test_plot_writes_three_figures_in_three_formats(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config(n_seeds=2)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _examples())
    paths = plot(run_dir, tmp_path / "figs")
    assert sorted(p.name for p in paths) == sorted(
        f"{stem}.{ext}"
        for stem in ("curves", "curves_val", "variance")
        for ext in ("pdf", "png", "svg")
    )
    assert all(p.stat().st_size > 0 for p in paths)


def test_run_loads_the_model_and_the_data(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[str] = []

    def fake_load_model(key: str) -> LoadedModel:
        asked.append(key)
        return tiny_arch

    monkeypatch.setattr(r2_utility, "load_model", fake_load_model)
    monkeypatch.setattr(r2_utility, "load_ethics_utility", _examples)
    config = _config(n_seeds=2)
    run_dir = RunDir.create(config, out_root=tmp_path)
    run(config, run_dir)
    assert asked == ["tiny"]
    assert (run_dir.path / "summary.json").is_file()
    assert run_dir.meta["finished_at"] is None


@pytest.mark.slow
def test_r2_on_gpt2_reads_utility_above_the_floor(tmp_path: Path) -> None:
    config = Config(n_seeds=3, n_pairs=128, n_eval_pairs=128)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(
        config,
        run_dir,
        load_model("gpt2"),
        load_ethics_utility(offline=True),
    )
    curves = run_dir.read_arrays("curves")
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    for name, row in zip(SERIES, curves["mean"], strict=True):
        print(name, np.round(row, 3).tolist())
    print(summary)
    assert summary["test_accuracy"] > 0.55
    assert (run_dir.path / "variance.npz").is_file()
