"""Tests for the R5 honesty reading experiment."""

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tomography.cli import load_experiment
from tomography.datasets.facts import load_facts
from tomography.datasets.records import Example
from tomography.datasets.records import Split
from tomography.experiments import r5_honesty
from tomography.experiments.base import RunDir
from tomography.experiments.r5_honesty import Config
from tomography.experiments.r5_honesty import execute
from tomography.experiments.r5_honesty import plot
from tomography.experiments.r5_honesty import run
from tomography.experiments.sweep import SERIES
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.templates import PAPER_TEMPLATES

PERSONAS = ("honest", "untruthful")


def _statements(
    count: int = 12, splits: tuple[Split, ...] = ("train", "val", "test")
) -> list[Example]:
    return [
        Example(
            uid=f"fact-{split}-{r}",
            text=f"The fact number {r} is " + "plain and long " * (r % 4 + 1),
            label=1,
            split=split,
        )
        for split in splits
        for r in range(count)
    ]


def _config(**overrides: Any) -> Config:
    base = Config(
        model="tiny",
        n_seeds=2,
        n_pairs=6,
        n_eval_pairs=4,
        n_scan_statements=2,
    )
    return dataclasses.replace(base, **overrides)


def test_config_defaults_and_round_trip() -> None:
    config = Config()
    assert config.experiment == "r5"
    assert config.model == "gpt2"
    assert config.position == "response"
    assert config.n_scan_statements == 4
    assert Config.from_dict(config.to_dict()) == config


def test_the_cli_discovers_the_experiment() -> None:
    experiment = load_experiment("r5")
    assert experiment.config_cls is Config
    assert experiment.run is run
    assert experiment.plot is plot


def _scan(run_dir: RunDir, i: int, persona: str) -> tuple[Any, list[str], Any]:
    stem = f"scan_{i}_{persona}"
    arrays = run_dir.read_arrays(stem)
    sidecar = run_dir.read_json(stem)
    assert isinstance(sidecar, dict)
    assert sidecar["mask_first"] is False
    return arrays, sidecar["tokens"], sidecar


def test_execute_writes_curves_scans_and_summary(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config()
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _statements())
    n_layers = tiny_arch.n_layers
    layers = list(range(1, n_layers + 1))

    curves = run_dir.read_arrays("curves")
    assert curves["x"].tolist() == layers
    assert curves["mean"].shape == (3, n_layers)
    assert curves["per_seed"].shape == (3, 2, n_layers)
    sidecar = run_dir.read_json("curves")
    assert isinstance(sidecar, dict)
    assert sidecar["series"] == list(SERIES)
    assert sidecar["chance"] == 0.5

    for i in range(2):
        tokens = {}
        for persona in PERSONAS:
            arrays, tokens[persona], _ = _scan(run_dir, i, persona)
            assert arrays["layers"].dtype == np.int64
            assert arrays["layers"].tolist() == layers
            scores = arrays["scores"]
            assert scores.shape == (n_layers, len(tokens[persona]))
            assert np.isfinite(scores).all()
            assert all(isinstance(t, str) and t for t in tokens[persona])
        assert tokens["honest"] == tokens["untruthful"]
    assert not (run_dir.path / "scan_2_honest.npz").exists()

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
        "n_scan",
    }
    assert summary["best_layer"] in layers
    at = layers.index(summary["best_layer"])
    assert summary["test_accuracy"] == pytest.approx(curves["mean"][0, at])
    assert summary["ci"] == pytest.approx(
        [curves["lo"][0, at], curves["hi"][0, at]]
    )
    assert summary["n_seeds"] == 2
    assert summary["n_pairs"] == 6
    assert summary["n_eval_pairs"] == {"val": 4, "test": 4}
    assert summary["n_scan"] == 2


def test_persona_pairing_and_splits_before_pairing(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[Example], dict[str, Any]]] = []
    real = r5_honesty.build_stimulus_set

    def spy(examples: Any, template: Any, chat: Any, **kwargs: Any) -> Any:
        calls.append((list(examples), {"template": template, **kwargs}))
        return real(examples, template, chat, **kwargs)

    positions: list[str] = []
    real_activations = r5_honesty.stimulus_activations

    def spy_activations(*args: Any, **kwargs: Any) -> Any:
        positions.append(kwargs["position"])
        return real_activations(*args, **kwargs)

    monkeypatch.setattr(r5_honesty, "build_stimulus_set", spy)
    monkeypatch.setattr(r5_honesty, "stimulus_activations", spy_activations)
    config = _config()
    statements = _statements()
    execute(
        config,
        RunDir.create(config, out_root=tmp_path),
        tiny_arch,
        statements,
    )
    n = config.n_seeds
    assert len(calls) == n + 3
    for _, kwargs in calls:
        assert kwargs["pairing"] == "persona"
        assert kwargs["template"] is PAPER_TEMPLATES["honesty"]
    splits = [{e.split for e in examples} for examples, _ in calls]
    assert splits == [{"train"}] * n + [{"val"}, {"test"}, {"test"}]
    uids = [{e.uid for e in examples} for examples, _ in calls]
    assert uids[0] | uids[n] | uids[n + 1] == {e.uid for e in statements}
    assert not uids[0] & uids[n]
    assert not uids[n] & uids[n + 1]
    assert not uids[0] & uids[n + 1]
    test = [e for e in statements if e.split == "test"]
    assert [e.uid for e in calls[n + 2][0]] == [e.uid for e in test[:2]]
    assert calls[n + 2][1]["n_pairs"] is None
    assert positions[-1] == "response"


def test_scans_follow_the_input_order_of_the_test_statements(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    statements = _statements()
    test = [e for e in statements if e.split == "test"]
    shuffled = [e for e in statements if e.split != "test"] + test[::-1]
    config = _config()
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, shuffled)
    for i, statement in enumerate(test[::-1][:2]):
        for persona in PERSONAS:
            _, tokens, _ = _scan(run_dir, i, persona)
            assert "".join(tokens).strip() == statement.text.strip()


def test_too_many_scan_statements_fail_before_the_model_runs(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the model must not run")

    monkeypatch.setattr(r5_honesty, "reading_sweep", forbidden)
    monkeypatch.setattr(r5_honesty, "stimulus_activations", forbidden)
    config = _config(n_scan_statements=13)
    run_dir = RunDir.create(config, out_root=tmp_path)
    with pytest.raises(ValueError, match="n_scan_statements"):
        execute(config, run_dir, tiny_arch, _statements())


def test_execute_is_deterministic(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config()
    first = RunDir.create(config, out_root=tmp_path / "a")
    second = RunDir.create(config, out_root=tmp_path / "b")
    execute(config, first, tiny_arch, _statements())
    execute(config, second, tiny_arch, _statements())
    stems = ["curves", *(f"scan_{i}_{p}" for i in range(2) for p in PERSONAS)]
    for stem in stems:
        a, b = first.read_arrays(stem), second.read_arrays(stem)
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])


def test_plot_writes_the_curves_and_scan_figures(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config()
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _statements())
    paths = plot(run_dir, tmp_path / "figs")
    stems = ["curves", *(f"scan_{i}_{p}" for i in range(2) for p in PERSONAS)]
    assert sorted(p.name for p in paths) == sorted(
        f"{stem}.{ext}" for stem in stems for ext in ("pdf", "png", "svg")
    )
    assert all(p.stat().st_size > 0 for p in paths)
    assert sorted(p.name for p in (tmp_path / "figs").iterdir()) == sorted(
        p.name for p in paths
    )


def test_run_keeps_only_the_true_statements(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mixed = [
        dataclasses.replace(e, label=r % 2)
        for r, e in enumerate(_statements(count=4))
    ]
    seen: list[list[Example]] = []

    def record(
        config: Config,
        run_dir: RunDir,
        loaded: LoadedModel,
        statements: Any,
    ) -> None:
        seen.append(list(statements))

    monkeypatch.setattr(r5_honesty, "load_facts", lambda: mixed)
    monkeypatch.setattr(r5_honesty, "load_model", lambda key: tiny_arch)
    monkeypatch.setattr(r5_honesty, "execute", record)
    config = _config()
    run(config, RunDir.create(config, out_root=tmp_path))
    assert len(seen) == 1
    assert seen[0]
    assert {e.label for e in seen[0]} == {1}
    assert [e.uid for e in seen[0]] == [e.uid for e in mixed if e.label == 1]


@pytest.mark.slow
def test_r5_on_gpt2_reads_honesty_above_the_floor(tmp_path: Path) -> None:
    config = Config(n_seeds=3, n_pairs=64, n_eval_pairs=64)
    run_dir = RunDir.create(config, out_root=tmp_path)
    statements = [e for e in load_facts(offline=True) if e.label == 1]
    execute(config, run_dir, load_model("gpt2"), statements)
    curves = run_dir.read_arrays("curves")
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    for name, row in zip(SERIES, curves["mean"], strict=True):
        print(name, np.round(row, 3).tolist())
    print(summary)
    assert summary["test_accuracy"] > 0.6
    assert (run_dir.path / "scan_0_honest.npz").is_file()
