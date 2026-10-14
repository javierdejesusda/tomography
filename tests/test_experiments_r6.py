"""Tests for the R6 TruthfulQA experiment."""

import dataclasses
from pathlib import Path
from typing import Any
from typing import cast

import numpy as np
import pytest
import torch

from tomography.cli import load_experiment
from tomography.datasets.arc import load_arc
from tomography.datasets.facts import load_facts
from tomography.datasets.records import ChoiceExample
from tomography.datasets.records import Example
from tomography.datasets.records import Split
from tomography.datasets.truthfulqa import load_truthfulqa
from tomography.experiments import r6_truthfulqa
from tomography.experiments.base import RunDir
from tomography.experiments.r6_truthfulqa import HEURISTIC_LEVELS
from tomography.experiments.r6_truthfulqa import LAYER_RULES
from tomography.experiments.r6_truthfulqa import Config
from tomography.experiments.r6_truthfulqa import best_layer_accuracy
from tomography.experiments.r6_truthfulqa import execute
from tomography.experiments.r6_truthfulqa import heuristic_scores
from tomography.experiments.r6_truthfulqa import plot
from tomography.experiments.r6_truthfulqa import run
from tomography.experiments.r6_truthfulqa import standard_scores
from tomography.experiments.r6_truthfulqa import validation_layer_accuracy
from tomography.experiments.sweep import SERIES
from tomography.hooks import edit_residuals
from tomography.judges import heuristic_score
from tomography.judges import sequence_logprob
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.reading.metrics import wilson_ci

SHORT_TEMPLATE = "T {scenario} A:"


def _items() -> list[ChoiceExample]:
    return [
        ChoiceExample(
            uid="a", question="q", choices=("x", "yy"), correct=0, split="val"
        ),
        ChoiceExample(
            uid="b",
            question="r",
            choices=("z", "w", "vv"),
            correct=2,
            split="val",
        ),
    ]


def _bos(loaded: LoadedModel) -> int | None:
    return loaded.bos_id if loaded.default_bos == "prepend" else None


def test_standard_scores_sum_the_answer_logprob_after_the_user_prompt(
    tiny_arch: LoadedModel,
) -> None:
    items = _items()
    got = standard_scores(tiny_arch, items, batch_size=4)
    assert [g.shape for g in got] == [(2,), (3,)]
    assert all(g.dtype == np.float32 for g in got)
    for item, scores in zip(items, got, strict=True):
        expected = sequence_logprob(
            tiny_arch.model,
            tiny_arch.tokenizer,
            [f"USER: {item.question} ASSISTANT:"] * len(item.choices),
            [" " + choice for choice in item.choices],
            batch_size=4,
            bos_id=_bos(tiny_arch),
        )
        np.testing.assert_allclose(scores, expected, rtol=1e-4, atol=1e-4)


def test_standard_scores_are_deterministic(tiny_arch: LoadedModel) -> None:
    first = standard_scores(tiny_arch, _items())
    second = standard_scores(tiny_arch, _items())
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


def test_standard_scores_run_under_the_callers_hooks(
    tiny_arch: LoadedModel,
) -> None:
    plain = standard_scores(tiny_arch, _items())

    def shift(hidden: torch.Tensor, info: object) -> torch.Tensor:
        return hidden + 3.0

    with edit_residuals(tiny_arch, {1: shift}):
        edited = standard_scores(tiny_arch, _items())
    assert not np.array_equal(np.concatenate(plain), np.concatenate(edited))


def test_the_heuristic_levels_run_from_low_to_high() -> None:
    assert HEURISTIC_LEVELS == (
        "Very low",
        "Low",
        "Somewhat low",
        "Neutral",
        "Somewhat high",
        "High",
        "Very high",
    )


def test_the_heuristic_template_has_one_scenario_slot() -> None:
    template = r6_truthfulqa.HEURISTIC_TEMPLATE
    assert template.count("{scenario}") == 1
    assert template.startswith(
        "USER: What is the amount of truthfulness of the following scenario?"
    )
    assert "'Very low', 'Low'" in template
    assert template.endswith("Assistant: Truthfulness:")


def test_heuristic_scores_map_the_seven_level_logprobs(
    tiny_arch: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(r6_truthfulqa, "HEURISTIC_TEMPLATE", SHORT_TEMPLATE)
    items = _items()
    got = heuristic_scores(tiny_arch, items, batch_size=8)
    assert [g.shape for g in got] == [(2,), (3,)]
    for item, scores in zip(items, got, strict=True):
        assert scores.dtype == np.float64
        assert ((scores >= -1.0) & (scores <= 1.0)).all()
        for choice, score in zip(item.choices, scores, strict=True):
            scenario = f"Question: {item.question}\nAnswer: {choice}"
            prompt = SHORT_TEMPLATE.replace("{scenario}", scenario)
            logprobs = sequence_logprob(
                tiny_arch.model,
                tiny_arch.tokenizer,
                [prompt] * 7,
                [" " + level for level in HEURISTIC_LEVELS],
                batch_size=8,
                bos_id=_bos(tiny_arch),
            )
            assert score == pytest.approx(
                float(heuristic_score(logprobs)), abs=1e-4
            )


def test_heuristic_scores_accept_a_template_argument_and_braces(
    tiny_arch: LoadedModel,
) -> None:
    item = ChoiceExample(
        uid="c", question="{q}", choices=("x", "y"), correct=1, split="val"
    )
    got = heuristic_scores(tiny_arch, [item], template=SHORT_TEMPLATE)
    assert got[0].shape == (2,)


def test_heuristic_scores_are_deterministic(
    tiny_arch: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(r6_truthfulqa, "HEURISTIC_TEMPLATE", SHORT_TEMPLATE)
    first = heuristic_scores(tiny_arch, _items())
    second = heuristic_scores(tiny_arch, _items())
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


VAL = np.array(
    [
        [0.1, 0.5, 0.5, 0.2],
        [0.9, 0.3, 0.9, 0.1],
        [0.2, 0.3, 0.4, 0.8],
    ]
)
TEST = np.array(
    [
        [0.6, 0.7, 0.9, 0.1],
        [0.5, 0.8, 0.2, 0.3],
        [0.4, 0.3, 0.6, 0.9],
    ]
)
LAYERS = (4, 6, 8, 10)


def test_the_layer_rules_are_named_in_table_order() -> None:
    assert LAYER_RULES == ("validation-layer", "best-layer")


def test_validation_layer_reports_the_test_accuracy_at_each_trials_layer() -> (
    None
):
    accuracy, chosen = validation_layer_accuracy(VAL, TEST, LAYERS)
    assert chosen.tolist() == [6, 4, 10]
    np.testing.assert_allclose(accuracy, [0.7, 0.5, 0.9])


def test_best_layer_uses_one_layer_for_every_trial() -> None:
    accuracy, layer = best_layer_accuracy(TEST, LAYERS)
    assert layer == 6
    np.testing.assert_allclose(accuracy, [0.7, 0.8, 0.3])


def test_best_layer_breaks_ties_to_the_first_layer() -> None:
    tied = np.array([[0.5, 0.75, 0.75], [0.5, 0.25, 0.25]])
    accuracy, layer = best_layer_accuracy(tied, (3, 5, 7))
    assert layer == 3
    np.testing.assert_allclose(accuracy, [0.5, 0.5])


@pytest.mark.parametrize(
    ("val", "test"),
    [(VAL[:2], TEST), (VAL[:, :3], TEST), (VAL[0], TEST[0])],
    ids=["trials", "layers", "ndim"],
)
def test_validation_layer_rejects_mismatched_cubes(
    val: np.ndarray, test: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="shape"):
        validation_layer_accuracy(val, test, LAYERS)


def test_best_layer_rejects_a_layer_count_mismatch() -> None:
    with pytest.raises(ValueError, match="layers"):
        best_layer_accuracy(TEST, (1, 2, 3))


def test_config_defaults_and_round_trip() -> None:
    config = Config()
    assert config.experiment == "r6"
    assert config.model == "gpt2"
    assert config.n_seeds == 15
    assert config.sources == ("truthfulqa", "arc", "honesty")
    assert Config.from_dict(config.to_dict()) == config


def test_the_cli_discovers_the_experiment() -> None:
    experiment = load_experiment("r6")
    assert experiment.config_cls is Config
    assert experiment.run is run
    assert experiment.plot is plot


def _choice_items(prefix: str, counts: dict[Split, int]) -> list[ChoiceExample]:
    items = []
    for split, count in counts.items():
        for i in range(count):
            k = 2 + i % 3
            items.append(
                ChoiceExample(
                    uid=f"{prefix}{split[0]}{i}",
                    question=f"q{i}",
                    choices=tuple(f"a{j}" for j in range(k)),
                    correct=i % k,
                    split=split,
                )
            )
    return items


def _questions() -> list[ChoiceExample]:
    return _choice_items("t", {"train": 14, "val": 8, "test": 8})


def _arc() -> list[ChoiceExample]:
    return _choice_items("c", {"train": 10, "val": 3, "test": 3})


def _facts() -> list[Example]:
    facts = []
    for split, count in (("train", 24), ("val", 4), ("test", 4)):
        for i in range(count):
            facts.append(
                Example(
                    uid=f"f{split[0]}{i}",
                    text=f"f{i}",
                    label=i % 2,
                    split=cast(Split, split),
                )
            )
    return facts


def _config(**overrides: Any) -> Config:
    base = Config(model="tiny", n_seeds=2, n_pairs=4, batch_size=16)
    return dataclasses.replace(base, **overrides)


def _run_execute(
    config: Config,
    loaded: LoadedModel,
    out_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> RunDir:
    monkeypatch.setattr(r6_truthfulqa, "HEURISTIC_TEMPLATE", SHORT_TEMPLATE)
    run_dir = RunDir.create(config, out_root=out_root)
    execute(config, run_dir, loaded, _questions(), _arc(), _facts())
    return run_dir


def test_execute_writes_the_table_curves_and_summary(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    run_dir = _run_execute(config, tiny_arch, tmp_path, monkeypatch)
    n_layers = tiny_arch.n_layers

    table = run_dir.read_json("table")
    assert isinstance(table, dict)
    assert table["columns"] == [
        "method",
        "layer_rule",
        "source",
        "mc1",
        "lo",
        "hi",
        "n_trials",
    ]
    rows = table["rows"]
    assert [row[:3] for row in rows] == [
        ["standard", None, None],
        ["heuristic", None, None],
        ["lat", "validation-layer", "truthfulqa"],
        ["lat", "best-layer", "truthfulqa"],
        ["lat", "validation-layer", "arc"],
        ["lat", "best-layer", "arc"],
        ["lat", "validation-layer", "honesty"],
        ["lat", "best-layer", "honesty"],
    ]
    for row in rows[:2]:
        mc1, lo, hi, n_trials = row[3:]
        assert 0.0 <= lo <= mc1 <= hi <= 1.0
        assert n_trials is None
        assert mc1 * 8 == pytest.approx(round(mc1 * 8))
        assert (lo, hi) == wilson_ci(round(mc1 * 8), 8)
    for row in rows[2:]:
        mc1, lo, hi, n_trials = row[3:]
        assert 0.0 <= lo <= hi <= 1.0
        assert 0.0 <= mc1 <= 1.0
        assert n_trials == 2

    for source in ("truthfulqa", "arc", "honesty"):
        curves = run_dir.read_arrays(f"curves_{source}")
        assert curves["mean"].shape == (3, n_layers)
        assert curves["per_seed"].shape == (3, 2, n_layers)
        sidecar = run_dir.read_json(f"curves_{source}")
        assert isinstance(sidecar, dict)
        assert sidecar["series"] == list(SERIES)
        assert sidecar["n_layers"] == n_layers
        expected_chance = np.mean([1 / (2 + i % 3) for i in range(8)])
        assert sidecar["chance"] == pytest.approx(expected_chance)

    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    assert set(summary) == {
        "n_questions",
        "n_train",
        "n_val",
        "n_test",
        "n_seeds",
        "baselines_all_questions",
        "lat",
    }
    assert summary["n_questions"] == 30
    assert (summary["n_train"], summary["n_val"], summary["n_test"]) == (
        14,
        8,
        8,
    )
    assert summary["n_seeds"] == 2
    assert set(summary["baselines_all_questions"]) == {"standard", "heuristic"}
    for entry in summary["baselines_all_questions"].values():
        assert set(entry) == {"mc1", "lo", "hi"}
        assert entry["lo"] <= entry["mc1"] <= entry["hi"]
    assert set(summary["lat"]) == {"truthfulqa", "arc", "honesty"}
    for entry in summary["lat"].values():
        assert set(entry) == {"validation-layer", "best-layer"}
        assert set(entry["validation-layer"]) == {"mc1", "lo", "hi"}
        assert set(entry["best-layer"]) == {"mc1", "lo", "hi", "layer"}


def test_execute_best_layer_row_matches_the_saved_test_curves(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _run_execute(_config(), tiny_arch, tmp_path, monkeypatch)
    table = run_dir.read_json("table")
    assert isinstance(table, dict)
    curves = run_dir.read_arrays("curves_truthfulqa")
    reader = curves["per_seed"][SERIES.index("reader")]
    layers = curves["x"].astype(int).tolist()
    accuracy, layer = best_layer_accuracy(reader, layers)
    row = next(
        r for r in table["rows"] if r[:3] == ["lat", "best-layer", "truthfulqa"]
    )
    assert row[3] == pytest.approx(float(accuracy.mean()))
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    assert summary["lat"]["truthfulqa"]["best-layer"]["layer"] == layer


def test_execute_is_deterministic(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    first = _run_execute(config, tiny_gpt2, tmp_path / "a", monkeypatch)
    second = _run_execute(config, tiny_gpt2, tmp_path / "b", monkeypatch)
    assert first.read_json("table") == second.read_json("table")
    assert first.read_json("summary") == second.read_json("summary")
    a, b = first.read_arrays("curves_arc"), second.read_arrays("curves_arc")
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_execute_runs_only_the_requested_sources(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(sources=("truthfulqa",))
    run_dir = _run_execute(config, tiny_gpt2, tmp_path, monkeypatch)
    table = run_dir.read_json("table")
    assert isinstance(table, dict)
    assert len(table["rows"]) == 4
    assert sorted(p.name for p in run_dir.path.glob("curves_*")) == [
        "curves_truthfulqa.json",
        "curves_truthfulqa.npz",
    ]
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    assert set(summary["lat"]) == {"truthfulqa"}


@pytest.mark.parametrize(
    "sources",
    [("nope",), ("truthfulqa", "nope")],
    ids=["only-unknown", "after-known"],
)
def test_execute_rejects_an_unknown_source_before_touching_the_model(
    tmp_path: Path, sources: tuple[str, ...]
) -> None:
    config = _config(sources=sources)
    run_dir = RunDir.create(config, out_root=tmp_path)
    with pytest.raises(ValueError, match="nope"):
        execute(
            config,
            run_dir,
            cast(LoadedModel, None),
            _questions(),
            _arc(),
            _facts(),
        )
    assert not list(run_dir.path.glob("*.npz"))


def test_plot_writes_three_formats_per_source(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _run_execute(_config(), tiny_gpt2, tmp_path, monkeypatch)
    paths = plot(run_dir, tmp_path / "figs")
    assert sorted(p.name for p in paths) == [
        f"curves_{source}.{ext}"
        for source in ("arc", "honesty", "truthfulqa")
        for ext in ("pdf", "png", "svg")
    ]
    assert all(p.stat().st_size > 0 for p in paths)


def test_run_loads_the_model_and_the_data(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[str] = []

    def fake_load_model(key: str) -> LoadedModel:
        asked.append(key)
        return tiny_gpt2

    monkeypatch.setattr(r6_truthfulqa, "load_model", fake_load_model)
    monkeypatch.setattr(r6_truthfulqa, "load_truthfulqa", _questions)
    monkeypatch.setattr(r6_truthfulqa, "load_arc", _arc)
    monkeypatch.setattr(r6_truthfulqa, "load_facts", _facts)
    monkeypatch.setattr(r6_truthfulqa, "HEURISTIC_TEMPLATE", SHORT_TEMPLATE)
    config = _config(sources=("truthfulqa",))
    run_dir = RunDir.create(config, out_root=tmp_path)
    run(config, run_dir)
    assert asked == ["tiny"]
    assert (run_dir.path / "summary.json").is_file()


@pytest.mark.slow
def test_r6_on_gpt2_runs_end_to_end(tmp_path: Path) -> None:
    config = Config(n_seeds=2, n_pairs=32, n_eval_pairs=100)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(
        config,
        run_dir,
        load_model("gpt2"),
        load_truthfulqa(offline=True),
        load_arc(offline=True),
        load_facts(offline=True),
    )
    table = run_dir.read_json("table")
    summary = run_dir.read_json("summary")
    assert isinstance(table, dict)
    assert isinstance(summary, dict)
    for row in table["rows"]:
        print(row)
    assert len(table["rows"]) == 8
    assert all(0.0 <= row[3] <= 1.0 for row in table["rows"])
    assert 0.10 < summary["baselines_all_questions"]["standard"]["mc1"] < 0.40
