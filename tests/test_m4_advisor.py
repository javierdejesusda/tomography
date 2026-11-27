"""CPU tests for the layer and cross-task reader experiment."""

from collections.abc import Sequence
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tomography.activations import Activations
from tomography.activations import PositionStrategy
from tomography.datasets.records import Example
from tomography.datasets.stimulus import StimulusSet
from tomography.experiments import m2_readers as m2
from tomography.experiments import m4_advisor as experiment
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.model import LoadedModel


def _write_baseline(
    root: Path,
    config: m2.Config,
    rows: list[list[Any]],
    *,
    legacy: bool = False,
) -> Path:
    payload = config.to_dict()
    if legacy:
        del payload["reader_series"]
    compact = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(compact.encode()).hexdigest()[:12]
    path = root / config.model / digest
    path.mkdir(parents=True)
    (path / "config.json").write_text(compact, encoding="utf-8")
    (path / "table.json").write_text(
        json.dumps({"columns": m2.COLUMNS, "rows": rows}), encoding="utf-8"
    )
    return path


def _gate_rows() -> list[list[Any]]:
    return [
        [
            "sentiment",
            "original",
            scorer,
            "last",
            rule,
            "reader",
            [[1]],
            *([0.5] * 11),
            1,
        ]
        for scorer in ("logistic", "mean_diff")
        for rule in ("best", "top_k")
    ]


@pytest.mark.parametrize("legacy", [False, True])
def test_gate_resolves_m2_configuration_then_checks_rows(
    tmp_path: Path, legacy: bool
) -> None:
    config = experiment.Config(model="tiny")
    baseline = config.m2_config()
    rows = _gate_rows()
    expected = _write_baseline(tmp_path, baseline, rows, legacy=legacy)
    _write_baseline(
        tmp_path, dataclasses.replace(baseline, reader_series=False), []
    )
    assert experiment.find_m2_run(config, tmp_path) == expected
    experiment.check_m2_equality(rows, rows)
    changed = [list(row) for row in rows]
    changed[0][m2.COLUMNS.index("auroc")] += 2e-6
    with pytest.raises(ValueError, match="auroc"):
        experiment.check_m2_equality(changed, rows)
    with pytest.raises(ValueError, match="missing"):
        experiment.check_m2_equality(rows[:1], rows)


def test_gate_rejects_missing_or_ambiguous_m2_configuration(
    tmp_path: Path,
) -> None:
    config = experiment.Config(model="tiny")
    with pytest.raises(ValueError, match="no matching M2"):
        experiment.find_m2_run(config, tmp_path)
    _write_baseline(tmp_path, config.m2_config(), [])
    _write_baseline(tmp_path, config.m2_config(), [], legacy=True)
    with pytest.raises(ValueError, match="multiple matching M2"):
        experiment.find_m2_run(config, tmp_path)


def test_lr_tuning_returns_scores_and_strongest_tied_penalty() -> None:
    data = np.array([[[-2.0], [2.0], [-1.0], [1.0]]], dtype=np.float32)
    labels = np.array([0, 1, 0, 1])
    groups = np.array([[0, 1], [2, 3]])
    rows = experiment._LabeledRows(data, (1,), labels, groups, groups)
    result = experiment._tuned_lr(
        experiment.Config(logistic_c=(0.01, 0.1, 1.0)), rows, [rows], seed=0
    )[0]
    assert result.penalties == (0.01,)
    np.testing.assert_array_equal(result.select_accuracy, [1.0])
    np.testing.assert_allclose(result.select_scores, [[-2, 2, -1, 1]])
    np.testing.assert_allclose(result.reader.score(data), result.select_scores)


def _sources() -> dict[str, list[Example]]:
    sentiment = [
        Example(
            f"review-{split}-{label}-{i}", f"a long review {i}", label, split
        )
        for split in ("train", "val")
        for label in (0, 1)
        for i in range(24)
    ]
    utility = [
        Example(
            f"{pair}:{label}",
            f"scenario {i} option {label}",
            label,
            split,
            {"pair": pair},
        )
        for split in ("train", "val", "test")
        for i in range(8)
        for pair in (f"pair-{split}-{i}",)
        for label in (0, 1)
    ]
    return {"sst2": sentiment, "ethics_utility": utility}


def _fake_positions(
    loaded: LoadedModel,
    stimuli: StimulusSet,
    positions: Sequence[PositionStrategy],
    config: m2.Config,
) -> dict[PositionStrategy, Activations]:
    del loaded
    layers = config.layers
    assert layers is not None
    n_rows = len(stimuli.labels)
    data = {}
    for position in positions:
        rows = []
        for label, uid in zip(stimuli.labels, stimuli.uids, strict=True):
            seed = derive_seed(config.seed, uid, position)
            row = np.random.default_rng(seed).normal(size=(len(layers), 3))
            row[:, 0] += 3 * label
            rows.append(row)
        data[position] = Activations(
            data=np.stack(rows, axis=1).astype(np.float32),
            layers=layers,
            position=position,
            row_stimulus=np.arange(n_rows),
            row_rank=np.zeros(n_rows, dtype=int),
            row_token=np.zeros(n_rows, dtype=int),
            row_token_id=np.zeros(n_rows, dtype=int),
            n_stimuli=n_rows,
        )
    return data


@pytest.fixture
def setup_run(
    tmp_path: Path, tiny_gpt2: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> tuple[experiment.Config, RunDir, RunDir]:
    monkeypatch.setattr(m2, "read_positions", _fake_positions)
    monkeypatch.setattr(experiment, "load_model", lambda *a, **kw: tiny_gpt2)
    monkeypatch.setattr(experiment, "load_sources", lambda sources: _sources())
    config = experiment.Config(
        model="tiny",
        tasks=("sentiment", "utility"),
        layers=(1, 4, 8),
        token_layers=None,
        readouts=("last", "max", "mean"),
        scorers=("logistic", "mean_diff"),
        layer_rules=("best", "top_k"),
        controls=("random", "text", "shuffled"),
        n_seeds=2,
        n_pairs=4,
        n_eval_pairs=3,
        top_k=2,
        n_boot=5,
        n_folds=2,
        n_trees=3,
        matched_pairs_per_task=2,
        logistic_c=(0.1, 1.0),
        mlp_alpha=(0.1,),
        shifts=(),
        m2_root=str(tmp_path / "m2"),
    )
    baseline = RunDir.create(config.m2_config(), out_root=tmp_path)
    m2.execute(config.m2_config(), baseline, tiny_gpt2, _sources())
    return config, baseline, RunDir.create(config, out_root=tmp_path)


def _table(run_dir: RunDir, name: str = "table") -> list[dict[str, Any]]:
    payload = run_dir.read_json(name)
    return [
        dict(zip(payload["columns"], row, strict=True))
        for row in payload["rows"]
    ]


def test_run_gates_before_writing_experimental_rows(
    setup_run: tuple[experiment.Config, RunDir, RunDir],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, baseline, run_dir = setup_run
    stored = baseline.read_json("table")
    index = stored["columns"].index("auroc")
    row = next(
        row
        for row in stored["rows"]
        if row[2:6] == ["logistic", "last", "best", "reader"]
    )
    row[index] += 2e-6
    baseline.write_json("table", stored)

    def early_analysis(*args: object) -> None:
        pytest.fail("draw effects were analyzed before the equality gate")

    monkeypatch.setattr(experiment, "_draw_delta", early_analysis)
    with pytest.raises(ValueError, match="M2 equality failed"):
        experiment.run(config, run_dir)
    assert not (run_dir.path / "table.json").exists()
    assert not (run_dir.path / "comparisons.json").exists()


def test_run_writes_selected_baselines_ties_and_stored_controls(
    setup_run: tuple[experiment.Config, RunDir, RunDir],
) -> None:
    config, baseline, run_dir = setup_run
    experiment.run(config, run_dir)
    rows = _table(run_dir)
    for row in _table(baseline):
        if row["scorer"] in ("text", "random"):
            assert row in rows
    deep = [row for row in rows if row["layer_rule"] == "best_deep"]
    assert {row["readout"] for row in deep} == {"last", "max", "mean"}
    assert {row["scorer"] for row in deep} == {"logistic", "mean_diff"}
    ties = run_dir.read_json("layer_ties")
    assert len(ties) == 12
    assert all(
        0 <= row["different_share"] <= row["tie_share"] <= 1 for row in ties
    )
    assert all(len(row["draw_ties"]) == 2 for row in ties)


def test_run_reader_protocol_at_each_readout(
    setup_run: tuple[experiment.Config, RunDir, RunDir],
) -> None:
    config, _, run_dir = setup_run
    experiment.run(config, run_dir)
    table = _table(run_dir)
    shuffled = {
        "equal_mean",
        "weighted_mean",
        "stack_lr",
        "stack_mlp",
        "stack_forest",
        "vector_forest",
    }
    originals = {"pca16_forest", "pca64_forest", "concat_lr", "bridge_forest"}
    for scorer in shuffled | originals:
        rows = [row for row in table if row["scorer"] == scorer]
        expected = {"reader", "shuffled"} if scorer in shuffled else {"reader"}
        assert {row["readout"] for row in rows} == {"last", "max", "mean"}, (
            scorer
        )
        assert {row["series"] for row in rows} == expected, scorer
        if scorer in ("vector_forest", "pca16_forest", "pca64_forest"):
            assert {row["layer_rule"] for row in rows} == {"default", "tuned"}
        assert all(0 <= row["auroc"] <= 1 for row in rows), scorer


def test_run_reports_mlp_seed_spread_in_a_two_draw_design(
    setup_run: tuple[experiment.Config, RunDir, RunDir],
) -> None:
    config, _, run_dir = setup_run
    experiment.run(dataclasses.replace(config, pilot_draws=1), run_dir)
    rows = run_dir.read_json("fit_diagnostics")["mlp_pilot"]
    assert rows
    assert {row["draw"] for row in rows} == {0}
    assert all(len(areas) == 5 for row in rows for areas in row["aurocs"])
    assert {
        row["n_seeds"]
        for row in _table(run_dir)
        if row["scorer"] == "stack_mlp"
    } == {1}


def _items() -> StimulusSet:
    return StimulusSet(
        texts=("a", "b", "c", "d", "e", "f"),
        response_spans=(None,) * 6,
        concept_spans=(None,) * 6,
        labels=np.tile(np.array([0, 1]), 3),
        pairs=np.arange(6).reshape(3, 2),
        groups=np.arange(6).reshape(3, 2),
        uids=("a", "b", "c", "d", "e", "f"),
    )


def test_comparisons_keep_equal_mean_out_of_families() -> None:
    items = _items()
    splits = m2._Splits([], items, {"original": items}, {}, {}, {}, {}, None)

    def entry(
        scorer: str,
        readout: str,
        rule: str,
        draws: tuple[float, float],
    ) -> m2._Entry:
        return m2._Entry(
            m2.Cell(scorer, readout, rule),
            "reader",
            [
                m2._Choice(
                    (1,),
                    items.labels.astype(float),
                    (sign * items.labels[None, :]).astype(float),
                    items.labels,
                )
                for sign in draws
            ],
        )

    entries = [
        entry("logistic", "last", "best", (1, -1)),
        entry("equal_mean", "last", "all", (1, 1)),
        entry("stack_lr", "last", "all", (1, 1)),
        entry("concat_lr", "last", "all", (-1, -1)),
        entry("bridge_forest", "mean", "fixed", (1, 1)),
    ]
    rows = experiment._comparison_rows(
        experiment.Config(n_boot=8), m2._task("sentiment"), splits, entries
    )
    baseline = [row for row in rows if row["baseline"] == "logistic/last/best"]
    equal = next(r for r in baseline if r["reader"] == "equal_mean/last/all")
    stack = next(r for r in baseline if r["reader"] == "stack_lr/last/all")
    bridge = next(
        r for r in baseline if r["reader"] == "bridge_forest/mean/fixed"
    )
    assert equal["family_size"] == 0
    assert "delong_holm" not in equal
    assert stack["family_size"] == 2
    assert stack["auroc_draw_diff"] == [0, 1]
    assert stack["pair_draw_diff"] == [0, 1]
    assert stack["auroc_draw_mean"] == 0.5
    assert stack["auroc_draw_wins"] == 1
    assert bridge["family_size"] == 1
    assert bridge["family"] != stack["family"]
    assert any(row["baseline"] == "equal_mean/last/all" for row in rows)


def test_utility_clusters_join_pairs_through_shared_scenarios() -> None:
    items = _items()
    texts = dict(zip(items.uids, ("A", "B", "B", "C", "D", "E"), strict=True))
    row_clusters, pair_clusters = experiment._report_clusters(
        m2._task("utility"), items, texts
    )
    np.testing.assert_array_equal(row_clusters, [0, 0, 0, 0, 2, 2])
    np.testing.assert_array_equal(pair_clusters, [0, 0, 2])
