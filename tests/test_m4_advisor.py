"""CPU tests for the layer and cross-task reader experiment."""

from collections.abc import Callable
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
from tomography.experiments import m3_matrix as m3
from tomography.experiments import m4_advisor as experiment
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.model import BosMode
from tomography.model import LoadedModel
from tomography.reading import token_forest


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


def _fake_prompt_cells(
    loaded: LoadedModel,
    stimuli: StimulusSet,
    layers: Sequence[int],
    *,
    bos: BosMode | None,
    batch_size: int,
    decoy: bool,
) -> tuple[Activations, np.ndarray]:
    del batch_size, decoy
    encoded = loaded.encode(stimuli.texts, bos=bos)
    positions = [
        np.array(
            [i for i, (lo, hi) in enumerate(offsets) if lo < hi],
            dtype=np.int64,
        )
        for offsets in encoded.offsets
    ]
    counts = [len(row) for row in positions]
    tokens = np.concatenate(positions)
    stimulus = np.repeat(np.arange(len(stimuli.labels)), counts)
    data = np.broadcast_to(
        (tokens + 10 * stimuli.labels[stimulus])[None, :, None],
        (len(layers), len(tokens), loaded.d_model),
    ).astype(np.float32)
    cells = Activations(
        data=data,
        layers=tuple(layers),
        position="response",
        row_stimulus=stimulus,
        row_rank=np.concatenate([np.arange(n) for n in counts]),
        row_token=tokens,
        row_token_id=encoded.input_ids[stimulus, tokens].numpy(),
        n_stimuli=len(stimuli.labels),
    )
    return cells, np.zeros(len(tokens), dtype=np.int8)


@pytest.fixture
def setup_run(
    tmp_path: Path, tiny_gpt2: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> tuple[experiment.Config, RunDir, RunDir]:
    # Only the tokenizer runs here, including the longer task templates.
    tiny_gpt2.model.config.n_positions = 512
    monkeypatch.setattr(m2, "read_positions", _fake_positions)
    monkeypatch.setattr(m3, "prompt_cells", _fake_prompt_cells)
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
    monkeypatch.setattr(m3, "prompt_cells", early_analysis)
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


def test_run_writes_the_literal_family_after_the_gate(
    setup_run: tuple[experiment.Config, RunDir, RunDir],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, run_dir = setup_run
    gate_passed = False
    check = experiment.check_m2_equality

    def gate(current: list[list[Any]], stored: list[list[Any]]) -> None:
        nonlocal gate_passed
        check(current, stored)
        gate_passed = True

    def read(*args: Any, **kwargs: Any) -> tuple[Activations, np.ndarray]:
        assert gate_passed
        assert args[2] == (4, 8)
        return _fake_prompt_cells(*args, **kwargs)

    monkeypatch.setattr(experiment, "check_m2_equality", gate)
    monkeypatch.setattr(m3, "prompt_cells", read)
    experiment.run(config, run_dir)

    rows = _table(run_dir)
    for name, layers in (("literal_forest", [4, 8]), ("length_control", [])):
        found = [row for row in rows if row["scorer"] == name]
        assert {row["task"] for row in found} == {"sentiment", "utility"}
        assert {row["readout"] for row in found} == {"tokens"}
        scores = run_dir.read_json("scores_sentiment")
        entry = next(
            entry for entry in scores["entries"] if entry.startswith(name)
        )
        assert scores["layers"][entry] == [layers, layers]

    comparisons = run_dir.read_json("comparisons")
    related = [row for row in comparisons if "/bridge/" in row["family"]]
    assert related
    for family in {row["family"] for row in related}:
        members = [row for row in related if row["family"] == family]
        assert {row["reader"].split("/")[0] for row in members} == {
            "literal_forest",
            "length_control",
            "bridge_forest",
        }
        assert len(members) == 5
        assert {row["family_size"] for row in members} == {5}
        assert {row["baseline"].split("/")[1] for row in members} == {"last"}
        assert all(len(row["auroc_draw_diff"]) == 2 for row in members)
    summary = run_dir.read_json("summary")
    assert summary["literal_forest"]["alignment"] == "right"
    assert "shift token positions" in summary["literal_forest"]["caveat"]


@pytest.mark.parametrize(
    ("alignment", "message"),
    [
        ("typo", "literal_alignment must be right or left"),
        ("left", "left alignment requires qwen3-0.6b-base"),
    ],
)
def test_run_validates_literal_alignment_before_looking_up_m2(
    alignment: Any, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def lookup(*args: object) -> None:
        pytest.fail("looked up M2 before validating literal alignment")

    monkeypatch.setattr(experiment, "find_m2_run", lookup)
    config = experiment.Config(
        model="qwen3-1.7b-base", literal_alignment=alignment
    )
    with pytest.raises(ValueError, match=message):
        experiment.run(config, None)


def test_run_records_each_literal_draw_width(
    setup_run: tuple[experiment.Config, RunDir, RunDir],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, run_dir = setup_run
    score_splits = token_forest.score_splits
    widths: dict[str, list[int]] = {}

    def record(*args: Any, **kwargs: Any) -> list[token_forest.LiteralDraw]:
        draws = score_splits(*args, **kwargs)
        # The select split's texts identify the task of this call.
        widths[args[2].uids[0]] = [draw.width for draw in draws]
        return draws

    monkeypatch.setattr(token_forest, "score_splits", record)
    experiment.run(config, run_dir)

    rows = [
        row
        for row in run_dir.read_json("fit_diagnostics")["draws"]
        if row["readout"] == "tokens"
    ]
    assert len(widths) == 2
    assert sorted(row["literal_width"] for row in rows) == sorted(
        width for draws in widths.values() for width in draws
    )
    for task in ("sentiment", "utility"):
        found = [row for row in rows if row["task"] == task]
        assert [row["draw"] for row in found] == [0, 1]
        assert {row["series"] for row in found} == {"reader"}


def test_left_alignment_runs_on_qwen3_0_6b_base_against_the_same_m2_run(
    setup_run: tuple[experiment.Config, RunDir, RunDir],
    tiny_gpt2: LoadedModel,
    tmp_path: Path,
) -> None:
    config, _, _ = setup_run
    right = dataclasses.replace(config, model="qwen3-0.6b-base")
    left = dataclasses.replace(right, literal_alignment="left")
    assert left.config_hash() != right.config_hash()
    assert left.m2_config() == right.m2_config()
    baseline = RunDir.create(right.m2_config(), out_root=tmp_path)
    m2.execute(right.m2_config(), baseline, tiny_gpt2, _sources())
    root = Path(config.m2_root)
    assert experiment.find_m2_run(left, root) == baseline.path
    assert experiment.find_m2_run(right, root) == baseline.path

    run_dir = RunDir.create(left, out_root=tmp_path)
    assert run_dir.path != RunDir.create(right, out_root=tmp_path).path
    experiment.run(left, run_dir)

    assert run_dir.read_json("summary")["literal_forest"]["alignment"] == "left"
    rows = [row for row in _table(run_dir) if row["scorer"] == "literal_forest"]
    assert {row["task"] for row in rows} == {"sentiment", "utility"}


def test_comparison_counts_report_the_length_control_apart() -> None:
    def row(task: str, reader: str, size: int) -> dict[str, Any]:
        return {
            "task": task,
            "reader": reader,
            "family_size": size,
            "auroc_draw_mean": 0.1,
            "pair_draw_mean": -0.1,
        }

    rows = [
        row("sentiment", "literal_forest/tokens/fixed", 5),
        row("truth", "literal_forest/tokens/fixed", 5),
        row("sentiment", "length_control/tokens/all", 5),
        row("sentiment", "length_control/tokens/all", 5),
        row("truth", "length_control/tokens/all", 5),
        row("sentiment", "equal_mean/last/all", 0),
    ]
    counts = experiment._comparison_counts(rows)
    assert counts["with_truth"] == {
        "comparisons": 2,
        "higher_auroc": 2,
        "higher_pair_accuracy": 0,
    }
    assert counts["without_truth"]["comparisons"] == 1
    assert counts["length_control"] == {
        "with_truth": {
            "comparisons": 3,
            "higher_auroc": 3,
            "higher_pair_accuracy": 0,
        },
        "without_truth": {
            "comparisons": 2,
            "higher_auroc": 2,
            "higher_pair_accuracy": 0,
        },
    }


def test_run_reports_mlp_seed_spread_in_a_two_draw_design(
    setup_run: tuple[experiment.Config, RunDir, RunDir],
) -> None:
    config, _, run_dir = setup_run
    experiment.run(dataclasses.replace(config, pilot_draws=1), run_dir)
    diagnostics = run_dir.read_json("fit_diagnostics")
    rows = diagnostics["mlp_pilot"]
    assert rows
    assert {row["draw"] for row in rows} == {0}
    assert {row["series"] for row in rows} == {"reader"}
    assert all(
        len(row["aurocs"]) == len(row["pair_accuracies"]) == 5 for row in rows
    )
    assert diagnostics["recommended_mlp_repeats"] == (
        5
        if any(
            max(row["auroc_spread"], row["pair_accuracy_spread"]) > 0.01
            for row in rows
        )
        else 1
    )
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
        entry("logistic", "last", "best_deep", (1, 1)),
        entry("mean_diff", "last", "best_deep", (1, 1)),
        entry("equal_mean", "last", "all", (1, 1)),
        entry("stack_lr", "last", "all", (1, 1)),
        entry("concat_lr", "last", "all", (-1, -1)),
        entry("bridge_forest", "mean", "fixed", (1, 1)),
        entry("bridge_forest", "max", "fixed", (1, 1)),
        entry("bridge_forest", "last", "fixed", (1, 1)),
        entry("stack_mlp", "last", "all", (1, 1))._replace(series="shuffled"),
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
    assert bridge["family_size"] == 3
    assert bridge["family"] != stack["family"]
    assert {
        r["reader"] for r in baseline if r["family"] == bridge["family"]
    } == {
        "bridge_forest/last/fixed",
        "bridge_forest/max/fixed",
        "bridge_forest/mean/fixed",
    }
    assert not any("best_deep" in r["reader"] for r in rows)
    assert (
        experiment._comparison_counts(rows)["without_truth"]["comparisons"]
        == 10
    )
    assert any(row["baseline"] == "equal_mean/last/all" for row in rows)


def test_utility_clusters_join_pairs_through_shared_scenarios() -> None:
    items = _items()
    texts = dict(zip(items.uids, ("A", "B", "B", "C", "D", "E"), strict=True))
    row_clusters, pair_clusters = experiment._report_clusters(
        m2._task("utility"), items, texts
    )
    np.testing.assert_array_equal(row_clusters, [0, 0, 0, 0, 2, 2])
    np.testing.assert_array_equal(pair_clusters, [0, 0, 2])


def test_run_cross_task_arms_share_only_shared_layer_and_budget(
    setup_run: tuple[experiment.Config, RunDir, RunDir],
) -> None:
    config, _, run_dir = setup_run
    experiment.run(config, run_dir)
    rows = run_dir.read_json("cross_task")
    assert {row["readout"] for row in rows} == {"last", "max"}
    assert {row["arm"] for row in rows} == {
        "transfer",
        "per_task",
        "pooled_all",
        "pooled_matched",
    }
    pooled = [r for r in rows if r["arm"].startswith("pooled")]
    assert {r["train_pairs"] for r in pooled if r["arm"] == "pooled_all"} == {8}
    assert {
        r["train_pairs"] for r in pooled if r["arm"] == "pooled_matched"
    } == {4}
    assert {r["tuning"] for r in pooled} == {"shared", "per_task"}
    shared = [r for r in pooled if r["tuning"] == "shared"]
    for row in shared:
        counterpart = next(
            r
            for r in shared
            if r["task"] != row["task"]
            and all(
                r[key] == row[key] for key in ("arm", "readout", "condition")
            )
        )
        assert row["layers"] == counterpart["layers"]
        assert row["penalties"] == counterpart["penalties"]
    assert all(
        r["auroc_draw_diff"] == [0.0, 0.0]
        for r in rows
        if r["arm"] == "per_task"
    )


def test_transfer_keeps_source_orientation_on_reversed_target() -> None:
    train_values = np.array([[[-2.0], [2.0], [-1.0], [1.0]]], dtype=np.float32)
    labels = np.array([0, 1, 0, 1])
    pairs = np.arange(4).reshape(-1, 2)
    train = experiment._LabeledRows(train_values, (3,), labels, pairs, pairs)
    tuned = experiment._tuned_lr(experiment.Config(), train, [train], seed=0)[0]
    choice, penalty = experiment._score_best(tuned, [(-train_values)], labels)
    assert choice.layers == (3,)
    assert penalty == 0.01
    assert (choice.report[0, pairs[:, 0]] > choice.report[0, pairs[:, 1]]).all()


def _truth_rows() -> list[Example]:
    return [
        Example(
            f"{subset}-{split}-{label}-{i}",
            f"the claim {i} is {label}",
            label,
            split,
            {"subset": subset},
        )
        for subset in ("cities", "neg_cities", "sp_en_trans", "neg_sp_en_trans")
        for split in ("train", "val", "test")
        for label in (0, 1)
        for i in range(5)
    ]


def test_holdout_rebuilds_pairs_before_scoring_full_held_split(
    tiny_gpt2: LoadedModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(m2, "read_positions", _fake_positions)
    config = experiment.Config(
        model="tiny",
        layers=(1, 4),
        n_seeds=1,
        n_pairs=3,
        n_eval_pairs=2,
        decoys=(),
        shifts=(),
        logistic_c=(0.1,),
        n_boot=5,
    )
    rows = experiment._lodo_rows(
        config, tiny_gpt2, {"geometry_of_truth": _truth_rows()}, 1
    )
    assert {r["holdout_kind"] for r in rows} == {"subset", "topic", "polarity"}
    assert {r["arm"] for r in rows} == {
        "held_out",
        "all_subsets",
        "within_held",
    }
    assert {r["readout"] for r in rows} == {"last", "max"}
    assert {r["train_pairs"] for r in rows} == {3}
    assert {
        r["report_pairs"] for r in rows if r["holdout_kind"] == "subset"
    } == {5}
    assert {
        r["report_pairs"] for r in rows if r["holdout_kind"] != "subset"
    } == {10}
    assert len(rows) == 8 * 3 * 2


def test_geometry_split_can_separate_a_statement_from_its_negation() -> None:
    from tomography.datasets.geometry_of_truth import parse_geometry_of_truth

    original = "statement,label\n" + "".join(
        f"city {i} is coastal,1\n" for i in range(32)
    )
    negated = "statement,label\n" + "".join(
        f"city {i} is not coastal,0\n" for i in range(32)
    )
    affirmative = parse_geometry_of_truth(original, subset="cities")
    negative = parse_geometry_of_truth(negated, subset="neg_cities")
    assert any(
        a.split != b.split for a, b in zip(affirmative, negative, strict=True)
    )


def test_gate_accepts_null_only_against_nonfinite_metric() -> None:
    rows = _gate_rows()
    column = m2.COLUMNS.index("auroc")
    rows[0][column] = None
    experiment.check_m2_equality(rows, rows)
    current = [list(row) for row in rows]
    current[0][column] = float("nan")
    experiment.check_m2_equality(current, rows)
    current[0][column] = 0.5
    with pytest.raises(ValueError, match=r"M2 equality failed.*auroc"):
        experiment.check_m2_equality(current, rows)


def test_mlp_pilot_ignores_report_labels_and_does_not_change_emitted_scores(
    setup_run: tuple[experiment.Config, RunDir, RunDir],
    tiny_gpt2: LoadedModel,
) -> None:
    config, _, _ = setup_run
    source = experiment._prepare(
        config, tiny_gpt2, [m2._task("sentiment")], _sources()
    )[0]
    flipped = dataclasses.replace(
        source,
        splits=dataclasses.replace(
            source.splits,
            report_items={
                condition: dataclasses.replace(items, labels=1 - items.labels)
                for condition, items in source.splits.report_items.items()
            },
        ),
    )
    pilot_config = dataclasses.replace(config, pilot_draws=1)
    original, _, pilot = experiment._ensemble_entries(pilot_config, source, 1)
    changed, _, changed_pilot = experiment._ensemble_entries(
        pilot_config, flipped, 1
    )
    regular, _, _ = experiment._ensemble_entries(config, source, 1)
    assert pilot == changed_pilot
    assert {row["series"] for row in pilot} == {"reader"}
    assert len(pilot) == 3
    for expected, actual, control in zip(
        original, changed, regular, strict=True
    ):
        if expected.cell.scorer == "stack_mlp":
            np.testing.assert_array_equal(
                expected.choices[0].report, actual.choices[0].report
            )
            np.testing.assert_array_equal(
                expected.choices[0].report, control.choices[0].report
            )


def test_equal_layer_weights_match_the_select_standardized_mean(
    setup_run: tuple[experiment.Config, RunDir, RunDir],
    tiny_gpt2: LoadedModel,
) -> None:
    config, _, _ = setup_run
    source = experiment._prepare(
        config, tiny_gpt2, [m2._task("sentiment")], _sources()
    )[0]
    train, select, report = experiment._draw_rows(source.splits, "last", 0)
    tuned = dataclasses.replace(
        source.baseline.logistic["last"][0],
        select_accuracy=np.full(len(train.layers), 0.75),
    )
    choices, _ = experiment._stack_choices(
        config,
        train,
        select,
        report,
        tuned,
        seed=2,
    )
    equal = choices["equal_mean", "all"]
    weighted = choices["weighted_mean", "all"]
    np.testing.assert_allclose(weighted.select, equal.select, rtol=1e-12)
    np.testing.assert_allclose(weighted.report, equal.report, rtol=1e-12)


def test_utility_descriptive_equal_mean_has_cluster_intervals() -> None:
    items = _items()
    texts = dict(zip(items.uids, ("A", "B", "B", "C", "D", "E"), strict=True))
    splits = m2._Splits([], items, {"original": items}, {}, {}, {}, texts, None)
    val = items.labels.astype(float)
    entries = [
        m2._Entry(
            m2.Cell(scorer, "last", rule),
            "reader",
            [m2._Choice((1,), val, (sign * val)[None, :], items.labels)],
        )
        for scorer, rule, sign in (
            ("logistic", "best", -1),
            ("equal_mean", "all", 1),
        )
    ]
    (row,) = experiment._comparison_rows(
        experiment.Config(n_boot=8), m2._task("utility"), splits, entries
    )
    assert row["family_size"] == 0
    assert row["descriptive"]
    assert row["auroc_cluster"]["difference"] == 1.0
    assert row["pair_cluster"]["difference"] == 1.0
    assert "delong_holm" not in row


def test_lodo_reads_each_unique_prompt_once_without_changing_metrics(
    tiny_gpt2: LoadedModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = experiment.Config(
        model="tiny",
        layers=(1, 4),
        n_seeds=1,
        n_pairs=3,
        n_eval_pairs=2,
        decoys=("random",),
        shifts=(),
        logistic_c=(0.1,),
        n_boot=5,
    )
    examples = {
        "geometry_of_truth": [
            dataclasses.replace(row, text=f"{row.text} {row.uid}")
            for row in _truth_rows()
        ]
    }
    texts: list[str] = []

    def counted_read(
        loaded: LoadedModel,
        items: StimulusSet,
        positions: Sequence[PositionStrategy],
        cfg: experiment.Config,
    ) -> dict[PositionStrategy, Activations]:
        texts.extend(items.texts)
        return _fake_positions(loaded, items, positions, cfg)

    def without_cache(
        cfg: experiment.Config,
        loaded: LoadedModel,
        plan: object,
    ) -> Callable[
        [str, StimulusSet], dict[m2.Readout, experiment._LabeledRows]
    ]:
        def read(
            condition: str, items: StimulusSet
        ) -> dict[m2.Readout, experiment._LabeledRows]:
            return experiment._truth_batch(cfg, loaded, items)

        return read

    monkeypatch.setattr(m2, "read_positions", counted_read)
    cached = experiment._lodo_rows(config, tiny_gpt2, examples, 1)
    cached_texts = list(texts)
    assert len(cached_texts) == len(set(cached_texts)) == 153
    texts.clear()
    monkeypatch.setattr(experiment, "_truth_reader", without_cache)
    uncached = experiment._lodo_rows(config, tiny_gpt2, examples, 1)
    assert len(texts) == 598
    assert set(cached_texts) == set(texts)
    assert cached == uncached


def test_fold_raw_scores_ignore_labels_of_held_out_pairs() -> None:
    from tomography.reading.ensembles import pair_folds

    values = np.array([[[-3.0], [3.0]] * 8], dtype=np.float32)
    labels = np.tile([0, 1], 8)
    pairs = np.arange(16).reshape(-1, 2)
    train = experiment._LabeledRows(values, (1,), labels, pairs, pairs)
    folds = list(pair_folds(pairs, n_rows=16, n_folds=2, seed=9))
    fit, held = folds[0]
    flipped = labels.copy()
    flipped[held] = 1 - flipped[held]
    changed = dataclasses.replace(train, labels=flipped)
    config = experiment.Config()
    for i, (fit, held) in enumerate(folds):
        actual = experiment._fit_fold(config, train, (0.1,), fit, held, seed=4)
        other = experiment._fit_fold(config, changed, (0.1,), fit, held, seed=4)
        if i == 0:
            np.testing.assert_array_equal(actual, other)
        else:
            np.testing.assert_array_equal(actual, -other)


def test_cli_discovers_m4_config_run_and_plot() -> None:
    from tomography.cli import load_experiment

    loaded = load_experiment("m4")
    assert loaded.config_cls is experiment.Config
    assert loaded.run is experiment.run
    assert loaded.plot is experiment.plot


def test_plot_writes_figures_from_a_tiny_run(
    setup_run: tuple[experiment.Config, RunDir, RunDir],
    tmp_path: Path,
) -> None:
    config, _, run_dir = setup_run
    experiment.run(config, run_dir)
    outputs = experiment.plot(run_dir, tmp_path / "plots")
    assert {path.stem for path in outputs} == {
        "auroc_sentiment",
        "auroc_utility",
    }
    assert {path.suffix for path in outputs} == {".pdf", ".png", ".svg"}
    assert len(outputs) == 6
    assert all(path.stat().st_size > 0 for path in outputs)
