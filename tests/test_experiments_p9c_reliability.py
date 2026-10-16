"""Tests for the P9c reliability experiment."""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
from pathlib import Path
import time
import types
from typing import Any
import zlib

import numpy as np
import pytest

from tomography.cli import load_experiment
from tomography.datasets.records import MetaValue
from tomography.datasets.records import QAExample
from tomography.datasets.records import Split
from tomography.experiments import p9c_reliability
from tomography.experiments.base import RunDir
from tomography.experiments.p9a_labels import Config as LabelsConfig
from tomography.experiments.p9c_reliability import CONFIDENCE
from tomography.experiments.p9c_reliability import COVERAGE_GRID
from tomography.experiments.p9c_reliability import PREDICTORS
from tomography.experiments.p9c_reliability import RISK_PANELS
from tomography.experiments.p9c_reliability import Config
from tomography.experiments.p9c_reliability import execute
from tomography.experiments.p9c_reliability import plot
from tomography.experiments.p9c_reliability import resolve_lat
from tomography.experiments.p9c_reliability import run
from tomography.knowledge import DEFAULT_SHOTS
from tomography.knowledge import ItemLabel
from tomography.knowledge import Label
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.p9.pool import LabeledItem
from tomography.p9.pool import pool_records
from tomography.p9.reading import ItemActivations
from tomography.p9.reading import draw_seed
from tomography.p9.reading import fit_type_reader
from tomography.p9.reading import p9_template
from tomography.p9.risk import aurc
from tomography.p9.risk import auroc_ci
from tomography.p9.risk import grouped_auroc_ci
from tomography.p9.signals import item_signals
from tomography.p9.unknowns import type_pools
from tomography.reading.metrics import auroc
from tomography.reading.reader import RepReader

SPLITS: tuple[Split, ...] = ("train", "train", "train", "val", "test")
RELATIONS = ("father", "mother", "occupation")
DATASETS = ("popqa", "triviaqa", "nq_open")
TABLE_COLUMNS = [
    "dataset",
    "predictor",
    "n",
    "n_error",
    "n_filled",
    "auroc",
    "lo",
    "hi",
    "aurc",
    "auroc_within",
    "lo_within",
    "hi_within",
    "n_pairs_within",
]
WITHIN_COLUMNS = TABLE_COLUMNS[-4:]
RELATION_COLUMNS = ["predictor", "relation", "n_correct", "n_error", "auroc"]
INCREMENTAL_COLUMNS = [
    "dataset",
    "features",
    "auroc",
    "diff_vs_confidence",
    "lo",
    "hi",
]
NO_META = ("popularity", "template", "name_logprob")
OUTPUT_SERIES = [
    "first_token_prob",
    "neg_entropy",
    "answer_logprob",
    "mean_token_logprob",
    "consistency",
    "p_true",
    "lat",
    "probe",
]
CONTROL_SERIES = [
    "lat",
    "probe",
    "popularity",
    "template",
    "length",
    "name_logprob",
]
RISK_STEMS = [
    f"riskcov_{d}_{p}" for d in DATASETS for p in ("outputs", "controls")
]
SHIFTS = {1: 0.0, 2: 8.0, 3: 0.0}


def _label(
    uid: str,
    label: Label,
    greedy: str,
    correct: bool,
    samples: tuple[str, ...],
) -> ItemLabel:
    return ItemLabel(
        uid=uid,
        label=label,
        basis="behavior",
        greedy=greedy,
        greedy_correct=correct,
        samples=samples,
        n_correct=sum(s == "gold" for s in samples),
    )


def _example(
    uid: str,
    split: Split,
    source: str,
    kind: str,
    meta: dict[str, MetaValue],
) -> QAExample:
    return QAExample(
        uid=uid,
        question=f"Who is {uid.replace(':', ' ')}?",
        answers=("gold",),
        kind=kind,
        split=split,
        meta={"source": source, **meta},
    )


def _answered(
    correct: bool, empty: bool = False, firm: bool = False
) -> tuple[str, tuple[str, ...]]:
    if correct:
        return "gold", ("gold",) * 5
    if empty:
        return "", ()
    if firm:
        return "other", ("other",) * 5
    return "other", ("other", "gold", "gold", "gold", "gold")


def _popqa(n: int = 150) -> list[LabeledItem]:
    items = []
    for i in range(n):
        correct = i % 6 in (0, 1, 4)
        label: Label = (
            "known" if correct else ("uncertain" if i % 6 == 5 else "unknown")
        )
        greedy, samples = _answered(correct, empty=i % 17 == 2, firm=i % 3 == 0)
        meta: dict[str, MetaValue] = {
            "relation": RELATIONS[i % 3],
            "popularity": None
            if i % 11 == 0
            else 1
            if i % 23 == 0 and i % 5 == 4
            else 10 ** (1 + (0 if correct else 3) + i % 2),
        }
        if i % 13:
            meta["subject"] = f"Person {i}"
        example = _example(f"popqa:{i}", SPLITS[i % 5], "popqa", "real", meta)
        items.append(
            LabeledItem(
                example, _label(example.uid, label, greedy, correct, samples)
            )
        )
    return items


def _open(source: str, n: int = 60) -> list[LabeledItem]:
    items = []
    for i in range(n):
        correct = i % 2 == 0
        greedy, samples = _answered(correct, firm=i % 4 == 1)
        example = _example(f"{source}:{i}", SPLITS[i % 5], source, "open", {})
        label: Label = "known" if correct else "unknown"
        items.append(
            LabeledItem(
                example, _label(example.uid, label, greedy, correct, samples)
            )
        )
    return items


def _fictional(n: int = 50) -> list[LabeledItem]:
    items = []
    for i in range(n):
        example = _example(
            f"generated:{i}",
            SPLITS[i % 5],
            "generated",
            "fictional",
            {"relation": RELATIONS[i % 3], "popularity": None},
        )
        greedy, samples = _answered(False)
        items.append(
            LabeledItem(
                example, _label(example.uid, "unknown", greedy, False, samples)
            )
        )
    return items


def _construction(n: int = 6) -> list[LabeledItem]:
    items = []
    for i in range(n):
        example = QAExample(
            uid=f"popqa:c{i}",
            question=f"Who is popqa c{i}?",
            answers=(),
            kind="real",
            split=SPLITS[i % 5],
            meta={"source": "popqa", "relation": RELATIONS[i % 3]},
        )
        items.append(
            LabeledItem(
                example,
                ItemLabel(
                    example.uid, "unknown", "construction", "", False, (), 0
                ),
            )
        )
    return items


def _items() -> list[LabeledItem]:
    return [
        *_popqa(),
        *_open("triviaqa"),
        *_open("nq_open"),
        *_fictional(),
        *_construction(),
    ]


def _starved_items() -> list[LabeledItem]:
    """`_items` where known and unknown PopQA share no relation."""
    popqa = [
        dataclasses.replace(
            i,
            example=dataclasses.replace(
                i.example,
                meta={
                    **i.example.meta,
                    "relation": RELATIONS[0 if i.label.label == "known" else 1],
                },
            ),
        )
        for i in _popqa()
    ]
    return [*popqa, *_items()[len(popqa) :]]


def _behavior(items: Sequence[LabeledItem], dataset: str) -> list[LabeledItem]:
    return [
        i for i in items if i.group == dataset and i.label.basis == "behavior"
    ]


def _test(items: Sequence[LabeledItem]) -> list[LabeledItem]:
    return [i for i in items if i.example.split == "test"]


def _correct(items: Sequence[LabeledItem]) -> np.ndarray:
    return np.array([i.label.greedy_correct for i in items], dtype=np.int64)


def _planted_acts(
    items: Sequence[LabeledItem],
    layers: Sequence[int],
    d_model: int,
    shifts: dict[int, float] = SHIFTS,
) -> ItemActivations:
    data = np.zeros((len(layers), len(items), d_model), dtype=np.float32)
    for j, item in enumerate(items):
        rng = np.random.default_rng(zlib.crc32(item.example.uid.encode()))
        data[:, j] = 0.1 * rng.standard_normal((len(layers), d_model))
        for k, layer in enumerate(layers):
            if item.label.greedy_correct:
                data[k, j, 0] += shifts.get(layer, 0.0)
            if item.group == "popqa" and item.label.label == "unknown":
                data[k, j, 1] += shifts.get(layer, 0.0)
    return ItemActivations(
        tuple(i.example.uid for i in items), tuple(layers), data
    )


def _fake_collect(
    calls: list[dict[str, Any]],
) -> Callable[..., ItemActivations]:
    def fake(
        loaded: LoadedModel,
        items: Sequence[LabeledItem],
        template: Any,
        chat: Any,
        *,
        layers: Sequence[int],
        bos: Any = None,
        batch_size: int = 8,
    ) -> ItemActivations:
        calls.append(
            {
                "items": list(items),
                "template": template,
                "chat": chat,
                "layers": tuple(layers),
                "bos": bos,
                "batch_size": batch_size,
            }
        )
        return _planted_acts(items, layers, loaded.d_model)

    return fake


def _noise_collect(d_model: int) -> Callable[..., ItemActivations]:
    def fake(
        loaded: LoadedModel,
        items: Sequence[LabeledItem],
        template: Any,
        chat: Any,
        *,
        layers: Sequence[int],
        bos: Any = None,
        batch_size: int = 8,
    ) -> ItemActivations:
        data = np.zeros((len(layers), len(items), d_model), dtype=np.float32)
        for j, item in enumerate(items):
            rng = np.random.default_rng(zlib.crc32(item.example.uid.encode()))
            data[:, j] = rng.standard_normal((len(layers), d_model))
        return ItemActivations(
            tuple(i.example.uid for i in items), tuple(layers), data
        )

    return fake


def _flip_test_popularity(
    items: Sequence[LabeledItem],
) -> list[LabeledItem]:
    flipped = []
    for item in items:
        if (
            item.example.split == "test"
            and item.example.meta["popularity"] is not None
        ):
            i = int(item.example.uid.split(":")[1])
            pop = 10 ** (1 + (3 if item.label.greedy_correct else 0) + i % 2)
            meta: dict[str, MetaValue] = {
                **item.example.meta,
                "popularity": pop,
            }
            item = dataclasses.replace(
                item, example=dataclasses.replace(item.example, meta=meta)
            )
        flipped.append(item)
    return flipped


@pytest.fixture
def planted(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replaces the collection by planted activations; records each call."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        p9c_reliability, "collect_item_activations", _fake_collect(calls)
    )
    return calls


def _config(**overrides: Any) -> Config:
    base = Config(
        model="tiny",
        labels_run="runs/p9a",
        n_shots=2,
        n_seeds=2,
        n_pairs=16,
        min_pairs=8,
        min_probe=20,
        n_boot=60,
        batch_size=16,
    )
    return dataclasses.replace(base, **overrides)


def _execute(
    loaded: LoadedModel,
    tmp_path: Path,
    *,
    items: Sequence[LabeledItem] | None = None,
    known_contrasts: bool = True,
    **overrides: Any,
) -> RunDir:
    config = _config(**overrides)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(
        config,
        run_dir,
        loaded,
        _items() if items is None else items,
        known_contrasts=known_contrasts,
    )
    return run_dir


def _wrong(item: LabeledItem) -> LabeledItem:
    return dataclasses.replace(
        item,
        label=dataclasses.replace(
            item.label,
            label="unknown",
            greedy="other",
            greedy_correct=False,
            samples=("other",) * 5,
            n_correct=0,
        ),
    )


def _labels_run(
    tmp_path: Path,
    items: Sequence[LabeledItem],
    *,
    known_contrasts: bool = True,
    finish: bool = True,
    n_shots: int = 2,
) -> RunDir:
    run_dir = RunDir.create(
        LabelsConfig(model="tiny"), out_root=tmp_path / "labels"
    )
    run_dir.write_json("items", pool_records(items))
    run_dir.write_json(
        "summary",
        {"known_contrasts": known_contrasts, "recipe": {"n_shots": n_shots}},
    )
    if finish:
        run_dir.finalize()
    return run_dir


_DEFAULT_RUNS: dict[str, RunDir] = {}


@pytest.fixture
def default_run(
    tiny_arch: LoadedModel, tmp_path_factory: pytest.TempPathFactory
) -> RunDir:
    """One run per architecture with the default settings."""
    key = type(tiny_arch.model).__name__
    if key not in _DEFAULT_RUNS:
        out = tmp_path_factory.mktemp(f"p9c-{key}")
        _DEFAULT_RUNS[key] = _execute(tiny_arch, out)
    return _DEFAULT_RUNS[key]


_PLANTED_RUNS: dict[str, RunDir] = {}


@pytest.fixture
def planted_run(
    tiny_gpt2: LoadedModel, tmp_path_factory: pytest.TempPathFactory
) -> RunDir:
    """One run on planted activations with the default settings."""
    if "default" not in _PLANTED_RUNS:
        out = tmp_path_factory.mktemp("p9c-planted")
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                p9c_reliability,
                "collect_item_activations",
                _fake_collect([]),
            )
            _PLANTED_RUNS["default"] = _execute(tiny_gpt2, out)
    return _PLANTED_RUNS["default"]


def _summary(run_dir: RunDir) -> dict[str, Any]:
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    return summary


def _rows(
    run_dir: RunDir, name: str, columns: list[str]
) -> list[dict[str, Any]]:
    table = run_dir.read_json(name)
    assert isinstance(table, dict)
    assert table["columns"] == columns
    return [dict(zip(columns, row, strict=True)) for row in table["rows"]]


def _table(run_dir: RunDir) -> list[dict[str, Any]]:
    return _rows(run_dir, "table", TABLE_COLUMNS)


def _predictors(rows: list[dict[str, Any]], dataset: str) -> list[str]:
    return [r["predictor"] for r in rows if r["dataset"] == dataset]


def _row(
    rows: list[dict[str, Any]], dataset: str, predictor: str
) -> dict[str, Any]:
    (row,) = [
        r
        for r in rows
        if r["dataset"] == dataset and r["predictor"] == predictor
    ]
    return row


def _files(run_dir: RunDir) -> list[str]:
    return sorted(p.name for p in run_dir.path.iterdir() if p.is_file())


def test_the_constants_name_the_predictors_and_the_panels() -> None:
    assert PREDICTORS == (
        "first_token_prob",
        "neg_entropy",
        "answer_logprob",
        "mean_token_logprob",
        "consistency",
        "p_true",
        "lat",
        "probe",
        "popularity",
        "template",
        "length",
        "name_logprob",
    )
    assert CONFIDENCE == (
        "first_token_prob",
        "neg_entropy",
        "mean_token_logprob",
        "consistency",
        "p_true",
    )
    assert {
        "outputs": tuple(OUTPUT_SERIES),
        "controls": tuple(CONTROL_SERIES),
    } == RISK_PANELS
    assert tuple(range(5, 101, 5)) == COVERAGE_GRID


def test_config_defaults_and_round_trip() -> None:
    config = Config(labels_run="runs/p9a")

    assert config.experiment == "p9c"
    assert config.model == "gpt2"
    assert config.template == "fewshot"
    assert config.n_shots == 15
    assert config.datasets == DATASETS
    assert config.lat_type == "obscure_real"
    assert config.lat_method == "logistic"
    assert config.layers is None
    assert config.layer is None
    assert config.n_seeds == 5
    assert config.n_pairs == 64
    assert config.min_pairs == 16
    assert config.n_boot == 1000
    assert config.min_probe == 50
    assert config.min_relation_class == 5
    assert config.bos is None
    assert config.batch_size == 8
    assert Config.from_dict(config.to_dict()) == config


def test_config_round_trips_with_every_field_set() -> None:
    config = Config(
        labels_run="runs/p9a",
        datasets=("popqa",),
        lat_type="real_vs_generated",
        lat_method="mean_diff",
        layers=(2, 4),
        layer=4,
        min_relation_class=3,
        bos="none",
    )

    assert Config.from_dict(config.to_dict()) == config


def test_config_requires_the_labels_run() -> None:
    with pytest.raises(ValueError, match="labels_run"):
        Config.from_dict({})


def test_config_rejects_a_method_outside_the_literal() -> None:
    data = Config(labels_run="runs/p9a").to_dict()
    data["lat_method"] = "magic"

    with pytest.raises(ValueError, match="lat_method"):
        Config.from_dict(data)


def test_the_cli_discovers_the_experiment() -> None:
    experiment = load_experiment("p9c")

    assert experiment.config_cls is Config
    assert experiment.run is run
    assert experiment.plot is plot


def test_resolve_lat_returns_a_usable_contrast() -> None:
    pool, reason = resolve_lat(
        _items(), "obscure_real", known_contrasts=True, min_pairs=8
    )

    assert reason is None
    assert pool is not None
    assert pool.name == "obscure_real"
    assert pool == type_pools(_items())["obscure_real"]


def test_resolve_lat_reads_an_unknown_type_contrast_under_a_closed_gate() -> (
    None
):
    pool, reason = resolve_lat(
        _items(), "real_vs_generated", known_contrasts=False, min_pairs=8
    )

    assert reason is None
    assert pool is not None
    assert pool.contrast == "unknown_type"


def test_resolve_lat_names_the_gate_for_a_knowledge_contrast() -> None:
    pool, reason = resolve_lat(
        _items(), "obscure_real", known_contrasts=False, min_pairs=8
    )

    assert pool is None
    assert reason == (
        "the p9a gate admits no known versus unknown contrast for this model"
    )


def test_resolve_lat_gives_the_check_pool_reason() -> None:
    pool, reason = resolve_lat(
        _items(), "obscure_real", known_contrasts=True, min_pairs=10_000
    )

    assert pool is None
    assert reason == "fewer than 10000 train items on one side"


def test_resolve_lat_gives_the_matched_pairs_reason() -> None:
    pool, reason = resolve_lat(
        _starved_items(), "obscure_real", known_contrasts=True, min_pairs=8
    )

    assert pool is None
    assert reason == "fewer than 8 matched train pairs"


def test_resolve_lat_only_needs_train_pairs() -> None:
    train_only = [
        dataclasses.replace(
            i, example=dataclasses.replace(i.example, split="train")
        )
        for i in _popqa()
    ]

    pool, reason = resolve_lat(
        train_only, "obscure_real", known_contrasts=True, min_pairs=8
    )

    assert reason is None
    assert pool is not None


@pytest.mark.parametrize("known_contrasts", [True, False])
def test_resolve_lat_names_a_misspelled_contrast(known_contrasts: bool) -> None:
    pool, reason = resolve_lat(
        _items(),
        "no_such_type",
        known_contrasts=known_contrasts,
        min_pairs=8,
    )

    assert pool is None
    assert reason == "no contrast named 'no_such_type' on these labels"


def test_execute_writes_the_files_of_a_run(default_run: RunDir) -> None:
    names = _files(default_run)

    expected = {
        "config.json",
        "meta.json",
        "table.json",
        "table_incremental.json",
        "summary.json",
        "curves.npz",
        "curves.json",
        "relations_popqa.json",
        *(f"signals_{d}.npz" for d in DATASETS),
    }
    for stem in RISK_STEMS:
        expected |= {f"{stem}.npz", f"{stem}.json"}
    assert set(names) == expected
    assert len(names) == 23


def test_summary_records_the_run(default_run: RunDir) -> None:
    summary = _summary(default_run)

    assert set(summary) == {
        "layer",
        "layer_source",
        "template",
        "datasets",
        "skipped",
        "probe_skipped",
        "lat_type",
        "lat_method",
        "lat_skipped",
        "known_contrasts",
    }
    assert summary["layer_source"] == "validation"
    assert summary["layer"] in (1, 2, 3)
    assert summary["template"] == "fewshot"
    assert summary["datasets"] == list(DATASETS)
    assert summary["skipped"] == {}
    assert summary["probe_skipped"] == {}
    assert summary["lat_type"] == "obscure_real"
    assert summary["lat_method"] == "logistic"
    assert summary["lat_skipped"] is None
    assert summary["known_contrasts"] is True


def test_popqa_rows_hold_all_twelve_predictors(default_run: RunDir) -> None:
    rows = _table(default_run)

    assert _predictors(rows, "popqa") == list(PREDICTORS)


@pytest.mark.parametrize("dataset", ["triviaqa", "nq_open"])
def test_open_datasets_lack_the_predictors_that_need_meta(
    default_run: RunDir, dataset: str
) -> None:
    rows = _table(default_run)

    assert _predictors(rows, dataset) == [
        p for p in PREDICTORS if p not in NO_META
    ]


def test_table_counts_the_test_items_of_the_behavior_population(
    default_run: RunDir,
) -> None:
    rows = _table(default_run)

    for dataset in DATASETS:
        test = _test(_behavior(_items(), dataset))
        n_error = int((_correct(test) == 0).sum())
        for row in (r for r in rows if r["dataset"] == dataset):
            assert row["n"] == len(test)
            assert row["n_error"] == n_error
    assert len(_test(_behavior(_items(), "popqa"))) < len(
        _test([i for i in _items() if i.group == "popqa"])
    )


def test_table_metrics_follow_the_saved_signals(default_run: RunDir) -> None:
    rows = _table(default_run)

    for dataset in DATASETS:
        arrays = default_run.read_arrays(f"signals_{dataset}")
        for row in (r for r in rows if r["dataset"] == dataset):
            scores = arrays[row["predictor"]]
            point, lo, hi = auroc_ci(
                scores, arrays["correct"], n_boot=60, seed=0
            )
            assert row["auroc"] == pytest.approx(point)
            assert row["lo"] == pytest.approx(lo)
            assert row["hi"] == pytest.approx(hi)
            assert row["aurc"] == pytest.approx(
                aurc(scores, arrays["correct"], seed=0)
            )
            assert 0.0 <= row["lo"] <= row["hi"] <= 1.0


def test_signals_hold_the_test_items_of_each_predictor(
    default_run: RunDir,
) -> None:
    rows = _table(default_run)

    for dataset in DATASETS:
        arrays = default_run.read_arrays(f"signals_{dataset}")
        test = _test(_behavior(_items(), dataset))
        relation = arrays.pop("relation", None)
        assert (relation is not None) == (dataset == "popqa")
        assert set(arrays) == {*_predictors(rows, dataset), "correct"}
        np.testing.assert_array_equal(arrays["correct"], _correct(test))
        assert all(a.shape == (len(test),) for a in arrays.values())
        assert all(np.all(np.isfinite(a)) for a in arrays.values())


def test_a_firm_wrong_answer_keeps_consistency_below_a_perfect_ranking(
    default_run: RunDir,
) -> None:
    rows = _table(default_run)

    for dataset in DATASETS:
        assert 0.5 < _row(rows, dataset, "consistency")["auroc"] < 1.0


def test_lat_adds_to_the_confidence_when_it_carries_the_errors(
    planted_run: RunDir,
) -> None:
    rows = _rows(planted_run, "table_incremental", INCREMENTAL_COLUMNS)

    (plus,) = [
        r
        for r in rows
        if r["dataset"] == "popqa" and r["features"] == "confidence+lat"
    ]
    assert plus["diff_vs_confidence"] > 0.0
    assert plus["lo"] > 0.0


def test_incremental_table_holds_the_three_feature_sets(
    default_run: RunDir,
) -> None:
    rows = _rows(default_run, "table_incremental", INCREMENTAL_COLUMNS)

    assert [(r["dataset"], r["features"]) for r in rows] == [
        (d, f)
        for d in DATASETS
        for f in ("confidence", "confidence+lat", "confidence+probe")
    ]
    for row in rows:
        if row["features"] == "confidence":
            assert (row["diff_vs_confidence"], row["lo"], row["hi"]) == (
                0.0,
                0.0,
                0.0,
            )
        else:
            assert row["lo"] <= row["hi"]
    for dataset in DATASETS:
        base = next(
            r["auroc"]
            for r in rows
            if r["dataset"] == dataset and r["features"] == "confidence"
        )
        for row in (r for r in rows if r["dataset"] == dataset):
            assert row["diff_vs_confidence"] == pytest.approx(
                row["auroc"] - base
            )


def test_risk_files_hold_the_series_of_each_panel(default_run: RunDir) -> None:
    for dataset in DATASETS:
        for panel, series in (
            ("outputs", OUTPUT_SERIES),
            ("controls", CONTROL_SERIES),
        ):
            stem = f"riskcov_{dataset}_{panel}"
            arrays = default_run.read_arrays(stem)
            sidecar = default_run.read_json(stem)
            assert isinstance(sidecar, dict)
            expected = [
                s
                for s in series
                if s in _predictors(_table(default_run), dataset)
            ]
            assert sidecar["series"] == expected
            assert sidecar["xlabel"] == "Coverage (%)"
            assert sidecar["ylabel"] == "Risk (error rate among answered)"
            assert sidecar["chance"] is None
            np.testing.assert_array_equal(
                arrays["x"], np.array(COVERAGE_GRID, dtype=np.float64)
            )
            assert arrays["mean"].shape == (len(expected), 20)
            assert (
                arrays["lo"].shape == arrays["hi"].shape == arrays["mean"].shape
            )
            assert np.all(arrays["lo"] <= arrays["mean"] + 1e-12)
            assert np.all(arrays["mean"] <= arrays["hi"] + 1e-12)


def test_the_outputs_panel_has_eight_series_and_the_controls_six(
    default_run: RunDir,
) -> None:
    outputs = default_run.read_arrays("riskcov_popqa_outputs")
    controls = default_run.read_arrays("riskcov_popqa_controls")
    open_controls = default_run.read_arrays("riskcov_triviaqa_controls")

    assert outputs["mean"].shape[0] == 8
    assert controls["mean"].shape[0] == 6
    assert open_controls["mean"].shape[0] == 3


def test_full_coverage_risk_is_the_test_error_rate(default_run: RunDir) -> None:
    for dataset in DATASETS:
        correct = default_run.read_arrays(f"signals_{dataset}")["correct"]
        error_rate = 1.0 - float(correct.mean())
        for panel in ("outputs", "controls"):
            mean = default_run.read_arrays(f"riskcov_{dataset}_{panel}")["mean"]
            np.testing.assert_allclose(mean[:, -1], error_rate)


def test_curves_hold_lat_and_probe_per_layer(
    tiny_arch: LoadedModel, default_run: RunDir
) -> None:
    arrays = default_run.read_arrays("curves")
    sidecar = default_run.read_json("curves")

    assert isinstance(sidecar, dict)
    assert sidecar["series"] == ["lat", "probe"]
    assert sidecar["ylabel"] == "AUROC (own errors)"
    assert sidecar["chance"] == 0.5
    np.testing.assert_array_equal(arrays["x"], [1.0, 2.0, 3.0])
    assert arrays["per_seed"].shape == (2, 2, tiny_arch.n_layers)
    np.testing.assert_array_equal(
        arrays["per_seed"][1, 0], arrays["per_seed"][1, 1]
    )
    assert np.all((arrays["mean"] >= 0.0) & (arrays["mean"] <= 1.0))


def test_the_plot_draws_curves_and_every_risk_file(
    default_run: RunDir, tmp_path: Path
) -> None:
    paths = plot(default_run, tmp_path / "figures")

    assert len(paths) == 21
    assert all(p.is_file() for p in paths)
    assert {p.stem for p in paths} == {"curves", *RISK_STEMS}
    assert {p.suffix for p in paths} == {".pdf", ".png", ".svg"}


def test_the_validation_split_chooses_the_layer_of_lat(
    planted_run: RunDir,
) -> None:
    run_dir = planted_run

    summary = _summary(run_dir)
    assert summary["layer"] == 2
    assert summary["layer_source"] == "validation"


@pytest.mark.parametrize(
    ("val_keep", "layer"),
    [((0,), 1), ((1,), 2)],
    ids=["validation-peaks-first", "validation-peaks-second"],
)
def test_no_test_item_chooses_the_layer(
    tiny_gpt2: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    val_keep: tuple[int, ...],
    layer: int,
) -> None:
    def split_acts(
        loaded: LoadedModel,
        items: Sequence[LabeledItem],
        template: Any,
        chat: Any,
        *,
        layers: Sequence[int],
        bos: Any = None,
        batch_size: int = 8,
    ) -> ItemActivations:
        acts = _planted_acts(items, layers, loaded.d_model)
        data = acts.data.copy()
        for j, item in enumerate(items):
            split = item.example.split
            keep = {"train": (0, 1, 2), "val": val_keep, "test": (2,)}[split]
            for k in range(len(layers)):
                if k not in keep:
                    data[k, j, :2] = 0.0
                else:
                    data[k, j, :2] = acts.data[1, j, :2]
        return ItemActivations(acts.uids, acts.layers, data)

    monkeypatch.setattr(p9c_reliability, "collect_item_activations", split_acts)

    run_dir = _execute(tiny_gpt2, tmp_path)

    summary = _summary(run_dir)
    assert summary["layer"] == layer
    assert summary["layer_source"] == "validation"
    curve = run_dir.read_arrays("curves")["per_seed"][1, 0]
    assert curve[2] > curve[0]


def test_the_layer_of_the_config_wins(
    tiny_gpt2: LoadedModel, tmp_path: Path, planted: list[dict[str, Any]]
) -> None:
    run_dir = _execute(tiny_gpt2, tmp_path, layer=3)

    summary = _summary(run_dir)
    assert summary["layer"] == 3
    assert summary["layer_source"] == "config"


def test_a_layer_that_is_not_read_is_an_error(
    tiny_gpt2: LoadedModel, tmp_path: Path, planted: list[dict[str, Any]]
) -> None:
    with pytest.raises(ValueError, match="layer 3 is not among the layers"):
        _execute(tiny_gpt2, tmp_path, layers=(1, 2), layer=3)


def test_the_layers_of_the_config_restrict_the_reading(
    tiny_gpt2: LoadedModel, tmp_path: Path, planted: list[dict[str, Any]]
) -> None:
    run_dir = _execute(tiny_gpt2, tmp_path, layers=(2, 3))

    (call,) = planted
    assert call["layers"] == (2, 3)
    np.testing.assert_array_equal(
        run_dir.read_arrays("curves")["x"], [2.0, 3.0]
    )
    assert _summary(run_dir)["layer"] == 2


def test_the_activations_are_collected_once_for_the_union(
    tiny_gpt2: LoadedModel, tmp_path: Path, planted: list[dict[str, Any]]
) -> None:
    _execute(
        tiny_gpt2,
        tmp_path,
        lat_type="real_vs_generated",
        template="plain",
        bos="none",
        batch_size=4,
    )

    (call,) = planted
    uids = [i.example.uid for i in call["items"]]
    assert len(uids) == len(set(uids))
    pool = type_pools(_items())["real_vs_generated"]
    expected = {
        i.example.uid for d in DATASETS for i in _behavior(_items(), d)
    } | {i.example.uid for i in (*pool.positive, *pool.negative)}
    assert set(uids) == expected
    assert any(u.startswith("generated:") for u in uids)
    assert call["layers"] == (1, 2, 3)
    assert call["bos"] == "none"
    assert call["batch_size"] == 4
    assert call["template"] == p9_template("plain", DEFAULT_SHOTS[:2])
    assert call["chat"].kind in ("plain", "template")


def test_the_signals_are_scored_once_per_dataset_with_the_shots(
    tiny_gpt2: LoadedModel,
    tmp_path: Path,
    planted: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    real = p9c_reliability.item_signals

    def spy(
        loaded: LoadedModel, items: Sequence[LabeledItem], **kwargs: Any
    ) -> dict[str, np.ndarray]:
        calls.append({"uids": [i.example.uid for i in items], **kwargs})
        return real(loaded, items, **kwargs)

    monkeypatch.setattr(p9c_reliability, "item_signals", spy)

    _execute(tiny_gpt2, tmp_path, n_shots=1, batch_size=4)

    assert [c["uids"] for c in calls] == [
        [i.example.uid for i in _behavior(_items(), d)] for d in DATASETS
    ]
    assert all(c["shots"] == DEFAULT_SHOTS[:1] for c in calls)
    assert all(c["batch_size"] == 4 for c in calls)


def test_lat_is_the_train_fitted_reader_at_the_layer(
    tiny_gpt2: LoadedModel, tmp_path: Path, planted: list[dict[str, Any]]
) -> None:
    run_dir = _execute(tiny_gpt2, tmp_path, layer=2)

    pool = type_pools(_items())["obscure_real"]
    union = [*pool.positive, *pool.negative]
    popqa = _behavior(_items(), "popqa")
    acts = _planted_acts([*union, *popqa], (1, 2, 3), tiny_gpt2.d_model)
    reader = fit_type_reader(
        acts, pool, method="logistic", n_pairs=None, seed=0
    )
    test = _test(popqa)
    scores = reader.score(acts.data[:, acts.rows(test)])
    arrays = run_dir.read_arrays("signals_popqa")
    np.testing.assert_allclose(arrays["lat"], scores[1], rtol=1e-4, atol=1e-5)
    assert _row(_table(run_dir), "popqa", "lat")["auroc"] > 0.9


def test_lat_is_never_reoriented_on_correctness(
    tiny_gpt2: LoadedModel, tmp_path: Path, planted: list[dict[str, Any]]
) -> None:
    run_dir = _execute(
        tiny_gpt2,
        tmp_path,
        known_contrasts=False,
        lat_type="real_vs_generated",
        layer=2,
    )

    summary = _summary(run_dir)
    assert summary["lat_skipped"] is None
    assert summary["layer_source"] == "config"
    assert _row(_table(run_dir), "popqa", "lat")["auroc"] < 0.5


def test_the_closed_gate_skips_lat_and_uses_the_probe(
    tiny_gpt2: LoadedModel, tmp_path: Path, planted: list[dict[str, Any]]
) -> None:
    run_dir = _execute(tiny_gpt2, tmp_path, known_contrasts=False)

    summary = _summary(run_dir)
    assert summary["lat_skipped"] == (
        "the p9a gate admits no known versus unknown contrast for this model"
    )
    assert summary["layer_source"] == "probe_validation"
    assert summary["layer"] == 2
    assert summary["known_contrasts"] is False
    rows = _table(run_dir)
    assert all(r["predictor"] != "lat" for r in rows)
    assert "probe" in _predictors(rows, "popqa")
    incremental = _rows(run_dir, "table_incremental", INCREMENTAL_COLUMNS)
    assert [r["features"] for r in incremental if r["dataset"] == "popqa"] == [
        "confidence",
        "confidence+probe",
    ]
    curves = run_dir.read_json("curves")
    assert isinstance(curves, dict)
    assert curves["series"] == ["probe"]
    assert run_dir.read_arrays("curves")["per_seed"].shape == (1, 2, 3)
    series = run_dir.read_json("riskcov_popqa_controls")
    assert isinstance(series, dict)
    assert series["series"] == [
        "probe",
        "popularity",
        "template",
        "length",
        "name_logprob",
    ]


def test_an_unknown_type_contrast_brings_lat_back_under_a_closed_gate(
    tiny_gpt2: LoadedModel, tmp_path: Path, planted: list[dict[str, Any]]
) -> None:
    run_dir = _execute(
        tiny_gpt2,
        tmp_path,
        known_contrasts=False,
        lat_type="real_vs_generated",
    )

    summary = _summary(run_dir)
    assert summary["lat_skipped"] is None
    assert summary["layer_source"] == "validation"
    assert summary["lat_type"] == "real_vs_generated"
    assert "lat" in _predictors(_table(run_dir), "popqa")


def test_a_misspelled_contrast_skips_lat_without_raising(
    tiny_gpt2: LoadedModel, tmp_path: Path, planted: list[dict[str, Any]]
) -> None:
    run_dir = _execute(tiny_gpt2, tmp_path, lat_type="no_such_type")

    assert _summary(run_dir)["lat_skipped"] == (
        "no contrast named 'no_such_type' on these labels"
    )
    assert "lat" not in _predictors(_table(run_dir), "popqa")


def test_a_lat_contrast_without_matched_pairs_is_skipped_not_fatal(
    tiny_gpt2: LoadedModel, tmp_path: Path, planted: list[dict[str, Any]]
) -> None:
    run_dir = _execute(tiny_gpt2, tmp_path, items=_starved_items())

    summary = _summary(run_dir)
    assert summary["lat_skipped"] == "fewer than 8 matched train pairs"
    assert summary["layer_source"] == "probe_validation"
    assert "lat" not in _predictors(_table(run_dir), "popqa")


def test_the_curves_fit_one_reader_per_draw_on_the_test_items(
    tiny_gpt2: LoadedModel, tmp_path: Path, planted: list[dict[str, Any]]
) -> None:
    run_dir = _execute(tiny_gpt2, tmp_path, n_seeds=3, n_pairs=12)

    pool = type_pools(_items())["obscure_real"]
    popqa = _behavior(_items(), "popqa")
    acts = _planted_acts(
        [*pool.positive, *pool.negative, *popqa], (1, 2, 3), tiny_gpt2.d_model
    )
    test = _test(popqa)
    rows = acts.rows(test)
    per_seed = run_dir.read_arrays("curves")["per_seed"]
    assert per_seed.shape == (2, 3, 3)
    for s in range(3):
        reader = fit_type_reader(
            acts,
            pool,
            method="logistic",
            n_pairs=12,
            seed=draw_seed(0, s),
        )
        expected = auroc(reader.score(acts.data[:, rows]), _correct(test))
        np.testing.assert_allclose(per_seed[0, s], expected, atol=1e-5)


def test_the_probe_curve_is_the_test_auroc_of_a_train_fit_per_layer(
    planted_run: RunDir,
) -> None:
    run_dir = planted_run

    per_seed = run_dir.read_arrays("curves")["per_seed"]
    assert per_seed[1, 0, 1] > 0.9
    assert per_seed[1, 0, 0] < per_seed[1, 0, 1]


def test_a_dataset_without_both_outcomes_is_skipped(
    tiny_gpt2: LoadedModel, tmp_path: Path, planted: list[dict[str, Any]]
) -> None:
    items = [
        *_popqa(),
        *_open("nq_open"),
        *_fictional(),
        *(_wrong(i) for i in _open("triviaqa")),
    ]

    run_dir = _execute(tiny_gpt2, tmp_path, items=items)

    summary = _summary(run_dir)
    assert summary["skipped"] == {
        "triviaqa": "the train split does not hold both outcomes"
    }
    assert summary["datasets"] == ["popqa", "nq_open"]
    rows = _table(run_dir)
    assert {r["dataset"] for r in rows} == {"popqa", "nq_open"}
    assert not (run_dir.path / "signals_triviaqa.npz").exists()
    assert not (run_dir.path / "riskcov_triviaqa_outputs.json").exists()


@pytest.mark.parametrize("split", ["val", "test"])
def test_the_skip_reason_names_the_first_split_that_lacks_an_outcome(
    tiny_gpt2: LoadedModel,
    tmp_path: Path,
    planted: list[dict[str, Any]],
    split: str,
) -> None:
    items = [
        *_popqa(),
        *(
            _wrong(i) if i.example.split == split else i
            for i in _open("nq_open")
        ),
    ]

    run_dir = _execute(
        tiny_gpt2, tmp_path, items=items, datasets=("popqa", "nq_open")
    )

    assert _summary(run_dir)["skipped"] == {
        "nq_open": f"the {split} split does not hold both outcomes"
    }


def test_every_dataset_skipped_is_an_error(
    tiny_gpt2: LoadedModel, tmp_path: Path, planted: list[dict[str, Any]]
) -> None:
    items = [*_popqa(), *(_wrong(i) for i in _open("triviaqa"))]

    with pytest.raises(
        ValueError, match="no dataset holds both outcomes in every split"
    ) as error:
        _execute(tiny_gpt2, tmp_path, items=items, datasets=("triviaqa",))

    assert "the train split does not hold both outcomes" in str(error.value)
    assert not planted


def test_the_probe_needs_min_probe_train_items(
    tiny_gpt2: LoadedModel, tmp_path: Path, planted: list[dict[str, Any]]
) -> None:
    run_dir = _execute(tiny_gpt2, tmp_path, min_probe=50)

    summary = _summary(run_dir)
    assert summary["probe_skipped"] == {
        "triviaqa": "fewer than 50 train items",
        "nq_open": "fewer than 50 train items",
    }
    rows = _table(run_dir)
    assert "probe" in _predictors(rows, "popqa")
    assert "probe" not in _predictors(rows, "triviaqa")
    incremental = _rows(run_dir, "table_incremental", INCREMENTAL_COLUMNS)
    assert [
        r["features"] for r in incremental if r["dataset"] == "nq_open"
    ] == [
        "confidence",
        "confidence+lat",
    ]


def test_the_probe_fits_at_exactly_min_probe_train_items(
    tiny_gpt2: LoadedModel, tmp_path: Path, planted: list[dict[str, Any]]
) -> None:
    n_train = int(
        sum(i.example.split == "train" for i in _behavior(_items(), "nq_open"))
    )

    at = _execute(
        tiny_gpt2, tmp_path / "at", datasets=("nq_open",), min_probe=n_train
    )
    above = _execute(
        tiny_gpt2,
        tmp_path / "above",
        datasets=("nq_open",),
        min_probe=n_train + 1,
    )

    assert _summary(at)["probe_skipped"] == {}
    assert _summary(above)["probe_skipped"] == {
        "nq_open": f"fewer than {n_train + 1} train items"
    }


def test_a_probe_below_the_minimum_still_chooses_the_layer(
    tiny_gpt2: LoadedModel, tmp_path: Path, planted: list[dict[str, Any]]
) -> None:
    run_dir = _execute(
        tiny_gpt2, tmp_path, known_contrasts=False, min_probe=10_000
    )

    summary = _summary(run_dir)
    assert summary["layer_source"] == "probe_validation"
    assert summary["layer"] == 2
    assert set(summary["probe_skipped"]) == set(DATASETS)
    curves = run_dir.read_json("curves")
    assert isinstance(curves, dict)
    assert curves["series"] == ["probe"]


def test_popularity_is_oriented_on_the_train_items(
    planted_run: RunDir,
) -> None:
    run_dir = planted_run

    rows = _table(run_dir)
    assert _row(rows, "popqa", "popularity")["auroc"] > 0.75


def test_a_nan_takes_the_minimum_finite_train_value(
    planted_run: RunDir,
) -> None:
    run_dir = planted_run

    popqa = _behavior(_items(), "popqa")
    missing = [i for i in popqa if i.example.meta["popularity"] is None]
    rows = _table(run_dir)
    assert _row(rows, "popqa", "popularity")["n_filled"] == len(missing)
    empty = [i for i in popqa if not i.label.greedy]
    for name in ("answer_logprob", "mean_token_logprob", "p_true"):
        assert _row(rows, "popqa", name)["n_filled"] == len(empty)
    assert _row(rows, "popqa", "consistency")["n_filled"] == len(empty)
    assert _row(rows, "popqa", "length")["n_filled"] == 0
    test = _test(popqa)
    arrays = run_dir.read_arrays("signals_popqa")
    filled = [
        k for k, i in enumerate(test) if i.example.meta["popularity"] is None
    ]
    assert filled
    train_min = min(
        np.log10(i.example.meta["popularity"])
        for i in popqa
        if i.example.split == "train" and i.example.meta["popularity"]
    )
    np.testing.assert_allclose(
        np.abs(arrays["popularity"][filled]), abs(train_min)
    )
    train_empty = [
        i for i in popqa if i.example.split == "train" and not i.label.greedy
    ]
    assert train_empty
    test_empty = [k for k, i in enumerate(test) if not i.label.greedy]
    assert test_empty
    assert len(set(arrays["consistency"][test_empty])) == 1
    assert arrays["consistency"][test_empty[0]] == pytest.approx(0.2)


def test_a_signal_without_a_finite_train_value_is_dropped(
    tiny_gpt2: LoadedModel, tmp_path: Path, planted: list[dict[str, Any]]
) -> None:
    items = [
        dataclasses.replace(
            i,
            example=dataclasses.replace(
                i.example,
                meta={
                    k: v
                    for k, v in i.example.meta.items()
                    if k != "subject" or i.example.split == "test"
                },
            ),
        )
        for i in _popqa()
    ]

    run_dir = _execute(
        tiny_gpt2,
        tmp_path,
        items=[*items, *_open("nq_open")],
        datasets=("popqa", "nq_open"),
    )

    assert "name_logprob" not in _predictors(_table(run_dir), "popqa")


def test_template_scores_the_smoothed_train_accuracy_of_the_relation(
    planted_run: RunDir,
) -> None:
    assert _row(_table(planted_run), "popqa", "template")["auroc"] > 0.85


def _relations(run_dir: RunDir, dataset: str) -> list[dict[str, Any]]:
    return _rows(run_dir, f"relations_{dataset}", RELATION_COLUMNS)


def _pairs_within(items: Sequence[LabeledItem]) -> int:
    total = 0
    for relation in RELATIONS:
        rows = [i for i in items if i.example.meta["relation"] == relation]
        correct = _correct(rows)
        total += int(correct.sum()) * int((1 - correct).sum())
    return total


@pytest.mark.parametrize("dataset", ["triviaqa", "nq_open"])
def test_the_within_relation_columns_are_null_without_relations(
    default_run: RunDir, dataset: str
) -> None:
    rows = [r for r in _table(default_run) if r["dataset"] == dataset]

    assert rows
    for row in rows:
        assert [row[c] for c in WITHIN_COLUMNS] == [None] * 4


def test_popqa_rows_follow_the_within_relation_interval(
    default_run: RunDir,
) -> None:
    arrays = default_run.read_arrays("signals_popqa")
    pairs = _pairs_within(_test(_behavior(_items(), "popqa")))
    config = _config()
    father = arrays["relation"] == "father"

    for row in (r for r in _table(default_run) if r["dataset"] == "popqa"):
        scores = arrays[row["predictor"]]
        point, lo, hi = grouped_auroc_ci(
            scores,
            arrays["correct"],
            arrays["relation"],
            n_boot=config.n_boot,
            seed=config.seed,
        )
        assert row["auroc_within"] == pytest.approx(point)
        assert row["auroc_within"] == pytest.approx(
            float(auroc(scores[father], arrays["correct"][father]))
        )
        assert row["lo_within"] == pytest.approx(lo)
        assert row["hi_within"] == pytest.approx(hi)
        assert 0.0 <= row["lo_within"] <= row["hi_within"] <= 1.0
        assert row["n_pairs_within"] == pairs
    assert pairs == 25


def _blocked_items() -> list[LabeledItem]:
    """`_items` where PopQA relations come in contiguous blocks of 50."""
    popqa = [
        dataclasses.replace(
            i,
            example=dataclasses.replace(
                i.example,
                meta={
                    **i.example.meta,
                    "relation": RELATIONS[min(int(i.example.uid[6:]) // 50, 2)],
                },
            ),
        )
        for i in _popqa()
    ]
    return [*popqa, *_items()[len(popqa) :]]


def _thin_blocked_items() -> list[LabeledItem]:
    """Blocked relations where `occupation` has 3 correct, 7 wrong tests."""
    items = _blocked_items()
    thin = [
        i
        for i in _test(_behavior(items, "popqa"))
        if i.example.meta["relation"] == "occupation" and i.label.greedy_correct
    ][:2]
    flipped = {id(i) for i in thin}
    return [_wrong(i) if id(i) in flipped else i for i in items]


def test_the_within_auroc_pools_every_relation_by_its_pairs(
    tiny_gpt2: LoadedModel, tmp_path: Path, planted: list[dict[str, Any]]
) -> None:
    items = _thin_blocked_items()
    test = _test(_behavior(items, "popqa"))
    relation = np.array([i.example.meta["relation"] for i in test])
    correct = _correct(test)
    pairs = {
        r: int(correct[relation == r].sum())
        * int((1 - correct[relation == r]).sum())
        for r in RELATIONS
    }
    assert sorted(pairs.values()) == [21, 25, 25]

    run_dir = _execute(tiny_gpt2, tmp_path, items=items)

    arrays = run_dir.read_arrays("signals_popqa")
    for row in (r for r in _table(run_dir) if r["dataset"] == "popqa"):
        scores = arrays[row["predictor"]]
        expected = sum(
            pairs[r]
            * float(auroc(scores[relation == r], correct[relation == r]))
            for r in RELATIONS
        ) / sum(pairs.values())
        assert row["auroc_within"] == pytest.approx(expected)
        assert row["n_pairs_within"] == sum(pairs.values()) == 71


def test_the_within_interval_follows_the_config_seed(
    tiny_gpt2: LoadedModel, tmp_path: Path, planted: list[dict[str, Any]]
) -> None:
    run_dir = _execute(tiny_gpt2, tmp_path, seed=3)
    arrays = run_dir.read_arrays("signals_popqa")

    for row in (r for r in _table(run_dir) if r["dataset"] == "popqa"):
        _, lo, hi = grouped_auroc_ci(
            arrays[row["predictor"]],
            arrays["correct"],
            arrays["relation"],
            n_boot=_config().n_boot,
            seed=3,
        )
        assert row["lo_within"] == pytest.approx(lo)
        assert row["hi_within"] == pytest.approx(hi)


def test_the_template_control_is_exactly_chance_within_a_relation(
    default_run: RunDir,
) -> None:
    row = _row(_table(default_run), "popqa", "template")

    assert row["auroc_within"] == 0.5


def test_the_relations_file_holds_a_row_per_predictor_and_relation(
    default_run: RunDir,
) -> None:
    rows = _relations(default_run, "popqa")
    arrays = default_run.read_arrays("signals_popqa")
    predictors = _predictors(_table(default_run), "popqa")

    assert [(r["predictor"], r["relation"]) for r in rows] == [
        (p, r) for p in predictors for r in sorted(RELATIONS)
    ]
    for row in rows:
        member = arrays["relation"] == row["relation"]
        n_correct = int(arrays["correct"][member].sum())
        assert row["n_correct"] == n_correct
        assert row["n_error"] == int(member.sum()) - n_correct
        if row["relation"] == "father":
            scores = arrays[row["predictor"]][member]
            expected = auroc(scores, arrays["correct"][member])
            assert row["auroc"] == pytest.approx(float(expected))
        else:
            assert row["auroc"] is None


def test_a_relation_below_min_relation_class_has_no_auroc(
    tiny_gpt2: LoadedModel, tmp_path: Path, planted: list[dict[str, Any]]
) -> None:
    fits = _execute(tiny_gpt2, tmp_path / "five", min_relation_class=5)
    thin = _execute(tiny_gpt2, tmp_path / "six", min_relation_class=6)

    father = [r for r in _relations(fits, "popqa") if r["relation"] == "father"]
    assert father
    assert all(r["auroc"] is not None for r in father)
    assert all(r["n_correct"] == 5 and r["n_error"] == 5 for r in father)
    assert all(r["auroc"] is None for r in _relations(thin, "popqa"))


def test_the_signals_save_the_relation_of_each_test_item(
    default_run: RunDir,
) -> None:
    arrays = default_run.read_arrays("signals_popqa")
    test = _test(_behavior(_items(), "popqa"))

    assert arrays["relation"].dtype.kind == "U"
    assert arrays["relation"].tolist() == [
        i.example.meta["relation"] for i in test
    ]
    np.testing.assert_array_equal(arrays["correct"], _correct(test))
    for dataset in ("triviaqa", "nq_open"):
        assert "relation" not in default_run.read_arrays(f"signals_{dataset}")


def test_no_pair_inside_a_relation_gives_null_columns_and_zero_pairs(
    tiny_gpt2: LoadedModel, tmp_path: Path, planted: list[dict[str, Any]]
) -> None:
    run_dir = _execute(tiny_gpt2, tmp_path, items=_starved_items())

    rows = [r for r in _table(run_dir) if r["dataset"] == "popqa"]
    assert rows
    for row in rows:
        assert [row[c] for c in WITHIN_COLUMNS] == [None, None, None, 0]
    assert all(r["auroc"] is None for r in _relations(run_dir, "popqa"))


def test_a_dataset_without_a_within_pair_still_lists_every_relation_row(
    tiny_gpt2: LoadedModel, tmp_path: Path, planted: list[dict[str, Any]]
) -> None:
    run_dir = _execute(tiny_gpt2, tmp_path, items=_starved_items())

    predictors = _predictors(_table(run_dir), "popqa")
    rows = _relations(run_dir, "popqa")
    assert [(r["predictor"], r["relation"]) for r in rows] == [
        (p, r) for p in predictors for r in ("father", "mother")
    ]
    assert all(r["auroc"] is None for r in rows)


def test_a_test_item_without_a_string_relation_makes_the_dataset_non_relational(
    tiny_gpt2: LoadedModel, tmp_path: Path, planted: list[dict[str, Any]]
) -> None:
    items = _items()
    victim = next(
        i
        for i in items
        if i.group == "popqa"
        and i.label.basis == "behavior"
        and i.example.split == "test"
    )
    meta: dict[str, MetaValue] = {**victim.example.meta, "relation": 7}
    stripped = dataclasses.replace(
        victim, example=dataclasses.replace(victim.example, meta=meta)
    )

    run_dir = _execute(
        tiny_gpt2,
        tmp_path,
        items=[stripped if i is victim else i for i in items],
    )

    for row in (r for r in _table(run_dir) if r["dataset"] == "popqa"):
        assert [row[c] for c in WITHIN_COLUMNS] == [None] * 4
    assert "relations_popqa.json" not in _files(run_dir)
    assert "relation" not in run_dir.read_arrays("signals_popqa")


def test_a_min_relation_class_of_one_is_valid(
    tiny_gpt2: LoadedModel, tmp_path: Path, planted: list[dict[str, Any]]
) -> None:
    run_dir = _execute(tiny_gpt2, tmp_path, min_relation_class=1)

    father = [
        r for r in _relations(run_dir, "popqa") if r["relation"] == "father"
    ]
    assert father
    assert all(r["auroc"] is not None for r in father)


def test_a_min_relation_class_below_one_fails_before_any_forward_pass(
    tiny_gpt2: LoadedModel, tmp_path: Path, planted: list[dict[str, Any]]
) -> None:
    with pytest.raises(ValueError, match=r"min_relation_class.*0"):
        _execute(tiny_gpt2, tmp_path, min_relation_class=0)

    assert not planted


def test_noise_activations_add_nothing_beyond_the_confidence(
    tiny_gpt2: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        p9c_reliability, "collect_item_activations", _noise_collect(512)
    )
    items = [
        *_popqa(600),
        *_open("triviaqa", 300),
        *_open("nq_open", 300),
        *_fictional(100),
    ]

    run_dir = _execute(tiny_gpt2, tmp_path, items=items)

    rows = _rows(run_dir, "table_incremental", INCREMENTAL_COLUMNS)
    probe = [r for r in rows if r["features"] == "confidence+probe"]
    lat = [
        r
        for r in rows
        if r["features"] == "confidence+lat" and r["dataset"] == "popqa"
    ]
    assert len(probe) == 3
    assert len(lat) == 1
    for row in probe:
        assert row["lo"] <= 0.0 <= row["hi"], row
        assert abs(row["diff_vs_confidence"]) < 0.05, row
    assert abs(lat[0]["diff_vs_confidence"]) < 0.02, lat[0]


def test_the_stacker_is_fitted_on_the_validation_rows(
    tiny_gpt2: LoadedModel,
    tmp_path: Path,
    planted: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    real = p9c_reliability.stack_scores

    def spy(
        train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, **kw: Any
    ) -> np.ndarray:
        calls.append((train_x, train_y, test_x))
        return real(train_x, train_y, test_x, **kw)

    monkeypatch.setattr(p9c_reliability, "stack_scores", spy)

    _execute(tiny_gpt2, tmp_path, datasets=("popqa",))

    popqa = _behavior(_items(), "popqa")
    val = [i for i in popqa if i.example.split == "val"]
    test = _test(popqa)
    first = item_signals(
        tiny_gpt2, popqa, shots=DEFAULT_SHOTS[:2], batch_size=16
    )["first_token_prob"]
    split = np.array([i.example.split for i in popqa])
    assert len(calls) == 3
    for train_x, train_y, test_x in calls:
        np.testing.assert_array_equal(train_y, _correct(val))
        assert train_x.shape[0] == len(val)
        assert test_x.shape[0] == len(test)
        np.testing.assert_allclose(
            train_x[:, 0], first[split == "val"], atol=1e-5
        )
        np.testing.assert_allclose(
            test_x[:, 0], first[split == "test"], atol=1e-5
        )


def _firm_on_val(items: Sequence[LabeledItem]) -> list[LabeledItem]:
    firm = []
    for item in items:
        if item.example.split == "val":
            greedy = "gold" if item.label.greedy_correct else "other"
            label = dataclasses.replace(
                item.label, greedy=greedy, samples=(greedy,) * 5
            )
            item = dataclasses.replace(item, label=label)
        firm.append(item)
    return firm


def _without_samples(items: Sequence[LabeledItem]) -> list[LabeledItem]:
    return [
        dataclasses.replace(
            item, label=dataclasses.replace(item.label, samples=())
        )
        for item in items
    ]


def test_a_confidence_column_constant_on_the_validation_rows_gets_no_weight(
    tiny_gpt2: LoadedModel, tmp_path: Path, planted: list[dict[str, Any]]
) -> None:
    items = _firm_on_val(_items())

    constant = _execute(
        tiny_gpt2, tmp_path / "constant", items=items, datasets=("popqa",)
    )
    dropped = _execute(
        tiny_gpt2,
        tmp_path / "dropped",
        items=_without_samples(items),
        datasets=("popqa",),
    )

    assert "consistency" in _predictors(_table(constant), "popqa")
    assert "consistency" not in _predictors(_table(dropped), "popqa")
    rows = _rows(constant, "table_incremental", INCREMENTAL_COLUMNS)
    reference = _rows(dropped, "table_incremental", INCREMENTAL_COLUMNS)
    assert [r["features"] for r in rows] == [r["features"] for r in reference]
    for row, expected in zip(rows, reference, strict=True):
        for column in ("auroc", "diff_vs_confidence", "lo", "hi"):
            assert np.isfinite(row[column]), row
            assert row[column] == pytest.approx(expected[column], abs=1e-6)


def test_the_probe_is_fitted_on_the_train_rows_only(
    tiny_gpt2: LoadedModel, planted_run: RunDir
) -> None:
    popqa = _behavior(_items(), "popqa")
    train = [i for i in popqa if i.example.split == "train"]
    test = _test(popqa)
    acts = _planted_acts(popqa, (1, 2, 3), tiny_gpt2.d_model)
    probe = RepReader.fit(
        "logistic",
        acts.data[1:2, acts.rows(train)],
        (2,),
        labels=_correct(train),
    )
    expected = probe.score(acts.data[1:2, acts.rows(test)])[0]

    arrays = planted_run.read_arrays("signals_popqa")

    assert _summary(planted_run)["layer"] == 2
    np.testing.assert_allclose(arrays["probe"], expected, rtol=1e-4, atol=1e-5)


def test_a_control_follows_the_train_orientation_not_the_test_one(
    tiny_gpt2: LoadedModel, tmp_path: Path, planted: list[dict[str, Any]]
) -> None:
    items = [*_flip_test_popularity(_popqa()), *_open("nq_open")]

    run_dir = _execute(
        tiny_gpt2, tmp_path, items=items, datasets=("popqa", "nq_open")
    )

    assert _row(_table(run_dir), "popqa", "popularity")["auroc"] < 0.5


def test_the_controls_are_oriented_on_the_train_rows(
    tiny_gpt2: LoadedModel,
    tmp_path: Path,
    planted: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    real = p9c_reliability.orient_on_train

    def spy(
        train_scores: np.ndarray,
        train_labels: np.ndarray,
        test_scores: np.ndarray,
    ) -> np.ndarray:
        calls.append((train_scores, train_labels, test_scores))
        return real(train_scores, train_labels, test_scores)

    monkeypatch.setattr(p9c_reliability, "orient_on_train", spy)

    _execute(tiny_gpt2, tmp_path, datasets=("popqa",))

    popqa = _behavior(_items(), "popqa")
    train = np.array([i.example.split == "train" for i in popqa])
    assert len(calls) == 3
    for train_scores, train_labels, column in calls:
        assert column.shape == (len(popqa),)
        assert len(np.unique(column)) > 2
        np.testing.assert_array_equal(train_scores, column[train])
        np.testing.assert_array_equal(train_labels, _correct(popqa)[train])


@pytest.mark.parametrize("known_contrasts", [True, False])
def test_the_all_layer_probe_is_fitted_once(
    tiny_gpt2: LoadedModel,
    tmp_path: Path,
    planted: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    known_contrasts: bool,
) -> None:
    shapes: list[tuple[int, ...]] = []
    real = p9c_reliability.RepReader.fit

    def spy(method: Any, h: np.ndarray, layers: Any, **kw: Any) -> Any:
        shapes.append(h.shape[:2])
        return real(method, h, layers, **kw)

    monkeypatch.setattr(
        p9c_reliability, "RepReader", types.SimpleNamespace(fit=spy)
    )
    n_train = sum(
        i.example.split == "train" for i in _behavior(_items(), "popqa")
    )

    _execute(tiny_gpt2, tmp_path, known_contrasts=known_contrasts)

    assert shapes.count((3, n_train)) == 1


def test_execute_is_deterministic(
    tiny_gpt2: LoadedModel, tmp_path: Path, planted: list[dict[str, Any]]
) -> None:
    first = _execute(tiny_gpt2, tmp_path / "a")
    second = _execute(tiny_gpt2, tmp_path / "b")

    assert first.read_json("table") == second.read_json("table")
    assert first.read_json("table_incremental") == second.read_json(
        "table_incremental"
    )
    np.testing.assert_array_equal(
        first.read_arrays("riskcov_popqa_outputs")["lo"],
        second.read_arrays("riskcov_popqa_outputs")["lo"],
    )


def test_a_dataset_without_behavior_items_has_its_own_reason(
    tiny_gpt2: LoadedModel, tmp_path: Path, planted: list[dict[str, Any]]
) -> None:
    run_dir = _execute(
        tiny_gpt2, tmp_path, datasets=("popqa", "triviaqa", "falseqa")
    )

    assert _summary(run_dir)["skipped"] == {"falseqa": "no behavior items"}


def test_a_group_of_construction_items_has_no_behavior_items(
    tiny_gpt2: LoadedModel, tmp_path: Path, planted: list[dict[str, Any]]
) -> None:
    items = [*_construction(), *_open("nq_open"), *_fictional()]

    run_dir = _execute(
        tiny_gpt2,
        tmp_path,
        items=items,
        datasets=("popqa", "nq_open"),
        known_contrasts=False,
    )

    assert _summary(run_dir)["skipped"] == {"popqa": "no behavior items"}


def test_a_dataset_named_twice_is_rejected(
    tiny_gpt2: LoadedModel, tmp_path: Path, planted: list[dict[str, Any]]
) -> None:
    with pytest.raises(ValueError, match=r"duplicate.*popqa"):
        _execute(tiny_gpt2, tmp_path, datasets=("popqa", "triviaqa", "popqa"))

    assert not planted


def test_run_reads_the_labels_and_the_gate(
    tiny_gpt2: LoadedModel,
    tmp_path: Path,
    planted: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = _labels_run(tmp_path, _items(), known_contrasts=False)
    asked: list[str] = []

    def fake_load(name: str) -> LoadedModel:
        asked.append(name)
        return tiny_gpt2

    monkeypatch.setattr(p9c_reliability, "load_model", fake_load)
    config = _config(labels_run=str(labels.path))
    run_dir = RunDir.create(config, out_root=tmp_path)

    run(config, run_dir)

    assert asked == ["tiny"]
    summary = _summary(run_dir)
    assert summary["known_contrasts"] is False
    assert summary["layer_source"] == "probe_validation"


def test_run_refuses_an_unfinished_labeling_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = _labels_run(tmp_path, _items(), finish=False)

    def never(name: str) -> LoadedModel:
        raise AssertionError("the model must not load first")

    monkeypatch.setattr(p9c_reliability, "load_model", never)
    config = _config(labels_run=str(labels.path))
    run_dir = RunDir.create(config, out_root=tmp_path)

    with pytest.raises(ValueError, match="not a finished run"):
        run(config, run_dir)


def test_run_refuses_labels_made_with_other_shots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = _labels_run(tmp_path, _items(), n_shots=15)

    def never(name: str) -> LoadedModel:
        raise AssertionError("the model must not load first")

    monkeypatch.setattr(p9c_reliability, "load_model", never)
    config = _config(labels_run=str(labels.path), n_shots=2)
    run_dir = RunDir.create(config, out_root=tmp_path)

    with pytest.raises(ValueError, match=r"n_shots.*15.*2"):
        run(config, run_dir)


def test_run_refuses_labels_made_for_another_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = _labels_run(tmp_path, _items())

    def never(name: str) -> LoadedModel:
        raise AssertionError("the model must not load first")

    monkeypatch.setattr(p9c_reliability, "load_model", never)
    config = _config(labels_run=str(labels.path), model="gpt2")
    run_dir = RunDir.create(config, out_root=tmp_path)

    with pytest.raises(ValueError, match="labeled 'tiny', not 'gpt2'"):
        run(config, run_dir)


@pytest.mark.slow
def test_p9c_on_gpt2_reads_a_synthetic_pool(tmp_path: Path) -> None:
    items = [*_popqa(300), *_open("triviaqa", 120), *_fictional()]
    config = Config(
        labels_run="runs/p9a",
        datasets=("popqa", "triviaqa"),
        n_seeds=2,
        n_pairs=16,
        min_pairs=8,
        n_boot=100,
    )
    run_dir = RunDir.create(config, out_root=tmp_path)
    loaded = load_model("gpt2")
    start = time.perf_counter()
    execute(config, run_dir, loaded, items, known_contrasts=True)
    seconds = time.perf_counter() - start

    summary = _summary(run_dir)
    print(summary["layer"], summary["layer_source"], summary["skipped"])
    print(f"{seconds / len(items):.2f} seconds per item")
    assert summary["datasets"] == ["popqa", "triviaqa"]
