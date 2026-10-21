"""Tests for the P9b reading experiment."""

from collections.abc import Sequence
import dataclasses
from pathlib import Path
import runpy
import time
from typing import Any

import numpy as np
import pytest

from tomography.cli import load_experiment
from tomography.datasets.records import QAExample
from tomography.datasets.records import Split
from tomography.experiments import p9b_reading
from tomography.experiments.base import RunDir
from tomography.experiments.p9b_reading import Config
from tomography.experiments.p9b_reading import execute
from tomography.experiments.p9b_reading import plot
from tomography.experiments.p9b_reading import run
from tomography.experiments.p9b_reading import usable_pools
from tomography.knowledge import DEFAULT_SHOTS
from tomography.knowledge import ItemLabel
from tomography.knowledge import Label
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.p9.pool import LabeledItem
from tomography.p9.reading import READING_METHODS
from tomography.p9.reading import ItemActivations
from tomography.p9.reading import ReadingCube
from tomography.p9.reading import collect_item_activations
from tomography.p9.reading import p9_template
from tomography.p9.reading import reading_cube
from tomography.p9.reading import surface_baselines
from tomography.p9.reading import within_relation_pairs
from tomography.p9.unknowns import UNKNOWN_TYPES
from tomography.p9.unknowns import type_pools
from tomography.reading.directions import Method
from tomography.reading.metrics import bootstrap_ci
from tomography.reading.reader import RepReader

# The planted pool and its labels run are shared with the V1 tests.
_INPUTS = runpy.run_path(str(Path(__file__).with_name("_p9b_inputs.py")))
SPLITS: tuple[Split, ...] = _INPUTS["SPLITS"]
LABELS: tuple[Label, ...] = _INPUTS["LABELS"]
RELATIONS: tuple[str, ...] = _INPUTS["RELATIONS"]
USABLE: list[str] = _INPUTS["USABLE"]
_items = _INPUTS["items"]
_labels_run = _INPUTS["labels_run"]
SKIPPED = ["kuq_false_assumption", "pre_vs_post_cutoff"]
RELATIONAL = [
    "fictional_synthal",
    "fictional_generated",
    "obscure_real",
    "obscure_real_pop",
    "post_cutoff",
    "real_vs_synthal",
    "real_vs_generated",
]
WITHIN_COLUMNS = [
    "logistic_within",
    "lo_within",
    "hi_within",
    "text_word_within",
    "text_char_within",
    "popularity_within",
    "n_pairs_within",
]
CURVE_STEMS = [
    *(f"curves_{name}" for name in USABLE),
    "curves_entity",
    "curves_question",
    "curves_unknown",
]


def _config(labels: RunDir, **overrides: Any) -> Config:
    base = Config(
        model="tiny",
        labels_run=str(labels.path),
        n_shots=2,
        n_seeds=2,
        n_pairs=16,
        min_pairs=8,
        min_eval=5,
        batch_size=16,
    )
    return dataclasses.replace(base, **overrides)


def _execute(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    *,
    known_contrasts: bool = True,
    **overrides: Any,
) -> RunDir:
    labels = _labels_run(tmp_path, _items(), known_contrasts=known_contrasts)
    config = _config(labels, **overrides)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(
        config,
        run_dir,
        tiny_arch,
        _items(),
        known_contrasts=known_contrasts,
    )
    return run_dir


_DEFAULT_RUNS: dict[str, RunDir] = {}


@pytest.fixture
def default_run(
    tiny_arch: LoadedModel, tmp_path_factory: pytest.TempPathFactory
) -> RunDir:
    """One run per architecture with the default settings."""
    key = type(tiny_arch.model).__name__
    if key not in _DEFAULT_RUNS:
        out = tmp_path_factory.mktemp(f"p9b-{key}")
        _DEFAULT_RUNS[key] = _execute(tiny_arch, out)
    return _DEFAULT_RUNS[key]


def _summary(run_dir: RunDir) -> dict[str, Any]:
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    return summary


def _files(run_dir: RunDir) -> list[str]:
    return sorted(p.name for p in run_dir.path.iterdir() if p.is_file())


def test_config_defaults_and_round_trip() -> None:
    config = Config(labels_run="runs/p9a")

    assert config.experiment == "p9b"
    assert config.model == "gpt2"
    assert config.template == "fewshot"
    assert config.n_shots == 15
    assert config.types is None
    assert config.methods == ("logistic", "pca_diff", "mean_diff", "random")
    assert config.methods == READING_METHODS
    assert config.preprocess == "center"
    assert config.n_seeds == 5
    assert config.n_pairs == 64
    assert config.layers is None
    assert config.layer is None
    assert config.min_pairs == 16
    assert config.min_eval == 10
    assert config.bos is None
    assert config.batch_size == 8
    assert Config.from_dict(config.to_dict()) == config


def test_config_round_trips_with_every_field_set() -> None:
    config = Config(
        labels_run="runs/p9a",
        types=("obscure_real", "real_vs_synthal"),
        methods=("pca_diff", "logistic"),
        layers=(2, 4),
        layer=4,
        bos="none",
    )

    assert Config.from_dict(config.to_dict()) == config


def test_config_requires_the_labels_run() -> None:
    with pytest.raises(ValueError, match="labels_run"):
        Config.from_dict({})


def test_config_rejects_a_method_outside_the_literal() -> None:
    data = Config(labels_run="runs/p9a").to_dict()
    data["methods"] = ["logistic", "magic"]

    with pytest.raises(ValueError, match="methods"):
        Config.from_dict(data)


def test_usable_pools_splits_the_contrasts_into_usable_and_skipped() -> None:
    config = Config(labels_run="runs/p9a", min_pairs=8, min_eval=5)

    usable, skipped = usable_pools(config, _items(), True)

    assert list(usable) == USABLE
    assert list(skipped) == SKIPPED


def test_the_cli_discovers_the_experiment() -> None:
    experiment = load_experiment("p9b")

    assert experiment.config_cls is Config
    assert experiment.run is run
    assert experiment.plot is plot


def test_execute_writes_the_files_of_a_run(
    tiny_arch: LoadedModel, default_run: RunDir
) -> None:
    run_dir = default_run

    names = _files(run_dir)
    expected = {"config.json", "meta.json", "table.json", "summary.json"}
    for stem in (
        *CURVE_STEMS,
        "matrix",
        "matrix_cosine",
        "transfer",
        "readers",
    ):
        expected |= {f"{stem}.npz", f"{stem}.json"}
    assert set(names) == expected
    assert len(names) == 40


def test_summary_records_the_run(
    tiny_arch: LoadedModel, default_run: RunDir
) -> None:
    run_dir = default_run

    summary = _summary(run_dir)
    assert set(summary) == {
        "layer",
        "layer_source",
        "template",
        "methods",
        "types",
        "contrasts",
        "known_contrasts",
        "skipped",
        "curve_stems",
        "matrix_stems",
        "best_layer_by_type",
    }
    assert summary["layer_source"] == "validation"
    assert summary["layer"] in range(1, tiny_arch.n_layers + 1)
    assert summary["template"] == "fewshot"
    assert summary["methods"] == list(READING_METHODS)
    assert summary["types"] == USABLE
    assert summary["known_contrasts"] is True
    assert summary["curve_stems"] == CURVE_STEMS
    assert summary["matrix_stems"] == ["matrix", "matrix_cosine"]
    assert summary["contrasts"] == {
        name: "unknown_type" if name in UNKNOWN_TYPES else "knowledge"
        for name in USABLE
    }
    assert list(summary["skipped"]) == SKIPPED
    for reason in summary["skipped"].values():
        assert reason == "fewer than 8 train items on one side"
    assert set(summary["best_layer_by_type"]) == set(USABLE)
    for layer in summary["best_layer_by_type"].values():
        assert layer in range(1, tiny_arch.n_layers + 1)


def test_the_arrays_have_the_documented_shapes(
    tiny_arch: LoadedModel, default_run: RunDir
) -> None:
    run_dir = default_run
    n_types, n_layers, d = len(USABLE), tiny_arch.n_layers, tiny_arch.d_model

    transfer = run_dir.read_arrays("transfer")
    assert transfer["layers"].tolist() == list(range(1, n_layers + 1))
    assert transfer["layers"].dtype == np.int64
    assert transfer["auroc"].shape == (2, n_layers, n_types, n_types)
    assert transfer["cosine"].shape == (n_layers, n_types, n_types)
    assert transfer["val"].shape == (4, n_types, 2, n_layers)
    assert transfer["test"].shape == (4, n_types, 2, n_layers)
    sidecar = run_dir.read_json("transfer")
    assert sidecar == {"types": USABLE, "methods": list(READING_METHODS)}
    for t in range(n_types):
        np.testing.assert_allclose(
            transfer["auroc"][:, :, t, t], transfer["test"][0, t], atol=1e-12
        )
    np.testing.assert_allclose(
        transfer["cosine"], transfer["cosine"].transpose(0, 2, 1), atol=1e-6
    )
    assert np.abs(transfer["cosine"]).max() <= 1.0 + 1e-6

    readers = run_dir.read_arrays("readers")
    assert readers["layers"].tolist() == list(range(1, n_layers + 1))
    assert readers["directions"].shape == (n_types, n_layers, d)
    assert readers["means"].shape == (n_types, n_layers, d)
    assert readers["signs"].shape == (n_types, n_layers)
    np.testing.assert_allclose(
        np.linalg.norm(readers["directions"], axis=-1), 1.0, atol=1e-5
    )
    assert run_dir.read_json("readers") == {"types": USABLE}


def _record_cubes(monkeypatch: pytest.MonkeyPatch) -> list[ReadingCube]:
    cubes: list[ReadingCube] = []

    def spy(*args: Any, **kw: Any) -> ReadingCube:
        cubes.append(reading_cube(*args, **kw))
        return cubes[-1]

    monkeypatch.setattr(p9b_reading, "reading_cube", spy)
    return cubes


@pytest.mark.parametrize("headline", ["logistic", "pca_diff"])
def test_the_saved_readers_score_like_the_fitted_ones(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    headline: Method,
) -> None:
    cubes = _record_cubes(monkeypatch)

    run_dir = _execute(tiny_arch, tmp_path, methods=(headline,))

    saved = run_dir.read_arrays("readers")
    (cube,) = cubes
    h = np.random.default_rng(0).standard_normal(
        (len(saved["layers"]), 6, saved["directions"].shape[-1])
    )
    for t, name in enumerate(cube.types):
        rebuilt = RepReader.from_arrays(
            headline,
            saved["layers"].tolist(),
            saved["directions"][t],
            saved["means"][t],
            saved["signs"][t],
        )
        np.testing.assert_allclose(
            rebuilt.score(h), cube.readers[name].score(h), atol=1e-5
        )


def test_the_saved_signs_are_those_of_the_fitted_readers(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cubes = _record_cubes(monkeypatch)

    run_dir = _execute(tiny_arch, tmp_path, methods=("pca_diff",))

    saved = run_dir.read_arrays("readers")
    (cube,) = cubes
    assert (saved["signs"] == -1).any()
    for t, name in enumerate(cube.types):
        np.testing.assert_array_equal(
            saved["signs"][t], cube.readers[name].signs
        )


def test_each_contrast_has_a_curve_file_of_the_methods(
    tiny_arch: LoadedModel, default_run: RunDir
) -> None:
    run_dir = default_run
    n_layers = tiny_arch.n_layers
    transfer = run_dir.read_arrays("transfer")

    for t, name in enumerate(USABLE):
        arrays = run_dir.read_arrays(f"curves_{name}")
        sidecar = run_dir.read_json(f"curves_{name}")
        assert isinstance(sidecar, dict)
        assert arrays["x"].tolist() == list(range(1, n_layers + 1))
        assert arrays["per_seed"].shape == (4, 2, n_layers)
        np.testing.assert_allclose(arrays["per_seed"], transfer["test"][:, t])
        assert sidecar["series"] == list(READING_METHODS)
        assert sidecar["chance"] == 0.5
        assert sidecar["n_layers"] == n_layers
        want = (
            "AUROC (unknown-type contrast)"
            if name in UNKNOWN_TYPES
            else "AUROC (known vs unknown)"
        )
        assert sidecar["ylabel"] == want


def test_the_panels_draw_the_first_method_of_their_contrasts(
    tiny_arch: LoadedModel, default_run: RunDir
) -> None:
    run_dir = default_run
    test = run_dir.read_arrays("transfer")["test"]
    panels = {
        "entity": USABLE[:6],
        "question": USABLE[6:9],
        "unknown": USABLE[9:],
    }

    for panel, names in panels.items():
        arrays = run_dir.read_arrays(f"curves_{panel}")
        sidecar = run_dir.read_json(f"curves_{panel}")
        assert isinstance(sidecar, dict)
        assert sidecar["series"] == names
        rows = [USABLE.index(n) for n in names]
        np.testing.assert_allclose(arrays["per_seed"], test[0][rows])
        assert sidecar["chance"] == 0.5
        want = (
            "AUROC (unknown-type contrast)"
            if panel == "unknown"
            else "AUROC (known vs unknown)"
        )
        assert sidecar["ylabel"] == want


def test_the_matrices_hold_the_transfer_and_the_cosines_at_the_layer(
    tiny_arch: LoadedModel, default_run: RunDir
) -> None:
    run_dir = default_run
    summary = _summary(run_dir)
    transfer = run_dir.read_arrays("transfer")
    k = transfer["layers"].tolist().index(summary["layer"])

    matrix = run_dir.read_arrays("matrix")["values"]
    sidecar = run_dir.read_json("matrix")
    assert isinstance(sidecar, dict)
    assert sidecar["rows"] == USABLE
    assert sidecar["cols"] == USABLE
    assert f"layer {summary['layer']}" in sidecar["title"]
    np.testing.assert_allclose(matrix, transfer["auroc"][:, k].mean(axis=0))
    cosine = run_dir.read_arrays("matrix_cosine")["values"]
    sidecar = run_dir.read_json("matrix_cosine")
    assert isinstance(sidecar, dict)
    assert sidecar["rows"] == USABLE
    assert f"layer {summary['layer']}" in sidecar["title"]
    np.testing.assert_allclose(cosine, transfer["cosine"][k])


def test_the_validation_split_chooses_the_layer(
    tiny_arch: LoadedModel, default_run: RunDir
) -> None:
    run_dir = default_run

    summary = _summary(run_dir)
    val = run_dir.read_arrays("transfer")["val"]
    mean = val[0].mean(axis=(0, 1))
    assert summary["layer"] == int(np.argmax(mean)) + 1
    for t, name in enumerate(USABLE):
        best = int(np.argmax(val[0, t].mean(axis=0))) + 1
        assert summary["best_layer_by_type"][name] == best


def test_the_table_reports_the_auroc_at_the_layer(
    tiny_arch: LoadedModel, default_run: RunDir
) -> None:
    run_dir = default_run
    summary = _summary(run_dir)
    transfer = run_dir.read_arrays("transfer")
    k = transfer["layers"].tolist().index(summary["layer"])
    pools = type_pools(_items())

    table = run_dir.read_json("table")
    assert isinstance(table, dict)
    assert table["columns"] == [
        "type",
        "contrast",
        "n_positive_train",
        "n_negative_train",
        "n_positive_test",
        "n_negative_test",
        "logistic",
        "lo",
        "hi",
        "pca_diff",
        "mean_diff",
        "random",
        "text_word",
        "text_char",
        "popularity",
        *WITHIN_COLUMNS,
    ]
    assert [row[0] for row in table["rows"]] == USABLE
    for t, row in enumerate(table["rows"]):
        row_dict = dict(zip(table["columns"], row, strict=True))
        name = row_dict["type"]
        pool = pools[name]
        assert row_dict["contrast"] == summary["contrasts"][name]
        assert row_dict["n_positive_train"] == len(pool.split("train")[0])
        assert row_dict["n_negative_train"] == len(pool.split("train")[1])
        assert row_dict["n_positive_test"] == len(pool.split("test")[0])
        assert row_dict["n_negative_test"] == len(pool.split("test")[1])
        for m, method in enumerate(READING_METHODS):
            assert row_dict[method] == pytest.approx(
                transfer["test"][m, t, :, k].mean()
            )
        assert row_dict["lo"] <= row_dict["logistic"] <= row_dict["hi"]


def test_the_table_holds_the_surface_baselines_of_every_contrast(
    tiny_arch: LoadedModel, default_run: RunDir
) -> None:
    table = default_run.read_json("table")
    assert isinstance(table, dict)
    pools = type_pools(_items())

    for row in table["rows"]:
        row_dict = dict(zip(table["columns"], row, strict=True))
        name = row_dict["type"]
        want = surface_baselines(pools[name])
        assert 0.0 <= row_dict["text_word"] <= 1.0, name
        assert 0.0 <= row_dict["text_char"] <= 1.0, name
        assert row_dict["text_word"] == pytest.approx(want["text_word"]), name
        assert row_dict["text_char"] == pytest.approx(want["text_char"]), name
        if name in ("obscure_real", "obscure_real_pop"):
            assert row_dict["popularity"] == pytest.approx(want["popularity"])
            assert isinstance(row_dict["popularity"], float)
        else:
            assert row_dict["popularity"] is None, name


def test_the_within_arrays_are_saved_next_to_the_pooled_ones(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cubes = _record_cubes(monkeypatch)

    run_dir = _execute(tiny_arch, tmp_path)

    (cube,) = cubes
    transfer = run_dir.read_arrays("transfer")
    assert set(transfer) == {
        "auroc",
        "cosine",
        "val",
        "test",
        "val_within",
        "test_within",
        "layers",
    }
    for name in ("val_within", "test_within"):
        assert transfer[name].shape == transfer["test"].shape
        np.testing.assert_array_equal(
            transfer[name], getattr(cube, name), strict=True
        )
        for t, type_name in enumerate(USABLE):
            held = transfer[name][:, t]
            if type_name in RELATIONAL:
                assert np.isfinite(held).all(), type_name
            else:
                assert np.isnan(held).all(), type_name


def test_the_within_values_leave_the_pooled_ones_and_the_layer_alone(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cubes = _record_cubes(monkeypatch)

    run_dir = _execute(tiny_arch, tmp_path)

    (cube,) = cubes
    transfer = run_dir.read_arrays("transfer")
    np.testing.assert_array_equal(transfer["val"], cube.val)
    np.testing.assert_array_equal(transfer["test"], cube.test)
    assert (
        _summary(run_dir)["layer"]
        == int(np.argmax(cube.val[0].mean(axis=(0, 1)))) + 1
    )


def test_the_table_holds_the_within_relation_columns(
    tiny_arch: LoadedModel, default_run: RunDir
) -> None:
    summary = _summary(default_run)
    transfer = default_run.read_arrays("transfer")
    k = transfer["layers"].tolist().index(summary["layer"])
    pools = type_pools(_items())
    seed = Config(labels_run="x").seed

    table = default_run.read_json("table")
    assert isinstance(table, dict)
    for t, row in enumerate(table["rows"]):
        row_dict = dict(zip(table["columns"], row, strict=True))
        name = row_dict["type"]
        if name not in RELATIONAL:
            for column in WITHIN_COLUMNS:
                assert row_dict[column] is None, (name, column)
            continue
        held = transfer["test_within"][0, t, :, k]
        lo, hi = bootstrap_ci(held, n_boot=200, seed=seed)
        want = surface_baselines(pools[name])
        assert row_dict["logistic_within"] == pytest.approx(held.mean())
        assert row_dict["lo_within"] == pytest.approx(float(lo))
        assert row_dict["hi_within"] == pytest.approx(float(hi))
        assert row_dict["lo_within"] <= row_dict["logistic_within"]
        assert row_dict["logistic_within"] <= row_dict["hi_within"]
        for column in ("text_word_within", "text_char_within"):
            assert row_dict[column] == pytest.approx(want[column]), name
        if name in ("obscure_real", "obscure_real_pop"):
            assert row_dict["popularity_within"] == pytest.approx(
                want["popularity_within"]
            )
        else:
            assert row_dict["popularity_within"] is None, name
        assert row_dict["n_pairs_within"] == within_relation_pairs(
            pools[name], "test"
        )
        assert row_dict["n_pairs_within"] > 0


def test_the_within_column_follows_the_headline_method(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(tiny_arch, tmp_path, methods=("pca_diff", "logistic"))

    table = run_dir.read_json("table")
    assert isinstance(table, dict)
    assert "pca_diff_within" in table["columns"]
    assert "logistic_within" not in table["columns"]
    assert table["columns"][-7:] == [
        "pca_diff_within",
        "lo_within",
        "hi_within",
        "text_word_within",
        "text_char_within",
        "popularity_within",
        "n_pairs_within",
    ]


def test_a_relational_contrast_without_a_shared_relation_has_null_columns() -> (
    None
):
    pool = type_pools(_items())["obscure_real"]

    def apart(item: LabeledItem) -> LabeledItem:
        if item.example.split != "test":
            return item
        relation = "alpha" if item in pool.positive else "beta"
        meta = {**item.example.meta, "relation": relation}
        example = dataclasses.replace(item.example, meta=meta)
        return dataclasses.replace(item, example=example)

    pool = dataclasses.replace(
        pool,
        positive=tuple(apart(i) for i in pool.positive),
        negative=tuple(apart(i) for i in pool.negative),
    )
    cube = ReadingCube(
        types=(pool.name,),
        methods=("logistic",),
        layers=(1, 2),
        val=np.full((1, 1, 2, 2), 0.7),
        test=np.full((1, 1, 2, 2), 0.7),
        val_within=np.full((1, 1, 2, 2), np.nan),
        test_within=np.full((1, 1, 2, 2), np.nan),
        transfer=np.zeros((2, 2, 1, 1)),
        cosine=np.zeros((2, 1, 1)),
        readers={},
    )
    baselines = surface_baselines(pool)

    (row,) = p9b_reading._table_rows(
        cube, {pool.name: pool}, {pool.name: baselines}, 0, 0
    )

    assert row[-7:] == [None, None, None, None, None, None, 0]
    assert baselines["text_word_within"] is None


def test_a_baseline_that_cannot_work_fails_before_the_forward_pass(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def never(*args: Any, **kw: Any) -> ItemActivations:
        raise AssertionError("the forward pass must not start")

    def broken(pool: Any) -> dict[str, float | None]:
        raise ValueError("no baseline")

    monkeypatch.setattr(p9b_reading, "collect_item_activations", never)
    monkeypatch.setattr(p9b_reading, "surface_baselines", broken)

    with pytest.raises(ValueError, match="no baseline"):
        _execute(tiny_arch, tmp_path)


def test_the_layer_of_the_config_wins(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(tiny_arch, tmp_path, layer=2)

    summary = _summary(run_dir)
    assert summary["layer"] == 2
    assert summary["layer_source"] == "config"
    sidecar = run_dir.read_json("matrix")
    assert isinstance(sidecar, dict)
    assert "layer 2" in sidecar["title"]
    transfer = run_dir.read_arrays("transfer")
    np.testing.assert_allclose(
        run_dir.read_arrays("matrix")["values"],
        transfer["auroc"][:, 1].mean(axis=0),
    )


def test_the_layers_of_the_config_restrict_the_reading(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(tiny_arch, tmp_path, layers=(1, 3))

    transfer = run_dir.read_arrays("transfer")
    assert transfer["layers"].tolist() == [1, 3]
    assert transfer["test"].shape[-1] == 2
    assert _summary(run_dir)["layer"] in (1, 3)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"n_seeds": 0}, "n_seeds"),
        ({"methods": ()}, "methods"),
        ({"n_pairs": 1}, "n_pairs"),
        ({"min_eval": 0}, "min_eval"),
        ({"template": "nonsense"}, "unknown template"),
        ({"types": ("nonsense",)}, "unknown or empty types"),
        ({"min_pairs": 10_000}, "no usable type"),
        ({"layers": (1, 2), "layer": 3}, "layer 3"),
    ],
    ids=[
        "n_seeds",
        "methods",
        "n_pairs",
        "min_eval",
        "template",
        "types",
        "no_usable_type",
        "layer",
    ],
)
def test_a_setting_that_cannot_work_fails_before_the_forward_pass(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
    match: str,
) -> None:
    def never(*args: Any, **kw: Any) -> ItemActivations:
        raise AssertionError("the forward pass must not start")

    monkeypatch.setattr(p9b_reading, "collect_item_activations", never)

    with pytest.raises(ValueError, match=match):
        _execute(tiny_arch, tmp_path, **overrides)


def test_the_smallest_valid_settings_run(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(
        tiny_arch,
        tmp_path,
        n_seeds=1,
        n_pairs=2,
        min_eval=1,
        methods=("logistic",),
    )

    assert _summary(run_dir)["types"] == USABLE


def test_a_layer_that_is_not_read_is_an_error(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="layer 3"):
        _execute(tiny_arch, tmp_path, layers=(1, 2), layer=3)


def test_the_layer_is_chosen_on_the_validation_split(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def peaked(*args: Any, **kw: Any) -> ReadingCube:
        cube = reading_cube(*args, **kw)
        val = np.full_like(cube.val, 0.5)
        val[..., 0] = 0.9
        test = np.full_like(cube.test, 0.5)
        test[..., -1] = 0.9
        return dataclasses.replace(cube, val=val, test=test)

    monkeypatch.setattr(p9b_reading, "reading_cube", peaked)

    run_dir = _execute(tiny_arch, tmp_path)

    assert _summary(run_dir)["layer"] == 1


def test_types_restricts_the_contrasts(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    types = ("real_vs_generated", "obscure_real")

    run_dir = _execute(tiny_arch, tmp_path, types=types)

    summary = _summary(run_dir)
    assert summary["types"] == list(types)
    assert summary["curve_stems"] == [
        "curves_real_vs_generated",
        "curves_obscure_real",
        "curves_entity",
        "curves_unknown",
    ]
    assert summary["skipped"] == {}
    assert not (run_dir.path / "curves_question.npz").exists()
    assert not (run_dir.path / "curves_post_cutoff.npz").exists()


def test_triviaqa_known_alone_reads_the_same_dataset_contrast(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(tiny_arch, tmp_path, types=("triviaqa_known",))

    summary = _summary(run_dir)
    assert summary["types"] == ["triviaqa_known"]
    assert summary["skipped"] == {}
    assert summary["curve_stems"] == ["curves_triviaqa_known", "curves_entity"]
    table = run_dir.read_json("table")
    assert isinstance(table, dict)
    row = table["rows"][0]
    assert row[0] == "triviaqa_known"
    assert row[1] == "knowledge"


def test_a_skipped_type_in_types_is_reported_not_fatal(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(
        tiny_arch, tmp_path, types=("obscure_real", "pre_vs_post_cutoff")
    )

    summary = _summary(run_dir)
    assert summary["types"] == ["obscure_real"]
    assert list(summary["skipped"]) == ["pre_vs_post_cutoff"]


def test_a_contrast_without_a_shared_stratum_is_skipped_not_fatal(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    items = _items(post_awards_apart=True)
    labels = _labels_run(tmp_path, items)
    config = _config(labels)
    run_dir = RunDir.create(config, out_root=tmp_path)

    execute(config, run_dir, tiny_arch, items, known_contrasts=True)

    summary = _summary(run_dir)
    assert summary["skipped"]["post_cutoff"] == (
        "fewer than 8 matched train pairs"
    )
    assert "post_cutoff" not in summary["types"]
    assert summary["types"] == [n for n in USABLE if n != "post_cutoff"]


def test_a_type_that_does_not_exist_is_an_error(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="unknown or empty types") as caught:
        _execute(tiny_arch, tmp_path, types=("obscure_real", "nonsense"))

    assert "['nonsense']" in str(caught.value)


def test_no_usable_contrast_is_an_error_with_the_reasons(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="no usable type") as caught:
        _execute(tiny_arch, tmp_path, min_pairs=10_000)

    assert "obscure_real: fewer than 10000 train items" in str(caught.value)


def test_a_named_type_that_is_too_small_is_an_error(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="no usable type"):
        _execute(tiny_arch, tmp_path, types=("kuq_false_assumption",))


def test_the_labeled_pool_is_collected_once_per_distinct_item(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def spy(
        loaded: LoadedModel, items: Sequence[LabeledItem], *args: Any, **kw: Any
    ) -> ItemActivations:
        calls.append({"items": list(items), "args": args, **kw})
        return collect_item_activations(loaded, items, *args, **kw)

    monkeypatch.setattr(p9b_reading, "collect_item_activations", spy)

    _execute(tiny_arch, tmp_path, template="plain", bos="none", batch_size=4)

    (call,) = calls
    uids = [i.example.uid for i in call["items"]]
    assert len(uids) == len(set(uids))
    pools = type_pools(_items())
    used = {
        i.example.uid
        for name in USABLE
        for i in (*pools[name].positive, *pools[name].negative)
    }
    assert set(uids) == used
    assert call["layers"] == tuple(range(1, tiny_arch.n_layers + 1))
    assert call["bos"] == "none"
    assert call["batch_size"] == 4
    template, chat = call["args"]
    assert template == p9_template("plain")
    assert chat.kind in ("plain", "template")


def test_the_number_of_shots_builds_the_prompt(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Any] = []

    def spy(
        loaded: LoadedModel, items: Sequence[LabeledItem], *args: Any, **kw: Any
    ) -> ItemActivations:
        seen.append(args[0])
        return collect_item_activations(loaded, items, *args, **kw)

    monkeypatch.setattr(p9b_reading, "collect_item_activations", spy)

    _execute(tiny_arch, tmp_path, n_shots=1, types=("obscure_real",))

    assert seen == [p9_template("fewshot", DEFAULT_SHOTS[:1])]


def test_an_unknown_template_is_an_error(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="verbose"):
        _execute(tiny_arch, tmp_path, template="verbose")


def test_execute_is_deterministic(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    first = _execute(tiny_arch, tmp_path / "a")
    second = _execute(tiny_arch, tmp_path / "b")

    for stem in ("transfer", "readers", "matrix", "matrix_cosine"):
        one, two = first.read_arrays(stem), second.read_arrays(stem)
        assert one.keys() == two.keys()
        for key in one:
            np.testing.assert_array_equal(one[key], two[key], err_msg=stem)
    assert first.read_json("table") == second.read_json("table")


def test_the_gate_leaves_the_unknown_type_contrasts_only(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(tiny_arch, tmp_path, known_contrasts=False)

    summary = _summary(run_dir)
    assert summary["types"] == ["real_vs_synthal", "real_vs_generated"]
    assert summary["known_contrasts"] is False
    assert set(summary["contrasts"].values()) == {"unknown_type"}
    assert list(summary["skipped"]) == ["pre_vs_post_cutoff"]
    assert summary["curve_stems"] == [
        "curves_real_vs_synthal",
        "curves_real_vs_generated",
        "curves_unknown",
    ]
    assert not (run_dir.path / "curves_entity.npz").exists()
    assert not (run_dir.path / "curves_question.npz").exists()
    table = run_dir.read_json("table")
    assert isinstance(table, dict)
    assert {row[1] for row in table["rows"]} == {"unknown_type"}


@pytest.mark.parametrize(
    "name", ["obscure_real", "kuq_future_unknown", "triviaqa_known"]
)
def test_the_gate_refuses_a_knowledge_contrast_by_name(
    tiny_arch: LoadedModel, tmp_path: Path, name: str
) -> None:
    with pytest.raises(ValueError, match=name) as caught:
        _execute(tiny_arch, tmp_path, known_contrasts=False, types=(name,))

    assert "unknown_type" in str(caught.value)
    assert "gate" in str(caught.value)


def test_a_missing_type_names_no_gate_when_the_gate_is_open(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="nonsense") as caught:
        _execute(tiny_arch, tmp_path, types=("nonsense",))

    assert "gate" not in str(caught.value)


def test_plot_draws_every_curve_and_both_matrices(
    tiny_arch: LoadedModel, tmp_path: Path, default_run: RunDir
) -> None:
    run_dir = default_run

    paths = plot(run_dir, tmp_path / "figures")

    assert len(paths) == 48
    assert all(p.is_file() for p in paths)
    stems = {p.stem for p in paths}
    assert stems == {*CURVE_STEMS, "matrix", "matrix_cosine"}
    assert {p.suffix for p in paths} == {".pdf", ".png", ".svg"}


def test_plot_follows_the_curve_stems_of_the_summary(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(tiny_arch, tmp_path, known_contrasts=False)

    paths = plot(run_dir, tmp_path / "figures")

    assert len(paths) == 3 * (3 + 2)


def test_plot_scales_the_matrices(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _execute(tiny_arch, tmp_path, types=("obscure_real",))
    calls: list[dict[str, Any]] = []
    real = p9b_reading.plot_matrix_file

    def spy(*args: Any, **kwargs: Any) -> list[Path]:
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(p9b_reading, "plot_matrix_file", spy)

    plot(run_dir, tmp_path / "figures")

    transfer, cosine = calls
    assert transfer["stem"] == "matrix"
    assert transfer.get("kind", "sequential") == "sequential"
    assert (transfer["vmin"], transfer["vmax"]) == (0.0, 1.0)
    assert cosine["stem"] == "matrix_cosine"
    assert cosine["kind"] == "diverging"
    assert (cosine["vmin"], cosine["vmax"]) == (-1.0, 1.0)


def test_run_reads_the_labels_and_the_gate(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = _labels_run(tmp_path, _items(), known_contrasts=False)
    asked: list[str] = []

    def fake_load(name: str) -> LoadedModel:
        asked.append(name)
        return tiny_arch

    monkeypatch.setattr(p9b_reading, "load_model", fake_load)
    config = _config(labels)
    run_dir = RunDir.create(config, out_root=tmp_path)

    run(config, run_dir)

    assert asked == ["tiny"]
    summary = _summary(run_dir)
    assert summary["known_contrasts"] is False
    assert summary["types"] == ["real_vs_synthal", "real_vs_generated"]


def test_run_refuses_an_unfinished_labeling_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = _labels_run(tmp_path, _items(), finish=False)

    def never(name: str) -> LoadedModel:
        raise AssertionError("the model must not load first")

    monkeypatch.setattr(p9b_reading, "load_model", never)
    config = _config(labels)
    run_dir = RunDir.create(config, out_root=tmp_path)

    with pytest.raises(ValueError, match="not a finished run"):
        run(config, run_dir)


def test_run_refuses_a_labels_run_of_another_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = _labels_run(tmp_path, _items())

    def never(name: str) -> LoadedModel:
        raise AssertionError("the model must not load first")

    monkeypatch.setattr(p9b_reading, "load_model", never)
    config = _config(labels, model="gpt2")
    run_dir = RunDir.create(config, out_root=tmp_path)

    with pytest.raises(ValueError, match=r"'tiny'.*'gpt2'"):
        run(config, run_dir)


@pytest.mark.slow
def test_p9b_on_gpt2_reads_a_synthetic_pool(tmp_path: Path) -> None:
    items = _popqa_shaped(200)
    labels = _labels_run(tmp_path, items)
    config = Config(
        labels_run=str(labels.path),
        n_seeds=2,
        n_pairs=16,
        min_pairs=8,
        min_eval=5,
    )
    run_dir = RunDir.create(config, out_root=tmp_path)
    loaded = load_model("gpt2")
    start = time.perf_counter()
    execute(config, run_dir, loaded, items, known_contrasts=True)
    seconds = time.perf_counter() - start

    summary = _summary(run_dir)
    print(summary["layer"], summary["types"], summary["skipped"])
    print(f"{seconds / len(items):.2f} seconds per item")
    assert summary["types"] == ["obscure_real", "obscure_real_pop"]


def _popqa_shaped(n: int) -> list[LabeledItem]:
    items: list[LabeledItem] = []
    for i in range(n):
        uid = f"popqa:{i}"
        label = LABELS[i % 6]
        correct = label == "known"
        example = QAExample(
            uid=uid,
            question=f"Who is the {RELATIONS[i % 3]} of Person Number{i}?",
            answers=("gold",),
            kind="real",
            split=SPLITS[i % 5],
            meta={
                "source": "popqa",
                "relation": RELATIONS[i % 3],
                "popularity": 10 ** (1 + i % 7),
            },
        )
        item_label = ItemLabel(
            uid=uid,
            label=label,
            basis="behavior",
            greedy="gold" if correct else "other",
            greedy_correct=correct,
            samples=("gold",) * 5 if correct else ("other",) * 5,
            n_correct=5 if correct else 0,
        )
        items.append(LabeledItem(example, item_label))
    return items


def test_the_layer_is_chosen_on_the_pooled_validation_auroc(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def skewed(*args: Any, **kw: Any) -> ReadingCube:
        cube = reading_cube(*args, **kw)
        n = len(cube.layers)
        steps = np.arange(n, dtype=np.float64)
        falling = np.broadcast_to(-steps, cube.val.shape).copy()
        rising = np.broadcast_to(100.0 * steps, cube.val.shape).copy()
        return dataclasses.replace(cube, val=falling, val_within=rising)

    monkeypatch.setattr(p9b_reading, "reading_cube", skewed)

    run_dir = _execute(tiny_arch, tmp_path)

    assert _summary(run_dir)["layer"] == 1


def test_the_within_interval_is_drawn_as_the_pooled_one() -> None:
    pool = type_pools(_items())["obscure_real"]
    draws = np.random.default_rng(4).uniform(0.5, 0.9, size=(1, 1, 8, 2))
    cube = ReadingCube(
        types=(pool.name,),
        methods=("logistic",),
        layers=(1, 2),
        val=draws.copy(),
        test=draws.copy(),
        val_within=draws.copy(),
        test_within=draws.copy(),
        transfer=np.zeros((8, 2, 1, 1)),
        cosine=np.zeros((2, 1, 1)),
        readers={},
    )

    (row,) = p9b_reading._table_rows(
        cube, {pool.name: pool}, {pool.name: surface_baselines(pool)}, 1, 7
    )

    assert row[12:15] == row[6:9]
