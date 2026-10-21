"""Tests for the knowledge views experiment."""

from collections.abc import Callable
import dataclasses
import hashlib
import json
from pathlib import Path
import runpy
import shutil
from typing import Any

from matplotlib.figure import Figure
from matplotlib.text import Text
import numpy as np
import pytest
import torch

from tomography.analysis import pca_coordinates
from tomography.cli import load_experiment
from tomography.experiments import p9b_reading
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.sweep import chat_format_for
import tomography.experiments.v1_knowledge_views as v1_module
from tomography.experiments.v1_knowledge_views import COUNT_COLUMNS
from tomography.experiments.v1_knowledge_views import MAX_DISPLAYED
from tomography.experiments.v1_knowledge_views import READING_FILES
from tomography.experiments.v1_knowledge_views import CommittedReading
from tomography.experiments.v1_knowledge_views import Config
from tomography.experiments.v1_knowledge_views import Plan
from tomography.experiments.v1_knowledge_views import check_labels_run
from tomography.experiments.v1_knowledge_views import check_model
from tomography.experiments.v1_knowledge_views import check_reading
from tomography.experiments.v1_knowledge_views import compare_auroc
from tomography.experiments.v1_knowledge_views import compare_surface
from tomography.experiments.v1_knowledge_views import displayed_items
from tomography.experiments.v1_knowledge_views import displayed_uids
from tomography.experiments.v1_knowledge_views import execute
from tomography.experiments.v1_knowledge_views import load_reading
from tomography.experiments.v1_knowledge_views import panel_contrasts
from tomography.experiments.v1_knowledge_views import panel_figure
from tomography.experiments.v1_knowledge_views import panel_union
from tomography.experiments.v1_knowledge_views import pca_views
from tomography.experiments.v1_knowledge_views import plot
from tomography.experiments.v1_knowledge_views import prepare
from tomography.experiments.v1_knowledge_views import published_views
from tomography.experiments.v1_knowledge_views import refit_methods
from tomography.experiments.v1_knowledge_views import refit_shuffled
from tomography.experiments.v1_knowledge_views import residual_axis
from tomography.experiments.v1_knowledge_views import run
from tomography.experiments.v1_knowledge_views import run_plan
from tomography.experiments.v1_knowledge_views import select_items
from tomography.experiments.v1_knowledge_views import shuffled_draw
from tomography.experiments.v1_knowledge_views import supervised_views
from tomography.experiments.v1_knowledge_views import text_view
from tomography.experiments.v1_knowledge_views import text_views
from tomography.experiments.v1_knowledge_views import train_draws
from tomography.experiments.v1_knowledge_views import view_layers
from tomography.export import ExportError
from tomography.export import export_runs
from tomography.export import is_exported
from tomography.knowledge import DEFAULT_SHOTS
from tomography.model import MODELS
from tomography.model import LoadedModel
from tomography.p9.reading import READING_METHODS
from tomography.p9.reading import SURFACE_BASELINES
from tomography.p9.reading import ItemActivations
from tomography.p9.reading import collect_item_activations
from tomography.p9.reading import draw_train
from tomography.p9.reading import p9_template
from tomography.p9.reading import reading_cube
from tomography.p9.reading import type_auroc
from tomography.p9.unknowns import TypePool
from tomography.reading.reader import RepReader

PINNED_HASH = "3257d1220951"
# The planted pool and the labels run of the P9b tests are the inputs here.
_P9B = runpy.run_path(str(Path(__file__).with_name("_p9b_inputs.py")))
_items = _P9B["items"]
_labels_run = _P9B["labels_run"]
USABLE: list[str] = _P9B["USABLE"]


@dataclasses.dataclass(frozen=True)
class Inputs:
    """A P9a labels run and the P9b run read from it, both finished."""

    labels: RunDir
    reading: RunDir


_INPUTS: dict[str, Inputs] = {}


@pytest.fixture
def inputs(
    tiny_arch: LoadedModel, tmp_path_factory: pytest.TempPathFactory
) -> Inputs:
    """One P9b run per architecture, read from planted labels."""
    key = type(tiny_arch.model).__name__
    if key not in _INPUTS:
        out = tmp_path_factory.mktemp(f"v1-{key}")
        labels = _labels_run(out, _items())
        config = p9b_reading.Config(
            model="tiny",
            labels_run=str(labels.path),
            n_shots=2,
            n_seeds=2,
            n_pairs=16,
            min_pairs=8,
            min_eval=5,
            batch_size=16,
        )
        reading = RunDir.create(config, out_root=out / "reading")
        p9b_reading.execute(
            config, reading, tiny_arch, _items(), known_contrasts=True
        )
        reading.finalize()
        _INPUTS[key] = Inputs(labels, reading)
    return _INPUTS[key]


def _pools(committed: CommittedReading) -> dict[str, TypePool]:
    usable, _ = p9b_reading.usable_pools(committed.config, _items(), True)
    return usable


@pytest.fixture
def committed(inputs: Inputs) -> CommittedReading:
    """The P9b run of the architecture, as `load_reading` reads it."""
    return load_reading(inputs.reading.path, model="tiny")


@pytest.fixture
def pools(committed: CommittedReading) -> dict[str, TypePool]:
    """The usable contrasts the labels give, fresh for every test."""
    return _pools(committed)


def _copy(source: RunDir, tmp_path: Path) -> Path:
    target = tmp_path / "copy"
    shutil.copytree(source.path, target)
    return target


def _edit_json(path: Path, **changes: Any) -> None:
    document = json.loads(path.read_text("utf-8"))
    document.update(changes)
    path.write_text(json.dumps(document), "utf-8")


def test_config_defaults_and_round_trip() -> None:
    config = Config(labels_run="runs/p9a", reading_run="runs/p9b")

    assert config.experiment == "v1"
    assert config.model == "gpt2"
    assert config.labels_run == "runs/p9a"
    assert config.reading_run == "runs/p9b"
    assert config.batch_size == 8
    assert config.seed == 0
    assert Config.from_dict(config.to_dict()) == config


@pytest.mark.parametrize(
    ("data", "missing"),
    [({}, "labels_run"), ({"labels_run": "x"}, "reading_run")],
    ids=["labels_run", "reading_run"],
)
def test_config_requires_both_runs(data: dict[str, str], missing: str) -> None:
    with pytest.raises(ValueError, match=missing):
        Config.from_dict(data)


def test_the_config_hash_is_pinned() -> None:
    config = Config(labels_run="runs/p9a", reading_run="runs/p9b")

    assert config.config_hash() == PINNED_HASH


def test_the_cli_discovers_the_experiment() -> None:
    experiment = load_experiment("v1")

    assert experiment.config_cls is Config
    assert experiment.run is run
    assert experiment.plot is plot


def test_load_reading_returns_the_committed_numbers(
    tiny_arch: LoadedModel, inputs: Inputs
) -> None:
    reading = inputs.reading

    committed = load_reading(reading.path, model="tiny")

    assert isinstance(committed, CommittedReading)
    assert committed.config == p9b_reading.Config.from_dict(reading.config)
    assert committed.types == tuple(USABLE)
    assert committed.layers == tuple(range(1, tiny_arch.n_layers + 1))
    summary = reading.read_json("summary")
    assert isinstance(summary, dict)
    assert committed.layer == summary["layer"]
    transfer = reading.read_arrays("transfer")
    np.testing.assert_array_equal(committed.test, transfer["test"])
    n_types, n_layers = len(USABLE), tiny_arch.n_layers
    assert committed.test.shape == (4, n_types, 2, n_layers)
    readers = reading.read_arrays("readers")
    assert committed.directions.shape == (n_types, n_layers, tiny_arch.d_model)
    np.testing.assert_array_equal(committed.directions, readers["directions"])
    np.testing.assert_array_equal(committed.means, readers["means"])
    np.testing.assert_array_equal(committed.signs, readers["signs"])
    table = reading.read_json("table")
    assert isinstance(table, dict)
    assert list(committed.table) == USABLE
    row = dict(zip(table["columns"], table["rows"][0], strict=True))
    assert committed.table[USABLE[0]] == row


@pytest.mark.parametrize(
    ("name", "change", "match"),
    [
        ("meta.json", {"finished_at": None}, "not a finished run"),
        ("meta.json", {"experiment": "p9c"}, "not a P9b run"),
        ("config.json", {"template": "plain"}, "template"),
        ("config.json", {"methods": ["pca_diff", "logistic"]}, "methods"),
        ("config.json", {"n_seeds": 3}, "draws"),
    ],
    ids=["unfinished", "experiment", "template", "methods", "draws"],
)
def test_load_reading_refuses_a_run_it_cannot_reproduce(
    inputs: Inputs,
    tmp_path: Path,
    name: str,
    change: dict[str, Any],
    match: str,
) -> None:
    path = _copy(inputs.reading, tmp_path)
    _edit_json(path / name, **change)

    with pytest.raises(ValueError, match=match):
        load_reading(path, model="tiny")


def test_load_reading_refuses_a_run_of_another_model(inputs: Inputs) -> None:
    with pytest.raises(ValueError, match=r"'tiny'.*'gpt2'"):
        load_reading(inputs.reading.path, model="gpt2")


@pytest.mark.parametrize("name", READING_FILES)
def test_load_reading_refuses_a_run_without_its_files(
    inputs: Inputs, tmp_path: Path, name: str
) -> None:
    path = _copy(inputs.reading, tmp_path)
    (path / name).unlink()

    with pytest.raises(ValueError, match=name.replace(".", r"\.")):
        load_reading(path, model="tiny")


def test_load_reading_refuses_arrays_that_disagree_on_the_contrasts(
    inputs: Inputs, tmp_path: Path
) -> None:
    path = _copy(inputs.reading, tmp_path)
    document = json.loads((path / "readers.json").read_text("utf-8"))
    document["data"]["types"] = document["data"]["types"][::-1]
    (path / "readers.json").write_text(json.dumps(document), "utf-8")

    with pytest.raises(ValueError, match="contrasts"):
        load_reading(path, model="tiny")


def _drop_table_column(path: Path, column: str) -> None:
    document = json.loads((path / "table.json").read_text("utf-8"))
    table = document["data"]
    at = table["columns"].index(column)
    del table["columns"][at]
    for row in table["rows"]:
        del row[at]
    (path / "table.json").write_text(json.dumps(document), "utf-8")


@pytest.mark.parametrize(
    "column", [*COUNT_COLUMNS, *SURFACE_BASELINES], ids=str
)
def test_load_reading_names_a_column_the_table_lacks(
    inputs: Inputs, tmp_path: Path, column: str
) -> None:
    path = _copy(inputs.reading, tmp_path)
    _drop_table_column(path, column)

    with pytest.raises(ValueError, match=rf"table\.json.*'{column}'"):
        load_reading(path, model="tiny")


def _rewrite_readers(path: Path, **changes: np.ndarray) -> None:
    run_dir = RunDir.open(path)
    arrays = run_dir.read_arrays("readers")
    for key, change in changes.items():
        arrays[key] = change
    run_dir.write_arrays("readers", **arrays)


@pytest.mark.parametrize(
    ("key", "at", "value", "match"),
    [
        ("directions", (2, 1, slice(None)), 0.0, "zero vector"),
        ("directions", (0, 0, 3), np.nan, "finite"),
        ("directions", (0, 0, 3), np.inf, "finite"),
        ("means", (1, 2, 5), np.nan, "finite"),
        ("signs", (3, 1), 0.5, "signs"),
        ("signs", (3, 1), np.nan, "finite"),
    ],
    ids=[
        "zero-direction",
        "nan-direction",
        "inf-direction",
        "nan-mean",
        "half-sign",
        "nan-sign",
    ],
)
def test_load_reading_refuses_readers_it_cannot_score_with(
    inputs: Inputs,
    tmp_path: Path,
    key: str,
    at: tuple[Any, ...],
    value: float,
    match: str,
) -> None:
    path = _copy(inputs.reading, tmp_path)
    array = RunDir.open(path).read_arrays("readers")[key].copy()
    array[at] = value
    _rewrite_readers(path, **{key: array})

    with pytest.raises(ValueError, match=match):
        load_reading(path, model="tiny")


def test_load_reading_names_the_contrast_of_a_reader_it_refuses(
    inputs: Inputs, tmp_path: Path
) -> None:
    path = _copy(inputs.reading, tmp_path)
    directions = RunDir.open(path).read_arrays("readers")["directions"].copy()
    directions[4, 0] = 0.0
    _rewrite_readers(path, directions=directions)

    with pytest.raises(ValueError, match=USABLE[4]):
        load_reading(path, model="tiny")


def test_load_reading_holds_the_published_reader_of_each_contrast(
    committed: CommittedReading,
) -> None:
    assert len(committed.readers) == len(USABLE)
    for t, reader in enumerate(committed.readers):
        want = RepReader.from_arrays(
            committed.config.methods[0],
            committed.layers,
            committed.directions[t],
            committed.means[t],
            committed.signs[t],
        )
        assert reader.layers == committed.layers
        np.testing.assert_array_equal(reader.directions, want.directions)
        np.testing.assert_array_equal(reader.means, want.means)
        np.testing.assert_array_equal(reader.signs, want.signs)


def test_the_selection_is_the_test_items_and_the_train_items_of_the_draws(
    committed: CommittedReading, pools: dict[str, TypePool]
) -> None:
    draws = train_draws(pools, committed.config)

    selected = select_items(pools, draws)

    uids = [i.example.uid for i in selected]
    assert len(uids) == len(set(uids))
    test = {
        i.example.uid
        for pool in pools.values()
        for i in (*pool.positive, *pool.negative)
        if i.example.split == "test"
    }
    train = {
        i.example.uid
        for by_draw in draws.values()
        for draw in by_draw
        for i in draw.items
    }
    assert set(uids) == test | train
    assert {i.example.split for i in selected} == {"train", "test"}
    assert {
        i.example.uid for i in selected if i.example.split == "train"
    } == train


def test_the_selection_leaves_out_the_train_items_no_draw_uses(
    committed: CommittedReading, pools: dict[str, TypePool]
) -> None:
    draws = train_draws(pools, committed.config)

    selected = {i.example.uid for i in select_items(pools, draws)}

    every_train = {
        i.example.uid
        for pool in pools.values()
        for i in (*pool.positive, *pool.negative)
        if i.example.split == "train"
    }
    assert every_train - selected


def test_check_reading_accepts_the_pools_the_run_was_read_from(
    committed: CommittedReading, pools: dict[str, TypePool]
) -> None:
    check_reading(committed, pools)


def test_check_reading_refuses_other_contrasts(
    committed: CommittedReading, pools: dict[str, TypePool]
) -> None:
    del pools["false_premise"]

    with pytest.raises(ValueError, match=r"contrasts.*false_premise"):
        check_reading(committed, pools)


def test_check_reading_refuses_other_item_counts(
    committed: CommittedReading, pools: dict[str, TypePool]
) -> None:
    pool = pools["unknowable_detail"]
    dropped = next(i for i in pool.negative if i.example.split == "test")
    pools["unknowable_detail"] = dataclasses.replace(
        pool, negative=tuple(i for i in pool.negative if i is not dropped)
    )

    with pytest.raises(ValueError, match="unknowable_detail"):
        check_reading(committed, pools)


def test_check_reading_refuses_layers_that_do_not_start_at_one(
    committed: CommittedReading, pools: dict[str, TypePool]
) -> None:
    skipped = dataclasses.replace(committed, layers=(2, 3, 4))

    with pytest.raises(ValueError, match="layers"):
        check_reading(skipped, pools)


def test_check_model_accepts_the_model_the_run_was_read_with(
    committed: CommittedReading, tiny_arch: LoadedModel
) -> None:
    check_model(committed, tiny_arch)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda c: dataclasses.replace(c, layers=c.layers[:2]), "layers"),
        (
            lambda c: dataclasses.replace(c, directions=c.directions[..., :8]),
            "width",
        ),
    ],
    ids=["depth", "width"],
)
def test_check_model_refuses_a_model_the_readers_do_not_fit(
    committed: CommittedReading,
    tiny_arch: LoadedModel,
    change: Callable[[CommittedReading], CommittedReading],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        check_model(change(committed), tiny_arch)


def test_check_labels_run_compares_the_directories_not_the_spellings(
    committed: CommittedReading, inputs: Inputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(inputs.labels.path.parent)
    recorded = dataclasses.replace(
        committed.config, labels_run=inputs.labels.path.name
    )
    relative = dataclasses.replace(committed, config=recorded)

    check_labels_run(_config(inputs), relative)
    check_labels_run(
        dataclasses.replace(
            _config(inputs), labels_run=inputs.labels.path.name
        ),
        committed,
    )


def test_check_labels_run_refuses_other_labels(
    committed: CommittedReading, inputs: Inputs, tmp_path: Path
) -> None:
    elsewhere = dataclasses.replace(
        committed.config, labels_run=str(tmp_path / "other" / "p9a")
    )
    moved = dataclasses.replace(committed, config=elsewhere)

    with pytest.raises(ValueError, match=r"labels.*other"):
        check_labels_run(_config(inputs), moved)


def test_prepare_lists_the_questions_without_the_model(
    committed: CommittedReading, pools: dict[str, TypePool], inputs: Inputs
) -> None:
    draws = train_draws(pools, committed.config)

    plan = prepare(
        _config(inputs), _items(), known_contrasts=True, reading=committed
    )

    assert isinstance(plan, Plan)
    assert list(plan.pools) == USABLE
    assert [i.example.uid for i in plan.selected] == [
        i.example.uid for i in select_items(pools, draws)
    ]
    assert plan.surface == compare_surface(pools, committed)
    assert list(plan.draws) == USABLE


def _set_table_cell(path: Path, name: str, column: str, value: Any) -> None:
    document = json.loads((path / "table.json").read_text("utf-8"))
    table = document["data"]
    at = table["columns"].index(column)
    for row in table["rows"]:
        if row[0] == name:
            row[at] = value
    (path / "table.json").write_text(json.dumps(document), "utf-8")


def _never(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Makes the step `name` of the module fail the test if it runs."""

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"{name} must not run")

    monkeypatch.setattr(v1_module, name, never)


def test_run_refuses_a_reading_run_read_from_other_labels_before_the_model(
    inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _never(monkeypatch, "load_model")
    path = _copy(inputs.reading, tmp_path / "reading")
    _edit_json(path / "config.json", labels_run=str(tmp_path / "other" / "p9a"))
    config = dataclasses.replace(_config(inputs), reading_run=str(path))
    run_dir = RunDir.create(config, out_root=tmp_path)

    with pytest.raises(ValueError, match=r"labels.*other"):
        run(config, run_dir)


def test_run_refuses_a_batch_size_before_the_model(
    inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _never(monkeypatch, "load_model")
    config = _config(inputs, batch_size=0)
    run_dir = RunDir.create(config, out_root=tmp_path)

    with pytest.raises(ValueError, match="batch_size"):
        run(config, run_dir)


def test_run_refuses_labels_that_differ_from_the_reading_before_the_model(
    inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _never(monkeypatch, "load_model")
    path = _copy(inputs.reading, tmp_path / "reading")
    _set_table_cell(path, "unknowable_detail", "n_positive_test", 999)
    config = dataclasses.replace(_config(inputs), reading_run=str(path))
    run_dir = RunDir.create(config, out_root=tmp_path)

    with pytest.raises(ValueError, match="unknowable_detail"):
        run(config, run_dir)


def test_run_computes_the_surface_controls_before_the_model(
    inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _never(monkeypatch, "load_model")

    def broken(pool: Any) -> dict[str, float | None]:
        raise ValueError("no control")

    monkeypatch.setattr(v1_module, "surface_baselines", broken)
    config = _config(inputs)
    run_dir = RunDir.create(config, out_root=tmp_path)

    with pytest.raises(ValueError, match="no control"):
        run(config, run_dir)


def test_the_forward_pass_waits_for_the_check_of_the_model(
    committed: CommittedReading,
    tiny_arch: LoadedModel,
    inputs: Inputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _never(monkeypatch, "collect_item_activations")
    plan = prepare(
        _config(inputs), _items(), known_contrasts=True, reading=committed
    )
    shallow = dataclasses.replace(committed, layers=committed.layers[:2])
    config = _config(inputs)
    run_dir = RunDir.create(config, out_root=tmp_path)

    with pytest.raises(ValueError, match="layers"):
        run_plan(config, run_dir, tiny_arch, plan, shallow)


@dataclasses.dataclass(frozen=True)
class Read:
    """The committed run, its contrasts and the activations behind them."""

    committed: CommittedReading
    pools: dict[str, TypePool]
    acts: ItemActivations


def _read(
    tiny_arch: LoadedModel, inputs: Inputs, *, layer: int | None = None
) -> Read:
    committed = load_reading(inputs.reading.path, model="tiny")
    if layer is not None:
        committed = dataclasses.replace(committed, layer=layer)
    pools = _pools(committed)
    union = {
        i.example.uid: i
        for pool in pools.values()
        for i in (*pool.positive, *pool.negative)
    }
    acts = collect_item_activations(
        tiny_arch,
        list(union.values()),
        p9_template("fewshot", DEFAULT_SHOTS[: committed.config.n_shots]),
        chat_format_for(tiny_arch),
        layers=committed.layers,
        bos=committed.config.bos,
        batch_size=16,
    )
    return Read(committed, pools, acts)


def test_refit_methods_repeats_the_cube_of_the_reading(
    tiny_arch: LoadedModel, inputs: Inputs
) -> None:
    read = _read(tiny_arch, inputs)
    protocol = read.committed.config
    cube = reading_cube(
        read.acts,
        read.pools,
        methods=protocol.methods,
        preprocess=protocol.preprocess,
        n_seeds=protocol.n_seeds,
        n_pairs=protocol.n_pairs,
        seed=protocol.seed,
    )

    test = refit_methods(
        read.acts, read.pools, train_draws(read.pools, protocol), protocol
    )

    assert test.shape == (4, len(USABLE), 2, tiny_arch.n_layers)
    np.testing.assert_array_equal(test, cube.test)


def test_refit_methods_needs_only_the_selected_items(
    tiny_arch: LoadedModel, inputs: Inputs
) -> None:
    read = _read(tiny_arch, inputs)
    protocol = read.committed.config
    draws = train_draws(read.pools, protocol)
    selected = select_items(read.pools, draws)
    rows = read.acts.rows(selected)
    partial = ItemActivations(
        tuple(i.example.uid for i in selected),
        read.acts.layers,
        read.acts.data[:, rows],
    )

    test = refit_methods(partial, read.pools, draws, protocol)

    np.testing.assert_array_equal(
        test, refit_methods(read.acts, read.pools, draws, protocol)
    )


def _planted(pool: TypePool) -> tuple[dict[str, TypePool], ItemActivations]:
    """Noise with one direction shifted on the positive side of a contrast."""
    items = [*pool.positive, *pool.negative]
    rng = np.random.default_rng(1)
    data = rng.standard_normal((3, len(items), 32)).astype(np.float32)
    direction = rng.standard_normal(32).astype(np.float32)
    direction /= np.linalg.norm(direction)
    data[1:, : len(pool.positive)] += 6.0 * direction
    uids = tuple(i.example.uid for i in items)
    return {pool.name: pool}, ItemActivations(uids, (1, 2, 3), data)


def test_shuffled_labels_read_nothing_where_the_true_labels_read_a_signal(
    committed: CommittedReading, pools: dict[str, TypePool]
) -> None:
    pools, acts = _planted(pools["obscure_real"])
    protocol = dataclasses.replace(committed.config, n_seeds=4, n_pairs=32)
    draws = train_draws(pools, protocol)

    true = refit_methods(acts, pools, draws, protocol)[0, 0]
    shuffled = refit_shuffled(acts, pools, draws, protocol, seed=0)[0]

    assert true[:, 1:].min() > 0.9
    assert shuffled.shape == true.shape
    assert abs(shuffled[:, 1:].mean() - 0.5) < 0.15


def test_refit_shuffled_is_deterministic_and_follows_the_seed(
    committed: CommittedReading, pools: dict[str, TypePool]
) -> None:
    pools, acts = _planted(pools["obscure_real"])
    protocol = committed.config
    draws = train_draws(pools, protocol)

    one = refit_shuffled(acts, pools, draws, protocol, seed=0)
    again = refit_shuffled(acts, pools, draws, protocol, seed=0)
    other = refit_shuffled(acts, pools, draws, protocol, seed=1)

    np.testing.assert_array_equal(one, again)
    assert not np.array_equal(one, other)


def test_shuffled_draw_permutes_the_labels_inside_each_stratum(
    committed: CommittedReading, pools: dict[str, TypePool]
) -> None:
    pool = pools["obscure_real_pop"]
    draw = draw_train(pool, n_pairs=16, seed=3)

    shuffled = shuffled_draw(draw, seed=5)

    assert shuffled.pairs is None
    assert shuffled.items == draw.items
    np.testing.assert_array_equal(shuffled.strata, draw.strata)
    assert not np.array_equal(shuffled.labels, draw.labels)
    for code in np.unique(draw.strata):
        inside = draw.strata == code
        assert sorted(shuffled.labels[inside]) == sorted(draw.labels[inside])
    again = shuffled_draw(draw, seed=5)
    np.testing.assert_array_equal(shuffled.labels, again.labels)
    other = shuffled_draw(draw, seed=6)
    assert not np.array_equal(shuffled.labels, other.labels)


def _hash(uid: str) -> str:
    return hashlib.sha256(uid.encode()).hexdigest()


def test_displayed_items_are_the_first_by_the_hash_of_the_uid() -> None:
    items = _items()[:40]

    shown = displayed_items(items, cap=7)

    want = sorted(items, key=lambda i: _hash(i.example.uid))[:7]
    assert [i.example.uid for i in shown] == [i.example.uid for i in want]


def test_displayed_items_keep_a_short_side_whole() -> None:
    items = _items()[:5]

    shown = displayed_items(items, cap=300)

    assert sorted(i.example.uid for i in shown) == sorted(
        i.example.uid for i in items
    )
    assert displayed_items([], cap=300) == []


def test_compare_auroc_reports_the_largest_differences() -> None:
    committed = np.full((2, 2, 2, 3), 0.5)
    recomputed = committed.copy()
    recomputed[0, 1, 0, 1] += 0.0625
    recomputed[1, 0, 1, 2] += 0.03125

    report = compare_auroc(recomputed, committed, 1, ("a", "b"), ("x", "y"))

    assert report["tolerance"] == 0.02
    assert report["within_tolerance"] is False
    assert report["max_abs_diff"] == {
        "draw_at_layer": 0.0625,
        "mean_at_layer": 0.03125,
        "draw_all_layers": 0.0625,
        "mean_all_layers": 0.03125,
    }
    assert report["mean_diff_at_layer"] == {
        "x": {"a": 0.0, "b": 0.0},
        "y": {"a": 0.03125, "b": 0.0},
    }


def test_compare_auroc_passes_a_mean_difference_inside_the_tolerance() -> None:
    committed = np.full((1, 1, 2, 2), 0.5)
    recomputed = committed.copy()
    recomputed[0, 0, 0, 0] += 0.03125

    report = compare_auroc(recomputed, committed, 0, ("a",), ("x",))

    assert report["within_tolerance"] is True
    assert report["max_abs_diff"]["draw_at_layer"] == 0.03125
    assert report["max_abs_diff"]["mean_at_layer"] == 0.015625


def test_compare_auroc_looks_at_the_chosen_layer_for_the_flag() -> None:
    committed = np.full((1, 1, 1, 3), 0.5)
    recomputed = committed.copy()
    recomputed[0, 0, 0, 2] += 0.25

    report = compare_auroc(recomputed, committed, 0, ("a",), ("x",))

    assert report["within_tolerance"] is True
    assert report["max_abs_diff"]["draw_at_layer"] == 0.0
    assert report["max_abs_diff"]["draw_all_layers"] == 0.25


def test_compare_surface_matches_the_committed_controls(
    committed: CommittedReading, pools: dict[str, TypePool]
) -> None:
    report = compare_surface(pools, committed)

    assert list(report) == USABLE
    for name, by_control in report.items():
        assert list(by_control) == list(SURFACE_BASELINES)
        for control, entry in by_control.items():
            assert entry["equal"] is True, (name, control)
            assert entry["committed"] == committed.table[name][control]
    assert report["obscure_real"]["popularity"]["recomputed"] is not None
    assert report["false_premise"]["popularity"] == {
        "recomputed": None,
        "committed": None,
        "equal": True,
    }


def test_compare_surface_flags_a_control_that_moved(
    committed: CommittedReading, pools: dict[str, TypePool]
) -> None:
    table = {name: dict(row) for name, row in committed.table.items()}
    table["obscure_real"]["text_word"] += 0.01
    table["false_premise"]["popularity"] = 0.7
    moved = dataclasses.replace(committed, table=table)

    report = compare_surface(pools, moved)

    unequal = [
        (name, control)
        for name, by_control in report.items()
        for control, entry in by_control.items()
        if not entry["equal"]
    ]
    assert unequal == [
        ("obscure_real", "text_word"),
        ("false_premise", "popularity"),
    ]


def test_published_views_score_the_displayed_items_on_the_published_reader(
    tiny_arch: LoadedModel, inputs: Inputs
) -> None:
    read = _read(tiny_arch, inputs)
    committed = read.committed
    k = committed.layers.index(committed.layer)

    views = published_views(committed, read.acts, read.pools, cap=3)

    assert views.auroc.shape == (len(USABLE), tiny_arch.n_layers)
    assert list(views.projections) == [f"projection_{n}" for n in USABLE]
    assert list(views.uids) == USABLE
    for t, (name, pool) in enumerate(read.pools.items()):
        reader = RepReader.from_arrays(
            committed.config.methods[0],
            committed.layers,
            committed.directions[t],
            committed.means[t],
            committed.signs[t],
        )
        np.testing.assert_array_equal(
            views.auroc[t], type_auroc(reader, read.acts, pool, "test")
        )
        positive, negative = pool.split("test")
        shown = {
            "positive": displayed_items(positive, cap=3),
            "negative": displayed_items(negative, cap=3),
        }
        assert views.uids[name] == {
            side: [i.example.uid for i in items]
            for side, items in shown.items()
        }
        rows = read.acts.rows([*shown["positive"], *shown["negative"]])
        want = reader.score(read.acts.data[:, rows])[k]
        projection = views.projections[f"projection_{name}"]
        assert projection.dtype == np.float32
        np.testing.assert_array_equal(projection, want)


def test_published_views_cap_each_side_not_the_contrast(
    tiny_arch: LoadedModel, inputs: Inputs
) -> None:
    read = _read(tiny_arch, inputs)

    views = published_views(read.committed, read.acts, read.pools, cap=2)

    for name, sides in views.uids.items():
        assert len(sides["positive"]) == 2, name
        assert len(sides["negative"]) == 2, name
        assert views.projections[f"projection_{name}"].shape == (4,)


@dataclasses.dataclass
class Captured:
    """The call `execute` makes to collect the activations."""

    items: list[Any] = dataclasses.field(default_factory=list)
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    acts: ItemActivations | None = None


def _capture(monkeypatch: pytest.MonkeyPatch) -> Captured:
    captured = Captured()

    def spy(
        loaded: LoadedModel, items: Any, *args: Any, **kwargs: Any
    ) -> ItemActivations:
        captured.items = list(items)
        captured.args = args
        captured.kwargs = kwargs
        captured.acts = collect_item_activations(loaded, items, *args, **kwargs)
        return captured.acts

    monkeypatch.setattr(v1_module, "collect_item_activations", spy)
    return captured


def _config(inputs: Inputs, **overrides: Any) -> Config:
    base = Config(
        model="tiny",
        labels_run=str(inputs.labels.path),
        reading_run=str(inputs.reading.path),
        batch_size=16,
    )
    return dataclasses.replace(base, **overrides)


def _execute(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    inputs: Inputs,
    *,
    reading: CommittedReading | None = None,
    items: Any = None,
    **overrides: Any,
) -> RunDir:
    config = _config(inputs, **overrides)
    run_dir = RunDir.create(config, out_root=tmp_path / "v1")
    execute(
        config,
        run_dir,
        tiny_arch,
        _items() if items is None else items,
        known_contrasts=True,
        reading=reading or load_reading(inputs.reading.path, model="tiny"),
    )
    return run_dir


_DEFAULT_RUNS: dict[str, RunDir] = {}


@pytest.fixture
def default_run(
    tiny_arch: LoadedModel,
    inputs: Inputs,
    tmp_path_factory: pytest.TempPathFactory,
) -> RunDir:
    """One V1 run per architecture with the default settings."""
    key = type(tiny_arch.model).__name__
    if key not in _DEFAULT_RUNS:
        out = tmp_path_factory.mktemp(f"v1-run-{key}")
        _DEFAULT_RUNS[key] = _execute(tiny_arch, out, inputs)
    return _DEFAULT_RUNS[key]


_LAYER_RUNS: dict[tuple[str, int], RunDir] = {}


@pytest.fixture
def run_at(
    tiny_arch: LoadedModel,
    inputs: Inputs,
    tmp_path_factory: pytest.TempPathFactory,
) -> Callable[[int], RunDir]:
    """V1 runs per architecture whose P9b run chose a given layer."""
    key = type(tiny_arch.model).__name__

    def build(layer: int) -> RunDir:
        if (key, layer) not in _LAYER_RUNS:
            out = tmp_path_factory.mktemp(f"v1-layer{layer}-{key}")
            reading = dataclasses.replace(
                load_reading(inputs.reading.path, model="tiny"), layer=layer
            )
            _LAYER_RUNS[(key, layer)] = _execute(
                tiny_arch, out, inputs, reading=reading
            )
        return _LAYER_RUNS[(key, layer)]

    return build


def _summary(run_dir: RunDir) -> dict[str, Any]:
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    return summary


def _sidecar(run_dir: RunDir) -> dict[str, Any]:
    sidecar = run_dir.read_json("projections")
    assert isinstance(sidecar, dict)
    return sidecar


def _figure(run_dir: RunDir, panel: str) -> Figure:
    return panel_figure(
        panel,
        _sidecar(run_dir),
        run_dir.read_arrays("pca2d"),
        run_dir.read_arrays("text2d"),
    )


def test_execute_writes_the_files_of_a_run(default_run: RunDir) -> None:
    names = sorted(p.name for p in default_run.path.iterdir() if p.is_file())

    assert names == [
        "config.json",
        "meta.json",
        "pca2d.npz",
        "projections.json",
        "projections.npz",
        "summary.json",
        "text2d.npz",
    ]


def test_the_arrays_have_the_documented_shapes(
    tiny_arch: LoadedModel, default_run: RunDir
) -> None:
    n_types, n_layers = len(USABLE), tiny_arch.n_layers

    arrays = default_run.read_arrays("projections")

    assert set(arrays) == {
        "layers",
        "test",
        "shuffled",
        "published",
        *(f"projection_{name}" for name in USABLE),
    }
    assert arrays["layers"].tolist() == list(range(1, n_layers + 1))
    assert arrays["layers"].dtype == np.int64
    assert arrays["test"].shape == (4, n_types, 2, n_layers)
    assert arrays["shuffled"].shape == (n_types, 2, n_layers)
    assert arrays["published"].shape == (n_types, n_layers)
    for name in ("test", "shuffled", "published"):
        assert arrays[name].dtype == np.float64
        assert arrays[name].min() >= 0.0
        assert arrays[name].max() <= 1.0
    sidecar = _sidecar(default_run)
    assert sidecar["types"] == USABLE
    assert sidecar["methods"] == list(READING_METHODS)
    assert sidecar["max_per_side"] == 300
    for name in USABLE:
        sides = sidecar["uids"][name]
        shown = len(sides["positive"]) + len(sides["negative"])
        assert arrays[f"projection_{name}"].shape == (shown,)


def test_the_summary_records_the_run(
    committed: CommittedReading, pools: dict[str, TypePool], default_run: RunDir
) -> None:
    summary = _summary(default_run)

    assert set(summary) == {
        "layer",
        "template",
        "methods",
        "types",
        "n_items",
        "extraction_dtype",
        "reading_run_dtype",
        "views",
        "reproduction",
        "surface",
        "surface_equal",
    }
    assert summary["layer"] == committed.layer
    assert summary["extraction_dtype"] == "float32"
    assert summary["reading_run_dtype"] is None
    assert summary["template"] == "fewshot"
    assert summary["methods"] == list(READING_METHODS)
    assert summary["types"] == USABLE
    selected = select_items(pools, train_draws(pools, committed.config))
    assert summary["n_items"] == {
        "train": sum(i.example.split == "train" for i in selected),
        "test": sum(i.example.split == "test" for i in selected),
    }
    assert set(summary["reproduction"]) == {
        "tolerance",
        "within_tolerance",
        "max_abs_diff",
        "mean_diff_at_layer",
    }
    assert list(summary["surface"]) == USABLE
    assert summary["surface_equal"] is True


def test_the_summary_names_the_dtype_of_this_run_and_of_the_reading_run(
    tiny_arch: LoadedModel,
    inputs: Inputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = dataclasses.replace(MODELS["gpt2"], key="tiny", dtype="bfloat16")
    monkeypatch.setattr(v1_module, "MODELS", {"tiny": spec})
    half = dataclasses.replace(tiny_arch, dtype=torch.float16)

    summary = _summary(_execute(half, tmp_path, inputs))

    assert summary["extraction_dtype"] == "float16"
    assert summary["reading_run_dtype"] == "bfloat16"


def test_the_recomputed_auroc_is_the_committed_one_on_the_same_activations(
    committed: CommittedReading, default_run: RunDir
) -> None:
    test = default_run.read_arrays("projections")["test"]

    np.testing.assert_array_equal(test, committed.test)
    reproduction = _summary(default_run)["reproduction"]
    assert reproduction["within_tolerance"] is True
    assert reproduction["tolerance"] == 0.02
    assert set(reproduction["max_abs_diff"].values()) == {0.0}


def test_the_verdict_looks_at_the_column_of_the_chosen_layer(
    committed: CommittedReading,
    tiny_arch: LoadedModel,
    inputs: Inputs,
    tmp_path: Path,
) -> None:
    assert committed.layers == (1, 2, 3)
    chosen = dataclasses.replace(committed, layer=2)
    at = chosen.layers.index(2)

    def reproduction(column: int | None, name: str) -> dict[str, Any]:
        test = committed.test.copy()
        if column is not None:
            test[..., column] -= 0.125
        reading = dataclasses.replace(chosen, test=test)
        run_dir = _execute(tiny_arch, tmp_path / name, inputs, reading=reading)
        return _summary(run_dir)["reproduction"]

    baseline = reproduction(None, "none")
    assert baseline["within_tolerance"] is True
    assert set(baseline["max_abs_diff"].values()) == {0.0}
    moved = reproduction(at, "chosen")
    assert moved["within_tolerance"] is False
    for by_method in moved["mean_diff_at_layer"].values():
        assert by_method == pytest.approx(dict.fromkeys(by_method, 0.125))
    for column in (0, 2):
        other = reproduction(column, f"other{column}")
        assert other["within_tolerance"] is True
        assert other["mean_diff_at_layer"] == baseline["mean_diff_at_layer"]
        assert other["max_abs_diff"]["mean_all_layers"] == pytest.approx(0.125)


def test_the_random_finder_of_the_run_is_the_one_of_the_p9b_run(
    committed: CommittedReading, default_run: RunDir
) -> None:
    m = READING_METHODS.index("random")

    random = default_run.read_arrays("projections")["test"][m]

    np.testing.assert_array_equal(random, committed.test[m])
    assert not np.allclose(random, 0.5)


def test_the_random_finder_draws_by_position_in_the_layer_axis(
    tiny_arch: LoadedModel, inputs: Inputs
) -> None:
    read = _read(tiny_arch, inputs)
    committed = read.committed
    draws = train_draws(read.pools, committed.config)
    later = ItemActivations(
        read.acts.uids, read.acts.layers[1:], read.acts.data[1:]
    )
    m = READING_METHODS.index("random")

    aligned = refit_methods(read.acts, read.pools, draws, committed.config)
    shifted = refit_methods(later, read.pools, draws, committed.config)

    np.testing.assert_array_equal(aligned[m], committed.test[m])
    assert not np.array_equal(shifted[m], committed.test[m][..., 1:])
    np.testing.assert_array_equal(shifted[0], committed.test[0][..., 1:])


def test_the_extraction_reads_layers_one_to_the_last_of_the_selected_items(
    committed: CommittedReading,
    pools: dict[str, TypePool],
    tiny_arch: LoadedModel,
    inputs: Inputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture(monkeypatch)

    _execute(tiny_arch, tmp_path, inputs, batch_size=4)

    selected = select_items(pools, train_draws(pools, committed.config))
    assert [i.example.uid for i in captured.items] == [
        i.example.uid for i in selected
    ]
    assert captured.acts is not None
    assert tuple(captured.acts.layers) == committed.layers
    template, chat = captured.args
    assert template == p9_template("fewshot", DEFAULT_SHOTS[:2])
    assert chat.kind == chat_format_for(tiny_arch).kind
    assert captured.kwargs == {
        "layers": tuple(range(1, tiny_arch.n_layers + 1)),
        "bos": committed.config.bos,
        "batch_size": 4,
        "share_prefix": True,
    }


def test_the_published_reader_and_its_projections_are_written(
    committed: CommittedReading,
    pools: dict[str, TypePool],
    tiny_arch: LoadedModel,
    inputs: Inputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture(monkeypatch)

    run_dir = _execute(tiny_arch, tmp_path, inputs)

    assert captured.acts is not None
    reads = captured.acts
    views = published_views(committed, reads, pools, cap=MAX_DISPLAYED)
    arrays = run_dir.read_arrays("projections")
    np.testing.assert_array_equal(arrays["published"], views.auroc)
    for key, projection in views.projections.items():
        np.testing.assert_array_equal(arrays[key], projection)
    sidecar = _sidecar(run_dir)
    assert sidecar["uids"] == views.uids


def test_the_shuffled_control_is_written_beside_the_methods(
    committed: CommittedReading,
    pools: dict[str, TypePool],
    tiny_arch: LoadedModel,
    inputs: Inputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture(monkeypatch)

    run_dir = _execute(tiny_arch, tmp_path, inputs, seed=3)

    assert captured.acts is not None
    reads = captured.acts
    want = refit_shuffled(
        reads,
        pools,
        train_draws(pools, committed.config),
        committed.config,
        seed=3,
    )
    np.testing.assert_array_equal(
        run_dir.read_arrays("projections")["shuffled"], want
    )


def test_the_surface_controls_are_recorded_beside_the_committed_ones(
    committed: CommittedReading, pools: dict[str, TypePool], default_run: RunDir
) -> None:
    surface = _summary(default_run)["surface"]

    assert surface == compare_surface(pools, committed)


def test_a_surface_control_that_moved_is_reported_not_hidden(
    committed: CommittedReading,
    tiny_arch: LoadedModel,
    inputs: Inputs,
    tmp_path: Path,
) -> None:
    table = {name: dict(row) for name, row in committed.table.items()}
    table["obscure_real"]["text_char"] += 0.05
    moved = dataclasses.replace(committed, table=table)

    run_dir = _execute(tiny_arch, tmp_path, inputs, reading=moved)

    summary = _summary(run_dir)
    assert summary["surface_equal"] is False
    assert summary["surface"]["obscure_real"]["text_char"]["equal"] is False


def test_execute_is_deterministic(
    tiny_arch: LoadedModel, inputs: Inputs, tmp_path: Path
) -> None:
    first = _execute(tiny_arch, tmp_path / "a", inputs)
    second = _execute(tiny_arch, tmp_path / "b", inputs)

    for name in ("projections", "pca2d", "text2d"):
        one, two = first.read_arrays(name), second.read_arrays(name)
        assert one.keys() == two.keys(), name
        for key in one:
            np.testing.assert_array_equal(
                one[key], two[key], err_msg=f"{name}.{key}"
            )
    assert first.read_json("projections") == second.read_json("projections")
    assert first.read_json("summary") == second.read_json("summary")


def test_run_loads_the_model_in_float32_on_the_default_device(
    tiny_arch: LoadedModel,
    inputs: Inputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[tuple[str, dict[str, Any]]] = []

    def fake_load(name: str, **kwargs: Any) -> LoadedModel:
        asked.append((name, kwargs))
        return tiny_arch

    monkeypatch.setattr(v1_module, "load_model", fake_load)
    config = _config(inputs)
    run_dir = RunDir.create(config, out_root=tmp_path)

    run(config, run_dir)

    assert asked == [("tiny", {"dtype": torch.float32})]
    assert _summary(run_dir)["types"] == USABLE
    assert (run_dir.path / "projections.npz").is_file()


def test_run_refuses_an_unfinished_labels_run(
    inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _never(monkeypatch, "load_model")
    path = _copy(inputs.labels, tmp_path / "labels")
    _edit_json(path / "meta.json", finished_at=None)
    config = dataclasses.replace(_config(inputs), labels_run=str(path))
    run_dir = RunDir.create(config, out_root=tmp_path)

    with pytest.raises(ValueError, match="not a finished run"):
        run(config, run_dir)


def _finished_copy(run_dir: RunDir, tmp_path: Path) -> Path:
    """Copies a run and stamps it finished, as the command line does."""
    target = tmp_path / "run"
    shutil.copytree(run_dir.path, target)
    RunDir.open(target).finalize()
    return target


@pytest.mark.usefixtures("fixed_machine")
def test_every_file_of_a_run_passes_the_export(
    default_run: RunDir, tmp_path: Path
) -> None:
    for name in (
        "projections.npz",
        "projections.json",
        "pca2d.npz",
        "text2d.npz",
        "summary.json",
    ):
        assert is_exported(name), name
    source = _finished_copy(default_run, tmp_path)

    report = export_runs([source], tmp_path / "dest")

    (exported,) = report.exported
    assert exported.status == "copied"
    copied = sorted(
        p.name for p in (tmp_path / "dest" / exported.key).iterdir()
    )
    assert copied == [
        "config.json",
        "meta.json",
        "pca2d.npz",
        "projections.json",
        "projections.npz",
        "summary.json",
        "text2d.npz",
    ]


@pytest.mark.usefixtures("fixed_machine")
def test_the_export_refuses_a_uid_that_names_the_machine(
    default_run: RunDir, tmp_path: Path
) -> None:
    source = _finished_copy(default_run, tmp_path)
    document = json.loads((source / "projections.json").read_text("utf-8"))
    sides = document["data"]["uids"][USABLE[0]]
    sides["positive"][0] = "popqa:tester-user-7"
    (source / "projections.json").write_text(json.dumps(document), "utf-8")

    with pytest.raises(ExportError, match=r"projections\.json"):
        export_runs([source], tmp_path / "dest")

    assert not (tmp_path / "dest").exists()


@pytest.mark.usefixtures("fixed_machine")
def test_the_export_refuses_a_uid_array_that_names_a_private_path(
    default_run: RunDir, tmp_path: Path
) -> None:
    source = _finished_copy(default_run, tmp_path)
    arrays = default_run.read_arrays("projections")
    RunDir.open(source).write_arrays(
        "projections", uids=np.array(["popqa:1", "D:\\private\\x"]), **arrays
    )

    with pytest.raises(ExportError, match=r"projections\.npz"):
        export_runs([source], tmp_path / "dest")


def test_panel_contrasts_group_the_shown_contrasts_by_panel() -> None:
    names = [
        "fictional_synthal",
        "fictional_generated",
        "obscure_real",
        "obscure_real_pop",
        "post_cutoff",
        "triviaqa_known",
        "unknowable_detail",
        "false_premise",
        "kuq_future_unknown",
        "kuq_false_assumption",
        "real_vs_synthal",
        "real_vs_generated",
        "pre_vs_post_cutoff",
    ]

    panels = panel_contrasts(names)

    assert list(panels) == ["entity", "question", "unknown"]
    assert panels["entity"] == ("fictional_generated", "obscure_real")
    assert panels["question"] == (
        "unknowable_detail",
        "false_premise",
        "kuq_future_unknown",
        "kuq_false_assumption",
    )
    assert panels["unknown"] == (
        "real_vs_synthal",
        "real_vs_generated",
        "pre_vs_post_cutoff",
    )


def test_a_panel_without_a_usable_contrast_is_absent() -> None:
    assert list(panel_contrasts(["real_vs_generated", "post_cutoff"])) == [
        "unknown"
    ]
    assert panel_contrasts(["obscure_real_pop", "triviaqa_known"]) == {}
    assert panel_contrasts([]) == {}


def test_a_panel_shows_each_item_once_in_the_order_of_its_contrasts() -> None:
    uids = {
        "a": {"positive": ["k1", "k2"], "negative": ["u1", "u2"]},
        "b": {"positive": ["k2", "k3"], "negative": ["u3", "u1"]},
        "c": {"positive": ["z"], "negative": ["y"]},
    }

    assert panel_union(("a", "b"), uids) == [
        "k1",
        "k2",
        "u1",
        "u2",
        "k3",
        "u3",
    ]
    assert panel_union(("c",), uids) == ["z", "y"]


def test_the_sidecar_lists_the_panels_and_the_items_they_show(
    default_run: RunDir,
) -> None:
    sidecar = _sidecar(default_run)

    panels = sidecar["panels"]

    assert list(panels) == ["entity", "question", "unknown"]
    assert panels["entity"]["types"] == ["fictional_generated", "obscure_real"]
    assert panels["question"]["types"] == [
        "unknowable_detail",
        "false_premise",
        "kuq_future_unknown",
    ]
    assert panels["unknown"]["types"] == [
        "real_vs_synthal",
        "real_vs_generated",
    ]
    for panel in panels.values():
        assert panel["uids"] == panel_union(panel["types"], sidecar["uids"])
        assert len(set(panel["uids"])) == len(panel["uids"])


CHOSEN_LAYERS = pytest.mark.parametrize(
    ("chosen", "shown"),
    [(1, [1, 3]), (2, [1, 2, 3]), (3, [1, 3])],
    ids=["first", "middle", "last"],
)


@pytest.mark.parametrize(
    ("chosen", "n_layers", "want"),
    [
        (2, 3, (1, 2, 3)),
        (1, 3, (1, 3)),
        (3, 3, (1, 3)),
        (1, 1, (1,)),
    ],
    ids=["middle", "first", "last", "only"],
)
def test_the_views_are_drawn_at_layer_one_the_chosen_layer_and_the_last(
    committed: CommittedReading,
    chosen: int,
    n_layers: int,
    want: tuple[int, ...],
) -> None:
    reading = dataclasses.replace(committed, layer=chosen)

    assert view_layers(reading, n_layers) == want


def _two_panels(read: Read) -> dict[str, list[str]]:
    shown = displayed_uids(read.pools, cap=4)
    return {
        panel: panel_union(types, shown)
        for panel, types in panel_contrasts(read.pools).items()
    }


def test_pca_views_fit_each_panel_on_its_own_items(
    tiny_arch: LoadedModel, inputs: Inputs
) -> None:
    read = _read(tiny_arch, inputs, layer=2)
    panels = _two_panels(read)
    layers = view_layers(read.committed, tiny_arch.n_layers)

    arrays = pca_views(read.acts, layers, panels)

    assert set(arrays) == {
        "view_layers",
        *(f"coordinates_{panel}" for panel in panels),
        *(f"ratios_{panel}" for panel in panels),
    }
    assert arrays["view_layers"].tolist() == list(layers)
    assert arrays["view_layers"].dtype == np.int64
    for panel, uids in panels.items():
        coordinates = arrays[f"coordinates_{panel}"]
        assert coordinates.shape == (3, len(uids), 2)
        assert coordinates.dtype == np.float32
        assert arrays[f"ratios_{panel}"].shape == (3, 2)
        index = {uid: row for row, uid in enumerate(read.acts.uids)}
        rows = [index[uid] for uid in uids]
        ks = [read.acts.layers.index(layer) for layer in layers]
        want, ratios = pca_coordinates(read.acts.data[np.ix_(ks, rows)])
        np.testing.assert_array_equal(coordinates, want)
        np.testing.assert_array_equal(arrays[f"ratios_{panel}"], ratios)


def test_pca_views_ignore_the_items_a_panel_does_not_show(
    tiny_arch: LoadedModel, inputs: Inputs
) -> None:
    read = _read(tiny_arch, inputs, layer=2)
    panels = _two_panels(read)
    layers = view_layers(read.committed, tiny_arch.n_layers)
    shown = {uid for uids in panels.values() for uid in uids}
    hidden = next(
        row for row, uid in enumerate(read.acts.uids) if uid not in shown
    )
    data = read.acts.data.copy()
    data[:, hidden] += 100.0
    moved = ItemActivations(read.acts.uids, read.acts.layers, data)

    before = pca_views(read.acts, layers, panels)
    after = pca_views(moved, layers, panels)

    for key in before:
        np.testing.assert_array_equal(before[key], after[key], err_msg=key)


def test_pca_views_refuse_an_item_without_activations(
    tiny_arch: LoadedModel, inputs: Inputs
) -> None:
    read = _read(tiny_arch, inputs)
    layers = view_layers(read.committed, tiny_arch.n_layers)

    with pytest.raises(ValueError, match="no activations for 'ghost:1'"):
        pca_views(read.acts, layers, {"entity": ["ghost:1"]})


@CHOSEN_LAYERS
def test_execute_writes_the_pca_views_of_each_panel(
    run_at: Callable[[int], RunDir], chosen: int, shown: list[int]
) -> None:
    run_dir = run_at(chosen)
    sidecar = _sidecar(run_dir)

    arrays = run_dir.read_arrays("pca2d")

    assert arrays["view_layers"].tolist() == shown
    for panel, entry in sidecar["panels"].items():
        n = len(entry["uids"])
        assert arrays[f"coordinates_{panel}"].shape == (len(shown), n, 2)
        assert arrays[f"ratios_{panel}"].shape == (len(shown), 2)
    assert set(arrays) == {
        "view_layers",
        *(f"coordinates_{panel}" for panel in sidecar["panels"]),
        *(f"ratios_{panel}" for panel in sidecar["panels"]),
        *(
            f"supervised_{name}"
            for entry in sidecar["panels"].values()
            for name in entry["types"]
        ),
    }


def test_a_small_cap_limits_every_view_of_a_run(
    pools: dict[str, TypePool],
    tiny_arch: LoadedModel,
    inputs: Inputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cap = 3
    monkeypatch.setattr(v1_module, "MAX_DISPLAYED", cap)
    want = {}
    for name, pool in pools.items():
        positive, negative = pool.split("test")
        want[name] = {
            "positive": [
                i.example.uid for i in displayed_items(positive, cap=cap)
            ],
            "negative": [
                i.example.uid for i in displayed_items(negative, cap=cap)
            ],
        }

    run_dir = _execute(tiny_arch, tmp_path, inputs)

    sidecar = _sidecar(run_dir)
    assert sidecar["max_per_side"] == cap
    assert sidecar["uids"] == want
    assert any(len(side) == cap for by in want.values() for side in by.values())
    arrays = run_dir.read_arrays("projections")
    for name, by in want.items():
        shown = len(by["positive"]) + len(by["negative"])
        assert arrays[f"projection_{name}"].shape == (shown,)
    pca = run_dir.read_arrays("pca2d")
    text = run_dir.read_arrays("text2d")
    for panel, entry in sidecar["panels"].items():
        assert entry["uids"] == panel_union(entry["types"], want)
        n = len(entry["uids"])
        assert pca[f"coordinates_{panel}"].shape[1] == n
        assert text[f"coordinates_{panel}"].shape[0] == n


@CHOSEN_LAYERS
def test_the_summary_names_the_views_and_the_items_they_show(
    run_at: Callable[[int], RunDir], chosen: int, shown: list[int]
) -> None:
    run_dir = run_at(chosen)
    sidecar = _sidecar(run_dir)

    views = _summary(run_dir)["views"]

    assert views["layers"] == shown
    assert views["panels"] == {
        panel: {"types": entry["types"], "n_items": len(entry["uids"])}
        for panel, entry in sidecar["panels"].items()
    }


def test_displayed_uids_are_those_the_published_views_show(
    tiny_arch: LoadedModel, inputs: Inputs
) -> None:
    read = _read(tiny_arch, inputs)

    views = published_views(read.committed, read.acts, read.pools, cap=3)

    assert displayed_uids(read.pools, cap=3) == views.uids


def _orthogonal_columns(n: int, k: int, seed: int) -> np.ndarray:
    """Centered columns that are orthonormal, so every variance is planted."""
    rng = np.random.default_rng(seed)
    centered = rng.standard_normal((n, k))
    centered -= centered.mean(axis=0)
    q, _ = np.linalg.qr(centered)
    return q


def test_residual_axis_is_the_top_component_once_the_direction_is_removed() -> (
    None
):
    q = _orthogonal_columns(20, 3, seed=0)
    h = np.zeros((20, 5))
    h[:, 0] = 10.0 * q[:, 0]
    h[:, 1] = 3.0 * q[:, 1]
    h[:, 2] = 1.0 * q[:, 2]
    direction = np.eye(5)[0]

    y = residual_axis(h, direction)

    assert y.dtype == np.float32
    np.testing.assert_allclose(y, 3.0 * q[:, 1], atol=1e-5)


def test_residual_axis_does_not_see_the_component_along_the_direction() -> None:
    rng = np.random.default_rng(1)
    h = rng.standard_normal((30, 6))
    direction = rng.standard_normal(6)
    shifted = h + np.outer(rng.standard_normal(30) * 50.0, direction)

    np.testing.assert_allclose(
        residual_axis(shifted, direction),
        residual_axis(h, direction),
        atol=1e-4,
    )
    np.testing.assert_allclose(
        residual_axis(h, 7.0 * direction),
        residual_axis(h, direction),
        atol=1e-5,
    )


def test_residual_axis_refuses_rows_that_vary_along_the_direction_alone() -> (
    None
):
    rng = np.random.default_rng(2)
    t = rng.standard_normal(12)
    direction = rng.standard_normal(6)
    rows = np.outer(t, direction) + 5.0

    with pytest.raises(ValueError, match="direction alone"):
        residual_axis(rows, direction)


def test_residual_axis_refuses_a_zero_direction() -> None:
    rows = np.random.default_rng(3).standard_normal((10, 4))

    with pytest.raises(ValueError, match="zero"):
        residual_axis(rows, np.zeros(4))


def test_supervised_views_set_the_published_projection_against_the_rest(
    tiny_arch: LoadedModel, inputs: Inputs
) -> None:
    read = _read(tiny_arch, inputs, layer=2)
    committed = read.committed
    panels = _two_panels(read)
    shown = panel_contrasts(read.pools)
    layers = view_layers(committed, tiny_arch.n_layers)

    arrays = supervised_views(committed, read.acts, layers, panels, shown)

    assert set(arrays) == {
        f"supervised_{name}" for names in shown.values() for name in names
    }
    index = {uid: row for row, uid in enumerate(read.acts.uids)}
    for panel, names in shown.items():
        rows = [index[uid] for uid in panels[panel]]
        for name in names:
            t = list(committed.types).index(name)
            view = arrays[f"supervised_{name}"]
            assert view.shape == (3, len(rows), 2)
            assert view.dtype == np.float32
            for v, layer in enumerate(layers):
                k = committed.layers.index(layer)
                reader = RepReader.from_arrays(
                    committed.config.methods[0],
                    [layer],
                    committed.directions[t, [k]],
                    committed.means[t, [k]],
                    committed.signs[t, [k]],
                )
                h = read.acts.data[k][rows]
                np.testing.assert_array_equal(
                    view[v, :, 0], reader.score(h[None])[0]
                )
                np.testing.assert_array_equal(
                    view[v, :, 1], residual_axis(h, reader.directions[0])
                )


def test_supervised_x_is_the_projection_the_run_publishes(
    tiny_arch: LoadedModel, inputs: Inputs
) -> None:
    read = _read(tiny_arch, inputs, layer=2)
    committed = read.committed
    panels = _two_panels(read)
    shown = panel_contrasts(read.pools)
    layers = view_layers(committed, tiny_arch.n_layers)
    arrays = supervised_views(committed, read.acts, layers, panels, shown)
    at = layers.index(committed.layer)
    own = displayed_uids(read.pools, cap=4)
    published = published_views(committed, read.acts, read.pools, cap=4)

    for panel, names in shown.items():
        for name in names:
            uids = [*own[name]["positive"], *own[name]["negative"]]
            places = [panels[panel].index(uid) for uid in uids]
            chosen = arrays[f"supervised_{name}"][at, places, 0]
            np.testing.assert_allclose(
                chosen, published.projections[f"projection_{name}"], rtol=1e-5
            )


@CHOSEN_LAYERS
def test_execute_writes_the_supervised_view_of_each_shown_contrast(
    run_at: Callable[[int], RunDir], chosen: int, shown: list[int]
) -> None:
    run_dir = run_at(chosen)
    sidecar = _sidecar(run_dir)

    arrays = run_dir.read_arrays("pca2d")

    for entry in sidecar["panels"].values():
        for name in entry["types"]:
            view = arrays[f"supervised_{name}"]
            assert view.shape == (len(shown), len(entry["uids"]), 2)
            assert view.dtype == np.float32
    assert "supervised_obscure_real_pop" not in arrays


def test_text_view_separates_questions_that_share_no_words() -> None:
    # Groups of unequal size give unequal singular values; equal groups would
    # leave any rotation of the two components a valid answer.
    first = [f"alpha beta gamma delta {i}" for i in range(8)]
    second = [f"red green blue yellow {i}" for i in range(4)]

    coordinates, ratios = text_view([*first, *second], seed=0)

    assert coordinates.shape == (12, 2)
    assert coordinates.dtype == np.float32
    assert ratios.shape == (2,)
    assert ratios.dtype == np.float64
    assert (ratios > 0.0).all()
    assert ratios.sum() <= 1.0 + 1e-9
    size = np.abs(coordinates)
    np.testing.assert_allclose(size[:8], [[1.0, 0.0]] * 8, atol=1e-5)
    np.testing.assert_allclose(size[8:], [[0.0, 1.0]] * 4, atol=1e-5)


@pytest.mark.parametrize(
    "texts",
    [
        ["alpha beta gamma"] * 6,
        [f"alpha beta rare{i}" for i in range(6)],
    ],
    ids=["identical", "differ_only_by_words_seen_once"],
)
def test_text_view_refuses_texts_with_a_single_component(
    texts: list[str],
) -> None:
    with pytest.raises(ValueError, match="two components"):
        text_view(texts, seed=0)


WORDS = ("river", "mountain", "harbor", "forest", "valley", "island")


def _sentence(i: int) -> str:
    return f"who found the {WORDS[i % 6]} near the {WORDS[(i * 5 + 1) % 6]}"


def test_text_view_is_deterministic_for_a_seed() -> None:
    texts = [_sentence(i) for i in range(30)]

    one = text_view(texts, seed=7)
    two = text_view(texts, seed=7)

    np.testing.assert_array_equal(one[0], two[0])
    np.testing.assert_array_equal(one[1], two[1])


def test_text_views_use_the_questions_of_the_items_a_panel_shows() -> None:
    questions = {f"q:{i}": _sentence(i) for i in range(30)}
    panels = {
        "entity": ["q:3", "q:1", "q:7", "q:12", "q:20", "q:25"],
        "unknown": ["q:0", "q:5", "q:9", "q:2", "q:8"],
    }

    arrays = text_views(questions, panels, seed=4)

    assert set(arrays) == {
        "coordinates_entity",
        "ratios_entity",
        "coordinates_unknown",
        "ratios_unknown",
    }
    for panel, uids in panels.items():
        coordinates, ratios = text_view(
            [questions[uid] for uid in uids], seed=derive_seed(4, "text", panel)
        )
        np.testing.assert_array_equal(
            arrays[f"coordinates_{panel}"], coordinates
        )
        np.testing.assert_array_equal(arrays[f"ratios_{panel}"], ratios)


def test_text_views_refuse_an_item_without_a_question() -> None:
    with pytest.raises(ValueError, match="no question for 'ghost:1'"):
        text_views({"q:1": "a b"}, {"entity": ["q:1", "ghost:1"]}, seed=0)


def test_execute_writes_the_text_view_of_each_panel(
    committed: CommittedReading, pools: dict[str, TypePool], default_run: RunDir
) -> None:
    sidecar = _sidecar(default_run)
    questions = {
        i.example.uid: i.example.question
        for pool in pools.values()
        for i in (*pool.positive, *pool.negative)
    }

    arrays = default_run.read_arrays("text2d")

    assert set(arrays) == {
        *(f"coordinates_{panel}" for panel in sidecar["panels"]),
        *(f"ratios_{panel}" for panel in sidecar["panels"]),
    }
    want = text_views(
        questions,
        {name: entry["uids"] for name, entry in sidecar["panels"].items()},
        seed=0,
    )
    for key, value in want.items():
        np.testing.assert_array_equal(arrays[key], value, err_msg=key)
    for panel, entry in sidecar["panels"].items():
        assert arrays[f"coordinates_{panel}"].shape == (len(entry["uids"]), 2)
        assert arrays[f"ratios_{panel}"].shape == (2,)


def test_the_seed_of_the_run_reaches_the_text_view(
    tiny_arch: LoadedModel,
    inputs: Inputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeds: list[int] = []
    original = v1_module.text_view

    def spy(texts: Any, *, seed: int) -> Any:
        seeds.append(seed)
        return original(texts, seed=seed)

    monkeypatch.setattr(v1_module, "text_view", spy)

    _execute(tiny_arch, tmp_path, inputs, seed=3)

    assert seeds == [
        derive_seed(3, "text", p) for p in ("entity", "question", "unknown")
    ]


def test_no_file_of_a_run_holds_a_question(default_run: RunDir) -> None:
    questions = [i.example.question for i in _items()]

    for path in default_run.path.iterdir():
        content = path.read_bytes()
        for question in questions:
            for encoding in ("utf-8", "utf-32-le"):
                assert question.encode(encoding) not in content, path.name


@pytest.mark.parametrize("name", ["pca2d", "text2d", "projections"])
def test_no_array_of_a_run_holds_text(default_run: RunDir, name: str) -> None:
    arrays = default_run.read_arrays(name)

    assert arrays
    for key, array in arrays.items():
        assert array.dtype.kind in "biuf", f"{name}[{key}] is {array.dtype}"


def test_no_text_of_a_figure_holds_a_question(default_run: RunDir) -> None:
    questions = [i.example.question for i in _items()]

    for panel in _sidecar(default_run)["panels"]:
        figure = _figure(default_run, panel)
        figure.canvas.draw()
        seen = [artist.get_text() for artist in figure.findobj(Text)]
        assert "two leading components" in seen
        assert "words only (LSA)" in seen
        assert "positive side" in seen
        assert any(label.strip().lstrip("-").isdigit() for label in seen)
        for label in seen:
            for question in questions:
                assert question not in label, panel


def test_run_fits_the_text_views_before_the_model(
    inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _never(monkeypatch, "load_model")

    def broken(texts: Any, *, seed: int) -> Any:
        raise ValueError("no text view")

    monkeypatch.setattr(v1_module, "text_view", broken)
    config = _config(inputs)
    run_dir = RunDir.create(config, out_root=tmp_path)

    with pytest.raises(ValueError, match="no text view"):
        run(config, run_dir)


def test_plot_needs_the_views_of_a_run(tmp_path: Path) -> None:
    config = Config(labels_run="runs/p9a", reading_run="runs/p9b")
    run_dir = RunDir.create(config, out_root=tmp_path)

    with pytest.raises(FileNotFoundError):
        plot(run_dir, tmp_path / "figures")


def test_plot_writes_one_check_figure_per_panel(
    default_run: RunDir, tmp_path: Path
) -> None:
    paths = plot(default_run, tmp_path / "figures")

    names = ["views_entity.png", "views_question.png", "views_unknown.png"]
    assert [path.name for path in paths] == names
    assert sorted(p.name for p in (tmp_path / "figures").iterdir()) == names
    for path in paths:
        assert path.read_bytes().startswith(b"\x89PNG")


def test_plot_is_deterministic(default_run: RunDir, tmp_path: Path) -> None:
    first = plot(default_run, tmp_path / "a")
    second = plot(default_run, tmp_path / "b")

    assert len(first) == len(second) == 3
    for one, two in zip(first, second, strict=True):
        assert one.read_bytes() == two.read_bytes(), one.name


def _points(ax: Any) -> list[int]:
    return [len(collection.get_offsets()) for collection in ax.collections]


@CHOSEN_LAYERS
def test_a_panel_figure_has_a_row_per_view_layer_and_one_for_the_text(
    run_at: Callable[[int], RunDir], chosen: int, shown: list[int]
) -> None:
    run_dir = run_at(chosen)
    sidecar = _sidecar(run_dir)
    entry = sidecar["panels"]["question"]
    n = len(entry["uids"])

    figure = _figure(run_dir, "question")

    visible = [ax for ax in figure.axes if ax.get_visible()]
    assert len(visible) == len(shown) * (1 + len(entry["types"])) + 1
    assert [ax.get_ylabel() for ax in visible if ax.get_ylabel()] == [
        *(f"layer {layer}" for layer in shown),
        "words only (LSA)",
    ]
    for ax in visible:
        assert sum(_points(ax)) == n


def _drawn_views() -> tuple[
    dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]
]:
    """Planted views of one panel whose items sit on different sides."""
    uids = [f"u{i}" for i in range(6)]
    sidecar = {
        "uids": {
            "a": {"positive": ["u0", "u1"], "negative": ["u2"]},
            "b": {"positive": ["u2", "u5"], "negative": ["u3", "u4"]},
        },
        "panels": {"p": {"types": ["a", "b"], "uids": uids}},
    }
    rng = np.random.default_rng(0)

    def planted(*shape: int) -> np.ndarray:
        return rng.standard_normal(shape).astype(np.float32)

    pca = {
        "view_layers": np.array([1, 2, 3]),
        "coordinates_p": planted(3, 6, 2),
        "supervised_a": planted(3, 6, 2),
        "supervised_b": planted(3, 6, 2),
    }
    return sidecar, pca, {"coordinates_p": planted(6, 2)}


def _assert_sides(
    ax: Any, points: np.ndarray, codes: list[int], where: str
) -> None:
    """Checks the points of each group of an axes against the planted ones."""
    drawn = [np.asarray(c.get_offsets()) for c in ax.collections]
    assert len(drawn) == 3, where
    for offsets, code in zip(drawn, (-1, 0, 1), strict=True):
        want = points[np.array(codes) == code]
        assert offsets.shape == want.shape, f"{where}, group {code}"
        np.testing.assert_allclose(offsets, want, err_msg=f"{where} {code}")


def test_a_panel_figure_draws_the_stored_coordinates_by_side() -> None:
    sidecar, pca, text = _drawn_views()
    both = [0, 0, 1, 1, 1, 0]
    only_a = [0, 0, 1, -1, -1, -1]
    only_b = [-1, -1, 0, 1, 1, 0]

    figure = panel_figure("p", sidecar, pca, text)

    axes = np.array(figure.axes).reshape(4, 3)
    for v in range(3):
        _assert_sides(axes[v, 0], pca["coordinates_p"][v], both, f"all {v}")
        _assert_sides(axes[v, 1], pca["supervised_a"][v], only_a, f"a {v}")
        _assert_sides(axes[v, 2], pca["supervised_b"][v], only_b, f"b {v}")
    _assert_sides(axes[3, 0], text["coordinates_p"], both, "words")
    assert [ax.get_ylabel() for ax in axes[:, 0]] == [
        "layer 1",
        "layer 2",
        "layer 3",
        "words only (LSA)",
    ]
    assert [ax.get_title() for ax in axes[0]] == [
        "two leading components",
        "a",
        "b",
    ]
