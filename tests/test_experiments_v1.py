"""Tests for the knowledge views experiment."""

import dataclasses
import getpass
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import socket
from typing import Any

import numpy as np
import pytest
import torch

from tomography.cli import load_experiment
from tomography.experiments import p9b_reading
from tomography.experiments.base import RunDir
from tomography.experiments.sweep import chat_format_for
import tomography.experiments.v1_knowledge_views as v1_module
from tomography.experiments.v1_knowledge_views import COUNT_COLUMNS
from tomography.experiments.v1_knowledge_views import CommittedReading
from tomography.experiments.v1_knowledge_views import Config
from tomography.experiments.v1_knowledge_views import check_reading
from tomography.experiments.v1_knowledge_views import compare_auroc
from tomography.experiments.v1_knowledge_views import compare_surface
from tomography.experiments.v1_knowledge_views import displayed_items
from tomography.experiments.v1_knowledge_views import execute
from tomography.experiments.v1_knowledge_views import load_reading
from tomography.experiments.v1_knowledge_views import plot
from tomography.experiments.v1_knowledge_views import published_views
from tomography.experiments.v1_knowledge_views import refit_methods
from tomography.experiments.v1_knowledge_views import refit_shuffled
from tomography.experiments.v1_knowledge_views import run
from tomography.experiments.v1_knowledge_views import select_items
from tomography.experiments.v1_knowledge_views import shuffled_draw
from tomography.experiments.v1_knowledge_views import train_draws
from tomography.export import ExportError
from tomography.export import export_runs
from tomography.export import is_exported
from tomography.knowledge import DEFAULT_SHOTS
from tomography.model import LoadedModel
from tomography.p9.reading import READING_METHODS
from tomography.p9.reading import SURFACE_BASELINES
from tomography.p9.reading import ItemActivations
from tomography.p9.reading import collect_item_activations
from tomography.p9.reading import draw_seed
from tomography.p9.reading import draw_train
from tomography.p9.reading import p9_template
from tomography.p9.reading import reading_cube
from tomography.p9.reading import type_auroc
from tomography.p9.unknowns import TypePool
from tomography.reading.reader import RepReader

PINNED_HASH = "3257d1220951"
# The planted pool and the labels run of the P9b tests are the inputs here.
_P9B = runpy.run_path(
    str(Path(__file__).with_name("test_experiments_p9b_reading.py"))
)
_items = _P9B["_items"]
_labels_run = _P9B["_labels_run"]
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


def test_plot_draws_nothing_before_the_views_exist(tmp_path: Path) -> None:
    config = Config(labels_run="runs/p9a", reading_run="runs/p9b")
    run_dir = RunDir.create(config, out_root=tmp_path)

    assert plot(run_dir, tmp_path / "figures") == []


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


def test_load_reading_refuses_an_unfinished_run(
    inputs: Inputs, tmp_path: Path
) -> None:
    path = _copy(inputs.reading, tmp_path)
    _edit_json(path / "meta.json", finished_at=None)

    with pytest.raises(ValueError, match="not a finished run"):
        load_reading(path, model="tiny")


def test_load_reading_refuses_a_run_of_another_model(inputs: Inputs) -> None:
    with pytest.raises(ValueError, match=r"'tiny'.*'gpt2'"):
        load_reading(inputs.reading.path, model="gpt2")


def test_load_reading_refuses_a_run_of_another_experiment(
    inputs: Inputs, tmp_path: Path
) -> None:
    path = _copy(inputs.reading, tmp_path)
    _edit_json(path / "meta.json", experiment="p9c")

    with pytest.raises(ValueError, match="not a P9b run"):
        load_reading(path, model="tiny")


@pytest.mark.parametrize(
    "name",
    [
        "summary.json",
        "table.json",
        "transfer.npz",
        "transfer.json",
        "readers.npz",
        "readers.json",
    ],
)
def test_load_reading_refuses_a_run_without_its_files(
    inputs: Inputs, tmp_path: Path, name: str
) -> None:
    path = _copy(inputs.reading, tmp_path)
    (path / name).unlink()

    with pytest.raises(ValueError, match=name.replace(".", r"\.")):
        load_reading(path, model="tiny")


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"template": "plain"}, "template"),
        ({"methods": ["pca_diff", "logistic"]}, "methods"),
    ],
    ids=["template", "methods"],
)
def test_load_reading_refuses_a_protocol_it_cannot_reproduce(
    inputs: Inputs, tmp_path: Path, change: dict[str, Any], match: str
) -> None:
    path = _copy(inputs.reading, tmp_path)
    _edit_json(path / "config.json", **change)

    with pytest.raises(ValueError, match=match):
        load_reading(path, model="tiny")


def test_load_reading_refuses_a_run_whose_draws_are_not_those_of_its_config(
    inputs: Inputs, tmp_path: Path
) -> None:
    path = _copy(inputs.reading, tmp_path)
    _edit_json(path / "config.json", n_seeds=3)

    with pytest.raises(ValueError, match="draws"):
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
    inputs: Inputs,
) -> None:
    committed = load_reading(inputs.reading.path, model="tiny")

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


def _pools(committed: CommittedReading) -> dict[str, TypePool]:
    usable, _ = p9b_reading.usable_pools(committed.config, _items(), True)
    return usable


def test_train_draws_are_the_draws_of_the_reading(inputs: Inputs) -> None:
    committed = load_reading(inputs.reading.path, model="tiny")
    pools = _pools(committed)

    draws = train_draws(pools, committed.config)

    assert list(draws) == USABLE
    for name, pool in pools.items():
        assert len(draws[name]) == committed.config.n_seeds
        for s, draw in enumerate(draws[name]):
            want = draw_train(
                pool,
                n_pairs=committed.config.n_pairs,
                seed=draw_seed(committed.config.seed, s),
            )
            assert [i.example.uid for i in draw.items] == [
                i.example.uid for i in want.items
            ]
            np.testing.assert_array_equal(draw.pairs, want.pairs)
    held = [
        [{i.example.uid for i in draw.items} for draw in by_draw]
        for by_draw in draws.values()
    ]
    assert any(first != second for first, second in held)


# Taken from `fit_type_reader` before it was split in two: the number of train
# items, a digest of their uids in order and the first pair, by contrast and
# draw, for the planted pool and the settings of the tiny P9b run.
PRE_SPLIT_DRAWS = {
    ("fictional_synthal", 0): (32, "fbbb5612ff70", ("synthal:21", "popqa:96")),
    ("fictional_synthal", 1): (32, "8d1b700cccb2", ("synthal:45", "popqa:132")),
    ("fictional_generated", 0): (
        32,
        "09f1b28f0bdd",
        ("generated:21", "popqa:96"),
    ),
    ("fictional_generated", 1): (
        32,
        "cc4750317a62",
        ("generated:45", "popqa:132"),
    ),
    ("obscure_real", 0): (30, "588accbb9e21", ("popqa:135", "popqa:120")),
    ("obscure_real", 1): (30, "588accbb9e21", ("popqa:135", "popqa:132")),
    ("obscure_real_pop", 0): (26, "2dc1f4820469", ("popqa:75", "popqa:102")),
    ("obscure_real_pop", 1): (26, "16054c398e2d", ("popqa:141", "popqa:0")),
    ("post_cutoff", 0): (32, "d1c3309a9f01", ("award-real:20", "award-post:6")),
    ("post_cutoff", 1): (
        32,
        "7ade9b0d8965",
        ("award-real:37", "award-post:11"),
    ),
    ("triviaqa_known", 0): (32, "e92d663864e3", ("triviaqa:46", "triviaqa:45")),
    ("triviaqa_known", 1): (32, "72ab35b6205c", ("triviaqa:50", "triviaqa:7")),
    ("unknowable_detail", 0): (32, "f52fb8e8cde4", ("detail:7", "nq_open:16")),
    ("unknowable_detail", 1): (32, "c94342f4cf40", ("detail:7", "triviaqa:42")),
    ("false_premise", 0): (32, "853da9720344", ("falseqa:7", "nq_open:16")),
    ("false_premise", 1): (32, "9fe2b4fe41d4", ("falseqa:7", "triviaqa:42")),
    ("kuq_future_unknown", 0): (
        32,
        "dfa780958d2b",
        ("kuq-future:7", "nq_open:16"),
    ),
    ("kuq_future_unknown", 1): (
        32,
        "b0144373cbda",
        ("kuq-future:7", "triviaqa:42"),
    ),
    ("real_vs_synthal", 0): (32, "98ddd719c2cc", ("popqa:81", "synthal:27")),
    ("real_vs_synthal", 1): (32, "0d7952c92ee5", ("popqa:147", "synthal:45")),
    ("real_vs_generated", 0): (
        32,
        "a976ee6d4091",
        ("popqa:81", "generated:27"),
    ),
    ("real_vs_generated", 1): (
        32,
        "91b23b5c4b8b",
        ("popqa:147", "generated:45"),
    ),
}


def _digest(uids: list[str]) -> str:
    return hashlib.sha256("\n".join(uids).encode()).hexdigest()[:12]


def test_train_draws_are_those_the_fit_of_the_p9b_run_drew(
    inputs: Inputs,
) -> None:
    committed = load_reading(inputs.reading.path, model="tiny")
    pools = _pools(committed)

    draws = train_draws(pools, committed.config)

    assert {name for name, _ in PRE_SPLIT_DRAWS} == set(USABLE)
    for (name, s), (count, digest, first) in PRE_SPLIT_DRAWS.items():
        draw = draws[name][s]
        uids = [i.example.uid for i in draw.items]
        assert len(uids) == count, (name, s)
        assert _digest(uids) == digest, (name, s)
        assert draw.pairs is not None
        pair = (uids[draw.pairs[0, 0]], uids[draw.pairs[0, 1]])
        assert pair == first, (name, s)


def test_the_selection_is_the_test_items_and_the_train_items_of_the_draws(
    inputs: Inputs,
) -> None:
    committed = load_reading(inputs.reading.path, model="tiny")
    pools = _pools(committed)
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
    inputs: Inputs,
) -> None:
    committed = load_reading(inputs.reading.path, model="tiny")
    pools = _pools(committed)
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
    tiny_arch: LoadedModel, inputs: Inputs
) -> None:
    committed = load_reading(inputs.reading.path, model="tiny")

    check_reading(committed, _pools(committed), tiny_arch.n_layers)


def test_check_reading_refuses_other_contrasts(
    tiny_arch: LoadedModel, inputs: Inputs
) -> None:
    committed = load_reading(inputs.reading.path, model="tiny")
    pools = _pools(committed)
    del pools["false_premise"]

    with pytest.raises(ValueError, match=r"contrasts.*false_premise"):
        check_reading(committed, pools, tiny_arch.n_layers)


def test_check_reading_refuses_other_item_counts(
    tiny_arch: LoadedModel, inputs: Inputs
) -> None:
    committed = load_reading(inputs.reading.path, model="tiny")
    pools = _pools(committed)
    pool = pools["unknowable_detail"]
    dropped = next(i for i in pool.negative if i.example.split == "test")
    pools["unknowable_detail"] = dataclasses.replace(
        pool, negative=tuple(i for i in pool.negative if i is not dropped)
    )

    with pytest.raises(ValueError, match="unknowable_detail"):
        check_reading(committed, pools, tiny_arch.n_layers)


def test_check_reading_refuses_a_model_of_another_depth(
    tiny_arch: LoadedModel, inputs: Inputs
) -> None:
    committed = load_reading(inputs.reading.path, model="tiny")

    with pytest.raises(ValueError, match="layers"):
        check_reading(committed, _pools(committed), tiny_arch.n_layers + 1)


@dataclasses.dataclass(frozen=True)
class Read:
    """The committed run, its contrasts and the activations behind them."""

    committed: CommittedReading
    pools: dict[str, TypePool]
    acts: ItemActivations


def _read(tiny_arch: LoadedModel, inputs: Inputs) -> Read:
    committed = load_reading(inputs.reading.path, model="tiny")
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
    inputs: Inputs,
) -> None:
    committed = load_reading(inputs.reading.path, model="tiny")
    pools, acts = _planted(_pools(committed)["obscure_real"])
    protocol = dataclasses.replace(committed.config, n_seeds=4, n_pairs=32)
    draws = train_draws(pools, protocol)

    true = refit_methods(acts, pools, draws, protocol)[0, 0]
    shuffled = refit_shuffled(acts, pools, draws, protocol, seed=0)[0]

    assert true[:, 1:].min() > 0.9
    assert shuffled.shape == true.shape
    assert abs(shuffled[:, 1:].mean() - 0.5) < 0.15


def test_refit_shuffled_is_deterministic_and_follows_the_seed(
    inputs: Inputs,
) -> None:
    committed = load_reading(inputs.reading.path, model="tiny")
    pools, acts = _planted(_pools(committed)["obscure_real"])
    protocol = committed.config
    draws = train_draws(pools, protocol)

    one = refit_shuffled(acts, pools, draws, protocol, seed=0)
    again = refit_shuffled(acts, pools, draws, protocol, seed=0)
    other = refit_shuffled(acts, pools, draws, protocol, seed=1)

    np.testing.assert_array_equal(one, again)
    assert not np.array_equal(one, other)


def test_shuffled_draw_permutes_the_labels_inside_each_stratum(
    inputs: Inputs,
) -> None:
    committed = load_reading(inputs.reading.path, model="tiny")
    pool = _pools(committed)["obscure_real_pop"]
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
    inputs: Inputs,
) -> None:
    committed = load_reading(inputs.reading.path, model="tiny")
    pools = _pools(committed)

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


def test_compare_surface_flags_a_control_that_moved(inputs: Inputs) -> None:
    committed = load_reading(inputs.reading.path, model="tiny")
    pools = _pools(committed)
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


def _summary(run_dir: RunDir) -> dict[str, Any]:
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    return summary


def test_execute_writes_the_files_of_a_run(default_run: RunDir) -> None:
    names = sorted(p.name for p in default_run.path.iterdir() if p.is_file())

    assert names == [
        "config.json",
        "meta.json",
        "projections.json",
        "projections.npz",
        "summary.json",
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
    sidecar = default_run.read_json("projections")
    assert isinstance(sidecar, dict)
    assert sidecar["types"] == USABLE
    assert sidecar["methods"] == list(READING_METHODS)
    assert sidecar["max_per_side"] == 300
    for name in USABLE:
        sides = sidecar["uids"][name]
        shown = len(sides["positive"]) + len(sides["negative"])
        assert arrays[f"projection_{name}"].shape == (shown,)


def test_the_summary_records_the_run(
    tiny_arch: LoadedModel, inputs: Inputs, default_run: RunDir
) -> None:
    committed = load_reading(inputs.reading.path, model="tiny")

    summary = _summary(default_run)

    assert set(summary) == {
        "layer",
        "template",
        "methods",
        "types",
        "n_items",
        "reproduction",
        "surface",
        "surface_equal",
    }
    assert summary["layer"] == committed.layer
    assert summary["template"] == "fewshot"
    assert summary["methods"] == list(READING_METHODS)
    assert summary["types"] == USABLE
    pools = _pools(committed)
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


def test_the_recomputed_auroc_is_the_committed_one_on_the_same_activations(
    inputs: Inputs, default_run: RunDir
) -> None:
    committed = load_reading(inputs.reading.path, model="tiny")

    test = default_run.read_arrays("projections")["test"]

    np.testing.assert_array_equal(test, committed.test)
    reproduction = _summary(default_run)["reproduction"]
    assert reproduction["within_tolerance"] is True
    assert reproduction["tolerance"] == 0.02
    assert set(reproduction["max_abs_diff"].values()) == {0.0}


def test_the_verdict_looks_at_the_column_of_the_chosen_layer(
    tiny_arch: LoadedModel, inputs: Inputs, tmp_path: Path
) -> None:
    committed = load_reading(inputs.reading.path, model="tiny")
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
    inputs: Inputs, default_run: RunDir
) -> None:
    committed = load_reading(inputs.reading.path, model="tiny")
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
    tiny_arch: LoadedModel,
    inputs: Inputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture(monkeypatch)
    committed = load_reading(inputs.reading.path, model="tiny")
    pools = _pools(committed)

    _execute(tiny_arch, tmp_path, inputs, batch_size=4)

    selected = select_items(pools, train_draws(pools, committed.config))
    assert [i.example.uid for i in captured.items] == [
        i.example.uid for i in selected
    ]
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
    tiny_arch: LoadedModel,
    inputs: Inputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture(monkeypatch)
    committed = load_reading(inputs.reading.path, model="tiny")

    run_dir = _execute(tiny_arch, tmp_path, inputs)

    assert captured.acts is not None
    reads = captured.acts
    views = published_views(committed, reads, _pools(committed))
    arrays = run_dir.read_arrays("projections")
    np.testing.assert_array_equal(arrays["published"], views.auroc)
    for key, projection in views.projections.items():
        np.testing.assert_array_equal(arrays[key], projection)
    sidecar = run_dir.read_json("projections")
    assert isinstance(sidecar, dict)
    assert sidecar["uids"] == views.uids


def test_the_shuffled_control_is_written_beside_the_methods(
    tiny_arch: LoadedModel,
    inputs: Inputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture(monkeypatch)
    committed = load_reading(inputs.reading.path, model="tiny")
    pools = _pools(committed)

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
    inputs: Inputs, default_run: RunDir
) -> None:
    committed = load_reading(inputs.reading.path, model="tiny")

    surface = _summary(default_run)["surface"]

    assert surface == compare_surface(_pools(committed), committed)


def test_a_difference_beyond_the_tolerance_is_reported_not_hidden(
    tiny_arch: LoadedModel, inputs: Inputs, tmp_path: Path
) -> None:
    committed = load_reading(inputs.reading.path, model="tiny")
    moved = dataclasses.replace(committed, test=committed.test - 0.125)

    run_dir = _execute(tiny_arch, tmp_path, inputs, reading=moved)

    reproduction = _summary(run_dir)["reproduction"]
    assert reproduction["within_tolerance"] is False
    assert reproduction["max_abs_diff"]["mean_at_layer"] > 0.02
    assert reproduction["max_abs_diff"]["mean_all_layers"] > 0.02


def test_a_surface_control_that_moved_is_reported_not_hidden(
    tiny_arch: LoadedModel, inputs: Inputs, tmp_path: Path
) -> None:
    committed = load_reading(inputs.reading.path, model="tiny")
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

    one, two = (
        first.read_arrays("projections"),
        second.read_arrays("projections"),
    )
    assert one.keys() == two.keys()
    for key in one:
        np.testing.assert_array_equal(one[key], two[key], err_msg=key)
    assert first.read_json("projections") == second.read_json("projections")
    assert first.read_json("summary") == second.read_json("summary")


def test_the_uids_hold_no_question_text(default_run: RunDir) -> None:
    text = (default_run.path / "projections.json").read_text("utf-8")

    assert "Who is" not in text
    sidecar = default_run.read_json("projections")
    assert isinstance(sidecar, dict)
    known = {i.example.uid for i in _items()}
    for sides in sidecar["uids"].values():
        assert set(sides["positive"]) | set(sides["negative"]) <= known


@pytest.mark.parametrize(
    ("overrides", "match"),
    [({"batch_size": 0}, "batch_size")],
    ids=["batch_size"],
)
def test_a_setting_that_cannot_work_fails_before_the_forward_pass(
    tiny_arch: LoadedModel,
    inputs: Inputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
    match: str,
) -> None:
    def never(*args: Any, **kwargs: Any) -> ItemActivations:
        raise AssertionError("the forward pass must not start")

    monkeypatch.setattr(v1_module, "collect_item_activations", never)

    with pytest.raises(ValueError, match=match):
        _execute(tiny_arch, tmp_path, inputs, **overrides)


def test_labels_that_differ_from_the_reading_fail_before_the_forward_pass(
    tiny_arch: LoadedModel,
    inputs: Inputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def never(*args: Any, **kwargs: Any) -> ItemActivations:
        raise AssertionError("the forward pass must not start")

    monkeypatch.setattr(v1_module, "collect_item_activations", never)
    items = [i for i in _items() if not i.example.uid.startswith("falseqa")]

    with pytest.raises(ValueError, match="contrasts"):
        _execute(tiny_arch, tmp_path, inputs, items=items)


def test_a_control_that_cannot_work_fails_before_the_forward_pass(
    tiny_arch: LoadedModel,
    inputs: Inputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def never(*args: Any, **kwargs: Any) -> ItemActivations:
        raise AssertionError("the forward pass must not start")

    def broken(pool: Any) -> dict[str, float | None]:
        raise ValueError("no control")

    monkeypatch.setattr(v1_module, "collect_item_activations", never)
    monkeypatch.setattr(v1_module, "surface_baselines", broken)

    with pytest.raises(ValueError, match="no control"):
        _execute(tiny_arch, tmp_path, inputs)


def test_run_reads_both_runs_and_loads_the_model_on_the_cpu_in_float32(
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

    assert asked == [("tiny", {"device": "cpu", "dtype": torch.float32})]
    assert _summary(run_dir)["types"] == USABLE
    assert (run_dir.path / "projections.npz").is_file()


def _never_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    def never(*args: Any, **kwargs: Any) -> LoadedModel:
        raise AssertionError("the model must not load first")

    monkeypatch.setattr(v1_module, "load_model", never)


def test_run_refuses_an_unfinished_labels_run(
    inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _never_loaded(monkeypatch)
    path = _copy(inputs.labels, tmp_path / "labels")
    _edit_json(path / "meta.json", finished_at=None)
    config = dataclasses.replace(_config(inputs), labels_run=str(path))
    run_dir = RunDir.create(config, out_root=tmp_path)

    with pytest.raises(ValueError, match="not a finished run"):
        run(config, run_dir)


def test_run_refuses_an_unfinished_reading_run(
    inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _never_loaded(monkeypatch)
    path = _copy(inputs.reading, tmp_path / "reading")
    _edit_json(path / "meta.json", finished_at=None)
    config = dataclasses.replace(_config(inputs), reading_run=str(path))
    run_dir = RunDir.create(config, out_root=tmp_path)

    with pytest.raises(ValueError, match="not a finished run"):
        run(config, run_dir)


def test_run_refuses_a_reading_run_of_another_model(
    inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _never_loaded(monkeypatch)
    path = _copy(inputs.reading, tmp_path / "reading")
    _edit_json(path / "config.json", model="gpt2")
    config = dataclasses.replace(_config(inputs), reading_run=str(path))
    run_dir = RunDir.create(config, out_root=tmp_path)

    with pytest.raises(ValueError, match=r"'gpt2'.*'tiny'"):
        run(config, run_dir)


def test_run_refuses_a_reading_run_without_its_arrays(
    inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _never_loaded(monkeypatch)
    path = _copy(inputs.reading, tmp_path / "reading")
    (path / "transfer.npz").unlink()
    config = dataclasses.replace(_config(inputs), reading_run=str(path))
    run_dir = RunDir.create(config, out_root=tmp_path)

    with pytest.raises(ValueError, match=r"transfer\.npz"):
        run(config, run_dir)


def _finished_copy(run_dir: RunDir, tmp_path: Path) -> Path:
    """Copies a run and stamps it finished, as the command line does."""
    target = tmp_path / "run"
    shutil.copytree(run_dir.path, target)
    RunDir.open(target).finalize()
    return target


@pytest.fixture
def fixed_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(getpass, "getuser", lambda: "tester-user")
    monkeypatch.setattr(socket, "gethostname", lambda: "testhost-box.lan")


@pytest.mark.usefixtures("fixed_machine")
def test_every_file_of_a_run_passes_the_export(
    default_run: RunDir, tmp_path: Path
) -> None:
    for name in ("projections.npz", "projections.json", "summary.json"):
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
        "projections.json",
        "projections.npz",
        "summary.json",
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
