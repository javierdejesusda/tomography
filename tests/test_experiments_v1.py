"""Tests for the knowledge views experiment."""

import dataclasses
import hashlib
import json
from pathlib import Path
import runpy
import shutil
from typing import Any

import numpy as np
import pytest

from tomography.cli import load_experiment
from tomography.experiments import p9b_reading
from tomography.experiments.base import RunDir
from tomography.experiments.sweep import chat_format_for
from tomography.experiments.v1_knowledge_views import CommittedReading
from tomography.experiments.v1_knowledge_views import Config
from tomography.experiments.v1_knowledge_views import check_reading
from tomography.experiments.v1_knowledge_views import compare_auroc
from tomography.experiments.v1_knowledge_views import compare_surface
from tomography.experiments.v1_knowledge_views import displayed_items
from tomography.experiments.v1_knowledge_views import load_reading
from tomography.experiments.v1_knowledge_views import plot
from tomography.experiments.v1_knowledge_views import refit_methods
from tomography.experiments.v1_knowledge_views import refit_shuffled
from tomography.experiments.v1_knowledge_views import run
from tomography.experiments.v1_knowledge_views import select_items
from tomography.experiments.v1_knowledge_views import shuffled_draw
from tomography.experiments.v1_knowledge_views import train_draws
from tomography.knowledge import DEFAULT_SHOTS
from tomography.model import LoadedModel
from tomography.p9.reading import SURFACE_BASELINES
from tomography.p9.reading import ItemActivations
from tomography.p9.reading import collect_item_activations
from tomography.p9.reading import draw_seed
from tomography.p9.reading import draw_train
from tomography.p9.reading import p9_template
from tomography.p9.reading import reading_cube
from tomography.p9.unknowns import TypePool

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


def test_load_reading_refuses_arrays_that_disagree_on_the_contrasts(
    inputs: Inputs, tmp_path: Path
) -> None:
    path = _copy(inputs.reading, tmp_path)
    document = json.loads((path / "readers.json").read_text("utf-8"))
    document["data"]["types"] = document["data"]["types"][::-1]
    (path / "readers.json").write_text(json.dumps(document), "utf-8")

    with pytest.raises(ValueError, match="contrasts"):
        load_reading(path, model="tiny")


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


def test_compare_surface_matches_the_committed_baselines(
    inputs: Inputs,
) -> None:
    committed = load_reading(inputs.reading.path, model="tiny")
    pools = _pools(committed)

    report = compare_surface(pools, committed)

    assert list(report) == USABLE
    for name, by_baseline in report.items():
        assert list(by_baseline) == list(SURFACE_BASELINES)
        for baseline, entry in by_baseline.items():
            assert entry["equal"] is True, (name, baseline)
            assert entry["committed"] == committed.table[name][baseline]
    assert report["obscure_real"]["popularity"]["recomputed"] is not None
    assert report["false_premise"]["popularity"] == {
        "recomputed": None,
        "committed": None,
        "equal": True,
    }


def test_compare_surface_flags_a_baseline_that_moved(inputs: Inputs) -> None:
    committed = load_reading(inputs.reading.path, model="tiny")
    pools = _pools(committed)
    table = {name: dict(row) for name, row in committed.table.items()}
    table["obscure_real"]["text_word"] += 0.01
    table["false_premise"]["popularity"] = 0.7
    moved = dataclasses.replace(committed, table=table)

    report = compare_surface(pools, moved)

    unequal = [
        (name, baseline)
        for name, by_baseline in report.items()
        for baseline, entry in by_baseline.items()
        if not entry["equal"]
    ]
    assert unequal == [
        ("obscure_real", "text_word"),
        ("false_premise", "popularity"),
    ]
