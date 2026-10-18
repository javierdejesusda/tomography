"""Tests for the P10c links experiment."""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.datasets.quantities import QuantityQuestion
from tomography.experiments import p10c_links
from tomography.experiments.base import RunDir
from tomography.experiments.p9b_reading import Config as P9bConfig
from tomography.experiments.p10a_quantities import Answer
from tomography.experiments.p10a_quantities import Config as P10aConfig
from tomography.experiments.p10b_reading import Config as P10bConfig
from tomography.experiments.p10c_links import Config

PINNED_HASH = "ff466673e0c1"
WIDTH = 8
LAYERS = (1, 2, 3)
READERS = ("plain/logistic", "concept/logistic")
TYPES = ("real_vs_synthal", "real_vs_generated")
USED = ("height", "wingspan")
SPLITS = (("train", 4), ("val", 2), ("test", 6))
SHA = "a" * 64
CELLS = (
    "free/fewshot/range_last",
    "free/fewshot/exact_last",
    "forced/fewshot/range_last",
    "forced/fewshot/exact_last",
)


def test_config_defaults_and_round_trip() -> None:
    config = Config(p10b_run="runs/p10b", knowledge_run="runs/p9b")

    assert config.experiment == "p10c"
    assert config.model == "gpt2"
    assert config.seed == 0
    assert config.p10b_run == "runs/p10b"
    assert config.knowledge_run == "runs/p9b"
    assert config.behavior_run is None
    assert config.reader == "plain/logistic"
    assert config.n_boot == 1000
    assert config.n_random == 200
    assert config.batch_size == 8
    assert config.offline is True
    assert Config.from_dict(config.to_dict()) == config


def test_both_links_round_trip() -> None:
    config = Config(
        p10b_run="a",
        knowledge_run="b",
        behavior_run="c",
        reader="concept/mean_diff",
    )

    assert Config.from_dict(config.to_dict()) == config


def test_the_p10b_run_is_required() -> None:
    with pytest.raises(ValueError, match="p10b_run"):
        Config.from_dict({"knowledge_run": "y"})


def test_the_config_hash_is_pinned() -> None:
    config = Config(p10b_run="x", knowledge_run="y")

    assert config.config_hash() == PINNED_HASH


def test_a_link_is_required() -> None:
    with pytest.raises(ValueError, match="knowledge_run"):
        Config(p10b_run="x")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_boot", 0),
        ("n_random", 0),
        ("batch_size", 0),
        ("reader", "plain"),
        ("reader", "plain/logistic/extra"),
        ("reader", "chat/logistic"),
        ("reader", "plain/kmeans"),
        ("reader", ""),
    ],
)
def test_bad_settings_raise_before_any_work(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        Config(p10b_run="x", knowledge_run="y", **{field: value})


def _item(
    uid: str,
    family: str,
    kind: str,
    split: str,
    *,
    question: str | None = None,
    sitelinks: int = 10,
    sha: str = SHA,
) -> QuantityQuestion:
    return QuantityQuestion(
        uid=uid,
        family=family,
        kind=kind,  # type: ignore[arg-type]
        entity=uid,
        entity_id=f"Q{abs(hash(uid)) % 10000}",
        question=question or f"What is the {family} of {uid}?",
        low=1.0,
        high=2.0 if kind == "range" else 1.0,
        unit="metres",
        sitelinks=sitelinks,
        split=split,  # type: ignore[arg-type]
        retrieved_on="2026-01-01",
        sha256=sha,
    )


def _items(families: Sequence[str] = USED) -> list[QuantityQuestion]:
    return [
        _item(
            f"{family}-{kind}-{split}{i}", family, kind, split, sitelinks=5 + i
        )
        for family in families
        for kind in ("range", "exact")
        for split, n in SPLITS
        for i in range(n)
    ]


def _people(n: int = 4) -> list[QuantityQuestion]:
    return [
        _item(f"person_height-exact-q{i}", "person_height", "exact", "test")
        for i in range(n)
    ]


def _stamps(
    items: Sequence[QuantityQuestion],
) -> dict[str, dict[str, dict[str, list[str]]]]:
    stamps: dict[str, dict[str, dict[str, list[str]]]] = {}
    for item in items:
        stamp = stamps.setdefault(item.family, {}).setdefault(
            item.kind, {"retrieved_on": [], "sha256": []}
        )
        stamp["retrieved_on"] = [item.retrieved_on]
        stamp["sha256"] = [item.sha256]
    return stamps


def _unit(vectors: NDArray[np.float64]) -> NDArray[np.float32]:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return (vectors / norms).astype(np.float32)


Score = Callable[[QuantityQuestion, str], float]


def _default_score(item: QuantityQuestion, reader: str) -> float:
    return float(len(item.uid) % 5) - 2.0


def _write_p10b(
    root: Path,
    items: Sequence[QuantityQuestion],
    *,
    model: str = "tiny",
    directions: NDArray[np.float32] | None = None,
    signs: NDArray[np.float32] | None = None,
    layer: int = 2,
    score: Score = _default_score,
    finished: bool = True,
    config: P10bConfig | None = None,
) -> RunDir:
    config = config or P10bConfig(model=model, families=USED)
    run_dir = RunDir.create(config, out_root=root)
    if directions is None:
        rng = np.random.default_rng(1)
        directions = _unit(rng.normal(size=(len(READERS), len(LAYERS), WIDTH)))
    if signs is None:
        signs = np.ones((len(READERS), len(LAYERS)), dtype=np.float32)
    run_dir.write_arrays(
        "readers",
        directions=directions,
        means=np.zeros_like(directions),
        signs=signs,
        layers=np.asarray(LAYERS, dtype=np.int64),
    )
    run_dir.write_json(
        "readers", {"readers": list(READERS), "train_set": "all"}
    )
    run_dir.write_json(
        "items",
        [
            {
                "uid": i.uid,
                "family": i.family,
                "kind": i.kind,
                "split": i.split,
                "sitelinks": i.sitelinks,
                "scores": {r: score(i, r) for r in READERS},
            }
            for i in items
        ],
    )
    run_dir.write_json(
        "summary",
        {
            "model": model,
            "layers": {f"{r}/all": layer for r in READERS},
            "snapshots": _stamps(items),
            "config": config.to_dict(),
        },
    )
    if finished:
        run_dir.finalize()
    return run_dir


def _write_p9b(
    root: Path,
    *,
    model: str = "tiny",
    types: Sequence[str] = TYPES,
    directions: NDArray[np.float32] | None = None,
    means: NDArray[np.float32] | None = None,
    signs: NDArray[np.float32] | None = None,
    layers: Sequence[int] = LAYERS,
    width: int = WIDTH,
    finished: bool = True,
    config: P9bConfig | None = None,
) -> RunDir:
    config = config or P9bConfig(
        model=model, labels_run="labels", template="fewshot", n_shots=3
    )
    run_dir = RunDir.create(config, out_root=root)
    if directions is None:
        rng = np.random.default_rng(2)
        directions = _unit(rng.normal(size=(len(types), len(layers), width)))
    if means is None:
        means = np.zeros_like(directions)
    if signs is None:
        signs = np.ones((len(types), len(layers)), dtype=np.float32)
    run_dir.write_arrays(
        "readers",
        directions=directions,
        means=means,
        signs=signs,
        layers=np.asarray(layers, dtype=np.int64),
    )
    run_dir.write_json("readers", {"types": list(types)})
    if finished:
        run_dir.finalize()
    return run_dir


def _answer(form: str) -> dict[str, object]:
    return dataclasses.asdict(
        Answer(
            text="",
            form=form,
            low=None,
            high=None,
            approximate=False,
            correct=None,
            unit_mismatch=False,
        )
    )


def _cell(greedy: str, *samples: str) -> dict[str, object]:
    return {
        "greedy": _answer(greedy),
        "samples": [_answer(form) for form in samples or (greedy,)],
    }


def _hedge_none(item: QuantityQuestion, cell: str) -> dict[str, object]:
    return _cell("point", "point", "point")


def _write_p10a(
    root: Path,
    items: Sequence[QuantityQuestion],
    *,
    model: str = "tiny",
    cells: Sequence[str] = CELLS,
    answers: Callable[[QuantityQuestion, str], dict[str, object]] = _hedge_none,
    finished: bool = True,
    stamps: dict[str, dict[str, dict[str, list[str]]]] | None = None,
) -> RunDir:
    config = P10aConfig(model=model)
    run_dir = RunDir.create(config, out_root=root)
    run_dir.write_json(
        "items",
        [
            {
                **dataclasses.asdict(i),
                "answers": {c: answers(i, c) for c in cells},
            }
            for i in items
        ],
    )
    run_dir.write_json(
        "summary",
        {
            "model": model,
            "snapshots": stamps if stamps is not None else _stamps(items),
            "config": config.to_dict(),
        },
    )
    if finished:
        run_dir.finalize()
    return run_dir


def test_the_p10b_run_gives_the_reader_its_layer_and_the_items(
    tmp_path: Path,
) -> None:
    items = _items()
    written = _write_p10b(tmp_path, items, layer=3)

    run = p10c_links.load_p10b(
        written.path, model="tiny", reader="concept/logistic"
    )

    stored = written.read_arrays("readers")
    np.testing.assert_array_equal(run.directions, stored["directions"][1])
    np.testing.assert_array_equal(run.signs, stored["signs"][1])
    assert run.layers == LAYERS
    assert run.layer == 3
    assert run.reader == "concept/logistic"
    assert run.config_hash == written.meta["config_hash"]
    assert run.config["families"] == ["height", "wingspan"]
    assert [r["uid"] for r in run.items] == [i.uid for i in items]
    assert run.items[0]["scores"]["concept/logistic"] == pytest.approx(
        _default_score(items[0], "x")
    )
    assert run.snapshots == _stamps(items)


def test_a_reader_missing_from_the_p10b_run_is_named(tmp_path: Path) -> None:
    written = _write_p10b(tmp_path, _items())

    with pytest.raises(ValueError, match="plain/mean_diff"):
        p10c_links.load_p10b(
            written.path, model="tiny", reader="plain/mean_diff"
        )


def test_the_p9b_run_gives_the_types_and_the_arrays(tmp_path: Path) -> None:
    written = _write_p9b(tmp_path)

    run = p10c_links.load_p9b(written.path, model="tiny")

    stored = written.read_arrays("readers")
    assert run.types == TYPES
    assert run.layers == LAYERS
    np.testing.assert_array_equal(run.directions, stored["directions"])
    np.testing.assert_array_equal(run.means, stored["means"])
    np.testing.assert_array_equal(run.signs, stored["signs"])
    assert run.config["template"] == "fewshot"
    assert run.config["n_shots"] == 3
    assert run.config_hash == written.meta["config_hash"]


def test_the_p10a_run_gives_the_items_by_uid_and_the_snapshots(
    tmp_path: Path,
) -> None:
    items = _items()
    written = _write_p10a(tmp_path, items)

    run = p10c_links.load_p10a(written.path, model="tiny")

    assert set(run.items) == {i.uid for i in items}
    first = run.items[items[0].uid]
    assert first["split"] == items[0].split
    assert set(first["answers"]) == set(CELLS)
    assert run.snapshots == _stamps(items)
    assert run.config_hash == written.meta["config_hash"]


def _load(kind: str, path: Path, model: str) -> object:
    if kind == "p10b":
        return p10c_links.load_p10b(path, model=model, reader=READERS[0])
    if kind == "p9b":
        return p10c_links.load_p9b(path, model=model)
    return p10c_links.load_p10a(path, model=model)


def _written(kind: str, root: Path, **changes: object) -> RunDir:
    if kind == "p10b":
        return _write_p10b(root, _items(), **changes)  # type: ignore[arg-type]
    if kind == "p9b":
        return _write_p9b(root, **changes)  # type: ignore[arg-type]
    return _write_p10a(root, _items(), **changes)  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", ["p10b", "p9b", "p10a"])
def test_an_unfinished_run_is_refused(kind: str, tmp_path: Path) -> None:
    written = _written(kind, tmp_path, finished=False)

    with pytest.raises(ValueError, match="not a finished run"):
        _load(kind, written.path, "tiny")


@pytest.mark.parametrize("kind", ["p10b", "p9b", "p10a"])
def test_a_run_of_another_model_is_refused(kind: str, tmp_path: Path) -> None:
    written = _written(kind, tmp_path, model="other")

    with pytest.raises(ValueError, match="'other'"):
        _load(kind, written.path, "tiny")


@pytest.mark.parametrize("kind", ["p10b", "p9b", "p10a"])
def test_a_folder_that_is_not_a_run_is_refused(
    kind: str, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="not a run directory"):
        _load(kind, tmp_path / "nowhere", "tiny")


@pytest.mark.parametrize(
    ("kind", "dropped"),
    [("p10b", "readers.npz"), ("p9b", "readers.npz"), ("p10a", "items.json")],
)
def test_an_export_without_the_local_files_is_refused(
    kind: str, dropped: str, tmp_path: Path
) -> None:
    written = _written(kind, tmp_path)
    (written.path / dropped).unlink()

    with pytest.raises(ValueError, match=f"no {dropped}.*full run"):
        _load(kind, written.path, "tiny")


def _linked(
    tmp_path: Path,
) -> tuple[
    p10c_links.P10bRun,
    p10c_links.P9bRun,
    p10c_links.P10aRun,
    list[QuantityQuestion],
]:
    items = _items()
    p10b = p10c_links.load_p10b(
        _write_p10b(tmp_path, items).path, model="tiny", reader=READERS[0]
    )
    p9b = p10c_links.load_p9b(_write_p9b(tmp_path).path, model="tiny")
    p10a = p10c_links.load_p10a(_write_p10a(tmp_path, items).path, model="tiny")
    return p10b, p9b, p10a, items


def test_matching_runs_pass_the_link_checks(tmp_path: Path) -> None:
    p10b, p9b, p10a, _ = _linked(tmp_path)

    p10c_links.check_links(p10b, p9b, p10a)
    p10c_links.check_links(p10b, p9b, None)
    p10c_links.check_links(p10b, None, p10a)


def test_other_layers_in_the_p9b_run_are_refused(tmp_path: Path) -> None:
    p10b, _, _, _ = _linked(tmp_path)
    shifted = p10c_links.load_p9b(
        _write_p9b(tmp_path / "shifted", layers=(2, 3, 4)).path, model="tiny"
    )

    with pytest.raises(ValueError, match="layers"):
        p10c_links.check_links(p10b, shifted, None)


def test_another_hidden_size_in_the_p9b_run_is_refused(tmp_path: Path) -> None:
    p10b, _, _, _ = _linked(tmp_path)
    wide = p10c_links.load_p9b(
        _write_p9b(tmp_path / "wide", width=WIDTH * 2).path, model="tiny"
    )

    with pytest.raises(ValueError, match="hidden size"):
        p10c_links.check_links(p10b, wide, None)


def test_a_p10a_run_without_a_test_item_of_the_p10b_run_is_refused(
    tmp_path: Path,
) -> None:
    p10b, _, _, items = _linked(tmp_path)
    dropped = [i for i in items if i.uid != "height-range-test3"]
    short = p10c_links.load_p10a(
        _write_p10a(tmp_path / "short", dropped).path, model="tiny"
    )

    with pytest.raises(ValueError, match="height-range-test3"):
        p10c_links.check_links(p10b, None, short)


def test_train_items_missing_from_the_p10a_run_do_not_matter(
    tmp_path: Path,
) -> None:
    p10b, _, _, items = _linked(tmp_path)
    tests_only = [i for i in items if i.split == "test"]
    run = p10c_links.load_p10a(
        _write_p10a(tmp_path / "tests", tests_only).path, model="tiny"
    )

    p10c_links.check_links(p10b, None, run)


def test_another_split_for_a_test_uid_is_refused(tmp_path: Path) -> None:
    p10b, _, _, items = _linked(tmp_path)
    moved = [
        dataclasses.replace(i, split="val")
        if i.uid == "wingspan-exact-test0"
        else i
        for i in items
    ]
    other = p10c_links.load_p10a(
        _write_p10a(tmp_path / "moved", moved).path, model="tiny"
    )

    with pytest.raises(ValueError, match="wingspan-exact-test0"):
        p10c_links.check_links(p10b, None, other)


def test_another_snapshot_is_refused(tmp_path: Path) -> None:
    p10b, _, _, items = _linked(tmp_path)
    newer = [dataclasses.replace(i, sha256="b" * 64) for i in items]
    other = p10c_links.load_p10a(
        _write_p10a(tmp_path / "newer", newer).path, model="tiny"
    )

    with pytest.raises(ValueError, match="sha256"):
        p10c_links.check_links(p10b, None, other)


def test_the_people_are_checked_only_when_both_runs_have_them(
    tmp_path: Path,
) -> None:
    items = _items()
    people = _people()
    both = p10c_links.load_p10b(
        _write_p10b(tmp_path / "b", [*items, *people]).path,
        model="tiny",
        reader=READERS[0],
    )
    without = p10c_links.load_p10a(
        _write_p10a(tmp_path / "a", items).path, model="tiny"
    )
    p10c_links.check_links(both, None, without)

    stale = [dataclasses.replace(p, sha256="b" * 64) for p in people]
    other = p10c_links.load_p10a(
        _write_p10a(tmp_path / "c", [*items, *stale]).path, model="tiny"
    )
    with pytest.raises(ValueError, match="person_height"):
        p10c_links.check_links(both, None, other)


def _range_run(
    directions: NDArray[np.float32],
    signs: NDArray[np.float32] | None = None,
    items: Sequence[dict[str, object]] = (),
    layer: int = 2,
) -> p10c_links.P10bRun:
    n_layers = directions.shape[0]
    return p10c_links.P10bRun(
        path="p10b",
        config_hash="h10b",
        config={},
        reader=READERS[0],
        layer=layer,
        layers=tuple(range(1, n_layers + 1)),
        directions=directions,
        signs=np.ones(n_layers, np.float32) if signs is None else signs,
        items=tuple(items),
        snapshots={},
    )


def _known_run(
    directions: NDArray[np.float32],
    signs: NDArray[np.float32] | None = None,
    means: NDArray[np.float32] | None = None,
    types: Sequence[str] = TYPES,
    config: dict[str, object] | None = None,
) -> p10c_links.P9bRun:
    n_types, n_layers = directions.shape[:2]
    return p10c_links.P9bRun(
        path="p9b",
        config_hash="h9b",
        config=config or {"template": "fewshot", "n_shots": 3, "bos": None},
        types=tuple(types[:n_types]),
        layers=tuple(range(1, n_layers + 1)),
        directions=directions,
        means=np.zeros_like(directions) if means is None else means,
        signs=np.ones((n_types, n_layers), np.float32)
        if signs is None
        else signs,
    )


def _random_directions(
    shape: tuple[int, ...], seed: int = 0
) -> NDArray[np.float32]:
    return _unit(np.random.default_rng(seed).normal(size=shape))


def _cosines(rows: list[dict[str, object]], p9_type: str) -> dict[int, float]:
    return {
        int(r["layer"]): float(r["cosine"])  # type: ignore[arg-type]
        for r in rows
        if r["p9_type"] == p9_type
    }


def test_cosine_rows_have_one_row_per_type_and_layer() -> None:
    range_dir = _random_directions((3, 16), 1)
    known = _random_directions((2, 3, 16), 2)

    rows = p10c_links.cosine_rows(
        _range_run(range_dir), _known_run(known), n_random=50, seed=0
    )

    assert len(rows) == 6
    assert tuple(rows[0]) == p10c_links.COSINE_COLUMNS
    assert [(r["p9_type"], r["layer"]) for r in rows] == [
        (t, k) for t in TYPES for k in (1, 2, 3)
    ]
    for row in rows:
        assert -1.0 <= float(row["cosine"]) <= 1.0  # type: ignore[arg-type]


def test_the_same_direction_has_cosine_one_and_the_opposite_minus_one() -> None:
    range_dir = _random_directions((3, 16), 1)
    known = _random_directions((2, 3, 16), 2)
    known[0] = range_dir
    known[1] = -range_dir

    rows = p10c_links.cosine_rows(
        _range_run(range_dir), _known_run(known), n_random=50, seed=0
    )

    assert _cosines(rows, TYPES[0]) == pytest.approx({1: 1.0, 2: 1.0, 3: 1.0})
    assert _cosines(rows, TYPES[1]) == pytest.approx(
        {1: -1.0, 2: -1.0, 3: -1.0}
    )


def test_a_sign_flip_of_the_p10b_reader_flips_that_layer() -> None:
    range_dir = _random_directions((3, 16), 1)
    known = _random_directions((1, 3, 16), 2)
    known[0] = range_dir
    signs = np.array([1.0, -1.0, 1.0], dtype=np.float32)

    rows = p10c_links.cosine_rows(
        _range_run(range_dir, signs), _known_run(known), n_random=50, seed=0
    )

    assert _cosines(rows, TYPES[0]) == pytest.approx({1: 1.0, 2: -1.0, 3: 1.0})


def test_a_sign_flip_of_the_p9_reader_flips_that_layer_and_type() -> None:
    range_dir = _random_directions((3, 16), 1)
    known = np.stack([range_dir, range_dir])
    signs = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, 1.0]], dtype=np.float32)

    rows = p10c_links.cosine_rows(
        _range_run(range_dir),
        _known_run(known, signs),
        n_random=50,
        seed=0,
    )

    assert _cosines(rows, TYPES[0]) == pytest.approx({1: 1.0, 2: 1.0, 3: 1.0})
    assert _cosines(rows, TYPES[1]) == pytest.approx({1: 1.0, 2: -1.0, 3: 1.0})


def test_flipping_both_signs_leaves_the_cosine_unchanged() -> None:
    range_dir = _random_directions((3, 16), 1)
    known = np.stack([range_dir])
    flipped = -np.ones((1, 3), dtype=np.float32)

    rows = p10c_links.cosine_rows(
        _range_run(range_dir, -np.ones(3, np.float32)),
        _known_run(known, flipped),
        n_random=50,
        seed=0,
    )

    assert _cosines(rows, TYPES[0]) == pytest.approx({1: 1.0, 2: 1.0, 3: 1.0})


def test_the_cosine_ignores_the_norm_of_the_stored_directions() -> None:
    range_dir = _random_directions((3, 16), 1)
    known = _random_directions((1, 3, 16), 2)
    base = p10c_links.cosine_rows(
        _range_run(range_dir), _known_run(known), n_random=50, seed=0
    )

    scaled = p10c_links.cosine_rows(
        _range_run(3.0 * range_dir),
        _known_run(5.0 * known),
        n_random=50,
        seed=0,
    )

    assert _cosines(scaled, TYPES[0]) == pytest.approx(_cosines(base, TYPES[0]))


def test_a_zero_direction_is_refused() -> None:
    range_dir = _random_directions((3, 16), 1)
    known = _random_directions((1, 3, 16), 2)
    known[0, 1] = 0.0

    with pytest.raises(ValueError, match="zero"):
        p10c_links.cosine_rows(
            _range_run(range_dir), _known_run(known), n_random=50, seed=0
        )


def test_the_band_brackets_zero_and_is_shared_by_the_types() -> None:
    range_dir = _random_directions((3, 64), 1)
    known = _random_directions((2, 3, 64), 2)

    rows = p10c_links.cosine_rows(
        _range_run(range_dir), _known_run(known), n_random=200, seed=0
    )

    for row in rows:
        assert float(row["band_lo"]) < 0.0 < float(row["band_hi"])  # type: ignore[arg-type]
    for layer in (1, 2, 3):
        bands = {
            (r["band_lo"], r["band_hi"]) for r in rows if r["layer"] == layer
        }
        assert len(bands) == 1


def test_the_band_narrows_as_the_width_grows() -> None:
    def width(d: int) -> float:
        rows = p10c_links.cosine_rows(
            _range_run(_random_directions((2, d), 1)),
            _known_run(_random_directions((1, 2, d), 2)),
            n_random=400,
            seed=0,
        )
        return float(rows[0]["band_hi"]) - float(rows[0]["band_lo"])  # type: ignore[arg-type]

    assert width(16) > 2.0 * width(256)


def test_the_band_follows_the_seed_and_repeats_for_one_seed() -> None:
    range_dir = _random_directions((2, 32), 1)
    known = _random_directions((1, 2, 32), 2)

    def band(seed: int) -> tuple[object, object]:
        row = p10c_links.cosine_rows(
            _range_run(range_dir), _known_run(known), n_random=100, seed=seed
        )[0]
        return row["band_lo"], row["band_hi"]

    assert band(0) == band(0)
    assert band(0) != band(1)


def test_the_band_holds_about_95_percent_of_random_cosines() -> None:
    range_dir = _random_directions((2, 32), 1)
    known = _random_directions((1, 2, 32), 2)
    row = p10c_links.cosine_rows(
        _range_run(range_dir), _known_run(known), n_random=400, seed=0
    )[0]
    fresh = _random_directions((5000, 32), 99).astype(np.float64)
    inside = (
        fresh @ range_dir[0].astype(np.float64) >= float(row["band_lo"])
    ) & (  # type: ignore[arg-type]
        fresh @ range_dir[0].astype(np.float64) <= float(row["band_hi"])  # type: ignore[arg-type]
    )

    assert 0.92 < float(inside.mean()) < 0.98
