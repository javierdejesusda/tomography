"""Tests for the P10c links experiment."""

from collections.abc import Callable
from collections.abc import Sequence
import csv
import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.cli import load_experiment
from tomography.datasets.quantities import QuantityQuestion
from tomography.experiments import p10c_links
from tomography.experiments.base import RunDir
from tomography.experiments.p3_size import spearman
from tomography.experiments.p9b_reading import Config as P9bConfig
from tomography.experiments.p10a_quantities import Answer
from tomography.experiments.p10a_quantities import Config as P10aConfig
from tomography.experiments.p10a_quantities import Outcome
from tomography.experiments.p10a_quantities import _item_rates
from tomography.experiments.p10b_reading import Config as P10bConfig
from tomography.experiments.p10c_links import Config
from tomography.experiments.sweep import chat_format_for
from tomography.knowledge import DEFAULT_SHOTS
from tomography.knowledge import few_shot_prompt
from tomography.model import LoadedModel
from tomography.p9.unknowns import UNKNOWN_TYPES
from tomography.templates import ChatFormat

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


def _contrast_of(name: str) -> str:
    return "unknown_type" if name in UNKNOWN_TYPES else "knowledge"


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
    summary: dict[str, object] | None = None,
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
    contrasts = {name: _contrast_of(name) for name in types}
    run_dir.write_json(
        "summary",
        summary
        if summary is not None
        else {
            "types": list(types),
            "contrasts": contrasts,
            "known_contrasts": "knowledge" in contrasts.values(),
        },
    )
    if finished:
        run_dir.finalize()
    return run_dir


def _real_answer(form: str) -> Answer:
    return Answer(
        text="",
        form=form,
        low=None,
        high=None,
        approximate=False,
        correct=None,
        unit_mismatch=False,
    )


def _answer(form: str) -> dict[str, object]:
    return dataclasses.asdict(_real_answer(form))


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


def test_the_p9b_run_gives_the_contrast_of_each_type(tmp_path: Path) -> None:
    types = ("obscure_real", "real_vs_synthal", "pre_vs_post_cutoff")
    written = _write_p9b(tmp_path, types=types)

    run = p10c_links.load_p9b(written.path, model="tiny")

    assert run.types == types
    assert run.contrasts == ("knowledge", "unknown_type", "unknown_type")


def _without_contrasts(types: Sequence[str]) -> dict[str, object]:
    return {"types": list(types)}


def _missing_a_type(types: Sequence[str]) -> dict[str, object]:
    return {"contrasts": {types[0]: "unknown_type"}}


def _with_another_kind(types: Sequence[str]) -> dict[str, object]:
    return {"contrasts": {name: "mixed" for name in types}}


@pytest.mark.parametrize(
    ("summary", "message"),
    [
        (_without_contrasts, "no contrasts"),
        (_missing_a_type, "no contrast for 'real_vs_generated'"),
        (_with_another_kind, "'mixed'"),
    ],
    ids=["no contrasts", "type without contrast", "unknown contrast kind"],
)
def test_a_p9b_run_without_a_usable_contrast_per_type_is_refused(
    summary: Callable[[Sequence[str]], dict[str, object]],
    message: str,
    tmp_path: Path,
) -> None:
    written = _write_p9b(tmp_path, summary=summary(TYPES))

    with pytest.raises(ValueError, match=message):
        p10c_links.load_p9b(written.path, model="tiny")


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
    [
        ("p10b", "readers.npz"),
        ("p9b", "readers.npz"),
        ("p9b", "summary.json"),
        ("p10a", "items.json"),
    ],
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


def _stamped_p10b(
    items: Sequence[QuantityQuestion], people: Sequence[QuantityQuestion]
) -> p10c_links.P10bRun:
    scored = [*items, *people]
    records = [{"uid": i.uid, "split": i.split} for i in scored]
    return dataclasses.replace(
        _range_run(_random_directions((3, 4)), items=records),
        snapshots=_stamps(scored),
    )


def test_rebuilt_items_with_the_stored_snapshots_pass_the_item_check() -> None:
    items, people = _items(), _people(4)

    p10c_links.check_items(items, people, _stamped_p10b(items, people))


@pytest.mark.parametrize(
    ("family", "kind"),
    [("height", "exact"), ("wingspan", "range"), ("person_height", "exact")],
)
def test_rebuilt_items_from_another_snapshot_are_refused(
    family: str, kind: str
) -> None:
    items, people = _items(), _people(4)
    p10b = _stamped_p10b(items, people)

    def moved(item: QuantityQuestion) -> QuantityQuestion:
        if (item.family, item.kind) == (family, kind):
            return dataclasses.replace(item, sha256="b" * 64)
        return item

    with pytest.raises(ValueError, match=f"{family}/{kind} snapshot"):
        p10c_links.check_items(
            [moved(i) for i in items], [moved(i) for i in people], p10b
        )


def test_a_stored_digest_that_the_rebuilt_items_lack_is_refused() -> None:
    items, people = _items(), _people(4)
    p10b = _stamped_p10b(items, people)
    p10b.snapshots["height"]["exact"]["sha256"] = sorted([SHA, "c" * 64])

    with pytest.raises(ValueError, match="height/exact snapshot"):
        p10c_links.check_items(items, people, p10b)


def test_rebuilt_items_without_a_stored_snapshot_are_refused() -> None:
    items, people = _items(), _people(4)
    p10b = _stamped_p10b(items, people)
    del p10b.snapshots["person_height"]

    with pytest.raises(ValueError, match="person_height/exact snapshot"):
        p10c_links.check_items(items, people, p10b)


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
    reader: str = READERS[0],
) -> p10c_links.P10bRun:
    n_layers = directions.shape[0]
    return p10c_links.P10bRun(
        path="p10b",
        config_hash="h10b",
        config={},
        reader=reader,
        layer=layer,
        layers=tuple(range(1, n_layers + 1)),
        directions=directions,
        signs=np.ones(n_layers, np.float32) if signs is None else signs,
        items=tuple(items),
        snapshots={},
    )


def _p9b_run(
    directions: NDArray[np.float32],
    signs: NDArray[np.float32] | None = None,
    means: NDArray[np.float32] | None = None,
    types: Sequence[str] = TYPES,
    config: dict[str, object] | None = None,
    contrasts: Sequence[str] | None = None,
) -> p10c_links.P9bRun:
    n_types, n_layers = directions.shape[:2]
    return p10c_links.P9bRun(
        path="p9b",
        config_hash="h9b",
        config=config or {"template": "fewshot", "n_shots": 3, "bos": None},
        types=tuple(types[:n_types]),
        contrasts=tuple(_contrast_of(name) for name in types[:n_types])
        if contrasts is None
        else tuple(contrasts),
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
    p9 = _random_directions((2, 3, 16), 2)

    rows = p10c_links.cosine_rows(
        _range_run(range_dir), _p9b_run(p9), n_random=50, seed=0
    )

    assert len(rows) == 6
    assert tuple(rows[0]) == p10c_links.COSINE_COLUMNS
    assert [(r["p9_type"], r["layer"]) for r in rows] == [
        (t, k) for t in TYPES for k in (1, 2, 3)
    ]
    for row in rows:
        assert -1.0 <= float(row["cosine"]) <= 1.0  # type: ignore[arg-type]


def test_cosine_rows_name_the_contrast_of_each_type() -> None:
    p9 = _p9b_run(
        _random_directions((2, 3, 16), 2),
        types=("obscure_real", "real_vs_synthal"),
    )

    rows = p10c_links.cosine_rows(
        _range_run(_random_directions((3, 16), 1)), p9, n_random=20, seed=0
    )

    assert {(r["p9_type"], r["contrast"]) for r in rows} == {
        ("obscure_real", "knowledge"),
        ("real_vs_synthal", "unknown_type"),
    }


def test_the_same_direction_has_cosine_one_and_the_opposite_minus_one() -> None:
    range_dir = _random_directions((3, 16), 1)
    p9 = _random_directions((2, 3, 16), 2)
    p9[0] = range_dir
    p9[1] = -range_dir

    rows = p10c_links.cosine_rows(
        _range_run(range_dir), _p9b_run(p9), n_random=50, seed=0
    )

    assert _cosines(rows, TYPES[0]) == pytest.approx({1: 1.0, 2: 1.0, 3: 1.0})
    assert _cosines(rows, TYPES[1]) == pytest.approx(
        {1: -1.0, 2: -1.0, 3: -1.0}
    )


def test_a_sign_flip_of_the_p10b_reader_flips_that_layer() -> None:
    range_dir = _random_directions((3, 16), 1)
    p9 = _random_directions((1, 3, 16), 2)
    p9[0] = range_dir
    signs = np.array([1.0, -1.0, 1.0], dtype=np.float32)

    rows = p10c_links.cosine_rows(
        _range_run(range_dir, signs), _p9b_run(p9), n_random=50, seed=0
    )

    assert _cosines(rows, TYPES[0]) == pytest.approx({1: 1.0, 2: -1.0, 3: 1.0})


def test_a_sign_flip_of_the_p9_reader_flips_that_layer_and_type() -> None:
    range_dir = _random_directions((3, 16), 1)
    p9 = np.stack([range_dir, range_dir])
    signs = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, 1.0]], dtype=np.float32)

    rows = p10c_links.cosine_rows(
        _range_run(range_dir),
        _p9b_run(p9, signs),
        n_random=50,
        seed=0,
    )

    assert _cosines(rows, TYPES[0]) == pytest.approx({1: 1.0, 2: 1.0, 3: 1.0})
    assert _cosines(rows, TYPES[1]) == pytest.approx({1: 1.0, 2: -1.0, 3: 1.0})


def test_flipping_both_signs_leaves_the_cosine_unchanged() -> None:
    range_dir = _random_directions((3, 16), 1)
    p9 = np.stack([range_dir])
    flipped = -np.ones((1, 3), dtype=np.float32)

    rows = p10c_links.cosine_rows(
        _range_run(range_dir, -np.ones(3, np.float32)),
        _p9b_run(p9, flipped),
        n_random=50,
        seed=0,
    )

    assert _cosines(rows, TYPES[0]) == pytest.approx({1: 1.0, 2: 1.0, 3: 1.0})


def test_the_cosine_ignores_the_norm_of_the_stored_directions() -> None:
    range_dir = _random_directions((3, 16), 1)
    p9 = _random_directions((1, 3, 16), 2)
    base = p10c_links.cosine_rows(
        _range_run(range_dir), _p9b_run(p9), n_random=50, seed=0
    )

    scaled = p10c_links.cosine_rows(
        _range_run(3.0 * range_dir),
        _p9b_run(5.0 * p9),
        n_random=50,
        seed=0,
    )

    assert _cosines(scaled, TYPES[0]) == pytest.approx(_cosines(base, TYPES[0]))


def test_a_zero_direction_is_refused() -> None:
    range_dir = _random_directions((3, 16), 1)
    p9 = _random_directions((1, 3, 16), 2)
    p9[0, 1] = 0.0

    with pytest.raises(ValueError, match="zero"):
        p10c_links.cosine_rows(
            _range_run(range_dir), _p9b_run(p9), n_random=50, seed=0
        )


def test_the_band_brackets_zero_and_is_shared_by_the_types() -> None:
    range_dir = _random_directions((3, 64), 1)
    p9 = _random_directions((2, 3, 64), 2)

    rows = p10c_links.cosine_rows(
        _range_run(range_dir), _p9b_run(p9), n_random=200, seed=0
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
            _p9b_run(_random_directions((1, 2, d), 2)),
            n_random=400,
            seed=0,
        )
        return float(rows[0]["band_hi"]) - float(rows[0]["band_lo"])  # type: ignore[arg-type]

    assert width(16) > 2.0 * width(256)


def test_the_band_follows_the_seed_and_repeats_for_one_seed() -> None:
    range_dir = _random_directions((2, 32), 1)
    p9 = _random_directions((1, 2, 32), 2)

    def band(seed: int) -> tuple[object, object]:
        row = p10c_links.cosine_rows(
            _range_run(range_dir), _p9b_run(p9), n_random=100, seed=seed
        )[0]
        return row["band_lo"], row["band_hi"]

    assert band(0) == band(0)
    assert band(0) != band(1)


def test_the_band_holds_about_95_percent_of_random_cosines() -> None:
    range_dir = _random_directions((2, 32), 1)
    p9 = _random_directions((1, 2, 32), 2)
    row = p10c_links.cosine_rows(
        _range_run(range_dir), _p9b_run(p9), n_random=400, seed=0
    )[0]
    fresh = _random_directions((5000, 32), 99).astype(np.float64)
    inside = (
        fresh @ range_dir[0].astype(np.float64) >= float(row["band_lo"])
    ) & (  # type: ignore[arg-type]
        fresh @ range_dir[0].astype(np.float64) <= float(row["band_hi"])  # type: ignore[arg-type]
    )

    assert 0.92 < float(inside.mean()) < 0.98


P9_WIDTH = 256
Shift = Callable[[QuantityQuestion], float]


def _planted(
    items: Sequence[QuantityQuestion],
    shift: Shift,
    direction: NDArray[np.float64],
    *,
    seed: int = 0,
) -> NDArray[np.float32]:
    """Noise plus `shift(item)` along `direction`, at every layer."""
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(len(LAYERS), len(items), direction.shape[-1]))
    for n, item in enumerate(items):
        data[:, n] += shift(item) * direction
    return data.astype(np.float32)


def _range_is_negative(item: QuantityQuestion) -> float:
    return -1.5 if item.kind == "range" else 1.5


def _range_is_positive(item: QuantityQuestion) -> float:
    return -_range_is_negative(item)


def _plant_direction() -> NDArray[np.float64]:
    return _random_directions((P9_WIDTH,), 5).astype(np.float64)


def _p9b_pair(
    plant: NDArray[np.float64],
    *,
    signs: NDArray[np.float32] | None = None,
) -> p10c_links.P9bRun:
    """Type 0 reads along `plant`, type 1 along an orthogonal direction."""
    other = _random_directions((P9_WIDTH,), 7).astype(np.float64)
    other -= plant * float(other @ plant)
    other /= np.linalg.norm(other)
    both = np.stack([plant, other]).astype(np.float32)
    return _p9b_run(np.stack([both] * len(LAYERS), axis=1), signs)


def _rows_of(
    rows: list[dict[str, object]], **where: object
) -> list[dict[str, object]]:
    return [r for r in rows if all(r[k] == v for k, v in where.items())]


def _one(rows: list[dict[str, object]], **where: object) -> dict[str, object]:
    (row,) = _rows_of(rows, **where)
    return row


def _knowledge(
    data: NDArray[np.float32],
    items: Sequence[QuantityQuestion],
    p9b: p10c_links.P9bRun,
    **changes: int,
) -> list[dict[str, object]]:
    settings = {"n_boot": 100, "n_random": 100, "seed": 0, **changes}
    return p10c_links.knowledge_rows(data, items, p9b, **settings)


def _auroc(row: dict[str, object]) -> float:
    return float(row["auroc"])  # type: ignore[arg-type]


def test_range_items_on_the_negative_side_fall_below_the_random_band() -> None:
    items = _items()
    plant = _plant_direction()
    data = _planted(items, _range_is_negative, plant)

    rows = _knowledge(data, items, _p9b_pair(plant))

    for layer in LAYERS:
        row = _one(
            rows,
            p9_type=TYPES[0],
            layer=layer,
            items="all",
            test_family="pooled",
        )
        assert _auroc(row) < 0.1
        assert _auroc(row) < float(row["rand_lo"])  # type: ignore[arg-type]


def test_range_items_on_the_positive_side_rise_above_the_random_band() -> None:
    items = _items()
    plant = _plant_direction()
    data = _planted(items, _range_is_positive, plant)

    rows = _knowledge(data, items, _p9b_pair(plant))

    for layer in LAYERS:
        row = _one(
            rows,
            p9_type=TYPES[0],
            layer=layer,
            items="test",
            test_family="pooled",
        )
        assert _auroc(row) > 0.9
        assert _auroc(row) > float(row["rand_hi"])  # type: ignore[arg-type]


def test_an_orthogonal_plant_stays_inside_the_random_band() -> None:
    items = _items()
    pair = _p9b_pair(_plant_direction())
    orthogonal = pair.directions[1, 0].astype(np.float64)
    data = _planted(items, _range_is_negative, orthogonal)

    rows = _knowledge(data, items, pair)

    for layer in LAYERS:
        where = {"layer": layer, "items": "all", "test_family": "pooled"}
        blind = _one(rows, p9_type=TYPES[0], **where)
        assert float(blind["rand_lo"]) <= _auroc(blind)  # type: ignore[arg-type]
        assert _auroc(blind) <= float(blind["rand_hi"])  # type: ignore[arg-type]
        aligned = _one(rows, p9_type=TYPES[1], **where)
        assert _auroc(aligned) < float(aligned["rand_lo"])  # type: ignore[arg-type]


def test_the_stored_sign_of_a_p9_reader_flips_its_auroc() -> None:
    items = _items()
    plant = _plant_direction()
    data = _planted(items, _range_is_negative, plant)
    signs = np.ones((2, len(LAYERS)), dtype=np.float32)
    signs[0, 1] = -1.0

    rows = _knowledge(data, items, _p9b_pair(plant, signs=signs))

    where = {"p9_type": TYPES[0], "items": "all", "test_family": "pooled"}
    kept = _auroc(_one(rows, layer=1, **where))
    flipped = _auroc(_one(rows, layer=2, **where))
    assert kept < 0.1
    assert flipped > 0.9


def test_knowledge_rows_name_the_contrast_of_each_type() -> None:
    items = _items()
    plant = _plant_direction()
    pair = dataclasses.replace(
        _p9b_pair(plant),
        types=("obscure_real", "real_vs_synthal"),
        contrasts=("knowledge", "unknown_type"),
    )

    rows = _knowledge(_planted(items, _range_is_negative, plant), items, pair)

    assert {(r["p9_type"], r["contrast"]) for r in rows} == {
        ("obscure_real", "knowledge"),
        ("real_vs_synthal", "unknown_type"),
    }


def test_knowledge_rows_cover_every_type_layer_set_and_family() -> None:
    items = _items()
    plant = _plant_direction()
    data = _planted(items, _range_is_negative, plant)

    rows = _knowledge(data, items, _p9b_pair(plant))

    assert tuple(rows[0]) == p10c_links.KNOWLEDGE_COLUMNS
    assert len(rows) == 2 * len(LAYERS) * 2 * 3
    assert {r["items"] for r in rows} == {"all", "test"}
    assert {r["test_family"] for r in rows} == {"height", "wingspan", "pooled"}
    counts = {
        (r["items"], r["test_family"]): (r["n_range"], r["n_exact"])
        for r in rows
    }
    assert counts == {
        ("all", "height"): (12, 12),
        ("all", "wingspan"): (12, 12),
        ("all", "pooled"): (24, 24),
        ("test", "height"): (6, 6),
        ("test", "wingspan"): (6, 6),
        ("test", "pooled"): (12, 12),
    }
    for row in rows:
        assert float(row["lo"]) <= _auroc(row) <= float(row["hi"])  # type: ignore[arg-type]
        assert float(row["rand_lo"]) < float(row["rand_hi"])  # type: ignore[arg-type]


def test_the_random_band_is_two_sided_around_one_half_on_noise() -> None:
    items = _items()
    data = _planted(items, lambda item: 0.0, _plant_direction())

    rows = _knowledge(data, items, _p9b_pair(_plant_direction()), n_random=2000)

    for row in rows:
        low, high = float(row["rand_lo"]), float(row["rand_hi"])  # type: ignore[arg-type]
        assert low < 0.5 < high
        assert low + high == pytest.approx(1.0, abs=0.06)


def test_the_random_band_depends_on_the_layer_set_and_family_only() -> None:
    items = _items()
    plant = _plant_direction()
    data = _planted(items, _range_is_negative, plant)

    rows = _knowledge(data, items, _p9b_pair(plant))

    bands = {
        (r["layer"], r["items"], r["test_family"], r["rand_lo"], r["rand_hi"])
        for r in rows
    }
    assert len(bands) == len(LAYERS) * 2 * 3


def test_a_group_without_both_kinds_has_no_row() -> None:
    items = [
        i
        for i in _items()
        if not (i.family == "wingspan" and i.kind == "range")
    ]
    plant = _plant_direction()
    data = _planted(items, _range_is_negative, plant)

    rows = _knowledge(data, items, _p9b_pair(plant))

    assert {r["test_family"] for r in rows} == {"height", "pooled"}
    pooled = _one(
        rows,
        p9_type=TYPES[0],
        layer=1,
        items="all",
        test_family="pooled",
    )
    assert (pooled["n_range"], pooled["n_exact"]) == (12, 24)


def _fewshot_run(n_shots: int, template: str = "fewshot") -> p10c_links.P9bRun:
    return _p9b_run(
        _random_directions((1, 3, 16)),
        config={"template": template, "n_shots": n_shots, "bos": None},
    )


def test_the_fewshot_prompt_has_the_demonstrations_of_the_p9b_run() -> None:
    items = _items()[:3]

    for n_shots in (2, 5):
        texts = p10c_links.render_questions(
            items, _fewshot_run(n_shots), ChatFormat("plain")
        )

        for text, item in zip(texts, items, strict=True):
            first_question, first_answer = DEFAULT_SHOTS[0]
            assert text.startswith(
                f"Q: {first_question}\nA: {first_answer}\n\n"
            )
            assert text.count("Q: ") == n_shots + 1
            assert text.endswith(f"Q: {item.question}\nA:")
            assert text == few_shot_prompt(
                item.question, DEFAULT_SHOTS[:n_shots]
            )


def test_the_plain_and_concept_templates_follow_the_p9b_run() -> None:
    items = _items()[:2]
    chat = ChatFormat("plain")

    plain = p10c_links.render_questions(items, _fewshot_run(3, "plain"), chat)
    concept = p10c_links.render_questions(
        items, _fewshot_run(3, "concept"), chat
    )

    assert plain == [i.question for i in items]
    for text, item in zip(concept, items, strict=True):
        assert "knowledge" in text
        assert item.question in text


def test_an_unknown_p9b_template_is_refused() -> None:
    with pytest.raises(ValueError, match="no such template"):
        p10c_links.render_questions(
            _items()[:1],
            _fewshot_run(3, "no such template"),
            ChatFormat("plain"),
        )


@pytest.mark.parametrize("bos", [None, "none", "prepend"])
def test_activations_are_read_at_the_last_token_with_the_p9b_bos(
    bos: str | None,
    tiny_gpt2: LoadedModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def spy(loaded: LoadedModel, texts: Sequence[str], **kwargs: Any) -> Any:
        seen.update(texts=list(texts), **kwargs)
        return SimpleNamespace(
            layers=tuple(kwargs["layers"]),
            data=np.zeros((len(kwargs["layers"]), len(texts), 32), np.float32),
        )

    monkeypatch.setattr(p10c_links, "collect_activations", spy)
    items = _items()[:3]
    p9b = _p9b_run(
        _random_directions((1, 3, 32)),
        config={"template": "fewshot", "n_shots": 2, "bos": bos},
    )

    data = p10c_links.collect_question_activations(
        tiny_gpt2, items, p9b, layers=LAYERS, batch_size=4
    )

    assert data.shape == (3, 3, 32)
    assert seen["position"] == "last"
    assert seen["layers"] == LAYERS
    assert seen["bos"] == bos
    assert seen["batch_size"] == 4
    assert seen["texts"] == p10c_links.render_questions(
        items, p9b, chat_format_for(tiny_gpt2)
    )


def _outcome(greedy: str, *samples: str) -> Outcome:
    return Outcome(
        greedy=_real_answer(greedy),
        samples=tuple(_real_answer(form) for form in samples),
    )


def _serialized(outcome: Outcome) -> dict[str, object]:
    return {
        "greedy": dataclasses.asdict(outcome.greedy),
        "samples": [dataclasses.asdict(a) for a in outcome.samples],
    }


@pytest.mark.parametrize("decoding", ["greedy", "sampled"])
def test_the_hedge_share_matches_the_item_rates_of_p10a(decoding: str) -> None:
    first = [
        _outcome("range", "range", "range", "range", "point"),
        _outcome("point", "range", "abstain"),
    ]
    second = [
        _outcome("point", "point", "point"),
        _outcome("range", "range", "range", "range", "point", "point"),
    ]
    for member in (0, 1):
        answers = {
            "free/fewshot/range_last": _serialized(first[member]),
            "free/fewshot/exact_last": _serialized(second[member]),
            "forced/fewshot/range_last": _serialized(
                _outcome("range", "range")
            ),
        }

        share = p10c_links.hedge_share(answers, "free", "fewshot", decoding)

        expected = _item_rates([first, second], [member], decoding)[0]
        assert share == pytest.approx(expected)


def _sample_cell(ranged: int, total: int = 5) -> dict[str, object]:
    forms = ["range"] * ranged + ["point"] * (total - ranged)
    return _serialized(_outcome("point", *forms))


def _two_cells(first: int, second: int) -> dict[str, dict[str, object]]:
    return {
        "free/fewshot/range_last": _sample_cell(first),
        "free/fewshot/exact_last": _sample_cell(second),
    }


def test_equal_shares_from_different_cell_splits_are_equal() -> None:
    narrow = _two_cells(1, 2)
    wide = _two_cells(0, 3)

    assert p10c_links.hedge_share(
        narrow, "free", "fewshot", "sampled"
    ) == p10c_links.hedge_share(wide, "free", "fewshot", "sampled")


def test_an_item_without_the_cell_is_refused() -> None:
    answers = {
        "free/fewshot/range_last": _serialized(_outcome("point", "point"))
    }

    with pytest.raises(ValueError, match="forced/chat"):
        p10c_links.hedge_share(answers, "forced", "chat", "sampled")


N_SAMPLES = 24
ZERO_PATTERN = (1, 4, 2, 5, 0, 3)


def _share_cell(share: float) -> dict[str, object]:
    ranged = round(share * N_SAMPLES)
    return {
        "greedy": _answer("range" if share >= 0.5 else "point"),
        "samples": [_answer("range")] * ranged
        + [_answer("point")] * (N_SAMPLES - ranged),
    }


def _position(item: QuantityQuestion) -> int:
    """Index 0 to 11 of a test item within its kind, over both families."""
    return int(item.uid[-1]) + (6 if item.family == "wingspan" else 0)


Link = Callable[[QuantityQuestion], tuple[float, float]]


def _views(
    items: Sequence[QuantityQuestion],
    link: Link,
    *,
    cells: Sequence[str] = CELLS,
    people: Sequence[QuantityQuestion] = (),
    outside: Link | None = None,
    reader: str = READERS[0],
    per_cell: Callable[[float, str], float] = lambda share, cell: share,
    directions: NDArray[np.float32] | None = None,
) -> tuple[p10c_links.P10bRun, p10c_links.P10aRun]:
    """Builds in-memory runs where link gives (score, hedge share).

    The other reader of the P10b run scores every item the other way round.
    """
    records = []
    answers = {}
    for item in [*items, *people]:
        score, share = (
            link(item)
            if item.split == "test" or outside is None
            else outside(item)
        )
        records.append(
            {
                "uid": item.uid,
                "family": item.family,
                "kind": item.kind,
                "split": item.split,
                "sitelinks": 1,
                "scores": {READERS[0]: score, READERS[1]: -score},
            }
        )
        answers[item.uid] = {
            **dataclasses.asdict(item),
            "answers": {c: _share_cell(per_cell(share, c)) for c in cells},
        }
    p10b = _range_run(
        np.zeros((3, 4), np.float32) if directions is None else directions,
        items=records,
        reader=reader,
    )
    p10a = p10c_links.P10aRun(
        path="p10a", config_hash="h", config={}, items=answers, snapshots={}
    )
    return p10b, p10a


def _behavior(
    items: Sequence[QuantityQuestion], link: Link, **changes: Any
) -> list[dict[str, object]]:
    settings = {"n_boot": 200, "seed": 0}
    views = {
        k: changes.pop(k)
        for k in ("cells", "people", "outside", "reader", "per_cell")
        if k in changes
    }
    p10b, p10a = _views(items, link, **views)
    return p10c_links.behavior_rows(p10b, p10a, **{**settings, **changes})


def _monotone(item: QuantityQuestion) -> tuple[float, float]:
    index = _position(item)
    return float(index), index / 12.0


def _sampled(
    rows: list[dict[str, object]], **where: object
) -> dict[str, object]:
    return _one(rows, decoding="sampled", **where)


def _rho(row: dict[str, object]) -> float:
    return float(row["spearman"])  # type: ignore[arg-type]


def test_a_monotone_link_within_each_kind_gives_a_spearman_near_one() -> None:
    rows = _behavior(_items(), _monotone)

    for kind in ("range", "exact"):
        for family in ("height", "wingspan", "pooled"):
            row = _sampled(
                rows,
                condition="free",
                format="fewshot",
                kind=kind,
                test_family=family,
            )
            assert _rho(row) == pytest.approx(1.0)
            assert row["constant"] is False
            assert float(row["lo"]) > 0.5  # type: ignore[arg-type]
            assert float(row["hi"]) <= 1.0  # type: ignore[arg-type]


def test_a_link_only_across_kinds_vanishes_within_a_kind() -> None:
    def across(item: QuantityQuestion) -> tuple[float, float]:
        pattern = ZERO_PATTERN[int(item.uid[-1])]
        if item.kind == "range":
            return 10.0 + _position(item), 0.5 + pattern / 12.0
        return float(_position(item)), pattern / 12.0

    rows = _behavior(_items(), across)

    for kind in ("range", "exact"):
        for family in ("height", "wingspan", "pooled"):
            row = _sampled(
                rows,
                condition="free",
                format="fewshot",
                kind=kind,
                test_family=family,
            )
            assert abs(_rho(row)) < 0.1
            assert float(row["lo"]) < 0.0 < float(row["hi"])  # type: ignore[arg-type]
    pooled = _sampled(
        rows,
        condition="free",
        format="fewshot",
        kind="range",
        test_family="pooled",
    )
    assert pooled["n"] == 12
    assert float(pooled["mean_hedge"]) > 0.5  # type: ignore[arg-type]
    exact = _sampled(
        rows,
        condition="free",
        format="fewshot",
        kind="exact",
        test_family="pooled",
    )
    assert float(exact["mean_hedge"]) < 0.5  # type: ignore[arg-type]


def test_only_the_test_items_of_the_p10b_run_enter() -> None:
    def backwards(item: QuantityQuestion) -> tuple[float, float]:
        score, share = _monotone(item)
        return score, 1.0 - share

    rows = _behavior(_items(), _monotone, outside=backwards)

    row = _sampled(
        rows,
        condition="free",
        format="fewshot",
        kind="range",
        test_family="pooled",
    )
    assert row["n"] == 12
    assert _rho(row) == pytest.approx(1.0)
    height = _sampled(
        rows,
        condition="free",
        format="fewshot",
        kind="exact",
        test_family="height",
    )
    assert height["n"] == 6


def test_a_constant_hedge_share_gives_nan_and_the_constant_flag() -> None:
    rows = _behavior(_items(), lambda item: (float(_position(item)), 0.0))

    for row in rows:
        assert row["constant"] is True
        assert np.isnan(_rho(row))
        assert np.isnan(float(row["lo"]))  # type: ignore[arg-type]
        assert np.isnan(float(row["hi"]))  # type: ignore[arg-type]
        assert float(row["mean_hedge"]) == 0.0  # type: ignore[arg-type]


def test_a_constant_score_is_flagged_as_constant_too() -> None:
    rows = _behavior(_items(), lambda item: (1.0, _monotone(item)[1]))

    assert all(r["constant"] is True for r in rows)
    assert all(np.isnan(_rho(r)) for r in rows)


def test_the_rows_cover_the_conditions_formats_decodings_and_groups() -> None:
    rows = _behavior(_items(), _monotone)

    assert tuple(rows[0]) == p10c_links.BEHAVIOR_COLUMNS
    assert {(r["condition"], r["format"]) for r in rows} == {
        ("free", "fewshot"),
        ("forced", "fewshot"),
    }
    assert {r["decoding"] for r in rows} == {"greedy", "sampled"}
    assert len(rows) == 2 * 2 * 2 * 3
    assert {(r["kind"], r["test_family"]) for r in rows} == {
        (k, f)
        for k in ("range", "exact")
        for f in ("height", "wingspan", "pooled")
    }
    assert {r["n"] for r in rows if r["test_family"] == "pooled"} == {12}


def test_the_chat_format_gets_rows_when_the_run_has_it() -> None:
    cells = ("free/chat/none", "free/fewshot/range_last")

    rows = _behavior(_items(), _monotone, cells=cells)

    assert {(r["condition"], r["format"]) for r in rows} == {
        ("free", "chat"),
        ("free", "fewshot"),
    }


TIE_SPLITS = {0.30: (1 / 24, 4 / 24), 0.31: (0.0, 5 / 24)}


def test_items_with_equal_shares_from_different_splits_tie_in_the_rank() -> (
    None
):
    def split_cells(share: float, cell: str) -> float:
        if share not in TIE_SPLITS:
            return share
        return TIE_SPLITS[share][0 if "range_last" in cell else 1]

    def link(item: QuantityQuestion) -> tuple[float, float]:
        position = _position(item) % 6
        shares = (0.05, 0.1, 0.30, 0.31, 0.5, 0.6)
        return float(position), shares[position]

    rows = _behavior(_items(), link, per_cell=split_cells)

    row = _sampled(
        rows,
        condition="free",
        format="fewshot",
        kind="range",
        test_family="height",
    )
    tied = 5 / 48
    exact = spearman(
        np.arange(6.0),
        np.array([1 / 24, 2 / 24, tied, tied, 12 / 24, 14 / 24]),
    )
    assert _rho(row) == pytest.approx(exact)
    assert _rho(row) < 1.0


def test_the_shot_orders_are_averaged_into_one_share() -> None:
    def opposite_orders(share: float, cell: str) -> float:
        return share if "range_last" in cell else 11.0 / 12.0 - share

    where = {
        "condition": "free",
        "format": "fewshot",
        "kind": "range",
        "test_family": "pooled",
    }

    first_only = _behavior(
        _items(), _monotone, cells=CELLS[:1], per_cell=opposite_orders
    )
    averaged = _behavior(
        _items(), _monotone, cells=CELLS[:2], per_cell=opposite_orders
    )

    assert _rho(_sampled(first_only, **where)) == pytest.approx(1.0)
    row = _sampled(averaged, **where)
    assert row["constant"] is True
    assert float(row["mean_hedge"]) == pytest.approx(11.0 / 24.0)  # type: ignore[arg-type]


def test_the_score_is_the_one_of_the_chosen_reader() -> None:
    where = {
        "condition": "free",
        "format": "fewshot",
        "kind": "range",
        "test_family": "pooled",
    }

    first = _behavior(_items(), _monotone, reader=READERS[0])
    second = _behavior(_items(), _monotone, reader=READERS[1])

    assert _rho(_sampled(first, **where)) == pytest.approx(1.0)
    assert _rho(_sampled(second, **where)) == pytest.approx(-1.0)


def test_the_people_get_their_own_row_when_both_runs_have_them() -> None:
    people = _people(6)

    rows = _behavior(_items(), _monotone, people=people)

    person = _sampled(
        rows, condition="free", format="fewshot", test_family="person_height"
    )
    assert person["kind"] == "exact"
    assert person["n"] == 6
    pooled = _sampled(
        rows,
        condition="free",
        format="fewshot",
        kind="exact",
        test_family="pooled",
    )
    assert pooled["n"] == 12


def test_a_family_with_fewer_than_three_items_has_no_row() -> None:
    items = [
        i
        for i in _items()
        if not (
            i.family == "wingspan" and i.split == "test" and i.uid[-1] > "1"
        )
    ]

    rows = _behavior(items, _monotone)

    assert {r["test_family"] for r in rows} == {"height", "pooled"}


def test_the_behavior_rows_repeat_for_one_seed() -> None:
    first = _behavior(_items(), _monotone, seed=3)
    again = _behavior(_items(), _monotone, seed=3)

    np.testing.assert_equal(first, again)


TINY_WIDTH = 32


def _tiny_views(
    *,
    knowledge: bool = True,
    behavior: bool = True,
    link: Link = _monotone,
    n_types: int = 2,
    types: Sequence[str] | None = None,
) -> tuple[
    Config,
    p10c_links.P10bRun,
    p10c_links.P9bRun | None,
    p10c_links.P10aRun | None,
]:
    """In-memory runs for a model of width 32 and three layers."""
    people = _people(4)
    names = list(types or [f"type_{n}" for n in range(n_types)])
    p10b, p10a = _views(
        _items(),
        link,
        people=people,
        directions=_random_directions((3, TINY_WIDTH), 1),
    )
    p9b = _p9b_run(
        _random_directions((len(names), 3, TINY_WIDTH), 2),
        types=names,
        config={"template": "fewshot", "n_shots": 2, "bos": None},
    )
    p10b = dataclasses.replace(p10b, path="p10b", config_hash="h10b")
    config = Config(
        model="tiny",
        p10b_run="p10b",
        knowledge_run="p9b" if knowledge else None,
        behavior_run="p10a" if behavior else None,
        n_boot=30,
        n_random=20,
        batch_size=4,
    )
    return (
        config,
        p10b,
        p9b if knowledge else None,
        p10a if behavior else None,
    )


def _executed(
    loaded: LoadedModel | None,
    tmp_path: Path,
    *,
    knowledge: bool = True,
    behavior: bool = True,
    **changes: Any,
) -> RunDir:
    config, p10b, p9b, p10a = _tiny_views(
        knowledge=knowledge, behavior=behavior, **changes
    )
    run_dir = RunDir.create(config, out_root=tmp_path)
    p10c_links.execute(
        config,
        run_dir,
        loaded if knowledge else None,
        _items() if knowledge else None,
        _people(4) if knowledge else None,
        p10b,
        p9b,
        p10a,
    )
    return run_dir


def _csv(run_dir: RunDir, name: str) -> list[dict[str, str]]:
    with (run_dir.path / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize(
    ("knowledge", "behavior"), [(True, False), (False, True), (True, True)]
)
def test_execute_writes_the_files_of_each_link(
    knowledge: bool, behavior: bool, tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _executed(
        tiny_gpt2, tmp_path, knowledge=knowledge, behavior=behavior
    )

    written = {p.name for p in run_dir.path.glob("*.*")}
    assert ("cosine.csv" in written) is knowledge
    assert ("knowledge.csv" in written) is knowledge
    assert ("behavior.csv" in written) is behavior
    assert "summary.json" in written


def test_the_csv_files_have_the_documented_columns_and_rows(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _executed(tiny_gpt2, tmp_path)

    cosine = _csv(run_dir, "cosine.csv")
    knowledge = _csv(run_dir, "knowledge.csv")
    behavior = _csv(run_dir, "behavior.csv")

    assert tuple(cosine[0]) == p10c_links.COSINE_COLUMNS
    assert tuple(knowledge[0]) == p10c_links.KNOWLEDGE_COLUMNS
    assert tuple(behavior[0]) == p10c_links.BEHAVIOR_COLUMNS
    assert len(cosine) == 2 * 3
    assert len(knowledge) == 2 * 3 * 2 * 3
    assert len(behavior) == 2 * 2 * (2 * 3 + 1)
    assert {r["test_family"] for r in behavior} == {
        "height",
        "wingspan",
        "pooled",
        "person_height",
    }


def test_the_files_name_the_contrast_of_each_type(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _executed(
        tiny_gpt2,
        tmp_path,
        types=("obscure_real", "real_vs_synthal"),
    )

    expected = {
        "obscure_real": "knowledge",
        "real_vs_synthal": "unknown_type",
    }
    for name in ("cosine.csv", "knowledge.csv"):
        rows = _csv(run_dir, name)
        assert {(r["p9_type"], r["contrast"]) for r in rows} == set(
            expected.items()
        )
    summary = run_dir.read_json("summary")
    for section in ("cosine", "knowledge"):
        assert {
            name: entry["contrast"]  # type: ignore[index]
            for name, entry in summary[section].items()  # type: ignore[index]
        } == expected


def test_the_knowledge_link_costs_one_forward_pass_over_the_questions(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    real = p10c_links.collect_activations

    def spy(loaded: LoadedModel, texts: Sequence[str], **kwargs: Any) -> Any:
        calls.append({"n": len(texts), **kwargs})
        return real(loaded, texts, **kwargs)

    monkeypatch.setattr(p10c_links, "collect_activations", spy)

    _executed(tiny_gpt2, tmp_path)

    assert len(calls) == 1
    assert calls[0]["n"] == len(_items())
    assert tuple(calls[0]["layers"]) == LAYERS
    assert calls[0]["position"] == "last"
    assert calls[0]["batch_size"] == 4


def test_the_behavior_link_alone_needs_no_model(tmp_path: Path) -> None:
    run_dir = _executed(None, tmp_path, knowledge=False)

    assert (run_dir.path / "behavior.csv").is_file()


def test_a_knowledge_link_without_the_model_or_the_items_is_refused(
    tmp_path: Path,
) -> None:
    config, p10b, p9b, _ = _tiny_views(behavior=False)
    run_dir = RunDir.create(config, out_root=tmp_path)

    with pytest.raises(ValueError, match="model"):
        p10c_links.execute(config, run_dir, None, None, None, p10b, p9b, None)


def test_a_constant_hedge_share_is_written_as_nan_with_the_flag(
    tmp_path: Path,
) -> None:
    run_dir = _executed(
        None,
        tmp_path,
        knowledge=False,
        link=lambda item: (float(_position(item)), 0.0),
    )

    rows = _csv(run_dir, "behavior.csv")

    assert {r["constant"] for r in rows} == {"True"}
    assert {r["spearman"] for r in rows} == {"nan"}
    assert {r["lo"] for r in rows} == {"nan"}


def test_the_summary_reports_the_runs_the_layer_and_the_links(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _executed(tiny_gpt2, tmp_path)

    summary = run_dir.read_json("summary")
    cosine = _csv(run_dir, "cosine.csv")
    knowledge = _csv(run_dir, "knowledge.csv")

    assert summary["model"] == "tiny"  # type: ignore[index]
    assert summary["runs"] == {  # type: ignore[index]
        "p10b": {"path": "p10b", "config_hash": "h10b"},
        "p9b": {"path": "p9b", "config_hash": "h9b"},
        "p10a": {"path": "p10a", "config_hash": "h"},
    }
    assert summary["reader"] == "plain/logistic"  # type: ignore[index]
    assert summary["layer"] == 2  # type: ignore[index]
    assert summary["p9_types"] == ["type_0", "type_1"]  # type: ignore[index]
    for name in ("type_0", "type_1"):
        row = next(
            r for r in cosine if r["p9_type"] == name and r["layer"] == "2"
        )
        assert summary["cosine"][name] == {  # type: ignore[index]
            "contrast": "knowledge",
            "cosine": pytest.approx(float(row["cosine"])),
            "band_lo": pytest.approx(float(row["band_lo"])),
            "band_hi": pytest.approx(float(row["band_hi"])),
        }
        pooled = next(
            r
            for r in knowledge
            if (r["p9_type"], r["layer"], r["items"], r["test_family"])
            == (name, "2", "all", "pooled")
        )
        assert set(summary["knowledge"][name]) == {  # type: ignore[index]
            "contrast",
            "n_range",
            "n_exact",
            "auroc",
            "lo",
            "hi",
            "rand_lo",
            "rand_hi",
        }
        assert summary["knowledge"][name]["auroc"] == pytest.approx(  # type: ignore[index]
            float(pooled["auroc"])
        )
    behavior = summary["behavior"]  # type: ignore[index]
    assert behavior
    assert {r["test_family"] for r in behavior} == {"pooled"}
    assert summary["config"]["reader"] == "plain/logistic"  # type: ignore[index]
    assert summary["counts"]["knowledge"]["height"] == {  # type: ignore[index]
        "range": 12,
        "exact": 12,
    }
    assert summary["counts"]["people"] == 4  # type: ignore[index]
    assert summary["counts"]["behavior"]["height"] == {  # type: ignore[index]
        "range": 6,
        "exact": 6,
    }
    assert summary["counts"]["behavior"]["person_height"] == {  # type: ignore[index]
        "exact": 4
    }


def test_the_summary_of_a_single_link_leaves_the_other_empty(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    only_behavior = _executed(None, tmp_path / "b", knowledge=False)
    only_knowledge = _executed(tiny_gpt2, tmp_path / "k", behavior=False)

    first = only_behavior.read_json("summary")
    second = only_knowledge.read_json("summary")

    assert first["runs"]["p9b"] is None  # type: ignore[index]
    assert first["cosine"] == {}  # type: ignore[index]
    assert first["knowledge"] == {}  # type: ignore[index]
    assert first["behavior"]  # type: ignore[index]
    assert second["runs"]["p10a"] is None  # type: ignore[index]
    assert second["behavior"] == []  # type: ignore[index]
    assert second["cosine"]  # type: ignore[index]


def test_the_experiment_is_registered() -> None:
    experiment = load_experiment("p10c")

    assert experiment.config_cls is Config
    assert experiment.run is p10c_links.run
    assert experiment.plot is p10c_links.plot


def _plotted(run_dir: RunDir, tmp_path: Path) -> list[Path]:
    paths = p10c_links.plot(run_dir, tmp_path / "figures")
    assert all(p.is_file() for p in paths)
    return paths


def _stems(paths: Sequence[Path]) -> set[str]:
    return {p.stem for p in paths}


def test_plot_draws_every_figure_whose_csv_exists(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    both = _executed(tiny_gpt2, tmp_path / "both")
    only_k = _executed(tiny_gpt2, tmp_path / "k", behavior=False)
    only_b = _executed(None, tmp_path / "b", knowledge=False)

    paths = _plotted(both, tmp_path / "both")

    assert _stems(paths) == {"cosine", "knowledge_auroc", "behavior_spearman"}
    assert sorted(p.suffix for p in paths) == sorted(
        [".pdf", ".png", ".svg"] * 3
    )
    assert _stems(_plotted(only_k, tmp_path / "k")) == {
        "cosine",
        "knowledge_auroc",
    }
    assert _stems(_plotted(only_b, tmp_path / "b")) == {"behavior_spearman"}


def test_more_than_eight_types_are_split_over_several_figures(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _executed(tiny_gpt2, tmp_path, behavior=False, n_types=10)

    paths = _plotted(run_dir, tmp_path)

    assert _stems(paths) == {
        "cosine",
        "cosine_2",
        "knowledge_auroc",
        "knowledge_auroc_2",
    }


def test_plot_survives_a_behavior_run_where_every_row_is_constant(
    tmp_path: Path,
) -> None:
    run_dir = _executed(
        None,
        tmp_path,
        knowledge=False,
        link=lambda item: (float(_position(item)), 0.0),
    )

    assert _stems(_plotted(run_dir, tmp_path)) == {"behavior_spearman"}


class _ForbiddenCallError(AssertionError):
    """Raised by a loader that a test says must not run."""


def _forbid(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise _ForbiddenCallError("must not be called")

    for name in names:
        monkeypatch.setattr(p10c_links, name, refuse)


def _runs_on_disk(
    root: Path,
    *,
    p10b_people: bool = True,
    p10a_people: bool = True,
    p10a_items: Sequence[QuantityQuestion] | None = None,
    p10b: dict[str, Any] | None = None,
    p9b: dict[str, Any] | None = None,
    p10a: dict[str, Any] | None = None,
) -> dict[str, RunDir]:
    """Writes a P10b, a P9b and a P10a run for the tiny model."""
    items = _items()
    people = _people(4)
    p10b_args = {"model": "tiny", **(p10b or {})}
    config = P10bConfig(
        model=p10b_args["model"],
        families=USED,
        n_per_side=7,
        min_sitelinks=9,
        seed=4,
        people=p10b_people,
    )
    runs = {
        "p10b": _write_p10b(
            root / "p10b",
            [*items, *(people if p10b_people else [])],
            directions=_random_directions((len(READERS), 3, TINY_WIDTH), 1),
            config=config,
            **p10b_args,
        ),
        "p9b": _write_p9b(
            root / "p9b",
            **{
                "width": TINY_WIDTH,
                "config": P9bConfig(
                    model=(p9b or {}).get("model", "tiny"),
                    labels_run="x",
                    template="fewshot",
                    n_shots=2,
                ),
                **(p9b or {}),
            },
        ),
        "p10a": _write_p10a(
            root / "p10a",
            [
                *(items if p10a_items is None else p10a_items),
                *(people if p10a_people else []),
            ],
            **(p10a or {}),
        ),
    }
    return runs


def _config_for(runs: dict[str, RunDir], **changes: Any) -> Config:
    settings: dict[str, Any] = {
        "model": "tiny",
        "p10b_run": str(runs["p10b"].path),
        "knowledge_run": str(runs["p9b"].path),
        "behavior_run": str(runs["p10a"].path),
        "n_boot": 20,
        "n_random": 20,
        "batch_size": 4,
    }
    return Config(**{**settings, **changes})


def _p10c_dir(config: Config, root: Path) -> RunDir:
    return RunDir.create(config, out_root=root / "out")


STALE = [dataclasses.replace(i, sha256="b" * 64) for i in _items()]
BAD_INPUTS: dict[str, tuple[dict[str, Any], str]] = {
    "unfinished p10b": ({"p10b": {"finished": False}}, "not a finished run"),
    "p10b of another model": ({"p10b": {"model": "other"}}, "'other'"),
    "unfinished p9b": ({"p9b": {"finished": False}}, "not a finished run"),
    "p9b of another model": ({"p9b": {"model": "other"}}, "'other'"),
    "p9b other layers": ({"p9b": {"layers": (2, 3, 4)}}, "layers"),
    "p9b other hidden size": (
        {"p9b": {"width": TINY_WIDTH * 2}},
        "hidden size",
    ),
    "unfinished p10a": ({"p10a": {"finished": False}}, "not a finished run"),
    "p10a of another model": ({"p10a": {"model": "other"}}, "'other'"),
    "p10a missing item": (
        {"p10a_items": [i for i in _items() if i.uid != "height-range-test3"]},
        "height-range-test3",
    ),
    "p10a other snapshot": ({"p10a": {"stamps": _stamps(STALE)}}, "sha256"),
}


@pytest.mark.parametrize("case", [*BAD_INPUTS, "reader missing"])
def test_run_refuses_bad_inputs_before_any_model_or_item_load(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid(monkeypatch, "load_model", "load_quantities", "load_people_heights")
    changes, message = BAD_INPUTS.get(case, ({}, "plain/random"))
    runs = _runs_on_disk(tmp_path, **changes)
    config = _config_for(
        runs, **({"reader": "plain/random"} if case == "reader missing" else {})
    )

    with pytest.raises(ValueError, match=message):
        p10c_links.run(config, _p10c_dir(config, tmp_path))


def test_run_refuses_items_that_differ_from_the_p10b_items_before_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid(monkeypatch, "load_model")
    runs = _runs_on_disk(tmp_path)
    fewer = [i for i in _items() if i.uid != "wingspan-exact-train1"]
    monkeypatch.setattr(p10c_links, "load_quantities", lambda **kw: fewer)
    monkeypatch.setattr(
        p10c_links, "load_people_heights", lambda **kw: _people(4)
    )
    config = _config_for(runs)

    with pytest.raises(ValueError, match="wingspan-exact-train1"):
        p10c_links.run(config, _p10c_dir(config, tmp_path))


def test_run_refuses_items_from_another_snapshot_before_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid(monkeypatch, "load_model")
    runs = _runs_on_disk(tmp_path)
    monkeypatch.setattr(p10c_links, "load_quantities", lambda **kw: STALE)
    monkeypatch.setattr(
        p10c_links, "load_people_heights", lambda **kw: _people(4)
    )
    config = _config_for(runs)

    with pytest.raises(ValueError, match=r"sha256.*rebuilt"):
        p10c_links.run(config, _p10c_dir(config, tmp_path))


def test_run_refuses_items_in_another_split_than_the_p10b_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid(monkeypatch, "load_model")
    runs = _runs_on_disk(tmp_path)
    moved = [
        dataclasses.replace(i, split="train")
        if i.uid == "height-range-test0"
        else i
        for i in _items()
    ]
    monkeypatch.setattr(p10c_links, "load_quantities", lambda **kw: moved)
    monkeypatch.setattr(
        p10c_links, "load_people_heights", lambda **kw: _people(4)
    )
    config = _config_for(runs)

    with pytest.raises(ValueError, match="height-range-test0"):
        p10c_links.run(config, _p10c_dir(config, tmp_path))


def test_run_rebuilds_the_items_from_the_p10b_config_and_writes_everything(
    tmp_path: Path, tiny_gpt2: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, dict[str, Any]] = {}

    def quantities(**kwargs: Any) -> list[QuantityQuestion]:
        seen["quantities"] = kwargs
        return _items()

    def people(**kwargs: Any) -> list[QuantityQuestion]:
        seen["people"] = kwargs
        return _people(4)

    monkeypatch.setattr(p10c_links, "load_quantities", quantities)
    monkeypatch.setattr(p10c_links, "load_people_heights", people)
    monkeypatch.setattr(p10c_links, "load_model", lambda key: tiny_gpt2)
    runs = _runs_on_disk(tmp_path)
    config = _config_for(runs, offline=False)
    run_dir = _p10c_dir(config, tmp_path)

    p10c_links.run(config, run_dir)

    loaded = seen["quantities"]
    assert [f.name for f in loaded["families"]] == ["height", "wingspan"]
    assert loaded["n_per_side"] == 7
    assert loaded["min_sitelinks"] == 9
    assert loaded["seed"] == 4
    assert loaded["offline"] is False
    assert seen["people"] == {"offline": False}
    for name in ("cosine.csv", "knowledge.csv", "behavior.csv", "summary.json"):
        assert (run_dir.path / name).is_file()
    summary = run_dir.read_json("summary")
    assert (
        summary["runs"]["p10b"]["config_hash"]
        == (  # type: ignore[index]
            runs["p10b"].meta["config_hash"]
        )
    )


def test_run_skips_the_people_the_p10b_run_did_not_use(
    tmp_path: Path, tiny_gpt2: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid(monkeypatch, "load_people_heights")
    monkeypatch.setattr(p10c_links, "load_quantities", lambda **kw: _items())
    monkeypatch.setattr(p10c_links, "load_model", lambda key: tiny_gpt2)
    runs = _runs_on_disk(tmp_path, p10b_people=False, p10a_people=False)
    config = _config_for(runs)
    run_dir = _p10c_dir(config, tmp_path)

    p10c_links.run(config, run_dir)

    assert (run_dir.path / "knowledge.csv").is_file()


def test_run_with_only_the_behavior_link_loads_no_model_and_no_items(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid(monkeypatch, "load_model", "load_quantities", "load_people_heights")
    runs = _runs_on_disk(tmp_path)
    config = _config_for(runs, knowledge_run=None)
    run_dir = _p10c_dir(config, tmp_path)

    p10c_links.run(config, run_dir)

    assert (run_dir.path / "behavior.csv").is_file()
    assert not (run_dir.path / "knowledge.csv").exists()
    assert not (run_dir.path / "cosine.csv").exists()


def test_a_p10a_run_without_the_people_still_links_the_other_items(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid(monkeypatch, "load_model", "load_quantities", "load_people_heights")
    runs = _runs_on_disk(tmp_path, p10a_people=False)
    config = _config_for(runs, knowledge_run=None)
    run_dir = _p10c_dir(config, tmp_path)

    p10c_links.run(config, run_dir)

    rows = _csv(run_dir, "behavior.csv")
    assert "person_height" not in {r["test_family"] for r in rows}
