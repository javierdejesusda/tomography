"""Tests for the blinded judge batches."""

import dataclasses
import json
import math
from pathlib import Path

import pytest

from tomography.judging.batches import Export
from tomography.judging.batches import Generation
from tomography.judging.batches import plan_export
from tomography.judging.batches import write_export

CRITERIA = ("concept", "fluency")
RUBRIC = "Rate every item.\n"
RUBRIC_SHA = "ab" * 32


def _keys(n: int) -> list[str]:
    return [f"sentiment/tune/caa/0.2/{i}/0" for i in range(n)]


def _generations(keys: list[str]) -> dict[str, Generation]:
    return {
        key: Generation(
            key=key,
            prompt=f"prompt {i}",
            concept="positive sentiment",
            text=f"generation {i} café",
        )
        for i, key in enumerate(keys)
    }


def _plan(
    keys: list[str],
    *,
    seed: int = 7,
    batch_size: int = 4,
    second_fraction: float = 0.25,
) -> Export:
    return plan_export(
        keys,
        stage="tune",
        seed=seed,
        criteria=CRITERIA,
        rubric_sha256=RUBRIC_SHA,
        batch_size=batch_size,
        second_fraction=second_fraction,
    )


def _judge(export: Export, judge: int) -> list[tuple[str, str]]:
    return [
        pair
        for batch in export.batches
        if batch.judge == judge
        for pair in zip(batch.item_ids, batch.keys, strict=True)
    ]


def test_the_first_judge_gets_every_generation_exactly_once() -> None:
    keys = _keys(10)

    export = _plan(keys)

    primary = _judge(export, 1)
    assert sorted(key for _, key in primary) == sorted(keys)
    sizes = [len(b.keys) for b in export.batches if b.judge == 1]
    assert sizes == [4, 4, 2]
    assert [key for _, key in primary] != sorted(keys)


def test_the_second_judge_gets_a_subset_under_other_item_ids() -> None:
    keys = _keys(10)

    export = _plan(keys, second_fraction=0.25)

    primary = dict((key, item) for item, key in _judge(export, 1))
    second = _judge(export, 2)
    assert len(second) == math.ceil(0.25 * 10)
    assert len({key for _, key in second}) == len(second)
    assert {key for _, key in second} <= set(keys)
    assert all(primary[key] != item for item, key in second)
    every_id = [i for b in export.batches for i in b.item_ids]
    assert len(set(every_id)) == len(every_id) == 13


def test_a_zero_fraction_exports_nothing_for_a_second_judge() -> None:
    export = _plan(_keys(6), second_fraction=0.0)

    assert {batch.judge for batch in export.batches} == {1}


def test_ids_carry_no_trace_of_the_generation_they_stand_for() -> None:
    export = _plan(_keys(10))

    for batch in export.batches:
        assert len(batch.batch_id) == 8
        int(batch.batch_id, 16)
        for item_id in batch.item_ids:
            assert len(item_id) == 16
            int(item_id, 16)
    assert len({b.batch_id for b in export.batches}) == len(export.batches)


def test_the_plan_depends_only_on_the_keys_and_the_seed() -> None:
    keys = _keys(10)

    first = _plan(keys)

    assert _plan(list(reversed(keys))) == first
    assert _plan(keys, seed=8) != first
    assert first.stage == "tune"
    assert first.seed == 7
    assert first.criteria == CRITERIA
    assert first.rubric_sha256 == RUBRIC_SHA


def test_the_plan_survives_a_round_trip_through_json() -> None:
    export = _plan(_keys(10))

    restored = Export.from_dict(json.loads(json.dumps(export.to_dict())))

    assert restored == export
    assert restored.batch(export.batches[1].batch_id) == export.batches[1]


def test_an_unknown_batch_id_is_an_error() -> None:
    with pytest.raises(ValueError, match="unknown batch 'nope'"):
        _plan(_keys(3)).batch("nope")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"keys": ["a", "a"]}, "duplicate"),
        ({"keys": []}, "no generations"),
        ({"batch_size": 0}, "batch_size"),
        ({"second_fraction": -0.1}, "second_fraction"),
        ({"second_fraction": 1.5}, "second_fraction"),
        ({"criteria": ()}, "criteria"),
        ({"criteria": ("concept", "style")}, "criteria"),
        ({"criteria": ("fluency", "concept")}, "criteria"),
        ({"stage": "Tune 1"}, "stage"),
    ],
    ids=[
        "duplicate_key",
        "no_keys",
        "batch_size",
        "negative_fraction",
        "large_fraction",
        "no_criteria",
        "unknown_criterion",
        "criteria_order",
        "stage_name",
    ],
)
def test_the_plan_rejects_invalid_arguments(
    change: dict[str, object], message: str
) -> None:
    arguments: dict[str, object] = {
        "keys": _keys(4),
        "stage": "tune",
        "seed": 0,
        "criteria": CRITERIA,
        "rubric_sha256": RUBRIC_SHA,
        "batch_size": 4,
        "second_fraction": 0.5,
    }
    arguments.update(change)
    keys = arguments.pop("keys")

    with pytest.raises(ValueError, match=message):
        plan_export(keys, **arguments)  # type: ignore[arg-type]


def test_a_batch_line_holds_only_what_the_judge_may_see(tmp_path: Path) -> None:
    keys = _keys(10)
    generations = _generations(keys)
    export = _plan(keys)

    write_export(export, generations, tmp_path, rubric=RUBRIC)

    for batch in export.batches:
        lines = (
            (tmp_path / f"{batch.batch_id}.jsonl")
            .read_text("utf-8")
            .splitlines()
        )
        rows = [json.loads(line) for line in lines]
        assert [row["item_id"] for row in rows] == list(batch.item_ids)
        for row, key in zip(rows, batch.keys, strict=True):
            assert row == {
                "item_id": row["item_id"],
                "prompt": generations[key].prompt,
                "concept": "positive sentiment",
                "generation": generations[key].text,
            }


def test_no_exported_file_names_an_arm_a_coefficient_or_a_split(
    tmp_path: Path,
) -> None:
    keys = _keys(10)
    export = _plan(keys)

    paths = write_export(export, _generations(keys), tmp_path, rubric=RUBRIC)

    assert sorted(paths) == sorted(tmp_path.iterdir())
    for path in paths:
        text = path.read_text("utf-8")
        assert "caa" not in text
        assert "0.2" not in text
        assert "sentiment/" not in text
        if path.suffix == ".jsonl":
            assert "tune" not in text
            assert "tune" not in path.name


def test_the_manifest_tells_which_file_goes_to_which_judge(
    tmp_path: Path,
) -> None:
    keys = _keys(10)
    export = _plan(keys)

    write_export(export, _generations(keys), tmp_path, rubric=RUBRIC)

    manifest = json.loads((tmp_path / "manifest_tune.json").read_text("utf-8"))
    assert manifest == {
        "stage": "tune",
        "criteria": ["concept", "fluency"],
        "scores": [0, 1, 2],
        "rubric_file": "rubric.txt",
        "rubric_sha256": RUBRIC_SHA,
        "batches": [
            {
                "batch_id": batch.batch_id,
                "file": f"{batch.batch_id}.jsonl",
                "judge": batch.judge,
                "n_items": len(batch.item_ids),
            }
            for batch in export.batches
        ],
    }
    assert (tmp_path / "rubric.txt").read_bytes() == RUBRIC.encode("utf-8")


def test_two_exports_of_one_plan_are_byte_identical(tmp_path: Path) -> None:
    keys = _keys(10)
    generations = _generations(keys)

    first = write_export(
        _plan(keys), generations, tmp_path / "a", rubric=RUBRIC
    )
    second = write_export(
        _plan(keys), generations, tmp_path / "b", rubric=RUBRIC
    )

    assert [p.name for p in first] == [p.name for p in second]
    for one, other in zip(first, second, strict=True):
        data = one.read_bytes()
        assert data == other.read_bytes()
        assert b"\r" not in data
    batch = next(p for p in first if p.suffix == ".jsonl")
    assert "café".encode() in b"".join(
        p.read_bytes() for p in first if p.suffix == ".jsonl"
    )
    assert batch.read_bytes().endswith(b"}\n")


def test_writing_needs_the_generation_of_every_planned_key(
    tmp_path: Path,
) -> None:
    keys = _keys(4)
    generations = _generations(keys[:3])

    with pytest.raises(ValueError, match="no generation for"):
        write_export(_plan(keys), generations, tmp_path, rubric=RUBRIC)

    assert list(tmp_path.iterdir()) == []


def test_a_batch_is_immutable() -> None:
    batch = _plan(_keys(3)).batches[0]

    with pytest.raises(dataclasses.FrozenInstanceError):
        batch.judge = 2  # type: ignore[misc]
