import collections
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tomography.datasets import arc
from tomography.datasets.arc import EASY_SOURCE
from tomography.datasets.arc import SOURCE
from tomography.datasets.arc import load_arc
from tomography.datasets.arc import parse_arc
from tomography.datasets.records import ChoiceExample

FILES = {
    "train": "ARC-Challenge/train-00000-of-00001.parquet",
    "validation": "ARC-Challenge/validation-00000-of-00001.parquet",
    "test": "ARC-Challenge/test-00000-of-00001.parquet",
}
EASY_FILES = {
    "train": "ARC-Easy/train-00000-of-00001.parquet",
    "validation": "ARC-Easy/validation-00000-of-00001.parquet",
    "test": "ARC-Easy/test-00000-of-00001.parquet",
}


def _row(
    row_id: str,
    texts: list[str],
    labels: list[str],
    answer: str,
) -> dict[str, object]:
    return {
        "id": row_id,
        "question": f"Question {row_id}? ",
        "choices": {"text": texts, "label": labels},
        "answerKey": answer,
    }


LETTERED = _row("Mercury_1", ["a", "b", "c", "d"], ["A", "B", "C", "D"], "C")
NUMBERED = _row("MCAS_2", ["w", "x", "y", "z"], ["1", "2", "3", "4"], "4")
EASY = _row("Mercury_7", ["p", "q", "r"], ["A", "B", "C"], "B")
SERVED = {
    FILES["train"]: [LETTERED, NUMBERED],
    FILES["validation"]: [LETTERED],
    FILES["test"]: [],
    EASY_FILES["train"]: [EASY],
    EASY_FILES["validation"]: [],
    EASY_FILES["test"]: [NUMBERED],
}


def parsed(files: dict[str, str]) -> list[ChoiceExample]:
    return [
        item
        for source_split, file in files.items()
        for item in parse_arc(SERVED[file], source_split=source_split)  # type: ignore[arg-type]
    ]


def test_source_pins_the_hub_revision() -> None:
    assert SOURCE.name == "arc"
    assert SOURCE.kind == "hub"
    assert SOURCE.location == "allenai/ai2_arc"
    assert SOURCE.revision == "210d026faf9955653af8916fad021475a3f00453"
    assert SOURCE.files == tuple(FILES.values())


def test_easy_source_pins_the_same_revision_with_the_easy_files() -> None:
    assert EASY_SOURCE.name == "arc_easy"
    assert EASY_SOURCE.kind == "hub"
    assert EASY_SOURCE.location == SOURCE.location
    assert EASY_SOURCE.revision == SOURCE.revision
    assert EASY_SOURCE.files == tuple(EASY_FILES.values())
    assert EASY_SOURCE.license == SOURCE.license


def test_parse_points_correct_at_the_answer_key() -> None:
    lettered, numbered = parse_arc([LETTERED, NUMBERED], source_split="train")

    assert isinstance(lettered, ChoiceExample)
    assert (lettered.choices, lettered.correct) == (("a", "b", "c", "d"), 2)
    assert (numbered.choices, numbered.correct) == (("w", "x", "y", "z"), 3)


def test_parse_builds_uids_and_strips_the_question() -> None:
    (item,) = parse_arc([LETTERED], source_split="train")

    assert item.uid == "arc-Mercury_1"
    assert item.question == "Question Mercury_1?"


@pytest.mark.parametrize(
    ("source_split", "split"),
    [("train", "train"), ("validation", "val"), ("test", "test")],
    ids=["train", "validation", "test"],
)
def test_parse_maps_the_source_split(source_split: str, split: str) -> None:
    (item,) = parse_arc([LETTERED], source_split=source_split)  # type: ignore[arg-type]

    assert item.split == split


def test_parse_keeps_questions_with_other_numbers_of_choices() -> None:
    three = _row("t", ["a", "b", "c"], ["A", "B", "C"], "A")
    five = _row("f", list("abcde"), list("ABCDE"), "E")

    first, second = parse_arc([three, five], source_split="test")

    assert (len(first.choices), first.correct) == (3, 0)
    assert (len(second.choices), second.correct) == (5, 4)


def test_parse_gives_every_item_its_own_meta() -> None:
    first, second = parse_arc([LETTERED, NUMBERED], source_split="train")

    assert first.meta == {}
    assert first.meta is not second.meta


def test_parse_rejects_an_answer_key_that_is_not_a_label() -> None:
    row = _row("bad_1", ["a", "b"], ["A", "B"], "C")

    with pytest.raises(ValueError, match="arc-bad_1"):
        parse_arc([row], source_split="train")


def test_parse_rejects_choices_and_labels_of_different_length() -> None:
    row = _row("bad_2", ["a", "b", "c"], ["A", "B"], "A")

    with pytest.raises(ValueError, match="arc-bad_2"):
        parse_arc([row], source_split="train")


@pytest.fixture
def served(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> list[dict[str, object]]:
    paths = {}
    for index, (file, split_rows) in enumerate(SERVED.items()):
        path = tmp_path / f"{index}.parquet"
        schema = pa.schema(
            [
                ("id", pa.string()),
                ("question", pa.string()),
                (
                    "choices",
                    pa.struct(
                        [
                            ("text", pa.list_(pa.string())),
                            ("label", pa.list_(pa.string())),
                        ]
                    ),
                ),
                ("answerKey", pa.string()),
            ]
        )
        pq.write_table(pa.Table.from_pylist(split_rows, schema=schema), path)
        paths[file] = path
    calls: list[dict[str, object]] = []

    def fake_fetch(source: object, *, file: str, **kwargs: object) -> Path:
        calls.append({"source": source, "file": file, **kwargs})
        return paths[file]

    monkeypatch.setattr(arc, "fetch", fake_fetch)
    return calls


def test_load_reads_train_validation_then_test(
    served: list[dict[str, object]], tmp_path: Path
) -> None:
    items = load_arc(cache=tmp_path, offline=True)

    assert [c["file"] for c in served] == list(FILES.values())
    assert all(c["source"] is SOURCE for c in served)
    assert all(c["cache"] == tmp_path for c in served)
    assert all(c["offline"] is True for c in served)
    assert [(i.uid, i.split) for i in items] == [
        ("arc-Mercury_1", "train"),
        ("arc-MCAS_2", "train"),
        ("arc-Mercury_1", "val"),
    ]


def test_load_default_equals_the_parsed_challenge_files(
    served: list[dict[str, object]], tmp_path: Path
) -> None:
    items = load_arc(cache=tmp_path, offline=True)

    assert items == parsed(FILES)


def test_load_reads_the_easy_files_for_arc_easy(
    served: list[dict[str, object]], tmp_path: Path
) -> None:
    items = load_arc(config="ARC-Easy", cache=tmp_path, offline=True)

    assert [c["file"] for c in served] == list(EASY_FILES.values())
    assert all(c["source"] is EASY_SOURCE for c in served)
    assert all(c["cache"] == tmp_path for c in served)
    assert all(c["offline"] is True for c in served)
    assert items == parsed(EASY_FILES)


@pytest.mark.network
def test_load_reads_the_pinned_easy_revision() -> None:
    items = load_arc(config="ARC-Easy")

    by_split = collections.Counter(i.split for i in items)
    assert by_split == {"train": 2_251, "val": 570, "test": 2_376}
    assert len({i.uid for i in items}) == len(items)
    assert {len(i.choices) for i in items} == {3, 4, 5}


@pytest.mark.network
def test_load_reads_the_pinned_revision() -> None:
    items = load_arc()

    by_split = collections.Counter(i.split for i in items)
    assert by_split == {"train": 1_119, "val": 299, "test": 1_172}
    assert len({i.uid for i in items}) == len(items)
    assert {len(i.choices) for i in items} == {3, 4, 5}
    assert all(0 <= i.correct < len(i.choices) for i in items)
