import collections
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tomography.datasets import commonsenseqa
from tomography.datasets.commonsenseqa import SOURCE
from tomography.datasets.commonsenseqa import load_commonsenseqa
from tomography.datasets.commonsenseqa import parse_commonsenseqa
from tomography.datasets.records import ChoiceExample

FILES = {
    "train": "data/train-00000-of-00001.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
}


def _row(row_id: str, texts: list[str], answer: str) -> dict[str, object]:
    return {
        "id": row_id,
        "question": f"Where is {row_id}? ",
        "question_concept": "place",
        "choices": {"label": list("ABCDE")[: len(texts)], "text": texts},
        "answerKey": answer,
    }


SHOP = _row("075e48", ["mall", "street", "town", "ice cream ", "home"], "C")
PARK = _row("1afa02", ["bench", "tree", "lake", "path", "gate"], "A")


def test_source_pins_the_hub_revision() -> None:
    assert SOURCE.name == "commonsenseqa"
    assert SOURCE.kind == "hub"
    assert SOURCE.location == "tau/commonsense_qa"
    assert SOURCE.revision == "94630fe30dad47192a8546eb75f094926d47e155"
    assert SOURCE.files == tuple(FILES.values())
    assert SOURCE.license == "MIT"


def test_parse_points_correct_at_the_answer_key() -> None:
    shop, park = parse_commonsenseqa([SHOP, PARK], source_split="train")

    assert isinstance(shop, ChoiceExample)
    assert shop.choices == ("mall", "street", "town", "ice cream", "home")
    assert shop.correct == 2
    assert park.correct == 0


def test_parse_builds_uids_and_strips_the_question() -> None:
    (item,) = parse_commonsenseqa([SHOP], source_split="train")

    assert item.uid == "commonsenseqa-075e48"
    assert item.question == "Where is 075e48?"
    assert item.meta == {}


@pytest.mark.parametrize(
    ("source_split", "split"),
    [("train", "train"), ("validation", "val")],
    ids=["train", "validation"],
)
def test_parse_maps_the_source_split(source_split: str, split: str) -> None:
    (item,) = parse_commonsenseqa([SHOP], source_split=source_split)  # type: ignore[arg-type]

    assert item.split == split


def test_parse_rejects_an_unlabeled_row() -> None:
    row = _row("hidden", ["a", "b", "c", "d", "e"], "")

    with pytest.raises(ValueError, match="commonsenseqa-hidden"):
        parse_commonsenseqa([row], source_split="validation")


def test_parse_rejects_choices_and_labels_of_different_length() -> None:
    row = _row("bad", ["a", "b", "c"], "A")
    row["choices"] = {"label": ["A", "B"], "text": ["a", "b", "c"]}

    with pytest.raises(ValueError, match="commonsenseqa-bad"):
        parse_commonsenseqa([row], source_split="train")


@pytest.fixture
def served(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> list[dict[str, object]]:
    rows = {"train": [SHOP, PARK], "validation": [PARK]}
    schema = pa.schema(
        [
            ("id", pa.string()),
            ("question", pa.string()),
            ("question_concept", pa.string()),
            (
                "choices",
                pa.struct(
                    [
                        ("label", pa.list_(pa.string())),
                        ("text", pa.list_(pa.string())),
                    ]
                ),
            ),
            ("answerKey", pa.string()),
        ]
    )
    paths = {}
    for name, split_rows in rows.items():
        path = tmp_path / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist(split_rows, schema=schema), path)
        paths[FILES[name]] = path
    calls: list[dict[str, object]] = []

    def fake_fetch(source: object, *, file: str, **kwargs: object) -> Path:
        calls.append({"source": source, "file": file, **kwargs})
        return paths[file]

    monkeypatch.setattr(commonsenseqa, "fetch", fake_fetch)
    return calls


def test_load_reads_train_then_validation(
    served: list[dict[str, object]], tmp_path: Path
) -> None:
    items = load_commonsenseqa(cache=tmp_path, offline=True)

    assert [c["file"] for c in served] == list(FILES.values())
    assert all(c["source"] is SOURCE for c in served)
    assert all(c["cache"] == tmp_path for c in served)
    assert all(c["offline"] is True for c in served)
    assert [(i.uid, i.split) for i in items] == [
        ("commonsenseqa-075e48", "train"),
        ("commonsenseqa-1afa02", "train"),
        ("commonsenseqa-1afa02", "val"),
    ]


@pytest.mark.network
def test_load_reads_the_pinned_revision() -> None:
    items = load_commonsenseqa()

    by_split = collections.Counter(i.split for i in items)
    assert by_split == {"train": 9_741, "val": 1_221}
    assert len({i.uid for i in items}) == len(items)
    assert {len(i.choices) for i in items} == {5}
