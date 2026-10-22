import collections
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tomography.datasets import openbookqa
from tomography.datasets.openbookqa import SOURCE
from tomography.datasets.openbookqa import load_openbookqa
from tomography.datasets.openbookqa import parse_openbookqa
from tomography.datasets.records import ChoiceExample

FILES = {
    "train": "main/train-00000-of-00001.parquet",
    "validation": "main/validation-00000-of-00001.parquet",
    "test": "main/test-00000-of-00001.parquet",
}


def _row(row_id: str, texts: list[str], answer: str) -> dict[str, object]:
    return {
        "id": row_id,
        "question_stem": f"The stem of {row_id} ",
        "choices": {"text": texts, "label": list("ABCD")[: len(texts)]},
        "answerKey": answer,
    }


SUN = _row("7-980", ["puppies", "children", "flowers", "plants "], "D")
MOON = _row("8-376", ["tides", "rain", "wind", "snow"], "A")


def test_source_pins_the_hub_revision() -> None:
    assert SOURCE.name == "openbookqa"
    assert SOURCE.kind == "hub"
    assert SOURCE.location == "allenai/openbookqa"
    assert SOURCE.revision == "388097ea7776314e93a529163e0fea805b8a6454"
    assert SOURCE.files == tuple(FILES.values())
    assert SOURCE.license == (
        "unknown on the dataset card, Apache-2.0 on the GitHub repository"
    )


def test_parse_points_correct_at_the_answer_key() -> None:
    sun, moon = parse_openbookqa([SUN, MOON], source_split="train")

    assert isinstance(sun, ChoiceExample)
    assert sun.choices == ("puppies", "children", "flowers", "plants")
    assert sun.correct == 3
    assert moon.choices == ("tides", "rain", "wind", "snow")
    assert moon.correct == 0


def test_parse_builds_uids_and_strips_the_stem() -> None:
    (item,) = parse_openbookqa([SUN], source_split="train")

    assert item.uid == "openbookqa-7-980"
    assert item.question == "The stem of 7-980"
    assert item.meta == {}


@pytest.mark.parametrize(
    ("source_split", "split"),
    [("train", "train"), ("validation", "val"), ("test", "test")],
    ids=["train", "validation", "test"],
)
def test_parse_maps_the_source_split(source_split: str, split: str) -> None:
    (item,) = parse_openbookqa([SUN], source_split=source_split)  # type: ignore[arg-type]

    assert item.split == split


def test_parse_rejects_an_answer_key_that_is_not_a_label() -> None:
    row = _row("bad-1", ["a", "b", "c"], "D")

    with pytest.raises(ValueError, match="openbookqa-bad-1"):
        parse_openbookqa([row], source_split="train")


def test_parse_rejects_choices_and_labels_of_different_length() -> None:
    row = _row("bad-2", ["a", "b", "c"], "A")
    row["choices"] = {"text": ["a", "b", "c"], "label": ["A", "B"]}

    with pytest.raises(ValueError, match="openbookqa-bad-2"):
        parse_openbookqa([row], source_split="train")


@pytest.fixture
def served(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> list[dict[str, object]]:
    rows = {"train": [SUN, MOON], "validation": [MOON], "test": [SUN]}
    schema = pa.schema(
        [
            ("id", pa.string()),
            ("question_stem", pa.string()),
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
    paths = {}
    for name, split_rows in rows.items():
        path = tmp_path / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist(split_rows, schema=schema), path)
        paths[FILES[name]] = path
    calls: list[dict[str, object]] = []

    def fake_fetch(source: object, *, file: str, **kwargs: object) -> Path:
        calls.append({"source": source, "file": file, **kwargs})
        return paths[file]

    monkeypatch.setattr(openbookqa, "fetch", fake_fetch)
    return calls


def test_load_reads_train_validation_then_test(
    served: list[dict[str, object]], tmp_path: Path
) -> None:
    items = load_openbookqa(cache=tmp_path, offline=True)

    assert [c["file"] for c in served] == list(FILES.values())
    assert all(c["source"] is SOURCE for c in served)
    assert all(c["cache"] == tmp_path for c in served)
    assert all(c["offline"] is True for c in served)
    assert [(i.uid, i.split) for i in items] == [
        ("openbookqa-7-980", "train"),
        ("openbookqa-8-376", "train"),
        ("openbookqa-8-376", "val"),
        ("openbookqa-7-980", "test"),
    ]


@pytest.mark.network
def test_load_reads_the_pinned_revision() -> None:
    items = load_openbookqa()

    by_split = collections.Counter(i.split for i in items)
    assert by_split == {"train": 4_957, "val": 500, "test": 500}
    assert len({i.uid for i in items}) == len(items)
    assert {len(i.choices) for i in items} == {4}
