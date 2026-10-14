from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tomography.datasets import sst2
from tomography.datasets.records import Example
from tomography.datasets.sst2 import SOURCE
from tomography.datasets.sst2 import load_sst2
from tomography.datasets.sst2 import parse_sst2

TRAIN_FILE = "data/train-00000-of-00001.parquet"
VALIDATION_FILE = "data/validation-00000-of-00001.parquet"


def _write_parquet(path: Path, rows: list[tuple[int, str, int]]) -> Path:
    table = pa.table(
        {
            "idx": pa.array([r[0] for r in rows], type=pa.int32()),
            "sentence": [r[1] for r in rows],
            "label": pa.array([r[2] for r in rows], type=pa.int64()),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def test_source_pins_the_stanfordnlp_revision() -> None:
    assert SOURCE.name == "sst2"
    assert SOURCE.kind == "hub"
    assert SOURCE.location == "stanfordnlp/sst2"
    assert SOURCE.revision == "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb"


def test_source_serves_the_labeled_splits_only() -> None:
    assert SOURCE.files == (TRAIN_FILE, VALIDATION_FILE)


def test_parse_keeps_the_binary_label() -> None:
    rows = [
        {"idx": 0, "sentence": "a gem", "label": 1},
        {"idx": 1, "sentence": "a bore", "label": 0},
    ]

    examples = parse_sst2(rows, source_split="train")

    assert [e.label for e in examples] == [1, 0]


def test_parse_strips_the_padding_around_a_phrase() -> None:
    rows = [{"idx": 0, "sentence": "hide new secretions ", "label": 0}]

    (example,) = parse_sst2(rows, source_split="train")

    assert example.text == "hide new secretions"


def test_parse_builds_a_uid_from_the_source_split_and_row_index() -> None:
    rows = [{"idx": 7, "sentence": "fine", "label": 1}]

    (train,) = parse_sst2(rows, source_split="train")
    (validation,) = parse_sst2(rows, source_split="validation")

    assert train.uid == "sst2-train-7"
    assert validation.uid == "sst2-validation-7"


@pytest.mark.parametrize(
    ("source_split", "split"),
    [("train", "train"), ("validation", "val")],
    ids=["train", "validation"],
)
def test_parse_maps_the_source_split(source_split: str, split: str) -> None:
    rows = [{"idx": 0, "sentence": "fine", "label": 1}]

    (example,) = parse_sst2(rows, source_split=source_split)  # type: ignore[arg-type]

    assert example.split == split


def test_parse_gives_every_example_its_own_meta() -> None:
    rows = [
        {"idx": 0, "sentence": "a", "label": 1},
        {"idx": 1, "sentence": "b", "label": 0},
    ]

    first, second = parse_sst2(rows, source_split="train")

    assert first.meta == {}
    assert first.meta is not second.meta


def test_parse_rejects_the_unlabeled_test_split() -> None:
    rows = [{"idx": 3, "sentence": "no gold", "label": -1}]

    with pytest.raises(ValueError, match="sst2-train-3"):
        parse_sst2(rows, source_split="train")


def test_parse_of_no_rows_is_empty() -> None:
    assert parse_sst2([], source_split="train") == []


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict]:
    files = {
        TRAIN_FILE: _write_parquet(
            tmp_path / "train.parquet",
            [(0, "a gem ", 1), (1, "a bore ", 0), (2, "fine ", 1)],
        ),
        VALIDATION_FILE: _write_parquet(
            tmp_path / "validation.parquet", [(0, "sharp .", 1)]
        ),
    }
    calls: list[dict] = []

    def fake_fetch(source: object, *, file: str, **kwargs: object) -> Path:
        calls.append({"source": source, "file": file, **kwargs})
        return files[file]

    monkeypatch.setattr(sst2, "fetch", fake_fetch)
    return calls


def test_load_returns_train_then_validation(served: list[dict]) -> None:
    examples = load_sst2()

    assert [e.uid for e in examples] == [
        "sst2-train-0",
        "sst2-train-1",
        "sst2-train-2",
        "sst2-validation-0",
    ]
    assert [e.split for e in examples] == ["train", "train", "train", "val"]
    assert [e.text for e in examples] == ["a gem", "a bore", "fine", "sharp ."]
    assert all(isinstance(e, Example) for e in examples)


def test_load_passes_the_cache_and_offline_flags_to_fetch(
    served: list[dict], tmp_path: Path
) -> None:
    load_sst2(cache=tmp_path, offline=True)

    assert [c["file"] for c in served] == [TRAIN_FILE, VALIDATION_FILE]
    assert all(c["source"] is SOURCE for c in served)
    assert all(c["cache"] == tmp_path for c in served)
    assert all(c["offline"] is True for c in served)


@pytest.mark.network
def test_load_reads_the_pinned_revision() -> None:
    examples = load_sst2()

    train = [e for e in examples if e.split == "train"]
    validation = [e for e in examples if e.split == "val"]
    assert len(train) == 67_349
    assert len(validation) == 872
    assert {e.label for e in examples} == {0, 1}
    assert train[0].text == "hide new secretions from the parental units"
    assert len({e.uid for e in examples}) == len(examples)
