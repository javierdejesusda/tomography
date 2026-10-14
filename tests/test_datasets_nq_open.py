from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tomography.datasets import nq_open
from tomography.datasets.records import QAExample
from tomography.datasets.records import assign_splits


def _row(
    question: str = "when was the last time anyone was on the moon",
    answer: tuple[str, ...] = ("14 December 1972 UTC", "December 1972"),
) -> dict[str, Any]:
    return {"question": question, "answer": list(answer)}


def test_source_pins_the_validation_parquet() -> None:
    source = nq_open.SOURCE

    assert source.name == "nq_open"
    assert source.kind == "hub"
    assert source.location == "google-research-datasets/nq_open"
    assert source.revision == "5dd9790a83002ad084ddeb7c420dc716852c6f28"
    assert source.files == ("nq_open/validation-00000-of-00001.parquet",)
    assert source.license == "CC-BY-SA-3.0"


def test_parse_builds_an_open_item_per_row_keyed_by_position() -> None:
    items = nq_open.parse_nq_open([_row(), _row("who sings it", ("Ann",))])

    assert items[0] == QAExample(
        uid="nq_open:0",
        question="when was the last time anyone was on the moon",
        answers=("14 December 1972 UTC", "December 1972"),
        kind="open",
        split=assign_splits(["nq_open:0"], dataset="nq_open")[0],
        meta={"source": "nq_open"},
    )
    assert items[1].uid == "nq_open:1"
    assert items[1].answers == ("Ann",)


def test_parse_drops_repeated_answers() -> None:
    (item,) = nq_open.parse_nq_open([_row(answer=("Ann", "Ann", "Bo"))])

    assert item.answers == ("Ann", "Bo")


def test_load_reads_the_parquet_through_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "validation.parquet"
    pq.write_table(pa.Table.from_pylist([_row()]), path)
    calls: list[tuple[object, Path | None, bool]] = []

    def fake_fetch(
        source: object,
        *,
        file: str | None = None,
        cache: Path | None = None,
        offline: bool = False,
    ) -> Path:
        calls.append((source, cache, offline))
        return path

    monkeypatch.setattr(nq_open, "fetch", fake_fetch)

    items = nq_open.load_nq_open(cache=tmp_path, offline=True)

    assert calls == [(nq_open.SOURCE, tmp_path, True)]
    assert [i.uid for i in items] == ["nq_open:0"]


@pytest.mark.network
def test_network_loads_the_validation_questions() -> None:
    items = nq_open.load_nq_open()

    assert len(items) == 3610
    assert len({i.uid for i in items}) == 3610
    assert all(i.answers for i in items)
