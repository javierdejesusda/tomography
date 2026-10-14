from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tomography.datasets import triviaqa
from tomography.datasets.records import QAExample
from tomography.datasets.records import assign_splits


def _row(
    question_id: str = "tc_2",
    question: str = "Who was the man behind The Chipmunks?",
    value: str = "David Seville",
    aliases: tuple[str, ...] = ("David Seville", "Ross Bagdasarian"),
) -> dict[str, Any]:
    return {
        "question": question,
        "question_id": question_id,
        "answer": {"value": value, "aliases": list(aliases)},
    }


def test_source_pins_the_validation_parquet_of_the_rc_config() -> None:
    source = triviaqa.SOURCE

    assert source.name == "triviaqa"
    assert source.kind == "hub"
    assert source.location == "mandarjoshi/trivia_qa"
    assert source.revision == "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f"
    assert source.files == ("rc.nocontext/validation-00000-of-00001.parquet",)


def test_parse_builds_an_open_item_with_the_value_and_aliases() -> None:
    items = triviaqa.parse_triviaqa([_row()])

    assert items == [
        QAExample(
            uid="triviaqa:tc_2",
            question="Who was the man behind The Chipmunks?",
            answers=("David Seville", "Ross Bagdasarian"),
            kind="open",
            split=assign_splits(["triviaqa:tc_2"], dataset="triviaqa")[0],
            meta={"source": "triviaqa"},
        )
    ]


def test_parse_puts_the_value_first_and_drops_repeated_aliases() -> None:
    row = _row(value="Sunset Boulevard", aliases=("Sunset Blvd", "Sunset Blvd"))

    (item,) = triviaqa.parse_triviaqa([row])

    assert item.answers == ("Sunset Boulevard", "Sunset Blvd")


def test_parse_keeps_one_item_per_question_id() -> None:
    rows = [_row("a", "first?"), _row("b", "other?"), _row("a", "second?")]

    items = triviaqa.parse_triviaqa(rows)

    assert [i.uid for i in items] == ["triviaqa:a", "triviaqa:b"]
    assert items[0].question == "first?"


def test_load_reads_the_parquet_through_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "validation.parquet"
    rows = [{**_row(), "entity_pages": {"title": ["Alvin"]}}]
    pq.write_table(pa.Table.from_pylist(rows), path)
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

    monkeypatch.setattr(triviaqa, "fetch", fake_fetch)

    items = triviaqa.load_triviaqa(cache=tmp_path, offline=True)

    assert calls == [(triviaqa.SOURCE, tmp_path, True)]
    assert [i.uid for i in items] == ["triviaqa:tc_2"]


@pytest.mark.network
def test_network_loads_the_unique_validation_questions() -> None:
    items = triviaqa.load_triviaqa()

    assert len(items) == 9960
    assert len({i.uid for i in items}) == 9960
    assert all(i.answers for i in items)
