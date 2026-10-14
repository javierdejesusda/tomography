from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tomography.datasets import wikitext
from tomography.datasets.wikitext import SOURCE
from tomography.datasets.wikitext import load_wikitext
from tomography.datasets.wikitext import parse_wikitext

FILE = "wikitext-103-raw-v1/test-00000-of-00001.parquet"
LINES = ["", " = Robert Boulter = \n", "", " He was born in 1985 . \n", ""]


def test_source_pins_the_test_shard_only() -> None:
    assert SOURCE.name == "wikitext"
    assert SOURCE.kind == "hub"
    assert SOURCE.location == "Salesforce/wikitext"
    assert SOURCE.revision == "b08601e04326c79dfdd32d625aee71d232d685c3"
    assert SOURCE.files == (FILE,)


def test_parse_skips_the_empty_lines() -> None:
    examples = parse_wikitext(LINES)

    assert [e.text for e in examples] == [
        "= Robert Boulter =",
        "He was born in 1985 .",
    ]


def test_parse_numbers_uids_by_the_line_of_the_source_file() -> None:
    examples = parse_wikitext(LINES)

    assert [e.uid for e in examples] == ["wikitext-test-1", "wikitext-test-3"]


def test_parse_marks_every_line_as_unlabeled_test_text() -> None:
    for example in parse_wikitext(LINES):
        assert example.label == -1
        assert example.split == "test"
        assert example.meta == {}


def test_parse_treats_a_whitespace_only_line_as_empty() -> None:
    examples = parse_wikitext([" \n", "\t", "Text."])

    assert [e.uid for e in examples] == ["wikitext-test-2"]


@pytest.fixture
def served(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> list[dict[str, object]]:
    path = tmp_path / "test.parquet"
    pq.write_table(pa.table({"text": LINES}), path)
    calls: list[dict[str, object]] = []

    def fake_fetch(source: object, **kwargs: object) -> Path:
        calls.append({"source": source, **kwargs})
        return path

    monkeypatch.setattr(wikitext, "fetch", fake_fetch)
    return calls


def test_load_reads_the_parquet(
    served: list[dict[str, object]], tmp_path: Path
) -> None:
    examples = load_wikitext(cache=tmp_path, offline=True)

    assert served == [{"source": SOURCE, "cache": tmp_path, "offline": True}]
    assert examples == parse_wikitext(LINES)


@pytest.mark.network
def test_load_reads_the_pinned_test_shard() -> None:
    examples = load_wikitext()

    assert len(examples) == 2_891
    assert len({e.uid for e in examples}) == len(examples)
    assert examples[0].text == "= Robert Boulter ="
    assert all(e.text for e in examples)
