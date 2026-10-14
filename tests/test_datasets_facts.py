from pathlib import Path

import pytest

from tomography.datasets import facts
from tomography.datasets.facts import SOURCE
from tomography.datasets.facts import load_facts
from tomography.datasets.facts import parse_facts
from tomography.datasets.records import assign_splits

CSV = (
    "statement,label\r\n"
    "The Earth is round.,1\r\n"
    "The moon is made of cheese.,0\r\n"
    '"Water boils at 100 degrees, at sea level.",1\r\n'
)


def test_source_pins_the_reference_repository_commit() -> None:
    assert SOURCE.name == "facts"
    assert SOURCE.kind == "url"
    assert SOURCE.location == (
        "https://raw.githubusercontent.com/andyzoujm/"
        "representation-engineering/"
        "5455d8a375d5fb1cb191f9ebcd089b7c21e9a31e/"
    )
    assert SOURCE.files == ("data/facts/facts_true_false.csv",)
    assert SOURCE.sha256 == {
        "data/facts/facts_true_false.csv": (
            "151b6163289460eb8b37f7c6a1998e23aed8e05b5e557bf1a2297079bf0e5af9"
        )
    }


def test_parse_reads_statements_and_binary_labels() -> None:
    examples = parse_facts(CSV)

    assert [e.text for e in examples] == [
        "The Earth is round.",
        "The moon is made of cheese.",
        "Water boils at 100 degrees, at sea level.",
    ]
    assert [e.label for e in examples] == [1, 0, 1]


def test_parse_uses_the_data_row_index_as_the_uid() -> None:
    examples = parse_facts(CSV)

    assert [e.uid for e in examples] == ["facts-0", "facts-1", "facts-2"]


def test_parse_strips_the_edges_of_a_statement() -> None:
    (example,) = parse_facts('statement,label\n" Padded. ",1\n')

    assert example.text == "Padded."


def test_parse_keeps_one_copy_of_a_repeated_statement() -> None:
    text = "statement,label\nA.,1\nB.,0\nA.,1\nA.,1\n"

    examples = parse_facts(text)

    assert [(e.uid, e.text) for e in examples] == [
        ("facts-0", "A."),
        ("facts-1", "B."),
    ]


def test_parse_drops_a_statement_that_carries_both_labels() -> None:
    text = "statement,label\nA.,1\nB.,0\nA.,0\n"

    examples = parse_facts(text)

    assert [e.text for e in examples] == ["B."]


def test_parse_assigns_splits_by_uid() -> None:
    lines = [f"Statement {i}.,{i % 2}" for i in range(300)]
    examples = parse_facts("statement,label\n" + "\n".join(lines) + "\n")

    expected = assign_splits([e.uid for e in examples], dataset="facts")

    assert [e.split for e in examples] == expected
    assert {"train", "val", "test"} == set(expected)


def test_parse_gives_every_example_its_own_meta() -> None:
    first, second = parse_facts(CSV)[:2]

    assert first.meta == {}
    assert first.meta is not second.meta


def test_parse_of_a_header_only_file_is_empty() -> None:
    assert parse_facts("statement,label\n") == []


@pytest.mark.parametrize(
    "label",
    ["2", "-1", "true", ""],
    ids=["two", "negative", "word", "empty"],
)
def test_parse_rejects_a_label_that_is_not_0_or_1(label: str) -> None:
    with pytest.raises(ValueError, match="facts-0"):
        parse_facts(f"statement,label\nA.,{label}\n")


def test_parse_rejects_a_file_without_the_expected_columns() -> None:
    with pytest.raises(ValueError, match="statement"):
        parse_facts("text,label\nA.,1\n")


def test_load_reads_the_fetched_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "facts.csv"
    path.write_bytes(CSV.encode("utf-8"))
    calls: list[dict[str, object]] = []

    def fake_fetch(source: object, **kwargs: object) -> Path:
        calls.append({"source": source, **kwargs})
        return path

    monkeypatch.setattr(facts, "fetch", fake_fetch)

    examples = load_facts(cache=tmp_path, offline=True)

    assert [e.label for e in examples] == [1, 0, 1]
    assert calls == [
        {"source": SOURCE, "cache": tmp_path, "offline": True},
    ]


@pytest.mark.network
def test_load_reads_the_pinned_file() -> None:
    examples = load_facts()

    assert len(examples) == 597
    assert sum(e.label for e in examples) == 297
    assert len({e.uid for e in examples}) == len(examples)
    assert len({e.text for e in examples}) == len(examples)
