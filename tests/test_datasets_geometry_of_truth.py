import collections
from pathlib import Path

import pytest

from tomography.datasets import geometry_of_truth
from tomography.datasets.geometry_of_truth import SOURCE
from tomography.datasets.geometry_of_truth import SUBSETS
from tomography.datasets.geometry_of_truth import load_geometry_of_truth
from tomography.datasets.geometry_of_truth import parse_geometry_of_truth
from tomography.datasets.records import assign_splits

CSV = (
    "statement,label,city,country,correct_country\n"
    "The city of Krasnodar is in Russia.,1,Krasnodar,Russia,Russia\n"
    "The city of Krasnodar is in South Africa.,0,Krasnodar,"
    "South Africa,Russia\n"
)


def test_subsets_are_the_nine_documented_files() -> None:
    assert SUBSETS == (
        "cities",
        "neg_cities",
        "sp_en_trans",
        "neg_sp_en_trans",
        "larger_than",
        "smaller_than",
        "cities_cities_conj",
        "cities_cities_disj",
        "common_claim_true_false",
    )


def test_source_pins_the_repository_commit() -> None:
    assert SOURCE.name == "geometry_of_truth"
    assert SOURCE.kind == "url"
    assert SOURCE.location == (
        "https://raw.githubusercontent.com/saprmarks/geometry-of-truth/"
        "5d1c630c44f7e50bda7ad86d601ccadf9abc5ddb/"
    )
    assert SOURCE.files == tuple(f"datasets/{name}.csv" for name in SUBSETS)
    assert SOURCE.sha256["datasets/cities.csv"] == (
        "f2560a0c1758c69935b50125111b574b71896bb75930dd87b1d8c973c1fa086e"
    )
    assert SOURCE.sha256["datasets/common_claim_true_false.csv"] == (
        "1889b4793c2ebd13c767eea8b65ebd4a5c2da700567677d512097b495a262576"
    )
    assert set(SOURCE.sha256) == set(SOURCE.files)


def test_parse_reads_statements_and_binary_labels() -> None:
    examples = parse_geometry_of_truth(CSV, subset="cities")

    assert [e.text for e in examples] == [
        "The city of Krasnodar is in Russia.",
        "The city of Krasnodar is in South Africa.",
    ]
    assert [e.label for e in examples] == [1, 0]


def test_parse_builds_uids_and_records_the_subset() -> None:
    examples = parse_geometry_of_truth(CSV, subset="cities")

    assert [e.uid for e in examples] == ["got-cities-0", "got-cities-1"]
    assert [dict(e.meta) for e in examples] == [{"subset": "cities"}] * 2


def test_parse_keeps_repeated_statements_as_separate_rows() -> None:
    text = (
        "statement,label\n"
        "The word 'a' means 'b'.,1\n"
        "The word 'a' means 'b'.,1\n"
    )

    examples = parse_geometry_of_truth(text, subset="sp_en_trans")

    assert len(examples) == 2
    assert examples[0].uid != examples[1].uid


def test_parse_strips_the_edges_of_a_statement() -> None:
    (example,) = parse_geometry_of_truth(
        'statement,label\n" Padded. ",1\n', subset="cities"
    )

    assert example.text == "Padded."


def test_parse_assigns_splits_by_uid() -> None:
    lines = [f"Statement {i}.,{i % 2}" for i in range(300)]
    text = "statement,label\n" + "\n".join(lines) + "\n"

    examples = parse_geometry_of_truth(text, subset="larger_than")

    expected = assign_splits(
        [e.uid for e in examples], dataset="geometry_of_truth"
    )
    assert [e.split for e in examples] == expected


@pytest.mark.parametrize(
    "label", ["2", "true", ""], ids=["two", "word", "empty"]
)
def test_parse_rejects_a_label_that_is_not_0_or_1(label: str) -> None:
    with pytest.raises(ValueError, match="got-cities-0"):
        parse_geometry_of_truth(
            f"statement,label\nA.,{label}\n", subset="cities"
        )


def test_parse_rejects_a_file_without_the_expected_columns() -> None:
    with pytest.raises(ValueError, match="statement"):
        parse_geometry_of_truth("text,label\nA.,1\n", subset="cities")


@pytest.fixture
def served(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def fake_fetch(source: object, *, file: str, **kwargs: object) -> Path:
        calls.append({"source": source, "file": file, **kwargs})
        path = tmp_path / Path(file).name
        path.write_text(
            f"statement,label\nTrue in {Path(file).stem}.,1\n"
            f"False in {Path(file).stem}.,0\n",
            encoding="utf-8",
        )
        return path

    monkeypatch.setattr(geometry_of_truth, "fetch", fake_fetch)
    return calls


def test_load_reads_every_subset_by_default(
    served: list[dict[str, object]], tmp_path: Path
) -> None:
    examples = load_geometry_of_truth(cache=tmp_path, offline=True)

    assert [c["file"] for c in served] == [
        f"datasets/{name}.csv" for name in SUBSETS
    ]
    assert all(c["source"] is SOURCE for c in served)
    assert all(c["cache"] == tmp_path for c in served)
    assert all(c["offline"] is True for c in served)
    assert len(examples) == 18
    assert [e.meta["subset"] for e in examples][::2] == list(SUBSETS)


def test_load_reads_only_the_requested_subsets_in_order(
    served: list[dict[str, object]],
) -> None:
    examples = load_geometry_of_truth(subsets=["larger_than", "cities"])

    assert [c["file"] for c in served] == [
        "datasets/larger_than.csv",
        "datasets/cities.csv",
    ]
    assert [e.uid for e in examples] == [
        "got-larger_than-0",
        "got-larger_than-1",
        "got-cities-0",
        "got-cities-1",
    ]


def test_load_rejects_an_unknown_subset_before_fetching(
    served: list[dict[str, object]],
) -> None:
    with pytest.raises(ValueError, match="counterfact"):
        load_geometry_of_truth(subsets=["cities", "counterfact"])

    assert served == []


@pytest.mark.network
def test_load_reads_the_pinned_files() -> None:
    examples = load_geometry_of_truth()

    counts = collections.Counter(e.meta["subset"] for e in examples)
    assert counts == {
        "cities": 1_496,
        "neg_cities": 1_496,
        "sp_en_trans": 354,
        "neg_sp_en_trans": 354,
        "larger_than": 1_980,
        "smaller_than": 1_980,
        "cities_cities_conj": 1_498,
        "cities_cities_disj": 1_498,
        "common_claim_true_false": 4_450,
    }
    assert len({e.uid for e in examples}) == len(examples)
    assert {e.label for e in examples} == {0, 1}
