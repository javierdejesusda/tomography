import collections
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tomography.datasets import truthfulqa
from tomography.datasets.records import ChoiceExample
from tomography.datasets.records import assign_splits
from tomography.datasets.truthfulqa import SOURCE
from tomography.datasets.truthfulqa import load_truthfulqa
from tomography.datasets.truthfulqa import parse_truthfulqa

FILE = "multiple_choice/validation-00000-of-00001.parquet"


def _row(question: str, n_choices: int) -> dict[str, object]:
    choices = [f"{question} answer {i}" for i in range(n_choices)]
    labels = [1] + [0] * (n_choices - 1)
    return {
        "question": question,
        "mc1_targets": {"choices": choices, "labels": labels},
        "mc2_targets": {"choices": choices, "labels": labels},
    }


def _rows(n: int, n_choices: int = 4) -> list[dict[str, object]]:
    return [_row(f"Question {i}?", n_choices) for i in range(n)]


def test_source_pins_the_hub_revision() -> None:
    assert SOURCE.name == "truthfulqa"
    assert SOURCE.kind == "hub"
    assert SOURCE.location == "truthfulqa/truthful_qa"
    assert SOURCE.revision == "741b8276f2d1982aa3d5b832d3ee81ed3b896490"
    assert SOURCE.files == (FILE,)


def test_parse_keeps_each_question_and_its_choice_set() -> None:
    items = parse_truthfulqa(_rows(20), seed=0)

    for i, item in enumerate(items):
        assert isinstance(item, ChoiceExample)
        assert item.question == f"Question {i}?"
        assert sorted(item.choices) == [
            f"Question {i}? answer {k}" for k in range(4)
        ]


def test_parse_points_correct_at_the_original_right_answer() -> None:
    for i, item in enumerate(parse_truthfulqa(_rows(50), seed=3)):
        assert item.choices[item.correct] == f"Question {i}? answer 0"


def test_parse_moves_the_correct_answer_away_from_index_zero() -> None:
    items = parse_truthfulqa(_rows(400), seed=0)

    positions = collections.Counter(item.correct for item in items)
    assert set(positions) == {0, 1, 2, 3}
    assert all(60 <= count <= 140 for count in positions.values())


def test_parse_shuffles_the_same_way_for_the_same_seed() -> None:
    first = parse_truthfulqa(_rows(50), seed=5)
    second = parse_truthfulqa(_rows(50), seed=5)

    assert first == second


def test_parse_shuffles_differently_for_another_seed() -> None:
    first = parse_truthfulqa(_rows(50), seed=5)
    second = parse_truthfulqa(_rows(50), seed=6)

    assert [i.choices for i in first] != [i.choices for i in second]


def test_parse_defaults_to_seed_zero() -> None:
    assert parse_truthfulqa(_rows(10)) == parse_truthfulqa(_rows(10), seed=0)


def test_parse_builds_uids_and_splits_by_row() -> None:
    items = parse_truthfulqa(_rows(300), seed=0)

    assert [i.uid for i in items[:2]] == ["truthfulqa-0", "truthfulqa-1"]
    expected = assign_splits([i.uid for i in items], dataset="truthfulqa")
    assert [i.split for i in items] == expected
    assert {"train", "val", "test"} == set(expected)


def test_parse_gives_every_item_its_own_meta() -> None:
    first, second = parse_truthfulqa(_rows(2), seed=0)

    assert first.meta == {}
    assert first.meta is not second.meta


def test_parse_handles_a_single_choice() -> None:
    (item,) = parse_truthfulqa([_row("Only?", 1)], seed=0)

    assert item.correct == 0
    assert item.choices == ("Only? answer 0",)


@pytest.mark.parametrize(
    "labels",
    [[0, 0, 0], [1, 1, 0], [1, 0]],
    ids=["no_correct", "two_correct", "length_mismatch"],
)
def test_parse_rejects_malformed_mc1_targets(labels: list[int]) -> None:
    row = {
        "question": "Q?",
        "mc1_targets": {"choices": ["a", "b", "c"], "labels": labels},
    }

    with pytest.raises(ValueError, match="truthfulqa-0"):
        parse_truthfulqa([row], seed=0)


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> Path:
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


@pytest.fixture
def served(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> list[dict[str, object]]:
    path = _write_parquet(tmp_path / "validation.parquet", _rows(30))
    calls: list[dict[str, object]] = []

    def fake_fetch(source: object, **kwargs: object) -> Path:
        calls.append({"source": source, **kwargs})
        return path

    monkeypatch.setattr(truthfulqa, "fetch", fake_fetch)
    return calls


def test_load_reads_the_parquet_and_shuffles_with_the_seed(
    served: list[dict[str, object]], tmp_path: Path
) -> None:
    items = load_truthfulqa(seed=4, cache=tmp_path, offline=True)

    assert served == [{"source": SOURCE, "cache": tmp_path, "offline": True}]
    assert items == parse_truthfulqa(_rows(30), seed=4)


@pytest.mark.network
def test_load_reads_the_pinned_revision() -> None:
    items = load_truthfulqa(seed=0)

    assert len(items) == 817
    assert len({i.uid for i in items}) == 817
    assert min(len(i.choices) for i in items) >= 2
    assert max(len(i.choices) for i in items) == 13
    assert sum(i.correct != 0 for i in items) > 817 * 0.5
    assert all(i.choices[i.correct] for i in items)
