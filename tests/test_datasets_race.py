import collections
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tomography.datasets import race
from tomography.datasets.race import SOURCE
from tomography.datasets.race import RacePassage
from tomography.datasets.race import load_race
from tomography.datasets.race import parse_race
from tomography.datasets.records import ChoiceExample

FILES = {
    "train": "high/train-00000-of-00001.parquet",
    "validation": "high/validation-00000-of-00001.parquet",
    "test": "high/test-00000-of-00001.parquet",
}


def _row(
    example_id: str,
    article: str,
    question: str,
    answer: str = "B",
    options: list[str] | None = None,
) -> dict[str, object]:
    return {
        "example_id": example_id,
        "article": article,
        "question": question,
        "options": options or [f"{question} {o}" for o in "wxyz"],
        "answer": answer,
    }


TOWN = "A town.\nIt rains."
FARM = "A farm.\nIt is dry."
ROWS = [
    _row("high1.txt", TOWN, "Where?", "A"),
    _row("high2.txt", FARM, "What?", "C"),
    _row("high1.txt", TOWN, "When  _  .", "D"),
    _row("high1.txt", TOWN, "Why? ", "B", [" yes", "no ", "maybe", "never"]),
    _row("high2.txt", FARM, "Who?", "D"),
]


def test_source_pins_the_hub_revision() -> None:
    assert SOURCE.name == "race"
    assert SOURCE.kind == "hub"
    assert SOURCE.location == "ehovy/race"
    assert SOURCE.revision == "2fec9fd81f1dc971569a9b729c43f2f0e6436637"
    assert SOURCE.files == tuple(FILES.values())
    assert SOURCE.license == "non-commercial research purpose only"


def test_parse_groups_questions_by_article_in_first_appearance_order() -> None:
    town, farm = parse_race(ROWS, source_split="train")

    assert isinstance(town, RacePassage)
    assert (town.article, farm.article) == (TOWN, FARM)
    assert town.last.question == "Why?"
    assert farm.last.question == "Who?"


def test_parse_keeps_the_last_question_as_a_four_way_item() -> None:
    town, farm = parse_race(ROWS, source_split="train")

    assert isinstance(town.last, ChoiceExample)
    assert town.last.uid == "race-high1.txt"
    assert town.last.choices == ("yes", "no", "maybe", "never")
    assert town.last.correct == 1
    assert (farm.last.uid, farm.last.correct) == ("race-high2.txt", 3)


def test_parse_keeps_each_earlier_question_with_its_gold_answer() -> None:
    town, farm = parse_race(ROWS, source_split="train")

    assert town.earlier == (
        ("Where?", "Where? w"),
        ("When  _  .", "When  _  . z"),
    )
    assert farm.earlier == (("What?", "What? y"),)


@pytest.mark.parametrize(
    ("source_split", "split"),
    [("train", "train"), ("validation", "val"), ("test", "test")],
    ids=["train", "validation", "test"],
)
def test_parse_maps_the_source_split(source_split: str, split: str) -> None:
    (passage, _) = parse_race(ROWS, source_split=source_split)  # type: ignore[arg-type]

    assert passage.last.split == split


def test_parse_keeps_an_article_with_repeated_options_as_one_group() -> None:
    blank = ["_ & _"] * 4
    rows = [
        _row("high3188.txt", "First.", "One?", "B"),
        _row("high3189.txt", "Skating.", "More in  _  .", "A", blank),
        _row("high3190.txt", "Third.", "Three?", "C"),
    ]

    first, skating, third = parse_race(rows, source_split="validation")

    assert skating.last.choices == ("_ & _",) * 4
    assert skating.last.correct == 0
    stimuli = [e for p in (first, skating, third) for e in p.last.flatten()]
    assert collections.Counter(e.meta["group"] for e in stimuli) == {
        "race-high3188.txt": 4,
        "race-high3189.txt": 4,
        "race-high3190.txt": 4,
    }
    assert [e.text for e in stimuli[-4:]] == [f"Three? {o}" for o in "wxyz"]
    assert [e.label for e in stimuli[-4:]] == [0, 0, 1, 0]


def test_parse_rejects_a_question_without_four_options() -> None:
    row = _row("high9.txt", "Short.", "Huh?", "A", ["a", "b", "c"])

    with pytest.raises(ValueError, match=r"race-high9\.txt"):
        parse_race([row], source_split="train")


def test_parse_rejects_an_answer_that_is_not_a_letter_a_to_d() -> None:
    row = _row("high8.txt", "Short.", "Huh?", "E")

    with pytest.raises(ValueError, match=r"race-high8\.txt"):
        parse_race([row], source_split="train")


@pytest.fixture
def served(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> list[dict[str, object]]:
    rows = {"train": ROWS, "validation": ROWS[1:2], "test": ROWS[:1]}
    schema = pa.schema(
        [
            ("example_id", pa.string()),
            ("article", pa.string()),
            ("answer", pa.string()),
            ("question", pa.string()),
            ("options", pa.list_(pa.string())),
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

    monkeypatch.setattr(race, "fetch", fake_fetch)
    return calls


def test_load_reads_train_validation_then_test(
    served: list[dict[str, object]], tmp_path: Path
) -> None:
    passages = load_race(cache=tmp_path, offline=True)

    assert [c["file"] for c in served] == list(FILES.values())
    assert all(c["source"] is SOURCE for c in served)
    assert all(c["cache"] == tmp_path for c in served)
    assert all(c["offline"] is True for c in served)
    assert [(p.last.uid, p.last.split) for p in passages] == [
        ("race-high1.txt", "train"),
        ("race-high2.txt", "train"),
        ("race-high2.txt", "val"),
        ("race-high1.txt", "test"),
    ]


@pytest.mark.network
def test_load_reads_the_pinned_revision() -> None:
    passages = load_race()

    articles = collections.Counter(p.last.split for p in passages)
    assert articles == {"train": 18_726, "val": 1_021, "test": 1_045}
    questions = collections.Counter()
    for p in passages:
        questions[p.last.split] += 1 + len(p.earlier)
    assert questions == {"train": 62_445, "val": 3_451, "test": 3_498}
    assert len({p.last.uid for p in passages}) == len(passages)
    val = [p for p in passages if p.last.split == "val"]
    assert val[444].last.uid == "race-high3189.txt"
    assert val[444].last.choices == ("_ & _",) * 4
