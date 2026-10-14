import csv
import dataclasses
from pathlib import Path

import pytest

from tomography.datasets.records import QAExample
from tomography.knowledge import DEFAULT_SHOTS
from tomography.knowledge import ItemLabel
from tomography.knowledge import Label
from tomography.knowledge import LabelRecipe
from tomography.knowledge import extract_answer
from tomography.knowledge import few_shot_prompt
from tomography.knowledge import is_correct
from tomography.knowledge import label_from_counts
from tomography.knowledge import normalize_answer
from tomography.knowledge import write_hand_check

SHOTS = (("What is 2 plus 2?", "four"), ("Capital of France?", "Paris"))


def test_few_shot_prompt_is_exact_for_two_shots() -> None:
    prompt = few_shot_prompt("Who wrote Hamlet?", SHOTS)

    assert prompt == (
        "Q: What is 2 plus 2?\nA: four\n\n"
        "Q: Capital of France?\nA: Paris\n\n"
        "Q: Who wrote Hamlet?\nA:"
    )


def test_default_shots_are_fifteen_distinct_single_line_pairs() -> None:
    assert len(DEFAULT_SHOTS) == 15
    for question, answer in DEFAULT_SHOTS:
        assert question.strip()
        assert answer.strip()
        assert "\n" not in question
        assert "\n" not in answer
    assert len({question for question, _ in DEFAULT_SHOTS}) == 15


@pytest.mark.parametrize(
    ("completion", "stop", "expected"),
    [
        (" Paris\nQ: next", "\n", "Paris"),
        (" Paris  ", "\n", "Paris"),
        ("\n Paris", "\n", ""),
        ("", "\n", ""),
        (" one. two. three", ".", "one"),
        (" Paris\nQ: next", "", "Paris\nQ: next"),
    ],
    ids=[
        "cut_at_newline",
        "strip_spaces",
        "leading_stop",
        "empty",
        "custom_stop",
        "empty_stop_keeps_all",
    ],
)
def test_extract_answer(completion: str, stop: str, expected: str) -> None:
    assert extract_answer(completion, stop=stop) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Café Münster", "cafe munster"),
        ("The Lord of the Rings", "lord of rings"),
        ("An apple, a pear!", "apple pear"),
        ("N.Y.C.", "n y c"),
        ("  Hello---World  ", "hello world"),
        ("Zoë's 3rd", "zoe s 3rd"),
        ("the", ""),
        ("", ""),
        ("?!", ""),
    ],
    ids=[
        "accents",
        "articles",
        "article_and_punctuation",
        "dotted_letters",
        "punctuation_runs",
        "apostrophe_and_digits",
        "only_article",
        "empty",
        "only_punctuation",
    ],
)
def test_normalize_answer(text: str, expected: str) -> None:
    assert normalize_answer(text) == expected


@pytest.mark.parametrize(
    ("answer", "aliases", "expected"),
    [
        ("Paris", ("paris",), True),
        ("It is the city of Paris, France", ("Paris",), True),
        ("party", ("art",), False),
        ("the art of war", ("Art",), True),
        ("New York City", ("new york",), True),
        ("York New", ("new york",), False),
        ("Café", ("cafe",), True),
        ("anything", ("", "!!"), False),
        ("anything", (), False),
        ("", ("Paris",), False),
        ("Lyon", ("", "Lyon"), True),
    ],
    ids=[
        "exact",
        "inside_sentence",
        "substring_is_not_a_word",
        "whole_word",
        "multiword_sequence",
        "word_order_matters",
        "accent_insensitive",
        "empty_aliases_ignored",
        "no_aliases",
        "empty_answer",
        "empty_alias_does_not_block_others",
    ],
)
def test_is_correct(
    answer: str, aliases: tuple[str, ...], expected: bool
) -> None:
    assert is_correct(answer, aliases) is expected


@pytest.mark.parametrize(
    ("greedy", "n_correct", "expected"),
    [
        *[(True, n, "uncertain") for n in range(4)],
        (True, 4, "known"),
        (True, 5, "known"),
        (False, 0, "unknown"),
        *[(False, n, "uncertain") for n in range(1, 6)],
    ],
    ids=[
        *[f"greedy_right_{n}_of_5" for n in range(6)],
        *[f"greedy_wrong_{n}_of_5" for n in range(6)],
    ],
)
def test_label_from_counts_over_every_case(
    greedy: bool, n_correct: int, expected: str
) -> None:
    assert label_from_counts(greedy, n_correct, n_samples=5) == expected


def test_label_from_counts_threshold_is_a_keyword() -> None:
    assert label_from_counts(True, 3, n_samples=5, known_min_correct=3) == (
        "known"
    )
    assert label_from_counts(True, 4, n_samples=5, known_min_correct=5) == (
        "uncertain"
    )


@pytest.mark.parametrize(
    ("n_correct", "n_samples", "known_min_correct"),
    [(-1, 5, 4), (6, 5, 4), (0, 5, 0), (0, 5, 6)],
    ids=["negative_count", "count_above_samples", "min_zero", "min_above"],
)
def test_label_from_counts_rejects_impossible_arguments(
    n_correct: int, n_samples: int, known_min_correct: int
) -> None:
    with pytest.raises(ValueError, match="must be in"):
        label_from_counts(
            True,
            n_correct,
            n_samples=n_samples,
            known_min_correct=known_min_correct,
        )


def test_label_recipe_defaults_match_the_frozen_rule() -> None:
    recipe = LabelRecipe()

    assert (recipe.n_samples, recipe.temperature) == (5, 1.0)
    assert (recipe.known_min_correct, recipe.max_new_tokens) == (4, 16)
    assert recipe.stop == "\n"


def test_records_are_frozen() -> None:
    item = ItemLabel("q1", "known", "behavior", "Paris", True, ("Paris",), 1)

    with pytest.raises(dataclasses.FrozenInstanceError):
        item.label = "unknown"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        LabelRecipe().n_samples = 3  # type: ignore[misc]


def _items(
    labels: tuple[Label, ...],
) -> tuple[list[QAExample], list[ItemLabel]]:
    examples = [
        QAExample(f"q{i}", f"Question {i}?", (f"a{i}", f"b{i}"), "real", "test")
        for i in range(len(labels))
    ]
    item_labels = [
        ItemLabel(f"q{i}", label, "behavior", f"g{i}", False, ("x", "y"), 0)
        for i, label in enumerate(labels)
    ]
    return examples, item_labels


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_write_hand_check_columns_and_cells(tmp_path: Path) -> None:
    examples, labels = _items(("known", "unknown"))
    path = tmp_path / "check.csv"

    written = write_hand_check(path, examples, labels, n=10)

    rows = _read(path)
    assert written == 2
    assert list(rows[0]) == [
        "uid",
        "kind",
        "question",
        "gold",
        "greedy",
        "samples",
        "label",
        "human_label",
    ]
    assert rows[0] == {
        "uid": "q0",
        "kind": "real",
        "question": "Question 0?",
        "gold": "a0 | b0",
        "greedy": "g0",
        "samples": "x | y",
        "label": "known",
        "human_label": "",
    }


def test_write_hand_check_round_robin_over_labels(tmp_path: Path) -> None:
    labels: tuple[Label, ...] = (
        ("known",) * 6 + ("unknown",) * 6 + ("uncertain",) * 2
    )
    examples, item_labels = _items(labels)
    path = tmp_path / "check.csv"

    written = write_hand_check(path, examples, item_labels, n=7)

    counts = {
        label: sum(row["label"] == label for row in _read(path))
        for label in ("known", "unknown", "uncertain")
    }
    assert written == 7
    assert counts == {"known": 3, "unknown": 2, "uncertain": 2}


def test_write_hand_check_is_deterministic_per_seed(tmp_path: Path) -> None:
    examples, labels = _items(("known", "unknown") * 10)

    write_hand_check(tmp_path / "a.csv", examples, labels, n=6, seed=1)
    write_hand_check(tmp_path / "b.csv", examples, labels, n=6, seed=1)
    write_hand_check(tmp_path / "c.csv", examples, labels, n=6, seed=2)

    a = (tmp_path / "a.csv").read_text(encoding="utf-8")
    assert a == (tmp_path / "b.csv").read_text(encoding="utf-8")
    assert a != (tmp_path / "c.csv").read_text(encoding="utf-8")


def test_write_hand_check_rows_keep_the_order_of_examples(
    tmp_path: Path,
) -> None:
    examples, labels = _items(("known", "unknown", "uncertain") * 5)
    path = tmp_path / "check.csv"

    write_hand_check(path, examples, labels, n=9)

    uids = [int(row["uid"][1:]) for row in _read(path)]
    assert uids == sorted(uids)


def test_write_hand_check_n_above_item_count_returns_all(
    tmp_path: Path,
) -> None:
    examples, labels = _items(("known", "known", "unknown"))
    path = tmp_path / "check.csv"

    assert write_hand_check(path, examples, labels, n=200) == 3
    assert [row["uid"] for row in _read(path)] == ["q0", "q1", "q2"]


def test_write_hand_check_missing_label_gives_its_quota_away(
    tmp_path: Path,
) -> None:
    examples, labels = _items(("known",) * 5 + ("unknown",) * 2)
    path = tmp_path / "check.csv"

    written = write_hand_check(path, examples, labels, n=6)

    unknown = [row for row in _read(path) if row["label"] == "unknown"]
    assert written == 6
    assert len(unknown) == 2


def test_write_hand_check_rejects_unpaired_labels(tmp_path: Path) -> None:
    examples, labels = _items(("known", "unknown"))

    with pytest.raises(ValueError, match="pair up"):
        write_hand_check(tmp_path / "a.csv", examples, labels[:1])
    with pytest.raises(ValueError, match="pair up"):
        write_hand_check(tmp_path / "b.csv", examples, labels[::-1])
