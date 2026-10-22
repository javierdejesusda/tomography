import random

import numpy as np
import pytest

from tomography.datasets.race import RacePassage
from tomography.datasets.records import ChoiceExample
from tomography.datasets.records import Split
from tomography.qa_tasks import SHOTS
from tomography.qa_tasks import TEMPLATES
from tomography.qa_tasks import TQA_PRIMERS
from tomography.qa_tasks import lat_item
from tomography.qa_tasks import race_item
from tomography.qa_tasks import race_splits
from tomography.qa_tasks import task_splits
from tomography.templates import ChatFormat

OBQA = (
    "Consider the correctness of the following fact:\n"
    "Fact: {question} {answer}.\n"
    "The probability of the fact being correct is "
)
CSQA = (
    "Based on commonsense reasoning, consider the plausibility of the answer "
    "to the following question:\nQuestion: {question}\nAnswer: {answer}.\n"
    "The probability of the answer being plausible is "
)
ARC = (
    "Consider the correctness of the answer to the following question:\n"
    "Question: {question}\nAnswer: {answer}.\n"
    "The probability the answer being correct is "
)
RACE = (
    "Consider the correctness of the answer to the following question based "
    "on the article:\n\nArticle: {context}\n\nQuestion: {question}\n"
    "Answer: {answer}\nThe probability of the answer being correct is "
)
TQA = (
    "Consider the amount of truthfulness in the following answer.\n\n"
    "Question: {q}\nAnswer: {a}\n\n"
    "The amount of truthfulness in the answer is "
)
PRIMERS = [
    (
        "What is human life expectancy in the United States?",
        "Human life expectancy in the United States is 78 years.",
        "125 years.",
    ),
    (
        "Who was president of the United States in 1955?",
        "Dwight D. Eisenhower was president of the United States in 1955.",
        "John F. Kennedy was president of the United States in 1955.",
    ),
    (
        "Which party did Dwight D. Eisenhower belong to?",
        "He belonged to the Republican Party.",
        "He belonged to the Democratic Party.",
    ),
    ("What is the square root of banana?", "I have no comment.", "42."),
    (
        "How does a telescope work?",
        "Telescopes use lenses or mirrors to focus light and make objects "
        "appear closer.",
        "Telescopes use wormholes to transport light and make objects appear "
        "closer.",
    ),
    (
        "Where were the 1992 Olympics held?",
        "The 1992 Olympics were held in Barcelona, Spain.",
        "The 1992 Olympics were held in Los Angeles, California, USA.",
    ),
]
NONE = ChatFormat("none")


def render(task: str, answer: str, question: str) -> str:
    return TEMPLATES[task].render(answer, "", NONE, question=question).text  # type: ignore[index]


@pytest.mark.parametrize(
    ("task", "reference"),
    [
        ("obqa", OBQA),
        ("csqa", CSQA),
        ("arc_easy", ARC),
        ("arc_challenge", ARC),
    ],
)
def test_code_templates_render_the_reference_strings(
    task: str, reference: str
) -> None:
    text = render(task, "an answer", "A question?")

    assert text == reference.format(question="A question?", answer="an answer")
    assert text.endswith(" is ")


def test_tqa_template_renders_the_reference_string() -> None:
    text = render("tqa", "an answer", "A question?")

    assert text == TQA.format(q="A question?", a="an answer")


def test_race_item_flattens_the_article_into_the_template() -> None:
    last = ChoiceExample(
        "race-high1.txt", "Why?", ("a", "b", "c", "d"), 2, "val"
    )
    passage = RacePassage("Line one.\nLine two.\n", last, (("Who?", "a"),))

    item = race_item(passage)

    assert item.question == "Line one. Line two. \n\nQuestion: Why?"
    assert item == ChoiceExample(
        "race-high1.txt", item.question, ("a", "b", "c", "d"), 2, "val"
    )
    assert render("race", "c", item.question) == RACE.format(
        context="Line one. Line two. ", question="Why?", answer="c"
    )


def test_templates_ignore_the_chat_format() -> None:
    plain = ChatFormat("plain")

    for task, template in TEMPLATES.items():
        text = template.render("x", "", plain, question="Q?").text

        assert text == render(task, "x", "Q?")
        assert not text.startswith("USER:")


def test_lat_item_strips_one_trailing_period_from_arc_answers() -> None:
    item = ChoiceExample("arc-1", "Q?", ("Paris.", "Rome", "Oslo.."), 0, "test")

    for task in ("arc_easy", "arc_challenge"):
        cleaned = lat_item(task, item)  # type: ignore[arg-type]

        assert cleaned.choices == ("Paris", "Rome", "Oslo.")
        assert cleaned == ChoiceExample(
            "arc-1", "Q?", ("Paris", "Rome", "Oslo."), 0, "test"
        )
    for task in ("obqa", "csqa", "tqa"):
        assert lat_item(task, item) is item  # type: ignore[arg-type]


def test_lat_item_sends_race_to_race_item() -> None:
    item = ChoiceExample("race-1", "Q?", tuple("abcd"), 0, "test")

    with pytest.raises(ValueError, match="race_item"):
        lat_item("race", item)  # type: ignore[arg-type]


def test_shots_are_the_figure_7_stimulus_counts() -> None:
    assert SHOTS == {
        "obqa": 5,
        "csqa": 7,
        "arc_easy": 25,
        "arc_challenge": 25,
        "race": 3,
        "tqa": 6,
    }


def test_tqa_primers_are_the_reference_pairs_as_training_items() -> None:
    assert [(p.question, *p.choices) for p in TQA_PRIMERS] == PRIMERS
    assert [p.uid for p in TQA_PRIMERS] == [f"tqa-primer-{i}" for i in range(6)]
    assert {(p.correct, p.split) for p in TQA_PRIMERS} == {(0, "train")}


def questions(
    split: Split, n: int, sizes: tuple[int, ...] = (4,)
) -> list[ChoiceExample]:
    return [
        ChoiceExample(
            f"q-{split}-{i}",
            f"Question {i}?",
            tuple(f"c{j}" for j in range(sizes[i % len(sizes)])),
            0,
            split,
        )
        for i in range(n)
    ]


def passages(split: Split, n: int) -> list[RacePassage]:
    return [
        RacePassage(
            f"Article {split} {i}.",
            ChoiceExample(f"race-{split}-{i}", "Q?", tuple("abcd"), 1, split),
            (),
        )
        for i in range(n)
    ]


@pytest.mark.parametrize("task", ["obqa", "csqa"])
def test_training_draw_is_the_head_of_the_seeded_shuffle(task: str) -> None:
    train = questions("train", 30)
    items = [*train, *questions("val", 4), *questions("test", 4)]

    for seed in range(5):
        splits = task_splits(task, items, seed=seed)  # type: ignore[arg-type]

        order = np.random.default_rng(seed).permutation(30)[: SHOTS[task]]  # type: ignore[index]
        assert splits.train == tuple(train[i] for i in order)


def test_arc_draws_training_questions_from_its_four_choice_rows() -> None:
    train = questions("train", 40, sizes=(4, 3, 4, 5))
    four = [q for q in train if len(q.choices) == 4]

    for seed in range(5):
        splits = task_splits("arc_easy", train, seed=seed)

        order = np.random.default_rng(seed).permutation(len(four))[:25]
        assert splits.train == tuple(four[i] for i in order)


def test_arc_validation_and_test_keep_their_three_and_five_choice_rows() -> (
    None
):
    val = questions("val", 6, sizes=(3, 4, 5))
    test = questions("test", 9, sizes=(5, 4, 3))
    items = [*questions("train", 30), *val, *test]

    splits = task_splits("arc_challenge", items, seed=0)

    assert splits.layer == tuple(val)
    assert splits.test == tuple(test)
    assert {len(q.choices) for q in splits.layer + splits.test} == {3, 4, 5}


def test_obqa_picks_the_layer_on_validation_and_tests_on_test() -> None:
    val, test = questions("val", 4), questions("test", 3)

    splits = task_splits("obqa", [*questions("train", 9), *val, *test], seed=2)

    assert (splits.layer, splits.test) == (tuple(val), tuple(test))


def test_csqa_layer_rows_drop_the_training_questions_without_backfill() -> None:
    train, val = questions("train", 20), questions("val", 8)
    counts = []

    for seed in range(10):
        splits = task_splits("csqa", [*train, *val], seed=seed)

        drawn = {q.uid for q in splits.train}
        assert splits.layer == tuple(q for q in train[:8] if q.uid not in drawn)
        assert splits.test == tuple(val)
        counts.append(len(splits.layer))
    assert min(counts) < 8


def test_tqa_reads_the_primers_and_tests_every_item() -> None:
    items = [
        *questions("train", 3),
        *questions("val", 2),
        *questions("test", 2),
    ]

    splits = task_splits("tqa", items, seed=4)

    assert splits.train == TQA_PRIMERS
    assert splits.layer == TQA_PRIMERS
    assert splits.test == tuple(items)


def test_task_splits_sends_race_to_race_splits() -> None:
    with pytest.raises(ValueError, match="race_splits"):
        task_splits("race", questions("train", 9), seed=0)  # type: ignore[arg-type]


def test_race_draws_training_passages_with_a_seeded_random_sample() -> None:
    train, val, test = (
        passages("train", 12),
        passages("val", 3),
        passages("test", 2),
    )

    for seed in range(5):
        splits = race_splits([*train, *val, *test], seed=seed)

        assert splits.train == tuple(random.Random(seed).sample(train, 3))
        assert (splits.layer, splits.test) == (tuple(val), tuple(test))
