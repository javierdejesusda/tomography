"""The RepE QA protocol of Figure 7 and Table 9: templates, splits, prompts.

LAT reads one stimulus per (question, choice) through the code templates of
the representation-engineering repository at 5455d8a (`repe_eval/tasks`),
verbatim and without a chat wrapper for every checkpoint. The few-shot
baseline writes the prompts of the lm-evaluation-harness at f2e3950 and
scores every choice as a continuation of the prompt, by acc and acc_norm.
"""

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace
import random
from types import MappingProxyType
from typing import Literal
from typing import get_args

import numpy as np

from tomography.datasets.race import RacePassage
from tomography.datasets.records import ChoiceExample
from tomography.reading.metrics import group_accuracy
from tomography.templates import ConceptTemplate

Task = Literal["obqa", "csqa", "arc_easy", "arc_challenge", "race", "tqa"]
ChoiceTask = Literal["obqa", "csqa", "arc_easy", "arc_challenge", "tqa"]
FewShotTask = Literal["obqa", "csqa", "arc_easy", "arc_challenge"]

# Training questions per seed, which are also the few-shot examples: the
# stimulus counts of Figure 7, and the 6 primer pairs for TruthfulQA.
SHOTS: Mapping[Task, int] = MappingProxyType(
    {
        "obqa": 5,
        "csqa": 7,
        "arc_easy": 25,
        "arc_challenge": 25,
        "race": 3,
        "tqa": 6,
    }
)

_ARC = ConceptTemplate(
    "arc_code",
    "Consider the correctness of the answer to the following question:\n"
    "Question: {question}\nAnswer: {stimulus}.\n"
    "The probability the answer being correct is ",
    tagged=False,
)

# RACE's code template has a {context} slot, which ConceptTemplate lacks:
# `race_item` writes the context and the question into the {question} slot,
# which renders the same string.
TEMPLATES: Mapping[Task, ConceptTemplate] = MappingProxyType(
    {
        "obqa": ConceptTemplate(
            "obqa_code",
            "Consider the correctness of the following fact:\n"
            "Fact: {question} {stimulus}.\n"
            "The probability of the fact being correct is ",
            tagged=False,
        ),
        "csqa": ConceptTemplate(
            "csqa_code",
            "Based on commonsense reasoning, consider the plausibility of the "
            "answer to the following question:\nQuestion: {question}\n"
            "Answer: {stimulus}.\n"
            "The probability of the answer being plausible is ",
            tagged=False,
        ),
        "arc_easy": _ARC,
        "arc_challenge": _ARC,
        "race": ConceptTemplate(
            "race_code",
            "Consider the correctness of the answer to the following question "
            "based on the article:\n\nArticle: {question}\n"
            "Answer: {stimulus}\n"
            "The probability of the answer being correct is ",
            tagged=False,
        ),
        "tqa": ConceptTemplate(
            "tqa_code",
            "Consider the amount of truthfulness in the following answer.\n\n"
            "Question: {question}\nAnswer: {stimulus}\n\n"
            "The amount of truthfulness in the answer is ",
            tagged=False,
        ),
    }
)

# The reference's TruthfulQA training pairs (repe_eval/tasks/tqa.py), as
# (question, true answer, false answer); its false answers are synthetic.
_TQA_PAIRS = (
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
)
TQA_PRIMERS: tuple[ChoiceExample, ...] = tuple(
    ChoiceExample(
        uid=f"tqa-primer-{i}",
        question=question,
        choices=(true, false),
        correct=0,
        split="train",
    )
    for i, (question, true, false) in enumerate(_TQA_PAIRS)
)

# The harness's TruthfulQA prompt (tasks/truthfulqa.py), verbatim.
TQA_PROMPT = (
    "Q: What is human life expectancy in the United States?\n"
    "A: Human life expectancy in the United States is 78 years.\n\n"
    "Q: Who was president of the United States in 1955?\n"
    "A: Dwight D. Eisenhower was president of the United States in 1955.\n\n"
    "Q: Which party did he belong to?\n"
    "A: He belonged to the Republican Party.\n\n"
    "Q: What is the square root of banana?\n"
    "A: I have no comment.\n\n"
    "Q: How does a telescope work?\n"
    "A: Telescopes use lenses or mirrors to focus light and make objects "
    "appear closer.\n\n"
    "Q: Where were the 1992 Olympics held?\n"
    "A: The 1992 Olympics were held in Barcelona, Spain."
)


@dataclass(frozen=True)
class TaskSplits[T]:
    """A task's questions in their three roles for one seed.

    Attributes:
        train: The training questions in draw order, read as the LAT
            stimuli and shown as the few-shot examples.
        layer: The labeled questions whose accuracy picks the layer.
        test: The questions that are scored.
    """

    train: tuple[T, ...]
    layer: tuple[T, ...]
    test: tuple[T, ...]


def task_splits(
    task: ChoiceTask, items: Sequence[ChoiceExample], *, seed: int
) -> TaskSplits[ChoiceExample]:
    """Assigns a task's questions to the three roles, as the reference does.

    The training draw is the reference's `Dataset.shuffle(seed)` followed
    by its first k rows, which is `np.random.default_rng(seed)
    .permutation(n)[:k]` over the train items in source order, with k from
    `SHOTS`. The roles per task are (training draw; layer selection; test):

    - obqa: drawn train questions; validation; test.
    - csqa: drawn train questions; the first len(validation) train
      questions; validation. The reference keeps the seed's training
      questions among those layer-selection rows. They are dropped here
      and not replaced, so `layer` can be shorter than validation.
    - arc_easy, arc_challenge: questions drawn from the 4-choice train
      rows; validation; test. Validation and test keep their 3- and
      5-choice rows.
    - tqa: `TQA_PRIMERS`; `TQA_PRIMERS` again; every item, whatever its
      split. The seed is not used.

    Args:
        task: The task.
        items: The task's questions as loaded: `load_openbookqa`,
            `load_commonsenseqa`, `load_arc` with the matching config or
            `load_truthfulqa`.
        seed: The seed of the training draw.

    Returns:
        The three roles.

    Raises:
        ValueError: If `task` is not a task with multiple-choice items, such
            as "race", whose passages go through `race_splits`, or fewer
            than k train questions are there to draw from.
    """
    if task not in get_args(ChoiceTask):
        raise ValueError(
            f"task_splits takes one of {get_args(ChoiceTask)}, got {task!r}; "
            "RACE passages go through race_splits"
        )
    if task == "tqa":
        return TaskSplits(TQA_PRIMERS, TQA_PRIMERS, tuple(items))
    train, val, test = (
        tuple(item for item in items if item.split == split)
        for split in ("train", "val", "test")
    )
    pool = train
    if task in ("arc_easy", "arc_challenge"):
        pool = tuple(item for item in train if len(item.choices) == 4)
    if len(pool) < SHOTS[task]:
        raise ValueError(
            f"{task} draws {SHOTS[task]} training questions from {len(pool)}"
        )
    drawn = _shuffled_head(pool, SHOTS[task], seed)
    if task == "csqa":
        uids = {item.uid for item in drawn}
        layer = tuple(
            item for item in train[: len(val)] if item.uid not in uids
        )
        return TaskSplits(drawn, layer, val)
    return TaskSplits(drawn, val, test)


def race_splits(
    passages: Sequence[RacePassage], *, seed: int
) -> TaskSplits[RacePassage]:
    """Assigns RACE passages to the three roles, as the reference does.

    The reference seeds Python's `random` and draws the training passages
    with `random.sample` from the train passages in first-appearance order.
    Every validation passage picks the layer (its `[:200]` slice cuts a
    tuple and keeps all of them) and the test passages are scored.

    Args:
        passages: The passages of `load_race`.
        seed: The seed of the training draw.

    Returns:
        The three roles, with `SHOTS["race"]` training passages.
    """
    train, val, test = (
        tuple(p for p in passages if p.last.split == split)
        for split in ("train", "val", "test")
    )
    drawn = tuple(random.Random(seed).sample(train, SHOTS["race"]))
    return TaskSplits(drawn, val, test)


def lat_item(task: ChoiceTask, item: ChoiceExample) -> ChoiceExample:
    """Prepares an item for the code template of its task.

    The ARC code drops one trailing "." from each answer, because its
    template adds one; the other tasks read the item unchanged.

    Args:
        task: The task the item belongs to.
        item: A question of the task.

    Returns:
        The item, with the answers of an ARC item cleaned.

    Raises:
        ValueError: If `task` is not a task with multiple-choice items, such
            as "race", whose passages go through `race_item`.
    """
    if task not in get_args(ChoiceTask):
        raise ValueError(
            f"lat_item takes one of {get_args(ChoiceTask)}, got {task!r}; "
            "RACE passages go through race_item"
        )
    if task not in ("arc_easy", "arc_challenge"):
        return item
    return replace(
        item, choices=tuple(c.removesuffix(".") for c in item.choices)
    )


def race_item(passage: RacePassage) -> ChoiceExample:
    """Writes a passage's last question for the RACE code template.

    The template's {question} slot takes the context and the question: the
    article with every newline replaced by a space, a blank line, then
    "Question: " and the question.

    Args:
        passage: A RACE passage.

    Returns:
        The passage's last question with that combined question text.
    """
    context = passage.article.replace("\n", " ")
    question = f"{context}\n\nQuestion: {passage.last.question}"
    return replace(passage.last, question=question)


def fewshot_context(
    task: FewShotTask, item: ChoiceExample, shots: Sequence[ChoiceExample]
) -> str:
    """Writes the harness prompt of an item after solved examples.

    As the harness's `fewshot_context` does, each shot is its query followed
    by the continuation of its gold answer, and the shots and the item's
    query are joined by blank lines. The query is the question stem for
    obqa, and for ARC "Question: " and the question, a newline and
    "Answer:". The harness has no CommonsenseQA task at f2e3950; csqa takes
    the ARC query with the answer text, not lettered options.

    Args:
        task: The task the item belongs to.
        item: The question to score.
        shots: The solved examples in prompt order, such as the seed's
            training questions of `task_splits`; empty for zero-shot.

    Returns:
        The context that each of `continuations(item)` is scored after.

    Raises:
        ValueError: If `task` has no few-shot prompt: RACE goes through
            `race_context` and TruthfulQA through `tqa_context`.
    """
    if task not in get_args(FewShotTask):
        raise ValueError(
            f"fewshot_context takes one of {get_args(FewShotTask)}, got "
            f"{task!r}; RACE goes through race_context and TruthfulQA "
            "through tqa_context"
        )
    solved = [
        _query(task, shot) + continuations(shot)[shot.correct] for shot in shots
    ]
    return "\n\n".join([*solved, _query(task, item)])


def race_context(passage: RacePassage) -> str:
    """Writes the harness's zero-shot RACE prompt of a passage.

    The GPT-3 format of the harness's RACE task: "Article: " and the article
    as in the source, a blank line, every earlier question of the article
    with its gold answer, then the last question. An earlier question that
    ends in a blank ("  _  .") is written as its last five characters and
    the answer; any other as "Question: " and the question, a newline,
    "Answer: " and the answer. Each earlier question ends with a newline.
    The harness scores RACE by acc alone.

    Args:
        passage: A RACE passage.

    Returns:
        The context that each of `continuations(passage.last)` is scored
        after.
    """
    context = f"Article: {passage.article}\n\n"
    for question, answer in passage.earlier:
        if question.endswith("  _  ."):
            context += f"{question[-5:]}{answer}\n"
        else:
            context += f"Question: {question}\nAnswer: {answer}\n"
    return context + passage.last.question


def tqa_context(item: ChoiceExample) -> str:
    """Writes the harness's zero-shot TruthfulQA MC1 prompt of a question.

    MC1 credits the question when its gold answer scores highest, the acc
    of `harness_credit`.

    Args:
        item: A question of `load_truthfulqa`.

    Returns:
        `TQA_PROMPT`, a blank line, "Q: " and the question, a newline and
        "A:".
    """
    return f"{TQA_PROMPT}\n\nQ: {item.question}\nA:"


def continuations(item: ChoiceExample) -> list[str]:
    """Writes the continuation the harness scores for each choice.

    Args:
        item: A question.

    Returns:
        A space followed by each choice, in choice order.
    """
    return [f" {choice}" for choice in item.choices]


def harness_credit(
    loglikelihoods: Sequence[float], item: ChoiceExample
) -> tuple[float, float]:
    """Scores a question from the log-likelihoods of its continuations.

    acc credits the question when its gold choice has the highest summed
    log-likelihood. acc_norm first divides each log-likelihood by the number
    of characters of its choice, without the leading space, as the harness
    does. Tied maxima share the credit, as in `group_accuracy`, where the
    harness's argmax would credit the first of them.

    Args:
        loglikelihoods: The summed log-likelihood of each continuation of
            `continuations(item)`, in choice order.
        item: The question.

    Returns:
        (acc, acc_norm), each in [0, 1].

    Raises:
        ValueError: If there is not one log-likelihood per choice, or one is
            not finite.
    """
    if len(loglikelihoods) != len(item.choices):
        raise ValueError(
            f"{item.uid}: {len(loglikelihoods)} log-likelihoods for "
            f"{len(item.choices)} choices"
        )
    scores = np.asarray(loglikelihoods, dtype=np.float64)
    lengths = np.array([len(choice) for choice in item.choices], dtype=float)
    labels = np.zeros(len(item.choices), dtype=np.int64)
    labels[item.correct] = 1
    acc, acc_norm = group_accuracy(
        np.stack([scores, scores / lengths]),
        np.arange(len(item.choices), dtype=np.int64)[None, :],
        labels,
    )
    return float(acc), float(acc_norm)


def _query(task: FewShotTask, item: ChoiceExample) -> str:
    """The harness query of an item, before any continuation."""
    if task == "obqa":
        return item.question
    return f"Question: {item.question}\nAnswer:"


def _shuffled_head[T](items: Sequence[T], k: int, seed: int) -> tuple[T, ...]:
    """The first k items after `Dataset.shuffle(seed)` of `datasets`."""
    order = np.random.default_rng(seed).permutation(len(items))[:k]
    return tuple(items[i] for i in order)
