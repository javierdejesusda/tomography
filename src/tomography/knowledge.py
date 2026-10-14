"""Behavioral known/unknown labels for closed-book questions."""

from collections.abc import Sequence
import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal
import unicodedata

import numpy as np

from tomography.control.generate import Completion
from tomography.control.generate import generate
from tomography.datasets.records import QAExample
from tomography.model import LoadedModel

Label = Literal["known", "unknown", "uncertain"]
Basis = Literal["behavior", "construction"]

_ARTICLES = frozenset(("a", "an", "the"))
_NON_ALNUM = re.compile(r"[^0-9a-z]+")

DEFAULT_SHOTS: tuple[tuple[str, str], ...] = (
    ("How many days are in a week?", "seven"),
    ("What is the chemical symbol for gold?", "Au"),
    ("Which planet is known as the Red Planet?", "Mars"),
    ("What is the largest ocean on Earth?", "the Pacific Ocean"),
    ("In which country is the city of Kyoto?", "Japan"),
    ("What is the boiling point of water in degrees Celsius?", "100"),
    ("Which language is spoken in Brazil?", "Portuguese"),
    ("How many sides does a hexagon have?", "six"),
    ("What gas do plants absorb from the air?", "carbon dioxide"),
    ("What is the square root of 81?", "nine"),
    ("In which year did World War II end?", "1945"),
    ("What is the capital of Canada?", "Ottawa"),
    ("How many centimeters are in a meter?", "100"),
    ("What is the hardest natural substance?", "diamond"),
    ("Which continent is Egypt in?", "Africa"),
)


@dataclass(frozen=True)
class LabelRecipe:
    """The knobs of the label rule, revisited after the label pilot.

    Attributes:
        n_samples: Temperature samples drawn per question.
        temperature: Sampling temperature of those samples.
        known_min_correct: Samples that must be correct for "known".
        max_new_tokens: Generation budget per answer.
        stop: Marker that ends an answer.
    """

    n_samples: int = 5
    temperature: float = 1.0
    known_min_correct: int = 4
    max_new_tokens: int = 16
    stop: str = "\n"


@dataclass(frozen=True)
class ItemLabel:
    """The behavioral label of one question and its evidence.

    Attributes:
        uid: The uid of the labeled `QAExample`.
        label: "known", "unknown" or "uncertain".
        basis: "behavior" for generated evidence, "construction" for an
            item without gold answers, unknown by construction.
        greedy: The extracted greedy answer ("" for basis "construction").
        greedy_correct: Whether the greedy answer matched a gold alias.
        samples: The extracted sampled answers.
        n_correct: How many of `samples` match a gold alias.
    """

    uid: str
    label: Label
    basis: Basis
    greedy: str
    greedy_correct: bool
    samples: tuple[str, ...]
    n_correct: int


_DEFAULT_RECIPE = LabelRecipe()


def few_shot_prompt(
    question: str, shots: Sequence[tuple[str, str]] = DEFAULT_SHOTS
) -> str:
    """Builds a few-shot closed-book prompt.

    Args:
        question: The question to ask after the demonstrations.
        shots: (question, answer) demonstrations, in prompt order.

    Returns:
        One "Q: ..." line and one "A: ..." line per shot, each shot followed
        by a blank line, then the question and a bare "A:" with no trailing
        space, so the model emits the leading space token.
    """
    demos = "".join(f"Q: {q}\nA: {a}\n\n" for q, a in shots)
    return f"{demos}Q: {question}\nA:"


def extract_answer(completion: str, *, stop: str = "\n") -> str:
    """Reduces a completion to the answer text.

    Args:
        completion: The generated continuation of a few-shot prompt.
        stop: Marker that ends the answer; empty keeps the whole completion.

    Returns:
        The completion up to the first `stop`, stripped.
    """
    head = completion.split(stop, 1)[0] if stop else completion
    return head.strip()


def normalize_answer(text: str) -> str:
    """Normalizes an answer for matching.

    Applies Unicode NFKD and drops combining marks, lower-cases, turns every
    run of non-alphanumeric characters into one space and removes the words
    "a", "an" and "the".

    Args:
        text: An answer or a gold alias.

    Returns:
        The normalized text, with single spaces between words.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    base = "".join(c for c in decomposed if not unicodedata.combining(c))
    words = _NON_ALNUM.sub(" ", base.lower()).split()
    return " ".join(w for w in words if w not in _ARTICLES)


def is_correct(answer: str, aliases: Sequence[str]) -> bool:
    """Checks whether an answer contains a gold alias as whole words.

    Args:
        answer: The extracted model answer.
        aliases: Gold aliases; those that normalize to nothing are ignored.

    Returns:
        True if some alias occurs in the normalized answer as a whole-word
        sequence ("art" is not in "party").
    """
    padded = f" {normalize_answer(answer)} "
    for alias in aliases:
        normalized = normalize_answer(alias)
        if normalized and f" {normalized} " in padded:
            return True
    return False


def label_from_counts(
    greedy_correct: bool,
    n_correct: int,
    *,
    n_samples: int,
    known_min_correct: int = 4,
) -> Label:
    """Applies the known/unknown/uncertain rule.

    Args:
        greedy_correct: Whether the greedy answer matched a gold alias.
        n_correct: How many sampled answers matched a gold alias.
        n_samples: Number of sampled answers.
        known_min_correct: Samples that must be correct for "known".

    Returns:
        "known" if the greedy answer is correct and `n_correct` is at least
        `known_min_correct`; "unknown" if the greedy answer is wrong and
        `n_correct` is 0; otherwise "uncertain".

    Raises:
        ValueError: If `n_correct` is outside 0..n_samples or
            `known_min_correct` is outside 1..n_samples.
    """
    if not 0 <= n_correct <= n_samples:
        raise ValueError(
            f"n_correct must be in 0..{n_samples}, got {n_correct}"
        )
    if not 1 <= known_min_correct <= n_samples:
        raise ValueError(
            f"known_min_correct must be in 1..{n_samples}, "
            f"got {known_min_correct}"
        )
    if greedy_correct and n_correct >= known_min_correct:
        return "known"
    if not greedy_correct and n_correct == 0:
        return "unknown"
    return "uncertain"


def label_questions(
    loaded: LoadedModel,
    examples: Sequence[QAExample],
    *,
    recipe: LabelRecipe = _DEFAULT_RECIPE,
    shots: Sequence[tuple[str, str]] = DEFAULT_SHOTS,
    seed: int = 0,
    batch_size: int = 8,
) -> list[ItemLabel]:
    """Labels questions by what one model does on them, closed book.

    Makes two `generate` calls over the prompts of all items with gold
    answers: one greedy, then one sampling `recipe.n_samples` completions at
    `recipe.temperature`. Items without gold answers are "unknown" by
    construction and cost no generation. Labels are per model: call this
    once per `LoadedModel`.

    Args:
        loaded: The wrapped model.
        examples: The questions to label.
        recipe: Sampling budget and label thresholds.
        shots: Few-shot demonstrations placed before every question.
        seed: Base seed of the sampling pass.
        batch_size: Prompts per generation call chunk.

    Returns:
        One `ItemLabel` per example, in the order of `examples`.

    Raises:
        ValueError: If the recipe or a generation argument is out of range,
            or a prompt exceeds the position limit of the model.
    """
    behavioral = [e for e in examples if e.answers]
    prompts = [few_shot_prompt(e.question, shots) for e in behavioral]
    greedy_rows: list[list[Completion]] = []
    sampled_rows: list[list[Completion]] = []
    if prompts:
        greedy_rows = generate(
            loaded,
            prompts,
            max_new_tokens=recipe.max_new_tokens,
            temperature=None,
            batch_size=batch_size,
        )
        sampled_rows = generate(
            loaded,
            prompts,
            max_new_tokens=recipe.max_new_tokens,
            temperature=recipe.temperature,
            num_samples=recipe.n_samples,
            seed=seed,
            batch_size=batch_size,
        )
    behavioral_labels: list[ItemLabel] = []
    for example, greedy_row, sampled_row in zip(
        behavioral, greedy_rows, sampled_rows, strict=True
    ):
        greedy = extract_answer(greedy_row[0].text, stop=recipe.stop)
        samples = tuple(
            extract_answer(c.text, stop=recipe.stop) for c in sampled_row
        )
        greedy_correct = is_correct(greedy, example.answers)
        n_correct = sum(is_correct(a, example.answers) for a in samples)
        behavioral_labels.append(
            ItemLabel(
                uid=example.uid,
                label=label_from_counts(
                    greedy_correct,
                    n_correct,
                    n_samples=recipe.n_samples,
                    known_min_correct=recipe.known_min_correct,
                ),
                basis="behavior",
                greedy=greedy,
                greedy_correct=greedy_correct,
                samples=samples,
                n_correct=n_correct,
            )
        )
    labeled = iter(behavioral_labels)
    return [
        next(labeled)
        if e.answers
        else ItemLabel(e.uid, "unknown", "construction", "", False, (), 0)
        for e in examples
    ]


_LABEL_ORDER: tuple[Label, ...] = ("known", "unknown", "uncertain")
_HAND_CHECK_COLUMNS = (
    "uid",
    "kind",
    "question",
    "gold",
    "greedy",
    "samples",
    "label",
    "human_label",
)


def _choose_rows(labels: Sequence[ItemLabel], n: int, seed: int) -> list[int]:
    """Picks up to `n` indices, filling one row per label in turn."""
    rng = np.random.default_rng(seed)
    queues: list[list[int]] = []
    for name in _LABEL_ORDER:
        members = [i for i, item in enumerate(labels) if item.label == name]
        if members:
            queues.append([members[j] for j in rng.permutation(len(members))])
    chosen: list[int] = []
    depth = 0
    while len(chosen) < n and any(depth < len(q) for q in queues):
        for queue in queues:
            if depth < len(queue) and len(chosen) < n:
                chosen.append(queue[depth])
        depth += 1
    return sorted(chosen)


def write_hand_check(
    path: Path,
    examples: Sequence[QAExample],
    labels: Sequence[ItemLabel],
    *,
    n: int = 200,
    seed: int = 0,
) -> int:
    """Writes a CSV of labeled items for manual checking.

    All given labels are considered: filter by `basis` beforehand to check
    behavioral labels only. Rows are chosen round-robin over the labels
    present, in the order known, unknown, uncertain, one row at a time until
    `n` rows are chosen or the items run out, so the label quotas are as
    even as the counts allow. Within a label the rows are a seeded random
    draw. The file keeps the order of `examples`; gold aliases and samples
    are joined with " | " and `human_label` is left empty.

    Args:
        path: Output file.
        examples: The labeled questions.
        labels: One label per example, in the same order.
        n: Maximum number of rows.
        seed: Seed of the draw.

    Returns:
        The number of data rows written.

    Raises:
        ValueError: If `examples` and `labels` do not pair up by uid.
    """
    if [e.uid for e in examples] != [item.uid for item in labels]:
        raise ValueError("examples and labels must pair up by uid")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_HAND_CHECK_COLUMNS)
        rows = _choose_rows(labels, n, seed)
        for i in rows:
            example, item = examples[i], labels[i]
            writer.writerow(
                [
                    example.uid,
                    example.kind,
                    example.question,
                    " | ".join(example.answers),
                    item.greedy,
                    " | ".join(item.samples),
                    item.label,
                    "",
                ]
            )
    return len(rows)
