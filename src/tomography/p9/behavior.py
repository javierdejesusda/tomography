"""What a model does about a question it may not know: abstain or answer.

A completion is an abstention when it says that the model does not know, by
one of eleven phrase patterns, and no judge model is called. Otherwise it is
scored against the gold aliases exactly as the labels are, so "correct" means
the same here as in the labeling run. The rates, the choice of a steering
coefficient and the interval of a shift are plain functions of those
outcomes.
"""

from collections.abc import Sequence
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import re

import numpy as np
from numpy.typing import NDArray

from tomography.control.generate import Completion
from tomography.knowledge import extract_answer
from tomography.knowledge import is_correct
from tomography.reading.metrics import bootstrap_ci

ABSTAIN_PATTERNS: tuple[str, ...] = (
    r"i (?:do not|don't|dont) know",
    r"i(?:'m| am) not sure",
    r"i(?:'m| am) (?:not )?(?:unable|not able) to",
    r"i (?:cannot|can't|can not|couldn't|could not)"
    r" (?:answer|find|verify|provide|determine|say)",
    r"no (?:reliable |public |such )?(?:information|record|data)",
    r"not (?:aware|familiar)",
    r"(?:does not|doesn't|do not|don't) (?:exist|have (?:any )?information)",
    r"unknown(?! soldiers?\b)",
    r"cannot be (?:determined|known|answered)",
    r"^\W*(?:\d+\W+)?(?:do not|don't|dont) know",
    r"(?:is|are|was|were) not"
    r" (?:(?:publicly|widely|generally|commonly|readily) ){0,2}"
    r"(?:known(?! (?:for|to|as)\b)|disclosed|publicized|publicised|specified)",
)
ABSTAIN_REGEXES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(rf"(?<!\w)(?:{pattern})(?!\w)", re.MULTILINE)
    for pattern in ABSTAIN_PATTERNS
)
CHECK_COLUMNS: tuple[str, ...] = (
    "uid",
    "kind",
    "coeff",
    "question",
    "completion",
    "abstained",
    "correct",
    "human_label",
)


def is_abstention(
    text: str, patterns: Sequence[re.Pattern[str]] = ABSTAIN_REGEXES
) -> bool:
    """Tells whether a completion says that the model does not know.

    The text is lower-cased and the typographic apostrophe is folded to a
    plain one before the patterns are searched.

    Args:
        text: The completion.
        patterns: The compiled patterns; any match is an abstention.

    Returns:
        Whether some pattern occurs in the text.
    """
    folded = text.lower().replace("\N{RIGHT SINGLE QUOTATION MARK}", "'")
    return any(pattern.search(folded) for pattern in patterns)


@dataclass(frozen=True)
class Outcome:
    """What one completion did.

    Attributes:
        abstained: Whether the completion is an abstention.
        correct: Whether it did not abstain and an alias of the gold answer
            matches its answer; False without gold answers.
    """

    abstained: bool
    correct: bool


def classify(completion: Completion | str, answers: Sequence[str]) -> Outcome:
    """Classifies a completion as an abstention, a correct or a wrong answer.

    Args:
        completion: The generated text, or the completion that holds it.
        answers: The gold aliases; empty for a question without one.

    Returns:
        The outcome. A completion that is not an abstention is correct when
        `knowledge.is_correct` matches its first line against an alias.
    """
    text = completion if isinstance(completion, str) else completion.text
    if is_abstention(text):
        return Outcome(abstained=True, correct=False)
    return Outcome(
        abstained=False, correct=is_correct(extract_answer(text), answers)
    )


@dataclass(frozen=True)
class BehaviorRates:
    """The behavior of one arm on one known set and one unknown set.

    Attributes:
        n_known: Items of the known side.
        n_unknown: Items of the unknown side.
        false_refusal: Share of the known items abstained on.
        correct_refusal: Share of the unknown items abstained on.
        coverage: Share of all items answered, not abstained on.
        wrong_among_answered: Share of the answered items that are not
            correct; NaN when nothing was answered.
        balanced: `0.5 * (correct_refusal + 1 - false_refusal)`.
    """

    n_known: int
    n_unknown: int
    false_refusal: float
    correct_refusal: float
    coverage: float
    wrong_among_answered: float
    balanced: float


def behavior_rates(
    known: Sequence[Outcome], unknown: Sequence[Outcome]
) -> BehaviorRates:
    """Summarizes the outcomes on a known side and an unknown side.

    The labels of the pool decide the side of an item, never the completion.

    Args:
        known: The outcomes on the known items.
        unknown: The outcomes on the unknown items.

    Returns:
        The rates.

    Raises:
        ValueError: If either side is empty.
    """
    if not known or not unknown:
        raise ValueError(
            f"both sides need outcomes, got {len(known)} known and "
            f"{len(unknown)} unknown (an empty side has no rate)"
        )
    false_refusal = sum(o.abstained for o in known) / len(known)
    correct_refusal = sum(o.abstained for o in unknown) / len(unknown)
    answered = [o for o in (*known, *unknown) if not o.abstained]
    wrong = math.nan
    if answered:
        wrong = sum(not o.correct for o in answered) / len(answered)
    return BehaviorRates(
        n_known=len(known),
        n_unknown=len(unknown),
        false_refusal=false_refusal,
        correct_refusal=correct_refusal,
        coverage=len(answered) / (len(known) + len(unknown)),
        wrong_among_answered=wrong,
        balanced=0.5 * (correct_refusal + 1.0 - false_refusal),
    )


def select_coefficient(
    coeffs: Sequence[float], rates: Sequence[BehaviorRates]
) -> float:
    """Chooses the coefficient with the best balanced behavior.

    Ties, after rounding `balanced` to 12 digits, go to the smaller
    magnitude and then to the smaller coefficient, so a direction that
    changes nothing selects 0.0.

    Args:
        coeffs: The coefficients.
        rates: The rates of each coefficient, on the calibration items.

    Returns:
        The chosen coefficient.

    Raises:
        ValueError: If the inputs are empty or differ in length.
    """
    if len(coeffs) != len(rates):
        raise ValueError(
            f"got {len(coeffs)} coefficients and {len(rates)} rates"
        )
    if not coeffs:
        raise ValueError("coeffs is empty")
    best = min(
        range(len(coeffs)),
        key=lambda i: (
            -round(rates[i].balanced, 12),
            abs(coeffs[i]),
            coeffs[i],
        ),
    )
    return float(coeffs[best])


def shift_ci(
    base: NDArray[np.float64],
    steered: NDArray[np.float64],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Computes the mean paired shift and its bootstrap interval.

    Args:
        base: The unsteered value of each item.
        steered: The steered value of the same items.
        n_boot: Bootstrap resamples of the pairs.
        alpha: One minus the coverage.
        seed: Seed of the resampling.

    Returns:
        The mean of `steered - base` over the finite pairs, and the lower and
        upper bounds of its percentile interval.

    Raises:
        ValueError: If the arrays differ in shape, no pair is finite, or
            `alpha` or `n_boot` is invalid.
    """
    before = np.asarray(base, dtype=np.float64)
    after = np.asarray(steered, dtype=np.float64)
    if before.shape != after.shape:
        raise ValueError(
            f"base and steered must have one shape, got {before.shape} "
            f"and {after.shape}"
        )
    keep = np.isfinite(before) & np.isfinite(after)
    if not keep.any():
        raise ValueError("no finite pair of values to compare")
    shift = (after - before)[keep]
    lo, hi = bootstrap_ci(shift, alpha=alpha, n_boot=n_boot, seed=seed)
    return float(shift.mean()), float(lo), float(hi)


@dataclass(frozen=True)
class CompletionRecord:
    """One generated completion, for the manual check.

    Attributes:
        uid: The question.
        kind: Its kind in the pool.
        coeff: The steering coefficient it was generated under.
        question: The question text.
        completion: The generated text.
        abstained: Whether it is an abstention.
        correct: Whether it is a correct answer.
    """

    uid: str
    kind: str
    coeff: float
    question: str
    completion: str
    abstained: bool
    correct: bool


def write_completion_check(
    path: Path,
    records: Sequence[CompletionRecord],
    *,
    n: int = 100,
    seed: int = 0,
) -> int:
    """Writes a CSV of completions for checking the abstention patterns.

    Rows alternate between abstained and answered completions, starting
    with an abstained one, until `n` rows are chosen or one kind runs out
    and the other fills the rest. Within a kind the rows are a seeded
    random draw and they keep the order they were drawn in.

    Args:
        path: Output file.
        records: The completions.
        n: Maximum number of rows.
        seed: Seed of the draw.

    Returns:
        The number of data rows written.
    """
    rng = np.random.default_rng(seed)
    queues: list[list[CompletionRecord]] = []
    for flag in (True, False):
        members = [r for r in records if r.abstained == flag]
        queues.append([members[j] for j in rng.permutation(len(members))])
    chosen: list[CompletionRecord] = []
    depth = 0
    while len(chosen) < n and any(depth < len(q) for q in queues):
        for queue in queues:
            if depth < len(queue) and len(chosen) < n:
                chosen.append(queue[depth])
        depth += 1
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CHECK_COLUMNS)
        for record in chosen:
            writer.writerow(
                [
                    record.uid,
                    record.kind,
                    f"{record.coeff:g}",
                    record.question,
                    record.completion,
                    int(record.abstained),
                    int(record.correct),
                    "",
                ]
            )
    return len(chosen)
