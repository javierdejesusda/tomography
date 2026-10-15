"""The guess control: a known label must beat a constant answer.

A model that always gives the same answer to a family of questions is right
on every question of that family whose gold answer is that one. Such a hit is
a guess and not knowledge, so a known label survives only where the constant
guess of its stratum would not have produced it.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
from dataclasses import dataclass

from tomography.knowledge import is_correct
from tomography.knowledge import normalize_answer
from tomography.p9.pool import GuessVerdict
from tomography.p9.pool import LabeledItem

GUESS_GROUPS: tuple[str, ...] = ("popqa", "award", "triviaqa", "nq_open")


@dataclass(frozen=True)
class StratumGuess:
    """The constant-guess baseline of one stratum for one model.

    Attributes:
        stratum: "popqa:<relation>", "award:<award>", "triviaqa" or
            "nq_open".
        n: Behavior-labeled items of the stratum.
        modal: The most frequent normalized greedy answer, "" when no
            answer repeats.
        modal_share: The share of the `n` items that gave `modal` (0.0
            without a modal answer).
        guess_accuracy: The share of the `n` items whose gold aliases match
            `modal`.
        excluded: Whether the stratum never supplies known items.
        eligible: Whether the stratum may supply known items: not excluded
            and a guess accuracy within the limit.
    """

    stratum: str
    n: int
    modal: str
    modal_share: float
    guess_accuracy: float
    excluded: bool
    eligible: bool


def guess_stratum(item: LabeledItem) -> str | None:
    """Names the stratum whose constant guess controls an item.

    Args:
        item: A labeled question.

    Returns:
        "popqa:<relation>" for PopQA, "award:<award>" for the award
        questions, "triviaqa" or "nq_open" for those datasets, and None for
        every other group (no gold answers or no shared answer space) and
        for items labeled by construction.
    """
    group = item.group
    if group not in GUESS_GROUPS or item.label.basis != "behavior":
        return None
    if group in ("popqa", "award"):
        return f"{group}:{item.example.meta['relation']}"
    return group


def stratum_guesses(
    items: Sequence[LabeledItem],
    *,
    max_guess_accuracy: float = 0.10,
    excluded: Sequence[str] = ("popqa:sport",),
    min_modal_count: int = 2,
) -> dict[str, StratumGuess]:
    """Measures the constant guess of every stratum.

    The modal answer of a stratum is its most frequent normalized greedy
    answer. An empty answer is never modal (a model that answers nothing has
    no guess), an answer seen fewer than `min_modal_count` times is not a
    pattern (without the floor a stratum of distinct answers would get a
    guess accuracy of 1/n), and ties go to the alphabetically first answer.

    Args:
        items: The labeled questions of one model.
        max_guess_accuracy: A stratum whose guess accuracy is above this
            supplies no known items.
        excluded: Strata that never supply known items.
        min_modal_count: Occurrences a modal answer needs.

    Returns:
        The guess of every stratum present, keyed and sorted by stratum.

    Raises:
        ValueError: If `min_modal_count` is below 1 or `max_guess_accuracy`
            is not a share in [0, 1].
    """
    if min_modal_count < 1:
        raise ValueError(
            f"min_modal_count must be at least 1, got {min_modal_count}"
        )
    if not 0.0 <= max_guess_accuracy <= 1.0:
        raise ValueError(
            "max_guess_accuracy must be a share in [0, 1], "
            f"got {max_guess_accuracy}"
        )
    members: dict[str, list[LabeledItem]] = {}
    for item in items:
        key = guess_stratum(item)
        if key is not None:
            members.setdefault(key, []).append(item)
    return {
        key: _stratum_guess(
            key,
            members[key],
            max_guess_accuracy=max_guess_accuracy,
            excluded=key in excluded,
            min_modal_count=min_modal_count,
        )
        for key in sorted(members)
    }


def _stratum_guess(
    key: str,
    members: Sequence[LabeledItem],
    *,
    max_guess_accuracy: float,
    excluded: bool,
    min_modal_count: int,
) -> StratumGuess:
    counts: dict[str, int] = {}
    for item in members:
        answer = normalize_answer(item.label.greedy)
        if answer:
            counts[answer] = counts.get(answer, 0) + 1
    modal, hits = "", 0
    if counts:
        modal, hits = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if hits < min_modal_count:
        modal, hits = "", 0
    accuracy = 0.0
    if modal:
        accuracy = sum(
            is_correct(modal, m.example.answers) for m in members
        ) / len(members)
    return StratumGuess(
        stratum=key,
        n=len(members),
        modal=modal,
        modal_share=hits / len(members),
        guess_accuracy=accuracy,
        excluded=excluded,
        eligible=not excluded and accuracy <= max_guess_accuracy,
    )


def guess_verdict(
    item: LabeledItem, guesses: Mapping[str, StratumGuess]
) -> GuessVerdict | None:
    """Decides whether a known label fails the guess control.

    The checks run in this order: the stratum is excluded, the stratum is
    guessable (its guess accuracy is above the limit), the greedy answer of
    the item is the modal answer of its stratum.

    Args:
        item: A labeled question.
        guesses: The strata, as `stratum_guesses` returns.

    Returns:
        The reason the label fails, or None when it stands: always None for
        a label that is not "known" and for items outside every stratum.

    Raises:
        ValueError: If a known item's stratum is missing from `guesses`.
    """
    if item.label.label != "known":
        return None
    key = guess_stratum(item)
    if key is None:
        return None
    if key not in guesses:
        raise ValueError(
            f"{item.example.uid}: no guess control for stratum {key!r}"
        )
    guess = guesses[key]
    if guess.excluded:
        return "excluded_stratum"
    if not guess.eligible:
        return "guessable_stratum"
    if normalize_answer(item.label.greedy) == guess.modal:
        return "modal_answer"
    return None


def apply_guess_control(
    items: Sequence[LabeledItem], guesses: Mapping[str, StratumGuess]
) -> list[LabeledItem]:
    """Sets the guess verdict of every item.

    The raw labels are kept: `LabeledItem.contrast_label` turns a failed
    known label into "uncertain", so it leaves the contrasts but stays
    available for error prediction.

    Args:
        items: The labeled questions.
        guesses: The strata, as `stratum_guesses` returns.

    Returns:
        The items in order, with `guess` set by `guess_verdict`.

    Raises:
        ValueError: If a known item's stratum is missing from `guesses`.
    """
    return [
        dataclasses.replace(item, guess=guess_verdict(item, guesses))
        for item in items
    ]
