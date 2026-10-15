"""The item pool of the knowledge boundary study and its labels.

One pool holds the questions of one model: real entities, invented ones,
unanswerable questions and open-domain trivia. Each question carries the
behavioral label of `tomography.knowledge.label_questions`, and the guess
control of `tomography.p9.guess` may later demote a known label.
"""

from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal
import zlib

import numpy as np

from tomography.datasets.records import QAExample
from tomography.knowledge import ItemLabel
from tomography.knowledge import Label

GROUPS: tuple[str, ...] = (
    "popqa",
    "synthal",
    "fictional",
    "award",
    "detail",
    "falseqa",
    "kuq",
    "triviaqa",
    "nq_open",
)

GuessVerdict = Literal["excluded_stratum", "guessable_stratum", "modal_answer"]

_SOURCE_GROUP = {
    "popqa": "popqa",
    "synthal": "synthal",
    "generated": "fictional",
    "kuq": "kuq",
    "falseqa": "falseqa",
    "triviaqa": "triviaqa",
    "nq_open": "nq_open",
}


def group_of(example: QAExample) -> str:
    """Names the pool group of a question.

    Args:
        example: A question whose meta holds "source".

    Returns:
        The group, one of `GROUPS`: the source name, except that generated
        names are "fictional" and Wikidata items are "detail" for the kind
        "unknowable_detail" and "award" otherwise.

    Raises:
        ValueError: If the source is missing or not one of the pool's.
    """
    source = example.meta.get("source")
    if source == "wikidata":
        return "detail" if example.kind == "unknowable_detail" else "award"
    if not isinstance(source, str) or source not in _SOURCE_GROUP:
        raise ValueError(f"{example.uid}: unknown source {source!r}")
    return _SOURCE_GROUP[source]


@dataclass(frozen=True)
class LabeledItem:
    """One question with the label of one model.

    Attributes:
        example: The question.
        label: The raw behavioral label.
        guess: Why a known label failed the guess control, or None when the
            label stands.

    Raises:
        ValueError: If the label belongs to another question or a verdict
            is set on a label that is not "known".
    """

    example: QAExample
    label: ItemLabel
    guess: GuessVerdict | None = None

    def __post_init__(self) -> None:
        """Checks that the label fits the question and the verdict."""
        if self.label.uid != self.example.uid:
            raise ValueError(
                f"label {self.label.uid!r} does not belong to "
                f"{self.example.uid!r}"
            )
        if self.guess is not None and self.label.label != "known":
            raise ValueError(
                f"{self.example.uid}: only a known label can fail the "
                "guess control"
            )

    @property
    def group(self) -> str:
        """The pool group of the question."""
        return group_of(self.example)

    @property
    def contrast_label(self) -> Label:
        """The label every contrast and count uses.

        "uncertain" when the guess control rejected a known label, else the
        raw label.
        """
        return "uncertain" if self.guess is not None else self.label.label


def subsample(
    items: Sequence[QAExample],
    n: int | None,
    *,
    seed: int,
    by: Callable[[QAExample], str] | None = None,
    prefer: Callable[[QAExample], bool] | None = None,
    share: float = 0.5,
) -> list[QAExample]:
    """Caps the number of questions per stratum with a seeded draw.

    A stratum larger than `n` keeps `n` questions chosen by a seeded
    permutation of its uid-sorted members, so the draw does not depend on
    the input order. The result keeps the order of `items`.

    With `prefer`, a capped stratum takes up to `round(share * n)` preferred
    questions first, fills the rest from the others, and tops up from the
    preferred ones when the others run short.

    Args:
        items: The questions.
        n: Questions kept per stratum; None keeps every question.
        seed: Seeds the draw.
        by: Stratum of a question; None puts all questions in one stratum.
        prefer: Marks the questions the draw favors, if any.
        share: Share of `n` drawn from the preferred questions first.

    Returns:
        The kept questions, in their original order.

    Raises:
        ValueError: If `n` is negative or `share` is outside [0, 1].
    """
    if not 0.0 <= share <= 1.0:
        raise ValueError(f"share must be in [0, 1], got {share}")
    if n is None:
        return list(items)
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    strata: dict[str, list[int]] = {}
    for index, example in enumerate(items):
        key = "" if by is None else by(example)
        strata.setdefault(key, []).append(index)
    keep: set[int] = set()
    for key, indices in strata.items():
        if len(indices) <= n:
            keep.update(indices)
            continue
        ordered = sorted(indices, key=lambda i: items[i].uid)
        rng = np.random.default_rng([seed, zlib.crc32(key.encode())])
        shuffled = [ordered[int(j)] for j in rng.permutation(len(ordered))]
        if prefer is None:
            keep.update(shuffled[:n])
            continue
        first = [i for i in shuffled if prefer(items[i])]
        rest = [i for i in shuffled if not prefer(items[i])]
        take = min(len(first), round(share * n))
        chosen = first[:take] + rest[: n - take]
        chosen += first[take : take + n - len(chosen)]
        keep.update(chosen)
    return [example for i, example in enumerate(items) if i in keep]
