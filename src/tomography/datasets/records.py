"""Example records returned by the dataset loaders, and split assignment."""

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
import hashlib
import json
import math
from typing import Literal

Split = Literal["train", "val", "test"]
MetaValue = str | int | float | None


@dataclass(frozen=True)
class Example:
    """One labeled text, the unit the stimulus builders consume.

    Attributes:
        uid: Stable identifier, unique per source statement. Splits are held
            out by uid.
        text: The raw stimulus, not yet rendered through a template.
        label: 1 for the positive pole (concept present, true, more
            pleasant, acceptable, positive sentiment), else 0 or a class id.
        split: Held-out split this example belongs to.
        meta: Extra per-example fields (for example "question").
    """

    uid: str
    text: str
    label: int
    split: Split
    meta: Mapping[str, MetaValue] = field(default_factory=dict)


@dataclass(frozen=True)
class ChoiceExample:
    """A multiple-choice item with one correct choice.

    Attributes:
        uid: Stable identifier, unique per question.
        question: The question text.
        choices: The answer choices, in presentation order.
        correct: Index into `choices` of the correct choice.
        split: Held-out split this item belongs to.
        meta: Extra per-item fields.
    """

    uid: str
    question: str
    choices: tuple[str, ...]
    correct: int
    split: Split
    meta: Mapping[str, MetaValue] = field(default_factory=dict)

    def flatten(self) -> list[Example]:
        """Turns the item into one `Example` per choice.

        Returns:
            One example per choice with uid "<uid>:<i>", the choice as text,
            label 1 only for the correct choice, and `meta` holding the item
            meta plus "group" (the item uid) and "question".

        Raises:
            ValueError: If `correct` is not an index into `choices`.
        """
        if not 0 <= self.correct < len(self.choices):
            raise ValueError(
                f"{self.uid}: correct index {self.correct} is outside "
                f"{len(self.choices)} choices"
            )
        meta = {**self.meta, "group": self.uid, "question": self.question}
        return [
            Example(
                uid=f"{self.uid}:{i}",
                text=choice,
                label=int(i == self.correct),
                split=self.split,
                meta=meta,
            )
            for i, choice in enumerate(self.choices)
        ]


@dataclass(frozen=True)
class QAExample:
    """An open question with gold aliases, used by the knowledge studies.

    Whether a model knows the answer is a per-model label and is not part of
    the record.

    Attributes:
        uid: Stable identifier, unique per question.
        question: The question text.
        answers: Gold answer aliases, empty when the item has none.
        kind: The item family: "real", "fictional", "post_cutoff",
            "unknowable_detail", "false_premise", "kuq:<category>" or "open".
        split: Held-out split this item belongs to.
        meta: Extra per-item fields (relation, subject, popularity, source).
    """

    uid: str
    question: str
    answers: tuple[str, ...]
    kind: str
    split: Split
    meta: Mapping[str, MetaValue] = field(default_factory=dict)


def assign_splits(
    uids: Sequence[str],
    *,
    dataset: str,
    seed: int = 0,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> list[Split]:
    """Assigns each uid to train, val or test by hashing.

    The split is a pure function of (dataset, uid, seed): it ignores row
    order and library versions, and a repeated uid always lands in the same
    split, so everything derived from one source statement stays together.

    Args:
        uids: Source identifiers, one per example.
        dataset: Dataset name, so the same uid splits differently across
            datasets.
        seed: Changes the assignment without changing its proportions.
        fractions: Shares of (train, val, test); non-negative, sum to 1.

    Returns:
        One split per uid, in the order of `uids`.

    Raises:
        ValueError: If `fractions` has a negative part or does not sum to 1.
    """
    if min(fractions) < 0 or not math.isclose(
        sum(fractions), 1.0, abs_tol=1e-9
    ):
        raise ValueError(
            f"fractions must be non-negative and sum to 1, got {fractions}"
        )
    train_edge = fractions[0]
    val_edge = fractions[0] + fractions[1]
    splits: list[Split] = []
    for uid in uids:
        point = _unit_hash(dataset, seed, uid)
        if point < train_edge:
            splits.append("train")
        elif point < val_edge:
            splits.append("val")
        else:
            splits.append("test")
    return splits


def _unit_hash(dataset: str, seed: int, uid: str) -> float:
    """Maps (dataset, seed, uid) to a stable point in [0, 1)."""
    key = json.dumps([dataset, seed, uid]).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") / 2**64
