"""The contrasts of the P9 reading study.

A contrast sets one side of labeled questions against another: either known
questions against one type of unknown ("knowledge"), or one kind of unknown
against another ("unknown_type"). The second kind needs no known side, so it
also exists for a model the guess control of `tomography.p9.guess` leaves
with too few known questions.
"""

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from tomography.datasets.records import Split
from tomography.p9.pool import LabeledItem

POP_EDGES: tuple[int, ...] = (1_000, 10_000, 100_000)
ENTITY_TYPES: tuple[str, ...] = (
    "fictional_synthal",
    "fictional_generated",
    "obscure_real",
    "obscure_real_pop",
    "post_cutoff",
)
UNKNOWN_TYPES: tuple[str, ...] = (
    "real_vs_synthal",
    "real_vs_generated",
    "pre_vs_post_cutoff",
)

Strata = Literal["relation", "relation_popularity", "none"]
Contrast = Literal["knowledge", "unknown_type"]


@dataclass(frozen=True)
class TypePool:
    """The two sides of one contrast.

    A reader fitted on the pool scores the positive side higher.

    Attributes:
        name: The contrast, also its config value and its table row.
        positive: The "known" questions of a knowledge contrast, the real or
            pre-cutoff unknowns of an unknown-type one.
        negative: The questions labeled "unknown", by behavior or by
            construction.
        strata: What a pair may not cross: a relation, a relation and
            popularity bin, or nothing.
        contrast: Whether the positive side is known questions or one kind
            of unknown.
    """

    name: str
    positive: tuple[LabeledItem, ...]
    negative: tuple[LabeledItem, ...]
    strata: Strata
    contrast: Contrast

    def split(
        self, split: Split
    ) -> tuple[list[LabeledItem], list[LabeledItem]]:
        """Returns the items of one split.

        Args:
            split: "train", "val" or "test".

        Returns:
            The positive and the negative items of that split, in the order
            of the pool.
        """
        return (
            [i for i in self.positive if i.example.split == split],
            [i for i in self.negative if i.example.split == split],
        )


def stratum_name(item: LabeledItem, mode: Strata) -> str:
    """Names the stratum of an item under a strata mode.

    Args:
        item: A labeled question.
        mode: What a pair may not cross.

    Returns:
        The empty string for "none", the relation for "relation", and the
        relation and the popularity bin ("relation|bin", the bin counting
        the `POP_EDGES` the popularity reaches) for "relation_popularity".

    Raises:
        ValueError: If the mode is "relation_popularity" and the item has no
            integer popularity.
    """
    if mode == "none":
        return ""
    relation = str(item.example.meta["relation"])
    if mode == "relation":
        return relation
    popularity = item.example.meta.get("popularity")
    if not isinstance(popularity, int):
        raise ValueError(
            f"{item.example.uid}: relation_popularity strata need an "
            "integer popularity"
        )
    return f"{relation}|{bisect_right(POP_EDGES, popularity)}"


def type_pools(
    items: Sequence[LabeledItem], *, knowledge: bool = True
) -> dict[str, TypePool]:
    """Builds the contrasts the labeled questions allow.

    Every side is chosen by `contrast_label`, so a known label that failed
    the guess control and an "uncertain" label are in no pool. A contrast
    exists only when both of its sides are non-empty.

    The known sides are the PopQA questions labeled known (`popqa_known`),
    the TriviaQA and NQ-open ones (`open_known`) and the award questions of
    kind "real" labeled known. `obscure` is the PopQA questions labeled
    unknown. The contrasts, in this order, are:

    - `fictional_synthal`: popqa_known against SyntHal.
    - `fictional_generated`: popqa_known against the generated names.
    - `obscure_real`: popqa_known against obscure.
    - `obscure_real_pop`: the same, paired inside a popularity bin.
    - `post_cutoff`: known real awards against unknown post-cutoff awards.
    - `unknowable_detail`: open_known against the unknowable details.
    - `false_premise`: open_known against FalseQA.
    - `kuq_<category>`: open_known against one KUQ category, sorted.
    - `real_vs_synthal`: obscure against SyntHal (unknown_type).
    - `real_vs_generated`: obscure against the generated names
      (unknown_type).
    - `pre_vs_post_cutoff`: unknown real awards against unknown post-cutoff
      awards (unknown_type).

    Args:
        items: The labeled questions.
        knowledge: Whether to build the contrasts that need a known side.
            False keeps the three "unknown_type" contrasts only.

    Returns:
        The non-empty contrasts by name, in the order above.
    """

    def pick(
        group: str, label: str, kind: str | None = None
    ) -> tuple[LabeledItem, ...]:
        return tuple(
            i
            for i in items
            if i.group == group
            and i.contrast_label == label
            and (kind is None or i.example.kind == kind)
        )

    popqa_known = pick("popqa", "known")
    open_known = (*pick("triviaqa", "known"), *pick("nq_open", "known"))
    obscure = pick("popqa", "unknown")
    pools: dict[str, TypePool] = {}

    def add(
        name: str,
        positive: Sequence[LabeledItem],
        negative: Sequence[LabeledItem],
        strata: Strata,
        contrast: Contrast = "knowledge",
    ) -> None:
        if positive and negative and (knowledge or contrast == "unknown_type"):
            pools[name] = TypePool(
                name, tuple(positive), tuple(negative), strata, contrast
            )

    add(
        "fictional_synthal", popqa_known, pick("synthal", "unknown"), "relation"
    )
    add(
        "fictional_generated",
        popqa_known,
        pick("fictional", "unknown"),
        "relation",
    )
    add("obscure_real", popqa_known, obscure, "relation")
    add("obscure_real_pop", popqa_known, obscure, "relation_popularity")
    add(
        "post_cutoff",
        pick("award", "known", "real"),
        pick("award", "unknown", "post_cutoff"),
        "relation",
    )
    add("unknowable_detail", open_known, pick("detail", "unknown"), "none")
    add("false_premise", open_known, pick("falseqa", "unknown"), "none")
    for kind in sorted({i.example.kind for i in items if i.group == "kuq"}):
        name = kind.replace(":", "_").replace(" ", "_")
        add(name, open_known, pick("kuq", "unknown", kind), "none")
    add(
        "real_vs_synthal",
        obscure,
        pick("synthal", "unknown"),
        "relation",
        "unknown_type",
    )
    add(
        "real_vs_generated",
        obscure,
        pick("fictional", "unknown"),
        "relation",
        "unknown_type",
    )
    add(
        "pre_vs_post_cutoff",
        pick("award", "unknown", "real"),
        pick("award", "unknown", "post_cutoff"),
        "relation",
        "unknown_type",
    )
    return pools


def panel_of(type_name: str) -> Literal["entity", "question", "unknown"]:
    """Names the figure panel of a contrast.

    The figure style allows eight series at most, and there can be sixteen
    contrasts, so the curves are drawn in three panels.

    Args:
        type_name: A contrast name.

    Returns:
        "entity" for `ENTITY_TYPES`, "unknown" for `UNKNOWN_TYPES` and
        "question" for the rest.
    """
    if type_name in UNKNOWN_TYPES:
        return "unknown"
    return "entity" if type_name in ENTITY_TYPES else "question"
