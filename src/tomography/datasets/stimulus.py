"""Stimulus sets: examples rendered through a template and paired for reading.

A ``StimulusSet`` holds exactly the stimuli that its reading pairs and its
evaluation groups reference, laid out pair by pair so that pair ``i`` of a
two-member layout is ``(2i, 2i + 1)``. The set is a pure function of the
examples, the template, the chat format and the seed.
"""

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal
from typing import Protocol
from typing import get_args

import numpy as np
from numpy.typing import NDArray

from tomography.pairing import label_contrastive_pairs
from tomography.pairing import one_vs_rest_pairs
from tomography.pairing import persona_pairs
from tomography.pairing import random_pairs
from tomography.templates import ChatFormat
from tomography.templates import ConceptTemplate
from tomography.templates import Rendered
from tomography.templates import Template

Span = tuple[int, int]
IntArray = NDArray[np.int64]
PairingMode = Literal[
    "random",
    "label_contrastive",
    "persona",
    "one_vs_rest",
    "by_group",
    "within_group",
]


class ExampleLike(Protocol):
    """The structural type of a source example (satisfied by ``Example``)."""

    @property
    def uid(self) -> str:
        """Stable id of the source statement, the unit of splitting."""
        ...

    @property
    def text(self) -> str:
        """The raw, untemplated stimulus text."""
        ...

    @property
    def label(self) -> int:
        """The source label (1 is the positive pole)."""
        ...

    @property
    def meta(self) -> Mapping[str, str | int | float | None]:
        """Extra fields; ``question`` and ``instruction`` fill slots."""
        ...


@dataclass(frozen=True, eq=False)
class StimulusSet:
    """Rendered stimuli with their reading pairs and evaluation groups.

    Attributes:
        texts: The rendered prompts.
        response_spans: The marked character span of each text.
        concept_spans: The concept-word span of each text, if any.
        labels: Binary int64 labels of shape ``(n,)``.
        pairs: Int64 ``(m, 2)`` indices into ``texts``, the reading pairs.
        groups: Int64 ``(g, k)`` indices into ``texts``, padded with -1, the
            evaluation groups.
        uids: The source example id of each stimulus (both personas of one
            statement share it).
    """

    texts: tuple[str, ...]
    response_spans: tuple[Span | None, ...]
    concept_spans: tuple[Span | None, ...]
    labels: IntArray
    pairs: IntArray
    groups: IntArray
    uids: tuple[str, ...]


@dataclass(frozen=True, eq=False)
class _Plan:
    """Which examples to render, in which order, and how they are paired."""

    order: IntArray
    labels: IntArray
    pairs: IntArray
    groups: IntArray
    poles: tuple[Literal["pos", "neg"], ...] | None = None
    persona_index: IntArray | None = None


def build_stimulus_set(
    examples: Sequence[ExampleLike],
    template: Template,
    chat: ChatFormat,
    *,
    pairing: PairingMode,
    concept: str | None = None,
    target_class: int | None = None,
    group_ids: Sequence[str] | None = None,
    strata: Sequence[str] | None = None,
    n_pairs: int | None = None,
    seed: int = 0,
) -> StimulusSet:
    """Renders examples through a template and pairs them for reading.

    Concept templates give one stimulus per example. Function templates give
    two per example (the two persona poles, in the ``persona_pairs`` layout,
    labels 1 and 0) and need ``pairing="persona"``, which in turn needs a
    function template. ``texts`` holds exactly the stimuli that ``pairs`` or
    ``groups`` reference, so ``n_pairs=64`` gives 64 pairs and 128 stimuli for
    concept and persona templates. ``groups`` equals ``pairs`` for the
    label-contrastive, persona and one-vs-rest modes, holds the label matches
    among the same stimuli for ``random``, and has all members of each group
    for ``by_group`` and ``within_group``.

    Args:
        examples: Source examples. Split them by source before calling, so
            both renderings of one statement land in the same split.
        template: The template to render.
        chat: The chat format that wraps each prompt.
        pairing: How stimuli are paired. ``random`` matches stimuli at
            random, ``label_contrastive`` one label-1 with one label-0 member,
            ``one_vs_rest`` a ``target_class`` member with a member of another
            class, ``persona`` the two poles of a statement, and ``by_group``
            takes the examples that share a ``group_ids`` entry as one group
            (one label-1 member among the rest) and pairs its label-1 member
            with one seeded label-0 member. ``within_group`` takes the same
            groups and pairs consecutive members of each seeded shuffle of a
            group, the reference's unsupervised pairing (see
            ``_within_group_plan``).
        concept: The concept word for a ``{concept}`` slot.
        target_class: The class that is label 1 for ``one_vs_rest``.
        group_ids: One group id per example, required for ``by_group`` and
            ``within_group``.
        strata: One stratum name per example (``label_contrastive`` only);
            pairs never cross strata.
        n_pairs: Number of pairs (statements for persona, groups for
            ``by_group``). ``None`` takes as many as the mode allows;
            ``within_group`` pairs every group and accepts only ``None``.
        seed: Seed for every random choice.

    Returns:
        The stimulus set.

    Raises:
        ValueError: If ``pairing`` is not a known mode, the arguments do not
            fit the template or the mode, a required argument or template slot
            value is missing, labels are not binary where they must be, or
            ``n_pairs`` is out of range.
    """
    if not examples:
        raise ValueError("at least one example is required")
    is_function = not isinstance(template, ConceptTemplate)
    if is_function != (pairing == "persona"):
        raise ValueError(
            "function templates need pairing='persona' and only they allow it"
        )
    if strata is not None and pairing != "label_contrastive":
        raise ValueError("strata is only valid for label_contrastive pairing")
    if (
        isinstance(template, ConceptTemplate)
        and "concept" in template.slots
        and concept is None
    ):
        raise ValueError(f"template {template.name!r} needs a concept")
    if n_pairs is not None and n_pairs < 1:
        raise ValueError(f"n_pairs must be positive, got {n_pairs}")
    rng = np.random.default_rng(seed)
    if pairing == "persona":
        assert not isinstance(template, ConceptTemplate)
        plan = _persona_plan(examples, template, n_pairs, rng)
    elif pairing == "by_group":
        plan = _group_plan(examples, group_ids, n_pairs, rng)
    elif pairing == "within_group":
        plan = _within_group_plan(examples, group_ids, n_pairs, rng)
    elif pairing == "one_vs_rest":
        plan = _one_vs_rest_plan(examples, target_class, n_pairs, rng)
    elif pairing == "random":
        plan = _random_plan(examples, n_pairs, rng)
    elif pairing == "label_contrastive":
        plan = _label_contrastive_plan(examples, strata, n_pairs, rng)
    else:
        raise ValueError(
            f"unknown pairing {pairing!r}, "
            f"expected one of {get_args(PairingMode)}"
        )
    rendered = [
        _render(examples[example], k, plan, template, chat, concept)
        for k, example in enumerate(plan.order)
    ]
    return StimulusSet(
        texts=tuple(r.text for r in rendered),
        response_spans=tuple(r.response_span for r in rendered),
        concept_spans=tuple(r.concept_span for r in rendered),
        labels=plan.labels,
        pairs=plan.pairs,
        groups=plan.groups,
        uids=tuple(examples[example].uid for example in plan.order),
    )


def _render(
    example: ExampleLike,
    position: int,
    plan: _Plan,
    template: Template,
    chat: ChatFormat,
    concept: str | None,
) -> Rendered:
    """Renders stimulus ``position`` of the plan from its source example."""
    if isinstance(template, ConceptTemplate):
        question = _slot_text(example, "question", template.slots)
        return template.render(
            example.text, concept or "", chat, question=question
        )
    assert plan.poles is not None
    assert plan.persona_index is not None
    return template.render(
        example.text,
        plan.poles[position],
        chat,
        index=int(plan.persona_index[position]),
        instruction=_slot_text(example, "instruction", template.slots),
    )


def _slot_text(
    example: ExampleLike, name: str, slots: frozenset[str]
) -> str | None:
    """Reads a text slot value from ``meta`` if the template has the slot."""
    if name not in slots:
        return None
    value = example.meta.get(name)
    if not isinstance(value, str):
        raise ValueError(
            f"example {example.uid!r} needs meta[{name!r}] as text for the "
            "template"
        )
    return value


def _binary_labels(examples: Sequence[ExampleLike]) -> IntArray:
    """Returns the example labels, which must all be 0 or 1."""
    labels = np.array([example.label for example in examples], dtype=np.int64)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("example labels must be binary (0 or 1)")
    return labels


def _count(requested: int | None, available: int, unit: str) -> int:
    """Resolves ``n_pairs`` against the number the mode allows."""
    if requested is None:
        return available
    if requested > available:
        raise ValueError(
            f"n_pairs={requested} exceeds the {available} {unit} available"
        )
    return requested


def _contiguous(matched: IntArray) -> tuple[IntArray, IntArray]:
    """Lays matched example pairs out pair by pair."""
    order = matched.reshape(-1)
    return order, np.arange(order.size, dtype=np.int64).reshape(-1, 2)


def _label_contrastive_plan(
    examples: Sequence[ExampleLike],
    strata: Sequence[str] | None,
    n_pairs: int | None,
    rng: np.random.Generator,
) -> _Plan:
    """Plans one label-1 and one label-0 stimulus per pair, per stratum.

    Args:
        examples: Source examples with binary labels.
        strata: Optional stratum name per example; pairs never cross strata.
        n_pairs: Number of pairs, or ``None`` for every match available.
        rng: Generator that supplies the seed of the pairing function.

    Returns:
        A plan whose groups equal its pairs.

    Raises:
        ValueError: If a label is not binary or ``strata`` has the wrong
            length.
    """
    labels = _binary_labels(examples)
    codes = None
    if strata is not None:
        if len(strata) != len(examples):
            raise ValueError("strata needs one entry per example")
        _, codes = np.unique(np.asarray(strata), return_inverse=True)
        codes = codes.astype(np.int64)
    matched = label_contrastive_pairs(
        labels, seed=_child_seed(rng), n_pairs=n_pairs, strata=codes
    )
    order, pairs = _contiguous(matched)
    return _Plan(order, labels[order], pairs, pairs)


def _one_vs_rest_plan(
    examples: Sequence[ExampleLike],
    target_class: int | None,
    n_pairs: int | None,
    rng: np.random.Generator,
) -> _Plan:
    """Plans pairs of one target-class stimulus and one from another class.

    Args:
        examples: Source examples whose label is a class id.
        target_class: The class that becomes label 1.
        n_pairs: Number of pairs, or ``None`` for ``min(#target, #others)``.
        rng: Generator that supplies the seed of the pairing function.

    Returns:
        A plan with binary labels (1 for the target class) whose groups equal
        its pairs.

    Raises:
        ValueError: If ``target_class`` is None.
    """
    if target_class is None:
        raise ValueError("one_vs_rest pairing needs a target_class")
    classes = np.array([example.label for example in examples], dtype=np.int64)
    matched = one_vs_rest_pairs(
        classes, target_class, seed=_child_seed(rng), n_pairs=n_pairs
    )
    order, pairs = _contiguous(matched)
    labels = (classes[order] == target_class).astype(np.int64)
    return _Plan(order, labels, pairs, pairs)


def _random_plan(
    examples: Sequence[ExampleLike],
    n_pairs: int | None,
    rng: np.random.Generator,
) -> _Plan:
    """Plans a random perfect matching over a random subset of examples.

    Args:
        examples: Source examples with binary labels.
        n_pairs: Number of pairs, or ``None`` for ``len(examples) // 2``.
        rng: Generator for the subset and the matching.

    Returns:
        A plan whose groups hold the label-contrastive matches among the
        chosen stimuli (empty when only one label is present).

    Raises:
        ValueError: If a label is not binary, ``n_pairs`` exceeds the pairs
            available, or fewer than two examples are given.
    """
    labels = _binary_labels(examples)
    size = _count(n_pairs, len(examples) // 2, "pairs")
    if size == 0:
        raise ValueError("random pairing needs at least two examples")
    chosen = rng.permutation(len(examples))[: 2 * size].astype(np.int64)
    matched = chosen[random_pairs(2 * size, seed=_child_seed(rng))]
    order, pairs = _contiguous(matched)
    stimulus_labels = labels[order]
    groups = np.empty((0, 2), dtype=np.int64)
    if stimulus_labels.any() and not stimulus_labels.all():
        groups = label_contrastive_pairs(stimulus_labels, seed=_child_seed(rng))
    return _Plan(order, stimulus_labels, pairs, groups)


def _persona_plan(
    examples: Sequence[ExampleLike],
    template: Template,
    n_pairs: int | None,
    rng: np.random.Generator,
) -> _Plan:
    """Plans two stimuli per example, one for each persona pole.

    Args:
        examples: Source examples; their labels are ignored.
        template: A function template that supplies the persona lists.
        n_pairs: Number of statements, or ``None`` for every example.
        rng: Generator for the example subset and the persona choices.

    Returns:
        A plan in the ``persona_pairs`` layout with labels 1 and 0 and the
        persona index of each stimulus; groups equal pairs.

    Raises:
        ValueError: If ``n_pairs`` exceeds the number of examples.
    """
    assert not isinstance(template, ConceptTemplate)
    size = _count(n_pairs, len(examples), "examples")
    chosen = rng.permutation(len(examples))[:size].astype(np.int64)
    persona_index = np.empty(2 * size, dtype=np.int64)
    persona_index[0::2] = rng.integers(len(template.pos_personas), size=size)
    persona_index[1::2] = rng.integers(len(template.neg_personas), size=size)
    pairs = persona_pairs(2 * size)
    return _Plan(
        order=np.repeat(chosen, 2),
        labels=np.tile(np.array([1, 0], dtype=np.int64), size),
        pairs=pairs,
        groups=pairs.copy(),
        poles=("pos", "neg") * size,
        persona_index=persona_index,
    )


def _group_plan(
    examples: Sequence[ExampleLike],
    group_ids: Sequence[str] | None,
    n_pairs: int | None,
    rng: np.random.Generator,
) -> _Plan:
    """Plans k-way groups and one label-1 versus label-0 pair per group.

    Args:
        examples: Source examples with binary labels.
        group_ids: One group id per example; the members of a group stay
            contiguous in the plan and keep their input order.
        n_pairs: Number of groups, or ``None`` for every group.
        rng: Generator for the group subset and the label-0 member choice.

    Returns:
        A plan whose groups are padded with -1 and whose pairs hold the
        label-1 member and one label-0 member of the same group.

    Raises:
        ValueError: If ``group_ids`` is missing or has the wrong length, a
            label is not binary, a group lacks exactly one label-1 member or
            any label-0 member, or ``n_pairs`` exceeds the groups available.
    """
    labels, members = _members(examples, group_ids, "by_group")
    keys = list(members)
    size = _count(n_pairs, len(keys), "groups")
    chosen = rng.permutation(len(keys))[:size]
    selected = [np.array(members[keys[g]], dtype=np.int64) for g in chosen]
    order = np.concatenate(selected)
    groups = np.full((size, max(map(len, selected))), -1, dtype=np.int64)
    pairs = np.empty((size, 2), dtype=np.int64)
    start = 0
    for row, block in enumerate(selected):
        local = np.arange(start, start + len(block), dtype=np.int64)
        groups[row, : len(block)] = local
        positive = local[labels[block] == 1][0]
        negative = rng.choice(local[labels[block] == 0])
        pairs[row] = (
            (positive, negative) if rng.random() < 0.5 else (negative, positive)
        )
        start += len(block)
    return _Plan(order, labels[order], pairs, groups)


def _within_group_plan(
    examples: Sequence[ExampleLike],
    group_ids: Sequence[str] | None,
    n_pairs: int | None,
    rng: np.random.Generator,
) -> _Plan:
    """Plans the reference's unsupervised pairs inside each group.

    The reference (representation-engineering ``repe_eval``) shuffles the
    choices of each training question and differences consecutive stimuli,
    so its pairs stay inside one question, include wrong-versus-wrong pairs
    and never read the labels. Here every group is shuffled with the seed
    and its consecutive members are paired. A group with an odd number of
    members first gets a copy of one seeded label-0 member, as the
    reference pads its 5-choice CommonsenseQA questions to six. The copy is
    paired like any member but is left out of the group's row, so the row
    keeps exactly one label-1 member for orientation. Unlike the reference,
    which draws the pad from all choices, the copy is never the label-1
    member.

    Args:
        examples: Source examples with binary labels.
        group_ids: One group id per example; groups keep their order of
            first appearance.
        n_pairs: Must be None: every group is paired.
        rng: Generator for the pads and the shuffles.

    Returns:
        A plan whose groups hold the members of each group, padded with -1,
        and whose pairs are consecutive stimuli of one shuffled group.

    Raises:
        ValueError: If ``n_pairs`` is given, ``group_ids`` is missing or has
            the wrong length, a label is not binary, or a group lacks
            exactly one label-1 member or any label-0 member.
    """
    if n_pairs is not None:
        raise ValueError(
            "within_group pairing pairs every group; n_pairs must be None, "
            f"got {n_pairs}"
        )
    labels, members = _members(examples, group_ids, "within_group")
    blocks = []
    rows = []
    start = 0
    for indices in members.values():
        block = np.array(indices, dtype=np.int64)
        is_member = np.ones(len(block), dtype=bool)
        if len(block) % 2:
            block = np.append(block, rng.choice(block[labels[block] == 0]))
            is_member = np.append(is_member, False)
        shuffle = rng.permutation(len(block))
        blocks.append(block[shuffle])
        rows.append(start + np.flatnonzero(is_member[shuffle]))
        start += len(block)
    groups = np.full((len(rows), max(map(len, rows))), -1, dtype=np.int64)
    for row, positions in enumerate(rows):
        groups[row, : len(positions)] = positions
    order, pairs = _contiguous(np.concatenate(blocks).reshape(-1, 2))
    return _Plan(order, labels[order], pairs, groups)


def _members(
    examples: Sequence[ExampleLike],
    group_ids: Sequence[str] | None,
    pairing: PairingMode,
) -> tuple[IntArray, dict[str, list[int]]]:
    """Validates the groups of a group-wise mode and lists their members.

    Args:
        examples: Source examples with binary labels.
        group_ids: One group id per example.
        pairing: The mode, named in the error messages.

    Returns:
        The binary labels and the example indices of each group, groups in
        order of first appearance and members in input order.

    Raises:
        ValueError: If ``group_ids`` is missing or has the wrong length, a
            label is not binary, or a group lacks exactly one label-1 member
            or any label-0 member.
    """
    if group_ids is None or len(group_ids) != len(examples):
        raise ValueError(
            f"{pairing} pairing needs one group_ids entry per example"
        )
    labels = _binary_labels(examples)
    members: dict[str, list[int]] = {}
    for index, group in enumerate(group_ids):
        members.setdefault(group, []).append(index)
    for group, indices in members.items():
        positives = int(labels[indices].sum())
        if positives != 1 or positives == len(indices):
            raise ValueError(
                f"group {group!r} needs exactly one label-1 member and at "
                "least one label-0 member"
            )
    return labels, members


def _child_seed(rng: np.random.Generator) -> int:
    """Draws an independent seed for a pairing function."""
    return int(rng.integers(2**32))
