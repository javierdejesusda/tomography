"""Template variants of the reading tasks, read with shared seeds and items.

The prompt studies (P1, P2, and later P7 and P8) read one concept through many
variants of its template. Every variant of a task is trained with the same
seeds on the same examples and evaluated on the same items, so a difference
between two variants belongs to the wording and not to the draw.
"""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
import itertools
from pathlib import Path
from types import MappingProxyType
from typing import Literal
from typing import get_args

import numpy as np
from numpy.typing import NDArray

from tomography.activations import PositionStrategy
from tomography.datasets.emotions import load_emotions
from tomography.datasets.ethics import load_ethics_utility
from tomography.datasets.facts import load_facts
from tomography.datasets.geometry_of_truth import load_geometry_of_truth
from tomography.datasets.records import Example
from tomography.datasets.records import Split
from tomography.datasets.sst2 import load_sst2
from tomography.datasets.stimulus import StimulusSet
from tomography.datasets.stimulus import build_stimulus_set
from tomography.experiments.base import derive_seed
from tomography.experiments.sweep import Prepared
from tomography.experiments.sweep import ReadingConfig
from tomography.experiments.sweep import stimulus_activations
from tomography.experiments.sweep import sweep_from_activations
from tomography.model import LoadedModel
from tomography.reading.directions import Method
from tomography.reading.directions import Preprocess
from tomography.reading.reader import RepReader
from tomography.templates import EMOTIONS
from tomography.templates import PAPER_TEMPLATES
from tomography.templates import ChatFormat
from tomography.templates import ConceptTemplate
from tomography.templates import FunctionTemplate
from tomography.templates import Template
from tomography.templates import ablate

VariantKind = Literal[
    "paper",
    "no_template",
    "bare",
    "no_chat_tags",
    "neutral_frame",
    "no_trailing_question",
    "no_trailing_space",
    "paraphrase",
    "scaffold",
    "alternate",
    "concept_swap",
    "framing_swap",
]
VARIANT_KINDS: tuple[VariantKind, ...] = get_args(VariantKind)
Family = Literal["frame", "wording", "concept"]
FAMILIES: tuple[Family, ...] = get_args(Family)
FAMILY_OF: Mapping[VariantKind, Family] = MappingProxyType(
    {
        "paper": "frame",
        "no_template": "frame",
        "bare": "frame",
        "no_chat_tags": "frame",
        "neutral_frame": "frame",
        "no_trailing_question": "frame",
        "no_trailing_space": "frame",
        "paraphrase": "wording",
        "scaffold": "wording",
        "alternate": "wording",
        "concept_swap": "concept",
        "framing_swap": "concept",
    }
)
TaskSource = Literal[
    "sst2", "facts", "ethics_utility", "emotions", "geometry_of_truth"
]
TaskPairing = Literal["label_contrastive", "by_group", "one_vs_rest"]


@dataclass(frozen=True)
class ReadingTask:
    """One concept, its paper template and its dataset.

    Attributes:
        name: Key of the task in `READING_TASKS`.
        source: The dataset the examples come from.
        template: The paper template, the "paper" variant.
        concept: Fills the `{concept}` slot; None when the template has no
            such slot.
        noun: What the decoy line and the zero-shot prompt of P2 ask about.
        pairing: How the stimuli are paired.
        splits: The (train, select, report) splits; when select and report
            are the same split, they are two halves of it.
        min_words: Training examples with fewer words are dropped.
        target_class: The class that is label 1 for `one_vs_rest`.
        alternates: Other wordings of the same template (kind "alternate").
        framing: The `index` of `ablate(template, "framing_swap")`; None when
            the task has no function framing.
        word: The literal concept word read at position "concept" when the
            template has no slot.
        subset: Keeps only the examples with `meta["subset"] == subset`.
        target: The task whose select and report items evaluate this one;
            None means itself.
    """

    name: str
    source: TaskSource
    template: ConceptTemplate
    concept: str | None
    noun: str
    pairing: TaskPairing
    splits: tuple[Split, Split, Split]
    min_words: int = 1
    target_class: int | None = None
    alternates: tuple[ConceptTemplate, ...] = ()
    framing: int | None = None
    word: str | None = None
    subset: str | None = None
    target: str | None = None


def _concept_template(name: str) -> ConceptTemplate:
    template = PAPER_TEMPLATES[name]
    assert isinstance(template, ConceptTemplate)
    return template


def _truth_source(name: str, subset: str) -> ReadingTask:
    return ReadingTask(
        name,
        "geometry_of_truth",
        _concept_template("concept"),
        "truthfulness",
        "truthfulness",
        "label_contrastive",
        ("train", "val", "test"),
        subset=subset,
        target="truth",
    )


READING_TASKS: Mapping[str, ReadingTask] = MappingProxyType(
    {
        task.name: task
        for task in (
            ReadingTask(
                "sentiment",
                "sst2",
                _concept_template("concept"),
                "sentiment",
                "positive sentiment",
                "label_contrastive",
                ("train", "val", "val"),
                min_words=3,
            ),
            ReadingTask(
                "truth",
                "facts",
                _concept_template("concept"),
                "truthfulness",
                "truthfulness",
                "label_contrastive",
                ("train", "val", "test"),
            ),
            _truth_source("truth_cities", "cities"),
            _truth_source("truth_common", "common_claim_true_false"),
            ReadingTask(
                "utility",
                "ethics_utility",
                _concept_template("utility"),
                None,
                "happiness",
                "by_group",
                ("train", "val", "test"),
                word="happiness",
            ),
            *(
                ReadingTask(
                    f"emotion_{emotion}",
                    "emotions",
                    _concept_template("emotion_concept"),
                    emotion,
                    emotion,
                    "one_vs_rest",
                    ("train", "val", "test"),
                    target_class=index,
                    alternates=(_concept_template("emotion_concept_paper"),),
                    framing=index,
                )
                for index, emotion in enumerate(EMOTIONS)
            ),
        )
    }
)


def expand_tasks(names: Sequence[str]) -> tuple[str, ...]:
    """Expands "emotion" into the six emotion tasks.

    Args:
        names: Keys of `READING_TASKS`, or "emotion" for the six tasks
            `emotion_<e>`.

    Returns:
        The task names in the order given, "emotion" replaced in place by
        the six, without repeats.

    Raises:
        ValueError: If a name is not a task.
    """
    expanded: list[str] = []
    for name in names:
        for item in (
            [f"emotion_{e}" for e in EMOTIONS] if name == "emotion" else [name]
        ):
            if item not in READING_TASKS:
                raise ValueError(
                    f"unknown task {item!r}, expected one of "
                    f"{list(READING_TASKS)} or 'emotion'"
                )
            if item not in expanded:
                expanded.append(item)
    return tuple(expanded)


@dataclass(frozen=True)
class Variant:
    """One wording of a task template, ready to render stimuli.

    Attributes:
        name: The kind, or `<kind>_<i>` for a kind with several variants.
        kind: The kind of ablation or wording.
        family: The family of the kind.
        template: The template to render.
        chat: The chat format the prompts are rendered with.
        concept: The concept word for a `{concept}` slot, None when the
            template has no such slot.
        position: The position the variant is trained at when it is forced;
            "response" for a function template, else None.
    """

    name: str
    kind: VariantKind
    family: Family
    template: Template
    chat: ChatFormat
    concept: str | None
    position: PositionStrategy | None = None

    @property
    def evaluable(self) -> bool:
        """Whether the variant is a concept template.

        An evaluable variant is scored on its own items; a function variant is
        only trained and is scored on the items of the paper variant.
        """
        return isinstance(self.template, ConceptTemplate)


def _variant(
    task: ReadingTask,
    name: str,
    kind: VariantKind,
    template: Template,
    chat: ChatFormat,
) -> Variant:
    return Variant(
        name,
        kind,
        FAMILY_OF[kind],
        template,
        chat,
        task.concept if "concept" in template.slots else None,
        "response" if isinstance(template, FunctionTemplate) else None,
    )


def _without_trailing_space(chat: ChatFormat) -> ChatFormat:
    """Renders the plain format without the space after the assistant tag."""

    def render(messages: list[dict[str, str]]) -> str:
        return (
            f"{chat.user_tag} {messages[0]['content']}{chat.turn_sep}"
            f"{chat.assistant_tag}"
        )

    return ChatFormat("template", render_fn=render)


def _build(
    task: ReadingTask, chat: ChatFormat, kind: VariantKind
) -> list[Variant]:
    """Builds the variants of one kind, raising if it does not apply."""
    source = task.template
    if kind == "paper":
        return [_variant(task, kind, kind, source, chat)]
    if kind == "bare":
        template = ablate(ablate(source, "no_template"), "no_chat_tags")
        return [_variant(task, kind, kind, template, chat)]
    if kind == "no_trailing_space":
        if chat.kind != "plain":
            raise ValueError(f"needs the plain chat format, got {chat.kind!r}")
        if source.assistant:
            raise ValueError(
                f"template {source.name!r} has assistant text after the tag"
            )
        return [
            _variant(task, kind, kind, source, _without_trailing_space(chat))
        ]
    if kind == "alternate":
        if not task.alternates:
            raise ValueError(f"task {task.name!r} has no alternate wording")
        return [
            _variant(task, f"{kind}_{i}", kind, template, chat)
            for i, template in enumerate(task.alternates)
        ]
    if kind == "framing_swap":
        if task.framing is None:
            raise ValueError(f"task {task.name!r} has no function framing")
        template = ablate(source, kind, index=task.framing)
        return [_variant(task, kind, kind, template, chat)]
    if kind in ("paraphrase", "scaffold", "concept_swap"):
        variants: list[Variant] = []
        for index in itertools.count():
            try:
                template = ablate(source, kind, index=index)
            except ValueError:
                if index == 0:
                    raise
                return variants
            variants.append(
                _variant(task, f"{kind}_{index}", kind, template, chat)
            )
    return [_variant(task, kind, kind, ablate(source, kind), chat)]


def template_variants(
    task: ReadingTask,
    chat: ChatFormat,
    *,
    kinds: Sequence[VariantKind] = VARIANT_KINDS,
) -> tuple[tuple[Variant, ...], dict[str, str]]:
    """Builds the variants of a task's template.

    The `bare` variant is the stimulus with nothing around it (no template and
    no chat tags). `no_trailing_space` renders the plain format without the
    lone space token after `ASSISTANT:`; it needs the plain format and a
    template without assistant text. `framing_swap` is the function template
    of an emotion (concept against function framing); it is trained at the
    `response` position and is never evaluated.

    Args:
        task: The reading task.
        chat: The chat format of the model.
        kinds: The kinds to build; `paper` is required.

    Returns:
        The variants in `VARIANT_KINDS` order, `paper` first, and the kinds
        that do not apply to the task or the chat format, each with its
        reason.

    Raises:
        ValueError: If a kind is unknown or `kinds` lacks `paper`.
    """
    unknown = [kind for kind in kinds if kind not in VARIANT_KINDS]
    if unknown:
        raise ValueError(
            f"unknown variant kinds {unknown}, expected one of {VARIANT_KINDS}"
        )
    if "paper" not in kinds:
        raise ValueError(
            "kinds must include 'paper', the reference of every comparison"
        )
    variants: list[Variant] = []
    skipped: dict[str, str] = {}
    for kind in VARIANT_KINDS:
        if kind not in kinds:
            continue
        try:
            variants.extend(_build(task, chat, kind))
        except ValueError as error:
            skipped[kind] = str(error)
    return tuple(variants), skipped


@dataclass(frozen=True)
class TaskSplits:
    """The examples of a task, split by what they are used for.

    Attributes:
        train: The examples the readers are fitted on.
        select: The examples that choose the layer.
        report: The examples every reported number is computed on.
    """

    train: tuple[Example, ...]
    select: tuple[Example, ...]
    report: tuple[Example, ...]


def task_examples(
    task: ReadingTask, examples: Mapping[str, Sequence[Example]]
) -> list[Example]:
    """Returns the examples of a task's source, filtered by its subset.

    Args:
        task: The reading task.
        examples: The examples of each source, by source name.

    Returns:
        The examples of `task.source` whose `meta["subset"]` is
        `task.subset`, or all of them when the task has no subset.

    Raises:
        ValueError: If `examples` has no entry for the source.
    """
    if task.source not in examples:
        raise ValueError(f"no examples for source {task.source!r}")
    return [
        e
        for e in examples[task.source]
        if task.subset is None or e.meta.get("subset") == task.subset
    ]


def target_of(task: ReadingTask) -> ReadingTask:
    """Returns the task whose items evaluate a task.

    Args:
        task: The reading task.

    Returns:
        `READING_TASKS[task.target]`, or the task itself when it has no
        target.

    Raises:
        ValueError: If the target reads a different template, concept or
            pairing than the task.
    """
    if task.target is None:
        return task
    target = READING_TASKS[task.target]
    if (target.template, target.concept, target.pairing) != (
        task.template,
        task.concept,
        task.pairing,
    ):
        raise ValueError(
            f"task {task.name!r} and its target {target.name!r} differ in "
            "template, concept or pairing"
        )
    return target


def eval_seed(task: ReadingTask, split: str, seed: int) -> int:
    """Returns the seed of a task's evaluation items.

    The seed is keyed by the target, so the sources of one target share their
    evaluation items.

    Args:
        task: The reading task.
        split: "select" or "report".
        seed: The base seed of the run.

    Returns:
        The seed that draws the evaluation pairs of `split`.
    """
    return derive_seed(seed, "eval", target_of(task).name, split)


def _group_key(task: ReadingTask, example: Example) -> str:
    """Returns the unit a split never separates: a pair id or a uid."""
    if task.pairing == "by_group":
        return str(example.meta["pair"])
    return example.uid


def task_splits(
    task: ReadingTask,
    examples: Mapping[str, Sequence[Example]],
    *,
    seed: int,
) -> TaskSplits:
    """Splits the examples of a task into train, select and report.

    The training examples come from the task (its source, subset and
    `min_words`); the select and report examples come from its target. When
    the select and report splits are the same split, it is halved by group (a
    pair id for `by_group`, else the uid) with a seeded draw.

    Args:
        task: The reading task.
        examples: The examples of each source, by source name.
        seed: The base seed of the run.

    Returns:
        The three parts.

    Raises:
        ValueError: If a source is missing, the target differs from the task
            (see `target_of`), or a part has no pair.
    """
    target = target_of(task)
    train_split = task.splits[0]
    _, select_split, report_split = target.splits
    train = tuple(
        e
        for e in task_examples(task, examples)
        if e.split == train_split and len(e.text.split()) >= task.min_words
    )
    pool = task_examples(target, examples)
    select = tuple(e for e in pool if e.split == select_split)
    if select_split == report_split:
        keys = sorted({_group_key(target, e) for e in select})
        order = np.random.default_rng(
            derive_seed(seed, "halve", target.name)
        ).permutation(len(keys))
        first = {keys[i] for i in order[: len(keys) // 2]}
        report = tuple(e for e in select if _group_key(target, e) not in first)
        select = tuple(e for e in select if _group_key(target, e) in first)
    else:
        report = tuple(e for e in pool if e.split == report_split)
    for name, part in (
        ("train", train),
        ("select", select),
        ("report", report),
    ):
        if pair_capacity(task, part) == 0:
            raise ValueError(
                f"task {task.name!r} has no pair in its {name} examples"
            )
    return TaskSplits(train, select, report)


def pair_capacity(task: ReadingTask, examples: Sequence[Example]) -> int:
    """Counts the pairs a task can draw from some examples.

    Args:
        task: The reading task.
        examples: Examples of the task, with the labels its pairing uses.

    Returns:
        The number of pairs: the rarer label for `label_contrastive`, the
        smaller of the target class and the others for `one_vs_rest`, and
        the groups with exactly one label-1 member and another member for
        `by_group`.
    """
    labels = np.array([e.label for e in examples], dtype=np.int64)
    if task.pairing == "label_contrastive":
        return int(min((labels == 1).sum(), (labels == 0).sum()))
    if task.pairing == "one_vs_rest":
        target = labels == task.target_class
        return int(min(target.sum(), (~target).sum()))
    groups: dict[str, list[int]] = {}
    for e in examples:
        groups.setdefault(str(e.meta["pair"]), []).append(e.label)
    return sum(1 for g in groups.values() if g.count(1) == 1 and len(g) >= 2)


def eval_pairs(
    task: ReadingTask, examples: Sequence[Example], n_eval_pairs: int | None
) -> int | None:
    """Caps the number of evaluation pairs by what the examples allow.

    Args:
        task: The reading task.
        examples: The select or report examples.
        n_eval_pairs: The requested number of pairs; None takes every pair.

    Returns:
        `min(n_eval_pairs, pair_capacity)`, or None when `n_eval_pairs` is
        None.
    """
    if n_eval_pairs is None:
        return None
    return min(n_eval_pairs, pair_capacity(task, examples))


def task_set(
    task: ReadingTask,
    variant: Variant,
    examples: Sequence[Example],
    *,
    n_pairs: int | None,
    seed: int,
) -> StimulusSet:
    """Renders examples through a variant and pairs them as the task asks.

    The pairing depends on the task and the seed only, never on the variant,
    so every evaluable variant of a task draws the same items from the same
    examples. A function variant renders persona pairs instead.

    Args:
        task: The reading task.
        variant: The variant to render.
        examples: The examples to draw from.
        n_pairs: Number of pairs; None takes every pair.
        seed: Seed of the draw.

    Returns:
        The stimulus set.
    """
    if not variant.evaluable:
        return build_stimulus_set(
            examples,
            variant.template,
            variant.chat,
            pairing="persona",
            n_pairs=n_pairs,
            seed=seed,
        )
    return build_stimulus_set(
        examples,
        variant.template,
        variant.chat,
        pairing=task.pairing,
        concept=variant.concept,
        target_class=task.target_class,
        group_ids=(
            [str(e.meta["pair"]) for e in examples]
            if task.pairing == "by_group"
            else None
        ),
        n_pairs=n_pairs,
        seed=seed,
    )


def prepare(
    loaded: LoadedModel,
    stimuli: StimulusSet,
    position: PositionStrategy,
    config: ReadingConfig,
) -> Prepared:
    """Collects the activations of a stimulus set.

    Args:
        loaded: The wrapped model.
        stimuli: The stimuli.
        position: Which token positions to read.
        config: Supplies the layers (None means 1 to L, as the reading sweep
            does), the BOS policy and the batch size.

    Returns:
        The stimuli with their activations.
    """
    layers = (
        tuple(range(1, loaded.n_layers + 1))
        if config.layers is None
        else config.layers
    )
    act = stimulus_activations(
        loaded,
        stimuli,
        layers=layers,
        position=position,
        bos=config.bos,
        batch_size=config.batch_size,
    )
    return stimuli, act


def _index(names: Sequence[str], name: str, what: str) -> int:
    if name not in names:
        raise ValueError(
            f"unknown {what} {name!r}, expected one of {tuple(names)}"
        )
    return names.index(name)


@dataclass(frozen=True, eq=False)
class VariantSweep:
    """Accuracy of every variant's reader on every variant's items.

    Attributes:
        splits: The evaluation splits, ("select", "report").
        train_variants: The variants the readers are trained on (V).
        eval_variants: The evaluable variants the readers are scored on (W).
        layers: Layer index of each entry of the last axis (K).
        accuracy: Array of shape (S, V, W, 3, N, K): split, trained on,
            evaluated on, `SERIES`, seed and layer.
        readers: The fitted "reader" series, (V, N).
    """

    splits: tuple[str, ...]
    train_variants: tuple[str, ...]
    eval_variants: tuple[str, ...]
    layers: tuple[int, ...]
    accuracy: NDArray[np.float64]
    readers: tuple[tuple[RepReader, ...], ...]

    @property
    def directions(self) -> NDArray[np.float32]:
        """The unit direction of every reader times its sign.

        Returns:
            Array of shape (V, N, K, d).
        """
        return np.stack(
            [
                np.stack([r.directions * r.signs[:, None] for r in row])
                for row in self.readers
            ]
        ).astype(np.float32)

    def at(self, split: str, train: str, evaluated: str) -> NDArray[np.float64]:
        """Returns the accuracy of one reader on one set of items.

        Args:
            split: A name from `splits`.
            train: A name from `train_variants`.
            evaluated: A name from `eval_variants`.

        Returns:
            Array of shape (3, N, K).

        Raises:
            ValueError: If a name is unknown.
        """
        return np.asarray(
            self.accuracy[
                _index(self.splits, split, "split"),
                _index(self.train_variants, train, "training variant"),
                _index(self.eval_variants, evaluated, "evaluation variant"),
            ]
        )

    def own(
        self, split: str, variant: str, *, reference: str = "paper"
    ) -> NDArray[np.float64]:
        """Returns the accuracy of a variant's reader on its own items.

        A variant that is only trained, never evaluated, is scored on the
        items of `reference`.

        Args:
            split: A name from `splits`.
            variant: A name from `train_variants`.
            reference: The evaluation variant a train-only variant is scored
                on.

        Returns:
            Array of shape (3, N, K).

        Raises:
            ValueError: If a name is unknown.
        """
        _index(self.train_variants, variant, "training variant")
        evaluated = variant if variant in self.eval_variants else reference
        return self.at(split, variant, evaluated)


def variant_sweep(
    train: Mapping[str, Sequence[Prepared]],
    evals: Mapping[str, Mapping[str, Prepared]],
    *,
    method: Method,
    preprocess: Preprocess,
    seed: int,
) -> VariantSweep:
    """Scores the reader of every training variant on every evaluation set.

    It runs one `sweep_from_activations` per training variant over every
    evaluation set, named `<split>/<variant>`, so the random and shuffled
    series come with it and the diagonal is exactly a plain sweep.

    Args:
        train: One prepared training set per seed, for each variant.
        evals: The prepared evaluation set of each evaluable variant, for
            each split.
        method: Direction finder of the `reader` series.
        preprocess: Vector preparation, see `fit_direction`.
        seed: Base seed of the fits and the shuffles.

    Returns:
        The accuracy of every reader on every set.

    Raises:
        ValueError: If there is no training variant or no evaluation
            variant, or the splits do not hold the same evaluation variants.
    """
    if not train:
        raise ValueError("train needs at least one training variant")
    splits = tuple(evals)
    eval_variants = tuple(evals[splits[0]]) if splits else ()
    if not eval_variants:
        raise ValueError("evals needs at least one evaluation variant")
    if any(tuple(evals[split]) != eval_variants for split in splits):
        raise ValueError("every split must hold the same evaluation variants")
    flat = {
        f"{split}/{name}": evals[split][name]
        for split in splits
        for name in eval_variants
    }
    rows = []
    readers = []
    layers: tuple[int, ...] = ()
    for sets in train.values():
        result = sweep_from_activations(
            sets, flat, method=method, preprocess=preprocess, seed=seed
        )
        layers = result.layers
        rows.append(
            np.stack(
                [
                    np.stack(
                        [result.per_seed(f"{s}/{w}") for w in eval_variants]
                    )
                    for s in splits
                ]
            )
        )
        readers.append(result.readers)
    return VariantSweep(
        splits,
        tuple(train),
        eval_variants,
        layers,
        np.stack(rows, axis=1),
        tuple(readers),
    )


def variant_cosines(directions: NDArray[np.float32]) -> NDArray[np.float64]:
    """Computes the cosines between the readers of every pair of variants.

    Off the diagonal it is the mean over seeds of the cosine between the
    readers of two variants fitted on the same seed (the same training
    items). On the diagonal it is the mean over pairs of seeds of the cosine
    between a variant's own readers: its stability, the ceiling a variant can
    reach against another.

    Args:
        directions: Oriented unit directions of shape (V, N, K, d), such as
            `VariantSweep.directions`.

    Returns:
        Array of shape (K, V, V).

    Raises:
        ValueError: If `directions` is not 4-D or has fewer than two seeds.
    """
    unit = np.asarray(directions, dtype=np.float64)
    if unit.ndim != 4 or unit.shape[1] < 2:
        raise ValueError(
            "directions must have shape (V, N, K, d) with N >= 2, "
            f"got {unit.shape}"
        )
    n_variants, n_seeds = unit.shape[:2]
    cosines = np.einsum("vnkd,wnkd->kvw", unit, unit) / n_seeds
    first, second = np.triu_indices(n_seeds, k=1)
    for v in range(n_variants):
        own = np.einsum("nkd,mkd->knm", unit[v], unit[v])
        cosines[:, v, v] = own[:, first, second].mean(axis=1)
    return cosines


def load_sources(
    sources: Sequence[TaskSource],
    *,
    cache: Path | None = None,
    offline: bool = False,
) -> dict[str, list[Example]]:
    """Loads the datasets of some task sources, each once.

    Args:
        sources: Source names; a repeated name is loaded once.
        cache: Cache directory; None uses the loaders' default.
        offline: If True, read only from the cache.

    Returns:
        The examples of each source, in the order of first appearance.

    Raises:
        ValueError: If a source is unknown (checked before any load).
    """
    loaders: dict[str, Callable[..., list[Example]]] = {
        "sst2": load_sst2,
        "facts": load_facts,
        "ethics_utility": load_ethics_utility,
        "emotions": load_emotions,
        "geometry_of_truth": load_geometry_of_truth,
    }
    unknown = [source for source in sources if source not in loaders]
    if unknown:
        raise ValueError(
            f"unknown source {unknown[0]!r}, expected one of {list(loaders)}"
        )
    return {
        source: loaders[source](cache=cache, offline=offline)
        for source in dict.fromkeys(sources)
    }
