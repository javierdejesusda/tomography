"""I1: data, prompts and the held-in stages of the invariance reader.

Implements the preregistration `docs/preregistration-invariance-reader.md`:
the partitions and draws of section 2.4, the frozen wordings and renderings
of section 3, the stage 0 prompt set of section 6.1 (and the stage 2 prompt
set of section 7.1, defined but not run here) and the stage 0 activation
cache of section 10. Stage 1 reads that cache, without loading a model.
Section numbers in the docstrings refer to that file.
"""

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
import io
import math
import multiprocessing
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import cast
from typing import get_args

import numpy as np
from numpy.typing import NDArray
import torch

from tomography.activations import collect_activations
from tomography.chain import THREAD_VARIABLES
from tomography.datasets.ethics import load_ethics_utility
from tomography.datasets.facts import load_facts
from tomography.datasets.geometry_of_truth import SOURCE as GOT_SOURCE
from tomography.datasets.geometry_of_truth import load_geometry_of_truth
from tomography.datasets.records import Example
from tomography.datasets.sources import fetch
from tomography.datasets.sst2 import load_sst2
from tomography.experiments.base import ExperimentConfig
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.m2_readers import POLARITY_TASKS
from tomography.experiments.variants import READING_TASKS
from tomography.experiments.variants import ReadingTask
from tomography.experiments.variants import eval_pairs
from tomography.experiments.variants import eval_seed
from tomography.experiments.variants import task_set
from tomography.experiments.variants import task_splits
from tomography.experiments.variants import template_variants
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.reading import invariance
from tomography.reading import invariance_stats
from tomography.templates import ChatFormat
from tomography.templates import Span

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

TaskName = Literal["sentiment", "truth", "truth_negations", "utility"]
TASKS: tuple[TaskName, ...] = get_args(TaskName)
Placement = Literal["pre", "head", "post"]
LevelRule = Literal["random", "against"]
WordingSet = Literal["fit", "score", "test"]
NEGATION_SUBSETS: tuple[str, ...] = POLARITY_TASKS["truth_polarity"]
_QUOTED = re.compile(r"'([^']+)'")

# The fallback date of the Llama 3.2 chat template; passing it keeps the
# rendered prompts independent of the day of the run.
FIXED_DATE_STRING = "26 Jul 2024"

_SOURCE_TASK: Mapping[str, str] = MappingProxyType(
    {
        "sentiment": "sentiment",
        "truth": "truth_common",
        "truth_negations": "truth_cities",
        "utility": "utility",
    }
)


@dataclass(frozen=True)
class Wording:
    """One frozen wording of section 3.

    Attributes:
        name: The identifier used in the preregistration, such as "F-D1".
        wording_set: "fit" (3.4), "score" (3.5) or "test" (3.3 and 3.6).
        text: The user text of a template variant, or the decoy line with
            its `{noun}` and `{level}` slots.
        placement: Where a decoy line goes; None for a template variant.
        levels: The (low, high) level words of a decoy.
    """

    name: str
    wording_set: WordingSet
    text: str
    placement: Placement | None = None
    levels: tuple[str, str] = ("low", "high")


FIT_TEMPLATES: tuple[Wording, ...] = (
    Wording(
        "F-T1",
        "fit",
        "Here is a passage.\n{stimulus}\n"
        "The level of {concept} in this passage is",
    ),
    Wording(
        "F-T2",
        "fit",
        "Text: {stimulus}\nTask: judge the {concept} of the text.\n"
        "Judgment: the {concept} is",
    ),
    Wording(
        "F-T3",
        "fit",
        "Evaluate the following with respect to {concept}:\n{stimulus}\n"
        "Evaluation: the {concept} is",
    ),
)
FIT_DECOYS: tuple[Wording, ...] = (
    Wording(
        "F-D1",
        "fit",
        "A colleague told me the {noun} here is {level}, though I have not "
        "checked it myself.",
        "head",
    ),
    Wording(
        "F-D2",
        "fit",
        "Before you answer: my first impression was that the {noun} of this "
        "one is {level}.",
        "pre",
    ),
    Wording(
        "F-D3",
        "fit",
        "Note from an earlier reader: the {noun} of this text is {level}.",
        "pre",
        ("weak", "strong"),
    ),
)
SCORE_TEMPLATES: tuple[Wording, ...] = (
    Wording(
        "S-T1",
        "score",
        "Take a look at this excerpt:\n{stimulus}\n"
        "The degree of {concept} it shows is",
    ),
    Wording(
        "S-T2",
        "score",
        "{stimulus}\nGiven the text above, determine its {concept}.\n"
        "The {concept} of the text is",
    ),
)
SCORE_DECOYS: tuple[Wording, ...] = (
    Wording(
        "S-D1",
        "score",
        "Someone on the forum insisted that the {noun} of the next text is "
        "{level}.",
        "pre",
    ),
    Wording(
        "S-D2",
        "score",
        "For what it is worth, an automatic tool scored the {noun} below as "
        "{level}.",
        "head",
    ),
)
M2_TEST_DECOY = Wording(
    "M2-D",
    "test",
    "I think the amount of {noun} in the scenario is {level}, but I'm "
    "curious to hear what you think.",
    "pre",
)
TEST_PARAPHRASES: tuple[Wording, ...] = (
    Wording(
        "P1",
        "test",
        "Rate how much {concept} the following text expresses:\n"
        "{stimulus}\nThe amount of {concept} is",
    ),
    Wording(
        "P2",
        "test",
        "How much {concept} is there in this text?\n{stimulus}\n"
        "The amount of {concept} is",
    ),
    Wording(
        "P3",
        "test",
        "Read the text below and assess its {concept}.\n{stimulus}\n"
        "Its {concept} is",
    ),
)
TEST_DECOYS: tuple[Wording, ...] = (
    Wording(
        "T-D1",
        "test",
        "Everyone I asked says the {noun} of what follows is {level}.",
        "pre",
    ),
    Wording(
        "T-D2",
        "test",
        "(A previous annotator marked the {noun} of the text above as "
        "{level}.)",
        "post",
    ),
    Wording(
        "T-D3",
        "test",
        "An expert reviewer is certain that the {noun} of the text below is "
        "{level}.",
        "head",
        ("minimal", "maximal"),
    ),
)
WORDINGS: Mapping[str, Wording] = MappingProxyType(
    {
        wording.name: wording
        for wording in (
            *FIT_TEMPLATES,
            *FIT_DECOYS,
            *SCORE_TEMPLATES,
            *SCORE_DECOYS,
            M2_TEST_DECOY,
            *TEST_PARAPHRASES,
            *TEST_DECOYS,
        )
    }
)


@dataclass(frozen=True)
class RenderedText:
    """A rendered prompt.

    Attributes:
        text: The full prompt text, chat format included.
        stimulus_span: The character span of the stimulus in `text`.
        level: The decoy level word, None without a decoy.
    """

    text: str
    stimulus_span: Span
    level: str | None


def reading_task(task: str) -> ReadingTask:
    """Returns the repository task whose template, concept and noun I1 uses.

    Args:
        task: One of `TASKS`.

    Returns:
        The `READING_TASKS` entry of section 2.2.

    Raises:
        ValueError: If the task is unknown.
    """
    if task not in _SOURCE_TASK:
        raise ValueError(f"unknown task {task!r}, expected one of {TASKS}")
    return READING_TASKS[_SOURCE_TASK[task]]


def wrapper_names(
    task: str, wording_set: Literal["fit", "score"]
) -> tuple[str, ...]:
    """Lists the wrappers of a held-in set for a task (section 3.7).

    Args:
        task: One of `TASKS`.
        wording_set: "fit" or "score".

    Returns:
        The template variants then the decoys; utility gets decoys only.
    """
    reading_task(task)
    if wording_set == "fit":
        templates, decoys = FIT_TEMPLATES, FIT_DECOYS
    else:
        templates, decoys = SCORE_TEMPLATES, SCORE_DECOYS
    if task == "utility":
        templates = ()
    return tuple(w.name for w in (*templates, *decoys))


def decoy_level(
    wording: Wording, example: Example, rule: LevelRule, *, seed: int = 0
) -> str:
    """Returns the level word of a decoy for one text (section 3.2).

    Args:
        wording: A decoy wording.
        example: The text, with a binary label for `against`.
        rule: "random" draws the level from a generator seeded by the base
            seed, the wording and the uid, independent of the label;
            "against" gives label 1 the low level and label 0 the high one.
        seed: The base seed.

    Returns:
        One of `wording.levels`.

    Raises:
        ValueError: If the rule is unknown or `against` meets a non-binary
            label.
    """
    low, high = wording.levels
    if rule == "against":
        if example.label not in (0, 1):
            raise ValueError(f"example {example.uid!r} needs a binary label")
        return low if example.label == 1 else high
    if rule != "random":
        raise ValueError(f"unknown level rule {rule!r}")
    draw = np.random.default_rng(
        derive_seed(seed, "i1", "level", wording.name, example.uid)
    )
    return wording.levels[int(draw.integers(2))]


def _with_decoy(user: str, line: str, placement: Placement) -> str:
    """Inserts a decoy line into the user text of the clean template."""
    if placement == "head":
        return f"{line}\n{user}"
    at = user.index("{stimulus}")
    if placement == "pre":
        start = user.rfind("\n", 0, at) + 1
        return f"{user[:start]}{line}\n{user[start:]}"
    end = user.find("\n", at)
    if end == -1:
        return f"{user}\n{line}"
    return f"{user[:end]}\n{line}{user[end:]}"


def render_text(
    task: str,
    wording: str | None,
    example: Example,
    chat: ChatFormat,
    *,
    rule: LevelRule | None = None,
    stage: int = 0,
    evaluation: bool = False,
    seed: int = 0,
) -> RenderedText:
    """Renders one text under the clean template or one wording (section 3).

    Decoys are added to the clean template only, as their own line in the
    user turn; template variants replace the clean user text and carry no
    decoy. The decoy level fills the `{question}` slot of the repository
    templates, so the stimulus span comes from `ConceptTemplate.render`.

    Args:
        task: One of `TASKS`.
        wording: A key of `WORDINGS`, or None for the clean template.
        example: The text.
        chat: The chat format of the checkpoint.
        rule: The level rule of a decoy; None for anything else.
        stage: The stage the prompt is rendered for; test wordings need 2
            (section 8.2).
        evaluation: Whether the prompt belongs to an evaluation set; an
            `against` decoy needs it (sections 3.2 and 8.2).
        seed: The base seed of the `random` levels.

    Returns:
        The prompt, its stimulus span and its decoy level.

    Raises:
        ValueError: If the task or wording is unknown, a test wording is
            rendered before stage 2, utility gets a template variant, the
            level rule does not fit the wording, or `against` is asked for
            outside an evaluation set.
    """
    source = reading_task(task)
    template = source.template
    level = None
    if wording is None:
        if rule is not None:
            raise ValueError("the clean template takes no level rule")
    else:
        if wording not in WORDINGS:
            raise ValueError(f"unknown wording {wording!r}")
        frozen = WORDINGS[wording]
        if frozen.wording_set == "test" and stage < 2:
            raise ValueError(
                f"{wording} is a test wording, rendered only at stage 2"
            )
        if frozen.placement is None:
            if rule is not None:
                raise ValueError(f"template {wording} takes no level rule")
            if task == "utility":
                raise ValueError("utility uses decoy wrappers only")
            template = replace(template, name=wording, user=frozen.text)
        else:
            if rule is None:
                raise ValueError(f"decoy {wording} needs a level rule")
            if rule == "against" and not evaluation:
                raise ValueError("against decoys exist only in evaluation sets")
            level = decoy_level(frozen, example, rule, seed=seed)
            line = frozen.text.replace("{noun}", source.noun).replace(
                "{level}", "{question}"
            )
            template = replace(
                template,
                name=wording,
                user=_with_decoy(template.user, line, frozen.placement),
            )
    rendered = template.render(
        example.text, source.concept or "", chat, question=level
    )
    assert rendered.response_span is not None
    return RenderedText(rendered.text, rendered.response_span, level)


def chat_from_tokenizer(tokenizer: "PreTrainedTokenizerBase") -> ChatFormat:
    """Builds a chat template format that does not depend on the run date.

    As `ChatFormat.from_tokenizer` (generation prompt, thinking disabled),
    and passes `date_string`, which the Llama 3.2 template would otherwise
    take from the current date.

    Args:
        tokenizer: A tokenizer that defines a chat template.

    Returns:
        A format of the template kind.
    """

    def render(messages: list[dict[str, str]]) -> str:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            date_string=FIXED_DATE_STRING,
        )
        if not isinstance(prompt, str):
            raise TypeError("the chat template did not return text")
        return prompt

    return ChatFormat("template", render_fn=render)


def chat_format(loaded: LoadedModel) -> ChatFormat:
    """Returns the chat format of a checkpoint (section 2.1).

    Args:
        loaded: The wrapped model.

    Returns:
        Its chat template iff the checkpoint is post-trained, else the plain
        `USER:` and `ASSISTANT:` format.
    """
    if loaded.checkpoint is not None and loaded.checkpoint.chat_format == (
        "template"
    ):
        return chat_from_tokenizer(loaded.tokenizer)
    return ChatFormat("plain")


BlockName = Literal["report", "validation", "fit", "pool"]
BLOCKS: tuple[BlockName, ...] = get_args(BlockName)
SENTIMENT_MIN_WORDS = 3
REPORT_MIN_WORDS = 10
M2_EVAL_PAIRS = 256
M2_SEED = 0


@dataclass(frozen=True)
class BlockSizes:
    """The block sizes of section 2.4.

    Attributes:
        report: Pairs of the reserved report block.
        validation: Pairs of the validation block.
        fit: Pairs of the fit reservoir.
        pool: Texts of the unlabelled pool.
    """

    report: int = 256
    validation: int = 200
    fit: int = 192
    pool: int = 640


PREREGISTERED_SIZES = BlockSizes()


@dataclass(frozen=True)
class Block:
    """One block of a partition.

    Attributes:
        name: One of `BLOCKS`.
        keys: The group keys the block holds, in hash order. A key belongs
            to one block only, even when some of its texts are not used.
        examples: The texts used: both members of each pair in pair order,
            or the pool texts.
        pairs: The (label 1 uid, label 0 uid) pairs; empty for the pool.
    """

    name: BlockName
    keys: tuple[str, ...]
    examples: tuple[Example, ...]
    pairs: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class TaskPartition:
    """The blocks of one task and the counts the preregistration states.

    Attributes:
        task: One of `TASKS`.
        blocks: The blocks by name, in `BLOCKS` order.
        counts: Source and filter counts, such as "burned_candidates".
        notes: Logged events: fallback group keys and a short pool.
    """

    task: str
    blocks: Mapping[str, Block]
    counts: Mapping[str, int]
    notes: tuple[str, ...]

    def item_ids(self) -> dict[str, dict[str, list[Any]]]:
        """Returns the keys, uids and pairs of every block, for the run dir.

        Returns:
            By block name, the lists "keys", "uids" and "pairs".
        """
        return {
            name: {
                "keys": list(block.keys),
                "uids": [e.uid for e in block.examples],
                "pairs": [list(pair) for pair in block.pairs],
            }
            for name, block in self.blocks.items()
        }


def hash_order(task: str, keys: Iterable[str], *, seed: int) -> list[str]:
    """Orders group keys by a seeded hash (section 2.4).

    Args:
        task: The task, part of the hash.
        keys: Group keys; repeats are dropped.
        seed: The base seed.

    Returns:
        The distinct keys sorted by `derive_seed(seed, "i1", "partition",
        task, key)`, ties broken by the key.
    """
    return sorted(
        set(keys),
        key=lambda key: (derive_seed(seed, "i1", "partition", task, key), key),
    )


def _rows(members: Sequence[Example]) -> list[tuple[Example, Example]]:
    """Pairs the two scenarios of each source row, label 1 first."""
    by_row: dict[str, list[Example]] = {}
    for example in members:
        by_row.setdefault(str(example.meta["pair"]), []).append(example)
    rows = []
    for row in by_row.values():
        first = [e for e in row if e.label == 1]
        second = [e for e in row if e.label == 0]
        if len(first) == 1 and len(second) == 1:
            rows.append((first[0], second[0]))
    return rows


def fill_pairs(
    name: BlockName,
    order: Sequence[str],
    groups: Mapping[str, Sequence[Example]],
    n_pairs: int,
    *,
    by_row: bool = False,
) -> Block:
    """Fills a labelled block by walking group keys in hash order.

    A group joins the block whole. For label-contrastive tasks it joins
    when one of its texts has a label still below `n_pairs`, and the pairs
    are the i-th label 1 text with the i-th label 0 text in block order. For
    utility (`by_row`) it joins while the block has fewer than `n_pairs`
    rows, and each source row is one pair. Texts of a joined group beyond
    the quota stay in the block unused.

    Args:
        name: The block name.
        order: Group keys in hash order.
        groups: The texts of each key, in source order.
        n_pairs: The number of pairs.
        by_row: Pair the two scenarios of each source row.

    Returns:
        The block.

    Raises:
        ValueError: If the groups hold fewer than `n_pairs` pairs.
    """
    keys: list[str] = []
    first: list[Example] = []
    second: list[Example] = []
    rows: list[tuple[Example, Example]] = []
    for key in order:
        members = groups[key]
        if by_row:
            if len(rows) >= n_pairs:
                break
            rows.extend(_rows(members))
        else:
            if len(first) >= n_pairs and len(second) >= n_pairs:
                break
            if not any(
                (e.label == 1 and len(first) < n_pairs)
                or (e.label == 0 and len(second) < n_pairs)
                for e in members
            ):
                continue
            first.extend(e for e in members if e.label == 1)
            second.extend(e for e in members if e.label == 0)
        keys.append(key)
    pairs = rows if by_row else list(zip(first, second, strict=False))
    if len(pairs) < n_pairs:
        raise ValueError(
            f"the {name} block has {len(pairs)} of {n_pairs} pairs"
        )
    pairs = pairs[:n_pairs]
    return Block(
        name,
        tuple(keys),
        tuple(e for pair in pairs for e in pair),
        tuple((a.uid, b.uid) for a, b in pairs),
    )


def fill_pool(
    name: BlockName,
    order: Sequence[str],
    groups: Mapping[str, Sequence[Example]],
    n_texts: int,
) -> Block:
    """Takes texts in hash order without reading their labels (section 4).

    Args:
        name: The block name.
        order: Group keys in hash order.
        groups: The texts of each key, in source order.
        n_texts: The number of texts; fewer are taken when fewer remain.

    Returns:
        The block, without pairs.
    """
    keys: list[str] = []
    texts: list[Example] = []
    for key in order:
        if len(texts) >= n_texts:
            break
        keys.append(key)
        texts.extend(groups[key][: n_texts - len(texts)])
    return Block(name, tuple(keys), tuple(texts), ())


def _without(
    groups: Mapping[str, Sequence[Example]], block: Block
) -> dict[str, Sequence[Example]]:
    taken = set(block.keys)
    return {k: v for k, v in groups.items() if k not in taken}


def _finish(
    task: str,
    blocks: Sequence[Block],
    rest: Mapping[str, Sequence[Example]],
    counts: dict[str, int],
    sizes: BlockSizes,
    seed: int,
) -> TaskPartition:
    """Adds the pool to the labelled blocks and builds the partition."""
    notes = []
    pool = fill_pool(
        "pool", hash_order(task, rest, seed=seed), rest, sizes.pool
    )
    counts["pool_texts"] = len(pool.examples)
    if len(pool.examples) < sizes.pool:
        notes.append(
            f"{task}: the pool takes all {len(pool.examples)} remaining texts "
            f"(of {sizes.pool})"
        )
    by_name = {block.name: block for block in (*blocks, pool)}
    return TaskPartition(
        task,
        MappingProxyType({name: by_name[name] for name in BLOCKS}),
        MappingProxyType(dict(counts)),
        tuple(notes),
    )


def normalize_words(text: str) -> str:
    """Lower-cases a text and collapses its whitespace (section 2.4).

    Args:
        text: The text.

    Returns:
        The lower-cased words joined by single spaces.
    """
    return " ".join(text.lower().split())


def normalize_statement(text: str) -> str:
    """Normalizes a statement: `normalize_words`, trailing period removed.

    Args:
        text: The statement.

    Returns:
        The normalized statement.
    """
    words = normalize_words(text)
    return words[:-1] if words.endswith(".") else words


def _runs(words: str, lengths: Iterable[int] | None = None) -> set[str]:
    """Returns the contiguous runs of whole words of a normalized text."""
    tokens = words.split()
    sizes = range(1, len(tokens) + 1) if lengths is None else lengths
    return {
        " ".join(tokens[i : i + size])
        for size in sizes
        for i in range(len(tokens) - size + 1)
    }


def contains_words(text: str, part: str) -> bool:
    """Whether `part` is a contiguous run of whole words of `text`.

    Both texts are normalized with `normalize_words` first.

    Args:
        text: The containing text.
        part: The contained text.

    Returns:
        True when the normalized `part` is a run of the normalized `text`.
    """
    return normalize_words(part) in _runs(normalize_words(text))


class _Overlap:
    """Tests whether a text contains, or is contained in, reference texts."""

    def __init__(self, texts: Iterable[str]) -> None:
        self._texts = {normalize_words(t) for t in texts}
        self._lengths = sorted({len(t.split()) for t in self._texts})
        self._runs: set[str] = set()
        for text in self._texts:
            self._runs |= _runs(text)

    def __call__(self, text: str) -> bool:
        words = normalize_words(text)
        if words in self._runs:
            return True
        return not self._texts.isdisjoint(_runs(words, self._lengths))


def _singletons(examples: Iterable[Example]) -> dict[str, Sequence[Example]]:
    return {e.uid: (e,) for e in examples}


def _held_in_blocks(
    task: str,
    groups: Mapping[str, Sequence[Example]],
    sizes: BlockSizes,
    seed: int,
    *,
    by_row: bool = False,
) -> tuple[Block, Block, dict[str, Sequence[Example]]]:
    """Fills validation then fit from the same groups; returns the rest."""
    validation = fill_pairs(
        "validation",
        hash_order(task, groups, seed=seed),
        groups,
        sizes.validation,
        by_row=by_row,
    )
    rest = _without(groups, validation)
    fit = fill_pairs(
        "fit",
        hash_order(task, rest, seed=seed),
        rest,
        sizes.fit,
        by_row=by_row,
    )
    return validation, fit, _without(rest, fit)


def partition_sentiment(
    examples: Sequence[Example],
    *,
    sizes: BlockSizes = PREREGISTERED_SIZES,
    seed: int = 0,
) -> TaskPartition:
    """Partitions SST-2 for the sentiment task (section 2.4).

    The group key is the uid; stage 0 and 1 items are train texts of at
    least 3 words. Report candidates are train texts of at least 10 words;
    a candidate that contains, or is contained in, a validation sentence is
    removed from the candidates. After the report block, every train item
    that contains or is contained in a report text is removed; after the
    validation block, the same holds for validation texts. Containment is
    on `normalize_words` text, as a contiguous run of whole words.

    Args:
        examples: The SST-2 loader output (train and val splits).
        sizes: The block sizes.
        seed: The base seed of the hash.

    Returns:
        The partition. Counts: "train", "candidates_positive" and
        "candidates_negative" (before the burn), "burned_candidates",
        "removed_by_report", "removed_by_validation", "pool_texts".

    Raises:
        ValueError: If a block cannot be filled.
    """
    task = "sentiment"
    train = [
        e
        for e in examples
        if e.split == "train" and len(e.text.split()) >= SENTIMENT_MIN_WORDS
    ]
    candidates = [e for e in train if len(e.text.split()) >= REPORT_MIN_WORDS]
    burned_by = _Overlap(e.text for e in examples if e.split == "val")
    kept = [e for e in candidates if not burned_by(e.text)]
    counts = {
        "train": len(train),
        "candidates_positive": sum(e.label == 1 for e in candidates),
        "candidates_negative": sum(e.label == 0 for e in candidates),
        "burned_candidates": len(candidates) - len(kept),
    }
    report_groups = _singletons(kept)
    report = fill_pairs(
        "report",
        hash_order(task, report_groups, seed=seed),
        report_groups,
        sizes.report,
    )
    overlaps_report = _Overlap(e.text for e in report.examples)
    rest = [e for e in train if not overlaps_report(e.text)]
    counts["removed_by_report"] = len(train) - len(rest)
    groups = _singletons(rest)
    validation = fill_pairs(
        "validation",
        hash_order(task, groups, seed=seed),
        groups,
        sizes.validation,
    )
    overlaps_validation = _Overlap(e.text for e in validation.examples)
    left = _without(groups, validation)
    after = {
        k: v for k, v in left.items() if not overlaps_validation(v[0].text)
    }
    counts["removed_by_validation"] = len(left) - len(after)
    fit = fill_pairs(
        "fit", hash_order(task, after, seed=seed), after, sizes.fit
    )
    return _finish(
        task,
        (report, validation, fit),
        _without(after, fit),
        counts,
        sizes,
        seed,
    )


def partition_truth(
    common: Sequence[Example],
    facts: Sequence[Example],
    *,
    sizes: BlockSizes = PREREGISTERED_SIZES,
    seed: int = 0,
) -> TaskPartition:
    """Partitions `common_claim_true_false` for the truth task (section 2.4).

    Statements equal to a facts statement under `normalize_statement` are
    removed. Repeats under the same normalization keep their first row, and
    a statement with copies in two source splits is removed from both. The
    report block comes from the test split, the other blocks from the train
    split; the group key is the uid.

    Args:
        common: The `common_claim_true_false` rows of the loader.
        facts: The facts loader output.
        sizes: The block sizes.
        seed: The base seed of the hash.

    Returns:
        The partition. Counts: "facts_matches", "train", "train_true",
        "test", "test_true", "pool_texts".

    Raises:
        ValueError: If a block cannot be filled.
    """
    task = "truth"
    burned = {normalize_statement(e.text) for e in facts}
    rows = [e for e in common if normalize_statement(e.text) not in burned]
    copies: dict[str, list[Example]] = {}
    for example in rows:
        copies.setdefault(normalize_statement(example.text), []).append(example)
    kept = [v[0] for v in copies.values() if len({e.split for e in v}) == 1]
    train = [e for e in kept if e.split == "train"]
    test = [e for e in kept if e.split == "test"]
    counts = {
        "facts_matches": len(common) - len(rows),
        "train": len(train),
        "train_true": sum(e.label == 1 for e in train),
        "test": len(test),
        "test_true": sum(e.label == 1 for e in test),
    }
    report_groups = _singletons(test)
    report = fill_pairs(
        "report",
        hash_order(task, report_groups, seed=seed),
        report_groups,
        sizes.report,
    )
    validation, fit, rest = _held_in_blocks(
        task, _singletons(train), sizes, seed
    )
    return _finish(task, (report, validation, fit), rest, counts, sizes, seed)


def negation_subjects(
    examples: Sequence[Example], cities: Mapping[str, Sequence[str]]
) -> tuple[dict[str, str], list[str]]:
    """Returns the subject of every negations statement (section 2.4).

    The subject is the `city` column of the city files (looked up by the
    row index in the uid) or the first quoted word of a translation
    statement. When neither is found, the key is the uid and a note logs
    the fallback and its reason.

    Args:
        examples: The rows of the four negation subsets.
        cities: The `city` column of each city file, by subset, in row
            order.

    Returns:
        The group key of each uid ("city:<name>", "es:<word>" or
        "uid:<uid>") and the fallback notes.
    """
    keys: dict[str, str] = {}
    notes: list[str] = []
    for example in examples:
        subset = str(example.meta.get("subset"))
        reason = ""
        if subset in cities:
            index = example.uid.removeprefix(f"got-{subset}-")
            column = cities[subset]
            if (
                index.isdigit()
                and int(index) < len(column)
                and column[int(index)].strip()
            ):
                keys[example.uid] = f"city:{column[int(index)].strip()}"
                continue
            reason = "no city in the city column"
        else:
            match = _QUOTED.search(example.text)
            if match:
                keys[example.uid] = f"es:{match.group(1)}"
                continue
            reason = "no quoted word in the statement"
        keys[example.uid] = f"uid:{example.uid}"
        notes.append(f"{example.uid}: {reason}; the group key is the uid")
    return keys, notes


def partition_negations(
    examples: Sequence[Example],
    subjects: Mapping[str, str],
    *,
    sizes: BlockSizes = PREREGISTERED_SIZES,
    seed: int = 0,
) -> TaskPartition:
    """Partitions the four negation subsets by subject (section 2.4).

    Repeats under `normalize_statement` collapse to their first row, and a
    train statement with a copy in the validation or test split is removed
    everywhere, so a train statement is kept iff all its copies are in the
    train split. The report block is drawn only from subjects whose every
    source row is in the train split; the other blocks take the remaining
    subjects of the kept train statements.

    Args:
        examples: The rows of the four subsets, in loader order.
        subjects: The group key of each uid, from `negation_subjects`.
        sizes: The block sizes.
        seed: The base seed of the hash.

    Returns:
        The partition. Counts: "repeated_statements",
        "train_repeating_eval", "train_distinct", "eligible_subjects",
        "eligible_statements", "eligible_city_statements",
        "eligible_spanish_statements", "eligible_pairs", "pool_texts".

    Raises:
        ValueError: If a block cannot be filled.
    """
    task = "truth_negations"
    copies: dict[str, list[Example]] = {}
    for example in examples:
        copies.setdefault(normalize_statement(example.text), []).append(example)
    splits_of = {k: {e.split for e in v} for k, v in copies.items()}
    kept = [v[0] for k, v in copies.items() if splits_of[k] == {"train"}]
    subject_splits: dict[str, set[str]] = {}
    for example in examples:
        subject_splits.setdefault(subjects[example.uid], set()).add(
            example.split
        )
    groups: dict[str, list[Example]] = {}
    for example in kept:
        groups.setdefault(subjects[example.uid], []).append(example)
    eligible = {
        k: v for k, v in groups.items() if subject_splits[k] == {"train"}
    }
    statements = [e for v in eligible.values() for e in v]
    counts = {
        "repeated_statements": sum(len(v) > 1 for v in copies.values()),
        "train_repeating_eval": sum(
            e.split == "train"
            and splits_of[normalize_statement(e.text)] != {"train"}
            for e in examples
        ),
        "train_distinct": len(kept),
        "eligible_subjects": len(eligible),
        "eligible_statements": len(statements),
        "eligible_city_statements": sum(
            "cities" in str(e.meta.get("subset")) for e in statements
        ),
        "eligible_spanish_statements": sum(
            "sp_en" in str(e.meta.get("subset")) for e in statements
        ),
        "eligible_pairs": min(
            sum(e.label == 1 for e in statements),
            sum(e.label == 0 for e in statements),
        ),
    }
    report = fill_pairs(
        "report", hash_order(task, eligible, seed=seed), eligible, sizes.report
    )
    validation, fit, rest = _held_in_blocks(
        task, _without(groups, report), sizes, seed
    )
    return _finish(task, (report, validation, fit), rest, counts, sizes, seed)


def partition_utility(
    examples: Sequence[Example],
    m2_report: frozenset[str],
    *,
    sizes: BlockSizes = PREREGISTERED_SIZES,
    seed: int = 0,
) -> TaskPartition:
    """Partitions ETHICS utilitarianism by linked group (section 2.4).

    Rows are linked by a shared scenario text, directly or through a chain
    of rows, over the source train and test files together; the group key
    is the first row of each linked group. A test row is a report candidate
    iff its group holds no row of the source train file (train or val
    split) and none of `m2_report`. Groups are never split. The other
    blocks take the train-split rows.

    Args:
        examples: The utility loader output.
        m2_report: The row ids (`meta["pair"]`) of the M2 report pairs.
        sizes: The block sizes.
        seed: The base seed of the hash.

    Returns:
        The partition. Counts: "test_pairs", "report_candidate_pairs",
        "report_candidate_groups", "pool_texts".

    Raises:
        ValueError: If a block cannot be filled.
    """
    task = "utility"
    rows: dict[str, list[Example]] = {}
    for example in examples:
        rows.setdefault(str(example.meta["pair"]), []).append(example)
    parent: dict[str, str] = {}

    def find(text: str) -> str:
        root = parent.setdefault(text, text)
        while parent[root] != root:
            root = parent[root]
        while parent[text] != root:
            parent[text], text = root, parent[text]
        return root

    for members in rows.values():
        for other in members[1:]:
            parent[find(members[0].text)] = find(other.text)
    key_of_root: dict[str, str] = {}
    group_of = {
        pair: key_of_root.setdefault(find(members[0].text), pair)
        for pair, members in rows.items()
    }
    excluded = {
        group_of[pair]
        for pair, members in rows.items()
        if members[0].split != "test" or pair in m2_report
    }
    test_groups: dict[str, list[Example]] = {}
    train_groups: dict[str, list[Example]] = {}
    for pair, members in rows.items():
        split = members[0].split
        if split == "test" and group_of[pair] not in excluded:
            test_groups.setdefault(group_of[pair], []).extend(members)
        elif split == "train":
            train_groups.setdefault(group_of[pair], []).extend(members)
    counts = {
        "test_pairs": sum(m[0].split == "test" for m in rows.values()),
        "report_candidate_pairs": sum(
            len(_rows(v)) for v in test_groups.values()
        ),
        "report_candidate_groups": len(test_groups),
    }
    report = fill_pairs(
        "report",
        hash_order(task, test_groups, seed=seed),
        test_groups,
        sizes.report,
        by_row=True,
    )
    validation, fit, rest = _held_in_blocks(
        task, train_groups, sizes, seed, by_row=True
    )
    return _finish(task, (report, validation, fit), rest, counts, sizes, seed)


def m2_utility_report_pairs(
    examples: Sequence[Example],
    *,
    n_pairs: int = M2_EVAL_PAIRS,
    seed: int = M2_SEED,
) -> frozenset[str]:
    """Recomputes the utility report pairs M2 read (section 2.4).

    Uses `task_splits` and `task_set` with the report seed of `eval_seed`,
    as M2 does; the pairing depends on the task and the seed only, never on
    the chat format.

    Args:
        examples: The utility loader output.
        n_pairs: The M2 number of evaluation pairs.
        seed: The M2 base seed.

    Returns:
        The row ids (`meta["pair"]`) of the pairs.
    """
    task = READING_TASKS["utility"]
    splits = task_splits(task, {task.source: examples}, seed=seed)
    variants, _ = template_variants(task, ChatFormat("plain"), kinds=("paper",))
    stimuli = task_set(
        task,
        variants[0],
        splits.report,
        n_pairs=eval_pairs(task, splits.report, n_pairs),
        seed=eval_seed(task, "report", seed),
    )
    return frozenset(uid.rpartition(":")[0] for uid in stimuli.uids)


# Labelled pairs of a draw (section 2.4), and the pairs of each draw kept in
# `partitions.json`: the largest budget of the label-budget curve (7.6).
N_PAIRS = 64
CURVE_PAIRS = 128


def draws(
    partition: TaskPartition,
    *,
    n_draws: int = 10,
    n_pairs: int = N_PAIRS,
    seed: int = 0,
) -> tuple[tuple[tuple[str, str], ...], ...]:
    """Returns the labelled training draws of a task (section 2.4).

    Args:
        partition: The task partition.
        n_draws: The number of draws.
        n_pairs: The pairs per draw.
        seed: The base seed.

    Returns:
        For each draw, the first `n_pairs` pairs of a seeded permutation of
        the fit reservoir. The draws do not depend on the model.
    """
    fit = partition.blocks["fit"].pairs
    out = []
    for index in range(n_draws):
        order = np.random.default_rng(
            derive_seed(seed, "i1", "draw", partition.task, index)
        ).permutation(len(fit))
        out.append(tuple(fit[i] for i in order[:n_pairs]))
    return tuple(out)


def _city_column(subset: str, cache: Path | None, offline: bool) -> list[str]:
    path = fetch(
        GOT_SOURCE, file=f"datasets/{subset}.csv", cache=cache, offline=offline
    )
    reader = csv.DictReader(io.StringIO(path.read_bytes().decode("utf-8")))
    if "city" not in (reader.fieldnames or ()):
        return []
    return [row["city"] for row in reader]


def load_partitions(
    *,
    sizes: BlockSizes = PREREGISTERED_SIZES,
    seed: int = 0,
    cache: Path | None = None,
    offline: bool = False,
) -> dict[str, TaskPartition]:
    """Loads the pinned sources and partitions the four tasks.

    Args:
        sizes: The block sizes.
        seed: The base seed of the hash.
        cache: Dataset cache directory; None uses the repository default.
        offline: Read only from the cache.

    Returns:
        The partition of each task, in `TASKS` order.
    """
    sst2 = load_sst2(cache=cache, offline=offline)
    common = load_geometry_of_truth(
        subsets=("common_claim_true_false",), cache=cache, offline=offline
    )
    facts = load_facts(cache=cache, offline=offline)
    negations = load_geometry_of_truth(
        subsets=NEGATION_SUBSETS, cache=cache, offline=offline
    )
    cities = {
        subset: _city_column(subset, cache, offline)
        for subset in NEGATION_SUBSETS
        if "cities" in subset
    }
    subjects, notes = negation_subjects(negations, cities)
    utility = load_ethics_utility(cache=cache, offline=offline)
    negation_part = partition_negations(
        negations, subjects, sizes=sizes, seed=seed
    )
    return {
        "sentiment": partition_sentiment(sst2, sizes=sizes, seed=seed),
        "truth": partition_truth(common, facts, sizes=sizes, seed=seed),
        "truth_negations": replace(
            negation_part, notes=(*notes, *negation_part.notes)
        ),
        "utility": partition_utility(
            utility, m2_utility_report_pairs(utility), sizes=sizes, seed=seed
        ),
    }


@dataclass(frozen=True)
class Prompt:
    """One rendered prompt of a prompt set.

    Attributes:
        task: One of `TASKS`.
        block: The block of the text.
        uid: The uid of the text.
        label: The label of the text. It is kept for the readers; the pool
            labels are never passed to an estimator (section 4).
        condition: "clean", a template name such as "F-T1", a decoy with its
            level rule such as "S-D1/against", or "oracle/<condition>".
        text: The full prompt.
        stimulus_span: The character span of the stimulus in `text`.
        level: The decoy level drawn or set, None without a decoy.
    """

    task: str
    block: BlockName
    uid: str
    label: int
    condition: str
    text: str
    stimulus_span: Span
    level: str | None


Condition = tuple[str | None, LevelRule | None]


def _held_in_conditions(
    task: str, wording_set: Literal["fit", "score"]
) -> list[Condition]:
    """The clean template and the wrappers of a held-in set.

    Fit-set decoys take the `random` level only; score-set decoys take
    `random` and `against` (sections 3.4, 3.5 and 6.1).
    """
    rules: tuple[LevelRule, ...] = (
        ("random",) if wording_set == "fit" else ("random", "against")
    )
    out: list[Condition] = [(None, None)]
    for name in wrapper_names(task, wording_set):
        if WORDINGS[name].placement is None:
            out.append((name, None))
        else:
            out.extend((name, rule) for rule in rules)
    return out


def _report_conditions(task: str) -> list[Condition]:
    """The stage 2 conditions of section 7.1, in the order of its table."""
    out: list[Condition] = [(None, None)]
    for wording in (M2_TEST_DECOY, *TEST_DECOYS):
        out.extend((wording.name, rule) for rule in ("random", "against"))
    if task != "utility":
        out.extend((w.name, None) for w in TEST_PARAPHRASES)
    return out


def _condition_name(wording: str | None, rule: LevelRule | None) -> str:
    if wording is None:
        return "clean"
    return wording if rule is None else f"{wording}/{rule}"


def _render_block(
    task: str,
    block: Block,
    conditions: Sequence[Condition],
    chat: ChatFormat,
    *,
    stage: int,
    seed: int,
) -> list[Prompt]:
    evaluation = block.name in ("validation", "report")
    prompts = []
    for wording, rule in conditions:
        for example in block.examples:
            rendered = render_text(
                task,
                wording,
                example,
                chat,
                rule=rule,
                stage=stage,
                evaluation=evaluation,
                seed=seed,
            )
            prompts.append(
                Prompt(
                    task,
                    block.name,
                    example.uid,
                    example.label,
                    _condition_name(wording, rule),
                    rendered.text,
                    rendered.stimulus_span,
                    rendered.level,
                )
            )
    return prompts


def _prompt_set(
    partitions: Mapping[str, TaskPartition],
    chat: ChatFormat,
    plan: Mapping[BlockName, Callable[[str], list[Condition]]],
    *,
    stage: int,
    seed: int,
) -> list[Prompt]:
    prompts: list[Prompt] = []
    for task in TASKS:
        blocks = partitions[task].blocks
        for name, conditions in plan.items():
            prompts += _render_block(
                task,
                blocks[name],
                conditions(task),
                chat,
                stage=stage,
                seed=seed,
            )
    return prompts


def _fit_conditions(task: str) -> list[Condition]:
    return _held_in_conditions(task, "fit")


def _score_conditions(task: str) -> list[Condition]:
    return _held_in_conditions(task, "score")


def stage0_prompts(
    partitions: Mapping[str, TaskPartition],
    chat: ChatFormat,
    *,
    seed: int = 0,
) -> list[Prompt]:
    """Builds the stage 0 prompt set of section 6.1.

    Fit reservoir and pool under the clean template and the fit-set
    wrappers (`random` levels); validation pairs under the clean template
    and the score-set wrappers (`random` and `against`). The report block is
    not rendered and no test wording is used.

    Args:
        partitions: The partition of each task.
        chat: The chat format of the checkpoint.
        seed: The base seed of the `random` levels.

    Returns:
        The prompts, task by task in `TASKS` order, then by block,
        condition and text.
    """
    plan: dict[BlockName, Callable[[str], list[Condition]]] = {
        "fit": _fit_conditions,
        "pool": _fit_conditions,
        "validation": _score_conditions,
    }
    return _prompt_set(partitions, chat, plan, stage=0, seed=seed)


def stage2_prompts(
    partitions: Mapping[str, TaskPartition],
    chat: ChatFormat,
    *,
    seed: int = 0,
) -> list[Prompt]:
    """Builds the stage 2 prompt set of one checkpoint (sections 7.1, 10).

    Fit reservoir and pool as in stage 0, and the report pairs under the
    conditions of section 7.1. Defined here, run only after the stage 1
    gate.

    Args:
        partitions: The partition of each task.
        chat: The chat format of the checkpoint.
        seed: The base seed of the `random` levels.

    Returns:
        The prompts, task by task in `TASKS` order.
    """
    plan: dict[BlockName, Callable[[str], list[Condition]]] = {
        "fit": _fit_conditions,
        "pool": _fit_conditions,
        "report": _report_conditions,
    }
    return _prompt_set(partitions, chat, plan, stage=2, seed=seed)


def oracle_prompts(prompts: Sequence[Prompt], chat: ChatFormat) -> list[Prompt]:
    """Renders every wrapped report text again under the clean template.

    The canonical re-prompt oracle of sections 5.5 and 8.3: the wrapper is
    stripped and the stimulus read in the clean template.

    Args:
        prompts: A stage 2 prompt set.
        chat: The chat format of the checkpoint.

    Returns:
        One clean prompt per wrapped report prompt, in the same order, with
        the condition "oracle/<condition>".
    """
    out = []
    for prompt in prompts:
        if prompt.block != "report" or prompt.condition == "clean":
            continue
        text = prompt.text[slice(*prompt.stimulus_span)]
        example = Example(prompt.uid, text, prompt.label, "test")
        rendered = render_text(prompt.task, None, example, chat)
        out.append(
            replace(
                prompt,
                condition=f"oracle/{prompt.condition}",
                text=rendered.text,
                stimulus_span=rendered.stimulus_span,
                level=None,
            )
        )
    return out


def pool_levels(
    prompts: Sequence[Prompt],
) -> dict[str, dict[str, dict[str, str]]]:
    """Returns the random level drawn for each pool text and decoy wording.

    Section 9.1 compares pool texts by their random level.

    Args:
        prompts: A prompt set.

    Returns:
        By task, decoy wording and uid, the level word.
    """
    out: dict[str, dict[str, dict[str, str]]] = {}
    for prompt in prompts:
        if prompt.block == "pool" and prompt.level is not None:
            wording = prompt.condition.split("/")[0]
            task = out.setdefault(prompt.task, {})
            task.setdefault(wording, {})[prompt.uid] = prompt.level
    return out


STAGE0_MODEL = "qwen3-0.6b-base"
MIN_RELATIVE_DEPTH = 0.4


@dataclass(frozen=True, kw_only=True)
class Config(ExperimentConfig):
    """I1 settings (sections 6 and 10).

    Attributes:
        batch_size: Texts per forward pass.
        stage: Stage to enter, 0 (cache) or 1 (tournament). Stage 2 only
            checks the stored gate until the confirmation driver exists.
        input_run: Stage 0 run directory for stage 1, or stage 1 run
            directory for the stage 2 preflight.
        go: Explicit authorization to start stage 0 or 1, recorded in the
            run log before any data is read.
        workers: Number of independent stage 1 draw workers.
    """

    experiment: str = "i1"
    model: str = STAGE0_MODEL
    batch_size: int = 8
    stage: Literal[0, 1, 2] = 0
    input_run: str = ""
    go: str = ""
    workers: int = 1


def candidate_layers(n_layers: int) -> tuple[int, ...]:
    """Returns the layers at relative depth 0.4-1.0 (section 2.1).

    Args:
        n_layers: The number of decoder blocks L, read from the model.

    Returns:
        The outputs of blocks `ceil(0.4 L)` to `L`.
    """
    return tuple(range(math.ceil(MIN_RELATIVE_DEPTH * n_layers), n_layers + 1))


def _partition_record(partition: TaskPartition, seed: int) -> dict[str, Any]:
    return {
        "blocks": partition.item_ids(),
        "draws": [
            [list(pair) for pair in draw]
            for draw in draws(partition, n_pairs=CURVE_PAIRS, seed=seed)
        ],
        "counts": dict(partition.counts),
        "notes": list(partition.notes),
    }


def encode_prompts(
    loaded: LoadedModel,
    prompts: Sequence[Prompt],
    run_dir: RunDir,
    *,
    batch_size: int = 8,
) -> None:
    """Encodes a prompt set at both positions and caches it per task.

    Writes per task, in `TASKS` order, `prompts_<task>.json` (one record
    per row) and `activations_<task>.npz` with the float32 arrays `last` and
    `stimulus` of shape (layers, prompts, d_model), the `layers` read
    (`candidate_layers` of the model) and the token index of each row
    (`last_token`, `stimulus_token`). The stimulus position is the last
    token overlapping the stimulus span, the overlap rule of
    `collect_activations` (section 5.1).

    Args:
        loaded: The wrapped model.
        prompts: The prompts.
        run_dir: The run directory to write into.
        batch_size: Texts per forward pass.
    """
    layers = candidate_layers(loaded.n_layers)
    for task in TASKS:
        rows = [p for p in prompts if p.task == task]
        if not rows:
            continue
        run_dir.write_json(
            f"prompts_{task}",
            [
                {
                    "block": p.block,
                    "uid": p.uid,
                    "label": p.label,
                    "condition": p.condition,
                    "level": p.level,
                }
                for p in rows
            ],
        )
        texts = [p.text for p in rows]
        last = collect_activations(
            loaded, texts, layers=layers, position="last", batch_size=batch_size
        )
        stimulus = collect_activations(
            loaded,
            texts,
            layers=layers,
            position="concept",
            concept_spans=[p.stimulus_span for p in rows],
            batch_size=batch_size,
        )
        run_dir.write_arrays(
            f"activations_{task}",
            last=last.data,
            stimulus=stimulus.data,
            layers=np.asarray(layers, dtype=np.int64),
            last_token=last.row_token,
            stimulus_token=stimulus.row_token,
        )


def encode_stage0(
    loaded: LoadedModel,
    partitions: Mapping[str, TaskPartition],
    run_dir: RunDir,
    *,
    seed: int = 0,
    batch_size: int = 8,
) -> None:
    """Encodes the stage 0 prompt set and caches it (sections 6.1 and 10).

    Writes `partitions.json` first (the item ids, draws, counts and notes of
    every task, step 1 of section 10), then `pool_levels.json`, then the
    cache of `encode_prompts`. Each draw keeps its first `CURVE_PAIRS`
    pairs; stage 1 fits on the first `N_PAIRS` of them.

    Args:
        loaded: The wrapped model, in float32 for the preregistered run.
        partitions: The partition of each task.
        run_dir: The run directory to write into.
        seed: The base seed of the draws and the `random` levels.
        batch_size: Texts per forward pass.
    """
    run_dir.write_json(
        "partitions",
        {task: _partition_record(partitions[task], seed) for task in TASKS},
    )
    prompts = stage0_prompts(partitions, chat_format(loaded), seed=seed)
    run_dir.write_json("pool_levels", pool_levels(prompts))
    encode_prompts(loaded, prompts, run_dir, batch_size=batch_size)


_STAGE1_ARMS = tuple(f"A{index}" for index in range(1, 9))


def _stage1_renderings(
    task: str, wording_set: Literal["fit", "score"]
) -> invariance.Renderings:
    conditions = _held_in_conditions(task, wording_set)
    return invariance.Renderings(
        tuple(_condition_name(*condition) for condition in conditions),
        tuple(rule or "none" for _, rule in conditions),
    )


def _block_features(
    features: invariance.FloatArray, rows: NDArray[np.int64]
) -> invariance.FloatArray:
    flat = rows.ravel()
    start = int(flat[0])
    # The stage 0 writer is condition-major. Keep a view on its cache
    # rather than duplicating the full fit reservoir or pool in each worker.
    if np.array_equal(flat, np.arange(start, start + flat.size)):
        selected = features[:, start : start + flat.size]
    else:
        selected = features[:, flat]
    return selected.reshape(features.shape[0], *rows.shape, -1).swapaxes(1, 2)


@dataclass(frozen=True)
class _Stage1Features:
    fit_index: dict[str, int]
    fit_stim: invariance.FloatArray
    fit_last: invariance.FloatArray
    pool: invariance.Pool
    score_renderings: invariance.Renderings
    score_stim: invariance.FloatArray
    score_last: invariance.FloatArray


def _pair_uids(pairs: Sequence[Sequence[str]]) -> list[str]:
    return [uid for positive, negative in pairs for uid in (positive, negative)]


_LOADED: dict[tuple[Path, str], _Stage1Features] = {}


def _stage1_features(cache: Path, task: str) -> _Stage1Features:
    # One task per worker at a time: drop the old cache before loading.
    if (cache, task) not in _LOADED:
        _LOADED.clear()
        _LOADED[cache, task] = _load_stage1_features(cache, task)
    return _LOADED[cache, task]


def _load_stage1_features(cache: Path, task: str) -> _Stage1Features:
    source = RunDir.open(cache)
    partitions = cast(dict[str, Any], source.read_json("partitions"))
    blocks = partitions[task]["blocks"]
    fit_uids = blocks["fit"]["uids"]
    fit = _stage1_renderings(task, "fit")
    score = _stage1_renderings(task, "score")
    rows = cast(list[dict[str, Any]], source.read_json(f"prompts_{task}"))
    by_key: dict[tuple[str, str, str], int] = {}
    valid = {"fit": fit.names, "pool": fit.names, "validation": score.names}
    for index, row in enumerate(rows):
        block, uid, condition = row["block"], row["uid"], row["condition"]
        key = (block, uid, condition)
        if condition not in valid.get(block, ()) or key in by_key:
            raise ValueError(f"unexpected or duplicate {task} cache row: {key}")
        by_key[key] = index

    def indices(block: str, uids: Sequence[str]) -> NDArray[np.int64]:
        return np.array(
            [
                [by_key[block, uid, name] for uid in uids]
                for name in valid[block]
            ],
            dtype=np.int64,
        )

    validation_uids = _pair_uids(blocks["validation"]["pairs"])
    arrays = source.read_arrays(f"activations_{task}")
    layers = tuple(map(int, arrays["layers"]))
    fit_rows = indices("fit", fit_uids)
    pool_rows = indices("pool", blocks["pool"]["uids"])
    # Load validation in its cached block order when it matches the explicit
    # pair axis; the positive/negative order always comes from the partition.
    score_rows = indices("validation", validation_uids)
    stim, last = arrays["stimulus"], arrays["last"]
    return _Stage1Features(
        {uid: index for index, uid in enumerate(fit_uids)},
        _block_features(stim, fit_rows),
        _block_features(last, fit_rows),
        invariance.Pool(
            _block_features(stim, pool_rows),
            _block_features(last, pool_rows),
            fit,
            layers,
        ),
        score,
        _block_features(stim, score_rows),
        _block_features(last, score_rows),
    )


@dataclass(frozen=True)
class _Stage1Job:
    cache: Path
    task: str
    draw: int
    pairs: list[list[str]]
    seed: int


@dataclass(frozen=True)
class _DrawResult:
    scores: dict[str, dict[str, invariance_stats.FloatArray]]
    audit: dict[str, Any]


def _fit_stage1_draw(job: _Stage1Job) -> _DrawResult:
    source = _stage1_features(job.cache, job.task)
    rows = [source.fit_index[uid] for uid in _pair_uids(job.pairs)]
    n_pairs = len(job.pairs)
    pool = source.pool
    data = invariance.LabelledSet(
        source.fit_stim[:, rows],
        source.fit_last[:, rows],
        np.tile(np.array([1, 0], dtype=np.int64), n_pairs),
        np.repeat(np.arange(n_pairs, dtype=np.int64), 2),
        pool.renderings,
        pool.layers,
    )
    choices: dict[str, invariance.Candidate] = {}
    scores: dict[str, dict[str, invariance_stats.FloatArray]] = {}
    arms = {}
    for arm in _STAGE1_ARMS:
        layers = None
        if len(invariance.ARMS[arm].positions) == 2:
            layers = (choices["A4"].layers[0], choices["A3"].layers[0])
        tuned = invariance.fit_arm(
            arm, data, pool, seed=job.seed, layers=layers
        )
        choices[arm] = tuned.choice
        reader = tuned.reader
        scores[arm] = {
            condition: reader.score(
                source.score_stim[:, :, index],
                source.score_last[:, :, index],
            ).reshape(-1, 2)
            for index, condition in enumerate(source.score_renderings.names)
        }
        arms[arm] = {
            "layers": {block.position: block.layer for block in reader.blocks},
            "params": reader.params,
            "criterion": reader.criterion,
            "fits": tuned.fits,
            "fallbacks": tuned.fallbacks,
            "refit_fallbacks": reader.fallbacks,
        }
    return _DrawResult(
        scores,
        {
            "draw": job.draw,
            "seed": job.seed,
            "pairs": job.pairs,
            "label_budget": len(data.labels),
            "fit_renderings": len(data.renderings.names),
            "pool_texts": pool.last.shape[1],
            "validation_pairs": source.score_last.shape[1] // 2,
            "arms": arms,
        },
    )


def _stage1_scores(
    task: str, draws_: Sequence[_DrawResult]
) -> invariance_stats.TaskScores:
    scores: dict[str, dict[str, invariance_stats.FloatArray]] = {}
    for arm, conditions in draws_[0].scores.items():
        scores[arm] = {
            name: np.stack([draw.scores[arm][name] for draw in draws_])
            for name in conditions
        }
    return invariance_stats.TaskScores(task, scores)


def _run_stage1(config: Config, run_dir: RunDir) -> None:
    if not config.go.strip():
        raise ValueError(
            "stage 1 needs an explicit go recorded with --set go=..."
        )
    if not config.input_run or config.workers < 1:
        raise ValueError("stage 1 needs input_run and at least one worker")
    cache = Path(config.input_run).resolve()
    if cache == run_dir.path.resolve():
        raise ValueError("stage 1 must not write into its stage 0 cache")
    source = RunDir.open(cache)
    settings = source.config
    if settings.get("experiment") != "i1" or settings.get("stage") != 0:
        raise ValueError("input_run must be an I1 stage 0 run")
    for field in ("model", "seed"):
        if settings.get(field) != getattr(config, field):
            raise ValueError(
                f"stage 1 {field} {getattr(config, field)!r} differs from"
                f" the stage 0 cache ({settings.get(field)!r})"
            )
    log = {"go": config.go, "stage0_cache": str(cache)}
    run_dir.write_json("run_log", log)
    partitions = cast(dict[str, Any], source.read_json("partitions"))
    jobs = []
    for task in TASKS:
        task_draws = partitions[task]["draws"]
        if not task_draws:
            raise ValueError(f"{task} has no stored training draws")
        for index, pairs in enumerate(task_draws):
            seed = derive_seed(config.seed, "i1", "stage1", task, index)
            jobs.append(_Stage1Job(cache, task, index, pairs[:N_PAIRS], seed))

    results: dict[str, list[_DrawResult]] = {task: [] for task in TASKS}
    previous = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    try:
        # Spawn fresh BLAS runtimes and give each worker one core.
        for name in previous:
            os.environ[name] = "1"
        with ProcessPoolExecutor(
            max_workers=config.workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            for job, result in zip(
                jobs, executor.map(_fit_stage1_draw, jobs), strict=True
            ):
                results[job.task].append(result)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    tasks = [_stage1_scores(task, entries) for task, entries in results.items()]
    n_resamples = invariance_stats.N_RESAMPLES
    gate = invariance_stats.stage1_gate(
        tasks, rng=config.seed, n_resamples=n_resamples
    )
    decision = {
        "comparator": gate.comparator,
        "statistic": gate.statistic,
        "interval": gate.interval,
        "passed": gate.passed,
        "reasons": gate.reasons,
    }
    run_dir.write_json(
        "stage1",
        {
            "selection": asdict(gate.selection),
            "gate": decision,
            "bootstrap_seed": config.seed,
            "n_resamples": n_resamples,
            "bootstrap": gate.bootstrap,
            "draws": {
                task: [draw.audit for draw in entries]
                for task, entries in results.items()
            },
        },
    )
    run_dir.write_json(
        "run_log",
        {**log, "gate": {"winner": gate.selection.winner, **decision}},
    )


def require_stage1_pass(stage1_run: RunDir) -> str:
    """Returns the frozen winner only after a passed gate is logged.

    Args:
        stage1_run: The existing I1 stage 1 run.

    Returns:
        The winning arm to use unchanged in stage 2.

    Raises:
        ValueError: If the log does not contain a passed stage 1 gate.
    """
    try:
        log = stage1_run.read_json("run_log")
    except (FileNotFoundError, ValueError):
        log = None
    if isinstance(log, dict) and stage1_run.config.get("stage") == 1:
        decision = log.get("gate")
        if isinstance(decision, dict) and decision.get("passed") is True:
            winner = decision.get("winner")
            if (
                isinstance(winner, str)
                and winner in invariance_stats.READER_ARMS
            ):
                return winner
    raise ValueError(
        "stage 2 needs a stored passed stage 1 gate in run_log.json"
    )


def run(config: Config, run_dir: RunDir) -> None:
    """Runs stage 0 or 1; guards the future stage 2 driver.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.

    Raises:
        ValueError: If a stage lacks its input, authorization or gate, or
            stage 0 is given a different checkpoint.
        NotImplementedError: If stage 2 clears the gate check.
    """
    if config.stage == 1:
        _run_stage1(config, run_dir)
        return
    if config.stage == 2:
        if not config.input_run:
            raise ValueError(
                "stage 2 needs input_run with a stored passed stage 1 gate"
            )
        require_stage1_pass(RunDir.open(config.input_run))
        raise NotImplementedError("stage 2 is not implemented")
    if config.model != STAGE0_MODEL:
        raise ValueError(
            f"stage 0 runs on {STAGE0_MODEL} only, got {config.model!r}"
        )
    if not config.go.strip():
        raise ValueError(
            "stage 0 needs an explicit go recorded with --set go=..."
        )
    run_dir.write_json("run_log", {"go": config.go})
    partitions = load_partitions(seed=config.seed)
    loaded = load_model(config.model, dtype=torch.float32)
    encode_stage0(
        loaded,
        partitions,
        run_dir,
        seed=config.seed,
        batch_size=config.batch_size,
    )


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Stage 0 writes a cache only and has no figure.

    Args:
        run_dir: The run directory.
        out_dir: The figure directory.

    Returns:
        An empty list.
    """
    del run_dir, out_dir
    return []
