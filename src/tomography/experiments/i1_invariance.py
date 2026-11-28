"""I1: data, prompts and stage 0 of the two-position invariance reader.

Implements the preregistration `docs/preregistration-invariance-reader.md`:
the partitions and draws of section 2.4, the frozen wordings and renderings
of section 3, the stage 0 prompt set of section 6.1 (and the stage 2 prompt
set of section 7.1, defined but not run here) and the stage 0 activation
cache of section 10. Section numbers in the docstrings refer to that file.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Literal
from typing import get_args

import numpy as np

from tomography.datasets.records import Example
from tomography.experiments.base import derive_seed
from tomography.experiments.variants import READING_TASKS
from tomography.experiments.variants import ReadingTask
from tomography.model import LoadedModel
from tomography.templates import ChatFormat
from tomography.templates import Span

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

TaskName = Literal["sentiment", "truth", "truth_negations", "utility"]
TASKS: tuple[TaskName, ...] = get_args(TaskName)
Placement = Literal["pre", "head", "post"]
LevelRule = Literal["random", "against"]
WordingSet = Literal["fit", "score", "test"]

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
