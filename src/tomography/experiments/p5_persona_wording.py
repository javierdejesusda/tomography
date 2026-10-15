"""P5: does the honesty reader read honesty or the persona's wording?"""

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace
from types import MappingProxyType

from tomography.activations import PositionStrategy
from tomography.datasets.records import Example
from tomography.datasets.stimulus import StimulusSet
from tomography.datasets.stimulus import build_stimulus_set
from tomography.experiments.base import derive_seed
from tomography.experiments.sweep import ReadingConfig
from tomography.experiments.sweep import shuffle_within_groups
from tomography.reading.directions import Method
from tomography.templates import PAPER_TEMPLATES
from tomography.templates import ChatFormat
from tomography.templates import ConceptTemplate
from tomography.templates import FunctionTemplate

PERSONA_PAIRS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "paper": ("an honest", "an untruthful"),
        "synonym_0": ("a truthful", "a deceitful"),
        "synonym_1": ("a sincere", "a lying"),
        "unrelated_0": ("a tall", "a short"),
        "unrelated_1": ("an old", "a young"),
    }
)
CONTROLS: tuple[str, ...] = ("instruction_only", "fact_only", "random_label")
CONDITIONS: tuple[str, ...] = (*PERSONA_PAIRS, *CONTROLS)
GOT_SUBSETS: tuple[str, ...] = (
    "cities",
    "neg_cities",
    "sp_en_trans",
    "larger_than",
    "common_claim_true_false",
)
PLAIN = ConceptTemplate("honesty_plain", "{stimulus}", tagged=False)
PLAIN_POSITION: PositionStrategy = "mean"
MAX_CONDITIONS = 8


@dataclass(frozen=True, kw_only=True)
class Config(ReadingConfig):
    """P5 settings: which persona words the honesty reader depends on.

    Every condition changes one thing against the paper persona prompt, the
    persona words or the stimuli, and is read the way R5 reads honesty: a
    reader trained on the response tokens of true statements, then scored on
    plain true against false statements at the token mean. Accuracies are
    raw, so a reader whose first persona aligns with false statements
    scores below 0.5.

    Attributes:
        conditions: Names from `CONDITIONS`, at most `MAX_CONDITIONS`, and
            `paper` is required.
        got_subsets: Geometry of Truth subsets scored on their test split.
        n_boot: Number of bootstrap resamples of every interval.
    """

    experiment: str = "p5"
    model: str = "gpt2"
    method: Method = "logistic"
    position: PositionStrategy = "response"
    n_eval_pairs: int | None = 256
    conditions: tuple[str, ...] = CONDITIONS
    got_subsets: tuple[str, ...] = GOT_SUBSETS
    n_boot: int = 2000


def persona_template(condition: str) -> FunctionTemplate:
    """Returns the paper honesty template with the persona words replaced.

    Args:
        condition: A key of `PERSONA_PAIRS`.

    Returns:
        The template of the paper, its instruction text unchanged, with the
        first word of the pair as the label-1 persona and the second as the
        label-0 persona.

    Raises:
        ValueError: If `condition` is not a key of `PERSONA_PAIRS`.
    """
    if condition not in PERSONA_PAIRS:
        raise ValueError(
            f"{condition!r} has no persona words, expected one of "
            f"{tuple(PERSONA_PAIRS)}"
        )
    paper = PAPER_TEMPLATES["honesty"]
    assert isinstance(paper, FunctionTemplate)
    pos, neg = PERSONA_PAIRS[condition]
    return replace(
        paper,
        name=f"honesty_{condition}",
        pos_personas=(pos,),
        neg_personas=(neg,),
    )


def plain_stimuli(
    statements: Sequence[Example], *, n_pairs: int | None, seed: int
) -> StimulusSet:
    """Pairs plain true statements with plain false ones, with no persona.

    Args:
        statements: Statements labeled 1 for true and 0 for false.
        n_pairs: Number of pairs, capped at the number of true statements
            and at the number of false ones; None takes every pair.
        seed: Seed of the pairing.

    Returns:
        The `PLAIN` stimuli in the label-contrastive layout.

    Raises:
        ValueError: If the statements hold no true or no false one.
    """
    n_true = sum(e.label == 1 for e in statements)
    n_false = sum(e.label == 0 for e in statements)
    if not n_true or not n_false:
        raise ValueError(
            "plain stimuli need true and false statements, got "
            f"{n_true} true and {n_false} false"
        )
    return build_stimulus_set(
        statements,
        PLAIN,
        ChatFormat("none"),
        pairing="label_contrastive",
        n_pairs=None if n_pairs is None else min(n_pairs, n_true, n_false),
        seed=seed,
    )


def condition_stimuli(
    condition: str,
    statements: Sequence[Example],
    chat: ChatFormat,
    *,
    n_pairs: int | None,
    seed: int,
    train: bool,
) -> StimulusSet:
    """Builds the stimuli of one condition for one split.

    The persona conditions and `random_label` use the true statements only,
    and the persona plan depends on the statements and the seed alone, so
    with the same seed they draw the same statements. `instruction_only`
    renders the paper personas with an empty response. `fact_only` drops the
    persona and pairs plain true with plain false statements.

    Args:
        condition: A name from `CONDITIONS`.
        statements: The statements of one split, true ones labeled 1.
        chat: The chat format of the persona prompts.
        n_pairs: Number of statements (pairs); None takes every one.
        seed: Seed of the draw.
        train: Whether the stimuli train a reader. Only `random_label`
            changes: its training labels are shuffled within each pair, with
            the seed `derive_seed(seed, "random_label")`, and its evaluation
            labels stay real.

    Returns:
        The stimuli of the condition.

    Raises:
        ValueError: If the condition is unknown or the statements do not fit
            the stimuli.
    """
    if condition == "fact_only":
        return plain_stimuli(statements, n_pairs=n_pairs, seed=seed)
    if condition not in CONDITIONS:
        raise ValueError(
            f"unknown condition {condition!r}, expected one of {CONDITIONS}"
        )
    true = [e for e in statements if e.label == 1]
    if condition == "instruction_only":
        true = [replace(e, text="") for e in true]
    template = (
        persona_template(condition)
        if condition in PERSONA_PAIRS
        else persona_template("paper")
    )
    stimuli = build_stimulus_set(
        true, template, chat, pairing="persona", n_pairs=n_pairs, seed=seed
    )
    if condition == "random_label" and train:
        labels = shuffle_within_groups(
            stimuli.labels,
            stimuli.groups,
            seed=derive_seed(seed, "random_label"),
        )
        stimuli = replace(stimuli, labels=labels)
    return stimuli


def condition_position(
    condition: str, position: PositionStrategy
) -> PositionStrategy:
    """Returns the token positions a condition reads.

    Args:
        condition: A name from `CONDITIONS`.
        position: The position of the persona conditions.

    Returns:
        `last` for `instruction_only`, whose response is empty,
        `PLAIN_POSITION` for `fact_only` and `position` otherwise.
    """
    if condition == "instruction_only":
        return "last"
    if condition == "fact_only":
        return PLAIN_POSITION
    return position
