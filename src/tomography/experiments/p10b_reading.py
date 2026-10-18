"""P10b: reading a range against an exact answer from hidden states.

Each question asks for a quantity whose sourced answer is a range (the height
of a dog breed) or one exact value (the height of a building). A reader is
fitted on the activations of the questions of some families and scored on the
held-out questions of every family, which asks whether a direction that
separates the two kinds transfers across families. In every family the range
side is living and the exact side an artifact, so a set of well-known people
with one exact height is scored as a diagnostic: a reader of range against
exact calls them exact, a reader of living against artifact calls them range.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from tomography.datasets.quantities import DEFAULT_MIN_SITELINKS
from tomography.datasets.quantities import FAMILIES
from tomography.experiments.base import ExperimentConfig
from tomography.p9.reading import READING_METHODS

TEMPLATES: Final = ("plain", "concept")
ALL_SET: Final = "all"


@dataclass(frozen=True, kw_only=True)
class Config(ExperimentConfig):
    """P10b settings: the questions, the prompts and the readers.

    Attributes:
        families: Names of the question families to read; None reads all.
        n_per_side: Largest number of items per family and kind.
        min_sitelinks: Smallest sitelink count of an entity.
        people: Whether to score the people heights as a diagnostic; it
            needs the "height" family.
        offline: Whether the questions are read from the cache only.
        templates: How a question is rendered: "plain" is the bare question,
            "concept" the paper's concept template wrapped in the chat
            format of post-trained models.
        concept: The concept word of the "concept" template.
        methods: The direction finders; "random" is the control.
        n_seeds: Draws of the pairs per template, method and train set.
        n_boot: Resamples of every bootstrap interval.
        batch_size: Prompts per forward pass.
    """

    experiment: str = "p10b"
    model: str = "gpt2"
    families: tuple[str, ...] | None = None
    n_per_side: int = 120
    min_sitelinks: int = DEFAULT_MIN_SITELINKS
    people: bool = True
    offline: bool = False
    templates: tuple[str, ...] = TEMPLATES
    concept: str = "variability"
    methods: tuple[str, ...] = READING_METHODS
    n_seeds: int = 5
    n_boot: int = 1000
    batch_size: int = 8

    def __post_init__(self) -> None:
        """Checks the settings that would act silently or fail late.

        Raises:
            ValueError: If `families`, `templates` or `methods` is empty,
                repeats a name or holds an unknown one, `concept` is empty
                or a count is below its minimum (`n_per_side`, `n_seeds`,
                `n_boot` and `batch_size` need 1, `min_sitelinks` needs 0).
        """
        if self.families is not None:
            _check_names("families", self.families, [f.name for f in FAMILIES])
        _check_names("templates", self.templates, TEMPLATES)
        _check_names("methods", self.methods, READING_METHODS)
        if not self.concept:
            raise ValueError("concept must not be empty")
        for name in ("n_per_side", "n_seeds", "n_boot", "batch_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.min_sitelinks < 0:
            raise ValueError("min_sitelinks must not be negative")


def _check_names(
    field: str, names: tuple[str, ...], allowed: Sequence[str]
) -> None:
    """Checks that a setting names at least one known option, each once.

    Args:
        field: The setting, for the message.
        names: The chosen options.
        allowed: The known options.

    Raises:
        ValueError: If `names` is empty, repeats an option or holds one that
            is not in `allowed`.
    """
    if not names:
        raise ValueError(f"{field} must not be empty")
    if len(set(names)) != len(names):
        raise ValueError(f"{field} repeats a name: {names}")
    unknown = sorted(set(names) - set(allowed))
    if unknown:
        raise ValueError(f"{field} names unknown {unknown}; known: {allowed}")


def train_sets(families: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """Derives the sets of families a reader is fitted on.

    The sets are each single family, then, with three or more families,
    each leave-one-family-out set, then every family together as "all".
    With one family the single set is that family alone. A set is named by
    joining its family names with "+", and the families of a set follow the
    order of `FAMILIES`.

    Args:
        families: The selected family names.

    Returns:
        The family names of each set, by set name.
    """
    order = [f.name for f in FAMILIES]
    chosen = tuple(sorted(families, key=order.index))
    sets: dict[str, tuple[str, ...]] = {name: (name,) for name in chosen}
    if len(chosen) >= 3:
        for left_out in chosen:
            rest = tuple(name for name in chosen if name != left_out)
            sets["+".join(rest)] = rest
    if len(chosen) >= 2:
        sets[ALL_SET] = chosen
    return sets
