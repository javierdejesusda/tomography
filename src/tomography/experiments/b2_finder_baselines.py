"""B2: published direction finders beside LAT on truth and sentiment.

LAT (`pca_diff`, the method under study) is compared with published
baselines fitted on the same training items and scored on the same test
pairs at a layer chosen on validation: the mass-mean probe (`mean_diff`) and
its whitened (IID) form (`whitened_mean_diff`) of Marks and Tegmark (2024),
the logistic-regression probe (`logistic`, Alain and Bengio 2016) and CCS
(Burns et al. 2023). Beside them sit the controls: a random direction, the
LAT reader scored against shuffled labels, and the word and character
n-gram text-only classifiers of TB1.

Truth reads three Geometry of Truth sets, each statement paired with its
negation (the row of the same index in the negated set), the contrast pairs
of Marks and Tegmark; a pair takes the loader's split of its statement.
Sentiment reads SST-2 as R1 does, with R1's training sets; SST-2 has no
labelled test split, so the validation split is halved into a layer-choice
half and a report half as for the reading task `sentiment`. CCS reads
contrast pairs: the statement and its negation for truth, and for sentiment
each text in one fixed prompt in the form of Burns et al., completed once
with "positive" and once with "negative".
"""

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType

from tomography.datasets.records import Example
from tomography.datasets.stimulus import StimulusSet
from tomography.datasets.stimulus import build_stimulus_set
from tomography.experiments import r1_sentiment
from tomography.experiments import r7_truth
from tomography.experiments.base import ExperimentConfig
from tomography.experiments.base import derive_seed
from tomography.experiments.variants import READING_TASKS
from tomography.experiments.variants import task_splits
from tomography.model import BosMode
from tomography.templates import PAPER_TEMPLATES
from tomography.templates import ChatFormat
from tomography.templates import ConceptTemplate

NEGATIONS: Mapping[str, str] = MappingProxyType(
    {
        "cities": "neg_cities",
        "larger_than": "smaller_than",
        "sp_en_trans": "neg_sp_en_trans",
    }
)
DATASETS: tuple[str, ...] = ("sentiment", *NEGATIONS)
_CCS_PROMPT = (
    "Consider the following example: ''' {stimulus} '''\n"
    "Between negative and positive, the sentiment of this example is"
)
# The first completion is the first member of each CCS contrast pair.
_COMPLETIONS = tuple(
    ConceptTemplate(f"ccs_{word}", _CCS_PROMPT, assistant=word)
    for word in ("positive", "negative")
)


@dataclass(frozen=True, kw_only=True)
class Config(ExperimentConfig):
    """B2 settings.

    Attributes:
        datasets: The datasets to read, from `DATASETS`: `sentiment` (SST-2
            as in R1) or a Geometry of Truth set paired with its negation.
        n_seeds: Number of training sets (seeds).
        n_pairs: Training pairs per seed, as in R1 and R7.
        n_eval_pairs: Validation and test pairs; None means every pair the
            split allows.
        layers: Layer indices; None means 1..L.
        bos: BOS policy; None uses the model's default.
        batch_size: Texts per forward pass.
        ccs_restarts: Random initialisations of each CCS fit.
        ccs_epochs: Full-batch optimiser steps of each CCS initialisation.
        ccs_lr: AdamW learning rate of CCS (the paper's 0.01).
        n_boot: Bootstrap resamples of the interval over seeds.
    """

    experiment: str = "b2"
    model: str = "gpt2"
    datasets: tuple[str, ...] = DATASETS
    n_seeds: int = 10
    n_pairs: int = 64
    n_eval_pairs: int | None = None
    layers: tuple[int, ...] | None = None
    bos: BosMode | None = None
    batch_size: int = 8
    ccs_restarts: int = 10
    ccs_epochs: int = 1000
    ccs_lr: float = 0.01
    n_boot: int = 2000

    def __post_init__(self) -> None:
        """Checks the dataset list.

        Raises:
            ValueError: If `datasets` is empty, names an unknown dataset or
                repeats one.
        """
        if not self.datasets:
            raise ValueError("datasets needs at least one dataset")
        for i, name in enumerate(self.datasets):
            if name not in DATASETS:
                raise ValueError(
                    f"unknown dataset {name!r}, expected one of {DATASETS}"
                )
            if name in self.datasets[:i]:
                raise ValueError(f"repeated dataset {name!r}")


@dataclass(frozen=True, eq=False)
class Items:
    """The stimuli of one dataset, before the model runs.

    Attributes:
        train: One training set per seed.
        val: The pairs that choose the layer.
        test: The pairs every reported number is computed on.
        text_of: The raw text of every example, by uid.
    """

    train: tuple[StimulusSet, ...]
    val: StimulusSet
    test: StimulusSet
    text_of: Mapping[str, str]


def dataset_items(
    config: Config,
    name: str,
    chat: ChatFormat,
    sentiment: Sequence[Example],
    truth_sets: Sequence[Example],
) -> Items:
    """Builds the training sets and the validation and test pairs.

    Sentiment uses R1's template, training examples and training seeds, so
    its training sets are R1's; its validation and test pairs are the
    halves of SST-2 validation that `task_splits` gives the reading task
    `sentiment`. A truth set uses R7's template and concept word; each of
    its groups is a statement and then its negation, exactly one of them
    true, in the loader's split of the statement.

    Args:
        config: The experiment settings.
        name: A name from `DATASETS`.
        chat: The chat format the prompts are rendered with.
        sentiment: SST-2 examples with `train` and `val` splits.
        truth_sets: Geometry of Truth examples, each with `meta["subset"]`.

    Returns:
        The items of the dataset.

    Raises:
        ValueError: If a set has no examples, a negated set does not flip
            the label of its statement set row by row, or a split has too
            few pairs.
    """
    if name == "sentiment":
        return _sentiment_items(config, chat, sentiment)
    return _truth_items(config, name, chat, truth_sets)


def _sentiment_items(
    config: Config, chat: ChatFormat, examples: Sequence[Example]
) -> Items:
    splits = task_splits(
        READING_TASKS["sentiment"], {"sst2": examples}, seed=config.seed
    )

    def eval_set(part: str, rows: Sequence[Example]) -> StimulusSet:
        seed = derive_seed(config.seed, "eval", "sentiment", part)
        return r1_sentiment._stimuli(rows, chat, config.n_eval_pairs, seed)

    return Items(
        train=tuple(
            r1_sentiment._stimuli(
                splits.train,
                chat,
                config.n_pairs,
                derive_seed(config.seed, "train", i),
            )
            for i in range(config.n_seeds)
        ),
        val=eval_set("val", splits.select),
        test=eval_set("test", splits.report),
        text_of={e.uid: e.text for e in examples},
    )


def _truth_items(
    config: Config,
    name: str,
    chat: ChatFormat,
    truth_sets: Sequence[Example],
) -> Items:
    negation = NEGATIONS[name]
    statements, negated = (
        [e for e in truth_sets if e.meta.get("subset") == subset]
        for subset in (name, negation)
    )
    missing = [
        s for s, rows in ((name, statements), (negation, negated)) if not rows
    ]
    if missing:
        raise ValueError(f"no examples for {' and '.join(missing)}")
    if len(statements) != len(negated):
        raise ValueError(
            f"{negation} does not negate {name} row by row: "
            f"{len(negated)} rows for {len(statements)}"
        )
    pairs = list(zip(statements, negated, strict=True))
    kept = [i for i, (a, b) in enumerate(pairs) if a.label == b.label]
    if kept:
        raise ValueError(
            f"{negation} does not negate {name} row by row: row {kept[0]} "
            "keeps its label"
        )

    def stimuli(split: str, n_pairs: int | None, seed: int) -> StimulusSet:
        own = [pair for pair in pairs if pair[0].split == split]
        return build_stimulus_set(
            [row for pair in own for row in pair],
            PAPER_TEMPLATES["concept"],
            chat,
            pairing="by_group",
            concept=r7_truth.Config().concept,
            group_ids=[pair[0].uid for pair in own for _ in pair],
            n_pairs=n_pairs,
            seed=seed,
        )

    return Items(
        train=tuple(
            stimuli(
                "train",
                config.n_pairs,
                derive_seed(config.seed, "train", name, i),
            )
            for i in range(config.n_seeds)
        ),
        val=stimuli(
            "val",
            config.n_eval_pairs,
            derive_seed(config.seed, "eval", name, "val"),
        ),
        test=stimuli(
            "test",
            config.n_eval_pairs,
            derive_seed(config.seed, "eval", name, "test"),
        ),
        text_of={e.uid: e.text for e in (*statements, *negated)},
    )


def completion_texts(
    stimuli: StimulusSet, text_of: Mapping[str, str], chat: ChatFormat
) -> list[str]:
    """Renders the CCS contrast pair of every sentiment stimulus.

    Args:
        stimuli: The sentiment stimuli.
        text_of: The raw text of every example, by uid.
        chat: The chat format the prompts are rendered with.

    Returns:
        Two texts per stimulus, in stimulus order: its raw text in the CCS
        prompt completed with "positive", then with "negative".
    """
    return [
        template.render(text_of[uid], "", chat).text
        for uid in stimuli.uids
        for template in _COMPLETIONS
    ]
