"""TB1: word and character n-gram baselines for the reading concepts.

A reader that does not beat a classifier on the surface form of its own
statements reads that form, not a model representation. For each reading
concept a text-only classifier (`tomography.p9.risk.text_scores`, word or
character n-grams) is fitted on the raw training statements of the reader and
scored on the reader's own test pairs, one fit per training set of the reader.
No model is loaded: the statements, the pairs and the seeds do not depend on
the checkpoint, because the chat format only changes how a statement is
rendered and never which statements are drawn.
"""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tomography.datasets.ethics import load_ethics_morality
from tomography.datasets.ethics import load_ethics_utility
from tomography.datasets.facts import load_facts
from tomography.datasets.records import Example
from tomography.datasets.sst2 import load_sst2
from tomography.datasets.stimulus import PairingMode
from tomography.datasets.stimulus import StimulusSet
from tomography.datasets.stimulus import build_stimulus_set
from tomography.experiments import p3_size
from tomography.experiments import r1_sentiment
from tomography.experiments import r2_utility
from tomography.experiments import r3_morality
from tomography.experiments import r5_honesty
from tomography.experiments import r7_truth
from tomography.experiments.base import ExperimentConfig
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.variants import READING_TASKS
from tomography.experiments.variants import eval_pairs
from tomography.experiments.variants import eval_seed
from tomography.experiments.variants import task_set
from tomography.experiments.variants import task_splits
from tomography.experiments.variants import template_variants
from tomography.p9.risk import TEXT_ANALYZERS
from tomography.p9.risk import TextAnalyzer
from tomography.p9.risk import text_scores
from tomography.reading.metrics import group_accuracy
from tomography.reading.metrics import wilson_ci
from tomography.templates import ChatFormat
from tomography.templates import ConceptTemplate

IntArray = NDArray[np.int64]
CONCEPTS: tuple[str, ...] = (
    "sentiment",
    "sentiment_r1",
    "utility",
    "morality",
    "truth",
    "honesty",
    "truth_p3",
    "utility_p3",
)
COLUMNS: tuple[str, ...] = (
    "concept",
    "source",
    "analyzer",
    "accuracy",
    "lo",
    "hi",
    "n_pairs",
    "n_seeds",
    "n_train_pairs",
    "seed_min",
    "seed_max",
)
P3_TRAIN_PAIRS = 64
_RAW = ConceptTemplate("raw", "{stimulus}", tagged=False)
_CHAT = ChatFormat("none")


@dataclass(frozen=True, kw_only=True)
class Config(ExperimentConfig):
    """TB1 settings.

    The sizes, seeds and splits are those of the readers, read from their own
    configs, so a baseline cannot drift from the reader it is compared with.
    `model` is a placeholder: the run does not depend on a checkpoint.

    Attributes:
        concepts: The concepts to score, from `CONCEPTS`. `sentiment` is the
            held-out report split of P3, `sentiment_r1` the validation pairs
            of R1, `truth_p3` and `utility_p3` the report splits of P3 for
            truth and utility, and `honesty` the plain transfer pairs of R5.
        offline: If True, load the datasets from the cache only.
    """

    experiment: str = "tb1"
    model: str = "none"
    concepts: tuple[str, ...] = CONCEPTS
    offline: bool = False

    def __post_init__(self) -> None:
        """Checks the concept list.

        Raises:
            ValueError: If `concepts` is empty or names an unknown concept.
        """
        if not self.concepts:
            raise ValueError("concepts needs at least one concept")
        unknown = [c for c in self.concepts if c not in CONCEPTS]
        if unknown:
            raise ValueError(
                f"unknown concept {unknown[0]!r}, expected one of {CONCEPTS}"
            )


@dataclass(frozen=True, eq=False)
class TextSet:
    """Raw statements with the labels and groups of a stimulus set.

    Attributes:
        texts: The raw statement of each stimulus, not rendered.
        uids: The source example id of each stimulus.
        labels: Binary labels of shape (n,).
        groups: Row indices of shape (g, k), one label-1 member per group.
    """

    texts: tuple[str, ...]
    uids: tuple[str, ...]
    labels: IntArray
    groups: IntArray


@dataclass(frozen=True, eq=False)
class ConceptItems:
    """What one reader is fitted and scored on.

    Attributes:
        concept: The concept id.
        source: The experiment that reports the reader.
        train: One training set per seed of the reader.
        test: The test pairs of the reader.
    """

    concept: str
    source: str
    train: tuple[TextSet, ...]
    test: TextSet


def _text_set(stimuli: StimulusSet, text_of: Mapping[str, str]) -> TextSet:
    return TextSet(
        texts=tuple(text_of[uid] for uid in stimuli.uids),
        uids=stimuli.uids,
        labels=stimuli.labels,
        groups=stimuli.groups,
    )


def _part(
    concept: str, examples: Sequence[Example], split: str
) -> list[Example]:
    rows = [e for e in examples if e.split == split]
    if not rows:
        raise ValueError(f"concept {concept!r} has no examples in {split}")
    return rows


def _pair(
    examples: Sequence[Example],
    *,
    pairing: PairingMode,
    n_pairs: int | None,
    seed: int,
    group_ids: Sequence[str] | None = None,
    strata: Sequence[str] | None = None,
) -> StimulusSet:
    return build_stimulus_set(
        examples,
        _RAW,
        _CHAT,
        pairing=pairing,
        group_ids=group_ids,
        strata=strata,
        n_pairs=n_pairs,
        seed=seed,
    )


def concept_items(
    concept: str,
    sources: Mapping[str, Sequence[Example]],
    config: Config,
) -> ConceptItems:
    """Builds the training sets and the test pairs of one reader.

    The items are those of the reader: the same examples, splits, pairing,
    pair counts and seeds. Only the statement is kept, as raw text.

    Args:
        concept: A name from `CONCEPTS`.
        sources: The examples of each dataset, by `sst2`, `facts`,
            `ethics_utility` and `ethics_morality`.
        config: The run settings, for the base seed.

    Returns:
        The items of the concept.

    Raises:
        ValueError: If the concept is unknown, a dataset is missing, or a
            split the reader needs has no example.
    """
    if concept not in _BUILDERS:
        raise ValueError(
            f"unknown concept {concept!r}, expected one of {CONCEPTS}"
        )
    return _BUILDERS[concept](concept, sources, config)


def _dataset(
    concept: str, sources: Mapping[str, Sequence[Example]], name: str
) -> Sequence[Example]:
    if name not in sources:
        raise ValueError(f"concept {concept!r} needs the dataset {name!r}")
    return sources[name]


def _items(
    concept: str,
    source: str,
    examples: Sequence[Example],
    train: Sequence[StimulusSet],
    test: StimulusSet,
) -> ConceptItems:
    text_of = {e.uid: e.text for e in examples}
    return ConceptItems(
        concept,
        source,
        tuple(_text_set(s, text_of) for s in train),
        _text_set(test, text_of),
    )


def _r1(
    concept: str, sources: Mapping[str, Sequence[Example]], config: Config
) -> ConceptItems:
    reader = r1_sentiment.Config()
    examples = _dataset(concept, sources, "sst2")
    train = r1_sentiment.select_examples(
        examples, split="train", min_words=reader.min_words
    )
    val = r1_sentiment.select_examples(
        examples, split="val", min_words=reader.min_words
    )
    for name, rows in (("train", train), ("val", val)):
        if not rows:
            raise ValueError(f"concept {concept!r} has no examples in {name}")
    return _items(
        concept,
        "r1",
        examples,
        [
            _pair(
                train,
                pairing="label_contrastive",
                n_pairs=reader.n_pairs,
                seed=derive_seed(config.seed, "train", i),
            )
            for i in range(reader.n_seeds)
        ],
        _pair(
            val,
            pairing="label_contrastive",
            n_pairs=reader.n_eval_pairs,
            seed=derive_seed(config.seed, "eval"),
        ),
    )


def _r2(
    concept: str, sources: Mapping[str, Sequence[Example]], config: Config
) -> ConceptItems:
    reader = r2_utility.Config()
    examples = _dataset(concept, sources, "ethics_utility")

    def pair(rows: Sequence[Example], n: int | None, seed: int) -> StimulusSet:
        return _pair(
            rows,
            pairing="by_group",
            group_ids=[str(e.meta["pair"]) for e in rows],
            n_pairs=n,
            seed=seed,
        )

    train = _part(concept, examples, "train")
    return _items(
        concept,
        "r2",
        examples,
        [
            pair(
                train,
                reader.n_pairs,
                derive_seed(config.seed, "train", i),
            )
            for i in range(reader.n_seeds)
        ],
        pair(
            _part(concept, examples, "test"),
            reader.n_eval_pairs,
            derive_seed(config.seed, "eval", "test"),
        ),
    )


def _r3(
    concept: str, sources: Mapping[str, Sequence[Example]], config: Config
) -> ConceptItems:
    reader = r3_morality.Config()
    examples = _dataset(concept, sources, "ethics_morality")
    kept = (
        [e for e in examples if e.meta["is_short"] == 1]
        if reader.short_only
        else list(examples)
    )

    def pair(rows: Sequence[Example], n: int | None, seed: int) -> StimulusSet:
        return _pair(
            rows,
            pairing="label_contrastive",
            strata=(
                [str(e.meta["is_short"]) for e in rows]
                if reader.match_length
                else None
            ),
            n_pairs=n,
            seed=seed,
        )

    train = _part(concept, kept, "train")
    return _items(
        concept,
        "r3",
        kept,
        [
            pair(
                train,
                reader.n_pairs,
                derive_seed(config.seed, "train", i),
            )
            for i in range(reader.n_seeds)
        ],
        pair(
            _part(concept, kept, "test"),
            reader.n_eval_pairs,
            derive_seed(config.seed, "eval", "test"),
        ),
    )


def _facts(
    concept: str,
    source: str,
    sources: Mapping[str, Sequence[Example]],
    config: Config,
    *,
    n_seeds: int,
    n_pairs: int | None,
    n_eval_pairs: int | None,
    eval_key: tuple[str, ...],
) -> ConceptItems:
    examples = _dataset(concept, sources, "facts")
    train = _part(concept, examples, "train")
    return _items(
        concept,
        source,
        examples,
        [
            _pair(
                train,
                pairing="label_contrastive",
                n_pairs=n_pairs,
                seed=derive_seed(config.seed, "train", i),
            )
            for i in range(n_seeds)
        ],
        _pair(
            _part(concept, examples, "test"),
            pairing="label_contrastive",
            n_pairs=n_eval_pairs,
            seed=derive_seed(config.seed, *eval_key),
        ),
    )


def _r7(
    concept: str, sources: Mapping[str, Sequence[Example]], config: Config
) -> ConceptItems:
    reader = r7_truth.Config()
    return _facts(
        concept,
        "r7",
        sources,
        config,
        n_seeds=reader.n_seeds,
        n_pairs=reader.n_pairs,
        n_eval_pairs=reader.n_eval_pairs,
        eval_key=("eval", "test"),
    )


def _r5(
    concept: str, sources: Mapping[str, Sequence[Example]], config: Config
) -> ConceptItems:
    reader = r5_honesty.Config()
    return _facts(
        concept,
        "r5",
        sources,
        config,
        n_seeds=reader.n_seeds,
        n_pairs=reader.n_pairs,
        n_eval_pairs=reader.n_eval_pairs,
        eval_key=("eval", "transfer", "test"),
    )


def _p3(task_name: str) -> Callable[..., ConceptItems]:
    def build(
        concept: str,
        sources: Mapping[str, Sequence[Example]],
        config: Config,
    ) -> ConceptItems:
        reader = p3_size.Config()
        task = READING_TASKS[task_name]
        examples = _dataset(concept, sources, task.source)
        (variant,), _ = template_variants(task, _CHAT, kinds=("paper",))
        splits = task_splits(task, sources, seed=config.seed)
        train = [
            task_set(
                task,
                variant,
                splits.train,
                n_pairs=P3_TRAIN_PAIRS,
                seed=derive_seed(
                    config.seed, "size", task_name, P3_TRAIN_PAIRS, r
                ),
            )
            for r in range(reader.n_seeds)
        ]
        test = task_set(
            task,
            variant,
            splits.report,
            n_pairs=eval_pairs(task, splits.report, reader.n_eval_pairs),
            seed=eval_seed(task, "report", config.seed),
        )
        return _items(concept, "p3", examples, train, test)

    return build


_BUILDERS: Mapping[
    str, Callable[[str, Mapping[str, Sequence[Example]], Config], ConceptItems]
] = {
    "sentiment": _p3("sentiment"),
    "sentiment_r1": _r1,
    "utility": _r2,
    "morality": _r3,
    "truth": _r7,
    "honesty": _r5,
    "truth_p3": _p3("truth"),
    "utility_p3": _p3("utility"),
}


def seed_accuracy(
    train_texts: Sequence[str],
    train_labels: IntArray,
    test: TextSet,
    *,
    analyzer: TextAnalyzer,
) -> float:
    """Scores one fit of the text classifier on the test pairs.

    Args:
        train_texts: The raw training statements.
        train_labels: Their 0/1 labels.
        test: The test pairs of the reader.
        analyzer: "word" or "char" n-grams.

    Returns:
        The share of test groups whose label-1 member scores highest. A tie
        shares the credit, so a constant scorer sits at one half for pairs.

    Raises:
        ValueError: If `analyzer` is unknown or the texts leave no feature.
    """
    scores = text_scores(
        train_texts, train_labels, test.texts, analyzer=analyzer
    )
    return float(group_accuracy(scores, test.groups, test.labels))


def interval(accuracy: float, n: int) -> tuple[float, float]:
    """Wilson 95 percent interval of an accuracy over some test groups.

    Args:
        accuracy: The accuracy, a mean over fits, so the expected wins may
            be fractional.
        n: The number of test groups.

    Returns:
        The lower and upper limits.

    Raises:
        ValueError: If `n` is below 1 or `accuracy` is outside [0, 1].
    """
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError(f"accuracy must be in [0, 1], got {accuracy}")
    return wilson_ci(accuracy * n, n)


def _needed(concepts: Sequence[str]) -> list[str]:
    needs = {
        "sentiment": "sst2",
        "sentiment_r1": "sst2",
        "utility": "ethics_utility",
        "morality": "ethics_morality",
        "truth": "facts",
        "honesty": "facts",
        "truth_p3": "facts",
        "utility_p3": "ethics_utility",
    }
    return list(dict.fromkeys(needs[c] for c in concepts))


def execute(
    config: Config,
    run_dir: RunDir,
    sources: Mapping[str, Sequence[Example]],
) -> None:
    """Scores the word and character baselines of every concept.

    Writes `table.json` (one row per concept and analyzer, see `COLUMNS`) and
    `summary.json` (the scope of the baseline and the accuracy of every fit).

    Args:
        config: The run settings.
        run_dir: The run to write into.
        sources: The examples of each dataset, by `sst2`, `facts`,
            `ethics_utility` and `ethics_morality`.

    Raises:
        ValueError: If a concept lacks a dataset or a split.
    """
    rows: list[list[object]] = []
    fits: dict[str, dict[str, list[float]]] = {}
    for concept in config.concepts:
        items = concept_items(concept, sources, config)
        n = int(items.test.groups.shape[0])
        fits[concept] = {}
        for analyzer in TEXT_ANALYZERS:
            accuracies = [
                seed_accuracy(
                    train.texts, train.labels, items.test, analyzer=analyzer
                )
                for train in items.train
            ]
            mean = float(np.mean(accuracies))
            lo, hi = interval(mean, n)
            fits[concept][analyzer] = accuracies
            rows.append(
                [
                    concept,
                    items.source,
                    analyzer,
                    mean,
                    lo,
                    hi,
                    n,
                    len(accuracies),
                    int(items.train[0].groups.shape[0]),
                    min(accuracies),
                    max(accuracies),
                ]
            )
    run_dir.write_json("table", {"columns": list(COLUMNS), "rows": rows})
    run_dir.write_json(
        "summary",
        {
            "model_independent": True,
            "concepts": list(config.concepts),
            "analyzers": list(TEXT_ANALYZERS),
            "ties": "share the credit",
            "interval": "wilson 95 percent",
            "seed": config.seed,
            "fit_accuracy": fits,
        },
    )


def run(config: Config, run_dir: RunDir) -> None:
    """Loads the datasets of the chosen concepts, then runs `execute`.

    Args:
        config: The run settings.
        run_dir: The run to write into.
    """
    loaders: dict[str, Callable[..., list[Example]]] = {
        "sst2": load_sst2,
        "facts": load_facts,
        "ethics_utility": load_ethics_utility,
        "ethics_morality": load_ethics_morality,
    }
    sources = {
        name: loaders[name](offline=config.offline)
        for name in _needed(config.concepts)
    }
    execute(config, run_dir, sources)


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws nothing: the run writes tables only.

    Args:
        run_dir: A finished TB1 run.
        out_dir: Folder of the figures.

    Returns:
        An empty list.
    """
    return []
