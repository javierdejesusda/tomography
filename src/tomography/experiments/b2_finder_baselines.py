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
half and a report half as for the reading task `sentiment`, each keeping
the sentences R1 keeps (at least 3 words). CCS reads
contrast pairs: the statement and its negation for truth, and for sentiment
each text in one fixed prompt in the form of Burns et al., completed once
with "positive" and once with "negative".
"""

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tomography.activations import Activations
from tomography.activations import collect_activations
from tomography.datasets.geometry_of_truth import load_geometry_of_truth
from tomography.datasets.records import Example
from tomography.datasets.sst2 import load_sst2
from tomography.datasets.stimulus import StimulusSet
from tomography.datasets.stimulus import build_stimulus_set
from tomography.experiments import r1_sentiment
from tomography.experiments import r7_truth
from tomography.experiments.base import ExperimentConfig
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.sweep import chat_format_for
from tomography.experiments.sweep import plot_curves_file
from tomography.experiments.sweep import sweep_from_activations
from tomography.experiments.sweep import write_curves
from tomography.experiments.tb1_text_baselines import _text_set
from tomography.experiments.tb1_text_baselines import interval
from tomography.experiments.tb1_text_baselines import seed_accuracy
from tomography.experiments.variants import READING_TASKS
from tomography.experiments.variants import task_splits
from tomography.model import BosMode
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.p9.risk import TEXT_ANALYZERS
from tomography.reading.ccs import CCSProbe
from tomography.reading.ccs import fit_ccs
from tomography.reading.directions import Method
from tomography.reading.metrics import best_layer
from tomography.reading.metrics import bootstrap_ci
from tomography.reading.metrics import group_accuracy
from tomography.templates import PAPER_TEMPLATES
from tomography.templates import ChatFormat
from tomography.templates import ConceptTemplate

FINDERS: tuple[Method, ...] = (
    "pca_diff",
    "mean_diff",
    "whitened_mean_diff",
    "logistic",
)
LAYER_SERIES: tuple[str, ...] = (*FINDERS, "ccs", "random", "shuffled")
TEXT_CONTROLS: tuple[str, ...] = tuple(f"text_{a}" for a in TEXT_ANALYZERS)
KIND_OF: Mapping[str, str] = MappingProxyType(
    {
        "pca_diff": "reader",
        **{name: "baseline" for name in (*FINDERS[1:], "ccs")},
        **{name: "control" for name in ("random", "shuffled", *TEXT_CONTROLS)},
    }
)
COLUMNS: tuple[str, ...] = (
    "dataset",
    "method",
    "kind",
    "val_layer",
    "accuracy",
    "wilson_lo",
    "wilson_hi",
    "boot_lo",
    "boot_hi",
    "n_test_pairs",
    "n_seeds",
)
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
    `sentiment`, without the phrases shorter than R1's `min_words`. A
    truth set uses R7's template and concept word; each of its groups is
    a statement and then its negation, exactly one of them true, in the
    loader's split of the statement.

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
        # R1 drops the short validation phrases, which task_splits keeps.
        kept = r1_sentiment.select_examples(
            rows, split="val", min_words=r1_sentiment.Config().min_words
        )
        seed = derive_seed(config.seed, "eval", "sentiment", part)
        return r1_sentiment._stimuli(kept, chat, config.n_eval_pairs, seed)

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

    val, test = (
        stimuli(
            s, config.n_eval_pairs, derive_seed(config.seed, "eval", name, s)
        )
        for s in ("val", "test")
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
        val=val,
        test=test,
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


@dataclass(frozen=True, eq=False)
class Contrast:
    """The CCS contrast pairs that stand for the stimuli of one part.

    Attributes:
        h: Activations of the contrast rows, shape (K, n_c, d).
        pairs: Rows of `h` of shape (m, 2), the first member first.
        labels: Shape (n_c,), 1 for a true contrast row.
        stimulus_row: The contrast row each stimulus of the part stands for,
            shape (n,).
    """

    h: NDArray[np.float32]
    pairs: NDArray[np.int64]
    labels: NDArray[np.int64]
    stimulus_row: NDArray[np.int64]


def pair_contrast(stimuli: StimulusSet, act: Activations) -> Contrast:
    """The contrast pairs of a truth part: its own statement-first groups.

    Args:
        stimuli: Stimuli whose groups are a statement and its negation.
        act: Their activations, one row per stimulus.

    Returns:
        The groups as CCS pairs over the part's own activations.
    """
    n = stimuli.labels.shape[0]
    return Contrast(
        act.data, stimuli.groups, stimuli.labels, np.arange(n, dtype=np.int64)
    )


def completion_contrast(stimuli: StimulusSet, act: Activations) -> Contrast:
    """The contrast pairs of a sentiment part: each text's two completions.

    Args:
        stimuli: The sentiment stimuli.
        act: Activations of `completion_texts(stimuli, ...)`, one row per
            text: rows 2j and 2j + 1 complete stimulus j with "positive"
            and "negative".

    Returns:
        One pair per stimulus; its first member is true when the stimulus
        is positive.
    """
    n = stimuli.labels.shape[0]
    return Contrast(
        act.data,
        np.arange(2 * n, dtype=np.int64).reshape(n, 2),
        np.stack([stimuli.labels, 1 - stimuli.labels], axis=1).reshape(-1),
        2 * np.arange(n, dtype=np.int64),
    )


@dataclass(frozen=True, eq=False)
class Part:
    """One training, validation or test set with its activations.

    Attributes:
        stimuli: The stimuli the finders read.
        act: Their activations, one row per stimulus.
        contrast: The contrast pairs CCS reads in their place.
    """

    stimuli: StimulusSet
    act: Activations
    contrast: Contrast


@dataclass(frozen=True, eq=False)
class Dataset:
    """Every part of one dataset, ready to score.

    Attributes:
        name: A name from `DATASETS`.
        train: One training part per seed.
        val: The part that chooses the layer.
        test: The part every reported number is computed on.
        text_of: The raw text of every example, by uid.
    """

    name: str
    train: tuple[Part, ...]
    val: Part
    test: Part
    text_of: Mapping[str, str]


@dataclass(frozen=True, eq=False)
class Scores:
    """The accuracy of every series on one dataset.

    Attributes:
        layers: The layer of each column.
        val: Validation group accuracy of each name in `LAYER_SERIES`,
            shape (N, K): seed and layer.
        test: Test group accuracy, the same shapes.
        text: Test group accuracy of each name in `TEXT_CONTROLS`, shape
            (N,).
        ccs_losses: The CCS loss on the training pairs, shape (N, K).
    """

    layers: tuple[int, ...]
    val: Mapping[str, NDArray[np.float64]]
    test: Mapping[str, NDArray[np.float64]]
    text: Mapping[str, NDArray[np.float64]]
    ccs_losses: NDArray[np.float64]


def score_dataset(config: Config, data: Dataset) -> Scores:
    """Fits every series on each training part and scores both held-out parts.

    The finders and the random and shuffled controls are the series of the
    shared reading sweep (`sweep_from_activations`), the controls those of
    LAT; CCS is fitted on the contrast pairs of each training part and
    scored from its pair probabilities, never from its reader; the
    text-only controls are TB1's classifiers on the raw texts.

    Args:
        config: The experiment settings.
        data: The dataset.

    Returns:
        The accuracy of every series.
    """
    train = [(part.stimuli, part.act) for part in data.train]
    evals = {
        "val": (data.val.stimuli, data.val.act),
        "test": (data.test.stimuli, data.test.act),
    }
    val: dict[str, NDArray[np.float64]] = {}
    test: dict[str, NDArray[np.float64]] = {}
    for finder in FINDERS:
        sweep = sweep_from_activations(
            train, evals, method=finder, seed=config.seed
        )
        for split, out in (("val", val), ("test", test)):
            reader, random, shuffled = sweep.per_seed(split)
            out[finder] = reader
            if KIND_OF[finder] == "reader":
                out["random"], out["shuffled"] = random, shuffled
    layers = data.val.act.layers
    probes = [
        fit_ccs(
            part.contrast.h,
            layers,
            pairs=part.contrast.pairs,
            labels=part.contrast.labels,
            restarts=config.ccs_restarts,
            epochs=config.ccs_epochs,
            lr=config.ccs_lr,
            seed=derive_seed(config.seed, "ccs", data.name, i),
        )
        for i, part in enumerate(data.train)
    ]
    val["ccs"] = np.stack([_ccs_accuracy(p, data.val) for p in probes])
    test["ccs"] = np.stack([_ccs_accuracy(p, data.test) for p in probes])
    texts = _text_set(data.test.stimuli, data.text_of)
    trains = [_text_set(part.stimuli, data.text_of) for part in data.train]
    text = {
        f"text_{analyzer}": np.array(
            [
                seed_accuracy(t.texts, t.labels, texts, analyzer=analyzer)
                for t in trains
            ]
        )
        for analyzer in TEXT_ANALYZERS
    }
    return Scores(
        layers, val, test, text, np.stack([probe.losses for probe in probes])
    )


def _ccs_accuracy(probe: CCSProbe, part: Part) -> NDArray[np.float64]:
    """Group accuracy of CCS from its pair probabilities, per layer.

    A stimulus scores the probability that its contrast row is true: the
    pair probability for a first member and its complement for a second.
    The probe's reader is not used: it drops the per-set centring, so on a
    statement set and a negation set with different means its comparisons
    are shifted.
    """
    contrast = part.contrast
    probability = probe.pair_probabilities(contrast.h, contrast.pairs)
    true = np.zeros((probability.shape[0], contrast.h.shape[1]))
    true[:, contrast.pairs[:, 0]] = probability
    true[:, contrast.pairs[:, 1]] = 1.0 - probability
    scores = true[:, contrast.stimulus_row]
    return group_accuracy(scores, part.stimuli.groups, part.stimuli.labels)


def table_rows(
    name: str, scores: Scores, n_test_pairs: int, config: Config
) -> list[list[Any]]:
    """Reports every series at the layer its own validation curve chooses.

    The layer is the best of the mean validation accuracy over seeds, the
    first one on ties; the text-only controls read no layer. The accuracy is
    the mean test accuracy of the seeds at that layer, with a Wilson
    interval over the test pairs and a bootstrap interval over the seeds.

    Args:
        name: The dataset.
        scores: Its accuracies.
        n_test_pairs: The number of test pairs (groups).
        config: The experiment settings, for the bootstrap.

    Returns:
        One row per series in `COLUMNS` order, the layer series first.
    """
    rows = []
    for series in (*LAYER_SERIES, *TEXT_CONTROLS):
        layer: int | None = None
        if series in scores.text:
            per_seed = scores.text[series]
        else:
            layer = best_layer(scores.val[series].mean(axis=0), scores.layers)
            per_seed = scores.test[series][:, scores.layers.index(layer)]
        accuracy = float(per_seed.mean())
        lo, hi = bootstrap_ci(per_seed, n_boot=config.n_boot, seed=config.seed)
        rows.append(
            [
                name,
                series,
                KIND_OF[series],
                layer,
                accuracy,
                *interval(accuracy, n_test_pairs),
                float(lo),
                float(hi),
                n_test_pairs,
                int(per_seed.shape[0]),
            ]
        )
    return rows


def prepare_dataset(
    config: Config,
    name: str,
    items: Items,
    loaded: LoadedModel,
    chat: ChatFormat,
) -> Dataset:
    """Reads the last-token activations of every part of a dataset.

    Args:
        config: The experiment settings.
        name: A name from `DATASETS`.
        items: Its stimuli, from `dataset_items`.
        loaded: The wrapped model.
        chat: The chat format the items were rendered with.

    Returns:
        The dataset; for sentiment the contrast rows are the activations of
        `completion_texts`, for truth the parts' own rows.
    """
    layers = (
        tuple(range(1, loaded.n_layers + 1))
        if config.layers is None
        else config.layers
    )

    def collect(texts: Sequence[str]) -> Activations:
        return collect_activations(
            loaded,
            texts,
            layers=layers,
            position="last",
            bos=config.bos,
            batch_size=config.batch_size,
        )

    def part(stimuli: StimulusSet) -> Part:
        act = collect(stimuli.texts)
        if name == "sentiment":
            texts = completion_texts(stimuli, items.text_of, chat)
            return Part(
                stimuli, act, completion_contrast(stimuli, collect(texts))
            )
        return Part(stimuli, act, pair_contrast(stimuli, act))

    return Dataset(
        name,
        tuple(part(stimuli) for stimuli in items.train),
        part(items.val),
        part(items.test),
        items.text_of,
    )


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    sentiment: Sequence[Example],
    truth_sets: Sequence[Example],
) -> None:
    """Scores every finder and control on every dataset.

    The items of all datasets are built before any forward pass. Writes
    `table.json` (one row per dataset and series, see `COLUMNS`),
    `curves_<dataset>_<val|test>` per dataset (the `LAYER_SERIES` accuracy
    per seed and layer) and `summary.json` (the pair counts and the mean
    CCS training loss at its validation layer per dataset, and the kind of
    every series).

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model.
        sentiment: SST-2 examples; may be empty without `sentiment`.
        truth_sets: Geometry of Truth examples; may be empty without a
            truth dataset.

    Raises:
        ValueError: If a dataset lacks examples or pairs.
    """
    chat = chat_format_for(loaded)
    items = {
        name: dataset_items(config, name, chat, sentiment, truth_sets)
        for name in config.datasets
    }
    rows: list[list[Any]] = []
    datasets: dict[str, dict[str, Any]] = {}
    for name, parts in items.items():
        scores = score_dataset(
            config, prepare_dataset(config, name, parts, loaded, chat)
        )
        for split, accuracy in (("val", scores.val), ("test", scores.test)):
            write_curves(
                run_dir,
                np.stack([accuracy[series] for series in LAYER_SERIES]),
                scores.layers,
                series=LAYER_SERIES,
                stem=f"curves_{name}_{split}",
                chance=0.5,
                n_layers=loaded.n_layers,
                n_boot=config.n_boot,
                seed=config.seed,
            )
        n_test = int(parts.test.groups.shape[0])
        rows += table_rows(name, scores, n_test, config)
        ccs_layer = best_layer(scores.val["ccs"].mean(axis=0), scores.layers)
        losses = scores.ccs_losses[:, scores.layers.index(ccs_layer)]
        datasets[name] = {
            "n_train_pairs": int(parts.train[0].groups.shape[0]),
            "n_val_pairs": int(parts.val.groups.shape[0]),
            "n_test_pairs": n_test,
            "ccs_loss": float(losses.mean()),
        }
    run_dir.write_json("table", {"columns": list(COLUMNS), "rows": rows})
    run_dir.write_json(
        "summary", {"datasets": datasets, "kinds": dict(KIND_OF)}
    )


def run(config: Config, run_dir: RunDir) -> None:
    """Loads the model and the datasets it reads, then runs `execute`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
    """
    subsets = [
        subset
        for name in config.datasets
        if name in NEGATIONS
        for subset in (name, NEGATIONS[name])
    ]
    loaded = load_model(config.model)
    execute(
        config,
        run_dir,
        loaded,
        load_sst2() if "sentiment" in config.datasets else [],
        load_geometry_of_truth(subsets=subsets) if subsets else [],
    )


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws the B2 layer curves of every dataset and held-out split.

    Args:
        run_dir: A finished B2 run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths of every `curves_*` file, sorted by stem.
    """
    stems = sorted(p.stem for p in run_dir.path.glob("curves_*.npz"))
    return [
        path
        for stem in stems
        for path in plot_curves_file(run_dir, out_dir, stem=stem)
    ]
