"""B1: published error detectors beside the P9c signals.

The question is the one of P9c: does a score predict whether the model's own
greedy answer is correct? B1 adds three published detectors as baselines,
scored on the same items, splits and test metric as the P9c predictors: the
exact-answer-token probe (Orgad et al., 2025), discrete semantic entropy
(Farquhar et al., 2024) and EigenScore (Chen et al., 2024). It consumes a
finished P9c run and the P9a run behind it, reuses the P9c split of the
labeled questions, and reads the stored P9c test signals for the paired
comparisons, so no P9c predictor is computed again.

Every score is oriented so that higher means more likely correct, as in
P9c. The probe is oriented on the train labels by its fit; semantic entropy
and EigenScore keep the sign of their papers (higher means more likely
wrong) and are negated. They are scored on the test items only, so there is
no train score to orient them on.
"""

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any
from typing import cast

import numpy as np
from numpy.typing import NDArray
import torch

from tomography.experiments.aggregate import open_finished
from tomography.experiments.base import ExperimentConfig
from tomography.experiments.base import RunDir
from tomography.experiments.p9a_labels import load_labeled
from tomography.experiments.p9c_reliability import PREDICTORS
from tomography.experiments.p9c_reliability import TABLE_COLUMNS
from tomography.experiments.p9c_reliability import Config as P9cConfig
from tomography.experiments.p9c_reliability import _Dataset
from tomography.experiments.p9c_reliability import _split_datasets
from tomography.experiments.p9c_reliability import _test_relations
from tomography.experiments.p9c_reliability import _within_columns
from tomography.experiments.p9c_reliability import _within_pairs
from tomography.judges import load_nli
from tomography.knowledge import DEFAULT_SHOTS
from tomography.knowledge import few_shot_prompt
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.p9.answer_probe import answer_activations
from tomography.p9.answer_probe import fit_answer_probe
from tomography.p9.eigenscore import eigen_layer
from tomography.p9.eigenscore import eigenscores
from tomography.p9.pool import LabeledItem
from tomography.p9.risk import aurc
from tomography.p9.risk import auroc_ci
from tomography.p9.risk import auroc_diff_ci
from tomography.p9.semantic_entropy import Entailer
from tomography.p9.semantic_entropy import entailment_pairs
from tomography.p9.semantic_entropy import semantic_entropy

BASELINES: tuple[str, ...] = ("answer_probe", "semantic_entropy", "eigenscore")
REFERENCES: tuple[str, ...] = ("lat", "p_true")
PAIRED_COLUMNS = ["dataset", "predictor", "reference", "diff", "lo", "hi"]


@dataclass(frozen=True, kw_only=True)
class Config(ExperimentConfig):
    """B1 settings: published error detectors on the items of a P9c run.

    The intervals take `n_boot` and `seed` from the P9c run, so the metric
    of every row is the one of the P9c table; the seed of this config is
    not used.

    Attributes:
        reliability_run: Path of a finished P9c run of the same model.
        baselines: The detectors to score, a subset of `BASELINES`.
        references: The P9c predictors every baseline is compared with in a
            paired difference, names of
            `tomography.experiments.p9c_reliability.PREDICTORS`.
        batch_size: Texts per forward pass.
    """

    experiment: str = "b1"
    model: str = "gpt2"
    reliability_run: str
    baselines: tuple[str, ...] = BASELINES
    references: tuple[str, ...] = REFERENCES
    batch_size: int = 8


def _check_names(
    kind: str, names: tuple[str, ...], allowed: tuple[str, ...]
) -> None:
    unknown = [n for n in names if n not in allowed]
    if unknown:
        raise ValueError(f"unknown {kind}: {', '.join(unknown)}")
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ValueError(f"repeated {kind}: {', '.join(repeated)}")


def _check_config(config: Config) -> None:
    if not config.baselines:
        raise ValueError("config.baselines needs at least one baseline")
    _check_names("baselines", config.baselines, BASELINES)
    _check_names("references", config.references, PREDICTORS)


def _check_alignment(
    dataset: _Dataset, stored: Mapping[str, NDArray[Any]]
) -> None:
    """Checks that the stored P9c test rows are the items of the labels."""
    test = dataset.masks["test"]
    if not np.array_equal(stored["correct"], dataset.correct[test]):
        raise ValueError(
            f"{dataset.name}: the test outcomes stored by the P9c run differ "
            "from those of its labels"
        )
    relation = _test_relations(dataset)
    stored_relation = stored.get("relation")
    if stored_relation is None or relation is None:
        same = stored_relation is None and relation is None
    else:
        same = np.array_equal(relation, stored_relation)
    if not same:
        raise ValueError(
            f"{dataset.name}: the test relations stored by the P9c run differ "
            "from those of its labels"
        )


def _filled(
    column: NDArray[np.float64], reference: NDArray[np.bool_] | None = None
) -> tuple[NDArray[np.float64], int]:
    """Fills a missing score with the lowest finite score of the reference.

    A missing score is the least confident case, as in P9c. A reference of
    None is every scored item.
    """
    finite = np.isfinite(column)
    pool = finite if reference is None else finite & reference
    if not pool.any():
        raise ValueError("a baseline has no finite score to fill with")
    return np.where(finite, column, column[pool].min()), int((~finite).sum())


@dataclass
class _Scores:
    """The test scores of the baselines of one dataset and their notes."""

    test: dict[str, NDArray[np.float64]]
    n_filled: dict[str, int]
    seconds: dict[str, float]


@dataclass(frozen=True)
class _Inputs:
    """What the baselines of one dataset read."""

    config: Config
    p9c: P9cConfig
    loaded: LoadedModel
    dataset: _Dataset
    prompts: list[str]
    entails: Entailer | None

    @property
    def test_items(self) -> list[LabeledItem]:
        """The test items, in the order of the stored P9c signals."""
        test = self.dataset.masks["test"]
        return [i for i, t in zip(self.dataset.items, test, strict=True) if t]

    @property
    def test_prompts(self) -> list[str]:
        """The prompts of the test items."""
        test = self.dataset.masks["test"]
        return [p for p, t in zip(self.prompts, test, strict=True) if t]


def _answer_probe(
    inputs: _Inputs, notes: dict[str, Any]
) -> tuple[NDArray[np.float64], int]:
    """Fits the probe on every layer read and keeps the validation best.

    The layers are those the P9c run read. An empty answer is filled with
    the lowest train score, the P9c rule for the answer-based signals.
    """
    p9c, dataset = inputs.p9c, inputs.dataset
    layers = (
        tuple(range(1, inputs.loaded.n_layers + 1))
        if p9c.layers is None
        else p9c.layers
    )
    h = answer_activations(
        inputs.loaded,
        inputs.prompts,
        [i.label.greedy for i in dataset.items],
        layers=layers,
        bos=p9c.bos,
        batch_size=inputs.config.batch_size,
        share_prefix=True,
    )
    probe = fit_answer_probe(
        h,
        layers,
        dataset.correct,
        train=dataset.masks["train"],
        val=dataset.masks["val"],
    )
    notes.setdefault("answer_probe", {})[dataset.name] = {
        "layer": probe.layer,
        "val_auroc": probe.val_auroc.tolist(),
    }
    column, n_filled = _filled(probe.scores, dataset.masks["train"])
    return column[dataset.masks["test"]], n_filled


def _semantic_entropy(
    inputs: _Inputs, notes: dict[str, Any]
) -> tuple[NDArray[np.float64], int]:
    """Negated discrete semantic entropy of the stored samples."""
    entails = inputs.entails
    assert entails is not None, "checked by execute"
    items = inputs.test_items
    entropy = np.array(
        [
            semantic_entropy(i.example.question, i.label.samples, entails)
            if any(s.strip() for s in i.label.samples)
            else np.nan
            for i in items
        ]
    )
    notes.setdefault("nli_pairs", {})[inputs.dataset.name] = sum(
        len(entailment_pairs(i.example.question, i.label.samples))
        for i in items
    )
    return _filled(-entropy)


def _eigenscore(
    inputs: _Inputs, notes: dict[str, Any]
) -> tuple[NDArray[np.float64], int]:
    """Negated EigenScore of the stored samples at the middle layer."""
    eigen = eigenscores(
        inputs.loaded,
        inputs.test_prompts,
        [i.label.samples for i in inputs.test_items],
        bos=inputs.p9c.bos,
        batch_size=inputs.config.batch_size,
        share_prefix=True,
    )
    notes["eigen_layer"] = eigen_layer(inputs.loaded)
    return _filled(-eigen)


_SCORERS = {
    "answer_probe": _answer_probe,
    "semantic_entropy": _semantic_entropy,
    "eigenscore": _eigenscore,
}


def _score_dataset(inputs: _Inputs, notes: dict[str, Any]) -> _Scores:
    scores = _Scores({}, {}, {})
    for name in inputs.config.baselines:
        start = time.perf_counter()
        values, n_filled = _SCORERS[name](inputs, notes)
        scores.test[name] = values
        scores.n_filled[name] = n_filled
        scores.seconds[name] = time.perf_counter() - start
    return scores


def _table_rows(
    p9c: P9cConfig, dataset: _Dataset, scores: _Scores
) -> list[list[object]]:
    test = dataset.masks["test"]
    correct = dataset.correct[test]
    relation = _test_relations(dataset)
    n_pairs = 0 if relation is None else _within_pairs(correct, relation)
    rows: list[list[object]] = []
    for name, values in scores.test.items():
        point, lo, hi = auroc_ci(
            values, correct, n_boot=p9c.n_boot, seed=p9c.seed
        )
        rows.append(
            [
                dataset.name,
                name,
                int(test.sum()),
                int((correct == 0).sum()),
                scores.n_filled[name],
                point,
                lo,
                hi,
                aurc(values, correct, seed=p9c.seed),
                *_within_columns(p9c, values, correct, relation, n_pairs),
            ]
        )
    return rows


def _paired_rows(
    p9c: P9cConfig,
    dataset: _Dataset,
    scores: _Scores,
    stored: Mapping[str, NDArray[Any]],
    references: Sequence[str],
) -> list[list[object]]:
    correct = stored["correct"]
    return [
        [
            dataset.name,
            name,
            reference,
            *auroc_diff_ci(
                values,
                stored[reference],
                correct,
                n_boot=p9c.n_boot,
                seed=p9c.seed,
            ),
        ]
        for name, values in scores.test.items()
        for reference in references
    ]


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    reliability: RunDir,
    items: Sequence[LabeledItem],
    *,
    entails: Entailer | None,
) -> None:
    """Scores the published detectors on the items of a P9c run.

    The datasets, their items and their splits are the ones P9c made from
    the same labels; before any model runs, the test outcomes (and, for a
    relational dataset, the test relations) that the P9c run stored must
    equal the ones made here. Writes `table` (the P9c rows of each dataset,
    copied, then one row per baseline in the P9c columns; `n_filled` counts
    the missing scores among the items the baseline scored: every item of
    the dataset for the probe, as P9c counts, and the test items for the
    other two), `table_paired` (the AUROC of every baseline minus that of
    every reference the P9c run stored, with a paired bootstrap interval
    over the test items), `signals_<dataset>` (the filled and oriented test
    scores of the
    baselines and the outcomes) and `summary` (the inputs, the layer and the
    validation AUROC per layer of the probe, the EigenScore layer, the
    entailment pairs judged, the references the P9c run lacks and the
    seconds each baseline took).

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model.
        reliability: The finished P9c run.
        items: The labeled questions of the P9a run behind it.
        entails: The entailment model of semantic entropy, called as
            `entails(premises, hypotheses)`; it may be None when semantic
            entropy is not a baseline.

    Raises:
        ValueError: If the config names an unknown or repeated baseline or
            reference, semantic entropy has no entailment model, the labels
            give other datasets than the P9c run evaluated, its table has
            other columns than the P9c ones, or the stored test outcomes or
            relations differ from those of the labels.
    """
    _check_config(config)
    if "semantic_entropy" in config.baselines and entails is None:
        raise ValueError("semantic entropy needs an entailment model")
    p9c = P9cConfig.from_dict(reliability.config)
    p9c_summary = cast(dict[str, Any], reliability.read_json("summary"))
    datasets, _ = _split_datasets(p9c, items)
    names = [d.name for d in datasets]
    if names != p9c_summary["datasets"]:
        raise ValueError(
            f"the labels give the datasets {names}, but the P9c run evaluated "
            f"{p9c_summary['datasets']}"
        )
    stored = {
        d.name: reliability.read_arrays(f"signals_{d.name}") for d in datasets
    }
    for dataset in datasets:
        _check_alignment(dataset, stored[dataset.name])
    p9c_table = cast(dict[str, Any], reliability.read_json("table"))
    if p9c_table["columns"] != TABLE_COLUMNS:
        raise ValueError(
            f"the P9c table has the columns {p9c_table['columns']}, not "
            f"{TABLE_COLUMNS}"
        )
    notes: dict[str, Any] = {}
    table: list[list[object]] = []
    paired: list[list[object]] = []
    seconds: dict[str, dict[str, float]] = {}
    missing: dict[str, list[str]] = {}
    shots = DEFAULT_SHOTS[: p9c.n_shots]
    for dataset in datasets:
        prompts = [
            few_shot_prompt(i.example.question, shots) for i in dataset.items
        ]
        scores = _score_dataset(
            _Inputs(config, p9c, loaded, dataset, prompts, entails), notes
        )
        absent = [r for r in config.references if r not in stored[dataset.name]]
        if absent:
            missing[dataset.name] = absent
        present = [r for r in config.references if r not in absent]
        table += [r for r in p9c_table["rows"] if r[0] == dataset.name]
        table += _table_rows(p9c, dataset, scores)
        paired += _paired_rows(
            p9c, dataset, scores, stored[dataset.name], present
        )
        seconds[dataset.name] = scores.seconds
        run_dir.write_arrays(
            f"signals_{dataset.name}",
            **scores.test,
            correct=stored[dataset.name]["correct"],
        )
    run_dir.write_json("table", {"columns": TABLE_COLUMNS, "rows": table})
    run_dir.write_json(
        "table_paired", {"columns": PAIRED_COLUMNS, "rows": paired}
    )
    run_dir.write_json(
        "summary",
        {
            "reliability_run": config.reliability_run,
            "labels_run": p9c.labels_run,
            "datasets": names,
            "baselines": list(config.baselines),
            "references": list(config.references),
            **notes,
            "missing_references": missing,
            "seconds": seconds,
        },
    )


def run(config: Config, run_dir: RunDir) -> None:
    """Loads the P9c run, its labels and the models, then runs `execute`.

    The model loads in float32, which the shared few-shot prefix of the
    activations needs (P9a and P9c ran Qwen3 in bfloat16 and `gpt2-xl` in
    float16). The entailment model loads only when semantic entropy is a
    baseline, and on the CPU, so it never shares the GPU with the model.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.

    Raises:
        ValueError: If the config names an unknown or repeated baseline or
            reference, `config.reliability_run` is not a finished run, it is
            a run of another model than `config.model`, or its labels run is
            not a finished run of that model.
    """
    _check_config(config)
    reliability = open_finished(
        config.reliability_run, model=config.model, files=()
    )
    items, _ = load_labeled(
        reliability.config["labels_run"], model=config.model
    )
    entails = (
        load_nli(device="cpu").entails
        if "semantic_entropy" in config.baselines
        else None
    )
    execute(
        config,
        run_dir,
        load_model(config.model, dtype=torch.float32),
        reliability,
        items,
        entails=entails,
    )


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws nothing: B1 writes tables only.

    Args:
        run_dir: A finished B1 run.
        out_dir: Folder of the figures.

    Returns:
        An empty list.
    """
    return []
