"""M4: layer ensembles, cross-task transfer and pooled readers."""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from typing import NamedTuple
from typing import cast

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
import torch

from tomography.datasets.records import Example
from tomography.datasets.stimulus import StimulusSet
from tomography.experiments import m2_readers as m2
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.p2_positions import misleading_template
from tomography.experiments.p2_positions import with_decoys
from tomography.experiments.sweep import chat_format_for
from tomography.experiments.sweep import shuffle_within_groups
from tomography.experiments.variants import ReadingTask
from tomography.experiments.variants import TaskSplits
from tomography.experiments.variants import Variant
from tomography.experiments.variants import eval_seed
from tomography.experiments.variants import load_sources
from tomography.experiments.variants import target_of
from tomography.experiments.variants import task_set
from tomography.experiments.variants import task_splits
from tomography.experiments.variants import template_variants
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.reading import token_forest
from tomography.reading.ensembles import GroupedForest
from tomography.reading.ensembles import cross_fit_scores
from tomography.reading.ensembles import fit_grouped_forest
from tomography.reading.ensembles import fit_mlp_stack
from tomography.reading.ensembles import tune_grouped_forest
from tomography.reading.ensembles import tune_stack
from tomography.reading.ensembles import weighted_mean
from tomography.reading.metrics import auroc
from tomography.reading.metrics import best_layer
from tomography.reading.metrics import bootstrap_ci
from tomography.reading.metrics import group_accuracy
from tomography.reading.reader import RepReader
from tomography.reading.stats import LayerChoice
from tomography.reading.stats import _standardize_layer_scores
from tomography.reading.stats import best_validation_layer
from tomography.reading.stats import delong_test
from tomography.reading.stats import holm_correction
from tomography.reading.stats import mcnemar_test
from tomography.reading.stats import paired_accuracy_diff
from tomography.reading.stats import paired_auroc_diff

Rows = NDArray[np.float32]
Scores = NDArray[np.float64]
Indices = NDArray[np.int64]
READOUTS: tuple[m2.Readout, ...] = ("last", "max", "mean")
CROSS_READOUTS: tuple[m2.Readout, ...] = ("last", "max")
BASE_SCORERS = ("logistic", "mean_diff")
BASE_RULES = ("best", "top_k")
BRIDGE_LAYERS = token_forest.LITERAL_LAYERS


@dataclass(frozen=True, kw_only=True)
class Config(m2.Config):
    """M4 settings layered over the reference run's M2 configuration.

    The inherited fields identify the M2 comparison. M4 reads last, max
    and mean regardless of which additional readouts M2 ran.

    Attributes:
        m2_root: Directory containing the stored M2 model directories.
        pilot_draws: Limits new readers to the first draws, after checking
            equality on the complete M2 draw set.
        n_folds: Pair-grouped cross-fitting folds.
        n_trees: Trees per forest fit.
        mlp_repeats: One or five seeds per selected MLP penalty.
        matched_pairs_per_task: Budget per task in the matched pooled arm.
        literal_alignment: Right alignment for the primary forest; left
            alignment for the qwen3-0.6b-base sensitivity run.
    """

    experiment: str = "m4"
    token_layers: tuple[int, ...] | None = BRIDGE_LAYERS
    m2_root: str = "results/m2"
    pilot_draws: int | None = None
    n_folds: int = 8
    n_trees: int = 500
    mlp_repeats: int = 1
    matched_pairs_per_task: int = 16
    literal_alignment: token_forest.Alignment = "right"

    def m2_config(self) -> m2.Config:
        """Returns the M2 comparison config without M4-only fields."""
        values = self.to_dict()
        fields = {f.name: values[f.name] for f in dataclasses.fields(m2.Config)}
        return m2.Config.from_dict({**fields, "experiment": "m2"})


def find_m2_run(config: Config, root: Path) -> Path:
    """Finds exactly one corresponding M2 full-run configuration.

    Args:
        config: Experiment config sharing its baseline fields with M2.
        root: Directory containing the M2 model directories.

    Returns:
        The matching, hash-verified result directory.

    Raises:
        ValueError: If no config matches, multiple configs match, or the
            matching directory has the wrong config digest.
    """
    expected = config.m2_config()
    fields = expected.to_dict()
    matches = []
    for path in sorted((root / config.model).glob("*/config.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        # Full runs predating reader_series used the implicit True default.
        normalized = {"reader_series": True, **payload}
        if normalized == fields:
            matches.append((path.parent, payload))
    if not matches:
        raise ValueError(f"no matching M2 config under {root / config.model}")
    if len(matches) != 1:
        raise ValueError(
            f"multiple matching M2 configs: {[p for p, _ in matches]}"
        )
    path, payload = matches[0]
    if "reader_series" in payload:
        digest = expected.config_hash()
    else:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode()).hexdigest()[:12]
    if path.name != digest:
        raise ValueError(
            f"M2 config hash mismatch at {path}: expected {digest}"
        )
    return path


def _gate_index(
    rows: Sequence[Sequence[Any]],
) -> dict[tuple[str, str, str, str], Sequence[Any]]:
    indexed = {}
    for row in rows:
        task, condition, scorer, readout, rule, series = row[:6]
        if (
            scorer in BASE_SCORERS
            and readout == "last"
            and rule in BASE_RULES
            and series == "reader"
        ):
            key = (task, condition, scorer, rule)
            if key in indexed:
                raise ValueError(f"duplicate M2 equality row: {key}")
            indexed[key] = row
    return indexed


def check_m2_equality(
    current: Sequence[Sequence[Any]], stored: Sequence[Sequence[Any]]
) -> None:
    """Requires matching M2 last-token baseline rows to absolute 1e-6.

    Args:
        current: The recomputed rows in M2 column order.
        stored: The stored M2 rows in the same column order.

    Raises:
        ValueError: If either table has missing, extra, duplicate or
            different baseline rows.
    """
    actual, expected = _gate_index(current), _gate_index(stored)
    if not expected or actual.keys() != expected.keys():
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        raise ValueError(f"M2 equality rows missing={missing}, extra={extra}")
    for key, target in expected.items():
        for column, value, reference in zip(
            m2.COLUMNS[6:], actual[key][6:], target[6:], strict=True
        ):
            if column in ("layers", "n_seeds"):
                equal = value == reference
            else:
                value_missing = value is None or not math.isfinite(value)
                reference_missing = reference is None or not math.isfinite(
                    reference
                )
                if value_missing or reference_missing:
                    equal = value_missing == reference_missing
                else:
                    equal = math.isclose(
                        value, reference, rel_tol=0.0, abs_tol=1e-6
                    )
            if not equal:
                raise ValueError(
                    f"M2 equality failed for {key}, {column}: "
                    f"{value!r} vs stored {reference!r}"
                )


@dataclass(frozen=True)
class _LabeledRows:
    """Layer-major row features with labels and M2 pair/group indices."""

    data: Rows
    layers: tuple[int, ...]
    labels: Indices
    pairs: Indices
    groups: Indices


@dataclass(frozen=True)
class _TunedLR:
    """One selected model per layer and its tuning evidence."""

    reader: RepReader
    penalties: tuple[float, ...]
    select_scores: Scores
    select_accuracy: Scores


def _fit_lr(
    config: Config, train: _LabeledRows, penalty: float, seed: int
) -> RepReader:
    return cast(
        RepReader,
        m2._row_reader(
            config,
            "logistic",
            train.data,
            train.layers,
            train.labels,
            train.pairs,
            train.groups,
            penalty,
            seed,
            False,
            None,
        ),
    )


def _choose_lr(
    fits: Sequence[RepReader],
    grid: tuple[float, ...],
    select: Scores,
    metrics: Scores,
) -> _TunedLR:
    best = m2.tune_per_layer(metrics)
    positions = np.arange(len(best))
    chosen = [fits[i] for i in best]
    reader = RepReader(
        "logistic",
        fits[0].layers,
        np.stack([fit.directions[i] for i, fit in enumerate(chosen)]),
        np.stack([fit.means[i] for i, fit in enumerate(chosen)]),
        np.array(
            [fit.signs[i] for i, fit in enumerate(chosen)], dtype=np.float32
        ),
    )
    return _TunedLR(
        reader,
        tuple(grid[i] for i in best),
        select[best, positions],
        metrics[best, positions],
    )


def _tuned_lr(
    config: Config,
    train: _LabeledRows,
    validation: Sequence[_LabeledRows],
    *,
    seed: int,
) -> list[_TunedLR]:
    """Fits M2's LR grid once for both per-task and shared tuning.

    Args:
        config: The M2 feature preprocessing and strongest-first grid.
        train: Training activations and orientation data.
        validation: Validation tasks, weighted equally for shared tuning.
        seed: Seed of the row readers.

    Returns:
        A fitted reader tuned on each validation set, then one tuned on
        mean select accuracy across all sets. The shared reader carries
        the first set's select scores and the mean select accuracy.
    """
    grid = config.logistic_c
    fits = [_fit_lr(config, train, value, seed) for value in grid]
    values = [
        np.stack([fit.score(rows.data) for fit in fits]).astype(np.float64)
        for rows in validation
    ]
    metrics = [
        group_accuracy(scores, rows.groups, rows.labels)
        for scores, rows in zip(values, validation, strict=True)
    ]
    choices = [
        _choose_lr(fits, grid, scores, accuracy)
        for scores, accuracy in zip(values, metrics, strict=True)
    ]
    if len(validation) > 1:
        choices.append(
            _choose_lr(fits, grid, values[0], np.mean(metrics, axis=0))
        )
    return choices


@dataclass(frozen=True)
class _Baseline:
    """The baseline scores, tuned logistic fits and draw-level ties."""

    entries: list[m2._Entry]
    logistic: dict[m2.Readout, list[_TunedLR]]
    ties: dict[tuple[str, m2.Readout], list[bool]]


def _labeled(
    data: Rows, layers: tuple[int, ...], items: StimulusSet
) -> _LabeledRows:
    return _LabeledRows(data, layers, items.labels, items.pairs, items.groups)


def _draw_rows(
    splits: m2._Splits, readout: m2.Readout, i: int
) -> tuple[_LabeledRows, _LabeledRows, list[Rows]]:
    position = m2.POSITION_OF[readout]
    train = splits.train[position][i]
    select = splits.select[position]
    return (
        _labeled(train.data, train.layers, splits.train_items[i]),
        _labeled(select.data, select.layers, splits.select_items),
        [act.data for act in splits.report[position]],
    )


def _layer_choice(
    read: LayerChoice, report: Scores, labels: Indices
) -> m2._Choice:
    return m2._Choice(
        read.layers,
        read.val_scores.copy(),
        read.test_scores.reshape(report.shape[1:]).copy(),
        labels,
    )


def _draw_delta(
    a: Sequence[m2._Choice],
    b: Sequence[m2._Choice],
    condition: int,
    items: StimulusSet,
) -> tuple[Scores, Scores]:
    scores_a = np.stack([choice.report[condition] for choice in a])
    scores_b = np.stack([choice.report[condition] for choice in b])
    return (
        auroc(scores_a, items.labels) - auroc(scores_b, items.labels),
        group_accuracy(scores_a, items.groups, items.labels)
        - group_accuracy(scores_b, items.groups, items.labels),
    )


def _baseline(
    config: Config, task: ReadingTask, splits: m2._Splits
) -> _Baseline:
    """Fits the baseline readers on every draw at every readout.

    For each readout, scorer in `BASE_SCORERS` and draw, the L2 LR is
    tuned per layer on the select pairs and diff-means comes from M2
    `_tuned`. Three layer rules read the same select metric: `best`
    (first maximum), `top_k` (M2 `read_top_k`) and `best_deep` (deepest
    maximum, `tie_break="last"`). A draw is tied when more than one layer
    shares the maximum select pair accuracy.

    Args:
        config: The run configuration.
        task: The reading task.
        splits: The task's train, select and report activations.

    Returns:
        The `reader` entries per scorer, readout and rule, the tuned LR of
        every draw per readout (reused by the new readers), and the
        per-draw tie flags per scorer and readout.
    """
    entries: list[m2._Entry] = []
    logistic: dict[m2.Readout, list[_TunedLR]] = {}
    ties: dict[tuple[str, m2.Readout], list[bool]] = {}
    for readout in READOUTS:
        logistic[readout] = []
        for scorer in BASE_SCORERS:
            choices: dict[str, list[m2._Choice]] = {
                rule: [] for rule in (*BASE_RULES, "best_deep")
            }
            tied = []
            for i in range(config.n_seeds):
                train, select, report = _draw_rows(splits, readout, i)
                seed = derive_seed(
                    config.seed, "fit", task.name, scorer, readout, i
                )
                if scorer == "logistic":
                    fitted = _tuned_lr(config, train, [select], seed=seed)[0]
                    logistic[readout].append(fitted)
                    val, metric = fitted.select_scores, fitted.select_accuracy
                    test = np.stack(
                        [fitted.reader.score(rows) for rows in report], axis=1
                    ).astype(np.float64)
                else:
                    val, test, metric = m2._tuned(
                        config,
                        scorer,
                        readout,
                        splits,
                        i,
                        train.labels,
                        select.labels,
                        seed,
                    )
                layers = train.layers
                flat = test.reshape(len(layers), -1)
                best = best_validation_layer(metric, layers, val, flat)
                deep = best_validation_layer(
                    metric, layers, val, flat, tie_break="last"
                )
                top = m2.read_top_k(metric, layers, val, flat, k=config.top_k)
                for rule, read in zip(choices, (best, top, deep), strict=True):
                    choices[rule].append(
                        _layer_choice(read, test, select.labels)
                    )
                tied.append(bool(np.count_nonzero(metric == metric.max()) > 1))
            entries.extend(
                m2._Entry(m2.Cell(scorer, readout, rule), "reader", draws)
                for rule, draws in choices.items()
            )
            ties[scorer, readout] = tied
    return _Baseline(entries, logistic, ties)


@dataclass(frozen=True)
class _Prepared:
    task: ReadingTask
    splits: m2._Splits
    baseline: _Baseline
    skipped: dict[str, str]


def _tie_rows(source: _Prepared, take: int) -> list[dict[str, Any]]:
    """Compares the `best` and `best_deep` layer rules per baseline.

    Args:
        source: A prepared task with its baseline fits.
        take: Number of leading draws to summarize.

    Returns:
        One row per scorer and readout: the share of draws tied at the
        maximum select pair accuracy, the share where the two rules pick
        different layers, the per-draw flags, and per report condition the
        per-draw AUROC and pair accuracy differences (`best_deep` minus
        `best`).
    """
    draws = {
        entry.cell: entry.choices[:take] for entry in source.baseline.entries
    }
    summaries = []
    for (scorer, readout), tied in source.baseline.ties.items():
        first = draws[m2.Cell(scorer, readout, "best")]
        deep = draws[m2.Cell(scorer, readout, "best_deep")]
        different = [
            a.layers != b.layers for a, b in zip(first, deep, strict=True)
        ]
        effects = {}
        for i, (condition, items) in enumerate(
            source.splits.report_items.items()
        ):
            area, pair = _draw_delta(deep, first, i, items)
            effects[condition] = {
                "auroc_draw_diff": area.tolist(),
                "pair_draw_diff": pair.tolist(),
            }
        summaries.append(
            {
                "task": source.task.name,
                "scorer": scorer,
                "readout": readout,
                "tie_share": float(np.mean(tied[:take])),
                "different_share": float(np.mean(different)),
                "draw_ties": tied[:take],
                "draw_layer_differs": different,
                "effects": effects,
            }
        )
    return summaries


def _prepare(
    config: Config,
    loaded: LoadedModel,
    tasks: Sequence[ReadingTask],
    examples: Mapping[str, Sequence[Example]],
) -> list[_Prepared]:
    prepared = []
    for task in tasks:
        splits, skipped = m2._prepare_task(
            config,
            loaded,
            task,
            READOUTS,
            m2._task_examples(task.name, examples),
        )
        prepared.append(
            _Prepared(task, splits, _baseline(config, task, splits), skipped)
        )
    return prepared


def _stored_controls(
    rows: Sequence[list[Any]],
) -> list[list[Any]]:
    return [
        row
        for row in rows
        if (
            row[2] == "text"
            or (
                row[3] in READOUTS
                and (
                    row[2] == "random"
                    or (row[2] == "mlp" and row[4] == "joint")
                )
            )
        )
    ]


def _write_entries(
    config: Config, run_dir: RunDir, source: _Prepared, entries: list[m2._Entry]
) -> list[list[Any]]:
    rows = [
        m2._metric_rows(config, source.task, source.splits, entry)
        for entry in entries
    ]
    m2._write_task(run_dir, source.task, source.splits, entries, rows)
    return [row for block in rows for row in block]


def _literal_entries(
    config: Config,
    loaded: LoadedModel,
    source: _Prepared,
    layers: tuple[int, ...],
    take: int,
) -> tuple[list[m2._Entry], list[dict[str, Any]]]:
    """Fits the literal token forest and its length control on each draw.

    Args:
        config: Forest size, alignment, BOS policy and seed.
        loaded: Causal model for tokenization and forward passes.
        source: The task's splits.
        layers: Retained literal layers; empty skips the family.
        take: Number of training draws.

    Returns:
        The forest and length-control entries, and one fit diagnostic per
        draw with its training-derived token width.
    """
    if not layers:
        return [], []
    splits = source.splits
    draws = token_forest.score_splits(
        loaded,
        splits.train_items[:take],
        splits.select_items,
        tuple(splits.report_items.values()),
        layers,
        bos=config.bos,
        batch_size=config.batch_size,
        alignment=config.literal_alignment,
        n_trees=config.n_trees,
        seeds=[
            derive_seed(
                config.seed, "fit", source.task.name, "logistic", "last", i
            )
            for i in range(take)
        ],
    )
    labels = splits.select_items.labels
    entries = [
        m2._Entry(
            m2.Cell("literal_forest", "tokens", "fixed"),
            "reader",
            [
                m2._Choice(layers, d.forest_select, d.forest_report, labels)
                for d in draws
            ],
        ),
        m2._Entry(
            m2.Cell("length_control", "tokens", "all"),
            "reader",
            [
                m2._Choice((), d.length_select, d.length_report, labels)
                for d in draws
            ],
        ),
    ]
    diagnostics = [
        {
            "task": source.task.name,
            "readout": "tokens",
            "draw": i,
            "series": "reader",
            "literal_width": d.width,
        }
        for i, d in enumerate(draws)
    ]
    return entries, diagnostics


def run(config: Config, run_dir: RunDir) -> None:
    """Runs the M2-matched protocol and writes the experimental tables.

    Args:
        config: M2 comparison settings and M4-specific settings.
        run_dir: Destination initialized for this M4 configuration.

    Raises:
        ValueError: If the configuration or reference M2 run is invalid,
            or the recomputed reference rows disagree with the stored run.
    """
    if config.literal_alignment not in token_forest.ALIGNMENTS:
        raise ValueError("literal_alignment must be right or left")
    if config.literal_alignment == "left" and config.model != "qwen3-0.6b-base":
        raise ValueError("left alignment requires qwen3-0.6b-base")
    tasks = [m2._task(name) for name in m2._check(config.m2_config())]
    if not config.reader_series:
        raise ValueError("M4 needs a full M2 reader run")
    if config.pilot_draws is not None and not (
        1 <= config.pilot_draws <= config.n_seeds
    ):
        raise ValueError("pilot_draws must be within the M2 draw count")
    if not 2 <= config.n_folds <= config.n_pairs:
        raise ValueError("n_folds must be between 2 and n_pairs")
    if config.n_trees < 1 or config.mlp_repeats not in (1, 5):
        raise ValueError("M4 needs n_trees >= 1 and mlp_repeats of 1 or 5")
    if not 1 <= config.matched_pairs_per_task <= config.n_pairs:
        raise ValueError("matched_pairs_per_task must be within n_pairs")
    baseline_path = find_m2_run(config, Path(config.m2_root))
    stored = cast(dict[str, Any], RunDir.open(baseline_path).read_json("table"))
    if stored["columns"] != list(m2.COLUMNS):
        raise ValueError(f"unexpected M2 table columns at {baseline_path}")
    sources = {task.source for task in tasks}
    sources.update(target_of(task).source for task in tasks)
    loaded = load_model(config.model, dtype=torch.float32)
    examples = load_sources(sorted(sources))
    layers = m2._read_layers(config, loaded)
    if tuple(sorted(layers)) != layers:
        raise ValueError("M4 expects layers in ascending depth order")
    prepared = _prepare(config, loaded, tasks, examples)

    baseline_rows = [
        row
        for source in prepared
        for entry in source.baseline.entries
        if entry.cell.readout == "last" and entry.cell.layer_rule in BASE_RULES
        for row in m2._metric_rows(config, source.task, source.splits, entry)
    ]
    check_m2_equality(baseline_rows, stored["rows"])

    take = config.pilot_draws or config.n_seeds
    literal_layers = token_forest.literal_layers(layers)
    table = _stored_controls(stored["rows"])
    ties = []
    fit_diagnostics = []
    comparisons = []
    mlp_pilot = []
    for index, source in enumerate(prepared):
        entries = [
            entry._replace(choices=entry.choices[:take])
            for entry in source.baseline.entries
        ]
        experimental, diagnostics, pilot = _ensemble_entries(
            config, source, take
        )
        literal, widths = _literal_entries(
            config, loaded, source, literal_layers, take
        )
        entries.extend(experimental)
        entries.extend(literal)
        fit_diagnostics.extend(diagnostics)
        fit_diagnostics.extend(widths)
        mlp_pilot.extend(pilot)
        table.extend(_write_entries(config, run_dir, source, entries))
        ties.extend(_tie_rows(source, take))
        comparisons.extend(
            _comparison_rows(config, source.task, source.splits, entries)
        )
        prepared[index] = _cross_cache(source)
    del source
    run_dir.write_json("cross_task", _cross_task_rows(config, prepared, take))
    have_truth = any(p.task.name == "truth_polarity" for p in prepared)
    summary_tasks = [p.task.name for p in prepared]
    summary_conditions = {p.task.name: p.splits.conditions for p in prepared}
    summary_skips = {p.task.name: p.skipped for p in prepared}
    n_eval_pairs = {
        p.task.name: {
            "select": len(p.splits.select_items.groups),
            "report": len(p.splits.report_items["original"].groups),
        }
        for p in prepared
    }
    del prepared
    if have_truth:
        run_dir.write_json(
            "truth_holdouts", _lodo_rows(config, loaded, examples, take)
        )
    run_dir.write_json("table", {"columns": list(m2.COLUMNS), "rows": table})
    run_dir.write_json(
        "fit_diagnostics",
        {
            "draws": fit_diagnostics,
            "mlp_pilot": mlp_pilot,
            "recommended_mlp_repeats": 5
            if any(r["recommend_five"] for r in mlp_pilot)
            else 1,
        },
    )
    run_dir.write_json("layer_ties", ties)
    run_dir.write_json("comparisons", comparisons)
    run_dir.write_json(
        "summary",
        {
            "tasks": summary_tasks,
            "conditions": summary_conditions,
            "skipped": summary_skips,
            "n_eval_pairs": n_eval_pairs,
            "m2_reference": str(baseline_path),
            "m2_equality_passed": True,
            "comparison_counts": _comparison_counts(comparisons),
            "truth_note": (
                "Descriptive; gaps below about 0.06 are not detectable."
            ),
            "negation_split_caveat": NEGATION_CAVEAT if have_truth else None,
            "n_draws": take,
            "literal_forest": {
                "alignment": config.literal_alignment,
                "layers": literal_layers,
                "caveat": (
                    "Decoy and paraphrase conditions shift token positions "
                    "by construction."
                ),
            }
            if literal_layers
            else None,
        },
    )


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws the report AUROC matrix of each task.

    Args:
        run_dir: A finished M4 run.
        out_dir: Destination directory of the figures.

    Returns:
        Paths to the generated figures.
    """
    return m2.plot(run_dir, out_dir)


def _take_pairs(rows: _LabeledRows, pairs: Indices) -> _LabeledRows:
    indices = np.sort(pairs.ravel())
    remapped = np.searchsorted(indices, pairs)
    return _LabeledRows(
        rows.data[:, indices],
        rows.layers,
        rows.labels[indices],
        remapped,
        remapped,
    )


def _fit_fold(
    config: Config,
    train: _LabeledRows,
    penalties: tuple[float, ...],
    fit: Indices,
    held: Indices,
    *,
    seed: int,
) -> Scores:
    """Refits and orients the tuned layer readers on the fit pairs only."""
    training_pairs = train.pairs[np.isin(train.pairs[:, 0], fit)]
    fold = _take_pairs(train, training_pairs)
    scores = np.empty((len(held), len(train.layers)), dtype=np.float64)
    for penalty in sorted(set(penalties)):
        keep = np.flatnonzero(np.asarray(penalties) == penalty)
        subset = dataclasses.replace(
            fold,
            data=fold.data[keep],
            layers=tuple(fold.layers[index] for index in keep),
        )
        scores[:, keep] = (
            _fit_lr(config, subset, penalty, seed)
            .score(train.data[keep[:, None], held])
            .T
        )
    return scores


def _stack_features(
    config: Config,
    train: _LabeledRows,
    tuned: _TunedLR,
    test: Scores,
    seed: int,
) -> tuple[Scores, Scores, Scores]:
    def score_fold(fit: Indices, held: Indices) -> Scores:
        return _fit_fold(config, train, tuned.penalties, fit, held, seed=seed)

    return cross_fit_scores(
        train.pairs,
        len(train.labels),
        tuned.select_scores.T,
        np.moveaxis(test, 0, -1),
        score_fold,
        n_folds=config.n_folds,
        seed=seed,
    )


def _matrix_choice(
    layers: tuple[int, ...],
    val: Scores,
    report: Scores,
    select: _LabeledRows,
) -> m2._Choice:
    return m2._Choice(layers, val, report, select.labels)


def _stack_choices(
    config: Config,
    train: _LabeledRows,
    select: _LabeledRows,
    report: Sequence[Rows],
    tuned: _TunedLR,
    seed: int,
    *,
    pilot_diagnostics: bool = False,
) -> tuple[dict[tuple[str, str], m2._Choice], dict[str, Any]]:
    """Reads the S14 layer aggregators from one draw's tuned layer LR.

    `equal_mean` is M2 `read_top_k` over all layers (select
    standardization). `weighted_mean` weights the same layers'
    select-standardized scores by max(select pair accuracy - 0.5, 0) and
    reuses the equal-mean choice when a chosen layer has constant select
    scores. The stackers train on out-of-fold layer scores: in each
    pair-grouped fold every layer reader is refit at its tuned penalty on
    the fit pairs only; the out-of-fold scores set each layer's
    standardization, and select and report scores come from the full-fit
    readers standardized with the same statistics. `stack_lr` and
    `stack_mlp` are tuned on select pair accuracy, `stack_mlp` averages
    `config.mlp_repeats` seeds, and `stack_forest` draws from all layers.

    Args:
        config: The run configuration.
        train: The draw's training rows.
        select: The select rows.
        report: Report activations per condition, shaped (layers, rows,
            width).
        tuned: The per-layer LR tuned on this draw's select pairs.
        seed: Seed for the fold assignment and every fit.
        pilot_diagnostics: Also fit five MLP seeds and record their select
            AUROC and pair accuracy spreads, recommending five seeds when
            the AUROC spread exceeds 0.01. The emitted MLP scores still
            use `config.mlp_repeats` seeds.

    Returns:
        The choices keyed by (scorer, rule) and the fit diagnostics: layer
        penalties, stacking LR penalty, MLP alpha, the weighted-mean
        fallback flag and, with `pilot_diagnostics`, `mlp_pilot`.
    """
    test = np.stack(
        [tuned.reader.score(rows) for rows in report], axis=1
    ).astype(np.float64)
    flat = test.reshape(len(train.layers), -1)
    equal = m2.read_top_k(
        tuned.select_accuracy,
        train.layers,
        tuned.select_scores,
        flat,
        k=len(train.layers),
    )
    choices = {("equal_mean", "all"): _layer_choice(equal, test, select.labels)}
    at = [train.layers.index(layer) for layer in equal.layers]
    # With no varying layer, read_top_k falls back to one constant layer,
    # which has no deviation to standardize; reuse that fallback choice.
    weighted_fallback = bool(
        (np.ptp(tuned.select_scores[at], axis=1) == 0.0).any()
    )
    if weighted_fallback:
        choices["weighted_mean", "all"] = _layer_choice(
            equal, test, select.labels
        )
    else:
        accuracy = tuned.select_accuracy[at]
        standard_val, standard_report = _standardize_layer_scores(
            tuned.select_scores[at], flat[at]
        )
        choices["weighted_mean", "all"] = _matrix_choice(
            equal.layers,
            weighted_mean(standard_val.T, accuracy),
            weighted_mean(standard_report.T, accuracy).reshape(test.shape[1:]),
            select,
        )
    oof, val, out = _stack_features(config, train, tuned, test, seed)
    lr, penalty = tune_stack(
        "lr",
        oof,
        train.labels,
        val,
        select.labels,
        select.groups,
        grid=config.logistic_c,
        seed=seed,
    )
    choices["stack_lr", "all"] = _matrix_choice(
        train.layers, lr(val), np.stack([lr(rows) for rows in out]), select
    )
    mlp, alpha = tune_stack(
        "mlp",
        oof,
        train.labels,
        val,
        select.labels,
        select.groups,
        grid=config.mlp_alpha,
        seed=seed,
    )
    n_seeds = 5 if pilot_diagnostics else config.mlp_repeats
    scorers = [mlp]
    for i in range(1, n_seeds):
        scorers.append(
            fit_mlp_stack(
                train=oof,
                labels=train.labels,
                alpha=alpha,
                seed=derive_seed(seed, "stack-mlp", i),
            )
        )
    scores = np.stack(
        [
            np.stack([score(rows) for rows in out])
            for score in scorers[: config.mlp_repeats]
        ]
    )
    scores_val = np.stack([score(val) for score in scorers])
    diagnostics: dict[str, Any] = {
        "layer_penalties": tuned.penalties,
        "stack_lr_penalty": penalty,
        "stack_mlp_alpha": alpha,
        "weighted_mean_fallback": weighted_fallback,
    }
    if pilot_diagnostics:
        areas = auroc(scores_val, select.labels)
        pairs = group_accuracy(scores_val, select.groups, select.labels)
        area_spread = float(np.ptp(areas))
        pair_spread = float(np.ptp(pairs))
        diagnostics["mlp_pilot"] = {
            "aurocs": areas.tolist(),
            "pair_accuracies": pairs.tolist(),
            "auroc_spread": area_spread,
            "pair_accuracy_spread": pair_spread,
            "recommend_five": area_spread > 0.01,
        }
    choices["stack_mlp", "all"] = _matrix_choice(
        train.layers,
        scores_val[: config.mlp_repeats].mean(axis=0),
        scores.mean(axis=0),
        select,
    )
    forest = fit_grouped_forest(
        oof,
        train.labels,
        train.pairs,
        tuple((i,) for i in range(oof.shape[1])),
        m=oof.shape[1],
        n_trees=config.n_trees,
        seed=seed,
    )
    choices["stack_forest", "all"] = _matrix_choice(
        train.layers,
        forest.score(val),
        np.stack([forest.score(rows) for rows in out]),
        select,
    )
    return choices, diagnostics


def _flatten(rows: Rows) -> Rows:
    n_layers, n_rows, width = rows.shape
    return np.ascontiguousarray(rows.transpose(1, 0, 2)).reshape(
        n_rows, n_layers * width
    )


def _pca_transform(
    train: Rows, components: int, seed: int
) -> Callable[[Rows], Rows]:
    n = min(components, train.shape[1], train.shape[2])
    fits = [
        PCA(
            n_components=n,
            svd_solver="randomized",
            random_state=derive_seed(seed, "pca", i),
        ).fit(rows)
        for i, rows in enumerate(train)
    ]

    def project(rows: Rows) -> Rows:
        return np.stack(
            [
                fit.transform(layer)
                for fit, layer in zip(fits, rows, strict=True)
            ]
        )

    return project


def _forest_choices(
    config: Config,
    train: _LabeledRows,
    select: _LabeledRows,
    report: Sequence[Rows],
    transform: Callable[[Rows], Rows],
    seed: int,
    *,
    fixed_m: int | None = None,
) -> tuple[dict[str, m2._Choice], dict[str, int]]:
    """Reads the grouped random-subspace forest on transformed layers.

    Each layer of `transform(rows)` is one feature group. Without
    `fixed_m`, m is tuned in (1, 6, 14, 28), capped at the number of
    layers, on select pair accuracy and reported as `tuned`, next to
    `default` with m = ceil(sqrt(layers)). The default reuses the grid fit
    when its m is in the grid, and report rows are scored once when both
    rules have the same m. With `fixed_m`, one forest is reported as
    `fixed`.

    Args:
        config: The run configuration.
        train: The draw's training rows.
        select: The select rows.
        report: Report activations per condition, shaped (layers, rows,
            width).
        transform: Maps rows shaped (layers, rows, width) to per-layer
            features.
        seed: Seed for the forest fits.
        fixed_m: The number of layers per tree, when not tuned.

    Returns:
        The choices by rule and the m each rule used.
    """
    fit_rows = transform(train.data)
    n_layers, _, width = fit_rows.shape
    train_features = _flatten(fit_rows)
    val = _flatten(transform(select.data))
    groups = tuple(
        tuple(range(i * width, (i + 1) * width)) for i in range(n_layers)
    )

    def fit(m: int) -> GroupedForest:
        return fit_grouped_forest(
            train_features,
            train.labels,
            train.pairs,
            groups,
            m=m,
            n_trees=config.n_trees,
            seed=seed,
        )

    if fixed_m is None:
        default = math.ceil(math.sqrt(n_layers))
        tuned, grid = tune_grouped_forest(
            train_features,
            train.labels,
            train.pairs,
            val,
            select.labels,
            select.groups,
            groups,
            grid=(1, min(6, n_layers), min(14, n_layers), min(28, n_layers)),
            n_trees=config.n_trees,
            seed=seed,
        )
        fits = {
            "default": grid[default] if default in grid else fit(default),
            "tuned": grid[tuned],
        }
        selected = {"default": default, "tuned": tuned}
    else:
        fits = {"fixed": fit(fixed_m)}
        selected = {"fixed": fixed_m}
    by_m = {selected[rule]: forest for rule, forest in fits.items()}
    val_scores = {m: forest.score(val) for m, forest in by_m.items()}
    scored: dict[int, list[Scores]] = {m: [] for m in by_m}
    for rows in report:
        features = _flatten(transform(rows))
        for m, forest in by_m.items():
            scored[m].append(forest.score(features))
    return {
        rule: _matrix_choice(
            train.layers, val_scores[m], np.stack(scored[m]), select
        )
        for rule, m in selected.items()
    }, selected


def _concat_choice(
    config: Config,
    train: _LabeledRows,
    select: _LabeledRows,
    report: Sequence[Rows],
    seed: int,
) -> tuple[m2._Choice, float]:
    flat_train = dataclasses.replace(
        train, data=_flatten(train.data)[None], layers=(0,)
    )
    flat_select = dataclasses.replace(
        select, data=_flatten(select.data)[None], layers=(0,)
    )
    tuned = _tuned_lr(config, flat_train, [flat_select], seed=seed)[0]
    scores = np.stack(
        [tuned.reader.score(_flatten(rows)[None])[0] for rows in report]
    ).astype(np.float64)
    return _matrix_choice(
        train.layers, tuned.select_scores[0], scores, select
    ), tuned.penalties[0]


def _ensemble_entries(
    config: Config,
    source: _Prepared,
    take: int,
) -> tuple[list[m2._Entry], list[dict[str, Any]], list[dict[str, Any]]]:
    """Reads every S14 and S15 (ii)-(iv) reader on the leading draws.

    For each readout and draw, the `reader` series reuses the baseline's
    tuned LR. With `shuffled` in `config.controls`, a `shuffled` series
    shuffles the train and select labels within pairs and retunes the
    layer LR on them. Both series get the stackers and the vector forest;
    the PCA forests, `concat_lr` and the bridge forest run on the `reader`
    series only, as does the MLP pilot when `config.pilot_draws` is set.

    Args:
        config: The run configuration.
        source: A prepared task with its baseline fits.
        take: Number of leading draws to read.

    Returns:
        The entries per cell and series, one diagnostics row per readout,
        draw and series, and the MLP pilot rows.
    """
    entries: dict[tuple[m2.Cell, str], list[m2._Choice]] = {}
    diagnostics = []
    pilot = []
    series_names = ["reader"]
    if "shuffled" in config.controls:
        series_names.append("shuffled")
    for readout in READOUTS:
        for i in range(take):
            train, select, report = _draw_rows(source.splits, readout, i)
            name = source.task.name
            seed = derive_seed(config.seed, "fit", name, "logistic", readout, i)
            for series in series_names:
                if series == "reader":
                    tuned = source.baseline.logistic[readout][i]
                    fit_train, fit_select = train, select
                else:
                    fit_train = dataclasses.replace(
                        train,
                        labels=shuffle_within_groups(
                            train.labels,
                            train.groups,
                            seed=derive_seed(config.seed, "shuffled", name, i),
                        ),
                    )
                    fit_select = dataclasses.replace(
                        select,
                        labels=shuffle_within_groups(
                            select.labels,
                            select.groups,
                            seed=derive_seed(
                                config.seed, "shuffled-select", name, i
                            ),
                        ),
                    )
                    tuned = _tuned_lr(
                        config, fit_train, [fit_select], seed=seed
                    )[0]
                choices, info = _stack_choices(
                    config,
                    fit_train,
                    fit_select,
                    report,
                    tuned,
                    seed,
                    pilot_diagnostics=config.pilot_draws is not None
                    and series == "reader",
                )
                metadata = {
                    "task": name,
                    "readout": readout,
                    "draw": i,
                    "series": series,
                }
                if "mlp_pilot" in info:
                    pilot.append({**metadata, **info.pop("mlp_pilot")})
                forest, sizes = _forest_choices(
                    config,
                    fit_train,
                    fit_select,
                    report,
                    lambda rows: rows,
                    seed,
                )
                choices.update(
                    {("vector_forest", rule): c for rule, c in forest.items()}
                )
                info["vector_m"] = sizes
                if series == "reader":
                    info.update(
                        _additional_choices(
                            config, train, select, report, seed, choices
                        )
                    )
                for (scorer, rule), choice in choices.items():
                    label = (m2.Cell(scorer, readout, rule), series)
                    entries.setdefault(label, []).append(choice)
                diagnostics.append({**metadata, **info})
    return (
        [
            m2._Entry(cell, series, choices)
            for (cell, series), choices in entries.items()
        ],
        diagnostics,
        pilot,
    )


def _additional_choices(
    config: Config,
    train: _LabeledRows,
    select: _LabeledRows,
    report: Sequence[Rows],
    seed: int,
    choices: dict[tuple[str, str], m2._Choice],
) -> dict[str, Any]:
    """Adds the `reader`-only S15 forests and the concatenation control.

    Updates `choices` in place with the PCA forests (16 and 64 components
    per layer, fitted on the training texts), `concat_lr` (one L2 LR on
    the concatenated layers, M2 grid) and, when any `BRIDGE_LAYERS` are
    present, the bridge forest on those layers with m = min(3, layers).

    Args:
        config: The run configuration.
        train: The draw's training rows.
        select: The select rows.
        report: Report activations per condition, shaped (layers, rows,
            width).
        seed: Seed for the PCA and every fit.
        choices: The draw's choices keyed by (scorer, rule), updated.

    Returns:
        The fit diagnostics: the m of each PCA forest rule, the
        concatenation penalty and the bridge m.
    """
    info: dict[str, Any] = {}
    for components in (16, 64):
        name = f"pca{components}_forest"
        forest, sizes = _forest_choices(
            config,
            train,
            select,
            report,
            _pca_transform(train.data, components, seed),
            seed,
        )
        choices.update(
            {(name, rule): choice for rule, choice in forest.items()}
        )
        info[f"pca{components}_m"] = sizes
    choices["concat_lr", "all"], info["concat_penalty"] = _concat_choice(
        config, train, select, report, seed
    )
    bridge = token_forest.literal_layers(train.layers)
    if bridge:
        at = [train.layers.index(layer) for layer in bridge]
        forest, sizes = _forest_choices(
            config,
            dataclasses.replace(train, layers=bridge),
            select,
            report,
            lambda rows: rows[at],
            seed,
            fixed_m=min(3, len(bridge)),
        )
        choices["bridge_forest", "fixed"] = forest["fixed"]
        info["bridge_m"] = sizes["fixed"]
    return info


BRIDGE_FAMILY = frozenset({"bridge_forest", "literal_forest", "length_control"})
EQUAL_MEAN = "equal_mean"


def _report_clusters(
    task: ReadingTask, items: StimulusSet, texts: Mapping[str, str]
) -> tuple[Indices, Indices]:
    """Gives each report row and pair its bootstrap cluster.

    Outside utility a row's cluster is its pair (loose rows their own)
    and each pair is its own cluster. Utility pairs share scenarios, so a
    union-find over the row and pair clusters merges every cluster whose
    rows have the same text; rows and pairs then take their root.

    Args:
        task: The reading task.
        items: The report items.
        texts: The text of each stimulus uid.

    Returns:
        Cluster ids per row and per pair, on one shared index space.
    """
    rows = m2._clusters(items.groups, len(items.labels))
    pairs = np.arange(len(items.groups))
    if task.name != "utility":
        return rows, pairs
    parent = np.arange(len(items.groups) + len(items.labels))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    by_text: dict[str, int] = {}
    for i, uid in enumerate(items.uids):
        text = texts[uid]
        current = root(int(rows[i]))
        other = root(by_text.setdefault(text, current))
        parent[max(current, other)] = min(current, other)
    return (
        np.array([root(int(index)) for index in rows]),
        np.array([root(int(index)) for index in pairs]),
    )


def _draw_summary(config: Config, prefix: str, diff: Scores) -> dict[str, Any]:
    return {
        f"{prefix}_draw_diff": diff.tolist(),
        f"{prefix}_draw_mean": float(diff.mean()),
        f"{prefix}_draw_ci": bootstrap_ci(
            diff, n_boot=config.n_boot, seed=config.seed
        ),
        f"{prefix}_draw_wins": int(np.count_nonzero(diff > 0)),
    }


def _comparison_row(
    config: Config,
    task: ReadingTask,
    condition: str,
    index: int,
    items: StimulusSet,
    reader: m2._Entry,
    baseline: m2._Entry,
    family: str,
    size: int,
) -> dict[str, Any]:
    area, pair = _draw_delta(reader.choices, baseline.choices, index, items)
    return {
        "task": task.name,
        "condition": condition,
        "readout": reader.cell.readout,
        "reader": reader.cell.name,
        "baseline": baseline.cell.name,
        "family": family,
        "family_size": size,
        "descriptive": task.name == "truth" or size == 0,
        **_draw_summary(config, "auroc", area),
        **_draw_summary(config, "pair", pair),
    }


def _cluster_intervals(
    config: Config,
    labels: Indices,
    clusters: tuple[Indices, Indices],
    scores: tuple[Scores, Scores],
    wins: tuple[Scores, Scores],
) -> dict[str, Any]:
    row_clusters, pair_clusters = clusters
    return {
        "auroc_cluster": paired_auroc_diff(
            *scores,
            labels,
            row_clusters,
            n_boot=config.n_boot,
            seed=config.seed,
        )._asdict(),
        "pair_cluster": paired_accuracy_diff(
            *wins,
            pair_clusters,
            n_boot=config.n_boot,
            seed=config.seed,
        )._asdict(),
    }


def _comparison_rows(
    config: Config,
    task: ReadingTask,
    splits: m2._Splits,
    entries: Sequence[m2._Entry],
) -> list[dict[str, Any]]:
    """Pairs draw metrics and tests draw-ensemble scores within each family.

    The separate bridge family keeps `bridge_forest` at last, max and mean
    against last-readout baselines. It compares pooling over tokens with
    concatenation in the literal forest.

    Args:
        config: Resampling settings and checkpoint.
        task: Evaluated task.
        splits: Labels, groups and raw scenario texts in report order.
        entries: Baselines and new readers with matched draw order.

    Returns:
        Descriptive matched-draw effects and secondary family-adjusted
        comparisons. Utility also includes scenario-cluster intervals.
    """
    readers = {
        entry.cell: entry for entry in entries if entry.series == "reader"
    }
    bases = {
        cell
        for cell in readers
        if cell.scorer == EQUAL_MEAN
        or (cell.scorer in BASE_SCORERS and cell.layer_rule in BASE_RULES)
    }
    members = sorted(
        cell
        for cell in readers
        if cell.scorer not in (*BASE_SCORERS, EQUAL_MEAN)
    )
    compared = bases.union(members)
    ensembles = {cell: m2._ensemble(readers[cell].choices) for cell in compared}
    output = []
    for j, (condition, items) in enumerate(splits.report_items.items()):
        clusters, pair_clusters = _report_clusters(task, items, splits.texts)
        wins = {
            cell: m2._group_wins(scores[j], items)
            for cell, scores in ensembles.items()
        }

        for base in sorted(bases):
            families = {
                base.readout: [
                    cell
                    for cell in members
                    if cell.readout == base.readout
                    and cell.scorer not in BRIDGE_FAMILY
                ]
            }
            if base.readout == "last":
                families["bridge"] = [
                    cell for cell in members if cell.scorer in BRIDGE_FAMILY
                ]
            for tag, family in families.items():
                name = (
                    f"{config.model}/{task.name}/{condition}/{tag}/{base.name}"
                )
                block = []
                for cell in family:
                    row = _comparison_row(
                        config,
                        task,
                        condition,
                        j,
                        items,
                        readers[cell],
                        readers[base],
                        name,
                        len(family),
                    )
                    score_a, score_b = ensembles[cell][j], ensembles[base][j]
                    delong = delong_test(score_a, score_b, items.labels)
                    mcnemar = mcnemar_test(wins[cell] == 1.0, wins[base] == 1.0)
                    row.update(
                        delong_p=delong.p_value,
                        mcnemar_p=mcnemar.p_value,
                        a_only=mcnemar.a_only,
                        b_only=mcnemar.b_only,
                    )
                    if task.name == "utility":
                        row.update(
                            _cluster_intervals(
                                config,
                                items.labels,
                                (clusters, pair_clusters),
                                (score_a, score_b),
                                (wins[cell], wins[base]),
                            )
                        )
                    block.append(row)
                if block:
                    for field in ("delong", "mcnemar"):
                        adjusted = holm_correction(
                            np.array([row[f"{field}_p"] for row in block])
                        ).adjusted
                        for row, value in zip(block, adjusted, strict=True):
                            row[f"{field}_holm"] = float(value)
                    output.extend(block)
            equal = m2.Cell(EQUAL_MEAN, base.readout, "all")
            if base.scorer != EQUAL_MEAN and equal in readers:
                row = _comparison_row(
                    config,
                    task,
                    condition,
                    j,
                    items,
                    readers[equal],
                    readers[base],
                    "descriptive",
                    0,
                )
                if task.name == "utility":
                    row.update(
                        _cluster_intervals(
                            config,
                            items.labels,
                            (clusters, pair_clusters),
                            (ensembles[equal][j], ensembles[base][j]),
                            (wins[equal], wins[base]),
                        )
                    )
                output.append(row)
    return output


def _comparison_counts(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Counts family comparisons won by the new readers.

    The length control is a family member but not a new reader, so its rows
    are counted apart under `length_control`.

    Args:
        rows: Comparison rows; descriptive rows have family size 0.

    Returns:
        Counts with and without truth for the new readers, and the same
        counts for the length control.
    """

    def subsets(kept: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            subset: {
                "comparisons": len(chosen),
                "higher_auroc": sum(r["auroc_draw_mean"] > 0 for r in chosen),
                "higher_pair_accuracy": sum(
                    r["pair_draw_mean"] > 0 for r in chosen
                ),
            }
            for subset, chosen in (
                ("with_truth", kept),
                ("without_truth", [r for r in kept if r["task"] != "truth"]),
            )
        }

    split: dict[bool, list[dict[str, Any]]] = {True: [], False: []}
    for r in rows:
        if r["family_size"] > 0:
            split[r["reader"].startswith("length_control/")].append(r)
    return {**subsets(split[False]), "length_control": subsets(split[True])}


class _ScoredDraw(NamedTuple):
    choice: m2._Choice
    penalty: float


def _score_best(
    tuned: _TunedLR,
    report: Sequence[Rows],
    select_labels: Indices,
    *,
    target_select: _LabeledRows | None = None,
) -> _ScoredDraw:
    layer = best_layer(tuned.select_accuracy, tuned.reader.layers)
    at = tuned.reader.layers.index(layer)
    one = slice(at, at + 1)
    reader = dataclasses.replace(
        tuned.reader,
        layers=(layer,),
        directions=tuned.reader.directions[one],
        means=tuned.reader.means[one],
        signs=tuned.reader.signs[one],
    )
    val = tuned.select_scores[at]
    if target_select is not None:
        val = reader.score(target_select.data[one])[0].astype(np.float64)
        select_labels = target_select.labels
    out = np.stack([reader.score(rows[one])[0] for rows in report])
    return _ScoredDraw(
        m2._Choice((layer,), val, out.astype(np.float64), select_labels),
        tuned.penalties[at],
    )


def _pool_rows(sets: Sequence[_LabeledRows], pairs_each: int) -> _LabeledRows:
    parts = [
        rows
        if len(rows.pairs) == pairs_each
        else _take_pairs(rows, rows.pairs[:pairs_each])
        for rows in sets
    ]
    offsets = np.cumsum([0, *(len(rows.labels) for rows in parts[:-1])])
    pairs = np.concatenate(
        [
            rows.pairs + offset
            for rows, offset in zip(parts, offsets, strict=True)
        ]
    )
    return _LabeledRows(
        np.concatenate([rows.data for rows in parts], axis=1),
        parts[0].layers,
        np.concatenate([rows.labels for rows in parts]),
        pairs,
        pairs,
    )


def _cross_metrics(
    config: Config,
    task: ReadingTask,
    splits: m2._Splits,
    readout: m2.Readout,
    arm: str,
    draws: Sequence[_ScoredDraw],
    metadata: Mapping[str, Any],
    reference: Sequence[m2._Choice],
) -> list[dict[str, Any]]:
    choices, penalties = zip(*draws, strict=True)
    entry = m2._Entry(
        m2.Cell("logistic", readout, "best"), "reader", list(choices)
    )
    rows = m2._metric_rows(config, task, splits, entry)
    results = []
    for i, row in enumerate(rows):
        values = dict(zip(m2.COLUMNS, row, strict=True))
        values.update(metadata, arm=arm, penalties=list(penalties))
        items = splits.report_items[values["condition"]]
        area, pair = _draw_delta(choices, reference, i, items)
        values.update(_draw_summary(config, "auroc", area))
        values.update(_draw_summary(config, "pair", pair))
        results.append(values)
    return results


def _cross_task_rows(
    config: Config,
    prepared: Sequence[_Prepared],
    take: int,
) -> list[dict[str, Any]]:
    """Reads the S16 transfer matrix and pooled arms.

    At each cross readout and draw, every task's tuned LR (rule `best`,
    penalty and layer chosen on its own select) scores every task's
    report items; AUROC is never re-oriented on the target task. With two
    or more tasks, `pooled_all` (`n_pairs` per task) and `pooled_matched`
    (the first `matched_pairs_per_task` pairs of each task's draw) fit one
    LR on the pooled rows, read as `shared` (penalty and layer by mean
    select pair accuracy over tasks) and `per_task` (chosen on the
    evaluated task's select). Every row is compared with the evaluated
    task's own classifier.

    Args:
        config: The run configuration.
        prepared: The cross-task caches of every task.
        take: Number of leading draws to read.

    Returns:
        The metric rows of the transfer and pooled arms.
    """
    budgets = {
        "pooled_all": config.n_pairs,
        "pooled_matched": config.matched_pairs_per_task,
    }
    output = []
    for readout in CROSS_READOUTS:
        transfers: dict[tuple[int, int], list[_ScoredDraw]] = {}
        pooled: dict[tuple[str, str, int], list[_ScoredDraw]] = {}
        for i in range(take):
            inputs = [_draw_rows(s.splits, readout, i) for s in prepared]
            training, validation, reports = zip(*inputs, strict=True)
            for origin, source in enumerate(prepared):
                tuned = source.baseline.logistic[readout][i]
                for target, report in enumerate(reports):
                    score = _score_best(
                        tuned, report, validation[origin].labels
                    )
                    transfers.setdefault((origin, target), []).append(score)
            if len(prepared) < 2:
                continue
            for arm, budget in budgets.items():
                train = _pool_rows(training, budget)
                fits = _tuned_lr(
                    config,
                    train,
                    validation,
                    seed=derive_seed(config.seed, "pool", arm, readout, i),
                )
                per_task, shared = fits[:-1], fits[-1]
                for target, report in enumerate(reports):
                    select = validation[target]
                    for method, tuned in (
                        ("shared", shared),
                        ("per_task", per_task[target]),
                    ):
                        score = _score_best(
                            tuned, report, select.labels, target_select=select
                        )
                        pooled.setdefault((arm, method, target), []).append(
                            score
                        )
        for (origin, target), draws in transfers.items():
            evaluated = prepared[target]
            reference = [d.choice for d in transfers[target, target]]
            output.extend(
                _cross_metrics(
                    config,
                    evaluated.task,
                    evaluated.splits,
                    readout,
                    "per_task" if origin == target else "transfer",
                    draws,
                    {
                        "train_task": prepared[origin].task.name,
                        "train_pairs": config.n_pairs,
                        "tuning": "source",
                    },
                    reference,
                )
            )
        for (arm, method, target), draws in pooled.items():
            evaluated = prepared[target]
            output.extend(
                _cross_metrics(
                    config,
                    evaluated.task,
                    evaluated.splits,
                    readout,
                    arm,
                    draws,
                    {
                        "train_pairs": len(prepared) * budgets[arm],
                        "tuning": method,
                        "train_tasks": [s.task.name for s in prepared],
                    },
                    [d.choice for d in transfers[target, target]],
                )
            )
    return output


TRUTH_SUBSETS = m2.POLARITY_TASKS["truth_polarity"]
TRUTH_HOLDOUTS = (
    *(("subset", subset, (subset,)) for subset in TRUTH_SUBSETS),
    ("topic", "cities", ("cities", "neg_cities")),
    ("topic", "spanish_translation", ("sp_en_trans", "neg_sp_en_trans")),
    ("polarity", "affirmative", ("cities", "sp_en_trans")),
    ("polarity", "negated", ("neg_cities", "neg_sp_en_trans")),
)
NEGATION_CAVEAT = (
    "A statement and its negation have independent subset-specific IDs "
    "and can land in different splits. Held-out subsets, topics and "
    "polarities do not guarantee held-out facts."
)


def _scope_examples(
    examples: Sequence[Example],
    subsets: Sequence[str],
) -> list[Example]:
    return [
        example for example in examples if example.meta.get("subset") in subsets
    ]


def _truth_training(
    config: Config,
    task: ReadingTask,
    paper: Variant,
    pool: TaskSplits,
    scope: Sequence[str],
    take: int,
) -> tuple[StimulusSet, list[StimulusSet]]:
    select = task_set(
        task,
        paper,
        _scope_examples(pool.select, scope),
        n_pairs=None,
        seed=eval_seed(task, "select", config.seed),
    )
    rows = _scope_examples(pool.train, scope)
    draws = [
        task_set(
            task,
            paper,
            rows,
            n_pairs=config.n_pairs,
            seed=derive_seed(config.seed, "train", i),
        )
        for i in range(take)
    ]
    return select, draws


def _truth_batch(
    config: Config,
    loaded: LoadedModel,
    items: StimulusSet,
) -> dict[m2.Readout, _LabeledRows]:
    """Reads the cross-task readouts of a prompt bank with M2's reader.

    Args:
        config: Layers, batching and BOS policy.
        loaded: The wrapped model.
        items: Prompts and their response spans.

    Returns:
        The `last` and `max` rows of every prompt, in input order.
    """
    positions = tuple(m2.POSITION_OF[r] for r in CROSS_READOUTS)
    read = m2.read_positions(loaded, items, positions, config)
    return {
        readout: _labeled(read[position].data, read[position].layers, items)
        for readout, position in zip(CROSS_READOUTS, positions, strict=True)
    }


_TruthRead = Callable[[str, StimulusSet], dict[m2.Readout, _LabeledRows]]


def _truth_reader(
    config: Config,
    loaded: LoadedModel,
    plan: Mapping[str, Sequence[StimulusSet]],
) -> _TruthRead:
    """Caches unique texts per condition while preserving split pairings.

    Args:
        config: Extraction settings.
        loaded: The wrapped model.
        plan: All stimulus sets needed by the holdouts, by condition.

    Returns:
        A reader that gathers activations in each set's original order.

    Raises:
        ValueError: If a duplicate prompt has conflicting response spans.
    """
    lookup: dict[str, dict[str, int]] = {}
    banks: dict[str, dict[m2.Readout, _LabeledRows]] = {}
    for condition, sets in plan.items():
        index: dict[str, int] = {}
        representatives: list[tuple[StimulusSet, int]] = []
        for items in sets:
            for i, text in enumerate(items.texts):
                if text not in index:
                    index[text] = len(representatives)
                    representatives.append((items, i))
                else:
                    previous, j = representatives[index[text]]
                    if previous.response_spans[j] != items.response_spans[i]:
                        raise ValueError(
                            "identical truth prompts have different spans"
                        )
        empty = np.empty((0, 2), dtype=np.int64)
        bank = StimulusSet(
            texts=tuple(index),
            response_spans=tuple(
                s.response_spans[i] for s, i in representatives
            ),
            concept_spans=(None,) * len(index),
            labels=np.array([s.labels[i] for s, i in representatives]),
            pairs=empty,
            groups=empty,
            uids=tuple(s.uids[i] for s, i in representatives),
        )
        lookup[condition] = index
        banks[condition] = _truth_batch(config, loaded, bank)

    def read(
        condition: str, items: StimulusSet
    ) -> dict[m2.Readout, _LabeledRows]:
        at = [lookup[condition][text] for text in items.texts]
        return {
            pos: _labeled(rows.data[:, at], rows.layers, items)
            for pos, rows in banks[condition].items()
        }

    return read


def _truth_fits(
    config: Config,
    task: ReadingTask,
    select: StimulusSet,
    draws: Sequence[StimulusSet],
    read: _TruthRead,
) -> tuple[dict[m2.Readout, list[_TunedLR]], Indices]:
    val = read("original", select)
    fitted: dict[m2.Readout, list[_TunedLR]] = {r: [] for r in CROSS_READOUTS}
    for i, items in enumerate(draws):
        train = read("original", items)
        for readout in CROSS_READOUTS:
            seed = derive_seed(
                config.seed, "fit", task.name, "logistic", readout, i
            )
            fitted[readout].append(
                _tuned_lr(
                    config,
                    train[readout],
                    [val[readout]],
                    seed=seed,
                )[0]
            )
    return fitted, select.labels


def _held_reports(
    config: Config,
    task: ReadingTask,
    paper: Variant,
    shifts: Sequence[Variant],
    pool: Sequence[Example],
) -> dict[str, StimulusSet]:
    biased = dataclasses.replace(
        paper,
        name="misleading",
        template=misleading_template(task.template, task.noun),
    )
    cases = [("original", paper, pool)]
    cases.extend(
        (decoy, biased, with_decoys(pool, decoy, seed=config.seed))
        for decoy in config.decoys
    )
    cases.extend(
        (variant.name, variant, pool) for variant in shifts if variant.evaluable
    )
    return {
        name: task_set(
            task,
            variant,
            rows,
            n_pairs=None,
            seed=eval_seed(task, "report", config.seed),
        )
        for name, variant, rows in cases
    }


def _lodo_rows(
    config: Config,
    loaded: LoadedModel,
    examples: Mapping[str, Sequence[Example]],
    take: int,
) -> list[dict[str, Any]]:
    """Reads the S16 (c) holdouts in truth with negations.

    For each holdout in `TRUTH_HOLDOUTS` (subset, topic, polarity), the
    training, select and report examples are filtered by subset before
    pairing. Three arms with the
    same pair budget are tuned on their own select pairs: `held_out`
    (the other subsets), `all_subsets` and `within_held` (the held
    subsets alone). A training scope shared by holdouts is fitted once,
    and every planned set is read in one `_truth_reader` pass. Per draw,
    the report scores of all conditions are stacked in condition order,
    and every arm is compared with `within_held`.

    Args:
        config: The run configuration.
        loaded: The wrapped model.
        examples: Examples by task name.
        take: Number of draws.

    Returns:
        The metric rows with the holdout metadata.
    """
    task = m2._task("truth_polarity")
    pool = task_splits(
        task, m2._task_examples(task.name, examples), seed=config.seed
    )
    variants, _ = template_variants(
        task, chat_format_for(loaded), kinds=("paper", *config.shifts)
    )
    paper, *shifts = variants
    holdouts = []
    training = {}
    plan: dict[str, list[StimulusSet]] = {"original": []}
    for kind, name, held in TRUTH_HOLDOUTS:
        report = _held_reports(
            config, task, paper, shifts, _scope_examples(pool.report, held)
        )
        for condition, items in report.items():
            plan.setdefault(condition, []).append(items)
        other = tuple(subset for subset in TRUTH_SUBSETS if subset not in held)
        base_arms = (
            ("held_out", other),
            ("all_subsets", TRUTH_SUBSETS),
            ("within_held", held),
        )
        holdouts.append((kind, name, held, report, base_arms))
        for _, scope in base_arms:
            if scope not in training:
                select, draws = _truth_training(
                    config, task, paper, pool, scope, take
                )
                training[scope] = (select, draws)
                plan["original"].extend((select, *draws))
    read = _truth_reader(config, loaded, plan)
    fits = {
        scope: _truth_fits(config, task, select, draws, read)
        for scope, (select, draws) in training.items()
    }
    output = []
    for kind, name, held, report, base_arms in holdouts:
        original = report["original"]
        splits = m2._Splits([], original, report, {}, {}, {}, {}, None)
        by_condition: dict[tuple[str, m2.Readout], list[list[_ScoredDraw]]] = {}
        for condition, items in report.items():
            activations = read(condition, items)
            for readout in CROSS_READOUTS:
                for arm, scope in base_arms:
                    models, labels = fits[scope]
                    by_condition.setdefault((arm, readout), []).append(
                        [
                            _score_best(
                                fit, [activations[readout].data], labels
                            )
                            for fit in models[readout]
                        ]
                    )
        results = {
            key: [
                _ScoredDraw(
                    dataclasses.replace(
                        draws[0].choice,
                        report=np.concatenate([d.choice.report for d in draws]),
                    ),
                    draws[0].penalty,
                )
                for draws in zip(*conditions, strict=True)
            ]
            for key, conditions in by_condition.items()
        }
        for (arm, readout), scores in results.items():
            scope = dict(base_arms)[arm]
            output.extend(
                _cross_metrics(
                    config,
                    task,
                    splits,
                    readout,
                    arm,
                    scores,
                    {
                        "holdout_kind": kind,
                        "holdout": name,
                        "held_subsets": held,
                        "train_subsets": scope,
                        "train_pairs": config.n_pairs,
                        "report_pairs": len(original.groups),
                    },
                    [d.choice for d in results["within_held", readout]],
                )
            )
    return output


def _cross_cache(source: _Prepared) -> _Prepared:
    """Keeps only what the cross-task reads need from a prepared task.

    Drops the activations at positions outside `CROSS_READOUTS` and every
    baseline entry and tie flag, keeping the tuned LR fits at those
    readouts, so all tasks fit in memory together.

    Args:
        source: A prepared task.

    Returns:
        A trimmed copy of `source`.
    """
    positions = tuple(m2.POSITION_OF[r] for r in CROSS_READOUTS)
    splits = source.splits
    return dataclasses.replace(
        source,
        splits=dataclasses.replace(
            splits,
            train={position: splits.train[position] for position in positions},
            select={
                position: splits.select[position] for position in positions
            },
            report={
                position: splits.report[position] for position in positions
            },
        ),
        baseline=_Baseline(
            [],
            {r: source.baseline.logistic[r] for r in CROSS_READOUTS},
            {},
        ),
    )
