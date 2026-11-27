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
from tomography.experiments.sweep import shuffle_within_groups
from tomography.experiments.variants import ReadingTask
from tomography.experiments.variants import load_sources
from tomography.experiments.variants import target_of
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.reading.ensembles import GroupedForest
from tomography.reading.ensembles import cross_fit_scores
from tomography.reading.ensembles import fit_grouped_forest
from tomography.reading.ensembles import fit_mlp_stack
from tomography.reading.ensembles import tune_grouped_forest
from tomography.reading.ensembles import tune_stack
from tomography.reading.ensembles import weighted_mean
from tomography.reading.metrics import auroc
from tomography.reading.metrics import group_accuracy
from tomography.reading.reader import RepReader
from tomography.reading.stats import LayerChoice
from tomography.reading.stats import best_validation_layer

Rows = NDArray[np.float32]
Scores = NDArray[np.float64]
Indices = NDArray[np.int64]
READOUTS: tuple[m2.Readout, ...] = ("last", "max", "mean")
CROSS_READOUTS: tuple[m2.Readout, ...] = ("last", "max")
BASE_SCORERS = ("logistic", "mean_diff")
BASE_RULES = ("best", "top_k")
BRIDGE_LAYERS = (4, 8, 12, 16, 20, 24, 28)


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
    """

    experiment: str = "m4"
    token_layers: tuple[int, ...] | None = BRIDGE_LAYERS
    m2_root: str = "results/m2"
    pilot_draws: int | None = None
    n_folds: int = 8
    n_trees: int = 500
    mlp_repeats: int = 1
    matched_pairs_per_task: int = 16

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
            equal = (
                value == reference
                if column in ("layers", "n_seeds")
                else math.isclose(value, reference, rel_tol=0.0, abs_tol=1e-6)
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


def run(config: Config, run_dir: RunDir) -> None:
    """Runs the M2-matched protocol and writes the experimental tables.

    Args:
        config: M2 comparison settings and M4-specific settings.
        run_dir: Destination initialized for this M4 configuration.

    Raises:
        ValueError: If the configuration or reference M2 run is invalid,
            or the recomputed reference rows disagree with the stored run.
    """
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
    table = _stored_controls(stored["rows"])
    ties = []
    fit_diagnostics = []
    mlp_pilot = []
    for source in prepared:
        entries = [
            entry._replace(choices=entry.choices[:take])
            for entry in source.baseline.entries
        ]
        experimental, diagnostics, pilot = _ensemble_entries(
            config, source, take
        )
        entries.extend(experimental)
        fit_diagnostics.extend(diagnostics)
        mlp_pilot.extend(pilot)
        table.extend(_write_entries(config, run_dir, source, entries))
        ties.extend(_tie_rows(source, take))

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
    run_dir.write_json("table", {"columns": list(m2.COLUMNS), "rows": table})
    run_dir.write_json(
        "fit_diagnostics",
        {
            "draws": fit_diagnostics,
            "mlp_pilot": mlp_pilot,
            "recommended_mlp_repeats": 5
            if any(r["recommend_five"] for r in mlp_pilot)
            else config.mlp_repeats,
        },
    )
    run_dir.write_json("layer_ties", ties)
    run_dir.write_json(
        "summary",
        {
            "tasks": summary_tasks,
            "conditions": summary_conditions,
            "skipped": summary_skips,
            "n_eval_pairs": n_eval_pairs,
            "m2_reference": str(baseline_path),
            "m2_equality_passed": True,
            "n_draws": take,
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


def _stack_features(
    config: Config,
    train: _LabeledRows,
    tuned: _TunedLR,
    test: Scores,
    seed: int,
) -> tuple[Scores, Scores, Scores]:
    def score_fold(fit: Indices, held: Indices) -> Scores:
        training_pairs = train.pairs[np.isin(train.pairs[:, 0], fit)]
        fold = _take_pairs(train, training_pairs)
        scores = np.empty((len(held), len(train.layers)), dtype=np.float64)
        for penalty in sorted(set(tuned.penalties)):
            keep = np.flatnonzero(np.asarray(tuned.penalties) == penalty)
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
    report_items: Sequence[StimulusSet],
    seed: int,
) -> tuple[dict[tuple[str, str], m2._Choice], dict[str, Any]]:
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
    oof, val, out = _stack_features(config, train, tuned, test, seed)
    choices["weighted_mean", "all"] = _matrix_choice(
        train.layers,
        weighted_mean(val, tuned.select_accuracy),
        weighted_mean(out, tuned.select_accuracy),
        select,
    )
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
    n_seeds = 5 if config.pilot_draws is not None else config.mlp_repeats
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
        [np.stack([score(rows) for rows in out]) for score in scorers]
    )
    scores_val = np.stack([score(val) for score in scorers])
    diagnostics: dict[str, Any] = {
        "layer_penalties": tuned.penalties,
        "stack_lr_penalty": penalty,
        "stack_mlp_alpha": alpha,
    }
    average = config.mlp_repeats == 5
    if config.pilot_draws is not None:
        areas = np.stack(
            [
                auroc(scores[:, i], items.labels)
                for i, items in enumerate(report_items)
            ]
        )
        spread = np.ptp(areas, axis=1)
        average = bool((spread > 0.01).any()) or average
        diagnostics["mlp_pilot"] = {
            "aurocs": areas.tolist(),
            "spread": spread.tolist(),
            "recommend_five": bool((spread > 0.01).any()),
        }
    choices["stack_mlp", "all"] = _matrix_choice(
        train.layers,
        scores_val.mean(axis=0) if average else scores_val[0],
        scores.mean(axis=0) if average else scores[0],
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
    scored: dict[str, list[Scores]] = {rule: [] for rule in fits}
    for rows in report:
        features = _flatten(transform(rows))
        unique = {
            m: fits[rule].score(features)
            for rule, m in selected.items()
            if rule == "default" or m != selected.get("default")
        }
        for rule, m in selected.items():
            scored[rule].append(unique[m])
    return {
        rule: _matrix_choice(
            train.layers, forest.score(val), np.stack(scored[rule]), select
        )
        for rule, forest in fits.items()
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
    entries: dict[tuple[m2.Cell, str], list[m2._Choice]] = {}
    diagnostics = []
    pilot = []
    items = list(source.splits.report_items.values())
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
                    items,
                    seed,
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
    bridge = tuple(layer for layer in BRIDGE_LAYERS if layer in train.layers)
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
