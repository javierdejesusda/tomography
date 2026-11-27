"""M4: layer ensembles, cross-task transfer and pooled readers."""

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
import torch

from tomography.datasets.records import Example
from tomography.datasets.stimulus import StimulusSet
from tomography.experiments import m2_readers as m2
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.variants import ReadingTask
from tomography.model import LoadedModel
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
        matched_pairs_per_task: Budget per task in the matched pooled arm.
    """

    experiment: str = "m4"
    token_layers: tuple[int, ...] | None = BRIDGE_LAYERS
    m2_root: str = "results/m2"
    pilot_draws: int | None = None
    n_folds: int = 8
    n_trees: int = 500
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
    ties: list[dict[str, Any]]


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
    entries = []
    logistic: dict[m2.Readout, list[_TunedLR]] = {}
    ties = []
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
            take = config.pilot_draws or config.n_seeds
            first = choices["best"][:take]
            deep_draws = choices["best_deep"][:take]
            different = [
                a.layers != b.layers
                for a, b in zip(first, deep_draws, strict=True)
            ]
            effects = {}
            for j, (condition, items) in enumerate(splits.report_items.items()):
                area, pair = _draw_delta(deep_draws, first, j, items)
                effects[condition] = {
                    "auroc_draw_diff": area.tolist(),
                    "pair_draw_diff": pair.tolist(),
                }
            ties.append(
                {
                    "task": task.name,
                    "scorer": scorer,
                    "readout": readout,
                    "tie_share": float(np.mean(tied[:take])),
                    "different_share": float(np.mean(different)),
                    "draw_ties": tied[:take],
                    "draw_layer_differs": different,
                    "effects": effects,
                }
            )
    return _Baseline(entries, logistic, ties)


@dataclass(frozen=True)
class _Prepared:
    task: ReadingTask
    splits: m2._Splits
    baseline: _Baseline
    skipped: dict[str, str]


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
            or (row[3] in READOUTS
            and (row[2] == "random" or (row[2] == "mlp" and row[4] == "joint")))
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
    baseline_path = find_m2_run(config, Path(config.m2_root))
    stored = cast(dict[str, Any], RunDir.open(baseline_path).read_json("table"))
    if stored["columns"] != list(m2.COLUMNS):
        raise ValueError(f"unexpected M2 table columns at {baseline_path}")
    sources = {task.source for task in tasks}
    sources.update(m2.target_of(task).source for task in tasks)
    loaded = m2.load_model(config.model, dtype=torch.float32)
    examples = m2.load_sources(sorted(sources))
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
    for source in prepared:
        entries = [
            entry._replace(choices=entry.choices[:take])
            for entry in source.baseline.entries
        ]
        table.extend(_write_entries(config, run_dir, source, entries))
        ties.extend(source.baseline.ties)
    run_dir.write_json("table", {"columns": list(m2.COLUMNS), "rows": table})
    run_dir.write_json("layer_ties", ties)
    run_dir.write_json(
        "summary",
        {
            "tasks": [source.task.name for source in prepared],
            "conditions": {s.task.name: s.splits.conditions for s in prepared},
            "skipped": {s.task.name: s.skipped for s in prepared},
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
