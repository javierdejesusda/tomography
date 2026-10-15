"""P3: how many pairs a reading direction needs, and how much one draw misleads.

For each task and each training-set size, many resampled training sets are
read at once, each compared with a reader fitted on all the training data. The
spread of the report accuracy, the share of failed draws, the agreement of the
pair differences as a warning sign and the size where returns diminish come
from the same sweeps. Tasks that name the same target are sources of one
concept, scored on one set of items.
"""

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
import itertools
from pathlib import Path
from typing import Any
from typing import cast

from matplotlib.ticker import NullLocator
from matplotlib.ticker import ScalarFormatter
import numpy as np
from numpy.typing import NDArray

from tomography.datasets.records import Example
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.matrices import plot_matrix_file
from tomography.experiments.matrices import write_matrix
from tomography.experiments.sweep import SERIES
from tomography.experiments.sweep import Prepared
from tomography.experiments.sweep import ReadingConfig
from tomography.experiments.sweep import chat_format_for
from tomography.experiments.sweep import sweep_from_activations
from tomography.experiments.sweep import write_curves
from tomography.experiments.variants import READING_TASKS
from tomography.experiments.variants import ReadingTask
from tomography.experiments.variants import TaskSplits
from tomography.experiments.variants import Variant
from tomography.experiments.variants import eval_pairs
from tomography.experiments.variants import eval_seed
from tomography.experiments.variants import expand_tasks
from tomography.experiments.variants import load_sources
from tomography.experiments.variants import pair_capacity
from tomography.experiments.variants import prepare
from tomography.experiments.variants import target_of
from tomography.experiments.variants import task_set
from tomography.experiments.variants import task_splits
from tomography.experiments.variants import template_variants
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.reading.directions import Method
from tomography.reading.metrics import auroc
from tomography.reading.metrics import best_layer
from tomography.reading.metrics import cosine_matrix
from tomography.templates import ChatFormat
from tomography.viz.reading import curves_from_arrays
from tomography.viz.reading import plot_curves
from tomography.viz.style import new_figure
from tomography.viz.style import save_figure
from tomography.viz.style import style_context

COLUMNS: tuple[str, ...] = (
    "task",
    "target",
    "size",
    "layer",
    "accuracy",
    "lo",
    "hi",
    "sd",
    "q05",
    "q95",
    "failure",
    "shuffled",
    "cosine",
    "cosine_sd",
    "agreement",
    "rho",
)
SPLITS: tuple[str, str] = ("select", "report")
SIZE_LABEL = "Training pairs"


@dataclass(frozen=True, kw_only=True)
class Config(ReadingConfig):
    """P3 settings: readers trained on growing sets of pairs.

    The shared defaults apply except for the experiment id, the model, the
    direction finder (`logistic`, as in R1), the number of resamples and the
    number of evaluation pairs. `n_pairs` is not read: the training size
    comes from `sizes`.

    Attributes:
        n_seeds: Resampled training sets per size, at least 2.
        tasks: Keys of `READING_TASKS` to read; binary tasks only.
        sizes: Training pairs per set, increasing, each at least 2. A size
            above a task's training capacity is skipped for that task.
        reference_pairs: The full-data reader takes all the training pairs
            up to this many.
        failure_accuracy: A resample whose report accuracy is below this has
            failed.
        saturation_delta: How far below the largest size a size's mean
            accuracy may fall and still count as saturated.
        n_boot: Number of bootstrap resamples of the bands and intervals.
    """

    experiment: str = "p3"
    model: str = "gpt2"
    method: Method = "logistic"
    n_seeds: int = 20
    n_eval_pairs: int | None = 256
    tasks: tuple[str, ...] = (
        "sentiment",
        "truth",
        "truth_cities",
        "truth_common",
        "utility",
    )
    sizes: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256, 512)
    reference_pairs: int = 2048
    failure_accuracy: float = 0.6
    saturation_delta: float = 0.01
    n_boot: int = 2000


def oriented_differences(prepared: Prepared) -> NDArray[np.float64]:
    """Takes the label-1 member minus the label-0 member of every pair.

    The order inside a stored pair is random, so a plain difference would
    flip sign from pair to pair; the oriented one does not.

    Args:
        prepared: The stimuli and activations of a training set, read at one
            row per stimulus.

    Returns:
        Array of shape (K, m, d): layer, pair and feature.

    Raises:
        ValueError: If a pair holds two members of the same label.
    """
    stimuli, act = prepared
    pairs = act.row_groups(stimuli.pairs)
    labels = act.row_values(stimuli.labels)
    first = labels[pairs[:, 0]]
    equal = np.flatnonzero(first == labels[pairs[:, 1]])
    if equal.size:
        raise ValueError(f"pair {int(equal[0])} has equal labels")
    swap = first == 0
    positive = np.where(swap, pairs[:, 1], pairs[:, 0])
    negative = np.where(swap, pairs[:, 0], pairs[:, 1])
    return act.data[:, positive].astype(np.float64) - act.data[
        :, negative
    ].astype(np.float64)


def directional_agreement(diffs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Averages the cosine between each difference and the mean difference.

    It is one when every pair points the same way and falls toward zero as
    the pairs disagree, so a training set can be judged without labels from
    the evaluation set.

    Args:
        diffs: Pair differences of shape (K, m, d), with m at least 2.

    Returns:
        Array of shape (K,).

    Raises:
        ValueError: If `diffs` is not 3-D, has fewer than two pairs, or a
            difference or the mean difference is the zero vector.
    """
    vectors = np.asarray(diffs, dtype=np.float64)
    if vectors.ndim != 3 or vectors.shape[1] < 2:
        raise ValueError(
            "directional_agreement needs differences of shape (K, m, d) with "
            f"m >= 2, got {vectors.shape}"
        )
    mean = vectors.mean(axis=1)
    norms = np.linalg.norm(vectors, axis=2)
    mean_norms = np.linalg.norm(mean, axis=1)
    if (norms == 0.0).any() or (mean_norms == 0.0).any():
        raise ValueError("directional_agreement is undefined for a zero vector")
    cosines = np.einsum("kmd,kd->km", vectors, mean) / (
        norms * mean_norms[:, None]
    )
    return np.asarray(cosines.mean(axis=1), dtype=np.float64)


def saturation_size(
    sizes: Sequence[int], accuracy: NDArray[np.float64], *, delta: float
) -> int:
    """Finds the smallest size that is within `delta` of the largest.

    Args:
        sizes: The training-set sizes, Z of them.
        accuracy: Report accuracy of shape (Z, R): size and resample.
        delta: How far below the mean accuracy of the largest size a size's
            mean may fall and still count as saturated.

    Returns:
        The smallest size whose mean accuracy is at least the largest
        size's mean minus `delta`.

    Raises:
        ValueError: If `accuracy` does not have shape (Z, R) or `delta` is
            negative.
    """
    grid, values = _checked_saturation_inputs(sizes, accuracy, delta)
    return int(_saturated(grid, values.mean(axis=1), delta))


def saturation_ci(
    sizes: Sequence[int],
    accuracy: NDArray[np.float64],
    *,
    delta: float,
    n_boot: int,
    seed: int,
) -> tuple[int, int, int]:
    """Bounds the saturation size by a percentile bootstrap over resamples.

    Each bootstrap draw resamples the R resamples of every size with
    replacement, independently per size, and finds the saturation size of
    the resampled means.

    Args:
        sizes: The training-set sizes, Z of them.
        accuracy: Report accuracy of shape (Z, R): size and resample.
        delta: As in `saturation_size`.
        n_boot: Number of bootstrap draws.
        seed: Seed of the draws.

    Returns:
        The saturation size and the 2.5 and 97.5 percent bounds of its
        bootstrap distribution, each one of `sizes`.

    Raises:
        ValueError: If `accuracy` does not have shape (Z, R), `delta` is
            negative or `n_boot` is below 1.
    """
    grid, values = _checked_saturation_inputs(sizes, accuracy, delta)
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot}")
    n_sizes, n_resamples = values.shape
    draws = np.random.default_rng(seed).integers(
        0, n_resamples, size=(n_boot, n_sizes, n_resamples)
    )
    means = np.take_along_axis(values[None], draws, axis=2).mean(axis=2)
    boot = _saturated(grid, means, delta)
    return (
        saturation_size(sizes, accuracy, delta=delta),
        int(np.percentile(boot, 2.5, method="lower")),
        int(np.percentile(boot, 97.5, method="higher")),
    )


def _checked_saturation_inputs(
    sizes: Sequence[int], accuracy: NDArray[np.float64], delta: float
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    grid = np.asarray(sizes, dtype=np.int64)
    values = np.asarray(accuracy, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != grid.shape[0]:
        raise ValueError(
            f"saturation needs accuracy of shape ({grid.shape[0]}, R), one "
            f"row per size, got {values.shape}"
        )
    if delta < 0.0:
        raise ValueError(f"saturation needs a delta of at least 0, got {delta}")
    return grid, values


def _saturated(
    sizes: NDArray[np.int64], means: NDArray[np.float64], delta: float
) -> NDArray[np.int64]:
    """Returns the smallest size reaching the largest size's mean minus delta.

    Args:
        sizes: The sizes, shape (Z,).
        means: Mean accuracy of shape (..., Z).
        delta: The tolerance, at least 0, so the largest size always counts.

    Returns:
        Array of shape (...,).
    """
    floor = means[..., np.argmax(sizes)][..., None] - delta
    return np.where(means >= floor, sizes, np.iinfo(np.int64).max).min(axis=-1)


def spearman(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Computes the Spearman rank correlation, averaging the ranks of ties.

    Args:
        x: Values of shape (n,).
        y: Values of shape (n,).

    Returns:
        The Pearson correlation of the ranks; NaN when one side is constant.

    Raises:
        ValueError: If `x` and `y` are not vectors of the same length, at
            least 2.
    """
    first = np.asarray(x, dtype=np.float64)
    second = np.asarray(y, dtype=np.float64)
    if first.ndim != 1 or first.shape != second.shape or first.shape[0] < 2:
        raise ValueError(
            "spearman needs two vectors of the same length, at least 2, got "
            f"shapes {first.shape} and {second.shape}"
        )
    ranks_x = _average_ranks(first)
    ranks_y = _average_ranks(second)
    centered_x = ranks_x - ranks_x.mean()
    centered_y = ranks_y - ranks_y.mean()
    norm = np.sqrt((centered_x**2).sum() * (centered_y**2).sum())
    if norm == 0.0:
        return float("nan")
    return float((centered_x * centered_y).sum() / norm)


def _average_ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    ends = np.r_[starts[1:], values.shape[0]]
    ranks = np.empty(values.shape[0], dtype=np.float64)
    ranks[order] = np.repeat((starts + ends + 1) / 2, ends - starts)
    return ranks


@dataclass(frozen=True, eq=False)
class _Plan:
    """What one task reads: its splits and the sizes that fit them."""

    task: ReadingTask
    variant: Variant
    splits: TaskSplits
    capacity: int
    sizes: tuple[int, ...]
    skipped: tuple[int, ...]


@dataclass(frozen=True, eq=False)
class _Result:
    """The measurements of one task.

    Attributes:
        task: The task.
        sizes: The sizes that ran, Z of them.
        layers: Layer index of each of the K layers.
        layer: The reading layer, an element of `layers`.
        pairs: Training pairs of the reference reader.
        accuracy: Array of shape (Z, 2, 3, R, K): size, select or report,
            `SERIES`, resample and layer.
        cosine: Cosine to the reference direction, shape (Z, R, K).
        agreement: Directional agreement of the training pairs, (Z, R, K).
        reference: Accuracy of the reference reader, shape (2, 3, K).
        direction: Oriented unit direction of the reference, shape (K, d).
    """

    task: ReadingTask
    sizes: tuple[int, ...]
    layers: tuple[int, ...]
    layer: int
    pairs: int
    accuracy: NDArray[np.float64]
    cosine: NDArray[np.float64]
    agreement: NDArray[np.float64]
    reference: NDArray[np.float64]
    direction: NDArray[np.float64]

    @property
    def at(self) -> int:
        """The position of the reading layer among the layers."""
        return self.layers.index(self.layer)

    @property
    def report(self) -> NDArray[np.float64]:
        """The report accuracy of the reader at the reading layer, (Z, R)."""
        return self.accuracy[:, 1, 0, :, self.at]


def _checked_tasks(config: Config) -> tuple[ReadingTask, ...]:
    if config.n_seeds < 2:
        raise ValueError(f"n_seeds must be at least 2, got {config.n_seeds}")
    if "response" in (config.position, config.eval_position):
        raise ValueError(
            "p3 reads one row per stimulus, not the 'response' position"
        )
    sizes = config.sizes
    if (
        len(sizes) < 2
        or sizes[0] < 2
        or any(later <= earlier for earlier, later in itertools.pairwise(sizes))
    ):
        raise ValueError(
            "sizes must be increasing, at least two of them and each at "
            f"least 2, got {sizes}"
        )
    tasks = tuple(READING_TASKS[name] for name in expand_tasks(config.tasks))
    for task in tasks:
        if task.pairing == "one_vs_rest":
            raise ValueError(f"p3 reads binary tasks only, not {task.name!r}")
    return tasks


def _plan(
    config: Config,
    task: ReadingTask,
    chat: ChatFormat,
    examples: Mapping[str, Sequence[Example]],
) -> _Plan:
    (variant,), _ = template_variants(task, chat, kinds=("paper",))
    splits = task_splits(task, examples, seed=config.seed)
    capacity = pair_capacity(task, splits.train)
    sizes = tuple(size for size in config.sizes if size <= capacity)
    if len(sizes) < 2:
        raise ValueError(
            f"task {task.name!r} has {capacity} training pairs, fewer than "
            f"two of the sizes {config.sizes}"
        )
    skipped = tuple(size for size in config.sizes if size > capacity)
    return _Plan(task, variant, splits, capacity, sizes, skipped)


def _evaluation(
    loaded: LoadedModel,
    config: Config,
    plan: _Plan,
    cache: dict[str, dict[str, Prepared]],
) -> dict[str, Prepared]:
    """Prepares the select and report items of a task's target, once."""
    target = target_of(plan.task).name
    if target not in cache:
        position = (
            config.position
            if config.eval_position is None
            else config.eval_position
        )
        cache[target] = {
            split: prepare(
                loaded,
                task_set(
                    plan.task,
                    plan.variant,
                    part,
                    n_pairs=eval_pairs(plan.task, part, config.n_eval_pairs),
                    seed=eval_seed(plan.task, split, config.seed),
                ),
                position,
                config,
            )
            for split, part in zip(
                SPLITS, (plan.splits.select, plan.splits.report), strict=True
            )
        }
    return cache[target]


def _train_sets(
    loaded: LoadedModel,
    config: Config,
    plan: _Plan,
    *,
    n_pairs: int,
    seeds: Sequence[int],
) -> list[Prepared]:
    return [
        prepare(
            loaded,
            task_set(
                plan.task,
                plan.variant,
                plan.splits.train,
                n_pairs=n_pairs,
                seed=seed,
            ),
            config.position,
            config,
        )
        for seed in seeds
    ]


def _read_task(
    loaded: LoadedModel,
    config: Config,
    plan: _Plan,
    evaluation: dict[str, Prepared],
) -> _Result:
    """Reads the reference and every size of one task."""
    name = plan.task.name
    pairs = min(plan.capacity, config.reference_pairs)
    reference = sweep_from_activations(
        _train_sets(
            loaded,
            config,
            plan,
            n_pairs=pairs,
            seeds=[derive_seed(config.seed, "reference", name)],
        ),
        evaluation,
        method=config.method,
        preprocess=config.preprocess,
        seed=derive_seed(config.seed, "reference", name, "sweep"),
    )
    layers = reference.layers
    reference_accuracy = np.stack(
        [reference.per_seed(split)[:, 0, :] for split in SPLITS]
    )
    layer = best_layer(reference_accuracy[0, 0], layers)
    direction = _oriented(
        reference.readers[0].directions, reference.readers[0].signs
    )
    n_sizes, n_layers = len(plan.sizes), len(layers)
    accuracy = np.zeros(
        (n_sizes, len(SPLITS), len(SERIES), config.n_seeds, n_layers)
    )
    cosine = np.zeros((n_sizes, config.n_seeds, n_layers))
    agreement = np.zeros((n_sizes, config.n_seeds, n_layers))
    for z, size in enumerate(plan.sizes):
        train = _train_sets(
            loaded,
            config,
            plan,
            n_pairs=size,
            seeds=[
                derive_seed(config.seed, "size", name, size, r)
                for r in range(config.n_seeds)
            ],
        )
        sweep = sweep_from_activations(
            train,
            evaluation,
            method=config.method,
            preprocess=config.preprocess,
            seed=derive_seed(config.seed, "sweep", name, size),
        )
        accuracy[z] = np.stack([sweep.per_seed(split) for split in SPLITS])
        for r, (reader, prepared) in enumerate(
            zip(sweep.readers, train, strict=True)
        ):
            oriented = _oriented(reader.directions, reader.signs)
            cosine[z, r] = np.clip(
                np.einsum("kd,kd->k", oriented, direction), -1.0, 1.0
            )
            agreement[z, r] = directional_agreement(
                oriented_differences(prepared)
            )
    return _Result(
        plan.task,
        plan.sizes,
        layers,
        layer,
        pairs,
        accuracy,
        cosine,
        agreement,
        reference_accuracy,
        direction,
    )


def _oriented(
    directions: NDArray[np.float32], signs: NDArray[np.float32]
) -> NDArray[np.float64]:
    return (directions * signs[:, None]).astype(np.float64)


def _write_task(
    run_dir: RunDir, config: Config, result: _Result
) -> tuple[list[dict[str, Any]], dict[str, Any], float | None]:
    """Writes the files of a task; returns its rows, saturation and AUROC."""
    name = result.task.name
    target = target_of(result.task).name
    at = result.at
    run_dir.write_arrays(
        f"sizes_{name}",
        sizes=np.asarray(result.sizes, dtype=np.int64),
        layers=np.asarray(result.layers, dtype=np.int64),
        accuracy=result.accuracy,
        cosine=result.cosine,
        agreement=result.agreement,
        reference=result.reference,
    )
    run_dir.write_json(
        f"sizes_{name}",
        {
            "sizes": list(result.sizes),
            "splits": list(SPLITS),
            "series": list(SERIES),
            "layer": result.layer,
            "target": target,
            "reference_pairs": result.pairs,
        },
    )
    stem = f"curves_{name}"
    write_curves(
        run_dir,
        np.moveaxis(result.accuracy[:, 1, :, :, at], 0, -1),
        result.sizes,
        series=SERIES,
        stem=stem,
        xlabel=SIZE_LABEL,
        chance=0.5,
        n_boot=config.n_boot,
        seed=derive_seed(config.seed, "curves", stem),
    )
    statistics = np.stack([result.cosine[:, :, at], result.agreement[:, :, at]])
    write_curves(
        run_dir,
        np.moveaxis(statistics, 1, -1),
        result.sizes,
        series=("cosine to reference", "pair agreement"),
        stem=f"{stem}_cosine",
        xlabel=SIZE_LABEL,
        ylabel="Cosine",
        n_boot=config.n_boot,
        seed=derive_seed(config.seed, "curves", f"{stem}_cosine"),
    )
    band = run_dir.read_arrays(stem)
    rows = []
    for z, size in enumerate(result.sizes):
        report = result.report[z]
        agreement = result.agreement[z, :, at]
        cosine = result.cosine[z, :, at]
        low, high = np.quantile(report, [0.05, 0.95])
        values = {
            "task": name,
            "target": target,
            "size": size,
            "layer": result.layer,
            "accuracy": report.mean(),
            "lo": band["lo"][0, z],
            "hi": band["hi"][0, z],
            "sd": report.std(ddof=1),
            "q05": low,
            "q95": high,
            "failure": (report < config.failure_accuracy).mean(),
            "shuffled": result.accuracy[z, 1, 2, :, at].mean(),
            "cosine": cosine.mean(),
            "cosine_sd": cosine.std(ddof=1),
            "agreement": agreement.mean(),
            "rho": spearman(agreement, report),
        }
        rows.append({column: values[column] for column in COLUMNS})
    point, lower, upper = saturation_ci(
        result.sizes,
        result.report,
        delta=config.saturation_delta,
        n_boot=config.n_boot,
        seed=derive_seed(config.seed, "saturation", name),
    )
    saturation = {
        "size": point,
        "lo": lower,
        "hi": upper,
        "delta": config.saturation_delta,
    }
    success = (result.report >= config.failure_accuracy).reshape(-1)
    failure_auroc = None
    if success.any() and not success.all():
        pooled = result.agreement[:, :, at].reshape(-1)
        failure_auroc = float(auroc(pooled, success.astype(np.int64)))
    return rows, saturation, failure_auroc


def _write_sources(
    run_dir: RunDir, config: Config, results: Mapping[str, _Result]
) -> dict[str, str]:
    """Writes the curve and matrix of every target with several sources."""
    groups: dict[str, list[str]] = {}
    for name, result in results.items():
        groups.setdefault(target_of(result.task).name, []).append(name)
    skipped: dict[str, str] = {}
    for target, names in groups.items():
        if target not in results:
            skipped[target] = f"{target!r} is not in tasks"
            continue
        if len(names) < 2:
            skipped[target] = (
                f"no other task in tasks has {target!r} as its target"
            )
            continue
        shared = sorted(
            set.intersection(*(set(results[n].sizes) for n in names))
        )
        stem = f"curves_sources_{target}"
        write_curves(
            run_dir,
            np.stack(
                [
                    results[n]
                    .report[[results[n].sizes.index(size) for size in shared]]
                    .T
                    for n in names
                ]
            ),
            shared,
            series=names,
            stem=stem,
            xlabel=SIZE_LABEL,
            chance=0.5,
            n_boot=config.n_boot,
            seed=derive_seed(config.seed, "curves", stem),
        )
        at = results[target].at
        write_matrix(
            run_dir,
            f"sources_{target}",
            cosine_matrix(np.stack([results[n].direction[at] for n in names])),
            rows=names,
            cols=names,
            title=(
                f"Cosine between the {target} reference directions at layer "
                f"{results[target].layer}"
            ),
        )
    return skipped


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    examples: Mapping[str, Sequence[Example]],
) -> None:
    """Reads every task at every size and writes the study.

    The settings and the sizes that fit each task's training pairs are
    checked before the model runs. Per task it writes `sizes_<task>` (the
    accuracy, cosine to the reference and pair agreement of every resample
    at every layer), `curves_<task>` and `curves_<task>_cosine`; per target
    with several sources `curves_sources_<target>` and `sources_<target>`;
    and `table.json` and `summary.json`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model.
        examples: The examples of each source that the tasks and their
            targets read.

    Raises:
        ValueError: If a setting is invalid, a task is not binary, a source
            is missing, or fewer than two sizes fit a task's training pairs.
    """
    tasks = _checked_tasks(config)
    chat = chat_format_for(loaded)
    plans = [_plan(config, task, chat, examples) for task in tasks]
    cache: dict[str, dict[str, Prepared]] = {}
    results: dict[str, _Result] = {}
    rows: list[dict[str, Any]] = []
    saturation: dict[str, dict[str, Any]] = {}
    failure_auroc: dict[str, float | None] = {}
    for plan in plans:
        name = plan.task.name
        result = _read_task(
            loaded, config, plan, _evaluation(loaded, config, plan, cache)
        )
        results[name] = result
        task_rows, saturation[name], failure_auroc[name] = _write_task(
            run_dir, config, result
        )
        rows.extend(task_rows)
    skipped_sources = _write_sources(run_dir, config, results)
    run_dir.write_json("table", rows)
    run_dir.write_json(
        "summary",
        {
            "tasks": list(results),
            "targets": {n: target_of(r.task).name for n, r in results.items()},
            "layers": {n: r.layer for n, r in results.items()},
            "reference": {
                n: {
                    "pairs": r.pairs,
                    "select": r.reference[0, 0, r.at],
                    "report": r.reference[1, 0, r.at],
                }
                for n, r in results.items()
            },
            "skipped_sizes": {p.task.name: list(p.skipped) for p in plans},
            "skipped_sources": skipped_sources,
            "saturation": saturation,
            "failure_auroc": failure_auroc,
        },
    )


def run(config: Config, run_dir: RunDir) -> None:
    """Loads the model and the datasets, then runs `execute`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
    """
    tasks = _checked_tasks(config)
    sources = list(
        dict.fromkeys(
            source
            for task in tasks
            for source in (task.source, target_of(task).source)
        )
    )
    execute(config, run_dir, load_model(config.model), load_sources(sources))


def plot_size_curves(
    run_dir: RunDir, out_dir: Path, *, stem: str
) -> list[Path]:
    """Draws a `curves` file whose x axis is a training size, on a log2 axis.

    Args:
        run_dir: The run holding the pair.
        out_dir: Folder of the figures.
        stem: File stem of the pair, also the name of the figures.

    Returns:
        The PDF, PNG and SVG paths.
    """
    arrays = run_dir.read_arrays(stem)
    sidecar = cast(dict[str, Any], run_dir.read_json(stem))
    curves = curves_from_arrays(
        arrays["x"], arrays["mean"], arrays["lo"], arrays["hi"]
    )
    with style_context():
        fig, axes = new_figure()
        ax = axes[0, 0]
        plot_curves(
            curves,
            series=sidecar["series"],
            per_seed=list(arrays["per_seed"]),
            xlabel=sidecar["xlabel"],
            ylabel=sidecar["ylabel"],
            chance=sidecar["chance"],
            n_layers=sidecar["n_layers"],
            ax=ax,
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(arrays["x"])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(NullLocator())
    return save_figure(fig, out_dir, stem)


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws every P3 curve on a log2 axis and every sources matrix.

    Args:
        run_dir: A finished P3 run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths.
    """
    paths: list[Path] = []
    for file in sorted(run_dir.path.glob("curves_*.npz")):
        paths += plot_size_curves(run_dir, out_dir, stem=file.stem)
    for file in sorted(run_dir.path.glob("sources_*.npz")):
        paths += plot_matrix_file(
            run_dir,
            out_dir,
            stem=file.stem,
            kind="diverging",
            vmin=-1.0,
            vmax=1.0,
        )
    return paths
