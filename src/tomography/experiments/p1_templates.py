"""P1: how much the reading depends on the paper's template.

Every variant of a task's template is read with the same seeds and the same
items as the paper variant, so a difference in accuracy, in direction or in
transfer belongs to the wording and not to the draw.
"""

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import cast

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
from tomography.experiments.sweep import plot_curves_file
from tomography.experiments.sweep import write_curves
from tomography.experiments.variants import FAMILIES
from tomography.experiments.variants import READING_TASKS
from tomography.experiments.variants import VARIANT_KINDS
from tomography.experiments.variants import ReadingTask
from tomography.experiments.variants import Variant
from tomography.experiments.variants import VariantKind
from tomography.experiments.variants import VariantSweep
from tomography.experiments.variants import eval_pairs
from tomography.experiments.variants import eval_seed
from tomography.experiments.variants import expand_tasks
from tomography.experiments.variants import load_sources
from tomography.experiments.variants import prepare
from tomography.experiments.variants import target_of
from tomography.experiments.variants import task_examples
from tomography.experiments.variants import task_set
from tomography.experiments.variants import task_splits
from tomography.experiments.variants import template_variants
from tomography.experiments.variants import variant_cosines
from tomography.experiments.variants import variant_sweep
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.reading.directions import Method
from tomography.reading.metrics import best_layer
from tomography.reading.metrics import bootstrap_ci
from tomography.templates import ChatFormat

READER = SERIES.index("reader")
RANDOM = SERIES.index("random")
SHUFFLED = SERIES.index("shuffled")
COLUMNS: tuple[str, ...] = (
    "task",
    "variant",
    "kind",
    "family",
    "layer",
    "select",
    "report",
    "lo",
    "hi",
    "random",
    "shuffled",
    "from_paper",
    "to_paper",
    "cosine_paper",
    "stability",
    "own_layer",
    "own_report",
)


@dataclass(frozen=True, kw_only=True)
class Config(ReadingConfig):
    """P1 settings: GPT-2 template variants.

    The shared defaults apply except for the experiment id, the model, the
    number of evaluation pairs and the direction finder, which is the GPT-2
    reading default (`logistic`); `pca_diff` is a second run.

    Attributes:
        tasks: Keys of `READING_TASKS`, where "emotion" stands for the six
            emotion tasks.
        kinds: The variant kinds to read; `paper` is required.
    """

    experiment: str = "p1"
    model: str = "gpt2"
    method: Method = "logistic"
    n_eval_pairs: int | None = 256
    tasks: tuple[str, ...] = ("sentiment", "truth", "emotion")
    kinds: tuple[VariantKind, ...] = VARIANT_KINDS


@dataclass(frozen=True)
class _TaskResult:
    rows: list[dict[str, Any]]
    layer: int
    skipped: dict[str, str]
    n_eval_pairs: dict[str, int]


def _prepared_sets(
    config: Config,
    loaded: LoadedModel,
    task: ReadingTask,
    variants: Sequence[Variant],
    examples: Mapping[str, Sequence[Example]],
) -> tuple[
    dict[str, list[Prepared]], dict[str, dict[str, Prepared]], dict[str, int]
]:
    """Prepares the training sets and the select and report sets of a task."""
    splits = task_splits(task, examples, seed=config.seed)
    eval_position = config.eval_position or config.position
    evals: dict[str, dict[str, Prepared]] = {}
    sizes: dict[str, int] = {}
    for split, part in (("select", splits.select), ("report", splits.report)):
        stimuli = {
            v.name: task_set(
                task,
                v,
                part,
                n_pairs=eval_pairs(task, part, config.n_eval_pairs),
                seed=eval_seed(task, split, config.seed),
            )
            for v in variants
            if v.evaluable
        }
        if len({s.uids for s in stimuli.values()}) != 1:
            raise ValueError(
                f"the evaluable variants of task {task.name!r} do not share "
                f"their {split} items"
            )
        sizes[split] = int(next(iter(stimuli.values())).groups.shape[0])
        evals[split] = {
            name: prepare(loaded, s, eval_position, config)
            for name, s in stimuli.items()
        }
    train = {
        v.name: [
            prepare(
                loaded,
                task_set(
                    task,
                    v,
                    splits.train,
                    n_pairs=config.n_pairs,
                    seed=derive_seed(config.seed, "train", i),
                ),
                v.position or config.position,
                config,
            )
            for i in range(config.n_seeds)
        ]
        for v in variants
    }
    return train, evals, sizes


def _write_task(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    task: ReadingTask,
    variants: Sequence[Variant],
    sweep: VariantSweep,
    cosines: NDArray[np.float64],
    layer: int,
) -> None:
    """Writes the sweep, the curves and the two matrices of a task."""
    name = task.name
    k = sweep.layers.index(layer)
    run_dir.write_arrays(
        f"sweep_{name}",
        accuracy=sweep.accuracy,
        cosine=cosines,
        layers=np.asarray(sweep.layers, dtype=np.int64),
    )
    run_dir.write_json(
        f"sweep_{name}",
        {
            "splits": list(sweep.splits),
            "train_variants": list(sweep.train_variants),
            "eval_variants": list(sweep.eval_variants),
            "series": list(SERIES),
            "layer": layer,
        },
    )

    def curves(stem: str, split: str, names: Sequence[str]) -> None:
        write_curves(
            run_dir,
            np.stack([sweep.own(split, n)[READER] for n in names]),
            sweep.layers,
            series=names,
            stem=stem,
            chance=0.5,
            n_layers=loaded.n_layers,
            seed=config.seed,
        )

    names = [v.name for v in variants]
    curves(f"curves_{name}", "report", names)
    curves(f"curves_{name}_val", "select", names)
    for family in FAMILIES:
        members = [
            v.name for v in variants if v.family == family or v.name == "paper"
        ]
        if len(members) > 1:
            curves(f"curves_{name}_{family}", "report", members)
    transfer = sweep.accuracy[1, :, :, READER, :, k].mean(axis=-1)
    write_matrix(
        run_dir,
        f"transfer_{name}",
        transfer,
        rows=sweep.train_variants,
        cols=sweep.eval_variants,
        title=f"{name}: report accuracy at layer {layer}",
    )
    write_matrix(
        run_dir,
        f"cosine_{name}",
        cosines[k],
        rows=sweep.train_variants,
        cols=sweep.train_variants,
        title=f"{name}: direction cosine at layer {layer}",
    )


def _table_rows(
    config: Config,
    task: ReadingTask,
    variants: Sequence[Variant],
    sweep: VariantSweep,
    cosines: NDArray[np.float64],
    layer: int,
) -> list[dict[str, Any]]:
    """Builds the table rows of a task, one per variant, at one layer."""
    k = sweep.layers.index(layer)
    paper = sweep.train_variants.index("paper")
    rows = []
    for v in variants:
        at = sweep.train_variants.index(v.name)
        report = sweep.own("report", v.name)
        select = sweep.own("select", v.name)
        lo, hi = bootstrap_ci(report[READER, :, k], seed=config.seed)
        own_layer = best_layer(select[READER].mean(axis=0), sweep.layers)
        rows.append(
            {
                "task": task.name,
                "variant": v.name,
                "kind": v.kind,
                "family": v.family,
                "layer": layer,
                "select": float(select[READER, :, k].mean()),
                "report": float(report[READER, :, k].mean()),
                "lo": float(lo),
                "hi": float(hi),
                "random": float(report[RANDOM, :, k].mean()),
                "shuffled": float(report[SHUFFLED, :, k].mean()),
                "from_paper": (
                    float(
                        sweep.at("report", "paper", v.name)[READER, :, k].mean()
                    )
                    if v.evaluable
                    else None
                ),
                "to_paper": float(
                    sweep.at("report", v.name, "paper")[READER, :, k].mean()
                ),
                "cosine_paper": float(cosines[k, at, paper]),
                "stability": float(cosines[k, at, at]),
                "own_layer": own_layer,
                "own_report": float(
                    report[READER, :, sweep.layers.index(own_layer)].mean()
                ),
            }
        )
    return rows


def _read_task(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    chat: ChatFormat,
    task: ReadingTask,
    examples: Mapping[str, Sequence[Example]],
) -> _TaskResult:
    variants, skipped = template_variants(task, chat, kinds=config.kinds)
    train, evals, sizes = _prepared_sets(
        config, loaded, task, variants, examples
    )
    sweep = variant_sweep(
        train,
        evals,
        method=config.method,
        preprocess=config.preprocess,
        seed=config.seed,
    )
    cosines = variant_cosines(sweep.directions)
    layer = best_layer(
        sweep.at("select", "paper", "paper")[READER].mean(axis=0), sweep.layers
    )
    _write_task(config, run_dir, loaded, task, variants, sweep, cosines, layer)
    rows = _table_rows(config, task, variants, sweep, cosines, layer)
    return _TaskResult(rows, layer, skipped, sizes)


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    examples: Mapping[str, Sequence[Example]],
) -> None:
    """Reads every task through every template variant.

    Per task it trains every variant on the same seeds and examples, scores
    every reader on the select and report items of every evaluable variant,
    and compares the readers at one layer: the best select layer of the paper
    reader. Writes, per task, `sweep_<task>` (the accuracy of every trained-on
    and evaluated-on pair, and the cosines), `curves_<task>` and
    `curves_<task>_val` (the reader of every variant on its own items, report
    and select split), `curves_<task>_<family>` (the report curves of `paper`
    and one family, for the figures), `transfer_<task>` and `cosine_<task>`;
    and `table.json` and `summary.json` for the whole run.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model.
        examples: The examples of each source, by source name.

    Raises:
        ValueError: If `n_seeds` is below 2, a task is unknown or a source is
            missing (all checked before any model call), or the evaluable
            variants of a task do not share their evaluation items.
    """
    tasks = expand_tasks(config.tasks)
    if config.n_seeds < 2:
        raise ValueError(
            "p1 needs n_seeds of at least 2, because the stability of a "
            "reader is a cosine between two seeds"
        )
    for name in tasks:
        task_examples(READING_TASKS[name], examples)
        task_examples(target_of(READING_TASKS[name]), examples)
    chat = chat_format_for(loaded)
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "tasks": list(tasks),
        "layers": {},
        "skipped": {},
        "n_eval_pairs": {},
        "method": config.method,
    }
    for name in tasks:
        result = _read_task(
            config, run_dir, loaded, chat, READING_TASKS[name], examples
        )
        rows.extend(result.rows)
        summary["layers"][name] = result.layer
        summary["skipped"][name] = result.skipped
        summary["n_eval_pairs"][name] = result.n_eval_pairs
    run_dir.write_json("table", rows)
    run_dir.write_json("summary", summary)


def run(config: Config, run_dir: RunDir) -> None:
    """Loads the model and the datasets of the tasks, then runs `execute`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.

    Raises:
        ValueError: If a task is unknown (before the model is loaded).
    """
    sources = list(
        dict.fromkeys(
            source
            for name in expand_tasks(config.tasks)
            for source in (
                READING_TASKS[name].source,
                target_of(READING_TASKS[name]).source,
            )
        )
    )
    execute(config, run_dir, load_model(config.model), load_sources(sources))


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws the P1 figures of every task.

    Per task: one curve figure for each family that has a member besides the
    paper variant, the transfer matrix (accuracy in 0 to 1) and the cosine
    matrix (signed, -1 to 1).

    Args:
        run_dir: A finished P1 run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths.
    """
    summary = cast(dict[str, Any], run_dir.read_json("summary"))
    paths: list[Path] = []
    for task in summary["tasks"]:
        for family in FAMILIES:
            stem = f"curves_{task}_{family}"
            if (run_dir.path / f"{stem}.npz").is_file():
                paths += plot_curves_file(run_dir, out_dir, stem=stem)
        paths += plot_matrix_file(
            run_dir, out_dir, stem=f"transfer_{task}", vmin=0.0, vmax=1.0
        )
        paths += plot_matrix_file(
            run_dir,
            out_dir,
            stem=f"cosine_{task}",
            kind="diverging",
            vmin=-1.0,
            vmax=1.0,
        )
    return paths
