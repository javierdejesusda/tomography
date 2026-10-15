"""P6: does the reading change with model size, across the GPT-2 sizes.

An aggregate experiment: it loads no model and reads the finished default
runs of the reading experiments, and of C1 for the dose-response half, on every
size. Each reading run keeps its own rule for the best layer: it is chosen on
the validation curves when the run writes them and reported on the test
curves, so a larger model, with more layers to take a maximum over, does not
gain from choosing and reporting on one split.
"""

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import cast

import numpy as np
from numpy.typing import NDArray
import torch
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import PretrainedConfig

from tomography.experiments.aggregate import out_root_of
from tomography.experiments.aggregate import read_curves
from tomography.experiments.aggregate import reading_point
from tomography.experiments.aggregate import relative_depth
from tomography.experiments.aggregate import resolve_runs
from tomography.experiments.base import ExperimentConfig
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.steering import dose_slope
from tomography.experiments.sweep import write_curves
from tomography.model import MODELS
from tomography.model import UnknownCheckpointError
from tomography.reading.metrics import bootstrap_ci
from tomography.viz.reading import plot_curves
from tomography.viz.style import COLUMN_WIDTH
from tomography.viz.style import INK
from tomography.viz.style import categorical_colors
from tomography.viz.style import new_figure
from tomography.viz.style import save_figure
from tomography.viz.style import style_context

GPT2_SIZES: tuple[str, ...] = ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")
MAX_PANELS = 6
MAX_MODELS = 8
LEGEND_COLUMNS = 4


@dataclass(frozen=True, kw_only=True)
class Config(ExperimentConfig):
    """P6 settings: which runs to read and how to summarize them.

    Attributes:
        models: The checkpoint keys, from 2 to 8, in size order.
        sources: The reading experiments whose default runs are read, from 1
            to `MAX_PANELS`.
        steering_source: The steering experiment whose default runs give the
            dose slopes, or None for the reading half only.
        extra_runs: Finished runs with other settings; each replaces the
            default run of its own experiment and model.
        n_boot: Bootstrap resamples of every interval.
    """

    experiment: str = "p6"
    model: str = "gpt2"
    models: tuple[str, ...] = GPT2_SIZES
    sources: tuple[str, ...] = ("r1", "r2", "r3", "r5", "r7")
    steering_source: str | None = "c1"
    extra_runs: tuple[str, ...] = ()
    n_boot: int = 2000


def count_parameters(config: PretrainedConfig) -> int:
    """Counts the parameters of the model a config describes.

    The model is built on the meta device, so no weight is allocated. A tied
    embedding is listed once by `parameters()` and counts once.

    Args:
        config: The model config.

    Returns:
        The total number of parameters.
    """
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)  # type: ignore[no-untyped-call]
    return sum(p.numel() for p in model.parameters())


def parameter_count(key: str) -> int:
    """Counts the parameters of a registered checkpoint from its config.

    Only the pinned config is read, from the cache the reading runs filled,
    and no weights.

    Args:
        key: A key of `MODELS`.

    Returns:
        The total number of parameters, with a tied embedding counted once.

    Raises:
        UnknownCheckpointError: If `key` is not in `MODELS`.
    """
    checkpoint = MODELS.get(key)
    if checkpoint is None:
        raise UnknownCheckpointError(
            f"Unknown checkpoint {key!r}; known: {sorted(MODELS)}"
        )
    config = AutoConfig.from_pretrained(
        checkpoint.repo_id, revision=checkpoint.revision
    )
    return count_parameters(config)


def copy_reader_curves(
    run_dir: RunDir,
    source: str,
    model: str,
    run: RunDir,
    *,
    n_boot: int,
    seed: int,
) -> None:
    """Copies the reader series of a source run into this run.

    The copy is written as `curves_<source>_<model>` with one series named
    after the model and a fresh band, so the figures read only this run's own
    files.

    Args:
        run_dir: The run to write into.
        source: The experiment id of the source run, such as "r1".
        model: The checkpoint key of the source run.
        run: The finished source run holding `curves`.
        n_boot: Bootstrap resamples of the band.
        seed: Seed of the resampling.

    Raises:
        ValueError: If the source curves have no `reader` series.
    """
    curves = read_curves(run)
    if "reader" not in curves.series:
        raise ValueError(
            f"the curves of {run.path} have no reader series, got "
            f"{curves.series}"
        )
    per_seed = curves.per_seed[curves.series.index("reader")]
    write_curves(
        run_dir,
        per_seed[None],
        [int(layer) for layer in curves.x],
        series=[model],
        stem=f"curves_{source}_{model}",
        chance=curves.chance,
        n_layers=curves.n_layers,
        n_boot=n_boot,
        seed=seed,
    )


def steering_slopes(run: RunDir) -> NDArray[np.float64]:
    """Fits the dose slope of every steering reader of a C1 run.

    Args:
        run: A finished C1 run holding `curves_operators`: the relative
            coefficients on the x axis, `none` at 0.0 and one row per
            steering reader in each operator series.

    Returns:
        The least-squares slope of score on coefficient for each reader, from
        the `linear` series, shape (R,).

    Raises:
        ValueError: If the run has no `linear` series, or a reader has fewer
            than two finite points.
    """
    curves = read_curves(run, "curves_operators")
    if "linear" not in curves.series:
        raise ValueError(
            f"the curves_operators of {run.path} have no linear series, "
            f"got {curves.series}"
        )
    readers = curves.per_seed[curves.series.index("linear")]
    coeffs = curves.x.tolist()
    return np.array(
        [dose_slope(coeffs, row) for row in readers], dtype=np.float64
    )


def _check(
    config: Config,
    runs: Mapping[tuple[str, str], RunDir],
    n_params: Mapping[str, int],
) -> None:
    """Checks the config and the inputs before anything is read."""
    if config.model not in config.models:
        raise ValueError(
            f"model {config.model!r} must be one of models {config.models}"
        )
    if not 2 <= len(config.models) <= MAX_MODELS:
        raise ValueError(
            f"models must hold 2 to {MAX_MODELS} keys, got {config.models}"
        )
    if len(set(config.models)) != len(config.models):
        raise ValueError(f"models must not repeat, got {config.models}")
    if not 1 <= len(config.sources) <= MAX_PANELS:
        raise ValueError(
            f"sources must hold 1 to {MAX_PANELS} ids, got {config.sources}"
        )
    if len(set(config.sources)) != len(config.sources):
        raise ValueError(f"sources must not repeat, got {config.sources}")
    experiments = list(config.sources)
    if config.steering_source is not None:
        experiments.append(config.steering_source)
    lacking = [
        (experiment, model)
        for experiment in experiments
        for model in config.models
        if (experiment, model) not in runs
    ]
    if lacking:
        raise ValueError(f"runs lacks the pairs {lacking}")
    unknown = [model for model in config.models if model not in n_params]
    if unknown:
        raise ValueError(f"n_params lacks the models {unknown}")


def _steering_row(
    model: str, params: int, n_layers: int, run: RunDir
) -> list[object]:
    summary = cast(dict[str, Any], run.read_json("summary"))
    slopes = steering_slopes(run)
    steer, reading = int(summary["steer_layer"]), int(summary["reading_layer"])
    return [
        model,
        params,
        steer,
        steer / n_layers,
        reading,
        reading / n_layers,
        float(slopes.mean()),
        float(slopes.min()),
        float(slopes.max()),
        int(slopes.size),
    ]


def execute(
    config: Config,
    run_dir: RunDir,
    runs: Mapping[tuple[str, str], RunDir],
    n_params: Mapping[str, int],
) -> None:
    """Tabulates the best reading of every source and size.

    Writes `curves_<source>_<model>` for every source and model and `table`
    with one row per source and model: the best layer, chosen on the
    validation curves when the run has them, and the mean accuracy of the
    seeds there with its percentile bootstrap interval. With a steering
    source it also writes `table_steering` with the mean dose slope of every
    model and the range over its readers, and `summary` holds the least-squares
    slope of the best accuracy on log10 of the parameter count of every
    source.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        runs: The finished run of every (experiment, model) pair of the
            sources and the steering source.
        n_params: The parameter count of every model.

    Raises:
        ValueError: If `model` is not one of 2 to 8 distinct `models`, there
            are not 1 to `MAX_PANELS` distinct sources, a run or a parameter
            count is missing, or a run cannot be read. The first checks run
            before any file is read.
    """
    _check(config, runs, n_params)
    rows: list[list[object]] = []
    best: dict[str, list[float]] = {source: [] for source in config.sources}
    depth: dict[str, int] = {}
    for source in config.sources:
        for model in config.models:
            run = runs[(source, model)]
            copy_reader_curves(
                run_dir,
                source,
                model,
                run,
                n_boot=config.n_boot,
                seed=config.seed,
            )
            point = reading_point(run)
            lo, hi = bootstrap_ci(
                point.values,
                n_boot=config.n_boot,
                seed=derive_seed(config.seed, "ci", source, model),
            )
            mean = float(point.values.mean())
            depth.setdefault(model, point.n_layers)
            best[source].append(mean)
            rows.append(
                [
                    source,
                    model,
                    n_params[model],
                    point.n_layers,
                    point.layer,
                    point.layer / point.n_layers,
                    point.selected_on,
                    mean,
                    float(lo),
                    float(hi),
                ]
            )
    run_dir.write_json(
        "table",
        {
            "columns": [
                "source",
                "model",
                "params",
                "n_layers",
                "layer",
                "depth",
                "selected_on",
                "accuracy",
                "lo",
                "hi",
            ],
            "rows": rows,
        },
    )
    if config.steering_source is not None:
        run_dir.write_json(
            "table_steering",
            {
                "columns": [
                    "model",
                    "params",
                    "steer_layer",
                    "steer_depth",
                    "reading_layer",
                    "reading_depth",
                    "slope",
                    "slope_min",
                    "slope_max",
                    "n_readers",
                ],
                "rows": [
                    _steering_row(
                        model,
                        n_params[model],
                        depth[model],
                        runs[(config.steering_source, model)],
                    )
                    for model in config.models
                ],
            },
        )
    log_params = np.log10([n_params[model] for model in config.models])
    run_dir.write_json(
        "summary",
        {
            "models": list(config.models),
            "params": {model: n_params[model] for model in config.models},
            "sources": list(config.sources),
            "steering": config.steering_source,
            "accuracy_per_decade": {
                source: dose_slope(log_params.tolist(), np.asarray(values))
                for source, values in best.items()
            },
        },
    )


def run(config: Config, run_dir: RunDir) -> None:
    """Finds the source runs and the parameter counts, then runs `execute`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into; the source runs are looked up in its
            out root.
    """
    experiments = list(config.sources)
    if config.steering_source is not None:
        experiments.append(config.steering_source)
    runs = resolve_runs(
        out_root_of(run_dir), experiments, config.models, config.extra_runs
    )
    execute(
        config,
        run_dir,
        runs,
        {model: parameter_count(model) for model in config.models},
    )


def plot_depth_panels(
    run_dir: RunDir,
    out_dir: Path,
    sources: Sequence[str],
    models: Sequence[str],
    name: str,
) -> list[Path]:
    """Draws the reader curve of every model on the depth axis, per source.

    One panel per source, at most three to a row, each with one curve and band
    per model over the relative depth `layer / L` so that models of different
    depth share an axis. One legend for the whole figure sits above the panels,
    in at most `LEGEND_COLUMNS` columns. Only the run's own
    `curves_<source>_<model>` files are read.

    Args:
        run_dir: A run holding a `curves_<source>_<model>` file for every
            source and model.
        out_dir: Folder of the figure.
        sources: The experiment ids, one panel each, from 1 to `MAX_PANELS`.
        models: The checkpoint keys, one curve each.
        name: File name of the figure without extension.

    Returns:
        The PDF, PNG and SVG paths.

    Raises:
        ValueError: If there are no sources, more than `MAX_PANELS`, or no
            models, or a curves file has no depth.
    """
    if not 1 <= len(sources) <= MAX_PANELS:
        raise ValueError(
            f"sources must hold 1 to {MAX_PANELS} ids, got {list(sources)}"
        )
    if not models:
        raise ValueError("models must hold at least one key")
    columns = min(3, len(sources))
    rows = -(-len(sources) // columns)
    fig, axes = new_figure(rows, columns, width=2 * COLUMN_WIDTH, aspect=0.8)
    with style_context():
        for i, source in enumerate(sources):
            curves = [
                read_curves(run_dir, f"curves_{source}_{model}")
                for model in models
            ]
            if any(curve.n_layers is None for curve in curves):
                raise ValueError(
                    f"the curves of {source} must give the depth of the model"
                )
            ax = axes[i // columns, i % columns]
            plot_curves(
                [
                    (
                        relative_depth(
                            [int(layer) for layer in curve.x],
                            cast(int, curve.n_layers),
                        ),
                        curve.mean[0],
                        curve.lo[0],
                        curve.hi[0],
                    )
                    for curve in curves
                ],
                series=list(models),
                xlabel="Relative depth" if i + columns >= len(sources) else "",
                ylabel="Accuracy" if i % columns == 0 else "",
                chance=curves[0].chance,
                n_layers=1,
                legend=False,
                ax=ax,
            )
            ax.set_title(source)
        for i in range(len(sources), rows * columns):
            fig.delaxes(axes[i // columns, i % columns])
        handles, names = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            names,
            loc="outside upper center",
            ncols=min(len(models), LEGEND_COLUMNS),
        )
    return save_figure(fig, out_dir, name)


def _table(run_dir: RunDir, stem: str) -> list[dict[str, Any]]:
    payload = cast(dict[str, Any], run_dir.read_json(stem))
    return [
        dict(zip(payload["columns"], row, strict=True))
        for row in payload["rows"]
    ]


def _spread(
    mean: Sequence[float], lo: Sequence[float], hi: Sequence[float]
) -> NDArray[np.float64]:
    """Error bar lengths below and above the mean, never negative."""
    middle = np.asarray(mean, dtype=np.float64)
    below = middle - np.asarray(lo, dtype=np.float64)
    above = np.asarray(hi, dtype=np.float64) - middle
    return np.stack([np.maximum(below, 0.0), np.maximum(above, 0.0)])


def _plot_best(
    run_dir: RunDir, out_dir: Path, config: Config, name: str
) -> list[Path]:
    table = _table(run_dir, "table")
    steering = (
        None
        if config.steering_source is None
        else _table(run_dir, "table_steering")
    )
    panels = 2 if steering is None else 3
    fig, axes = new_figure(
        1,
        panels,
        width=2 * COLUMN_WIDTH,
        aspect=0.7 if steering is None else 1.0,
    )
    with style_context():
        accuracy, depth = axes[0, 0], axes[0, 1]
        for source, color in zip(
            config.sources, categorical_colors(len(config.sources)), strict=True
        ):
            mine = [row for row in table if row["source"] == source]
            params = [row["params"] for row in mine]
            best = [row["accuracy"] for row in mine]
            accuracy.errorbar(
                params,
                best,
                yerr=_spread(
                    best,
                    [row["lo"] for row in mine],
                    [row["hi"] for row in mine],
                ),
                color=color,
                marker="o",
                ms=3,
                label=source,
            )
            depth.plot(
                params,
                [row["depth"] for row in mine],
                color=color,
                marker="o",
                ms=3,
                label=source,
            )
        accuracy.set_ylabel("Best accuracy")
        accuracy.legend(
            loc="lower left",
            bbox_to_anchor=(0.0, 1.0),
            borderaxespad=0.2,
            ncols=min(len(config.sources), 3),
            handlelength=1.5,
            columnspacing=1.2,
        )
        depth.set_ylabel("Best relative depth")
        depth.set_ylim(0.0, 1.0)
        if steering is not None:
            slope = [row["slope"] for row in steering]
            axes[0, 2].errorbar(
                [row["params"] for row in steering],
                slope,
                yerr=_spread(
                    slope,
                    [row["slope_min"] for row in steering],
                    [row["slope_max"] for row in steering],
                ),
                color=INK,
                marker="o",
                ms=3,
            )
            axes[0, 2].set_ylabel("Dose slope")
        for ax in axes[0]:
            ax.set_xscale("log")
            ax.set_xlabel("Parameters")
    return save_figure(fig, out_dir, name)


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws the curves by relative depth and the best reading by size.

    `scale_depth` has one panel per source with a curve for every model.
    `scale_best` plots, against the parameter count on a log axis, the best
    accuracy with its interval, the best relative depth, and with a steering
    source the dose slope with the range over its readers.

    Args:
        run_dir: A finished P6 run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths of both figures.
    """
    config = Config.from_dict(run_dir.config)
    return [
        *plot_depth_panels(
            run_dir, out_dir, config.sources, config.models, "scale_depth"
        ),
        *_plot_best(run_dir, out_dir, config, "scale_best"),
    ]
