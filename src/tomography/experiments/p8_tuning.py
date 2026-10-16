"""P8: does instruction tuning change the reading, on the Qwen3 pairs.

An aggregate experiment: it loads no model and reads the finished default
runs of the reading experiments on every base model and its post-trained
pair, the P1 runs for the template sensitivity and the C1 runs for the
dose-response half. The seed is the paired unit: seed `i` draws the same
stimuli on a base model and on its tuned pair, so the per-seed difference of
the two readings removes the variation of the stimuli.
"""

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tomography.experiments.aggregate import ReadingPoint
from tomography.experiments.aggregate import finished_run
from tomography.experiments.aggregate import out_root_of
from tomography.experiments.aggregate import read_curves
from tomography.experiments.aggregate import read_table
from tomography.experiments.aggregate import reading_point
from tomography.experiments.aggregate import resolve_runs
from tomography.experiments.aggregate import source_reading_point
from tomography.experiments.base import ExperimentConfig
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.effects import write_dose_curves
from tomography.experiments.matrices import plot_matrix_file
from tomography.experiments.matrices import write_matrix
from tomography.experiments.p6_scale import MAX_MODELS
from tomography.experiments.p6_scale import MAX_PANELS
from tomography.experiments.p6_scale import copy_reader_curves
from tomography.experiments.p6_scale import plot_depth_panels
from tomography.experiments.p6_scale import steering_slopes
from tomography.experiments.sweep import plot_curves_file
from tomography.reading.metrics import bootstrap_ci
from tomography.viz.style import COLUMN_WIDTH
from tomography.viz.style import INK_MUTED
from tomography.viz.style import PX
from tomography.viz.style import categorical_colors
from tomography.viz.style import new_figure
from tomography.viz.style import save_figure
from tomography.viz.style import style_context


@dataclass(frozen=True, kw_only=True)
class Config(ExperimentConfig):
    """P8 settings: which runs to read and how to summarize them.

    Attributes:
        base_models: The base checkpoint keys.
        tuned_models: The post-trained checkpoint keys, paired with
            `base_models` by position. `model` is one of them.
        reference: A pure base model shown beside the pairs, or None.
        concept_sources: The reading experiments of the concept templates.
        function_sources: The reading experiments of the persona (function)
            templates.
        template_source: The experiment whose default runs on every model
            give the template sensitivity, or None for no such part.
        template_task: The task of the template runs that is read.
        steering_source: The steering experiment whose default runs on the
            paired models give the dose response, or None for no such part.
            It is none of the reading sources and not `template_source`, so
            that no run serves two roles.
        extra_runs: Finished runs with other settings; each replaces the
            default run of its own experiment and model.
        n_boot: Bootstrap resamples of every interval.
    """

    experiment: str = "p8"
    model: str = "qwen3-1.7b"
    base_models: tuple[str, ...] = ("qwen3-1.7b-base", "qwen3-0.6b-base")
    tuned_models: tuple[str, ...] = ("qwen3-1.7b", "qwen3-0.6b")
    reference: str | None = "gpt2"
    concept_sources: tuple[str, ...] = ("r1", "r2", "r3", "r7")
    function_sources: tuple[str, ...] = ("r5",)
    template_source: str | None = "p1"
    template_task: str = "sentiment"
    steering_source: str | None = "c1"
    extra_runs: tuple[str, ...] = ()
    n_boot: int = 2000


def paired_difference(
    base: NDArray[np.float64], tuned: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Subtracts the base reading from the tuned one, seed by seed.

    Args:
        base: The per-seed values of the base model, shape (N_base,).
        tuned: The per-seed values of the tuned model, shape (N_tuned,).

    Returns:
        `tuned[:n] - base[:n]` with `n = min(N_base, N_tuned)`.

    Raises:
        ValueError: If an array is not 1-D or has no seed.
    """
    if base.ndim != 1 or tuned.ndim != 1 or base.size == 0 or tuned.size == 0:
        raise ValueError(
            "base and tuned need one seed axis with at least one seed, got "
            f"shapes {base.shape} and {tuned.shape}"
        )
    n = min(base.size, tuned.size)
    return tuned[:n] - base[:n]


def _boot_means(
    values: NDArray[np.float64], n_boot: int, seed: int
) -> NDArray[np.float64]:
    """Means of `n_boot` resamples of the seeds, shape (n_boot,)."""
    n = values.size
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot)
    return np.asarray(draws @ values / n)


def gap_interval(
    diffs: Mapping[str, NDArray[np.float64]],
    function_sources: Sequence[str],
    concept_sources: Sequence[str],
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    """Compares the gain of the function sources with that of the concepts.

    The point is the mean over the function sources of each source's mean
    difference, minus the same mean over the concept sources. The 95%
    percentile interval resamples the seeds of every source on its own, since
    the sources are different data.

    Args:
        diffs: The per-seed paired difference of every source.
        function_sources: The sources of the function templates.
        concept_sources: The sources of the concept templates.
        n_boot: Number of bootstrap resamples.
        seed: Seed of the resampling.

    Returns:
        The point, the lower and the upper bound.

    Raises:
        ValueError: If either list is empty or names a source without a
            difference.
    """
    kinds = {"function": function_sources, "concept": concept_sources}
    for kind, sources in kinds.items():
        if not sources:
            raise ValueError(f"the {kind} sources must not be empty")
        unknown = [source for source in sources if source not in diffs]
        if unknown:
            raise ValueError(f"no difference for the sources {unknown}")
    means = {}
    for kind, sources in kinds.items():
        means[kind] = np.mean(
            [
                _boot_means(diffs[source], n_boot, derive_seed(seed, source))
                for source in sources
            ],
            axis=0,
        )
    point = float(
        np.mean([diffs[s].mean() for s in function_sources])
        - np.mean([diffs[s].mean() for s in concept_sources])
    )
    lo, hi = np.percentile(means["function"] - means["concept"], [2.5, 97.5])
    return point, float(lo), float(hi)


def template_scores(
    run: RunDir, task: str = "sentiment"
) -> dict[str, NDArray[np.float64]]:
    """Scores every template variant of one task of a P1 run.

    Every series of the `curves_<task>` pair is one variant. Each variant is
    read at the layer its own validation curves pick, when the run has them.

    Args:
        run: A finished P1 run.
        task: The task whose curves are read.

    Returns:
        The per-seed reader accuracy of every variant, shape (N,), in the
        order of the series.

    Raises:
        ValueError: If the run has no `curves_<task>` pair or it holds fewer
            than two variants.
    """
    stem = f"curves_{task}"
    if not (run.path / f"{stem}.npz").is_file():
        raise ValueError(f"the run {run.path} has no {stem} curves")
    variants = read_curves(run, stem).series
    if len(variants) < 2:
        raise ValueError(
            f"{stem} of {run.path} needs at least two variants, got {variants}"
        )
    return {
        variant: reading_point(
            run, series=variant, report=stem, select=f"{stem}_val"
        ).values
        for variant in variants
    }


def _sources(config: Config) -> list[str]:
    return [*config.concept_sources, *config.function_sources]


def _pairs(config: Config) -> list[tuple[str, str]]:
    if len(config.base_models) != len(config.tuned_models):
        raise ValueError(
            "base_models and tuned_models must have the same length, got "
            f"{len(config.base_models)} and {len(config.tuned_models)}"
        )
    return list(zip(config.base_models, config.tuned_models, strict=True))


def _paired_models(config: Config) -> list[str]:
    """The pairs in order, base then tuned, without the reference."""
    return [model for pair in _pairs(config) for model in pair]


def _models(config: Config) -> list[str]:
    """The pairs in order, base then tuned, then the reference."""
    models = _paired_models(config)
    if config.reference is not None:
        models.append(config.reference)
    return models


def _check_config(config: Config) -> None:
    """Checks the settings alone, so that no run is looked up for a bad one."""
    _pairs(config)
    if not config.tuned_models:
        raise ValueError("at least one pair of models is needed")
    if config.model not in config.tuned_models:
        raise ValueError(
            f"model {config.model!r} must be one of tuned_models "
            f"{config.tuned_models}"
        )
    for name in ("concept_sources", "function_sources"):
        if not getattr(config, name):
            raise ValueError(f"{name} must not be empty")
    sources, models = _sources(config), _models(config)
    if len(sources) > MAX_PANELS:
        raise ValueError(
            f"at most {MAX_PANELS} sources are drawn, got {len(sources)}"
        )
    if len(set(sources)) != len(sources):
        raise ValueError(f"each source may appear only once, got {sources}")
    steering = config.steering_source
    if steering is not None and steering in (*sources, config.template_source):
        raise ValueError(
            f"steering_source {steering!r} must not also be a reading or "
            f"template source, got sources {sources} and template_source "
            f"{config.template_source!r}"
        )
    if len(models) > MAX_MODELS:
        raise ValueError(
            f"at most {MAX_MODELS} models are drawn, got {len(models)}"
        )
    if len(set(models)) != len(models):
        raise ValueError(f"each model may appear only once, got {models}")


def _check(config: Config, runs: Mapping[tuple[str, str], RunDir]) -> None:
    """Checks the config and the inputs before anything is read."""
    _check_config(config)
    sources, models = _sources(config), _models(config)
    needed = [(source, model) for source in sources for model in models]
    if config.template_source is not None:
        needed += [(config.template_source, model) for model in models]
    if config.steering_source is not None:
        needed += [
            (config.steering_source, model) for model in _paired_models(config)
        ]
    lacking = [pair for pair in needed if pair not in runs]
    if lacking:
        raise ValueError(f"runs lacks the pairs {lacking}")


def _read_sources(
    config: Config,
    run_dir: RunDir,
    runs: Mapping[tuple[str, str], RunDir],
) -> dict[tuple[str, str], ReadingPoint]:
    """Copies the reader curves and picks the layer of every run.

    R5 reads its transfer test, as R5 itself does, because its in-template
    test is not a held-out measure.
    """
    points: dict[tuple[str, str], ReadingPoint] = {}
    for source in _sources(config):
        for model in _models(config):
            run = runs[(source, model)]
            copy_reader_curves(
                run_dir,
                source,
                model,
                run,
                n_boot=config.n_boot,
                seed=config.seed,
            )
            points[(source, model)] = source_reading_point(run, source)
    return points


def _pair_rows(
    config: Config,
    points: Mapping[tuple[str, str], ReadingPoint],
) -> tuple[list[list[object]], dict[str, dict[str, NDArray[np.float64]]]]:
    """Builds the rows of `table` and the paired differences behind them."""
    rows: list[list[object]] = []
    diffs: dict[str, dict[str, NDArray[np.float64]]] = {}
    for base, tuned in _pairs(config):
        diffs[tuned] = {}
        for source in _sources(config):
            before, after = points[(source, base)], points[(source, tuned)]
            diff = paired_difference(before.values, after.values)
            diffs[tuned][source] = diff
            lo, hi = bootstrap_ci(
                diff,
                n_boot=config.n_boot,
                seed=derive_seed(config.seed, "delta", source, tuned),
            )
            selected = {before.selected_on, after.selected_on}
            rows.append(
                [
                    source,
                    "function"
                    if source in config.function_sources
                    else "concept",
                    base,
                    tuned,
                    float(before.values.mean()),
                    float(after.values.mean()),
                    float(diff.mean()),
                    float(lo),
                    float(hi),
                    before.layer / before.n_layers,
                    after.layer / after.n_layers,
                    (
                        before.selected_on
                        if len(selected) == 1
                        else f"{before.selected_on}/{after.selected_on}"
                    ),
                ]
            )
    return rows, diffs


def _reference_rows(
    config: Config,
    points: Mapping[tuple[str, str], ReadingPoint],
    reference: str,
) -> list[list[object]]:
    rows: list[list[object]] = []
    for source in _sources(config):
        point = points[(source, reference)]
        lo, hi = bootstrap_ci(
            point.values,
            n_boot=config.n_boot,
            seed=derive_seed(config.seed, "reference", source, reference),
        )
        rows.append(
            [
                source,
                reference,
                float(point.values.mean()),
                float(lo),
                float(hi),
                point.layer / point.n_layers,
            ]
        )
    return rows


def _mean_interval(
    values: NDArray[np.float64], n_boot: int, seed: int
) -> dict[str, float]:
    """The mean of per-seed values with its percentile bootstrap interval."""
    lo, hi = bootstrap_ci(values, n_boot=n_boot, seed=seed)
    return {"mean": float(values.mean()), "lo": float(lo), "hi": float(hi)}


def _template_part(
    config: Config,
    run_dir: RunDir,
    runs: Mapping[tuple[str, str], RunDir],
    template_source: str,
) -> dict[str, Any]:
    """Writes `matrix_templates` and returns the template summary entries."""
    models = _models(config)
    scores = {
        model: template_scores(
            runs[(template_source, model)], config.template_task
        )
        for model in models
    }
    shared = [
        variant
        for variant in scores[models[0]]
        if all(variant in scores[model] for model in models)
    ]
    if len(shared) < 2:
        raise ValueError(
            "fewer than two template variants are shared by every model, "
            f"got {shared}"
        )
    write_matrix(
        run_dir,
        "matrix_templates",
        np.array(
            [[scores[model][v].mean() for v in shared] for model in models]
        ),
        rows=models,
        cols=shared,
        title=f"{config.template_task} accuracy by template",
    )
    spread = {
        model: np.stack([scores[model][v] for v in shared]).std(axis=0)
        for model in models
    }
    return {
        "template_variants": shared,
        "template_dropped": {
            model: [v for v in scores[model] if v not in shared]
            for model in models
        },
        "template_sd": {
            model: _mean_interval(
                spread[model],
                config.n_boot,
                derive_seed(config.seed, "sd", model),
            )
            for model in models
        },
        "template_sd_delta": {
            tuned: _mean_interval(
                paired_difference(spread[base], spread[tuned]),
                config.n_boot,
                derive_seed(config.seed, "sd_delta", tuned),
            )
            for base, tuned in _pairs(config)
        },
    }


def _linear_readers(
    run: RunDir,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The coefficients and the `linear` curve of every reader of a C1 run."""
    curves = read_curves(run, "curves_operators")
    return curves.x, curves.per_seed[curves.series.index("linear")]


def _dose_part(
    config: Config,
    run_dir: RunDir,
    runs: Mapping[tuple[str, str], RunDir],
    steering_source: str,
) -> dict[str, Any]:
    """Writes `curves_dose_<tuned>` and returns the slope summary entry."""
    slopes: dict[str, dict[str, Any]] = {}
    for base, tuned in _pairs(config):
        before, after = (
            runs[(steering_source, base)],
            runs[(steering_source, tuned)],
        )
        slopes_before, slopes_after = (
            steering_slopes(before),
            steering_slopes(after),
        )
        coeffs, readers_before = _linear_readers(before)
        coeffs_after, readers_after = _linear_readers(after)
        if not np.array_equal(coeffs, coeffs_after):
            raise ValueError(
                f"the {steering_source} runs of {base} and {tuned} use "
                "different coefficient grids"
            )
        n = min(len(readers_before), len(readers_after))
        write_dose_curves(
            run_dir,
            np.stack([readers_before[:n], readers_after[:n]]),
            coeffs.tolist(),
            series=[base, tuned],
            stem=f"curves_dose_{tuned}",
            xlabel="Relative coefficient",
            ylabel="Score",
            n_boot=config.n_boot,
            seed=derive_seed(config.seed, "dose", tuned),
        )
        slopes[tuned] = {
            "base": slopes_before.tolist(),
            "tuned": slopes_after.tolist(),
            "delta": float(
                paired_difference(slopes_before, slopes_after).mean()
            ),
        }
    return {"slopes": slopes}


def execute(
    config: Config,
    run_dir: RunDir,
    runs: Mapping[tuple[str, str], RunDir],
) -> None:
    """Compares every base model with its tuned pair.

    Writes `curves_<source>_<model>` for every source and model and `table`
    with one row per pair and source: the best accuracy of the base and the
    tuned model, their mean per-seed difference with its percentile bootstrap
    interval, and the layers as depths. The reference gets `table_reference`.
    R5 reports its transfer test, as R5 itself does, because its in-template
    test is not a held-out measure.
    `summary` holds the `function_gap` of every tuned model, the mean gain of
    the function sources minus that of the concept sources.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        runs: The finished run of every (experiment, model) pair of the
            sources, of the template source on every model, and of the
            steering source on the paired models.

    Raises:
        ValueError: If `base_models` and `tuned_models` are not equal in
            length and at least one, `model` is not a tuned model, the models
            (with the reference) are not at most `MAX_MODELS` and distinct,
            the sources are not distinct, at most `MAX_PANELS` and of both
            kinds, the steering source is also a reading or template source,
            a run is missing, or a run cannot be read. The first checks run
            before any file is read.
    """
    _check(config, runs)
    points = _read_sources(config, run_dir, runs)
    rows, diffs = _pair_rows(config, points)
    run_dir.write_json(
        "table",
        {
            "columns": [
                "source",
                "kind",
                "base",
                "tuned",
                "base_accuracy",
                "tuned_accuracy",
                "delta",
                "lo",
                "hi",
                "base_depth",
                "tuned_depth",
                "selected_on",
            ],
            "rows": rows,
        },
    )
    if config.reference is not None:
        run_dir.write_json(
            "table_reference",
            {
                "columns": ["source", "model", "accuracy", "lo", "hi", "depth"],
                "rows": _reference_rows(config, points, config.reference),
            },
        )
    function_gap = {}
    for _, tuned in _pairs(config):
        point, lo, hi = gap_interval(
            diffs[tuned],
            config.function_sources,
            config.concept_sources,
            n_boot=config.n_boot,
            seed=derive_seed(config.seed, "gap"),
        )
        function_gap[tuned] = {"point": point, "lo": lo, "hi": hi}
    summary: dict[str, Any] = {
        "pairs": [list(pair) for pair in _pairs(config)],
        "sources": {
            "concept": list(config.concept_sources),
            "function": list(config.function_sources),
        },
        "models": _models(config),
        "function_gap": function_gap,
    }
    if config.template_source is not None:
        summary.update(
            _template_part(config, run_dir, runs, config.template_source)
        )
    if config.steering_source is not None:
        summary.update(
            _dose_part(config, run_dir, runs, config.steering_source)
        )
    run_dir.write_json("summary", summary)


def _steering_extra_runs(config: Config) -> list[str]:
    """The extra runs that belong to the steering source."""
    if config.steering_source is None:
        return []
    return [
        path
        for path in config.extra_runs
        if finished_run(path).config["experiment"] == config.steering_source
    ]


def run(config: Config, run_dir: RunDir) -> None:
    """Finds the source runs, then runs `execute`.

    The sources and the template source are looked up on every model, the
    reference included, and the steering source on the paired models only,
    since the reference is not steered. Each extra run goes to the lookup of
    its own experiment.

    Args:
        config: The experiment settings.
        run_dir: The run to write into; the source runs are looked up in its
            out root.

    Raises:
        ValueError: If the settings are not valid. This is checked before any
            run is looked up.
    """
    _check_config(config)
    out_root = out_root_of(run_dir)
    experiments = _sources(config)
    if config.template_source is not None:
        experiments.append(config.template_source)
    steering_extras = _steering_extra_runs(config)
    runs = resolve_runs(
        out_root,
        experiments,
        _models(config),
        [path for path in config.extra_runs if path not in steering_extras],
    )
    if config.steering_source is not None:
        runs.update(
            resolve_runs(
                out_root,
                [config.steering_source],
                _paired_models(config),
                steering_extras,
            )
        )
    execute(config, run_dir, runs)


def _plot_pairs(run_dir: RunDir, out_dir: Path, config: Config) -> list[Path]:
    """Joins the base and tuned best accuracy of every source, per pair."""
    table = read_table(run_dir, "table")
    pairs, sources = _pairs(config), _sources(config)
    columns = min(2, len(pairs))
    rows = -(-len(pairs) // columns)
    fig, axes = new_figure(
        rows, columns, width=2 * COLUMN_WIDTH, aspect=0.7, sharey=True
    )
    labels = [
        f"{source}\n(function)" if source in config.function_sources else source
        for source in sources
    ]
    base_color, tuned_color = categorical_colors(2)
    x = np.arange(len(sources))
    with style_context():
        for i, (base, tuned) in enumerate(pairs):
            ax = axes[i // columns, i % columns]
            mine = {
                row["source"]: row for row in table if row["tuned"] == tuned
            }
            before = [mine[source]["base_accuracy"] for source in sources]
            after = [mine[source]["tuned_accuracy"] for source in sources]
            for k, (low, high) in enumerate(zip(before, after, strict=True)):
                ax.plot(
                    [k, k], [low, high], color=INK_MUTED, lw=PX, label="_link"
                )
            ax.plot(
                x,
                before,
                ls="",
                marker="o",
                color=base_color,
                markersize=8,
                markerfacecolor="white",
                label="base",
            )
            ax.plot(
                x,
                after,
                ls="",
                marker="o",
                markersize=4,
                color=tuned_color,
                label="tuned",
            )
            ax.set_xticks(x, labels)
            ax.set_title(f"{base} vs {tuned}")
            if i % columns == 0:
                ax.set_ylabel("Best accuracy")
        for i in range(len(pairs), rows * columns):
            fig.delaxes(axes[i // columns, i % columns])
        handles, names = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, names, loc="outside upper center", ncols=2)
    return save_figure(fig, out_dir, "tuning_pairs")


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws the curves, the pairs, the templates and the dose responses.

    `tuning_depth` has one panel per source with a curve for every model.
    `tuning_pairs` has one panel per pair with the base and tuned best
    accuracy of every source joined by a line, the function sources labeled.
    The template matrix and the dose curves of every tuned model are drawn
    when the run has them.

    Args:
        run_dir: A finished P8 run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths of every figure.
    """
    config = Config.from_dict(run_dir.config)
    paths = [
        *plot_depth_panels(
            run_dir, out_dir, _sources(config), _models(config), "tuning_depth"
        ),
        *_plot_pairs(run_dir, out_dir, config),
    ]
    if (run_dir.path / "matrix_templates.npz").is_file():
        paths += plot_matrix_file(
            run_dir,
            out_dir,
            stem="matrix_templates",
            kind="sequential",
            vmin=0.0,
            vmax=1.0,
        )
    for _, tuned in _pairs(config):
        stem = f"curves_dose_{tuned}"
        if (run_dir.path / f"{stem}.npz").is_file():
            paths += plot_curves_file(run_dir, out_dir, stem=stem)
    return paths
