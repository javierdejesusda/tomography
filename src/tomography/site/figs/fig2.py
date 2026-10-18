"""Figure 2: Where and how well a concept can be read.

Row one shows selectivity (reader accuracy minus a random direction of the
same run) over relative depth for sentiment, morality and truth. Row two
shows accuracy at the selected layer against parameter count for the four
concepts, with the random direction at that layer and the GPT-2 trend that
P6 reports.
"""

import json
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np

from tomography.site import params
from tomography.site import reading
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import axes as shared
from tomography.viz import paper

MARGIN = 0.1
GPT2 = ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")
DEPTH_PANELS = (("r1", "Sentiment"), ("r3", "Morality"), ("r7", "Truth"))
SIZE_PANELS = (
    ("r1", "Sentiment"),
    ("r2", "Utility"),
    ("r3", "Morality"),
    ("r7", "Truth"),
)
P6 = ("p6", "gpt2")


def _slopes(manifest: Manifest) -> dict[str, float]:
    run = manifest.run(*P6)
    if run is None:
        raise FileNotFoundError("results/p6/gpt2 is missing")
    data = json.loads((run / "summary.json").read_text("utf-8"))["data"]
    return {k: float(v) for k, v in data["accuracy_per_decade"].items()}


def _marker_kwargs(model: str) -> dict[str, Any]:
    style = shared.model_style(model)
    return {
        "marker": style.marker,
        "mfc": style.color if style.filled else "white",
        "mec": style.color,
        "color": style.color,
    }


def _depth_panel(
    ax: Axes, manifest: Manifest, exp: str, rows: list[list[str]]
) -> list[float]:
    firsts = []
    ax.axhline(0.0, color=paper.CHANCE, lw=0.5, zorder=1)
    for model in manifest.models(exp):
        found = reading.load(manifest, exp, model)
        if found is None:
            continue
        style = shared.model_style(model)
        ax.plot(
            found.depth,
            found.selectivity,
            color=style.color,
            ls=style.linestyle,
            lw=1.0,
            zorder=3,
        )
        at = found.layer - 1
        ax.plot(
            [found.depth[at]],
            [found.selectivity[at]],
            ls="none",
            ms=3.5,
            zorder=4,
            **_marker_kwargs(model),
        )
        first = reading.first_depth_above(found, MARGIN)
        if first is not None:
            firsts.append(first)
        for depth, value in zip(found.depth, found.selectivity, strict=True):
            rows.append(
                [
                    f"depth {exp}",
                    model,
                    f"{depth:.6g}",
                    f"{value:.6g}",
                    "",
                    "",
                    "0",
                ]
            )
    shared.depth_axis(ax)
    return firsts


def _size_panel(
    ax: Axes,
    manifest: Manifest,
    exp: str,
    slope: float,
    rows: list[list[str]],
) -> list[float]:
    ax.axhline(0.5, color=paper.CHANCE, lw=0.5, zorder=1)
    counts: list[float] = []
    gpt2: list[tuple[float, float]] = []
    accuracies: list[float] = []
    for model in manifest.models(exp):
        found = reading.load(manifest, exp, model)
        if found is None:
            continue
        count = float(params.param_count(model))
        counts.append(count)
        style = shared.model_style(model)
        ax.plot(
            [count, count],
            [found.lo, found.hi],
            color=style.color,
            lw=0.8,
            zorder=3,
        )
        ax.plot(
            [count],
            [found.accuracy],
            ls="none",
            ms=4,
            zorder=4,
            **_marker_kwargs(model),
        )
        ax.plot(
            [count],
            [found.random_at_layer],
            ls="none",
            marker="x",
            ms=4,
            mew=0.8,
            color=paper.CHANCE,
            zorder=2,
        )
        if model in GPT2:
            gpt2.append((np.log10(count), found.accuracy))
        accuracies.append(found.accuracy)
        rows.append(
            [
                f"size {exp}",
                model,
                f"{count:.0f}",
                f"{found.accuracy:.6g}",
                f"{found.lo:.6g}",
                f"{found.hi:.6g}",
                f"{found.random_at_layer:.6g}",
            ]
        )
    x = np.array([g[0] for g in gpt2])
    y = np.array([g[1] for g in gpt2])
    line = 10 ** np.array([x.min(), x.max()])
    fit = y.mean() + slope * (np.array([x.min(), x.max()]) - x.mean())
    ax.plot(line, fit, color=paper.INK, ls="--", lw=0.8, zorder=2)
    shared.size_axis(ax, counts, ticks=(1e8, 1e9))
    ax.set_ylim(0.4, 1.0)
    ax.set_yticks([0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.5", "0.75", "1"])
    return accuracies


def _legend(fig: Figure, models: list[str]) -> None:
    handles = []
    for model in models:
        style = shared.model_style(model)
        handles.append(
            Line2D(
                [],
                [],
                color=style.color,
                ls=style.linestyle,
                lw=1.0,
                ms=4,
                **{
                    k: v
                    for k, v in _marker_kwargs(model).items()
                    if k != "color"
                },
            )
        )
    handles.append(
        Line2D([], [], color=paper.CHANCE, marker="x", ls="none", ms=4, mew=0.8)
    )
    handles.append(Line2D([], [], color=paper.INK, ls="--", lw=0.8))
    labels = [paper.display_name(m) for m in models]
    labels += ["random direction", "GPT-2 trend"]
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=4,
        frameon=False,
        fontsize=paper.TICK,
        handlelength=2.0,
        columnspacing=1.2,
    )


def _claim(numbers: dict[str, float]) -> str:
    n = numbers
    trend = n["slope_r7"]
    fall = "falls" if trend < 0 else "rises"
    return (
        "A sentiment reader beats a random direction by more than "
        f"{MARGIN:.1f} by {n['r1_depth_max']:.2f} of the depth in every "
        f"model, morality needs {n['r3_depth_min']:.2f} to "
        f"{n['r3_depth_max']:.2f} and truth {n['r7_depth_min']:.2f} to "
        f"{n['r7_depth_max']:.2f}; across the GPT-2 sizes accuracy rises by "
        f"{n['slope_r1']:.3f} per tenfold in parameters for sentiment and "
        f"{n['slope_r3']:.3f} for morality, and {fall} by "
        f"{abs(trend):.3f} for truth."
    )


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    slopes = _slopes(manifest)
    rows: list[list[str]] = []
    firsts: dict[str, list[float]] = {}
    models = manifest.models("r1")
    with paper.paper_style():
        fig = Figure(figsize=(paper.FULL_WIDTH, 4.9), layout="constrained")
        grid = fig.add_gridspec(2, 12, height_ratios=[1, 1])
        letters = iter("abcdefg")
        top_axes = []
        for i, (exp, title) in enumerate(DEPTH_PANELS):
            ax = fig.add_subplot(grid[0, 4 * i : 4 * i + 4])
            firsts[exp] = _depth_panel(ax, manifest, exp, rows)
            ax.set_title(title, fontsize=paper.TITLE, x=0.06)
            paper.panel_letter(ax, next(letters))
            top_axes.append(ax)
        top_axes[0].set_ylabel("selectivity")
        low = min(ax.get_ylim()[0] for ax in top_axes)
        high = max(ax.get_ylim()[1] for ax in top_axes)
        for ax in top_axes:
            ax.set_ylim(low, high)
        for i, (exp, title) in enumerate(SIZE_PANELS):
            ax = fig.add_subplot(grid[1, 3 * i : 3 * i + 3])
            _size_panel(ax, manifest, exp, slopes[exp], rows)
            ax.set_title(title, fontsize=paper.TITLE, x=0.06)
            paper.panel_letter(ax, next(letters))
            if i == 0:
                ax.set_ylabel("accuracy")
        _legend(fig, models)
    numbers = {
        "r1_depth_max": max(firsts["r1"]),
        "r3_depth_min": min(firsts["r3"]),
        "r3_depth_max": max(firsts["r3"]),
        "r7_depth_min": min(firsts["r7"]),
        "r7_depth_max": max(firsts["r7"]),
        "slope_r1": slopes["r1"],
        "slope_r3": slopes["r3"],
        "slope_r7": slopes["r7"],
    }
    detail = (
        "Top: selectivity, the reader's accuracy minus a random direction "
        "of the same run at the same layer, against relative depth (layer "
        "over layer count); the dot marks the layer chosen on validation "
        "data and the grey line is zero. "
        "Bottom: accuracy at that layer against parameter count; whiskers "
        "are 95 percent intervals, grey crosses the random direction at the "
        "same layer, the dashed line the least squares trend over the four "
        "GPT-2 sizes with the slope P6 stores. Filled symbols are GPT-2 and "
        "Qwen3 base models, open dashed ones post-trained models; the two "
        "kinds are never pooled."
    )
    return FigureOutput(fig, _claim(numbers), detail, numbers, _csv(rows))


def _csv(rows: list[list[str]]) -> str:
    header = "panel,model,x,y,lo,hi,baseline"
    return "\n".join([header, *(",".join(r) for r in rows)]) + "\n"
