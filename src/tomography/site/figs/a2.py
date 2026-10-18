"""Figure A2: All finders on all models.

Report-split accuracy of the six P4 finders over relative depth, one panel
per checkpoint and task, with the random-direction band.
"""

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from tomography.site import finders
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import paper

BAND = 0.05
STYLES = {
    "pca_diff": (paper.INK, "-", 1.0),
    "pca_raw": ("#666666", "--", 0.8),
    "mean_diff": (paper.SERIES[2], "-", 1.0),
    "logistic": (paper.SERIES[3], "-", 1.0),
    "kmeans": ("#999999", "-.", 0.8),
}
LABELS = {
    "pca_diff": "PCA of differences (paper)",
    "pca_raw": "PCA of raw activations",
    "mean_diff": "mean difference",
    "logistic": "logistic regression",
    "kmeans": "k-means",
}
TASK_NAMES = {"sentiment": "Sentiment", "truth": "Truth", "utility": "Utility"}


def _panel(
    ax: Axes,
    cell: finders.Cell,
    model: str,
    task: str,
    rows: list[list[str]],
) -> None:
    ax.axhline(0.5, color=paper.CHANCE, lw=0.5, ls=(0, (3, 2)), zorder=1)
    mean, lo, hi = cell.curves["random"]
    ax.fill_between(
        cell.depth, lo, hi, color=paper.RANDOM_BAND, lw=0, zorder=1.5
    )
    ax.plot(cell.depth, mean, color=paper.CHANCE, lw=0.6, zorder=2)
    for method, (color, style, width) in STYLES.items():
        values, low, high = cell.curves[method]
        ax.plot(cell.depth, values, color=color, ls=style, lw=width, zorder=3)
        for i, depth in enumerate(cell.depth):
            rows.append(
                [
                    f"{task} {method}",
                    model,
                    f"{depth:.6g}",
                    f"{values[i]:.6g}",
                    f"{low[i]:.6g}",
                    f"{high[i]:.6g}",
                    f"{mean[i]:.6g}",
                ]
            )
    ax.set_xlim(0, 1)
    ax.set_ylim(0.4, 1.0)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels(["0.5", "1"])


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    models = manifest.models("p4")
    loaded = {m: finders.load(manifest, m) for m in models}
    rows: list[list[str]] = []
    cells = raw = kmeans = 0
    with paper.paper_style():
        fig = Figure(
            figsize=(paper.FULL_WIDTH, 1.0 * len(models) + 0.8),
            layout="constrained",
        )
        grid = fig.add_gridspec(len(models), 3)
        for r, model in enumerate(models):
            found = loaded[model]
            if found is None:
                continue
            for c, task in enumerate(finders.TASKS):
                ax = fig.add_subplot(grid[r, c])
                cell = found[task]
                _panel(ax, cell, model, task, rows)
                cells += 1
                ref = cell.rows
                reference = ref["pca_diff"].report
                raw += abs(ref["pca_raw"].report - reference) <= BAND
                kmeans += abs(ref["kmeans"].report - reference) <= BAND
                if r == 0:
                    ax.set_title(TASK_NAMES[task], fontsize=paper.TITLE)
                if c == 0:
                    ax.set_ylabel(
                        paper.display_name(model), fontsize=paper.LABEL
                    )
                if r < len(models) - 1:
                    ax.set_xlabel("")
                else:
                    ax.set_xlabel("relative depth")
        handles: list[Line2D | Patch] = [
            Line2D([], [], color=c, ls=s, lw=w) for c, s, w in STYLES.values()
        ]
        handles.append(Patch(fc=paper.RANDOM_BAND, ec="none"))
        fig.legend(
            handles,
            [*LABELS.values(), "random directions"],
            loc="outside upper center",
            ncol=3,
            frameon=False,
            fontsize=paper.TICK,
            columnspacing=1.0,
        )
    numbers = {
        "cells": float(cells),
        "pca_raw_within": float(raw),
        "kmeans_within": float(kmeans),
    }
    claim = (
        f"At the selected layer, PCA of raw activations is within {BAND:.2f} "
        f"of the paper's PCA finder in {raw} of {cells} task and model cells "
        f"and k-means in {kmeans} of {cells}."
    )
    detail = (
        "Report-split accuracy of the six P4 finders over relative depth for "
        "each checkpoint (rows) and task (columns), on the reference "
        "pairing of the task with centered activations; the grey band is the "
        "interval over seeds of random directions and the dashed line is "
        "chance. The claim compares the report-split accuracy at each "
        "finder's layer chosen on the select split."
    )
    return FigureOutput(fig, claim, detail, numbers, _csv(rows))


def _csv(rows: list[list[str]]) -> str:
    header = "panel,model,x,y,lo,hi,baseline"
    return "\n".join([header, *(",".join(r) for r in rows)]) + "\n"
