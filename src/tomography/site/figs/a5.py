"""Figure A5: Dose curves of the post-trained P9d runs.

One row per post-trained run and one column per question set: the abstention
rate on known and unknown questions against the relative coefficient of the
knowledge direction (lat) and of random directions of the same norm.
"""

import json
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np

from tomography.site import store
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import paper

TYPES = (
    "obscure_real",
    "fictional_generated",
    "unknowable_detail",
    "false_premise",
)
TEMPLATES = ("fewshot", "concept")
SERIES = {
    "lat: known": (paper.SERIES[0], "-"),
    "lat: unknown": (paper.SERIES[1], "-"),
    "random: known": (paper.CHANCE, "--"),
    "random: unknown": ("#444444", ":"),
}


def _runs(manifest: Manifest) -> list[tuple[str, str, Path]]:
    found = []
    for model in manifest.models("p9d"):
        if not store.is_post_trained(model):
            continue
        for template in TEMPLATES:
            run = manifest.run("p9d", model, template=template)
            if run is not None:
                found.append((model, template, run))
    return found


def _panel(
    ax: Axes,
    run: Path,
    kind: str,
    model: str,
    rows: list[list[str]],
) -> bool:
    meta = json.loads((run / f"curves_{kind}.json").read_text("utf-8"))["data"]
    with np.load(run / f"curves_{kind}.npz") as npz:
        x = np.asarray(npz["x"], dtype=float)
        mean, lo, hi = (np.asarray(npz[k], float) for k in ["mean", "lo", "hi"])
    names = list(meta["series"])
    for name, (color, style) in SERIES.items():
        i = names.index(name)
        ax.fill_between(x, lo[i], hi[i], color=color, alpha=0.15, lw=0)
        ax.plot(x, mean[i], color=color, ls=style, lw=1.0, zorder=3)
        for j, value in enumerate(x):
            rows.append(
                [
                    f"{kind} {name}",
                    model,
                    f"{value:.6g}",
                    f"{mean[i][j]:.6g}",
                    f"{lo[i][j]:.6g}",
                    f"{hi[i][j]:.6g}",
                    "",
                ]
            )
    lat = mean[names.index("lat: unknown")]
    rand = mean[names.index("random: unknown")]
    ax.axvline(0.0, color=paper.CHANCE, lw=0.4, zorder=1)
    return bool(np.ptp(lat) > np.ptp(rand))


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    runs = _runs(manifest)
    rows: list[list[str]] = []
    moved = panels = 0
    with paper.paper_style():
        fig = Figure(
            figsize=(paper.FULL_WIDTH, 1.25 * len(runs) + 0.7),
            layout="constrained",
        )
        grid = fig.add_gridspec(len(runs), len(TYPES))
        for r, (model, template, run) in enumerate(runs):
            for c, kind in enumerate(TYPES):
                ax = fig.add_subplot(grid[r, c])
                moved += _panel(ax, run, kind, model, rows)
                panels += 1
                if r == 0:
                    ax.set_title(
                        kind.replace("_", " "), fontsize=paper.TITLE, x=0.05
                    )
                if c == 0:
                    ax.set_ylabel(
                        f"{paper.display_name(model + ' ' + template)}",
                        fontsize=paper.LABEL,
                    )
                if r == len(runs) - 1:
                    ax.set_xlabel("relative coefficient")
                ax.tick_params(labelsize=paper.TICK)
        handles = [
            Line2D([], [], color=color, ls=style, lw=1.0)
            for color, style in SERIES.values()
        ]
        fig.legend(
            handles,
            list(SERIES),
            loc="outside upper center",
            ncol=4,
            frameon=False,
            fontsize=paper.TICK,
            columnspacing=1.0,
        )
    numbers = {"panels": float(panels), "lat_moves_more": float(moved)}
    claim = (
        f"In {moved} of {panels} panels the knowledge direction moves the "
        "abstention rate of unknown questions over the dose range by more "
        "than random directions of the same norm do."
    )
    detail = (
        "Abstention rate on known and unknown questions against the "
        "relative coefficient added along the knowledge direction (lat) "
        "and along random directions of the same norm, for the post-trained "
        "runs of P9d with the few-shot and the concept template; bands are "
        "the stored intervals. The measure compares the range of the mean "
        "curve of unknown questions over the doses."
    )
    return FigureOutput(fig, claim, detail, numbers, _csv(rows))


def _csv(rows: list[list[str]]) -> str:
    header = "panel,model,x,y,lo,hi,baseline"
    return "\n".join([header, *(",".join(r) for r in rows)]) + "\n"
