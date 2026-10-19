"""Appendix figure: Dose curves of the post-trained runs by question type.

One row per post-trained run and one column per question type, with the
encoding of the dose panels of the main knowledge-steering figure. All
panels share one vertical range.
"""

from typing import Any

from matplotlib.collections import PolyCollection
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
import numpy as np

from tomography.site import runfiles
from tomography.site import stats
from tomography.site.figs import _group
from tomography.site.figs import _rows
from tomography.site.figs import knowledge_steering
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import axes as viz_axes
from tomography.viz import paper

HEIGHT = 7.2
PAD = 0.04
HEADER = ["panel", "run", "type", "x", "series", "value", "lo", "hi"]
SERIES = ("lat: known", "lat: unknown", "random: unknown")
ARMS = ("random0", "random1", "random2")
CLAIM = (
    "The direction moves the abstention rate of unknown questions more than "
    "all 3 random directions in {count} of {panels} panels."
)
FRAME_WIDTH = 1.6
DETAIL = (
    "Abstention rate on unknown questions (circle) and known questions "
    "(left triangle) against the relative coefficient of the knowledge "
    "direction, for each post-trained run (rows, ordered by the number of "
    "framed panels) and question type (columns), with 95 percent intervals "
    "that resample the questions, and the mean over the 3 random directions "
    "on unknown questions with its 95 percent interval that resamples the "
    "questions (grey band). A panel is framed in the colour of its "
    "checkpoint when the interval of the difference in the absolute slope on "
    "unknown questions between the direction and each of the 3 random "
    "directions lies above zero. The test uses each of the 3 random "
    "directions, while the plot shows only their mean, so a panel can look "
    "closer to the grey band than its test is; with 3 random directions a "
    "direction passes by chance with probability at most 1/4. The encoding "
    "is that of [[fig:knowledge_steering]]."
)


def _curves(
    run: knowledge_steering.Run, kind: str
) -> tuple[list[str], dict[str, Any]]:
    names = runfiles.data(run.path, f"curves_{kind}.json")["series"]
    return names, runfiles.arrays(run.path, f"curves_{kind}.npz")


def _moves_more(run: knowledge_steering.Run, kind: str) -> bool:
    found = [
        r
        for r in run.dose
        if r["type"] == kind
        and r["metric"] == "abstention"
        and r["side"] == "unknown"
        and r["arm"] == "lat0"
        and r["vs"] in ARMS
    ]
    return len(found) == len(ARMS) and all(r["abs_lo"] > 0 for r in found)


def _types(run: knowledge_steering.Run) -> list[str]:
    stems = sorted(p.stem for p in run.path.glob("curves_*.json"))
    kinds = [s.removeprefix("curves_") for s in stems]
    return [k for k in kinds if not k.endswith("_entropy")]


def _recolour(ax: Any, color: str) -> None:
    """Gives the panel one grey for random and the run colour to the rest.

    Args:
        ax: A panel drawn by `knowledge_steering.draw_curves`.
        color: The colour of the run's checkpoint.
    """
    for band in ax.collections:
        if not isinstance(band, PolyCollection):
            continue
        if np.allclose(
            np.asarray(band.get_facecolor())[0][:3],
            mcolors.to_rgb(paper.RANDOM_BAND),
        ):
            band.set_facecolor(paper.RANDOM_GREY)
            band.set_alpha(paper.RANDOM_BAND_ALPHA)
        else:
            band.set_facecolor(color)
            band.set_alpha(0.2)


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    runs = [
        r
        for r in knowledge_steering.load_runs(manifest)
        if stats.role(r.model) == "post"
    ]
    kinds = _types(runs[0])
    passes = {
        id(run): [_moves_more(run, kind) for kind in kinds] for run in runs
    }
    runs.sort(key=lambda run: -sum(passes[id(run)]))
    low, high = 0.0, 0.0
    for run in runs:
        for kind in kinds:
            names, arrays = _curves(run, kind)
            for series in SERIES:
                k = names.index(series)
                low = min(low, float(arrays["lo"][k].min()))
                high = max(high, float(arrays["hi"][k].max()))
    pad = PAD * (high - low)
    csv_rows: list[list[Any]] = []
    moved = 0

    with paper.paper_style():
        fig = _rows.new_figure(HEIGHT)
        grid = fig.subplots(len(runs), len(kinds), sharey=True, sharex=True)
        for r, run in enumerate(runs):
            letter = chr(ord("a") + r)
            for c, kind in enumerate(kinds):
                ax = grid[r][c]
                rows: list[list[Any]] = []
                knowledge_steering.draw_curves(
                    ax, run.model, run.label, _curves(run, kind), letter, rows
                )
                csv_rows.extend(
                    [row[0], row[1], kind, *row[2:]] for row in rows
                )
                _recolour(ax, viz_axes.model_style(run.model).color)
                passed = passes[id(run)][c]
                moved += passed
                if passed:
                    ax.add_patch(
                        Rectangle(
                            (0, 0),
                            1,
                            1,
                            transform=ax.transAxes,
                            fill=False,
                            ec=viz_axes.model_style(run.model).color,
                            lw=FRAME_WIDTH,
                            gid="pass-frame",
                            zorder=5,
                            clip_on=False,
                        )
                    )
                ax.set_ylim(low - pad, high + pad)
                ax.set_xlabel(
                    "relative coefficient" if r == len(runs) - 1 else ""
                )
                if c == 0:
                    ax.set_ylabel(run.name.replace(", ", "\n"), fontsize=6.5)
                    paper.panel_title(
                        ax,
                        letter,
                        kind.replace("_", " ") if r == 0 else "",
                        6.5,
                    )
                elif r == 0:
                    ax.set_title(
                        kind.replace("_", " "), loc="left", fontsize=6.5
                    )
        legend = fig.legend(
            handles=[
                _rows.series_handle("unknown questions", "o", paper.INK),
                _rows.series_handle(
                    "known questions",
                    paper.QUESTION_SET_MARKERS["known"],
                    paper.INK,
                ),
                Line2D(
                    [],
                    [],
                    color=paper.ENC["random_marker"],
                    lw=paper.DATA_WIDTH,
                    label="mean of 3 random directions",
                ),
                Patch(
                    facecolor=viz_axes.model_style("qwen3-1.7b").color,
                    alpha=0.2,
                    lw=0,
                    label="interval of the direction",
                ),
                Patch(
                    facecolor=paper.RANDOM_GREY,
                    alpha=paper.RANDOM_BAND_ALPHA,
                    lw=0,
                    label="interval of the random mean",
                ),
                Patch(
                    facecolor="none",
                    edgecolor=paper.INK,
                    lw=FRAME_WIDTH,
                    label="framed: beyond all 3 random directions",
                ),
                _rows.series_handle(
                    "post-trained", "h", paper.INK, filled=False
                ),
            ],
            loc="outside lower center",
            ncol=4,
            handlelength=1.0,
            columnspacing=1.0,
            fontsize=6.5,
        )
        legend.get_frame().set_edgecolor("none")

    panels = len(runs) * len(kinds)
    numbers = {"panels": float(panels), "moves_more": float(moved)}
    return FigureOutput(
        fig,
        CLAIM.format(count=moved, panels=panels),
        DETAIL,
        numbers,
        _group.csv_text(HEADER, csv_rows),
        {},
    )
