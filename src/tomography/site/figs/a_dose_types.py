"""Figure A7: Dose curves of the post-trained runs for each question type.

One row per post-trained run and one column per question type, with the
encoding of the dose panels of Figure 9. All panels share one vertical range.
"""

from typing import Any

from matplotlib.lines import Line2D

from tomography.site import runfiles
from tomography.site import stats
from tomography.site.figs import _group
from tomography.site.figs import _rows
from tomography.site.figs import knowledge_steering
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
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
DETAIL = (
    "Abstention rate on unknown questions (circle) and known questions "
    "(triangle) against the relative coefficient of the knowledge "
    "direction, for each post-trained run (rows) and question type "
    "(columns), with 95 percent intervals that resample the questions, and "
    "the mean over the 3 random directions on unknown questions with its "
    "95 percent interval that resamples the questions (grey band). A panel "
    "counts when the interval of the difference in the absolute slope on "
    "unknown questions between the direction and each of the 3 random "
    "directions lies above zero. The encoding is that of Figure 9."
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
                moved += _moves_more(run, kind)
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
                _rows.series_handle("known questions", "^", paper.INK),
                Line2D(
                    [],
                    [],
                    color=paper.ENC["random_marker"],
                    lw=paper.DATA_WIDTH,
                    label="mean of 3 random directions",
                ),
                _rows.series_handle(
                    "post-trained", "o", paper.INK, filled=False
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
