"""Figure 6: Dose response of pushing along the knowledge direction.

Dose panels draw the stored curves of the base and GPT-2 runs (entropy shift)
and of the largest post-trained run (abstention rate); the last panel gives
one unit-free ratio per run, the slope of the direction over the largest
absolute slope of the random arms, from the headline of each run.
"""

import math
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np

from tomography.site import params
from tomography.site import runfiles
from tomography.site.figs import _group
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import axes
from tomography.viz import paper

COLUMNS = 3
PANEL_WIDTH = 1.2
PANEL_HEIGHT = 0.95
ROW_PITCH = 1.38
LEFT = 0.62
SUMMARY_LEFT = 2.0
ROW_INCHES = 0.15
ENTROPY_COLOR = paper.SERIES[1]
ABSTENTION_COLOR = paper.SERIES[0]
HEADER = ["panel", "run", "x", "series", "value", "lo", "hi"]
SERIES = ("lat: unknown", "lat: known", "random: unknown", "random: known")
DETAIL = " ".join(
    [
        "(a to g) Change of the outcome while the direction (colored) or a",
        "random direction of the same norm (grey) is added at the stored",
        "layer, against the coefficient relative to the mean residual norm;",
        "solid lines are unknown questions, dashed lines known questions;",
        "bands are 95 percent bootstrap intervals that resample the items.",
        "Base and GPT-2 models show the shift of next-token entropy; the",
        "largest post-trained run shows the abstention rate. The y axes",
        "differ between panels. Random is the pooled series of three random",
        "directions.",
        "(h) One ratio per run of the {total} runs: the slope of the",
        "direction on unknown questions over the largest absolute slope of",
        "the three random arms of the same run, so the unit cancels.",
        "Whiskers scale the 95 percent bootstrap interval of the direction",
        "slope, which resamples the items. Filled markers are runs where the",
        "direction moves the outcome more than every random arm; open",
        "markers are runs where it does not.",
    ]
)


def _runs(manifest: Manifest) -> list[dict[str, Any]]:
    found = []
    for model in manifest.models("p9d"):
        templates = sorted(
            {
                str(r.config["template"])
                for r in manifest.runs("p9d", model)
                if r.config.get("template") is not None
            }
        )
        for template in templates:
            path = manifest.run("p9d", model, template=template)
            if path is None:
                continue
            summary = runfiles.summary(path)
            found.append(
                {
                    "model": model,
                    "template": template,
                    "key": f"{model} {template}",
                    "path": path,
                    "headline": summary["headline"],
                }
            )
    return found


def _dose_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fewshot = [r for r in runs if r["template"] == "fewshot"]
    entropy = [r for r in fewshot if r["headline"]["metric"] == "entropy"]
    post = [r for r in fewshot if r["headline"]["metric"] != "entropy"]
    if not post:
        return entropy
    largest = max(post, key=lambda r: params.param_count(r["model"]))
    return [*entropy, largest]


def _curves(run: dict[str, Any]) -> tuple[list[str], dict[str, np.ndarray]]:
    stem = (
        "curves_obscure_real_entropy"
        if run["headline"]["metric"] == "entropy"
        else "curves_obscure_real"
    )
    names = runfiles.data(run["path"], f"{stem}.json")["series"]
    with np.load(run["path"] / f"{stem}.npz") as arrays:
        return names, {
            k: np.asarray(arrays[k]) for k in ("x", "mean", "lo", "hi")
        }


def _place(
    fig: Figure, left: float, top: float, width: float, height: float
) -> Axes:
    w, h = fig.get_size_inches()
    return fig.add_axes(
        (left / w, 1 - (top + height) / h, width / w, height / h)
    )


def _dose_panel(
    ax: Axes,
    run: dict[str, Any],
    names: list[str],
    curves: dict[str, np.ndarray],
    bottom: bool,
) -> list[list[Any]]:
    color = axes.model_style(run["model"]).color
    metric = run["headline"]["metric"]
    rows: list[list[Any]] = []
    x = curves["x"]
    for name in SERIES:
        index = names.index(name)
        shade = paper.RANDOM_BAND if name.startswith("random") else color
        line = color if name.startswith("lat") else paper.NEUTRALS[1]
        style = "--" if name.endswith(": known") else "-"
        ax.plot(
            x, curves["mean"][index], color=line, ls=style, lw=1.0, zorder=3
        )
        if name.endswith(": unknown"):
            ax.fill_between(
                x,
                curves["lo"][index],
                curves["hi"][index],
                color=shade,
                alpha=0.3 if name.startswith("random") else 0.18,
                lw=0,
                zorder=2,
            )
        for j, value in enumerate(curves["mean"][index]):
            rows.append(
                [
                    run["key"],
                    float(x[j]),
                    name,
                    float(value),
                    float(curves["lo"][index][j]),
                    float(curves["hi"][index][j]),
                ]
            )
    ax.set_xticks([-0.4, 0.0, 0.4])
    ax.set_xticklabels(["-0.4", "0", "0.4"])
    ax.set_xlim(-0.44, 0.44)
    ax.axhline(0.0, color=paper.CHANCE, lw=0.4, zorder=1)
    ax.set_ylabel("entropy shift" if metric == "entropy" else "abstention rate")
    if bottom:
        ax.set_xlabel("relative coefficient")
    ax.set_title(
        paper.display_name(run["model"]), fontsize=8, loc="center", pad=3
    )
    _group.clip_ticks(ax)
    return rows


def _summary_panel(ax: Axes, runs: list[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    count = len(runs)
    for i, run in enumerate(runs):
        head = run["headline"]
        scale = head["max_abs_random_slope"]
        ratio = head["lat_slope"] / scale
        lo, hi = head["lat_lo"] / scale, head["lat_hi"] / scale
        color = (
            ENTROPY_COLOR if head["metric"] == "entropy" else ABSTENTION_COLOR
        )
        y = count - 1 - i
        ax.plot([lo, hi], [y, y], color=color, lw=0.7, zorder=2)
        ax.plot(
            [ratio],
            [y],
            marker="o",
            ms=3.6,
            mfc=color if head["distinguishable"] else "white",
            mec=color,
            mew=0.9,
            ls="none",
            zorder=3,
        )
        rows.append([run["key"], None, "slope_ratio", ratio, lo, hi])
    ax.set_yticks(range(count))
    ax.set_yticklabels(
        [paper.display_name(r["key"]) for r in reversed(runs)], fontsize=7
    )
    ax.set_ylim(-0.7, count - 0.3)
    for x in (-1.0, 1.0):
        ax.axvline(x, color=paper.CHANCE, lw=0.5, ls=(0, (3, 2)), zorder=1)
    ax.axvline(0.0, color=paper.INK, lw=0.5, zorder=1)
    ax.set_xlabel("slope of the direction / largest absolute random slope")
    low = min(r[4] for r in rows)
    ax.set_xlim(math.floor(low) - 1, max(2.0, max(r[5] for r in rows) + 1))
    _group.clip_ticks(ax)
    return rows


def _legend_handles() -> list[Line2D]:
    grey, ink = paper.NEUTRALS[1], paper.INK
    return [
        Line2D([], [], color=ink, lw=1.0, label="direction, unknown"),
        Line2D([], [], color=ink, lw=1.0, ls="--", label="direction, known"),
        Line2D([], [], color=grey, lw=1.0, label="random, unknown"),
        Line2D([], [], color=grey, lw=1.0, ls="--", label="random, known"),
    ]


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    runs = _runs(manifest)
    dose = _dose_runs(runs)
    cells = len(dose) + 1
    grid_rows = math.ceil(cells / COLUMNS)
    summary_height = len(runs) * ROW_INCHES
    summary_top = 0.3 + (grid_rows - 1) * ROW_PITCH + PANEL_HEIGHT + 0.6
    height = summary_top + summary_height + 0.55
    pitch = (paper.FULL_WIDTH - LEFT - 0.1 - PANEL_WIDTH) / (COLUMNS - 1)

    with paper.paper_style():
        fig = Figure(figsize=(paper.FULL_WIDTH, height))
        csv_rows: list[list[Any]] = []
        letters = "abcdefghijklmnopqrstuvwxyz"
        for i, run in enumerate(dose):
            ax = _place(
                fig,
                LEFT + (i % COLUMNS) * pitch,
                0.3 + (i // COLUMNS) * ROW_PITCH,
                PANEL_WIDTH,
                PANEL_HEIGHT,
            )
            names, curves = _curves(run)
            rows = _dose_panel(ax, run, names, curves, i + COLUMNS >= len(dose))
            csv_rows.extend([letters[i], *row] for row in rows)
            paper.panel_letter(ax, letters[i])
        slot = len(dose)
        legend_ax = _place(
            fig,
            LEFT + (slot % COLUMNS) * pitch,
            0.3 + (slot // COLUMNS) * ROW_PITCH,
            PANEL_WIDTH,
            PANEL_HEIGHT,
        )
        legend_ax.axis("off")
        legend_ax.legend(
            handles=_legend_handles(),
            loc="center left",
            bbox_to_anchor=(0.0, 0.5),
            borderaxespad=0.0,
            handlelength=1.8,
        )
        summary = _place(
            fig,
            SUMMARY_LEFT,
            summary_top,
            paper.FULL_WIDTH - SUMMARY_LEFT - 0.15,
            summary_height,
        )
        rows = _summary_panel(summary, runs)
        csv_rows.extend([letters[len(dose)], *row] for row in rows)
        paper.panel_letter(summary, letters[len(dose)])
        summary.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    ls="none",
                    marker="o",
                    ms=3.6,
                    mfc=ENTROPY_COLOR,
                    mec=ENTROPY_COLOR,
                    label="entropy shift",
                ),
                Line2D(
                    [],
                    [],
                    ls="none",
                    marker="o",
                    ms=3.6,
                    mfc=ABSTENTION_COLOR,
                    mec=ABSTENTION_COLOR,
                    label="abstention rate",
                ),
                Line2D(
                    [],
                    [],
                    ls="none",
                    marker="o",
                    ms=3.6,
                    mfc="white",
                    mec=paper.INK,
                    label="not distinguishable",
                ),
            ],
            loc="lower left",
            bbox_to_anchor=(0.0, 1.03),
            ncol=3,
            borderaxespad=0.0,
            columnspacing=1.0,
            handlelength=1.0,
        )

    distinguishable = sum(bool(r["headline"]["distinguishable"]) for r in runs)
    selective = sum(bool(r["headline"]["selective"]) for r in runs)
    failed_post = sum(
        1
        for r in runs
        if not r["headline"]["distinguishable"]
        and _group.role(r["model"]) == "Qwen3 post-trained"
    )
    numbers = {
        "runs": len(runs),
        "distinguishable": distinguishable,
        "selective": selective,
        "not_distinguishable": len(runs) - distinguishable,
        "not_distinguishable_post_trained": failed_post,
    }
    which = (
        f"the {len(runs) - distinguishable} that it does not are all "
        "post-trained"
        if failed_post == len(runs) - distinguishable
        else f"{failed_post} of the {len(runs) - distinguishable} that it "
        "does not are post-trained"
    )
    claim = (
        "Adding the known-versus-unknown direction while the model answers "
        "moves the outcome on unknown questions more than random directions "
        f"of the same norm in {distinguishable} of {len(runs)} runs "
        f"({which}), and moves unknown questions more than known ones in "
        f"{selective} of {len(runs)}."
    )
    return FigureOutput(
        fig,
        claim,
        DETAIL.format(total=len(runs)),
        {k: float(v) for k, v in numbers.items()},
        _group.csv_text(HEADER, csv_rows),
    )
