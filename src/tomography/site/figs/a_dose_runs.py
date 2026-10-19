"""Appendix figure: Dose curves of every knowledge-steering run.

One panel per run on the headline question type, with the encoding of the
dose panels of the main knowledge-steering figure. Panels in the same unit
share their vertical range.
"""

from typing import Any

from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from tomography.site import stats
from tomography.site.figs import _group
from tomography.site.figs import _rows
from tomography.site.figs import knowledge_steering
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import axes
from tomography.viz import paper

HEIGHT = 5.6
KEY_LINE_WIDTH = 1.8
COLUMNS = 3
A7_TAG_KEY = (
    "n of 3: random directions whose gap interval lies above zero; "
    "smallest one-sided p is 1/4"
)
PAD = 0.04
SERIES = ("lat: known", "lat: unknown", "random: unknown")
CLAIM = (
    "Dose curves of every knowledge-steering run on unknown and known "
    "questions."
)
DETAIL = (
    "Mean outcome on unknown questions (circle) and known questions "
    "(left triangle) against the relative coefficient of the knowledge "
    "direction, "
    "with 95 percent intervals that resample the questions, and the mean "
    "over the 3 random directions on unknown questions with its 95 percent "
    "interval that resamples the questions (grey band). Entropy shift is shown "
    "for GPT-2 and base checkpoints and abstention rate for post-trained "
    "ones, as in [[fig:knowledge_steering]]; rows are GPT-2, Qwen3 base and "
    "the post-trained few-shot and concept runs, and panels in the same unit "
    "share their vertical range. The tag in each panel is the outcome of "
    "[[fig:knowledge_steering]]: the number of the 3 random directions the "
    "direction is above. Open markers and dashed lines are post-trained "
    "checkpoints."
)


def _range(runs: list[knowledge_steering.Run]) -> tuple[float, float]:
    low, high = 0.0, 0.0
    for run in runs:
        names, arrays = run.curves()
        for series in SERIES:
            k = names.index(series)
            low = min(low, float(arrays["lo"][k].min()))
            high = max(high, float(arrays["hi"][k].max()))
    pad = PAD * (high - low)
    return low - pad, high + pad


def _row(run: knowledge_steering.Run) -> int:
    role = stats.role(run.model)
    if role == "gpt2":
        return 0
    if role == "base":
        return 1
    return 2 if run.template == "fewshot" else 3


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    runs = sorted(knowledge_steering.load_runs(manifest), key=_row)
    limits = {
        True: _range([r for r in runs if r.entropy]),
        False: _range([r for r in runs if not r.entropy]),
    }
    rows = -(-len(runs) // COLUMNS)
    csv_rows: list[list[Any]] = []

    with paper.paper_style():
        fig = _rows.new_figure(HEIGHT)
        grid = fig.subplots(rows, COLUMNS)
        flat = [ax for line in grid for ax in line]
        for i, (ax, run) in enumerate(zip(flat, runs, strict=False)):
            letter = chr(ord("a") + i)
            knowledge_steering.draw_dose(
                ax, run, letter, csv_rows, axes.model_style(run.model).color
            )
            ax.set_ylim(*limits[run.entropy])
            ax.set_title(
                f"{run.beats} of 3",
                loc="right",
                fontsize=paper.MIN_TEXT,
                color=paper.ENC["secondary_text"],
            )
            paper.panel_title(ax, letter, run.name, paper.MIN_TEXT)
            if i // COLUMNS < rows - 1:
                ax.set_xlabel("")
            first = i % COLUMNS == 0 and (
                i == 0 or runs[i - COLUMNS].entropy != run.entropy
            )
            if not first:
                ax.set_ylabel("")
            if i % COLUMNS:
                ax.tick_params(labelleft=False)
        for ax in flat[len(runs) :]:
            ax.set_axis_off()
        keys = knowledge_steering.direction_keys()
        legend = fig.legend(
            handles=[
                *keys[0],
                Line2D(
                    [],
                    [],
                    color=paper.ENC["random_marker"],
                    lw=KEY_LINE_WIDTH,
                ),
                tuple(
                    Patch(
                        facecolor=axes.model_style(model).color,
                        alpha=0.4,
                        lw=0,
                    )
                    for model in ("gpt2-xl", "qwen3-1.7b")
                ),
                Line2D([], [], ls="none"),
            ],
            labels=[
                *keys[1],
                "mean of 3 random directions",
                "interval of the direction (run colour)",
                A7_TAG_KEY,
            ],
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0.1)},
            loc="outside upper center",
            ncol=1,
            handlelength=1.0,
            columnspacing=1.0,
            fontsize=paper.MIN_TEXT,
        )
        legend.get_frame().set_edgecolor("none")

    numbers = {"runs": float(len(runs))}
    for role in stats.ROLES:
        numbers[f"n_{role}"] = float(
            sum(stats.role(r.model) == role for r in runs)
        )
    return FigureOutput(
        fig,
        CLAIM,
        DETAIL,
        numbers,
        _group.csv_text(knowledge_steering.HEADER, csv_rows),
        {},
    )
