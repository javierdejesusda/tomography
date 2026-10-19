"""Figure 4: Prompt wording and reader transfer.

Panels (a) and (b) are the transfer matrices of sentiment readers on one
Qwen3 and one GPT-2 checkpoint: the row is the wording a reader was fitted
on and the column the wording it is evaluated on. Panel (c) shows, for every
P1 checkpoint, the accuracy of the reader fitted on the paper wording on each
other wording.
"""

import json
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np

from tomography.site import store
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import axes as shared
from tomography.viz import paper

DROPPED = "no_trailing_space"
MATRIX_MODEL = "qwen3-1.7b-base"
TASK = "sentiment"


def _table(manifest: Manifest, model: str) -> list[dict[str, Any]]:
    run = manifest.run("p1", model)
    if run is None:
        raise FileNotFoundError(f"results/p1/{model} has no logistic run")
    rows = json.loads((run / "table.json").read_text("utf-8"))["data"]
    return [r for r in rows if r["task"] == TASK and r["variant"] != DROPPED]


def _matrix(manifest: Manifest, model: str) -> tuple[list[str], np.ndarray]:
    run = manifest.run("p1", model)
    if run is None:
        raise FileNotFoundError(f"results/p1/{model} has no logistic run")
    labels = json.loads((run / f"transfer_{TASK}.json").read_text("utf-8"))[
        "data"
    ]
    with np.load(run / f"transfer_{TASK}.npz") as npz:
        values = np.asarray(npz["values"], dtype=float)
    keep = [i for i, v in enumerate(labels["rows"]) if v != DROPPED]
    names = [labels["rows"][i] for i in keep]
    return names, values[np.ix_(keep, keep)]


def _heat(
    ax: Axes, manifest: Manifest, model: str, rows: list[list[str]]
) -> Any:
    names, matrix = _matrix(manifest, model)
    image = shared.heatmap(ax, matrix, vmin=0.5, vmax=1.0)
    n = len(names)
    start = names.index("paper")
    for rect in (
        Rectangle((-0.5, start - 0.5), n, 1, fill=False, ec=paper.SERIES[1]),
        Rectangle((start - 0.5, -0.5), 1, n, fill=False, ec=paper.SERIES[1]),
    ):
        rect.set_linewidth(0.9)
        rect.set_zorder(4)
        ax.add_patch(rect)
    ax.set_xticks([start])
    ax.set_xticklabels(["paper"])
    ax.set_yticks([start])
    ax.set_yticklabels(["paper"])
    ax.set_xlabel("evaluated on wording")
    ax.set_ylabel("fitted on wording")
    ax.set_title(paper.display_name(model), fontsize=paper.TITLE, x=0.12)
    for i, fit in enumerate(names):
        for j, ev in enumerate(names):
            rows.append(
                [
                    f"matrix {fit}",
                    model,
                    ev,
                    f"{matrix[i, j]:.6g}",
                    "",
                    "",
                    "",
                ]
            )
    return image


def _strip(
    ax: Axes, manifest: Manifest, models: list[str], rows: list[list[str]]
) -> dict[str, float]:
    spans: list[float] = []
    cosines: list[float] = []
    ax.set_xlim(0.45, 1.0)
    for y, model in enumerate(models):
        table = _table(manifest, model)
        style = shared.model_style(model)
        random = [r["random"] for r in table]
        ax.barh(
            y,
            max(random) - min(random),
            left=min(random),
            height=0.7,
            color=paper.RANDOM_BAND,
            lw=0,
            zorder=1,
        )
        values = [r for r in table if r["from_paper"] is not None]
        for r in values:
            is_paper = r["variant"] == "paper"
            ax.plot(
                [r["from_paper"]],
                [y],
                ls="none",
                marker="D" if is_paper else style.marker,
                ms=4.5 if is_paper else 3.5,
                mfc=style.color if style.filled or is_paper else "white",
                mec=paper.INK if is_paper else style.color,
                mew=0.8,
                zorder=4 if is_paper else 3,
            )
            rows.append(
                [
                    "strip",
                    model,
                    r["variant"],
                    f"{r['from_paper']:.6g}",
                    "",
                    "",
                    f"{r['random']:.6g}",
                ]
            )
        reach = [r["from_paper"] for r in values]
        span = max(reach) - min(reach)
        cos = [r["cosine_paper"] for r in table if r["variant"] != "paper"]
        spans.append(span)
        cosines += cos
        ax.annotate(
            f"{span:.3f}",
            xy=(1.03, y),
            xycoords=("axes fraction", "data"),
            va="center",
            ha="left",
            fontsize=paper.TICK,
        )
        ax.annotate(
            f"{min(cos):.2f} to {max(cos):.2f}",
            xy=(1.17, y),
            xycoords=("axes fraction", "data"),
            va="center",
            ha="left",
            fontsize=paper.TICK,
        )
    ax.annotate(
        "span",
        xy=(1.03, -0.8),
        xycoords=("axes fraction", "data"),
        va="center",
        ha="left",
        fontsize=paper.TICK,
    )
    ax.annotate(
        "cosine",
        xy=(1.17, -0.8),
        xycoords=("axes fraction", "data"),
        va="center",
        ha="left",
        fontsize=paper.TICK,
    )
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([paper.display_name(m) for m in models])
    ax.set_ylim(len(models) - 0.5, -1.2)
    ax.set_xticks([0.5, 0.75, 1.0])
    ax.set_xticklabels(["0.5", "0.75", "1"])
    ax.set_xlabel("accuracy of the reader fitted on the paper wording")
    return {
        "span_min": min(spans),
        "span_max": max(spans),
        "cosine_min": min(cosines),
        "cosine_max": max(cosines),
    }


def _claim(n: dict[str, float], count: int) -> str:
    return (
        "Changing only the wording of the prompt moves the accuracy of a "
        "sentiment reader fitted on the paper wording by "
        f"{n['span_min']:.3f} to {n['span_max']:.3f} between the best and "
        f"the worst wording on {count} checkpoints, and the directions "
        "fitted on other wordings have a cosine with the paper direction "
        f"from {n['cosine_min']:.2f} to {n['cosine_max']:.2f}."
    )


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    models = [m for m in manifest.models("p1") if manifest.run("p1", m)]
    gpt2 = [m for m in models if m.startswith("gpt2")]
    second = max(gpt2, key=store.model_rank)
    rows: list[list[str]] = []
    with paper.paper_style():
        fig = Figure(figsize=(paper.FULL_WIDTH, 5.0), layout="constrained")
        grid = fig.add_gridspec(
            2, 2, height_ratios=[1.1, 1], width_ratios=[1, 1]
        )
        left = fig.add_subplot(grid[0, 0])
        right = fig.add_subplot(grid[0, 1])
        image = _heat(left, manifest, MATRIX_MODEL, rows)
        _heat(right, manifest, second, rows)
        right.set_ylabel("")
        bar = fig.colorbar(
            image, ax=[left, right], fraction=0.04, pad=0.02, aspect=22
        )
        bar.set_ticks([0.5, 0.75, 1.0])
        bar.set_ticklabels(["0.5", "0.75", "1"])
        bar.set_label("accuracy")
        random = [r["random"] for r in _table(manifest, MATRIX_MODEL)]
        bar.ax.axhspan(
            min(random),
            max(random),
            xmin=0.0,
            xmax=0.45,
            color=paper.CHANCE,
            lw=0,
        )
        strip = fig.add_subplot(grid[1, :])
        numbers = _strip(strip, manifest, models, rows)
        for ax, letter in zip((left, right, strip), "abc", strict=True):
            paper.panel_letter(ax, letter)
    detail = (
        "(a, b) Accuracy of the sentiment reader fitted on the wording in "
        "the row and evaluated on the wording in the column, at the layer "
        "of the paper wording, for two checkpoints (the other checkpoints "
        "are in the appendix); the diagonal is outlined and the paper row "
        "and column in orange. The grey bar on the colour scale spans the "
        f"random-direction accuracy of {paper.display_name(MATRIX_MODEL)} "
        "over the wordings. (c) For each logistic P1 checkpoint, the "
        "accuracy of the reader fitted on the paper wording (diamond) on "
        "each wording (dots); the grey band spans the random-direction "
        "accuracy over the wordings, span is the largest minus the "
        "smallest accuracy, and cosine is the range of the cosine between "
        "the paper direction and the direction fitted on each other "
        "wording. The wording with no trailing space is dropped because "
        "the post-trained models do not have it."
    )
    return FigureOutput(
        fig, _claim(numbers, len(models)), detail, numbers, _csv(rows)
    )


def _csv(rows: list[list[str]]) -> str:
    header = "panel,model,x,y,lo,hi,baseline"
    return "\n".join([header, *(",".join(r) for r in rows)]) + "\n"
