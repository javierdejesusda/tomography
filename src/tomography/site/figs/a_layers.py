"""Figure A1: Layer curves of every concept and model.

One panel per concept and model family. Each model draws the reader, a
random direction of the same run and a shuffled-label reader over relative
depth, in raw accuracy.
"""

from collections.abc import Callable
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from tomography.site import reading
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import axes as shared
from tomography.viz import paper

MARGIN = 0.1
CONCEPTS = (
    ("r1", "Sentiment"),
    ("r2", "Utility"),
    ("r3", "Morality"),
    ("r7", "Truth"),
)
GROUPS: tuple[tuple[str, Callable[[str], bool]], ...] = (
    ("GPT-2", lambda m: m.startswith("gpt2")),
    ("Qwen3 base", lambda m: m.startswith("qwen3") and m.endswith("-base")),
    (
        "Qwen3 post-trained",
        lambda m: m.startswith("qwen3") and not m.endswith("-base"),
    ),
)


def _panel(
    ax: Axes,
    manifest: Manifest,
    exp: str,
    models: list[str],
    rows: list[list[str]],
) -> list[reading.Reading]:
    ax.axhline(0.5, color=paper.CHANCE, lw=0.5, ls=(0, (3, 2)), zorder=1)
    found = []
    for model in models:
        item = reading.load(manifest, exp, model)
        if item is None:
            continue
        found.append(item)
        style = shared.model_style(model)
        drawn: tuple[tuple[str, Any, dict[str, Any]], ...] = (
            ("reader", item.reader, {"lw": 1.0, "ls": style.linestyle}),
            ("random", item.random, {"lw": 0.6, "ls": ":", "alpha": 0.9}),
            ("shuffled", item.shuffled, {"lw": 0.6, "ls": "-.", "alpha": 0.6}),
        )
        for name, values, kwargs in drawn:
            ax.plot(item.depth, values, color=style.color, zorder=3, **kwargs)
            for depth, value in zip(item.depth, values, strict=True):
                rows.append(
                    [
                        f"{exp} {name}",
                        model,
                        f"{depth:.6g}",
                        f"{value:.6g}",
                        "",
                        "",
                        "0.5",
                    ]
                )
    shared.depth_axis(ax)
    ax.set_ylim(0.3, 1.0)
    ax.set_yticks([0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.5", "0.75", "1"])
    return found


def _legend(fig: Figure) -> None:
    handles = [
        Line2D([], [], color=paper.INK, lw=1.0),
        Line2D([], [], color=paper.INK, lw=0.6, ls=":"),
        Line2D([], [], color=paper.INK, lw=0.6, ls="-."),
    ]
    fig.legend(
        handles,
        ["reader", "random direction", "shuffled labels"],
        loc="outside upper center",
        ncol=3,
        frameon=False,
        fontsize=paper.TICK,
    )


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    rows: list[list[str]] = []
    cells = above = 0
    worst = 0.0
    with paper.paper_style():
        fig = Figure(figsize=(paper.FULL_WIDTH, 6.6), layout="constrained")
        grid = fig.add_gridspec(len(CONCEPTS), len(GROUPS))
        letters = iter("abcdefghijkl")
        for r, (exp, title) in enumerate(CONCEPTS):
            for c, (group, member) in enumerate(GROUPS):
                ax = fig.add_subplot(grid[r, c])
                models = [m for m in manifest.models(exp) if member(m)]
                for item in _panel(ax, manifest, exp, models, rows):
                    cells += 1
                    above += (item.reader[-1] - item.random[-1]) > MARGIN
                    worst = max(worst, float(abs(item.shuffled - 0.5).max()))
                paper.panel_letter(ax, next(letters))
                if r == 0:
                    ax.set_title(group, fontsize=paper.TITLE, x=0.12)
                if c == 0:
                    ax.set_ylabel(f"{title} accuracy")
                if r < len(CONCEPTS) - 1:
                    ax.set_xlabel("")
        _legend(fig)
    numbers = {
        "cells": float(cells),
        "above_at_last_layer": float(above),
        "shuffled_max_gap": worst,
    }
    claim = (
        f"At the last layer the reader is more than {MARGIN:.1f} above its "
        f"random direction in {above} of {cells} concept and model cells, and "
        f"a shuffled-label reader never leaves {worst:.3f} of chance."
    )
    detail = (
        "Raw accuracy of the reader (solid, dashed for post-trained models), "
        "of a random direction of the same run (dotted) and of a "
        "shuffled-label reader (dash-dot) at every layer, against relative "
        "depth; colors follow model size within a family and the dashed "
        "line is chance. Readers are logistic regressions on the test pairs "
        "of each concept, averaged over seeds."
    )
    return FigureOutput(fig, claim, detail, numbers, _csv(rows))


def _csv(rows: list[list[str]]) -> str:
    header = "panel,model,x,y,lo,hi,baseline"
    return "\n".join([header, *(",".join(r) for r in rows)]) + "\n"
