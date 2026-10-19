"""Figure A1: Reader accuracy at every layer of every concept and checkpoint.

One panel per concept (row) and family (column). Each checkpoint draws its
reader over relative depth; one grey band per panel spans the random
directions of the panel's checkpoints.
"""

import re

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

from tomography.site import reading
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import axes as shared
from tomography.viz import paper

HEIGHT = 6.0
CONCEPTS = (
    ("r1", "Sentiment"),
    ("r2", "Utility"),
    ("r3", "Morality"),
    ("r7", "Truth"),
)
GROUPS = (
    ("gpt2", "GPT-2"),
    ("base", "Qwen3 base"),
    ("post", "Qwen3 post-trained"),
)
GRID = np.linspace(0.0, 1.0, 101)
GPT2_LABELS = {
    "gpt2": "S",
    "gpt2-medium": "M",
    "gpt2-large": "L",
    "gpt2-xl": "XL",
}
XLIM = (0.0, 1.24)
LABEL_X = 1.02
LABEL_GAP = 0.07
YLIM = (0.3, 1.0)
TEXT = 6.5


def _label(model: str) -> str:
    if model in GPT2_LABELS:
        return GPT2_LABELS[model]
    match = re.match(r"qwen3-(\d+(?:\.\d+)?)b", model)
    assert match is not None, model
    return f"{match.group(1)}B"


def _spread(values: list[float]) -> list[float]:
    """Moves label heights apart so that none is closer than the gap."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = [0.0] * len(values)
    previous = -np.inf
    for index in order:
        previous = max(values[index], previous + LABEL_GAP)
        placed[index] = previous
    return placed


def _panel(
    ax: Axes,
    found: dict[str, reading.Reading],
    name: str,
    group: str,
    rows: list[list[str]],
) -> None:
    ax.axhline(
        0.5,
        color=paper.ENC["reference"],
        linestyle=paper.LINESTYLES["reference"],
        linewidth=paper.REFERENCE_WIDTH,
        zorder=1,
    )
    panel = f"{name.lower()} {group}"
    randoms = np.array(
        [np.interp(GRID, r.depth, r.random) for r in found.values()]
    )
    low, high = randoms.min(axis=0), randoms.max(axis=0)
    ax.fill_between(GRID, low, high, color=paper.RANDOM_BAND, lw=0, zorder=1.5)
    for series, values in (("band_lo", low), ("band_hi", high)):
        rows.extend(
            [panel, name.lower(), group, "", series, f"{x:.6g}", f"{v:.6g}"]
            for x, v in zip(GRID, values, strict=True)
        )
    ends = []
    for model, item in found.items():
        style = shared.model_style(model)
        ax.plot(
            item.depth,
            item.reader,
            color=style.color,
            ls=style.linestyle,
            lw=1.0,
            zorder=3,
        )
        ends.append(float(item.reader[-1]))
        for series, values in (
            ("reader", item.reader),
            ("random", item.random),
        ):
            rows.extend(
                [
                    panel,
                    name.lower(),
                    group,
                    model,
                    series,
                    f"{x:.6g}",
                    f"{v:.6g}",
                ]
                for x, v in zip(item.depth, values, strict=True)
            )
    for model, y in zip(found, _spread(ends), strict=True):
        ax.text(
            LABEL_X,
            y,
            _label(model),
            fontsize=TEXT,
            va="center",
            color=paper.INK,
        )
    ax.set_xlim(*XLIM)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_ylim(*YLIM)
    ax.set_yticks([0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.5", "0.75", "1"])


def _csv(rows: list[list[str]]) -> str:
    header = "panel,concept,group,model,series,depth,value"
    return "\n".join([header, *(",".join(r) for r in rows)]) + "\n"


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    rows: list[list[str]] = []
    with paper.paper_style():
        fig = Figure(figsize=(paper.FULL_WIDTH, HEIGHT), layout="constrained")
        grid = fig.add_gridspec(len(CONCEPTS), len(GROUPS))
        for r, (exp, title) in enumerate(CONCEPTS):
            for c, (group, group_title) in enumerate(GROUPS):
                ax = fig.add_subplot(grid[r, c])
                found = {}
                for model in manifest.models(exp):
                    if stats.role(model) != group:
                        continue
                    item = reading.load(manifest, exp, model)
                    if item is not None:
                        found[model] = item
                _panel(ax, found, title, group, rows)
                if r == 0:
                    ax.set_title(group_title, fontsize=paper.TITLE, x=0.1)
                if r == len(CONCEPTS) - 1:
                    ax.set_xlabel("relative depth")
                if c == 0:
                    ax.set_ylabel(f"{title} accuracy")
                    paper.panel_letter(ax, "abcd"[r])
                else:
                    ax.tick_params(labelleft=False)
    claim = (
        "Reader accuracy at every layer for every concept and checkpoint, "
        "against the range of the random directions of the same panel "
        "(grey)."
    )
    detail = (
        "Sentiment curves are scored on the 256 SST-2 validation pairs on "
        "which the sentiment layer is also chosen. Lines are the mean over "
        "seeds of a logistic regression reader; colour darkens with model "
        "size within a family, post-trained checkpoints are dashed, labels "
        "at the right name the size, and the dotted line is chance."
    )
    return FigureOutput(fig, claim, detail, {}, _csv(rows))
