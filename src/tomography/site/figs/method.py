"""Figure 2: the recipe, from a pair of inputs to a reading and a steering.

Five stages from left to right joined by straight arrows: a pair of inputs,
the residual stream at the last token, a finder that turns activations into a
unit direction, a pairwise reading, and steering by addition. A random unit
direction is drawn as the control of the reading. The figure reads no results.
"""

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import Rectangle
import numpy as np

from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import paper

WIDTH = 5.5
HEIGHT = 2.1
TEXT = paper.MIN_TEXT
TITLE_SIZE = 7.5
EQUATION_SIZE = 7.5
ELL = chr(0x2113)
STEER = r"$h'_\ell = h_\ell + c\,r_\ell\,v_\ell$"
INK = paper.INK
GREY = paper.ENC["secondary_text"]
TITLE_Y = 0.97
BODY_TOP = 0.74
ARROWS = ((0.17, 0.20), (0.30, 0.39), (0.595, 0.62), (0.795, 0.82))


def _title(ax: Axes, x: float, text: str) -> None:
    ax.text(
        x,
        TITLE_Y,
        text,
        ha="left",
        va="top",
        fontsize=TITLE_SIZE,
        color=INK,
        linespacing=1.2,
    )


def _text(
    ax: Axes,
    x: float,
    y: float,
    text: str,
    *,
    color: str = INK,
    ha: str = "left",
    size: float = TEXT,
) -> None:
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va="top",
        fontsize=size,
        color=color,
        linespacing=1.3,
    )


def _arrow(ax: Axes, x0: float, x1: float, y: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y),
            (x1, y),
            arrowstyle="-|>",
            mutation_scale=7,
            lw=0.7,
            color=INK,
            shrinkA=0,
            shrinkB=0,
        )
    )


def _pair(ax: Axes) -> None:
    _title(ax, 0.01, "pair")
    ax.add_patch(
        FancyBboxPatch(
            (0.01, 0.40),
            0.155,
            0.30,
            boxstyle="round,pad=0,rounding_size=0.012",
            fill=False,
            ec=INK,
            lw=0.6,
        )
    )
    _text(ax, 0.02, 0.66, "a positive input")
    _text(ax, 0.02, 0.52, "a negative input")
    ax.plot(
        [0.0875], [0.372], marker="^", ms=3.5, mew=0, color=INK, clip_on=False
    )
    _text(ax, 0.0875, 0.345, "last token", ha="center", color=GREY)


def _stream(ax: Axes) -> None:
    _title(ax, 0.205, "residual stream\nat layer $\\ell$")
    for i in range(5):
        y = BODY_TOP - 0.10 * (i + 1)
        cell = (0.225, y), 0.07, 0.10
        if i == 4:
            ax.add_patch(Rectangle(*cell, fc=INK, ec="none", lw=0))
        ax.add_patch(Rectangle(*cell, fill=False, ec=INK, lw=0.6))
    _text(ax, 0.26, 0.20, "last token", ha="center", color=GREY)


def _finder(ax: Axes) -> None:
    _title(ax, 0.395, "find a unit direction")
    lines = (
        (0.66, paper.ENC["paper"], "PCA of paired\ndifferences (paper)"),
        (0.46, INK, "logistic regression"),
        (
            0.30,
            paper.ENC["random_marker"],
            "random unit\ndirection $u$\n(control)",
        ),
    )
    for y, color, text in lines:
        ax.plot(
            [0.405],
            [y - 0.025],
            ls="none",
            marker="s",
            ms=4,
            mew=0,
            color=color,
        )
        _text(ax, 0.42, y, text)


def _read(ax: Axes) -> None:
    _title(ax, 0.625, "read")
    inset = ax.inset_axes((0.625, 0.40, 0.165, 0.34))
    inset.axis("off")
    x = np.linspace(-3.0, 5.0, 200)

    def density(mean: float) -> np.ndarray:
        return np.asarray(np.exp(-0.5 * ((x - mean) / 0.6) ** 2))

    for lift, gap, face, edge in (
        (1.15, 3.2, INK, INK),
        (0.0, 0.5, paper.ENC["random_band"], paper.ENC["reference"]),
    ):
        inset.fill_between(
            x, lift, lift + density(gap), fc=face, ec="none", lw=0
        )
        inset.plot(x, lift + density(gap), color=edge, lw=0.6)
        inset.plot(x, lift + density(0.0), color=edge, lw=0.6)
        inset.plot([x[0], x[-1]], [lift, lift], color=edge, lw=0.6)
    inset.set_xlim(-3.0, 5.0)
    inset.set_ylim(-0.05, 2.25)
    inset.text(-3.0, 1.9, "$v$", fontsize=TEXT, color=INK, va="center")
    inset.text(-3.0, 0.75, "$u$", fontsize=TEXT, color=INK, va="center")
    _text(
        ax,
        0.625,
        0.34,
        "a pair is read\ncorrectly when its\npositive member"
        "\nprojects higher\non $v$",
    )


def _steer(ax: Axes) -> None:
    _title(ax, 0.825, "steer")
    _text(ax, 0.825, 0.69, STEER, size=EQUATION_SIZE)
    _text(
        ax,
        0.825,
        0.57,
        "$c$: relative\ncoefficient,\n$r_\\ell$: mean residual\n"
        "norm at layer $\\ell$",
    )
    _text(
        ax,
        0.825,
        0.31,
        "added at the\nresponse positions\nof 3 consecutive\nlayers",
    )


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure; it reads no results.

    Args:
        manifest: The run resolver, unused by the schematic.

    Returns:
        The figure and its caption parts.
    """
    del manifest
    with paper.paper_style():
        fig = Figure(figsize=(WIDTH, HEIGHT))
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        _pair(ax)
        _stream(ax)
        _finder(ax)
        _read(ax)
        _steer(ax)
        for x0, x1 in ARROWS:
            _arrow(ax, x0, x1, 0.55)
    claim = "The recipe: find a direction, read with it, and steer with it."
    detail = (
        "A pair of inputs that differ in the concept is passed through the "
        "model; a finder turns the residual-stream activations of the last "
        f"token at layer {ELL} into a unit direction v; a pair is read "
        "correctly when its positive member projects higher on v, so a "
        "constant score sits at 0.5; steering adds c r v to the residual "
        "stream. A random unit direction, oriented on the training pairs "
        "like the reader, is the control of the readings. The densities in "
        "the read stage are illustrative."
    )
    return FigureOutput(fig, claim, detail, {}, "")
