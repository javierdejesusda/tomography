"""Figure 1: Method schematic.

Stimulus pairs go through a model, a direction is found in the residual
stream, and the same unit direction is used to read a concept and to steer
with `h' = h + c r v`, where `r` is the mean residual norm at the layer (the
`relative` coefficient mode). A random unit direction is the baseline of both.
"""

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import FancyBboxPatch

from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import paper

WIDTH = 5.5
HEIGHT = 2.5
STEER = r"$h'_\ell = h_\ell + c\,r_\ell\,v_\ell$"


def _box(
    ax: Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    dashed: bool = False,
    letter: str = "",
) -> None:
    color = paper.CHANCE if dashed else paper.INK
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.012",
            fc="white",
            ec=color,
            lw=0.7,
            ls="--" if dashed else "-",
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=paper.TICK,
        color=color,
        linespacing=1.35,
    )
    if letter:
        ax.text(
            x,
            y + h + 0.015,
            rf"$\mathbf{{({letter})}}$",
            ha="left",
            va="bottom",
            fontsize=paper.LETTER,
        )


def _arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    dashed: bool = False,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7,
            lw=0.7,
            color=paper.CHANCE if dashed else paper.INK,
            ls="--" if dashed else "-",
            shrinkA=0,
            shrinkB=0,
        )
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
        _box(
            ax,
            0.01,
            0.40,
            0.19,
            0.38,
            "pairs of\ncontrasting\nstimuli",
            letter="a",
        )
        _box(
            ax,
            0.26,
            0.40,
            0.20,
            0.38,
            "residual stream\n$h_\\ell$ at the\nlast token",
            letter="b",
        )
        _box(
            ax,
            0.52,
            0.40,
            0.21,
            0.38,
            "direction $v_\\ell$,\nunit norm; PCA,\nmean difference,\nlogistic",
            letter="c",
        )
        _box(
            ax,
            0.79,
            0.62,
            0.20,
            0.30,
            "read: score $h_\\ell\\cdot v_\\ell$\nagainst a threshold",
            letter="d",
        )
        _box(
            ax,
            0.79,
            0.18,
            0.20,
            0.30,
            "steer:\n" + STEER,
            letter="e",
        )
        _box(
            ax,
            0.52,
            0.04,
            0.21,
            0.22,
            "random unit direction\n(baseline of both)",
            dashed=True,
            letter="f",
        )
        _arrow(ax, (0.20, 0.59), (0.26, 0.59))
        _arrow(ax, (0.46, 0.59), (0.52, 0.59))
        _arrow(ax, (0.73, 0.64), (0.79, 0.77))
        _arrow(ax, (0.73, 0.54), (0.79, 0.35))
        _arrow(ax, (0.73, 0.15), (0.79, 0.20), dashed=True)
        _arrow(ax, (0.73, 0.19), (0.79, 0.68), dashed=True)
        ax.text(
            0.89,
            0.06,
            "$r_\\ell$: mean residual norm\nat layer $\\ell$; $c$: strength",
            ha="center",
            va="center",
            fontsize=paper.TICK,
        )
    claim = (
        "The same unit direction found in the residual stream of a model is "
        "used to read a concept and to steer with it, and a random unit "
        "direction is the baseline of both."
    )
    detail = (
        "Boxes (a) to (c) find the direction; (d) reads it by the score of "
        "the residual stream along it, and (e) adds it to the residual stream "
        "with a coefficient proportional to the mean residual norm of the "
        "layer. The dashed box is the baseline used in every comparison."
    )
    return FigureOutput(fig, claim, detail, {}, "")
