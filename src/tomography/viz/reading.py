"""Layer curves: a quantity per layer with confidence bands.

The primitive takes plain arrays, one `(x, mean, lo, hi)` tuple per series,
so models of different depth share a figure once the caller has rescaled `x`.
"""

from collections.abc import Sequence
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import numpy as np
from numpy.typing import NDArray

from tomography.viz import style

FloatArray = NDArray[np.floating[Any]]
CurveArrays = tuple[FloatArray, FloatArray, FloatArray, FloatArray]

BAND_ALPHA = 0.1
SEED_ALPHA = 0.3


def curves_from_arrays(
    x: FloatArray, mean: FloatArray, lo: FloatArray, hi: FloatArray
) -> list[CurveArrays]:
    """Splits the arrays of one `curves` result file into per-series tuples.

    Args:
        x: Shared abscissa, shape (K,).
        mean: Series means, shape (S, K).
        lo: Lower band bounds, shape (S, K).
        hi: Upper band bounds, shape (S, K).

    Returns:
        One `(x, mean, lo, hi)` tuple per series.

    Raises:
        ValueError: If the arrays do not share the shapes above.
    """
    if mean.ndim != 2 or lo.shape != mean.shape or hi.shape != mean.shape:
        raise ValueError(
            "mean, lo and hi must share the shape (S, K), got "
            f"{mean.shape}, {lo.shape} and {hi.shape}"
        )
    if x.shape != mean.shape[1:]:
        raise ValueError(
            f"x must have shape (K,) = {mean.shape[1:]}, got shape {x.shape}"
        )
    return [(x, mean[i], lo[i], hi[i]) for i in range(mean.shape[0])]


def _check_inputs(
    curves: Sequence[CurveArrays],
    series: Sequence[str],
    per_seed: Sequence[FloatArray | None] | None,
) -> None:
    if not curves:
        raise ValueError("plot_curves needs at least one series")
    if len(series) != len(curves):
        raise ValueError(
            f"got {len(curves)} series of arrays but {len(series)} labels"
        )
    if per_seed is not None and len(per_seed) != len(curves):
        raise ValueError(
            f"per_seed needs one entry per series, got {len(per_seed)} "
            f"for {len(curves)}"
        )
    for i, (x, mean, lo, hi) in enumerate(curves):
        label = series[i]
        if x.ndim != 1 or not x.shape == mean.shape == lo.shape == hi.shape:
            raise ValueError(
                f"x, mean, lo and hi of series {label!r} must share the "
                f"shape (K,), got {x.shape}, {mean.shape}, {lo.shape} and "
                f"{hi.shape}"
            )
        if np.any(lo > hi):
            raise ValueError(f"lo exceeds hi for series {label!r}")
        seeds = None if per_seed is None else per_seed[i]
        if seeds is not None and (seeds.ndim != 2 or seeds.shape[1] != len(x)):
            raise ValueError(
                f"per_seed of series {label!r} must have shape (N, K) with "
                f"K = {len(x)}, got {seeds.shape}"
            )


def plot_curves(
    curves: Sequence[CurveArrays],
    *,
    series: Sequence[str],
    per_seed: Sequence[FloatArray | None] | None = None,
    xlabel: str = "Layer",
    ylabel: str = "Accuracy",
    chance: float | None = None,
    n_layers: int | None = None,
    ylim: tuple[float, float] | None = None,
    legend: bool | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plots one line with a confidence band per series.

    Series take the categorical hues in order and a band at 10% opacity. A
    legend above the plot names two or more series; it replaces end labels
    because layer curves converge and would collide. The chance label sits
    in the right margin for the same reason.

    Args:
        curves: One `(x, mean, lo, hi)` tuple per series, each of shape (K,)
            with its own K.
        series: Legend label of each series.
        per_seed: Optional per-series arrays of shape (N, K), or None for a
            series without them, drawn as faint hairlines under the mean.
        xlabel: Label of the x axis.
        ylabel: Label of the y axis.
        chance: Level of a labeled reference line, or None for no line.
        n_layers: Depth of the model; fixes the x range to 0..n_layers.
        ylim: Fixed y range, or None to autoscale.
        legend: Whether to draw the legend; None draws it for two or more
            series.
        ax: Axes to draw on. A new styled figure is created when None.

    Returns:
        The figure holding the axes.

    Raises:
        ValueError: If there is no series, the labels, bands or per-seed
            arrays do not match the curves, or there are more than eight
            series.
    """
    _check_inputs(curves, series, per_seed)
    colors = style.categorical_colors(len(curves))
    if ax is None:
        _, axes = style.new_figure()
        ax = axes[0, 0]
    with style.style_context():
        for i, ((x, mean, lo, hi), label, color) in enumerate(
            zip(curves, series, colors, strict=True)
        ):
            ax.fill_between(x, lo, hi, color=color, alpha=BAND_ALPHA, lw=0)
            seeds = None if per_seed is None else per_seed[i]
            if seeds is not None:
                for values in seeds:
                    ax.plot(
                        x,
                        values,
                        color=color,
                        lw=style.PX,
                        alpha=SEED_ALPHA,
                        label="_seed",
                    )
            ax.plot(x, mean, color=color, label=label)
        if chance is not None:
            ax.axhline(
                chance, color=style.INK_MUTED, lw=style.PX, label="_chance"
            )
            ax.text(
                1.01,
                chance,
                "chance",
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                color=style.INK_SECONDARY,
            )
        every_x = np.concatenate([x for x, *_ in curves])
        if np.all(every_x == np.round(every_x)):
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if n_layers is not None:
            ax.set_xlim(0, n_layers)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if legend is None:
            legend = len(curves) > 1
        if legend:
            ax.legend(
                loc="lower left",
                bbox_to_anchor=(0.0, 1.0),
                borderaxespad=0.2,
                ncols=min(len(curves), 4),
                handlelength=1.5,
                columnspacing=1.2,
            )
    return style.figure_of(ax)
