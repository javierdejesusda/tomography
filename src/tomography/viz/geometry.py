"""Geometry plots: labeled matrices and per-layer embeddings.

Matrices cover cosine and transfer tables and layer by coefficient grids;
embeddings show how the classes of a stimulus set separate at each layer.
"""

from collections.abc import Sequence
import math
from typing import Any
from typing import Literal

from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
from numpy.typing import NDArray

from tomography.viz import style

FloatArray = NDArray[np.floating[Any]]

ANNOTATE_UP_TO = 64
LONG_LABEL = 4
HUE_CLASSES = 3
MARKERS = ("o", "s", "^")
POINT_SIZE = 12
MAX_COLUMNS = 4


def plot_matrix(
    values: FloatArray,
    *,
    rows: Sequence[str],
    cols: Sequence[str],
    title: str | None = None,
    kind: Literal["sequential", "diverging"] = "sequential",
    vmin: float | None = None,
    vmax: float | None = None,
    annotate: bool | None = None,
    fmt: str = "{:.2f}",
    cbar_label: str | None = None,
    colorbar: bool = True,
    ax: Axes | None = None,
) -> Figure:
    """Plots a labeled matrix as a heatmap with a surface gap between cells.

    Args:
        values: Cell values of shape (R, C); NaN cells stay blank.
        rows: The R row labels, top to bottom.
        cols: The C column labels, left to right.
        title: Title of the axes, if any.
        kind: "sequential" for magnitudes (light to dark blue) or
            "diverging" for signed values (blue, gray at zero, red); the
            diverging range is symmetric around zero.
        vmin: Lower end of the color range; None takes the data minimum.
        vmax: Upper end of the color range; None takes the data maximum.
        annotate: Whether to write each value in its cell; None annotates
            matrices of up to 64 cells.
        fmt: Format string of the cell values.
        cbar_label: Label of the color bar.
        colorbar: Whether to add a color bar beside the matrix.
        ax: Axes to draw on. A new styled figure is created when None.

    Returns:
        The figure holding the axes.

    Raises:
        ValueError: If `values` is not 2-D, the labels do not match its
            shape, `kind` is unknown or `vmin` is not below `vmax`.
    """
    if values.ndim != 2:
        raise ValueError(
            f"values must be 2-D (rows, cols), got shape {values.shape}"
        )
    n_rows, n_cols = values.shape
    if len(rows) != n_rows:
        raise ValueError(f"got {len(rows)} rows for {n_rows} value rows")
    if len(cols) != n_cols:
        raise ValueError(f"got {len(cols)} cols for {n_cols} value columns")
    if kind not in ("sequential", "diverging"):
        raise ValueError(f"unknown kind {kind!r}")
    low = float(np.nanmin(values)) if vmin is None else vmin
    high = float(np.nanmax(values)) if vmax is None else vmax
    if low >= high and not (vmin is None and vmax is None):
        raise ValueError(f"vmin {low} must be below vmax {high}")
    if kind == "diverging":
        limit = max(abs(low), abs(high)) or 1.0
        low, high = -limit, limit
        cmap = style.diverging_cmap()
    else:
        cmap = style.sequential_cmap()
    cmap = cmap.with_extremes(bad=style.SURFACE)
    norm = Normalize(low, high)
    if annotate is None:
        annotate = values.size <= ANNOTATE_UP_TO

    if ax is None:
        _, axes = style.new_figure()
        ax = axes[0, 0]
    with style.style_context():
        image = ax.imshow(
            np.ma.masked_invalid(values),
            cmap=cmap,
            norm=norm,
            origin="upper",
            aspect="auto",
            interpolation="nearest",
        )
        long_labels = any(len(c) > LONG_LABEL for c in cols)
        ax.set_xticks(
            range(n_cols),
            cols,
            rotation=45 if long_labels else 0,
            ha="right" if long_labels else "center",
        )
        ax.set_yticks(range(n_rows), rows)
        ax.set_xticks(np.arange(-0.5, n_cols), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows), minor=True)
        ax.set_axisbelow(False)
        ax.grid(False, which="major")
        ax.grid(
            True,
            which="minor",
            axis="both",
            color=style.SURFACE,
            linewidth=2 * style.PX,
        )
        ax.tick_params(which="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if title is not None:
            ax.set_title(title)
        if annotate:
            for i, j in np.ndindex(values.shape):
                if np.isnan(values[i, j]):
                    continue
                ax.text(
                    j,
                    i,
                    fmt.format(values[i, j]),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=style.ink_on(cmap(norm(values[i, j]))),
                )
        fig = style.figure_of(ax)
        if colorbar:
            bar = fig.colorbar(image, ax=ax, label=cbar_label, pad=0.02)
            bar.outline.set_visible(False)
            bar.ax.tick_params(length=0)
    return fig


def _frame(ax: Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_aspect("equal", adjustable="datalim")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(style.GRID)


def _scatter(
    ax: Axes,
    points: FloatArray,
    color: str,
    marker: str,
    *,
    alpha: float = 1.0,
    zorder: int = 2,
) -> None:
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=POINT_SIZE,
        color=color,
        marker=marker,
        alpha=alpha,
        edgecolors=style.SURFACE,
        linewidths=0.4,
        zorder=zorder,
        rasterized=True,
    )


def _key(color: str, marker: str, label: str) -> Line2D:
    return Line2D(
        [],
        [],
        linestyle="",
        marker=marker,
        markerfacecolor=color,
        markeredgecolor=style.SURFACE,
        markersize=5,
        label=label,
    )


def plot_embedding(
    xy: FloatArray,
    labels: NDArray[np.int64],
    layers: Sequence[int] | NDArray[np.int64],
    *,
    classes: Sequence[str],
    ncols: int | None = None,
    width: float = style.TEXT_WIDTH,
) -> Figure:
    """Plots a two-dimensional embedding of the stimuli, one panel per layer.

    Up to three classes take a hue and a marker shape each. Past three, an
    all-pairs hue scheme fails the palette's separation floor, so the figure
    facets one-vs-rest instead: one row per class with that class in the
    first hue and every other class in gray, one column per layer.

    Args:
        xy: Embedded coordinates of shape (K, N, 2).
        labels: Class index of each of the N points, in 0..len(classes)-1.
        layers: The K layer indices, in panel order.
        classes: Name of each class, indexed by label value.
        ncols: Panel columns of the hue layout (default: up to four); not
            allowed for the one-vs-rest layout, whose columns are layers.
        width: Figure width in inches.

    Returns:
        A new figure with a legend and no axis scales (embedding axes are
        arbitrary).

    Raises:
        ValueError: If the shapes of `xy`, `labels` and `layers` disagree,
            a label has no class name, or `ncols` is given for more than
            three classes.
    """
    if xy.ndim != 3 or xy.shape[2] != 2:
        raise ValueError(f"xy must have shape (K, N, 2), got {xy.shape}")
    n_layers, n_points, _ = xy.shape
    if len(labels) != n_points:
        raise ValueError(f"got {len(labels)} labels for {n_points} points")
    if len(layers) != n_layers:
        raise ValueError(f"got {len(layers)} layers for {n_layers} panels")
    if not classes:
        raise ValueError("classes must name at least one class")
    if labels.size and labels.max() >= len(classes):
        raise ValueError(
            f"labels reach class {labels.max()} but only {len(classes)} "
            "classes were named"
        )
    facet = len(classes) > HUE_CLASSES
    if facet and ncols is not None:
        raise ValueError(
            "ncols cannot wrap the one-vs-rest rows of more than three classes"
        )
    n_cols = n_layers if facet else ncols or min(n_layers, MAX_COLUMNS)
    n_rows = len(classes) if facet else math.ceil(n_layers / n_cols)

    fig, axes = style.new_figure(
        n_rows, n_cols, width=width, aspect=0.7 if facet else 1.0
    )
    with style.style_context():
        for index, ax in enumerate(axes.flat):
            row, col = divmod(index, n_cols)
            panel = col if facet else index
            if panel >= n_layers:
                ax.set_visible(False)
                continue
            _frame(ax)
            points = xy[panel]
            if facet:
                focus = labels == row
                _scatter(ax, points[~focus], style.INK_MUTED, "o", alpha=0.4)
                _scatter(ax, points[focus], style.CATEGORICAL[0], "o", zorder=3)
                if col == 0:
                    ax.set_ylabel(classes[row])
                if row == 0:
                    ax.set_title(f"Layer {layers[panel]}")
            else:
                for c in range(len(classes)):
                    _scatter(
                        ax,
                        points[labels == c],
                        style.CATEGORICAL[c],
                        MARKERS[c],
                    )
                ax.set_title(f"Layer {layers[panel]}")
        if facet:
            keys = [
                _key(style.CATEGORICAL[0], "o", "class of the row"),
                _key(style.INK_MUTED, "o", "other classes"),
            ]
        else:
            keys = [
                _key(style.CATEGORICAL[c], MARKERS[c], name)
                for c, name in enumerate(classes)
            ]
        fig.legend(handles=keys, loc="outside upper center", ncols=len(keys))
    return fig
