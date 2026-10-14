"""Geometry plots: labeled matrices and per-layer embeddings.

Matrices cover cosine and transfer tables and layer by coefficient grids;
embeddings show how the classes of a stimulus set separate at each layer.
"""

from collections.abc import Sequence
from typing import Any
from typing import Literal

from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray

from tomography.viz import style

FloatArray = NDArray[np.floating[Any]]

ANNOTATE_UP_TO = 64
LONG_LABEL = 4


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
