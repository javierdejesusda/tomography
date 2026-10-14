"""Shared figure style: color tokens, rc parameters and the save helper.

Colors come from the validated reference palette of the dataviz method.
Categorical hues are assigned in a fixed order and never cycled, so a series
keeps its color across figures. The figure surface is white: every contrast
and separation check of the palette was run against the slightly darker
`#fcfcfb`, so it only gets easier on white.
"""

from collections.abc import Iterator
from collections.abc import Sequence
import contextlib
from pathlib import Path
from typing import Any
from typing import cast

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Colormap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.rcsetup import cycler
import numpy as np
from numpy.typing import NDArray

CATEGORICAL: tuple[str, ...] = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)

SEQUENTIAL_STEPS: tuple[str, ...] = (
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)
NEUTRAL = "#f0efec"
DIVERGING_STEPS: tuple[str, ...] = (
    "#104281",
    "#256abf",
    "#5598e7",
    "#9ec5f4",
    NEUTRAL,
    "#f1aea8",
    "#dd716a",
    "#b13f3c",
    "#762221",
)

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

FORMATS: tuple[str, ...] = ("pdf", "png", "svg")
NO_STAMP: dict[str, dict[str, None]] = {
    "pdf": {"CreationDate": None, "Creator": None, "Producer": None},
    "png": {"Software": None},
    "svg": {"Date": None, "Creator": None},
}
PX = 0.75
COLUMN_WIDTH = 3.5
TEXT_WIDTH = 7.0

RC_PARAMS: dict[Any, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8,
    "text.color": INK,
    "axes.titlesize": 8,
    "axes.titlecolor": INK,
    "axes.labelsize": 8,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "xtick.color": BASELINE,
    "ytick.color": BASELINE,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "axes.edgecolor": BASELINE,
    "axes.linewidth": PX,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linestyle": "-",
    "grid.linewidth": PX,
    "axes.prop_cycle": cycler(color=CATEGORICAL),
    "lines.linewidth": 2 * PX,
    "lines.markersize": 8 * PX,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "path",
    "svg.hashsalt": "tomography",
}


def categorical_colors(n: int) -> list[str]:
    """Returns the first `n` categorical hues in their fixed order.

    Args:
        n: Number of series to color.

    Returns:
        Hex colors, one per series.

    Raises:
        ValueError: If `n` exceeds the eight documented hues. A ninth series
            is never a generated hue: fold the tail or facet instead.
    """
    if n > len(CATEGORICAL):
        raise ValueError(
            f"at most eight categorical colors exist, got {n}: fold the "
            "tail into one series or split the figure into panels"
        )
    return list(CATEGORICAL[:n])


def sequential_cmap() -> Colormap:
    """Returns the one-hue magnitude ramp, light (low) to dark (high).

    Returns:
        A blue ramp from palette step 100 to step 700.
    """
    return LinearSegmentedColormap.from_list(
        "tomography_sequential", SEQUENTIAL_STEPS
    )


def diverging_cmap() -> Colormap:
    """Returns the two-pole ramp for signed values around zero.

    Blue marks the negative pole and red the positive one, joined by a gray
    midpoint so that zero reads as nothing. The blue arm takes steps of the
    documented blue ramp; the red arm repeats their OKLCH lightness and
    chroma at the hue of the categorical red, so both arms are equally
    strong.

    Returns:
        A blue, gray, red colormap with the gray exactly at 0.5.
    """
    return LinearSegmentedColormap.from_list(
        "tomography_diverging", DIVERGING_STEPS
    )


@contextlib.contextmanager
def style_context() -> Iterator[None]:
    """Applies the figure style to everything created or saved inside.

    Yields:
        None. The rc parameters set here are restored on exit, including on
        exceptions.
    """
    with mpl.rc_context(RC_PARAMS):
        yield


def new_figure(
    nrows: int = 1,
    ncols: int = 1,
    *,
    width: float = COLUMN_WIDTH,
    aspect: float = 0.7,
    sharex: bool = False,
    sharey: bool = False,
) -> tuple[Figure, NDArray[np.object_]]:
    """Creates a styled figure with a grid of axes, outside pyplot.

    The figure is not registered with pyplot, so no backend is selected and
    no figure outlives its last reference.

    Args:
        nrows: Number of panel rows.
        ncols: Number of panel columns.
        width: Figure width in inches.
        aspect: Height over width of one panel, which fixes the figure height.
        sharex: Whether panels share the x axis.
        sharey: Whether panels share the y axis.

    Returns:
        The figure and its axes as an array of shape (nrows, ncols).
    """
    with style_context():
        fig = Figure(
            figsize=(width, width * aspect * nrows / ncols),
            layout="constrained",
        )
        FigureCanvasAgg(fig)
        axes = fig.subplots(
            nrows, ncols, sharex=sharex, sharey=sharey, squeeze=False
        )
    return fig, axes


def save_figure(
    fig: Figure,
    out_dir: str | Path,
    name: str,
    *,
    formats: Sequence[str] = FORMATS,
    dpi: int = 300,
) -> list[Path]:
    """Writes a figure as `<out_dir>/<name>.<format>` for every format.

    Args:
        fig: The figure to save.
        out_dir: Destination folder, created when missing.
        name: File name without extension.
        formats: Any of "pdf", "png" and "svg".
        dpi: Resolution of the PNG; vector formats ignore it.

    Returns:
        The written paths, in the order of `formats`.

    Raises:
        ValueError: If a format is not one of "pdf", "png" and "svg".
    """
    unknown = [fmt for fmt in formats if fmt not in FORMATS]
    if unknown:
        raise ValueError(
            f"unsupported figure format {unknown[0]!r}, expected one of "
            f"{FORMATS}"
        )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.{fmt}" for fmt in formats]
    with style_context():
        for path, fmt in zip(paths, formats, strict=True):
            with path.open("wb") as handle:
                fig.savefig(handle, format=fmt, dpi=dpi, metadata=NO_STAMP[fmt])
    return paths


def figure_of(ax: Axes) -> Figure:
    """Returns the root figure that holds `ax`, also for sub-figure axes.

    Args:
        ax: Axes created by `new_figure` or by any figure's `subplots`.

    Returns:
        The figure to save or return from a primitive.
    """
    return cast(Figure, ax.get_figure(root=True))
