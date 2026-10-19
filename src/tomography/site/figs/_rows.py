"""Scaffolding shared by the steering, knowledge and post-training figures.

Figures use constrained layout, one fill rule (filled for GPT-2 and base
checkpoints, open for post-trained ones) and the page's true minus.
"""

from collections.abc import Mapping
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.lines import Line2D

from tomography.site import cells
from tomography.site import stats
from tomography.viz import paper

number = cells.num
PAD = 0.07
STAR_SIZE = 6.4
TRIANGLE_SIZE = 5.4
CATEGORY_EDGE = 0.3
MARKER_SIZE = paper.MARKER_SIZE
EDGE_WIDTH = paper.MARKER_EDGE


def new_figure(height: float) -> Figure:
    """Creates an empty, constrained-layout figure of the page width.

    Args:
        height: Height in inches.

    Returns:
        The figure.
    """
    return Figure(
        figsize=(paper.FULL_WIDTH, height),
        layout=ConstrainedLayoutEngine(w_pad=PAD, h_pad=PAD),
    )


def freeze_layout(fig: Figure) -> None:
    """Pins every axes where constrained layout put it, to six decimals.

    Nested grids can differ in the last bits between two builds, and the ids
    of the SVG clip paths are hashed from those bits, so two builds of one
    figure would write different bytes.

    Args:
        fig: A figure with constrained layout.
    """
    fig.draw_without_rendering()
    for ax in fig.axes:
        x0, y0, width, height = ax.get_position().bounds
        ax.set_position(
            (round(x0, 6), round(y0, 6), round(width, 6), round(height, 6))
        )
    fig.set_layout_engine("none")


def fill(model: str, color: str) -> dict[str, Any]:
    """Gives the marker face and edge for a checkpoint.

    Args:
        model: Checkpoint key.
        color: Edge and, for GPT-2 and base checkpoints, face color.

    Returns:
        Keyword arguments for `Axes.plot`: filled for GPT-2 and base
        checkpoints, a white face with a colored edge for post-trained ones.
    """
    post = stats.role(model) == "post"
    return {
        "markerfacecolor": "white" if post else color,
        "markeredgecolor": color,
        "markeredgewidth": EDGE_WIDTH,
        "markersize": MARKER_SIZE,
    }


def run_counts(
    flags: Mapping[tuple[str, str], bool],
) -> dict[str, stats.RoleCount]:
    """Counts a per-run statement by the role of the run's checkpoint.

    Args:
        flags: (checkpoint key, run label) to whether the statement holds;
            a checkpoint may have more than one run.

    Returns:
        Role to its count, for every role in `stats.ROLES`.
    """
    counts = {r: [0, 0] for r in stats.ROLES}
    for (model, _label), flag in flags.items():
        cell = counts[stats.role(model)]
        cell[0] += int(flag)
        cell[1] += 1
    return {r: stats.RoleCount(h, t) for r, (h, t) in counts.items()}


def dot(
    ax: Axes,
    model: str,
    y: float,
    x: float,
    marker: str,
    color: str,
    interval: tuple[float, float] | None = None,
) -> None:
    """Draws one value of one checkpoint on a row, with an optional whisker.

    Args:
        ax: The panel; rows run down the y axis.
        model: Checkpoint key, which fixes the fill.
        y: Row position, including any dodge.
        x: The value.
        marker: Marker shape, which carries the meaning of the series.
        color: Series color.
        interval: Lower and upper whisker ends, or None.
    """
    if interval is not None:
        ax.plot(
            list(interval),
            [y, y],
            color=color,
            lw=paper.WHISKER_WIDTH,
            solid_capstyle="butt",
            zorder=2,
        )
    ax.plot(
        [x],
        [y],
        marker=marker,
        ls="none",
        color=color,
        zorder=3,
        **_category_edge(
            marker,
            {**fill(model, color), "markersize": marker_size(marker)},
            color,
        ),
    )


def row_axis(ax: Axes, rows: Mapping[str, float], labels: bool) -> None:
    """Sets up a panel whose rows are checkpoints.

    Args:
        ax: The panel.
        rows: Checkpoint key to row position, from `axes.model_rows`.
        labels: Whether to write the checkpoint names beside the rows.
    """
    ax.set_ylim(max(rows.values()) + 0.6, -0.6)
    ax.set_yticks(list(rows.values()))
    if labels:
        ax.set_yticklabels([paper.display_name(m) for m in rows])
    else:
        ax.tick_params(labelleft=False)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def marker_size(marker: str) -> float:
    """Gives the size that makes a marker shape legible.

    Args:
        marker: Marker shape.

    Returns:
        The size in points: larger for the star.
    """
    if marker == "*":
        return STAR_SIZE
    if marker in ("<", ">"):
        return TRIANGLE_SIZE
    return MARKER_SIZE


def _category_edge(
    marker: str, style: dict[str, Any], color: str
) -> dict[str, Any]:
    """Gives filled category markers a thin white edge so overlaps separate.

    Args:
        marker: Marker shape.
        style: Marker keyword arguments of `Axes.plot`.
        color: The series color.

    Returns:
        The keyword arguments, with the white edge when the marker is a
        filled category shape.
    """
    if marker in paper.CATEGORY_MARKERS and style["markerfacecolor"] == color:
        return {
            **style,
            "markeredgecolor": "white",
            "markeredgewidth": CATEGORY_EDGE,
        }
    return style


def series_handle(
    label: str, marker: str, color: str, filled: bool = True
) -> Line2D:
    """Builds a legend handle for one marker series.

    Args:
        label: Legend text.
        marker: Marker shape.
        color: Series color.
        filled: Whether the marker is filled or open.

    Returns:
        The handle.
    """
    return Line2D(
        [],
        [],
        ls="none",
        marker=marker,
        color=color,
        label=label,
        **_category_edge(
            marker,
            {
                "markerfacecolor": color if filled else "white",
                "markeredgecolor": color,
                "markeredgewidth": EDGE_WIDTH,
                "markersize": marker_size(marker),
            },
            color,
        ),
    )
