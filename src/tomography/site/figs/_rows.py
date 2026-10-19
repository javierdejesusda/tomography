"""Scaffolding shared by the steering, knowledge and post-training figures.

Figures use constrained layout, one fill rule (filled for GPT-2 and base
checkpoints, open for post-trained ones) and the page's true minus.
"""

from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from tomography.site import stats
from tomography.viz import axes
from tomography.viz import paper

PAD = 0.07
MARKER_SIZE = 3.5
EDGE_WIDTH = 0.8


def new_figure(height: float) -> Figure:
    """Creates an empty, constrained-layout figure of the page width.

    Args:
        height: Height in inches.

    Returns:
        The figure.
    """
    fig = Figure(figsize=(paper.FULL_WIDTH, height), layout="constrained")
    engine = fig.get_layout_engine()
    assert engine is not None
    engine.set(w_pad=PAD, h_pad=PAD)
    return fig


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


def tick_text(value: float) -> str:
    """Writes a tick label whose minus the Computer Modern font can draw.

    Args:
        value: The tick value.

    Returns:
        The trimmed number, as mathtext when it is negative.
    """
    text = f"{round(value, 6):g}"
    if text in ("-0", "0"):
        return "0"
    return rf"$\mathdefault{{{text}}}$" if value < 0 else text


def tick_formatter() -> ticker.FuncFormatter:
    """Builds a tick formatter that uses `tick_text`.

    Returns:
        The formatter.
    """
    return ticker.FuncFormatter(lambda value, _pos: tick_text(value))


def join_and(parts: Sequence[str]) -> str:
    """Joins phrases as a list: `a`, `a and b` or `a, b and c`.

    Args:
        parts: The phrases.

    Returns:
        The joined text.
    """
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def all_or_per_role(counts: Mapping[str, stats.RoleCount]) -> str:
    """Writes a count as `all N checkpoints` or as its per-role phrase.

    Args:
        counts: Role to its count.

    Returns:
        `all 9 checkpoints` when every checkpoint of every role satisfies the
        statement, else the per-role text.
    """
    total = sum(c.total for c in counts.values())
    if total and all(c.hits == c.total for c in counts.values()):
        return f"all {total} checkpoints"
    return stats.per_role_text(counts)


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
        **fill(model, color),
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
        markerfacecolor=color if filled else "white",
        markeredgecolor=color,
        markeredgewidth=EDGE_WIDTH,
        markersize=MARKER_SIZE,
        label=label,
    )


def number(value: float, digits: int = 3) -> str:
    """Writes a number for a caption with the true minus.

    Args:
        value: The number.
        digits: Decimals.

    Returns:
        The text; a value that rounds to zero has no sign.
    """
    text = f"{value:.{digits}f}"
    if float(text) == 0.0:
        text = text.lstrip("-")
    return text.replace("-", axes.MINUS)
