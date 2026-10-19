"""Scaffolding shared by the steering, knowledge and post-training figures.

Figures use constrained layout, one fill rule (filled for GPT-2 and base
checkpoints, open for post-trained ones) and the page's true minus.
"""

from collections.abc import Sequence
from typing import Any

from matplotlib import ticker
from matplotlib.figure import Figure

from tomography.site import stats
from tomography.viz import axes
from tomography.viz import paper

MARKER_SIZE = 3.5
EDGE_WIDTH = 0.8


def new_figure(height: float) -> Figure:
    """Creates an empty, constrained-layout figure of the page width.

    Args:
        height: Height in inches.

    Returns:
        The figure.
    """
    return Figure(figsize=(paper.FULL_WIDTH, height), layout="constrained")


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
