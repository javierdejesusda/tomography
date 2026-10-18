"""Layout checks that every paper figure must pass.

A figure is drawn once under the paper style; the checks then read the text
boxes and marker positions in pixels. They catch the faults a reviewer sees
at a glance: overlapping labels, text cut off by the figure edge, and
markers that sit on a tick label.
"""

import itertools

from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.transforms import Bbox
import numpy as np

from tomography.viz import paper

TOLERANCE = 0.5
MARKER_MARGIN = 2.0


def draw(fig: Figure) -> RendererBase:
    """Draws a figure under the paper style and returns its renderer.

    Args:
        fig: The figure.

    Returns:
        The renderer, so text boxes can be measured.
    """
    canvas = FigureCanvasAgg(fig)
    with paper.paper_style():
        canvas.draw()
    renderer: RendererBase = canvas.get_renderer()  # type: ignore[no-untyped-call]
    return renderer


def _unused_tick_labels(ax: Axes) -> list[Text]:
    """Lists tick labels whose tick lies outside the view and is not drawn."""
    unused = []
    for axis in (ax.xaxis, ax.yaxis):
        low, high = sorted(axis.get_view_interval())
        slack = (high - low) * 1e-6
        for tick in axis.get_major_ticks():
            loc = tick.get_loc()
            if not low - slack <= loc <= high + slack:
                unused += [tick.label1, tick.label2]
    return unused


def _tick_labels(ax: Axes, renderer: RendererBase) -> list[tuple[str, Bbox]]:
    skip = {id(t) for t in _unused_tick_labels(ax)}
    labels = [*ax.get_xticklabels(), *ax.get_yticklabels()]
    return [
        (t.get_text(), t.get_window_extent(renderer))
        for t in labels
        if t.get_text() and t.get_visible() and id(t) not in skip
    ]


def text_boxes(fig: Figure, renderer: RendererBase) -> list[tuple[str, Bbox]]:
    """Measures every visible, non-empty text artist.

    Args:
        fig: The figure.
        renderer: A renderer from `draw`.

    Returns:
        Text and pixel box for each artist.
    """
    hidden = {id(t) for ax in fig.axes for t in _unused_tick_labels(ax)}
    unique = {id(t): t for t in fig.findobj(Text) if id(t) not in hidden}
    return [
        (t.get_text(), t.get_window_extent(renderer))
        for t in unique.values()
        if t.get_text() and t.get_visible()
    ]


def _overlap(a: Bbox, b: Bbox) -> bool:
    return bool(
        a.x0 < b.x1 - TOLERANCE
        and b.x0 < a.x1 - TOLERANCE
        and a.y0 < b.y1 - TOLERANCE
        and b.y0 < a.y1 - TOLERANCE
    )


def text_problems(fig: Figure, renderer: RendererBase) -> list[str]:
    """Finds overlapping text and text outside the figure.

    Args:
        fig: The figure.
        renderer: A renderer from `draw`.

    Returns:
        One message per fault.
    """
    boxes = text_boxes(fig, renderer)
    width, height = fig.bbox.width, fig.bbox.height
    found = []
    for text, box in boxes:
        if (
            box.x0 < -TOLERANCE
            or box.y0 < -TOLERANCE
            or box.x1 > width + TOLERANCE
            or box.y1 > height + TOLERANCE
        ):
            found.append(f"text outside the figure: {text!r}")
    for (a, box_a), (b, box_b) in itertools.combinations(boxes, 2):
        if _overlap(box_a, box_b):
            found.append(f"text overlap: {a!r} and {b!r}")
    return found


def marker_problems(fig: Figure, renderer: RendererBase) -> list[str]:
    """Finds markers that sit on a tick label.

    Args:
        fig: The figure.
        renderer: A renderer from `draw`.

    Returns:
        One message per fault.
    """
    labels = [item for ax in fig.axes for item in _tick_labels(ax, renderer)]
    found = []
    for ax in fig.axes:
        for line in ax.get_lines():
            if line.get_marker() in ("None", None, ""):
                continue
            data = np.asarray(line.get_xydata(), dtype=float)
            for point in ax.transData.transform(data):
                px, py = float(point[0]), float(point[1])
                for text, box in labels:
                    if (
                        box.x0 - MARKER_MARGIN <= px <= box.x1 + MARKER_MARGIN
                        and box.y0 - MARKER_MARGIN
                        <= py
                        <= box.y1 + MARKER_MARGIN
                    ):
                        found.append(f"marker covers tick label {text!r}")
    return found


def problems(fig: Figure) -> list[str]:
    """Runs every layout check on a figure.

    Args:
        fig: The figure.

    Returns:
        Messages for all faults; empty when the figure is clean.
    """
    renderer = draw(fig)
    return [*text_problems(fig, renderer), *marker_problems(fig, renderer)]
