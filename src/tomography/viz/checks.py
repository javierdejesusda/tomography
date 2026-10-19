"""Layout checks that every paper figure must pass.

A figure is drawn once under the paper style; the checks then read the text
boxes and marker positions in pixels. They catch the faults a reviewer sees
at a glance: overlapping labels, text cut off by the figure edge, and
markers that sit on a tick label.
"""

import itertools
import re

from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import Collection
from matplotlib.collections import QuadMesh
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.transforms import Bbox
import numpy as np

from tomography.viz import paper

TOLERANCE = 0.5
MARKER_MARGIN = 2.0
MIN_FONT_MAIN = 6.5
MIN_FONT_APPENDIX = 6.0
WHITE = "#ffffff"
MATHTEXT = re.compile(r"(?<!\\)\$.*?(?<!\\)\$", re.DOTALL)


def draw(fig: Figure) -> RendererBase:
    """Draws a figure under the paper style and returns its renderer.

    Args:
        fig: The figure.

    Returns:
        The renderer, so text boxes can be measured.
    """
    canvas = FigureCanvasAgg(fig)
    with paper.paper_style():
        canvas.draw()  # type: ignore[no-untyped-call]
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


def overlap_problems(fig: Figure, renderer: RendererBase) -> list[str]:
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
    return [*overlap_problems(fig, renderer), *marker_problems(fig, renderer)]


def _visible_texts(fig: Figure) -> list[Text]:
    """Lists the drawn, non-empty text artists of a figure once each."""
    draw(fig)
    hidden = {id(t) for ax in fig.axes for t in _unused_tick_labels(ax)}
    unique = {id(t): t for t in fig.findobj(Text) if id(t) not in hidden}
    return [t for t in unique.values() if t.get_text() and t.get_visible()]


def _allowed_colours() -> set[str]:
    allowed = {WHITE}
    for value in paper.ENC.values():
        allowed.update(
            c.lower() for c in ([value] if isinstance(value, str) else value)
        )
    allowed.discard(paper.ENC["secondary_text"].lower())
    return allowed


def _hex(color: object) -> str | None:
    """Writes a colour as lower-case hex, or None when it is transparent."""
    red, green, blue, alpha = mcolors.to_rgba(color)  # type: ignore[arg-type]
    if alpha == 0.0:
        return None
    return mcolors.to_hex((red, green, blue)).lower()


def _artist_colours(artist: Artist) -> list[str]:
    """Reads the face, edge and line colours that an artist draws."""
    colours: list[object] = []
    if isinstance(artist, Line2D):
        if artist.get_linestyle() not in ("None", "none", " ", ""):
            colours.append(artist.get_color())
        if artist.get_marker() not in ("None", None, ""):
            colours += [
                artist.get_markerfacecolor(),
                artist.get_markeredgecolor(),
            ]
    elif isinstance(artist, QuadMesh):
        return []
    elif isinstance(artist, Collection):
        colours += list(artist.get_facecolor()) + list(artist.get_edgecolor())
    elif isinstance(artist, Patch):
        if artist.get_fill():
            colours.append(artist.get_facecolor())
        if artist.get_linewidth() > 0:
            colours.append(artist.get_edgecolor())
    found = []
    for colour in colours:
        text = _hex(colour)
        if text is not None:
            found.append(text)
    return found


def palette_problems(fig: Figure) -> list[str]:
    """Finds colours that are not in the encoding dictionary.

    Faces, edges and lines may use a dictionary hue or white; text may use the
    ink or the secondary grey, and white too on a text artist whose gid is
    `paper.TILE_LABEL` (a number on a heat-map tile); heat maps and colour
    bars are exempt.

    Args:
        fig: The figure.

    Returns:
        One message per colour not allowed.
    """
    allowed = _allowed_colours()
    found = []
    seen = {
        id(tick.gridline)
        for ax in fig.axes
        for axis in (ax.xaxis, ax.yaxis)
        for tick in axis.get_major_ticks() + axis.get_minor_ticks()
    }
    for artist in fig.findobj(
        lambda a: isinstance(a, Line2D | Patch | Collection)
    ):
        if id(artist) in seen or not artist.get_visible():
            continue
        seen.add(id(artist))
        for shade in _artist_colours(artist):
            if shade not in allowed:
                found.append(f"colour {shade} on {type(artist).__name__}")
    text_ok = {paper.INK.lower(), paper.ENC["secondary_text"].lower()}
    for text in _visible_texts(fig):
        tint = _hex(text.get_color())
        allowed_ink = text_ok
        if text.get_gid() == paper.TILE_LABEL:
            allowed_ink = text_ok | {paper.WHITE}
        if tint is not None and tint not in allowed_ink:
            found.append(f"text colour {tint}: {text.get_text()!r}")
    return found


def font_problems(fig: Figure, appendix: bool = False) -> list[str]:
    """Finds text smaller than the minimum size.

    Args:
        fig: The figure.
        appendix: Whether the figure is an appendix figure (6 pt minimum
            instead of 6.5 pt).

    Returns:
        One message per text artist below the minimum.
    """
    minimum = MIN_FONT_APPENDIX if appendix else MIN_FONT_MAIN
    return [
        f"font {float(t.get_fontsize()):g} pt below {minimum:g} pt: "
        f"{t.get_text()!r}"
        for t in _visible_texts(fig)
        if float(t.get_fontsize()) < minimum
    ]


def text_problems(fig: Figure) -> list[str]:
    """Finds underscores written outside mathtext.

    The cm font draws an underscore as a dot accent, and an underscore in
    reader-facing text is a raw identifier.

    Args:
        fig: The figure.

    Returns:
        One message per text artist that contains an underscore.
    """
    return [
        f"underscore in text: {t.get_text()!r}"
        for t in _visible_texts(fig)
        if "_" in MATHTEXT.sub("", t.get_text())
    ]


def guard_problems(fig: Figure, appendix: bool = False) -> list[str]:
    """Runs the palette, font and underscore guards.

    Args:
        fig: The figure.
        appendix: Whether the figure is an appendix figure.

    Returns:
        Messages for all faults; empty when the figure passes.
    """
    return [
        *palette_problems(fig),
        *font_problems(fig, appendix),
        *text_problems(fig),
    ]
