"""Inline SVG charts for the results site.

Two chart forms cover every figure: a dot panel (one row per model or run,
marks with optional intervals, an optional grey band for a control) and a
line panel (series over a numeric axis, optionally logarithmic). Colors come
from CSS classes (`s1` to `s6`), so the page controls the palette and both
themes. Every panel also has an HTML table with the same numbers.
"""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
import html
import math

WIDTH = 340
LABEL_X = 112
PLOT_L = 120
PLOT_R = 330
ROW_H = 22
GROUP_GAP = 10
TOP = 18
MARK_R = 4.5
STAGGER = 4.5

SVG_NS = "http://www.w3.org/2000/svg"


@dataclasses.dataclass(frozen=True)
class Style:
    """How one series is drawn.

    Attributes:
        css: Palette class, `s1` to `s6`.
        shape: One of circle, diamond, square or triangle.
        label: Legend and tooltip name.
    """

    css: str
    shape: str
    label: str


@dataclasses.dataclass(frozen=True)
class Mark:
    """One value on a row, with an optional interval.

    Attributes:
        series: Key into the panel's styles.
        value: Plotted value.
        lo: Lower end of the stored interval.
        hi: Upper end of the stored interval.
        hollow: Draw an open marker (post-trained checkpoints, failed tests).
        tip: Tooltip text; derived from the value when empty.
    """

    series: str
    value: float
    lo: float | None = None
    hi: float | None = None
    hollow: bool = False
    tip: str = ""


@dataclasses.dataclass(frozen=True)
class Row:
    """One labelled row of a dot panel.

    Attributes:
        label: Row label.
        marks: Values drawn on the row.
        band: Control range drawn behind the marks, as (low, high).
        group: Rows with a different group than the previous row are
            separated by a gap.
    """

    label: str
    marks: tuple[Mark, ...]
    band: tuple[float, float] | None = None
    group: str = ""


@dataclasses.dataclass(frozen=True)
class DotPanel:
    """A dot plot with one row per model or run.

    Attributes:
        uid: Unique id on the page, used for accessibility ids.
        title: Panel title, also the accessible name.
        rows: Rows from top to bottom.
        styles: Series key to style.
        domain: Axis range.
        ticks: Axis tick values.
        refs: Dashed reference lines as (value, label).
        band_label: What the grey band means, used in the description.
        fmt: Number format for tooltips and tables.
        stagger: Offset marks vertically so crowded rows stay readable.
        whiskers: Draw stored intervals as whiskers.
    """

    uid: str
    title: str
    rows: tuple[Row, ...]
    styles: dict[str, Style]
    domain: tuple[float, float]
    ticks: tuple[float, ...]
    refs: tuple[tuple[float, str], ...] = ()
    band_label: str = ""
    fmt: str = "{:.3f}"
    stagger: bool = False
    whiskers: bool = True


@dataclasses.dataclass(frozen=True)
class LineSeries:
    """One line of a line panel.

    Attributes:
        key: Series key.
        label: Legend and tooltip name.
        css: Palette class.
        points: (x, y) pairs in increasing x.
        dashed: Draw dashed (post-trained checkpoints).
    """

    key: str
    label: str
    css: str
    points: tuple[tuple[float, float], ...]
    dashed: bool = False


@dataclasses.dataclass(frozen=True)
class LinePanel:
    """Series over a numeric axis.

    Attributes:
        uid: Unique id on the page.
        title: Panel title, also the accessible name.
        series: Lines to draw.
        x_domain: Horizontal range.
        x_ticks: Horizontal tick values.
        y_domain: Vertical range.
        y_ticks: Vertical tick values.
        x_label: Horizontal axis name.
        y_log: Use a logarithmic vertical axis.
        refs: Dashed horizontal lines as (value, label).
        fmt: Number format for tooltips and tables.
    """

    uid: str
    title: str
    series: tuple[LineSeries, ...]
    x_domain: tuple[float, float]
    x_ticks: tuple[float, ...]
    y_domain: tuple[float, float]
    y_ticks: tuple[float, ...]
    x_label: str = ""
    y_log: bool = False
    refs: tuple[tuple[float, str], ...] = ()
    fmt: str = "{:.3f}"


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def _num(value: float) -> str:
    if abs(value) >= 10:
        return f"{value:g}"
    one = f"{value:.1f}"
    return one if float(one) == value else f"{value:.2f}"


def _scale(
    value: float, domain: tuple[float, float], lo: float, hi: float
) -> float:
    clamped = min(max(value, domain[0]), domain[1])
    return lo + (clamped - domain[0]) / (domain[1] - domain[0]) * (hi - lo)


def _shape(kind: str, x: float, y: float, r: float) -> str:
    if kind == "square":
        return (
            f'<rect class="shape" x="{x - r:.1f}" y="{y - r:.1f}" '
            f'width="{2 * r:.1f}" height="{2 * r:.1f}" rx="1"/>'
        )
    if kind == "diamond":
        r *= 1.25
        pts = f"{x:.1f},{y - r:.1f} {x + r:.1f},{y:.1f} "
        pts += f"{x:.1f},{y + r:.1f} {x - r:.1f},{y:.1f}"
        return f'<polygon class="shape" points="{pts}"/>'
    if kind == "triangle":
        r *= 1.2
        pts = f"{x:.1f},{y - r:.1f} {x + r:.1f},{y + r * 0.8:.1f} "
        pts += f"{x - r:.1f},{y + r * 0.8:.1f}"
        return f'<polygon class="shape" points="{pts}"/>'
    return f'<circle class="shape" cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}"/>'


def _tip(panel: DotPanel, row: Row, mark: Mark) -> str:
    if mark.tip:
        return f"{row.label}: {mark.tip}"
    style = panel.styles[mark.series]
    text = f"{row.label}: {style.label} {panel.fmt.format(mark.value)}"
    if mark.lo is not None and mark.hi is not None:
        text += f" [{panel.fmt.format(mark.lo)}, {panel.fmt.format(mark.hi)}]"
    return text


def _open(panel: DotPanel | LinePanel, height: float, desc: str) -> list[str]:
    return [
        f'<svg xmlns="{SVG_NS}" class="chart" viewBox="0 0 {WIDTH} {height}" '
        f'role="img" aria-labelledby="{panel.uid}-t {panel.uid}-d">',
        f'<title id="{panel.uid}-t">{_esc(panel.title)}</title>',
        f'<desc id="{panel.uid}-d">{desc} The same numbers are in the data '
        "table below the chart.</desc>",
    ]


def render_dot_panel(panel: DotPanel) -> str:
    """Renders a dot panel as an inline SVG string.

    Args:
        panel: Panel description.

    Returns:
        The SVG markup.

    Raises:
        KeyError: If a mark names a series missing from the panel styles.
    """
    gaps = []
    previous = None
    for row in panel.rows:
        new_group = previous is not None and row.group != previous
        gaps.append(GROUP_GAP if new_group else 0)
        previous = row.group
    axis_y = TOP + len(panel.rows) * ROW_H + sum(gaps) + 4
    names = ", ".join(_esc(s.label) for s in panel.styles.values())
    desc = f"{len(panel.rows)} rows; series: {names}."
    if panel.band_label:
        desc += f" Grey band: {_esc(panel.band_label)}."

    def x_of(value: float) -> float:
        return _scale(value, panel.domain, PLOT_L, PLOT_R)

    parts = _open(panel, axis_y + 24, desc)
    for tick in panel.ticks:
        x = x_of(tick)
        parts.append(
            f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{TOP - 4}" '
            f'y2="{axis_y}"/>'
        )
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{axis_y + 15}" '
            f'text-anchor="middle">{_num(tick)}</text>'
        )
    parts.append(
        f'<line class="axis" x1="{PLOT_L}" x2="{PLOT_R}" y1="{axis_y}" '
        f'y2="{axis_y}"/>'
    )
    for value, label in panel.refs:
        x = x_of(value)
        parts.append(
            f'<line class="ref" x1="{x:.1f}" x2="{x:.1f}" y1="{TOP - 4}" '
            f'y2="{axis_y}"/>'
        )
        parts.append(
            f'<text class="reflabel" x="{x + 3:.1f}" y="{TOP - 7}">'
            f"{_esc(label)}</text>"
        )
    offset = 0.0
    for row, gap in zip(panel.rows, gaps, strict=True):
        offset += gap
        cy = TOP + offset + ROW_H / 2
        offset += ROW_H
        parts.append(
            f'<text class="rowlabel" x="{LABEL_X}" y="{cy + 4:.1f}" '
            f'text-anchor="end">{_esc(row.label)}</text>'
        )
        if row.band is not None:
            x0, x1 = x_of(row.band[0]), x_of(row.band[1])
            parts.append(
                f'<rect class="band" x="{x0:.1f}" y="{cy - ROW_H / 2 + 3:.1f}" '
                f'width="{max(x1 - x0, 1.5):.1f}" height="{ROW_H - 6}" rx="2"/>'
            )
        for i, mark in enumerate(row.marks):
            style = panel.styles[mark.series]
            y = cy + ((i % 3) - 1) * STAGGER if panel.stagger else cy
            x = x_of(mark.value)
            if panel.whiskers and mark.lo is not None and mark.hi is not None:
                parts.append(
                    f'<line class="whisker {style.css}" '
                    f'x1="{x_of(mark.lo):.1f}" x2="{x_of(mark.hi):.1f}" '
                    f'y1="{y:.1f}" y2="{y:.1f}"/>'
                )
            hollow = " hollow" if mark.hollow else ""
            parts.append(
                f'<g class="mark {style.css}{hollow}" '
                f'data-series="{_esc(mark.series)}" '
                f'data-tip="{_esc(_tip(panel, row, mark))}">'
                f'<circle class="hit" cx="{x:.1f}" cy="{y:.1f}" r="9"/>'
                f"{_shape(style.shape, x, y, MARK_R)}</g>"
            )
    parts.append("</svg>")
    return "".join(parts)


def _y_map(
    panel: LinePanel, top: float, bottom: float
) -> Callable[[float], float]:
    if not panel.y_log:
        return lambda v: _scale(v, panel.y_domain, bottom, top)
    lo, hi = math.log(panel.y_domain[0]), math.log(panel.y_domain[1])
    return lambda v: _scale(math.log(max(v, 1e-12)), (lo, hi), bottom, top)


def render_line_panel(panel: LinePanel) -> str:
    """Renders a line panel as an inline SVG string.

    Args:
        panel: Panel description.

    Returns:
        The SVG markup.
    """
    left, right, top, bottom = 44, 330, 22, 168
    y_of = _y_map(panel, top, bottom)

    def x_of(value: float) -> float:
        return _scale(value, panel.x_domain, left, right)

    names = ", ".join(_esc(s.label) for s in panel.series)
    parts = _open(panel, bottom + 40, f"{len(panel.series)} lines: {names}.")
    for tick in panel.y_ticks:
        y = y_of(tick)
        parts.append(
            f'<line class="grid" x1="{left}" x2="{right}" y1="{y:.1f}" '
            f'y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{left - 6}" y="{y + 4:.1f}" '
            f'text-anchor="end">{_num(tick)}</text>'
        )
    for tick in panel.x_ticks:
        parts.append(
            f'<text class="tick" x="{x_of(tick):.1f}" y="{bottom + 16}" '
            f'text-anchor="middle">{_num(tick)}</text>'
        )
    parts.append(
        f'<line class="axis" x1="{left}" x2="{right}" y1="{bottom}" '
        f'y2="{bottom}"/>'
    )
    if panel.x_label:
        parts.append(
            f'<text class="axislabel" x="{(left + right) / 2:.1f}" '
            f'y="{bottom + 33}" text-anchor="middle">{_esc(panel.x_label)}'
            "</text>"
        )
    for value, label in panel.refs:
        y = y_of(value)
        parts.append(
            f'<line class="ref" x1="{left}" x2="{right}" y1="{y:.1f}" '
            f'y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text class="reflabel" x="{right}" y="{y - 4:.1f}" '
            f'text-anchor="end">{_esc(label)}</text>'
        )
    for s in panel.series:
        pts = " ".join(f"{x_of(x):.1f},{y_of(y):.1f}" for x, y in s.points)
        dashed = " dashed" if s.dashed else ""
        parts.append(
            f'<polyline class="line {s.css}{dashed}" '
            f'data-series="{_esc(s.key)}" points="{pts}"/>'
        )
    for s in panel.series:
        for x, y in s.points:
            tip = f"{s.label}: x {_num(x)}, y {panel.fmt.format(y)}"
            parts.append(
                f'<circle class="mark {s.css}" data-series="{_esc(s.key)}" '
                f'data-tip="{_esc(tip)}" cx="{x_of(x):.1f}" '
                f'cy="{y_of(y):.1f}" r="3"/>'
            )
    parts.append("</svg>")
    return "".join(parts)


def panel_table(panel: DotPanel | LinePanel) -> str:
    """Renders the numbers behind a panel as an HTML table.

    Args:
        panel: Dot or line panel.

    Returns:
        The table markup.
    """
    if isinstance(panel, LinePanel):
        xs = sorted({x for s in panel.series for x, _ in s.points})
        head = "".join(f"<th>{_esc(s.label)}</th>" for s in panel.series)
        lines = [f"<tr><th>{_esc(panel.x_label or 'x')}</th>{head}</tr>"]
        for x in xs:
            cells = []
            for s in panel.series:
                match = [y for px, y in s.points if px == x]
                cells.append(panel.fmt.format(match[0]) if match else "n/a")
            body = "".join(f"<td>{c}</td>" for c in cells)
            lines.append(f"<tr><td>{_num(x)}</td>{body}</tr>")
        return f"<table>{''.join(lines)}</table>"
    keys = list(panel.styles)
    head = "".join(f"<th>{_esc(panel.styles[k].label)}</th>" for k in keys)
    lines = [f"<tr><th>Row</th>{head}</tr>"]
    for row in panel.rows:
        cells = []
        for key in keys:
            found = [m for m in row.marks if m.series == key]
            if not found:
                cells.append("n/a")
                continue
            mark = found[0]
            text = panel.fmt.format(mark.value)
            if mark.lo is not None and mark.hi is not None:
                lo, hi = panel.fmt.format(mark.lo), panel.fmt.format(mark.hi)
                text += f" [{lo}, {hi}]"
            cells.append(text)
        body = "".join(f"<td>{_esc(c)}</td>" for c in cells)
        lines.append(f"<tr><td>{_esc(row.label)}</td>{body}</tr>")
    return f"<table>{''.join(lines)}</table>"


def legend(
    styles: dict[str, Style], extras: Sequence[tuple[str, str]] = ()
) -> str:
    """Renders an HTML legend for a set of series styles.

    Args:
        styles: Series key to style.
        extras: Extra keys as (kind, label), where kind is one of filled,
            hollow, band or dashed.

    Returns:
        The legend markup.
    """
    items = []
    for style in styles.values():
        swatch = _shape(style.shape, 7, 7, 4.5)
        items.append(
            f'<li class="{style.css}"><svg viewBox="0 0 14 14" width="14" '
            f'height="14" aria-hidden="true">{swatch}</svg>'
            f"{_esc(style.label)}</li>"
        )
    for kind, label in extras:
        items.append(f'<li class="key key-{kind}">{_esc(label)}</li>')
    return f'<ul class="legend">{"".join(items)}</ul>'
