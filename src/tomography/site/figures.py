"""Paper-style matplotlib figures for the results site.

Each `findings.Figure` becomes one matplotlib figure that follows the print
standard of `tomography.viz.paper`: 5.5 in wide, serif type of 7 to 8.5 pt,
panel letters, one legend above the panels and grey random bands. The same
objects give the CSV of the plotted numbers.
"""

from collections.abc import Sequence
import csv
import io
import math
import textwrap

from matplotlib import ticker
from matplotlib import transforms
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle

from tomography.site import findings
from tomography.site import panels
from tomography.viz import paper

Panel = panels.DotPanel | panels.LinePanel
Handle = Line2D | Patch

GROUP_GAP = 0.6
LEGEND_ROW = 0.17
TITLE_BAND = 0.22
TITLE_LINE = 0.125
X_BAND = 0.24
X_LABEL_BAND = 0.16
ROW_GAP = 0.1
LINE_HEIGHT = 1.35
LEFT_PAD = 0.07
RIGHT_PAD = 0.1
COLUMN_GAP = 0.14
FAMILY_BAND = 0.15
TICK_ROOM = 0.07
LINE_Y_ROOM = 0.34
CHAR_WIDTH = 0.056
DOMAIN_PAD = 0.03

COLORS = {
    "s0": paper.CHANCE,
    "s1": paper.SERIES[0],
    "s2": paper.SERIES[1],
    "s3": paper.SERIES[2],
    "s4": paper.SERIES[3],
    "s5": paper.NEUTRALS[0],
    "s6": paper.NEUTRALS[1],
}
MARKERS = {"circle": "o", "diamond": "D", "square": "s", "triangle": "^"}
FAMILIES = {"gpt2": "GPT-2", "qwen3": "Qwen3"}
PREFIXES = {"gpt2": "GPT-2 ", "qwen3": "Qwen3-"}


def _measure(strings: Sequence[str]) -> float:
    """Measures the widest string in inches at the tick size."""
    fig = Figure()
    renderer = FigureCanvasAgg(fig).get_renderer()  # type: ignore[no-untyped-call]
    widest = 0.0
    for text in strings:
        artist = fig.text(0, 0, text, fontsize=paper.TICK)
        widest = max(widest, artist.get_window_extent(renderer).width)
    return widest / fig.dpi


def _tick_text(value: float) -> str:
    return "0" if value == 0 else f"{value:.3g}"


def _row_label(row: panels.Row) -> str:
    return paper.display_name(row.label)


def _wrap(title: str, width: float) -> str:
    return "\n".join(textwrap.wrap(title, max(8, int(width / CHAR_WIDTH))))


def _dot_slots(rows: Sequence[panels.Row]) -> tuple[list[float], float]:
    """Places rows on a vertical scale with a gap between groups."""
    positions: list[float] = []
    pos = 0.0
    for i, row in enumerate(rows):
        if i and row.group != rows[i - 1].group:
            pos += GROUP_GAP
        positions.append(pos)
        pos += 1.0
    return positions, pos


class _Block:
    """Layout of one grid row of panels, in inches."""

    def __init__(self, items: Sequence[Panel], width: float) -> None:
        self.items = items
        self.is_dot = isinstance(items[0], panels.DotPanel)
        self.base: Sequence[panels.Row] = ()
        self.shared = False
        self.positions: list[float] = []
        self.span = 0.0
        self.label_w: list[float] = []
        self.families: list[bool] = []
        if self.is_dot:
            self._plan_dots(width)
        else:
            self.span = LINE_HEIGHT / paper.ROW_HEIGHT
        self.wide_x = any(
            isinstance(p, panels.LinePanel) and p.x_label for p in items
        )

    def _plan_dots(self, width: float) -> None:
        dots = [p for p in self.items if isinstance(p, panels.DotPanel)]
        base = max(dots, key=lambda p: len(p.rows)).rows
        names = {r.label for r in base}
        self.shared = {r.label for p in dots for r in p.rows} <= names
        self.base = base
        sources = [base] if self.shared else [p.rows for p in dots]
        self.positions, self.span = _dot_slots(base)
        for rows in sources:
            strip = bool(rows) and all(
                r.group in PREFIXES
                and paper.display_name(r.label).startswith(PREFIXES[r.group])
                for r in rows
            )
            self.families.append(strip)
            labels = [self.label(r, strip) for r in rows]
            self.label_w.append(_measure(labels))

    def label(self, row: panels.Row, strip: bool) -> str:
        name = _row_label(row)
        return name.removeprefix(PREFIXES[row.group]) if strip else name

    def plot_height(self) -> float:
        return self.span * paper.ROW_HEIGHT

    def title_lines(self, plot_w: float) -> int:
        return max(_wrap(p.title, plot_w).count("\n") + 1 for p in self.items)


def _color_of(css: str) -> str:
    return COLORS[css]


def _draw_dots(
    ax: Axes, panel: panels.DotPanel, rows: Sequence[panels.Row]
) -> None:
    positions, total = _dot_slots(rows)
    pad = DOMAIN_PAD * (panel.domain[1] - panel.domain[0])
    ax.set_xlim(panel.domain[0] - pad, panel.domain[1] + pad)
    ax.set_ylim(total - 0.5, -0.5)
    ax.set_xticks(panel.ticks)
    ax.set_xticklabels([_tick_text(t) for t in panel.ticks])
    for value, _label in panel.refs:
        ax.axvline(value, color=paper.CHANCE, lw=0.5, ls=":", zorder=2)
    by_label = {r.label: r for r in panel.rows}
    for row, pos in zip(rows, positions, strict=True):
        own = by_label.get(row.label)
        if own is None:
            continue
        if own.band is not None:
            lo, hi = own.band
            ax.add_patch(
                Rectangle(
                    (lo, pos - 0.38),
                    hi - lo,
                    0.76,
                    fc=paper.RANDOM_BAND,
                    ec="none",
                    zorder=1,
                )
            )
        count = len(own.marks)
        for i, mark in enumerate(own.marks):
            style = panel.styles[mark.series]
            y = pos + (i - (count - 1) / 2) * 0.28 if panel.stagger else pos
            color = _color_of(style.css)
            if panel.whiskers and mark.lo is not None and mark.hi is not None:
                ax.plot(
                    [mark.lo, mark.hi],
                    [y, y],
                    color=color,
                    lw=0.8,
                    solid_capstyle="butt",
                    zorder=3,
                )
            ax.plot(
                [mark.value],
                [y],
                ls="none",
                marker=MARKERS[style.shape],
                ms=3.5 if style.shape != "diamond" else 3.0,
                mfc="white" if mark.hollow else color,
                mec=color,
                mew=0.8,
                zorder=4,
            )


def _line_color(panel: panels.LinePanel) -> dict[str, str]:
    names = [s.label for s in panel.series]
    if not all(n.startswith(("gpt2", "qwen3")) for n in names):
        return {s.css: _color_of(s.css) for s in panel.series}
    fam = "gpt2" if names[0].startswith("gpt2") else "qwen3"
    order = sorted({s.css for s in panel.series})
    ramp = paper.size_ramp(fam, len(order))
    return {css: ramp[i] for i, css in enumerate(order)}


def _draw_lines(ax: Axes, panel: panels.LinePanel) -> None:
    colors = _line_color(panel)
    if panel.y_log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _pos: f"{v:g}")
        )
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlim(*panel.x_domain)
    ax.set_ylim(*panel.y_domain)
    ax.set_xticks(panel.x_ticks)
    ax.set_xticklabels([_tick_text(t) for t in panel.x_ticks])
    ax.set_yticks(panel.y_ticks)
    ax.set_yticklabels([_tick_text(t) for t in panel.y_ticks])
    ax.yaxis.grid(True, lw=0.3, color="#e6e6e6")
    ax.set_axisbelow(True)
    for value, _label in panel.refs:
        ax.axhline(value, color=paper.CHANCE, lw=0.5, ls=":", zorder=2)
    for line in panel.series:
        xs = [p[0] for p in line.points]
        ys = [p[1] for p in line.points]
        color = colors[line.css]
        ax.plot(
            xs,
            ys,
            color=color,
            lw=1.0,
            ls="--" if line.dashed else "-",
            marker="o",
            ms=2.5,
            mfc="white" if line.dashed else color,
            mec=color,
            mew=0.6,
            zorder=3,
        )
    if panel.x_label:
        ax.set_xlabel(panel.x_label, labelpad=1.5)


def _legend_handles(figure: findings.Figure) -> list[Handle]:
    handles: list[Handle] = []
    seen: set[str] = set()

    def add(label: str, handle: Handle) -> None:
        if label not in seen:
            seen.add(label)
            handles.append(handle)

    for panel in figure.panels:
        if isinstance(panel, panels.DotPanel):
            for style in panel.styles.values():
                add(
                    style.label,
                    Line2D(
                        [],
                        [],
                        ls="none",
                        marker=MARKERS[style.shape],
                        ms=3.5,
                        mfc=_color_of(style.css),
                        mec=_color_of(style.css),
                        mew=0.8,
                        label=style.label,
                    ),
                )
        else:
            colors = _line_color(panel)
            for line in panel.series:
                size = line.label.removesuffix("-base")
                name = paper.display_name(size)
                add(
                    name,
                    Line2D([], [], color=colors[line.css], lw=1.0, label=name),
                )
    for kind, label in figure.keys:
        add(label, _key_handle(kind, label))
    for panel in figure.panels:
        for _value, label in panel.refs:
            add(
                label,
                Line2D([], [], color=paper.CHANCE, lw=0.5, ls=":", label=label),
            )
    return handles


def _key_handle(kind: str, label: str) -> Handle:
    if kind == "band":
        return Patch(fc=paper.RANDOM_BAND, ec="none", label=label)
    if kind == "dashed":
        return Line2D([], [], color=paper.INK, lw=1.0, ls="--", label=label)
    return Line2D(
        [],
        [],
        ls="none",
        marker="o",
        ms=3.5,
        mfc="white" if kind == "hollow" else paper.INK,
        mec=paper.INK,
        mew=0.8,
        label=label,
    )


def _legend_rows(handles: Sequence[Handle]) -> int:
    """Picks the fewest rows in which the legend fits the text width."""
    widths = [_measure([str(h.get_label())]) + 0.33 for h in handles]
    room = paper.FULL_WIDTH - LEFT_PAD - RIGHT_PAD
    for rows in (1, 2):
        total = sum(
            max(widths[start : start + rows])
            for start in range(0, len(widths), rows)
        )
        if total <= room:
            return rows
    return 3


def _plan_widths(
    block: _Block, columns: int, total: float
) -> tuple[list[float], float]:
    """Gives the label area of each panel and the shared plot width."""
    areas: list[float] = []
    for i in range(columns):
        if block.is_dot:
            if block.shared:
                width = block.label_w[0] if i == 0 else 0.0
                fam = FAMILY_BAND if block.families[0] and i == 0 else 0.0
            elif i < len(block.label_w):
                width = block.label_w[i]
                fam = FAMILY_BAND if block.families[i] else 0.0
            else:
                width, fam = block.label_w[-1], 0.0
            areas.append(fam + width + (TICK_ROOM if width else 0.0))
        else:
            areas.append(LINE_Y_ROOM)
    plot_w = (total - sum(areas) - (columns - 1) * COLUMN_GAP) / columns
    return areas, plot_w


def render(figure: findings.Figure) -> Figure:
    """Draws a finding's figure in the paper style.

    Args:
        figure: The panels, legend keys and layout of one figure.

    Returns:
        A matplotlib figure 5.5 in wide whose height follows the content.
    """
    with paper.paper_style():
        return _render(figure)


def _render(figure: findings.Figure) -> Figure:
    columns = max(1, figure.columns)
    items = list(figure.panels)
    blocks = [
        _Block(items[i : i + columns], paper.FULL_WIDTH)
        for i in range(0, len(items), columns)
    ]
    total = paper.FULL_WIDTH - LEFT_PAD - RIGHT_PAD
    handles = _legend_handles(figure)
    legend_rows = _legend_rows(handles) if handles else 0
    header = legend_rows * LEGEND_ROW + (0.1 if legend_rows else 0.0)
    plans = [_plan_widths(b, columns, total) for b in blocks]
    heights = []
    for block, (_areas, plot_w) in zip(blocks, plans, strict=True):
        lines = block.title_lines(plot_w)
        heights.append(
            TITLE_BAND
            + (lines - 1) * TITLE_LINE
            + block.plot_height()
            + X_BAND
            + (X_LABEL_BAND if block.wide_x else 0.0)
            + ROW_GAP
        )
    height = header + sum(heights) + 0.04
    fig = Figure(figsize=(paper.FULL_WIDTH, height))
    FigureCanvasAgg(fig)
    y_top = header
    letter = iter("abcdefghijklmnopqrstuvwxyz")
    if handles:
        fig.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(LEFT_PAD / paper.FULL_WIDTH, 1 - 0.04 / height),
            ncol=math.ceil(len(handles) / legend_rows),
            frameon=False,
            borderaxespad=0.0,
            borderpad=0.0,
            handlelength=1.5,
            handletextpad=0.5,
            columnspacing=1.3,
            labelspacing=0.3,
        )
    for block, (areas, plot_w), block_h in zip(
        blocks, plans, heights, strict=True
    ):
        lines = block.title_lines(plot_w)
        top = y_top + TITLE_BAND + (lines - 1) * TITLE_LINE
        x = LEFT_PAD
        for col, panel in enumerate(block.items):
            x += areas[col]
            ax = fig.add_axes(
                (
                    x / paper.FULL_WIDTH,
                    1 - (top + block.plot_height()) / height,
                    plot_w / paper.FULL_WIDTH,
                    block.plot_height() / height,
                )
            )
            if isinstance(panel, panels.DotPanel):
                _style_dots(ax, panel, block, col)
            else:
                _draw_lines(ax, panel)
            paper.panel_letter(ax, next(letter))
            ax.annotate(
                _wrap(panel.title, plot_w),
                xy=(0.0, 1.0),
                xycoords="axes fraction",
                xytext=(3, 5),
                textcoords="offset points",
                fontsize=paper.LABEL,
                ha="left",
                va="bottom",
                linespacing=1.0,
            )
            x += plot_w + COLUMN_GAP
        y_top += block_h
    return fig


def _style_dots(
    ax: Axes, panel: panels.DotPanel, block: _Block, col: int
) -> None:
    rows = block.base if block.shared else panel.rows
    _draw_dots(ax, panel, rows)
    positions, _total = _dot_slots(rows)
    ax.set_yticks(positions)
    index = 0 if block.shared else min(col, len(block.families) - 1)
    strip = block.families[index]
    show = col == 0 or not block.shared
    ax.set_yticklabels([block.label(r, strip) for r in rows] if show else [])
    if not show:
        ax.tick_params(axis="y", labelleft=False)
    if show and strip:
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        width_axes = ax.get_position().width * paper.FULL_WIDTH
        offset = (block.label_w[index] + TICK_ROOM + 0.075) / width_axes
        groups: dict[str, list[float]] = {}
        for row, pos in zip(rows, positions, strict=True):
            groups.setdefault(row.group, []).append(pos)
        for group, spots in groups.items():
            ax.text(
                -offset,
                (spots[0] + spots[-1]) / 2,
                FAMILIES[group],
                transform=trans,
                rotation=90,
                fontsize=paper.TICK,
                ha="center",
                va="center",
            )


def csv_text(figure: findings.Figure) -> str:
    """Lists the exact numbers a figure plots, one per line.

    Args:
        figure: The figure.

    Returns:
        CSV with a header; dot panels fill `row` and `value` with the
        interval and band, line panels fill `row` with the series and `x`.
    """
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(
        [
            "panel",
            "row",
            "series",
            "x",
            "value",
            "lo",
            "hi",
            "band_lo",
            "band_hi",
        ]
    )
    for panel in figure.panels:
        if isinstance(panel, panels.DotPanel):
            for row in panel.rows:
                band = row.band or ("", "")
                for mark in row.marks:
                    writer.writerow(
                        [
                            panel.title,
                            paper.display_name(row.label),
                            panel.styles[mark.series].label,
                            "",
                            repr(mark.value),
                            "" if mark.lo is None else repr(mark.lo),
                            "" if mark.hi is None else repr(mark.hi),
                            band[0],
                            band[1],
                        ]
                    )
        else:
            for line in panel.series:
                for x, y in line.points:
                    writer.writerow(
                        [
                            panel.title,
                            paper.display_name(line.label),
                            panel.x_label,
                            repr(x),
                            repr(y),
                            "",
                            "",
                            "",
                            "",
                        ]
                    )
    return out.getvalue()


def svg(figure: findings.Figure) -> bytes:
    """Renders a figure as deterministic SVG bytes."""
    fig = render(figure)
    with paper.paper_style():
        return paper.svg_bytes(fig)


def pdf(figure: findings.Figure) -> bytes:
    """Renders a figure as deterministic vector PDF bytes."""
    fig = render(figure)
    with paper.paper_style():
        return paper.pdf_bytes(fig)
