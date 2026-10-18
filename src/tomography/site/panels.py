"""Chart descriptions shared by the findings and the figure renderer.

A dot panel has one row per checkpoint or run, with marks, optional intervals
and an optional control band. A line panel plots series over a numeric axis.
"""

import dataclasses


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
