"""Figure 4: where and how well a concept is read.

Rows one to three are the roles (GPT-2, Qwen3 base, Qwen3 post-trained), never
pooled, with a column per concept. Each panel draws, for every checkpoint of
the role, the reader's accuracy minus that of a random direction of the same
run against relative depth, with a dot at the layer read and a tick on the x
axis at the onset (the earliest layer above 0.1); the grey band is the stored
interval of the random direction over seeds, drawn around the zero line. Row
four is the held-out accuracy at the layer read against the checkpoint size,
with the random direction at the same layer as a grey tick. Sentiment curves
are the report-split curves of the finder comparison; its size dot is the
held-out value of the sample-size run.
"""

import dataclasses
import re

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np

from tomography.site import params
from tomography.site import reading
from tomography.site import stats
from tomography.site import store
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.site.tables import _cells
from tomography.viz import axes as shared
from tomography.viz import paper

HEIGHT = 6.0
MARGIN = 0.1
CONCEPTS = (
    ("sentiment", "Sentiment", "p4"),
    ("utility", "Utility", "r2"),
    ("morality", "Morality", "r3"),
    ("truth", "Truth", "r7"),
)
ROLE_ROWS = (
    ("gpt2", "GPT-2"),
    ("base", "Qwen3 base"),
    ("post", "Qwen3 post-trained"),
)
MIN_TREND_SIZES = 3
LINE = 1.1
DOT = 3.2
MARKER = 3.5
TEXT = paper.MIN_TEXT
ROLE_LABEL = 8
CURVE_LABEL_Y = 0.6
CURVE_YLIM = (-0.1, 0.5)
CURVE_TICKS = (0.0, 0.2, 0.4)
SIZE_YLIM = (0.4, 1.0)
SIZE_X = {
    "gpt2": 0.0,
    "gpt2-medium": 1.0,
    "gpt2-large": 2.0,
    "gpt2-xl": 3.0,
    "qwen3-0.6b": 4.0,
    "qwen3-1.7b": 5.0,
    "qwen3-4b": 6.0,
}
SIZE_TICKS = ("S", "M", "L", "XL", "0.6B", "1.7B", "4B")
POST_SHIFT = 0.22
GRID = np.linspace(0.0, 1.0, 101)
WIDTH_RATIOS = [1.0, 1.0, 1.0, 1.0, 0.34]
ROW_RATIOS = [1.0, 1.0, 1.0, 1.15]
TICK_HEIGHT = 0.035
GPT2_LABELS = {
    "gpt2": "S",
    "gpt2-medium": "M",
    "gpt2-large": "L",
    "gpt2-xl": "XL",
}


@dataclasses.dataclass(frozen=True)
class Cell:
    """One concept on one checkpoint.

    Attributes:
        curve: The reading over every layer.
        at: The reader and the random direction at the layer read.
    """

    curve: reading.Reading
    at: reading.Score


def cells_of(manifest: Manifest, concept: str, exp: str) -> dict[str, Cell]:
    """Loads every checkpoint of a concept.

    Args:
        manifest: The run resolver.
        concept: One of the concept keys of `CONCEPTS`.
        exp: The experiment the concept reads from.

    Returns:
        Checkpoint key to its cell, for the checkpoints with a run.
    """
    found: dict[str, Cell] = {}
    for model in manifest.models(exp):
        curve = (
            reading.load_sentiment(manifest, model)
            if concept == "sentiment"
            else reading.load(manifest, exp, model)
        )
        if curve is None:
            continue
        if concept == "sentiment":
            at = reading.heldout_sentiment(manifest, model)
        else:
            at = reading.Score(
                curve.layer,
                curve.accuracy,
                curve.random_at_layer,
                params.eval_pairs(manifest, exp, model),
            )
        if at is not None:
            found[model] = Cell(curve, at)
    return found


def _end_label(model: str) -> str:
    if model in GPT2_LABELS:
        return GPT2_LABELS[model]
    match = re.match(r"qwen3-(\d+(?:\.\d+)?)b", model)
    assert match is not None, model
    return f"{match.group(1)}B"


def _band(cells: dict[str, Cell]) -> tuple[np.ndarray, np.ndarray]:
    low, high = [], []
    for cell in cells.values():
        curve = cell.curve
        assert curve.random_lo is not None
        assert curve.random_hi is not None
        low.append(np.interp(GRID, curve.depth, curve.random_lo - curve.random))
        high.append(
            np.interp(GRID, curve.depth, curve.random_hi - curve.random)
        )
    return np.min(low, axis=0), np.max(high, axis=0)


def _curve_panel(ax: Axes, cells: dict[str, Cell]) -> None:
    low, high = _band(cells)
    band = ax.fill_between(
        GRID, low, high, color=paper.RANDOM_BAND, lw=0, zorder=1.5
    )
    band.set_gid("random_band")
    zero = ax.axhline(0.0, color=paper.INK, lw=0.5, zorder=2)
    zero.set_gid("zero")
    onsets, colours = [], []
    for model in sorted(cells, key=store.model_rank):
        cell = cells[model]
        colour = shared.model_style(model).color
        margin = cell.curve.selectivity
        (line,) = ax.plot(
            cell.curve.depth, margin, color=colour, lw=LINE, zorder=3
        )
        line.set_gid(f"curve:{model}")
        index = cell.at.layer - 1
        (dot,) = ax.plot(
            [cell.curve.depth[index]],
            [margin[index]],
            ls="none",
            marker=paper.MEASURE_MARKERS["direction"],
            ms=DOT,
            color=colour,
            mec=paper.INK,
            mew=0.5,
            zorder=4,
        )
        dot.set_gid(f"read:{model}")
        onset = reading.first_depth_above(cell.curve, MARGIN)
        if onset is not None:
            onsets.append(onset)
            colours.append(colour)
    floor = CURVE_YLIM[0]
    ticks = ax.vlines(
        onsets,
        floor,
        floor + TICK_HEIGHT * (CURVE_YLIM[1] - floor),
        colors=colours,
        lw=1.4,
        zorder=4,
    )
    ticks.set_gid("onset")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(*CURVE_YLIM)
    ax.set_xticks([0.0, 0.5, 1.0], ["0", "0.5", "1"])
    ax.set_yticks(CURVE_TICKS, ["0", "0.2", "0.4"])


def _label_column(ax: Axes, models: list[str]) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    count = len(models)
    for i, model in enumerate(sorted(models, key=store.model_rank)):
        y = 0.82 - i * (0.64 / max(count - 1, 1))
        ax.plot(
            [0.0, 0.28],
            [y, y],
            color=shared.model_style(model).color,
            lw=LINE + 0.4,
            transform=ax.transAxes,
            solid_capstyle="butt",
        )
        text = ax.text(
            0.34,
            y,
            _end_label(model),
            transform=ax.transAxes,
            fontsize=TEXT,
            va="center",
            ha="left",
            color=paper.INK,
        )
        text.set_gid("end_label")


def _is_post(model: str) -> bool:
    return model.startswith("qwen3") and not model.endswith("-base")


def _size_panel(ax: Axes, cells: dict[str, Cell]) -> None:
    for model, cell in cells.items():
        style = shared.model_style(model)
        x = SIZE_X[model.removesuffix("-base")]
        x += POST_SHIFT if _is_post(model) else 0.0
        lo, hi = stats.wilson(cell.at.reader, cell.at.n)
        (stem,) = ax.plot(
            [x, x],
            [cell.at.random, cell.at.reader],
            color=paper.RANDOM_BAND,
            lw=1.0,
            zorder=1.5,
        )
        stem.set_gid(f"stem:{model}")
        (tick,) = ax.plot(
            [x - 0.2, x + 0.2],
            [cell.at.random, cell.at.random],
            color=paper.ENC["random_marker"],
            lw=1.2,
            solid_capstyle="butt",
            zorder=2,
        )
        tick.set_gid(f"random:{model}")
        (whisker,) = ax.plot(
            [x, x],
            [lo, hi],
            color=style.color,
            lw=paper.WHISKER_WIDTH,
            zorder=3,
        )
        whisker.set_gid(f"whisker:{model}")
        (dot,) = ax.plot(
            [x],
            [cell.at.reader],
            ls="none",
            marker=paper.MEASURE_MARKERS["direction"],
            ms=MARKER,
            color=style.color,
            mfc=style.color if style.filled else "white",
            mew=0.8,
            zorder=4,
        )
        dot.set_gid(f"reader:{model}")
    ax.set_xlim(-0.6, 6.9)
    ax.set_ylim(*SIZE_YLIM)
    (chance,) = ax.plot(
        ax.get_xlim(),
        [0.5, 0.5],
        color=paper.ENC["reference"],
        ls=paper.LINESTYLES["reference"],
        lw=paper.REFERENCE_WIDTH,
        zorder=1,
    )
    chance.set_gid("chance")
    ax.set_xlim(-0.6, 6.9)
    ax.set_xticks(list(SIZE_X.values()), list(SIZE_TICKS))
    ax.set_yticks([0.5, 0.75, 1.0], ["0.5", "0.75", "1"])
    ax.tick_params(axis="x", length=2, pad=1.5, labelrotation=90)
    for x, label in ((1.5, "GPT-2"), (5.0, "Qwen3")):
        ax.annotate(
            label,
            xy=(x, 0.0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -26),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=TEXT,
            color=paper.ENC["secondary_text"],
            annotation_clip=False,
        )


def _trend(cells: dict[str, Cell]) -> tuple[float, float, float]:
    gpt2 = [m for m in cells if stats.role(m) == "gpt2"]
    if len(gpt2) < MIN_TREND_SIZES:
        return 0.0, 0.0, 0.0
    return stats.ols_slope(
        [float(np.log10(params.param_count(m))) for m in gpt2],
        [cells[m].at.reader for m in gpt2],
    )


def _numbers(
    found: dict[str, dict[str, Cell]],
    trends: dict[str, tuple[float, float, float]],
) -> dict[str, float]:
    numbers: dict[str, float] = {}
    for concept, _, _ in CONCEPTS:
        onsets = [
            d
            for c in found[concept].values()
            if (d := reading.first_depth_above(c.curve, MARGIN)) is not None
        ]
        if onsets:
            numbers[f"onset_{concept}_min"] = min(onsets)
            numbers[f"onset_{concept}_max"] = max(onsets)
        slope, lo, hi = trends[concept]
        numbers[f"slope_{concept}"] = slope
        numbers[f"slope_{concept}_lo"] = lo
        numbers[f"slope_{concept}_hi"] = hi
    numbers["n_gpt2"] = float(
        sum(stats.role(m) == "gpt2" for m in found["truth"])
    )
    numbers["truth_n"] = next(iter(found["truth"].values())).at.n
    return numbers


def _slope_text(numbers: dict[str, float], names: dict[str, str]) -> str:
    rising, flat = [], []
    for concept in names:
        s = numbers[f"slope_{concept}"]
        lo = numbers[f"slope_{concept}_lo"]
        hi = numbers[f"slope_{concept}_hi"]
        if lo > 0:
            rising.append(
                f"{_cells.num(s)} for {concept} "
                f"({_cells.num(lo)} to {_cells.num(hi)})"
            )
        else:
            flat.append(
                f"the {concept} slope ({_cells.num(s)}, {_cells.num(lo)} to "
                f"{_cells.num(hi)})"
            )
    parts = []
    if rising:
        parts.append(
            f"across the {int(numbers['n_gpt2'])} GPT-2 sizes accuracy "
            "rises by "
            + " and ".join(rising)
            + " per tenfold in parameters (95 percent intervals)"
        )
    if flat:
        verb = "has an interval" if len(flat) == 1 else "have intervals"
        include = "includes" if len(flat) == 1 else "include"
        parts.append(" and ".join(flat) + f" {verb} that {include} zero")
    return "; ".join(parts)


def _claim(numbers: dict[str, float], names: dict[str, str]) -> str:
    spans = [
        f"{numbers[f'onset_{c}_min']:.3f} to {numbers[f'onset_{c}_max']:.3f} "
        f"of the depth for {c}"
        for c in names
        if f"onset_{c}_min" in numbers
    ]
    return (
        "Where and how well a concept is read. The margin of the reader "
        "over a random direction reaches 0.1 at "
        + ", ".join(spans[:-1])
        + " and "
        + spans[-1]
        + "."
    )


def _csv(found: dict[str, dict[str, Cell]]) -> str:
    lines = ["panel,concept,model,x,y,lo,hi,baseline"]
    for concept, _, _ in CONCEPTS:
        for model, cell in found[concept].items():
            for depth, value in zip(
                cell.curve.depth, cell.curve.selectivity, strict=True
            ):
                lines.append(
                    f"depth,{concept},{model},{depth:.10g},{value:.10g},,,"
                )
            lo, hi = stats.wilson(cell.at.reader, cell.at.n)
            lines.append(
                f"size,{concept},{model},{params.param_count(model)},"
                f"{cell.at.reader:.10g},{lo:.10g},{hi:.10g},"
                f"{cell.at.random:.10g}"
            )
    return "\n".join(lines) + "\n"


def _detail(
    found: dict[str, dict[str, Cell]],
    numbers: dict[str, float],
    names: dict[str, str],
) -> str:
    sizes = {c: next(iter(found[c].values())).at.n for c in names}
    gpt2_truth = [
        stats.diff_interval(c.at.reader, c.at.random, c.at.n)[1]
        for m, c in found["truth"].items()
        if stats.role(m) == "gpt2"
    ]
    tail = ""
    if gpt2_truth and all(lo <= 0 for lo in gpt2_truth):
        tail = (
            f", and on GPT-2 the truth reader is not separable from its "
            f"random direction at {sizes['truth']} test pairs (Figure 1, "
            "row 3)"
        )
    return (
        "(a to d) Reader accuracy minus random-direction accuracy at every "
        "layer, one line per checkpoint in the panel of its role (rows: "
        "GPT-2, Qwen3 base, Qwen3 post-trained; darker lines are larger "
        "checkpoints, named in the right column of each row). The dot marks "
        "the layer read, chosen on a validation split (for sentiment the layer "
        "of the finder comparison; the size dot in (e) is the sample-size run "
        "at its own layer); the tick "
        "on the x axis marks the onset, the earliest layer at which the "
        "margin curve crosses 0.1, which is not an estimate with an "
        "interval; for truth the curve rests on "
        f"{sizes['truth']} truth pairs. The grey band is the stored 95 "
        "percent interval over seeds of the random direction, drawn around "
        "the zero line and spanning the checkpoints of the panel. Sentiment "
        "curves are the report-split accuracy of the logistic regression of "
        "the finder comparison, and for sentiment the size dot below is the "
        "held-out value of the sample-size run at 64 training pairs, read at "
        "its own layer. (e to h) "
        "Held-out accuracy at the layer read against the checkpoint size "
        "(S to XL: GPT-2 small to XL; 0.6, 1.7 and 4: Qwen3 in billions of "
        "parameters; filled for GPT-2 and Qwen3 base checkpoints, open and "
        "shifted right for post-trained ones), with 95 percent Wilson "
        f"intervals for the test pairs ({sizes['sentiment']:,} sentiment "
        f"pairs, {sizes['utility']:,} utility, {sizes['morality']:,} "
        f"morality and {sizes['truth']:,} truth pairs) and the random "
        "direction at the same layer as a grey tick; no fitted line is drawn. "
        "The GPT-2 slopes are least-squares slopes over the four GPT-2 sizes "
        "against the logarithm of the parameters, with a t interval from the "
        "scatter of those four points that does not include the sampling "
        "error of each point's accuracy: "
        + _slope_text(numbers, names)
        + tail
        + "."
    )


def _key(fig: Figure) -> None:
    ink = paper.INK
    grey = paper.ENC["random_marker"]
    handles = [
        Line2D(
            [],
            [],
            ls="none",
            marker=paper.MEASURE_MARKERS["direction"],
            ms=DOT,
            mfc=ink,
            mec=ink,
        ),
        Line2D([], [], ls="none", marker="|", ms=6, mew=1.4, color=ink),
        Line2D([], [], color=paper.RANDOM_BAND, lw=5, solid_capstyle="butt"),
        Line2D([], [], ls="none", marker="_", ms=6, mew=1.2, color=grey),
        Line2D(
            [],
            [],
            ls="none",
            marker=paper.MEASURE_MARKERS["direction"],
            ms=MARKER,
            mfc="white",
            mec=ink,
            mew=0.8,
        ),
    ]
    fig.legend(
        handles,
        [
            "layer read (a to d); GPT-2 and Qwen3 base (e to h)",
            "onset (earliest layer above 0.1)",
            "random direction, 95 percent interval",
            "random direction at the layer read",
            "post-trained (e to h)",
        ],
        loc="outside upper center",
        ncol=2,
        frameon=False,
        fontsize=TEXT,
        columnspacing=1.0,
        handletextpad=0.4,
    )


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    found = {c: cells_of(manifest, c, exp) for c, _, exp in CONCEPTS}
    names = {c: label for c, label, _ in CONCEPTS}
    trends = {c: _trend(found[c]) for c in names}
    numbers = _numbers(found, trends)
    with paper.paper_style():
        fig = Figure(figsize=(paper.FULL_WIDTH, HEIGHT), layout="constrained")
        grid = fig.add_gridspec(
            4, 5, width_ratios=WIDTH_RATIOS, height_ratios=ROW_RATIOS
        )
        for r, (role, role_name) in enumerate(ROLE_ROWS):
            members: list[str] = []
            for col, (concept, label, _) in enumerate(CONCEPTS):
                ax = fig.add_subplot(grid[r, col])
                cells = {
                    m: c
                    for m, c in found[concept].items()
                    if stats.role(m) == role
                }
                members = list(cells)
                ax.set_gid(f"curve_{role}_{concept}")
                _curve_panel(ax, cells)
                if r == 0:
                    ax.set_title(
                        label, fontsize=paper.TITLE, loc="left", x=0.14
                    )
                    paper.panel_letter(ax, "abcd"[col])
                if r < len(ROLE_ROWS) - 1:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel("relative depth")
                if col == 0:
                    ax.set_ylabel(role_name, fontsize=ROLE_LABEL)
                else:
                    ax.tick_params(labelleft=False)
            label_ax = fig.add_subplot(grid[r, 4])
            label_ax.set_gid(f"labels_{role}")
            _label_column(label_ax, members)
        for col, (concept, _, _) in enumerate(CONCEPTS):
            ax = fig.add_subplot(grid[3, col])
            ax.set_gid(f"size_{concept}")
            _size_panel(ax, found[concept])
            paper.panel_letter(ax, "efgh"[col])
            if col == 0:
                ax.set_ylabel("held-out accuracy", fontsize=TEXT)
            else:
                ax.tick_params(labelleft=False)
        shared_label = fig.supylabel(
            "reader minus random accuracy", y=CURVE_LABEL_Y, fontsize=ROLE_LABEL
        )
        shared_label.set_gid("curve_ylabel")
        _key(fig)
    return FigureOutput(
        fig,
        _claim(numbers, names),
        _detail(found, numbers, names),
        numbers,
        _csv(found),
    )
