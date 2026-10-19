"""Figure 4: how well a concept is read, against the checkpoint size.

One panel per concept (sentiment, utility, morality, truth). Each shows the
held-out accuracy at the layer read against the checkpoint size, with a 95
percent Wilson interval for the test pairs and the accuracy of a random
direction at the same layer as a grey tick. The layer curves behind the layer
read are in the appendix figure `a_reading_curves`. The sentiment dot is the
held-out value of the sample-size run at its own layer.
"""

import dataclasses

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np

from tomography.site import params
from tomography.site import reading
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.site.registry import RoleCount
from tomography.site.tables import _cells
from tomography.viz import axes as shared
from tomography.viz import paper

HEIGHT = 2.9
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


def _slope_text(numbers: dict[str, float], names: dict[str, str]) -> str:
    rising, falling, flat = [], [], []
    for concept in names:
        s = numbers[f"slope_{concept}"]
        lo = numbers[f"slope_{concept}_lo"]
        hi = numbers[f"slope_{concept}_hi"]
        if lo > 0:
            rising.append(
                f"{_cells.num(s)} for {concept} "
                f"({_cells.num(lo)} to {_cells.num(hi)})"
            )
        elif hi < 0:
            falling.append(
                f"{_cells.num(-s)} for {concept} "
                f"(slope {_cells.num(s)}, {_cells.num(lo)} to "
                f"{_cells.num(hi)})"
            )
        else:
            flat.append(
                f"the {concept} slope ({_cells.num(s)}, {_cells.num(lo)} to "
                f"{_cells.num(hi)})"
            )
    parts = []
    sizes = int(numbers["n_gpt2"])
    if rising:
        parts.append(
            f"across the {sizes} GPT-2 sizes accuracy rises by "
            + " and ".join(rising)
            + " per tenfold in parameters (95 percent intervals)"
        )
    if falling:
        parts.append(
            f"across the {sizes} GPT-2 sizes accuracy falls by "
            + " and ".join(falling)
            + " per tenfold in parameters (95 percent intervals)"
        )
    if flat:
        verb = "has an interval" if len(flat) == 1 else "have intervals"
        include = "includes" if len(flat) == 1 else "include"
        parts.append(" and ".join(flat) + f" {verb} that {include} zero")
    return "; ".join(parts)


def _numbers(
    found: dict[str, dict[str, Cell]],
    trends: dict[str, tuple[float, float, float]],
    counts: dict[str, dict[str, RoleCount]],
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
        for role, count in counts[concept].items():
            numbers[f"above_{concept}_{role}"] = count.hits
            numbers[f"above_{concept}_{role}_n"] = count.total
    numbers["n_gpt2"] = float(
        sum(stats.role(m) == "gpt2" for m in found["truth"])
    )
    numbers["truth_n"] = next(iter(found["truth"].values())).at.n
    return numbers


def _counts(
    found: dict[str, dict[str, Cell]],
) -> dict[str, dict[str, RoleCount]]:
    return {
        concept: stats.per_role(
            {
                m: stats.diff_interval(c.at.reader, c.at.random, c.at.n)[1] > 0
                for m, c in found[concept].items()
            }
        )
        for concept, _, _ in CONCEPTS
    }


def _list(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _claim(
    counts: dict[str, dict[str, RoleCount]], names: dict[str, str]
) -> str:
    full = [
        c for c in names if all(n.hits == n.total for n in counts[c].values())
    ]
    totals = {c: sum(n.total for n in counts[c].values()) for c in names}
    full = [c for c in full if totals[c] == totals[full[0]]] if full else []
    parts = []
    if full:
        total = totals[full[0]]
        parts.append(f"{_list(full)} on all {total} checkpoints")
    parts += [
        f"{c} on {stats.per_role_text(counts[c])}"
        for c in names
        if c not in full
    ]
    return (
        "The held-out accuracy of a logistic direction is above that of its "
        "random direction, with an interval excluding zero, for "
        + _list(parts)
        + "."
    )


def _csv(found: dict[str, dict[str, Cell]]) -> str:
    lines = ["panel,concept,model,x,y,lo,hi,baseline"]
    for concept, _, _ in CONCEPTS:
        for model, cell in found[concept].items():
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
    return (
        "Held-out accuracy at the layer read against the checkpoint size "
        "(S to XL: GPT-2; 0.6B to 4B: Qwen3; filled for GPT-2 and base, "
        "open and shifted right for post-trained), with 95 "
        f"percent Wilson intervals for the test pairs ({sizes['sentiment']:,} "
        f"sentiment, {sizes['utility']:,} utility, {sizes['morality']:,} "
        f"morality and {sizes['truth']:,} truth pairs) and the random "
        "direction at the same layer as a grey tick; the sentiment dot is "
        "the sample-size run at 64 training pairs, read at its own layer. "
        "GPT-2 slopes are least-squares over the four sizes against log "
        "parameters, with a t interval from those four points: "
        + _slope_text(numbers, names)
        + ". Layer curves and onsets are in [[fig:a_reading_curves]]."
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
            ms=MARKER,
            mfc=ink,
            mec=ink,
            mew=0.8,
        ),
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
        Line2D([], [], ls="none", marker="_", ms=6, mew=1.2, color=grey),
    ]
    fig.legend(
        handles,
        [
            "GPT-2 and Qwen3 base",
            "post-trained",
            "random direction at the layer read",
        ],
        loc="outside upper center",
        ncol=3,
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
    counts = _counts(found)
    numbers = _numbers(found, trends, counts)
    with paper.paper_style():
        fig = Figure(figsize=(paper.FULL_WIDTH, HEIGHT), layout="constrained")
        axes = fig.subplots(1, len(CONCEPTS), sharey=True)
        for col, (ax, (concept, label, _)) in enumerate(
            zip(axes, CONCEPTS, strict=True)
        ):
            ax.set_gid(f"size_{concept}")
            _size_panel(ax, found[concept])
            ax.set_title(label, fontsize=paper.TITLE, loc="left", x=0.14)
            paper.panel_letter(ax, "abcd"[col])
            if col == 0:
                ax.set_ylabel("held-out accuracy", fontsize=TEXT)
            else:
                ax.tick_params(labelleft=False)
        _key(fig)
    return FigureOutput(
        fig,
        _claim(counts, names),
        _detail(found, numbers, names),
        numbers,
        _csv(found),
        counts,
    )
