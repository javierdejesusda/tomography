"""Figure 4: where and how well a concept is read.

Row one is a heat map per concept with a row per checkpoint: the reader's
accuracy minus that of a random direction of the same run at every layer, with
the lowest layer above 0.1 marked and the layer read in row two circled. Row
two is the held-out accuracy at that layer against the number of parameters,
with the random direction at the same layer and a Wilson interval for the test
pairs. Sentiment is the held-out value of the sample-size run, not the
optimistic one of the layer-selection run.
"""

import dataclasses

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
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

HEIGHT = 4.4
MARGIN = 0.1
VMAX = 0.4
CONCEPTS = (
    ("sentiment", "Sentiment", "p4"),
    ("utility", "Utility", "r2"),
    ("morality", "Morality", "r3"),
    ("truth", "Truth", "r7"),
)
MIN_TREND_SIZES = 3
HALO = 1.6
TICK = 0.8
MARKER = 3.5


@dataclasses.dataclass(frozen=True)
class Cell:
    """One concept on one checkpoint.

    Attributes:
        curve: The reading over every layer.
        at: The reader and the random direction at the layer read.
    """

    curve: reading.Reading
    at: reading.Score


def _cells_of(manifest: Manifest, concept: str, exp: str) -> dict[str, Cell]:
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


def _heat(
    ax: Axes, cells: dict[str, Cell], rows: dict[str, float]
) -> AxesImage:
    image = None
    for model, cell in cells.items():
        y = rows[model]
        margin = cell.curve.selectivity
        image = ax.imshow(
            margin[None, :],
            extent=(0.0, 1.0, y + 0.5, y - 0.5),
            aspect="auto",
            cmap="cividis",
            vmin=0.0,
            vmax=VMAX,
            interpolation="nearest",
        )
        count = cell.curve.n_layers
        first = reading.first_depth_above(cell.curve, MARGIN)
        if first is not None:
            x = first - 0.5 / count
            ax.plot([x, x], [y - 0.45, y + 0.45], color="white", lw=HALO)
            ax.plot([x, x], [y - 0.45, y + 0.45], color=paper.INK, lw=TICK)
        ax.plot(
            [(cell.at.layer - 0.5) / count],
            [y],
            ls="none",
            marker="o",
            ms=2.6,
            mfc="white",
            mec=paper.INK,
            mew=0.6,
        )
    assert image is not None
    return image


def _math_minus(text: str) -> str:
    """Sets the true minus of a label in mathtext, which the text font lacks."""
    return text.replace(_cells.MINUS, f"${_cells.MINUS}$")


def _size(
    ax: Axes, cells: dict[str, Cell], trend: tuple[float, float, float]
) -> None:
    counts = [float(params.param_count(m)) for m in cells]
    shared.ref_line(ax, 0.5, "y")
    for model, cell in cells.items():
        style = shared.model_style(model)
        x = params.param_count(model) * 10 ** shared.role_dodge(model)
        lo, hi = stats.wilson(cell.at.reader, cell.at.n)
        ax.plot(
            [x, x],
            [cell.at.random, cell.at.reader],
            color=paper.RANDOM_BAND,
            lw=1.0,
            zorder=1.5,
        )
        ax.plot(
            [x, x],
            [lo, hi],
            color=style.color,
            lw=paper.WHISKER_WIDTH,
            zorder=3,
        )
        ax.plot(
            [x],
            [cell.at.random],
            ls="none",
            marker="x",
            ms=MARKER,
            mew=0.8,
            color=paper.ENC["random_marker"],
            zorder=2,
        )
        ax.plot(
            [x],
            [cell.at.reader],
            ls="none",
            marker="o",
            ms=MARKER,
            color=style.color,
            mfc=style.color if style.filled else "white",
            mew=0.8,
            zorder=4,
        )
    slope, lo, hi = trend
    gpt2 = [m for m in cells if stats.role(m) == "gpt2"]
    if len(gpt2) >= MIN_TREND_SIZES:
        logs = np.log10([params.param_count(m) for m in gpt2])
        ys = np.array([cells[m].at.reader for m in gpt2])
        ends = np.array([logs.min(), logs.max()])
        ax.plot(
            10**ends,
            ys.mean() + slope * (ends - logs.mean()),
            color=paper.INK,
            ls=paper.LINESTYLES["trend"],
            lw=paper.TREND_WIDTH,
            zorder=2,
        )
        ax.set_title(
            _math_minus(
                f"GPT-2 trend\n{'+' if slope >= 0 else ''}"
                f"{_cells.num(slope)} per decade\n"
                f"[{_cells.num(lo)}, {_cells.num(hi)}]"
            ),
            fontsize=6.5,
            loc="left",
            x=0.1,
            color=paper.INK,
        )
    shared.size_axis(ax, counts, ticks=(1e8, 1e9))
    ax.set_ylim(0.45, 1.0)
    shared.unit_axis(ax, "y")


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


def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:]


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
        "layer, one row per checkpoint; the colour of a cell is the margin "
        "(colour bar). The black tick marks the lowest layer where the "
        "margin exceeds 0.1 and the circle the layer read in (e to h), "
        "chosen on a validation split (sentiment: on the select split of "
        "the sample-size run). The onset is the earliest layer at which the "
        "margin curve crosses 0.1, not an estimate with an interval; for "
        f"truth the curve rests on {sizes['truth']} truth pairs. "
        "(e to h) Held-out accuracy at that layer against the number of "
        "parameters, with the random direction at the same layer (grey x) "
        "and 95 percent Wilson intervals for the "
        f"test pairs ({sizes['sentiment']:,} sentiment pairs from the "
        f"sample-size run at 64 training pairs, {sizes['utility']:,} "
        f"utility, {sizes['morality']:,} morality and {sizes['truth']:,} "
        "truth pairs); marker colour darkens with size within a family, "
        "filled for GPT-2 and Qwen3 base checkpoints and open for "
        "post-trained ones. The dashed line is the least-squares line "
        "through the GPT-2 sizes against the logarithm of the parameters; "
        "its interval is a t interval from the scatter of the GPT-2 points "
        "and does not include the sampling error of each point's "
        "accuracy. " + _sentence(_slope_text(numbers, names)) + tail + "."
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


def _keys(fig: Figure) -> None:
    handles = [
        Line2D([], [], color=paper.INK, lw=TICK),
        Line2D(
            [],
            [],
            ls="none",
            marker="o",
            ms=2.6,
            mfc="white",
            mec=paper.INK,
            mew=0.6,
        ),
    ]
    fig.legend(
        handles,
        ["lowest layer above 0.1", "layer read below"],
        loc="outside upper right",
        ncol=2,
        frameon=False,
        fontsize=6.5,
        columnspacing=1.0,
        handlelength=1.2,
    )


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    found = {c: _cells_of(manifest, c, exp) for c, _, exp in CONCEPTS}
    names = {c: label for c, label, _ in CONCEPTS}
    trends = {c: _trend(found[c]) for c in names}
    numbers = _numbers(found, trends)
    models = sorted(
        {m for cells in found.values() for m in cells}, key=store.model_rank
    )
    rows = shared.model_rows(models)
    with paper.paper_style():
        fig = Figure(figsize=(paper.FULL_WIDTH, HEIGHT), layout="constrained")
        grid = fig.add_gridspec(2, 4, height_ratios=[2.0, 1.35])
        image = None
        tops = []
        for col, (concept, label, _) in enumerate(CONCEPTS):
            top = fig.add_subplot(grid[0, col])
            bottom = fig.add_subplot(grid[1, col])
            tops.append(top)
            image = _heat(top, found[concept], rows)
            top.set_ylim(max(rows.values()) + 0.5, -0.5)
            top.set_xlim(0.0, 1.0)
            top.set_title(label, fontsize=paper.TITLE, loc="left", x=0.05)
            top.set_xticks([0.0, 0.5, 1.0], ["0", "0.5", "1"])
            top.set_xlabel("relative depth")
            top.tick_params(axis="y", length=0)
            if col == 0:
                top.set_yticks(
                    list(rows.values()),
                    [paper.display_name(m) for m in rows],
                    fontsize=6.5,
                )
            else:
                top.set_yticks([])
            paper.panel_letter(top, "abcd"[col])
            _size(bottom, found[concept], trends[concept])
            paper.panel_letter(bottom, "efgh"[col])
            if col:
                bottom.tick_params(labelleft=False)
        assert image is not None
        bar = fig.colorbar(
            image,
            ax=tops[0],
            orientation="horizontal",
            location="top",
            fraction=0.05,
            pad=0.04,
            shrink=0.9,
        )
        _keys(fig)
        bar.set_ticks([0.0, 0.1, 0.2, 0.3, 0.4])
        bar.ax.tick_params(labelsize=6.5)
        bar.set_label("reader minus random accuracy", fontsize=6.5)
        bar.outline.set_linewidth(0.4)
    return FigureOutput(
        fig,
        _claim(numbers, names),
        _detail(found, numbers, names),
        numbers,
        _csv(found),
    )
