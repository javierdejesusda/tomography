"""Figure 11: What post-training changes in the readings.

One row per measure, each with the post-trained minus base difference of the
three Qwen3 sizes and a conservative 95 percent interval for the test items.
"""

import math
from typing import Any

from matplotlib import ticker
from matplotlib.lines import Line2D

from tomography.site import params
from tomography.site import reading
from tomography.site import runfiles
from tomography.site import stats
from tomography.site.figs import _group
from tomography.site.figs import _rows
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import axes
from tomography.viz import paper

HEIGHT = 5.4
LABEL_FONT = 7
LABEL_PAD = 0.07
HEADER = ["panel", "row", "size", "value", "lo", "hi", "interval"]
NARROW = 0.15
WIDE = 0.3
DODGE = 0.3
ROWS = (
    ("sentiment", "sentiment, held out", "sentiment"),
    ("utility", "utility", "utility"),
    ("morality", "morality", "morality"),
    ("truth", "truth", "truth"),
    ("honesty", "honesty, transfer", "honesty transfer"),
    ("truthfulqa", "TruthfulQA", "TruthfulQA"),
    ("knowledge", "knowledge", "knowledge AUROC"),
    ("steering", "steering", "steering win rate"),
)
SECTIONS = {
    "sentiment": "reading accuracy",
    "utility": "reading accuracy",
    "morality": "reading accuracy",
    "truth": "reading accuracy",
    "honesty": "reading accuracy",
    "truthfulqa": "reading accuracy",
    "knowledge": "knowledge AUROC",
    "steering": "steering win rate",
}
GROUP_GAP = 1
CONSERVATIVE_DASH = (0, (6, 3))
DASH_GAP = 3.0
MIN_DASHES = 2
OTHER = ("utility", "morality", "truth", "honesty", "truthfulqa")
PAIRED = {"r2": "utility", "r3": "morality", "r5": "honesty", "r7": "truth"}
CLAIM = (
    "Post-training lowers held-out sentiment reading; on the other readings "
    "most intervals include zero, which is not evidence of no change."
)
DETAIL = (
    "Sentiment falls by {low} to {high}, with an interval below zero on "
    "{below} of 3 sizes; utility, morality, truth, honesty transfer and "
    "TruthfulQA move by at most {most}, with {excluding}; {falling}. "
    "Post-trained minus base checkpoint for each Qwen3 size (0.6B, 1.7B and "
    "4B, light to dark). A marker is filled where the interval excludes "
    "zero, open otherwise. A solid interval is the "
    "seed-paired 95 percent bootstrap interval that [[table:base_post]] "
    "stores (10 seeds, 0.6B and 1.7B); a dashed interval is a conservative "
    "95 percent interval, used for 4B and the measures without seeds. The "
    "axes differ between the panels: (a) shows the rows whose intervals "
    "stay within {narrow}, (b) the rows with wider intervals, on {wide}. "
    "A post-trained checkpoint also differs from its base in prompt format "
    "and distillation, so the contrast is with this recipe "
    "([[sec:limitations]])."
)


def _accuracy(manifest: Manifest, row: str, model: str) -> tuple[float, int]:
    if row == "sentiment":
        score = reading.heldout_sentiment(manifest, model)
        assert score is not None
        return score.reader, score.n
    exp = {"utility": "r2", "morality": "r3", "truth": "r7", "honesty": "r5"}
    found = reading.load(manifest, exp[row], model)
    assert found is not None
    return found.accuracy, params.eval_pairs(manifest, exp[row], model)


def _table(manifest: Manifest, exp: str, model: str) -> list[dict[str, Any]]:
    run = manifest.run(exp, model)
    assert run is not None
    return runfiles.rows(run, "table.json")


def _truthfulqa(manifest: Manifest, model: str) -> tuple[float, int]:
    (row,) = [
        r
        for r in _table(manifest, "r6", model)
        if r["method"] == "lat"
        and r["layer_rule"] == "validation-layer"
        and r["source"] == "truthfulqa"
    ]
    return float(row["mc1"]), params.eval_pairs(manifest, "r6", model)


def _knowledge(manifest: Manifest, model: str) -> dict[str, Any]:
    (row,) = [
        r for r in _table(manifest, "p9b", model) if r["type"] == "obscure_real"
    ]
    return row


def _steering(manifest: Manifest, model: str) -> dict[str, Any]:
    (row,) = [
        r
        for r in _table(manifest, "m1", model)
        if r["task"] == "manipulation"
        and r["setting"] == "0.3"
        and r["method"] == "logistic"
    ]
    return row


def _difference(
    manifest: Manifest, row: str, size: str
) -> tuple[float, float, float]:
    base, post = f"qwen3-{size}-base", f"qwen3-{size}"
    if row == "truthfulqa":
        value, n = _truthfulqa(manifest, post)
        return stats.diff_interval(value, _truthfulqa(manifest, base)[0], n)
    if row == "knowledge":
        a, b = _knowledge(manifest, post), _knowledge(manifest, base)
        half = stats.Z * math.hypot(
            *(
                stats.hanley_mcneil_se(
                    r["logistic_within"],
                    r["n_positive_test"],
                    r["n_negative_test"],
                )
                for r in (a, b)
            )
        )
        d = a["logistic_within"] - b["logistic_within"]
        return d, d - half, d + half
    if row == "steering":
        a, b = _steering(manifest, post), _steering(manifest, base)
        return stats.diff_interval(a["value"], b["value"], a["n"], b["n"])
    value, n = _accuracy(manifest, row, post)
    return stats.diff_interval(value, _accuracy(manifest, row, base)[0], n)


def _paired_rows(manifest: Manifest) -> list[dict[str, Any]]:
    """Reads the stored seed-paired differences the figure counts over.

    Args:
        manifest: The run resolver.

    Returns:
        One record per measure and size; held-out sentiment is not stored
        among them.
    """
    found: list[dict[str, Any]] = []
    for model in manifest.models("p8"):
        run = manifest.run("p8", model)
        assert run is not None
        found.extend(
            r for r in runfiles.rows(run, "table.json") if r["source"] in PAIRED
        )
    return found


def _interval(
    manifest: Manifest,
    paired: dict[tuple[str, str], dict[str, Any]],
    row: str,
    size: str,
) -> tuple[float, float, float, str]:
    """Gives a difference with the sharpest interval the results store.

    Args:
        manifest: The run resolver.
        paired: The seed-paired records by (measure, size such as `0.6b`).
        row: The measure key.
        size: The Qwen3 size such as `0.6b`.

    Returns:
        The post-trained minus base value, its lower and upper bound, and
        `paired` or `conservative` for the kind of interval.
    """
    stored = paired.get((row, size))
    if stored is not None:
        return (
            float(stored["delta"]),
            float(stored["lo"]),
            float(stored["hi"]),
            "paired",
        )
    value, lo, hi = _difference(manifest, row, size)
    return value, lo, hi, "conservative"


def _paired_excluding(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Finds the seed-paired differences whose interval excludes zero.

    Args:
        rows: The records of `_paired_rows`.

    Returns:
        (measure name, size such as `0.6B`) in size then measure order.
    """
    measures = list(PAIRED.values())
    found = [
        (PAIRED[r["source"]], str(r["tuned"]).split("-")[1].upper())
        for r in rows
        if r["lo"] > 0 or r["hi"] < 0
    ]
    return sorted(found, key=lambda f: (float(f[1][:-1]), measures.index(f[0])))


def _signed(value: float) -> str:
    text = _rows.number(value)
    if float(text.replace(axes.MINUS, "-")) == 0.0:
        return text
    return text if text.startswith(axes.MINUS) else f"+{text}"


def _sizes_text(sizes: list[str]) -> str:
    return stats.join_and([s.upper() for s in sizes])


def _excluding_text(found: list[tuple[str, str, float]]) -> str:
    if not found:
        return "no interval excluding zero"
    names = {key: phrase for key, _, phrase in ROWS}
    parts = [f"{names[r]} on {s.upper()} ({_signed(v)})" for r, s, v in found]
    return f"an interval excluding zero only for {stats.join_and(parts)}"


def _falling_text(knowledge: list[str], steering: list[str]) -> str:
    if knowledge == steering and knowledge:
        return (
            "the knowledge AUROC and the steering win rate fall with an "
            f"interval below zero on {_sizes_text(knowledge)} only"
        )
    parts = []
    for name, sizes in (
        ("knowledge AUROC", knowledge),
        ("steering win rate", steering),
    ):
        parts.append(
            f"the {name} has an interval below zero on "
            + (_sizes_text(sizes) if sizes else "no size")
        )
    return " and ".join(parts)


def _excludes_zero(lo: float, hi: float) -> bool:
    return lo > 0 or hi < 0


def _draw_row(
    ax: Any,
    y: float,
    cells: dict[str, tuple[float, float, float, str]],
    sizes: tuple[str, ...],
    limit: float,
    named: bool,
) -> None:
    for offset, size in zip((-DODGE, 0.0, DODGE), sizes, strict=True):
        value, lo, hi, kind = cells[size]
        color = axes.model_style(f"qwen3-{size}").color
        ax.plot(
            [lo, hi],
            [y + offset] * 2,
            color=color,
            ls="-" if kind == "paired" else CONSERVATIVE_DASH,
            lw=paper.WHISKER_WIDTH,
            solid_capstyle="butt",
            gid="whisker",
            zorder=2,
        )
        ax.plot(
            [value],
            [y + offset],
            ls="none",
            marker=paper.SHAPES["direction"],
            color=color,
            markerfacecolor=color if _excludes_zero(lo, hi) else "white",
            markeredgecolor=color,
            markeredgewidth=_rows.EDGE_WIDTH,
            markersize=_rows.MARKER_SIZE,
            gid="point",
            zorder=3,
        )
        if named:
            right = 0 <= hi < 0.6 * limit
            pad = LABEL_PAD * limit
            ax.text(
                hi + pad if right else lo - pad,
                y + offset,
                size.upper(),
                color=paper.INK,
                fontsize=LABEL_FONT,
                va="center",
                ha="left" if right else "right",
            )


def _positions(rows: list[tuple[str, str, str]]) -> list[int]:
    ys: list[int] = []
    for i, (key, _, _) in enumerate(rows):
        gap = i > 0 and SECTIONS[key] != SECTIONS[rows[i - 1][0]]
        ys.append((ys[-1] + 1 if ys else 0) + (GROUP_GAP if gap else 0))
    return ys


def _groups(
    ax: Any, rows: list[tuple[str, str, str]], ys: list[int], limit: float
) -> None:
    seen: list[str] = []
    for (key, _, _), y in zip(rows, ys, strict=True):
        section = SECTIONS[key]
        if section in seen:
            continue
        if seen:
            ax.axhline(
                y - 1.0,
                color=paper.RANDOM_BAND,
                lw=0.5,
                gid="separator",
                zorder=1,
            )
        seen.append(section)
        ax.text(
            -limit * 0.98,
            y - 0.65,
            section,
            color=paper.ENC["secondary_text"],
            fontsize=LABEL_FONT,
            va="center",
            ha="left",
            gid="group",
        )


def _panel(
    ax: Any,
    rows: list[tuple[str, str, str]],
    found: dict[tuple[str, str], tuple[float, float, float, str]],
    sizes: tuple[str, ...],
    limit: float,
    letter: str,
    csv_rows: list[list[Any]],
) -> None:
    ys = _positions(rows)
    for y, (key, label, _) in zip(ys, rows, strict=True):
        cells = {size: found[(key, size)] for size in sizes}
        _draw_row(ax, y, cells, sizes, limit, named=y == ys[0])
        for size in sizes:
            value, lo, hi, kind = cells[size]
            csv_rows.append([letter, label, size, value, lo, hi, kind])
    _groups(ax, rows, ys, limit)
    ax.axvline(0.0, color=paper.INK, lw=0.5, gid="zero", zorder=2.5)
    ax.set_xlim(-limit, limit)
    step = 0.05 if limit <= NARROW else 0.1
    count = round(limit / step)
    ticks = [round(k * step, 2) for k in range(-count, count + 1)]
    ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(axes.minus_formatter())
    ax.set_ylim(ys[-1] + 0.6, -1.45)
    ax.set_yticks(ys)
    ax.set_yticklabels([label for _, label, _ in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    for text, sign, side in (
        ("base higher", -1, "right"),
        ("post-trained higher", 1, "left"),
    ):
        ax.text(
            sign * LABEL_PAD * limit,
            -1.1,
            text,
            color=paper.ENC["secondary_text"],
            fontsize=LABEL_FONT,
            va="center",
            ha=side,
        )


def _key(fig: Any) -> None:
    ink = paper.INK
    marker: dict[str, Any] = {
        "marker": paper.SHAPES["direction"],
        "markeredgecolor": ink,
        "markeredgewidth": _rows.EDGE_WIDTH,
        "markersize": _rows.MARKER_SIZE,
        "color": ink,
        "ls": "none",
    }
    handles = [
        Line2D(
            [],
            [],
            markerfacecolor=ink,
            label="interval excludes zero",
            **marker,
        ),
        Line2D(
            [],
            [],
            markerfacecolor="white",
            label="interval includes zero",
            **marker,
        ),
        Line2D(
            [],
            [],
            color=ink,
            lw=paper.WHISKER_WIDTH,
            label="seed-paired interval",
        ),
        Line2D(
            [],
            [],
            color=ink,
            lw=paper.WHISKER_WIDTH,
            ls=CONSERVATIVE_DASH,
            label="conservative interval",
        ),
        *(
            Line2D(
                [],
                [],
                marker=paper.SHAPES["direction"],
                markerfacecolor=axes.model_style(f"qwen3-{size}").color,
                markeredgecolor=axes.model_style(f"qwen3-{size}").color,
                markersize=_rows.MARKER_SIZE,
                ls="none",
                label=size.upper(),
            )
            for size in axes.QWEN3_SIZES
        ),
    ]
    legend = fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=4,
        handlelength=1.6,
        columnspacing=1.0,
        fontsize=LABEL_FONT,
    )
    legend.get_frame().set_edgecolor("none")


def _fit(line: Any, length: float) -> None:
    """Sets a dash pattern that starts and ends on a whole dash.

    Args:
        line: The line to restyle.
        length: Its drawn length in points.
    """
    width = line.get_linewidth()
    gap = DASH_GAP * width
    nominal = CONSERVATIVE_DASH[1][0] * width
    count = max(MIN_DASHES, round((length + gap) / (nominal + gap)))
    dash = (length - (count - 1) * gap) / count
    line.set_linestyle((0, (dash / width, gap / width)))


def _fit_dashes(fig: Any) -> None:
    """Fits every conservative interval and its key sample to its length.

    A fixed pattern leaves a short stub at the end of a whisker, which
    reads as a dot after a dash.

    Args:
        fig: The figure with its layout frozen.
    """
    fig.draw_without_rendering()
    for ax in fig.axes:
        for line in ax.lines:
            if line.get_gid() != "whisker" or line.get_linestyle() == "-":
                continue
            x0, x1 = line.get_xdata()
            pixels = abs(
                ax.transData.transform((x1, 0))[0]
                - ax.transData.transform((x0, 0))[0]
            )
            _fit(line, pixels * 72 / fig.dpi)
    for legend in fig.legends:
        for handle, text in zip(
            legend.legend_handles, legend.get_texts(), strict=True
        ):
            if text.get_text() == "conservative interval":
                _fit(handle, legend.handlelength * text.get_fontsize())


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    sizes = axes.QWEN3_SIZES
    paired_rows = _paired_rows(manifest)
    by_measure = {
        (PAIRED[r["source"]], str(r["tuned"]).split("-")[1]): r
        for r in paired_rows
    }
    found = {
        (key, size): _interval(manifest, by_measure, key, size)
        for key, _, _ in ROWS
        for size in sizes
    }
    narrow = [
        row
        for row in ROWS
        if all(
            max(abs(found[(row[0], s)][1]), abs(found[(row[0], s)][2]))
            <= NARROW
            for s in sizes
        )
    ]
    wide = [row for row in ROWS if row not in narrow]
    csv_rows: list[list[Any]] = []
    numbers: dict[str, float] = {}
    for key, _, _ in ROWS:
        for size in sizes:
            value, lo, hi, _ = found[(key, size)]
            tag = f"{key}_{size.replace('.', '')}"
            numbers[tag] = value
            numbers[f"{tag}_lo"] = lo
            numbers[f"{tag}_hi"] = hi

    with paper.paper_style():
        fig = _rows.new_figure(HEIGHT)
        top, bottom = fig.subplots(
            2,
            1,
            gridspec_kw={
                "height_ratios": [
                    _positions(narrow)[-1] + 2.6,
                    _positions(wide)[-1] + 2.6,
                ]
            },
        )
        _panel(top, narrow, found, sizes, NARROW, "a", csv_rows)
        _panel(bottom, wide, found, sizes, WIDE, "b", csv_rows)
        paper.panel_letter(top, "a")
        paper.panel_letter(bottom, "b")
        for ax in (top, bottom):
            ax.set_xlabel("post-trained minus base")
        _key(fig)
        _rows.freeze_layout(fig)
        _fit_dashes(fig)

    sentiment = [found[("sentiment", s)] for s in sizes]
    numbers["sent_below"] = float(sum(hi < 0 for _, _, hi, _ in sentiment))
    moves = [abs(found[(r, s)][0]) for r in OTHER for s in sizes]
    numbers["max_abs_other"] = max(moves)
    excluding = [
        (r, s, found[(r, s)][0])
        for r in OTHER
        for s in sizes
        if _excludes_zero(found[(r, s)][1], found[(r, s)][2])
    ]
    numbers["other_excluding"] = float(len(excluding))
    falls = sorted(abs(v) for v, _, _, _ in sentiment)
    below = {
        r: [s for s in sizes if found[(r, s)][2] < 0]
        for r in ("knowledge", "steering")
    }
    paired = _paired_excluding(paired_rows)
    numbers["paired_total"] = float(len(paired_rows))
    numbers["paired_excluding_zero"] = float(len(paired))
    for name, size in paired:
        numbers[f"paired_{name}_{size.lower().replace('.', '')}"] = 1.0
    detail = DETAIL.format(
        narrow=_rows.number(NARROW),
        wide=f"{axes.MINUS}{_rows.number(WIDE)} to {_rows.number(WIDE)}",
        low=_rows.number(falls[0]),
        high=_rows.number(falls[-1]),
        below=int(numbers["sent_below"]),
        most=_rows.number(numbers["max_abs_other"]),
        excluding=_excluding_text(excluding),
        falling=_falling_text(below["knowledge"], below["steering"]),
    )
    return FigureOutput(
        fig, CLAIM, detail, numbers, _group.csv_text(HEADER, csv_rows), {}
    )
