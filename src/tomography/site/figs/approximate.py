"""Figure 10: Approximate answers and range readers.

Panel (a) reads the greedy few-shot behavior of P10a on range items, exact
items and the people diagnostic. Panels (b) and (c) read the held-out
duration cell and the people position of the concept-template reader of
P10b, fitted on height and wingspan, with its random control and the text-word
classifier. The three panels share the checkpoint rows.
"""

from collections.abc import Collection
import json
import math
from typing import Any

from matplotlib import ticker
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle

from tomography.site import runfiles
from tomography.site import stats
from tomography.site.figs import _group
from tomography.site.figs import _rows
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import axes
from tomography.viz import paper

CELL = ("free", "fewshot", "range_last", "greedy")
STATE_MARKER = "h"
FORCED = ("forced", "fewshot", "range_last", "greedy")
TRAIN = "height+wingspan"
HEIGHT = 3.4
HEADER = ["panel", "model", "series", "value", "lo", "hi"]
TOP = -1.9
HEAD_Y = -1.3
WHISKER_Y = -0.75
WORDS_OFFSET_PT = 6.0
BAR_HALF = 0.17
TICK_HALF = 0.36
ARROW = 0.14
POSITION_LIMITS = (-0.5, 1.5)
PANEL_A = (
    ("range_items", "range\nitems"),
    ("exact_items", "exact\nitems"),
    ("people", "people's\nheights"),
)
CLAIM = (
    "Qwen3 writes a range for {base_min} to {base_max} percent of range "
    "items (base) and {post_min} to {post_max} percent (post-trained), and a "
    "range reader is above the interval of its random control on "
    "{above_random}."
)
DETAIL = (
    "(a) Share of few-shot answers written as a range on range items, exact "
    "items and {n_people} people's heights. "
    "Wilson 95 percent intervals lie inside the markers, at most "
    "{wilson_gpt2} (GPT-2), {wilson_base} (Qwen3 base) and {wilson_post} "
    "(post-trained) points wide. Exact items, heights: {share_base} "
    "(Qwen3 base), {share_post} (post-trained); GPT-2 small {gpt2_range} and "
    "{gpt2_exact} percent on range and exact items. (b) AUROC on "
    "durations of a range reader fitted on heights and wingspans, with its "
    "random control (grey bar and tick: bootstrap interval and mean of "
    "{draws} random directions) and a word "
    "classifier (green line {text}, whisker {text_lo} to {text_hi}); "
    "{states}. (c) People's position between buildings (0) and breeds "
    "(1) against the random band, the range of the control's position over "
    "its {sets} training sets. See [[sec:appendix_approximate]]."
)


STATE_ORDER = ("above", "inside", "below")
STATE_LABELS = {
    "above": "above the control",
    "inside": "inside the control",
    "below": "below the control",
}
STATE_PHRASES = {
    "above": "a filled hexagon is above the control's interval",
    "inside": "an open hexagon inside it",
    "below": "a left arrow below it",
}
BELOW_MARKER = r"$\leftarrow$"
BELOW_OFFSET = 0.075


def state_sentence(states: Collection[str]) -> str:
    """Writes the caption clause that names the states which occur.

    Args:
        states: The states drawn in panel (b).

    Returns:
        The clause, in the order above, inside, below. The first phrase
        always names the control's interval, which the later ones refer to
        as "it".
    """
    phrases = [STATE_PHRASES[s] for s in STATE_ORDER if s in states]
    if phrases and not phrases[0].endswith("interval"):
        phrases[0] = phrases[0].removesuffix(" it") + " the control's interval"
    return ", ".join(phrases)


def share_phrase(exact: int, people: int) -> str:
    """Writes the highest range shares on exact items and on heights.

    A zero is said plainly and never follows "at most".

    Args:
        exact: The highest percent of exact items answered with a range.
        people: The highest percent of heights answered with a range.

    Returns:
        The phrase for one role.
    """
    if exact and people:
        return f"at most {exact} and {people} percent"
    if exact:
        return f"at most {exact} percent of exact items, none of the heights"
    if people:
        return (
            f"none of the exact items, at most {people} percent of the heights"
        )
    return "none of either"


def _percent(value: float) -> int:
    return math.floor(value * 100 + 0.5 + 1e-9)


def _behavior(manifest: Manifest, model: str) -> dict[str, dict[str, Any]]:
    run = manifest.run("p10a", model)
    assert run is not None
    table = {
        r["kind"]: r
        for r in runfiles.rows(run, "table.csv")
        if r["family"] == "all"
        and (r["condition"], r["format"], r["shot_order"], r["decoding"])
        == CELL
    }
    (diag,) = [
        r
        for r in runfiles.rows(run, "diagnostic.csv")
        if (r["condition"], r["format"], r["shot_order"], r["decoding"]) == CELL
    ]
    return {
        "range_items": {
            "value": table["range"]["range_rate"],
            "lo": table["range"]["range_lo"],
            "hi": table["range"]["range_hi"],
        },
        "exact_items": {
            "value": table["exact"]["range_rate"],
            "lo": table["exact"]["range_lo"],
            "hi": table["exact"]["range_hi"],
        },
        "people": {
            "value": diag["people_rate"],
            "lo": diag["people_lo"],
            "hi": diag["people_hi"],
        },
    }


def _forced_rate(manifest: Manifest, model: str) -> float:
    """Gives the range share when the prompt asks for one exact number.

    Args:
        manifest: The run resolver.
        model: Checkpoint key.

    Returns:
        The range rate of the forced condition in the plotted cell.
    """
    run = manifest.run("p10a", model)
    assert run is not None
    (row,) = [
        r
        for r in runfiles.rows(run, "table.csv")
        if r["family"] == "all"
        and r["kind"] == "range"
        and (r["condition"], r["format"], r["shot_order"], r["decoding"])
        == FORCED
    ]
    return float(row["range_rate"])


class _Reader:
    """What panels (b) and (c) read for one checkpoint."""

    def __init__(self, manifest: Manifest, model: str) -> None:
        run = manifest.run("p10b", model)
        assert run is not None
        cell = {
            r["method"]: r
            for r in runfiles.rows(run, "transfer.csv")
            if r["template"] == "concept"
            and r["train_set"] == TRAIN
            and r["test_family"] == "duration"
        }
        self.reader = cell["logistic"]
        self.random = cell["random"]
        (self.text,) = [
            r
            for r in runfiles.rows(run, "baselines.csv")
            if r["baseline"] == "text_word"
            and r["train_set"] == TRAIN
            and r["test_family"] == "duration"
        ]
        people = [
            r
            for r in runfiles.rows(run, "people.csv")
            if r["source"] == "reader" and r["template"] == "concept"
        ]
        (self.position,) = [
            r
            for r in people
            if r["method"] == "logistic" and r["train_set"] == TRAIN
        ]
        self.draws = [r["position"] for r in people if r["method"] == "random"]
        self.n_people = int(self.position["n_people"])
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        self.n_seeds = int(config["n_seeds"])


def _colour(model: str) -> str:
    """Gives the checkpoint colour of the family ramp."""
    return str(axes.model_style(model).color)


def _clip_arrow(ax: Any, y: float, edge: float, inward: float) -> None:
    ax.annotate(
        "",
        xy=(edge, y),
        xytext=(edge + inward, y),
        arrowprops={
            "arrowstyle": "->",
            "color": paper.RANDOM_GREY,
            "lw": paper.WHISKER_WIDTH,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        annotation_clip=False,
    )


def _band(ax: Any, y: float, lo: float, hi: float) -> None:
    low, high = POSITION_LIMITS
    clipped = (max(lo, low), min(hi, high))
    ax.add_patch(
        Rectangle(
            (clipped[0], y - 0.2),
            clipped[1] - clipped[0],
            0.4,
            facecolor=paper.RANDOM_BAND,
            edgecolor="none",
            zorder=1,
        )
    )
    if lo < low:
        _clip_arrow(ax, y, low, ARROW)
    if hi > high:
        _clip_arrow(ax, y, high, -ARROW)


def _unclip(ax: Any, count: int) -> None:
    """Lets the last `count` lines of a panel draw past its axes edge."""
    for line in ax.get_lines()[-count:]:
        line.set_clip_on(False)


def _head(ax: Any, x: float, text: str, ha: str = "center") -> None:
    ax.text(
        x,
        HEAD_Y,
        text,
        color=paper.ENC["secondary_text"],
        fontsize=paper.MIN_TEXT,
        ha=ha,
        va="center",
        linespacing=0.95,
    )


def _state(reader: Any) -> str:
    if reader.reader["auroc"] > reader.random["hi"]:
        return "above"
    if reader.reader["auroc"] < reader.random["lo"]:
        return "below"
    return "inside"


def _draw_reader(ax: Any, model: str, y: float, reader: "_Reader") -> str:
    """Draws the control bar and the reader of one row and gives its state."""
    state = _state(reader)
    lo, hi = float(reader.random["lo"]), float(reader.random["hi"])
    mean = float(reader.random["auroc"])
    bar = Rectangle(
        (lo, y - BAR_HALF),
        hi - lo,
        2 * BAR_HALF,
        facecolor=paper.RANDOM_GREY,
        alpha=paper.RANDOM_BAND_ALPHA,
        edgecolor="none",
        zorder=5,
    )
    bar.set_gid(f"control-{model}")
    ax.add_patch(bar)
    (control_tick,) = ax.plot(
        [mean, mean],
        [y - TICK_HALF, y + TICK_HALF],
        color=paper.RANDOM_GREY,
        lw=paper.DATA_WIDTH,
        solid_capstyle="butt",
        zorder=5,
    )
    control_tick.set_gid(f"control-tick-{model}")
    value = float(reader.reader["auroc"])
    edge = paper.INK if state == "above" else paper.RANDOM_GREY
    ax.plot(
        [reader.reader["lo"], reader.reader["hi"]],
        [y, y],
        color=edge,
        lw=paper.WHISKER_WIDTH,
        solid_capstyle="butt",
        zorder=3,
    )
    (point,) = ax.plot(
        [value],
        [y],
        marker=STATE_MARKER,
        ls="none",
        color=edge,
        markerfacecolor=paper.INK if state == "above" else "none",
        markeredgecolor=edge,
        markeredgewidth=_rows.EDGE_WIDTH,
        markersize=_rows.MARKER_SIZE,
        zorder=4,
    )
    point.set_gid(f"reader-{model}")
    point.set_clip_on(False)
    if state == "below":
        (mark,) = ax.plot(
            [value - BELOW_OFFSET],
            [y],
            marker=BELOW_MARKER,
            ls="none",
            color=paper.RANDOM_GREY,
            markersize=_rows.MARKER_SIZE,
            zorder=3,
        )
        mark.set_gid(f"below-mark-{model}")
    return state


def _percent_axis(ax: Any, ticks: tuple[float, ...] = (0.0, 1.0)) -> None:
    ax.set_xlim(-0.07, 1.07)
    ax.xaxis.set_major_locator(ticker.FixedLocator(list(ticks)))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))


def _draw(
    fig: Any,
    models: list[str],
    behavior: dict[str, dict[str, dict[str, Any]]],
    readers: dict[str, "_Reader"],
    text: dict[str, Any],
    rows: dict[str, float],
    csv_rows: list[list[Any]],
) -> None:
    grid = fig.add_gridspec(1, 5, width_ratios=[0.62, 0.62, 0.62, 1.4, 1.0])
    ax_a = fig.add_subplot(grid[0, 0])
    others = [fig.add_subplot(grid[0, i], sharey=ax_a) for i in range(1, 5)]
    columns = [ax_a, *others[:2]]
    ax_b, ax_c = others[2], others[3]
    for ax, (series, head) in zip(columns, PANEL_A, strict=True):
        for model in models:
            cell = behavior[model][series]
            colour = _colour(model)
            _rows.dot(
                ax,
                model,
                rows[model],
                float(cell["value"]),
                paper.MEASURE_MARKERS["direction"],
                colour,
                (float(cell["lo"]), float(cell["hi"])),
            )
            _unclip(ax, 2)
            csv_rows.append(
                ["a", model, series, cell["value"], cell["lo"], cell["hi"]]
            )
        _head(ax, 0.5, head)
        _percent_axis(ax)
    for model, reader in readers.items():
        _draw_reader(ax_b, model, rows[model], reader)
        for series, cell in (
            ("reader", reader.reader),
            ("random", reader.random),
        ):
            csv_rows.append(
                ["b", model, series, cell["auroc"], cell["lo"], cell["hi"]]
            )
        y = rows[model]
        _rows.dot(
            ax_c,
            model,
            y,
            float(reader.position["position"]),
            paper.MEASURE_MARKERS["direction"],
            _colour(model),
        )
        _band(ax_c, y, min(reader.draws), max(reader.draws))
        csv_rows.append(
            ["c", model, "reader", reader.position["position"], None, None]
        )
        csv_rows.append(
            ["c", model, "random", None, min(reader.draws), max(reader.draws)]
        )
    csv_rows.append(["b", None, "words", text["auroc"], text["lo"], text["hi"]])
    line = ax_b.axvline(
        text["auroc"], color=paper.ENC["text"], lw=paper.DATA_WIDTH, zorder=2
    )
    line.set_gid("words-line")
    (whisker,) = ax_b.plot(
        [text["lo"], text["hi"]],
        [WHISKER_Y, WHISKER_Y],
        color=paper.ENC["text"],
        lw=paper.WHISKER_WIDTH,
        solid_capstyle="butt",
    )
    whisker.set_gid("words-whisker")
    ax_b.annotate(
        "words",
        (text["auroc"], HEAD_Y),
        xytext=(-WORDS_OFFSET_PT, 0),
        textcoords="offset points",
        color=paper.ENC["secondary_text"],
        fontsize=paper.MIN_TEXT,
        ha="right",
        va="center",
        linespacing=0.95,
    )
    _head(ax_b, 0.48, "chance", "right")
    axes.ref_line(ax_b, 0.5, "x")
    _rows.row_axis(ax_a, rows, labels=True)
    for ax in others:
        _rows.row_axis(ax, rows, labels=False)
    ax_a.set_ylim(max(rows.values()) + 0.6, TOP)
    _percent_axis(ax_b, (0.0, 0.5, 1.0))
    ax_b.set_xlim(-0.03, 1.07)
    ax_b.set_xlabel("AUROC on held-out\ndurations")
    columns[1].set_xlabel("share written\nas a range")
    ax_c.set_xlim(*POSITION_LIMITS)
    ax_c.xaxis.set_major_locator(ticker.FixedLocator([-0.5, 0.5, 1.5]))
    ax_c.xaxis.set_major_formatter(axes.minus_formatter())
    for edge, label in ((0.0, "buildings"), (1.0, "breeds")):
        axes.ref_line(ax_c, edge, "x")
        _head(
            ax_c,
            edge + (0.05 if edge else -0.05),
            label,
            "left" if edge else "right",
        )
    ax_c.set_xlabel("people's position\narrows: band clipped")
    for ax, letter in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c")):
        paper.panel_letter(ax, letter)
    main = fig.legend(
        handles=[
            (
                Patch(
                    facecolor=paper.RANDOM_GREY,
                    alpha=paper.RANDOM_BAND_ALPHA,
                    lw=0,
                ),
                Line2D(
                    [],
                    [],
                    ls="none",
                    marker="|",
                    color=paper.RANDOM_GREY,
                    markersize=_rows.MARKER_SIZE,
                    markeredgewidth=paper.DATA_WIDTH,
                ),
            ),
            Line2D(
                [], [], color=paper.ENC["text"], lw=1.0, label="word classifier"
            ),
            _rows.series_handle(
                "(a, c) GPT-2 and Qwen3 base",
                "o",
                paper.ENC["reference"],
            ),
            _rows.series_handle(
                "(a, c) post-trained",
                "o",
                paper.ENC["reference"],
                filled=False,
            ),
        ],
        labels=[
            "random control",
            "word classifier",
            "(a, c) GPT-2 and Qwen3 base",
            "(a, c) post-trained",
        ],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
        loc="outside upper center",
        ncol=4,
        handlelength=1.0,
        columnspacing=1.0,
        fontsize=7,
    )
    main.get_frame().set_edgecolor("none")
    open_grey: dict[str, Any] = {
        "ls": "none",
        "marker": STATE_MARKER,
        "markerfacecolor": paper.WHITE,
        "markeredgecolor": paper.RANDOM_GREY,
        "markeredgewidth": _rows.EDGE_WIDTH,
        "markersize": _rows.MARKER_SIZE,
    }
    handles: dict[str, Any] = {
        "above": _rows.series_handle(
            STATE_LABELS["above"], STATE_MARKER, paper.INK
        ),
        "inside": Line2D([], [], **open_grey),
        "below": (
            Line2D(
                [],
                [],
                ls="none",
                marker=BELOW_MARKER,
                color=paper.RANDOM_GREY,
                markersize=_rows.MARKER_SIZE,
            ),
            Line2D([], [], **open_grey),
        ),
    }
    occurring = {_state(r) for r in readers.values()}
    shown = [k for k in STATE_ORDER if k in occurring]
    states = fig.legend(
        handles=[handles[k] for k in shown],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
        labels=[STATE_LABELS[k] for k in shown],
        title="(b) range reader against its control",
        title_fontsize=7,
        loc="outside lower center",
        ncol=len(shown),
        handlelength=1.4,
        columnspacing=1.2,
        fontsize=7,
    )
    states.get_frame().set_edgecolor("none")


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    models = manifest.models("p10a")
    behavior = {m: _behavior(manifest, m) for m in models}
    readers = {m: _Reader(manifest, m) for m in manifest.models("p10b")}
    words = {r.text["auroc"] for r in readers.values()}
    if len(words) != 1:
        raise ValueError("the word classifier differs between checkpoints")
    text = next(iter(readers.values())).text
    rows = axes.model_rows(models)
    csv_rows: list[list[Any]] = []

    with paper.paper_style():
        fig = _rows.new_figure(HEIGHT)
        _draw(fig, models, behavior, readers, text, rows, csv_rows)

    rate = {m: behavior[m]["range_items"]["value"] for m in models}
    exact = {m: behavior[m]["exact_items"]["value"] for m in models}
    people = {m: behavior[m]["people"]["value"] for m in models}
    qwen = [m for m in models if m.startswith("qwen3")]

    def _role_values(base: bool) -> list[float]:
        return [rate[m] for m in qwen if m.endswith("-base") == base]

    above = stats.per_role(
        {m: r.reader["auroc"] > text["auroc"] for m, r in readers.items()}
    )
    above_random = stats.per_role(
        {m: r.reader["auroc"] > r.random["hi"] for m, r in readers.items()}
    )
    below_half = stats.per_role(
        {m: r.random["auroc"] < 0.5 for m, r in readers.items()}
    )
    above_95 = stats.per_role(
        {m: r.random["auroc"] > 0.95 for m, r in readers.items()}
    )
    other_layer = stats.per_role(
        {m: r.random["layer"] != r.reader["layer"] for m, r in readers.items()}
    )
    outside = stats.per_role(
        {
            m: not min(r.draws) <= r.position["position"] <= max(r.draws)
            for m, r in readers.items()
        }
    )
    first = next(iter(readers.values()))
    numbers: dict[str, float] = {
        "range_base_min": float(_percent(min(_role_values(True)))),
        "range_base_max": float(_percent(max(_role_values(True)))),
        "range_post_min": float(_percent(min(_role_values(False)))),
        "range_post_max": float(_percent(max(_role_values(False)))),
        "gpt2_range": float(_percent(rate["gpt2"])),
        "gpt2_exact": float(_percent(exact["gpt2"])),
        "text_word": float(text["auroc"]),
        "text_lo": float(text["lo"]),
        "text_hi": float(text["hi"]),
        "n_people": float(first.n_people),
        "n_random_draws": float(first.n_seeds),
        "n_band_sets": float(len(first.draws)),
    }
    forced = {m: _forced_rate(manifest, m) for m in models}
    for role in stats.ROLES:
        members = [m for m in models if stats.role(m) == role]
        numbers[f"exact_{role}_max"] = float(
            _percent(max(exact[m] for m in members))
        )
        numbers[f"people_{role}_max"] = float(
            _percent(max(people[m] for m in members))
        )
        widest = max(
            float(c["hi"]) - float(c["lo"])
            for m in members
            for c in behavior[m].values()
        )
        numbers[f"wilson_{role}_points"] = math.ceil(widest * 1000 - 1e-9) / 10
        own = [r for m, r in readers.items() if stats.role(m) == role]
        place = [r.position["position"] for r in own]
        draws = [d for r in own for d in r.draws]
        numbers[f"position_{role}_min"] = float(min(place))
        numbers[f"position_{role}_max"] = float(max(place))
        numbers[f"band_{role}_min"] = float(min(draws))
        numbers[f"band_{role}_max"] = float(max(draws))
        numbers[f"forced_{role}_max"] = float(
            _percent(max(v for m, v in forced.items() if stats.role(m) == role))
        )
        numbers[f"above_text_{role}"] = float(above[role].hits)
        numbers[f"reader_above_random_{role}"] = float(above_random[role].hits)
        numbers[f"control_other_layer_{role}"] = float(other_layer[role].hits)
        numbers[f"control_below_half_{role}"] = float(below_half[role].hits)
        numbers[f"control_above_95_{role}"] = float(above_95[role].hits)
    claim = CLAIM.format(
        base_min=int(numbers["range_base_min"]),
        base_max=int(numbers["range_base_max"]),
        post_min=int(numbers["range_post_min"]),
        post_max=int(numbers["range_post_max"]),
        above_random=stats.per_role_text(above_random),
    )
    detail = DETAIL.format(
        states=state_sentence({_state(r) for r in readers.values()}),
        n_people=int(numbers["n_people"]),
        **{
            f"share_{role}": share_phrase(
                int(numbers[f"exact_{role}_max"]),
                int(numbers[f"people_{role}_max"]),
            )
            for role in ("base", "post")
        },
        **{
            f"wilson_{role}": f"{numbers[f'wilson_{role}_points']:.1f}"
            for role in stats.ROLES
        },
        gpt2_range=int(numbers["gpt2_range"]),
        gpt2_exact=int(numbers["gpt2_exact"]),
        draws=int(numbers["n_random_draws"]),
        sets=int(numbers["n_band_sets"]),
        text=_rows.number(text["auroc"], 3),
        text_lo=_rows.number(text["lo"], 3),
        text_hi=_rows.number(text["hi"], 3),
    )
    return FigureOutput(
        fig,
        claim,
        detail,
        numbers,
        _group.csv_text(HEADER, csv_rows),
        {
            "above_text": above,
            "reader_above_random": above_random,
            "control_other_layer": other_layer,
            "control_below_half": below_half,
            "control_above_95": above_95,
            "outside": outside,
        },
    )
