"""Figure 10: Approximate answers and range readers.

Panel (a) reads the greedy few-shot behavior of P10a on range items, exact
items and the people diagnostic. Panels (b) and (c) read the held-out
duration cell and the people position of the concept-template reader of
P10b, fitted on height and wingspan, with its random control and the text-word
classifier. The three panels share the checkpoint rows.
"""

import json
import math
from typing import Any

from matplotlib import ticker
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
TRAIN = "height+wingspan"
HEIGHT = 3.4
HEADER = ["panel", "model", "series", "value", "lo", "hi"]
TOP = -1.0
POSITION_LIMITS = (-0.5, 1.5)
PANEL_A = (
    ("range_items", paper.ITEM_SET_MARKERS["range"], -0.28),
    ("exact_items", paper.ITEM_SET_MARKERS["exact"], 0.0),
    ("people", paper.ITEM_SET_MARKERS["people"], 0.28),
)
CLAIM = (
    "Qwen3 writes ranges where the sourced answer is a range, but a range "
    "reader beats its random control only on some checkpoints."
)
DETAIL = (
    "(a) Share of greedy answers written as a range (few-shot, range "
    "example last) on range items (star), exact items (left triangle) and "
    "{n_people} people's heights (right triangle), with Wilson intervals on the "
    "item counts: Qwen3 base checkpoints write a range on {base_min} to "
    "{base_max} percent of the range items and post-trained ones on "
    "{post_min} to {post_max} percent, on at most {exact} percent of the "
    "exact items and {heights}, while GPT-2 small writes a range on "
    "{gpt2_range} percent of the range items and {gpt2_exact} percent of "
    "the exact items. (b) AUROC on held-out durations of a range reader "
    "fitted on heights and wingspans (concept prompt), with its random "
    "control (grey: the mean score of {draws} random directions oriented on "
    "the training labels, with the 95 percent bootstrap interval of the "
    "mean over the draws) and a classifier on the words of the question "
    "(green, {text}, interval {text_lo} to {text_hi}). The reader is above "
    "the classifier's value on {above}, {overlap}, and above "
    "the random control's interval on {above_random}; the control itself "
    "is below 0.5 on {below_half}, a sign flip between the training and "
    "the test concepts, and above 0.95 on {above_95}. (c) Where the reader "
    "places the "
    "people between the test buildings (0) and the breeds (1), against the "
    "range of its "
    "random control over {sets} training sets (grey): at {pos_min} to "
    "{pos_max}, next to the breeds, so the reader does not separate a "
    "range-valued answer from a living subject. The position lies outside "
    "the random band on {outside}, but the random bands span {band_min} to "
    "{band_max} across checkpoints, far outside 0 to 1 (clipped at the plot "
    "edges, arrows), so the position cannot separate the reader from random."
)


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


def _band(ax: Any, y: float, lo: float, hi: float, model: str) -> None:
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
    for marker, edge, clip in ((4, low, lo < low), (5, high, hi > high)):
        if clip:
            ax.plot(
                [edge],
                [y],
                marker=marker,
                ls="none",
                color=paper.ENC["random_marker"],
                markersize=3.0,
                zorder=2,
            )


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
        grid = fig.add_gridspec(1, 3, width_ratios=[2.1, 1.6, 1.6])
        ax_a = fig.add_subplot(grid[0, 0])
        ax_b = fig.add_subplot(grid[0, 1], sharey=ax_a)
        ax_c = fig.add_subplot(grid[0, 2], sharey=ax_a)
        for model in models:
            for series, marker, dodge in PANEL_A:
                cell = behavior[model][series]
                _rows.dot(
                    ax_a,
                    model,
                    rows[model] + dodge,
                    float(cell["value"]),
                    marker,
                    paper.INK,
                    (float(cell["lo"]), float(cell["hi"])),
                )
                csv_rows.append(
                    ["a", model, series, cell["value"], cell["lo"], cell["hi"]]
                )
        for model, reader in readers.items():
            y = rows[model]
            _rows.dot(
                ax_b,
                model,
                y - 0.15,
                float(reader.reader["auroc"]),
                paper.MEASURE_MARKERS["direction"],
                paper.INK,
                (float(reader.reader["lo"]), float(reader.reader["hi"])),
            )
            ax_b.plot(
                [reader.random["lo"], reader.random["hi"]],
                [y + 0.15, y + 0.15],
                color=paper.ENC["random_marker"],
                lw=paper.WHISKER_WIDTH,
                zorder=2,
            )
            ax_b.plot(
                [reader.random["auroc"]],
                [y + 0.15],
                marker="x",
                ls="none",
                color=paper.ENC["random_marker"],
                markersize=_rows.MARKER_SIZE,
                markeredgewidth=0.8,
                zorder=3,
            )
            csv_rows.append(
                [
                    "b",
                    model,
                    "reader",
                    reader.reader["auroc"],
                    reader.reader["lo"],
                    reader.reader["hi"],
                ]
            )
            csv_rows.append(
                [
                    "b",
                    model,
                    "random",
                    reader.random["auroc"],
                    reader.random["lo"],
                    reader.random["hi"],
                ]
            )
            _rows.dot(
                ax_c,
                model,
                y,
                float(reader.position["position"]),
                paper.MEASURE_MARKERS["direction"],
                paper.INK,
            )
            _band(ax_c, y, min(reader.draws), max(reader.draws), model)
            csv_rows.append(
                ["c", model, "reader", reader.position["position"], None, None]
            )
            csv_rows.append(
                [
                    "c",
                    model,
                    "random",
                    None,
                    min(reader.draws),
                    max(reader.draws),
                ]
            )
        csv_rows.append(
            ["b", None, "words", text["auroc"], text["lo"], text["hi"]]
        )
        ax_b.axvspan(
            text["lo"], text["hi"], color=paper.ENC["text"], alpha=0.15, lw=0
        )
        ax_b.axvline(
            text["auroc"], color=paper.ENC["text"], lw=paper.DATA_WIDTH
        )
        ax_b.text(
            text["auroc"],
            TOP + 0.4,
            "words",
            color=paper.ENC["secondary_text"],
            fontsize=6.5,
            ha="center",
            va="center",
        )
        _rows.row_axis(ax_a, rows, labels=True)
        _rows.row_axis(ax_b, rows, labels=False)
        _rows.row_axis(ax_c, rows, labels=False)
        ax_a.set_ylim(max(rows.values()) + 0.6, TOP)
        ax_a.set_xlim(-0.03, 1.03)
        axes.unit_axis(ax_a, "x")
        ax_a.set_xlabel("share written as a range")
        ax_b.set_xlim(-0.03, 1.03)
        axes.unit_axis(ax_b, "x")
        axes.ref_line(ax_b, 0.5, "x")
        ax_b.set_xlabel("AUROC on held-out\ndurations")
        ax_c.set_xlim(*POSITION_LIMITS)
        ax_c.xaxis.set_major_locator(
            ticker.FixedLocator([-0.5, 0.0, 0.5, 1.0, 1.5])
        )
        ax_c.xaxis.set_major_formatter(axes.minus_formatter())
        for edge, label in ((0.0, "buildings"), (1.0, "breeds")):
            axes.ref_line(ax_c, edge, "x")
            ax_c.text(
                edge,
                TOP + 0.4,
                label,
                color=paper.ENC["secondary_text"],
                fontsize=6.5,
                ha="center",
                va="center",
            )
        ax_c.set_xlabel("people's position\n(arrows: band clipped)")
        for ax, letter in zip((ax_a, ax_b, ax_c), "abc", strict=True):
            paper.panel_letter(ax, letter)
        legend = fig.legend(
            handles=[
                _rows.series_handle(
                    "range items", paper.ITEM_SET_MARKERS["range"], paper.INK
                ),
                _rows.series_handle(
                    "exact items", paper.ITEM_SET_MARKERS["exact"], paper.INK
                ),
                _rows.series_handle(
                    "people's heights",
                    paper.ITEM_SET_MARKERS["people"],
                    paper.INK,
                ),
                _rows.series_handle(
                    "range reader",
                    paper.MEASURE_MARKERS["direction"],
                    paper.INK,
                ),
                _rows.series_handle(
                    "random direction", "x", paper.ENC["random_marker"]
                ),
                Patch(
                    facecolor=paper.RANDOM_BAND,
                    lw=0,
                    label="random range of the position",
                ),
                Line2D(
                    [],
                    [],
                    color=paper.ENC["text"],
                    lw=1.0,
                    label="word classifier",
                ),
                _rows.series_handle("GPT-2 and base", "h", paper.INK),
                _rows.series_handle(
                    "post-trained", "h", paper.INK, filled=False
                ),
            ],
            loc="outside upper center",
            ncol=4,
            handlelength=1.0,
            columnspacing=1.2,
            fontsize=7,
        )
        legend.get_frame().set_edgecolor("none")

    rate = {m: behavior[m]["range_items"]["value"] for m in models}
    exact = {m: behavior[m]["exact_items"]["value"] for m in models}
    people = {m: behavior[m]["people"]["value"] for m in models}
    qwen = [m for m in models if m.startswith("qwen3")]

    def _role_values(base: bool) -> list[float]:
        return [rate[m] for m in qwen if m.endswith("-base") == base]

    above = stats.per_role(
        {m: r.reader["auroc"] > text["auroc"] for m, r in readers.items()}
    )
    inside = stats.per_role(
        {
            m: r.random["lo"] <= r.reader["auroc"] <= r.random["hi"]
            for m, r in readers.items()
        }
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
    outside = stats.per_role(
        {
            m: not min(r.draws) <= r.position["position"] <= max(r.draws)
            for m, r in readers.items()
        }
    )
    overlapping = sum(
        r.reader["hi"] >= text["lo"] and r.reader["lo"] <= text["hi"]
        for r in readers.values()
    )
    positions = [r.position["position"] for r in readers.values()]
    first = next(iter(readers.values()))
    numbers: dict[str, float] = {
        "range_base_min": float(_percent(min(_role_values(True)))),
        "range_base_max": float(_percent(max(_role_values(True)))),
        "range_post_min": float(_percent(min(_role_values(False)))),
        "range_post_max": float(_percent(max(_role_values(False)))),
        "exact_qwen_max": float(_percent(max(exact[m] for m in qwen))),
        "people_qwen_max": float(_percent(max(people[m] for m in qwen))),
        "gpt2_range": float(_percent(rate["gpt2"])),
        "gpt2_exact": float(_percent(exact["gpt2"])),
        "text_word": float(text["auroc"]),
        "text_lo": float(text["lo"]),
        "text_hi": float(text["hi"]),
        "position_min": float(min(positions)),
        "position_max": float(max(positions)),
        "n_people": float(first.n_people),
        "n_random_draws": float(first.n_seeds),
        "band_min": float(min(min(r.draws) for r in readers.values())),
        "band_max": float(max(max(r.draws) for r in readers.values())),
    }
    for role in stats.ROLES:
        numbers[f"above_text_{role}"] = float(above[role].hits)
        numbers[f"reader_in_random_{role}"] = float(inside[role].hits)
        numbers[f"people_outside_band_{role}"] = float(outside[role].hits)
        numbers[f"reader_above_random_{role}"] = float(above_random[role].hits)
        numbers[f"control_below_half_{role}"] = float(below_half[role].hits)
        numbers[f"control_above_95_{role}"] = float(above_95[role].hits)
    heights = (
        "on none of the heights"
        if numbers["people_qwen_max"] == 0
        else f"on at most {int(numbers['people_qwen_max'])} percent of the "
        "heights"
    )
    detail = DETAIL.format(
        n_people=int(numbers["n_people"]),
        base_min=int(numbers["range_base_min"]),
        base_max=int(numbers["range_base_max"]),
        post_min=int(numbers["range_post_min"]),
        post_max=int(numbers["range_post_max"]),
        exact=int(numbers["exact_qwen_max"]),
        heights=heights,
        gpt2_range=int(numbers["gpt2_range"]),
        gpt2_exact=int(numbers["gpt2_exact"]),
        draws=int(numbers["n_random_draws"]),
        text=f"{text['auroc']:.3f}",
        text_lo=f"{text['lo']:.3f}",
        text_hi=f"{text['hi']:.3f}",
        above=stats.per_role_text(above),
        overlap=(
            "every interval overlapping the classifier's"
            if overlapping == len(readers)
            else f"{overlapping} of {len(readers)} intervals overlapping "
            "the classifier's"
        ),
        above_random=stats.per_role_text(above_random),
        below_half=stats.per_role_text(below_half),
        above_95=stats.per_role_text(above_95),
        band_min=f"{numbers['band_min']:.1f}".replace("-", axes.MINUS),
        band_max=f"{numbers['band_max']:.1f}",
        outside=stats.per_role_text(outside),
        sets=len(first.draws),
        pos_min=f"{min(positions):.3f}",
        pos_max=f"{max(positions):.3f}",
    )
    return FigureOutput(
        fig,
        CLAIM,
        detail,
        numbers,
        _group.csv_text(HEADER, csv_rows),
        {
            "above_text": above,
            "reader_in_random": inside,
            "reader_above_random": above_random,
            "control_below_half": below_half,
            "control_above_95": above_95,
            "people_outside_band": outside,
        },
    )
