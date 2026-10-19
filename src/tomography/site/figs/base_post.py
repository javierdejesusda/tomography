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

HEIGHT = 2.6
HEADER = ["panel", "row", "size", "value", "lo", "hi"]
LIMIT = 0.3
DODGE = 0.25
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
OTHER = ("utility", "morality", "truth", "honesty", "truthfulqa")
PAIRED = {"r2": "utility", "r3": "morality", "r5": "honesty", "r7": "truth"}
CLAIM = (
    "Post-training lowers held-out sentiment reading; on the other readings "
    "most intervals include zero, which is not evidence of no change."
)
DETAIL = (
    "Post-trained minus base checkpoint for each Qwen3 size (0.6B, 1.7B and "
    "4B, light to dark), with conservative 95 percent intervals for the "
    "test items. Sentiment falls by {low} to {high}, with an interval below "
    "zero on {below} of 3 sizes; utility, morality, truth, honesty transfer "
    "and TruthfulQA move by at most {most}, with {excluding}; {falling}. A "
    "post-trained checkpoint also differs from its base in prompt format and "
    "is distilled from a larger model, so the contrast is with this recipe "
    "and not with instruction tuning in general ([[sec:limitations]]). The "
    "stored paired comparisons of the 0.6B and 1.7B pairs are in "
    "[[table:base_post]]. An interval that includes zero does not show that "
    "nothing changed: the seed-paired differences of [[table:base_post]] "
    "exclude zero for {paired}."
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


def _paired_excluding(manifest: Manifest) -> list[tuple[str, str]]:
    """Finds the stored seed-paired differences whose interval excludes zero.

    Args:
        manifest: The run resolver.

    Returns:
        (measure name, size such as `0.6B`) in size then measure order.
    """
    measures = list(PAIRED.values())
    found = []
    for model in manifest.models("p8"):
        run = manifest.run("p8", model)
        assert run is not None
        for row in runfiles.rows(run, "table.json"):
            if row["source"] in PAIRED and (row["lo"] > 0 or row["hi"] < 0):
                size = str(row["tuned"]).split("-")[1].upper()
                found.append((PAIRED[row["source"]], size))
    return sorted(found, key=lambda f: (float(f[1][:-1]), measures.index(f[0])))


def _signed(value: float) -> str:
    text = _rows.number(value)
    return text if text.startswith(axes.MINUS) else f"+{text}"


def _sizes_text(sizes: list[str]) -> str:
    return _rows.join_and([s.upper() for s in sizes])


def _excluding_text(found: list[tuple[str, str, float]]) -> str:
    if not found:
        return "no interval excluding zero"
    names = {key: phrase for key, _, phrase in ROWS}
    parts = [f"{names[r]} on {s.upper()} ({_signed(v)})" for r, s, v in found]
    return f"an interval excluding zero only for {_rows.join_and(parts)}"


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


def _marker(size: str, **extra: Any) -> dict[str, Any]:
    color = axes.model_style(f"qwen3-{size}").color
    return {
        "color": color,
        "marker": paper.MEASURE_MARKERS["direction"],
        "fillstyle": "left",
        "markerfacecolor": color,
        "markerfacecoloralt": "white",
        "markeredgecolor": color,
        "markeredgewidth": _rows.EDGE_WIDTH,
        "markersize": _rows.MARKER_SIZE,
        "ls": "none",
        **extra,
    }


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    sizes = axes.QWEN3_SIZES
    found = {
        (key, size): _difference(manifest, key, size)
        for key, _, _ in ROWS
        for size in sizes
    }
    csv_rows: list[list[Any]] = []
    numbers: dict[str, float] = {}

    with paper.paper_style():
        fig = _rows.new_figure(HEIGHT)
        ax = fig.add_subplot()
        for y, (key, label, _) in enumerate(ROWS):
            for offset, size in zip((-DODGE, 0.0, DODGE), sizes, strict=True):
                value, lo, hi = found[(key, size)]
                color = axes.model_style(f"qwen3-{size}").color
                ax.plot(
                    [lo, hi],
                    [y + offset] * 2,
                    color=color,
                    lw=paper.WHISKER_WIDTH,
                    solid_capstyle="butt",
                    zorder=2,
                )
                ax.plot([value], [y + offset], zorder=3, **_marker(size))
                csv_rows.append(["a", label, size, value, lo, hi])
                tag = f"{key}_{size.replace('.', '')}"
                numbers[tag] = value
                numbers[f"{tag}_lo"] = lo
                numbers[f"{tag}_hi"] = hi
        axes.ref_line(ax, 0.0, "x")
        ax.set_xlim(-LIMIT, LIMIT)
        ax.xaxis.set_major_locator(
            ticker.FixedLocator([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
        )
        ax.xaxis.set_major_formatter(axes.minus_formatter())
        ax.set_xlabel("post-trained minus base")
        ax.set_ylim(len(ROWS) - 0.4, -0.6)
        ax.set_yticks(range(len(ROWS)))
        ax.set_yticklabels([label for _, label, _ in ROWS])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        legend = fig.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    label=f"Qwen3 {size.upper()}",
                    **_marker(size),
                )
                for size in sizes
            ],
            loc="outside upper center",
            ncol=3,
            handlelength=1.0,
            columnspacing=1.2,
            fontsize=7,
        )
        legend.get_frame().set_edgecolor("none")

    sentiment = [found[("sentiment", s)] for s in sizes]
    numbers["sent_below"] = float(sum(hi < 0 for _, _, hi in sentiment))
    moves = [abs(found[(r, s)][0]) for r in OTHER for s in sizes]
    numbers["max_abs_other"] = max(moves)
    excluding = [
        (r, s, found[(r, s)][0])
        for r in OTHER
        for s in sizes
        if found[(r, s)][1] > 0 or found[(r, s)][2] < 0
    ]
    numbers["other_excluding"] = float(len(excluding))
    falls = sorted(abs(v) for v, _, _ in sentiment)
    below = {
        r: [s for s in sizes if found[(r, s)][2] < 0]
        for r in ("knowledge", "steering")
    }
    paired = _paired_excluding(manifest)
    numbers["paired_excluding_zero"] = float(len(paired))
    for name, size in paired:
        numbers[f"paired_{name}_{size.lower().replace('.', '')}"] = 1.0
    detail = DETAIL.format(
        paired=_rows.join_and([f"{n} on {s}" for n, s in paired]),
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
