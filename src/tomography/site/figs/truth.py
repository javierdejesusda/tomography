"""Figure 6: where reading truth does not beat text or negation.

Panel (a) puts the MC1 accuracy of the truth reader and of the honesty
persona reader on TruthfulQA beside the answer log-probability and the range
of two classifiers that see only the words. Panel (b) gives the accuracy of
the truth reader on four statement sets, with an arrow from each affirmative
set to its negation.
"""

import dataclasses
import json
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from tomography.site import runfiles
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.site.registry import RoleCount
from tomography.viz import axes as shared
from tomography.viz import paper

HEIGHT = 3.0
TEXT_METHODS = ("text_word", "text_char")
READERS = {
    "truthfulqa_reader": ("truthfulqa", "o", -0.18),
    "persona_reader": ("honesty", "D", 0.0),
}
LOG_PROBABILITY = ("log_probability", 0.18)
PAIRS = (
    ("cities", "neg_cities", "o", -0.2),
    ("sp_en_trans", "neg_sp_en_trans", "s", 0.2),
)
NEGATED = tuple(neg for _, neg, _, _ in PAIRS)
XLIM_A = (0.1, 0.85)
XLIM_B = (0.15, 1.04)
MARKER = 3.0
TEXT = 6.5


@dataclasses.dataclass(frozen=True)
class Cell:
    """One value with its interval.

    Attributes:
        value: The accuracy.
        lo: Lower end of the interval.
        hi: Upper end of the interval.
    """

    value: float
    lo: float
    hi: float


@dataclasses.dataclass(frozen=True)
class Model:
    """Everything one checkpoint contributes to the figure.

    Attributes:
        reads: Panel (a) series name to its cell.
        sets: Statement set to its cell.
        random: Negated set to the accuracy of its random direction.
        text: Text-only classifier name to its cell.
        words: Largest point accuracy of the text-only classifiers.
        n_questions: Test questions behind the TruthfulQA intervals.
    """

    reads: dict[str, Cell]
    sets: dict[str, Cell]
    random: dict[str, float]
    text: dict[str, Cell]
    words: float
    n_questions: int


def _random_at(path: Any, name: str, layer: int) -> float:
    series = json.loads((path / f"curves_{name}.json").read_text("utf-8"))[
        "data"
    ]["series"]
    with np.load(path / f"curves_{name}.npz") as curves:
        column = list(curves["x"]).index(layer)
        return float(curves["mean"][series.index("random"), column])


def _model(manifest: Manifest, model: str) -> Model:
    tqa = manifest.run("r6", model)
    other = manifest.run("r7", model)
    if tqa is None or other is None:
        raise FileNotFoundError(f"r6 or r7 has no run for {model}")
    table = runfiles.rows(tqa, "table.json")
    n_test = int(runfiles.data(tqa, "summary.json")["n_test"])

    def lat(source: str) -> Cell:
        (row,) = [
            r
            for r in table
            if r["method"] == "lat"
            and r["layer_rule"] == "validation-layer"
            and r["source"] == source
        ]
        lo, hi = stats.wilson(row["mc1"], n_test)
        return Cell(row["mc1"], lo, hi)

    def stored(method: str) -> Cell:
        (row,) = [r for r in table if r["method"] == method]
        return Cell(row["mc1"], row["lo"], row["hi"])

    reads = {name: lat(src) for name, (src, _, _) in READERS.items()}
    reads[LOG_PROBABILITY[0]] = stored("standard")
    summary = runfiles.data(other, "summary.json")
    layer = int(summary["val_layer"])
    sets = {}
    for name in (*(a for a, _, _, _ in PAIRS), *NEGATED):
        cell = summary["sets"][name]
        value, n = cell["accuracy_at_val_layer"], int(cell["n_eval_pairs"])
        lo, hi = stats.wilson(value, n)
        sets[name] = Cell(value, lo, hi)
    text = {m: stored(m) for m in TEXT_METHODS}
    return Model(
        reads,
        sets,
        {name: _random_at(other, name, layer) for name in NEGATED},
        text,
        max(c.value for c in text.values()),
        n_test,
    )


def _whisker(ax: Axes, y: float, cell: Cell, color: str) -> None:
    ax.plot(
        [cell.lo, cell.hi],
        [y, y],
        color=color,
        lw=paper.WHISKER_WIDTH,
        solid_capstyle="butt",
        zorder=3,
    )


def _point(
    ax: Axes,
    x: float,
    y: float,
    shape: str,
    color: str,
    filled: bool,
) -> None:
    ax.plot(
        [x],
        [y],
        ls="none",
        marker=shape,
        ms=MARKER,
        color=color,
        mfc=color if filled else "white",
        mew=0.8,
        zorder=4,
    )


def _panel_a(
    ax: Axes,
    rows: dict[str, float],
    data: dict[str, Model],
    band: tuple[float, float],
    chance: float,
) -> None:
    shared.ref_line(ax, chance, "x")
    ax.axvspan(*band, color=paper.ENC["text"], alpha=0.15, lw=0, zorder=1.5)
    ax.text(
        sum(band) / 2,
        -0.75,
        "text only",
        fontsize=TEXT,
        ha="center",
        va="center",
        color=paper.ENC["secondary_text"],
    )
    colors = {
        "truthfulqa_reader": paper.ENC["direction"],
        "persona_reader": paper.ENC["paper"],
        LOG_PROBABILITY[0]: paper.ENC["own_probability"],
    }
    for model, y in rows.items():
        filled = shared.model_style(model).filled
        for name, cell in data[model].reads.items():
            shape = "P" if name == LOG_PROBABILITY[0] else READERS[name][1]
            dodge = (
                LOG_PROBABILITY[1]
                if name == LOG_PROBABILITY[0]
                else READERS[name][2]
            )
            _whisker(ax, y + dodge, cell, colors[name])
            _point(
                ax,
                cell.value,
                y + dodge,
                shape,
                colors[name],
                filled or name == LOG_PROBABILITY[0],
            )
    ax.set_xlim(*XLIM_A)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("MC1 accuracy on TruthfulQA")
    ax.spines["left"].set_visible(False)


def _panel_b(ax: Axes, rows: dict[str, float], data: dict[str, Model]) -> None:
    shared.ref_line(ax, 0.5, "x")
    ink = paper.ENC["direction"]
    for model, y in rows.items():
        filled = shared.model_style(model).filled
        for plain, neg, shape, dodge in PAIRS:
            line = y + dodge
            _whisker(ax, line, data[model].sets[neg], ink)
            _point(
                ax,
                data[model].random[neg],
                line,
                "x",
                paper.ENC["random_marker"],
                True,
            )
            _point(ax, data[model].sets[neg].value, line, shape, ink, filled)
            ax.annotate(
                "",
                xy=(data[model].sets[neg].value, line),
                xytext=(data[model].sets[plain].value, line),
                arrowprops={
                    "arrowstyle": "-|>",
                    "lw": 0.8,
                    "color": ink,
                    "shrinkA": 1.5,
                    "shrinkB": 1.5,
                    "mutation_scale": 5,
                },
                zorder=3,
            )
            _whisker(ax, line, data[model].sets[plain], ink)
            _point(ax, data[model].sets[plain].value, line, shape, ink, filled)
    ax.set_xlim(*XLIM_B)
    shared.unit_axis(ax, "x")
    ax.set_xlabel("reader accuracy at the validation layer")
    ax.spines["left"].set_visible(False)


def _legend(fig: Figure) -> None:
    ink = paper.ENC["direction"]

    def mark(shape: str, color: str) -> Line2D:
        return Line2D([], [], ls="none", marker=shape, ms=MARKER, color=color)

    handles: list[Any] = [
        mark("o", ink),
        mark("D", paper.ENC["paper"]),
        mark("P", paper.ENC["own_probability"]),
        Patch(fc=paper.ENC["text"], alpha=0.15, ec="none"),
        mark("o", ink),
        mark("s", ink),
        Line2D([], [], color=ink, lw=0.8, marker=">", ms=2.5, markevery=[1]),
        mark("x", paper.ENC["random_marker"]),
    ]
    labels = [
        "(a) truth reader",
        "honesty persona reader",
        "answer log-probability",
        "text-only classifiers",
        "(b) cities",
        "Spanish to English",
        "to the negated set",
        "random direction, negated",
    ]
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=4,
        frameon=False,
        fontsize=TEXT,
        columnspacing=1.0,
        handletextpad=0.4,
    )


def _counts(
    models: list[str], data: dict[str, Model]
) -> dict[str, dict[str, RoleCount]]:
    def inside(m: str, band: tuple[float, float]) -> bool:
        return band[0] <= data[m].reads["truthfulqa_reader"].value <= band[1]

    band = _band(data)
    tqa = stats.per_role({m: inside(m, band) for m in models})
    below = stats.per_role(
        {m: any(data[m].sets[n].hi < 0.5 for n in NEGATED) for m in models}
    )
    return {"tqa_inside": tqa, "neg_below_half": below}


def _band(data: dict[str, Model]) -> tuple[float, float]:
    cells = [c for m in data.values() for c in m.text.values()]
    return min(c.lo for c in cells), max(c.hi for c in cells)


def _numbers(
    models: list[str],
    data: dict[str, Model],
    counts: dict[str, dict[str, RoleCount]],
    chance: float,
) -> dict[str, float]:
    lo, hi = _band(data)
    persona = [data[m].reads["persona_reader"].value for m in models]
    numbers: dict[str, float] = {
        "persona_min": min(persona),
        "persona_max": max(persona),
        "text_lo": lo,
        "text_hi": hi,
        "chance": chance,
        "text_word": data[models[0]].text["text_word"].value,
        "text_char": data[models[0]].text["text_char"].value,
        "n_questions": float(data[models[0]].n_questions),
    }
    for role in stats.ROLES:
        group = [m for m in models if stats.role(m) == role]
        numbers[f"tqa_inside_{role}"] = counts["tqa_inside"][role].hits
        numbers[f"tqa_above_words_{role}"] = sum(
            data[m].reads["truthfulqa_reader"].value > data[m].words
            for m in group
        )
        numbers[f"neg_below_half_{role}"] = counts["neg_below_half"][role].hits
        numbers[f"neg_below_half_n_{role}"] = counts["neg_below_half"][
            role
        ].total
    numbers["negcities_below_random_gpt2"] = sum(
        data[m].sets["neg_cities"].value < data[m].random["neg_cities"]
        for m in models
        if stats.role(m) == "gpt2"
    )
    return numbers


def _claim(
    numbers: dict[str, float], counts: dict[str, dict[str, RoleCount]]
) -> str:
    return (
        "A truth reader does not beat words alone on TruthfulQA, and "
        "negated statements pull it below chance on some checkpoints. Its "
        "MC1 accuracy lies inside the span of the 95 percent intervals of "
        f"two text-only classifiers ({numbers['text_lo']:.3f} to "
        f"{numbers['text_hi']:.3f}; their accuracies are "
        f"{numbers['text_word']:.3f} and {numbers['text_char']:.3f}) on "
        f"{stats.per_role_text(counts['tqa_inside'])}, while the honesty "
        f"persona reader reaches {numbers['persona_min']:.3f} to "
        f"{numbers['persona_max']:.3f}, against a chance level of "
        f"{numbers['chance']:.3f}."
    )


def _detail(
    numbers: dict[str, float], counts: dict[str, dict[str, RoleCount]]
) -> str:
    gpt2 = counts["tqa_inside"]["gpt2"].total
    return (
        "(a) MC1 accuracy on TruthfulQA of the truth reader and of the "
        "honesty persona reader at the layer chosen on validation (whiskers "
        f"are 95 percent Wilson intervals over {numbers['n_questions']:.0f} "
        "questions), of the answer "
        "log-probability (its stored interval), and the range of the "
        "intervals of two classifiers that see only the words (green); the "
        "dotted line is chance. (b) Accuracy of the truth reader on "
        "statements about cities and on Spanish to English translations, "
        "each joined by an arrow to its negation, with the accuracy of a "
        "random direction on the negated set (grey crosses); whiskers in "
        "(b) are 95 percent Wilson intervals over the evaluation pairs of "
        "each set. Checkpoints with at least one negated set whose "
        "interval lies below 0.5 are "
        f"{stats.per_role_text(counts['neg_below_half'])}. On negated "
        "cities the reader falls below the "
        f"random direction on {int(numbers['negcities_below_random_gpt2'])} "
        f"of {gpt2} GPT-2 sizes."
    )


def _csv(models: list[str], data: dict[str, Model]) -> str:
    lines = ["panel,model,role,series,value,lo,hi"]

    def add(panel: str, model: str, series: str, cell: Cell) -> None:
        lines.append(
            f"{panel},{model},{stats.role(model)},{series},"
            f"{cell.value:.10g},{cell.lo:.10g},{cell.hi:.10g}"
        )

    for model in models:
        for series, cell in data[model].reads.items():
            add("a", model, series, cell)
        for series, cell in data[model].sets.items():
            add("b", model, series, cell)
        for series, value in data[model].random.items():
            add("b", model, f"random_{series}", Cell(value, value, value))
    return "\n".join(lines) + "\n"


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    models = manifest.models("r6")
    data = {m: _model(manifest, m) for m in models}
    path = manifest.run("r6", models[0])
    assert path is not None
    chance = float(runfiles.data(path, "curves_truthfulqa.json")["chance"])
    counts = _counts(models, data)
    numbers = _numbers(models, data, counts, chance)
    rows = shared.model_rows(models)
    with paper.paper_style():
        fig = Figure(figsize=(paper.FULL_WIDTH, HEIGHT), layout="constrained")
        left, right = fig.subplots(
            1, 2, sharey=True, gridspec_kw={"width_ratios": [1.0, 1.0]}
        )
        _panel_a(left, rows, data, _band(data), chance)
        _panel_b(right, rows, data)
        left.set_yticks(
            list(rows.values()), [paper.display_name(m) for m in rows]
        )
        left.set_ylim(max(rows.values()) + 0.7, -1.1)
        for ax, letter in ((left, "a"), (right, "b")):
            ax.tick_params(axis="y", length=0)
            paper.panel_letter(ax, letter)
        _legend(fig)
    return FigureOutput(
        fig,
        _claim(numbers, counts),
        _detail(numbers, counts),
        numbers,
        _csv(models, data),
        counts,
    )
