"""Figure 6: where reading truth does not beat text or negation.

Panel (a) puts the MC1 accuracy of the truth reader and of the honesty
persona reader on TruthfulQA beside two classifiers that see only the words,
their difference with a conservative interval, and the answer log-probability
with its chance level. Panel (b) has a dumbbell per checkpoint from the
accuracy on each affirmative statement set to its negation.
"""

import dataclasses
import json
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np

from tomography.site import runfiles
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.site.registry import RoleCount
from tomography.viz import axes as shared
from tomography.viz import paper

HEIGHT = 5.4
TEXT_METHODS = ("text_word", "text_char")
TEXT_NAMES = {"text_word": "words", "text_char": "characters"}
DIRECTION = paper.MEASURE_MARKERS["direction"]
PERSONA = paper.ENC["popularity"]
PROBABILITY = paper.MEASURE_MARKERS["answer_probability"]
READERS = {
    "truthfulqa_reader": ("truthfulqa", DIRECTION, -0.18),
    "persona_reader": ("honesty", DIRECTION, 0.18),
}
LOG_PROBABILITY = "log_probability"
PAIRS = (
    ("cities", "neg_cities", "cities"),
    ("sp_en_trans", "neg_sp_en_trans", "translations"),
)
NEGATED = tuple(neg for _, neg, _ in PAIRS)
XLIM_A = (0.1, 0.85)
XLIM_C = (0.1, 0.6)
XLIM_D = (-0.3, 0.3)
XLIM_B = (0.15, 1.04)
MARKER = 3.0
SMALL_TICK = 6.0
TEXT = paper.MIN_TEXT
WIDTHS_TOP = [1.5, 1.1, 1.0]
BAND_ALPHA = 0.14


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
    (standard,) = [r for r in table if r["method"] == "standard"]
    lo, hi = stats.wilson(standard["mc1"], n_test)
    reads[LOG_PROBABILITY] = Cell(standard["mc1"], lo, hi)
    summary = runfiles.data(other, "summary.json")
    layer = int(summary["val_layer"])
    sets = {}
    for name in (*(a for a, _, _ in PAIRS), *NEGATED):
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


def _difference(m: Model) -> Cell:
    reader = m.reads["truthfulqa_reader"].value
    d, lo, hi = stats.diff_interval(reader, m.words, m.n_questions)
    return Cell(d, lo, hi)


def _whisker(
    ax: Axes, y: float, cell: Cell, gid: str, color: str = paper.INK
) -> None:
    (line,) = ax.plot(
        [cell.lo, cell.hi],
        [y, y],
        color=color,
        lw=paper.WHISKER_WIDTH,
        solid_capstyle="butt",
        zorder=3,
    )
    line.set_gid(f"whisker:{gid}")


def _point(
    ax: Axes, x: float, y: float, shape: str, gid: str, **kw: Any
) -> None:
    (line,) = ax.plot(
        [x],
        [y],
        ls="none",
        marker=shape,
        ms=kw.pop("ms", MARKER),
        color=kw.pop("color", paper.INK),
        mew=0.8,
        zorder=4,
        **kw,
    )
    line.set_gid(gid)


def _chance_line(ax: Axes, value: float) -> None:
    line = ax.axvline(
        value,
        color=paper.ENC["reference"],
        ls=paper.LINESTYLES["reference"],
        lw=paper.REFERENCE_WIDTH,
        zorder=1,
    )
    line.set_gid("chance")


def _panel_levels(
    ax: Axes, rows: dict[str, float], data: dict[str, Model]
) -> None:
    first = next(iter(data.values()))
    for method in TEXT_METHODS:
        cell = first.text[method]
        band = ax.axvspan(
            cell.lo,
            cell.hi,
            color=paper.ENC["text"],
            alpha=BAND_ALPHA,
            lw=0,
            zorder=1.5,
        )
        band.set_gid(f"textband:{method}")
        line = ax.axvline(cell.value, color=paper.ENC["text"], lw=0.8, zorder=2)
        line.set_gid(f"text:{method}")
    for model, y in rows.items():
        for name, (_, shape, dodge) in READERS.items():
            cell = data[model].reads[name]
            color = PERSONA if name == "persona_reader" else paper.INK
            _whisker(ax, y + dodge, cell, f"{name}:{model}", color)
            _point(
                ax,
                cell.value,
                y + dodge,
                shape,
                f"{name}:{model}",
                color=color,
            )
    ax.set_xlim(*XLIM_A)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("MC1 accuracy on TruthfulQA")
    ax.spines["left"].set_visible(False)


def _panel_difference(
    ax: Axes, rows: dict[str, float], data: dict[str, Model]
) -> None:
    zero = ax.axvline(0.0, color=paper.INK, lw=0.6, zorder=1.5, label="zero")
    zero.set_gid("zero")
    for model, y in rows.items():
        cell = _difference(data[model])
        _whisker(ax, y, cell, f"diff:{model}")
        _point(ax, cell.value, y, DIRECTION, f"diff:{model}")
    ax.set_xlim(*XLIM_D)
    ax.set_xticks([-0.2, 0.0, 0.2])
    ax.set_xlabel("reader minus words")
    ax.spines["left"].set_visible(False)


def _panel_logprob(
    ax: Axes, rows: dict[str, float], data: dict[str, Model], chance: float
) -> None:
    ax.set_xlim(*XLIM_C)
    _chance_line(ax, chance)
    for model, y in rows.items():
        cell = data[model].reads[LOG_PROBABILITY]
        _whisker(ax, y, cell, f"{LOG_PROBABILITY}:{model}")
        _point(
            ax,
            cell.value,
            y,
            PROBABILITY,
            f"{LOG_PROBABILITY}:{model}",
            color=paper.ENC["own_probability"],
        )
    ax.set_xticks([0.2, 0.4, 0.6])
    ax.set_xlabel("MC1 accuracy")
    ax.spines["left"].set_visible(False)


def _panel_dumbbell(
    ax: Axes,
    rows: dict[str, float],
    data: dict[str, Model],
    pair: tuple[str, str, str],
) -> None:
    plain, neg, _ = pair
    ax.set_xlim(*XLIM_B)
    shade = ax.axvspan(
        XLIM_B[0], 0.5, color=paper.RANDOM_BAND, alpha=0.5, lw=0, zorder=0.5
    )
    shade.set_gid("below_chance")
    _chance_line(ax, 0.5)
    for model, y in rows.items():
        affirmative = data[model].sets[plain]
        negated = data[model].sets[neg]
        (bell,) = ax.plot(
            [affirmative.value, negated.value],
            [y, y],
            color=paper.INK,
            lw=0.8,
            zorder=2,
        )
        bell.set_gid(f"dumbbell:{plain}:{model}")
        (tick,) = ax.plot(
            [data[model].random[neg]] * 2,
            [y - 0.32, y + 0.32],
            color=paper.ENC["random_marker"],
            lw=1.0,
            zorder=3,
        )
        tick.set_gid(f"random:{plain}:{model}")
        _point(
            ax,
            affirmative.value,
            y,
            "|",
            f"aff:{plain}:{model}",
            ms=SMALL_TICK,
        )
        below = negated.hi < 0.5
        _point(
            ax,
            negated.value,
            y,
            DIRECTION,
            f"neg:{plain}:{model}",
            ms=MARKER + 0.6,
            mfc=paper.INK if below else "white",
        )
    ax.set_xticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.5", "0.75", "1"])
    ax.set_xlabel("accuracy, validation layer")
    ax.spines["left"].set_visible(False)


def _legend(fig: Figure) -> None:
    ink = paper.INK

    def mark(shape: str, color: str, **kw: Any) -> Line2D:
        return Line2D(
            [],
            [],
            ls="none",
            marker=shape,
            ms=kw.pop("ms", MARKER),
            color=color,
            **kw,
        )

    handles: list[Any] = [
        mark(DIRECTION, ink),
        mark(DIRECTION, PERSONA),
        Line2D([], [], color=paper.ENC["text"], lw=0.8),
        mark(PROBABILITY, paper.ENC["own_probability"]),
        mark("|", ink, ms=SMALL_TICK, mew=1.0),
        mark(DIRECTION, ink, ms=MARKER + 0.6, mfc="white", mew=0.8),
        mark("|", paper.ENC["random_marker"], ms=6, mew=1.0),
    ]
    labels = [
        "truth reader (a), negated set with interval below 0.5 (b)",
        "honesty persona reader",
        "text-only classifier (line) and its interval (band)",
        "answer log-probability",
        "affirmative set",
        "negated set, otherwise",
        "random direction, negated set",
    ]
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=2,
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
    excludes = stats.per_role(
        {
            m: not (_difference(data[m]).lo <= 0.0 <= _difference(data[m]).hi)
            for m in models
        }
    )
    return {
        "tqa_inside": tqa,
        "neg_below_half": below,
        "diff_excludes_zero": excludes,
    }


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
        numbers[f"diff_excludes_zero_{role}"] = counts["diff_excludes_zero"][
            role
        ].hits
        numbers[f"diff_excludes_zero_n_{role}"] = counts["diff_excludes_zero"][
            role
        ].total
    numbers["negcities_below_random_gpt2"] = sum(
        data[m].sets["neg_cities"].value < data[m].random["neg_cities"]
        for m in models
        if stats.role(m) == "gpt2"
    )
    cities = [
        data[m].sets["cities"].value for m in models if stats.role(m) == "gpt2"
    ]
    numbers["gpt2_cities_min"] = min(cities)
    numbers["gpt2_cities_max"] = max(cities)
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
        "(a) MC1 accuracy on TruthfulQA of the truth reader (circle) and of "
        "the honesty persona reader (plus) at the layer chosen on "
        f"validation (whiskers are 95 percent Wilson intervals over "
        f"{numbers['n_questions']:.0f} questions); each of the two "
        "classifiers that see only the words is a green line at its "
        "accuracy with its own interval as a band. The answer "
        "log-probability is in its own panel with a Wilson interval and the "
        f"dotted chance level of {numbers['chance']:.3f}. The difference "
        "panel gives the reader minus the more accurate text-only "
        "classifier with the conservative interval d plus or minus 1.96 "
        "times the square root of p1(1-p1)/n plus p2(1-p2)/n over the same "
        f"{numbers['n_questions']:.0f} questions; intervals that exclude "
        f"zero: {stats.per_role_text(counts['diff_excludes_zero'])}. (b) "
        "Accuracy of the truth reader on statements about cities and on "
        "Spanish to English translations: a dumbbell from the affirmative "
        "set to its negation, filled where the Wilson interval of the "
        "negated set lies wholly below 0.5, with the accuracy of a random "
        "direction on the negated set (grey tick) and the region below "
        "chance shaded; whiskers in (b) are not drawn, the criterion uses "
        "95 percent Wilson intervals over the evaluation pairs of each "
        "set. GPT-2 reads affirmative cities at only "
        f"{numbers['gpt2_cities_min']:.2f} to {numbers['gpt2_cities_max']:.2f}"
        ", so small excursions there are not inversions of a strong reader. "
        "Checkpoints with at least one negated set whose interval lies "
        f"below 0.5 are {stats.per_role_text(counts['neg_below_half'])}. On "
        "negated cities the reader falls below the random direction by "
        f"point estimate on {int(numbers['negcities_below_random_gpt2'])} "
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
        add("a", model, "difference", _difference(data[model]))
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
    names = [paper.display_name(m) for m in rows]
    with paper.paper_style():
        fig = Figure(figsize=(paper.FULL_WIDTH, HEIGHT), layout="constrained")
        grid = fig.add_gridspec(2, 1)
        top = grid[0].subgridspec(1, 3, width_ratios=WIDTHS_TOP)
        bottom = grid[1].subgridspec(1, 2)
        slots = (
            (top[0, 0], "levels"),
            (top[0, 1], "difference"),
            (top[0, 2], "logprob"),
            (bottom[0, 0], "cities"),
            (bottom[0, 1], "translations"),
        )
        panels: list[Axes] = []
        for slot, gid in slots:
            ax = fig.add_subplot(slot, sharey=panels[0] if panels else None)
            ax.set_gid(gid)
            ax.tick_params(axis="y", length=0)
            panels.append(ax)
        _panel_levels(panels[0], rows, data)
        _panel_difference(panels[1], rows, data)
        _panel_logprob(panels[2], rows, data, chance)
        _panel_dumbbell(panels[3], rows, data, PAIRS[0])
        _panel_dumbbell(panels[4], rows, data, PAIRS[1])
        first, second, third, cities, translations = panels
        for ax in panels:
            ax.set_ylim(max(rows.values()) + 0.7, -0.7)
        for ax in (first, cities):
            ax.set_yticks(list(rows.values()), names)
        for ax in (second, third, translations):
            ax.tick_params(labelleft=False)
        for ax, letter in zip(panels, "abcde", strict=True):
            paper.panel_letter(ax, letter)
        for ax, title in (
            (first, "truth reader and text"),
            (second, "difference"),
            (third, "log-probability"),
            (cities, "cities"),
            (translations, "Spanish to English"),
        ):
            ax.set_title(title, fontsize=8, loc="center")
        _legend(fig)
    return FigureOutput(
        fig,
        _claim(numbers, counts),
        _detail(numbers, counts),
        numbers,
        _csv(models, data),
        counts,
    )
