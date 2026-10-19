"""Figure 6: a truth reader does not beat words alone on TruthfulQA.

Panel (a) puts the MC1 accuracy of the truth reader and of the honesty
persona reader on TruthfulQA beside two classifiers that see only the words.
Panel (b) is their difference with a conservative interval, and panel (c) the
answer log-probability with its chance level. The statement sets with
negation, where the reader falls below chance, are in the appendix figure
`a_truth_cities`, which reuses the loaders of this module.
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

HEIGHT = 3.6
TOP = -2.2
READER_NAMES = {
    "truthfulqa_reader": "truth reader",
    "persona_reader": "honesty persona reader",
}
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
XLIM_A = (0.1, 1.0)
BAND_NOTE_GAP = 0.012
XLIM_C = (0.1, 0.8)
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
        affirmative_random: Affirmative set to the accuracy of its random
            direction at the same layer.
    """

    reads: dict[str, Cell]
    sets: dict[str, Cell]
    random: dict[str, float]
    text: dict[str, Cell]
    words: float
    n_questions: int
    affirmative_random: dict[str, float]


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
        {name: _random_at(other, name, layer) for name, _, _ in PAIRS},
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


def point(
    ax: Axes, x: float, y: float, shape: str, gid: str, **kw: Any
) -> None:
    """Draws one marker with a gid.

    Args:
        ax: The panel.
        x: Horizontal position.
        y: Vertical position.
        shape: Marker shape.
        gid: Artist id.
        **kw: Further `Axes.plot` keywords; `ms` and `color` have defaults.
    """
    (line,) = ax.plot(
        [x],
        [y],
        ls="none",
        marker=shape,
        ms=kw.pop("ms", MARKER),
        color=kw.pop("color", paper.INK),
        mew=0.8,
        zorder=kw.pop("zorder", 4),
        **kw,
    )
    line.set_gid(gid)


def chance_line(ax: Axes, value: float) -> None:
    """Draws the dotted vertical chance line.

    Args:
        ax: The panel.
        value: Where the line sits.
    """
    line = ax.axvline(
        value,
        color=paper.ENC["reference"],
        ls=paper.LINESTYLES["reference"],
        lw=paper.REFERENCE_WIDTH,
        zorder=1,
    )
    line.set_gid("chance")


def _chance_label(ax: Axes, value: float) -> None:
    label = ax.annotate(
        "chance",
        xy=(value, TOP + 0.15),
        xytext=(4, 0),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=TEXT,
        color=paper.ENC["secondary_text"],
    )
    label.set_gid("chance_label")


def _is_post(model: str) -> bool:
    return model.startswith("qwen3") and not model.endswith("-base")


def _open(model: str) -> dict[str, str]:
    return {"mfc": "white"} if _is_post(model) else {}


def _text_levels(ax: Axes, first: Model) -> None:
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
        line = ax.axvline(cell.value, color=paper.ENC["text"], lw=1.2, zorder=2)
        line.set_gid(f"text:{method}")


def _panel_levels(
    ax: Axes, rows: dict[str, float], data: dict[str, Model], chance: float
) -> None:
    first = next(iter(data.values()))
    chance_line(ax, chance)
    _chance_label(ax, chance)
    _text_levels(ax, first)
    for model, y in rows.items():
        for name, (_, shape, dodge) in READERS.items():
            cell = data[model].reads[name]
            color = PERSONA if name == "persona_reader" else paper.INK
            _whisker(ax, y + dodge, cell, f"{name}:{model}", color)
            point(
                ax,
                cell.value,
                y + dodge,
                shape,
                f"{name}:{model}",
                color=color,
                **_open(model),
            )
    top_model = min(rows, key=lambda m: rows[m])
    top = rows[top_model]
    name = "truthfulqa_reader"
    cell = data[top_model].reads[name]
    note = ax.annotate(
        READER_NAMES[name],
        xy=(cell.lo, top + READERS[name][2]),
        xytext=(-4, 4),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=TEXT,
        color=paper.ENC["secondary_text"],
    )
    note.set_gid(f"reader_label:{name}")
    hi = max(first.text[m].hi for m in TEXT_METHODS)
    band_note = ax.text(
        hi + BAND_NOTE_GAP,
        TOP + 0.15,
        "words,\ncharacters",
        ha="left",
        va="top",
        fontsize=TEXT,
        color=paper.ENC["secondary_text"],
    )
    band_note.set_gid("band_label")
    ax.set_xlim(*XLIM_A)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("MC1 accuracy on TruthfulQA")
    ax.spines["left"].set_visible(False)


def _panel_difference(
    ax: Axes, rows: dict[str, float], data: dict[str, Model]
) -> None:
    zero = ax.axvline(
        0.0,
        color=paper.ENC["reference"],
        ls=paper.LINESTYLES["reference"],
        lw=paper.REFERENCE_WIDTH,
        zorder=1.5,
        label="zero",
    )
    zero.set_gid("zero")
    for model, y in rows.items():
        cell = _difference(data[model])
        _whisker(ax, y, cell, f"diff:{model}")
        point(ax, cell.value, y, DIRECTION, f"diff:{model}", **_open(model))
    ax.set_xlim(*XLIM_D)
    ax.set_xticks([-0.2, 0.0, 0.2])
    ax.set_xlabel("reader minus words")
    ax.spines["left"].set_visible(False)


def _panel_logprob(
    ax: Axes, rows: dict[str, float], data: dict[str, Model], chance: float
) -> None:
    ax.set_xlim(*XLIM_C)
    chance_line(ax, chance)
    _chance_label(ax, chance)
    _text_levels(ax, next(iter(data.values())))
    for model, y in rows.items():
        cell = data[model].reads[LOG_PROBABILITY]
        _whisker(ax, y, cell, f"{LOG_PROBABILITY}:{model}")
        point(
            ax,
            cell.value,
            y,
            PROBABILITY,
            f"{LOG_PROBABILITY}:{model}",
            color=paper.ENC["own_probability"],
            **_open(model),
        )
    ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("MC1 accuracy")
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
        Line2D([], [], color=paper.ENC["text"], lw=1.2),
        mark(PROBABILITY, paper.ENC["own_probability"]),
        mark(DIRECTION, PERSONA, mfc=PERSONA, mew=0.8),
        mark(DIRECTION, ink, mfc=ink, mew=0.8),
        mark(DIRECTION, ink, mfc="white", mew=0.8),
    ]
    labels = [
        "text-only classifiers on words and\n"
        "characters (lines), each with its\n"
        "95 percent interval (band)",
        "answer log-probability",
        READER_NAMES["persona_reader"],
        "GPT-2 and Qwen3 base",
        "post-trained",
    ]
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=2,
        frameon=False,
        fontsize=TEXT,
        columnspacing=2.5,
        labelspacing=0.8,
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
    numbers: dict[str, float] = {
        "text_lo": lo,
        "text_hi": hi,
        "chance": chance,
        "text_word": data[models[0]].text["text_word"].value,
        "text_char": data[models[0]].text["text_char"].value,
        "n_questions": float(data[models[0]].n_questions),
    }
    for role in stats.ROLES:
        group = [m for m in models if stats.role(m) == role]
        reads = [data[m].reads["persona_reader"].value for m in group]
        numbers[f"persona_{role}_min"] = min(reads)
        numbers[f"persona_{role}_max"] = max(reads)
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
    random = [
        data[m].affirmative_random["cities"]
        for m in models
        if stats.role(m) == "gpt2"
    ]
    numbers["gpt2_cities_random_min"] = min(random)
    numbers["gpt2_cities_random_max"] = max(random)
    return numbers


def _persona_text(numbers: dict[str, float]) -> str:
    return stats.join_and(
        [
            f"{numbers[f'persona_{r}_min']:.3f} to "
            f"{numbers[f'persona_{r}_max']:.3f} on {stats.ROLE_NAMES[r]}"
            for r in stats.ROLES
            if f"persona_{r}_min" in numbers
        ]
    )


def _claim(
    numbers: dict[str, float], counts: dict[str, dict[str, RoleCount]]
) -> str:
    return (
        "A truth reader does not beat words alone on TruthfulQA: its MC1 "
        "accuracy lies inside the span of the 95 percent intervals of two "
        f"text-only classifiers ({numbers['text_lo']:.3f} to "
        f"{numbers['text_hi']:.3f}; their accuracies are "
        f"{numbers['text_word']:.3f} and {numbers['text_char']:.3f}) on "
        f"{stats.per_role_text(counts['tqa_inside'])}, while the honesty "
        f"persona reader reaches {_persona_text(numbers)}, against a chance "
        f"level of {numbers['chance']:.3f}."
    )


def _detail(
    numbers: dict[str, float], counts: dict[str, dict[str, RoleCount]]
) -> str:
    n = f"{numbers['n_questions']:.0f}"
    return (
        "(a) MC1 accuracy of the truth reader (circle) and of the honesty "
        "persona reader (dark grey circle; open for post-trained) at the "
        "layer chosen on validation, with 95 percent Wilson intervals over "
        f"{n} questions; each text-only classifier is a green line (the two "
        "nearly coincide) with its own interval as a band; the dotted line "
        f"is the chance level of {numbers['chance']:.3f}. (b) The reader "
        "minus the more accurate classifier, with the conservative "
        "1.96-standard-error interval of a difference of two proportions "
        "over the same "
        f"{n} questions; intervals that exclude zero: "
        f"{stats.per_role_text(counts['diff_excludes_zero'])}. (c) The "
        "answer log-probability with a Wilson interval beside the band. "
        "Negation statements are in [[fig:a_truth_cities]]."
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
    return "\n".join(lines) + "\n"


def summarize(
    models: list[str], data: dict[str, Model], chance: float
) -> tuple[dict[str, float], dict[str, dict[str, RoleCount]]]:
    """Computes the numbers and the per-role counts of the truth figures.

    Args:
        models: Checkpoint keys.
        data: Their data.
        chance: The TruthfulQA chance level.

    Returns:
        The numbers and the counts.
    """
    counts = _counts(models, data)
    return _numbers(models, data, counts, chance), counts


def load(manifest: Manifest) -> tuple[list[str], dict[str, Model], float]:
    """Loads the checkpoints of the truth figures.

    Args:
        manifest: The run resolver.

    Returns:
        The checkpoint keys, their data and the TruthfulQA chance level.
    """
    models = manifest.models("r6")
    data = {m: _model(manifest, m) for m in models}
    path = manifest.run("r6", models[0])
    assert path is not None
    chance = float(runfiles.data(path, "curves_truthfulqa.json")["chance"])
    return models, data, chance


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    models, data, chance = load(manifest)
    counts = _counts(models, data)
    numbers = _numbers(models, data, counts, chance)
    rows = shared.model_rows(models)
    names = [paper.display_name(m) for m in rows]
    with paper.paper_style():
        fig = Figure(figsize=(paper.FULL_WIDTH, HEIGHT), layout="constrained")
        grid = fig.add_gridspec(1, 3, width_ratios=WIDTHS_TOP)
        panels: list[Axes] = []
        for col, gid in enumerate(("levels", "difference", "logprob")):
            ax = fig.add_subplot(
                grid[0, col], sharey=panels[0] if panels else None
            )
            ax.set_gid(gid)
            ax.tick_params(axis="y", length=0)
            panels.append(ax)
        first, second, third = panels
        _panel_levels(first, rows, data, chance)
        _panel_difference(second, rows, data)
        _panel_logprob(third, rows, data, chance)
        for ax in panels:
            ax.set_ylim(max(rows.values()) + 0.7, TOP)
        first.set_yticks(list(rows.values()), names)
        for ax in (second, third):
            ax.tick_params(labelleft=False)
        for ax, letter in zip(panels, "abc", strict=True):
            paper.panel_letter(ax, letter)
        for ax, title in (
            (first, "truth reader and text"),
            (second, "difference"),
            (third, "log-probability"),
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
