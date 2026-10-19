"""Appendix figure: statements with negation pull a truth reader below chance.

Two panels, cities and Spanish to English translations. Each draws, for every
checkpoint, a dumbbell from the accuracy of the truth reader on the
affirmative statement set to its accuracy on the negated set. The negated end
is filled where its 95 percent Wilson interval lies wholly below 0.5, and a
grey tick marks the random direction on the negated set.
"""

from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from tomography.site import stats
from tomography.site.figs import truth
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.site.registry import RoleCount
from tomography.viz import axes as shared
from tomography.viz import paper

HEIGHT = 3.2


TICK = paper.ENC["own_probability"]
TICK_WIDTH = 1.0


def _panel_dumbbell(
    ax: Axes,
    rows: dict[str, float],
    data: dict[str, truth.Model],
    pair: tuple[str, str, str],
) -> None:
    plain, neg, _ = pair
    ax.set_xlim(*truth.XLIM_B)
    shade = ax.axvspan(
        truth.XLIM_B[0],
        0.5,
        color=paper.RANDOM_BAND,
        alpha=0.5,
        lw=0,
        zorder=0.5,
    )
    shade.set_gid("below_chance")
    truth.chance_line(ax, 0.5)
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
            color=TICK,
            lw=TICK_WIDTH,
            zorder=3,
        )
        tick.set_gid(f"random:{plain}:{model}")
        truth.point(
            ax,
            affirmative.value,
            y,
            "|",
            f"aff:{plain}:{model}",
            ms=truth.SMALL_TICK,
        )
        below = negated.hi < 0.5
        truth.point(
            ax,
            negated.value,
            y,
            truth.DIRECTION,
            f"neg:{plain}:{model}",
            ms=truth.MARKER + 0.6,
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
            ms=kw.pop("ms", truth.MARKER),
            color=color,
            **kw,
        )

    handles = [
        mark("|", ink, ms=truth.SMALL_TICK, mew=1.0),
        mark(truth.DIRECTION, ink, ms=truth.MARKER + 0.6, mfc=ink, mew=0.8),
        mark(truth.DIRECTION, ink, ms=truth.MARKER + 0.6, mfc="white", mew=0.8),
        mark("|", TICK, ms=7, mew=TICK_WIDTH),
    ]
    fig.legend(
        handles,
        [
            "affirmative set",
            "negated set, interval below 0.5",
            "negated set, otherwise",
            "random direction, negated set",
        ],
        loc="outside upper center",
        ncol=2,
        frameon=False,
        fontsize=truth.TEXT,
        columnspacing=2.5,
        handletextpad=0.4,
    )


def _claim(counts: dict[str, dict[str, RoleCount]]) -> str:
    return (
        "Statements with negation pull a truth reader below chance on some "
        "checkpoints: the accuracy on a negated set has a 95 percent Wilson "
        "interval wholly below 0.5 for at least one set on "
        f"{stats.per_role_text(counts['neg_below_half'])}."
    )


def _detail(
    numbers: dict[str, float], counts: dict[str, dict[str, RoleCount]]
) -> str:
    gpt2 = counts["neg_below_half"]["gpt2"].total
    return (
        "Accuracy of the truth reader, at the layer chosen on validation, on "
        "statements about cities (a) and on Spanish to English translations "
        "(b): a dumbbell from the affirmative set (bar) to its negation "
        "(circle, filled where the Wilson interval of the negated set lies "
        "wholly below 0.5), with the accuracy of a random direction on the "
        "negated set as a grey tick and the region below chance shaded. "
        "Whiskers are not drawn; the criterion uses 95 percent Wilson "
        "intervals over the evaluation pairs of each set. GPT-2 reads "
        "affirmative cities at only "
        f"{numbers['gpt2_cities_min']:.2f} to {numbers['gpt2_cities_max']:.2f}"
        f", against {numbers['gpt2_cities_random_min']:.2f} to "
        f"{numbers['gpt2_cities_random_max']:.2f} for its random direction, "
        "so small excursions there are not inversions of a strong reader. "
        "On negated cities the reader falls below the random direction by "
        f"point estimate on {int(numbers['negcities_below_random_gpt2'])} "
        f"of {gpt2} GPT-2 sizes. The truth reader on TruthfulQA is in "
        "[[fig:truth]]."
    )


def _csv(models: list[str], data: dict[str, truth.Model]) -> str:
    lines = ["panel,model,role,series,value,lo,hi"]
    for model in models:
        for series, cell in data[model].sets.items():
            lines.append(
                f"b,{model},{stats.role(model)},{series},"
                f"{cell.value:.10g},{cell.lo:.10g},{cell.hi:.10g}"
            )
        for series, value in data[model].random.items():
            lines.append(
                f"b,{model},{stats.role(model)},random_{series},"
                f"{value:.10g},{value:.10g},{value:.10g}"
            )
    return "\n".join(lines) + "\n"


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    models, data, chance = truth.load(manifest)
    numbers, counts = truth.summarize(models, data, chance)
    rows = shared.model_rows(models)
    names = [paper.display_name(m) for m in rows]
    with paper.paper_style():
        fig = Figure(figsize=(paper.FULL_WIDTH, HEIGHT), layout="constrained")
        left, right = fig.subplots(1, 2, sharey=True)
        left.set_gid("cities")
        right.set_gid("translations")
        _panel_dumbbell(left, rows, data, truth.PAIRS[0])
        _panel_dumbbell(right, rows, data, truth.PAIRS[1])
        for ax in (left, right):
            ax.tick_params(axis="y", length=0)
            ax.set_ylim(max(rows.values()) + 0.7, -0.7)
        left.set_yticks(list(rows.values()), names)
        right.tick_params(labelleft=False)
        for ax, letter, title in (
            (left, "a", "cities"),
            (right, "b", "Spanish to English"),
        ):
            paper.panel_letter(ax, letter)
            ax.set_title(title, fontsize=8, loc="center")
        _legend(fig)
    return FigureOutput(
        fig,
        _claim(counts),
        _detail(numbers, counts),
        numbers,
        _csv(models, data),
        counts,
    )
