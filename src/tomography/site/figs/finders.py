"""Figure 3: the paper's finder and logistic regression against a random one.

Three panels (sentiment, truth and utility) share rows, one per checkpoint.
Each marker is a finder's report-split accuracy minus that of a random
direction of the same run, at the layer each chose on its select split, with a
conservative interval for the test pairs. A grey band around zero is the
spread that test-pair noise alone gives for a random direction.
"""

import dataclasses
import math

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from tomography.site import finders as loader
from tomography.site import params
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.site.registry import RoleCount
from tomography.viz import axes as shared
from tomography.viz import paper

HEIGHT = 3.0
TASKS = ("sentiment", "truth", "utility")
TITLES = {"sentiment": "Sentiment", "truth": "Truth", "utility": "Utility"}
FINDERS = ("logistic", "pca_diff")
DODGE = {"logistic": -0.2, "pca_diff": 0.2}
STYLE = {
    "logistic": (paper.ENC["direction"], "o"),
    "pca_diff": (paper.ENC["paper"], "D"),
}
LABELS = {
    "logistic": "logistic regression",
    "pca_diff": "PCA of paired differences (paper)",
}
XLIM = (-0.30, 0.45)
BAND_HEIGHT = 0.6
MARKER = 3.5


def fmt3(value: float) -> str:
    """Writes a number with three decimals and no sign on a rounded zero.

    Args:
        value: The number.

    Returns:
        The text, such as `0.000` for minus 1e-16.
    """
    text = f"{value:.3f}"
    return text[1:] if text == "-0.000" else text


@dataclasses.dataclass(frozen=True)
class Point:
    """One finder on one task and checkpoint.

    Attributes:
        margin: Finder accuracy minus random accuracy.
        lo: Lower end of the conservative interval of the margin.
        hi: Upper end of the conservative interval of the margin.
        random: Accuracy of the random direction.
        n: Test pairs.
    """

    margin: float
    lo: float
    hi: float
    random: float
    n: int


def _points(manifest: Manifest, model: str) -> dict[tuple[str, str], Point]:
    cells = loader.load(manifest, model)
    if cells is None:
        raise FileNotFoundError(f"results/p4/{model} is missing")

    def report(task: str, method: str) -> float:
        return cells[task].rows[method].report

    found: dict[tuple[str, str], Point] = {}
    for task in TASKS:
        n = params.REPORT_PAIRS[task]
        random = report(task, "random")
        for finder in FINDERS:
            margin, lo, hi = stats.diff_interval(
                report(task, finder), random, n
            )
            found[(task, finder)] = Point(margin, lo, hi, random, n)
    return found


def _marker(ax: Axes, y: float, model: str, finder: str, point: Point) -> None:
    color, shape = STYLE[finder]
    filled = shared.model_style(model).filled
    y += DODGE[finder]
    ax.plot(
        [point.lo, point.hi],
        [y, y],
        color=color,
        lw=paper.WHISKER_WIDTH,
        solid_capstyle="butt",
        zorder=3,
    )
    ax.plot(
        [point.margin],
        [y],
        ls="none",
        marker=shape,
        ms=MARKER,
        color=color,
        mfc=color if filled else "white",
        mew=0.8,
        zorder=4,
    )


def _panel(
    ax: Axes,
    task: str,
    rows: dict[str, float],
    data: dict[str, dict[tuple[str, str], Point]],
) -> None:
    shared.ref_line(ax, 0.0, "x")
    for model, y in rows.items():
        point = data[model][(task, "logistic")]
        half = stats.Z * math.sqrt(
            2 * point.random * (1 - point.random) / point.n
        )
        ax.fill_between(
            [-half, half],
            y - BAND_HEIGHT / 2,
            y + BAND_HEIGHT / 2,
            color=paper.RANDOM_BAND,
            lw=0,
            zorder=1.5,
        )
        for finder in FINDERS:
            _marker(ax, y, model, finder, data[model][(task, finder)])
    ax.set_xlim(*XLIM)
    ax.set_xticks([-0.2, 0.0, 0.2, 0.4])
    ax.xaxis.set_major_formatter(shared.minus_formatter())
    ax.set_title(TITLES[task], fontsize=paper.TITLE, loc="left", x=0.04)
    ax.spines["left"].set_visible(False)


def _legend(fig: Figure) -> None:
    handles: list[Line2D | Patch] = [
        Line2D(
            [],
            [],
            ls="none",
            marker=STYLE[f][1],
            ms=MARKER,
            color=STYLE[f][0],
        )
        for f in FINDERS
    ]
    handles.append(Patch(fc=paper.RANDOM_BAND, ec="none"))
    fig.legend(
        handles,
        [*(LABELS[f] for f in FINDERS), "random direction, test-pair noise"],
        loc="outside upper center",
        ncol=3,
        frameon=False,
        fontsize=7,
        columnspacing=1.4,
    )


def _numbers(
    models: list[str], data: dict[str, dict[tuple[str, str], Point]]
) -> tuple[dict[str, float], dict[str, dict[str, RoleCount]]]:
    def flags(task: str, finder: str, interval: bool) -> dict[str, bool]:
        return {
            m: (
                data[m][(task, finder)].lo > 0
                if interval
                else data[m][(task, finder)].margin > 0
            )
            for m in models
        }

    counts = {
        "pca_sentiment": stats.per_role(flags("sentiment", "pca_diff", True)),
        "logistic_sentiment": stats.per_role(
            flags("sentiment", "logistic", True)
        ),
        "pca_utility_point": stats.per_role(
            flags("utility", "pca_diff", False)
        ),
    }
    numbers: dict[str, float] = {}
    for role in stats.ROLES:
        numbers[f"pca_sentiment_above_{role}"] = counts["pca_sentiment"][
            role
        ].hits
        numbers[f"pca_sentiment_point_above_{role}"] = stats.per_role(
            flags("sentiment", "pca_diff", False)
        )[role].hits
        numbers[f"logistic_sentiment_above_{role}"] = counts[
            "logistic_sentiment"
        ][role].hits
        numbers[f"pca_utility_point_above_{role}"] = counts[
            "pca_utility_point"
        ][role].hits
    gpt2 = [m for m in models if stats.role(m) == "gpt2"]
    numbers["pca_truth_gpt2_max"] = max(
        data[m][("truth", "pca_diff")].margin for m in gpt2
    )
    utility = [data[m][("utility", "pca_diff")].margin for m in models]
    numbers["pca_utility_min"] = min(utility)
    numbers["pca_utility_max"] = max(utility)
    numbers["random_max"] = max(
        data[m][(task, "pca_diff")].random for m in models for task in TASKS
    )
    return numbers, counts


def _where(counts: dict[str, RoleCount]) -> str:
    total = sum(c.total for c in counts.values())
    if all(c.hits == c.total for c in counts.values()):
        return f"all {total} checkpoints"
    return stats.per_role_text(counts)


def _winners(
    models: list[str],
    data: dict[str, dict[tuple[str, str], Point]],
    task: str,
    finder: str,
) -> str:
    names = [
        paper.display_name(m) for m in models if data[m][(task, finder)].lo > 0
    ]
    return f" ({', '.join(names)})" if names else ""


def _claim(
    models: list[str],
    data: dict[str, dict[tuple[str, str], Point]],
    numbers: dict[str, float],
    counts: dict[str, dict[str, RoleCount]],
) -> str:
    sentiment = counts["pca_sentiment"]
    return (
        "The paper's finder needs matched pairs. On pairs of unrelated "
        "sentences with opposite labels, the leading principal component of "
        "paired differences is above a random direction, with an interval "
        f"excluding zero, for sentiment on {_where(sentiment)}"
        f"{_winners(models, data, 'sentiment', 'pca_diff')}, and for truth "
        "its largest margin on a GPT-2 size is "
        f"{fmt3(numbers['pca_truth_gpt2_max'])}. Logistic regression is above "
        "the random direction for sentiment, with an interval excluding "
        f"zero, on {_where(counts['logistic_sentiment'])}. On utility, whose "
        "pairs are matched scenarios, the paper's finder is above the "
        "random direction by point estimate on "
        f"{_where(counts['pca_utility_point'])}, by "
        f"{fmt3(numbers['pca_utility_min'])} to "
        f"{fmt3(numbers['pca_utility_max'])}."
    )


_PAIRS = (
    f"{params.REPORT_PAIRS['sentiment']} sentiment, "
    f"{params.REPORT_PAIRS['truth']} truth and "
    f"{params.REPORT_PAIRS['utility']} utility pairs"
)
DETAIL = (
    "Each marker is a finder's accuracy on the report split minus that of "
    "a random direction of the same run, each at the layer its select split "
    "chose (filled for GPT-2 and Qwen3 base checkpoints, open for "
    "post-trained ones); whiskers are conservative 95 percent intervals for "
    f"the test pairs ({_PAIRS}); the "
    "grey band is the spread that test-pair noise alone gives around the "
    "random direction. Layer curves of these finders are in Figures A2 and "
    "A3, all six finders in Table A2."
)


def _csv(
    models: list[str], data: dict[str, dict[tuple[str, str], Point]]
) -> str:
    lines = ["task,model,role,finder,margin,lo,hi,random,n"]
    for task in TASKS:
        for model in models:
            for finder in FINDERS:
                p = data[model][(task, finder)]
                lines.append(
                    f"{task},{model},{stats.role(model)},{finder},"
                    f"{p.margin:.10g},{p.lo:.10g},{p.hi:.10g},"
                    f"{p.random:.10g},{p.n}"
                )
    return "\n".join(lines) + "\n"


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    models = manifest.models("p4")
    data = {m: _points(manifest, m) for m in models}
    rows = shared.model_rows(models)
    numbers, counts = _numbers(models, data)
    with paper.paper_style():
        fig = Figure(figsize=(paper.FULL_WIDTH, HEIGHT), layout="constrained")
        axes = fig.subplots(
            1, 3, sharey=True, gridspec_kw={"width_ratios": [1.45, 1, 1]}
        )
        for ax, task, letter in zip(axes, TASKS, "abc", strict=True):
            _panel(ax, task, rows, data)
            paper.panel_letter(ax, letter)
        axes[0].set_yticks(
            list(rows.values()), [paper.display_name(m) for m in rows]
        )
        axes[0].set_ylim(max(rows.values()) + 0.7, -0.7)
        for ax in axes:
            ax.tick_params(axis="y", length=0)
        fig.supxlabel(
            "finder accuracy minus random direction", fontsize=paper.LABEL
        )
        _legend(fig)
    return FigureOutput(
        fig,
        _claim(models, data, numbers, counts),
        DETAIL,
        numbers,
        _csv(models, data),
        counts,
    )
