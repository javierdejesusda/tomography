"""Figure A3: The paper's finder and logistic regression on GPT-2, by layer.

Rows are checkpoints and columns are tasks. Each panel draws report-split
accuracy at every layer for the paper's finder and for logistic regression,
the seed interval of a random direction of the same run, and a dot at the
layer each finder chose on its select split. The Qwen3 finder figure reuses `split` for
Qwen3.
"""

from collections.abc import Callable

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from tomography.site import finders
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import axes as shared
from tomography.viz import paper

HEIGHT = 4.6
FINDERS = ("pca_diff", "logistic")
COLORS = {"pca_diff": paper.ENC["paper"], "logistic": paper.ENC["direction"]}
MARKERS = {
    "pca_diff": paper.MEASURE_MARKERS["paper_finder"],
    "logistic": paper.MEASURE_MARKERS["direction"],
}
TASK_NAMES = {"sentiment": "Sentiment", "truth": "Truth", "utility": "Utility"}
MARKER = 3.0
TEXT = 6.5
YLIM = (0.4, 1.0)


def _panel(
    ax: Axes,
    cell: finders.Cell,
    model: str,
    task: str,
    rows: list[list[str]],
) -> None:
    ax.axhline(
        0.5,
        color=paper.ENC["reference"],
        linestyle=paper.LINESTYLES["reference"],
        linewidth=paper.REFERENCE_WIDTH,
        zorder=1,
    )
    mean, lo, hi = cell.curves["random"]
    ax.fill_between(
        cell.depth, lo, hi, color=paper.RANDOM_BAND, lw=0, zorder=1.5
    )
    ax.plot(
        cell.depth,
        mean,
        color=paper.ENC["random_marker"],
        lw=paper.TREND_WIDTH,
        zorder=2,
    )
    series = {name: cell.curves[name] for name in (*FINDERS, "random")}
    for name, (values, low, high) in series.items():
        rows.extend(
            [task, model, name, f"{d:.6g}", f"{v:.6g}", f"{a:.6g}", f"{b:.6g}"]
            for d, v, a, b in zip(cell.depth, values, low, high, strict=True)
        )
    for name in FINDERS:
        values = cell.curves[name][0]
        ax.plot(
            cell.depth,
            values,
            color=COLORS[name],
            ls=paper.LINESTYLES[
                "solid" if shared.model_style(model).filled else "post_trained"
            ],
            lw=1.0,
            zorder=3,
        )
        row = cell.rows[name]
        depth = cell.row_depth(name)
        ax.plot(
            [depth],
            [row.report],
            ls="none",
            marker=MARKERS[name],
            ms=MARKER,
            color=COLORS[name],
            mec=paper.INK,
            mew=0.4,
            zorder=4,
        )
        rows.append(
            [
                task,
                model,
                f"selected_{name}",
                f"{depth:.6g}",
                f"{row.report:.6g}",
                "",
                "",
            ]
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(*YLIM)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels(["0.5", "1"])


def _legend(fig: Figure, post_trained: bool) -> None:
    handles = [
        Line2D(
            [],
            [],
            color=COLORS[name],
            lw=1.0,
            marker=MARKERS[name],
            ms=MARKER,
            mec=paper.INK,
            mew=0.4,
        )
        for name in FINDERS
    ]
    handles.append(
        Line2D(
            [],
            [],
            color=paper.RANDOM_BAND,
            lw=5,
            solid_capstyle="butt",
        )
    )
    labels = [
        "paper's finder",
        "logistic regression",
        "random direction (95 percent interval over seeds)",
    ]
    if post_trained:
        handles.append(
            Line2D(
                [],
                [],
                color=paper.INK,
                lw=1.0,
                ls=paper.LINESTYLES["post_trained"],
            )
        )
        labels.append("post-trained")
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


def split(
    manifest: Manifest,
    member: Callable[[str], bool],
    height: Callable[[int], float],
    family: str,
) -> FigureOutput:
    """Builds the finder curves of the checkpoints a predicate selects.

    Args:
        manifest: The run resolver.
        member: Whether a checkpoint key belongs to this figure.
        height: Figure height in inches from the number of rows.
        family: Family name used in the caption.

    Returns:
        The figure and its caption parts.
    """
    models = [m for m in manifest.models("p4") if member(m)]
    rows: list[list[str]] = []
    with paper.paper_style():
        fig = Figure(
            figsize=(paper.FULL_WIDTH, height(len(models))),
            layout="constrained",
        )
        grid = fig.add_gridspec(len(models), len(finders.TASKS))
        for r, model in enumerate(models):
            found = finders.load(manifest, model)
            if found is None:
                raise FileNotFoundError(f"results/p4/{model} is missing")
            for c, task in enumerate(finders.TASKS):
                ax = fig.add_subplot(grid[r, c])
                _panel(ax, found[task], model, task, rows)
                if r == 0:
                    ax.set_title(TASK_NAMES[task], fontsize=paper.TITLE, x=0.5)
                if c == 0:
                    ax.set_ylabel(
                        paper.display_name(model), fontsize=paper.LABEL
                    )
                    paper.panel_letter(ax, "abcdefgh"[r])
                else:
                    ax.tick_params(labelleft=False)
                if r == len(models) - 1:
                    ax.set_xlabel("relative depth")
                else:
                    ax.tick_params(labelbottom=False)
        _legend(fig, any(not shared.model_style(m).filled for m in models))
    claim = (
        "Report-split accuracy at every layer of the paper's finder "
        f"(purple) and logistic regression (black) on {family}, against the "
        "random direction of the same run (grey)."
    )
    detail = (
        "Dots mark the layer each finder's select split chose. The grey "
        "band is the 95 percent interval over seeds of the random "
        "direction, with its mean as a thin line, and the dotted line is "
        "chance. The other finders are in [[table:finders_all]]."
    )
    return FigureOutput(fig, claim, detail, {}, _csv(rows))


def _csv(rows: list[list[str]]) -> str:
    header = "task,model,series,depth,value,lo,hi"
    return "\n".join([header, *(",".join(r) for r in rows)]) + "\n"


def build(manifest: Manifest) -> FigureOutput:
    """Builds the GPT-2 figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    return split(
        manifest,
        lambda m: stats.role(m) == "gpt2",
        lambda _: HEIGHT,
        "GPT-2",
    )
