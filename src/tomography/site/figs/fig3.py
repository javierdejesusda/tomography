"""Figure 3: The paper's PCA finder against supervised finders.

Rows are the three P4 tasks and columns are GPT-2 XL and two Qwen3 base
models. Each panel draws the report-split accuracy of three finders over
relative depth against the band of random directions, with a dot at the layer
each finder selected on its select split.
"""

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from tomography.site import finders
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import axes as shared
from tomography.viz import paper

GPT2 = ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")
FIRST = "gpt2-xl"
QWEN = "qwen3-1.7b-base"
EXTRA = ("qwen3-4b-base", "qwen3-0.6b-base")
COLORS = {
    "pca_diff": paper.INK,
    "mean_diff": paper.SERIES[2],
    "logistic": paper.SERIES[3],
}
LABELS = {
    "pca_diff": "PCA of differences (paper)",
    "mean_diff": "mean difference",
    "logistic": "logistic regression",
}
TASK_NAMES = {"sentiment": "Sentiment", "truth": "Truth", "utility": "Utility"}


def _columns(manifest: Manifest) -> list[str]:
    available = manifest.models("p4")
    extra = next(m for m in EXTRA if m in available)
    return sorted({FIRST, QWEN, extra}, key=_rank)


def _rank(model: str) -> tuple[int, str]:
    from tomography.site.store import model_rank

    return model_rank(model)


def _margin(
    data: dict[str, dict[str, finders.Cell]],
    model: str,
    task: str,
    method: str,
) -> float:
    rows = data[model][task].rows
    return round(rows[method].report - rows["random"].report, 6) + 0.0


def _panel(
    ax: Axes,
    cell: finders.Cell,
    model: str,
    task: str,
    rows: list[list[str]],
) -> None:
    ax.axhline(0.5, color=paper.CHANCE, lw=0.5, ls=(0, (3, 2)), zorder=1)
    mean, lo, hi = cell.curves["random"]
    ax.fill_between(
        cell.depth, lo, hi, color=paper.RANDOM_BAND, lw=0, zorder=1.5
    )
    ax.plot(cell.depth, mean, color=paper.CHANCE, lw=0.7, zorder=2)
    for method, color in COLORS.items():
        mean, _, _ = cell.curves[method]
        ax.plot(cell.depth, mean, color=color, lw=1.0, zorder=3)
        row = cell.rows[method]
        ax.plot(
            [cell.row_depth(method)],
            [row.report],
            ls="none",
            marker="o",
            ms=3.5,
            color=color,
            zorder=4,
        )
    random_row = cell.rows["random"]
    for method in (*COLORS, "random"):
        mean, lo, hi = cell.curves[method]
        for i, depth in enumerate(cell.depth):
            rows.append(
                [
                    f"{task} {method}",
                    model,
                    f"{depth:.6g}",
                    f"{mean[i]:.6g}",
                    f"{lo[i]:.6g}",
                    f"{hi[i]:.6g}",
                    f"{cell.curves['random'][0][i]:.6g}",
                ]
            )
    del random_row
    shared.depth_axis(ax)
    ax.set_ylim(0.4, 1.0)
    ax.set_yticks([0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.5", "0.75", "1"])


def _legend(fig: Figure) -> None:
    handles: list[Line2D | Patch] = [
        Line2D([], [], color=c, lw=1.0, marker="o", ms=3.5)
        for c in COLORS.values()
    ]
    handles.append(Patch(fc=paper.RANDOM_BAND, ec="none"))
    labels = [*LABELS.values(), "random directions"]
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=4,
        frameon=False,
        fontsize=paper.TICK,
        columnspacing=1.2,
        handlelength=1.6,
    )


def _claim(numbers: dict[str, float]) -> str:
    return (
        "On every GPT-2 size the paper's PCA finder is at most "
        f"{numbers['gpt2_pca_over_random_max']:.3f} above a random direction "
        "for sentiment and truth, where logistic regression is at least "
        f"{numbers['gpt2_logistic_over_random_min']:.3f} above it, while on "
        "Qwen3-1.7B-Base the PCA finder reads all three tasks, at least "
        f"{numbers['qwen_pca_over_random_min']:.3f} above random."
    )


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    columns = _columns(manifest)
    data: dict[str, dict[str, finders.Cell]] = {}
    for model in {*columns, *GPT2, QWEN}:
        loaded = finders.load(manifest, model)
        if loaded is None:
            raise FileNotFoundError(f"results/p4/{model} is missing")
        data[model] = loaded
    rows: list[list[str]] = []
    with paper.paper_style():
        fig = Figure(figsize=(paper.FULL_WIDTH, 5.4), layout="constrained")
        grid = fig.add_gridspec(3, 3)
        letters = iter("abcdefghi")
        for r, task in enumerate(finders.TASKS):
            for c, model in enumerate(columns):
                ax = fig.add_subplot(grid[r, c])
                _panel(ax, data[model][task], model, task, rows)
                paper.panel_letter(ax, next(letters))
                if r == 0:
                    ax.set_title(
                        paper.display_name(model),
                        fontsize=paper.TITLE,
                        x=0.06,
                    )
                if c == 0:
                    ax.set_ylabel(f"{TASK_NAMES[task]} accuracy")
                if r < 2:
                    ax.set_xlabel("")
        _legend(fig)
    hard = ("sentiment", "truth")
    numbers = {
        "gpt2_pca_over_random_max": max(
            _margin(data, m, t, "pca_diff") for m in GPT2 for t in hard
        ),
        "gpt2_logistic_over_random_min": min(
            _margin(data, m, t, "logistic") for m in GPT2 for t in hard
        ),
        "qwen_pca_over_random_min": min(
            _margin(data, QWEN, t, "pca_diff") for t in finders.TASKS
        ),
    }
    detail = (
        "Report-split accuracy of three direction finders over relative "
        "depth on the pairing the task has (label-contrastive for sentiment "
        "and truth, matched for utility) with centered activations; dots "
        "mark each finder's layer chosen on the select split. The grey "
        "band is the interval over seeds of random directions and the "
        "grey line its mean; the dashed line is chance. The claim compares "
        "report-split accuracies at each finder's selected layer. Qwen3 "
        "4B has no P4 run, so the third column is the 0.6B base model when "
        "it is missing. The other finders and sizes are in Table 3 and the "
        "appendix."
    )
    return FigureOutput(fig, _claim(numbers), detail, numbers, _csv(rows))


def _csv(rows: list[list[str]]) -> str:
    header = "panel,model,x,y,lo,hi,baseline"
    return "\n".join([header, *(",".join(r) for r in rows)]) + "\n"
