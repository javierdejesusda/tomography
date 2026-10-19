"""Figure A5: Wording transfer as a margin over a random direction.

One matrix per checkpoint with a wording run. A cell is the accuracy of the
sentiment reader fitted on the wording of its row and scored on the wording of
its column, minus the accuracy of a random direction on the scored wording.
"""

import math
import string

from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
import numpy as np

from tomography.site import runfiles
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import paper
from tomography.viz import style

DROPPED = "no_trailing_space"
ROW_HEIGHT = 1.85
MAX_HEIGHT = 8.0
WIDE_LIMIT = 8
VMAX = 0.45
VMIN = -0.15
BAR_TICKS = [-0.15, 0.0, 0.15, 0.3, 0.45]
TEXT = 6.0
DOT = 1.6


def _matrix(
    manifest: Manifest, model: str, names: list[str] | None
) -> tuple[list[str], np.ndarray]:
    path = manifest.run("p1", model)
    if path is None:
        raise FileNotFoundError(f"results/p1/{model} has no logistic run")
    labels = list(runfiles.data(path, "transfer_sentiment.json")["rows"])
    values = runfiles.arrays(path, "transfer_sentiment.npz")["values"]
    keep_names = names or [v for v in labels if v != DROPPED]
    keep = [labels.index(n) for n in keep_names]
    random = {
        r["variant"]: float(r["random"])
        for r in runfiles.rows(path, "table.json")
        if r["task"] == "sentiment"
    }
    scored = np.array([random[n] for n in keep_names])
    margin = np.asarray(values, dtype=float)[np.ix_(keep, keep)]
    return keep_names, margin - scored[None, :]


def summarize(margins: dict[str, np.ndarray]) -> tuple[dict[str, float], str]:
    """Computes the minima and the caption sentence of the matrices.

    Args:
        margins: Checkpoint key to its margin matrix.

    Returns:
        The numbers (`min_<model>`, `min_<role>_lo`, `min_<role>_hi` and
        `n_below`) and the sentence: the smallest margin per role when every
        cell is above zero, otherwise the count of cells at or below zero.
    """
    numbers: dict[str, float] = {}
    for model, matrix in margins.items():
        numbers[f"min_{model}"] = float(matrix.min())
    parts = []
    for role in stats.ROLES:
        group = [numbers[f"min_{m}"] for m in margins if stats.role(m) == role]
        if group:
            numbers[f"min_{role}_lo"] = min(group)
            numbers[f"min_{role}_hi"] = max(group)
            parts.append(
                f"{min(group):.3f} to {max(group):.3f} "
                f"on {stats.ROLE_NAMES[role]}"
            )
    numbers["n_below"] = float(
        sum(int((m <= 0).sum()) for m in margins.values())
    )
    if numbers["n_below"] == 0:
        listed = (
            ", ".join(parts[:-1]) + " and " + parts[-1]
            if len(parts) > 1
            else parts[0]
        )
        return numbers, (
            "Every cell is above its random direction: the smallest margin "
            f"in a matrix is {listed} checkpoints."
        )
    flags = {m: bool((v <= 0).any()) for m, v in margins.items()}
    where = stats.per_role_text(stats.per_role(flags))
    return numbers, (
        f"The margin is at or below zero in {int(numbers['n_below'])} cells, "
        f"in {where}."
    )


def _draw(ax: Axes, matrix: np.ndarray, paper_index: int) -> AxesImage:
    k = matrix.shape[0]
    image = ax.imshow(
        matrix,
        cmap=style.diverging_cmap().reversed(),
        norm=TwoSlopeNorm(vmin=VMIN, vcenter=0.0, vmax=VMAX),
        interpolation="nearest",
    )
    ax.plot(
        range(k),
        range(k),
        ls="none",
        marker="o",
        ms=DOT,
        mfc="white",
        mec="white",
        zorder=3,
    )
    ticks = [str(i + 1) for i in range(k)]
    ax.set_xticks(range(k), ticks, fontsize=TEXT, rotation=90)
    ax.set_yticks(range(k), ticks, fontsize=TEXT)
    ax.tick_params(length=2, pad=1)
    for axis in (ax.xaxis, ax.yaxis):
        tick = axis.get_major_ticks()[paper_index]
        line = tick.tick1line
        line.set_markeredgecolor(paper.ENC["paper"])
        line.set_markersize(4)
        line.set_markeredgewidth(1.2)
    return image


def _csv(margins: dict[str, np.ndarray], names: list[str]) -> str:
    lines = ["model,role,fit,score,margin"]
    for model, matrix in margins.items():
        for i, fit in enumerate(names):
            for j, score in enumerate(names):
                lines.append(
                    f"{model},{stats.role(model)},{fit},{score},"
                    f"{matrix[i, j]:.10g}"
                )
    return "\n".join(lines) + "\n"


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    models = [m for m in manifest.models("p1") if manifest.run("p1", m)]
    names, first = _matrix(manifest, models[0], None)
    margins = {models[0]: first}
    for model in models[1:]:
        margins[model] = _matrix(manifest, model, names)[1]
    numbers, sentence = summarize(margins)
    cols = 2 if len(models) <= WIDE_LIMIT else 3
    rows = math.ceil(len(models) / cols)
    paper_index = names.index("paper")
    with paper.paper_style():
        fig = Figure(
            figsize=(paper.FULL_WIDTH, min(MAX_HEIGHT, ROW_HEIGHT * rows)),
            layout="constrained",
        )
        grid = fig.add_gridspec(rows, cols)
        axes = []
        image = None
        for i, model in enumerate(models):
            ax = fig.add_subplot(grid[i // cols, i % cols])
            image = _draw(ax, margins[model], paper_index)
            ax.set_title(
                paper.display_name(model), fontsize=paper.TITLE, x=0.18
            )
            paper.panel_letter(ax, string.ascii_lowercase[i])
            if i % cols:
                ax.tick_params(labelleft=False)
            axes.append(ax)
        assert image is not None
        empty = rows * cols - len(models)
        if empty >= 2:
            slot = grid[rows - 1, cols - empty :].subgridspec(
                3, 1, height_ratios=[1.0, 0.16, 1.0]
            )
            bar = fig.colorbar(
                image, cax=fig.add_subplot(slot[1]), orientation="horizontal"
            )
        else:
            bar = fig.colorbar(
                image,
                ax=axes,
                location="top",
                fraction=0.03,
                pad=0.02,
                shrink=0.6,
            )
        bar.set_ticks(BAR_TICKS)
        bar.ax.tick_params(labelsize=TEXT)
        bar.set_label("accuracy minus random direction", fontsize=TEXT)
    claim = (
        "Accuracy of a reader fitted on one wording (row) and scored on "
        "another (column), minus the random direction on the scored "
        "wording, for every checkpoint with a wording run."
    )
    detail = (
        f"{sentence} Index numbers refer to [[table:wordings]]; the purple "
        "tick marks "
        "the paper's wording and white dots the diagonal, where the fitted "
        "and the scored wording are the same."
    )
    return FigureOutput(fig, claim, detail, numbers, _csv(margins, names))
