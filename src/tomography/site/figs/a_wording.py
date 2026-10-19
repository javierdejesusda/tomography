"""Appendix figure: Wording transfer as a margin over a random direction.

One matrix per checkpoint with a wording run. A cell is the accuracy of the
sentiment reader fitted on the wording of its row and scored on the wording of
its column, minus the accuracy of a random direction on the scored wording.
"""

import math
import string

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.patches import Rectangle
import numpy as np

from tomography.site import runfiles
from tomography.site import stats
from tomography.site.figs import wording
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import paper

DROPPED = "no_trailing_space"
ROW_HEIGHT = 1.9
MAX_HEIGHT = 9.5
VMAX = 0.45
VMIN = -0.15
BAR_TICKS = [-0.15, 0.0, 0.15, 0.3, 0.45]
TEXT = paper.MIN_TEXT
ORDER = (
    "qwen3-0.6b-base",
    "qwen3-0.6b",
    "qwen3-1.7b-base",
    "qwen3-1.7b",
    "qwen3-4b-base",
    "qwen3-4b",
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
)
KIND_LABELS = {
    "paper": "paper",
    "paraphrase": "paraphrase",
    "scaffold": "scaffold",
    "concept_swap": "concept swap",
}
FRAME = "frame"
STIPPLE = "////"


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


def _kinds(manifest: Manifest, model: str, names: list[str]) -> list[str]:
    path = manifest.run("p1", model)
    assert path is not None
    kind = {
        r["variant"]: r["kind"]
        for r in runfiles.rows(path, "table.json")
        if r["task"] == "sentiment"
    }
    return [KIND_LABELS.get(kind[n], FRAME) for n in names]


def _groups(labels: list[str]) -> list[tuple[str, int, int]]:
    found: list[tuple[str, int, int]] = []
    for i, label in enumerate(labels):
        if found and found[-1][0] == label:
            found[-1] = (label, found[-1][1], i + 1)
        else:
            found.append((label, i, i + 1))
    return found


def _frame(
    ax: Axes, xy: tuple[float, float], w: float, h: float, gid: str
) -> None:
    patch = Rectangle(
        xy,
        w,
        h,
        fill=False,
        edgecolor=paper.ENC["paper"],
        lw=0.9,
        zorder=4,
    )
    patch.set_gid(gid)
    ax.add_patch(patch)


def _draw(
    ax: Axes,
    matrix: np.ndarray,
    paper_index: int,
    groups: list[tuple[str, int, int]],
    noise: float,
    labelled: bool,
) -> AxesImage:
    k = matrix.shape[0]
    cmap = mpl.colormaps[paper.SEQUENTIAL_CMAP].with_extremes(
        bad=paper.ENC["random_band"]
    )
    norm = Normalize(vmin=VMIN, vmax=VMAX)
    shown = np.ma.masked_array(matrix, mask=np.eye(k, dtype=bool))
    image = ax.imshow(shown, cmap=cmap, norm=norm, interpolation="nearest")
    for i in range(k):
        for j in range(k):
            if i == j or matrix[i, j] > noise:
                continue
            cell = Rectangle(
                (j - 0.5, i - 0.5),
                1,
                1,
                fill=False,
                hatch=STIPPLE,
                edgecolor="white",
                lw=0,
                zorder=3,
            )
            cell.set_gid("stipple")
            ax.add_patch(cell)
            if matrix[i, j] > 0.0:
                continue
            (dot,) = ax.plot(
                [j],
                [i],
                ls="none",
                marker="o",
                ms=3.4,
                mfc="white",
                mec=paper.INK,
                mew=0.7,
                zorder=5,
            )
            dot.set_gid("zero_dot")
    _frame(ax, (-0.5, paper_index - 0.5), k, 1, "paper_row")
    _frame(ax, (paper_index - 0.5, -0.5), 1, k, "paper_col")
    for _, _, end in groups[:-1]:
        for line in (
            ax.axhline(end - 0.5, color="white", lw=0.4, zorder=3),
            ax.axvline(end - 0.5, color="white", lw=0.4, zorder=3),
        ):
            line.set_gid("separator")
    at = [start for _, start, _ in groups[1:]] + [k - 1]
    ticks = [str(i + 1) for i in at]
    ax.set_xticks(at, ticks, fontsize=TEXT)
    ax.set_yticks(at, ticks, fontsize=TEXT)
    ax.tick_params(length=2, pad=1)
    if labelled:
        for n, (label, start, end) in enumerate(groups):
            if end - start < 2:
                continue
            note = ax.annotate(
                label,
                xy=(0.0, (start + end - 1) / 2),
                xycoords=("axes fraction", "data"),
                xytext=(-19 - 11 * (n % 3), 0),
                textcoords="offset points",
                rotation=90,
                ha="center",
                va="center",
                fontsize=TEXT,
                color=paper.ENC["secondary_text"],
                annotation_clip=False,
            )
            note.set_gid("kind")
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
    available = [m for m in manifest.models("p1") if manifest.run("p1", m)]
    models = [m for m in ORDER if m in available]
    models += [m for m in available if m not in models]
    names, first = _matrix(manifest, models[0], None)
    margins = {models[0]: first}
    for model in models[1:]:
        margins[model] = _matrix(manifest, model, names)[1]
    numbers, sentence = summarize(margins)
    noise = {m: wording.noise(manifest, m) for m in models}
    numbers["min_margin"] = min(numbers[f"min_{m}"] for m in margins)
    value = numbers["min_margin"]
    low = (chr(0x2212) if value < 0 else "") + f"{abs(value):.2f}"
    numbers["noise_min"] = min(noise.values())
    numbers["noise_max"] = max(noise.values())
    numbers["n_below_noise"] = float(
        sum(
            int(
                (
                    (margins[m] <= noise[m]) & ~np.eye(len(names), dtype=bool)
                ).sum()
            )
            for m in models
        )
    )
    groups = _groups(_kinds(manifest, models[0], names))
    cols = 2
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
            image = _draw(
                ax,
                margins[model],
                paper_index,
                groups,
                noise[model],
                i % cols == 0,
            )
            ax.set_title(
                paper.display_name(model), fontsize=paper.TITLE, x=0.18
            )
            paper.panel_letter(ax, string.ascii_lowercase[i])
            if i % cols:
                ax.tick_params(labelleft=False)
            if i < len(models) - cols:
                ax.tick_params(labelbottom=False)
            axes.append(ax)
        assert image is not None
        bar = fig.colorbar(
            image,
            ax=axes,
            location="bottom",
            fraction=0.025,
            pad=0.02,
            shrink=0.6,
        )
        bar.set_ticks(BAR_TICKS)
        bar.ax.tick_params(labelsize=TEXT)
        bar.set_label("accuracy minus random direction", fontsize=TEXT)
        for edge in (numbers["noise_min"], numbers["noise_max"]):
            tick = bar.ax.axvline(edge, color="white", lw=1.2, ymax=0.7)
            tick.set_gid("noise_tick")
    claim = (
        "Accuracy of a reader fitted on one wording (row) and scored on "
        "another (column), minus the random direction on the scored "
        "wording, for every checkpoint with a wording run."
    )
    detail = (
        f"{sentence} Index numbers refer to [[table:wordings]], in the order "
        "of that table, with the kind of change named at the left; the "
        "purple frame marks the row and the column of the paper's wording and "
        "the grey diagonal, where the fitted and the scored wording are the "
        "same, is not coloured. Cells at or below the noise expected from test "
        "pairs alone for that checkpoint (the noise column of "
        "[[fig:wording]]) are stippled; the "
        f"{int(numbers['n_below'])} cells at or below zero (lowest "
        f"{low}) carry a dot; "
        "the white ticks on the colour bar mark that noise range, "
        f"{numbers['noise_min']:.3f} to {numbers['noise_max']:.3f}, and "
        f"{int(numbers['n_below_noise'])} cells "
        "fall there."
    )
    return FigureOutput(fig, claim, detail, numbers, _csv(margins, names))
