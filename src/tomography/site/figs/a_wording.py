"""Figure A3: Annotated transfer matrices.

The sentiment transfer matrix of every checkpoint with a logistic P1 run,
with the accuracy written in each cell. A row is the wording a reader was
fitted on and a column the wording it is evaluated on.
"""

import json
import string

from matplotlib.figure import Figure
import numpy as np

from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import axes as shared
from tomography.viz import paper

DROPPED = "no_trailing_space"
NOTE_SIZE = 3.6
TICK_SIZE = 4.2


def _matrix(manifest: Manifest, model: str) -> tuple[list[str], np.ndarray]:
    run = manifest.run("p1", model)
    if run is None:
        raise FileNotFoundError(f"results/p1/{model} has no logistic run")
    labels = json.loads((run / "transfer_sentiment.json").read_text("utf-8"))[
        "data"
    ]
    with np.load(run / "transfer_sentiment.npz") as npz:
        values = np.asarray(npz["values"], dtype=float)
    keep = [i for i, v in enumerate(labels["rows"]) if v != DROPPED]
    return [labels["rows"][i] for i in keep], values[np.ix_(keep, keep)]


def _text(value: float) -> str:
    return "1." if value >= 0.995 else f"{value:.2f}"[1:]


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    models = [m for m in manifest.models("p1") if manifest.run("p1", m)]
    rows: list[list[str]] = []
    lows: list[float] = []
    with paper.paper_style():
        fig = Figure(figsize=(paper.FULL_WIDTH, 8.4), layout="constrained")
        grid = fig.add_gridspec((len(models) + 1) // 2, 2)
        letters = iter(string.ascii_lowercase)
        for i, model in enumerate(models):
            ax = fig.add_subplot(grid[i // 2, i % 2])
            names, matrix = _matrix(manifest, model)
            shared.heatmap(ax, matrix, vmin=0.5, vmax=1.0)
            lows.append(float(matrix.min()))
            for r, fit in enumerate(names):
                for c, ev in enumerate(names):
                    ax.text(
                        c,
                        r,
                        _text(matrix[r, c]),
                        ha="center",
                        va="center",
                        fontsize=NOTE_SIZE,
                        color=paper.INK if matrix[r, c] < 0.85 else "white",
                    )
                    rows.append(
                        [
                            f"matrix {fit}",
                            model,
                            ev,
                            f"{matrix[r, c]:.6g}",
                            "",
                            "",
                            "",
                        ]
                    )
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=90, fontsize=TICK_SIZE)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=TICK_SIZE)
            ax.tick_params(length=1.5, pad=1)
            ax.set_title(
                paper.display_name(model), fontsize=paper.TITLE, x=0.18
            )
            paper.panel_letter(ax, next(letters))
    numbers = {"lowest_min": min(lows), "lowest_max": max(lows)}
    claim = (
        "The lowest transfer accuracy in a matrix is "
        f"{min(lows):.3f} to {max(lows):.3f} across the {len(models)} "
        "checkpoints, so every cell is above chance."
    )
    detail = (
        "Accuracy of the sentiment reader fitted on the wording in the row "
        "and evaluated on the wording in the column, at the layer of the "
        "paper wording, for every checkpoint with a logistic P1 run; the "
        "diagonal is outlined and the color scale runs from 0.5 to 1. The "
        "wording with no trailing space is dropped because the post-trained "
        "models do not have it."
    )
    return FigureOutput(fig, claim, detail, numbers, _csv(rows))


def _csv(rows: list[list[str]]) -> str:
    header = "panel,model,x,y,lo,hi,baseline"
    return "\n".join([header, *(",".join(r) for r in rows)]) + "\n"
