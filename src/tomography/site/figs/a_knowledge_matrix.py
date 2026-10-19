"""Figure A4: P9b transfer matrix with text baselines flagged.

The AUROC of a reader fitted on one question type and tested on another,
for the largest base checkpoint. A test type is starred when a classifier on
the words or the characters of the questions alone comes within a small
margin of the reader fitted on that type.
"""

import json

from matplotlib.figure import Figure
import numpy as np

from tomography.site import store
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import axes as shared
from tomography.viz import paper

SLACK = 0.02
NOTE_SIZE = 4.6
TICK_SIZE = 5.5


def _model(manifest: Manifest) -> str:
    bases = [m for m in manifest.models("p9b") if not store.is_post_trained(m)]
    return max(bases, key=store.model_rank)


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    model = _model(manifest)
    run = manifest.run("p9b", model)
    if run is None:
        raise FileNotFoundError(f"results/p9b/{model} is missing")
    meta = json.loads((run / "matrix.json").read_text("utf-8"))["data"]
    with np.load(run / "matrix.npz") as npz:
        values = np.asarray(npz["values"], dtype=float)
    table = json.loads((run / "table.json").read_text("utf-8"))["data"]
    by_type = {
        r[0]: dict(zip(table["columns"], r, strict=True)) for r in table["rows"]
    }
    names: list[str] = list(meta["cols"])
    flagged = [
        max(by_type[t]["text_word"] or 0.0, by_type[t]["text_char"] or 0.0)
        >= by_type[t]["logistic"] - SLACK
        for t in names
    ]
    rows: list[list[str]] = []
    with paper.paper_style():
        fig = Figure(figsize=(paper.FULL_WIDTH, 5.6), layout="constrained")
        ax = fig.add_subplot()
        shared.heatmap(ax, values, vmin=0.5, vmax=1.0)
        for r, fit in enumerate(meta["rows"]):
            for c, ev in enumerate(names):
                ax.text(
                    c,
                    r,
                    f"{values[r, c]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=NOTE_SIZE,
                    color="white" if values[r, c] > 0.85 else paper.INK,
                )
                rows.append(
                    [
                        f"matrix {fit}",
                        model,
                        ev,
                        f"{values[r, c]:.6g}",
                        "",
                        "",
                        "",
                    ]
                )
        labels = [
            t.replace("_", " ") + (" *" if f else "")
            for t, f in zip(names, flagged, strict=True)
        ]
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(labels, rotation=90, fontsize=TICK_SIZE)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(
            [t.replace("_", " ") for t in meta["rows"]], fontsize=TICK_SIZE
        )
        ax.set_xlabel("tested on question type")
        ax.set_ylabel("fitted on question type")
        ax.set_title(paper.display_name(model), fontsize=paper.TITLE, x=0.06)
        paper.panel_letter(ax, "a")
    numbers = {"flagged": float(sum(flagged)), "types": float(len(names))}
    claim = (
        f"A classifier on the words or characters alone comes within "
        f"{SLACK:.3f} of the reader on {sum(flagged)} of {len(names)} "
        "question types."
    )
    detail = (
        "AUROC of a logistic reader fitted on the question type in the row "
        "and tested on the type in the column, for "
        f"{paper.display_name(model)} at its validation layer; the diagonal "
        "is outlined. A star marks a test type where a classifier on the "
        "words or on the characters of the questions reaches an AUROC "
        "within the stated margin of the reader fitted on that type, so a "
        "high value in that column does not show that the direction reads "
        "knowledge."
    )
    return FigureOutput(fig, claim, detail, numbers, _csv(rows))


def _csv(rows: list[list[str]]) -> str:
    header = "panel,model,x,y,lo,hi,baseline"
    return "\n".join([header, *(",".join(r) for r in rows)]) + "\n"
