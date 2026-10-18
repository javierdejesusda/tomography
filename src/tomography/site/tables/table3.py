"""Table 3: Direction finders.

The reference cell of each P4 task for the random direction, the paper's PCA
finder and logistic regression, on every checkpoint with a P4 run. The footer
adds the M1 steering win rates.
"""

import json

from tomography.site import finders
from tomography.site.manifest import Manifest
from tomography.site.registry import TableOutput
from tomography.viz import paper

MARGIN = 0.1
METHODS = ("random", "pca_diff", "logistic")
M1_METHODS = ("pca_diff", "mean_diff", "logistic", "random")
TASK_NAMES = {"sentiment": "Sentiment", "truth": "Truth", "utility": "Utility"}


def _text(row: finders.Row) -> str:
    return f"{row.report:.3f} [{row.lo:.3f}, {row.hi:.3f}]"


def _m1_note(manifest: Manifest) -> str:
    parts = []
    for model in manifest.models("m1"):
        run = manifest.run("m1", model)
        if run is None:
            continue
        rows = json.loads((run / "table.json").read_text("utf-8"))["data"][
            "rows"
        ]
        wins = {
            r[0]: float(r[3])
            for r in rows
            if r[1] == "manipulation" and r[2] == "0.3"
        }
        values = " / ".join(f"{wins[m]:.3f}" for m in M1_METHODS)
        parts.append(f"{paper.display_name(model)} {values}")
    return (
        "M1 steering win rate on utility at strength 0.3 (paper PCA finder / "
        "mean difference / logistic regression / random direction), where "
        "0.5 is no effect: " + "; ".join(parts) + "."
    )


def build(manifest: Manifest) -> TableOutput:
    """Builds the table from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The table and its caption parts.
    """
    models = manifest.models("p4")
    loaded = {m: finders.load(manifest, m) for m in models}
    rows: list[list[str]] = []
    at_least = reads = cells = 0
    for task in finders.TASKS:
        for i, model in enumerate(models):
            found = loaded[model]
            if found is None:
                continue
            cell = found[task].rows
            rows.append(
                [
                    TASK_NAMES[task] if i == 0 else "",
                    paper.display_name(model),
                    *(_text(cell[m]) for m in METHODS),
                ]
            )
            cells += 1
            at_least += cell["logistic"].report >= cell["pca_diff"].report
            reads += cell["pca_diff"].report - cell["random"].report > MARGIN
    numbers = {
        "cells": float(cells),
        "logistic_at_least_pca": float(at_least),
        "pca_above_random": float(reads),
    }
    claim = (
        "Logistic regression reads at least as well as the paper's PCA "
        f"finder in {at_least} of {cells} task and model cells, and the PCA "
        f"finder is more than {MARGIN:.1f} above a random direction in "
        f"{reads} of {cells}."
    )
    detail = (
        "Report-split accuracy at the layer each finder chose on its select "
        "split, with the 95 percent interval over seeds, on the reference "
        "pairing of each task (label-contrastive for sentiment and truth, "
        "matched for utility) with centered activations. Random is the "
        "accuracy of a random direction in the same setting. Base and "
        "post-trained checkpoints are separate rows; there is no P4 run for "
        "Qwen3 4B."
    )
    return TableOutput(
        claim,
        detail,
        [
            "Task",
            "Model",
            "Random",
            "PCA of differences (paper)",
            "Logistic regression",
        ],
        rows,
        "llrrr",
        [_m1_note(manifest)],
        numbers,
    )
