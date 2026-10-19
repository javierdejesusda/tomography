"""Table 2: Reading accuracy by model.

One row per checkpoint. For each concept a pair of columns gives the reader's
accuracy at the layer chosen on validation data (LAT) and the random
direction of the same run at that layer (Rand.). Intervals and the selected
layers are in Table A1.
"""

import json

from tomography.site import reading
from tomography.site import store
from tomography.site.manifest import Manifest
from tomography.site.registry import TableOutput
from tomography.viz import paper

MARGIN = 0.1
BLOCKS = (
    ("r1", "R1 sentiment"),
    ("r2", "R2 utility"),
    ("r3", "R3 morality"),
    ("r7", "R7 truth"),
)
EXPLORATORY = ("r5", "R5 honesty (exploratory)")
MISSING = "n/a"


def _slopes(manifest: Manifest) -> dict[str, float]:
    run = manifest.run("p6", "gpt2")
    if run is None:
        raise FileNotFoundError("results/p6/gpt2 is missing")
    data = json.loads((run / "summary.json").read_text("utf-8"))["data"]
    return {k: float(v) for k, v in data["accuracy_per_decade"].items()}


def build(manifest: Manifest) -> TableOutput:
    """Builds the table from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The table and its caption parts.
    """
    blocks = (*BLOCKS, EXPLORATORY)
    found: dict[tuple[str, str], reading.Reading] = {}
    for exp, _ in blocks:
        for model in manifest.models(exp):
            loaded = reading.load(manifest, exp, model)
            if loaded is not None:
                found[(exp, model)] = loaded
    models = sorted({m for _, m in found}, key=store.model_rank)
    rows: list[list[str]] = []
    for model in models:
        row = [paper.display_name(model)]
        for exp, _ in blocks:
            cell = found.get((exp, model))
            if cell is None:
                row += [MISSING, MISSING]
            elif exp == EXPLORATORY[0]:
                row += [f"{cell.accuracy:.3f}", MISSING]
            else:
                row += [f"{cell.accuracy:.3f}", f"{cell.random_at_layer:.3f}"]
        rows.append(row)
    margins = [
        c.accuracy - c.random_at_layer
        for (exp, _), c in found.items()
        if exp != EXPLORATORY[0]
    ]
    above = sum(m > MARGIN for m in margins)
    numbers = {
        "cells": float(len(margins)),
        "above": float(above),
        "smallest_margin": min(margins),
    }
    claim = (
        f"The reader beats a random direction by more than {MARGIN:.1f} in "
        f"{above} of {len(margins)} concept and model cells, and the "
        f"smallest margin is {min(margins):.3f}."
    )
    detail = (
        "LAT is the accuracy of the reader on the test pairs at the layer "
        "chosen on validation data, and Rand. is the accuracy of a random "
        "direction of the same run at the same layer, the baseline each "
        "reader must beat. The 95 percent intervals over the evaluation "
        "pairs and the selected layers are in Table A1 of the appendix. "
        "R5 is the transfer test from plain true and false statements to "
        "honesty; it is exploratory and has no random direction at its "
        "layer. Base and post-trained checkpoints are separate rows."
    )
    slopes = _slopes(manifest)
    notes = [
        "Change in accuracy per tenfold in parameters over the four GPT-2 "
        "sizes (P6): "
        + ", ".join(f"{k.upper()} {v:.3f}" for k, v in slopes.items())
        + "."
    ]
    groups = tuple((title, 2, exp == EXPLORATORY[0]) for exp, title in blocks)
    return TableOutput(
        claim,
        detail,
        ["Model"] + ["LAT", "Rand."] * len(blocks),
        rows,
        "l" + "rr" * len(blocks),
        notes,
        numbers,
        groups,
    )
