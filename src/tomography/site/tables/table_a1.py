"""Table 2: Reading accuracy by model.

One block per concept; each row is a checkpoint with the reader's accuracy
at the layer chosen on validation data, the random direction of the same run
at that layer, and that layer.
"""

import json

from tomography.site import reading
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
EXPLORATORY = ("r5", "R5 honesty transfer (exploratory)")


def _interval(found: reading.Reading) -> str:
    return f"{found.accuracy:.3f} [{found.lo:.3f}, {found.hi:.3f}]"


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
    rows: list[list[str]] = []
    margins: list[float] = []
    for exp, title in (*BLOCKS, EXPLORATORY):
        first = True
        for model in manifest.models(exp):
            found = reading.load(manifest, exp, model)
            if found is None:
                continue
            if exp == "r5":
                random = "n/a"
            else:
                random = f"{found.random_at_layer:.3f}"
                margins.append(found.accuracy - found.random_at_layer)
            rows.append(
                [
                    title if first else "",
                    paper.display_name(model),
                    _interval(found),
                    random,
                    str(found.layer),
                ]
            )
            first = False
    slopes = _slopes(manifest)
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
        "Accuracy is read at the layer chosen on validation data, on the "
        "test pairs; intervals are 95 percent intervals over the evaluation "
        "pairs. Random is the accuracy of a random direction of the same "
        "run at the same layer, the baseline each reader must beat. R5 is "
        "the transfer test from plain true and false statements to honesty "
        "and is exploratory; it has no random direction at its layer. "
        "Base and post-trained checkpoints are separate rows."
    )
    notes = [
        "Change in accuracy per tenfold in parameters over the four GPT-2 "
        "sizes (P6): "
        + ", ".join(f"{k.upper()} {v:.3f}" for k, v in slopes.items())
        + "."
    ]
    return TableOutput(
        claim,
        detail,
        ["Concept", "Model", "Reader [95% interval]", "Random", "Layer"],
        rows,
        "llrrr",
        notes,
        numbers,
    )
