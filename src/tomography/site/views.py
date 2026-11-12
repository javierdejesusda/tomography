"""Reads the views of the knowledge readers next to the run they follow.

A V1 run scores the questions of a P9b run again: it refits the committed
draws of each method, scores the published reader and adds a shuffled-label
control. This module joins its numbers at the layer of the reader with the
committed row of the same contrast, so a table or a figure shows the
recomputed value beside the committed one.
"""

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from tomography.site import runfiles
from tomography.site.manifest import Manifest

EXPERIMENT = "v1"
READING = "p9b"
READER = "logistic"
RANDOM = "random"


@dataclasses.dataclass(frozen=True)
class Spread:
    """A value over the draws of a contrast.

    Attributes:
        mean: Mean over the draws.
        low: The lowest draw.
        high: The highest draw.
    """

    mean: float
    low: float
    high: float


@dataclasses.dataclass(frozen=True)
class Contrast:
    """The AUROC values of one contrast at the layer of the reader.

    Attributes:
        name: The contrast.
        committed: The reader of the committed run, a mean over its draws.
        recomputed: The same draws refitted by the V1 run.
        published: The published reader, fitted on every training pair.
        random: The random direction of each draw, refitted.
        shuffled: The reader of each draw refitted on shuffled labels.
        text_word: The committed classifier on the words.
        text_char: The committed classifier on the characters.
        popularity: The committed popularity control, None where the
            contrast has none.
        gaps: Method to the distance between the recomputed and the
            committed mean over the draws.
    """

    name: str
    committed: float
    recomputed: float
    published: float
    random: Spread
    shuffled: Spread
    text_word: float
    text_char: float
    popularity: float | None
    gaps: dict[str, float]


@dataclasses.dataclass(frozen=True)
class Views:
    """The views of one checkpoint.

    Attributes:
        run: The V1 run directory.
        layer: The layer of the reader.
        tolerance: The largest gap that counts as a reproduction.
        draws: How many draws each contrast was refitted on.
        contrasts: The contrasts in the order of the run.
    """

    run: Path
    layer: int
    tolerance: float
    draws: int
    contrasts: tuple[Contrast, ...]


def _spread(values: np.ndarray) -> Spread:
    return Spread(
        float(values.mean()), float(values.min()), float(values.max())
    )


def load(manifest: Manifest, model: str) -> Views | None:
    """Reads the views of a checkpoint.

    Args:
        manifest: The run resolver.
        model: Checkpoint key.

    Returns:
        The views, or None when the checkpoint has no V1 run.

    Raises:
        ValueError: If the V1 run follows another reading run than the one
            the page reads its committed values from.
    """
    run = manifest.run(EXPERIMENT, model)
    if run is None:
        return None
    reading = manifest.run(READING, model)
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    followed = Path(config["reading_run"]).name
    if reading is None or reading.name != followed:
        raise ValueError(
            f"{EXPERIMENT}/{model} follows {READING} run {followed}, not "
            "the run of the page"
        )
    summary = runfiles.summary(run)
    sidecar = runfiles.data(run, "projections.json")
    stored = runfiles.arrays(run, "projections.npz")
    methods: list[str] = sidecar["methods"]
    k = stored["layers"].tolist().index(summary["layer"])
    table: dict[str, dict[str, Any]] = {
        r["type"]: r for r in runfiles.rows(reading, "table.json")
    }
    check = summary["reproduction"]
    contrasts = []
    for t, name in enumerate(sidecar["types"]):
        row = table[name]
        test = stored["test"][:, t, :, k]
        contrasts.append(
            Contrast(
                name=name,
                committed=float(row[READER]),
                recomputed=float(test[methods.index(READER)].mean()),
                published=float(stored["published"][t, k]),
                random=_spread(test[methods.index(RANDOM)]),
                shuffled=_spread(stored["shuffled"][t, :, k]),
                text_word=float(row["text_word"]),
                text_char=float(row["text_char"]),
                popularity=row["popularity"],
                gaps=dict(check["mean_diff_at_layer"][name]),
            )
        )
    return Views(
        run,
        int(summary["layer"]),
        float(check["tolerance"]),
        int(stored["shuffled"].shape[1]),
        tuple(contrasts),
    )
