"""Reads the views of the knowledge readers next to the run they follow.

A V1 run scores the questions of a P9b run again: it refits the committed
draws of each method, scores the published reader and adds a shuffled-label
control. This module reads its numbers at the layer of the reader and checks
that the run follows the reading run the page reads.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np

from tomography.site import runfiles
from tomography.site.manifest import Manifest

EXPERIMENT = "v1"
READING = "p9b"
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
        published: The published reader, fitted on every training pair.
        random: The random direction of each draw, refitted.
        shuffled: The reader of each draw refitted on shuffled labels.
    """

    name: str
    published: float
    random: Spread
    shuffled: Spread


@dataclasses.dataclass(frozen=True)
class Views:
    """The views of one checkpoint.

    Attributes:
        run: The V1 run directory.
        layer: The layer of the reader.
        draws: How many draws each contrast was refitted on.
        contrasts: The contrasts in the order of the run.
    """

    run: Path
    layer: int
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
    contrasts = []
    for t, name in enumerate(sidecar["types"]):
        test = stored["test"][:, t, :, k]
        contrasts.append(
            Contrast(
                name=name,
                published=float(stored["published"][t, k]),
                random=_spread(test[methods.index(RANDOM)]),
                shuffled=_spread(stored["shuffled"][t, :, k]),
            )
        )
    return Views(
        run,
        int(summary["layer"]),
        int(stored["shuffled"].shape[1]),
        tuple(contrasts),
    )
