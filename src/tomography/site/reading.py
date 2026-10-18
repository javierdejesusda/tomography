"""Reading accuracy of one concept on one checkpoint, from stored runs.

R1, R2, R3 and R7 store a summary with the accuracy at the layer chosen on
validation data, and a curve of the reader, a random direction and a
shuffled-label reader over every layer. This module joins them.
"""

import dataclasses
import json
from typing import Any

import numpy as np

from tomography.site.manifest import Manifest

ACCURACY_KEY = {"r1": "best_accuracy", "r2": "test_accuracy"}


@dataclasses.dataclass(frozen=True)
class Reading:
    """One concept on one checkpoint.

    Attributes:
        layer: Selected layer, counted from 1.
        accuracy: Reader accuracy at the selected layer.
        lo: Lower end of the 95 percent interval.
        hi: Upper end of the 95 percent interval.
        n_layers: Number of layers of the checkpoint.
        depth: Relative depth of each layer.
        reader: Reader accuracy at each layer.
        random: Accuracy of a random direction at each layer.
        shuffled: Accuracy of a shuffled-label reader at each layer.
    """

    layer: int
    accuracy: float
    lo: float
    hi: float
    n_layers: int
    depth: np.ndarray
    reader: np.ndarray
    random: np.ndarray
    shuffled: np.ndarray

    @property
    def selectivity(self) -> np.ndarray:
        """The reader minus the random direction of the same run."""
        return self.reader - self.random

    @property
    def random_at_layer(self) -> float:
        """Random-direction accuracy at the selected layer."""
        return float(self.random[self.layer - 1])


def _selected(exp: str, data: dict[str, Any]) -> tuple[int, float, list[float]]:
    if exp == "r7":
        test = data["sets"]["test"]
        return (
            int(data["val_layer"]),
            float(test["accuracy_at_val_layer"]),
            list(test["ci_at_val_layer"]),
        )
    key = ACCURACY_KEY.get(exp, "test_accuracy")
    return int(data["best_layer"]), float(data[key]), list(data["ci"])


def load(manifest: Manifest, exp: str, model: str) -> Reading | None:
    """Reads one cell.

    Args:
        manifest: The run resolver.
        exp: `r1`, `r2`, `r3` or `r7`.
        model: Checkpoint key.

    Returns:
        The reading, or None when the cell has no run.
    """
    run = manifest.run(exp, model)
    if run is None:
        return None
    data = json.loads((run / "summary.json").read_text("utf-8"))["data"]
    meta = json.loads((run / "curves.json").read_text("utf-8"))["data"]
    layer, accuracy, ci = _selected(exp, data)
    count = int(meta["n_layers"])
    with np.load(run / "curves.npz") as npz:
        mean = np.asarray(npz["mean"], dtype=float)
    series = list(meta["series"])
    return Reading(
        layer=layer,
        accuracy=accuracy,
        lo=float(ci[0]),
        hi=float(ci[1]),
        n_layers=count,
        depth=np.arange(1, count + 1) / count,
        reader=mean[series.index("reader")],
        random=mean[series.index("random")],
        shuffled=mean[series.index("shuffled")],
    )


def first_depth_above(found: Reading, margin: float) -> float | None:
    """Finds the earliest relative depth whose selectivity exceeds a margin.

    Args:
        found: A reading.
        margin: Selectivity threshold.

    Returns:
        The relative depth, or None if no layer exceeds it.
    """
    above = np.flatnonzero(found.selectivity > margin)
    return float(found.depth[above[0]]) if above.size else None
