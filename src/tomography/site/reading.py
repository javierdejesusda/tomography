"""Reading accuracy of one concept on one checkpoint, from stored runs.

R1, R2, R3 and R7 store a summary with the accuracy at the layer chosen on
validation data, and a curve of the reader, a random direction and a
shuffled-label reader over every layer. This module joins them. R5 scores a
reader fitted on honesty prompts on plain statements, so its curves are the
transfer curves; the held-out sentiment value comes from the sample-size run
of P3; `load_sentiment` reads the curves of the P4 logistic reader.
"""

import dataclasses
import json
from typing import Any

import numpy as np

from tomography.site import finders
from tomography.site import params
from tomography.site import stats
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
        random_lo: Lower end of the stored 95 percent interval over seeds of
            the random direction at each layer, or None when not stored.
        random_hi: Upper end of that interval, or None.
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
    random_lo: np.ndarray | None = None
    random_hi: np.ndarray | None = None

    @property
    def selectivity(self) -> np.ndarray:
        """The reader minus the random direction of the same run."""
        return np.asarray(self.reader - self.random)

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


@dataclasses.dataclass(frozen=True)
class Score:
    """A reader and its random direction at one layer.

    Attributes:
        layer: The layer, counted from 1.
        reader: Reader accuracy.
        random: Accuracy of a random direction at the same layer.
        n: Number of test pairs both accuracies are read from.
    """

    layer: int
    reader: float
    random: float
    n: int


def margin(score: Score) -> tuple[float, float, float]:
    """Reader minus random with the conservative interval.

    Args:
        score: A reader and its random direction.

    Returns:
        The margin with its lower and upper limits.
    """
    return stats.diff_interval(score.reader, score.random, score.n)


def load(manifest: Manifest, exp: str, model: str) -> Reading | None:
    """Reads one cell.

    Args:
        manifest: The run resolver.
        exp: `r1`, `r2`, `r3`, `r5` or `r7`.
        model: Checkpoint key.

    Returns:
        The reading, or None when the cell has no run.
    """
    run = manifest.run(exp, model)
    if run is None:
        return None
    data = json.loads((run / "summary.json").read_text("utf-8"))["data"]
    stem = "curves_transfer" if exp == "r5" else "curves"
    meta = json.loads((run / f"{stem}.json").read_text("utf-8"))["data"]
    layer, accuracy, ci = _selected(exp, data)
    count = int(meta["n_layers"])
    with np.load(run / f"{stem}.npz") as npz:
        mean = np.asarray(npz["mean"], dtype=float)
        low = np.asarray(npz["lo"], dtype=float)
        high = np.asarray(npz["hi"], dtype=float)
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
        random_lo=low[series.index("random")],
        random_hi=high[series.index("random")],
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


def in_template(manifest: Manifest, model: str) -> Score | None:
    """Reads the in-template block of R5.

    The reader is scored on held-out honesty prompts of the template it was
    fitted on; its random direction is read from the in-template curves at
    that layer.

    Args:
        manifest: The run resolver.
        model: Checkpoint key.

    Returns:
        The score, or None when the checkpoint has no R5 run.
    """
    run = manifest.run("r5", model)
    if run is None:
        return None
    data = json.loads((run / "summary.json").read_text("utf-8"))["data"]
    block = data["in_template"]
    meta = json.loads((run / "curves.json").read_text("utf-8"))["data"]
    layer = int(block["best_layer"])
    with np.load(run / "curves.npz") as npz:
        mean = np.asarray(npz["mean"], dtype=float)
    random = float(mean[list(meta["series"]).index("random")][layer - 1])
    return Score(
        layer,
        float(block["test_accuracy"]),
        random,
        int(data["n_eval_pairs"]["in_template"]["test"]),
    )


def heldout_sentiment(manifest: Manifest, model: str) -> Score | None:
    """Reads held-out sentiment from the sample-size run of P3.

    The layer is chosen on the select split and the accuracy is the mean over
    resamples on the report split, at 64 training pairs.

    Args:
        manifest: The run resolver.
        model: Checkpoint key.

    Returns:
        The score, or None when the checkpoint has no P3 run.
    """
    run = manifest.run("p3", model)
    if run is None:
        return None
    meta = json.loads((run / "sizes_sentiment.json").read_text("utf-8"))
    layer = int(meta["data"]["layer"])
    with np.load(run / "sizes_sentiment.npz") as npz:
        size = int(np.flatnonzero(npz["sizes"] == 64)[0])
        col = int(np.flatnonzero(npz["layers"] == layer)[0])
        accuracy = np.asarray(npz["accuracy"], dtype=float)
    return Score(
        layer,
        float(accuracy[size, 1, 0, :, col].mean()),
        float(accuracy[size, 1, 1, :, col].mean()),
        params.REPORT_PAIRS["sentiment"],
    )


def load_sentiment(manifest: Manifest, model: str) -> Reading | None:
    """Reads sentiment from the report-split curves of the finder comparison.

    The reader is the logistic regression of the reference cell of P4, so the
    curve is held-out accuracy at every layer and the layer is the one its
    select split chose. No shuffled-label curve is stored, so `shuffled` is
    NaN.

    Args:
        manifest: The run resolver.
        model: Checkpoint key.

    Returns:
        The reading, or None when the checkpoint has no P4 run.
    """
    cells = finders.load(manifest, model)
    if cells is None:
        return None
    cell = cells["sentiment"]
    row = cell.rows["logistic"]
    mean, _, _ = cell.curves["logistic"]
    random, random_lo, random_hi = cell.curves["random"]
    return Reading(
        layer=row.layer,
        accuracy=row.report,
        lo=row.lo,
        hi=row.hi,
        n_layers=cell.n_layers,
        depth=cell.depth,
        reader=mean,
        random=random,
        shuffled=np.full(cell.n_layers, np.nan),
        random_lo=random_lo,
        random_hi=random_hi,
    )
