"""Direction finders on the P4 reference cells, from stored runs.

P4 compares six finders per task. The reference cell of a task is the pairing
named in the run summary with centered preprocessing; its table row gives the
layer chosen on the select split and the report-split accuracy there, and the
curves file gives the report-split accuracy at every layer.
"""

import dataclasses
import json

import numpy as np

from tomography.site.manifest import Manifest

TASKS = ("sentiment", "truth", "utility")
PREPROCESS = "center"


@dataclasses.dataclass(frozen=True)
class Row:
    """One finder in the reference cell of a task.

    Attributes:
        layer: Layer chosen on the select split, counted from 1.
        report: Report-split accuracy at that layer.
        lo: Lower end of the 95 percent interval over seeds.
        hi: Upper end of the 95 percent interval over seeds.
    """

    layer: int
    report: float
    lo: float
    hi: float


@dataclasses.dataclass(frozen=True)
class Cell:
    """The reference cell of one task on one checkpoint.

    Attributes:
        pairing: Pairing of the reference cell.
        n_layers: Number of layers of the checkpoint.
        rows: Method name to its row.
        curves: Method name to (mean, lo, hi) over layers, in stored order.
    """

    pairing: str
    n_layers: int
    rows: dict[str, Row]
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]

    @property
    def depth(self) -> np.ndarray:
        """Relative depth of each layer."""
        return np.arange(1, self.n_layers + 1) / self.n_layers

    def row_depth(self, method: str) -> float:
        """Gives the relative depth of a finder's selected layer.

        Args:
            method: Finder name.

        Returns:
            Selected layer over layer count.
        """
        return self.rows[method].layer / self.n_layers


def load(manifest: Manifest, model: str) -> dict[str, Cell] | None:
    """Reads the reference cells of every task for one checkpoint.

    Args:
        manifest: The run resolver.
        model: Checkpoint key.

    Returns:
        Task name to cell, or None when the checkpoint has no P4 run.
    """
    run = manifest.run("p4", model)
    if run is None:
        return None
    reference = json.loads((run / "summary.json").read_text("utf-8"))["data"][
        "reference"
    ]
    table = json.loads((run / "table.json").read_text("utf-8"))["data"]
    cells: dict[str, Cell] = {}
    for task in TASKS:
        pairing = reference[task]["pairing"]
        rows = {
            r["method"]: Row(
                int(r["layer"]),
                float(r["report"]),
                float(r["lo"]),
                float(r["hi"]),
            )
            for r in table
            if r["task"] == task
            and r["pairing"] == pairing
            and r["preprocess"] == PREPROCESS
        }
        meta = json.loads(
            (run / f"curves_{task}_methods.json").read_text("utf-8")
        )["data"]
        with np.load(run / f"curves_{task}_methods.npz") as npz:
            mean, lo, hi = (
                np.asarray(npz[k], float) for k in ["mean", "lo", "hi"]
            )
        curves = {
            name: (mean[i], lo[i], hi[i])
            for i, name in enumerate(meta["series"])
        }
        cells[task] = Cell(pairing, int(meta["n_layers"]), rows, curves)
    return cells
