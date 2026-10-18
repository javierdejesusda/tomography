"""Read access to the committed `results/` tree.

Results live at `results/<experiment>/<model>/<config hash>/`. Every JSON file
is `{"_meta": ..., "data": ...}`; tables are stored as `{columns, rows}`, as a
list of records, or as CSV. The store hides those shapes behind a few calls
and never writes anything.
"""

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

MODEL_ORDER = (
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "qwen3-0.6b-base",
    "qwen3-0.6b",
    "qwen3-1.7b-base",
    "qwen3-1.7b",
    "qwen3-4b-base",
    "qwen3-4b",
)

Record = dict[str, Any]


def is_post_trained(model: str) -> bool:
    """Tells whether a checkpoint key names a post-trained model.

    Args:
        model: Checkpoint key such as `qwen3-1.7b` or `gpt2-xl`.

    Returns:
        True for the Qwen3 post-trained checkpoints.
    """
    return model.startswith("qwen3") and not model.endswith("-base")


def model_rank(model: str) -> tuple[int, str]:
    """Sorts checkpoint keys in paper order, unknown keys last.

    Args:
        model: Checkpoint key.

    Returns:
        A sort key.
    """
    if model in MODEL_ORDER:
        return MODEL_ORDER.index(model), model
    return len(MODEL_ORDER), model


def _number(text: str) -> float | str | None:
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return text


class Store:
    """Reads runs from a results directory.

    Attributes:
        root: The `results/` directory.
    """

    def __init__(self, root: Path) -> None:
        """Creates a store.

        Args:
            root: The `results/` directory.
        """
        self.root = root

    def models(self, experiment: str) -> list[str]:
        """Lists the models that have runs for an experiment.

        Args:
            experiment: Experiment id such as `r1`.

        Returns:
            Model keys in the fixed display order.
        """
        base = self.root / experiment
        if not base.is_dir():
            return []
        found = [p.name for p in base.iterdir() if p.is_dir()]
        return sorted(found, key=model_rank)

    def runs(self, experiment: str, model: str) -> list[Path]:
        """Lists the run directories of one cell, oldest first.

        Args:
            experiment: Experiment id.
            model: Model key.

        Returns:
            Run directories ordered by their recorded start time.
        """
        base = self.root / experiment / model
        if not base.is_dir():
            return []
        return sorted(
            (p for p in base.iterdir() if p.is_dir()),
            key=lambda p: (self._started(p), p.name),
        )

    @staticmethod
    def _started(run: Path) -> str:
        meta = run / "meta.json"
        if not meta.is_file():
            return ""
        return str(
            json.loads(meta.read_text(encoding="utf-8")).get("started_at")
        )

    def latest(self, experiment: str, model: str, name: str) -> Path | None:
        """Finds the newest run of a cell that holds a given file.

        Args:
            experiment: Experiment id.
            model: Model key.
            name: File the run must contain.

        Returns:
            The run directory, or None when no run has the file.
        """
        for run in reversed(self.runs(experiment, model)):
            if (run / name).is_file():
                return run
        return None

    def data(self, run: Path, name: str) -> Any:
        """Reads the `data` payload of a JSON result file.

        Args:
            run: Run directory.
            name: File name such as `summary.json`.

        Returns:
            The decoded payload.
        """
        payload = json.loads((run / name).read_text(encoding="utf-8"))
        return payload["data"]

    def summary(self, run: Path) -> Record:
        """Reads `summary.json` of a run.

        Args:
            run: Run directory.

        Returns:
            The summary payload.
        """
        result: Record = self.data(run, "summary.json")
        return result

    def rows(self, run: Path, name: str) -> list[Record]:
        """Reads a table file as a list of records.

        Args:
            run: Run directory.
            name: `table.json`-style or CSV file name.

        Returns:
            One dict per row; CSV numbers are parsed and blanks become None.
        """
        if name.endswith(".csv"):
            with (run / name).open(newline="", encoding="utf-8") as handle:
                return [
                    {k: _number(v) for k, v in row.items()}
                    for row in csv.DictReader(handle)
                ]
        payload = self.data(run, name)
        if isinstance(payload, dict):
            cols = payload["columns"]
            return [
                dict(zip(cols, row, strict=True)) for row in payload["rows"]
            ]
        result: list[Record] = payload
        return result

    def curve_at(
        self, run: Path, name: str, series: str, x: float
    ) -> tuple[float, float, float] | None:
        """Reads one point of a stored curve.

        Args:
            run: Run directory holding `<name>.json` and `<name>.npz`.
            name: Curve file stem such as `curves`.
            series: Series name listed in the JSON file.
            x: Axis value (a layer) to read.

        Returns:
            The (mean, lo, hi) at that x, or None when the curve is missing.
        """
        if not (run / f"{name}.npz").is_file():
            return None
        names = self.data(run, f"{name}.json")["series"]
        if series not in names:
            return None
        with np.load(run / f"{name}.npz") as arrays:
            xs = arrays["x"]
            hit = np.flatnonzero(xs == x)
            if hit.size == 0:
                return None
            row, col = names.index(series), int(hit[0])
            return (
                float(arrays["mean"][row, col]),
                float(arrays["lo"][row, col]),
                float(arrays["hi"][row, col]),
            )
