"""Raw readers of the committed results for recomputation tests.

These read the files directly, never through the site loaders they check. The
run of a cell is the manifest's choice, so the tests and the page agree on
which run is meant.
"""

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from tomography.site.manifest import Manifest

ROOT = Path(__file__).resolve().parents[2] / "results"
MANIFEST = Manifest(ROOT)


def run(exp: str, model: str, **config: str) -> Path:
    """Finds the run directory the page uses for a cell.

    Args:
        exp: Experiment id.
        model: Checkpoint key.
        **config: Config values that select among runs.

    Returns:
        The run directory.

    Raises:
        FileNotFoundError: If the cell has no run.
    """
    found = MANIFEST.run(exp, model, **config)
    if found is None:
        raise FileNotFoundError(f"results/{exp}/{model} has no run")
    return found


def models(exp: str) -> list[str]:
    """Lists the checkpoints with a run of an experiment, in paper order.

    Args:
        exp: Experiment id.

    Returns:
        Checkpoint keys.
    """
    return MANIFEST.models(exp)


def data(path: Path, name: str) -> Any:
    """Reads the `data` payload of a JSON file of a run.

    Args:
        path: Run directory.
        name: File name.

    Returns:
        The decoded payload.
    """
    return json.loads((path / name).read_text("utf-8"))["data"]


def npz(path: Path, name: str) -> dict[str, np.ndarray]:
    """Reads every array of an npz file of a run.

    Args:
        path: Run directory.
        name: File name.

    Returns:
        Array name to array.
    """
    with np.load(path / name) as stored:
        return {key: stored[key] for key in stored.files}


def csv_rows(path: Path, name: str) -> list[dict[str, str]]:
    """Reads a CSV file of a run as raw strings.

    Args:
        path: Run directory.
        name: File name.

    Returns:
        One dict of strings per row.
    """
    with (path / name).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))
