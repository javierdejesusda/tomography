"""Resolves which committed run stands for an experiment and a model.

Results live at `results/<experiment>/<model>/<config hash>/`. Most cells
hold one run; the exceptions are named here, each with its rule, and any
other second hash fails the build instead of picking one silently.
"""

import dataclasses
import json
from pathlib import Path
from typing import Any

from tomography.site import store

LATEST = frozenset({"c2", "c4", "c5", "m1", "p7"})
P1_DEFAULT_METHOD = "logistic"
P9D_DEFAULT_TEMPLATE = "fewshot"
MODEL_FREE = "none"


class AmbiguousRunError(ValueError):
    """Raised when a cell holds several runs and no rule picks one."""


@dataclasses.dataclass(frozen=True)
class RunInfo:
    """One run directory with the fields the rules read.

    Attributes:
        path: Run directory.
        config: Parsed `config.json` (or the config stored in `meta.json`).
        started_at: ISO start time from `meta.json`, empty when missing.
    """

    path: Path
    config: dict[str, Any]
    started_at: str


def _load(path: Path) -> RunInfo:
    meta: dict[str, Any] = {}
    meta_file = path / "meta.json"
    if meta_file.is_file():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    config_file = path / "config.json"
    if config_file.is_file():
        config = json.loads(config_file.read_text(encoding="utf-8"))
    else:
        config = meta.get("config", {})
    return RunInfo(path, config, str(meta.get("started_at", "")))


class Manifest:
    """Finds the run that stands for an experiment on a model.

    Attributes:
        root: The `results/` directory.
    """

    def __init__(self, root: Path) -> None:
        """Creates a manifest.

        Args:
            root: The `results/` directory.
        """
        self.root = root

    def experiments(self) -> list[str]:
        """Lists the experiment ids that have a directory.

        Returns:
            Sorted experiment ids.
        """
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def models(self, exp: str) -> list[str]:
        """Lists the models with at least one run, in paper order.

        Args:
            exp: Experiment id.

        Returns:
            Checkpoint keys.
        """
        base = self.root / exp
        if not base.is_dir():
            return []
        found = [p.name for p in base.iterdir() if p.is_dir()]
        return sorted(found, key=store.model_rank)

    def checkpoints(self) -> list[str]:
        """Lists the checkpoints of every experiment, in paper order.

        Model-free runs (`MODEL_FREE`) are not checkpoints and are skipped.

        Returns:
            Checkpoint keys found under any experiment.
        """
        found = {
            m
            for e in self.experiments()
            for m in self.models(e)
            if m != MODEL_FREE
        }
        return sorted(found, key=store.model_rank)

    def runs(self, exp: str, model: str) -> list[RunInfo]:
        """Lists every run directory of a cell, sorted by directory name.

        The directory names are hashes, so the order carries no age.

        Args:
            exp: Experiment id.
            model: Checkpoint key.

        Returns:
            The runs, possibly empty.
        """
        cell = self.root / exp / model
        if not cell.is_dir():
            return []
        return [_load(p) for p in sorted(cell.iterdir()) if p.is_dir()]

    def run(self, exp: str, model: str, **select: str) -> Path | None:
        """Resolves the run directory of a cell.

        Args:
            exp: Experiment id.
            model: Checkpoint key.
            **select: `method` for `p1` and `template` for `p9d`.

        The rule per experiment: `p1` keeps the runs of the selected method
        (logistic by default) and, if several remain, those with the most
        tasks; `p9d` keeps the runs of the selected template; the latest run
        (by start time) is taken for the experiments in `LATEST`; any other
        cell must hold one run.

        Returns:
            The run directory, or None when the cell has no matching run.

        Raises:
            AmbiguousRunError: If several runs remain after the rule.
        """
        runs = self.runs(exp, model)
        if exp == "p1":
            method = select.get("method", P1_DEFAULT_METHOD)
            runs = [r for r in runs if r.config.get("method") == method]
            if len(runs) > 1:
                most = max(len(r.config.get("tasks") or []) for r in runs)
                runs = [
                    r for r in runs if len(r.config.get("tasks") or []) == most
                ]
        elif exp == "p9d":
            template = select.get("template", P9D_DEFAULT_TEMPLATE)
            runs = [r for r in runs if r.config.get("template") == template]
        elif exp in LATEST and len(runs) > 1:
            runs = [max(runs, key=lambda r: r.started_at)]
        if not runs:
            return None
        if len(runs) > 1:
            names = ", ".join(r.path.name for r in runs)
            raise AmbiguousRunError(f"{exp}/{model} holds {names}")
        return runs[0].path
