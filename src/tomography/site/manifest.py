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
R8_DEFAULT_TASK = "obqa"
M3_DEFAULT_CELL_SETS = "prompt"
M3_DEFAULT_LOGISTIC_C = "0.01,0.1,1,10"
M3_DEFAULT_CELL_HEAD = "linear"
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


def _n_tasks(run: RunInfo) -> int:
    return len(run.config.get("tasks") or [])


def _most_tasks(runs: list[RunInfo]) -> list[RunInfo]:
    """Keeps the runs that read the most tasks."""
    most = max(map(_n_tasks, runs), default=0)
    return [r for r in runs if _n_tasks(r) == most]


def _only_task(runs: list[RunInfo], task: str) -> list[RunInfo]:
    """Keeps the runs that read `task` and no other task."""
    return [r for r in runs if r.config.get("tasks") == [task]]


def _grid(values: str) -> list[float]:
    """Parses a comma separated C grid, such as "0.01,0.1,1,10"."""
    return [float(v) for v in values.split(",")]


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
        """Lists the model directories of an experiment, in paper order.

        Every subdirectory counts, whether or not it holds a run.

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

        The rule per experiment: `p1` keeps the runs of the selected method
        (logistic by default) and, if several remain, those with the most
        tasks; `p9d` keeps the runs of the selected template; `m2` skips the
        shuffled-only runs (`reader_series` false), which only supplement a
        full run; `r8` keeps the run of the selected task (OpenBookQA by
        default), since it runs one task per run; `m3` keeps the runs of the
        selected cell sets (`prompt` by default) and then the run of the
        selected task alone or, with no task selected, those with the most
        tasks; the latest run (by start time) is taken for the experiments in
        `LATEST`; any other cell must hold one run. The `m3` rule first keeps
        the runs of the selected `logistic_c` grid (the default grid unless
        another is selected), so a run of a wider grid never stands in for
        the default one, and of the selected `cell_head` (`linear` unless
        another is selected; a run with no such field is linear). It keeps
        the runs of the selected `poolings` or, with none selected, skips
        the runs whose poolings include `flat`.

        Args:
            exp: Experiment id.
            model: Checkpoint key.
            **select: `method` for `p1`, `template` for `p9d`, `task` for
                `r8`, and `cell_sets` (comma separated, in config order),
                `logistic_c` (comma separated numbers, in config order),
                `cell_head`, `poolings` (comma separated, in config order)
                and `task` for `m3`.

        Returns:
            The run directory, or None when the cell has no matching run.

        Raises:
            AmbiguousRunError: If several runs remain after the rule.
        """
        runs = self.runs(exp, model)
        if exp == "p1":
            method = select.get("method", P1_DEFAULT_METHOD)
            runs = _most_tasks(
                [r for r in runs if r.config.get("method") == method]
            )
        elif exp == "p9d":
            template = select.get("template", P9D_DEFAULT_TEMPLATE)
            runs = [r for r in runs if r.config.get("template") == template]
        elif exp == "m2":
            runs = [r for r in runs if r.config.get("reader_series", True)]
        elif exp == "r8":
            runs = _only_task(runs, select.get("task", R8_DEFAULT_TASK))
        elif exp == "m3":
            sets = select.get("cell_sets", M3_DEFAULT_CELL_SETS).split(",")
            grid = _grid(select.get("logistic_c", M3_DEFAULT_LOGISTIC_C))
            head = select.get("cell_head", M3_DEFAULT_CELL_HEAD)
            runs = [
                r
                for r in runs
                if r.config.get("cell_sets") == sets
                and [float(c) for c in r.config.get("logistic_c", [])] == grid
                and r.config.get("cell_head", "linear") == head
            ]
            if "poolings" in select:
                poolings = select["poolings"].split(",")
                runs = [r for r in runs if r.config.get("poolings") == poolings]
            else:
                runs = [
                    r
                    for r in runs
                    if "flat" not in r.config.get("poolings", [])
                ]
            if "task" in select:
                runs = _only_task(runs, select["task"])
            else:
                runs = _most_tasks(runs)
        elif exp in LATEST and len(runs) > 1:
            runs = [max(runs, key=lambda r: r.started_at)]
        if not runs:
            return None
        if len(runs) > 1:
            names = ", ".join(r.path.name for r in runs)
            raise AmbiguousRunError(f"{exp}/{model} holds {names}")
        return runs[0].path
