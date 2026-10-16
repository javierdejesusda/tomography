"""Command line entry point: `python -m tomography run|plot|export`."""

import argparse
from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
import importlib
import json
from pathlib import Path
import sys
import typing
from typing import Any

import tomography.experiments as experiments_package
from tomography.experiments.base import EXPERIMENT_ID
from tomography.experiments.base import ExperimentConfig
from tomography.experiments.base import RunDir
from tomography.export import ExportError
from tomography.export import export_runs


@dataclasses.dataclass(frozen=True)
class Experiment:
    """An experiment module found by its id.

    Attributes:
        id: The experiment id, the file-name prefix of the module.
        config_cls: The module's `Config` class.
        run: The module's `run(config, run_dir)`.
        plot: The module's `plot(run_dir, out_dir)`.
    """

    id: str
    config_cls: type[ExperimentConfig]
    run: Callable[[Any, RunDir], None]
    plot: Callable[[RunDir, Path], list[Path]]


def _module_stems() -> dict[str, set[str]]:
    """Maps each experiment id to the module names that carry it."""
    found: dict[str, set[str]] = {}
    for folder in experiments_package.__path__:
        for file in Path(folder).glob("*_*.py"):
            experiment_id = file.stem.partition("_")[0]
            if EXPERIMENT_ID.fullmatch(experiment_id):
                found.setdefault(experiment_id, set()).add(file.stem)
    return found


def load_experiment(experiment_id: str) -> Experiment:
    """Imports the module `experiments/<id>_<slug>.py` and only that module.

    Args:
        experiment_id: The experiment id, such as "r1".

    Returns:
        The experiment.

    Raises:
        ValueError: If no module or several modules carry the id, or the
            module does not define `Config` (an `ExperimentConfig` subclass),
            `run` and `plot`.
    """
    stems = _module_stems()
    matches = sorted(stems.get(experiment_id, ()))
    if not matches:
        raise ValueError(
            f"unknown experiment {experiment_id!r}; "
            f"available: {', '.join(sorted(stems)) or 'none'}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"experiment id {experiment_id!r} is used by several modules: "
            f"{', '.join(matches)}"
        )
    module = importlib.import_module(
        f"{experiments_package.__name__}.{matches[0]}"
    )
    config_cls = getattr(module, "Config", None)
    run = getattr(module, "run", None)
    plot = getattr(module, "plot", None)
    if not (
        isinstance(config_cls, type)
        and issubclass(config_cls, ExperimentConfig)
        and callable(run)
        and callable(plot)
    ):
        raise ValueError(
            f"{module.__name__} must define Config (an ExperimentConfig "
            "subclass), run and plot"
        )
    return Experiment(experiment_id, config_cls, run, plot)


def _parse_override(
    config_cls: type[ExperimentConfig], item: str
) -> tuple[str, Any]:
    """Splits a `KEY=VALUE` option and decodes the value for the field.

    A value is JSON when it parses as JSON and plain text otherwise, except
    for `str` fields, which always keep the text as typed.

    Args:
        config_cls: The experiment's config class.
        item: The option text.

    Returns:
        The field name and the decoded value, not yet type-checked.

    Raises:
        ValueError: If the option is malformed, names an unknown field or
            tries to set the experiment id.
    """
    key, separator, text = item.partition("=")
    if not separator:
        raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
    if key == "experiment":
        raise ValueError("the experiment id comes from the command, not --set")
    names = [field.name for field in dataclasses.fields(config_cls)]
    if key not in names:
        raise ValueError(
            f"unknown field {key!r} for {config_cls.__name__}; "
            f"fields: {', '.join(names)}"
        )
    if typing.get_type_hints(config_cls)[key] is str:
        return key, text
    try:
        return key, json.loads(text)
    except json.JSONDecodeError:
        return key, text


def _build_config(
    experiment: Experiment, args: argparse.Namespace
) -> ExperimentConfig:
    data: dict[str, Any] = {}
    if args.model is not None:
        data["model"] = args.model
    if args.seed is not None:
        data["seed"] = args.seed
    for item in args.overrides:
        key, value = _parse_override(experiment.config_cls, item)
        data[key] = value
    data["experiment"] = experiment.id
    return experiment.config_cls.from_dict(data)


def _is_finished(path: Path) -> bool:
    try:
        return RunDir.open(path).meta["finished_at"] is not None
    except ValueError:
        return False


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        experiment = load_experiment(args.experiment)
        config = _build_config(experiment, args)
        path = RunDir.path_for(config, out_root=args.out)
    except ValueError as error:
        parser.error(str(error))
    if _is_finished(path):
        parser.error(
            f"run directory already exists: {path} (delete it to rerun)"
        )
    run_dir = RunDir.create(config, out_root=args.out, exist_ok=True)
    experiment.run(config, run_dir)
    run_dir.finalize()
    print(run_dir.path)
    return 0


def _select_runs(args: argparse.Namespace) -> list[RunDir]:
    """Returns the runs to plot: the one given by --run or all finished runs."""
    if args.run is not None:
        run = RunDir.open(args.run)
        found = run.meta["experiment"]
        if args.experiment not in ("all", found):
            raise ValueError(
                f"{run.path} is a run of {found!r}, not {args.experiment!r}"
            )
        return [run]
    runs: list[RunDir] = []
    for path in sorted(Path(args.out).glob("*/*/*")):
        if args.experiment not in ("all", path.parent.parent.name):
            continue
        try:
            run = RunDir.open(path)
        except ValueError:
            continue
        if run.meta["finished_at"] is None:
            print(f"skipping unfinished run: {path}", file=sys.stderr)
        else:
            runs.append(run)
    if not runs:
        raise ValueError(
            f"no finished runs of {args.experiment!r} under {args.out}"
        )
    return runs


def _plot(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        runs = _select_runs(args)
        experiments = {
            experiment_id: load_experiment(experiment_id)
            for experiment_id in sorted({r.meta["experiment"] for r in runs})
        }
    except ValueError as error:
        parser.error(str(error))
    for run in runs:
        out_dir = run.path / "figures"
        out_dir.mkdir(exist_ok=True)
        for figure in experiments[run.meta["experiment"]].plot(run, out_dir):
            print(figure)
    return 0


def _export(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        report = export_runs([Path(root) for root in args.roots], args.out)
    except ExportError as error:
        parser.error(str(error))
    for path in report.skipped:
        print(f"skipping unfinished run: {path}", file=sys.stderr)
    for warning in report.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    for run in report.exported:
        print(f"{run.status} {run.key} {run.size} bytes")
    print(
        f"exported {len(report.exported)} runs, "
        f"{report.total_bytes} bytes, to {args.out}"
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tomography", description="Run experiments and plot results."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run one experiment")
    run.add_argument("experiment", help="experiment id, such as r1")
    run.add_argument("--model", help="checkpoint key, such as gpt2")
    run.add_argument("--seed", type=int, help="base seed")
    run.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="KEY=VALUE",
        help="set a config field; the value is JSON or plain text",
    )
    run.add_argument("--out", default="results", help="results root folder")
    run.set_defaults(handler=_run)
    plot = commands.add_parser(
        "plot", help="regenerate the figures of saved runs"
    )
    plot.add_argument(
        "experiment", metavar="id|all", help="experiment id, or all of them"
    )
    plot.add_argument(
        "--run", help="plot only this run directory, finished or not"
    )
    plot.add_argument("--out", default="results", help="results root folder")
    plot.set_defaults(handler=_plot)
    export = commands.add_parser(
        "export", help="copy finished runs into a committable results tree"
    )
    export.add_argument(
        "--out", required=True, help="results root folder to write"
    )
    export.add_argument(
        "roots", nargs="+", help="folders holding run directories"
    )
    export.set_defaults(handler=_export)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the command line.

    Args:
        argv: Arguments without the program name; `sys.argv[1:]` if None.

    Returns:
        The process exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    code: int = args.handler(args, parser)
    return code
