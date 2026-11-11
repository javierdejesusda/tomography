"""Command line entry point: `python -m tomography <command>`.

The commands are `run`, `plot`, `export`, `chain` and `judge`.
"""

import argparse
from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
import importlib
import json
from pathlib import Path
import sys
import types
import typing
from typing import Any

from tomography import chain
import tomography.experiments as experiments_package
from tomography.experiments.base import EXPERIMENT_ID
from tomography.experiments.base import ExperimentConfig
from tomography.experiments.base import RunDir
from tomography.export import ExportError
from tomography.export import export_runs
from tomography.judging import run as judging


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


def _import_experiment(experiment_id: str) -> types.ModuleType:
    """Imports the module `experiments/<id>_<slug>.py` and only that module.

    Raises:
        ValueError: If no module or several modules carry the id.
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
    return importlib.import_module(
        f"{experiments_package.__name__}.{matches[0]}"
    )


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
    module = _import_experiment(experiment_id)
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


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        experiment = load_experiment(args.experiment)
        config = _build_config(experiment, args)
        path = RunDir.path_for(config, out_root=args.out)
    except ValueError as error:
        parser.error(str(error))
    if chain.is_finished_run(path):
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
        report = export_runs(
            [Path(root) for root in args.roots], Path(args.out)
        )
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


def _chain(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        spec = chain.load_spec(args.spec, check_experiment=load_experiment)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.dry_run:
        for line in chain.dry_run_lines(spec):
            print(line)
        return 0
    return chain.run_chain(spec)


def _judging_hook(run: RunDir, name: str) -> Callable[[RunDir], Any]:
    """Returns a judging function of the experiment that made a run.

    An experiment whose generations an external judge can rate defines
    `judged_generations(run_dir)`, which returns them as a
    `judging.run.Judged`, and `judge_report(run_dir)`, which writes the
    judged tables and returns their paths.

    Raises:
        ValueError: If the module of the experiment lacks the function.
    """
    experiment_id = run.meta["experiment"]
    hook = getattr(_import_experiment(experiment_id), name, None)
    if not callable(hook):
        raise ValueError(
            f"experiment {experiment_id!r} stores no generations for a judge "
            f"(its module defines no {name})"
        )
    return typing.cast(Callable[[RunDir], Any], hook)


def _judge_export(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> int:
    try:
        run = RunDir.open(args.run)
        paths = judging.export_stage(
            run,
            _judging_hook(run, "judged_generations")(run),
            args.stage,
            Path(args.out),
            seed=args.seed,
            batch_size=args.batch_size,
            second_fraction=args.second_fraction,
        )
        plan = judging.read_export(run, args.stage)
    except ValueError as error:
        parser.error(str(error))
    batches = {} if plan is None else {b.batch_id: b for b in plan.batches}
    for path in paths:
        batch = batches.get(path.stem)
        print(
            path
            if batch is None
            else f"{path}\tjudge {batch.judge}\t{len(batch.keys)} items"
        )
    return 0


def _read_judgments(path: str) -> str:
    """Reads a judgments file, without the byte order mark some editors add.

    Raises:
        ValueError: If the file cannot be read as UTF-8 text.
    """
    try:
        return Path(path).read_text("utf-8-sig")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error


def _judge_import(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> int:
    try:
        run = RunDir.open(args.run)
        stored = judging.import_judgments(
            run,
            args.batch,
            _read_judgments(args.judgments),
            judge_model=args.judge_model,
            export_dir=Path(args.export_dir),
        )
    except ValueError as error:
        parser.error(str(error))
    print(stored)
    return 0


def _judge_report(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> int:
    try:
        run = RunDir.open(args.run)
        paths = _judging_hook(run, "judge_report")(run)
    except ValueError as error:
        parser.error(str(error))
    for path in paths:
        print(path)
    return 0


def _add_judge_commands(commands: Any) -> None:
    """Adds `judge export|import|report` to the command line."""
    judge = commands.add_parser(
        "judge",
        help="judge the stored generations of a run with an external judge",
        description=(
            "Judge the stored generations of a run offline. Export the tune "
            "stage, have every batch file judged and import the judgments; "
            "export and import the held_out stage the same way; then write "
            "the report."
        ),
    )
    judge_commands = judge.add_subparsers(dest="judge_command", required=True)
    export = judge_commands.add_parser(
        "export",
        help="write blinded, shuffled judge batches",
        description=(
            "Write the blinded, shuffled batch files of one stage, with "
            "rubric.txt and criteria.json. A judge is given the folder and "
            "nothing else: a batch line holds an item id, the prompt, the "
            "concept description and the generation, and criteria.json "
            "lists the criteria to score. No file names the stage, the "
            "arm, the coefficient, the checkpoint or the judge slot; they "
            "stay in the run, with the SHA-256 of every batch file and of "
            "the rubric. Each batch file is printed with its judge slot "
            "and its number of items: the files of slot 2 hold a share of "
            "the items again, under other ids, for a second, independent "
            "judge. "
            "The held_out stage is refused until every batch of the tune "
            "stage is imported; it holds each arm at the coefficient that "
            "won there and stores that choice with its rule in "
            "judge_selection.json. An empty generation is never exported "
            "and scores 0 on every criterion."
        ),
    )
    export.add_argument("run", help="run directory")
    export.add_argument(
        "--out", required=True, help="folder to write the batch files into"
    )
    export.add_argument(
        "--stage",
        choices=judging.STAGES,
        default=judging.TUNE,
        help="which half of the prompts to export (default: tune)",
    )
    export.add_argument(
        "--seed",
        type=int,
        help="seed of the shuffle and the ids (default: from the run seed)",
    )
    export.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="largest number of items per batch file (default: 50)",
    )
    export.add_argument(
        "--second-fraction",
        type=float,
        default=0.1,
        help=(
            "share of the items exported again under other ids for a second, "
            "independent judge (default: 0.1)"
        ),
    )
    export.set_defaults(handler=_judge_export)
    import_ = judge_commands.add_parser(
        "import",
        help="check and store the judgments of one batch",
        description=(
            "Store the judgments of one exported batch. The file is JSON "
            "lines with exactly one line per item of the batch: the item_id "
            "and one integer score of 0, 1 or 2 per criterion. A file with "
            "a missing, repeated or unknown item, another field or another "
            "value is refused and the run stays as it was. So is an import "
            "whose batch file or rubric.txt is not unchanged since the "
            "export: their SHA-256 must equal the one the run stored."
        ),
    )
    import_.add_argument("run", help="run directory")
    import_.add_argument("judgments", help="judgments file (JSON lines)")
    import_.add_argument(
        "--batch", required=True, help="id of the exported batch"
    )
    import_.add_argument(
        "--judge-model",
        required=True,
        help="model id of the judge, stored with the judgments",
    )
    import_.add_argument(
        "--export-dir",
        required=True,
        help=(
            "folder the batch was exported to (the --out of the export), "
            "with the batch file and rubric.txt the judge was given"
        ),
    )
    import_.set_defaults(handler=_judge_import)
    report = judge_commands.add_parser(
        "report",
        help="write the judged tables of a run",
        description=(
            "Write the judged tables of a run once every exported batch is "
            "imported: the coefficient each arm won on the tune stage, the "
            "held-out scores at that coefficient with intervals, and the "
            "agreement of the two judges."
        ),
    )
    report.add_argument("run", help="run directory")
    report.set_defaults(handler=_judge_report)


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
    chain_parser = commands.add_parser(
        "chain", help="run a chain of experiment jobs"
    )
    chain_commands = chain_parser.add_subparsers(
        dest="chain_command", required=True
    )
    chain_run = chain_commands.add_parser(
        "run", help="run the jobs of a TOML chain spec in order"
    )
    chain_run.add_argument("spec", help="chain spec file (TOML)")
    chain_run.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the spec and print the commands without running them",
    )
    chain_run.set_defaults(handler=_chain)
    _add_judge_commands(commands)
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
