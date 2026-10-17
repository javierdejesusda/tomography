"""Runs a chain of experiment jobs described by a TOML spec."""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import tomllib
from typing import Any
from typing import Protocol

JOB_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
ON_MISSING = ("skip", "drop")
CHAIN_KEYS = {"name", "out", "env"}
JOB_KEYS = {
    "name",
    "experiment",
    "model",
    "set",
    "retry_batch_sizes",
    "run_dir_from",
    "on_missing",
}


@dataclasses.dataclass(frozen=True)
class Job:
    """One experiment run of a chain.

    Attributes:
        name: The unique job name, also its folder name under `_chain`.
        experiment: The experiment id.
        model: The checkpoint key.
        settings: Config fields passed as `--set key=value`.
        retry_batch_sizes: Batch sizes to try, in order, after a failure.
        run_dir_from: Config key to the name of an earlier job whose run
            directory becomes that key's value.
        on_missing: What to do when such a job has no run directory:
            "skip" the job or "drop" the key.
    """

    name: str
    experiment: str
    model: str
    settings: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    retry_batch_sizes: tuple[int, ...] = ()
    run_dir_from: Mapping[str, str] = dataclasses.field(default_factory=dict)
    on_missing: str = "skip"


@dataclasses.dataclass(frozen=True)
class ChainSpec:
    """A chain: a name, an out root, an environment and ordered jobs.

    Attributes:
        name: The chain name, which names its log file.
        out: The results root every job writes to.
        env: Variables added to every job's environment.
        jobs: The jobs, in run order.
    """

    name: str
    out: Path
    env: Mapping[str, str]
    jobs: tuple[Job, ...]


def _string(table: Mapping[str, Any], key: str, where: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where}.{key} must be a non-empty string")
    return value


def _check_keys(
    table: Mapping[str, Any], allowed: set[str], where: str
) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ValueError(f"{where}: unknown key {unknown[0]!r}")


def _table(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a table")
    return value


def _parse_job(table: Mapping[str, Any], index: int) -> Job:
    where = f"job #{index + 1}"
    _check_keys(table, JOB_KEYS, where)
    name = _string(table, "name", where)
    where = f"job {name!r}"
    if not JOB_NAME.fullmatch(name):
        raise ValueError(
            f"{where}: job name must be letters, digits, '.', '_' or '-'"
        )
    on_missing = table.get("on_missing", "skip")
    if on_missing not in ON_MISSING:
        raise ValueError(
            f"{where}: on_missing must be one of {', '.join(ON_MISSING)}, "
            f"got {on_missing!r}"
        )
    sizes = table.get("retry_batch_sizes", [])
    if not isinstance(sizes, list) or not all(
        isinstance(size, int) and not isinstance(size, bool) and size > 0
        for size in sizes
    ):
        raise ValueError(
            f"{where}: retry_batch_sizes must be a list of positive integers"
        )
    run_dir_from = _table(
        table.get("run_dir_from", {}), f"{where}.run_dir_from"
    )
    if not all(isinstance(dep, str) for dep in run_dir_from.values()):
        raise ValueError(f"{where}: run_dir_from values must be job names")
    settings = _table(table.get("set", {}), f"{where}.set")
    clash = sorted(set(settings) & set(run_dir_from))
    if clash:
        raise ValueError(
            f"{where}: {clash[0]!r} is in both set and run_dir_from"
        )
    return Job(
        name=name,
        experiment=_string(table, "experiment", where),
        model=_string(table, "model", where),
        settings=dict(settings),
        retry_batch_sizes=tuple(sizes),
        run_dir_from=dict(run_dir_from),
        on_missing=on_missing,
    )


def _check_dependencies(jobs: tuple[Job, ...]) -> None:
    seen: set[str] = set()
    attempts: set[str] = set()
    names = {job.name for job in jobs}
    for job in jobs:
        if job.name in seen:
            raise ValueError(f"duplicate job name {job.name!r}")
        for key, dep in job.run_dir_from.items():
            if dep not in names:
                raise ValueError(
                    f"job {job.name!r}: run_dir_from {key!r} names "
                    f"unknown job {dep!r}"
                )
            if dep not in seen:
                raise ValueError(
                    f"job {job.name!r}: run_dir_from {key!r} names "
                    f"later job {dep!r}"
                )
        seen.add(job.name)
    for job in jobs:
        for attempt in attempt_names(job):
            if attempt in attempts:
                raise ValueError(f"attempt name {attempt!r} is used twice")
            attempts.add(attempt)


def attempt_names(job: Job) -> list[str]:
    """Returns the job name followed by the name of each retry."""
    return [job.name] + [f"{job.name}-b{n}" for n in job.retry_batch_sizes]


def load_spec(
    path: str | Path,
    check_experiment: Callable[[str], object] | None = None,
) -> ChainSpec:
    """Reads and validates a chain spec.

    Args:
        path: The TOML file.
        check_experiment: Called with each experiment id; it raises
            `ValueError` for an unknown one.

    Returns:
        The spec, with a relative `out` resolved against the spec's folder.

    Raises:
        ValueError: If the file is not valid TOML or the spec is invalid.
    """
    spec_path = Path(path)
    try:
        with spec_path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"{spec_path} is not valid TOML: {error}") from error
    if "chain" not in data:
        raise ValueError("missing [chain] table")
    chain = _table(data["chain"], "chain")
    _check_keys(chain, CHAIN_KEYS, "chain")
    name = _string(chain, "name", "chain")
    out = Path(_string(chain, "out", "chain"))
    env = _table(chain.get("env", {}), "chain.env")
    if not all(isinstance(value, str) for value in env.values()):
        raise ValueError("chain.env values must be strings")
    _check_keys(data, {"chain", "job"}, "spec")
    tables = data.get("job", [])
    if not tables:
        raise ValueError("the spec needs at least one job")
    jobs = tuple(_parse_job(table, i) for i, table in enumerate(tables))
    _check_dependencies(jobs)
    if check_experiment is not None:
        for job in jobs:
            check_experiment(job.experiment)
    if not out.is_absolute():
        out = spec_path.resolve().parent / out
    return ChainSpec(name=name, out=out, env=dict(env), jobs=jobs)


def _encode(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except TypeError as error:
        raise ValueError(f"cannot pass {value!r} to --set: {error}") from error


def build_command(
    spec: ChainSpec,
    job: Job,
    run_dirs: Mapping[str, str],
    batch_size: int | None = None,
) -> list[str]:
    """Builds the command line that runs one job attempt.

    Args:
        spec: The chain, for its out root.
        job: The job.
        run_dirs: Run directory by job name of the jobs that finished.
        batch_size: A retry batch size, which replaces any `batch_size` in
            the job's settings.

    Returns:
        The argument vector.
    """
    command = [
        sys.executable,
        "-m",
        "tomography",
        "run",
        job.experiment,
        "--model",
        job.model,
        "--out",
        str(spec.out),
    ]
    for key, value in job.settings.items():
        if batch_size is not None and key == "batch_size":
            continue
        command += ["--set", f"{key}={_encode(value)}"]
    for key, dep in job.run_dir_from.items():
        if dep in run_dirs:
            command += ["--set", f"{key}={run_dirs[dep]}"]
    if batch_size is not None:
        command += ["--set", f"batch_size={batch_size}"]
    return command


def dry_run_lines(spec: ChainSpec) -> list[str]:
    """Lists `attempt: command` for every attempt, without running any.

    Args:
        spec: The chain.

    Returns:
        One line per attempt, with dependencies shown as
        `<run dir of JOB>`.
    """
    lines: list[str] = []
    for job in spec.jobs:
        placeholders = {
            dep: f"<run dir of {dep}>" for dep in job.run_dir_from.values()
        }
        sizes: list[int | None] = [None, *job.retry_batch_sizes]
        for attempt, size in zip(attempt_names(job), sizes, strict=True):
            command = build_command(spec, job, placeholders, size)
            lines.append(f"{attempt}: {' '.join(command)}")
    return lines


class Process(Protocol):
    """The part of a running process the runner needs."""

    def poll(self) -> int | None:
        """Returns the exit code, or None while the process runs."""


Launcher = Callable[[Sequence[str], Mapping[str, str], Path], Process]


def launch_process(
    command: Sequence[str], env: Mapping[str, str], log_path: Path
) -> subprocess.Popen[bytes]:
    """Starts a command with stdout and stderr appended to a log file.

    Args:
        command: The argument vector.
        env: The environment of the process.
        log_path: The log file, created if missing.

    Returns:
        The running process.
    """
    with log_path.open("ab") as log:
        return subprocess.Popen(
            list(command),
            env=dict(env),
            stdout=log,
            stderr=subprocess.STDOUT,
        )


def query_gpu_used() -> int | None:
    """Reads the memory in use on the first GPU, in MiB.

    Returns:
        The memory used, or None when `nvidia-smi` is missing or fails.
    """
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            return None
        return int(result.stdout.splitlines()[0].strip())
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _last_line(path: Path) -> str | None:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in reversed(lines):
        if line.strip():
            return line.strip()
    return None


def _done_dir(folder: Path) -> str | None:
    """Returns the run directory saved in a folder's DONE.txt, if it exists."""
    marker = folder / "DONE.txt"
    if not marker.is_file():
        return None
    text = marker.read_text(encoding="utf-8").strip()
    return text if text and Path(text).is_dir() else None


def _show(mebibytes: int | None) -> str:
    return "na" if mebibytes is None else str(mebibytes)


@dataclasses.dataclass
class _Runner:
    spec: ChainSpec
    heartbeat_seconds: float
    poll_seconds: float
    clock: Callable[[], float]
    sleep: Callable[[float], None]
    launch: Launcher
    gpu_query: Callable[[], int | None]

    def log(self, event: str) -> None:
        stamp = datetime.fromtimestamp(self.clock()).isoformat(
            timespec="seconds"
        )
        path = self.spec.out / "_chain" / f"{self.spec.name}.log"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{stamp} {event}\n")

    def find_done(self, job: Job) -> str | None:
        for attempt in attempt_names(job):
            found = _done_dir(self.spec.out / "_chain" / attempt)
            if found is not None:
                return found
        return None

    def attempt(
        self, name: str, command: Sequence[str]
    ) -> tuple[int, str | None]:
        folder = self.spec.out / "_chain" / name
        folder.mkdir(parents=True, exist_ok=True)
        log_path = folder / "run.log"
        env = {
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            **os.environ,
            **self.spec.env,
        }
        self.log(f"START {name}")
        start = self.clock()
        last_beat = start
        peak: int | None = None
        process = self.launch(command, env, log_path)
        while True:
            used = self.gpu_query()
            if used is not None:
                peak = used if peak is None else max(peak, used)
            code = process.poll()
            if code is not None:
                break
            now = self.clock()
            if now - last_beat >= self.heartbeat_seconds:
                last_beat = now
                self.log(
                    f"HEARTBEAT {name} elapsed={int(now - start)} "
                    f"gpu_used={_show(used)}"
                )
            self.sleep(self.poll_seconds)
        run_dir = _last_line(log_path) if code == 0 else None
        if run_dir is not None:
            (folder / "DONE.txt").write_text(run_dir + "\n", encoding="utf-8")
        self.log(
            f"END {name} exit={code} seconds={int(self.clock() - start)} "
            f"peak_gpu={_show(peak)} last={run_dir or '-'}"
        )
        return code, run_dir

    def run_job(self, job: Job, run_dirs: dict[str, str]) -> bool:
        sizes: list[int | None] = [None, *job.retry_batch_sizes]
        for name, size in zip(attempt_names(job), sizes, strict=True):
            if size is not None:
                self.log(f"RETRY {job.name} batch_size={size}")
            command = build_command(self.spec, job, run_dirs, size)
            code, run_dir = self.attempt(name, command)
            if code == 0:
                if run_dir is not None:
                    run_dirs[job.name] = run_dir
                return True
        return False


def run_chain(
    spec: ChainSpec,
    *,
    heartbeat_seconds: float = 300.0,
    poll_seconds: float = 5.0,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    launch: Launcher = launch_process,
    gpu_query: Callable[[], int | None] = query_gpu_used,
) -> int:
    """Runs the jobs of a chain one after another.

    Each attempt logs to `<out>/_chain/<attempt>/run.log` and the chain
    appends its events to `<out>/_chain/<name>.log`. A job whose DONE.txt
    names an existing directory is skipped, so a restart resumes the chain.

    Args:
        spec: The chain.
        heartbeat_seconds: Seconds between HEARTBEAT lines of a running job.
        poll_seconds: Seconds between checks of a running job and the GPU.
        clock: Returns the time in seconds since the epoch.
        sleep: Waits for the given seconds.
        launch: Starts a command, with the log file it writes to.
        gpu_query: Returns the GPU memory used in MiB, or None.

    Returns:
        0 when no job failed, 1 otherwise.
    """
    runner = _Runner(
        spec, heartbeat_seconds, poll_seconds, clock, sleep, launch, gpu_query
    )
    (spec.out / "_chain").mkdir(parents=True, exist_ok=True)
    runner.log(
        f"CHAIN_START name={spec.name} jobs={len(spec.jobs)} "
        f"python={sys.version.split()[0]}"
    )
    run_dirs: dict[str, str] = {}
    failed = 0
    for job in spec.jobs:
        done = runner.find_done(job)
        if done is not None:
            run_dirs[job.name] = done
            runner.log(f"SKIP {job.name} reason=done")
            continue
        missing = {
            key: dep
            for key, dep in job.run_dir_from.items()
            if dep not in run_dirs
        }
        if missing and job.on_missing == "skip":
            first = next(iter(missing.values()))
            runner.log(f"SKIP {job.name} reason=missing:{first}")
            failed += 1
            continue
        for key, dep in missing.items():
            runner.log(f"DROP {job.name} key={key} reason=missing:{dep}")
        if not runner.run_job(job, run_dirs):
            failed += 1
    runner.log(f"CHAIN_DONE failed={failed}")
    return 1 if failed else 0
