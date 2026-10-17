"""Runs a chain of experiment jobs described by a TOML spec."""

from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
import contextlib
import dataclasses
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import tomllib
from typing import Any
from typing import Protocol

from tomography.experiments.base import RunDir

JOB_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
ON_MISSING = ("skip", "drop")
TAIL_BYTES = 64 * 1024
TERMINATE_SECONDS = 10.0
STDOUT_LINES = 20
POSIX = os.name == "posix"
EXISTING_RUN = re.compile(
    r"run directory already exists: (.+?) \(delete it to rerun\)"
)
CHAIN_KEYS = {
    "name",
    "out",
    "env",
    "slots",
    "min_free_gpu_mib",
    "start_gap_seconds",
}
DEFAULT_START_GAP_SECONDS = 60.0
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
        slots: The most jobs that run at the same time.
        min_free_gpu_mib: Free GPU memory a job needs before it starts
            while another job runs; 0 turns the check off.
        start_gap_seconds: The least time between two job starts when
            `slots` is above 1.
    """

    name: str
    out: Path
    env: Mapping[str, str]
    jobs: tuple[Job, ...]
    slots: int = 1
    min_free_gpu_mib: int = 0
    start_gap_seconds: float = DEFAULT_START_GAP_SECONDS


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


def _integer(
    table: Mapping[str, Any], key: str, default: int, minimum: int, where: str
) -> int:
    """Reads an optional integer setting of at least `minimum`.

    Args:
        table: The TOML table.
        key: The setting name.
        default: The value when the key is absent.
        minimum: The smallest accepted value.
        where: The table name for error messages.

    Returns:
        The value.

    Raises:
        ValueError: If the value is not an integer or is below `minimum`.
    """
    value = table.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(
            f"{where}.{key} must be an integer of at least {minimum}"
        )
    return value


def _seconds(
    table: Mapping[str, Any], key: str, default: float, where: str
) -> float:
    """Reads an optional finite number of at least 0.

    Args:
        table: The TOML table.
        key: The setting name.
        default: The value when the key is absent.
        where: The table name for error messages.

    Returns:
        The value as a float.

    Raises:
        ValueError: If the value is not a finite number of at least 0.
    """
    value = table.get(key, default)
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{where}.{key} must be a number of at least 0")
    return float(value)


def _safe_name(name: str) -> bool:
    return JOB_NAME.fullmatch(name) is not None and ".." not in name


def _table(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a table")
    return value


def _parse_job(table: Mapping[str, Any], index: int) -> Job:
    where = f"job #{index + 1}"
    _check_keys(table, JOB_KEYS, where)
    name = _string(table, "name", where)
    where = f"job {name!r}"
    if not _safe_name(name):
        raise ValueError(
            f"{where}: job name must be letters, digits, '.', '_' or '-'"
        )
    if name.endswith(".log"):
        raise ValueError(f"{where}: job name must not end in '.log'")
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
    if not _safe_name(name):
        raise ValueError("chain.name must be letters, digits, '.', '_' or '-'")
    out = Path(_string(chain, "out", "chain"))
    env = _table(chain.get("env", {}), "chain.env")
    for key, value in env.items():
        if not key or "=" in key or "\0" in key:
            raise ValueError(f"chain.env key {key!r} is not a variable name")
        if not isinstance(value, str) or "\0" in value:
            raise ValueError(f"chain.env value of {key!r} must be a string")
    slots = _integer(chain, "slots", 1, 1, "chain")
    min_free = _integer(chain, "min_free_gpu_mib", 0, 0, "chain")
    gap = _seconds(
        chain, "start_gap_seconds", DEFAULT_START_GAP_SECONDS, "chain"
    )
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
    return ChainSpec(
        name=name,
        out=out,
        env=dict(env),
        jobs=jobs,
        slots=slots,
        min_free_gpu_mib=min_free,
        start_gap_seconds=gap,
    )


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

    pid: int

    def poll(self) -> int | None:
        """Returns the exit code, or None while the process runs."""

    def terminate(self) -> None:
        """Asks the process to stop."""

    def kill(self) -> None:
        """Stops the process without asking."""

    def wait(self, timeout: float | None = None) -> int:
        """Waits for the exit code, raising `TimeoutExpired` on timeout."""


Launcher = Callable[[Sequence[str], Mapping[str, str], Path, Path], Process]


@dataclasses.dataclass(frozen=True)
class _Attempt:
    code: int
    run_dir: str | None = None
    reused: bool = False
    launched: bool = True


def launch_process(
    command: Sequence[str],
    env: Mapping[str, str],
    stdout_path: Path,
    log_path: Path,
) -> subprocess.Popen[bytes]:
    """Starts a command with stdout and stderr in separate files.

    Args:
        command: The argument vector.
        env: The environment of the process.
        stdout_path: Receives stdout, truncated first.
        log_path: Receives stderr, appended to.

    Returns:
        The running process, in its own session on POSIX.
    """
    with stdout_path.open("wb") as out, log_path.open("ab") as log:
        return subprocess.Popen(
            list(command),
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=log,
            start_new_session=POSIX,
        )


def _smi_query(smi: str | None, field: str) -> Callable[[], int | None]:
    """Builds a query of one `nvidia-smi` field of the first GPU.

    Args:
        smi: The `nvidia-smi` path, or None when the tool is missing.
        field: The `--query-gpu` field, for example `memory.free`.

    Returns:
        A function giving the integer value, or None when the tool is
        missing, fails or prints something else.
    """
    if smi is None:
        return lambda: None
    command = [smi, f"--query-gpu={field}", "--format=csv,noheader,nounits"]

    def query() -> int | None:
        try:
            result = subprocess.run(
                command,
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

    return query


@dataclasses.dataclass(frozen=True)
class GpuQueries:
    """Queries of the first GPU, each giving None when it cannot answer.

    Attributes:
        used: The memory in use, in MiB.
        free: The free memory, in MiB.
        util: The utilization of the last sample period, in percent.
    """

    used: Callable[[], int | None]
    free: Callable[[], int | None]
    util: Callable[[], int | None]


def make_gpu_queries() -> GpuQueries:
    """Builds the `nvidia-smi` queries of the first GPU.

    The `nvidia-smi` lookup happens once, here.

    Returns:
        The queries; each gives None when `nvidia-smi` is missing or fails.
    """
    smi = shutil.which("nvidia-smi")
    return GpuQueries(
        used=_smi_query(smi, "memory.used"),
        free=_smi_query(smi, "memory.free"),
        util=_smi_query(smi, "utilization.gpu"),
    )


def read_tail(path: Path, size: int = TAIL_BYTES, floor: int = 0) -> str:
    """Reads at most the last `size` bytes of a file as text.

    Args:
        path: The file.
        size: The most bytes to read.
        floor: Byte offset before which nothing is read.

    Returns:
        The decoded tail; undecodable bytes are replaced.
    """
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(floor, handle.tell() - size))
        return handle.read().decode("utf-8", errors="replace")


def _last_lines(path: Path) -> list[str]:
    """Returns the last non-empty stdout lines, newest first."""
    lines = [line.strip() for line in read_tail(path).splitlines()]
    return [line for line in reversed(lines) if line][:STDOUT_LINES]


def is_finished_run(path: str | Path) -> bool:
    """Tells whether a path is a run directory whose run finished."""
    try:
        return RunDir.open(path).meta.get("finished_at") is not None
    except (OSError, ValueError, AttributeError):
        return False


def job_digest(spec: ChainSpec, job: Job, run_dirs: Mapping[str, str]) -> str:
    """Hashes what a job runs, so a changed job does not resume.

    Args:
        spec: The chain.
        job: The job.
        run_dirs: Run directory by job name of the jobs that finished.

    Returns:
        A short hex digest of the job's first-attempt arguments.
    """
    arguments = build_command(spec, job, run_dirs)[3:]
    text = json.dumps(arguments)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _write_done(folder: Path, run_dir: str, digest: str) -> None:
    temporary = folder / "DONE.txt.tmp"
    text = json.dumps({"run_dir": run_dir, "digest": digest})
    temporary.write_text(text + "\n", encoding="utf-8")
    os.replace(temporary, folder / "DONE.txt")


def _read_done(folder: Path, digest: str) -> str | None:
    """Returns the finished run directory a valid DONE.txt names."""
    try:
        data = json.loads((folder / "DONE.txt").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("digest") != digest:
        return None
    run_dir = data.get("run_dir")
    if isinstance(run_dir, str) and is_finished_run(run_dir):
        return run_dir
    return None


def _show(mebibytes: int | None) -> str:
    return "na" if mebibytes is None else str(mebibytes)


def _signals(*extra: signal.Signals) -> list[signal.Signals]:
    names = ["SIGTERM", "SIGHUP"]
    found = [getattr(signal, name, None) for name in names]
    return [sig for sig in (*found, *extra) if sig is not None]


@contextlib.contextmanager
def _handlers(
    signals: Sequence[signal.Signals], handler: Any
) -> Iterator[None]:
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {sig: signal.signal(sig, handler) for sig in signals}
    try:
        yield
    finally:
        for sig, old in previous.items():
            if old is not None:
                signal.signal(sig, old)


def _raise_interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def interrupt_on_sigterm() -> contextlib.AbstractContextManager[None]:
    """Makes SIGTERM and SIGHUP raise `KeyboardInterrupt` in the block.

    Does nothing outside the main thread.

    Returns:
        A context manager that restores the previous handlers on exit.
    """
    return _handlers(_signals(), _raise_interrupt)


def _signal_child(process: Process, sig: int, fallback: str) -> None:
    killpg = getattr(os, "killpg", None)
    if POSIX and killpg is not None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            killpg(process.pid, sig)
    else:
        getattr(process, fallback)()


def stop_processes(processes: Sequence[Process]) -> None:
    """Stops running children: SIGTERM to all, then SIGKILL.

    Every child gets SIGTERM first, then the children are waited on against
    one shared deadline of `TERMINATE_SECONDS`, so a child that ignores
    SIGTERM does not delay the others. On POSIX the signals go to each
    child's whole process group, and the SIGKILL also reaches group members
    that outlive the leader. Further SIGTERM, SIGHUP and SIGINT are ignored
    while stopping.

    Args:
        processes: The children.
    """
    running = [process for process in processes if process.poll() is None]
    if not running:
        return
    with _handlers(_signals(signal.SIGINT), signal.SIG_IGN):
        for process in running:
            _signal_child(process, signal.SIGTERM, "terminate")
        deadline = time.monotonic() + TERMINATE_SECONDS
        for process in running:
            timed_out = False
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                timed_out = True
            if timed_out or POSIX:
                _signal_child(process, getattr(signal, "SIGKILL", 9), "kill")
            if timed_out:
                process.wait()


@dataclasses.dataclass
class _Active:
    """A job that holds a slot, at one of its attempts."""

    job: Job
    digest: str
    names: list[str]
    sizes: list[int | None]
    index: int = 0
    process: Process | None = None
    offset: int = 0
    start: float = 0.0
    last_beat: float = 0.0
    peak: int | None = None

    @property
    def name(self) -> str:
        return self.names[self.index]


@dataclasses.dataclass
class _Runner:
    spec: ChainSpec
    heartbeat_seconds: float
    poll_seconds: float
    clock: Callable[[], float]
    sleep: Callable[[float], None]
    launch: Launcher
    gpu_query: Callable[[], int | None]
    gpu_free_query: Callable[[], int | None]
    gpu_util_query: Callable[[], int | None]
    run_dirs: dict[str, str] = dataclasses.field(default_factory=dict)
    active: list[_Active] = dataclasses.field(default_factory=list)
    pending: list[Job] = dataclasses.field(default_factory=list)
    ended: set[str] = dataclasses.field(default_factory=set)
    failed: int = 0
    last_start: float | None = None

    def _stamp(self) -> str:
        return datetime.fromtimestamp(self.clock()).isoformat(
            timespec="seconds"
        )

    def log(self, event: str) -> None:
        stamp = self._stamp()
        path = self.spec.out / "_chain" / f"{self.spec.name}.log"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{stamp} {event}\n")

    def find_done(self, job: Job, run_dirs: Mapping[str, str]) -> str | None:
        digest = job_digest(self.spec, job, run_dirs)
        for attempt in attempt_names(job):
            found = _read_done(self.spec.out / "_chain" / attempt, digest)
            if found is not None:
                return found
        return None

    def _begin(self, running: _Active) -> bool:
        """Starts the current attempt of a job.

        Logs RETRY for a retry and START, prepares the attempt folder and
        launches the command. A launch error is logged as an END line.

        Args:
            running: The job and the attempt it is at.

        Returns:
            True when the process runs, False when it could not launch.
        """
        name = running.name
        size = running.sizes[running.index]
        if size is not None:
            self.log(f"RETRY {running.job.name} batch_size={size}")
        command = build_command(self.spec, running.job, self.run_dirs, size)
        folder = self.spec.out / "_chain" / name
        stdout_path = folder / "stdout.txt"
        log_path = folder / "run.log"
        env = {
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            **os.environ,
            **self.spec.env,
        }
        started = self.clock()
        try:
            folder.mkdir(parents=True, exist_ok=True)
            self.log(f"START {name}")
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"# {self._stamp()} {name}\n")
            running.offset = log_path.stat().st_size
            running.process = self.launch(command, env, stdout_path, log_path)
        except (OSError, ValueError) as error:
            with (
                contextlib.suppress(OSError),
                log_path.open("a", encoding="utf-8") as handle,
            ):
                handle.write(f"launch failed: {error}\n")
            self.log(f"END {name} exit=-1 seconds=0 peak_gpu=na last=-")
            return False
        running.start = running.last_beat = started
        running.peak = None
        return True

    def _heartbeat(
        self, running: _Active, used: int | None, util: int | None
    ) -> None:
        """Logs a HEARTBEAT line of a running attempt.

        Args:
            running: The attempt.
            used: The latest GPU memory used in MiB, or None.
            util: The GPU utilization in percent, or None to leave it out.
        """
        now = self.clock()
        line = (
            f"HEARTBEAT {running.name} elapsed={int(now - running.start)} "
            f"gpu_used={_show(used)}"
        )
        if util is not None:
            line += f" gpu_util={util}"
        running.last_beat = now
        self.log(line)

    def _outcome(
        self, code: int, stdout_path: Path, log_path: Path, offset: int
    ) -> _Attempt:
        if code == 0:
            for line in _last_lines(stdout_path):
                if is_finished_run(line):
                    return _Attempt(code, line)
        elif code == 2:
            found = EXISTING_RUN.findall(read_tail(log_path, floor=offset))
            if found and is_finished_run(found[-1]):
                return _Attempt(code, found[-1], reused=True)
        return _Attempt(code)

    def _finish_attempt(self, running: _Active, code: int) -> None:
        """Logs a finished attempt and moves the job on.

        A retry is not launched here: the job keeps its slot without a
        process until `_start_ready` passes the start gate for it.

        Args:
            running: The job whose attempt ended.
            code: The exit code of the attempt.
        """
        folder = self.spec.out / "_chain" / running.name
        result = self._outcome(
            code, folder / "stdout.txt", folder / "run.log", running.offset
        )
        self.log(
            f"END {running.name} exit={code} "
            f"seconds={int(self.clock() - running.start)} "
            f"peak_gpu={_show(running.peak)} last={result.run_dir or '-'}"
        )
        if code == 0 and result.run_dir is None:
            self.log(f"FAILED {running.name} reason=no-run-dir")
        if result.run_dir is not None:
            if result.reused:
                self.log(f"SKIP {running.job.name} reason=existing")
            _write_done(folder, result.run_dir, running.digest)
            self.run_dirs[running.job.name] = result.run_dir
            self._retire(running.job.name, ok=True)
        elif running.index + 1 < len(running.names):
            running.index += 1
            running.process = None
        else:
            self._retire(running.job.name, ok=False)

    def _retire(self, name: str, ok: bool) -> None:
        """Takes a job out of the queue and the slots and marks it ended.

        Args:
            name: The job name.
            ok: Whether the job finished a run directory; a job that did
                not counts as failed.
        """
        self.pending = [job for job in self.pending if job.name != name]
        self.active = [r for r in self.active if r.job.name != name]
        self.ended.add(name)
        if not ok:
            self.failed += 1

    def _launch(self, running: _Active) -> None:
        """Records a start and begins an attempt, retiring a failed launch.

        Args:
            running: The job, already holding its slot.
        """
        self.last_start = self.clock()
        if not self._begin(running):
            self._retire(running.job.name, ok=False)

    def _gate_open(self) -> bool:
        """Tells whether a job may start now.

        Always true while no process runs. Otherwise the last start must be
        `start_gap_seconds` ago when `slots` is above 1, and the GPU must
        report `min_free_gpu_mib` free when that is set; a failed query
        keeps the gate closed.

        Returns:
            True when a start is allowed.
        """
        if not any(r.process is not None for r in self.active):
            return True
        if (
            self.spec.slots > 1
            and self.last_start is not None
            and self.clock() - self.last_start < self.spec.start_gap_seconds
        ):
            return False
        if self.spec.min_free_gpu_mib <= 0:
            return True
        free = self.gpu_free_query()
        return free is not None and free >= self.spec.min_free_gpu_mib

    def _resume_retries(self) -> None:
        """Starts the retries that wait in a slot once the gate is open."""
        for running in list(self.active):
            if running.process is not None or not self._gate_open():
                continue
            self._launch(running)

    def _start_ready(self) -> None:
        """Starts the first ready jobs, in spec order, while slots are free."""
        self._resume_retries()
        for job in list(self.pending):
            if len(self.active) >= self.spec.slots:
                return
            if not all(dep in self.ended for dep in job.run_dir_from.values()):
                continue
            done = self.find_done(job, self.run_dirs)
            if done is not None:
                self.run_dirs[job.name] = done
                self.log(f"SKIP {job.name} reason=done")
                self._retire(job.name, ok=True)
                continue
            missing = {
                key: dep
                for key, dep in job.run_dir_from.items()
                if dep not in self.run_dirs
            }
            if missing and job.on_missing == "skip":
                first = next(iter(missing.values()))
                self.log(f"SKIP {job.name} reason=missing:{first}")
                self._retire(job.name, ok=False)
                continue
            if not self._gate_open():
                return
            for key, dep in missing.items():
                self.log(f"DROP {job.name} key={key} reason=missing:{dep}")
            self.pending.remove(job)
            running = _Active(
                job,
                job_digest(self.spec, job, self.run_dirs),
                attempt_names(job),
                [None, *job.retry_batch_sizes],
            )
            self.active.append(running)
            self._launch(running)

    def _poll_active(self) -> bool:
        """Samples the GPU and checks every running job once.

        Returns:
            True when a job ended, so the caller should not sleep first.
        """
        used = self.gpu_query()
        if used is not None:
            for running in self.active:
                if running.process is not None:
                    peak = running.peak
                    running.peak = used if peak is None else max(peak, used)
        util: int | None = None
        util_read = False
        reaped = False
        for running in list(self.active):
            if running.process is None:
                continue
            code = running.process.poll()
            if code is not None:
                reaped = True
                self._finish_attempt(running, code)
                continue
            due = self.clock() - running.last_beat >= self.heartbeat_seconds
            if due and not util_read:
                util = self.gpu_util_query()
                util_read = True
            if due:
                self._heartbeat(running, used, util)
        return reaped

    def run_all(self) -> int:
        self.pending = list(self.spec.jobs)
        try:
            while self.pending or self.active:
                self._start_ready()
                if self.active and self._poll_active():
                    continue
                self.sleep(self.poll_seconds)
        except KeyboardInterrupt:
            for running in self.active:
                if running.process is not None:
                    self.log(f"INTERRUPTED {running.name}")
            raise
        finally:
            with _handlers(_signals(signal.SIGINT), signal.SIG_IGN):
                stop_processes(
                    [r.process for r in self.active if r.process is not None]
                )
        self.log(f"CHAIN_DONE failed={self.failed}")
        return 1 if self.failed else 0


def _with_defaults(
    used: Callable[[], int | None] | None,
    free: Callable[[], int | None] | None,
    util: Callable[[], int | None] | None,
) -> GpuQueries:
    """Fills each missing GPU query from `nvidia-smi`, looked up once.

    Args:
        used: The memory used query, or None.
        free: The free memory query, or None.
        util: The utilization query, or None.

    Returns:
        The three queries.
    """
    if used and free and util:
        return GpuQueries(used, free, util)
    defaults = make_gpu_queries()
    return GpuQueries(
        used or defaults.used, free or defaults.free, util or defaults.util
    )


def run_chain(
    spec: ChainSpec,
    *,
    heartbeat_seconds: float = 300.0,
    poll_seconds: float = 5.0,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    launch: Launcher = launch_process,
    gpu_query: Callable[[], int | None] | None = None,
    gpu_free_query: Callable[[], int | None] | None = None,
    gpu_util_query: Callable[[], int | None] | None = None,
) -> int:
    """Runs the jobs of a chain, up to `spec.slots` of them at a time.

    Each attempt writes `<out>/_chain/<attempt>/stdout.txt` and `run.log`
    (stderr), and the chain appends its events to `<out>/_chain/<name>.log`.
    A job whose DONE.txt names a finished run directory for the same
    arguments is skipped, so a restart resumes the chain. Jobs start in spec
    order; a job is ready once every job in its `run_dir_from` has ended, and
    a blocked job does not hold back a later ready one. The retries of a job
    run one after another in its slot and pass the same start gate. While
    another job runs, a start also
    waits for `spec.start_gap_seconds` since the last start and, when
    `spec.min_free_gpu_mib` is set, for that much free GPU memory (a failed
    query keeps the gate closed); an idle runner always starts. With
    several jobs the log lines interleave, and the `peak_gpu` of an END line
    is the device-wide peak while that job ran, not the memory of that job
    alone. SIGTERM, SIGHUP and Ctrl-C stop every running child at once,
    log the interruption and return 130.

    Args:
        spec: The chain.
        heartbeat_seconds: Seconds between HEARTBEAT lines of a running job.
        poll_seconds: Seconds between checks of the running jobs and the GPU.
        clock: Returns the time in seconds since the epoch.
        sleep: Waits for the given seconds.
        launch: Starts a command with its stdout and stderr files.
        gpu_query: Returns the GPU memory used in MiB, or None.
        gpu_free_query: Returns the free GPU memory in MiB, or None; read
            only for the start gate.
        gpu_util_query: Returns the GPU utilization in percent, or None; read
            only for a HEARTBEAT line, which adds `gpu_util` when it answers.
            Each query that is omitted defaults on its own to the matching
            `nvidia-smi` query, with one lookup of the tool for all of them.

    Returns:
        0 when no job failed, 1 when one did, 130 when interrupted.
    """
    queries = _with_defaults(gpu_query, gpu_free_query, gpu_util_query)
    runner = _Runner(
        spec,
        heartbeat_seconds,
        poll_seconds,
        clock,
        sleep,
        launch,
        queries.used,
        queries.free,
        queries.util,
    )
    (spec.out / "_chain").mkdir(parents=True, exist_ok=True)
    runner.log(
        f"CHAIN_START name={spec.name} jobs={len(spec.jobs)} "
        f"python={sys.version.split()[0]}"
    )
    try:
        with interrupt_on_sigterm():
            return runner.run_all()
    except KeyboardInterrupt:
        runner.log("CHAIN_INTERRUPTED")
        return 130
