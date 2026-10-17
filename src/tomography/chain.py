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
    value = table.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(
            f"{where}.{key} must be an integer of at least {minimum}"
        )
    return value


def _seconds(
    table: Mapping[str, Any], key: str, default: float, where: str
) -> float:
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


def make_gpu_query() -> Callable[[], int | None]:
    """Builds a query for the memory in use on the first GPU, in MiB.

    The `nvidia-smi` lookup happens once, here.

    Returns:
        A function giving the memory used, or None when `nvidia-smi` is
        missing or fails.
    """
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return lambda: None
    command = [smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"]

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


def stop_process(process: Process) -> None:
    """Stops a running child: SIGTERM, then SIGKILL.

    On POSIX the signals go to the child's whole process group, and the
    SIGKILL also reaches group members that outlive the leader. Further
    SIGTERM, SIGHUP and SIGINT are ignored while stopping.

    Args:
        process: The child.
    """
    if process.poll() is not None:
        return
    with _handlers(_signals(signal.SIGINT), signal.SIG_IGN):
        _signal_child(process, signal.SIGTERM, "terminate")
        timed_out = False
        try:
            process.wait(timeout=TERMINATE_SECONDS)
        except subprocess.TimeoutExpired:
            timed_out = True
        if timed_out or POSIX:
            _signal_child(process, getattr(signal, "SIGKILL", 9), "kill")
        if timed_out:
            process.wait()


@dataclasses.dataclass
class _Runner:
    spec: ChainSpec
    heartbeat_seconds: float
    poll_seconds: float
    clock: Callable[[], float]
    sleep: Callable[[float], None]
    launch: Launcher
    gpu_query: Callable[[], int | None]

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

    def attempt(self, name: str, command: Sequence[str]) -> _Attempt:
        folder = self.spec.out / "_chain" / name
        stdout_path = folder / "stdout.txt"
        log_path = folder / "run.log"
        env = {
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            **os.environ,
            **self.spec.env,
        }
        start = self.clock()
        try:
            folder.mkdir(parents=True, exist_ok=True)
            self.log(f"START {name}")
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"# {self._stamp()} {name}\n")
            offset = log_path.stat().st_size
            process = self.launch(command, env, stdout_path, log_path)
        except (OSError, ValueError) as error:
            with (
                contextlib.suppress(OSError),
                log_path.open("a", encoding="utf-8") as handle,
            ):
                handle.write(f"launch failed: {error}\n")
            self.log(f"END {name} exit=-1 seconds=0 peak_gpu=na last=-")
            return _Attempt(-1, launched=False)
        peak: int | None = None
        last_beat = start
        try:
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
        except KeyboardInterrupt:
            self.log(f"INTERRUPTED {name}")
            raise
        finally:
            stop_process(process)
        result = self._outcome(code, stdout_path, log_path, offset)
        self.log(
            f"END {name} exit={code} seconds={int(self.clock() - start)} "
            f"peak_gpu={_show(peak)} last={result.run_dir or '-'}"
        )
        if code == 0 and result.run_dir is None:
            self.log(f"FAILED {name} reason=no-run-dir")
        return result

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

    def run_job(self, job: Job, run_dirs: dict[str, str]) -> bool:
        digest = job_digest(self.spec, job, run_dirs)
        sizes: list[int | None] = [None, *job.retry_batch_sizes]
        for name, size in zip(attempt_names(job), sizes, strict=True):
            if size is not None:
                self.log(f"RETRY {job.name} batch_size={size}")
            command = build_command(self.spec, job, run_dirs, size)
            result = self.attempt(name, command)
            if not result.launched:
                return False
            if result.run_dir is None:
                continue
            if result.reused:
                self.log(f"SKIP {job.name} reason=existing")
            folder = self.spec.out / "_chain" / name
            _write_done(folder, result.run_dir, digest)
            run_dirs[job.name] = result.run_dir
            return True
        return False

    def run_all(self) -> int:
        run_dirs: dict[str, str] = {}
        failed = 0
        for job in self.spec.jobs:
            done = self.find_done(job, run_dirs)
            if done is not None:
                run_dirs[job.name] = done
                self.log(f"SKIP {job.name} reason=done")
                continue
            missing = {
                key: dep
                for key, dep in job.run_dir_from.items()
                if dep not in run_dirs
            }
            if missing and job.on_missing == "skip":
                first = next(iter(missing.values()))
                self.log(f"SKIP {job.name} reason=missing:{first}")
                failed += 1
                continue
            for key, dep in missing.items():
                self.log(f"DROP {job.name} key={key} reason=missing:{dep}")
            if not self.run_job(job, run_dirs):
                failed += 1
        self.log(f"CHAIN_DONE failed={failed}")
        return 1 if failed else 0


def run_chain(
    spec: ChainSpec,
    *,
    heartbeat_seconds: float = 300.0,
    poll_seconds: float = 5.0,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    launch: Launcher = launch_process,
    gpu_query: Callable[[], int | None] | None = None,
) -> int:
    """Runs the jobs of a chain one after another.

    Each attempt writes `<out>/_chain/<attempt>/stdout.txt` and `run.log`
    (stderr), and the chain appends its events to `<out>/_chain/<name>.log`.
    A job whose DONE.txt names a finished run directory for the same
    arguments is skipped, so a restart resumes the chain. SIGTERM and Ctrl-C
    stop the running child, log the interruption and return 130.

    Args:
        spec: The chain.
        heartbeat_seconds: Seconds between HEARTBEAT lines of a running job.
        poll_seconds: Seconds between checks of a running job and the GPU.
        clock: Returns the time in seconds since the epoch.
        sleep: Waits for the given seconds.
        launch: Starts a command with its stdout and stderr files.
        gpu_query: Returns the GPU memory used in MiB, or None; by default
            a query of `nvidia-smi`, looked up once.

    Returns:
        0 when no job failed, 1 when one did, 130 when interrupted.
    """
    runner = _Runner(
        spec,
        heartbeat_seconds,
        poll_seconds,
        clock,
        sleep,
        launch,
        gpu_query or make_gpu_query(),
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
