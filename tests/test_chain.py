"""Tests for the portable chain runner."""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import textwrap

import pytest

import tomography.chain as chain_module
from tomography.chain import ChainSpec
from tomography.chain import Job
from tomography.chain import build_command
from tomography.chain import dry_run_lines
from tomography.chain import interrupt_on_sigterm
from tomography.chain import job_digest
from tomography.chain import launch_process
from tomography.chain import load_spec
from tomography.chain import make_gpu_query
from tomography.chain import read_tail
from tomography.chain import run_chain

HEADER = """
[chain]
name = "c1"
out = "results"
"""


def _write_spec(tmp_path: Path, body: str, header: str = HEADER) -> Path:
    path = tmp_path / "chain.toml"
    path.write_text(
        textwrap.dedent(header) + textwrap.dedent(body), encoding="utf-8"
    )
    return path


def test_load_spec_reads_the_chain_and_its_jobs(tmp_path: Path) -> None:
    path = _write_spec(
        tmp_path,
        """
        [[job]]
        name = "r4-a"
        experiment = "r4"
        model = "gpt2"
        set = { lorra = "x", n = 3 }
        retry_batch_sizes = [8, 4]

        [[job]]
        name = "c4-a"
        experiment = "c4"
        model = "gpt2"
        run_dir_from = { lorra_run = "r4-a" }
        on_missing = "drop"
        """,
        HEADER + 'env = { HF_HUB_OFFLINE = "1" }\n',
    )

    spec = load_spec(path)

    assert spec.name == "c1"
    assert spec.out == tmp_path / "results"
    assert spec.env == {"HF_HUB_OFFLINE": "1"}
    assert [job.name for job in spec.jobs] == ["r4-a", "c4-a"]
    first, second = spec.jobs
    assert first == Job(
        name="r4-a",
        experiment="r4",
        model="gpt2",
        settings={"lorra": "x", "n": 3},
        retry_batch_sizes=(8, 4),
        run_dir_from={},
        on_missing="skip",
    )
    assert second.run_dir_from == {"lorra_run": "r4-a"}
    assert second.on_missing == "drop"
    assert dataclasses.is_dataclass(spec)


def test_load_spec_keeps_an_absolute_out_root(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere"
    header = f'[chain]\nname = "c1"\nout = {str(target)!r}\n'
    path = _write_spec(
        tmp_path,
        '[[job]]\nname = "a"\nexperiment = "r1"\nmodel = "gpt2"\n',
        header,
    )

    assert load_spec(path).out == target


JOB_A = '[[job]]\nname = "a"\nexperiment = "r1"\nmodel = "gpt2"\n'


@pytest.mark.parametrize(
    ("header", "body", "message"),
    [
        ("", JOB_A, "missing .chain. table"),
        ("[chain]\nout = 'r'\n", JOB_A, "chain.name"),
        ("[chain]\nname = 'c'\n", JOB_A, "chain.out"),
        (HEADER + "env = { A = 1 }\n", JOB_A, "env"),
        (HEADER + "colour = 'x'\n", JOB_A, "unknown key"),
        (HEADER, "", "at least one job"),
        (HEADER, '[[job]]\nexperiment = "r1"\nmodel = "gpt2"\n', "name"),
        (HEADER, '[[job]]\nname = "a"\nmodel = "gpt2"\n', "experiment"),
        (HEADER, '[[job]]\nname = "a"\nexperiment = "r1"\n', "model"),
        (HEADER, JOB_A + JOB_A, "duplicate job name 'a'"),
        (
            HEADER,
            '[[job]]\nname = "a/b"\nexperiment = "r1"\nmodel = "m"\n',
            "job name",
        ),
        (
            HEADER,
            JOB_A + 'on_missing = "explode"\n',
            "on_missing",
        ),
        (
            HEADER,
            JOB_A + "retry_batch_sizes = [4, 0]\n",
            "retry_batch_sizes",
        ),
        (
            HEADER,
            JOB_A + 'retry_batch_sizes = ["4"]\n',
            "retry_batch_sizes",
        ),
        (
            HEADER,
            JOB_A + "retry_batch_sizes = [true]\n",
            "retry_batch_sizes",
        ),
        (
            HEADER,
            JOB_A + 'run_dir_from = { key = "ghost" }\n',
            "unknown job 'ghost'",
        ),
        (
            HEADER,
            JOB_A + 'run_dir_from = { key = "b" }\n'
            '[[job]]\nname = "b"\nexperiment = "r1"\nmodel = "m"\n',
            "later job 'b'",
        ),
        (
            HEADER,
            JOB_A + 'run_dir_from = { key = "a" }\n',
            "later job 'a'",
        ),
        (
            HEADER,
            JOB_A + "colour = 1\n",
            "unknown key",
        ),
        (
            HEADER,
            JOB_A + "set = { k = 1 }\n"
            '[[job]]\nname = "b"\nexperiment = "r1"\nmodel = "m"\n'
            'set = { k = 1 }\nrun_dir_from = { k = "a" }\n',
            "both set and run_dir_from",
        ),
        (
            HEADER,
            JOB_A + "retry_batch_sizes = [4]\n"
            '[[job]]\nname = "a-b4"\nexperiment = "r1"\nmodel = "m"\n',
            "attempt name 'a-b4'",
        ),
    ],
)
def test_load_spec_rejects_a_bad_spec(
    tmp_path: Path, header: str, body: str, message: str
) -> None:
    path = _write_spec(tmp_path, body, header)

    with pytest.raises(ValueError, match=message):
        load_spec(path)


def test_load_spec_rejects_text_that_is_not_toml(tmp_path: Path) -> None:
    path = tmp_path / "chain.toml"
    path.write_text("[chain\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid TOML"):
        load_spec(path)


def test_load_spec_asks_the_registry_about_every_experiment(
    tmp_path: Path,
) -> None:
    path = _write_spec(
        tmp_path,
        JOB_A + '[[job]]\nname = "b"\nexperiment = "zz"\nmodel = "m"\n',
    )
    asked: list[str] = []

    def check(experiment_id: str) -> object:
        asked.append(experiment_id)
        if experiment_id == "zz":
            raise ValueError("unknown experiment 'zz'")
        return object()

    with pytest.raises(ValueError, match="unknown experiment 'zz'"):
        load_spec(path, check_experiment=check)

    assert asked == ["r1", "zz"]


def _spec(*jobs: Job, out: Path = Path("out")) -> ChainSpec:
    return ChainSpec(name="c1", out=out, env={}, jobs=tuple(jobs))


def _job(name: str = "a", **fields: object) -> Job:
    values: dict[str, object] = {
        "name": name,
        "experiment": "r1",
        "model": "gpt2",
    }
    values.update(fields)
    return Job(**values)  # type: ignore[arg-type]


def test_build_command_runs_the_cli_for_the_experiment() -> None:
    command = build_command(_spec(_job(), out=Path("out")), _job(), {})

    assert command == [
        sys.executable,
        "-m",
        "tomography",
        "run",
        "r1",
        "--model",
        "gpt2",
        "--out",
        str(Path("out")),
    ]


def test_build_command_encodes_set_values_for_the_cli() -> None:
    job = _job(
        settings={
            "text": "plain words",
            "count": 3,
            "ratio": 0.5,
            "flag": True,
            "layers": [1, 2],
        }
    )

    command = build_command(_spec(job), job, {})

    assert command[command.index("--out") + 2 :] == [
        "--set",
        "text=plain words",
        "--set",
        "count=3",
        "--set",
        "ratio=0.5",
        "--set",
        "flag=true",
        "--set",
        "layers=[1, 2]",
    ]


def test_build_command_replaces_batch_size_on_a_retry() -> None:
    job = _job(settings={"batch_size": 16, "n": 2})

    command = build_command(_spec(job), job, {}, batch_size=4)

    sets = [c for c in command if c.startswith("batch_size=")]
    assert sets == ["batch_size=4"]
    assert "n=2" in command


def test_build_command_passes_the_run_dir_of_the_earlier_job() -> None:
    job = _job("b", run_dir_from={"lorra_run": "a"})

    command = build_command(_spec(_job(), job), job, {"a": "/runs/x"})

    assert command[-2:] == ["--set", "lorra_run=/runs/x"]


def test_build_command_drops_a_key_whose_job_has_no_run_dir() -> None:
    job = _job("b", run_dir_from={"lorra_run": "a"}, on_missing="drop")

    command = build_command(_spec(_job(), job), job, {})

    assert not any("lorra_run" in item for item in command)


def test_dry_run_lines_show_every_command_and_placeholders() -> None:
    first = _job("a", retry_batch_sizes=(4,))
    second = _job("b", run_dir_from={"lorra_run": "a"})

    lines = dry_run_lines(_spec(first, second))

    assert [line.partition(": ")[0] for line in lines] == ["a", "a-b4", "b"]
    assert "--set batch_size=4" in lines[1]
    assert lines[2].endswith("--set lorra_run=<run dir of a>")


@pytest.fixture(autouse=True)
def _no_real_process_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(pid: int, sig: int) -> None:
        pytest.fail(f"real os.killpg({pid}, {sig}) reached")

    monkeypatch.setattr(chain_module, "POSIX", False)
    monkeypatch.setattr(os, "killpg", forbidden, raising=False)


class FakeClock:
    """A clock that only moves when the runner sleeps."""

    def __init__(self) -> None:
        self.now = 1_700_000_000.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _make_run(path: Path, finished: bool = True) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    meta = {"finished_at": "2026-01-01T00:00:00" if finished else None}
    (path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return path


class FakeProcess:
    """A process that exits once the fake clock reaches its end time."""

    def __init__(self, clock: FakeClock, seconds: float, code: int) -> None:
        self.clock = clock
        self.end = clock.now + seconds
        self.code = code
        self.pid = 4242
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        if self.terminated or self.killed:
            return -15
        return self.code if self.clock.now >= self.end else None

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: float | None = None) -> int:
        return self.code if self.poll() is None else self.poll() or 0


@dataclasses.dataclass
class Outcome:
    code: int = 0
    seconds: float = 10.0
    stdout: str | None = None
    stderr: str | None = None
    run_dir: Path | None = None
    finished: bool = True


class FakeLauncher:
    """Records launches and answers each one with a planned outcome."""

    def __init__(
        self,
        clock: FakeClock,
        runs: Path,
        plan: Callable[[Sequence[str]], Outcome] | None = None,
    ) -> None:
        self.clock = clock
        self.runs = runs
        self.plan = plan or (lambda command: Outcome())
        self.commands: list[list[str]] = []
        self.envs: list[Mapping[str, str]] = []
        self.logs: list[Path] = []
        self.stdouts: list[Path] = []
        self.processes: list[FakeProcess] = []

    def __call__(
        self,
        command: Sequence[str],
        env: Mapping[str, str],
        stdout_path: Path,
        log_path: Path,
    ) -> FakeProcess:
        self.commands.append(list(command))
        self.envs.append(env)
        self.logs.append(log_path)
        self.stdouts.append(stdout_path)
        outcome = self.plan(command)
        run_dir = outcome.run_dir or self.runs / f"run{len(self.commands)}"
        if outcome.code == 0 and outcome.run_dir is None:
            _make_run(run_dir, outcome.finished)
        stdout = outcome.stdout
        if stdout is None:
            stdout = f"{run_dir}\n\n" if outcome.code == 0 else ""
        stderr = outcome.stderr
        if stderr is None:
            stderr = "loading\n" if outcome.code == 0 else "boom\n"
        stdout_path.write_text(stdout, encoding="utf-8")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(stderr)
        process = FakeProcess(self.clock, outcome.seconds, outcome.code)
        self.processes.append(process)
        return process

    def sets(self, index: int) -> list[str]:
        command = self.commands[index]
        return [command[i + 1] for i, c in enumerate(command) if c == "--set"]


def _run(
    spec: ChainSpec,
    launcher: FakeLauncher,
    clock: FakeClock,
    gpu: Callable[[], int | None] | None = None,
    **options: float,
) -> int:
    return run_chain(
        spec,
        clock=clock,
        sleep=clock.sleep,
        launch=launcher,
        gpu_query=gpu or (lambda: None),
        **options,
    )


def _chain(
    tmp_path: Path, *jobs: Job, env: Mapping[str, str] | None = None
) -> ChainSpec:
    return ChainSpec(
        name="c1", out=tmp_path / "out", env=env or {}, jobs=tuple(jobs)
    )


def _events(spec: ChainSpec) -> list[tuple[str, str]]:
    path = spec.out / "_chain" / f"{spec.name}.log"
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stamp, _, text = line.partition(" ")
        events.append((stamp, text))
    return events


def _texts(spec: ChainSpec) -> list[str]:
    return [text for _, text in _events(spec)]


def _mark_done(folder: Path, run_dir: Path, digest: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    text = json.dumps({"run_dir": str(run_dir), "digest": digest})
    (folder / "DONE.txt").write_text(text, encoding="utf-8")


def _fresh(tmp_path: Path) -> tuple[FakeClock, Path]:
    return FakeClock(), tmp_path / "runs"


def test_a_clean_chain_logs_each_event_and_returns_zero(
    tmp_path: Path,
) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _chain(tmp_path, _job("a"), _job("b"))
    launcher = FakeLauncher(clock, runs)

    code = _run(spec, launcher, clock)

    texts = _texts(spec)
    assert code == 0
    assert texts[0] == (
        f"CHAIN_START name=c1 jobs=2 python={sys.version.split()[0]}"
    )
    assert texts[1] == "START a"
    assert texts[2] == (
        f"END a exit=0 seconds=10 peak_gpu=na last={runs / 'run1'}"
    )
    assert texts[3] == "START b"
    assert texts[-1] == "CHAIN_DONE failed=0"
    stamp = _events(spec)[0][0]
    assert datetime.fromisoformat(stamp) == datetime.fromtimestamp(
        1_700_000_000.0
    ).replace(microsecond=0)


def test_the_log_is_appended_across_runs(tmp_path: Path) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _chain(tmp_path, _job("a"))

    _run(spec, FakeLauncher(clock, runs), clock)
    _run(spec, FakeLauncher(clock, runs), clock)

    starts = [t for t in _texts(spec) if t.startswith("CHAIN_START")]
    assert len(starts) == 2


def test_each_job_gets_a_log_and_a_done_marker(tmp_path: Path) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _chain(tmp_path, _job("a"))
    launcher = FakeLauncher(clock, runs)

    _run(spec, launcher, clock)

    folder = spec.out / "_chain" / "a"
    done = json.loads((folder / "DONE.txt").read_text(encoding="utf-8"))
    assert launcher.logs == [folder / "run.log"]
    assert launcher.stdouts == [folder / "stdout.txt"]
    assert "loading" in (folder / "run.log").read_text(encoding="utf-8")
    assert done == {
        "run_dir": str(runs / "run1"),
        "digest": job_digest(spec, spec.jobs[0], {}),
    }


def test_the_job_environment_is_the_process_env_plus_the_chain_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TOMOGRAPHY_TEST_INHERITED", "yes")
    clock, runs = _fresh(tmp_path)
    spec = _chain(tmp_path, _job("a"), env={"HF_HUB_OFFLINE": "1"})
    launcher = FakeLauncher(clock, runs)

    _run(spec, launcher, clock)

    env = launcher.envs[0]
    assert env["TOMOGRAPHY_TEST_INHERITED"] == "yes"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["PYTHONUNBUFFERED"] == "1"


def test_a_failure_retries_each_batch_size_in_order(tmp_path: Path) -> None:
    clock, runs = _fresh(tmp_path)
    job = _job("a", settings={"batch_size": 16}, retry_batch_sizes=(8, 4))
    spec = _chain(tmp_path, job)
    launcher = FakeLauncher(
        clock,
        runs,
        lambda command: Outcome(code=0 if "batch_size=4" in command else 3),
    )

    code = _run(spec, launcher, clock)

    texts = _texts(spec)
    chain = spec.out / "_chain"
    assert code == 0
    assert [launcher.sets(i)[-1] for i in range(3)] == [
        "batch_size=16",
        "batch_size=8",
        "batch_size=4",
    ]
    assert texts.index("RETRY a batch_size=8") < texts.index(
        "RETRY a batch_size=4"
    )
    assert "START a-b8" in texts
    assert any(t.startswith("END a-b4 exit=0") for t in texts)
    assert (chain / "a-b8" / "run.log").exists()
    assert not (chain / "a" / "DONE.txt").exists()
    assert (chain / "a-b4" / "DONE.txt").exists()
    assert texts[-1] == "CHAIN_DONE failed=0"


def test_a_job_whose_attempts_all_fail_counts_once(tmp_path: Path) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _chain(
        tmp_path,
        _job("a", retry_batch_sizes=(4,)),
        _job("b", experiment="r2"),
    )
    launcher = FakeLauncher(
        clock, runs, lambda c: Outcome(code=0 if "r2" in c else 1)
    )

    code = _run(spec, launcher, clock)

    texts = _texts(spec)
    assert code == 1
    assert texts[-1] == "CHAIN_DONE failed=1"
    assert any(t.startswith("END a exit=1") for t in texts)
    assert any(t.startswith("END a-b4 exit=1") for t in texts)


def test_a_finished_job_is_skipped_on_restart(tmp_path: Path) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _chain(tmp_path, _job("a"), _job("b", experiment="r2"))
    _run(spec, FakeLauncher(clock, runs), clock)
    second = FakeLauncher(clock, runs)

    code = _run(spec, second, clock)

    assert code == 0
    assert second.commands == []
    assert _texts(spec).count("SKIP a reason=done") == 1
    assert _texts(spec).count("SKIP b reason=done") == 1


def test_a_done_marker_naming_a_missing_directory_does_not_skip(
    tmp_path: Path,
) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _chain(tmp_path, _job("a"))
    folder = spec.out / "_chain" / "a"
    folder.mkdir(parents=True)
    _mark_done(folder, runs / "gone", job_digest(spec, spec.jobs[0], {}))
    launcher = FakeLauncher(clock, runs)

    _run(spec, launcher, clock)

    assert len(launcher.commands) == 1


def test_a_retry_that_finished_earlier_is_found_on_restart(
    tmp_path: Path,
) -> None:
    clock, runs = _fresh(tmp_path)
    first = _chain(tmp_path, _job("a", retry_batch_sizes=(4,)))
    _run(
        first,
        FakeLauncher(
            clock, runs, lambda c: Outcome(code=0 if "batch_size=4" in c else 1)
        ),
        clock,
    )
    second = FakeLauncher(clock, runs)

    _run(first, second, clock)

    assert second.commands == []
    assert "SKIP a reason=done" in _texts(first)


def test_a_dependent_job_gets_the_run_dir_of_the_earlier_job(
    tmp_path: Path,
) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _chain(
        tmp_path, _job("a"), _job("b", run_dir_from={"lorra_run": "a"})
    )
    launcher = FakeLauncher(clock, runs)

    _run(spec, launcher, clock)

    assert launcher.sets(1) == [f"lorra_run={runs / 'run1'}"]


def test_a_skipped_job_still_feeds_its_dependents(tmp_path: Path) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _chain(
        tmp_path, _job("a"), _job("b", run_dir_from={"lorra_run": "a"})
    )
    _run(spec, FakeLauncher(clock, runs), clock)
    (spec.out / "_chain" / "b" / "DONE.txt").unlink()
    second = FakeLauncher(clock, runs)

    _run(spec, second, clock)

    assert second.sets(0) == [f"lorra_run={runs / 'run1'}"]


def test_on_missing_skip_skips_a_job_whose_dependency_failed(
    tmp_path: Path,
) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _chain(
        tmp_path,
        _job("a"),
        _job("b", experiment="r2", run_dir_from={"lorra_run": "a"}),
    )
    launcher = FakeLauncher(clock, runs, lambda c: Outcome(code=1))

    code = _run(spec, launcher, clock)

    texts = _texts(spec)
    assert code == 1
    assert len(launcher.commands) == 1
    assert "SKIP b reason=missing:a" in texts
    assert "START b" not in texts
    assert texts[-1] == "CHAIN_DONE failed=2"


def test_on_missing_drop_runs_the_job_without_the_key(tmp_path: Path) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _chain(
        tmp_path,
        _job("a"),
        _job(
            "b",
            experiment="r2",
            run_dir_from={"lorra_run": "a"},
            on_missing="drop",
        ),
    )
    launcher = FakeLauncher(
        clock, runs, lambda c: Outcome(code=1 if "r1" in c else 0)
    )

    code = _run(spec, launcher, clock)

    texts = _texts(spec)
    assert code == 1
    assert launcher.sets(1) == []
    assert "DROP b key=lorra_run reason=missing:a" in texts
    assert texts[-1] == "CHAIN_DONE failed=1"


def test_a_running_job_logs_a_heartbeat_every_interval(
    tmp_path: Path,
) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _chain(tmp_path, _job("a"))
    launcher = FakeLauncher(clock, runs, lambda c: Outcome(seconds=700))

    _run(spec, launcher, clock, heartbeat_seconds=300, poll_seconds=5)

    beats = [t for t in _texts(spec) if t.startswith("HEARTBEAT")]
    assert beats == [
        "HEARTBEAT a elapsed=300 gpu_used=na",
        "HEARTBEAT a elapsed=600 gpu_used=na",
    ]


def test_a_short_job_logs_no_heartbeat(tmp_path: Path) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _chain(tmp_path, _job("a"))

    _run(spec, FakeLauncher(clock, runs), clock)

    assert not [t for t in _texts(spec) if t.startswith("HEARTBEAT")]


def test_the_peak_gpu_memory_is_the_largest_sample(tmp_path: Path) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _chain(tmp_path, _job("a"))
    samples = iter([100, 900, 400, 300, 200])
    launcher = FakeLauncher(clock, runs, lambda c: Outcome(seconds=12))

    _run(spec, launcher, clock, gpu=lambda: next(samples), poll_seconds=5)

    end = next(t for t in _texts(spec) if t.startswith("END"))
    assert " peak_gpu=900 " in end


def test_a_heartbeat_reports_the_latest_gpu_sample(tmp_path: Path) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _chain(tmp_path, _job("a"))
    launcher = FakeLauncher(clock, runs, lambda c: Outcome(seconds=400))

    _run(
        spec,
        launcher,
        clock,
        gpu=lambda: 1234,
        heartbeat_seconds=300,
        poll_seconds=5,
    )

    assert "HEARTBEAT a elapsed=300 gpu_used=1234" in _texts(spec)


def test_the_peak_is_na_when_the_gpu_query_has_no_answer(
    tmp_path: Path,
) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _chain(tmp_path, _job("a"))

    _run(spec, FakeLauncher(clock, runs), clock, gpu=lambda: None)

    end = next(t for t in _texts(spec) if t.startswith("END"))
    assert " peak_gpu=na " in end


def test_a_failed_attempt_logs_no_run_dir(tmp_path: Path) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _chain(tmp_path, _job("a"))

    _run(spec, FakeLauncher(clock, runs, lambda c: Outcome(code=2)), clock)

    end = next(t for t in _texts(spec) if t.startswith("END"))
    assert end == "END a exit=2 seconds=10 peak_gpu=na last=-"


def test_make_gpu_query_is_none_without_nvidia_smi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)

    assert make_gpu_query()() is None


def _fake_smi(
    monkeypatch: pytest.MonkeyPatch, code: int, out: str
) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> object:
        calls.append(command)
        return subprocess.CompletedProcess(command, code, out, "")

    monkeypatch.setattr(shutil, "which", lambda name: "smi")
    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_make_gpu_query_reads_the_first_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_smi(monkeypatch, 0, "2048\n512\n")

    assert make_gpu_query()() == 2048
    assert calls[0] == [
        "smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]


@pytest.mark.parametrize(
    ("code", "out"), [(1, "2048\n"), (0, "N/A\n"), (0, "")]
)
def test_make_gpu_query_is_none_when_the_query_fails(
    monkeypatch: pytest.MonkeyPatch, code: int, out: str
) -> None:
    _fake_smi(monkeypatch, code, out)

    assert make_gpu_query()() is None


def test_launch_process_splits_stdout_and_appends_stderr(
    tmp_path: Path,
) -> None:
    out, log = tmp_path / "stdout.txt", tmp_path / "run.log"
    out.write_text("old\n", encoding="utf-8")
    log.write_text("old\n", encoding="utf-8")
    script = "import sys; print('out'); print('err', file=sys.stderr)"

    process = launch_process(
        [sys.executable, "-c", script], os.environ, out, log
    )

    assert process.wait(timeout=60) == 0
    assert out.read_text(encoding="utf-8").split() == ["out"]
    assert log.read_text(encoding="utf-8").split() == ["old", "err"]


def test_run_chain_runs_real_subprocesses_end_to_end(tmp_path: Path) -> None:
    script = (
        "import json, pathlib, sys; "
        "d = pathlib.Path(sys.argv[1]); d.mkdir(exist_ok=True); "
        "(d / 'config.json').write_text('{}'); "
        "(d / 'meta.json').write_text(json.dumps({'finished_at': 'x'})); "
        "print(d)"
    )
    runs = tmp_path / "runs"
    runs.mkdir()

    def launch(
        command: Sequence[str],
        env: Mapping[str, str],
        stdout_path: Path,
        log_path: Path,
    ) -> subprocess.Popen[bytes]:
        target = str(runs / command[4])
        return launch_process(
            [sys.executable, "-c", script, target], env, stdout_path, log_path
        )

    spec = _chain(tmp_path, _job("a", experiment="x1"))

    code = run_chain(spec, launch=launch, poll_seconds=0.01)

    done = json.loads(
        (spec.out / "_chain" / "a" / "DONE.txt").read_text(encoding="utf-8")
    )
    assert code == 0
    assert done["run_dir"] == str(runs / "x1")


def _exists_message(path: Path) -> str:
    return (
        "tomography: error: run directory already exists: "
        f"{path} (delete it to rerun)\n"
    )


def test_an_existing_finished_run_dir_is_reused_without_a_retry(
    tmp_path: Path,
) -> None:
    clock, runs = _fresh(tmp_path)
    existing = _make_run(tmp_path / "old")
    spec = _chain(
        tmp_path,
        _job("a", retry_batch_sizes=(4,)),
        _job("b", experiment="r2", run_dir_from={"lorra_run": "a"}),
    )
    launcher = FakeLauncher(
        clock,
        runs,
        lambda c: (
            Outcome(code=2, stderr=_exists_message(existing))
            if "r1" in c
            else Outcome()
        ),
    )

    code = _run(spec, launcher, clock)

    texts = _texts(spec)
    done = json.loads(
        (spec.out / "_chain" / "a" / "DONE.txt").read_text(encoding="utf-8")
    )
    assert code == 0
    assert len(launcher.commands) == 2
    assert "SKIP a reason=existing" in texts
    assert "RETRY a batch_size=4" not in texts
    assert done["run_dir"] == str(existing)
    assert launcher.sets(1) == [f"lorra_run={existing}"]
    assert texts[-1] == "CHAIN_DONE failed=0"


def test_an_existing_unfinished_run_dir_is_a_failure(tmp_path: Path) -> None:
    clock, runs = _fresh(tmp_path)
    existing = _make_run(tmp_path / "old", finished=False)
    spec = _chain(tmp_path, _job("a", retry_batch_sizes=(4,)))
    launcher = FakeLauncher(
        clock, runs, lambda c: Outcome(code=2, stderr=_exists_message(existing))
    )

    code = _run(spec, launcher, clock)

    assert code == 1
    assert len(launcher.commands) == 2
    assert "SKIP a reason=existing" not in _texts(spec)


@pytest.mark.parametrize("stdout", ["", "\n\n", "not a directory\n"])
def test_exit_zero_without_a_run_dir_is_a_failure(
    tmp_path: Path, stdout: str
) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _chain(tmp_path, _job("a"))
    launcher = FakeLauncher(clock, runs, lambda c: Outcome(stdout=stdout))

    code = _run(spec, launcher, clock)

    assert code == 1
    assert not (spec.out / "_chain" / "a" / "DONE.txt").exists()
    assert _texts(spec)[-1] == "CHAIN_DONE failed=1"


def test_exit_zero_naming_an_unfinished_run_dir_is_a_failure(
    tmp_path: Path,
) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _chain(tmp_path, _job("a"))
    launcher = FakeLauncher(clock, runs, lambda c: Outcome(finished=False))

    assert _run(spec, launcher, clock) == 1


def test_a_stderr_line_after_the_run_dir_does_not_hide_it(
    tmp_path: Path,
) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _chain(tmp_path, _job("a"))
    launcher = FakeLauncher(
        clock, runs, lambda c: Outcome(stderr="warning: late\n")
    )

    assert _run(spec, launcher, clock) == 0


def test_the_run_dir_is_found_in_a_huge_stdout(tmp_path: Path) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _chain(tmp_path, _job("a"))
    run_dir = _make_run(runs / "big")
    stdout = "x" * 300_000 + f"\n{run_dir}\n"
    launcher = FakeLauncher(
        clock, runs, lambda c: Outcome(stdout=stdout, run_dir=run_dir)
    )

    assert _run(spec, launcher, clock) == 0


def test_read_tail_returns_only_the_last_bytes(tmp_path: Path) -> None:
    path = tmp_path / "f.txt"
    path.write_text("abcdefghij" * 5, encoding="utf-8")

    assert read_tail(path, 10) == "abcdefghij"
    assert read_tail(path, 1000) == "abcdefghij" * 5


@pytest.mark.parametrize(
    "content", ['{"run_di', "garbage", "", "[]", '{"run_dir": 3}']
)
def test_a_damaged_done_marker_is_ignored(tmp_path: Path, content: str) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _chain(tmp_path, _job("a"))
    folder = spec.out / "_chain" / "a"
    folder.mkdir(parents=True)
    (folder / "DONE.txt").write_text(content, encoding="utf-8")
    launcher = FakeLauncher(clock, runs)

    _run(spec, launcher, clock)

    assert len(launcher.commands) == 1


def test_a_done_marker_naming_an_unfinished_run_dir_is_ignored(
    tmp_path: Path,
) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _chain(tmp_path, _job("a"))
    unfinished = _make_run(runs / "u", finished=False)
    _mark_done(
        spec.out / "_chain" / "a",
        unfinished,
        job_digest(spec, spec.jobs[0], {}),
    )
    launcher = FakeLauncher(clock, runs)

    _run(spec, launcher, clock)

    assert len(launcher.commands) == 1


def test_a_done_marker_of_a_changed_job_is_ignored(tmp_path: Path) -> None:
    clock, runs = _fresh(tmp_path)
    before = _chain(tmp_path, _job("a", settings={"n": 1}))
    _run(before, FakeLauncher(clock, runs), clock)
    after = _chain(tmp_path, _job("a", settings={"n": 2}))
    launcher = FakeLauncher(clock, runs)

    _run(after, launcher, clock)

    assert len(launcher.commands) == 1
    assert "SKIP a reason=done" not in _texts(after)


def test_a_done_marker_survives_a_retry_batch_size_change(
    tmp_path: Path,
) -> None:
    clock, runs = _fresh(tmp_path)
    before = _chain(tmp_path, _job("a", retry_batch_sizes=(4,)))
    _run(before, FakeLauncher(clock, runs), clock)
    after = _chain(tmp_path, _job("a", retry_batch_sizes=(4, 2)))
    launcher = FakeLauncher(clock, runs)

    _run(after, launcher, clock)

    assert launcher.commands == []


def test_the_done_marker_is_written_through_a_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _chain(tmp_path, _job("a"))
    replaced: list[tuple[str, str]] = []
    real_replace = os.replace

    def spy(src: str | Path, dst: str | Path) -> None:
        replaced.append((Path(src).name, Path(dst).name))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)

    _run(spec, FakeLauncher(clock, runs), clock)

    folder = spec.out / "_chain" / "a"
    assert replaced == [("DONE.txt.tmp", "DONE.txt")]
    assert not (folder / "DONE.txt.tmp").exists()


def test_an_interrupt_stops_the_child_and_logs_it(tmp_path: Path) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _chain(tmp_path, _job("a"), _job("b"))
    launcher = FakeLauncher(clock, runs, lambda c: Outcome(seconds=1000))
    sleeps = 0

    def sleep(seconds: float) -> None:
        nonlocal sleeps
        sleeps += 1
        if sleeps == 3:
            raise KeyboardInterrupt
        clock.sleep(seconds)

    code = run_chain(
        spec,
        clock=clock,
        sleep=sleep,
        launch=launcher,
        gpu_query=lambda: None,
    )

    texts = _texts(spec)
    assert code == 130
    assert launcher.processes[0].terminated
    assert len(launcher.commands) == 1
    assert "INTERRUPTED a" in texts
    assert texts[-1] == "CHAIN_INTERRUPTED"


def test_a_child_that_ignores_terminate_is_killed(tmp_path: Path) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _chain(tmp_path, _job("a"))
    launcher = FakeLauncher(clock, runs, lambda c: Outcome(seconds=1000))
    sleeps = 0

    def sleep(seconds: float) -> None:
        nonlocal sleeps
        sleeps += 1
        if sleeps == 2:
            raise KeyboardInterrupt
        clock.sleep(seconds)

    original = FakeProcess.wait

    def stubborn(self: FakeProcess, timeout: float | None = None) -> int:
        if not self.killed:
            raise subprocess.TimeoutExpired("x", timeout or 0)
        return original(self, timeout)

    FakeProcess.wait = stubborn  # type: ignore[method-assign]
    try:
        run_chain(
            spec,
            clock=clock,
            sleep=sleep,
            launch=launcher,
            gpu_query=lambda: None,
        )
    finally:
        FakeProcess.wait = original  # type: ignore[method-assign]

    assert launcher.processes[0].terminated
    assert launcher.processes[0].killed


def test_sigterm_becomes_a_keyboard_interrupt_and_is_restored() -> None:
    before = signal.getsignal(signal.SIGTERM)

    with pytest.raises(KeyboardInterrupt), interrupt_on_sigterm():
        signal.raise_signal(signal.SIGTERM)

    assert signal.getsignal(signal.SIGTERM) == before


@pytest.mark.parametrize("error", [OSError("no exec"), ValueError("bad env")])
def test_a_launch_error_fails_the_job_and_the_chain_goes_on(
    tmp_path: Path, error: Exception
) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _chain(tmp_path, _job("a"), _job("b", experiment="r2"))
    inner = FakeLauncher(clock, runs)

    def launch(
        command: Sequence[str],
        env: Mapping[str, str],
        stdout_path: Path,
        log_path: Path,
    ) -> FakeProcess:
        if "r1" in command:
            raise error
        return inner(command, env, stdout_path, log_path)

    code = run_chain(
        spec,
        clock=clock,
        sleep=clock.sleep,
        launch=launch,
        gpu_query=lambda: None,
    )

    texts = _texts(spec)
    assert code == 1
    assert "END a exit=-1 seconds=0 peak_gpu=na last=-" in texts
    assert "START b" in texts
    assert texts[-1] == "CHAIN_DONE failed=1"
    assert str(error) in (spec.out / "_chain" / "a" / "run.log").read_text(
        encoding="utf-8"
    )


def test_the_gpu_tool_is_looked_up_once_per_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _chain(tmp_path, _job("a"), _job("b", experiment="r2"))
    lookups: list[str] = []

    def which(name: str) -> None:
        lookups.append(name)

    monkeypatch.setattr(shutil, "which", which)
    launcher = FakeLauncher(clock, runs, lambda c: Outcome(seconds=60))

    run_chain(
        spec, clock=clock, sleep=clock.sleep, launch=launcher, poll_seconds=5
    )

    assert lookups == ["nvidia-smi"]


@pytest.mark.parametrize(
    "line",
    ['env = { "A=B" = "x" }', 'env = { "" = "x" }', 'env = { A = "x\\u0000" }'],
)
def test_load_spec_rejects_bad_env_names_and_values(
    tmp_path: Path, line: str
) -> None:
    path = _write_spec(tmp_path, JOB_A, HEADER + line + "\n")

    with pytest.raises(ValueError, match="env"):
        load_spec(path)


@pytest.mark.parametrize("name", ["a/b", "a\\\\b", "a..b", ".hidden", "x y"])
def test_load_spec_rejects_an_unsafe_chain_name(
    tmp_path: Path, name: str
) -> None:
    header = f'[chain]\nname = "{name}"\nout = "r"\n'
    path = _write_spec(tmp_path, JOB_A, header)

    with pytest.raises(ValueError, match=r"chain\.name"):
        load_spec(path)


def test_load_spec_rejects_a_job_name_that_collides_with_the_log(
    tmp_path: Path,
) -> None:
    body = '[[job]]\nname = "x.log"\nexperiment = "r1"\nmodel = "m"\n'
    path = _write_spec(tmp_path, body)

    with pytest.raises(ValueError, match=r"\.log"):
        load_spec(path)


def test_a_launch_error_does_not_try_the_retry_batch_sizes(
    tmp_path: Path,
) -> None:
    clock, _ = _fresh(tmp_path)
    spec = _chain(tmp_path, _job("a", retry_batch_sizes=(4, 2)))
    attempts: list[Sequence[str]] = []

    def launch(
        command: Sequence[str],
        env: Mapping[str, str],
        stdout_path: Path,
        log_path: Path,
    ) -> FakeProcess:
        attempts.append(command)
        raise OSError("no exec")

    code = run_chain(
        spec,
        clock=clock,
        sleep=clock.sleep,
        launch=launch,
        gpu_query=lambda: None,
    )

    texts = _texts(spec)
    assert code == 1
    assert len(attempts) == 1
    assert not [t for t in texts if t.startswith("RETRY")]


def test_a_new_attempt_keeps_the_previous_stderr_under_a_header(
    tmp_path: Path,
) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _chain(tmp_path, _job("a"))
    _run(spec, FakeLauncher(clock, runs, lambda c: Outcome(code=1)), clock)
    clock.sleep(60)

    _run(spec, FakeLauncher(clock, runs, lambda c: Outcome(code=1)), clock)

    text = (spec.out / "_chain" / "a" / "run.log").read_text(encoding="utf-8")
    headers = [line for line in text.splitlines() if line.startswith("#")]
    assert text.count("boom") == 2
    assert len(headers) == 2
    assert all(" a " in f"{h} " for h in headers)
    assert headers[0] != headers[1]


def test_a_trailing_library_line_does_not_hide_the_run_dir(
    tmp_path: Path,
) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _chain(tmp_path, _job("a"))
    run_dir = _make_run(runs / "r")
    stdout = f"{run_dir}\nlibrary says bye\nanother line\n"
    launcher = FakeLauncher(
        clock, runs, lambda c: Outcome(stdout=stdout, run_dir=run_dir)
    )

    assert _run(spec, launcher, clock) == 0
    done = json.loads(
        (spec.out / "_chain" / "a" / "DONE.txt").read_text(encoding="utf-8")
    )
    assert done["run_dir"] == str(run_dir)


def test_an_attempt_folder_error_fails_the_job_not_the_chain(
    tmp_path: Path,
) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _chain(tmp_path, _job("a"), _job("b", experiment="r2"))
    (spec.out / "_chain").mkdir(parents=True)
    (spec.out / "_chain" / "a").write_text("in the way", encoding="utf-8")
    launcher = FakeLauncher(clock, runs)

    code = _run(spec, launcher, clock)

    texts = _texts(spec)
    assert code == 1
    assert "END a exit=-1 seconds=0 peak_gpu=na last=-" in texts
    assert "START b" in texts
    assert texts[-1] == "CHAIN_DONE failed=1"


def test_sighup_is_handled_like_sigterm_and_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed: list[tuple[int, object]] = []
    monkeypatch.setattr(signal, "SIGHUP", 1, raising=False)
    monkeypatch.setattr(signal, "getsignal", lambda sig: f"old{sig}")

    def fake_signal(sig: int, handler: object) -> object:
        installed.append((sig, handler))
        return f"old{sig}"

    monkeypatch.setattr(signal, "signal", fake_signal)

    with interrupt_on_sigterm():
        pass

    signals = [sig for sig, _ in installed]
    assert signal.SIGTERM in signals
    assert 1 in signals
    restored = {(1, "old1"), (signal.SIGTERM, f"old{signal.SIGTERM}")}
    assert restored <= set(installed)


def test_a_second_sigterm_while_stopping_does_not_skip_the_kill(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    process = FakeProcess(clock, 1000, 0)

    def wait(timeout: float | None = None) -> int:
        signal.raise_signal(signal.SIGTERM)
        if timeout is not None:
            raise subprocess.TimeoutExpired("x", timeout)
        return 0

    process.wait = wait  # type: ignore[method-assign]

    with interrupt_on_sigterm():
        chain_module.stop_process(process)

    assert process.terminated
    assert process.killed


def test_stopping_signals_the_whole_process_group_on_posix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    process = FakeProcess(clock, 1000, 0)
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(chain_module, "POSIX", True)
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(
        os, "killpg", lambda pid, sig: sent.append((pid, sig)), raising=False
    )

    def wait(timeout: float | None = None) -> int:
        if timeout is not None:
            raise subprocess.TimeoutExpired("x", timeout)
        return 0

    process.wait = wait  # type: ignore[method-assign]

    chain_module.stop_process(process)

    assert sent == [(4242, signal.SIGTERM), (4242, 9)]
    assert not process.terminated


def test_a_group_member_that_outlives_the_leader_is_killed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = FakeProcess(FakeClock(), 1000, 0)
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(chain_module, "POSIX", True)
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(
        os, "killpg", lambda pid, sig: sent.append((pid, sig)), raising=False
    )

    chain_module.stop_process(process)

    assert sent == [(4242, signal.SIGTERM), (4242, 9)]


@pytest.mark.parametrize("error", [ProcessLookupError, PermissionError])
def test_a_vanished_process_group_is_not_an_error(
    monkeypatch: pytest.MonkeyPatch, error: type[OSError]
) -> None:
    process = FakeProcess(FakeClock(), 1000, 0)
    monkeypatch.setattr(chain_module, "POSIX", True)
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)

    def gone(pid: int, sig: int) -> None:
        raise error

    monkeypatch.setattr(os, "killpg", gone, raising=False)

    chain_module.stop_process(process)


def test_exit_two_ignores_an_already_exists_line_of_an_earlier_attempt(
    tmp_path: Path,
) -> None:
    clock, runs = _fresh(tmp_path)
    existing = _make_run(tmp_path / "old")
    spec = _chain(tmp_path, _job("a"))
    folder = spec.out / "_chain" / "a"
    folder.mkdir(parents=True)
    (folder / "run.log").write_text(_exists_message(existing), encoding="utf-8")
    launcher = FakeLauncher(
        clock, runs, lambda c: Outcome(code=2, stderr="usage error\n")
    )

    code = _run(spec, launcher, clock)

    assert code == 1
    assert "SKIP a reason=existing" not in _texts(spec)


def test_read_tail_can_start_at_a_floor(tmp_path: Path) -> None:
    path = tmp_path / "f.txt"
    path.write_text("old-new", encoding="utf-8")

    assert read_tail(path, 1000, floor=4) == "new"


def test_load_spec_reads_the_concurrency_keys(tmp_path: Path) -> None:
    header = HEADER + "slots = 3\nmin_free_gpu_mib = 6000\n"
    header += "start_gap_seconds = 12.5\n"
    path = _write_spec(tmp_path, JOB_A, header)

    spec = load_spec(path)

    assert spec.slots == 3
    assert spec.min_free_gpu_mib == 6000
    assert spec.start_gap_seconds == 12.5


def test_the_concurrency_keys_default_to_one_job_at_a_time(
    tmp_path: Path,
) -> None:
    spec = load_spec(_write_spec(tmp_path, JOB_A))

    assert spec.slots == 1
    assert spec.min_free_gpu_mib == 0
    assert spec.start_gap_seconds == 60.0


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("slots = 0", r"chain\.slots"),
        ("slots = -2", r"chain\.slots"),
        ("slots = 1.5", r"chain\.slots"),
        ('slots = "2"', r"chain\.slots"),
        ("slots = true", r"chain\.slots"),
        ("min_free_gpu_mib = -1", r"chain\.min_free_gpu_mib"),
        ("min_free_gpu_mib = 2.5", r"chain\.min_free_gpu_mib"),
        ("min_free_gpu_mib = false", r"chain\.min_free_gpu_mib"),
        ("start_gap_seconds = -1", r"chain\.start_gap_seconds"),
        ('start_gap_seconds = "5"', r"chain\.start_gap_seconds"),
        ("start_gap_seconds = true", r"chain\.start_gap_seconds"),
        ("start_gap_seconds = nan", r"chain\.start_gap_seconds"),
        ("start_gap_seconds = inf", r"chain\.start_gap_seconds"),
    ],
)
def test_load_spec_rejects_bad_concurrency_values(
    tmp_path: Path, line: str, message: str
) -> None:
    path = _write_spec(tmp_path, JOB_A, HEADER + line + "\n")

    with pytest.raises(ValueError, match=message):
        load_spec(path)


def _planned(
    clock: FakeClock,
    seconds: Mapping[str, float],
    codes: Mapping[str, int] | None = None,
) -> tuple[Callable[[Sequence[str]], Outcome], dict[str, float]]:
    """Plans an outcome per experiment id and records each launch time."""
    started: dict[str, float] = {}

    def plan(command: Sequence[str]) -> Outcome:
        experiment = command[4]
        started[experiment] = clock.now - 1_700_000_000.0
        return Outcome(
            code=(codes or {}).get(experiment, 0),
            seconds=seconds[experiment],
        )

    return plan, started


def _run_slots(
    spec: ChainSpec,
    launcher: FakeLauncher,
    clock: FakeClock,
    *,
    used: Callable[[], int | None] | None = None,
    free: Callable[[], int | None] | None = None,
    util: Callable[[], int | None] | None = None,
    sleep: Callable[[float], None] | None = None,
    **options: float,
) -> int:
    return run_chain(
        spec,
        clock=clock,
        sleep=sleep or clock.sleep,
        launch=launcher,
        gpu_query=used or (lambda: None),
        gpu_free_query=free or (lambda: None),
        gpu_util_query=util or (lambda: None),
        **options,
    )


def _slotted(tmp_path: Path, *jobs: Job, **fields: float) -> ChainSpec:
    values: dict[str, float] = {"slots": 2, "start_gap_seconds": 0}
    values.update(fields)
    return dataclasses.replace(_chain(tmp_path, *jobs), **values)


def _forbidden_query() -> int | None:
    pytest.fail("the free memory was read with the gate off")


def test_two_independent_jobs_run_at_the_same_time(tmp_path: Path) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _slotted(tmp_path, _job("a"), _job("b", experiment="r2"))
    plan, started = _planned(clock, {"r1": 100, "r2": 100})
    launcher = FakeLauncher(clock, runs, plan)

    code = _run_slots(spec, launcher, clock, free=_forbidden_query)

    texts = _texts(spec)
    ends = [t for t in texts if t.startswith("END")]
    assert code == 0
    assert started == {"r1": 0, "r2": 0}
    assert texts.index("START b") < texts.index(ends[0])
    assert [e.split()[1] for e in ends] == ["a", "b"]
    assert all(" seconds=100 " in e for e in ends)
    assert texts[-1] == "CHAIN_DONE failed=0"


def test_a_third_job_waits_for_a_free_slot(tmp_path: Path) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _slotted(
        tmp_path,
        _job("a"),
        _job("b", experiment="r2"),
        _job("c", experiment="r3"),
    )
    plan, started = _planned(clock, {"r1": 50, "r2": 100, "r3": 10})
    launcher = FakeLauncher(clock, runs, plan)

    _run_slots(spec, launcher, clock)

    texts = _texts(spec)
    end_a = next(i for i, t in enumerate(texts) if t.startswith("END a "))
    end_b = next(i for i, t in enumerate(texts) if t.startswith("END b "))
    assert started == {"r1": 0, "r2": 0, "r3": 50}
    assert end_a < texts.index("START c") < end_b


def test_a_dependent_job_waits_while_a_later_independent_job_starts(
    tmp_path: Path,
) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _slotted(
        tmp_path,
        _job("a"),
        _job("b", experiment="r2", run_dir_from={"lorra_run": "a"}),
        _job("c", experiment="r3"),
    )
    plan, started = _planned(clock, {"r1": 100, "r2": 10, "r3": 20})
    launcher = FakeLauncher(clock, runs, plan)

    _run_slots(spec, launcher, clock)

    texts = _texts(spec)
    starts = [t for t in texts if t.startswith("START")]
    assert starts == ["START a", "START c", "START b"]
    assert started == {"r1": 0, "r3": 0, "r2": 100}
    assert texts.index("START b") > next(
        i for i, t in enumerate(texts) if t.startswith("END a ")
    )
    assert launcher.sets(2) == [f"lorra_run={runs / 'run1'}"]


def test_a_failed_dependency_skips_the_dependent_while_others_run(
    tmp_path: Path,
) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _slotted(
        tmp_path,
        _job("a"),
        _job("b", experiment="r2", run_dir_from={"lorra_run": "a"}),
        _job("c", experiment="r3"),
    )
    plan, _ = _planned(clock, {"r1": 10, "r2": 10, "r3": 40}, {"r1": 1})
    launcher = FakeLauncher(clock, runs, plan)

    code = _run_slots(spec, launcher, clock)

    texts = _texts(spec)
    assert code == 1
    assert "SKIP b reason=missing:a" in texts
    assert "START b" not in texts
    assert any(t.startswith("END c exit=0") for t in texts)
    assert texts[-1] == "CHAIN_DONE failed=2"


def test_a_failed_job_does_not_stop_the_others(tmp_path: Path) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _slotted(tmp_path, _job("a"), _job("b", experiment="r2"))
    plan, _ = _planned(clock, {"r1": 10, "r2": 30}, {"r1": 3})
    launcher = FakeLauncher(clock, runs, plan)

    code = _run_slots(spec, launcher, clock)

    texts = _texts(spec)
    assert code == 1
    assert any(t.startswith("END a exit=3") for t in texts)
    assert any(t.startswith("END b exit=0") for t in texts)
    assert texts[-1] == "CHAIN_DONE failed=1"


def test_the_retries_of_a_job_run_one_after_another_in_its_slot(
    tmp_path: Path,
) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _slotted(
        tmp_path,
        _job("a", settings={"batch_size": 16}, retry_batch_sizes=(8,)),
        _job("b", experiment="r2"),
        _job("c", experiment="r3"),
    )
    launcher = FakeLauncher(
        clock,
        runs,
        lambda c: (
            Outcome(code=1, seconds=10)
            if "batch_size=16" in c
            else Outcome(seconds=100 if "r1" in c else 30)
        ),
    )

    code = _run_slots(spec, launcher, clock)

    texts = _texts(spec)
    assert code == 0
    assert texts.index("RETRY a batch_size=8") < texts.index("START a-b8")
    assert texts.index("START a-b8") < texts.index("START c")
    assert any(t.startswith("END a-b8 exit=0") for t in texts)


def test_the_gate_holds_a_start_until_enough_memory_is_free(
    tmp_path: Path,
) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _slotted(
        tmp_path,
        _job("a"),
        _job("b", experiment="r2"),
        min_free_gpu_mib=5000,
    )
    plan, started = _planned(clock, {"r1": 100, "r2": 10})
    launcher = FakeLauncher(clock, runs, plan)

    def free() -> int:
        return 8000 if clock.now - 1_700_000_000.0 >= 30 else 1000

    code = _run_slots(spec, launcher, clock, free=free)

    assert code == 0
    assert started == {"r1": 0, "r2": 30}


def test_the_first_job_never_waits_on_the_gate(tmp_path: Path) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _slotted(tmp_path, _job("a"), min_free_gpu_mib=5000)
    plan, started = _planned(clock, {"r1": 10})

    code = _run_slots(
        spec, FakeLauncher(clock, runs, plan), clock, free=lambda: 0
    )

    assert code == 0
    assert started == {"r1": 0}


def test_a_failed_gpu_query_holds_the_gate_only_while_a_job_runs(
    tmp_path: Path,
) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _slotted(
        tmp_path,
        _job("a"),
        _job("b", experiment="r2"),
        _job("c", experiment="r3"),
        min_free_gpu_mib=5000,
    )
    plan, started = _planned(clock, {"r1": 50, "r2": 30, "r3": 10})

    code = _run_slots(
        spec, FakeLauncher(clock, runs, plan), clock, free=lambda: None
    )

    assert code == 0
    assert started == {"r1": 0, "r2": 50, "r3": 80}


def test_the_start_gap_spaces_the_starts(tmp_path: Path) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _slotted(
        tmp_path,
        _job("a"),
        _job("b", experiment="r2"),
        _job("c", experiment="r3"),
        slots=3,
        start_gap_seconds=30,
    )
    plan, started = _planned(clock, {"r1": 200, "r2": 200, "r3": 200})

    _run_slots(spec, FakeLauncher(clock, runs, plan), clock)

    assert started == {"r1": 0, "r2": 30, "r3": 60}


def test_the_start_gap_does_not_delay_a_job_when_none_runs(
    tmp_path: Path,
) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _slotted(
        tmp_path,
        _job("a"),
        _job("b", experiment="r2"),
        start_gap_seconds=600,
    )
    plan, started = _planned(clock, {"r1": 10, "r2": 10})

    _run_slots(spec, FakeLauncher(clock, runs, plan), clock)

    assert started == {"r1": 0, "r2": 10}


def test_an_interrupt_stops_every_running_job(tmp_path: Path) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _slotted(tmp_path, _job("a"), _job("b", experiment="r2"))
    plan, _ = _planned(clock, {"r1": 1000, "r2": 1000})
    launcher = FakeLauncher(clock, runs, plan)
    sleeps = 0

    def sleep(seconds: float) -> None:
        nonlocal sleeps
        sleeps += 1
        if sleeps == 3:
            raise KeyboardInterrupt
        clock.sleep(seconds)

    code = _run_slots(spec, launcher, clock, sleep=sleep)

    texts = _texts(spec)
    assert code == 130
    assert len(launcher.processes) == 2
    assert all(process.terminated for process in launcher.processes)
    assert "INTERRUPTED a" in texts
    assert "INTERRUPTED b" in texts
    assert texts[-1] == "CHAIN_INTERRUPTED"


def test_the_peak_of_a_job_is_the_device_wide_peak_while_it_ran(
    tmp_path: Path,
) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _slotted(
        tmp_path,
        _job("a"),
        _job("b", experiment="r2"),
        start_gap_seconds=20,
    )
    plan, _ = _planned(clock, {"r1": 100, "r2": 10})

    def used() -> int:
        elapsed = clock.now - 1_700_000_000.0
        return 900 if 20 <= elapsed < 30 else 100

    _run_slots(spec, FakeLauncher(clock, runs, plan), clock, used=used)

    ends = {t.split()[1]: t for t in _texts(spec) if t.startswith("END")}
    assert " peak_gpu=900 " in ends["a"]
    assert " peak_gpu=900 " in ends["b"]


def test_a_heartbeat_reports_the_gpu_utilization(tmp_path: Path) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _chain(tmp_path, _job("a"))
    launcher = FakeLauncher(clock, runs, lambda c: Outcome(seconds=400))

    _run_slots(
        spec,
        launcher,
        clock,
        used=lambda: 1234,
        util=lambda: 87,
        heartbeat_seconds=300,
        poll_seconds=5,
    )

    assert "HEARTBEAT a elapsed=300 gpu_used=1234 gpu_util=87" in _texts(spec)


def test_every_running_job_logs_its_own_heartbeat(tmp_path: Path) -> None:
    clock, runs = _fresh(tmp_path)
    spec = _slotted(tmp_path, _job("a"), _job("b", experiment="r2"))
    plan, _ = _planned(clock, {"r1": 400, "r2": 400})

    _run_slots(
        spec,
        FakeLauncher(clock, runs, plan),
        clock,
        heartbeat_seconds=300,
        poll_seconds=5,
    )

    beats = [t for t in _texts(spec) if t.startswith("HEARTBEAT")]
    assert sorted(beats) == [
        "HEARTBEAT a elapsed=300 gpu_used=na",
        "HEARTBEAT b elapsed=300 gpu_used=na",
    ]


def test_make_gpu_queries_read_the_free_memory_and_the_utilization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_smi(monkeypatch, 0, "42\n7\n")

    queries = chain_module.make_gpu_queries()

    assert queries.free() == 42
    assert queries.util() == 42
    assert [c[1] for c in calls] == [
        "--query-gpu=memory.free",
        "--query-gpu=utilization.gpu",
    ]


def test_make_gpu_queries_are_none_without_nvidia_smi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)

    queries = chain_module.make_gpu_queries()

    assert (queries.used(), queries.free(), queries.util()) == (
        None,
        None,
        None,
    )
