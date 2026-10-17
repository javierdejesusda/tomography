"""Tests for the portable chain runner."""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest

from tomography.chain import ChainSpec
from tomography.chain import Job
from tomography.chain import build_command
from tomography.chain import dry_run_lines
from tomography.chain import launch_process
from tomography.chain import load_spec
from tomography.chain import query_gpu_used
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


class FakeClock:
    """A clock that only moves when the runner sleeps."""

    def __init__(self) -> None:
        self.now = 1_700_000_000.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakeProcess:
    """A process that exits once the fake clock reaches its end time."""

    def __init__(self, clock: FakeClock, seconds: float, code: int) -> None:
        self.clock = clock
        self.end = clock.now + seconds
        self.code = code

    def poll(self) -> int | None:
        return self.code if self.clock.now >= self.end else None


@dataclasses.dataclass
class Outcome:
    code: int = 0
    seconds: float = 10.0
    stdout: str | None = None


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

    def __call__(
        self, command: Sequence[str], env: Mapping[str, str], log_path: Path
    ) -> FakeProcess:
        self.commands.append(list(command))
        self.envs.append(env)
        self.logs.append(log_path)
        outcome = self.plan(command)
        run_dir = self.runs / f"run{len(self.commands)}"
        run_dir.mkdir(parents=True, exist_ok=True)
        text = outcome.stdout
        if text is None:
            text = f"loading\n{run_dir}\n\n" if outcome.code == 0 else "boom\n"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(text)
        return FakeProcess(self.clock, outcome.seconds, outcome.code)

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
    done = (folder / "DONE.txt").read_text(encoding="utf-8").strip()
    assert launcher.logs == [folder / "run.log"]
    assert (
        (folder / "run.log").read_text(encoding="utf-8").startswith("loading")
    )
    assert done == str(runs / "run1")


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
    (folder / "DONE.txt").write_text(str(runs / "gone"), encoding="utf-8")
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


def test_query_gpu_used_is_none_without_nvidia_smi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)

    assert query_gpu_used() is None


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


def test_query_gpu_used_reads_the_first_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_smi(monkeypatch, 0, "2048\n512\n")

    assert query_gpu_used() == 2048
    assert calls[0] == [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]


@pytest.mark.parametrize(
    ("code", "out"), [(1, "2048\n"), (0, "N/A\n"), (0, "")]
)
def test_query_gpu_used_is_none_when_the_query_fails(
    monkeypatch: pytest.MonkeyPatch, code: int, out: str
) -> None:
    _fake_smi(monkeypatch, code, out)

    assert query_gpu_used() is None


def test_launch_process_appends_stdout_and_stderr_to_the_log(
    tmp_path: Path,
) -> None:
    log = tmp_path / "run.log"
    log.write_text("before\n", encoding="utf-8")
    script = "import sys; print('out'); print('err', file=sys.stderr)"

    process = launch_process([sys.executable, "-c", script], os.environ, log)

    assert process.wait(timeout=60) == 0
    lines = log.read_text(encoding="utf-8").split()
    assert lines[0] == "before"
    assert sorted(lines[1:]) == ["err", "out"]


def test_run_chain_runs_real_subprocesses_end_to_end(tmp_path: Path) -> None:
    script = (
        "import pathlib, sys; "
        "d = pathlib.Path(sys.argv[1]); d.mkdir(exist_ok=True); print(d)"
    )
    runs = tmp_path / "runs"
    runs.mkdir()

    def launch(
        command: Sequence[str], env: Mapping[str, str], log_path: Path
    ) -> subprocess.Popen[bytes]:
        target = str(runs / command[4])
        return launch_process(
            [sys.executable, "-c", script, target], env, log_path
        )

    spec = _chain(tmp_path, _job("a", experiment="x1"))

    code = run_chain(spec, launch=launch, poll_seconds=0.01)

    assert code == 0
    done = (spec.out / "_chain" / "a" / "DONE.txt").read_text(encoding="utf-8")
    assert done.strip() == str(runs / "x1")
