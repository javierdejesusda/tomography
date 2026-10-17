"""Tests for the RunPod pod lifecycle commands, using fakes only."""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import pytest

from tomography import cloud

API_KEY = "rpa_SECRET_SENTINEL_123"
PUBLIC_KEY = "ssh-ed25519 AAAAC3Nza-test laptop"
BASE = "https://rest.runpod.io/v1"

Response = tuple[int, Any]


class FakeTransport:
    """Replays queued responses and records every request."""

    def __init__(self, responses: Sequence[Response] = ()) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, Mapping[str, str], bytes | None]] = []

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
    ) -> Response:
        self.calls.append((method, url, headers, body))
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


class FakeRunner:
    """Records commands; answers from a handler or returns success."""

    def __init__(
        self,
        handler: Callable[[Sequence[str]], cloud.CommandResult] | None = None,
    ) -> None:
        self.handler = handler
        self.runs: list[tuple[list[str], bytes | None]] = []
        self.pipes: list[tuple[list[str], list[str]]] = []

    def run(
        self, argv: Sequence[str], input_bytes: bytes | None = None
    ) -> cloud.CommandResult:
        self.runs.append((list(argv), input_bytes))
        if self.handler is not None:
            return self.handler(argv)
        return cloud.CommandResult(0, "")

    def pipe(self, source: Sequence[str], sink: Sequence[str]) -> int:
        self.pipes.append((list(source), list(sink)))
        return 0


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def ready_pod(pod_id: str = "pod1") -> dict[str, Any]:
    return {
        "id": pod_id,
        "name": "tomography",
        "desiredStatus": "RUNNING",
        "publicIp": "203.0.113.7",
        "portMappings": {"22": 10341},
        "costPerHr": 0.34,
    }


def starting_pod(pod_id: str = "pod1") -> dict[str, Any]:
    return {
        "id": pod_id,
        "desiredStatus": "RUNNING",
        "publicIp": "",
        "portMappings": None,
        "costPerHr": 0.34,
    }


class Harness:
    def __init__(
        self,
        tmp_path: Path,
        responses: Sequence[Response] = (),
        handler: Callable[[Sequence[str]], cloud.CommandResult] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.tmp_path = tmp_path
        self.transport = FakeTransport(responses)
        self.runner = FakeRunner(handler)
        self.clock = FakeClock()
        self.printed: list[str] = []
        self.state_path = tmp_path / "state" / "pod.json"
        self.pub_path = tmp_path / "id_test.pub"
        self.pub_path.write_text(PUBLIC_KEY + "\n", encoding="utf-8")
        self.deps = cloud.Deps(
            transport=self.transport,
            runner=self.runner,
            clock=self.clock,
            sleep=self.clock.sleep,
            env={"RUNPOD_API_KEY": API_KEY} if env is None else env,
            out=self.printed.append,
        )

    def main(self, *args: str) -> int:
        return cloud.main(
            [args[0], "--state", str(self.state_path), *args[1:]],
            deps=self.deps,
        )

    def state(self) -> dict[str, Any]:
        loaded: dict[str, Any] = json.loads(
            self.state_path.read_text(encoding="utf-8")
        )
        return loaded

    def output(self) -> str:
        return "\n".join(self.printed)


def up_harness(
    tmp_path: Path, responses: Sequence[Response] | None = None
) -> Harness:
    if responses is None:
        responses = [(201, starting_pod()), (200, ready_pod())]
    return Harness(tmp_path, responses)


def up(h: Harness, *extra: str) -> int:
    return h.main("up", "--public-key", str(h.pub_path), *extra)


def write_state(h: Harness, **overrides: Any) -> None:
    data: dict[str, Any] = {
        "pod_id": "pod1",
        "name": "tomography",
        "ip": "203.0.113.7",
        "ssh_port": 10341,
        "identity": str(h.tmp_path / "id_test"),
        "status": "running",
    }
    data.update(overrides)
    h.state_path.parent.mkdir(parents=True, exist_ok=True)
    h.state_path.write_text(json.dumps(data), encoding="utf-8")


def test_up_posts_the_community_pod_request(tmp_path: Path) -> None:
    h = up_harness(tmp_path)
    assert up(h) == 0
    method, url, headers, body = h.transport.calls[0]
    assert (method, url) == ("POST", f"{BASE}/pods")
    assert headers["Authorization"] == f"Bearer {API_KEY}"
    assert headers["Content-Type"] == "application/json"
    assert body is not None
    assert json.loads(body) == {
        "name": "tomography",
        "imageName": cloud.DEFAULT_IMAGE,
        "gpuTypeIds": ["NVIDIA GeForce RTX 4090"],
        "gpuCount": 1,
        "cloudType": "COMMUNITY",
        "containerDiskInGb": 30,
        "volumeInGb": 50,
        "volumeMountPath": "/workspace",
        "ports": ["22/tcp"],
        "allowedCudaVersions": ["13.0"],
        "supportPublicIp": True,
        "interruptible": False,
        "env": {"PUBLIC_KEY": PUBLIC_KEY},
    }


def test_up_secure_with_network_volume(tmp_path: Path) -> None:
    h = up_harness(tmp_path)
    assert up(h, "--cloud", "secure", "--volume-id", "vol9", "--gpu", "X") == 0
    body = json.loads(h.transport.calls[0][3] or b"")
    assert body["cloudType"] == "SECURE"
    assert body["networkVolumeId"] == "vol9"
    assert body["gpuTypeIds"] == ["X"]
    assert "volumeInGb" not in body
    assert "supportPublicIp" not in body


def test_up_polls_until_ip_and_port_are_exposed(tmp_path: Path) -> None:
    h = up_harness(
        tmp_path,
        [
            (201, starting_pod()),
            (200, starting_pod()),
            (200, starting_pod()),
            (200, ready_pod()),
        ],
    )
    assert up(h) == 0
    methods = [call[0] for call in h.transport.calls]
    assert methods == ["POST", "GET", "GET", "GET"]
    assert h.transport.calls[1][1] == f"{BASE}/pods/pod1"
    assert len(h.clock.sleeps) == 3
    state = h.state()
    assert state["ip"] == "203.0.113.7"
    assert state["ssh_port"] == 10341
    assert state["pod_id"] == "pod1"
    assert state["cost_per_hr"] == 0.34
    assert state["gpu"] == "NVIDIA GeForce RTX 4090"
    assert state["cloud"] == "COMMUNITY"
    assert state["status"] == "running"
    assert state["created"]


def test_up_timeout_keeps_the_pod_id_for_down(tmp_path: Path) -> None:
    h = up_harness(tmp_path, [(201, starting_pod()), (200, starting_pod())])
    assert up(h, "--timeout-minutes", "2") == 1
    assert h.state()["pod_id"] == "pod1"
    assert h.state()["status"] == "starting"
    assert "down" in h.output()
    assert sum(h.clock.sleeps) >= 120


def test_up_never_leaks_the_api_key(tmp_path: Path) -> None:
    h = up_harness(tmp_path)
    assert up(h) == 0
    assert API_KEY not in h.state_path.read_text(encoding="utf-8")
    assert API_KEY not in h.output()
    assert API_KEY not in repr(h.runner.runs) + repr(h.runner.pipes)
    body = h.transport.calls[0][3] or b""
    assert API_KEY.encode() not in body


def test_http_error_prints_status_and_message_without_the_key(
    tmp_path: Path,
) -> None:
    h = up_harness(tmp_path, [(401, {"error": f"bad key {API_KEY}"})])
    assert up(h) == 1
    assert "401" in h.output()
    assert "bad key" in h.output()
    assert API_KEY not in h.output()


def test_up_requires_the_key_in_the_environment(tmp_path: Path) -> None:
    h = Harness(tmp_path, [(201, ready_pod())], env={})
    assert up(h) == 1
    assert "RUNPOD_API_KEY" in h.output()
    assert h.transport.calls == []


def ssh_prefix(h: Harness) -> list[str]:
    known_hosts = (h.state_path.parent / "known_hosts").as_posix()
    return [
        "ssh",
        "-p",
        "10341",
        "-i",
        str(h.tmp_path / "id_test"),
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-o",
        "ServerAliveInterval=30",
        "root@203.0.113.7",
    ]


def test_ssh_without_a_command_prints_the_ssh_line(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    write_state(h)
    assert h.main("ssh") == 0
    assert h.runner.runs == []
    printed = h.output()
    assert printed.startswith("ssh -p 10341 -i ")
    assert "root@203.0.113.7" in printed
    assert "StrictHostKeyChecking=accept-new" in printed


def test_ssh_with_a_command_runs_it_as_one_remote_string(
    tmp_path: Path,
) -> None:
    h = Harness(tmp_path)
    write_state(h)
    assert h.main("ssh", "--", "ls", "/work space") == 0
    assert h.runner.runs == [([*ssh_prefix(h), "ls '/work space'"], None)]


def push_handler(
    cache_present: bool,
) -> Callable[[Sequence[str]], cloud.CommandResult]:
    def handle(argv: Sequence[str]) -> cloud.CommandResult:
        if argv[:2] == ["git", "rev-parse"]:
            return cloud.CommandResult(0, "abc123def456\n")
        if "test -d" in argv[-1]:
            return cloud.CommandResult(0 if cache_present else 1, "")
        return cloud.CommandResult(0, "")

    return handle


def push_harness(tmp_path: Path, cache_present: bool) -> tuple[Harness, Path]:
    h = Harness(tmp_path, handler=push_handler(cache_present))
    write_state(h)
    spec = tmp_path / "chain11.toml"
    spec.write_text("[chain]\nname = 'x'\n", encoding="utf-8")
    return h, spec


def test_push_streams_git_archive_into_the_source_folder(
    tmp_path: Path,
) -> None:
    h, spec = push_harness(tmp_path, cache_present=True)
    assert h.main("push", "--ref", "HEAD", "--spec", str(spec)) == 0
    assert h.runner.runs[0][0] == ["git", "rev-parse", "--short=12", "HEAD"]
    src = "/workspace/src/abc123def456"
    assert h.runner.pipes[0] == (
        ["git", "archive", "--format=tar", "abc123def456"],
        [*ssh_prefix(h), f"mkdir -p {src} && tar -x -C {src}"],
    )
    assert len(h.runner.pipes) == 1
    state = h.state()
    assert state["remote_src"] == src
    assert state["remote_spec"] == "/workspace/chains/chain11.toml"


def test_push_copies_the_spec_through_stdin(tmp_path: Path) -> None:
    h, spec = push_harness(tmp_path, cache_present=True)
    assert h.main("push", "--ref", "HEAD", "--spec", str(spec)) == 0
    copies = [r for r in h.runner.runs if "cat >" in r[0][-1]]
    assert copies == [
        (
            [
                *ssh_prefix(h),
                "mkdir -p /workspace/chains && "
                "cat > /workspace/chains/chain11.toml",
            ],
            spec.read_bytes(),
        )
    ]


def test_push_sends_the_cache_only_when_missing(tmp_path: Path) -> None:
    h, spec = push_harness(tmp_path, cache_present=False)
    cache = tmp_path / "cache"
    cache.mkdir()
    args = ("push", "--ref", "HEAD", "--spec", str(spec), "--cache", str(cache))
    assert h.main(*args) == 0
    dest = "/workspace/cache/tomography"
    assert h.runner.pipes[1] == (
        ["tar", "-c", "-C", str(cache), "."],
        [*ssh_prefix(h), f"mkdir -p {dest} && tar -x -C {dest}"],
    )
    assert h.state()["remote_cache"] == dest


def test_push_skips_a_present_cache_unless_sync_cache(tmp_path: Path) -> None:
    h, spec = push_harness(tmp_path, cache_present=True)
    cache = tmp_path / "cache"
    cache.mkdir()
    args = ("push", "--ref", "HEAD", "--spec", str(spec), "--cache", str(cache))
    assert h.main(*args) == 0
    assert len(h.runner.pipes) == 1
    assert h.main(*args, "--sync-cache") == 0
    assert len(h.runner.pipes) == 3
    assert h.runner.pipes[-1][0][:2] == ["tar", "-c"]


def test_push_fails_when_the_pipeline_fails(tmp_path: Path) -> None:
    h, spec = push_harness(tmp_path, cache_present=True)
    h.runner.pipe = lambda source, sink: 2  # type: ignore[method-assign]
    assert h.main("push", "--ref", "HEAD", "--spec", str(spec)) == 1
    assert "exit code 2" in h.output()


SRC = "/workspace/src/abc123def456"
SMI = "nvidia-smi --query-gpu=driver_version,name,memory.total"


def pushed(h: Harness, **extra: Any) -> None:
    write_state(
        h,
        remote_src=SRC,
        remote_spec="/workspace/chains/chain11.toml",
        remote_cache="/workspace/cache/tomography",
        **extra,
    )


def smi_handler(
    driver: str, torch_ok: bool = True
) -> Callable[[Sequence[str]], cloud.CommandResult]:
    def handle(argv: Sequence[str]) -> cloud.CommandResult:
        command = argv[-1]
        if command.startswith(SMI):
            return cloud.CommandResult(
                0, f"{driver}, NVIDIA GeForce RTX 4090, 24564 MiB\n"
            )
        if "torch.cuda.is_available" in command:
            return cloud.CommandResult(0 if torch_ok else 1, "")
        return cloud.CommandResult(0, "")

    return handle


def test_setup_refuses_a_driver_older_than_580(tmp_path: Path) -> None:
    h = Harness(tmp_path, handler=smi_handler("570.133.20"))
    pushed(h)
    assert h.main("setup") == 1
    assert "580" in h.output()
    assert "570" in h.output()
    assert len(h.runner.runs) == 1


def test_setup_installs_uv_syncs_and_checks_cuda(tmp_path: Path) -> None:
    h = Harness(tmp_path, handler=smi_handler("580.65.06"))
    pushed(h)
    assert h.main("setup") == 0
    commands = [run[0][-1] for run in h.runner.runs]
    assert len(commands) == 3
    assert commands[0].startswith(SMI)
    install = commands[1]
    assert "command -v uv" in install
    assert "https://astral.sh/uv/install.sh" in install
    assert f"cd {SRC}" in install
    assert "UV_CACHE_DIR=/workspace/uv-cache" in install
    assert "UV_PROJECT_ENVIRONMENT=/workspace/venv/abc123def456" in install
    assert "uv sync --locked" in install
    assert "import torch; assert torch.cuda.is_available()" in commands[2]
    assert "UV_PROJECT_ENVIRONMENT=/workspace/venv/abc123def456" in commands[2]


def test_setup_fails_when_cuda_is_unavailable(tmp_path: Path) -> None:
    h = Harness(tmp_path, handler=smi_handler("580.65.06", torch_ok=False))
    pushed(h)
    assert h.main("setup") == 1
    assert "cuda" in h.output().lower()


def test_setup_needs_a_pushed_source(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    write_state(h)
    assert h.main("setup") == 1
    assert "push" in h.output()


def started(h: Harness, *extra: str) -> tuple[str, str, list[str]]:
    code = h.main("start", "--spec", "chain11.toml", *extra)
    assert code == 0
    uploads = {
        run[0][-1]: (run[1] or b"").decode() for run in h.runner.runs if run[1]
    }
    chain = next(v for k, v in uploads.items() if k.endswith("chain.sh"))
    dog = next(v for k, v in uploads.items() if k.endswith("watchdog.sh"))
    launches = [run[0][-1] for run in h.runner.runs if not run[1]]
    return chain, dog, launches


def test_start_writes_the_chain_launcher(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    pushed(h)
    chain, _, _ = started(h, "--max-hours", "2")
    assert "export HF_HOME=/workspace/hf" in chain
    assert "export TOMOGRAPHY_CACHE=/workspace/cache/tomography" in chain
    assert "export UV_CACHE_DIR=/workspace/uv-cache" in chain
    assert "export UV_PROJECT_ENVIRONMENT=/workspace/venv/abc123def456" in chain
    assert f"cd {SRC}" in chain
    assert "echo $$ > /workspace/run/chain.pid" in chain
    assert (
        "uv run python -m tomography chain run /workspace/chains/chain11.toml"
        in chain
    )
    assert "HF_HUB_ENABLE_HF_TRANSFER" not in chain


def test_start_writes_a_watchdog_that_removes_the_pod(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    pushed(h)
    _, dog, _ = started(h, "--max-hours", "2", "--grace-minutes", "5")
    assert "MAX_SECONDS=7200" in dog
    assert "GRACE_SECONDS=300" in dog
    assert "CHAIN_DONE" in dog
    assert "/workspace/out/_chain" in dog
    assert "/proc/1/environ" in dog
    assert 'runpodctl remove pod "$RUNPOD_POD_ID"' in dog
    assert API_KEY not in dog


def test_start_launches_both_detached(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    pushed(h)
    _, _, launches = started(h, "--max-hours", "1")
    joined = " ".join(launches)
    for script in ("chain.sh", "watchdog.sh"):
        assert (
            f"setsid nohup bash /workspace/run/{script} "
            f"> /workspace/run/{script}.out 2>&1 < /dev/null &"
        ) in joined
    assert "touch /workspace/run/started" in joined
    assert API_KEY not in repr(h.runner.runs)


def test_start_uses_the_chosen_out_root(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    pushed(h)
    _, dog, _ = started(h, "--max-hours", "1", "--out", "/workspace/results")
    assert "/workspace/results/_chain" in dog
    assert h.state()["remote_out"] == "/workspace/results"


@pytest.mark.parametrize("hours", ["0", "-1", "24.5", "abc"])
def test_start_rejects_a_bad_max_hours(tmp_path: Path, hours: str) -> None:
    h = Harness(tmp_path)
    pushed(h)
    with pytest.raises(SystemExit):
        h.main("start", "--spec", "c.toml", "--max-hours", hours)


def test_start_requires_a_pushed_source(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    write_state(h)
    assert h.main("start", "--spec", "c.toml", "--max-hours", "1") == 1
    assert "push" in h.output()


def test_status_tails_the_log_and_reports_the_process(tmp_path: Path) -> None:
    def handle(argv: Sequence[str]) -> cloud.CommandResult:
        return cloud.CommandResult(0, "END r1 exit=0\nCHAIN_ALIVE\n")

    h = Harness(tmp_path, handler=handle)
    pushed(h, remote_out="/workspace/out")
    assert h.main("status") == 0
    command = h.runner.runs[0][0][-1]
    assert "ls -t /workspace/out/_chain/*.log" in command
    assert "tail -n 20" in command
    assert "kill -0" in command
    assert "CHAIN_ALIVE" in h.output()


def test_pull_streams_a_tar_into_the_destination(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    pushed(h, remote_out="/workspace/out")
    dest = tmp_path / "pulled"
    assert h.main("pull", "--dest", str(dest)) == 0
    assert dest.is_dir()
    assert h.runner.pipes == [
        (
            [*ssh_prefix(h), "tar -c -C /workspace/out ."],
            ["tar", "-x", "-C", str(dest)],
        )
    ]


def test_pull_accepts_another_remote_path(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    pushed(h)
    dest = tmp_path / "pulled"
    assert h.main("pull", "--dest", str(dest), "--path", "/workspace/a b") == 0
    assert h.runner.pipes[0][0][-1] == "tar -c -C '/workspace/a b' ."


def test_down_deletes_and_confirms_the_pod_is_gone(tmp_path: Path) -> None:
    h = Harness(tmp_path, [(204, None), (404, {"error": "pod not found"})])
    write_state(h)
    assert h.main("down") == 0
    assert [c[0] for c in h.transport.calls] == ["DELETE", "GET"]
    assert h.transport.calls[0][1] == f"{BASE}/pods/pod1"
    assert h.state()["status"] == "deleted"
    assert "deleted" in h.output()


def test_down_accepts_a_terminated_pod_as_confirmation(tmp_path: Path) -> None:
    h = Harness(tmp_path, [(200, None), (200, {"desiredStatus": "TERMINATED"})])
    write_state(h)
    assert h.main("down") == 0
    assert h.state()["status"] == "deleted"


def test_down_treats_an_unknown_pod_as_already_gone(tmp_path: Path) -> None:
    h = Harness(tmp_path, [(404, {"error": "pod not found"})])
    write_state(h)
    assert h.main("down") == 0
    assert h.state()["status"] == "deleted"
    assert "already" in h.output()


def test_down_reports_a_failed_delete_without_the_key(tmp_path: Path) -> None:
    h = Harness(tmp_path, [(500, {"message": f"boom {API_KEY}"})])
    write_state(h)
    assert h.main("down") == 1
    assert "500" in h.output()
    assert "boom" in h.output()
    assert API_KEY not in h.output()
    assert h.state()["status"] == "running"


def test_down_fails_when_the_pod_never_disappears(tmp_path: Path) -> None:
    h = Harness(tmp_path, [(204, None), (200, {"desiredStatus": "RUNNING"})])
    write_state(h)
    assert h.main("down", "--timeout-minutes", "1") == 1
    assert "still" in h.output()
    assert h.state()["status"] == "running"


@pytest.mark.parametrize(
    "argv",
    [
        ["up"],
        ["up", "--gpu", "G", "--cloud", "secure", "--volume-id", "v"],
        ["ssh"],
        ["ssh", "--", "nvidia-smi"],
        ["push", "--ref", "HEAD", "--spec", "s.toml", "--sync-cache"],
        ["setup"],
        ["start", "--spec", "s.toml", "--max-hours", "3"],
        ["status"],
        ["pull", "--dest", "d"],
        ["down"],
    ],
)
def test_every_subcommand_parses(argv: list[str]) -> None:
    args = cloud.build_parser().parse_args(argv)
    assert args.command == argv[0]
    assert callable(args.handler)
    assert args.state.name == "pod.json"


def test_remote_commands_never_carry_the_key(tmp_path: Path) -> None:
    h = Harness(tmp_path, handler=smi_handler("580.65.06"))
    pushed(h)
    for command in (
        ["setup"],
        ["start", "--spec", "c.toml", "--max-hours", "1"],
        ["status"],
    ):
        assert h.main(*command) == 0
    assert API_KEY not in repr(h.runner.runs)
    assert API_KEY not in h.output()
    assert API_KEY not in h.state_path.read_text(encoding="utf-8")


def _bash_works() -> bool:
    if shutil.which("bash") is None:
        return False
    probe = subprocess.run(
        ["bash", "-c", "true"], capture_output=True, check=False
    )
    return probe.returncode == 0


@pytest.mark.skipif(not _bash_works(), reason="needs a working bash")
def test_generated_scripts_are_valid_bash(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    pushed(h)
    chain, dog, _ = started(h, "--max-hours", "1")
    for name, text in (("chain.sh", chain), ("watchdog.sh", dog)):
        script = tmp_path / name
        script.write_text(text, encoding="utf-8", newline="\n")
        done = subprocess.run(
            ["bash", "-n", str(script)], capture_output=True, check=False
        )
        assert done.returncode == 0, done.stderr.decode()
