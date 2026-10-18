"""Tests for the RunPod pod lifecycle commands, using fakes only."""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from typing import IO
from typing import Any

import pytest

from tomography import cloud

API_KEY = "rpa_SECRET_SENTINEL_123"
PUBLIC_KEY = "ssh-ed25519 AAAAC3Nza-test laptop"
BASE = "https://rest.runpod.io/v1"
FULL = "abc123def456" + "0" * 28
SHORT = "abc123def456"
SRC = f"/workspace/src/{SHORT}"
OUT = "/workspace/out/chain11"
CHAIN_LOG = f"{OUT}/_chain/chain11.log"
WALL0 = 1_700_000_000.0
SMI = "nvidia-smi --query-gpu=driver_version,name,memory.total"
GPU_CHECK = "nvidia-smi --query-gpu=name --format=csv,noheader"

Response = tuple[int, Any]


class FakeTransport:
    """Replays queued responses and records every request."""

    def __init__(self, responses: Sequence[Response] = ()) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, Mapping[str, str], bytes | None]] = []
        self.error_on_call: dict[int, Exception] = {}

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
    ) -> Response:
        index = len(self.calls)
        self.calls.append((method, url, headers, body))
        if index in self.error_on_call:
            raise self.error_on_call[index]
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


class FakeRunner:
    """Records commands and answers them through a handler."""

    def __init__(
        self,
        handler: Callable[[Sequence[str]], cloud.CommandResult],
    ) -> None:
        self.handler = handler
        self.runs: list[tuple[list[str], bytes | None]] = []
        self.pipes: list[tuple[list[str], list[str]]] = []
        self.streams: list[list[str]] = []
        self.stream_data = b""
        self.stream_code = 0
        self.pipe_code = 0

    def run(
        self, argv: Sequence[str], input_bytes: bytes | None = None
    ) -> cloud.CommandResult:
        self.runs.append((list(argv), input_bytes))
        return self.handler(argv)

    def pipe(self, source: Sequence[str], sink: Sequence[str]) -> int:
        self.pipes.append((list(source), list(sink)))
        return self.pipe_code

    def stream(
        self, argv: Sequence[str], consumer: Callable[[IO[bytes]], None]
    ) -> int:
        self.streams.append(list(argv))
        consumer(io.BytesIO(self.stream_data))
        return self.stream_code


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def wall(self) -> float:
        return WALL0 + (self.now - 1000.0)

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
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.tmp_path = tmp_path
        self.driver = "580.65.06"
        self.torch_ok = True
        self.git_ok = True
        self.cache_present = True
        self.gpu_ok = True
        self.preflight_rc = 0
        self.missing: list[str] = []
        self.chain_live = False
        self.ssh_failures = 0
        self.transport = FakeTransport(responses)
        self.runner = FakeRunner(self._handle)
        self.clock = FakeClock()
        self.printed: list[str] = []
        self.state_path = tmp_path / "state dir" / "pod.json"
        self.pub_path = tmp_path / "id_test.pub"
        self.pub_path.write_text(PUBLIC_KEY + "\n", encoding="utf-8")
        self.deps = cloud.Deps(
            transport=self.transport,
            runner=self.runner,
            clock=self.clock,
            wall=self.clock.wall,
            sleep=self.clock.sleep,
            env={"RUNPOD_API_KEY": API_KEY} if env is None else env,
            out=self.printed.append,
        )

    def _handle(self, argv: Sequence[str]) -> cloud.CommandResult:
        if argv[:2] == ["git", "rev-parse"]:
            return cloud.CommandResult(0, FULL + "\n")
        if argv[0] != "ssh":
            return cloud.CommandResult(0, "")
        command = argv[-1]
        if command == "true":
            if self.ssh_failures > 0:
                self.ssh_failures -= 1
                return cloud.CommandResult(255, "")
            return cloud.CommandResult(0, "")
        if "/proc/1/environ" in command and "runpodctl" in command:
            return cloud.CommandResult(
                self.preflight_rc,
                "".join(f"missing {name}\n" for name in self.missing),
            )
        if "CHAIN_LIVE" in command:
            return cloud.CommandResult(
                0, "CHAIN_LIVE\n" if self.chain_live else ""
            )
        if "test -d" in command:
            return cloud.CommandResult(0 if self.cache_present else 1, "")
        if command.startswith(GPU_CHECK):
            return cloud.CommandResult(
                0 if self.gpu_ok else 9,
                "NVIDIA GeForce RTX 4090\n" if self.gpu_ok else "",
            )
        if command.startswith(SMI):
            return cloud.CommandResult(
                0, f"{self.driver}, NVIDIA GeForce RTX 4090, 24564 MiB\n"
            )
        if "command -v git" in command:
            return cloud.CommandResult(0 if self.git_ok else 1, "")
        if "torch.cuda.is_available" in command:
            return cloud.CommandResult(0 if self.torch_ok else 1, "")
        return cloud.CommandResult(0, "")

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

    def known_hosts(self, pod_id: str = "pod1") -> Path:
        return self.state_path.parent / f"known_hosts_{pod_id}"

    def commands(self) -> list[str]:
        return [run[0][-1] for run in self.runner.runs if run[0][0] == "ssh"]


def up_harness(
    tmp_path: Path, responses: Sequence[Response] | None = None
) -> Harness:
    if responses is None:
        responses = [(201, starting_pod()), (200, ready_pod())]
    return Harness(tmp_path, responses)


def up(h: Harness, *extra: str) -> int:
    return h.main(
        "up", "--public-key", str(h.pub_path), "--max-hours", "1", *extra
    )


def write_state(h: Harness, **overrides: Any) -> None:
    data: dict[str, Any] = {
        "pod_id": "pod1",
        "name": "tomography",
        "ip": "203.0.113.7",
        "ssh_port": 10341,
        "identity": str(h.tmp_path / "id_test"),
        "status": "running",
        "deadline": WALL0 + 10 * 3600,
    }
    data.update(overrides)
    h.state_path.parent.mkdir(parents=True, exist_ok=True)
    h.state_path.write_text(json.dumps(data), encoding="utf-8")


def pushed(h: Harness, **extra: Any) -> None:
    write_state(
        h,
        remote_src=SRC,
        remote_spec="/workspace/chains/chain11.toml",
        remote_cache="/workspace/cache/tomography",
        remote_out=OUT,
        remote_chain_log=CHAIN_LOG,
        **extra,
    )


def ssh_prefix(h: Harness, no_stdin: bool = False) -> list[str]:
    known_hosts = h.known_hosts().as_posix()
    return [
        "ssh",
        *(["-n"] if no_stdin else []),
        "-p",
        "10341",
        "-i",
        str(h.tmp_path / "id_test"),
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f'UserKnownHostsFile="{known_hosts}"',
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ConnectTimeout=15",
        "root@203.0.113.7",
    ]


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


def test_up_min_vcpu_asks_for_that_many_vcpus_per_gpu(tmp_path: Path) -> None:
    h = up_harness(tmp_path)
    assert up(h, "--min-vcpu", "16") == 0
    body = json.loads(h.transport.calls[0][3] or b"")
    assert body["minVCPUPerGPU"] == 16


@pytest.mark.parametrize(
    ("count", "reason"),
    [
        ("0", "must be at least 1"),
        ("-1", "must be at least 1"),
        ("abc", "not an integer"),
        ("1.5", "not an integer"),
    ],
)
def test_up_rejects_a_bad_min_vcpu(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], count: str, reason: str
) -> None:
    h = up_harness(tmp_path)
    with pytest.raises(SystemExit) as exc:
        up(h, "--min-vcpu", count)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--min-vcpu" in err
    assert reason in err
    assert h.transport.calls == []


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


def test_up_timeout_deletes_the_pod(tmp_path: Path) -> None:
    h = up_harness(tmp_path, [(201, starting_pod()), (200, starting_pod())])
    assert up(h, "--timeout-minutes", "2") == 1
    assert h.transport.calls[-1][:2] == ("DELETE", f"{BASE}/pods/pod1")
    assert h.state()["status"] == "deleted"
    assert h.state()["pod_id"] == "pod1"
    assert "deleted" in h.output()
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


def test_up_refuses_to_overwrite_a_live_pod(tmp_path: Path) -> None:
    h = up_harness(tmp_path)
    write_state(h, pod_id="old77")
    assert up(h) == 1
    assert "old77" in h.output()
    assert h.transport.calls == []
    assert h.state()["pod_id"] == "old77"


def test_up_force_replaces_the_state_and_names_the_old_pod(
    tmp_path: Path,
) -> None:
    h = up_harness(tmp_path)
    write_state(h, pod_id="old77")
    assert up(h, "--force") == 0
    assert "old77" in h.output()
    assert h.state()["pod_id"] == "pod1"


def test_up_allows_a_new_pod_after_a_deleted_one(tmp_path: Path) -> None:
    h = up_harness(tmp_path)
    write_state(h, pod_id="old77", status="deleted")
    assert up(h) == 0
    assert h.state()["pod_id"] == "pod1"


def test_a_corrupt_state_file_is_reported(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    h.state_path.parent.mkdir(parents=True)
    h.state_path.write_text("{not json", encoding="utf-8")
    assert h.main("status") == 1
    assert "state file" in h.output()
    assert "not valid JSON" in h.output()


def test_poll_failure_keeps_the_pod_id_in_the_message(tmp_path: Path) -> None:
    h = up_harness(tmp_path, [(201, starting_pod()), (200, starting_pod())])
    h.transport.error_on_call[1] = TimeoutError("read timed out")
    assert up(h) == 1
    assert "pod1" in h.output()
    assert h.transport.calls[-1][:2] == ("DELETE", f"{BASE}/pods/pod1")
    assert h.state()["status"] == "deleted"
    assert h.state()["pod_id"] == "pod1"
    assert "deleted" in h.output()


def test_create_timeout_is_a_clean_error(tmp_path: Path) -> None:
    h = up_harness(tmp_path)
    h.transport.error_on_call[0] = TimeoutError("read timed out")
    assert up(h) == 1
    assert "timed out" in h.output()
    assert "console" in h.output()


def test_up_waits_until_ssh_answers(tmp_path: Path) -> None:
    h = up_harness(tmp_path)
    h.ssh_failures = 2
    assert up(h) == 0
    probes = [r[0] for r in h.runner.runs if r[0][-1] == "true"]
    assert len(probes) == 3
    assert probes[0] == [*ssh_prefix(h, no_stdin=True), "true"]
    assert h.state()["status"] == "running"


def test_up_reports_when_ssh_never_answers(tmp_path: Path) -> None:
    h = up_harness(tmp_path)
    h.ssh_failures = 10_000
    assert up(h, "--ssh-timeout-minutes", "1") == 1
    assert "ssh" in h.output()
    assert h.transport.calls[-1][:2] == ("DELETE", f"{BASE}/pods/pod1")
    assert h.state()["status"] == "deleted"
    assert h.state()["pod_id"] == "pod1"
    assert "deleted" in h.output()


def test_up_creates_a_known_hosts_file_per_pod(tmp_path: Path) -> None:
    h = up_harness(tmp_path)
    assert up(h) == 0
    assert h.known_hosts().exists()


def test_up_records_the_deadline_and_arms_a_pod_watchdog(
    tmp_path: Path,
) -> None:
    h = up_harness(tmp_path)
    assert up(h) == 0
    assert h.state()["deadline"] == pytest.approx(WALL0 + 3600, abs=60)
    uploads = [
        (r[0][-1], (r[1] or b"").decode()) for r in h.runner.runs if r[1]
    ]
    _command, script = next(u for u in uploads if u[0].endswith("deadline.sh"))
    assert "MAX_SECONDS=3600" in script or "MAX_SECONDS=36" in script
    assert 'runpodctl stop pod "$RUNPOD_POD_ID"' in script
    assert "remove pod" not in script
    assert API_KEY not in script
    launches = [c for c in h.commands() if "setsid nohup" in c]
    assert len(launches) == 1
    assert "deadline.sh" in launches[0]


def test_up_deletes_the_pod_when_self_removal_is_missing(
    tmp_path: Path,
) -> None:
    h = up_harness(
        tmp_path,
        [(201, starting_pod()), (200, ready_pod()), (204, None)],
    )
    h.missing = ["RUNPOD_API_KEY"]
    assert up(h) == 1
    assert [c[0] for c in h.transport.calls][-1] == "DELETE"
    assert "RUNPOD_API_KEY" in h.output()
    assert h.state()["status"] == "deleted"


def test_up_keeps_the_pod_with_no_self_removal_and_warns(
    tmp_path: Path,
) -> None:
    h = up_harness(tmp_path)
    h.missing = ["runpodctl"]
    assert up(h, "--no-self-removal") == 0
    assert "DELETE" not in [c[0] for c in h.transport.calls]
    assert "WARNING" in h.output()
    assert "bills" in h.output()
    assert not [c for c in h.commands() if "setsid nohup" in c]


@pytest.mark.parametrize("hours", ["0", "24.5", "abc"])
def test_up_rejects_a_bad_max_hours(tmp_path: Path, hours: str) -> None:
    h = up_harness(tmp_path)
    with pytest.raises(SystemExit):
        h.main("up", "--max-hours", hours)


def test_up_requires_max_hours(tmp_path: Path) -> None:
    h = up_harness(tmp_path)
    with pytest.raises(SystemExit):
        h.main("up")


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
    assert h.runner.runs == [
        ([*ssh_prefix(h, no_stdin=True), "ls '/work space'"], None)
    ]


@pytest.mark.parametrize("state_dir", ["state dir", "state"])
def test_known_hosts_path_is_quoted_only_when_it_has_a_space(
    tmp_path: Path, state_dir: str
) -> None:
    h = Harness(tmp_path)
    h.state_path = tmp_path / state_dir / "pod.json"
    write_state(h)
    assert h.main("ssh", "--", "true") == 0
    option = next(
        a for a in h.runner.runs[0][0] if a.startswith("UserKnownHostsFile=")
    )
    path = h.known_hosts().as_posix()
    expected = f'"{path}"' if " " in state_dir else path
    assert option == f"UserKnownHostsFile={expected}"


def write_spec(tmp_path: Path, name: str = "chain11", out: str = OUT) -> Path:
    spec = tmp_path / "chain11.toml"
    spec.write_text(
        f'[chain]\nname = "{name}"\nout = "{out}"\n', encoding="utf-8"
    )
    return spec


def push_args(h: Harness, spec: Path, *extra: str) -> tuple[str, ...]:
    return ("push", "--ref", "HEAD", "--spec", str(spec), *extra)


def test_push_bundles_a_temporary_ref_and_deletes_it(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    write_state(h)
    spec = write_spec(tmp_path)
    assert h.main(*push_args(h, spec)) == 0
    assert h.runner.runs[0][0] == [
        "git",
        "rev-parse",
        "--verify",
        "HEAD^{commit}",
    ]
    ref = f"refs/tomography-cloud/{SHORT}"
    local = [r[0] for r in h.runner.runs if r[0][0] == "git"]
    assert local[1] == ["git", "update-ref", ref, FULL]
    assert local[-1] == ["git", "update-ref", "-d", ref]
    bundle = f"/workspace/src/{SHORT}.bundle"
    assert h.runner.pipes[0] == (
        ["git", "bundle", "create", "-", ref],
        [*ssh_prefix(h), f"mkdir -p /workspace/src && cat > {bundle}"],
    )


def test_push_clones_the_bundle_and_checks_out_the_commit(
    tmp_path: Path,
) -> None:
    h = Harness(tmp_path)
    write_state(h)
    assert h.main(*push_args(h, write_spec(tmp_path))) == 0
    fetch = next(c for c in h.commands() if "git fetch" in c)
    assert f"git init -q {SRC}" in fetch
    assert f"/workspace/src/{SHORT}.bundle" in fetch
    assert "refs/tomography-cloud/*:refs/tomography-cloud/*" in fetch
    assert f"git checkout -q --detach {FULL}" in fetch
    assert h.state()["remote_src"] == SRC


def test_push_deletes_the_temporary_ref_when_the_bundle_fails(
    tmp_path: Path,
) -> None:
    h = Harness(tmp_path)
    write_state(h)
    h.runner.pipe_code = 3
    assert h.main(*push_args(h, write_spec(tmp_path))) == 1
    local = [r[0] for r in h.runner.runs if r[0][0] == "git"]
    assert local[-1][:3] == ["git", "update-ref", "-d"]
    assert "exit code 3" in h.output()


@pytest.mark.parametrize("ref", ["--upload-pack=x", "-h"])
def test_push_refuses_a_ref_starting_with_a_dash(
    tmp_path: Path, ref: str
) -> None:
    h = Harness(tmp_path)
    write_state(h)
    code = h.main("push", f"--ref={ref}", "--spec", str(write_spec(tmp_path)))
    assert code == 1
    assert h.runner.runs == []
    assert "ref" in h.output()


def test_push_refuses_a_ref_git_cannot_resolve(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    write_state(h)
    original = h.runner.handler

    def failing(argv: Sequence[str]) -> cloud.CommandResult:
        if argv[:2] == ["git", "rev-parse"]:
            return cloud.CommandResult(128, "")
        return original(argv)

    h.runner.handler = failing
    assert h.main(*push_args(h, write_spec(tmp_path))) == 1
    assert "cannot resolve" in h.output()
    assert h.runner.pipes == []


def test_push_copies_the_spec_through_stdin(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    write_state(h)
    spec = write_spec(tmp_path)
    assert h.main(*push_args(h, spec)) == 0
    copies = [r for r in h.runner.runs if r[1] is not None]
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
    h = Harness(tmp_path)
    h.cache_present = False
    write_state(h)
    cache = tmp_path / "cache"
    cache.mkdir()
    spec = write_spec(tmp_path)
    assert h.main(*push_args(h, spec, "--cache", str(cache))) == 0
    dest = "/workspace/cache/tomography"
    assert h.runner.pipes[1] == (
        ["tar", "-c", "-C", str(cache), "."],
        [*ssh_prefix(h), f"mkdir -p {dest} && tar -x -C {dest}"],
    )
    assert h.state()["remote_cache"] == dest


def test_push_skips_a_present_cache_unless_sync_cache(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    write_state(h)
    cache = tmp_path / "cache"
    cache.mkdir()
    spec = write_spec(tmp_path)
    assert h.main(*push_args(h, spec, "--cache", str(cache))) == 0
    assert len(h.runner.pipes) == 1
    args = push_args(h, spec, "--cache", str(cache), "--sync-cache")
    assert h.main(*args) == 0
    assert len(h.runner.pipes) == 3
    assert h.runner.pipes[-1][0][:2] == ["tar", "-c"]


def test_push_records_the_out_root_from_the_spec(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    write_state(h)
    assert h.main(*push_args(h, write_spec(tmp_path))) == 0
    state = h.state()
    assert state["remote_out"] == OUT
    assert state["remote_chain_log"] == CHAIN_LOG
    assert state["remote_spec"] == "/workspace/chains/chain11.toml"


def test_push_resolves_a_relative_out_against_the_chains_folder(
    tmp_path: Path,
) -> None:
    h = Harness(tmp_path)
    write_state(h)
    spec = write_spec(tmp_path, out="../out/c11")
    assert h.main(*push_args(h, spec)) == 0
    assert h.state()["remote_out"] == "/workspace/out/c11"


@pytest.mark.parametrize("out", ["C:/results", "D:\\\\results", "out\\\\x"])
def test_push_refuses_windows_out_paths(tmp_path: Path, out: str) -> None:
    h = Harness(tmp_path)
    write_state(h)
    assert h.main(*push_args(h, write_spec(tmp_path, out=out))) == 1
    assert "out" in h.output()
    assert h.runner.pipes == []


def test_push_refuses_a_spec_without_a_chain_name(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    write_state(h)
    spec = tmp_path / "bad.toml"
    spec.write_text('[chain]\nout = "/workspace/out"\n', encoding="utf-8")
    assert h.main(*push_args(h, spec)) == 1
    assert "name" in h.output()


def test_pull_extracts_the_stream_in_python(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    pushed(h)
    h.runner.stream_data = make_tar([("a/b.txt", b"hello")])
    dest = tmp_path / "pulled"
    assert h.main("pull", "--dest", str(dest)) == 0
    assert (dest / "a" / "b.txt").read_bytes() == b"hello"
    assert h.runner.streams == [
        [*ssh_prefix(h, no_stdin=True), f"tar -c -C {OUT} ."]
    ]
    assert h.runner.pipes == []


def test_pull_accepts_another_remote_path(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    pushed(h)
    h.runner.stream_data = make_tar([])
    dest = tmp_path / "pulled"
    assert h.main("pull", "--dest", str(dest), "--path", "/workspace/a b") == 0
    assert h.runner.streams[0][-1] == "tar -c -C '/workspace/a b' ."


def make_tar(
    members: Sequence[tuple[str, bytes]],
    links: Sequence[tuple[str, str]] = (),
) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for name, target in links:
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            tar.addfile(info)
    return buffer.getvalue()


@pytest.mark.parametrize(
    "members",
    [
        [("/abs_evil.txt", b"x")],
        [("../escape.txt", b"x")],
        [("ok/../../escape.txt", b"x")],
    ],
)
def test_pull_rejects_paths_that_leave_the_destination(
    tmp_path: Path, members: list[tuple[str, bytes]]
) -> None:
    h = Harness(tmp_path)
    pushed(h)
    h.runner.stream_data = make_tar(members)
    dest = tmp_path / "sandbox" / "pulled"
    assert h.main("pull", "--dest", str(dest)) == 1
    assert "unsafe" in h.output()
    assert not (tmp_path / "sandbox" / "escape.txt").exists()
    assert not (tmp_path / "escape.txt").exists()
    assert list(dest.rglob("*")) == []


def test_pull_rejects_a_symlink_pointing_outside(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    pushed(h)
    h.runner.stream_data = make_tar(
        [("data.txt", b"x")], links=[("link", "../../outside")]
    )
    dest = tmp_path / "sandbox" / "pulled"
    assert h.main("pull", "--dest", str(dest)) == 1
    assert "unsafe" in h.output()
    assert not (dest / "link").is_symlink()


def test_pull_refuses_a_non_empty_destination(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    pushed(h)
    dest = tmp_path / "pulled"
    dest.mkdir()
    (dest / "keep.txt").write_text("x", encoding="utf-8")
    assert h.main("pull", "--dest", str(dest)) == 1
    assert "not empty" in h.output()
    assert h.runner.streams == []


def test_pull_fails_when_the_remote_tar_fails(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    pushed(h)
    h.runner.stream_code = 2
    assert h.main("pull", "--dest", str(tmp_path / "pulled")) == 1
    assert "exit code 2" in h.output()


def test_setup_refuses_a_driver_older_than_580(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    h.driver = "570.133.20"
    pushed(h)
    assert h.main("setup") == 1
    assert "580" in h.output()
    assert "570" in h.output()
    assert len(h.runner.runs) == 1


def test_setup_installs_uv_syncs_and_checks_cuda(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    pushed(h)
    assert h.main("setup") == 0
    commands = h.commands()
    assert len(commands) == 4
    assert commands[0].startswith(SMI)
    assert "command -v git" in commands[1]
    install = commands[2]
    assert "command -v uv" in install
    assert f"https://astral.sh/uv/{cloud.UV_VERSION}/install.sh" in install
    assert f"cd {SRC}" in install
    assert "UV_CACHE_DIR=/workspace/uv-cache" in install
    assert f"UV_PROJECT_ENVIRONMENT=/workspace/venv/{SHORT}" in install
    assert "uv sync --locked" in install
    assert "import torch; assert torch.cuda.is_available()" in commands[3]
    assert f"UV_PROJECT_ENVIRONMENT=/workspace/venv/{SHORT}" in commands[3]


def test_uv_version_is_pinned() -> None:
    assert cloud.UV_VERSION.count(".") == 2
    assert cloud.UV_VERSION[0].isdigit()


def test_setup_fails_without_git(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    h.git_ok = False
    pushed(h)
    assert h.main("setup") == 1
    assert "git" in h.output()


def test_setup_fails_when_cuda_is_unavailable(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    h.torch_ok = False
    pushed(h)
    assert h.main("setup") == 1
    assert "cuda" in h.output().lower()


def test_setup_needs_a_pushed_source(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    write_state(h)
    assert h.main("setup") == 1
    assert "push" in h.output()


def started(h: Harness, *extra: str) -> tuple[str, str, list[str]]:
    code = h.main("start", "--spec", "chain11.toml", "--max-hours", "2", *extra)
    assert code == 0
    uploads = {
        run[0][-1]: (run[1] or b"").decode() for run in h.runner.runs if run[1]
    }
    chain = next(v for k, v in uploads.items() if k.endswith("chain.sh"))
    dog = next(v for k, v in uploads.items() if k.endswith("watchdog.sh"))
    launches = [c for c in h.commands() if "setsid nohup" in c]
    return chain, dog, launches


def test_start_writes_the_chain_launcher(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    pushed(h)
    chain, _, _ = started(h)
    assert "export HF_HOME=/workspace/hf" in chain
    assert "export TOMOGRAPHY_CACHE=/workspace/cache/tomography" in chain
    assert "export UV_CACHE_DIR=/workspace/uv-cache" in chain
    assert f"export UV_PROJECT_ENVIRONMENT=/workspace/venv/{SHORT}" in chain
    assert f"cd {SRC}" in chain
    assert "echo $$ > /workspace/run/chain.pid" in chain
    assert (
        "uv run python -m tomography chain run /workspace/chains/chain11.toml"
        in chain
    )
    assert "HF_HUB_ENABLE_HF_TRANSFER" not in chain


def test_start_writes_a_watchdog_that_retries_the_removal(
    tmp_path: Path,
) -> None:
    h = Harness(tmp_path)
    pushed(h)
    _, dog, _ = started(h, "--grace-minutes", "5")
    assert "MAX_SECONDS=7200" in dog
    assert "GRACE_SECONDS=300" in dog
    assert "CHAIN_DONE" in dog
    assert f"CHAINLOG={CHAIN_LOG}" in dog
    assert "/proc/1/environ" in dog
    assert 'runpodctl stop pod "$RUNPOD_POD_ID"' in dog
    assert "remove pod" not in dog
    assert "stop attempt" in dog
    assert "sleep 60" in dog
    assert "until " in dog
    assert API_KEY not in dog


def test_start_counts_only_chain_done_lines_after_the_launch_offset(
    tmp_path: Path,
) -> None:
    h = Harness(tmp_path)
    pushed(h)
    _, dog, _ = started(h)
    assert "log_offset" in dog
    assert "tail -c +" in dog
    joined = " ".join(h.commands())
    assert "stat -c %s" in joined
    assert "/workspace/run/log_offset" in joined
    assert CHAIN_LOG in joined


def test_start_launches_both_detached(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    pushed(h)
    _, _, launches = started(h)
    joined = " ".join(launches)
    for script in ("chain.sh", "watchdog.sh"):
        assert (
            f"setsid nohup bash /workspace/run/{script} "
            f"> /workspace/run/{script}.out 2>&1 < /dev/null &"
        ) in joined
    commands = h.commands()
    prepare = next(c for c in commands if "touch /workspace/run/started" in c)
    assert commands.index(prepare) < commands.index(launches[0])
    assert API_KEY not in repr(h.runner.runs)
    launch_run = next(r for r in h.runner.runs if "setsid nohup" in r[0][-1])
    assert launch_run[0][1] == "-n"


def test_start_groups_the_background_jobs_in_braces(tmp_path: Path) -> None:
    # `a && b &` backgrounds the whole list, so its subshell keeps the ssh
    # channel open until the chain ends and `start` never returns. The
    # braces keep each `&` inside the group whatever precedes it.
    h = Harness(tmp_path)
    pushed(h)
    _, _, launches = started(h)
    assert launches == [
        "{ setsid nohup bash /workspace/run/chain.sh "
        "> /workspace/run/chain.sh.out 2>&1 < /dev/null & "
        "setsid nohup bash /workspace/run/watchdog.sh "
        "> /workspace/run/watchdog.sh.out 2>&1 < /dev/null & }"
    ]


def test_start_stops_when_the_launch_preparation_fails(
    tmp_path: Path,
) -> None:
    h = Harness(tmp_path)
    pushed(h)
    original = h.runner.handler

    def fail_prepare(argv: Sequence[str]) -> cloud.CommandResult:
        if argv[0] == "ssh" and "touch /workspace/run/started" in argv[-1]:
            return cloud.CommandResult(1, "")
        return original(argv)

    h.runner.handler = fail_prepare
    code = h.main("start", "--spec", "chain11.toml", "--max-hours", "1")
    assert code == 1
    assert not [c for c in h.commands() if "setsid nohup" in c]


def test_start_uses_the_earlier_of_the_two_deadlines(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    pushed(h, deadline=WALL0 + 3 * 3600)
    h.main("start", "--spec", "chain11.toml", "--max-hours", "20")
    dog = next(
        (r[1] or b"").decode()
        for r in h.runner.runs
        if r[1] and r[0][-1].endswith("watchdog.sh")
    )
    assert "MAX_SECONDS=10800" in dog


def test_start_refuses_after_the_up_deadline(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    pushed(h, deadline=WALL0 - 5)
    assert h.main("start", "--spec", "chain11.toml", "--max-hours", "1") == 1
    assert "deadline" in h.output()
    assert h.runner.runs == []


@pytest.mark.parametrize("hours", ["0", "-1", "24.5", "abc"])
def test_start_rejects_a_bad_max_hours(tmp_path: Path, hours: str) -> None:
    h = Harness(tmp_path)
    pushed(h)
    with pytest.raises(SystemExit):
        h.main("start", "--spec", "chain11.toml", "--max-hours", hours)


def test_start_requires_a_pushed_source(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    write_state(h)
    code = h.main("start", "--spec", "chain11.toml", "--max-hours", "1")
    assert code == 1
    assert "push" in h.output()


def test_start_refuses_a_spec_that_was_not_pushed(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    pushed(h)
    assert h.main("start", "--spec", "other.toml", "--max-hours", "1") == 1
    assert "push" in h.output()
    assert h.runner.runs == []


def test_start_has_no_out_option() -> None:
    with pytest.raises(SystemExit):
        cloud.build_parser().parse_args(
            ["start", "--spec", "s.toml", "--max-hours", "1", "--out", "x"]
        )


def test_start_refuses_when_self_removal_is_unavailable(
    tmp_path: Path,
) -> None:
    h = Harness(tmp_path)
    h.missing = ["RUNPOD_POD_ID", "runpodctl"]
    pushed(h)
    code = h.main("start", "--spec", "chain11.toml", "--max-hours", "1")
    assert code == 1
    assert "RUNPOD_POD_ID" in h.output()
    assert "runpodctl" in h.output()
    assert not [c for c in h.commands() if "setsid nohup" in c]
    assert not [r for r in h.runner.runs if r[1]]


def test_start_can_skip_self_removal_with_a_loud_warning(
    tmp_path: Path,
) -> None:
    h = Harness(tmp_path)
    h.missing = ["runpodctl"]
    pushed(h)
    _, _, launches = started(h, "--no-self-removal")
    assert "WARNING" in h.output()
    assert "bills" in h.output()
    assert launches


def test_preflight_never_asks_for_values(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    pushed(h)
    started(h)
    command = next(c for c in h.commands() if "/proc/1/environ" in c)
    assert "grep -q" in command
    assert "runpodctl" in command
    assert "RUNPOD_API_KEY" in command
    assert "cat /proc/1/environ" not in command


def test_start_refuses_a_second_chain(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    h.chain_live = True
    pushed(h)
    code = h.main("start", "--spec", "chain11.toml", "--max-hours", "1")
    assert code == 1
    assert "already running" in h.output()
    assert not [c for c in h.commands() if "setsid nohup" in c]


def test_status_tails_the_recorded_log_and_reports_the_exit(
    tmp_path: Path,
) -> None:
    h = Harness(tmp_path, [(200, ready_pod())])
    original = h.runner.handler

    def handle(argv: Sequence[str]) -> cloud.CommandResult:
        if argv[0] == "ssh":
            return cloud.CommandResult(
                0, "END r1 exit=0\nCHAIN_EXIT 0\nCHAIN_ALIVE\n"
            )
        return original(argv)

    h.runner.handler = handle
    pushed(h)
    assert h.main("status") == 0
    command = h.runner.runs[0][0][-1]
    assert f"tail -n 20 {CHAIN_LOG}" in command
    assert "kill -0" in command
    assert "/workspace/run/chain.exit" in command
    assert h.runner.runs[0][0][1] == "-n"
    assert "CHAIN_ALIVE" in h.output()


def test_down_deletes_and_confirms_the_pod_is_gone(tmp_path: Path) -> None:
    h = Harness(tmp_path, [(204, None), (404, {"error": "pod not found"})])
    write_state(h)
    h.known_hosts().write_text("x", encoding="utf-8")
    assert h.main("down") == 0
    assert [c[0] for c in h.transport.calls] == ["DELETE", "GET"]
    assert h.transport.calls[0][1] == f"{BASE}/pods/pod1"
    assert h.state()["status"] == "deleted"
    assert not h.known_hosts().exists()
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
        ["up", "--max-hours", "2"],
        [
            "up",
            "--max-hours",
            "2",
            "--gpu",
            "G",
            "--cloud",
            "secure",
            "--volume-id",
            "v",
            "--force",
            "--no-self-removal",
        ],
        ["ssh"],
        ["ssh", "--", "nvidia-smi"],
        ["push", "--ref", "HEAD", "--spec", "s.toml", "--sync-cache"],
        ["setup"],
        ["start", "--spec", "s.toml", "--max-hours", "3"],
        ["start", "--spec", "s.toml", "--max-hours", "3", "--no-self-removal"],
        ["resume", "--max-hours", "2", "--no-self-removal"],
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
    h = Harness(tmp_path, [(200, ready_pod())])
    pushed(h)
    for command in (
        ["setup"],
        ["start", "--spec", "chain11.toml", "--max-hours", "1"],
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
    chain, dog, _ = started(h)
    deadline = cloud._deadline_script(3600)
    for name, text in (
        ("chain.sh", chain),
        ("watchdog.sh", dog),
        ("deadline.sh", deadline),
    ):
        script = tmp_path / name
        script.write_text(text, encoding="utf-8", newline="\n")
        done = subprocess.run(
            ["bash", "-n", str(script)], capture_output=True, check=False
        )
        assert done.returncode == 0, done.stderr.decode()


def test_the_real_runner_streams_stdout_and_reports_the_exit() -> None:
    chunks: list[bytes] = []
    code = cloud.SubprocessRunner().stream(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.buffer.write(b'abc'); sys.exit(3)",
        ],
        lambda stream: chunks.append(stream.read()),
    )
    assert chunks == [b"abc"]
    assert code == 3


def test_the_real_runner_kills_the_command_when_the_consumer_fails() -> None:
    def refuse(stream: IO[bytes]) -> None:
        raise cloud.CloudError("unsafe")

    with pytest.raises(cloud.CloudError):
        cloud.SubprocessRunner().stream(
            [sys.executable, "-c", "import time; time.sleep(60)"], refuse
        )


def resumed_pod() -> dict[str, Any]:
    pod = ready_pod()
    pod["publicIp"] = "198.51.100.9"
    pod["portMappings"] = {"22": 20222}
    return pod


def stopped_harness(
    tmp_path: Path, responses: Sequence[Response] | None = None
) -> Harness:
    if responses is None:
        responses = [(200, None), (200, starting_pod()), (200, resumed_pod())]
    h = Harness(tmp_path, responses)
    pushed(h, status="stopped")
    return h


def test_resume_starts_the_pod_and_polls_for_the_new_endpoint(
    tmp_path: Path,
) -> None:
    h = stopped_harness(tmp_path)
    assert h.main("resume", "--max-hours", "2") == 0
    methods = [(c[0], c[1]) for c in h.transport.calls]
    assert methods == [
        ("POST", f"{BASE}/pods/pod1/start"),
        ("GET", f"{BASE}/pods/pod1"),
        ("GET", f"{BASE}/pods/pod1"),
    ]
    state = h.state()
    assert state["ip"] == "198.51.100.9"
    assert state["ssh_port"] == 20222
    assert state["status"] == "running"
    assert state["remote_src"] == SRC
    assert state["deadline"] == pytest.approx(WALL0 + 7200, abs=60)


def test_resume_resets_the_known_hosts_file(tmp_path: Path) -> None:
    h = stopped_harness(tmp_path)
    h.known_hosts().write_text("old host key", encoding="utf-8")
    assert h.main("resume", "--max-hours", "2") == 0
    assert h.known_hosts().read_text(encoding="utf-8") == ""


def test_resume_probes_ssh_checks_the_gpu_and_rearms_the_deadline(
    tmp_path: Path,
) -> None:
    h = stopped_harness(tmp_path)
    h.ssh_failures = 1
    assert h.main("resume", "--max-hours", "2") == 0
    probes = [r[0] for r in h.runner.runs if r[0][-1] == "true"]
    assert len(probes) == 2
    assert "20222" in probes[0]
    assert "root@198.51.100.9" in probes[0]
    commands = h.commands()
    assert any(c.startswith(GPU_CHECK) for c in commands)
    uploads = [
        (r[0][-1], (r[1] or b"").decode()) for r in h.runner.runs if r[1]
    ]
    script = next(u[1] for u in uploads if u[0].endswith("deadline.sh"))
    assert "MAX_SECONDS=7200" in script
    assert 'runpodctl stop pod "$RUNPOD_POD_ID"' in script
    assert [c for c in commands if "setsid nohup" in c]


def test_resume_refuses_a_deleted_pod(tmp_path: Path) -> None:
    h = Harness(tmp_path, [(200, None)])
    pushed(h, status="deleted")
    assert h.main("resume", "--max-hours", "2") == 1
    assert "deleted" in h.output()
    assert h.transport.calls == []


def test_resume_reports_a_start_error_without_the_key(tmp_path: Path) -> None:
    h = stopped_harness(tmp_path, [(400, {"error": f"no gpu {API_KEY}"})])
    assert h.main("resume", "--max-hours", "2") == 1
    assert "400" in h.output()
    assert "no gpu" in h.output()
    assert API_KEY not in h.output()


def test_resume_timeout_stops_the_pod_again(tmp_path: Path) -> None:
    h = stopped_harness(tmp_path, [(200, None), (200, starting_pod())])
    assert h.main("resume", "--max-hours", "2", "--timeout-minutes", "1") == 1
    assert h.transport.calls[-1][:2] == ("POST", f"{BASE}/pods/pod1/stop")
    assert "resume" in h.output()
    assert h.state()["status"] == "stopped"
    assert h.state()["pod_id"] == "pod1"


def test_resume_stops_the_pod_again_when_no_gpu_is_attached(
    tmp_path: Path,
) -> None:
    h = stopped_harness(
        tmp_path,
        [(200, None), (200, resumed_pod()), (200, None)],
    )
    h.gpu_ok = False
    assert h.main("resume", "--max-hours", "2") == 1
    assert h.transport.calls[-1][:2] == ("POST", f"{BASE}/pods/pod1/stop")
    assert "GPU" in h.output()
    assert "resume" in h.output()
    assert h.state()["status"] == "stopped"


def test_resume_stops_the_pod_when_it_cannot_stop_itself(
    tmp_path: Path,
) -> None:
    h = stopped_harness(
        tmp_path,
        [(200, None), (200, resumed_pod()), (200, None)],
    )
    h.missing = ["runpodctl"]
    assert h.main("resume", "--max-hours", "2") == 1
    assert h.transport.calls[-1][:2] == ("POST", f"{BASE}/pods/pod1/stop")
    assert "runpodctl" in h.output()
    assert h.state()["status"] == "stopped"


def test_resume_can_skip_self_removal_with_a_warning(tmp_path: Path) -> None:
    h = stopped_harness(tmp_path, [(200, None), (200, resumed_pod())])
    h.missing = ["runpodctl"]
    assert h.main("resume", "--max-hours", "2", "--no-self-removal") == 0
    assert "WARNING" in h.output()
    assert not [c for c in h.commands() if "setsid nohup" in c]


def test_status_says_when_the_pod_is_stopped(tmp_path: Path) -> None:
    h = Harness(tmp_path, [(200, {"id": "pod1", "desiredStatus": "EXITED"})])
    pushed(h)
    assert h.main("status") == 0
    assert "stopped" in h.output()
    assert "resume" in h.output()
    assert h.runner.runs == []


def test_status_says_when_the_pod_no_longer_exists(tmp_path: Path) -> None:
    h = Harness(tmp_path, [(404, {"error": "pod not found"})])
    pushed(h)
    assert h.main("status") == 0
    assert "no longer exists" in h.output()
    assert h.runner.runs == []


def test_status_still_reads_the_log_without_an_api_key(tmp_path: Path) -> None:
    h = Harness(tmp_path, env={})
    pushed(h)
    assert h.main("status") == 0
    assert h.transport.calls == []
    assert h.runner.runs


def test_pull_names_the_folder_when_the_extraction_is_partial(
    tmp_path: Path,
) -> None:
    h = Harness(tmp_path)
    pushed(h)
    h.runner.stream_data = make_tar([("ok.txt", b"x"), ("/abs_evil.txt", b"y")])
    dest = tmp_path / "pulled"
    assert h.main("pull", "--dest", str(dest)) == 1
    assert "partial extraction" in h.output()
    assert str(dest) in h.output()
    assert (dest / "ok.txt").exists()


def test_pull_does_not_claim_a_partial_extraction_when_nothing_landed(
    tmp_path: Path,
) -> None:
    h = Harness(tmp_path)
    pushed(h)
    h.runner.stream_data = make_tar([("/abs_evil.txt", b"y")])
    assert h.main("pull", "--dest", str(tmp_path / "pulled")) == 1
    assert "partial" not in h.output()


def test_resume_stops_the_pod_again_when_ssh_never_answers(
    tmp_path: Path,
) -> None:
    h = stopped_harness(
        tmp_path, [(200, None), (200, resumed_pod()), (200, None)]
    )
    h.ssh_failures = 10_000
    code = h.main("resume", "--max-hours", "2", "--ssh-timeout-minutes", "1")
    assert code == 1
    assert h.transport.calls[-1][:2] == ("POST", f"{BASE}/pods/pod1/stop")
    assert h.state()["status"] == "stopped"


def test_pull_records_the_time_it_finished(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    pushed(h, started="2026-01-01T00:00:00+00:00")
    h.runner.stream_data = make_tar([("a.txt", b"x")])
    assert h.main("pull", "--dest", str(tmp_path / "pulled")) == 0
    assert h.state()["pulled"]


def test_a_failed_pull_does_not_record_anything(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    pushed(h, started="2026-01-01T00:00:00+00:00")
    h.runner.stream_code = 2
    assert h.main("pull", "--dest", str(tmp_path / "pulled")) == 1
    assert "pulled" not in h.state()


def test_start_records_the_launch_and_clears_an_old_pull(
    tmp_path: Path,
) -> None:
    h = Harness(tmp_path)
    pushed(h, pulled="2026-01-01T00:00:00+00:00")
    started(h)
    state = h.state()
    assert state["started"]
    assert "pulled" not in state


def test_down_refuses_to_discard_unpulled_results(tmp_path: Path) -> None:
    h = Harness(tmp_path, [(204, None), (404, None)])
    pushed(h, started="2026-01-01T00:00:00+00:00")
    assert h.main("down") == 1
    assert OUT in h.output()
    assert "pull" in h.output()
    assert "--discard" in h.output()
    assert h.transport.calls == []
    assert h.state()["status"] == "running"


def test_down_discard_deletes_with_a_loud_warning(tmp_path: Path) -> None:
    h = Harness(tmp_path, [(204, None), (404, None)])
    pushed(h, started="2026-01-01T00:00:00+00:00")
    assert h.main("down", "--discard") == 0
    assert "WARNING" in h.output()
    assert OUT in h.output()
    assert h.state()["status"] == "deleted"


def test_down_deletes_after_a_pull(tmp_path: Path) -> None:
    h = Harness(tmp_path, [(204, None), (404, None)])
    pushed(
        h,
        started="2026-01-01T00:00:00+00:00",
        pulled="2026-01-02T00:00:00+00:00",
    )
    assert h.main("down") == 0
    assert h.state()["status"] == "deleted"


def test_down_needs_no_pull_before_any_start(tmp_path: Path) -> None:
    h = Harness(tmp_path, [(204, None), (404, None)])
    pushed(h)
    assert h.main("down") == 0


def test_a_dropped_connection_fails_the_preflight_on_up(
    tmp_path: Path,
) -> None:
    h = up_harness(tmp_path)
    h.preflight_rc = 255
    assert up(h) == 1
    assert "preflight" in h.output()
    assert "pod1" in h.output()
    assert h.transport.calls[-1][:2] == ("DELETE", f"{BASE}/pods/pod1")
    assert h.state()["status"] == "deleted"


def test_a_dropped_connection_fails_the_preflight_on_start(
    tmp_path: Path,
) -> None:
    h = Harness(tmp_path)
    h.preflight_rc = 255
    pushed(h)
    code = h.main("start", "--spec", "chain11.toml", "--max-hours", "1")
    assert code == 1
    assert "preflight" in h.output()
    assert not [r for r in h.runner.runs if r[1]]


def test_a_dropped_connection_fails_the_preflight_on_resume(
    tmp_path: Path,
) -> None:
    h = stopped_harness(
        tmp_path, [(200, None), (200, resumed_pod()), (200, None)]
    )
    h.preflight_rc = 255
    assert h.main("resume", "--max-hours", "2") == 1
    assert "preflight" in h.output()
    assert "pod1" in h.output()


def test_up_keeps_the_pod_id_when_the_port_mapping_is_malformed(
    tmp_path: Path,
) -> None:
    pod = ready_pod()
    pod["portMappings"] = {"22": "not-a-port"}
    h = up_harness(tmp_path, [(201, pod)])
    assert up(h) == 1
    assert "pod1" in h.output()
    assert "ValueError" in h.output()
    assert h.transport.calls[-1][:2] == ("DELETE", f"{BASE}/pods/pod1")
    assert h.state()["status"] == "deleted"
    assert h.state()["pod_id"] == "pod1"
    assert "deleted" in h.output()


def test_up_reports_a_create_response_without_an_id(tmp_path: Path) -> None:
    h = up_harness(tmp_path, [(201, {"unexpected": True})])
    assert up(h) == 1
    assert "no pod id" in h.output()
    assert "console" in h.output()


def test_up_survives_a_known_hosts_write_error(tmp_path: Path) -> None:
    h = up_harness(tmp_path)
    h.known_hosts().parent.mkdir(parents=True, exist_ok=True)
    h.known_hosts().mkdir()
    assert up(h) == 1
    assert "pod1" in h.output()
    assert h.transport.calls[-1][:2] == ("DELETE", f"{BASE}/pods/pod1")
    assert h.state()["status"] == "deleted"
    assert h.state()["pod_id"] == "pod1"
    assert "deleted" in h.output()


def test_resume_names_the_pod_when_the_probe_blows_up(tmp_path: Path) -> None:
    h = stopped_harness(tmp_path, [(200, None), (200, resumed_pod())])

    def explode(argv: Sequence[str]) -> cloud.CommandResult:
        raise RuntimeError("boom")

    h.runner.handler = explode
    assert h.main("resume", "--max-hours", "2") == 1
    assert "pod1" in h.output()
    assert "pull" in h.output()


def test_status_on_a_stopped_pod_reminds_about_the_volume_bill(
    tmp_path: Path,
) -> None:
    h = Harness(tmp_path, [(200, {"id": "pod1", "desiredStatus": "EXITED"})])
    pushed(h)
    assert h.main("status") == 0
    assert "bills" in h.output()
    assert "volume" in h.output()


def test_start_reminds_that_a_stopped_pod_still_bills(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    pushed(h)
    started(h)
    assert "volume" in h.output()
    assert "down" in h.output()


def test_the_live_chain_guard_checks_the_process_command_line(
    tmp_path: Path,
) -> None:
    h = Harness(tmp_path)
    pushed(h)
    started(h)
    command = next(c for c in h.commands() if "CHAIN_LIVE" in c)
    assert "/proc/$pid/cmdline" in command
    assert "chain.sh" in command


def test_up_deletes_the_pod_when_interrupted(tmp_path: Path) -> None:
    h = up_harness(tmp_path)

    def interrupt(argv: Sequence[str]) -> cloud.CommandResult:
        raise KeyboardInterrupt

    h.runner.handler = interrupt
    assert up(h) == 1
    assert h.transport.calls[-1][:2] == ("DELETE", f"{BASE}/pods/pod1")
    assert "was deleted" in h.output()
    assert h.state()["status"] == "deleted"


def test_up_reports_a_failed_delete_without_hiding_the_cause(
    tmp_path: Path,
) -> None:
    h = up_harness(
        tmp_path,
        [(201, starting_pod()), (200, ready_pod()), (500, {"error": "nope"})],
    )

    def explode(argv: Sequence[str]) -> cloud.CommandResult:
        raise RuntimeError("probe broke")

    h.runner.handler = explode
    assert up(h) == 1
    assert "probe broke" in h.output()
    assert "deleting it failed" in h.output()
    assert "500" in h.output()
    assert "down" in h.output()


def test_resume_stops_the_pod_when_a_step_blows_up(tmp_path: Path) -> None:
    h = stopped_harness(
        tmp_path, [(200, None), (200, resumed_pod()), (200, None)]
    )

    def explode(argv: Sequence[str]) -> cloud.CommandResult:
        raise RuntimeError("boom")

    h.runner.handler = explode
    assert h.main("resume", "--max-hours", "2") == 1
    assert h.transport.calls[-1][:2] == ("POST", f"{BASE}/pods/pod1/stop")
    assert "boom" in h.output()
    assert h.state()["status"] == "stopped"


def test_start_records_the_launch_before_sending_it(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    pushed(h, pulled="2026-01-01T00:00:00+00:00")
    original = h.runner.handler

    def drop_launch(argv: Sequence[str]) -> cloud.CommandResult:
        if argv[0] == "ssh" and "setsid nohup" in argv[-1]:
            return cloud.CommandResult(255, "")
        return original(argv)

    h.runner.handler = drop_launch
    code = h.main("start", "--spec", "chain11.toml", "--max-hours", "1")
    assert code == 1
    state = h.state()
    assert state["started"]
    assert "pulled" not in state


def test_pull_while_the_chain_runs_does_not_record_a_pull(
    tmp_path: Path,
) -> None:
    h = Harness(tmp_path)
    h.chain_live = True
    pushed(h, started="2026-01-01T00:00:00+00:00")
    h.runner.stream_data = make_tar([("a.txt", b"x")])
    assert h.main("pull", "--dest", str(tmp_path / "pulled")) == 0
    assert "WARNING" in h.output()
    assert "still" in h.output()
    assert "pulled" not in h.state()
