"""Manages a RunPod GPU pod for chain runs from a laptop.

Run as `python -m tomography.cloud <command>`. The commands create a pod, push
a commit and the data cache over ssh, set the environment up, start a chain
detached, report its status, pull the results and delete the pod.

The API key is read from `RUNPOD_API_KEY` only. It is sent in the
`Authorization` header to the RunPod API and nowhere else: it is never written
to the state file, printed, put on a command line or sent to the pod. The pod
stops itself through its own pod-scoped key, which RunPod sets inside the
pod. It never removes itself: stopping keeps /workspace (the volume disk) and
clears the container disk, so results survive until `pull`, and only `down`
deletes the pod. A stopped pod still bills for its volume disk. `resume`
starts it again; RunPod may attach no GPU if the host capacity changed, and
network volumes exist on Secure Cloud only, so a Community pod keeps its
data on the pod volume.

References:
    https://docs.runpod.io/api-reference/pods/POST/pods
    https://docs.runpod.io/pods/configuration/use-ssh
    https://docs.runpod.io/pods/references/environment-variables
    https://docs.runpod.io/runpodctl/manage-pods
    https://docs.runpod.io/api-reference/pods/POST/pods/podId/start
    https://docs.runpod.io/api-reference/pods/POST/pods/podId/stop
    https://docs.runpod.io/pods/manage-pods
    https://docs.runpod.io/runpodctl/reference/runpodctl-stop-pod
    https://docs.runpod.io/storage/network-volumes
"""

import argparse
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import contextlib
import dataclasses
import datetime
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import posixpath
import re
import shlex
import subprocess
import tarfile
import time
import tomllib
from typing import IO
from typing import Any
import urllib.error
import urllib.request

API_BASE = "https://rest.runpod.io/v1"
DEFAULT_GPU = "NVIDIA GeForce RTX 4090"
DEFAULT_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
DEFAULT_STATE = Path("~/.cache/tomography-cloud/pod.json")
DEFAULT_PUBLIC_KEY = Path("~/.ssh/id_ed25519.pub")
DEFAULT_CACHE = Path("~/.cache/tomography")
UV_VERSION = "0.12.17"
REMOTE_SRC = "/workspace/src"
REMOTE_CHAINS = "/workspace/chains"
REMOTE_CACHE = "/workspace/cache/tomography"
REMOTE_RUN = "/workspace/run"
REMOTE_VENVS = "/workspace/venv"
REMOTE_UV_CACHE = "/workspace/uv-cache"
MIN_DRIVER_MAJOR = 580
MAX_HOURS_LIMIT = 24
_POLL_SECONDS = 10.0
_USER_AGENT = "tomography-cloud/1.0"
_REF_NAMESPACE = "refs/tomography-cloud"
_SAFE_NAME = re.compile(r"[A-Za-z0-9._-]+")
_DRIVE = re.compile(r"[A-Za-z]:")

Transport = Callable[
    [str, str, Mapping[str, str], bytes | None], tuple[int, Any]
]

_WATCHDOG_COMMON = r"""LOG="$RUN/watchdog.log"
log() { echo "$(date -Is) $*" >> "$LOG"; }
while IFS= read -r -d '' kv; do
  case "$kv" in
    RUNPOD_POD_ID=*|RUNPOD_API_KEY=*) export "$kv" ;;
  esac
done < /proc/1/environ
stop_pod() {
  log "stopping the pod: $1"
  attempt=1
  until runpodctl stop pod "$RUNPOD_POD_ID" >> "$LOG" 2>&1 \
      || runpodctl pod stop "$RUNPOD_POD_ID" >> "$LOG" 2>&1; do
    log "stop attempt $attempt failed; retrying in 60 s"
    attempt=$((attempt + 1))
    sleep 60
  done
  log "pod stop confirmed"
  exit 0
}
"""

_WATCHDOG_LOOP = r"""chain_finished() {
  offset=$(cat "$RUN/log_offset" 2>/dev/null || echo 0)
  if [ -f "$CHAINLOG" ] \
      && tail -c +$((offset + 1)) "$CHAINLOG" | grep -q CHAIN_DONE; then
    return 0
  fi
  pid=$(cat "$RUN/chain.pid" 2>/dev/null)
  if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi
  return 1
}
start=$(date +%s)
finished_at=""
while true; do
  now=$(date +%s)
  if [ $((now - start)) -ge "$MAX_SECONDS" ]; then
    stop_pod "max runtime reached"
  fi
  if [ -z "$finished_at" ] && chain_finished; then
    finished_at=$now
    log "chain finished; grace period started"
  fi
  waited=$((now - ${finished_at:-$now}))
  if [ -n "$finished_at" ] && [ "$waited" -ge "$GRACE_SECONDS" ]; then
    stop_pod "grace period after the chain ended"
  fi
  sleep 30
done
"""

_PREFLIGHT = (
    "for v in RUNPOD_API_KEY RUNPOD_POD_ID; do "
    "tr '\\0' '\\n' </proc/1/environ | grep -q \"^$v=\" "
    '|| echo "missing $v"; done; '
    "command -v runpodctl >/dev/null 2>&1 || echo 'missing runpodctl'; true"
)


class CloudError(Exception):
    """A failure the command line reports to the user without a traceback."""


@dataclasses.dataclass(frozen=True)
class CommandResult:
    """The outcome of a local command.

    Attributes:
        returncode: Process exit code.
        stdout: Decoded standard output.
    """

    returncode: int
    stdout: str


def _urllib_transport(
    method: str, url: str, headers: Mapping[str, str], body: bytes | None
) -> tuple[int, Any]:
    request = urllib.request.Request(
        url, data=body, method=method, headers=dict(headers)
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            status = response.status
            raw = response.read()
    except urllib.error.HTTPError as error:
        status = error.code
        raw = error.read()
    if not raw:
        return status, None
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, raw.decode("utf-8", "replace")


class SubprocessRunner:
    """Runs local commands with `subprocess`."""

    def run(
        self, argv: Sequence[str], input_bytes: bytes | None = None
    ) -> CommandResult:
        """Runs a command and captures its standard output.

        Args:
            argv: The command and its arguments.
            input_bytes: Optional bytes written to standard input.

        Returns:
            The exit code and the decoded standard output.

        Raises:
            CloudError: When the program is not installed.
        """
        try:
            done = subprocess.run(
                list(argv),
                input=input_bytes,
                stdout=subprocess.PIPE,
                check=False,
            )
        except FileNotFoundError as error:
            raise CloudError(f"command not found: {argv[0]}") from error
        return CommandResult(
            done.returncode, done.stdout.decode("utf-8", "replace")
        )

    def pipe(self, source: Sequence[str], sink: Sequence[str]) -> int:
        """Streams the standard output of one command into another.

        Args:
            source: The producing command.
            sink: The consuming command.

        Returns:
            0 when both commands succeed, else the first non-zero exit code.

        Raises:
            CloudError: When a program is not installed.
        """
        try:
            producer = subprocess.Popen(list(source), stdout=subprocess.PIPE)
            consumer = subprocess.Popen(list(sink), stdin=producer.stdout)
        except FileNotFoundError as error:
            raise CloudError(f"command not found: {error.filename}") from error
        if producer.stdout is not None:
            producer.stdout.close()
        consumer_code = consumer.wait()
        producer_code = producer.wait()
        return producer_code or consumer_code

    def stream(
        self, argv: Sequence[str], consumer: Callable[[IO[bytes]], None]
    ) -> int:
        """Hands the standard output of a command to a Python consumer.

        Args:
            argv: The command and its arguments.
            consumer: Reads the output stream. When it raises, the command
                is killed and the exception propagates.

        Returns:
            The exit code of the command.

        Raises:
            CloudError: When the program is not installed.
        """
        try:
            process = subprocess.Popen(list(argv), stdout=subprocess.PIPE)
        except FileNotFoundError as error:
            raise CloudError(f"command not found: {argv[0]}") from error
        assert process.stdout is not None
        try:
            consumer(process.stdout)
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            process.stdout.close()
        return process.wait()


@dataclasses.dataclass
class Deps:
    """The side effects of the module, replaceable in tests.

    Attributes:
        transport: Sends one HTTP request and returns status and JSON.
        runner: Runs local commands.
        clock: Returns a monotonic time in seconds, for timeouts.
        wall: Returns the epoch time in seconds, for stored deadlines.
        sleep: Waits for a number of seconds.
        env: The environment variables the module may read.
        out: Prints one line to the user.
    """

    transport: Transport = _urllib_transport
    runner: Any = dataclasses.field(default_factory=SubprocessRunner)
    clock: Callable[[], float] = time.monotonic
    wall: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep
    env: Mapping[str, str] = dataclasses.field(
        default_factory=lambda: dict(os.environ)
    )
    out: Callable[[str], None] = print


class Api:
    """A thin client for the RunPod REST API."""

    def __init__(self, deps: Deps) -> None:
        """Reads the API key from the environment.

        Args:
            deps: Side effects.

        Raises:
            CloudError: When `RUNPOD_API_KEY` is not set.
        """
        key = deps.env.get("RUNPOD_API_KEY")
        if not key:
            raise CloudError(
                "RUNPOD_API_KEY is not set; export it in the environment"
            )
        self._key = key
        self._transport = deps.transport

    def request(
        self, method: str, path: str, payload: Mapping[str, Any] | None = None
    ) -> tuple[int, Any]:
        """Sends one request and returns the status and decoded body.

        Args:
            method: HTTP method.
            path: Path below the API base, starting with a slash.
            payload: Optional JSON body.

        Returns:
            The HTTP status and the decoded JSON body (None when empty).

        Raises:
            CloudError: When the request times out or cannot be sent.
        """
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": _USER_AGENT,
        }
        body = None if payload is None else json.dumps(payload).encode()
        try:
            return self._transport(method, API_BASE + path, headers, body)
        except TimeoutError as error:
            raise CloudError(f"{method} {path} timed out") from error
        except OSError as error:
            text = str(error).replace(self._key, "[redacted]")
            raise CloudError(f"{method} {path} failed: {text}") from error

    def check(self, status: int, body: Any, action: str) -> Any:
        """Returns the body of a successful response or raises.

        Args:
            status: HTTP status.
            body: Decoded response body.
            action: What was attempted, for the message.

        Returns:
            The body when the status is 2xx.

        Raises:
            CloudError: On any other status, with the API message and
                without the key.
        """
        if 200 <= status < 300:
            return body
        message = body
        if isinstance(body, Mapping):
            message = body.get("error") or body.get("message") or body
        text = str(message).replace(self._key, "[redacted]")
        raise CloudError(f"{action} failed: HTTP {status}: {text}")


def _now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise CloudError(f"no state file at {path}; run `up` first")
    try:
        loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CloudError(
            f"state file {path} is not valid JSON ({error}); fix or delete it"
        ) from error
    return loaded


def _save_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _known_hosts(state_path: Path, pod_id: str) -> Path:
    return state_path.parent / f"known_hosts_{pod_id}"


def _pod_endpoint(pod: Mapping[str, Any]) -> tuple[str, int] | None:
    ip = pod.get("publicIp")
    port = (pod.get("portMappings") or {}).get("22")
    if pod.get("desiredStatus") == "RUNNING" and ip and port:
        return str(ip), int(port)
    return None


def _ssh_argv(
    args: argparse.Namespace,
    state: Mapping[str, Any],
    no_stdin: bool = False,
) -> list[str]:
    known_hosts = _known_hosts(args.state, state["pod_id"]).as_posix()
    # Quotes only where ssh needs them: the Windows argv escaping turns
    # quotes in an argument without spaces into \" and ssh rejects that.
    if any(c.isspace() for c in known_hosts):
        known_hosts = f'"{known_hosts}"'
    return [
        "ssh",
        *(["-n"] if no_stdin else []),
        "-p",
        str(state["ssh_port"]),
        "-i",
        str(state["identity"]),
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ConnectTimeout=15",
        f"root@{state['ip']}",
    ]


def _require_endpoint(state: Mapping[str, Any]) -> None:
    if not state.get("ip") or not state.get("ssh_port"):
        raise CloudError("the pod has no ssh endpoint yet; run `up` first")


def _require_source(state: Mapping[str, Any]) -> str:
    src = state.get("remote_src")
    if not src:
        raise CloudError("no source on the pod; run `push` first")
    return str(src)


def _remote(
    args: argparse.Namespace,
    deps: Deps,
    state: Mapping[str, Any],
    command: str,
    input_bytes: bytes | None = None,
) -> CommandResult:
    _require_endpoint(state)
    argv = [*_ssh_argv(args, state, input_bytes is None), command]
    result: CommandResult = deps.runner.run(argv, input_bytes)
    return result


def _stream(deps: Deps, source: Sequence[str], sink: Sequence[str]) -> None:
    code = deps.runner.pipe(source, sink)
    if code:
        raise CloudError(f"{source[0]} pipeline failed with exit code {code}")


def _preflight(
    args: argparse.Namespace, deps: Deps, state: Mapping[str, Any]
) -> list[str]:
    result = _remote(args, deps, state, _PREFLIGHT)
    if result.returncode:
        raise CloudError(
            f"the preflight over ssh failed (exit {result.returncode}); "
            "is the connection up?"
        )
    return [
        line[len("missing ") :].strip()
        for line in result.stdout.splitlines()
        if line.startswith("missing ")
    ]


def _warn_no_self_removal(deps: Deps, missing: Sequence[str]) -> None:
    deps.out(
        "WARNING: the pod cannot stop itself (missing: "
        f"{', '.join(missing)}); it bills until you run `down`"
    )


def _script(header: Sequence[str], tail: str) -> str:
    return "\n".join(["#!/bin/bash", *header]) + "\n" + tail


def _deadline_script(max_seconds: int) -> str:
    return _script(
        [f"MAX_SECONDS={max_seconds}", f"RUN={REMOTE_RUN}"],
        _WATCHDOG_COMMON
        + 'sleep "$MAX_SECONDS"\nstop_pod "deadline reached"\n',
    )


def _watchdog_script(
    chain_log: str, max_seconds: int, grace_minutes: float
) -> str:
    return _script(
        [
            f"MAX_SECONDS={max_seconds}",
            f"GRACE_SECONDS={int(grace_minutes * 60)}",
            f"CHAINLOG={shlex.quote(chain_log)}",
            f"RUN={REMOTE_RUN}",
        ],
        _WATCHDOG_COMMON + _WATCHDOG_LOOP,
    )


def _launch_command(names: Sequence[str]) -> str:
    # The braces keep each `&` inside the group: `a && b &` would background
    # the whole list, and its subshell would hold the ssh channel open until
    # the chain ends.
    jobs = " ".join(
        f"setsid nohup bash {REMOTE_RUN}/{name} "
        f"> {REMOTE_RUN}/{name}.out 2>&1 < /dev/null &"
        for name in names
    )
    return f"{{ {jobs} }}"


def _upload_script(
    args: argparse.Namespace,
    deps: Deps,
    state: Mapping[str, Any],
    name: str,
    text: str,
) -> None:
    upload = _remote(
        args,
        deps,
        state,
        f"mkdir -p {REMOTE_RUN} && cat > {REMOTE_RUN}/{name}",
        text.encode(),
    )
    if upload.returncode:
        raise CloudError(f"uploading {name} failed")


def _guard_existing_state(args: argparse.Namespace, deps: Deps) -> None:
    if not args.state.exists():
        return
    try:
        old = _load_state(args.state)
    except CloudError:
        if args.force:
            return
        raise
    pod_id = old.get("pod_id")
    if not pod_id or old.get("status") == "deleted":
        return
    if not args.force:
        raise CloudError(
            f"the state file tracks pod {pod_id} ({old.get('status')}); "
            "run `down` first, or pass --force to replace it and leave "
            "that pod running"
        )
    deps.out(f"WARNING: replacing the state of pod {pod_id}, still running")


def _wait_for_ssh(
    args: argparse.Namespace, deps: Deps, state: Mapping[str, Any]
) -> bool:
    deadline = deps.clock() + args.ssh_timeout_minutes * 60
    probe = [*_ssh_argv(args, state, no_stdin=True), "true"]
    while deps.runner.run(probe).returncode:
        if deps.clock() >= deadline:
            return False
        deps.sleep(_POLL_SECONDS)
    return True


def _error_detail(error: BaseException, deps: Deps) -> str:
    if isinstance(error, CloudError):
        detail = str(error)
    elif isinstance(error, KeyboardInterrupt):
        detail = "interrupted"
    else:
        detail = f"{type(error).__name__}: {error}"
    key = deps.env.get("RUNPOD_API_KEY")
    return detail.replace(key, "[redacted]") if key else detail


def _stop_pod(
    args: argparse.Namespace, deps: Deps, api: Api, state: dict[str, Any]
) -> None:
    status, body = api.request("POST", f"/pods/{state['pod_id']}/stop")
    api.check(status, body, "stop pod")
    state["status"] = "stopped"
    _save_state(args.state, state)


def _reach_pod(
    args: argparse.Namespace,
    deps: Deps,
    api: Api,
    state: dict[str, Any],
    pod: Mapping[str, Any],
    next_step: str,
) -> tuple[str, int] | None:
    pod_id = state["pod_id"]
    deadline = deps.clock() + args.timeout_minutes * 60
    while (endpoint := _pod_endpoint(pod)) is None:
        if deps.clock() >= deadline:
            deps.out(
                f"pod {pod_id} did not expose ssh within "
                f"{args.timeout_minutes:g} minutes; {next_step}"
            )
            return None
        deps.sleep(_POLL_SECONDS)
        status, got = api.request("GET", f"/pods/{pod_id}")
        pod = api.check(status, got, "get pod")
    state.update(ip=endpoint[0], ssh_port=endpoint[1], status="running")
    _save_state(args.state, state)
    known_hosts = _known_hosts(args.state, pod_id)
    known_hosts.unlink(missing_ok=True)
    known_hosts.touch()
    if not _wait_for_ssh(args, deps, state):
        deps.out(
            f"ssh to pod {pod_id} did not answer within "
            f"{args.ssh_timeout_minutes:g} minutes; {next_step}"
        )
        return None
    return endpoint


def _arm_deadline(
    args: argparse.Namespace,
    deps: Deps,
    state: Mapping[str, Any],
    missing: Sequence[str],
) -> None:
    if missing:
        _warn_no_self_removal(deps, missing)
        return
    script = _deadline_script(int(args.max_hours * 3600))
    _upload_script(args, deps, state, "deadline.sh", script)
    launch = _launch_command(["deadline.sh"])
    if _remote(args, deps, state, launch).returncode:
        raise CloudError("launching the deadline watchdog failed")


def _delete_pod(
    args: argparse.Namespace, deps: Deps, api: Api, state: dict[str, Any]
) -> None:
    pod_id = state["pod_id"]
    status, body = api.request("DELETE", f"/pods/{pod_id}")
    if status != 404:
        api.check(status, body, "delete pod")
    state["status"] = "deleted"
    _save_state(args.state, state)
    with contextlib.suppress(OSError):
        _known_hosts(args.state, pod_id).unlink(missing_ok=True)


def _fail_pod(
    error: BaseException,
    args: argparse.Namespace,
    deps: Deps,
    api: Api,
    state: dict[str, Any],
    delete: bool,
    hint: str = "",
) -> CloudError:
    pod_id = state["pod_id"]
    detail = _error_detail(error, deps)
    done, doing = ("deleted", "deleting") if delete else ("stopped", "stopping")
    try:
        (_delete_pod if delete else _stop_pod)(args, deps, api, state)
    except Exception as release_error:
        return CloudError(
            f"{detail}; pod {pod_id} may still be running and {doing} it "
            f"failed ({_error_detail(release_error, deps)}), "
            f"{'run `down`' if delete else hint}"
        )
    return CloudError(
        f"{detail}; pod {pod_id} was {done}" + (f", {hint}" if hint else "")
    )


def _bring_up(
    args: argparse.Namespace,
    deps: Deps,
    api: Api,
    state: dict[str, Any],
    pod: Mapping[str, Any],
) -> int:
    pod_id = state["pod_id"]
    endpoint = _reach_pod(args, deps, api, state, pod, "deleting it")
    if endpoint is None:
        _delete_pod(args, deps, api, state)
        deps.out(f"deleted pod {pod_id}")
        return 1
    missing = _preflight(args, deps, state)
    if missing and not args.no_self_removal:
        _delete_pod(args, deps, api, state)
        deps.out(
            f"the pod cannot stop itself (missing: {', '.join(missing)}); "
            f"deleted pod {pod_id}. Use --no-self-removal to keep such a pod"
        )
        return 1
    _arm_deadline(args, deps, state, missing)
    deps.out(f"pod {pod_id} ready at {endpoint[0]}:{endpoint[1]}")
    return 0


def cmd_up(args: argparse.Namespace, deps: Deps) -> int:
    """Creates a pod and waits until ssh works and it can stop itself.

    Args:
        args: Parsed command line.
        deps: Side effects.

    Returns:
        0 when the pod is ready, 1 on a timeout or a refused pod.

    Raises:
        CloudError: On an API failure.
    """
    api = Api(deps)
    _guard_existing_state(args, deps)
    public_key = args.public_key.expanduser().read_text(encoding="utf-8")
    body: dict[str, Any] = {
        "name": args.name,
        "imageName": args.image,
        "gpuTypeIds": [args.gpu],
        "gpuCount": 1,
        "cloudType": args.cloud.upper(),
        "containerDiskInGb": args.disk_gb,
        "volumeMountPath": "/workspace",
        "ports": ["22/tcp"],
        "allowedCudaVersions": ["13.0"],
        "interruptible": False,
        "env": {"PUBLIC_KEY": public_key.strip()},
    }
    if args.volume_id:
        body["networkVolumeId"] = args.volume_id
    else:
        body["volumeInGb"] = args.volume_gb
    if args.cloud == "community":
        body["supportPublicIp"] = True
    try:
        status, created = api.request("POST", "/pods", body)
    except CloudError as error:
        raise CloudError(
            f"{error}; the pod may have been created, check the RunPod "
            "console and delete it there"
        ) from error
    pod = api.check(status, created, "create pod")
    if not isinstance(pod, Mapping) or not pod.get("id"):
        raise CloudError(
            "the create response had no pod id; check the RunPod console "
            "and delete any new pod there"
        )
    pod_id = str(pod["id"])
    identity = args.public_key.expanduser().with_suffix("")
    state: dict[str, Any] = {
        "pod_id": pod_id,
        "name": args.name,
        "gpu": args.gpu,
        "cloud": body["cloudType"],
        "created": _now_iso(),
        "cost_per_hr": pod.get("costPerHr"),
        "identity": str(identity),
        "deadline": deps.wall() + args.max_hours * 3600,
        "status": "starting",
    }
    _save_state(args.state, state)
    deps.out(f"created pod {pod_id}; waiting for ssh")
    try:
        return _bring_up(args, deps, api, state, pod)
    except (Exception, KeyboardInterrupt) as error:
        raise _fail_pod(error, args, deps, api, state, delete=True) from error


def cmd_ssh(args: argparse.Namespace, deps: Deps) -> int:
    """Prints the ssh command, or runs a command on the pod.

    Args:
        args: Parsed command line.
        deps: Side effects.

    Returns:
        The exit code of the remote command, or 0 when printing.
    """
    state = _load_state(args.state)
    _require_endpoint(state)
    if not args.remote:
        deps.out(shlex.join(_ssh_argv(args, state)))
        return 0
    result = _remote(args, deps, state, shlex.join(args.remote))
    if result.stdout:
        deps.out(result.stdout.rstrip("\n"))
    return result.returncode


def _read_spec(path: Path) -> tuple[str, str]:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise CloudError(
            f"cannot read the chain spec {path}: {error}"
        ) from error
    chain = data.get("chain")
    name = chain.get("name") if isinstance(chain, dict) else None
    out = chain.get("out") if isinstance(chain, dict) else None
    if not isinstance(name, str) or not _SAFE_NAME.fullmatch(name):
        raise CloudError(
            "the spec needs [chain] name made of letters, digits, . _ -"
        )
    if not isinstance(out, str) or not out:
        raise CloudError("the spec needs a [chain] out path")
    if "\\" in out or _DRIVE.match(out):
        raise CloudError(
            f"[chain] out {out!r} is a Windows path; use a POSIX path"
        )
    root = posixpath.normpath(posixpath.join(REMOTE_CHAINS, out))
    if not PurePosixPath(root).is_relative_to("/workspace"):
        raise CloudError(f"[chain] out resolves to {root}, outside /workspace")
    return name, root


def cmd_push(args: argparse.Namespace, deps: Deps) -> int:
    """Uploads a commit with its history, a chain spec and the data cache.

    Args:
        args: Parsed command line.
        deps: Side effects.

    Returns:
        0 on success.

    Raises:
        CloudError: When a ref, a spec or a transfer is invalid or fails.
    """
    state = _load_state(args.state)
    _require_endpoint(state)
    if args.ref.startswith("-"):
        raise CloudError(f"the ref {args.ref!r} must not start with '-'")
    chain_name, out_root = _read_spec(args.spec)
    resolved = deps.runner.run(
        ["git", "rev-parse", "--verify", f"{args.ref}^{{commit}}"]
    )
    full = resolved.stdout.strip()
    if resolved.returncode or not full:
        raise CloudError(f"cannot resolve git ref {args.ref}")
    short = full[:12]
    src = f"{REMOTE_SRC}/{short}"
    bundle = f"{REMOTE_SRC}/{short}.bundle"
    ref = f"{_REF_NAMESPACE}/{short}"
    deps.runner.run(["git", "update-ref", ref, full])
    try:
        _stream(
            deps,
            ["git", "bundle", "create", "-", ref],
            [
                *_ssh_argv(args, state),
                f"mkdir -p {REMOTE_SRC} && cat > {shlex.quote(bundle)}",
            ],
        )
    finally:
        deps.runner.run(["git", "update-ref", "-d", ref])
    quoted_src = shlex.quote(src)
    quoted_bundle = shlex.quote(bundle)
    unpacked = _remote(
        args,
        deps,
        state,
        f"([ -d {quoted_src}/.git ] || git init -q {quoted_src}) && "
        f"cd {quoted_src} && git fetch -q {quoted_bundle} "
        f"'+{_REF_NAMESPACE}/*:{_REF_NAMESPACE}/*' && "
        f"git checkout -q --detach {full} && rm -f {quoted_bundle}",
    )
    if unpacked.returncode:
        raise CloudError(
            f"unpacking the commit on the pod failed (exit "
            f"{unpacked.returncode}); is git installed there?"
        )
    remote_spec = f"{REMOTE_CHAINS}/{args.spec.name}"
    copied = _remote(
        args,
        deps,
        state,
        f"mkdir -p {REMOTE_CHAINS} && cat > {shlex.quote(remote_spec)}",
        args.spec.read_bytes(),
    )
    if copied.returncode:
        raise CloudError(f"copying the spec failed: exit {copied.returncode}")
    cache_dir = shlex.quote(REMOTE_CACHE)
    present = _remote(args, deps, state, f"test -d {cache_dir}/hub")
    if args.sync_cache or present.returncode:
        _stream(
            deps,
            ["tar", "-c", "-C", str(args.cache.expanduser()), "."],
            [
                *_ssh_argv(args, state),
                f"mkdir -p {cache_dir} && tar -x -C {cache_dir}",
            ],
        )
        deps.out("data cache uploaded")
    state.update(
        remote_src=src,
        remote_spec=remote_spec,
        remote_cache=REMOTE_CACHE,
        remote_out=out_root,
        remote_chain_log=f"{out_root}/_chain/{chain_name}.log",
    )
    _save_state(args.state, state)
    deps.out(f"pushed {short} to {src}; chain output goes to {out_root}")
    return 0


def _uv_env(src: str) -> list[str]:
    venv = f"{REMOTE_VENVS}/{Path(src).name}"
    return [
        'export PATH="$HOME/.local/bin:$PATH"',
        f"export UV_CACHE_DIR={REMOTE_UV_CACHE}",
        f"export UV_PROJECT_ENVIRONMENT={venv}",
    ]


def cmd_setup(args: argparse.Namespace, deps: Deps) -> int:
    """Checks the driver and git, installs uv, syncs, checks CUDA.

    Args:
        args: Parsed command line.
        deps: Side effects.

    Returns:
        0 on success.

    Raises:
        CloudError: When the driver is too old or a step fails.
    """
    state = _load_state(args.state)
    src = _require_source(state)
    smi = _remote(
        args,
        deps,
        state,
        "nvidia-smi --query-gpu=driver_version,name,memory.total "
        "--format=csv,noheader",
    )
    if smi.returncode or not smi.stdout.strip():
        raise CloudError("nvidia-smi failed on the pod")
    first = smi.stdout.strip().splitlines()[0]
    driver = first.split(",")[0].strip()
    deps.out(f"gpu: {first}")
    if int(driver.split(".")[0]) < MIN_DRIVER_MAJOR:
        raise CloudError(
            f"driver {driver} is older than {MIN_DRIVER_MAJOR}, which the "
            "cu130 wheels need; run `down` and `up` again to get another host"
        )
    if _remote(args, deps, state, "command -v git >/dev/null 2>&1").returncode:
        raise CloudError("git is missing on the pod")
    env = " && ".join(_uv_env(src))
    sync = (
        f"{env} && (command -v uv >/dev/null 2>&1 || "
        f"curl -LsSf https://astral.sh/uv/{UV_VERSION}/install.sh | sh) && "
        f"cd {shlex.quote(src)} && uv sync --locked"
    )
    if _remote(args, deps, state, sync).returncode:
        raise CloudError("uv sync failed on the pod")
    check = (
        f"{env} && cd {shlex.quote(src)} && uv run python -c "
        "'import torch; assert torch.cuda.is_available()'"
    )
    if _remote(args, deps, state, check).returncode:
        raise CloudError("torch cannot use CUDA on the pod")
    deps.out("setup complete")
    return 0


def _chain_script(src: str, spec: str) -> str:
    lines = [
        "#!/bin/bash",
        *_uv_env(src),
        "export HF_HOME=/workspace/hf",
        f"export TOMOGRAPHY_CACHE={REMOTE_CACHE}",
        f"cd {shlex.quote(src)}",
        f"echo $$ > {REMOTE_RUN}/chain.pid",
        f"uv run python -m tomography chain run {shlex.quote(spec)}",
        f'echo "CHAIN_EXIT $?" >> {REMOTE_RUN}/chain.exit',
    ]
    return "\n".join(lines) + "\n"


def _chain_live(
    args: argparse.Namespace, deps: Deps, state: Mapping[str, Any]
) -> bool:
    pid_file = f"{REMOTE_RUN}/chain.pid"
    live = _remote(
        args,
        deps,
        state,
        f"pid=$(cat {pid_file} 2>/dev/null); "
        'if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null '
        "&& tr '\\0' ' ' </proc/$pid/cmdline | grep -q chain.sh; "
        "then echo CHAIN_LIVE; fi",
    )
    return "CHAIN_LIVE" in live.stdout


def cmd_start(args: argparse.Namespace, deps: Deps) -> int:
    """Starts the chain and its watchdog detached on the pod.

    Args:
        args: Parsed command line.
        deps: Side effects.

    Returns:
        0 when both processes were launched.

    Raises:
        CloudError: When a check fails or the launch fails.
    """
    state = _load_state(args.state)
    src = _require_source(state)
    spec = f"{REMOTE_CHAINS}/{args.spec}"
    chain_log = state.get("remote_chain_log")
    if state.get("remote_spec") != spec or not chain_log:
        raise CloudError(f"spec {args.spec} was not pushed; run `push` first")
    max_seconds = int(args.max_hours * 3600)
    if state.get("deadline") is not None:
        remaining = state["deadline"] - deps.wall()
        if remaining <= 0:
            raise CloudError("the deadline set by `up` has passed; run `down`")
        max_seconds = int(min(max_seconds, remaining))
    pid_file = f"{REMOTE_RUN}/chain.pid"
    if _chain_live(args, deps, state):
        raise CloudError(
            f"a chain is already running on the pod (see {pid_file})"
        )
    missing = _preflight(args, deps, state)
    if missing and not args.no_self_removal:
        raise CloudError(
            f"the pod cannot stop itself (missing: {', '.join(missing)}); "
            "fix that, or pass --no-self-removal to run anyway"
        )
    if missing:
        _warn_no_self_removal(deps, missing)
    scripts = {
        "chain.sh": _chain_script(src, spec),
        "watchdog.sh": _watchdog_script(
            chain_log, max_seconds, args.grace_minutes
        ),
    }
    for name, text in scripts.items():
        _upload_script(args, deps, state, name, text)
    offset = f"{REMOTE_RUN}/log_offset"
    prepare = (
        f"rm -f {pid_file} {REMOTE_RUN}/chain.exit && "
        f"(stat -c %s {shlex.quote(chain_log)} 2>/dev/null || echo 0) "
        f"> {offset} && touch {REMOTE_RUN}/started"
    )
    state["started"] = _now_iso()
    state.pop("pulled", None)
    _save_state(args.state, state)
    if _remote(args, deps, state, prepare).returncode:
        raise CloudError("preparing the chain launch failed")
    launch = _launch_command(list(scripts))
    if _remote(args, deps, state, launch).returncode:
        raise CloudError("launching the chain failed")
    deps.out(
        "chain started; the watchdog stops the pod after "
        f"{max_seconds / 3600:g} h or {args.grace_minutes:g} min after the "
        "end. "
        "A stopped pod still bills its volume disk until `pull` and `down`"
    )
    return 0


def _report_if_not_running(deps: Deps, state: Mapping[str, Any]) -> bool:
    if not deps.env.get("RUNPOD_API_KEY"):
        return False
    api = Api(deps)
    pod_id = state["pod_id"]
    status, body = api.request("GET", f"/pods/{pod_id}")
    if status == 404:
        deps.out(f"pod {pod_id} no longer exists; nothing to read")
        return True
    if (
        status == 200
        and isinstance(body, Mapping)
        and body.get("desiredStatus") not in (None, "RUNNING")
    ):
        deps.out(
            f"pod {pod_id} is stopped ({body.get('desiredStatus')}); "
            "run `resume` to read its log or pull the results. A stopped "
            "pod still bills its volume disk until you `pull` and `down`"
        )
        return True
    return False


def cmd_status(args: argparse.Namespace, deps: Deps) -> int:
    """Prints the tail of the chain log and whether the chain is alive.

    Args:
        args: Parsed command line.
        deps: Side effects.

    Returns:
        0 when the remote command ran.

    Raises:
        CloudError: When no chain log is recorded.
    """
    state = _load_state(args.state)
    chain_log = state.get("remote_chain_log")
    if not chain_log:
        raise CloudError("no chain log is recorded; run `push` first")
    if _report_if_not_running(deps, state):
        return 0
    log = shlex.quote(chain_log)
    pid_file = f"{REMOTE_RUN}/chain.pid"
    exit_file = f"{REMOTE_RUN}/chain.exit"
    command = (
        f"if [ -f {log} ]; then tail -n 20 {log}; "
        "else echo 'no chain log yet'; fi; "
        f"if [ -f {exit_file} ]; then cat {exit_file}; fi; "
        f"if [ -f {pid_file} ] && kill -0 $(cat {pid_file}) 2>/dev/null; "
        "then echo CHAIN_ALIVE; else echo CHAIN_NOT_RUNNING; fi"
    )
    result = _remote(args, deps, state, command)
    deps.out(result.stdout.rstrip("\n"))
    return result.returncode


def _check_member(member: tarfile.TarInfo) -> None:
    name = member.name.replace("\\", "/")
    parts = PurePosixPath(name).parts
    if name.startswith("/") or ".." in parts or _DRIVE.match(name):
        raise CloudError(f"unsafe path in the archive: {member.name}")


def _extract(
    stream: IO[bytes],
    dest: Path,
    errors: list[Exception],
    extracted: list[str],
) -> None:
    try:
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            for member in tar:
                _check_member(member)
                try:
                    tar.extract(member, path=dest, filter="data")
                    extracted.append(member.name)
                except tarfile.FilterError as error:
                    raise CloudError(
                        f"unsafe member in the archive: {error}"
                    ) from error
    except tarfile.TarError as error:
        errors.append(error)


def _with_partial_note(
    message: object, dest: Path, extracted: list[str]
) -> str:
    if not extracted:
        return str(message)
    return (
        f"{message}; {dest} holds a partial extraction "
        f"({len(extracted)} members), not a full pull"
    )


def cmd_pull(args: argparse.Namespace, deps: Deps) -> int:
    """Streams the remote out root back into a new or empty local folder.

    Args:
        args: Parsed command line.
        deps: Side effects.

    Returns:
        0 on success.

    Raises:
        CloudError: When the folder is not empty, the archive holds an
            unsafe member or the remote command fails.
    """
    state = _load_state(args.state)
    _require_endpoint(state)
    remote = args.path or state.get("remote_out")
    if not remote:
        raise CloudError("no out root is recorded; run `push` or pass --path")
    if args.dest.exists() and any(args.dest.iterdir()):
        raise CloudError(f"destination {args.dest} is not empty")
    args.dest.mkdir(parents=True, exist_ok=True)
    errors: list[Exception] = []
    extracted: list[str] = []
    try:
        code = _pull_stream(args, deps, state, remote, errors, extracted)
    except CloudError as error:
        raise CloudError(
            _with_partial_note(error, args.dest, extracted)
        ) from error
    if code:
        raise CloudError(
            _with_partial_note(
                f"tar over ssh failed with exit code {code}",
                args.dest,
                extracted,
            )
        )
    if errors:
        raise CloudError(
            _with_partial_note(
                f"the archive is not valid: {errors[0]}", args.dest, extracted
            )
        )
    if _chain_live(args, deps, state):
        deps.out(
            "WARNING: the chain is still writing results, so this pull is "
            "not recorded; pull again when it has finished"
        )
    else:
        state["pulled"] = _now_iso()
        _save_state(args.state, state)
    deps.out(f"pulled {remote} into {args.dest}")
    return 0


def _pull_stream(
    args: argparse.Namespace,
    deps: Deps,
    state: Mapping[str, Any],
    remote: str,
    errors: list[Exception],
    extracted: list[str],
) -> int:
    code: int = deps.runner.stream(
        [
            *_ssh_argv(args, state, no_stdin=True),
            f"tar -c -C {shlex.quote(remote)} .",
        ],
        lambda stream: _extract(stream, args.dest, errors, extracted),
    )
    return code


def cmd_resume(args: argparse.Namespace, deps: Deps) -> int:
    """Starts a stopped pod again and waits until ssh works.

    Args:
        args: Parsed command line.
        deps: Side effects.

    Returns:
        0 when the pod is ready, 1 on a timeout, a missing GPU or a pod
        that cannot stop itself (it is stopped again in the last two cases).

    Raises:
        CloudError: On an API failure or a deleted pod.
    """
    state = _load_state(args.state)
    if state.get("status") == "deleted":
        raise CloudError("the pod was deleted; there is nothing to resume")
    api = Api(deps)
    pod_id = state["pod_id"]
    status, body = api.request("POST", f"/pods/{pod_id}/start")
    api.check(status, body, "start pod")
    state["deadline"] = deps.wall() + args.max_hours * 3600
    state["status"] = "starting"
    _save_state(args.state, state)
    try:
        endpoint = _reach_pod(
            args,
            deps,
            api,
            state,
            {},
            "stopping it again; run `resume` to retry, then `pull` "
            "before `down`",
        )
        if endpoint is None:
            _stop_pod(args, deps, api, state)
            return 1
        gpu = _remote(
            args,
            deps,
            state,
            "nvidia-smi --query-gpu=name --format=csv,noheader",
        )
        if gpu.returncode or not gpu.stdout.strip():
            _stop_pod(args, deps, api, state)
            deps.out(
                f"pod {pod_id} came back without a GPU (the host capacity "
                "changed); stopped it again, try `resume` later"
            )
            return 1
        missing = _preflight(args, deps, state)
        if missing and not args.no_self_removal:
            _stop_pod(args, deps, api, state)
            deps.out(
                f"the pod cannot stop itself (missing: {', '.join(missing)}); "
                f"stopped pod {pod_id}. Use --no-self-removal to run anyway"
            )
            return 1
        _arm_deadline(args, deps, state, missing)
    except (Exception, KeyboardInterrupt) as error:
        raise _fail_pod(
            error,
            args,
            deps,
            api,
            state,
            delete=False,
            hint="run `resume` to retry, then `pull` before `down`",
        ) from error
    deps.out(
        f"pod {pod_id} resumed at {endpoint[0]}:{endpoint[1]}; the container "
        "disk was cleared, so run `setup` again before `start`"
    )
    return 0


def cmd_down(args: argparse.Namespace, deps: Deps) -> int:
    """Deletes the pod and waits until the API no longer lists it.

    Args:
        args: Parsed command line.
        deps: Side effects.

    Returns:
        0 when the pod is gone, 1 when it is still there at the timeout.

    Raises:
        CloudError: On an API failure.
    """
    state = _load_state(args.state)
    api = Api(deps)
    pod_id = state["pod_id"]
    if state.get("started") and not state.get("pulled"):
        lost = f"results under {state.get('remote_out')} on pod {pod_id}"
        if not args.discard:
            raise CloudError(
                f"the {lost} were not pulled; run `resume` if the pod is "
                "stopped, then `pull`, or pass --discard to delete them"
            )
        deps.out(f"WARNING: discarding the {lost}")
    status, body = api.request("DELETE", f"/pods/{pod_id}")
    if status == 404:
        deps.out(f"pod {pod_id} was already gone")
    else:
        api.check(status, body, "delete pod")
        deadline = deps.clock() + args.timeout_minutes * 60
        while True:
            status, body = api.request("GET", f"/pods/{pod_id}")
            if status == 404 or (
                isinstance(body, Mapping)
                and body.get("desiredStatus") == "TERMINATED"
            ):
                break
            api.check(status, body, "get pod")
            if deps.clock() >= deadline:
                deps.out(f"pod {pod_id} still exists after the delete")
                return 1
            deps.sleep(_POLL_SECONDS)
        deps.out(f"pod {pod_id} deleted")
    state["status"] = "deleted"
    _save_state(args.state, state)
    _known_hosts(args.state, pod_id).unlink(missing_ok=True)
    return 0


def _max_hours(text: str) -> float:
    try:
        hours = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"not a number: {text}") from error
    if not 0 < hours <= MAX_HOURS_LIMIT:
        raise argparse.ArgumentTypeError(
            f"must be above 0 and at most {MAX_HOURS_LIMIT}"
        )
    return hours


def build_parser() -> argparse.ArgumentParser:
    """Builds the command line parser.

    Returns:
        The parser with one subcommand per lifecycle step.
    """
    parser = argparse.ArgumentParser(prog="tomography.cloud")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, handler: Callable[..., int]) -> argparse.ArgumentParser:
        command = sub.add_parser(name)
        command.add_argument("--state", type=Path, default=DEFAULT_STATE)
        command.set_defaults(handler=handler)
        return command

    up = add("up", cmd_up)
    up.add_argument("--gpu", default=DEFAULT_GPU)
    up.add_argument(
        "--cloud", choices=("community", "secure"), default="community"
    )
    up.add_argument("--image", default=DEFAULT_IMAGE)
    up.add_argument("--volume-id")
    up.add_argument("--volume-gb", type=int, default=50)
    up.add_argument("--disk-gb", type=int, default=30)
    up.add_argument("--public-key", type=Path, default=DEFAULT_PUBLIC_KEY)
    up.add_argument("--name", default="tomography")
    up.add_argument("--max-hours", type=_max_hours, required=True)
    up.add_argument("--timeout-minutes", type=float, default=15)
    up.add_argument("--ssh-timeout-minutes", type=float, default=5)
    up.add_argument("--force", action="store_true")
    up.add_argument("--no-self-removal", action="store_true")

    ssh = add("ssh", cmd_ssh)
    ssh.add_argument("remote", nargs="*")

    push = add("push", cmd_push)
    push.add_argument("--ref", required=True)
    push.add_argument("--spec", type=Path, required=True)
    push.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    push.add_argument("--sync-cache", action="store_true")

    add("setup", cmd_setup)

    start = add("start", cmd_start)
    start.add_argument("--spec", required=True)
    start.add_argument("--max-hours", type=_max_hours, required=True)
    start.add_argument("--grace-minutes", type=float, default=30)
    start.add_argument("--no-self-removal", action="store_true")

    add("status", cmd_status)

    pull = add("pull", cmd_pull)
    pull.add_argument("--dest", type=Path, required=True)
    pull.add_argument("--path")

    resume = add("resume", cmd_resume)
    resume.add_argument("--max-hours", type=_max_hours, required=True)
    resume.add_argument("--timeout-minutes", type=float, default=15)
    resume.add_argument("--ssh-timeout-minutes", type=float, default=5)
    resume.add_argument("--no-self-removal", action="store_true")

    down = add("down", cmd_down)
    down.add_argument("--timeout-minutes", type=float, default=2)
    down.add_argument("--discard", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None, deps: Deps | None = None) -> int:
    """Runs one command.

    Args:
        argv: Command line arguments, without the program name.
        deps: Side effects; the real ones when omitted.

    Returns:
        The process exit code.
    """
    deps = deps or Deps()
    args = build_parser().parse_args(argv)
    args.state = args.state.expanduser()
    try:
        code: int = args.handler(args, deps)
    except CloudError as error:
        deps.out(f"error: {error}")
        return 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
