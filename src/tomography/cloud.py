"""Manages a RunPod GPU pod for chain runs from a laptop.

Run as `python -m tomography.cloud <command>`. The commands create a pod, push
a commit and the data cache over ssh, set the environment up, start a chain
detached, report its status, pull the results and delete the pod.

The API key is read from `RUNPOD_API_KEY` only. It is sent in the
`Authorization` header to the RunPod API and nowhere else: it is never written
to the state file, printed, put on a command line or sent to the pod.

References:
    https://docs.runpod.io/api-reference/pods/POST/pods
    https://docs.runpod.io/pods/configuration/use-ssh
    https://docs.runpod.io/pods/references/environment-variables
    https://docs.runpod.io/runpodctl/manage-pods
"""

import argparse
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any
import urllib.error
import urllib.request

API_BASE = "https://rest.runpod.io/v1"
DEFAULT_GPU = "NVIDIA GeForce RTX 4090"
DEFAULT_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
DEFAULT_STATE = Path("~/.cache/tomography-cloud/pod.json")
DEFAULT_PUBLIC_KEY = Path("~/.ssh/id_ed25519.pub")
DEFAULT_CACHE = Path("~/.cache/tomography")
REMOTE_SRC = "/workspace/src"
REMOTE_CHAINS = "/workspace/chains"
REMOTE_CACHE = "/workspace/cache/tomography"
REMOTE_RUN = "/workspace/run"
REMOTE_VENVS = "/workspace/venv"
REMOTE_UV_CACHE = "/workspace/uv-cache"
DEFAULT_OUT = "/workspace/out"
MIN_DRIVER_MAJOR = 580
MAX_HOURS_LIMIT = 24
_POLL_SECONDS = 10.0
_USER_AGENT = "tomography-cloud/1.0"

Transport = Callable[
    [str, str, Mapping[str, str], bytes | None], tuple[int, Any]
]


_WATCHDOG_BODY = r"""LOG="$RUN/watchdog.log"
log() { echo "$(date -Is) $*" >> "$LOG"; }
while IFS= read -r -d '' kv; do
  case "$kv" in
    RUNPOD_POD_ID=*|RUNPOD_API_KEY=*) export "$kv" ;;
  esac
done < /proc/1/environ
remove_pod() {
  log "removing the pod: $1"
  if [ -z "${RUNPOD_POD_ID:-}" ] || ! command -v runpodctl >/dev/null 2>&1; then
    log "cannot remove: RUNPOD_POD_ID or runpodctl is missing"
    exit 1
  fi
  runpodctl remove pod "$RUNPOD_POD_ID" >> "$LOG" 2>&1
  exit 0
}
chain_finished() {
  if [ -n "$(find "$CHAINS" -maxdepth 1 -name '*.log' -newer "$RUN/started" \
      -exec grep -l CHAIN_DONE {} + 2>/dev/null)" ]; then
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
    remove_pod "max runtime reached"
  fi
  if [ -z "$finished_at" ] && chain_finished; then
    finished_at=$now
    log "chain finished; grace period started"
  fi
  waited=$((now - ${finished_at:-$now}))
  if [ -n "$finished_at" ] && [ "$waited" -ge "$GRACE_SECONDS" ]; then
    remove_pod "grace period after the chain ended"
  fi
  sleep 30
done
"""


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
    except urllib.error.URLError as error:
        raise CloudError(f"cannot reach {url}: {error.reason}") from error
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


@dataclasses.dataclass
class Deps:
    """The side effects of the module, replaceable in tests.

    Attributes:
        transport: Sends one HTTP request and returns status and JSON.
        runner: Runs local commands.
        clock: Returns the current time in seconds.
        sleep: Waits for a number of seconds.
        env: The environment variables the module may read.
        out: Prints one line to the user.
    """

    transport: Transport = _urllib_transport
    runner: Any = dataclasses.field(default_factory=SubprocessRunner)
    clock: Callable[[], float] = time.monotonic
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
        """
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": _USER_AGENT,
        }
        body = None if payload is None else json.dumps(payload).encode()
        return self._transport(method, API_BASE + path, headers, body)

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
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def _save_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _pod_endpoint(pod: Mapping[str, Any]) -> tuple[str, int] | None:
    ip = pod.get("publicIp")
    port = (pod.get("portMappings") or {}).get("22")
    if pod.get("desiredStatus") == "RUNNING" and ip and port:
        return str(ip), int(port)
    return None


def _ssh_argv(args: argparse.Namespace, state: Mapping[str, Any]) -> list[str]:
    known_hosts = (args.state.parent / "known_hosts").as_posix()
    return [
        "ssh",
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
        f"root@{state['ip']}",
    ]


def _require_endpoint(state: Mapping[str, Any]) -> None:
    if not state.get("ip") or not state.get("ssh_port"):
        raise CloudError("the pod has no ssh endpoint yet; run `up` first")


def _remote(
    args: argparse.Namespace,
    deps: Deps,
    state: Mapping[str, Any],
    command: str,
    input_bytes: bytes | None = None,
) -> CommandResult:
    _require_endpoint(state)
    result: CommandResult = deps.runner.run(
        [*_ssh_argv(args, state), command], input_bytes
    )
    return result


def _stream(deps: Deps, source: Sequence[str], sink: Sequence[str]) -> None:
    code = deps.runner.pipe(source, sink)
    if code:
        raise CloudError(f"{source[0]} pipeline failed with exit code {code}")


def cmd_up(args: argparse.Namespace, deps: Deps) -> int:
    """Creates a pod and waits until ssh can reach it.

    Args:
        args: Parsed command line.
        deps: Side effects.

    Returns:
        0 when the pod is reachable, 1 on a timeout.

    Raises:
        CloudError: On an API failure.
    """
    api = Api(deps)
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
    status, created = api.request("POST", "/pods", body)
    pod = api.check(status, created, "create pod")
    pod_id = str(pod["id"])
    identity = args.identity or args.public_key.expanduser().with_suffix("")
    state: dict[str, Any] = {
        "pod_id": pod_id,
        "name": args.name,
        "gpu": args.gpu,
        "cloud": body["cloudType"],
        "created": _now_iso(),
        "cost_per_hr": pod.get("costPerHr"),
        "identity": str(identity),
        "status": "starting",
    }
    _save_state(args.state, state)
    deps.out(f"created pod {pod_id}; waiting for ssh")
    deadline = deps.clock() + args.timeout_minutes * 60
    while True:
        endpoint = _pod_endpoint(pod)
        if endpoint is not None:
            break
        if deps.clock() >= deadline:
            deps.out(
                f"pod {pod_id} did not expose ssh within "
                f"{args.timeout_minutes} minutes; run `down` to remove it"
            )
            return 1
        deps.sleep(_POLL_SECONDS)
        status, got = api.request("GET", f"/pods/{pod_id}")
        pod = api.check(status, got, "get pod")
    state.update(ip=endpoint[0], ssh_port=endpoint[1], status="running")
    _save_state(args.state, state)
    deps.out(f"pod {pod_id} ready at {endpoint[0]}:{endpoint[1]}")
    return 0


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


def cmd_push(args: argparse.Namespace, deps: Deps) -> int:
    """Uploads a commit, a chain spec and the data cache to the pod.

    Args:
        args: Parsed command line.
        deps: Side effects.

    Returns:
        0 on success.

    Raises:
        CloudError: When a transfer fails.
    """
    state = _load_state(args.state)
    _require_endpoint(state)
    ssh = _ssh_argv(args, state)
    resolved = deps.runner.run(["git", "rev-parse", "--short=12", args.ref])
    if resolved.returncode:
        raise CloudError(f"cannot resolve git ref {args.ref}")
    sha = resolved.stdout.strip()
    src = f"{REMOTE_SRC}/{sha}"
    quoted = shlex.quote(src)
    _stream(
        deps,
        ["git", "archive", "--format=tar", sha],
        [*ssh, f"mkdir -p {quoted} && tar -x -C {quoted}"],
    )
    spec_name = args.spec.name
    remote_spec = f"{REMOTE_CHAINS}/{spec_name}"
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
            [*ssh, f"mkdir -p {cache_dir} && tar -x -C {cache_dir}"],
        )
        deps.out("data cache uploaded")
    state.update(
        remote_src=src, remote_spec=remote_spec, remote_cache=REMOTE_CACHE
    )
    _save_state(args.state, state)
    deps.out(f"pushed {sha} to {src}")
    return 0


def _require_source(state: Mapping[str, Any]) -> str:
    src = state.get("remote_src")
    if not src:
        raise CloudError("no source on the pod; run `push` first")
    return str(src)


def _uv_env(src: str) -> list[str]:
    venv = f"{REMOTE_VENVS}/{Path(src).name}"
    return [
        'export PATH="$HOME/.local/bin:$PATH"',
        f"export UV_CACHE_DIR={REMOTE_UV_CACHE}",
        f"export UV_PROJECT_ENVIRONMENT={venv}",
    ]


def cmd_setup(args: argparse.Namespace, deps: Deps) -> int:
    """Checks the driver, installs uv, syncs the environment, checks CUDA.

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
    env = " && ".join(_uv_env(src))
    sync = (
        f"{env} && (command -v uv >/dev/null 2>&1 || "
        "curl -LsSf https://astral.sh/uv/install.sh | sh) && "
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


def _watchdog_script(out: str, max_hours: float, grace_minutes: float) -> str:
    header = [
        "#!/bin/bash",
        f"MAX_SECONDS={int(max_hours * 3600)}",
        f"GRACE_SECONDS={int(grace_minutes * 60)}",
        f"CHAINS={shlex.quote(out + '/_chain')}",
        f"RUN={REMOTE_RUN}",
    ]
    return "\n".join(header) + "\n" + _WATCHDOG_BODY


def cmd_start(args: argparse.Namespace, deps: Deps) -> int:
    """Starts the chain and its watchdog detached on the pod.

    Args:
        args: Parsed command line.
        deps: Side effects.

    Returns:
        0 when both processes were launched.

    Raises:
        CloudError: When the launch fails.
    """
    state = _load_state(args.state)
    src = _require_source(state)
    spec = f"{REMOTE_CHAINS}/{args.spec}"
    scripts = {
        "chain.sh": _chain_script(src, spec),
        "watchdog.sh": _watchdog_script(
            args.out, args.max_hours, args.grace_minutes
        ),
    }
    for name, text in scripts.items():
        upload = _remote(
            args,
            deps,
            state,
            f"mkdir -p {REMOTE_RUN} && cat > {REMOTE_RUN}/{name}",
            text.encode(),
        )
        if upload.returncode:
            raise CloudError(f"uploading {name} failed")
    launches = " ".join(
        f"setsid nohup bash {REMOTE_RUN}/{name} "
        f"> {REMOTE_RUN}/{name}.out 2>&1 < /dev/null &"
        for name in scripts
    )
    command = (
        f"rm -f {REMOTE_RUN}/chain.pid && "
        f"touch {REMOTE_RUN}/started && {launches}"
    )
    if _remote(args, deps, state, command).returncode:
        raise CloudError("launching the chain failed")
    state.update(remote_out=args.out, remote_spec=spec)
    _save_state(args.state, state)
    deps.out(
        "chain started; the watchdog removes the pod after "
        f"{args.max_hours:g} h or {args.grace_minutes:g} min after the end"
    )
    return 0


def cmd_status(args: argparse.Namespace, deps: Deps) -> int:
    """Prints the tail of the chain log and whether the chain is alive.

    Args:
        args: Parsed command line.
        deps: Side effects.

    Returns:
        0 when the remote command ran.
    """
    state = _load_state(args.state)
    chains = shlex.quote(f"{state.get('remote_out', DEFAULT_OUT)}/_chain")
    pid_file = f"{REMOTE_RUN}/chain.pid"
    command = (
        f"f=$(ls -t {chains}/*.log 2>/dev/null | head -n 1); "
        'if [ -n "$f" ]; then tail -n 20 "$f"; '
        "else echo 'no chain log yet'; fi; "
        f"if [ -f {pid_file} ] && kill -0 $(cat {pid_file}) 2>/dev/null; "
        "then echo CHAIN_ALIVE; else echo CHAIN_NOT_RUNNING; fi"
    )
    result = _remote(args, deps, state, command)
    deps.out(result.stdout.rstrip("\n"))
    return result.returncode


def cmd_pull(args: argparse.Namespace, deps: Deps) -> int:
    """Streams the remote out root back as a tar into a local folder.

    Args:
        args: Parsed command line.
        deps: Side effects.

    Returns:
        0 on success.
    """
    state = _load_state(args.state)
    _require_endpoint(state)
    remote = args.path or state.get("remote_out", DEFAULT_OUT)
    args.dest.mkdir(parents=True, exist_ok=True)
    _stream(
        deps,
        [*_ssh_argv(args, state), f"tar -c -C {shlex.quote(remote)} ."],
        ["tar", "-x", "-C", str(args.dest)],
    )
    deps.out(f"pulled {remote} into {args.dest}")
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
    up.add_argument("--identity", type=Path)
    up.add_argument("--name", default="tomography")
    up.add_argument("--timeout-minutes", type=float, default=15)

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
    start.add_argument("--out", default=DEFAULT_OUT)

    add("status", cmd_status)

    pull = add("pull", cmd_pull)
    pull.add_argument("--dest", type=Path, required=True)
    pull.add_argument("--path")

    down = add("down", cmd_down)
    down.add_argument("--timeout-minutes", type=float, default=2)
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
