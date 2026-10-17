"""Copies finished runs into a results tree that is safe to commit.

A run directory is any folder holding `config.json` and `meta.json`, however
deep it sits under a root. Each finished run is copied to
`<dest>/<experiment>/<model>/<config hash>/`, keeping only the files the plots
and the docs read. Absolute paths stored in the copied JSON and in the metadata
of the `.npz` files are rewritten, and a last check refuses to write anything
that still names the local machine or user.
"""

from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import fnmatch
import getpass
import io
import json
import os
from pathlib import Path
import re
import shutil
import socket
from typing import Any
from typing import Literal
import uuid
import zipfile

import numpy as np

_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_EXPORTED_SUFFIXES = frozenset({".json", ".npz", ".csv"})
_EXCLUDED_NAMES = (
    "readers*.npz",
    "sweep*.npz",
    "items.json",
    "trials*",
    "transfer.npz",
    "signals*.npz",
)
_DRIVE_START = r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]"
_UNC_START = r"(?<![\\\w])\\\\[^\\/\s]+[\\/]"
_POSIX_START = r"(?:(?<![\w/.\-~])|(?<=file://))/[^/\s\"',;)\]]+/"
# A path continues across a space when the next word holds a separator.
_PATH_BODY = r"[^\s\"',;)\]]*(?:[ \t]+(?=\S*[\\/])[^\s\"',;)\]]+)*"
_EMBEDDED_PATH = re.compile(
    rf"(?:{_DRIVE_START}|{_UNC_START}|{_POSIX_START}){_PATH_BODY}"
)
_PRIVATE_PATTERNS = (
    ("a drive-letter path", re.compile(_DRIVE_START)),
    ("a UNC path", re.compile(_UNC_START + r"[^\\/\s]")),
    ("a POSIX absolute path", re.compile(_POSIX_START)),
    (
        "a 'Users' path segment",
        re.compile(r"(?:^|[\\/:])Users(?=[\\/]|$)", re.MULTILINE),
    ),
)
_WHOLE_PATH = re.compile(rf"{_DRIVE_START}|{_UNC_START}|//?[^/\s]+/")
_SNIPPET_MARGIN = 20
_MIN_USER_NAME = 3
_FILE_ERRORS = (
    ValueError,
    OSError,
    EOFError,
    KeyError,
    TypeError,
    zipfile.BadZipFile,
)
RESULTS_PREFIX = "results"
EXTERNAL_PREFIX = "<external>"


class ExportError(Exception):
    """An export that cannot be done safely; nothing was written."""


@dataclasses.dataclass(frozen=True)
class RunInfo:
    """A run directory found under a root.

    Attributes:
        path: The run directory.
        experiment: The experiment id, from `meta.json`.
        model: The checkpoint key, from `config.json`.
        config_hash: The config hash, from `meta.json`.
        git_sha: The code revision of the run, from `meta.json`.
        finished: Whether `meta.json` has a finish time.
    """

    path: Path
    experiment: str
    model: str
    config_hash: str
    git_sha: str | None
    finished: bool

    @property
    def key(self) -> str:
        """The `<experiment>/<model>/<config hash>` identity of the run."""
        return f"{self.experiment}/{self.model}/{self.config_hash}"


@dataclasses.dataclass(frozen=True)
class ExportedRun:
    """One run of the destination after an export.

    Attributes:
        key: The `<experiment>/<model>/<config hash>` identity of the run.
        size: Total bytes of the run's files in the destination.
        status: "copied" if files were written, "unchanged" if the
            destination already held the same run.
    """

    key: str
    size: int
    status: Literal["copied", "unchanged"]


@dataclasses.dataclass
class ExportReport:
    """What an export did.

    Attributes:
        exported: The finished runs now in the destination, in key order.
        skipped: The unfinished run directories that were not exported.
        warnings: One message per path that could not be mapped to a run.
    """

    exported: list[ExportedRun]
    skipped: list[Path]
    warnings: list[str]

    @property
    def total_bytes(self) -> int:
        """Total bytes of the exported runs."""
        return sum(run.size for run in self.exported)


def is_exported(relative: str) -> bool:
    """Tells whether a file of a run is copied by the export.

    Args:
        relative: The file path inside the run directory, with "/" separators.

    Returns:
        True for the JSON, NPZ and CSV files that plots and docs read; False
        for figures, logs and the raw vectors, trials and item files.
    """
    parts = relative.split("/")
    if "figures" in parts[:-1]:
        return False
    name = parts[-1]
    if Path(name).suffix not in _EXPORTED_SUFFIXES:
        return False
    return not any(fnmatch.fnmatch(name, glob) for glob in _EXCLUDED_NAMES)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError) as error:
        raise ExportError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ExportError(f"{path} does not hold a JSON object")
    return value


def _load_run(path: Path) -> RunInfo:
    config = _read_json(path / "config.json")
    meta = _read_json(path / "meta.json")
    parts = {
        "experiment": meta.get("experiment"),
        "model": config.get("model"),
        "config_hash": meta.get("config_hash"),
    }
    for label, part in parts.items():
        if not isinstance(part, str) or not _PART.fullmatch(part):
            raise ExportError(f"{path} has no usable {label}: {part!r}")
    git_sha = meta.get("git_sha")
    return RunInfo(
        path=path,
        experiment=str(parts["experiment"]),
        model=str(parts["model"]),
        config_hash=str(parts["config_hash"]),
        git_sha=git_sha if isinstance(git_sha, str) else None,
        finished=meta.get("finished_at") is not None,
    )


def find_runs(roots: Iterable[Path]) -> list[RunInfo]:
    """Finds the run directories under the roots.

    Args:
        roots: Folders to walk; a run is found by its files at any depth.

    Returns:
        The runs, each once, sorted by path.

    Raises:
        ExportError: If a root is not a folder or a run has unreadable files.
    """
    found: dict[Path, RunInfo] = {}
    for root in roots:
        if not Path(root).is_dir():
            raise ExportError(f"not a directory: {root}")
        for folder, subfolders, files in os.walk(root):
            if "config.json" in files and "meta.json" in files:
                subfolders.clear()
                path = Path(folder)
                found.setdefault(path.resolve(), _load_run(path))
    return [found[path] for path in sorted(found)]


def _check_collisions(runs: Sequence[RunInfo]) -> None:
    by_key: dict[str, RunInfo] = {}
    for run in runs:
        other = by_key.setdefault(run.key, run)
        if other is not run:
            raise ExportError(
                f"two runs share {run.key}: {other.path} "
                f"(git {other.git_sha}) and {run.path} (git {run.git_sha}); "
                "choose one of the roots"
            )


def _segments(text: str) -> list[str]:
    return text.replace("\\", "/").rstrip("/").split("/")


def _index_key(segments: Sequence[str]) -> str:
    joined = "/".join(segments).lower()
    return re.sub(r"^/(?:mnt/)?([a-z])/", r"\1:/", joined)


def _run_index(runs: Sequence[RunInfo]) -> dict[str, RunInfo]:
    return {_index_key(_segments(str(run.path.resolve()))): run for run in runs}


def _is_path(text: str) -> bool:
    return "\n" not in text and _WHOLE_PATH.match(text) is not None


class _PathRewriter:
    """Replaces absolute paths in JSON values by portable names.

    A path into an exported run becomes `results/<key>` (plus what follows
    the run folder), into a run that is not exported `<key>`, and any other
    path `<external>/<last name>`.
    """

    def __init__(
        self, index: Mapping[str, RunInfo], exported_keys: set[str]
    ) -> None:
        self._index = index
        self._exported = exported_keys
        self._seen: set[str] = set()
        self.warnings: list[str] = []
        self.context = ""

    def rewrite(self, value: Any) -> Any:
        """Returns a copy of a JSON value with every path rewritten."""
        if isinstance(value, str):
            return self._string(value)
        if isinstance(value, list):
            return [self.rewrite(item) for item in value]
        if isinstance(value, dict):
            return {
                self._string(key): self.rewrite(item)
                for key, item in value.items()
            }
        return value

    def _string(self, text: str) -> str:
        if _is_path(text):
            return self._path(text)
        return _EMBEDDED_PATH.sub(
            lambda match: self._embedded(text, match), text
        )

    def _embedded(self, text: str, match: re.Match[str]) -> str:
        replacement = self._path(match.group(0))
        rest = text[match.end() :]
        if replacement.startswith(EXTERNAL_PREFIX) and re.match(r"[ \t]", rest):
            tail = _EMBEDDED_PATH.sub("", rest).split("\n")[0]
            if re.search(r"[\\/]", tail):
                raise ValueError(
                    "a path tail after a rewritten path near "
                    f"{_snippet(text, *match.span())!r}"
                )
        return replacement

    def _path(self, text: str) -> str:
        segments = _segments(text)
        for end in range(len(segments), 0, -1):
            run = self._index.get(_index_key(segments[:end]))
            if run is not None:
                return self._named(run, segments[end:])
        located = self._locate_on_disk(segments)
        if located is not None:
            run, rest = located
            return self._named(run, rest)
        name = next((part for part in reversed(segments) if part), "")
        replacement = f"{EXTERNAL_PREFIX}/{name}"
        if text not in self._seen:
            self._seen.add(text)
            self.warnings.append(
                f"{self.context}: path {text} is outside the given roots, "
                f"written as {replacement}"
            )
        return replacement

    def _named(self, run: RunInfo, rest: Sequence[str]) -> str:
        prefix = f"{RESULTS_PREFIX}/" if run.key in self._exported else ""
        return "/".join([f"{prefix}{run.key}", *rest])

    def _locate_on_disk(
        self, segments: Sequence[str]
    ) -> tuple[RunInfo, Sequence[str]] | None:
        if not Path("/".join(segments)).is_absolute():
            return None
        for end in range(len(segments), 0, -1):
            folder = Path("/".join(segments[:end]))
            if (folder / "config.json").is_file() and (
                folder / "meta.json"
            ).is_file():
                return _load_run(folder), segments[end:]
        return None


def _dump_json(value: Any) -> bytes:
    return f"{json.dumps(value, indent=2, allow_nan=False)}\n".encode()


def _rewrite_json(data: bytes, rewriter: _PathRewriter) -> bytes:
    value = json.loads(data)
    rewritten = rewriter.rewrite(value)
    return data if rewritten == value else _dump_json(rewritten)


def _rewrite_npz(data: bytes, rewriter: _PathRewriter) -> bytes:
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        if "_meta" not in archive.files:
            return data
        arrays = {key: archive[key] for key in archive.files}
    meta = json.loads(str(arrays["_meta"]))
    rewritten = rewriter.rewrite(meta)
    if rewritten == meta:
        return data
    arrays["_meta"] = np.array(json.dumps(rewritten))
    buffer = io.BytesIO()
    np.savez(buffer, allow_pickle=False, **arrays)
    return buffer.getvalue()


def _rewrite(name: str, data: bytes, rewriter: _PathRewriter) -> bytes:
    if name.endswith(".json"):
        return _rewrite_json(data, rewriter)
    if name.endswith(".npz"):
        return _rewrite_npz(data, rewriter)
    return data


def _json_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _json_strings(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _json_strings(item)


def _scanned_members(name: str, data: bytes) -> Iterator[tuple[str, str]]:
    """Yields the (member, text) pairs of a file that the check reads."""
    if name.endswith(".json"):
        yield name, "\n".join(_json_strings(json.loads(data)))
    elif name.endswith(".npz"):
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            for key in archive.files:
                array = archive[key]
                if key == "_meta":
                    meta = json.loads(str(array))
                    yield f"{name}:{key}", "\n".join(_json_strings(meta))
                elif array.dtype.kind in "US":
                    text = "\n".join(str(item) for item in array.ravel())
                    yield f"{name}:{key}", text
    else:
        yield name, data.decode("utf-8", errors="replace")


def _machine_names() -> list[str]:
    """Returns the user name and host name (full and first label)."""
    try:
        host = socket.gethostname()
    except OSError:
        host = ""
    names = {getpass.getuser(), host, host.partition(".")[0]}
    return sorted(name for name in names if len(name) >= _MIN_USER_NAME)


def _snippet(text: str, start: int, end: int) -> str:
    around = text[max(0, start - _SNIPPET_MARGIN) : end + _SNIPPET_MARGIN]
    return " ".join(around.split())


def _private_marker(text: str) -> str | None:
    """Describes the first thing in the text that names the local machine."""
    for label, pattern in _PRIVATE_PATTERNS:
        match = pattern.search(text)
        if match:
            return f"{label} near {_snippet(text, *match.span())!r}"
    for name in _machine_names():
        match = re.search(re.escape(name), text, re.IGNORECASE)
        if match:
            return (
                "the local user or host name near "
                f"{_snippet(text, *match.span())!r}"
            )
    return None


def _private_problem(name: str, data: bytes) -> str | None:
    for member, text in _scanned_members(name, data):
        marker = _private_marker(text)
        if marker is not None:
            where = "" if member == name else f" ({member})"
            return f"{marker}{where}"
    return None


def _plan_run(run: RunInfo, rewriter: _PathRewriter) -> dict[str, bytes]:
    rewriter.context = run.key
    files: dict[str, bytes] = {}
    for file in sorted(run.path.rglob("*")):
        name = file.relative_to(run.path).as_posix()
        if not (file.is_file() and is_exported(name)):
            continue
        try:
            data = _rewrite(name, file.read_bytes(), rewriter)
            problem = _private_problem(name, data)
        except _FILE_ERRORS as error:
            raise ExportError(f"{run.key}/{name}: {error}") from error
        if problem is not None:
            raise ExportError(f"{run.key}/{name} still holds {problem}")
        files[name] = data
    return files


def _existing_files(folder: Path) -> dict[str, bytes]:
    existing: dict[str, bytes] = {}
    if folder.is_dir():
        for file in sorted(folder.rglob("*")):
            name = file.relative_to(folder).as_posix()
            if file.is_file() and is_exported(name):
                existing[name] = file.read_bytes()
    return existing


def _differences(
    planned: dict[str, bytes], existing: dict[str, bytes]
) -> list[str]:
    names = sorted(set(planned) | set(existing))
    return [name for name in names if planned.get(name) != existing.get(name)]


def _write_file(target: Path, data: bytes) -> None:
    target.write_bytes(data)


def _write_run(folder: Path, files: dict[str, bytes]) -> None:
    """Writes a run next to its place and moves it in, so none is partial."""
    staging = folder.parent / f".{folder.name}.{uuid.uuid4().hex}.tmp"
    try:
        for name, data in files.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_file(target, data)
        os.replace(staging, folder)
    except OSError as error:
        raise ExportError(f"cannot write {folder}: {error}") from error
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def export_runs(roots: Sequence[Path], dest: Path) -> ExportReport:
    """Copies the finished runs under the roots into a results tree.

    Everything is planned and checked before the first file is written, so a
    planning error leaves the destination untouched. Each run is moved into
    place whole, so a write error leaves no partial run.

    Args:
        roots: Folders holding run directories, at any nesting.
        dest: The results root to write `<experiment>/<model>/<hash>` into.

    Returns:
        The exported runs, the skipped unfinished runs and the warnings.

    Raises:
        ExportError: If two finished runs share an identity, a destination run
            differs from the one to copy, a root is not a folder, or a copied
            file would still name the local machine or user.
    """
    runs = find_runs(roots)
    finished = [run for run in runs if run.finished]
    _check_collisions(finished)
    rewriter = _PathRewriter(_run_index(runs), {run.key for run in finished})
    plans: list[tuple[RunInfo, dict[str, bytes], bool]] = []
    for run in sorted(finished, key=lambda item: item.key):
        files = _plan_run(run, rewriter)
        folder = Path(dest) / run.key
        existing = _existing_files(folder)
        if folder.exists() and not existing:
            raise ExportError(
                f"{folder} exists without the run's files; "
                "delete it to export again"
            )
        differing = _differences(files, existing) if existing else []
        if differing:
            raise ExportError(
                f"{Path(dest) / run.key} differs from {run.path} in "
                f"{', '.join(differing)}; delete it to export again"
            )
        plans.append((run, files, bool(existing)))
    exported = []
    for run, files, present in plans:
        if not present:
            _write_run(Path(dest) / run.key, files)
        exported.append(
            ExportedRun(
                key=run.key,
                size=sum(len(data) for data in files.values()),
                status="unchanged" if present else "copied",
            )
        )
    return ExportReport(
        exported=exported,
        skipped=[run.path for run in runs if not run.finished],
        warnings=rewriter.warnings,
    )
