"""Experiment configuration, run directories, seeding and results I/O."""

from collections.abc import Callable
from collections.abc import Mapping
import copy
import dataclasses
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import types
import typing
from typing import Any
from typing import BinaryIO
from typing import Literal
from typing import Self
import uuid

import numpy as np
from numpy.typing import NDArray

EXPERIMENT_ID = re.compile(r"[a-z][a-z0-9]*")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_EXTENSION = re.compile(r"[a-z0-9]+")
# Folder whose git state is recorded in the metadata of every run.
_SOURCE_DIR = Path(__file__).resolve().parent
_LIBRARIES = (
    "torch",
    "transformers",
    "numpy",
    "scikit-learn",
    "datasets",
    "peft",
    "matplotlib",
    "plotly",
)
_RESERVED_NAMES = frozenset({"config", "meta"})
_RESERVED_ARRAY_NAMES = frozenset({"_meta", "file", "allow_pickle"})


class _TypeMismatchError(Exception):
    """Raised when a value does not fit a field annotation."""


_SCALAR_CHECKS: Mapping[Any, Callable[[Any], bool]] = {
    bool: lambda value: isinstance(value, bool),
    int: lambda value: isinstance(value, int) and not isinstance(value, bool),
    float: lambda value: (
        isinstance(value, int | float) and not isinstance(value, bool)
    ),
    str: lambda value: isinstance(value, str),
}


def _coerce(annotation: Any, value: Any) -> Any:
    """Checks a JSON value against an annotation, turning lists into tuples.

    Annotations other than scalars, None, Literal, unions and tuples are not
    checked.

    Args:
        annotation: A resolved type annotation.
        value: A value decoded from JSON.

    Returns:
        The value, with lists converted where the annotation asks for a tuple
        and ints widened where it asks for a float.

    Raises:
        _TypeMismatchError: If the value does not fit the annotation,
            including an int too large to convert to a float.
    """
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if annotation is type(None):
        if value is not None:
            raise _TypeMismatchError
        return None
    if origin is Literal:
        if value not in args:
            raise _TypeMismatchError
        return value
    if origin in (typing.Union, types.UnionType):
        for member in args:
            try:
                return _coerce(member, value)
            except _TypeMismatchError:
                continue
        raise _TypeMismatchError
    if origin is tuple:
        if not isinstance(value, list | tuple):
            raise _TypeMismatchError
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(args[0], item) for item in value)
        if len(args) != len(value):
            raise _TypeMismatchError
        return tuple(
            _coerce(member, item)
            for member, item in zip(args, value, strict=True)
        )
    check = _SCALAR_CHECKS.get(annotation)
    if check is not None and not check(value):
        raise _TypeMismatchError
    if annotation is float:
        try:
            return float(value)
        except OverflowError:
            raise _TypeMismatchError from None
    return value


@dataclasses.dataclass(frozen=True, kw_only=True)
class ExperimentConfig:
    """Base class of every experiment configuration.

    Subclasses are frozen, keyword-only dataclasses that add JSON-serializable
    fields. Two configs with equal fields share one run directory.

    Attributes:
        experiment: Lowercase experiment id used in paths and on the command
            line, such as "r1" or "p3".
        model: Key of a registered checkpoint, kept as a plain string.
        seed: Base seed of the run.
    """

    experiment: str
    model: str
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Returns the fields as JSON values, with tuples turned into lists."""
        data: dict[str, Any] = json.loads(json.dumps(dataclasses.asdict(self)))
        return data

    def config_hash(self) -> str:
        """Returns a stable 12-character hex digest of the fields.

        Returns:
            The first 12 hex digits of the SHA-256 of the canonical JSON
            (sorted keys, no whitespace).

        Raises:
            ValueError: If a field holds a non-finite float.
        """
        canonical = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Builds a config from decoded JSON, checking names and types.

        Args:
            data: Field values as produced by `to_dict`. Missing fields take
                their defaults; lists become tuples where a field is a tuple.

        Returns:
            The config.

        Raises:
            ValueError: If a key is not a field, a required field is missing
                or a value does not fit its field's annotation.
        """
        fields = {field.name: field for field in dataclasses.fields(cls)}
        unknown = sorted(set(data) - set(fields))
        if unknown:
            raise ValueError(f"unknown fields for {cls.__name__}: {unknown}")
        missing = sorted(
            name
            for name, field in fields.items()
            if name not in data
            and field.default is dataclasses.MISSING
            and field.default_factory is dataclasses.MISSING
        )
        if missing:
            raise ValueError(
                f"missing required fields for {cls.__name__}: {missing}"
            )
        hints = typing.get_type_hints(cls)
        values: dict[str, Any] = {}
        for name, value in data.items():
            try:
                values[name] = _coerce(hints[name], value)
            except _TypeMismatchError:
                raise ValueError(
                    f"{cls.__name__}.{name}: expected {hints[name]}, "
                    f"got {value!r}"
                ) from None
        return cls(**values)


def derive_seed(base: int, *keys: str | int) -> int:
    """Derives a reproducible seed from a base seed and identifying keys.

    Args:
        base: The base seed.
        *keys: Strings or integers that identify the consumer, such as a
            layer index or a trial name.

    Returns:
        An integer in [0, 2**32) taken from the SHA-256 of `repr((base, keys))`,
        so it is stable across processes and Python versions.
    """
    digest = hashlib.sha256(repr((base, keys)).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def seed_everything(seed: int) -> np.random.Generator:
    """Seeds the global random number generators of a run.

    Seeds `random`, the legacy NumPy global state and torch on the CPU and on
    every CUDA device.

    Args:
        seed: An integer in [0, 2**32).

    Returns:
        A new NumPy generator seeded with `seed`.

    Raises:
        ValueError: If `seed` is outside [0, 2**32).
    """
    if not 0 <= seed < 2**32:
        raise ValueError(f"seed must be in [0, 2**32), got {seed}")
    # Imported here so that reading results never pays for importing torch.
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def _check_path_part(label: str, value: str, pattern: re.Pattern[str]) -> None:
    if not pattern.fullmatch(value):
        raise ValueError(
            f"{label} {value!r} is not usable in a path "
            f"(expected {pattern.pattern})"
        )


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def _git(args: list[str], cwd: Path) -> str | None:
    """Returns the stdout of a git command, or None if it cannot run."""
    try:
        completed = subprocess.run(
            ["git", "--no-optional-locks", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _git_state(cwd: Path) -> tuple[str | None, bool | None]:
    """Returns the HEAD commit and whether tracked files differ from it.

    Untracked files are ignored, so result folders that are not committed yet
    do not mark the code as dirty. Both values are None outside a repository.
    """
    sha = _git(["rev-parse", "HEAD"], cwd)
    if sha is None:
        return None, None
    status = _git(["status", "--porcelain", "--untracked-files=no"], cwd)
    return sha, None if status is None else bool(status)


def _device_name() -> str:
    # Imported here so that reading results never pays for importing torch.
    import torch

    if torch.cuda.is_available():
        return f"cuda:0 ({torch.cuda.get_device_name(0)})"
    return "cpu"


def _library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("tomography", *_LIBRARIES):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _atomic_write(path: Path, write: Callable[[BinaryIO], object]) -> None:
    """Writes a file so that readers see the old or the new content only.

    Args:
        path: Destination file.
        write: Called with a binary handle of a temporary file that sits next
            to `path`; the file replaces `path` only if `write` succeeds.
    """
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp_path.open("xb") as handle:
            write(handle)
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _to_jsonable(value: object) -> object:
    """Converts NumPy values and tuples, mapping non-finite floats to None."""
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _to_jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(item) for item in value]
    return value


def _atomic_write_json(path: Path, value: object) -> None:
    text = json.dumps(value, indent=2, allow_nan=False)
    data = f"{text}\n".encode()
    _atomic_write(path, lambda handle: handle.write(data))


class RunDir:
    """One run's directory: `<out_root>/<experiment>/<model>/<config_hash>/`.

    Use `create` to start a run and `open` to read a finished one.

    Attributes:
        path: The run directory.
    """

    def __init__(
        self, path: Path, config: dict[str, Any], meta: dict[str, Any]
    ) -> None:
        """Wraps an existing directory; prefer `create` and `open`.

        Args:
            path: The run directory.
            config: The run's config as JSON values.
            meta: The run's metadata.
        """
        self.path = path
        self._config = config
        self._meta = meta

    @property
    def config(self) -> dict[str, Any]:
        """The run's config as JSON values (a copy)."""
        return copy.deepcopy(self._config)

    @property
    def meta(self) -> dict[str, Any]:
        """The run's metadata, as stored in `meta.json` (a copy)."""
        return copy.deepcopy(self._meta)

    @classmethod
    def path_for(
        cls, config: ExperimentConfig, *, out_root: str | Path = "results"
    ) -> Path:
        """Returns the run directory of a config without creating it.

        Args:
            config: The experiment configuration.
            out_root: Root folder of all runs.

        Returns:
            `<out_root>/<experiment>/<model>/<config_hash>`.

        Raises:
            ValueError: If the experiment id or the model key cannot be used
                as a path part, or the config has a non-finite float.
        """
        _check_path_part("experiment", config.experiment, EXPERIMENT_ID)
        _check_path_part("model", config.model, _SAFE_NAME)
        return (
            Path(out_root)
            / config.experiment
            / config.model
            / config.config_hash()
        )

    @classmethod
    def create(
        cls,
        config: ExperimentConfig,
        *,
        out_root: str | Path = "results",
        exist_ok: bool = False,
    ) -> Self:
        """Creates the run directory with its config and metadata files.

        Args:
            config: The experiment configuration.
            out_root: Root folder of all runs.
            exist_ok: Whether an existing directory of the same config may be
                reused. Its metadata is then rewritten for the new run.

        Returns:
            The new run directory.

        Raises:
            ValueError: If the experiment id or the model key cannot be used
                as a path part.
            FileExistsError: If the directory exists and `exist_ok` is false.
        """
        path = cls.path_for(config, out_root=out_root)
        config_data = config.to_dict()
        git_sha, git_dirty = _git_state(_SOURCE_DIR)
        meta: dict[str, Any] = {
            "experiment": config.experiment,
            "config_hash": config.config_hash(),
            "seed": config.seed,
            "git_sha": git_sha,
            "git_dirty": git_dirty,
            "device": _device_name(),
            "platform": platform.platform(),
            "versions": _library_versions(),
            "started_at": _now(),
            "finished_at": None,
        }
        path.mkdir(parents=True, exist_ok=exist_ok)
        _atomic_write_json(path / "config.json", config_data)
        _atomic_write_json(path / "meta.json", meta)
        return cls(path, config_data, meta)

    @classmethod
    def open(cls, path: str | Path) -> Self:
        """Opens an existing run directory.

        Args:
            path: A directory written by `create`.

        Returns:
            The run directory.

        Raises:
            ValueError: If `path` has no `config.json` and `meta.json`.
        """
        directory = Path(path)
        config_file = directory / "config.json"
        meta_file = directory / "meta.json"
        if not (config_file.is_file() and meta_file.is_file()):
            raise ValueError(f"not a run directory: {directory}")
        return cls(
            directory,
            json.loads(config_file.read_text("utf-8")),
            json.loads(meta_file.read_text("utf-8")),
        )

    def _embedded_meta(self) -> dict[str, Any]:
        return {**self._meta, "config": self._config}

    def _file(self, name: str, extension: str) -> Path:
        _check_path_part("name", name, _SAFE_NAME)
        if name.lower() in _RESERVED_NAMES:
            raise ValueError(f"name {name!r} is reserved")
        return self.path / f"{name}.{extension}"

    def write_json(self, name: str, payload: object) -> Path:
        """Writes `<name>.json` atomically with the run metadata embedded.

        The file holds `{"_meta": <meta and config>, "data": <payload>}`.
        NumPy scalars and arrays become JSON numbers and lists, and
        non-finite floats become null.

        Args:
            name: File stem. It must be a plain name, not `config` or `meta`.
            payload: The JSON-serializable result.

        Returns:
            The path of the file.

        Raises:
            ValueError: If `name` is not usable.
            TypeError: If `payload` holds a value JSON cannot represent.
        """
        path = self._file(name, "json")
        document = {
            "_meta": self._embedded_meta(),
            "data": _to_jsonable(payload),
        }
        _atomic_write_json(path, document)
        return path

    def read_json(self, name: str) -> object:
        """Reads the payload of `<name>.json`, without the metadata.

        Args:
            name: File stem given to `write_json`.

        Returns:
            The payload.

        Raises:
            ValueError: If the file has no embedded metadata.
        """
        document = json.loads(self._file(name, "json").read_text("utf-8"))
        if not (
            isinstance(document, dict)
            and "_meta" in document
            and "data" in document
        ):
            raise ValueError(f"{name}.json has no embedded metadata")
        return document["data"]

    def write_arrays(self, name: str, **arrays: NDArray[Any]) -> Path:
        """Writes `<name>.npz` atomically with the run metadata embedded.

        The metadata is stored as the 0-d unicode array `_meta` holding JSON.

        Args:
            name: File stem. It must be a plain name, not `config` or `meta`.
            **arrays: The arrays to store, by key.

        Returns:
            The path of the file.

        Raises:
            ValueError: If `name` or a key is not usable, or an array needs
                pickling (object dtype).
        """
        path = self._file(name, "npz")
        reserved = sorted(_RESERVED_ARRAY_NAMES.intersection(arrays))
        if reserved:
            raise ValueError(f"reserved array names: {reserved}")
        marker = np.array(json.dumps(self._embedded_meta()))
        _atomic_write(
            path,
            lambda handle: np.savez(
                handle, allow_pickle=False, _meta=marker, **arrays
            ),
        )
        return path

    def read_arrays(self, name: str) -> dict[str, NDArray[Any]]:
        """Reads the arrays of `<name>.npz`, without the metadata.

        Args:
            name: File stem given to `write_arrays`.

        Returns:
            The arrays by key.

        Raises:
            ValueError: If the file has no embedded metadata.
        """
        with np.load(self._file(name, "npz"), allow_pickle=False) as archive:
            if "_meta" not in archive.files:
                raise ValueError(f"{name}.npz has no embedded metadata")
            return {
                key: archive[key] for key in archive.files if key != "_meta"
            }

    def figure_path(self, name: str, ext: str) -> Path:
        """Returns `<run>/figures/<name>.<ext>`, creating the folder.

        Args:
            name: File stem of the figure.
            ext: File extension without the dot, such as "pdf".

        Returns:
            The path, not yet written.

        Raises:
            ValueError: If `name` or `ext` is not usable.
        """
        _check_path_part("name", name, _SAFE_NAME)
        _check_path_part("extension", ext, _EXTENSION)
        folder = self.path / "figures"
        folder.mkdir(exist_ok=True)
        return folder / f"{name}.{ext}"

    def finalize(self) -> None:
        """Stamps `finished_at` in `meta.json` after a successful run."""
        self._meta["finished_at"] = _now()
        _atomic_write_json(self.path / "meta.json", self._meta)
