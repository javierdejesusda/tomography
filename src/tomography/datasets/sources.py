"""Pinned data sources: Hub files at a commit SHA and checksummed URL files.

Every loader module declares one `PinnedSource` row and reads its files through
`fetch`, so a download is tied to an exact revision or digest. The tomography
cache holds every fetched file (Hub files under `<cache>/hub`, URL files under
`<cache>/url`). `huggingface_hub` may keep its own Xet chunk cache under
`HF_HOME`, which this library does not manage.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Literal
import urllib.request

from huggingface_hub import constants as hub_constants
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError

hub_constants.HF_HUB_DISABLE_SYMLINKS_WARNING = True

_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_PATH_SEGMENT = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*")
_TIMEOUT_SECONDS = 60


class ChecksumError(ValueError):
    """A file does not match its pinned SHA-256 digest."""


class OfflineCacheMissError(FileNotFoundError):
    """A pinned file is not in the cache and downloads are disabled."""


@dataclass(frozen=True)
class PinnedSource:
    """One pinned data source.

    Attributes:
        name: Short identifier. A "url" row caches its files under
            `<cache>/url/<name>/<digest prefix>/`; a "hub" row never uses it
            as a path (Hub files live under `<cache>/hub`).
        kind: "hub" for a dataset repository on the Hugging Face Hub, "url"
            for plain files under an https base URL.
        location: The repository id for "hub", the base URL for "url".
        revision: Full 40-character commit SHA for "hub", None for "url".
        files: Relative paths `fetch` serves. A hub source loaded through
            `datasets.load_dataset` may leave this empty.
        license: License of the data.
        sha256: Hex digest per file; required for every "url" file and
            enforced for any "hub" file that has an entry.

    Raises:
        ValueError: If the row is not fully pinned or names an unsafe path.
    """

    name: str
    kind: Literal["hub", "url"]
    location: str
    revision: str | None
    files: tuple[str, ...]
    license: str
    sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        """Validates that the row is pinned and its paths are safe."""
        if not _NAME.fullmatch(self.name):
            raise ValueError(f"unsafe source name {self.name!r}")
        for file in self.files:
            if not _is_safe_relative_path(file):
                raise ValueError(
                    f"{self.name}: file {file!r} must be a relative path "
                    "inside the source"
                )
        for file, digest in self.sha256.items():
            if file not in self.files or not _DIGEST.fullmatch(digest):
                raise ValueError(
                    f"{self.name}: sha256 entry for {file!r} must be a "
                    "64-character hex digest of a declared file"
                )
        if self.kind == "hub":
            self._check_hub()
        elif self.kind == "url":
            self._check_url()
        else:
            raise ValueError(f"{self.name}: unknown kind {self.kind!r}")

    def _check_hub(self) -> None:
        if not _COMMIT_SHA.fullmatch(self.revision or ""):
            raise ValueError(
                f"{self.name}: hub revision must be a full 40-character "
                f"lowercase commit SHA, got {self.revision!r}"
            )

    def _check_url(self) -> None:
        if self.revision is not None:
            raise ValueError(
                f"{self.name}: a url source has no revision, got "
                f"{self.revision!r}"
            )
        if not self.location.startswith("https://"):
            raise ValueError(
                f"{self.name}: url location must use https, got "
                f"{self.location!r}"
            )
        if not self.files:
            raise ValueError(f"{self.name}: a url source needs files")
        missing = [f for f in self.files if f not in self.sha256]
        if missing:
            raise ValueError(f"{self.name}: no sha256 for {missing}")


def cache_dir() -> Path:
    """Returns the cache directory, which is kept outside the repository.

    Returns:
        `$TOMOGRAPHY_CACHE` when set and non-empty, else
        `~/.cache/tomography`. The directory is not created here.
    """
    override = os.environ.get("TOMOGRAPHY_CACHE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "tomography"


def fetch(
    source: PinnedSource,
    *,
    file: str | None = None,
    cache: Path | None = None,
    offline: bool = False,
) -> Path:
    """Returns the local path of one pinned file, downloading it if needed.

    Hub files come from `hf_hub_download` at the pinned commit SHA and are
    stored under `<cache>/hub`. URL files are stored under
    `<cache>/url/<name>/<digest prefix>/`. Any file with a pinned SHA-256
    digest is checked against it on every call, cache hits included.

    Args:
        source: The pinned source to read from.
        file: One of `source.files`; may be omitted when there is exactly one.
        cache: Cache directory; defaults to `cache_dir()`.
        offline: If True, never touch the network.

    Returns:
        Path of the file inside the cache.

    Raises:
        ValueError: If `file` is missing or not declared by the source.
        OfflineCacheMissError: If `offline` is True and the file is not cached.
        ChecksumError: If a file does not match its pinned digest.
    """
    name = _resolve_file(source, file)
    root = cache_dir() if cache is None else cache
    if source.kind == "hub":
        return _fetch_hub(source, name, root, offline)
    return _fetch_url(source, name, root, offline)


def _is_safe_relative_path(file: str) -> bool:
    return all(_PATH_SEGMENT.fullmatch(part) for part in file.split("/"))


def _resolve_file(source: PinnedSource, file: str | None) -> str:
    if file is None:
        if len(source.files) != 1:
            raise ValueError(
                f"{source.name}: pass file=, the source has "
                f"{len(source.files)} files"
            )
        return source.files[0]
    if file not in source.files:
        raise ValueError(f"{source.name}: {file!r} is not in {source.files}")
    return file


def _fetch_hub(
    source: PinnedSource, file: str, root: Path, offline: bool
) -> Path:
    try:
        local = Path(
            hf_hub_download(
                repo_id=source.location,
                filename=file,
                repo_type="dataset",
                revision=source.revision,
                cache_dir=root / "hub",
                local_files_only=offline,
            )
        )
    except LocalEntryNotFoundError as error:
        if not offline:
            raise
        raise _offline_miss(source, file) from error
    expected = source.sha256.get(file)
    if expected is not None:
        _verify(local, expected, f"{source.name}/{file}")
    return local


def _fetch_url(
    source: PinnedSource, file: str, root: Path, offline: bool
) -> Path:
    expected = source.sha256[file]
    label = f"{source.name}/{file}"
    target = root / "url" / source.name / expected[:16] / file
    if target.is_file():
        _verify(target, expected, label)
        return target
    if offline:
        raise _offline_miss(source, file)
    url = source.location.rstrip("/") + "/" + file
    _download(url, target, expected, label)
    return target


def _offline_miss(source: PinnedSource, file: str) -> OfflineCacheMissError:
    return OfflineCacheMissError(
        f"{source.name}/{file} is not cached and offline=True"
    )


def _download(url: str, target: Path, expected: str, label: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "tomography"})
    with tempfile.NamedTemporaryFile(
        dir=target.parent,
        prefix=target.name + ".",
        suffix=".part",
        delete_on_close=False,
    ) as out:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as reply:
            shutil.copyfileobj(reply, out)
        out.close()
        _verify(Path(out.name), expected, label)
        os.replace(out.name, target)


def _verify(path: Path, expected: str, label: str) -> None:
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != expected:
        raise ChecksumError(
            f"{label}: sha256 {actual} does not match the pinned {expected}"
        )
