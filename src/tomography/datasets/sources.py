"""Pinned data sources: Hub files at a commit SHA and checksummed URL files.

Every loader module declares one `PinnedSource` row and reads its files through
`fetch`, so a download is tied to an exact revision or digest and nothing is
written outside the cache directory.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from pathlib import PurePosixPath
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
_TIMEOUT_SECONDS = 60


class ChecksumError(ValueError):
    """A file does not match its pinned SHA-256 digest."""


class OfflineCacheMissError(FileNotFoundError):
    """A pinned file is not in the cache and downloads are disabled."""


@dataclass(frozen=True)
class PinnedSource:
    """One row of the data-source license table.

    Attributes:
        name: Short identifier, also a folder name inside the cache.
        kind: "hub" for a dataset repository on the Hugging Face Hub, "url"
            for plain files under an https base URL.
        location: The repository id for "hub", the base URL for "url".
        revision: Full 40-character commit SHA for "hub", None for "url".
        files: Relative paths `fetch` serves. A hub source loaded through
            `datasets.load_dataset` may leave this empty.
        license: License of the data, as shown in the README table.
        sha256: Hex digest per file; required for every "url" file.

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
            if self.revision is None or not _COMMIT_SHA.fullmatch(
                self.revision
            ):
                raise ValueError(
                    f"{self.name}: hub revision must be a full 40-character "
                    f"lowercase commit SHA, got {self.revision!r}"
                )
        elif self.kind == "url":
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
        else:
            raise ValueError(f"{self.name}: unknown kind {self.kind!r}")


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

    Hub files come from `hf_hub_download` at the pinned commit SHA. URL files
    are checked against their SHA-256 digest on every call, cache hits
    included.

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
        ChecksumError: If a URL file does not match its pinned digest.
    """
    name = _resolve_file(source, file)
    root = cache_dir() if cache is None else cache
    if source.kind == "hub":
        return _fetch_hub(source, name, root, offline)
    return _fetch_url(source, name, root, offline)


def _is_safe_relative_path(file: str) -> bool:
    if not file or "\\" in file or ":" in file:
        return False
    path = PurePosixPath(file)
    return not path.is_absolute() and ".." not in path.parts


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
        local = hf_hub_download(
            repo_id=source.location,
            filename=file,
            repo_type="dataset",
            revision=source.revision,
            cache_dir=root / "hub",
            local_files_only=offline,
        )
    except LocalEntryNotFoundError as error:
        if not offline:
            raise
        raise OfflineCacheMissError(
            f"{source.name}/{file} is not cached and offline=True"
        ) from error
    return Path(local)


def _fetch_url(
    source: PinnedSource, file: str, root: Path, offline: bool
) -> Path:
    expected = source.sha256[file]
    target = root / "url" / source.name / expected[:16] / file
    if target.is_file():
        actual = _sha256_of(target)
        if actual != expected:
            raise ChecksumError(
                f"{source.name}/{file}: cached sha256 {actual} does not match "
                f"the pinned {expected}; delete {target} to download it again"
            )
        return target
    if offline:
        raise OfflineCacheMissError(
            f"{source.name}/{file} is not cached and offline=True"
        )
    url = source.location.rstrip("/") + "/" + file
    _download(url, target, expected, label=f"{source.name}/{file}")
    return target


def _download(url: str, target: Path, expected: str, *, label: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=target.parent, prefix=target.name + ".", suffix=".part"
    )
    temp = Path(temp_name)
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "tomography"}
        )
        with (
            os.fdopen(descriptor, "wb") as out,
            urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as reply,
        ):
            shutil.copyfileobj(reply, out)
        actual = _sha256_of(temp)
        if actual != expected:
            raise ChecksumError(
                f"{label}: downloaded sha256 {actual} does not match the "
                f"pinned {expected}"
            )
        os.replace(temp, target)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def _sha256_of(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
