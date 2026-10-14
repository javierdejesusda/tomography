from collections.abc import Mapping
import dataclasses
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

import pytest

from tomography.datasets import sources
from tomography.datasets.sources import ChecksumError
from tomography.datasets.sources import OfflineCacheMissError
from tomography.datasets.sources import PinnedSource
from tomography.datasets.sources import cache_dir
from tomography.datasets.sources import fetch

REVISION = "0123456789abcdef0123456789abcdef01234567"


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _url_source(
    files: Mapping[str, bytes],
    *,
    digests: Mapping[str, str] | None = None,
    location: str = "https://example.org/data/",
) -> PinnedSource:
    return PinnedSource(
        name="demo",
        kind="url",
        location=location,
        revision=None,
        files=tuple(files),
        license="MIT",
        sha256=digests or {k: _digest(v) for k, v in files.items()},
    )


def _hub_source(**overrides: object) -> PinnedSource:
    fields: dict[str, object] = {
        "name": "demo",
        "kind": "hub",
        "location": "owner/demo",
        "revision": REVISION,
        "files": ("data/train.parquet",),
        "license": "MIT",
        "sha256": {},
    }
    fields.update(overrides)
    return PinnedSource(**fields)  # type: ignore[arg-type]


class _Network:
    """Stands in for urllib.request.urlopen and records the URLs asked."""

    def __init__(self, payloads: Mapping[str, bytes]) -> None:
        self.payloads = payloads
        self.urls: list[str] = []

    def __call__(
        self, request: urllib.request.Request, timeout: float | None = None
    ) -> io.BytesIO:
        self.urls.append(request.full_url)
        return io.BytesIO(self.payloads[request.full_url])


@pytest.fixture
def network(monkeypatch: pytest.MonkeyPatch) -> _Network:
    fake = _Network({})
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return fake


def test_pinned_source_is_frozen() -> None:
    source = _hub_source()

    with pytest.raises(dataclasses.FrozenInstanceError):
        source.name = "other"  # type: ignore[misc]


def test_hub_source_accepts_a_full_commit_sha() -> None:
    assert _hub_source().revision == REVISION


@pytest.mark.parametrize(
    "revision",
    [None, "main", "0123456", REVISION.upper(), REVISION + "0"],
    ids=["missing", "branch", "short_sha", "uppercase", "too_long"],
)
def test_hub_source_rejects_an_unpinned_revision(revision: str | None) -> None:
    with pytest.raises(ValueError, match="revision"):
        _hub_source(revision=revision)


def test_url_source_requires_a_digest_for_every_file() -> None:
    with pytest.raises(ValueError, match="sha256"):
        _url_source({"a.csv": b"a", "b.csv": b"b"}, digests={"a.csv": "0" * 64})


def test_url_source_rejects_a_malformed_digest() -> None:
    with pytest.raises(ValueError, match="sha256"):
        _url_source({"a.csv": b"a"}, digests={"a.csv": "abc"})


def test_url_source_requires_https() -> None:
    with pytest.raises(ValueError, match="https"):
        _url_source({"a.csv": b"a"}, location="http://example.org/")


def test_url_source_requires_at_least_one_file() -> None:
    with pytest.raises(ValueError, match="files"):
        _url_source({})


def test_source_rejects_a_digest_for_an_unknown_file() -> None:
    with pytest.raises(ValueError, match="sha256"):
        _hub_source(sha256={"other.parquet": "0" * 64})


@pytest.mark.parametrize(
    "file",
    ["../escape.csv", "/etc/passwd", "a/../../b.csv", "a\\b.csv", ""],
    ids=["parent", "absolute", "nested_parent", "backslash", "empty"],
)
def test_source_rejects_files_that_leave_the_cache(file: str) -> None:
    with pytest.raises(ValueError, match="file"):
        _hub_source(files=(file,))


@pytest.mark.parametrize(
    "name",
    ["", "a/b", "..", "with space"],
    ids=["empty", "slash", "dots", "space"],
)
def test_source_rejects_an_unsafe_name(name: str) -> None:
    with pytest.raises(ValueError, match="name"):
        _hub_source(name=name)


def test_cache_dir_follows_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TOMOGRAPHY_CACHE", str(tmp_path / "mine"))

    assert cache_dir() == tmp_path / "mine"


@pytest.mark.parametrize("value", [None, ""], ids=["unset", "empty"])
def test_cache_dir_defaults_to_a_folder_under_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("TOMOGRAPHY_CACHE", raising=False)
    else:
        monkeypatch.setenv("TOMOGRAPHY_CACHE", value)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    assert cache_dir() == tmp_path / ".cache" / "tomography"


def test_fetch_downloads_the_pinned_file(
    network: _Network, tmp_path: Path
) -> None:
    source = _url_source({"facts.csv": b"statement,label\n"})
    network.payloads = {
        "https://example.org/data/facts.csv": b"statement,label\n"
    }

    path = fetch(source, cache=tmp_path)

    assert path.read_bytes() == b"statement,label\n"
    assert network.urls == ["https://example.org/data/facts.csv"]


def test_fetch_keeps_the_download_inside_the_cache(
    network: _Network, tmp_path: Path
) -> None:
    source = _url_source({"facts.csv": b"x"})
    network.payloads = {"https://example.org/data/facts.csv": b"x"}
    cache = tmp_path / "cache"

    path = fetch(source, cache=cache)

    assert path.is_relative_to(cache)
    assert [p.name for p in tmp_path.iterdir()] == ["cache"]


def test_fetch_joins_a_nested_file_to_the_base_url(
    network: _Network, tmp_path: Path
) -> None:
    source = _url_source({"data/facts/true_false.csv": b"x"})
    network.payloads = {
        "https://example.org/data/data/facts/true_false.csv": b"x"
    }

    path = fetch(source, cache=tmp_path)

    assert path.name == "true_false.csv"
    assert network.urls == [
        "https://example.org/data/data/facts/true_false.csv"
    ]


def test_fetch_joins_a_base_url_without_a_trailing_slash(
    network: _Network, tmp_path: Path
) -> None:
    source = _url_source({"a.csv": b"x"}, location="https://example.org/data")
    network.payloads = {"https://example.org/data/a.csv": b"x"}

    fetch(source, cache=tmp_path)

    assert network.urls == ["https://example.org/data/a.csv"]


def test_fetch_serves_a_second_call_from_the_cache(
    network: _Network, tmp_path: Path
) -> None:
    source = _url_source({"a.csv": b"x"})
    network.payloads = {"https://example.org/data/a.csv": b"x"}

    first = fetch(source, cache=tmp_path)
    second = fetch(source, cache=tmp_path)

    assert first == second
    assert len(network.urls) == 1


def test_fetch_rejects_a_download_with_the_wrong_digest(
    network: _Network, tmp_path: Path
) -> None:
    source = _url_source({"a.csv": b"expected"})
    network.payloads = {"https://example.org/data/a.csv": b"tampered"}

    with pytest.raises(ChecksumError, match=r"a\.csv"):
        fetch(source, cache=tmp_path)


def test_fetch_leaves_nothing_in_the_cache_after_a_bad_download(
    network: _Network, tmp_path: Path
) -> None:
    source = _url_source({"a.csv": b"expected"})
    network.payloads = {"https://example.org/data/a.csv": b"tampered"}

    with pytest.raises(ChecksumError):
        fetch(source, cache=tmp_path)

    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []


def test_fetch_rejects_a_corrupted_cache_hit(
    network: _Network, tmp_path: Path
) -> None:
    source = _url_source({"a.csv": b"expected"})
    network.payloads = {"https://example.org/data/a.csv": b"expected"}
    path = fetch(source, cache=tmp_path)
    path.write_bytes(b"corrupted")

    with pytest.raises(ChecksumError, match=r"a\.csv"):
        fetch(source, cache=tmp_path)


def test_fetch_downloads_again_when_the_pinned_digest_changes(
    network: _Network, tmp_path: Path
) -> None:
    url = "https://example.org/data/a.csv"
    network.payloads = {url: b"old"}
    fetch(_url_source({"a.csv": b"old"}), cache=tmp_path)
    network.payloads = {url: b"new"}

    path = fetch(_url_source({"a.csv": b"new"}), cache=tmp_path)

    assert path.read_bytes() == b"new"


def test_fetch_offline_raises_on_a_cache_miss(
    network: _Network, tmp_path: Path
) -> None:
    source = _url_source({"a.csv": b"x"})

    with pytest.raises(OfflineCacheMissError, match=r"a\.csv"):
        fetch(source, cache=tmp_path, offline=True)

    assert network.urls == []


def test_fetch_offline_returns_a_cached_file(
    network: _Network, tmp_path: Path
) -> None:
    source = _url_source({"a.csv": b"x"})
    network.payloads = {"https://example.org/data/a.csv": b"x"}
    online = fetch(source, cache=tmp_path)

    offline = fetch(source, cache=tmp_path, offline=True)

    assert offline == online


def test_fetch_defaults_to_the_only_file(
    network: _Network, tmp_path: Path
) -> None:
    source = _url_source({"a.csv": b"x"})
    network.payloads = {"https://example.org/data/a.csv": b"x"}

    assert fetch(source, file="a.csv", cache=tmp_path) == fetch(
        source, cache=tmp_path
    )


def test_fetch_needs_a_file_name_when_there_are_several(
    tmp_path: Path,
) -> None:
    source = _url_source({"a.csv": b"a", "b.csv": b"b"})

    with pytest.raises(ValueError, match="file"):
        fetch(source, cache=tmp_path)


def test_fetch_rejects_a_file_the_source_does_not_declare(
    tmp_path: Path,
) -> None:
    source = _url_source({"a.csv": b"a"})

    with pytest.raises(ValueError, match=r"other\.csv"):
        fetch(source, file="other.csv", cache=tmp_path)


def test_fetch_asks_the_hub_for_the_pinned_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []
    served = tmp_path / "served.parquet"

    def fake_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(served)

    monkeypatch.setattr(sources, "hf_hub_download", fake_download)

    path = fetch(_hub_source(), cache=tmp_path / "cache", offline=True)

    assert path == served
    assert len(calls) == 1
    assert calls[0]["repo_id"] == "owner/demo"
    assert calls[0]["filename"] == "data/train.parquet"
    assert calls[0]["repo_type"] == "dataset"
    assert calls[0]["revision"] == REVISION
    assert calls[0]["local_files_only"] is True
    cache_arg = calls[0]["cache_dir"]
    assert isinstance(cache_arg, Path)
    assert cache_arg.is_relative_to(tmp_path / "cache")


def test_fetch_hub_is_online_unless_offline_is_asked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[object] = []

    def fake_download(**kwargs: object) -> str:
        seen.append(kwargs["local_files_only"])
        return str(tmp_path / "f")

    monkeypatch.setattr(sources, "hf_hub_download", fake_download)

    fetch(_hub_source(), cache=tmp_path)

    assert seen == [False]


def test_fetch_hub_offline_miss_raises_the_named_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(OfflineCacheMissError, match=r"data/train\.parquet"):
        fetch(_hub_source(), cache=tmp_path, offline=True)


def test_importing_sources_disables_the_symlink_warning() -> None:
    script = (
        "import huggingface_hub.constants as c\n"
        "assert c.HF_HUB_DISABLE_SYMLINKS_WARNING is False\n"
        "import tomography.datasets.sources\n"
        "assert c.HF_HUB_DISABLE_SYMLINKS_WARNING is True\n"
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if k != "HF_HUB_DISABLE_SYMLINKS_WARNING"
    }

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
