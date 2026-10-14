from collections.abc import Mapping
import dataclasses
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
from types import TracebackType
import urllib.request

import pytest

from tomography.datasets import sources
from tomography.datasets.sources import ChecksumError
from tomography.datasets.sources import OfflineCacheMissError
from tomography.datasets.sources import PinnedSource
from tomography.datasets.sources import cache_dir
from tomography.datasets.sources import fetch

REVISION = "0123456789abcdef0123456789abcdef01234567"
BASE_URL = "https://example.org/data/"
HUB_SOURCE = PinnedSource(
    name="demo",
    kind="hub",
    location="owner/demo",
    revision=REVISION,
    files=("data/train.parquet",),
    license="MIT",
    sha256={},
)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _url_source(
    files: Mapping[str, bytes],
    *,
    digests: Mapping[str, str] | None = None,
    location: str = BASE_URL,
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


class _Network:
    """Stands in for urllib.request.urlopen and records the URLs asked."""

    def __init__(self) -> None:
        self.payloads: dict[str, bytes] = {}
        self.urls: list[str] = []

    def __call__(
        self, request: urllib.request.Request, timeout: float | None = None
    ) -> io.BytesIO:
        self.urls.append(request.full_url)
        return io.BytesIO(self.payloads[request.full_url])


class _BrokenReply:
    """A reply that fails after a partial read."""

    def __enter__(self) -> "_BrokenReply":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        if not getattr(self, "sent", False):
            self.sent = True
            return b"partial"
        raise OSError("connection reset")


class _Hub:
    """Stands in for hf_hub_download and records the keyword arguments."""

    def __init__(self, served: Path) -> None:
        self.served = served
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        self.served.write_bytes(b"hub bytes")
        return str(self.served)


@pytest.fixture
def network(monkeypatch: pytest.MonkeyPatch) -> _Network:
    fake = _Network()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return fake


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Hub:
    fake = _Hub(tmp_path / "served.parquet")
    monkeypatch.setattr(sources, "hf_hub_download", fake)
    return fake


def _serve(
    network: _Network,
    files: Mapping[str, bytes],
    *,
    digests: Mapping[str, str] | None = None,
    location: str = BASE_URL,
) -> PinnedSource:
    for name, data in files.items():
        network.payloads[location.rstrip("/") + "/" + name] = data
    return _url_source(files, digests=digests, location=location)


def test_pinned_source_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        HUB_SOURCE.name = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    "revision",
    [None, "main", "0123456", REVISION.upper(), REVISION + "0"],
    ids=["missing", "branch", "short_sha", "uppercase", "too_long"],
)
def test_hub_source_rejects_an_unpinned_revision(revision: str | None) -> None:
    with pytest.raises(ValueError, match="revision"):
        dataclasses.replace(HUB_SOURCE, revision=revision)


def test_url_source_rejects_a_revision() -> None:
    source = _url_source({"a.csv": b"a"})

    with pytest.raises(ValueError, match="revision"):
        dataclasses.replace(source, revision=REVISION)


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
        dataclasses.replace(HUB_SOURCE, sha256={"other.parquet": "0" * 64})


@pytest.mark.parametrize(
    "file",
    [
        "../escape.csv",
        "/etc/passwd",
        "a/../../b.csv",
        "a\\b.csv",
        "",
        ".",
        "a/b/",
        "a b.csv",
        "a?x=1",
        "a#f",
        "a%2e%2e/b",
        "c:evil.csv",
        "a:b",
    ],
    ids=[
        "parent",
        "absolute",
        "nested_parent",
        "backslash",
        "empty",
        "dot",
        "trailing_slash",
        "space",
        "query",
        "fragment",
        "percent_escape",
        "drive_letter",
        "colon",
    ],
)
def test_source_rejects_files_that_leave_the_cache(file: str) -> None:
    with pytest.raises(ValueError, match="file"):
        dataclasses.replace(HUB_SOURCE, files=(file,))


@pytest.mark.parametrize(
    "file",
    [
        "data/train.parquet",
        "data/emotions/all_truncated_outputs.json",
        "data/emotions/happiness.json",
        "ethics.tar",
        "data/facts/facts_true_false.csv",
        "datasets/cities_cities_conj.csv",
        "data/train-00000-of-00001.parquet",
        "dataset/test.csv",
        "knowns_unknowns.jsonl",
        "nq_open/validation-00000-of-00001.parquet",
        "test.tsv",
        "data/00_dataset_SyntHal/SyntHal_prompts.jsonl",
        "rc.nocontext/validation-00000-of-00001.parquet",
    ],
)
def test_source_accepts_the_pinned_file_names(file: str) -> None:
    assert dataclasses.replace(HUB_SOURCE, files=(file,)).files == (file,)


@pytest.mark.parametrize(
    "name",
    ["", "a/b", "..", "with space"],
    ids=["empty", "slash", "dots", "space"],
)
def test_source_rejects_an_unsafe_name(name: str) -> None:
    with pytest.raises(ValueError, match="name"):
        dataclasses.replace(HUB_SOURCE, name=name)


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
    source = _serve(network, {"facts.csv": b"statement,label\n"})

    path = fetch(source, cache=tmp_path)

    assert path.read_bytes() == b"statement,label\n"
    assert network.urls == [BASE_URL + "facts.csv"]


def test_fetch_keeps_the_download_inside_the_cache(
    network: _Network, tmp_path: Path
) -> None:
    source = _serve(network, {"facts.csv": b"x"})
    cache = tmp_path / "cache"

    path = fetch(source, cache=cache)

    assert path.is_relative_to(cache)
    assert [p.name for p in tmp_path.iterdir()] == ["cache"]


@pytest.mark.parametrize(
    ("location", "file", "expected_url"),
    [
        (
            BASE_URL,
            "data/facts/true_false.csv",
            "https://example.org/data/data/facts/true_false.csv",
        ),
        ("https://example.org/data", "a.csv", "https://example.org/data/a.csv"),
    ],
    ids=["nested_file", "no_trailing_slash"],
)
def test_fetch_joins_the_file_to_the_base_url(
    network: _Network,
    tmp_path: Path,
    location: str,
    file: str,
    expected_url: str,
) -> None:
    source = _serve(network, {file: b"x"}, location=location)

    path = fetch(source, cache=tmp_path)

    assert path.name == file.rsplit("/", maxsplit=1)[-1]
    assert network.urls == [expected_url]


def test_fetch_serves_a_second_call_from_the_cache(
    network: _Network, tmp_path: Path
) -> None:
    source = _serve(network, {"a.csv": b"x"})

    first = fetch(source, cache=tmp_path)
    second = fetch(source, cache=tmp_path)

    assert first == second
    assert len(network.urls) == 1


def test_fetch_rejects_a_bad_download_and_leaves_nothing_cached(
    network: _Network, tmp_path: Path
) -> None:
    source = _serve(
        network,
        {"a.csv": b"tampered"},
        digests={"a.csv": _digest(b"expected")},
    )

    with pytest.raises(ChecksumError, match=r"a\.csv"):
        fetch(source, cache=tmp_path)

    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []


def test_fetch_leaves_nothing_in_the_cache_after_a_failed_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda request, timeout=None: _BrokenReply()
    )
    source = _url_source({"a.csv": b"x"})

    with pytest.raises(OSError, match="connection reset"):
        fetch(source, cache=tmp_path)

    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []


def test_fetch_rejects_a_corrupted_cache_hit(
    network: _Network, tmp_path: Path
) -> None:
    source = _serve(network, {"a.csv": b"expected"})
    path = fetch(source, cache=tmp_path)
    path.write_bytes(b"corrupted")

    with pytest.raises(ChecksumError, match=r"a\.csv"):
        fetch(source, cache=tmp_path)


def test_fetch_downloads_again_when_the_pinned_digest_changes(
    network: _Network, tmp_path: Path
) -> None:
    fetch(_serve(network, {"a.csv": b"old"}), cache=tmp_path)

    path = fetch(_serve(network, {"a.csv": b"new"}), cache=tmp_path)

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
    source = _serve(network, {"a.csv": b"x"})
    online = fetch(source, cache=tmp_path)

    offline = fetch(source, cache=tmp_path, offline=True)

    assert offline == online


def test_fetch_defaults_to_the_only_file(
    network: _Network, tmp_path: Path
) -> None:
    source = _serve(network, {"a.csv": b"x"})

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
    hub: _Hub, tmp_path: Path
) -> None:
    path = fetch(HUB_SOURCE, cache=tmp_path / "cache")

    assert path == hub.served
    assert len(hub.calls) == 1
    call = hub.calls[0]
    assert call["repo_id"] == "owner/demo"
    assert call["filename"] == "data/train.parquet"
    assert call["repo_type"] == "dataset"
    assert call["revision"] == REVISION
    cache_arg = call["cache_dir"]
    assert isinstance(cache_arg, Path)
    assert cache_arg.is_relative_to(tmp_path / "cache")


@pytest.mark.parametrize("offline", [False, True], ids=["online", "offline"])
def test_fetch_hub_passes_offline_through(
    hub: _Hub, tmp_path: Path, offline: bool
) -> None:
    fetch(HUB_SOURCE, cache=tmp_path, offline=offline)

    assert hub.calls[0]["local_files_only"] is offline


def test_fetch_hub_checks_a_declared_digest(hub: _Hub, tmp_path: Path) -> None:
    file = HUB_SOURCE.files[0]
    good = dataclasses.replace(HUB_SOURCE, sha256={file: _digest(b"hub bytes")})
    bad = dataclasses.replace(HUB_SOURCE, sha256={file: _digest(b"other")})

    assert fetch(good, cache=tmp_path) == hub.served
    with pytest.raises(ChecksumError, match=r"train\.parquet"):
        fetch(bad, cache=tmp_path)


def test_fetch_hub_offline_miss_raises_the_named_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(OfflineCacheMissError, match=r"data/train\.parquet"):
        fetch(HUB_SOURCE, cache=tmp_path, offline=True)


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
