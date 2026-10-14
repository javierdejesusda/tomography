"""Cached SPARQL queries against Wikidata for the knowledge-boundary study.

Queries go to the public Wikidata Query Service (CC0). Every result is
cached with the query text, the retrieval date and the SHA-256 of the
response: a later run reuses the exact snapshot, and callers can record
the date and digest that identify it.
"""

from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC
from datetime import date
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.parse
import urllib.request

from tomography.datasets.sources import ChecksumError
from tomography.datasets.sources import OfflineCacheMissError
from tomography.datasets.sources import cache_dir

ENDPOINT = "https://query.wikidata.org/sparql"
_USER_AGENT = "tomography/0.1 (https://github.com/javierdejesusda/tomography)"
_TIMEOUT_SECONDS = 90
_BODY = "result.json"
_RECORD = "record.json"

QueryFetch = Callable[[str], bytes]


@dataclass(frozen=True)
class QueryResult:
    """One SPARQL result with its provenance.

    Attributes:
        query: The query text that produced the result.
        retrieved_on: ISO date the response was first retrieved.
        sha256: Hex digest of the raw response body.
        rows: One mapping per binding, from variable name to value.
    """

    query: str
    retrieved_on: str
    sha256: str
    rows: tuple[Mapping[str, str], ...]


def parse_results(body: bytes) -> tuple[Mapping[str, str], ...]:
    """Flattens a SPARQL JSON response into plain rows.

    Args:
        body: The raw `application/sparql-results+json` response.

    Returns:
        One mapping per binding, from variable name to its string value;
        variables left unbound in a binding are absent.
    """
    bindings = json.loads(body)["results"]["bindings"]
    return tuple(
        {name: cell["value"] for name, cell in binding.items()}
        for binding in bindings
    )


def sparql_fetch(query: str) -> bytes:
    """Asks the public Wikidata Query Service for one query's results.

    Args:
        query: The SPARQL query text.

    Returns:
        The raw `application/sparql-results+json` response body.

    Raises:
        urllib.error.URLError: If the service cannot be reached, answers with
            an error, or times out.
    """
    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": query})
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/sparql-results+json",
            "User-Agent": _USER_AGENT,
        },
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as reply:
        body: bytes = reply.read()
    return body


def run_query(
    query: str,
    *,
    fetch: QueryFetch | None = None,
    cache: Path | None = None,
    offline: bool = False,
    today: Callable[[], date] | None = None,
) -> QueryResult:
    """Runs a SPARQL query and records where the answer came from.

    The response is cached under the query's digest together with the query
    text and the retrieval date. A later call returns that snapshot, after
    checking the body against its recorded SHA-256, and does not call `fetch`.

    Args:
        query: The SPARQL query text.
        fetch: Returns the raw response body for a query; defaults to
            `sparql_fetch`.
        cache: Cache directory; defaults to the shared tomography cache.
        offline: If True, never call `fetch`.
        today: Gives the retrieval date; defaults to today in UTC.

    Returns:
        The rows with the query, retrieval date and response digest.

    Raises:
        ChecksumError: If a cached body does not match its recorded digest.
        OfflineCacheMissError: If `offline` is True and the query is not
            cached.
        KeyError: If the response is not a SPARQL JSON result.
    """
    root = cache_dir() if cache is None else cache
    key = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    folder = root / "wikidata" / key
    if (folder / _RECORD).is_file():
        return _read_snapshot(query, folder)
    if offline:
        raise OfflineCacheMissError(
            f"query {key} is not cached and offline=True"
        )
    body = (fetch or sparql_fetch)(query)
    rows = parse_results(body)
    stamp = (today or _utc_today)().isoformat()
    digest = hashlib.sha256(body).hexdigest()
    folder.mkdir(parents=True, exist_ok=True)
    _write_atomic(folder / _BODY, body)
    record = {"query": query, "retrieved_on": stamp, "sha256": digest}
    _write_atomic(folder / _RECORD, json.dumps(record).encode("utf-8"))
    return QueryResult(query, stamp, digest, rows)


def _read_snapshot(query: str, folder: Path) -> QueryResult:
    record = json.loads((folder / _RECORD).read_text(encoding="utf-8"))
    body = (folder / _BODY).read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    if digest != record["sha256"]:
        raise ChecksumError(
            f"{folder / _BODY}: sha256 {digest} does not match the recorded "
            f"{record['sha256']}; delete {folder} to query again"
        )
    return QueryResult(
        query, record["retrieved_on"], digest, parse_results(body)
    )


def _utc_today() -> date:
    return datetime.now(UTC).date()


def _write_atomic(path: Path, data: bytes) -> None:
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=path.name + ".", suffix=".part"
    )
    temp = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as out:
            out.write(data)
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
