from datetime import date
import hashlib
import io
import json
from pathlib import Path
import urllib.request

import pytest

from tomography.datasets import wikidata
from tomography.datasets.sources import ChecksumError
from tomography.datasets.sources import OfflineCacheMissError


def _body(rows: list[dict[str, str]]) -> bytes:
    variables = sorted({key for row in rows for key in row})
    bindings = [
        {key: {"type": "literal", "value": value} for key, value in row.items()}
        for row in rows
    ]
    payload = {"head": {"vars": variables}, "results": {"bindings": bindings}}
    return json.dumps(payload).encode("utf-8")


class _Endpoint:
    """Stands in for the SPARQL endpoint and records the queries asked."""

    def __init__(self, body: bytes) -> None:
        self.body = body
        self.queries: list[str] = []

    def __call__(self, query: str) -> bytes:
        self.queries.append(query)
        return self.body


def test_parse_results_flattens_each_binding_to_variable_and_value() -> None:
    body = _body([{"a": "1", "b": "x"}, {"a": "2"}])

    rows = wikidata.parse_results(body)

    assert rows == ({"a": "1", "b": "x"}, {"a": "2"})


def test_run_query_records_query_date_digest_and_rows(tmp_path: Path) -> None:
    body = _body([{"a": "1"}])
    endpoint = _Endpoint(body)

    result = wikidata.run_query(
        "SELECT 1",
        fetch=endpoint,
        cache=tmp_path,
        today=lambda: date(2026, 10, 14),
    )

    assert result == wikidata.QueryResult(
        query="SELECT 1",
        retrieved_on="2026-10-14",
        sha256=hashlib.sha256(body).hexdigest(),
        rows=({"a": "1"},),
    )
    assert endpoint.queries == ["SELECT 1"]


def test_run_query_reuses_the_cached_snapshot(tmp_path: Path) -> None:
    first = _Endpoint(_body([{"a": "1"}]))
    again = _Endpoint(_body([{"a": "2"}]))

    one = wikidata.run_query(
        "SELECT 1", fetch=first, cache=tmp_path, today=lambda: date(2026, 1, 1)
    )
    two = wikidata.run_query(
        "SELECT 1", fetch=again, cache=tmp_path, today=lambda: date(2026, 2, 2)
    )

    assert two == one
    assert again.queries == []


def test_run_query_caches_each_query_apart(tmp_path: Path) -> None:
    endpoint = _Endpoint(_body([{"a": "1"}]))

    wikidata.run_query("SELECT 1", fetch=endpoint, cache=tmp_path)
    wikidata.run_query("SELECT 2", fetch=endpoint, cache=tmp_path)

    assert endpoint.queries == ["SELECT 1", "SELECT 2"]


def test_run_query_keeps_the_query_text_beside_the_cached_body(
    tmp_path: Path,
) -> None:
    body = _body([{"a": "1"}])

    wikidata.run_query(
        "SELECT 1",
        fetch=_Endpoint(body),
        cache=tmp_path,
        today=lambda: date(2026, 10, 14),
    )

    (folder,) = (tmp_path / "wikidata").iterdir()
    assert (folder / "result.json").read_bytes() == body
    assert json.loads((folder / "record.json").read_text("utf-8")) == {
        "query": "SELECT 1",
        "retrieved_on": "2026-10-14",
        "sha256": hashlib.sha256(body).hexdigest(),
    }


def test_run_query_rejects_a_cached_body_that_changed(tmp_path: Path) -> None:
    endpoint = _Endpoint(_body([{"a": "1"}]))
    wikidata.run_query("SELECT 1", fetch=endpoint, cache=tmp_path)
    (folder,) = (tmp_path / "wikidata").iterdir()
    (folder / "result.json").write_bytes(_body([{"a": "tampered"}]))

    with pytest.raises(ChecksumError, match=r"result.json"):
        wikidata.run_query("SELECT 1", fetch=endpoint, cache=tmp_path)


def test_run_query_offline_serves_a_hit_and_refuses_a_miss(
    tmp_path: Path,
) -> None:
    endpoint = _Endpoint(_body([{"a": "1"}]))
    cached = wikidata.run_query("SELECT 1", fetch=endpoint, cache=tmp_path)

    again = wikidata.run_query(
        "SELECT 1", fetch=endpoint, cache=tmp_path, offline=True
    )

    assert again == cached
    with pytest.raises(OfflineCacheMissError, match="offline"):
        wikidata.run_query(
            "SELECT 2", fetch=endpoint, cache=tmp_path, offline=True
        )
    assert endpoint.queries == ["SELECT 1"]


def test_run_query_does_not_cache_a_response_it_cannot_parse(
    tmp_path: Path,
) -> None:
    with pytest.raises(KeyError):
        wikidata.run_query(
            "SELECT 1", fetch=_Endpoint(b'{"error": 1}'), cache=tmp_path
        )

    assert not (tmp_path / "wikidata").exists()


def test_sparql_fetch_asks_the_endpoint_and_names_the_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[urllib.request.Request] = []

    def fake_urlopen(
        request: urllib.request.Request, timeout: float | None = None
    ) -> io.BytesIO:
        seen.append(request)
        return io.BytesIO(b"{}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    body = wikidata.sparql_fetch("SELECT ?x WHERE { ?x ?p ?o }")

    (request,) = seen
    assert body == b"{}"
    assert request.full_url == (
        "https://query.wikidata.org/sparql?query="
        "SELECT+%3Fx+WHERE+%7B+%3Fx+%3Fp+%3Fo+%7D"
    )
    headers = {k.lower(): v for k, v in request.header_items()}
    assert headers["accept"] == "application/sparql-results+json"
    assert "github.com/javierdejesusda/tomography" in headers["user-agent"]


def test_run_query_asks_the_endpoint_unless_told_otherwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = _body([{"a": "1"}])
    monkeypatch.setattr(wikidata, "sparql_fetch", lambda query: body)

    result = wikidata.run_query("SELECT 1", cache=tmp_path)

    assert result.rows == ({"a": "1"},)
