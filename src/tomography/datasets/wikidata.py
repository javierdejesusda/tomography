"""Questions generated from Wikidata for the knowledge-boundary study.

Award-winner questions whose year falls before or after a model's
training cutoff come from SPARQL queries against the public Wikidata
Query Service (CC0). Every query result is cached with the query text,
the retrieval date and the SHA-256 of the response: a later run reuses
the exact snapshot, and each item carries the date and digest that
identify it.
"""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC
from datetime import date
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import urllib.parse
import urllib.request

from tomography.datasets.records import QAExample
from tomography.datasets.records import assign_splits
from tomography.datasets.sources import ChecksumError
from tomography.datasets.sources import OfflineCacheMissError
from tomography.datasets.sources import cache_dir

ENDPOINT = "https://query.wikidata.org/sparql"
_USER_AGENT = "tomography/0.1 (https://github.com/javierdejesusda/tomography)"
_TIMEOUT_SECONDS = 90
_ITEM_ID = re.compile(r"Q\d+")
_BODY = "result.json"
_RECORD = "record.json"

QueryFetch = Callable[[str], bytes]

AWARDS: tuple[str, ...] = (
    "Q38104",
    "Q44585",
    "Q80061",
    "Q37922",
    "Q35637",
    "Q47170",
    "Q185667",
    "Q28835",
)
AWARD_TEMPLATE = "Who won the {award} in {year}?"


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


def award_winners_query(awards: Sequence[str]) -> str:
    """Builds the query for the winners of annual awards, with their year.

    Args:
        awards: Wikidata item ids of the awards, such as "Q38104".

    Returns:
        A SPARQL query with the variables prize, prizeLabel, year and
        personLabel: one row per laureate whose award statement carries a
        point in time.

    Raises:
        ValueError: If an award is not an item id.
    """
    values = " ".join(f"wd:{_item_id(award)}" for award in awards)
    lines = [
        "SELECT ?prize ?prizeLabel ?year ?personLabel WHERE {",
        f"  VALUES ?prize {{ {values} }}",
        "  ?person p:P166 ?statement .",
        "  ?statement ps:P166 ?prize ; pq:P585 ?date .",
        "  BIND(YEAR(?date) AS ?year)",
        '  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }',
        "}",
    ]
    return "\n".join(lines) + "\n"


def load_post_cutoff(
    *,
    cutoff_year: int,
    awards: Sequence[str] = AWARDS,
    first_year: int = 1990,
    template: str = AWARD_TEMPLATE,
    cache: Path | None = None,
    offline: bool = False,
    fetch: QueryFetch | None = None,
    today: Callable[[], date] | None = None,
) -> list[QAExample]:
    """Builds award-winner questions that straddle a training cutoff.

    Args:
        cutoff_year: Last year the model's training data covers.
        awards: Wikidata item ids of the awards to ask about.
        first_year: First year to ask about.
        template: Question with "{award}" and "{year}" slots.
        cache: Cache directory; defaults to the shared tomography cache.
        offline: If True, use only a cached query result.
        fetch: Returns the raw response for a query; defaults to
            `sparql_fetch`.
        today: Gives the retrieval date; defaults to today in UTC.

    Returns:
        The items `parse_award_winners` builds from the query result.

    Raises:
        ValueError: If an award is not an item id.
        ChecksumError: If a cached result does not match its digest.
        OfflineCacheMissError: If `offline` is True and the query is not
            cached.
    """
    result = run_query(
        award_winners_query(awards),
        fetch=fetch,
        cache=cache,
        offline=offline,
        today=today,
    )
    return parse_award_winners(
        result,
        cutoff_year=cutoff_year,
        first_year=first_year,
        template=template,
    )


def parse_award_winners(
    result: QueryResult,
    *,
    cutoff_year: int,
    first_year: int = 1990,
    template: str = AWARD_TEMPLATE,
) -> list[QAExample]:
    """Turns award-winner rows into questions where only the year changes.

    Years up to `cutoff_year` give "real" items and later years give
    "post_cutoff" items; both are asked with the same template, so surface
    form does not tell them apart. The answers are every laureate of that
    prize in that year. Laureates with no English label (Wikidata then
    returns the item id) are dropped.

    Args:
        result: Rows with "prize" (an entity URI), "prizeLabel", "year" and
            "personLabel".
        cutoff_year: Last year the model's training data covers.
        first_year: First year to keep.
        template: Question with "{award}" and "{year}" slots.

    Returns:
        One item per prize and year, ordered by prize id and year.
    """
    winners: dict[tuple[str, int], tuple[str, set[str]]] = {}
    for row in result.rows:
        year = int(row["year"])
        person = row["personLabel"]
        if year < first_year or _ITEM_ID.fullmatch(person):
            continue
        award_id = row["prize"].rsplit("/", 1)[-1]
        _, people = winners.setdefault(
            (award_id, year), (row["prizeLabel"], set())
        )
        people.add(person)
    keys = sorted(winners)
    uids = [f"wikidata:award:{award_id}:{year}" for award_id, year in keys]
    splits = assign_splits(uids, dataset="wikidata")
    items = []
    for (award_id, year), uid, split in zip(keys, uids, splits, strict=True):
        label, people = winners[(award_id, year)]
        items.append(
            QAExample(
                uid=uid,
                question=template.format(award=label, year=year),
                answers=tuple(sorted(people)),
                kind="post_cutoff" if year > cutoff_year else "real",
                split=split,
                meta={
                    "relation": label,
                    "subject": label,
                    "year": year,
                    "cutoff_year": cutoff_year,
                    "popularity": None,
                    "source": "wikidata",
                    "retrieved_on": result.retrieved_on,
                    "result_sha256": result.sha256,
                },
            )
        )
    return items


def _item_id(value: str) -> str:
    if not _ITEM_ID.fullmatch(value):
        raise ValueError(f"not a Wikidata item id: {value!r}")
    return value
