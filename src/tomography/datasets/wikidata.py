"""Questions generated from Wikidata for the knowledge-boundary study.

Two kinds of item come from SPARQL queries against the public Wikidata Query
Service (CC0): award-winner questions whose year falls before or after a
model's training cutoff, and questions about a famous person's ordinary day
that no record can answer. Every query result is cached with the query text,
the retrieval date and the SHA-256 of the response: a later run reuses the
exact snapshot, and each item carries the date and digest that identify it.
"""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC
from datetime import date
from datetime import datetime
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import urllib.parse
import urllib.request

import numpy as np

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
_LABEL_SERVICE = (
    '  SERVICE wikibase:label { bd:serviceParam wikibase:language "en,mul". }'
)

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
UNKNOWABLE_TEMPLATES: tuple[str, ...] = (
    "What did {name} have for breakfast on {date}?",
    "Who was the first person {name} spoke to on {date}?",
    "What was the first thing {name} did on {date}?",
)
_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


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
        TimeoutError: If reading the response times out.
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
        json.JSONDecodeError: If the response is not JSON.
        TypeError: If the response is JSON but not an object.
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
        point in time. Labels are English, falling back to the
        language-neutral `mul` label.

    Raises:
        ValueError: If an award is not an item id or no award is given.
    """
    values = _award_values(awards)
    lines = [
        "SELECT ?prize ?prizeLabel ?year ?personLabel WHERE {",
        f"  VALUES ?prize {{ {values} }}",
        "  ?person p:P166 ?statement .",
        "  ?statement ps:P166 ?prize ; pq:P585 ?date .",
        "  BIND(YEAR(?date) AS ?year)",
        _LABEL_SERVICE,
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
        ValueError: If an award is not an item id or no award is given.
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


def famous_people_query(awards: Sequence[str], *, limit: int) -> str:
    """Builds the query for the most widely documented dead laureates.

    Fame is the sitelink count of a person's item (`wikibase:sitelinks`),
    which covers every Wikimedia site (Wikipedia editions, Commons,
    Wikisource and so on), not only Wikipedia. The pool is the people who
    won one of `awards`, because a query over every human with a sitelink
    count times out on the public service.

    Args:
        awards: Wikidata item ids of the awards, such as "Q38104".
        limit: Number of rows to ask for, the most linked people first.

    Returns:
        A SPARQL query with the variables person, personLabel, birth, death
        and links: people with a birth and a death date given to the day.
        Labels are English, falling back to the language-neutral `mul`
        label, which some people (Marie Curie) have instead of an English
        one.

    Raises:
        ValueError: If an award is not an item id, no award is given or
            `limit` is below 1.
    """
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    values = _award_values(awards)
    lines = [
        "SELECT DISTINCT ?person ?personLabel ?birth ?death ?links WHERE {",
        f"  VALUES ?award {{ {values} }}",
        "  ?person wdt:P166 ?award ;",
        "          wikibase:sitelinks ?links ;",
        "          p:P569 ?birthStatement ;",
        "          p:P570 ?deathStatement .",
        "  ?birthStatement a wikibase:BestRank .",
        "  ?birthStatement psv:P569 ?birthValue .",
        "  ?birthValue wikibase:timeValue ?birth ;",
        "              wikibase:timePrecision 11 .",
        "  ?deathStatement a wikibase:BestRank .",
        "  ?deathStatement psv:P570 ?deathValue .",
        "  ?deathValue wikibase:timeValue ?death ;",
        "              wikibase:timePrecision 11 .",
        _LABEL_SERVICE,
        "}",
        "ORDER BY DESC(?links) ?person",
        f"LIMIT {limit}",
    ]
    return "\n".join(lines) + "\n"


def load_unknowable_detail(
    *,
    seed: int,
    awards: Sequence[str] = AWARDS,
    limit: int = 300,
    templates: Sequence[str] = UNKNOWABLE_TEMPLATES,
    min_age_years: int = 10,
    min_birth_year: int = 1700,
    cache: Path | None = None,
    offline: bool = False,
    fetch: QueryFetch | None = None,
    today: Callable[[], date] | None = None,
) -> list[QAExample]:
    """Builds questions about a famous person's day that no record answers.

    Args:
        seed: Chooses each person's template and day.
        awards: Wikidata item ids of the awards that define the pool.
        limit: Number of most linked people to ask for.
        templates: Questions with "{name}" and "{date}" slots.
        min_age_years: Earliest age whose days may be asked about.
        min_birth_year: Earliest birth year kept.
        cache: Cache directory; defaults to the shared tomography cache.
        offline: If True, use only a cached query result.
        fetch: Returns the raw response for a query; defaults to
            `sparql_fetch`.
        today: Gives the retrieval date; defaults to today in UTC.

    Returns:
        The items `parse_unknowable_detail` builds from the query result.

    Raises:
        ValueError: If an award is not an item id, no award is given or
            `limit` is below 1.
        ChecksumError: If a cached result does not match its digest.
        OfflineCacheMissError: If `offline` is True and the query is not
            cached.
    """
    result = run_query(
        famous_people_query(awards, limit=limit),
        fetch=fetch,
        cache=cache,
        offline=offline,
        today=today,
    )
    return parse_unknowable_detail(
        result,
        seed=seed,
        templates=templates,
        min_age_years=min_age_years,
        min_birth_year=min_birth_year,
    )


def parse_unknowable_detail(
    result: QueryResult,
    *,
    seed: int,
    templates: Sequence[str] = UNKNOWABLE_TEMPLATES,
    min_age_years: int = 10,
    min_birth_year: int = 1700,
) -> list[QAExample]:
    """Turns person rows into questions about an ordinary day.

    Each person gets one seeded day between `min_age_years` after birth and
    the day before death, and one seeded template. The draw depends only on
    the seed and the person, so a person's item does not change when the
    query returns other people. The items carry no answer: nobody recorded
    what the person did that day. Rows are skipped for a label that is an
    item id, a birth before `min_birth_year` (calendar changes make older
    dates ambiguous), dates that do not parse, or a life shorter than
    `min_age_years`. A person with several date statements counts once.

    Args:
        result: Rows with "person" (an entity URI), "personLabel", "birth",
            "death" (ISO timestamps) and "links".
        seed: Chooses each person's template and day.
        templates: Questions with "{name}" and "{date}" slots.
        min_age_years: Earliest age whose days may be asked about.
        min_birth_year: Earliest birth year kept.

    Returns:
        One "unknowable_detail" item per person, ordered by item id number.
        Its `meta["popularity"]` is the item's sitelink count, not on the
        scale of PopQA's `s_pop`.
    """
    first_row: dict[int, Mapping[str, str]] = {}
    for row in sorted(result.rows, key=lambda r: (r["birth"], r["death"])):
        number = int(row["person"].rsplit("/Q", 1)[-1])
        first_row.setdefault(number, row)
    numbers = sorted(first_row)
    uids = [f"wikidata:detail:Q{number}" for number in numbers]
    splits = assign_splits(uids, dataset="wikidata")
    items = []
    for number, uid, split in zip(numbers, uids, splits, strict=True):
        row = first_row[number]
        name = row["personLabel"]
        span = _adult_span(row, min_age_years, min_birth_year)
        if _ITEM_ID.fullmatch(name) or span is None:
            continue
        start, days = span
        rng = np.random.default_rng([seed, number])
        day = start + timedelta(days=int(rng.integers(days)))
        template = templates[int(rng.integers(len(templates)))]
        text = f"{day.day} {_MONTHS[day.month - 1]} {day.year}"
        items.append(
            QAExample(
                uid=uid,
                question=template.format(name=name, date=text),
                answers=(),
                kind="unknowable_detail",
                split=split,
                meta={
                    "relation": "ordinary_date",
                    "subject": name,
                    "popularity": int(row["links"]),
                    "source": "wikidata",
                    "date": day.isoformat(),
                    "retrieved_on": result.retrieved_on,
                    "result_sha256": result.sha256,
                },
            )
        )
    return items


def _award_values(awards: Sequence[str]) -> str:
    if not awards:
        raise ValueError("awards must name at least one item id")
    return " ".join(f"wd:{_item_id(award)}" for award in awards)


def _item_id(value: str) -> str:
    if not _ITEM_ID.fullmatch(value):
        raise ValueError(f"not a Wikidata item id: {value!r}")
    return value


def _adult_span(
    row: Mapping[str, str], min_age_years: int, min_birth_year: int
) -> tuple[date, int] | None:
    try:
        birth = date.fromisoformat(row["birth"][:10])
        death = date.fromisoformat(row["death"][:10])
    except ValueError:
        return None
    if birth.year < min_birth_year:
        return None
    start = _add_years(birth, min_age_years)
    days = (death - start).days
    return (start, days) if days >= 1 else None


def _add_years(day: date, years: int) -> date:
    try:
        return day.replace(year=day.year + years)
    except ValueError:
        return day.replace(year=day.year + years, day=28)
