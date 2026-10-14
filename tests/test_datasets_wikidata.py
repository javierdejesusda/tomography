from datetime import date
import hashlib
import io
import json
from pathlib import Path
import urllib.request

import pytest

from tomography.datasets import wikidata
from tomography.datasets.records import QAExample
from tomography.datasets.records import assign_splits
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


def test_run_query_does_not_cache_a_body_that_is_not_json(
    tmp_path: Path,
) -> None:
    with pytest.raises(json.JSONDecodeError):
        wikidata.run_query(
            "SELECT 1", fetch=_Endpoint(b"<html>"), cache=tmp_path
        )

    assert not (tmp_path / "wikidata").exists()


def test_run_query_does_not_cache_a_body_that_is_not_an_object(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError):
        wikidata.run_query("SELECT 1", fetch=_Endpoint(b"[]"), cache=tmp_path)

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


PHYSICS = "http://www.wikidata.org/entity/Q38104"
LITERATURE = "http://www.wikidata.org/entity/Q37922"
DIGEST = "d" * 64


def _winner(prize: str, label: str, year: int, person: str) -> dict[str, str]:
    return {
        "prize": prize,
        "prizeLabel": label,
        "year": str(year),
        "personLabel": person,
    }


def _result(rows: list[dict[str, str]]) -> wikidata.QueryResult:
    return wikidata.QueryResult(
        query="SELECT 1",
        retrieved_on="2026-10-14",
        sha256=DIGEST,
        rows=tuple(rows),
    )


def test_award_winners_query_names_each_award_and_the_year_qualifier() -> None:
    query = wikidata.award_winners_query(("Q38104", "Q37922"))

    assert "VALUES ?prize { wd:Q38104 wd:Q37922 }" in query
    assert "p:P166" in query
    assert "pq:P585" in query


def test_award_winners_query_rejects_anything_but_an_item_id() -> None:
    with pytest.raises(ValueError, match="DROP"):
        wikidata.award_winners_query(("Q1 } DROP",))


def test_award_winners_query_needs_at_least_one_award() -> None:
    with pytest.raises(ValueError, match="at least one"):
        wikidata.award_winners_query(())


def test_default_awards_are_eight_distinct_item_ids() -> None:
    assert len(set(wikidata.AWARDS)) == len(wikidata.AWARDS) == 8
    wikidata.award_winners_query(wikidata.AWARDS)


def test_parse_award_winners_builds_one_item_per_prize_and_year() -> None:
    rows = [
        _winner(PHYSICS, "Nobel Prize in Physics", 2016, "Duncan Haldane"),
        _winner(PHYSICS, "Nobel Prize in Physics", 2016, "David J. Thouless"),
        _winner(LITERATURE, "Nobel Prize in Literature", 2016, "Bob Dylan"),
    ]

    items = wikidata.parse_award_winners(_result(rows), cutoff_year=2017)

    assert [i.uid for i in items] == [
        "wikidata:award:Q37922:2016",
        "wikidata:award:Q38104:2016",
    ]
    physics = items[1]
    assert physics.question == "Who won the Nobel Prize in Physics in 2016?"
    assert physics.answers == ("David J. Thouless", "Duncan Haldane")
    assert physics.kind == "real"
    assert physics.meta == {
        "relation": "Nobel Prize in Physics",
        "subject": "Nobel Prize in Physics",
        "year": 2016,
        "cutoff_year": 2017,
        "popularity": None,
        "source": "wikidata",
        "retrieved_on": "2026-10-14",
        "result_sha256": DIGEST,
    }


def test_parse_award_winners_marks_the_years_after_the_cutoff() -> None:
    rows = [
        _winner(PHYSICS, "Nobel Prize in Physics", year, "Someone Famous")
        for year in (2016, 2017, 2018)
    ]

    items = wikidata.parse_award_winners(_result(rows), cutoff_year=2017)

    assert [(i.meta["year"], i.kind) for i in items] == [
        (2016, "real"),
        (2017, "real"),
        (2018, "post_cutoff"),
    ]


def test_parse_award_winners_skips_years_before_the_first_year() -> None:
    rows = [
        _winner(PHYSICS, "Nobel Prize in Physics", year, "Someone Famous")
        for year in (1989, 1990, 1991)
    ]

    items = wikidata.parse_award_winners(
        _result(rows), cutoff_year=2017, first_year=1990
    )

    assert [i.meta["year"] for i in items] == [1990, 1991]


def test_parse_award_winners_drops_laureates_without_an_english_label() -> None:
    rows = [
        _winner(PHYSICS, "Nobel Prize in Physics", 2000, "Q7186"),
        _winner(PHYSICS, "Nobel Prize in Physics", 2000, "Ada Smith"),
        _winner(PHYSICS, "Nobel Prize in Physics", 2001, "Q123"),
    ]

    items = wikidata.parse_award_winners(_result(rows), cutoff_year=2017)

    assert [(i.meta["year"], i.answers) for i in items] == [
        (2000, ("Ada Smith",))
    ]


def test_parse_award_winners_template_is_a_keyword() -> None:
    rows = [_winner(PHYSICS, "Nobel Prize in Physics", 2000, "Ada Smith")]

    (item,) = wikidata.parse_award_winners(
        _result(rows), cutoff_year=2017, template="In {year}, who won {award}?"
    )

    assert item.question == "In 2000, who won Nobel Prize in Physics?"


def test_load_post_cutoff_without_awards_asks_nothing_and_caches_nothing(
    tmp_path: Path,
) -> None:
    endpoint = _Endpoint(_body([]))

    with pytest.raises(ValueError, match="at least one"):
        wikidata.load_post_cutoff(
            cutoff_year=2017, awards=(), cache=tmp_path, fetch=endpoint
        )

    assert endpoint.queries == []
    assert not (tmp_path / "wikidata").exists()


def test_load_post_cutoff_runs_the_award_query_through_the_cache(
    tmp_path: Path,
) -> None:
    rows = [_winner(PHYSICS, "Nobel Prize in Physics", 2019, "Jim Peebles")]
    endpoint = _Endpoint(_body(rows))

    items = wikidata.load_post_cutoff(
        cutoff_year=2017,
        awards=("Q38104",),
        cache=tmp_path,
        fetch=endpoint,
        today=lambda: date(2026, 10, 14),
    )

    assert endpoint.queries == [wikidata.award_winners_query(("Q38104",))]
    assert [(i.uid, i.kind, i.answers) for i in items] == [
        ("wikidata:award:Q38104:2019", "post_cutoff", ("Jim Peebles",))
    ]
    assert items[0].meta["retrieved_on"] == "2026-10-14"


def _person(
    qid: str = "Q937",
    label: str = "Albert Einstein",
    birth: str = "1879-03-14T00:00:00Z",
    death: str = "1955-04-18T00:00:00Z",
    links: int = 322,
) -> dict[str, str]:
    return {
        "person": f"http://www.wikidata.org/entity/{qid}",
        "personLabel": label,
        "birth": birth,
        "death": death,
        "links": str(links),
    }


def _day(item: QAExample) -> date:
    return date.fromisoformat(str(item.meta["date"]))


def test_famous_people_query_asks_for_the_top_dead_laureates() -> None:
    query = wikidata.famous_people_query(("Q38104", "Q37922"), limit=50)

    assert "VALUES ?award { wd:Q38104 wd:Q37922 }" in query
    assert "wdt:P166 ?award" in query
    assert "wikibase:timePrecision 11" in query
    assert "ORDER BY DESC(?links)" in query
    assert query.rstrip().endswith("LIMIT 50")


def test_famous_people_query_rejects_anything_but_item_ids_and_counts() -> None:
    with pytest.raises(ValueError, match="DROP"):
        wikidata.famous_people_query(("Q1 } DROP",), limit=5)
    with pytest.raises(ValueError, match="limit"):
        wikidata.famous_people_query(("Q1",), limit=0)


def test_famous_people_query_keeps_only_the_best_ranked_dates() -> None:
    query = wikidata.famous_people_query(("Q38104",), limit=5)

    assert "  ?birthStatement a wikibase:BestRank ." in query
    assert "  ?deathStatement a wikibase:BestRank ." in query


def test_famous_people_query_needs_at_least_one_award() -> None:
    with pytest.raises(ValueError, match="at least one"):
        wikidata.famous_people_query((), limit=5)


def test_parse_unknowable_detail_builds_an_unanswerable_item() -> None:
    (item,) = wikidata.parse_unknowable_detail(_result([_person()]), seed=0)

    assert item.uid == "wikidata:detail:Q937"
    assert item.answers == ()
    assert item.kind == "unknowable_detail"
    assert item.split == assign_splits([item.uid], dataset="wikidata")[0]
    assert "Albert Einstein" in item.question
    day = _day(item)
    assert f"{day.day} " in item.question
    assert str(day.year) in item.question
    assert item.meta == {
        "relation": "ordinary_date",
        "subject": "Albert Einstein",
        "popularity": 322,
        "source": "wikidata",
        "date": day.isoformat(),
        "retrieved_on": "2026-10-14",
        "result_sha256": DIGEST,
    }


def test_parse_unknowable_detail_picks_a_day_inside_the_adult_life() -> None:
    rows = [_person(f"Q{n}", f"Person {n}") for n in range(1, 40)]

    items = wikidata.parse_unknowable_detail(_result(rows), seed=3)

    assert len(items) == 39
    for item in items:
        assert date(1889, 3, 14) <= _day(item) < date(1955, 4, 18)


def test_parse_unknowable_detail_is_deterministic_in_the_seed() -> None:
    rows = [_person(f"Q{n}", f"Person {n}") for n in range(1, 30)]

    first = wikidata.parse_unknowable_detail(_result(rows), seed=1)
    again = wikidata.parse_unknowable_detail(_result(rows), seed=1)
    other = wikidata.parse_unknowable_detail(_result(rows), seed=2)

    assert first == again
    assert [i.question for i in first] != [i.question for i in other]


def test_parse_unknowable_detail_draws_each_person_independently() -> None:
    a = _person("Q1", "Ada Smith")
    b = _person("Q2", "Ben Jones")

    both = wikidata.parse_unknowable_detail(_result([a, b]), seed=5)
    only_b = wikidata.parse_unknowable_detail(_result([b]), seed=5)

    assert both[1] == only_b[0]


def test_parse_unknowable_detail_skips_rows_it_cannot_use() -> None:
    rows = [
        _person("Q1", "Q1"),
        _person("Q2", "Old Master", birth="1650-01-01T00:00:00Z"),
        _person(
            "Q3",
            "Child Star",
            birth="1950-01-01T00:00:00Z",
            death="1955-01-01T00:00:00Z",
        ),
        _person(
            "Q4",
            "Ancient",
            birth="-0400-01-01T00:00:00Z",
            death="-0350-01-01T00:00:00Z",
        ),
        _person("Q5", "Good Person"),
    ]

    items = wikidata.parse_unknowable_detail(_result(rows), seed=0)

    assert [i.meta["subject"] for i in items] == ["Good Person"]


def test_parse_unknowable_detail_keeps_one_item_per_person() -> None:
    rows = [
        _person("Q7241", "Rabindranath Tagore", birth="1861-05-07T00:00:00Z"),
        _person("Q7241", "Rabindranath Tagore", birth="1861-05-06T00:00:00Z"),
    ]

    items = wikidata.parse_unknowable_detail(_result(rows), seed=0)

    assert [i.uid for i in items] == ["wikidata:detail:Q7241"]


def test_parse_unknowable_detail_templates_and_age_are_keywords() -> None:
    (item,) = wikidata.parse_unknowable_detail(
        _result([_person()]),
        seed=0,
        templates=("Where was {name} on {date}?",),
        min_age_years=40,
    )

    assert item.question.startswith("Where was Albert Einstein on ")
    assert _day(item) >= date(1919, 3, 14)


def test_load_unknowable_detail_runs_the_people_query_through_the_cache(
    tmp_path: Path,
) -> None:
    endpoint = _Endpoint(_body([_person()]))

    items = wikidata.load_unknowable_detail(
        seed=0,
        awards=("Q38104",),
        limit=10,
        cache=tmp_path,
        fetch=endpoint,
        today=lambda: date(2026, 10, 14),
    )

    expected = wikidata.famous_people_query(("Q38104",), limit=10)
    assert endpoint.queries == [expected]
    assert [i.uid for i in items] == ["wikidata:detail:Q937"]
    assert items[0].meta["retrieved_on"] == "2026-10-14"


@pytest.mark.network
def test_network_award_winners_straddle_the_gpt2_cutoff(tmp_path: Path) -> None:
    items = wikidata.load_post_cutoff(cutoff_year=2017, cache=tmp_path)

    kinds = {kind: 0 for kind in ("real", "post_cutoff")}
    for item in items:
        kinds[item.kind] += 1
    assert kinds["real"] > 100
    assert kinds["post_cutoff"] >= 8
    physics = next(i for i in items if i.uid == "wikidata:award:Q38104:2019")
    assert any("Peebles" in name for name in physics.answers)
    assert len(str(physics.meta["result_sha256"])) == 64
    offline = wikidata.load_post_cutoff(
        cutoff_year=2017, cache=tmp_path, offline=True
    )
    assert offline == items


@pytest.mark.network
def test_network_famous_people_give_unanswerable_items(tmp_path: Path) -> None:
    items = wikidata.load_unknowable_detail(seed=0, limit=100, cache=tmp_path)

    assert len(items) > 50
    subjects = {str(i.meta["subject"]) for i in items}
    assert "Albert Einstein" in subjects
    assert all(not i.answers for i in items)


@pytest.mark.network
def test_network_famous_people_appear_once_per_row(tmp_path: Path) -> None:
    query = wikidata.famous_people_query(wikidata.AWARDS, limit=100)

    result = wikidata.run_query(query, cache=tmp_path)

    people = {row["person"] for row in result.rows}
    assert len(people) == len(result.rows)
