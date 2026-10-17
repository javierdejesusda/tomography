from datetime import date
import json
from pathlib import Path

import pytest

from tomography.datasets import quantities
from tomography.datasets.quantities import Family
from tomography.datasets.quantities import Side
from tomography.datasets.sources import ChecksumError
from tomography.datasets.sources import OfflineCacheMissError
from tomography.datasets.wikidata import QueryResult


def _side(**overrides: object) -> Side:
    fields: dict[str, object] = {
        "kind": "range",
        "prop": "P2048",
        "classes": ("Q39367",),
        "template": "How tall is a {entity}, in centimetres?",
        "unit": "centimetres",
        "scale": 100.0,
    }
    fields.update(overrides)
    return Side(**fields)  # type: ignore[arg-type]


def test_side_query_asks_for_normalized_best_rank_values() -> None:
    query = quantities.side_query(_side(), limit=50)

    assert "p:P2048 ?st" in query
    assert "?st a wikibase:BestRank" in query
    assert "psn:P2048 ?v" in query
    assert "wikibase:quantityAmount ?amount" in query
    assert "wikibase:quantityLowerBound ?lower" in query
    assert "wikibase:quantityUpperBound ?upper" in query
    assert "wd:Q39367" in query
    assert "wikibase:sitelinks ?links" in query


def test_side_query_bounds_are_optional() -> None:
    query = quantities.side_query(_side(), limit=50)

    assert "OPTIONAL { ?v wikibase:quantityLowerBound" in query


def test_side_query_orders_by_links_then_item_and_limits() -> None:
    query = quantities.side_query(_side(), limit=50)

    assert "ORDER BY DESC(?links) ?item" in query
    assert query.rstrip().endswith("LIMIT 50")


def test_side_query_labels_in_english_with_mul_fallback() -> None:
    assert 'wikibase:language "en,mul"' in quantities.side_query(
        _side(), limit=5
    )


def test_side_query_lists_every_class() -> None:
    query = quantities.side_query(_side(classes=("Q11303", "Q12518")), limit=5)

    assert "wd:Q11303 wd:Q12518" in query


def test_side_query_asks_for_the_scientific_name_only_for_taxa() -> None:
    plain = quantities.side_query(_side(), limit=5)
    taxon = quantities.side_query(_side(taxon=True), limit=5)

    assert "P225" not in plain
    assert "OPTIONAL { ?item wdt:P225 ?taxonName }" in taxon


@pytest.mark.parametrize("prop", ["2048", "Q2048", "P2048; DROP", ""])
def test_side_query_rejects_a_bad_property(prop: str) -> None:
    with pytest.raises(ValueError, match="property"):
        quantities.side_query(_side(prop=prop), limit=5)


@pytest.mark.parametrize("bad", ["39367", "P31", "Q1 Q2", ""])
def test_side_query_rejects_a_bad_class(bad: str) -> None:
    with pytest.raises(ValueError, match="item id"):
        quantities.side_query(_side(classes=(bad,)), limit=5)


def test_side_query_rejects_no_class_and_a_small_limit() -> None:
    with pytest.raises(ValueError, match="class"):
        quantities.side_query(_side(classes=()), limit=5)
    with pytest.raises(ValueError, match="limit"):
        quantities.side_query(_side(), limit=0)


def test_side_query_filters_by_links_only_when_asked() -> None:
    plain = quantities.side_query(_side(), limit=5)
    floor = quantities.side_query(_side(query_min_sitelinks=20), limit=5)

    assert "FILTER" not in plain
    assert "FILTER(?links >= 20)" in floor


def test_side_rejects_a_negative_query_floor() -> None:
    with pytest.raises(ValueError, match="query_min_sitelinks"):
        _side(query_min_sitelinks=-1)


def test_side_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValueError, match="kind"):
        _side(kind="point")


def test_default_families_pair_a_range_side_with_an_exact_side() -> None:
    assert quantities.FAMILIES
    names = [family.name for family in quantities.FAMILIES]
    assert len(set(names)) == len(names)
    assert {"wingspan", "duration"} <= set(names)
    for family in quantities.FAMILIES:
        assert isinstance(family, Family)
        assert family.range_side.kind == "range"
        assert family.exact_side.kind == "exact"
        assert "{entity}" in family.range_side.template
        assert "{entity}" in family.exact_side.template


RANGE = _side()
EXACT = _side(
    kind="exact",
    classes=("Q11303",),
    template="How tall is the {entity}, in metres?",
    unit="metres",
    scale=1.0,
)
TAXON = _side(
    prop="P3063",
    classes=("Q16521",),
    template="How long is the gestation period of the {entity}, in days?",
    unit="days",
    scale=1 / 86400,
    taxon=True,
)


def _row(
    number: int,
    *,
    label: str | None = None,
    links: int = 20,
    statement: str | None = None,
    amount: float = 0.5,
    lower: float | None = 0.4,
    upper: float | None = 0.6,
    taxon_name: str | None = None,
) -> dict[str, str]:
    row = {
        "item": f"http://www.wikidata.org/entity/Q{number}",
        "itemLabel": f"Thing {number}" if label is None else label,
        "links": str(links),
        "st": (
            "http://www.wikidata.org/entity/statement/"
            f"Q{number}-{statement or 'a'}"
        ),
        "amount": str(amount),
    }
    if lower is not None:
        row["lower"] = str(lower)
    if upper is not None:
        row["upper"] = str(upper)
    if taxon_name is not None:
        row["taxonName"] = taxon_name
    return row


def _result(
    rows: list[dict[str, str]], *, on: str = "2026-10-17"
) -> QueryResult:
    return QueryResult(
        "SELECT", on, "ab" * 32, tuple(dict(row) for row in rows)
    )


def _parse(
    rows: list[dict[str, str]],
    side: Side = RANGE,
    *,
    min_sitelinks: int = 10,
    seed: int = 0,
) -> list[quantities.QuantityQuestion]:
    return quantities.parse_side(
        _result(rows),
        family="height",
        side=side,
        min_sitelinks=min_sitelinks,
        seed=seed,
    )


def test_parse_side_builds_a_question_with_provenance() -> None:
    (item,) = _parse([_row(7, label="Beagle", links=33)])

    assert item.uid == "height-range-Q7"
    assert item.family == "height"
    assert item.kind == "range"
    assert item.entity == "Beagle"
    assert item.entity_id == "Q7"
    assert item.question == "How tall is a Beagle, in centimetres?"
    assert item.unit == "centimetres"
    assert item.sitelinks == 33
    assert item.split in {"train", "val", "test"}
    assert item.retrieved_on == "2026-10-17"
    assert item.sha256 == "ab" * 32


def test_parse_side_converts_the_normalized_value_to_the_question_unit() -> (
    None
):
    (item,) = _parse([_row(7, amount=0.5, lower=0.4, upper=0.6)])

    assert (item.low, item.high) == (40.0, 60.0)


def test_parse_side_converts_seconds_to_days_and_rounds() -> None:
    day = 86400.0
    row = _row(
        7,
        amount=63 * day,
        lower=58 * day,
        upper=66.123456 * day,
        taxon_name="Canis lupus",
        label="wolf",
    )

    (item,) = _parse([row], TAXON)

    assert (item.low, item.high) == (58.0, 66.12)


def test_parse_side_drops_an_entity_with_two_best_rank_statements() -> None:
    rows = [
        _row(1, statement="a"),
        _row(1, statement="b", amount=0.9, lower=0.8, upper=1.0),
        _row(2),
    ]

    assert [i.entity_id for i in _parse(rows)] == ["Q2"]


def test_parse_side_counts_one_statement_once_across_duplicate_rows() -> None:
    rows = [
        _row(1, taxon_name="Canis lupus", label="wolf"),
        _row(1, taxon_name="Canis lupus lupus", label="wolf"),
    ]

    assert [i.entity_id for i in _parse(rows, TAXON)] == ["Q1"]


def test_parse_side_drops_entities_below_the_sitelink_floor() -> None:
    rows = [_row(1, links=9), _row(2, links=10)]

    assert [i.entity_id for i in _parse(rows, min_sitelinks=10)] == ["Q2"]


@pytest.mark.parametrize("label", ["Q5", "Q123456"])
def test_parse_side_drops_an_entity_labelled_by_its_id(label: str) -> None:
    assert _parse([_row(5, label=label)]) == []


def test_parse_side_drops_a_taxon_labelled_by_its_scientific_name() -> None:
    rows = [
        _row(1, label="Canis lupus", taxon_name="canis lupus"),
        _row(2, label="wolf", taxon_name="Canis lupus"),
        _row(3, label="Felis catus", taxon_name="Felis catus"),
    ]

    assert [i.entity_id for i in _parse(rows, TAXON)] == ["Q2"]


def test_parse_side_keeps_a_taxon_without_a_scientific_name() -> None:
    assert [i.entity_id for i in _parse([_row(1, label="ox")], TAXON)] == ["Q1"]


def test_parse_side_range_needs_bounds() -> None:
    rows = [_row(1, lower=None, upper=None), _row(2, lower=0.4, upper=None)]

    assert _parse(rows) == []


def test_parse_side_range_needs_lower_below_upper() -> None:
    rows = [
        _row(1, amount=0.5, lower=0.5, upper=0.5),
        _row(2, amount=0.5, lower=0.6, upper=0.4),
    ]

    assert _parse(rows) == []


def test_parse_side_range_needs_the_amount_inside_the_bounds() -> None:
    rows = [
        _row(1, amount=0.9, lower=0.4, upper=0.6),
        _row(2, amount=0.3, lower=0.4, upper=0.6),
        _row(3, amount=0.4, lower=0.4, upper=0.6),
        _row(4, amount=0.6, lower=0.4, upper=0.6),
    ]

    assert [i.entity_id for i in _parse(rows)] == ["Q3", "Q4"]


def test_parse_side_range_needs_a_relative_width_of_ten_percent() -> None:
    assert quantities.MIN_RANGE_RELATIVE_WIDTH == 0.10
    rows = [
        _row(1, amount=1.0, lower=0.96, upper=1.04),
        _row(2, amount=1.0, lower=0.95, upper=1.05),
    ]

    assert [i.entity_id for i in _parse(rows)] == ["Q2"]


def test_parse_side_exact_keeps_an_unbounded_statement_as_a_point() -> None:
    (item,) = _parse([_row(1, amount=381.5, lower=None, upper=None)], EXACT)

    assert (item.low, item.high) == (381.5, 381.5)
    assert item.kind == "exact"
    assert item.uid == "height-exact-Q1"


def test_parse_side_exact_keeps_precision_bounds_and_uses_the_amount() -> None:
    assert quantities.MAX_EXACT_RELATIVE_HALF_WIDTH == 0.01
    rows = [
        _row(1, amount=100.0, lower=99.0, upper=101.0),
        _row(2, amount=100.0, lower=98.9, upper=101.1),
        _row(3, amount=100.0, lower=90.0, upper=110.0),
    ]

    items = _parse(rows, EXACT)

    assert [i.entity_id for i in items] == ["Q1"]
    assert (items[0].low, items[0].high) == (100.0, 100.0)


def test_parse_side_orders_by_links_then_entity_id() -> None:
    rows = [
        _row(9, links=30),
        _row(3, links=50),
        _row(2, links=30),
    ]

    assert [i.entity_id for i in _parse(rows)] == ["Q3", "Q2", "Q9"]


def test_parse_side_splits_depend_on_family_uid_and_seed() -> None:
    rows = [_row(n) for n in range(1, 61)]

    first = [i.split for i in _parse(rows, seed=0)]
    again = [i.split for i in _parse(rows, seed=0)]
    other = [i.split for i in _parse(rows, seed=1)]

    assert first == again
    assert first != other
    assert set(first) == {"train", "val", "test"}
    assert 24 <= first.count("train") <= 48


def test_parse_side_splits_do_not_depend_on_the_other_rows() -> None:
    rows = [_row(n) for n in range(1, 21)]

    alone = _parse(rows[4:5])
    among = _parse(rows)

    assert alone[0].split == next(i.split for i in among if i.entity_id == "Q5")


def _body(rows: list[dict[str, str]]) -> bytes:
    bindings = [
        {key: {"type": "literal", "value": value} for key, value in row.items()}
        for row in rows
    ]
    return json.dumps(
        {"head": {"vars": []}, "results": {"bindings": bindings}}
    ).encode("utf-8")


class _Endpoint:
    """Answers each side's query from canned rows and records the queries."""

    def __init__(self, bodies: dict[str, bytes]) -> None:
        self.bodies = bodies
        self.queries: list[str] = []

    def __call__(self, query: str) -> bytes:
        self.queries.append(query)
        for marker, body in self.bodies.items():
            if marker in query:
                return body
        raise AssertionError(f"unexpected query: {query}")


def _family(name: str, range_class: str, exact_class: str) -> Family:
    return Family(
        name=name,
        range_side=_side(classes=(range_class,)),
        exact_side=_side(
            kind="exact",
            classes=(exact_class,),
            template="How tall is the {entity}, in metres?",
            unit="metres",
            scale=1.0,
        ),
    )


def _range_rows(count: int, start: int = 100) -> list[dict[str, str]]:
    return [
        _row(start + n, links=1000 - n, amount=0.5, lower=0.4, upper=0.6)
        for n in range(count)
    ]


def _exact_rows(count: int, start: int = 500) -> list[dict[str, str]]:
    return [
        _row(start + n, links=900 - n, amount=50.0 + n, lower=None, upper=None)
        for n in range(count)
    ]


def _endpoint(range_count: int, exact_count: int) -> _Endpoint:
    return _Endpoint(
        {
            "wd:Q1001": _body(_range_rows(range_count)),
            "wd:Q1002": _body(_exact_rows(exact_count)),
        }
    )


FAMILY = _family("height", "Q1001", "Q1002")


def _load(
    fetch: _Endpoint, tmp_path: Path, **kwargs: object
) -> list[quantities.QuantityQuestion]:
    options: dict[str, object] = {
        "families": (FAMILY,),
        "n_per_side": 5,
        "min_sitelinks": 10,
        "cache": tmp_path,
        "fetch": fetch,
        "today": lambda: date(2026, 10, 17),
    }
    options.update(kwargs)
    return quantities.load_quantities(**options)  # type: ignore[arg-type]


def test_load_quantities_keeps_the_most_linked_n_per_side(
    tmp_path: Path,
) -> None:
    items = _load(_endpoint(8, 8), tmp_path)

    by_kind = {
        kind: sorted(i.entity_id for i in items if i.kind == kind)
        for kind in ("range", "exact")
    }
    assert by_kind["range"] == [f"Q{100 + n}" for n in range(5)]
    assert by_kind["exact"] == [f"Q{500 + n}" for n in range(5)]


def test_load_quantities_balances_to_the_smaller_side(tmp_path: Path) -> None:
    items = _load(_endpoint(8, 3), tmp_path)

    assert sum(i.kind == "range" for i in items) == 3
    assert sum(i.kind == "exact" for i in items) == 3
    assert {i.entity_id for i in items if i.kind == "range"} == {
        "Q100",
        "Q101",
        "Q102",
    }


def test_load_quantities_balances_after_the_filters(tmp_path: Path) -> None:
    rows = _range_rows(8)
    for row in rows[:4]:
        row["links"] = "3"
    fetch = _Endpoint(
        {"wd:Q1001": _body(rows), "wd:Q1002": _body(_exact_rows(8))}
    )

    items = _load(fetch, tmp_path)

    assert sum(i.kind == "range" for i in items) == 4
    assert sum(i.kind == "exact" for i in items) == 4


def test_load_quantities_returns_items_sorted_by_uid(tmp_path: Path) -> None:
    items = _load(_endpoint(8, 8), tmp_path)

    assert [i.uid for i in items] == sorted(i.uid for i in items)
    assert len({i.uid for i in items}) == len(items)


def test_load_quantities_asks_for_four_times_n_rows_per_side(
    tmp_path: Path,
) -> None:
    fetch = _endpoint(8, 8)

    _load(fetch, tmp_path, n_per_side=7)

    assert len(fetch.queries) == 2
    assert all(q.rstrip().endswith("LIMIT 28") for q in fetch.queries)


def test_load_quantities_covers_every_family(tmp_path: Path) -> None:
    other = _family("duration", "Q2001", "Q2002")
    fetch = _Endpoint(
        {
            "wd:Q1001": _body(_range_rows(6)),
            "wd:Q1002": _body(_exact_rows(6)),
            "wd:Q2001": _body(_range_rows(6, start=700)),
            "wd:Q2002": _body(_exact_rows(6, start=900)),
        }
    )

    items = _load(fetch, tmp_path, families=(FAMILY, other))

    assert {i.family for i in items} == {"height", "duration"}
    assert len(items) == 20


def test_load_quantities_splits_are_deterministic_in_the_seed(
    tmp_path: Path,
) -> None:
    fetch = _endpoint(30, 30)

    first = _load(fetch, tmp_path, n_per_side=30, seed=0)
    again = _load(fetch, tmp_path, n_per_side=30, seed=0)
    other = _load(fetch, tmp_path, n_per_side=30, seed=1)

    assert [i.split for i in first] == [i.split for i in again]
    assert [i.split for i in first] != [i.split for i in other]


def test_load_quantities_records_the_snapshot(tmp_path: Path) -> None:
    items = _load(_endpoint(8, 8), tmp_path)

    assert {i.retrieved_on for i in items} == {"2026-10-17"}
    assert all(len(i.sha256) == 64 for i in items)


def test_load_quantities_reuses_the_cache_and_works_offline(
    tmp_path: Path,
) -> None:
    fetch = _endpoint(8, 8)
    first = _load(fetch, tmp_path)

    again = _load(fetch, tmp_path, offline=True)

    assert again == first
    assert len(fetch.queries) == 2


def test_load_quantities_offline_miss_propagates(tmp_path: Path) -> None:
    with pytest.raises(OfflineCacheMissError):
        _load(_endpoint(8, 8), tmp_path, offline=True)


def test_load_quantities_changed_cache_propagates(tmp_path: Path) -> None:
    _load(_endpoint(8, 8), tmp_path)
    for body in (tmp_path / "wikidata").glob("*/result.json"):
        body.write_bytes(body.read_bytes() + b" ")

    with pytest.raises(ChecksumError):
        _load(_endpoint(8, 8), tmp_path)


def test_load_quantities_rejects_a_bad_n_and_no_families(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="n_per_side"):
        _load(_endpoint(8, 8), tmp_path, n_per_side=0)
    with pytest.raises(ValueError, match="families"):
        _load(_endpoint(8, 8), tmp_path, families=())
