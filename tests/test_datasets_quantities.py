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


def test_side_query_asks_for_original_best_rank_values_with_their_unit() -> (
    None
):
    query = quantities.side_query(_side(), limit=50)

    assert "p:P2048 ?st" in query
    assert "?st a wikibase:BestRank" in query
    assert "psv:P2048 ?v" in query
    assert "psn:" not in query
    assert "wikibase:quantityAmount ?amount" in query
    assert "wikibase:quantityUnit ?unit" in query
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


def test_side_query_excludes_classes_only_when_asked() -> None:
    plain = quantities.side_query(_side(), limit=5)
    excl = quantities.side_query(
        _side(exclude_classes=("Q15056993", "Q1")), limit=5
    )

    assert "NOT EXISTS" not in plain
    assert "FILTER NOT EXISTS" in excl
    assert "VALUES ?excluded { wd:Q15056993 wd:Q1 }" in excl
    assert "?item wdt:P31 ?excluded" in excl


def test_side_query_rejects_a_bad_excluded_class() -> None:
    with pytest.raises(ValueError, match="item id"):
        quantities.side_query(_side(exclude_classes=("x",)), limit=5)


def test_side_rejects_an_unknown_dimension_and_an_exact_union() -> None:
    with pytest.raises(ValueError, match="dimension"):
        _side(dimension="mass")
    with pytest.raises(ValueError, match="union"):
        _side(kind="exact", union_statements=True)


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
    dimension="time",
)
DOG = _side(union_statements=True)

METRE = "http://www.wikidata.org/entity/Q11573"
SECOND = "http://www.wikidata.org/entity/Q11574"
DAY = "http://www.wikidata.org/entity/Q573"


def _unit(qid: str) -> str:
    return f"http://www.wikidata.org/entity/{qid}"


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
    unit: str | None = None,
) -> dict[str, str]:
    default_unit = METRE if taxon_name is None else DAY
    row = {
        "item": f"http://www.wikidata.org/entity/Q{number}",
        "itemLabel": f"Thing {number}" if label is None else label,
        "links": str(links),
        "st": (
            "http://www.wikidata.org/entity/statement/"
            f"Q{number}-{statement or 'a'}"
        ),
        "amount": str(amount),
        "unit": default_unit if unit is None else unit,
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
        unit=SECOND,
    )

    (item,) = _parse([row], TAXON)

    assert (item.low, item.high) == (58.0, 66.12)


@pytest.mark.parametrize(
    ("qid", "factor"),
    [
        ("Q11573", 1.0),
        ("Q174728", 0.01),
        ("Q174789", 0.001),
        ("Q828224", 1000.0),
        ("Q3710", 0.3048),
        ("Q218593", 0.0254),
    ],
)
def test_parse_side_converts_each_length_unit_to_the_question_unit(
    qid: str, factor: float
) -> None:
    row = _row(1, amount=100.0, lower=80.0, upper=120.0, unit=_unit(qid))

    (item,) = _parse([row])

    expected = (round(80 * factor * 100, 2), round(120 * factor * 100, 2))
    assert (item.low, item.high) == expected


@pytest.mark.parametrize(
    ("qid", "lower", "upper", "days"),
    [
        ("Q11574", 5_184_000.0, 6_048_000.0, (60.0, 70.0)),
        ("Q7727", 100_800.0, 115_200.0, (70.0, 80.0)),
        ("Q25235", 1_920.0, 2_160.0, (80.0, 90.0)),
        ("Q573", 90.0, 100.0, (90.0, 100.0)),
        ("Q23387", 14.0, 16.0, (98.0, 112.0)),
        ("Q5151", 4.0, 5.0, (121.75, 152.19)),
        ("Q577", 1.0, 2.0, (365.25, 730.5)),
    ],
)
def test_parse_side_converts_each_time_unit_to_days(
    qid: str, lower: float, upper: float, days: tuple[float, float]
) -> None:
    row = _row(
        1,
        amount=(lower + upper) / 2,
        lower=lower,
        upper=upper,
        unit=_unit(qid),
        taxon_name="x",
    )

    (item,) = _parse([row], TAXON)

    assert (item.low, item.high) == days


def test_parse_side_drops_a_range_that_rounds_to_one_value() -> None:
    rows = [
        _row(1, amount=100.0, lower=80.0, upper=120.0, unit=SECOND),
        _row(2, amount=4320.0, lower=3974.4, upper=4665.6, unit=SECOND),
        _row(
            3,
            amount=5_616_000.0,
            lower=5_184_000.0,
            upper=6_048_000.0,
            unit=SECOND,
        ),
    ]

    assert [i.entity_id for i in _parse(rows, TAXON)] == ["Q3"]


def test_parse_side_drops_a_union_that_rounds_to_one_value() -> None:
    rows = [
        _row(1, statement="a", amount=0.00001, lower=None, upper=None),
        _row(1, statement="b", amount=0.00002, lower=None, upper=None),
    ]

    assert _parse(rows, DOG) == []


def test_parse_side_keeps_the_original_bounds_exactly() -> None:
    row = _row(
        1,
        amount=130.0,
        lower=97.0,
        upper=163.0,
        unit=_unit("Q573"),
        taxon_name="x",
    )

    (item,) = _parse([row], TAXON)

    assert (item.low, item.high) == (97.0, 163.0)


@pytest.mark.parametrize("unit", ["Q99999", "Q573", "", "not a uri"])
def test_parse_side_drops_a_statement_in_an_unknown_unit(unit: str) -> None:
    assert _parse([_row(1, unit=unit)]) == []


def test_parse_side_drops_a_statement_without_a_unit() -> None:
    row = _row(1)
    del row["unit"]

    assert _parse([row]) == []


def test_parse_side_counts_a_statement_in_an_unknown_unit_as_ambiguity() -> (
    None
):
    rows = [
        _row(1, statement="a"),
        _row(1, statement="b", unit=_unit("Q99999")),
    ]

    assert _parse(rows) == []


def test_parse_side_exact_converts_feet() -> None:
    row = _row(1, amount=1000.0, lower=None, upper=None, unit=_unit("Q3710"))

    (item,) = _parse([row], EXACT)

    assert (item.low, item.high) == (304.8, 304.8)


def test_parse_side_union_spans_every_statement() -> None:
    rows = [
        _row(1, statement="a", amount=0.60, lower=None, upper=None),
        _row(1, statement="b", amount=0.50, lower=0.48, upper=0.52),
        _row(1, statement="c", amount=0.66, lower=0.64, upper=0.68),
    ]

    (item,) = _parse(rows, DOG)

    assert (item.low, item.high) == (48.0, 68.0)


def test_parse_side_union_needs_ten_percent_of_the_midpoint() -> None:
    wide = [
        _row(1, statement="a", amount=0.50, lower=None, upper=None),
        _row(1, statement="b", amount=0.60, lower=None, upper=None),
    ]
    narrow = [
        _row(2, statement="a", amount=0.50, lower=None, upper=None),
        _row(2, statement="b", amount=0.52, lower=None, upper=None),
    ]

    assert [i.entity_id for i in _parse(wide + narrow, DOG)] == ["Q1"]


def test_parse_side_union_takes_a_single_bounded_statement() -> None:
    (item,) = _parse([_row(1, amount=0.5, lower=0.4, upper=0.6)], DOG)

    assert (item.low, item.high) == (40.0, 60.0)


def test_parse_side_union_drops_an_entity_with_any_unusable_statement() -> None:
    rows = [
        _row(1, statement="a", amount=0.5, lower=None, upper=None),
        _row(1, statement="b", amount=0.7, lower=None, upper=None, unit="x"),
        _row(2, statement="a", amount=0.5, lower=0.6, upper=0.7),
        _row(2, statement="b", amount=0.9, lower=None, upper=None),
        _row(3, statement="a", amount=0.0, lower=None, upper=None),
        _row(3, statement="b", amount=0.9, lower=None, upper=None),
    ]

    assert _parse(rows, DOG) == []


def test_parse_side_union_still_filters_popularity_and_labels() -> None:
    rows = [
        _row(1, links=2, statement="a", amount=0.3, lower=None, upper=None),
        _row(1, links=2, statement="b", amount=0.6, lower=None, upper=None),
        _row(2, label="Q2", statement="a", amount=0.3, lower=None, upper=None),
        _row(2, label="Q2", statement="b", amount=0.6, lower=None, upper=None),
    ]

    assert _parse(rows, DOG) == []


def test_parse_side_drops_the_entity_a_full_limit_may_have_cut() -> None:
    rows = [
        _row(1, links=30),
        _row(2, links=20, statement="a"),
        _row(2, links=20, statement="b"),
        _row(3, links=10, statement="a"),
    ]
    result = _result(rows)

    kept = quantities.parse_side(
        result,
        family="height",
        side=RANGE,
        min_sitelinks=1,
        row_limit=4,
    )
    all_kept = quantities.parse_side(
        result,
        family="height",
        side=RANGE,
        min_sitelinks=1,
        row_limit=5,
    )

    assert [i.entity_id for i in kept] == ["Q1"]
    assert [i.entity_id for i in all_kept] == ["Q1", "Q3"]


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
    assert [
        i.entity_id for i in _parse([_row(1, label="ox", unit=DAY)], TAXON)
    ] == ["Q1"]


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


def test_load_quantities_asks_for_eight_times_n_rows_per_side(
    tmp_path: Path,
) -> None:
    fetch = _endpoint(8, 8)

    _load(fetch, tmp_path, n_per_side=7)

    assert len(fetch.queries) == 2
    assert all(q.rstrip().endswith("LIMIT 56") for q in fetch.queries)


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


def _default(name: str) -> Family:
    return next(f for f in quantities.FAMILIES if f.name == name)


def test_default_families_are_height_wingspan_and_duration() -> None:
    assert [f.name for f in quantities.FAMILIES] == [
        "height",
        "wingspan",
        "duration",
    ]


def test_the_wingspan_exact_side_is_aircraft_models_without_families() -> None:
    side = _default("wingspan").exact_side

    assert side.classes == ("Q15056995",)
    assert side.exclude_classes == ("Q15056993",)


def test_only_the_height_range_side_unions_statements() -> None:
    unions = [
        (f.name, s.kind)
        for f in quantities.FAMILIES
        for s in (f.range_side, f.exact_side)
        if s.union_statements
    ]

    assert unions == [("height", "range")]
    assert _default("height").range_side.classes == ("Q39367",)


def test_time_families_convert_from_time_units() -> None:
    assert _default("duration").range_side.dimension == "time"
    assert _default("duration").exact_side.dimension == "time"
    assert _default("height").exact_side.dimension == "length"
