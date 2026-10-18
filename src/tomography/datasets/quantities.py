"""Quantity questions whose true answer is a range or one exact value.

A range question asks for a quantity that varies across a population (a dog
breed's height, a species' gestation period); an exact question asks for a
quantity that has one value (a building's height, a film's running time).
Both come from Wikidata (CC0) through the snapshot cache in
`tomography.datasets.wikidata`, so every item carries the query's retrieval
date and response digest.
"""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re
from typing import Literal
import unicodedata

from tomography.datasets.records import Split
from tomography.datasets.records import assign_splits
from tomography.datasets.wikidata import _LABEL_SERVICE
from tomography.datasets.wikidata import QueryFetch
from tomography.datasets.wikidata import QueryResult
from tomography.datasets.wikidata import run_query

SideKind = Literal["range", "exact"]

_PROPERTY_ID = re.compile(r"P\d+")
_ITEM_ID = re.compile(r"Q\d+")
_KINDS = ("range", "exact")
_ENTITY_URI = "http://www.wikidata.org/entity/"
_SECONDS_PER_DAY = 86400.0
_TOLERANCE = 1e-9

MIN_RANGE_RELATIVE_WIDTH = 0.10
MAX_EXACT_RELATIVE_HALF_WIDTH = 0.01
DECIMALS = 2
DEFAULT_MIN_SITELINKS = 5
QUERY_ROWS_PER_ITEM = 8

LENGTH_UNITS: Mapping[str, float] = {
    "Q11573": 1.0,
    "Q174728": 0.01,
    "Q174789": 0.001,
    "Q828224": 1000.0,
    "Q3710": 0.3048,
    "Q218593": 0.0254,
}
TIME_UNITS: Mapping[str, float] = {
    "Q11574": 1.0,
    "Q7727": 60.0,
    "Q25235": 3600.0,
    "Q573": _SECONDS_PER_DAY,
    "Q23387": 7 * _SECONDS_PER_DAY,
    "Q5151": 30.4375 * _SECONDS_PER_DAY,
    "Q577": 365.25 * _SECONDS_PER_DAY,
}
UNITS: Mapping[str, Mapping[str, float]] = {
    "length": LENGTH_UNITS,
    "time": TIME_UNITS,
}

# Classes (P31) and states of use (P5817) of a structure that was never
# completed: under construction, proposed, proposed tower, proposed entity,
# unbuilt, unfinished, abandoned project and on hold.
UNBUILT_STRUCTURES: tuple[str, ...] = (
    "Q12377751",
    "Q811683",
    "Q124835105",
    "Q64728694",
    "Q105821051",
    "Q1570262",
    "Q21514702",
    "Q97317113",
)


@dataclass(frozen=True)
class Side:
    """One half of a family: where its questions and values come from.

    Attributes:
        kind: "range" if the true answer is a population range, "exact" if
            it is one value.
        prop: Wikidata property id of the quantity, such as "P2048".
        classes: Item ids of the direct P31 classes the entities belong to.
        template: Question text with an "{entity}" slot.
        unit: Word for the unit used in the question.
        scale: Factor from the SI base unit of `dimension` (metre or
            second) to the question unit.
        taxon: True if the entities are taxa, whose scientific names are
            fetched so that items named by them can be dropped.
        dimension: "length" or "time": which table of unit item ids
            (`LENGTH_UNITS`, `TIME_UNITS`, factors to metres or seconds)
            converts the statement's own unit. A statement in another unit
            is dropped.
        exclude_classes: Item ids of direct P31 classes whose entities are
            left out, such as aircraft families in a pool of aircraft
            models.
        exclude_states: Item ids of states of use (P5817) whose entities
            are left out, such as "building or structure under
            construction".
        exclude_subclasses_of: Item ids of classes, such as missile, whose
            instances and subclasses at any depth (P31 or P279, then any
            chain of P279) are left out.
        dated_by: Property ids of dates, such as official opening (P1619).
            If given, an entity is kept only with a value of one of them no
            later than the query time.
        union_statements: If True, an entity's several best-rank
            statements are not ambiguous: its answer is the union of them
            (the lowest low to the highest high). Only for a range side
            whose entities give one statement per sex or variety, as dog
            breed heights do.
        query_min_sitelinks: If positive, the query itself keeps only
            entities with at least this many sitelinks. A class with
            hundreds of thousands of statements (films) times out on the
            public service without it.
        occupations: Item ids of occupations (P106). If given, the query
            keeps only entities with one of them, which keeps a query over
            a huge class such as humans within the service's time limit.
        unbounded_only: If True, an exact statement with any lower or upper
            bound is dropped, however narrow.

    Raises:
        ValueError: If `kind` is not "range" or "exact", `dimension` is
            unknown, `union_statements` is set on an exact side,
            `unbounded_only` is set on a range side or
            `query_min_sitelinks` is negative.
    """

    kind: SideKind
    prop: str
    classes: tuple[str, ...]
    template: str
    unit: str
    scale: float
    taxon: bool = False
    dimension: str = "length"
    exclude_classes: tuple[str, ...] = ()
    exclude_states: tuple[str, ...] = ()
    exclude_subclasses_of: tuple[str, ...] = ()
    dated_by: tuple[str, ...] = ()
    union_statements: bool = False
    query_min_sitelinks: int = 0
    occupations: tuple[str, ...] = ()
    unbounded_only: bool = False

    def __post_init__(self) -> None:
        """Validates the kind, dimension, union, bounds and query floor."""
        if self.unbounded_only and self.kind != "exact":
            raise ValueError("unbounded_only applies to an exact side only")
        if self.dimension not in UNITS:
            raise ValueError(f"unknown dimension {self.dimension!r}")
        if self.union_statements and self.kind != "range":
            raise ValueError("union_statements applies to a range side only")
        if self.query_min_sitelinks < 0:
            raise ValueError(
                "query_min_sitelinks must not be negative, got "
                f"{self.query_min_sitelinks}"
            )
        if self.kind not in _KINDS:
            raise ValueError(
                f"kind must be 'range' or 'exact', got {self.kind!r}"
            )


@dataclass(frozen=True)
class Family:
    """A range side and an exact side that share a question frame.

    Attributes:
        name: Family name, such as "height".
        range_side: The side whose answers are population ranges.
        exact_side: The side whose answers are single values.
    """

    name: str
    range_side: Side
    exact_side: Side


FAMILIES: tuple[Family, ...] = (
    Family(
        name="height",
        range_side=Side(
            kind="range",
            prop="P2048",
            classes=("Q39367",),
            template="How tall is a {entity}, in centimetres?",
            unit="centimetres",
            scale=100.0,
            union_statements=True,
        ),
        exact_side=Side(
            kind="exact",
            prop="P2048",
            classes=("Q11303", "Q12518", "Q18142"),
            exclude_classes=UNBUILT_STRUCTURES,
            exclude_states=UNBUILT_STRUCTURES,
            dated_by=("P1619", "P571"),
            template="How tall is the {entity}, in metres?",
            unit="metres",
            scale=1.0,
        ),
    ),
    Family(
        name="wingspan",
        range_side=Side(
            kind="range",
            prop="P2050",
            classes=("Q16521",),
            template="What is the wingspan of the {entity}, in centimetres?",
            unit="centimetres",
            scale=100.0,
            taxon=True,
        ),
        exact_side=Side(
            kind="exact",
            prop="P2050",
            classes=("Q15056995",),
            exclude_classes=("Q15056993",),
            exclude_subclasses_of=("Q974850",),
            template=(
                "What is the wingspan of the {entity} aircraft, in metres?"
            ),
            unit="metres",
            scale=1.0,
        ),
    ),
    Family(
        name="duration",
        range_side=Side(
            kind="range",
            prop="P3063",
            classes=("Q16521",),
            template=(
                "How long is the gestation period of the {entity}, in days?"
            ),
            unit="days",
            scale=1 / 86400,
            taxon=True,
            dimension="time",
        ),
        exact_side=Side(
            kind="exact",
            prop="P2047",
            classes=("Q11424",),
            template="How long is the film {entity}, in minutes?",
            unit="minutes",
            scale=1 / 60,
            dimension="time",
            query_min_sitelinks=20,
        ),
    ),
)


@dataclass(frozen=True)
class QuantityQuestion:
    """One quantity question with its sourced answer.

    Attributes:
        uid: Stable identifier "<family>-<kind>-<entity id>".
        family: Family name, such as "height".
        kind: "range" or "exact".
        entity: English label of the entity asked about.
        entity_id: Wikidata item id of the entity.
        question: The question text.
        low: Lower end of the true answer, in the question unit.
        high: Upper end of the true answer; equal to `low` for an exact
            item.
        unit: Word for the unit used in the question.
        sitelinks: Number of Wikimedia sitelinks of the entity, a
            popularity measure.
        split: Held-out split this item belongs to.
        retrieved_on: ISO date of the query snapshot the item came from.
        sha256: Hex digest of the snapshot's response body.
    """

    uid: str
    family: str
    kind: SideKind
    entity: str
    entity_id: str
    question: str
    low: float
    high: float
    unit: str
    sitelinks: int
    split: Split
    retrieved_on: str
    sha256: str


def side_query(side: Side, *, limit: int) -> str:
    """Builds the query for one side's entities and normalized quantities.

    Values are the statement's own (`psv:`) amount, bounds and unit, not
    the normalized SI value, whose bounds Wikidata rounds to about two
    significant figures. Only best-rank statements are returned; one entity
    with several of them gives several rows.

    Args:
        side: The side to query.
        limit: Number of rows to ask for, the most linked entities first.

    Returns:
        A SPARQL query with the variables item, itemLabel, links, st,
        amount, unit, lower, upper and, for a taxon side, taxonName. Bounds are
        unbound when the statement has none. Labels are English, falling
        back to the language-neutral `mul` label.

    Raises:
        ValueError: If the quantity or a date property, a kept or excluded
            class, an excluded state or an excluded superclass is not an id,
            no class is given or `limit` is below 1.
    """
    prop = _property_id(side.prop)
    if not side.classes:
        raise ValueError("a side must name at least one class")
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    values = " ".join(f"wd:{_item_id(item)}" for item in side.classes)
    floor = (
        [f"  FILTER(?links >= {side.query_min_sitelinks})"]
        if side.query_min_sitelinks
        else []
    )
    variables = "?item ?itemLabel ?links ?st ?amount ?unit ?lower ?upper"
    lines = [
        f"SELECT {variables}{' ?taxonName' if side.taxon else ''} WHERE {{",
        f"  VALUES ?class {{ {values} }}",
        "  ?item wdt:P31 ?class ;",
        "        wikibase:sitelinks ?links ;",
        f"        p:{prop} ?st .",
        *floor,
        f"  ?st a wikibase:BestRank ; psv:{prop} ?v .",
        "  ?v wikibase:quantityAmount ?amount ;",
        "     wikibase:quantityUnit ?unit .",
        "  OPTIONAL { ?v wikibase:quantityLowerBound ?lower ;"
        " wikibase:quantityUpperBound ?upper . }",
    ]
    if side.occupations:
        jobs = " ".join(f"wd:{_item_id(item)}" for item in side.occupations)
        lines += [
            f"  VALUES ?occupation {{ {jobs} }}",
            "  ?item wdt:P106 ?occupation .",
        ]
    if side.exclude_classes:
        lines.append(_none_of("excluded", "wdt:P31", side.exclude_classes))
    if side.exclude_states:
        lines.append(_none_of("state", "wdt:P5817", side.exclude_states))
    if side.exclude_subclasses_of:
        path = "(wdt:P31|wdt:P279)/wdt:P279*"
        lines.append(_none_of("kind", path, side.exclude_subclasses_of))
    if side.dated_by:
        dates = "|".join(f"wdt:{_property_id(p)}" for p in side.dated_by)
        lines.append(
            f"  FILTER EXISTS {{ ?item {dates} ?built ."
            " FILTER(?built <= NOW()) }"
        )
    if side.taxon:
        lines.append("  OPTIONAL { ?item wdt:P225 ?taxonName }")
    lines += [
        _LABEL_SERVICE,
        "}",
        "ORDER BY DESC(?links) ?item",
        f"LIMIT {limit}",
    ]
    return "\n".join(lines) + "\n"


def _none_of(variable: str, path: str, items: tuple[str, ...]) -> str:
    values = " ".join(f"wd:{_item_id(item)}" for item in items)
    return (
        f"  FILTER NOT EXISTS {{ VALUES ?{variable} {{ {values} }}"
        f" ?item {path} ?{variable} }}"
    )


def _item_id(value: str) -> str:
    if not _ITEM_ID.fullmatch(value):
        raise ValueError(f"not a Wikidata item id: {value!r}")
    return value


def _property_id(value: str) -> str:
    if not _PROPERTY_ID.fullmatch(value):
        raise ValueError(f"not a Wikidata property id: {value!r}")
    return value


def parse_side(
    result: QueryResult,
    *,
    family: str,
    side: Side,
    min_sitelinks: int,
    seed: int = 0,
    row_limit: int | None = None,
) -> list[QuantityQuestion]:
    """Turns one side's query rows into questions.

    An entity is dropped for: more than one best-rank statement (the value
    is ambiguous, for example heights by sex), unless the side sets
    `union_statements`; fewer than `min_sitelinks` sitelinks; a missing
    label, a label that is just its item id or a label with a letter outside
    the Latin script (accented Latin letters are kept); for a taxon side, a
    label equal to its scientific name (compared ignoring case), so only
    common names remain; a statement without a unit in the side's table
    (`LENGTH_UNITS` or `TIME_UNITS`) or with a non-positive amount.

    A range statement must have bounds with lower < upper, the amount
    inside them and a relative width (upper - lower) / amount of at least
    `MIN_RANGE_RELATIVE_WIDTH`; narrower bounds are measurement precision,
    not a population range. With `union_statements`, each statement
    contributes its bounds (or its amount, if it has none), the answer runs
    from the lowest low to the highest high, and the relative width is taken
    over the midpoint. An exact statement must have no bounds or a relative
    half-width of at most `MAX_EXACT_RELATIVE_HALF_WIDTH`, and then low =
    high = amount. Values are converted from the statement's own unit and
    rounded to `DECIMALS` places; a range whose rounded low is not below its
    rounded high is dropped. The split comes from the family, uid and seed
    alone, so an item keeps its split when other rows change.

    Args:
        result: Rows from `side_query`.
        family: Family name, used in the uid and the split.
        side: The side the rows belong to.
        min_sitelinks: Smallest sitelink count kept.
        seed: Changes the split assignment.
        row_limit: The LIMIT the query had. If the result reached it, the
            last entity may have lost some of its statements and is dropped.

    Returns:
        The kept questions, most linked first, ties by item id number.
    """
    entities: dict[str, dict[str, Mapping[str, str]]] = {}
    for row in result.rows:
        statements = entities.setdefault(_entity_id(row), {})
        statements.setdefault(row["st"], row)
    if row_limit is not None and len(result.rows) >= row_limit:
        entities.pop(_entity_id(result.rows[-1]), None)
    kept: list[tuple[str, Mapping[str, str], tuple[float, float]]] = []
    for entity_id, statements in entities.items():
        rows = list(statements.values())
        first = rows[0]
        if len(rows) > 1 and not side.union_statements:
            continue
        if int(first["links"]) < min_sitelinks:
            continue
        if not _has_common_name(rows, side):
            continue
        interval = _interval(rows, side)
        if interval is None:
            continue
        low, high = (round(end * side.scale, DECIMALS) for end in interval)
        if side.kind == "range" and low >= high:
            continue
        kept.append((entity_id, first, (low, high)))
    kept.sort(key=lambda entry: (-int(entry[1]["links"]), int(entry[0][1:])))
    uids = [f"{family}-{side.kind}-{entity_id}" for entity_id, _, _ in kept]
    splits = assign_splits(uids, dataset=f"quantities:{family}", seed=seed)
    return [
        QuantityQuestion(
            uid=uid,
            family=family,
            kind=side.kind,
            entity=row["itemLabel"],
            entity_id=entity_id,
            question=side.template.format(entity=row["itemLabel"]),
            low=low,
            high=high,
            unit=side.unit,
            sitelinks=int(row["links"]),
            split=split,
            retrieved_on=result.retrieved_on,
            sha256=result.sha256,
        )
        for (entity_id, row, (low, high)), uid, split in zip(
            kept, uids, splits, strict=True
        )
    ]


def _entity_id(row: Mapping[str, str]) -> str:
    return row["item"].rsplit("/", 1)[-1]


def _has_common_name(rows: list[Mapping[str, str]], side: Side) -> bool:
    label = rows[0].get("itemLabel", "")
    if not label or _ITEM_ID.fullmatch(label) or not _is_latin(label):
        return False
    if side.taxon:
        names = {
            row["taxonName"].casefold() for row in rows if "taxonName" in row
        }
        return label.casefold() not in names
    return True


def _is_latin(text: str) -> bool:
    return all(
        unicodedata.name(char, "").startswith("LATIN")
        for char in text
        if char.isalpha()
    )


def _in_base_unit(
    row: Mapping[str, str], side: Side
) -> tuple[float, float | None, float | None] | None:
    """Returns (amount, lower, upper) in metres or seconds, or None."""
    unit = row.get("unit", "")
    if not unit.startswith(_ENTITY_URI):
        return None
    factor = UNITS[side.dimension].get(unit[len(_ENTITY_URI) :])
    amount = float(row["amount"])
    if factor is None or amount <= 0:
        return None
    if ("lower" in row) != ("upper" in row):
        return None
    if "lower" not in row:
        return amount * factor, None, None
    return (
        amount * factor,
        float(row["lower"]) * factor,
        float(row["upper"]) * factor,
    )


def _interval(
    rows: list[Mapping[str, str]], side: Side
) -> tuple[float, float] | None:
    values = [_in_base_unit(row, side) for row in rows]
    points = [value for value in values if value is not None]
    if len(points) < len(values):
        return None
    if side.union_statements:
        return _union(points)
    amount, lower, upper = points[0]
    if lower is None or upper is None:
        return (amount, amount) if side.kind == "exact" else None
    if side.unbounded_only:
        return None
    if side.kind == "range":
        inside = lower <= amount <= upper
        wide = (upper - lower) / amount >= MIN_RANGE_RELATIVE_WIDTH - _TOLERANCE
        return (lower, upper) if lower < upper and inside and wide else None
    half = (upper - lower) / 2 / amount
    if half <= MAX_EXACT_RELATIVE_HALF_WIDTH + _TOLERANCE:
        return amount, amount
    return None


def _union(
    points: list[tuple[float, float | None, float | None]],
) -> tuple[float, float] | None:
    lows, highs = [], []
    for amount, lower, upper in points:
        if lower is not None and upper is not None:
            if not lower <= amount <= upper:
                return None
            lows.append(lower)
            highs.append(upper)
        else:
            lows.append(amount)
            highs.append(amount)
    low, high = min(lows), max(highs)
    midpoint = (low + high) / 2
    wide = (high - low) / midpoint >= MIN_RANGE_RELATIVE_WIDTH - _TOLERANCE
    return (low, high) if low < high and wide else None


def load_quantities(
    *,
    families: Sequence[Family] = FAMILIES,
    n_per_side: int = 120,
    min_sitelinks: int = DEFAULT_MIN_SITELINKS,
    seed: int = 0,
    cache: Path | None = None,
    offline: bool = False,
    fetch: QueryFetch | None = None,
    today: Callable[[], date] | None = None,
) -> list[QuantityQuestion]:
    """Loads balanced range and exact quantity questions.

    Each side is one cached query for `QUERY_ROWS_PER_ITEM * n_per_side`
    rows, the most linked first: the filters of `parse_side` discard some
    rows (a taxon can also fill several rows), so the query asks for more
    than it needs. Per family, the most linked n items of each side are
    kept, where n is the smaller of `n_per_side` and the two sides' usable
    counts, so both sides have the same size and similar popularity. A
    family with no usable item on a side gives no items.

    Args:
        families: The families to load.
        n_per_side: Largest number of items per side and family.
        min_sitelinks: Smallest sitelink count kept.
        seed: Changes the split assignment.
        cache: Cache directory; defaults to the shared tomography cache.
        offline: If True, use only cached query results.
        fetch: Returns the raw response for a query; defaults to
            `sparql_fetch`.
        today: Gives the retrieval date; defaults to today in UTC.

    Returns:
        The questions of every family, sorted by uid.

    Raises:
        ValueError: If `n_per_side` is below 1 or `families` is empty.
        ChecksumError: If a cached result does not match its digest.
        OfflineCacheMissError: If `offline` is True and a query is not
            cached.
        http.client.IncompleteRead: If the connection closed before the
            whole response arrived on every attempt.
    """
    if n_per_side < 1:
        raise ValueError(f"n_per_side must be at least 1, got {n_per_side}")
    if not families:
        raise ValueError("families must name at least one family")
    limit = QUERY_ROWS_PER_ITEM * n_per_side
    items: list[QuantityQuestion] = []
    for family in families:
        sides = []
        for side in (family.range_side, family.exact_side):
            result = run_query(
                side_query(side, limit=limit),
                fetch=fetch,
                cache=cache,
                offline=offline,
                today=today,
            )
            sides.append(
                parse_side(
                    result,
                    family=family.name,
                    side=side,
                    min_sitelinks=min_sitelinks,
                    seed=seed,
                    row_limit=limit,
                )
            )
        keep = min(n_per_side, *(len(side) for side in sides))
        for parsed in sides:
            items.extend(parsed[:keep])
    return sorted(items, key=lambda item: item.uid)
