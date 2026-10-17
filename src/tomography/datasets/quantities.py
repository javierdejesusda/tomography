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
_TOLERANCE = 1e-9

MIN_RANGE_RELATIVE_WIDTH = 0.10
MAX_EXACT_RELATIVE_HALF_WIDTH = 0.01
DECIMALS = 2
DEFAULT_MIN_SITELINKS = 5
QUERY_ROWS_PER_ITEM = 4


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
        scale: Factor from the normalized SI value to the question unit.
        taxon: True if the entities are taxa, whose scientific names are
            fetched so that items named by them can be dropped.
        query_min_sitelinks: If positive, the query itself keeps only
            entities with at least this many sitelinks. A class with
            hundreds of thousands of statements (films) times out on the
            public service without it.

    Raises:
        ValueError: If `kind` is not "range" or "exact" or
            `query_min_sitelinks` is negative.
    """

    kind: SideKind
    prop: str
    classes: tuple[str, ...]
    template: str
    unit: str
    scale: float
    taxon: bool = False
    query_min_sitelinks: int = 0

    def __post_init__(self) -> None:
        """Validates the kind and the query floor."""
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
            classes=("Q15056993", "Q15056995"),
            template="What is the wingspan of the {entity}, in metres?",
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
        ),
        exact_side=Side(
            kind="exact",
            prop="P2047",
            classes=("Q11424",),
            template="How long is the film {entity}, in minutes?",
            unit="minutes",
            scale=1 / 60,
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

    Values are the normalized (`psn:`) quantities, so every statement is in
    SI units whatever unit its editor used. Only best-rank statements are
    returned; one entity with several of them gives several rows.

    Args:
        side: The side to query.
        limit: Number of rows to ask for, the most linked entities first.

    Returns:
        A SPARQL query with the variables item, itemLabel, links, st,
        amount, lower, upper and, for a taxon side, taxonName. Bounds are
        unbound when the statement has none. Labels are English, falling
        back to the language-neutral `mul` label.

    Raises:
        ValueError: If the property or a class is not an id, no class is
            given or `limit` is below 1.
    """
    prop = side.prop
    if not _PROPERTY_ID.fullmatch(prop):
        raise ValueError(f"not a Wikidata property id: {prop!r}")
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
    variables = "?item ?itemLabel ?links ?st ?amount ?lower ?upper"
    lines = [
        f"SELECT {variables}{' ?taxonName' if side.taxon else ''} WHERE {{",
        f"  VALUES ?class {{ {values} }}",
        "  ?item wdt:P31 ?class ;",
        "        wikibase:sitelinks ?links ;",
        f"        p:{prop} ?st .",
        *floor,
        f"  ?st a wikibase:BestRank ; psn:{prop} ?v .",
        "  ?v wikibase:quantityAmount ?amount .",
        "  OPTIONAL { ?v wikibase:quantityLowerBound ?lower ;"
        " wikibase:quantityUpperBound ?upper . }",
    ]
    if side.taxon:
        lines.append("  OPTIONAL { ?item wdt:P225 ?taxonName }")
    lines += [
        _LABEL_SERVICE,
        "}",
        "ORDER BY DESC(?links) ?item",
        f"LIMIT {limit}",
    ]
    return "\n".join(lines) + "\n"


def _item_id(value: str) -> str:
    if not _ITEM_ID.fullmatch(value):
        raise ValueError(f"not a Wikidata item id: {value!r}")
    return value


def parse_side(
    result: QueryResult,
    *,
    family: str,
    side: Side,
    min_sitelinks: int,
    seed: int = 0,
) -> list[QuantityQuestion]:
    """Turns one side's query rows into questions.

    An entity is dropped for: more than one best-rank statement (the value
    is ambiguous, for example heights by sex); fewer than `min_sitelinks`
    sitelinks; a missing label or a label that is just its item id; for a
    taxon side, a label equal to its scientific name (compared ignoring
    case), so only common names remain. A range statement must have bounds
    with lower < upper, the amount inside them and a relative width
    (upper - lower) / amount of at least `MIN_RANGE_RELATIVE_WIDTH`;
    narrower bounds are measurement precision, not a population range. An
    exact statement must have no bounds or a relative half-width of at most
    `MAX_EXACT_RELATIVE_HALF_WIDTH`, and then low = high = amount. Values
    are multiplied by the side's `scale` and rounded to `DECIMALS` places.
    The split comes from the family, uid and seed alone, so an item keeps
    its split when other rows change.

    Args:
        result: Rows from `side_query`.
        family: Family name, used in the uid and the split.
        side: The side the rows belong to.
        min_sitelinks: Smallest sitelink count kept.
        seed: Changes the split assignment.

    Returns:
        The kept questions, most linked first, ties by item id number.
    """
    entities: dict[str, list[Mapping[str, str]]] = {}
    for row in result.rows:
        entities.setdefault(row["item"].rsplit("/", 1)[-1], []).append(row)
    kept: list[tuple[str, Mapping[str, str], tuple[float, float]]] = []
    for entity_id, rows in entities.items():
        if len({row["st"] for row in rows}) > 1:
            continue
        first = rows[0]
        if int(first["links"]) < min_sitelinks:
            continue
        if not _has_common_name(entity_id, rows, side):
            continue
        interval = _interval(first, side)
        if interval is not None:
            kept.append((entity_id, first, interval))
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
            low=round(low * side.scale, DECIMALS),
            high=round(high * side.scale, DECIMALS),
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


def _has_common_name(
    entity_id: str, rows: list[Mapping[str, str]], side: Side
) -> bool:
    label = rows[0].get("itemLabel", "")
    if not label or _ITEM_ID.fullmatch(label):
        return False
    if side.taxon:
        names = {
            row["taxonName"].casefold() for row in rows if "taxonName" in row
        }
        return label.casefold() not in names
    return True


def _interval(row: Mapping[str, str], side: Side) -> tuple[float, float] | None:
    amount = float(row["amount"])
    if amount <= 0:
        return None
    if "lower" in row and "upper" in row:
        lower, upper = float(row["lower"]), float(row["upper"])
    elif "lower" in row or "upper" in row:
        return None
    elif side.kind == "exact":
        return amount, amount
    else:
        return None
    if side.kind == "range":
        inside = lower <= amount <= upper
        wide = (upper - lower) / amount >= MIN_RANGE_RELATIVE_WIDTH - _TOLERANCE
        return (lower, upper) if lower < upper and inside and wide else None
    half = (upper - lower) / 2 / amount
    if half <= MAX_EXACT_RELATIVE_HALF_WIDTH + _TOLERANCE:
        return amount, amount
    return None


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
                )
            )
        keep = min(n_per_side, *(len(side) for side in sides))
        for parsed in sides:
            items.extend(parsed[:keep])
    return sorted(items, key=lambda item: item.uid)
