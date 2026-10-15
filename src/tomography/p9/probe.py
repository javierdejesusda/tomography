"""The training-cutoff probe and the label tables of a labeled pool.

A model knows the award winners of the years it was trained on and few of the
later ones, so the known rate by year steps down at its training cutoff. The
tables count how many known items a model has per group and popularity bin,
which is what sizes the known versus unknown contrasts.
"""

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
import itertools
import math

from tomography.knowledge import Label
from tomography.p9.pool import GROUPS
from tomography.p9.pool import LabeledItem

_LABELS: tuple[Label, ...] = ("known", "unknown", "uncertain")


@dataclass(frozen=True)
class YearRate:
    """The award questions of one year.

    Attributes:
        year: The year of the award.
        n: Award questions of that year.
        known: How many of them have the contrast label "known".
    """

    year: int
    n: int
    known: int

    @property
    def rate(self) -> float:
        """The known share of the year."""
        return self.known / self.n


@dataclass(frozen=True)
class CutoffEstimate:
    """The two-rate step model fitted to the known rate by year.

    Attributes:
        year: Last year of the high-knowledge regime.
        rate_before: Known rate over the years up to `year`.
        rate_after: Known rate over the later years.
        gain: Log-likelihood gain of the two-rate model over one rate.
    """

    year: int
    rate_before: float
    rate_after: float
    gain: float


def year_rates(items: Sequence[LabeledItem]) -> list[YearRate]:
    """Counts the award questions and the known ones per year.

    Args:
        items: The labeled questions; only the "award" group counts.

    Returns:
        One entry per year present, sorted by year. "uncertain" and
        "unknown" both count as not known, and so does a known label that
        failed the guess control.

    Raises:
        ValueError: If an award question has a year that is not an int.
    """
    counts: dict[int, list[int]] = {}
    for item in items:
        if item.group != "award":
            continue
        year = item.example.meta.get("year")
        if not isinstance(year, int):
            raise ValueError(f"{item.example.uid}: meta['year'] must be an int")
        cell = counts.setdefault(year, [0, 0])
        cell[0] += 1
        cell[1] += item.contrast_label == "known"
    return [YearRate(y, n, k) for y, (n, k) in sorted(counts.items())]


def _log_likelihood(known: int, n: int) -> float:
    """Binomial log-likelihood at the maximum-likelihood rate."""
    if n == 0:
        return 0.0
    rate = known / n
    total = 0.0
    if known:
        total += known * math.log(rate)
    if n - known:
        total += (n - known) * math.log1p(-rate)
    return total


def estimate_cutoff(
    rates: Sequence[YearRate], *, min_years: int = 3
) -> CutoffEstimate:
    """Finds the year after which the known rate drops.

    For every split with at least `min_years` years on each side, compares
    one binomial rate for the years up to the split and one for the later
    years against a single rate over all years. Only splits whose later rate
    is lower are kept, since a knowledge cutoff only lowers the known rate.

    Args:
        rates: The rates by year, as `year_rates` returns.
        min_years: Years needed on each side of a split.

    Returns:
        The split with the largest log-likelihood gain; `year` is the last
        year before the drop.

    Raises:
        ValueError: If `min_years` is below 1, there are fewer than
            `2 * min_years` years, or no split has a lower later rate.
    """
    if min_years < 1:
        raise ValueError(f"min_years must be at least 1, got {min_years}")
    ordered = sorted(rates, key=lambda r: r.year)
    if len(ordered) < 2 * min_years:
        raise ValueError(
            f"need at least {2 * min_years} years, got {len(ordered)}"
        )
    base = _log_likelihood(
        sum(r.known for r in ordered), sum(r.n for r in ordered)
    )
    best: CutoffEstimate | None = None
    for cut in range(min_years, len(ordered) - min_years + 1):
        before, after = ordered[:cut], ordered[cut:]
        n_before, n_after = sum(r.n for r in before), sum(r.n for r in after)
        k_before = sum(r.known for r in before)
        k_after = sum(r.known for r in after)
        if (
            n_before == 0
            or n_after == 0
            or k_after / n_after >= k_before / n_before
        ):
            continue
        gain = (
            _log_likelihood(k_before, n_before)
            + _log_likelihood(k_after, n_after)
            - base
        )
        if best is None or gain > best.gain:
            best = CutoffEstimate(
                ordered[cut - 1].year,
                k_before / n_before,
                k_after / n_after,
                gain,
            )
    if best is None:
        raise ValueError("the known rate never drops after any year")
    return best


def _counts(members: Sequence[LabeledItem]) -> list[object]:
    """The count columns of a table row: n, known_raw and the three labels."""
    return [
        len(members),
        sum(m.label.label == "known" for m in members),
        *(sum(m.contrast_label == name for m in members) for name in _LABELS),
    ]


def label_table(items: Sequence[LabeledItem]) -> list[list[object]]:
    """Counts the labels per group and kind.

    Args:
        items: The labeled questions.

    Returns:
        Rows `[group, kind, n, known_raw, known, unknown, uncertain]`, one
        per (group, kind) present, in `GROUPS` order and then by kind.
        `known_raw` counts the raw "known" labels; the last three count the
        contrast labels and add up to `n`.
    """
    rows: list[list[object]] = []
    for group in GROUPS:
        members = [i for i in items if i.group == group]
        for kind in sorted({i.example.kind for i in members}):
            of_kind = [i for i in members if i.example.kind == kind]
            rows.append([group, kind, *_counts(of_kind)])
    return rows


def popularity_table(
    items: Sequence[LabeledItem],
    *,
    edges: Sequence[int] = (1_000, 10_000, 100_000),
) -> list[list[object]]:
    """Counts the labels of the PopQA questions per relation and popularity.

    Args:
        items: The labeled questions; only the "popqa" group counts.
        edges: Increasing bin edges of `meta["popularity"]`. The bins are
            `[0, edges[0])`, `[edges[0], edges[1])` and so on up to
            `[edges[-1], inf)`.

    Returns:
        Rows `[relation, popularity, n, known_raw, known, unknown,
        uncertain]`, one per (relation, bin) present, sorted by relation and
        then by bin.

    Raises:
        ValueError: If `edges` is empty or not strictly increasing, or a
            PopQA question has a popularity that is not an int.
    """
    if not edges or any(a >= b for a, b in itertools.pairwise(edges)):
        raise ValueError(f"edges must be non-empty and increasing, got {edges}")
    names = [
        f"[0, {edges[0]})",
        *(f"[{lo}, {hi})" for lo, hi in itertools.pairwise(edges)),
        f"[{edges[-1]}, inf)",
    ]
    cells: dict[tuple[str, int], list[LabeledItem]] = {}
    for item in items:
        if item.group != "popqa":
            continue
        popularity = item.example.meta.get("popularity")
        if not isinstance(popularity, int):
            raise ValueError(
                f"{item.example.uid}: PopQA popularity must be an int"
            )
        key = (
            str(item.example.meta["relation"]),
            bisect_right(edges, popularity),
        )
        cells.setdefault(key, []).append(item)
    return [
        [relation, names[index], *_counts(members)]
        for (relation, index), members in sorted(cells.items())
    ]
