"""Table 6: Steering and fluency.

One row per model: the judged sentiment score, the perplexity ratio to the
unsteered text and distinct-1 of the C1 runs at relative coefficients -0.1,
+0.1 and +0.3, with the unsteered and prompting rows, and the M1 win rate at
strength 0.3 of a random direction and of the mean-difference direction where
the run is stored.
"""

from typing import Any

import numpy as np

from tomography.site.manifest import Manifest
from tomography.site.registry import TableOutput
from tomography.site.tables import _cells
from tomography.viz import paper

MAX_RATIO = 1.5
MIN_DISTINCT = 0.15
COEFFS = (("-0.1", -0.1), ("+0.1", 0.1), ("+0.3", 0.3))
COLUMNS = [
    "Model",
    "Unsteered",
    *(name for name, _ in COEFFS),
    "Prompting +1",
    "M1 random",
    "M1 mean_diff",
]
ALIGN = "lrrrrrrr"
DETAIL = (
    "Cells of the first five data columns read score / perplexity ratio / "
    "distinct-1: the judged sentiment score, the perplexity as a ratio to "
    "the unsteered text and distinct-1, each the mean over the three readers "
    "of the C1 run at the relative coefficient named in the header. The "
    "prompting cell has the score and distinct-1 of prompting for positive "
    "sentiment and no perplexity. The last two columns are the win rate of "
    "the steered text against the unsteered text at relative strength 0.3 in "
    "M1 for a random direction and for the mean-difference direction, with "
    "the 95 percent bootstrap interval over the items, which resamples the "
    "750 judged pairs; they exist only for the models with an M1 run."
)


def _none(rows: list[dict[str, Any]]) -> dict[str, Any]:
    (row,) = [r for r in rows if r["arm"] == "none"]
    return row


def _mean(rows: list[dict[str, Any]], coeff: float, key: str) -> float:
    cell = [
        r[key] for r in rows if r["arm"] == "linear" and r["coeff"] == coeff
    ]
    return float(np.mean(cell))


def _ratio(rows: list[dict[str, Any]], coeff: float) -> float:
    return float(_mean(rows, coeff, "perplexity") / _none(rows)["perplexity"])


def _fluent(rows: list[dict[str, Any]], size: float) -> bool:
    return all(
        _ratio(rows, c) <= MAX_RATIO
        and _mean(rows, c, "distinct_1") >= MIN_DISTINCT
        for c in (size, -size)
    )


def _m1(manifest: Manifest, model: str) -> dict[str, dict[str, Any]]:
    run = manifest.run("m1", model)
    if run is None:
        return {}
    rows = _cells.read_table(run / "table.json")
    return {
        r["method"]: r
        for r in rows
        if r["task"] == "manipulation" and r["setting"] == "0.3"
    }


def _win(arms: dict[str, dict[str, Any]], method: str) -> str:
    hit = arms.get(method)
    if hit is None:
        return _cells.MISSING
    return _cells.interval(hit["value"], hit["lo"], hit["hi"], 2)


def build(manifest: Manifest) -> TableOutput:
    """Builds the table from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The table and its caption parts.
    """
    models = manifest.models("c1")
    tables: dict[str, list[dict[str, Any]]] = {}
    arms: dict[str, dict[str, dict[str, Any]]] = {}
    rows: list[list[str]] = []
    for model in models:
        run = manifest.run("c1", model)
        assert run is not None
        table = _cells.read_table(run / "table.json")
        tables[model] = table
        none = _none(table)
        (prompt,) = [r for r in table if r["arm"] == "prompting+1"]
        found = _m1(manifest, model)
        if found:
            arms[model] = found
        rows.append(
            [
                paper.display_name(model),
                f"{none['score']:.2f} / 1.00 / {none['distinct_1']:.2f}",
                *(
                    f"{_mean(table, c, 'score'):.2f} / {_ratio(table, c):.2f} "
                    f"/ {_mean(table, c, 'distinct_1'):.2f}"
                    for _, c in COEFFS
                ),
                f"{prompt['score']:.2f} / n/a / {prompt['distinct_1']:.2f}",
                _win(found, "random"),
                _win(found, "mean_diff"),
            ]
        )
    numbers = {
        "models": float(len(models)),
        "score_unsteered": float(
            np.mean([_none(t)["score"] for t in tables.values()])
        ),
        "score_minus": float(
            np.mean([_mean(t, -0.1, "score") for t in tables.values()])
        ),
        "score_plus": float(
            np.mean([_mean(t, 0.1, "score") for t in tables.values()])
        ),
        "fluent_at_0.1": float(sum(_fluent(t, 0.1) for t in tables.values())),
        "fluent_at_0.3": float(sum(_fluent(t, 0.3) for t in tables.values())),
        "m1_models": float(len(arms)),
        "m1_random_max": max(a["random"]["value"] for a in arms.values()),
        "m1_mean_diff_min": min(a["mean_diff"]["value"] for a in arms.values()),
    }
    n = len(models)
    claim = (
        f"Across the {n} checkpoints the mean judged sentiment score is "
        f"{numbers['score_unsteered']:.3f} unsteered, "
        f"{numbers['score_minus']:.3f} at relative coefficient -0.1 and "
        f"{numbers['score_plus']:.3f} at +0.1, and "
        f"{int(numbers['fluent_at_0.1'])} of {n} keep fluent text at both "
        f"signs of 0.1 against {int(numbers['fluent_at_0.3'])} of {n} at 0.3; "
        f"in M1 on {len(arms)} models a random direction wins at most "
        f"{numbers['m1_random_max']:.3f} of the pairs while the "
        f"mean-difference direction wins at least "
        f"{numbers['m1_mean_diff_min']:.3f}."
    )
    notes = [
        "Fluent means a perplexity ratio of at most 1.5 and distinct-1 of at "
        "least 0.15 at both signs of the coefficient; the rule is "
        "descriptive.",
        "C1 has no random-direction arm; the random baseline appears only in "
        "the M1 columns.",
    ]
    return TableOutput(
        claim,
        DETAIL,
        COLUMNS,
        rows,
        _cells.check_align(ALIGN, len(COLUMNS)),
        notes,
        numbers,
    )
