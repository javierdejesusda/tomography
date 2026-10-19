"""Table 5: Approximate answers and links.

One row per model: how often the greedy few-shot answer is a range on range
items, exact items and people's heights (P10a), how many stored cells of the
range reader, the text-word baseline and a random direction reach the
ceiling (P10b), and how many cosines between the range direction and the
knowledge directions lie outside their random band (P10c).
"""

import json

from tomography.site.manifest import Manifest
from tomography.site.registry import TableOutput
from tomography.site.tables import _cells
from tomography.viz import paper

CELL = ("free", "fewshot", "range_last", "greedy")
FORCED = ("all", "range", "fewshot", "range_last")
RANDOM_FLOOR = 0.899
COLUMNS = [
    "Model",
    "Range items (%)",
    "Exact items (%)",
    "People (%)",
    "Forced in range (%)",
    "Reader 1.000",
    "Text words 1.000",
    "Random 0.899 or more",
    "Cosines outside band",
]
ALIGN = "lrrrrrrrr"
DETAIL = (
    "Range items, Exact items and People are the share of greedy free "
    "few-shot answers written as a range when the range example comes last "
    "(P10a, family all; people's heights are a diagnostic, not a clean "
    "contrast). Forced in range is the share of forced point answers on "
    "range items that fall inside the sourced range. The next three columns "
    "count the stored P10b cells (both templates, every train set and test "
    "family) in which the logistic reader or the text-word baseline reads an "
    "AUROC of 1.000, and in which a random direction reads 0.899 or more. "
    "The last column counts the cosines between the range direction at the "
    "reader layer and the P9 reader directions that lie outside the random "
    "band of the run, out of all P9 types. These are counts, not estimates: "
    "nothing here is resampled."
)


def _count_cosines(manifest: Manifest, model: str) -> tuple[int, int] | None:
    run = manifest.run("p10c", model)
    if run is None:
        return None
    path = run / "summary.json"
    cosine = json.loads(path.read_text(encoding="utf-8"))["data"]["cosine"]
    outside = sum(
        not c["band_lo"] <= c["cosine"] <= c["band_hi"] for c in cosine.values()
    )
    return outside, len(cosine)


def _count_cells(manifest: Manifest, model: str) -> dict[str, tuple[int, int]]:
    run = manifest.run("p10b", model)
    assert run is not None
    transfer = _cells.read_csv(run / "transfer.csv")
    reader = [r["auroc"] for r in transfer if r["method"] == "logistic"]
    random = [r["auroc"] for r in transfer if r["method"] == "random"]
    text = [
        r["auroc"]
        for r in _cells.read_csv(run / "baselines.csv")
        if r["baseline"] == "text_word"
    ]
    return {
        "reader": (sum(round(v, 3) == 1.0 for v in reader), len(reader)),
        "text": (sum(round(v, 3) == 1.0 for v in text), len(text)),
        "random": (sum(v >= RANDOM_FLOOR for v in random), len(random)),
    }


def _pair(counts: tuple[int, int] | None) -> str:
    return _cells.MISSING if counts is None else _cells.count(*counts)


def _behavior(manifest: Manifest, model: str) -> list[str]:
    run = manifest.run("p10a", model)
    assert run is not None
    table = {
        r["kind"]: r
        for r in _cells.read_csv(run / "table.csv")
        if r["family"] == "all"
        and (r["condition"], r["format"], r["shot_order"], r["decoding"])
        == CELL
    }
    (diag,) = [
        r
        for r in _cells.read_csv(run / "diagnostic.csv")
        if (r["condition"], r["format"], r["shot_order"], r["decoding"]) == CELL
    ]
    forced = [
        r
        for r in _cells.read_csv(run / "forced_in_range.csv")
        if (r["family"], r["kind"], r["format"], r["shot_order"]) == FORCED
    ]
    inside = forced[0]["inside_rate"] if forced else None
    return [
        _cells.percent(table["range"]["range_rate"]),
        _cells.percent(table["exact"]["range_rate"]),
        _cells.percent(diag["people_rate"]),
        _cells.percent(inside),
    ]


def build(manifest: Manifest) -> TableOutput:
    """Builds the table from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The table and its caption parts.
    """
    models = manifest.models("p10a")
    rows: list[list[str]] = []
    totals: dict[str, list[int]] = {
        k: [0, 0] for k in ("reader", "text", "random", "cosine")
    }
    for model in models:
        counts = _count_cells(manifest, model)
        cosine = _count_cosines(manifest, model)
        for key, value in counts.items():
            totals[key][0] += value[0]
            totals[key][1] += value[1]
        if cosine is not None:
            totals["cosine"][0] += cosine[0]
            totals["cosine"][1] += cosine[1]
        rows.append(
            [
                paper.display_name(model),
                *_behavior(manifest, model),
                _pair(counts["reader"]),
                _pair(counts["text"]),
                _pair(counts["random"]),
                _pair(cosine),
            ]
        )
    numbers = {
        "models": float(len(models)),
        "reader_cells": float(totals["reader"][0]),
        "reader_total": float(totals["reader"][1]),
        "text_cells": float(totals["text"][0]),
        "text_total": float(totals["text"][1]),
        "random_cells": float(totals["random"][0]),
        "random_total": float(totals["random"][1]),
        "cosine_outside": float(totals["cosine"][0]),
        "cosine_total": float(totals["cosine"][1]),
    }
    claim = (
        f"Over the {len(models)} models the logistic range reader reads an "
        f"AUROC of 1.000 in {totals['reader'][0]} of {totals['reader'][1]} "
        f"stored cells, the text-word baseline in {totals['text'][0]} of "
        f"{totals['text'][1]}, and a random direction reads 0.899 or more in "
        f"{totals['random'][0]} of {totals['random'][1]}; "
        f"{totals['cosine'][0]} of {totals['cosine'][1]} cosines between the "
        "range direction and the knowledge directions lie outside their "
        "random band."
    )
    notes = [
        "The range reader of P10c is the plain-template logistic reader "
        "fitted on all three families at layer 1."
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
