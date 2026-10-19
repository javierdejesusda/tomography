"""Table 8: Approximate answers.

One row per checkpoint: how often the greedy few-shot answer is written as a
range on range items, exact items and the heights of 120 people, the AUROC on
held-out durations of a range reader fitted on heights and wingspans next to
its random control and a classifier on the question words, and where the
reader places the people between the test buildings and the breeds against
the range of its random control.
"""

from tomography.site import runfiles
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.site.registry import TableOutput
from tomography.site.tables import _cells
from tomography.site.tables import _rows
from tomography.viz import paper

CELL = {
    "condition": "free",
    "format": "fewshot",
    "shot_order": "range_last",
    "decoding": "greedy",
}
TRAIN = "height+wingspan"
TEMPLATE = "concept"
GROUPS = (
    ("Greedy answers written as a range (percent)", 3, False),
    ("AUROC on held-out durations", 3, False),
    ("People between buildings (0) and breeds (1)", 3, False),
)
COLUMNS = [
    "Model",
    "Range items",
    "Exact items",
    "People",
    "Reader",
    "Random control",
    f"Reader {_cells.MINUS} words",
    "Position",
    "Random band",
    "Inside band",
]


def _in_cell(row: dict) -> bool:
    return all(row[k] == v for k, v in CELL.items())


def _shares(manifest: Manifest, model: str) -> list[str]:
    run = manifest.run("p10a", model)
    assert run is not None
    kinds = {
        r["kind"]: r["range_rate"]
        for r in runfiles.rows(run, "table.csv")
        if r["family"] == "all" and _in_cell(r)
    }
    people = [
        r["people_rate"]
        for r in runfiles.rows(run, "diagnostic.csv")
        if _in_cell(r)
    ]
    return [
        _cells.percent(v) for v in (kinds["range"], kinds["exact"], people[0])
    ]


def _transfer(rows: list[dict], method: str) -> dict:
    (hit,) = [
        r
        for r in rows
        if r["template"] == TEMPLATE
        and r["method"] == method
        and r["train_set"] == TRAIN
        and r["test_family"] == "duration"
    ]
    return hit


def build(manifest: Manifest) -> TableOutput:
    """Builds the table from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The table and its caption parts.
    """
    models = manifest.models("p10b")
    rows: list[list[str]] = []
    above: dict[str, bool] = {}
    above_point: dict[str, bool] = {}
    inside: dict[str, bool] = {}
    words_value = 0.0
    for model in models:
        run = manifest.run("p10b", model)
        assert run is not None
        transfer = runfiles.rows(run, "transfer.csv")
        reader = _transfer(transfer, "logistic")
        random = _transfer(transfer, "random")
        (word,) = [
            r
            for r in runfiles.rows(run, "baselines.csv")
            if r["baseline"] == "text_word"
            and r["train_set"] == TRAIN
            and r["test_family"] == "duration"
        ]
        words_value = float(word["auroc"])
        n_pos, n_neg = int(reader["n_range"]), int(reader["n_exact"])
        d, lo, hi = stats.auc_diff_interval(
            reader["auroc"], word["auroc"], n_pos, n_neg
        )
        above[model] = lo > 0
        above_point[model] = d > 0
        people = [
            r
            for r in runfiles.rows(run, "people.csv")
            if r["source"] == "reader" and r["template"] == TEMPLATE
        ]
        (position,) = [
            r["position"]
            for r in people
            if r["method"] == "logistic" and r["train_set"] == TRAIN
        ]
        band = [r["position"] for r in people if r["method"] == "random"]
        low, high = min(band), max(band)
        inside[model] = low <= position <= high
        rows.append(
            [
                paper.display_name(model),
                *_shares(manifest, model),
                _cells.num(reader["auroc"]),
                _cells.num(random["auroc"]),
                _cells.pm(d, (hi - lo) / 2) + _rows.star(lo, hi),
                _cells.num(position),
                f"[{_cells.num(low)}, {_cells.num(high)}]",
                "yes" if inside[model] else "no",
            ]
        )
    claim = (
        "Qwen3 writes ranges where the sourced answer is a range, but a "
        "range reader is matched by a classifier on the question words and "
        "by its random control."
    )
    detail = (
        "The first three columns give the percentage of greedy answers "
        "written as a range on range items, exact items and the heights of "
        "120 people. Reader is the AUROC on held-out durations of a range "
        "reader fitted on heights and wingspans with the concept prompt, "
        "next to its random control (the mean score of 5 random directions "
        "oriented on the training labels) and its difference from a "
        f"classifier on the question words ({_cells.num(words_value)}), with "
        "the Hanley and McNeil half-width and an asterisk where it excludes "
        "zero. Position places the 120 people between the test buildings (0) "
        "and the breeds (1); the random band is the range of the random "
        "control's position over its 7 training sets, and the last column "
        "says whether the reader's position lies inside it."
    )
    numbers = {
        "text_word": words_value,
        "inside_band": float(sum(inside.values())),
    }
    counts = {
        "above_words": stats.per_role(above),
        "above_words_point": stats.per_role(above_point),
        "inside_band": stats.per_role(inside),
    }
    return TableOutput(
        claim,
        detail,
        COLUMNS,
        rows,
        "l" + "r" * 9,
        [],
        numbers,
        groups=GROUPS,
        row_groups=_rows.role_groups(models),
        counts=counts,
    )
