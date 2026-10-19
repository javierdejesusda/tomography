"""Table 7: Knowing what one knows.

One row per checkpoint: the within-relation AUROC of the knowledge direction
for separating known from unknown PopQA questions next to the subject's
popularity and two text classifiers, the difference from popularity with its
Hanley and McNeil half-width, the change in the AUROC for predicting wrong
answers when the direction or a supervised probe is added to the model's own
confidence, and how many steering runs distinguish the direction from random
directions and from known questions.
"""

from typing import Any

from tomography.site import runfiles
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.site.registry import TableOutput
from tomography.site.tables import _cells
from tomography.site.tables import _rows
from tomography.viz import paper

NO_CONTRAST = "no contrast"
NO_RUN = "none"
GROUPS = (
    ("Within-relation AUROC", 4, False),
    ("Difference", 1, False),
    ("Added to the model's confidence", 2, False),
    ("Steering runs", 2, False),
)
COLUMNS = [
    "Model",
    "Direction",
    "Popularity",
    "Words",
    "Characters",
    f"Direction {_cells.MINUS} popularity",
    "+ direction",
    "+ probe",
    "Distinguishable",
    "Selective",
]


def _p9b(manifest: Manifest, model: str) -> dict[str, Any] | None:
    run = manifest.run("p9b", model)
    assert run is not None
    found = [
        r
        for r in runfiles.rows(run, "table.json")
        if r["type"] == "obscure_real"
    ]
    return found[0] if found else None


def _increments(manifest: Manifest, model: str) -> dict[str, dict[str, Any]]:
    run = manifest.run("p9c", model)
    assert run is not None
    return {
        r["features"]: r
        for r in runfiles.rows(run, "table_incremental.json")
        if r["dataset"] == "popqa"
    }


def _gain(hit: dict[str, Any] | None) -> str:
    if hit is None:
        return NO_CONTRAST
    return _cells.interval(hit["diff_vs_confidence"], hit["lo"], hit["hi"])


def _steering(manifest: Manifest, model: str) -> tuple[int, int, int]:
    runs = manifest.runs("p9d", model)
    hits = selective = 0
    for info in runs:
        head = runfiles.summary(info.path)["headline"]
        hits += bool(head["distinguishable"])
        selective += bool(head["selective"])
    return len(runs), hits, selective


def build(manifest: Manifest) -> TableOutput:
    """Builds the table from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The table and its caption parts.
    """
    models = manifest.models("p9a")
    rows: list[list[str]] = []
    above: dict[str, bool] = {}
    distinct: dict[str, int] = {}
    select: dict[str, int] = {}
    totals: dict[str, int] = {}
    increments_above = {"direction": 0, "probe": 0}
    gain_upper: list[float] = []
    for model in models:
        reader = _p9b(manifest, model)
        inc = _increments(manifest, model)
        if reader is None:
            reading = [NO_CONTRAST] * 5
        else:
            n_pos = int(reader["n_positive_test"])
            n_neg = int(reader["n_negative_test"])
            d, lo, hi = stats.auc_diff_interval(
                reader["logistic_within"],
                reader["popularity_within"],
                n_pos,
                n_neg,
            )
            above[model] = lo > 0
            reading = [
                _cells.num(reader["logistic_within"]),
                _cells.num(reader["popularity_within"]),
                _cells.num(reader["text_word_within"]),
                _cells.num(reader["text_char_within"]),
                _cells.pm(d, (hi - lo) / 2) + _rows.star(lo, hi),
            ]
        for key, name in (("direction", "lat"), ("probe", "probe")):
            hit = inc.get(f"confidence+{name}")
            increments_above[key] += bool(hit and hit["lo"] > 0)
            if hit and key == "direction":
                gain_upper.append(float(hit["hi"]))
        found, hits, selective = _steering(manifest, model)
        totals[model] = found
        distinct[model], select[model] = hits, selective
        steer = (
            [f"{hits} of {found}", f"{selective} of {found}"]
            if found
            else [NO_RUN, NO_RUN]
        )
        rows.append(
            [
                paper.display_name(model),
                *reading,
                _gain(inc.get("confidence+lat")),
                _gain(inc.get("confidence+probe")),
                *steer,
            ]
        )

    def by_role(values: dict[str, int]) -> dict[str, stats.RoleCount]:
        out = {r: [0, 0] for r in stats.ROLES}
        for model, value in values.items():
            cell = out[stats.role(model)]
            cell[0] += value
            cell[1] += totals[model]
        return {r: stats.RoleCount(h, t) for r, (h, t) in out.items()}

    counts = {
        "above_popularity": stats.per_role(above),
        "distinguishable": by_role(distinct),
        "selective": by_role(select),
    }
    numbers = {
        "with_contrast": float(len(above)),
        "above_popularity": float(sum(above.values())),
        "direction_increment_above": float(increments_above["direction"]),
        "probe_increment_above": float(increments_above["probe"]),
        "gain_upper_max": max(gain_upper),
    }
    claim = (
        "Within-relation AUROC for PopQA questions the model answers "
        "correctly against those it does not, for the knowledge direction, "
        "the popularity of the subject and classifiers on the words and the "
        "characters of the question."
    )
    detail = (
        "Popularity is the monthly Wikipedia page views of the subject "
        "[[cite:mallen2023when]]; the difference from popularity carries the "
        "Hanley and McNeil half-width [[cite:hanley1982meaning]], with an "
        "asterisk where it excludes zero. The added columns give the change "
        "in AUROC for predicting wrong PopQA answers when the direction or a "
        "supervised probe is added to the confidence of the model, which "
        "stacks five output signals, with "
        "the stored paired bootstrap interval over the test questions. The "
        "steering columns count the runs whose direction moves unknown "
        "questions more than each of 3 random directions (distinguishable) "
        "and more than known questions (selective). GPT-2 small has too few "
        "known questions for a contrast."
    )
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
