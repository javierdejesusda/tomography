"""Table A2: Every finder, and the manipulation win rate of utility finders.

Rows are finders grouped by task; columns are checkpoints. A reading cell is
the finder's report accuracy minus the random direction's, with a star where
the conservative interval for the report pairs excludes zero; the random row
holds the random direction's accuracy. The last block gives the win rate of
the directions fitted on utility pairs when they steer sentiment.
"""

from typing import Any

from tomography.site import cells
from tomography.site import finders
from tomography.site import params
from tomography.site import runfiles
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.site.registry import TableOutput
from tomography.site.tables import _rows
from tomography.viz import paper

METHODS = ("pca_diff", "pca_raw", "mean_diff", "logistic", "kmeans", "random")
TASK_NAMES = {"sentiment": "Sentiment", "truth": "Truth", "utility": "Utility"}
STEERING = "Win rate at 0.3 (utility directions, sentiment judge)"


def _report(manifest: Manifest, model: str, task: str, method: str) -> float:
    run = manifest.run("p4", model)
    assert run is not None
    pairing = runfiles.summary(run)["reference"][task]["pairing"]
    (hit,) = [
        r
        for r in runfiles.rows(run, "table.json")
        if r["task"] == task
        and r["method"] == method
        and r["pairing"] == pairing
        and r["preprocess"] == finders.PREPROCESS
    ]
    return float(hit["report"])


def _win(manifest: Manifest, model: str, method: str) -> dict[str, Any]:
    run = manifest.run("m1", model)
    assert run is not None
    (hit,) = [
        r
        for r in runfiles.rows(run, "table.json")
        if r["method"] == method
        and r["task"] == "manipulation"
        and r["setting"] == "0.3"
    ]
    return hit


def build(manifest: Manifest) -> TableOutput:
    """Builds the table from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The table and its caption parts.
    """
    models = manifest.models("p4")
    rows: list[list[str]] = []
    starts: list[int] = []
    for task in finders.TASKS:
        starts.append(len(rows))
        n = params.REPORT_PAIRS[task]
        for k, method in enumerate(METHODS):
            row = [TASK_NAMES[task] if k == 0 else "", params.display(method)]
            for model in models:
                random = _report(manifest, model, task, "random")
                if method == "random":
                    row.append(cells.num(random))
                    continue
                d, lo, hi = stats.diff_interval(
                    _report(manifest, model, task, method), random, n
                )
                row.append(cells.num(d) + _rows.star(lo, hi))
            rows.append(row)
    starts.append(len(rows))
    for k, method in enumerate(METHODS):
        row = [STEERING if k == 0 else "", params.display(method)]
        for model in models:
            hit = _win(manifest, model, method)
            row.append(cells.pm(hit["value"], (hit["hi"] - hit["lo"]) / 2))
        rows.append(row)
    claim = (
        "Every finder on the report split, as accuracy minus the random "
        "direction (random row: its accuracy), and the manipulation win rate "
        "of directions fitted on utility pairs."
    )
    detail = (
        "Reading cells use the layer each finder chose on its select split "
        "and an asterisk marks a conservative 95 percent interval for the "
        "report pairs that excludes zero. The win rate is the share of "
        "decisive prompt pairs in which the completion steered at +0.3 is "
        "judged more positive than the one steered at "
        f"{cells.MINUS}0.3 by the sentiment judge, with the Wilson "
        "half-width. Three random directions of the same norm are the "
        "reference arm for the win rate; 0.5 is not. Mean difference is the "
        "mass-mean probe [[cite:marks2024geometry]] without whitening and "
        "logistic regression the linear probe [[cite:alain2016understanding]]; "
        "both are published baselines for the finder of the paper. The random "
        "direction is the control."
    )
    return TableOutput(
        claim,
        detail,
        ["Group", "Finder", *(paper.display_name(m) for m in models)],
        rows,
        "ll" + "r" * len(models),
        [],
        {"models": float(len(models))},
        row_groups=tuple(starts),
        compact_print=True,
    )
