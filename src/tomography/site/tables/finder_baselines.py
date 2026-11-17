"""Appendix table: The finder of the paper beside published finders.

One row per checkpoint and dataset: the test accuracy of the finder of the
paper, of four published baselines fitted on the same training pairs and of
two controls, each in the column group of the kind its run gives it. The two
text controls read no model, so a note gives them once per dataset and the
data file keeps them on every row.
"""

import json
from typing import Any

from tomography.site import cells
from tomography.site import params
from tomography.site import runfiles
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.site.registry import TableOutput
from tomography.viz import paper

EXPERIMENT = "b2"
NO_BREAK = chr(0x2011)
HEADERS = {
    "pca_diff": "PCA of differences",
    "mean_diff": f"Mass{NO_BREAK}mean",
    "whitened_mean_diff": f"Whitened mass{NO_BREAK}mean",
    "logistic": "Logistic regression",
    "ccs": "CCS",
    "random": "Random direction",
    "shuffled": "Shuffled labels",
}
TEXT = {"text_word": "Words", "text_char": "Characters"}
KIND_NAMES = {
    "reader": "Finder of the paper",
    "baseline": "Published baselines",
    "control": "Controls",
}
RANDOM = "random"


def _by_dataset(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Groups the rows of a run as dataset to method to row."""
    found: dict[str, dict[str, Any]] = {}
    for row in rows:
        found.setdefault(row["dataset"], {})[row["method"]] = row
    return found


def _mark(lo: float, hi: float) -> str:
    """Marks a difference from the finder of the paper by its direction.

    Args:
        lo: Lower limit of the interval of the difference.
        hi: Upper limit.

    Returns:
        An asterisk when the interval lies above zero, a dagger when it lies
        below, else an empty string.
    """
    return "*" if lo > 0 else "†" if hi < 0 else ""


def _above(row: dict[str, Any], other: dict[str, Any]) -> bool:
    """Tells whether one accuracy is above another by its interval."""
    _, lo, _ = stats.diff_interval(
        row["accuracy"], other["accuracy"], other["n_test_pairs"]
    )
    return lo > 0


def build(manifest: Manifest) -> TableOutput:
    """Builds the table from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The table and its caption parts.

    Raises:
        ValueError: If two runs give a method different kinds, differ in the
            number or the size of their training sets, or give a text
            control different accuracies.
    """
    rows: list[list[str]] = []
    text_cells: list[tuple[str, ...]] = []
    text_once: dict[str, set[tuple[str, ...]]] = {}
    starts: list[int] = []
    kinds: dict[str, str] = {}
    sizes: set[tuple[int, int]] = set()
    above_random: dict[str, bool] = {}
    baselines_above: dict[str, bool] = {}
    reach = 0.0
    for model in manifest.models(EXPERIMENT):
        run = manifest.run(EXPERIMENT, model)
        assert run is not None
        found = runfiles.summary(run)["kinds"]
        if kinds and found != kinds:
            raise ValueError(f"{EXPERIMENT}/{model} gives other kinds")
        kinds = found
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        sizes.add((config["n_seeds"], config["n_pairs"]))
        methods = sorted(
            (m for m in kinds if m not in TEXT),
            key=lambda m: list(KIND_NAMES).index(kinds[m]),
        )
        (reader,) = [m for m in methods if kinds[m] == "reader"]
        baselines = [m for m in methods if kinds[m] == "baseline"]
        table = _by_dataset(runfiles.rows(run, "table.json"))
        starts.append(len(rows))
        for place, (dataset, by) in enumerate(table.items()):
            ours = by[reader]
            row = [
                paper.display_name(model) if place == 0 else "",
                params.display(dataset),
                cells.num(ours["accuracy"]),
            ]
            reach = max(
                reach,
                *(r["accuracy"] - r["wilson_lo"] for r in by.values()),
                *(r["wilson_hi"] - r["accuracy"] for r in by.values()),
            )
            for method in methods[1:]:
                _, lo, hi = stats.diff_interval(
                    by[method]["accuracy"],
                    ours["accuracy"],
                    ours["n_test_pairs"],
                )
                row.append(cells.num(by[method]["accuracy"]) + _mark(lo, hi))
            rows.append(row)
            text = tuple(cells.num(by[m]["accuracy"]) for m in TEXT)
            text_cells.append(text)
            text_once.setdefault(params.display(dataset), set()).add(text)
        above_random[model] = any(
            _above(by[reader], by[RANDOM]) for by in table.values()
        )
        baselines_above[model] = all(
            any(_above(by[method], by[reader]) for by in table.values())
            for method in baselines
        )
    if len(sizes) != 1:
        raise ValueError(f"{EXPERIMENT} runs differ in their training sets")
    if any(len(found) != 1 for found in text_once.values()):
        raise ValueError(f"{EXPERIMENT} runs differ in their text controls")
    ((seeds, pairs),) = sizes
    counts = {
        "with_run": stats.per_role(
            {m: m in above_random for m in manifest.checkpoints()}
        ),
        "finder_above_random": stats.per_role(above_random),
        "every_baseline_above": stats.per_role(baselines_above),
    }
    spans = [
        (name, sum(kinds[m] == key for m in methods), False)
        for key, name in KIND_NAMES.items()
    ]
    claim = (
        "Test accuracy of the finder of the paper, four published finders "
        "and four controls on sentiment and three sets of true and false "
        f"statements: the mean over {seeds} training sets of {pairs} pairs "
        "at the layer each series chose on validation pairs."
    )
    detail = (
        "An asterisk (dagger) marks a value above (below) the finder of the "
        "paper by the conservative 95 percent interval of their difference. "
        "Unlike their references, the whitened mass-mean probe "
        "[[cite:marks2024geometry]] shrinks the covariance, and "
        "contrast-consistent search [[cite:burns2023discovering]] takes its "
        "sign from the training labels and reads one prompt on sentiment. "
        "The controls are a random direction, the finder scored against "
        "labels shuffled inside each pair, and the words alone (note under "
        "the table). The finder is above the random direction on at least "
        "one dataset, by that interval, on "
        f"{stats.per_role_text(counts['finder_above_random'])}."
    )
    return TableOutput(
        claim,
        detail,
        ["Model", "Dataset", *(HEADERS[m] for m in methods)],
        rows,
        "ll" + "r" * len(methods),
        [
            "Words and Characters read no model, so they give the same "
            "accuracy on every checkpoint and are listed here once: "
            + stats.join_and(
                [
                    f"{dataset} {word} and {char}"
                    for dataset, ((word, char),) in text_once.items()
                ]
            )
            + "."
        ],
        {
            "rows": float(len(rows)),
            "seeds": float(seeds),
            "train_pairs": float(pairs),
            "wilson_reach": reach,
        },
        groups=(("", 1, False), *spans),
        row_groups=tuple(starts),
        counts=counts,
        compact_print=True,
        data_columns=tuple(TEXT.values()),
        data_cells=tuple(text_cells),
        wide=True,
        roomy=True,
    )
