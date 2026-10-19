"""Table A4: Reader accuracy on every truth set.

One row per checkpoint and one column per truth set of the transfer
experiment. Each cell is the reader at the layer chosen on validation next to
the random direction at that layer.
"""

import numpy as np

from tomography.site import runfiles
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.site.registry import TableOutput
from tomography.site.tables import _cells
from tomography.site.tables import _rows
from tomography.viz import paper

SETS = (
    ("cities", "cities"),
    ("neg_cities", "negated cities"),
    ("sp_en_trans", "Spanish to English"),
    ("neg_sp_en_trans", "negated Spanish to English"),
    ("larger_than", "larger than"),
    ("smaller_than", "smaller than"),
    ("cities_cities_conj", "cities conjunction"),
    ("cities_cities_disj", "cities disjunction"),
    ("common_claim_true_false", "common claims"),
)


def build(manifest: Manifest) -> TableOutput:
    """Builds the table from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The table and its caption parts.
    """
    models = manifest.models("r7")
    rows: list[list[str]] = []
    below = 0
    for model in models:
        run = manifest.run("r7", model)
        assert run is not None
        summary = runfiles.summary(run)
        layer = int(summary["val_layer"])
        row = [paper.display_name(model)]
        for key, _ in SETS:
            block = summary["sets"][key]
            reader = float(block["accuracy_at_val_layer"])
            series = runfiles.data(run, f"curves_{key}.json")["series"]
            mean = runfiles.arrays(run, f"curves_{key}.npz")["mean"]
            random = float(np.asarray(mean)[series.index("random")][layer - 1])
            _, hi = stats.wilson(reader, int(block["n_eval_pairs"]))
            mark = "*" if hi < 0.5 else ""
            below += bool(mark)
            row.append(f"{_cells.num(reader)}{mark} ({_cells.num(random)})")
        rows.append(row)
    claim = "Reader accuracy on every truth set, next to the random direction."
    detail = (
        "Each cell is the accuracy of the reader at the layer chosen on "
        "validation pairs, with the accuracy of the random direction at that "
        "layer in parentheses. A star marks a reader whose 95 percent Wilson "
        "interval lies below 0.5, that is, a reader that gets the label "
        "reversed on that set."
    )
    columns = ["Model", *(name for _, name in SETS)]
    return TableOutput(
        claim,
        detail,
        columns,
        rows,
        _cells.check_align("l" + "r" * len(SETS), len(columns)),
        [],
        {"below_half": float(below)},
        row_groups=_rows.role_groups(models),
    )
