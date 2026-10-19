"""Table 3: The paper's finder and logistic regression against random.

For each checkpoint and task the random direction's report accuracy, and the
margin of the paper's PCA finder and of logistic regression over it, each at
the layer its own select split chose. A star marks a margin whose conservative
interval for the report pairs excludes zero.
"""

from tomography.site import finders
from tomography.site import params
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.site.registry import TableOutput
from tomography.site.tables import _cells
from tomography.site.tables import _rows
from tomography.viz import paper

TASK_NAMES = {"sentiment": "Sentiment", "truth": "Truth", "utility": "Utility"}
HEADERS = ("Random", "PCA of paired differences (paper)", "Logistic regression")


def build(manifest: Manifest) -> TableOutput:
    """Builds the table from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The table and its caption parts.
    """
    models = [m for m in manifest.models("p4") if finders.load(manifest, m)]
    rows: list[list[str]] = []
    starred = 0
    for model in models:
        found = finders.load(manifest, model)
        assert found is not None
        row = [paper.display_name(model)]
        for task in finders.TASKS:
            cell = found[task].rows
            random = cell["random"].report
            row.append(_cells.num(random))
            for method in ("pca_diff", "logistic"):
                d, lo, hi = stats.diff_interval(
                    cell[method].report, random, params.REPORT_PAIRS[task]
                )
                mark = _rows.star(lo, hi)
                starred += bool(mark)
                row.append(_cells.num(d) + mark)
        rows.append(row)
    sizes = params.REPORT_PAIRS
    claim = (
        "The paper's finder and logistic regression against a random "
        "direction, on the report split at the layer the select split chose."
    )
    detail = (
        "Random is the accuracy of the random direction; the two finder "
        "columns give the finder's accuracy minus it, with an asterisk where "
        "the conservative 95 percent interval for the test pairs "
        f"({sizes['sentiment']} sentiment, {sizes['truth']} truth and "
        f"{sizes['utility']} utility pairs) excludes zero. Sentiment and "
        "truth use pairs of unrelated inputs with opposite labels; utility "
        "uses matched scenarios. Every finder and preprocessing is in "
        "[[table:finders_all]]."
    )
    return TableOutput(
        claim,
        detail,
        ["Model", *HEADERS * len(finders.TASKS)],
        rows,
        "l" + "r" * 3 * len(finders.TASKS),
        [],
        {
            "starred": float(starred),
            **{f"n_{t}": float(sizes[t]) for t in sizes},
        },
        groups=tuple((TASK_NAMES[t], 3, False) for t in finders.TASKS),
        row_groups=_rows.role_groups(models),
    )
