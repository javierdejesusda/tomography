"""Table A8: Sentiment reading under a decoy sentence.

One row per checkpoint: the accuracy of the sentiment reader on the original
prompts and on prompts with a sentence added that states a low or a high
amount of the concept, either drawn at random or contradicting the label.
"""

from tomography.site import params
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.site.registry import TableOutput
from tomography.site.tables import _cells
from tomography.site.tables import _rows
from tomography.viz import paper

CONDITIONS = ("original", "random", "against")
COLUMNS = ["Model", "Original", "Random decoy", "Against decoy"]


def build(manifest: Manifest) -> TableOutput:
    """Builds the table from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The table and its caption parts.
    """
    models = manifest.models("p2")
    n = params.REPORT_PAIRS["sentiment"]
    rows: list[list[str]] = []
    lowers: dict[str, bool] = {}
    for model in models:
        run = manifest.run("p2", model)
        assert run is not None
        acc = {
            r["condition"]: float(r["accuracy"])
            for r in _cells.read_table(run / "table.json")
            if r["task"] == "sentiment" and r["position"] == "last"
        }
        row = [paper.display_name(model)]
        for cond in CONDITIONS:
            lo, hi = stats.wilson(acc[cond], n)
            row.append(_cells.pm(acc[cond], (hi - lo) / 2))
        rows.append(row)
        _, _, upper = stats.diff_interval(acc["against"], acc["original"], n)
        lowers[model] = upper < 0
    claim = (
        "Sentiment reading when a sentence stating a low or a high amount of "
        "the concept is added to the prompt."
    )
    detail = (
        "The sentence is drawn at random (the paper's decoy) or always "
        "contradicting the label. The reader is trained on the original "
        f"prompts and read at the last token; each cell is the accuracy on "
        f"{n} report pairs with its 95 percent Wilson half-width."
    )
    return TableOutput(
        claim,
        detail,
        COLUMNS,
        rows,
        _cells.check_align("lrrr", len(COLUMNS)),
        [],
        {"report_pairs": float(n)},
        row_groups=_rows.role_groups(models),
        counts={"against_lowers": stats.per_role(lowers)},
    )
