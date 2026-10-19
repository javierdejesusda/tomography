"""Table A4: Held-out sentiment accuracy against the training pairs.

One row per checkpoint and one column per number of training pairs. Each cell
is the reader next to the random direction at the same layer, on the report
split, averaged over the draws of training pairs.
"""

import numpy as np

from tomography.site import params
from tomography.site import runfiles
from tomography.site.manifest import Manifest
from tomography.site.registry import TableOutput
from tomography.site.tables import _cells
from tomography.site.tables import _rows
from tomography.viz import paper

SIZES = (2, 8, 32, 64, 128, 512)


def build(manifest: Manifest) -> TableOutput:
    """Builds the table from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The table and its caption parts.

    Raises:
        ValueError: If the runs average over different numbers of draws.
    """
    models = manifest.models("p3")
    rows: list[list[str]] = []
    draws: set[int] = set()
    for model in models:
        run = manifest.run("p3", model)
        assert run is not None
        meta = runfiles.data(run, "sizes_sentiment.json")
        accuracy = np.asarray(
            runfiles.arrays(run, "sizes_sentiment.npz")["accuracy"]
        )
        layer = int(meta["layer"]) - 1
        split = meta["splits"].index("report")
        reader = meta["series"].index("reader")
        random = meta["series"].index("random")
        draws.add(accuracy.shape[3])
        row = [paper.display_name(model)]
        for size in SIZES:
            at = accuracy[meta["sizes"].index(size), split, :, :, layer]
            row.append(
                f"{_cells.num(float(at[reader].mean()))} "
                f"({_cells.num(float(at[random].mean()))})"
            )
        rows.append(row)
    if len(draws) != 1:
        raise ValueError(f"the runs differ in their draws: {sorted(draws)}")
    n_draws = draws.pop()
    n_report = params.REPORT_PAIRS["sentiment"]
    claim = (
        "Held-out sentiment accuracy against the number of training pairs, "
        "with the random direction at the same layer in parentheses."
    )
    detail = (
        f"Each cell is the mean over {n_draws} draws of training pairs on "
        f"{n_report} report pairs, at the layer the run reports."
    )
    columns = ["Model", *(str(s) for s in SIZES)]
    return TableOutput(
        claim,
        detail,
        columns,
        rows,
        _cells.check_align("l" + "r" * len(SIZES), len(columns)),
        [],
        {"draws": float(n_draws), "report_pairs": float(n_report)},
        row_groups=_rows.role_groups(models),
    )
