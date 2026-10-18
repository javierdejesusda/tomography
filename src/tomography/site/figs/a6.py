"""Figure A6: P10c cosine and AUROC against the random band.

For every checkpoint, two heatmaps over P9 type and relative depth: the
cosine between the range direction and the P9 knowledge direction, and the
AUROC of the range direction for the pooled knowledge contrast. A value is
drawn only when it lies outside the band that random directions give; values
inside the band are masked.
"""

import csv
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
import numpy as np

from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import paper

DIVERGING = LinearSegmentedColormap.from_list(
    "paper_diverging", [paper.SERIES[1], "#ffffff", paper.SERIES[0]]
)
COSINE_LIMIT = 0.3
TICK_SIZE = 4.5


def _read(run: Path, name: str) -> list[dict[str, str]]:
    with (run / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _grid(
    types: list[str],
    layers: int,
    cells: dict[tuple[str, int], tuple[float, bool]],
) -> np.ma.MaskedArray:
    values = np.zeros((len(types), layers))
    masked = np.ones((len(types), layers), dtype=bool)
    for (kind, layer), (value, outside) in cells.items():
        values[types.index(kind), layer - 1] = value
        masked[types.index(kind), layer - 1] = not outside
    return np.ma.MaskedArray(values, mask=masked)


def _draw(
    ax: Axes,
    grid: np.ma.MaskedArray,
    types: list[str],
    vmin: float,
    vmax: float,
) -> None:
    ax.imshow(
        grid,
        cmap=DIVERGING,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
        extent=(0.0, 1.0, len(types) - 0.5, -0.5),
    )
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"], fontsize=TICK_SIZE)
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels([t.replace("_", " ") for t in types], fontsize=TICK_SIZE)
    ax.tick_params(length=1.5, pad=1)


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    models = manifest.models("p10c")
    csv_rows: list[list[str]] = []
    totals = {"cos_out": 0, "cos_all": 0, "auc_out": 0, "auc_all": 0}
    with paper.paper_style():
        fig = Figure(
            figsize=(paper.FULL_WIDTH, 1.55 * ((len(models) + 1) // 2) + 0.6),
            layout="constrained",
        )
        grid = fig.add_gridspec((len(models) + 1) // 2, 4)
        letters = iter("abcdefghijklmnopqrst")
        for i, model in enumerate(models):
            run = manifest.run("p10c", model)
            if run is None:
                continue
            cosine = _read(run, "cosine.csv")
            know = [
                r
                for r in _read(run, "knowledge.csv")
                if r["items"] == "all" and r["test_family"] == "pooled"
            ]
            types = list(dict.fromkeys(r["p9_type"] for r in cosine))
            layers = max(int(r["layer"]) for r in cosine)
            cos_cells = {}
            for r in cosine:
                value = float(r["cosine"])
                outside = (
                    not float(r["band_lo"]) <= value <= float(r["band_hi"])
                )
                cos_cells[(r["p9_type"], int(r["layer"]))] = (value, outside)
                totals["cos_all"] += 1
                totals["cos_out"] += outside
                csv_rows.append(
                    [
                        "cosine",
                        model,
                        f"{int(r['layer']) / layers:.6g}",
                        f"{value:.6g}",
                        r["band_lo"],
                        r["band_hi"],
                        r["p9_type"],
                    ]
                )
            auc_cells = {}
            for r in know:
                outside = not (
                    float(r["lo"]) <= float(r["rand_hi"])
                    and float(r["hi"]) >= float(r["rand_lo"])
                )
                auc_cells[(r["p9_type"], int(r["layer"]))] = (
                    float(r["auroc"]),
                    outside,
                )
                totals["auc_all"] += 1
                totals["auc_out"] += outside
                csv_rows.append(
                    [
                        "auroc",
                        model,
                        f"{int(r['layer']) / layers:.6g}",
                        r["auroc"],
                        r["rand_lo"],
                        r["rand_hi"],
                        r["p9_type"],
                    ]
                )
            row, col = divmod(i, 2)
            left = fig.add_subplot(grid[row, 2 * col])
            right = fig.add_subplot(grid[row, 2 * col + 1])
            _draw(
                left,
                _grid(types, layers, cos_cells),
                types,
                -COSINE_LIMIT,
                COSINE_LIMIT,
            )
            auc_types = list(dict.fromkeys(r["p9_type"] for r in know))
            _draw(
                right, _grid(auc_types, layers, auc_cells), auc_types, 0.0, 1.0
            )
            right.set_yticklabels([])
            left.set_title(
                paper.display_name(model), fontsize=paper.TITLE, x=0.2
            )
            paper.panel_letter(left, next(letters))
            if row == (len(models) - 1) // 2:
                left.set_xlabel("relative depth", fontsize=paper.LABEL)
                right.set_xlabel("relative depth", fontsize=paper.LABEL)
    numbers = {
        "cosine_outside": float(totals["cos_out"]),
        "cosine_cells": float(totals["cos_all"]),
        "auroc_outside": float(totals["auc_out"]),
        "auroc_cells": float(totals["auc_all"]),
    }
    claim = (
        f"Over all layers {totals['cos_out']} of {totals['cos_all']} cosines "
        "between the range direction and a P9 knowledge direction lie "
        f"outside the random band, and {totals['auc_out']} of "
        f"{totals['auc_all']} pooled AUROC intervals do."
    )
    detail = (
        "For each checkpoint, left: the cosine between the range direction "
        "and the direction of each P9 type at every layer (color from -0.3 "
        "to 0.3); right: the AUROC of the range direction for the pooled "
        "knowledge contrast (color from 0 to 1, white at 0.5). A cell is "
        "drawn only when the cosine lies outside the band of cosines of "
        "random directions, or the AUROC interval does not overlap the band "
        "of AUROCs of random directions; the other cells are masked."
    )
    return FigureOutput(fig, claim, detail, numbers, _csv(csv_rows))


def _csv(rows: list[list[str]]) -> str:
    header = "panel,model,x,y,band_lo,band_hi,p9_type"
    return "\n".join([header, *(",".join(r) for r in rows)]) + "\n"
