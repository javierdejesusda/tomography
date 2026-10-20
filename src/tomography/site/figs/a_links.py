"""Appendix figure: The range direction against the knowledge directions.

For every checkpoint, the share of layer-by-type cells whose cosine with the
range direction, or whose pooled AUROC interval, lies outside the band that
random directions give: one dot per checkpoint in role blocks, and a heat
strip of the cosine share by relative depth.
"""

from collections import defaultdict
from collections.abc import Mapping
from collections.abc import Sequence
import csv
import json
import math
from typing import Any

from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.colors import Normalize
import numpy as np

from tomography.site import stats
from tomography.site.figs import _group
from tomography.site.figs import _rows
from tomography.site.figs import knowledge_reading as rows_of
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import axes
from tomography.viz import paper

HEIGHT = 3.8
CHANCE = 0.05
BINS = 6
HEAD_Y = -0.95
TOP = -1.4
HEADER = ["row", "model", "layer", "depth", "share", "outside", "types"]
CLAIM = "The range direction against the knowledge directions at every layer."
DETAIL = (
    "Share of the layer-by-type cells whose cosine with the range direction "
    "lies outside the band of random directions (a) and whose pooled AUROC "
    "interval lies outside the band of random directions (b), one circle "
    "per checkpoint in role blocks (filled: GPT-2 and Qwen3 base; open: "
    "post-trained), with the number of cells printed at the right of each "
    "row; the dotted line is the share expected for random vectors, 0.05. "
    "(c) The cosine share by relative depth, in six bins of layers, one row "
    "per checkpoint, with 0.05 marked on the colour bar. {layer} Over all "
    "layers the cosine lies outside its random band in {cos_gpt2} of "
    "{cells_gpt2} GPT-2 cells, {cos_base} of {cells_base} Qwen3 base cells "
    "and {cos_post} of {cells_post} post-trained cells, and the AUROC "
    "interval outside its band in {auc_gpt2}, {auc_base} and {auc_post} of "
    "the same cells. Each share is a count over few question types "
    "({types_min} to {types_max} per layer), so it moves in steps, and "
    "adjacent layers share a residual stream, so the cells are not "
    "independent and 0.05 is only a rough reference."
)
LAYER_ONE = (
    "At layer 1, the layer the plain-template range reader chose on every "
    "checkpoint because ties go to the first layer, every cosine is at most "
    "{value} in absolute value; these values describe that layer and do not "
    "locate a link."
)
LAYER_OTHER = (
    "At the layer the plain-template range reader chose on each checkpoint, "
    "every cosine is at most {value} in absolute value."
)
LIMITS = {"cosine": (-0.04, 1.15), "auroc": (-0.012, 0.16)}
TICKS = {"cosine": [0.0, 0.5], "auroc": [0.0, 0.1]}
LABELS = {
    "cosine": "cosine outside\nthe random band",
    "auroc": "AUROC outside\nthe random band",
}
SEPARATOR_WIDTH = 0.5
GROUP_X = -0.68


def layer_sentence(layers: Sequence[int], biggest: Mapping[str, float]) -> str:
    """Writes the caption sentence on the layer the range reader chose.

    Args:
        layers: The chosen layer of every checkpoint.
        biggest: Role to the largest absolute cosine at the chosen layers.

    Returns:
        The sentence.
    """
    value = stats.join_and(
        [f"{biggest[r]:.3f} ({stats.ROLE_NAMES[r]})" for r in stats.ROLES]
    )
    if set(layers) == {1}:
        return LAYER_ONE.format(value=value)
    return LAYER_OTHER.format(value=value)


def _read(run: Any, name: str) -> list[dict[str, str]]:
    with (run / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _cosine_outside(row: dict[str, str]) -> bool:
    return (
        not float(row["band_lo"])
        <= float(row["cosine"])
        <= float(row["band_hi"])
    )


def _auroc_outside(row: dict[str, str]) -> bool:
    return not (
        float(row["lo"]) <= float(row["rand_hi"])
        and float(row["hi"]) >= float(row["rand_lo"])
    )


def _shares(
    rows: list[dict[str, str]], outside: Any
) -> list[tuple[int, int, int]]:
    cells: dict[int, list[bool]] = defaultdict(list)
    for r in rows:
        cells[int(r["layer"])].append(bool(outside(r)))
    return [(k, sum(v), len(v)) for k, v in sorted(cells.items())]


def _tally(
    shares: dict[str, list[tuple[int, int, int]]],
) -> dict[str, list[int]]:
    tally = {role: [0, 0] for role in stats.ROLES}
    for model, cells in shares.items():
        cell = tally[stats.role(model)]
        for _, hits, total in cells:
            cell[0] += hits
            cell[1] += total
    return tally


def _totals(
    found: dict[str, list[tuple[int, int, int]]],
) -> dict[str, tuple[int, int]]:
    return {
        m: (sum(h for _, h, _ in c), sum(t for _, _, t in c))
        for m, c in found.items()
    }


def _bins(cells: list[tuple[int, int, int]]) -> list[float]:
    layers = cells[-1][0]
    hits = [0] * BINS
    total = [0] * BINS
    for k, h, t in cells:
        index = min(math.ceil(k / layers * BINS) - 1, BINS - 1)
        hits[index] += h
        total[index] += t
    return [h / t for h, t in zip(hits, total, strict=True)]


def _head(ax: Axes, x: float, text: str, ha: str) -> None:
    ax.text(
        x,
        HEAD_Y,
        text,
        transform=ax.get_yaxis_transform(),
        color=paper.ENC["secondary_text"],
        fontsize=paper.MIN_TEXT,
        ha=ha,
        va="center",
        linespacing=0.95,
    )


def _dots(
    ax: Axes,
    row: str,
    totals: dict[str, tuple[int, int]],
    rows: dict[str, float],
) -> None:
    for model in rows:
        hits, total = totals[model]
        _rows.dot(
            ax,
            model,
            rows[model],
            hits / total,
            paper.MEASURE_MARKERS["direction"],
            axes.model_style(model).color,
        )
        ax.text(
            0.99,
            rows[model],
            f"{hits:,} of {total:,}",
            transform=ax.get_yaxis_transform(),
            color=paper.ENC["secondary_text"],
            fontsize=paper.MIN_TEXT,
            ha="right",
            va="center",
        )
    axes.ref_line(ax, CHANCE, "x")
    ax.set_xlim(*LIMITS[row])
    ax.xaxis.set_major_locator(ticker.FixedLocator(TICKS[row]))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel(LABELS[row])
    low, high = ax.get_xlim()
    ax.text(
        CHANCE + 0.015 * (high - low),
        HEAD_Y,
        "expected for\nrandom vectors",
        color=paper.ENC["secondary_text"],
        fontsize=paper.MIN_TEXT,
        ha="left",
        va="center",
        linespacing=0.95,
    )


def _heat(
    ax: Axes,
    cosine: dict[str, list[tuple[int, int, int]]],
    rows: dict[str, float],
) -> tuple[QuadMesh, float]:
    profile = {m: _bins(c) for m, c in cosine.items()}
    top = math.ceil(max(max(v) for v in profile.values()) * 10) / 10
    norm = Normalize(vmin=0.0, vmax=top)
    edges = np.linspace(0.0, 1.0, BINS + 1)
    mesh = None
    for members in _group.lines_by_role(list(rows)).values():
        ys = [rows[m] for m in members]
        mesh = ax.pcolormesh(
            edges,
            np.array([ys[0] - 0.5] + [y + 0.5 for y in ys]),
            np.array([profile[m] for m in members]),
            cmap=paper.SEQUENTIAL_CMAP,
            norm=norm,
            edgecolors="none",
            zorder=2,
        )
    assert mesh is not None
    ax.set_xlim(0.0, 1.0)
    ax.xaxis.set_major_locator(ticker.FixedLocator([0.0, 0.5, 1.0]))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("relative depth")
    _head(ax, 0.5, "cosine share\nby depth", "center")
    return mesh, top


def _draw(
    fig: Any,
    found: dict[str, dict[str, list[tuple[int, int, int]]]],
    rows: dict[str, float],
    separators: list[float],
) -> None:
    grid = fig.add_gridspec(1, 4, width_ratios=[1.0, 0.9, 1.0, 0.07])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1], sharey=ax_a)
    ax_c = fig.add_subplot(grid[0, 2], sharey=ax_a)
    cax = fig.add_subplot(grid[0, 3])
    _dots(ax_a, "cosine", _totals(found["cosine"]), rows)
    _dots(ax_b, "auroc", _totals(found["auroc"]), rows)
    mesh, top = _heat(ax_c, found["cosine"], rows)
    bar = fig.colorbar(mesh, cax=cax)
    bar.set_ticks([0.0, CHANCE, top])
    bar.ax.set_yticklabels(["0", "0.05 random", f"{top:g}"])
    bar.set_label("share of cells\noutside the band")
    bar.ax.axhline(CHANCE, color=paper.INK, lw=paper.REFERENCE_WIDTH)
    bar.outline.set_linewidth(paper.REFERENCE_WIDTH)
    panels = [ax_a, ax_b, ax_c]
    for position, ax in enumerate(panels):
        _rows.row_axis(ax, rows, labels=position == 0)
        ax.set_ylim(max(rows.values()) + 0.6, TOP)
        for y in separators:
            ax.axhline(y, color=paper.RANDOM_BAND, lw=SEPARATOR_WIDTH, zorder=0)
        paper.panel_letter(ax, "abc"[position])
    ax_a.set_yticklabels([rows_of.row_label(m) for m in rows])
    ax_c.tick_params(axis="y", labelleft=True)
    ax_c.set_yticklabels([rows_of.row_label(m) for m in rows])
    rows_of.draw_group_labels(ax_a, rows, GROUP_X)
    legend = fig.legend(
        handles=[
            _rows.series_handle(
                "GPT-2 and Qwen3 base",
                paper.MEASURE_MARKERS["direction"],
                paper.ENC["reference"],
            ),
            _rows.series_handle(
                "post-trained",
                paper.MEASURE_MARKERS["direction"],
                paper.ENC["reference"],
                filled=False,
            ),
        ],
        loc="outside upper center",
        ncol=2,
        handlelength=1.0,
        columnspacing=1.8,
        fontsize=7,
    )
    legend.get_frame().set_edgecolor("none")


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    models = manifest.models("p10c")
    sources: dict[str, dict[str, list[dict[str, str]]]] = {
        "cosine": {},
        "auroc": {},
    }
    chosen: list[int] = []
    biggest = {r: 0.0 for r in stats.ROLES}
    for model in models:
        run = manifest.run("p10c", model)
        assert run is not None
        cosine = _read(run, "cosine.csv")
        sources["cosine"][model] = cosine
        sources["auroc"][model] = [
            r
            for r in _read(run, "knowledge.csv")
            if r["items"] == "all" and r["test_family"] == "pooled"
        ]
        layer = int(
            json.loads((run / "summary.json").read_text("utf-8"))["data"][
                "layer"
            ]
        )
        chosen.append(layer)
        role = stats.role(model)
        biggest[role] = max(
            biggest[role],
            max(
                abs(float(r["cosine"]))
                for r in cosine
                if int(r["layer"]) == layer
            ),
        )
    flags = {"cosine": _cosine_outside, "auroc": _auroc_outside}
    found = {
        row: {m: _shares(sources[row][m], flags[row]) for m in models}
        for row in sources
    }
    csv_rows: list[list[Any]] = []
    for row, by_model in found.items():
        for model, shares in by_model.items():
            layers = shares[-1][0]
            for k, hits, total in shares:
                csv_rows.append(
                    [row, model, k, k / layers, hits / total, hits, total]
                )
        for model, (hits, total) in _totals(by_model).items():
            csv_rows.append(
                [f"{row}_all", model, None, None, hits / total, hits, total]
            )
    rows, separators = rows_of.block_rows(models)

    with paper.paper_style():
        fig = _rows.new_figure(HEIGHT)
        _draw(fig, found, rows, separators)
        _rows.freeze_layout(fig)

    counts: dict[str, dict[str, stats.RoleCount]] = {}
    numbers: dict[str, float] = {
        f"layer1_cos_max_{role}": biggest[role] for role in stats.ROLES
    }
    values: dict[str, str] = {}
    for name, row in (("cos", "cosine"), ("auc", "auroc")):
        tally = _tally(found[row])
        counts[f"{name}_out"] = {
            role: stats.RoleCount(h, t) for role, (h, t) in tally.items()
        }
        for role, (h, t) in tally.items():
            numbers[f"{name}_out_{role}"] = float(h)
            numbers[f"cells_{role}"] = float(t)
            values[f"{name}_{role}"] = f"{h:,}"
            values[f"cells_{role}"] = f"{t:,}"
    types = [t for cells in found["cosine"].values() for _, _, t in cells]
    detail = DETAIL.format(
        layer=layer_sentence(chosen, biggest),
        types_min=min(types),
        types_max=max(types),
        **values,
    )
    return FigureOutput(
        fig, CLAIM, detail, numbers, _group.csv_text(HEADER, csv_rows), counts
    )
