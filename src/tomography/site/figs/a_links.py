"""Figure A8: The range direction against the knowledge directions.

For every checkpoint, the share of question types at each layer whose cosine
with the range direction, or whose pooled AUROC interval, lies outside the
band that random directions give, against relative depth.
"""

from collections import defaultdict
from collections.abc import Sequence
import csv
import json
from typing import Any

from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from tomography.site import stats
from tomography.site.figs import _group
from tomography.site.figs import _rows
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import axes
from tomography.viz import paper

HEIGHT = 3.4
CHANCE = 0.05
HEADER = ["row", "model", "layer", "depth", "share", "outside", "types"]
CLAIM = "The range direction against the knowledge directions at every layer."
DETAIL = (
    "Share of the question types whose cosine with the range direction lies "
    "outside the band of random directions (top row) and whose pooled AUROC "
    "interval lies outside the band of random directions (bottom row), "
    "against relative depth, one line per checkpoint (dashed: "
    "post-trained); the dotted line is the share expected for random "
    "vectors, 0.05. {layer} Over all layers the cosine lies outside its random "
    "band in {cos_gpt2} of {cells_gpt2} GPT-2 cells, {cos_base} of "
    "{cells_base} Qwen3 base cells and {cos_post} of {cells_post} "
    "post-trained cells, and the AUROC interval outside its band in "
    "{auc_gpt2}, {auc_base} and {auc_post} of the same cells."
)
LAYER_ONE = (
    "At layer 1, the layer the range reader chose on every checkpoint "
    "because ties go to the first layer, every cosine is at most {value} in "
    "absolute value; these values describe that layer and do not locate a "
    "link."
)
LAYER_OTHER = (
    "At the layer the range reader chose on each checkpoint, every cosine is "
    "at most {value} in absolute value."
)


def layer_sentence(layers: Sequence[int], biggest: float) -> str:
    """Writes the caption sentence on the layer the range reader chose.

    Args:
        layers: The chosen layer of every checkpoint.
        biggest: The largest absolute cosine at the chosen layers.

    Returns:
        The sentence.
    """
    value = f"{biggest:.3f}"
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


def _panel(ax: Axes, found: dict[str, list[tuple[int, int, int]]]) -> None:
    for model, shares in found.items():
        style = axes.model_style(model)
        layers = shares[-1][0]
        ax.plot(
            [k / layers for k, _, _ in shares],
            [hits / total for _, hits, total in shares],
            color=style.color,
            ls=style.linestyle,
            lw=paper.DATA_WIDTH,
            zorder=3,
        )
    axes.ref_line(ax, CHANCE)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.03, 1.03)
    ax.xaxis.set_major_locator(ticker.FixedLocator([0, 0.5, 1]))
    ax.xaxis.set_major_formatter(axes.minus_formatter())
    ax.yaxis.set_major_locator(ticker.FixedLocator([0, 0.5, 1]))
    ax.yaxis.set_major_formatter(axes.minus_formatter())


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
    biggest = 0.0
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
        biggest = max(
            biggest,
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

    with paper.paper_style():
        fig = _rows.new_figure(HEIGHT)
        grid = fig.subplots(2, 2, sharex=True, sharey=True)
        groups = (
            ("GPT-2", [m for m in models if m.startswith("gpt2")]),
            ("Qwen3", [m for m in models if not m.startswith("gpt2")]),
        )
        for r, row in enumerate(("cosine", "auroc")):
            for c, (title, members) in enumerate(groups):
                ax = grid[r][c]
                _panel(ax, {m: found[row][m] for m in members})
                paper.panel_letter(ax, "abcd"[2 * r + c])
                if r == 0:
                    ax.set_title(title, loc="left")
                if c == 0:
                    ax.set_ylabel(
                        "cosine outside band"
                        if row == "cosine"
                        else "AUROC outside band"
                    )
                if r == 1:
                    ax.set_xlabel("relative depth")
        grid[1][1].annotate(
            "expected for random vectors",
            xy=(0.85, CHANCE),
            xytext=(0.97, 0.4),
            ha="right",
            fontsize=6.5,
            color=paper.ENC["secondary_text"],
            arrowprops={
                "arrowstyle": "-",
                "lw": paper.REFERENCE_WIDTH,
                "color": paper.ENC["reference"],
            },
        )
        legend = fig.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    color=axes.model_style(m).color,
                    ls=axes.model_style(m).linestyle,
                    lw=paper.DATA_WIDTH,
                    label=paper.display_name(m),
                )
                for m in models
            ],
            loc="outside lower center",
            ncol=5,
            handlelength=1.0,
            columnspacing=1.0,
            fontsize=6.5,
        )
        legend.get_frame().set_edgecolor("none")

    counts: dict[str, dict[str, stats.RoleCount]] = {}
    numbers: dict[str, float] = {"layer1_cos_max": biggest}
    values: dict[str, str] = {}
    for name, row in (("cos", "cosine"), ("auc", "auroc")):
        tally = {role: [0, 0] for role in stats.ROLES}
        for model in models:
            cell = tally[stats.role(model)]
            for _, hits, total in found[row][model]:
                cell[0] += hits
                cell[1] += total
        counts[f"{name}_out"] = {
            role: stats.RoleCount(h, t) for role, (h, t) in tally.items()
        }
        for role, (h, t) in tally.items():
            numbers[f"{name}_out_{role}"] = float(h)
            numbers[f"cells_{role}"] = float(t)
            values[f"{name}_{role}"] = f"{h:,}"
            values[f"cells_{role}"] = f"{t:,}"
    detail = DETAIL.format(layer=layer_sentence(chosen, biggest), **values)
    return FigureOutput(
        fig, CLAIM, detail, numbers, _group.csv_text(HEADER, csv_rows), counts
    )
