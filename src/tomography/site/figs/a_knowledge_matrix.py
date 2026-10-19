"""Figure A5: Transfer of the knowledge reader between question types.

Panel (a) is the AUROC of a reader fitted on one question type and tested on
another, for the largest base checkpoint. Panel (b) puts the reader on its own
type next to classifiers on the words and on the characters of the questions
and next to a random direction, one row per type.
"""

from typing import Any

from matplotlib import ticker

from tomography.site import runfiles
from tomography.site import store
from tomography.site.figs import _group
from tomography.site.figs import _rows
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import axes
from tomography.viz import paper

HEIGHT = 3.6
SLACK = 0.02
HEADER = ["panel", "fitted", "tested", "series", "value"]
SERIES = (
    ("reader", "o", paper.INK),
    ("words", "^", paper.ENC["text"]),
    ("characters", "v", paper.ENC["text"]),
    ("random", "x", paper.ENC["random_marker"]),
)
DETAIL = (
    "(a) AUROC of a logistic reader fitted on the question type in the row "
    "and tested on the type in the column, for {model} at its validation "
    "layer; the diagonal is outlined and the columns carry the numbers of "
    "the rows. (b) The reader on its own type (black circle) next to a "
    "classifier on the words of the questions (green triangle up), one on "
    "the characters (green triangle down) and a random direction (grey "
    "cross). A high value in a column of (a) does not show that the "
    "direction reads knowledge where a text classifier comes this close."
)


def _model(manifest: Manifest) -> str:
    bases = [m for m in manifest.models("p9b") if not store.is_post_trained(m)]
    return max(bases, key=store.model_rank)


def _name(value: str) -> str:
    return value.replace("_", " ")


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    model = _model(manifest)
    run = manifest.run("p9b", model)
    assert run is not None
    meta = runfiles.data(run, "matrix.json")
    values = runfiles.arrays(run, "matrix.npz")["values"]
    table = {r["type"]: r for r in runfiles.rows(run, "table.json")}
    names: list[str] = list(meta["rows"])
    own = [float(values[i, i]) for i in range(len(names))]
    close = [
        max(table[t]["text_word"] or 0.0, table[t]["text_char"] or 0.0)
        >= own[i] - SLACK
        for i, t in enumerate(names)
    ]
    csv_rows: list[list[Any]] = []
    for i, fitted in enumerate(names):
        for j, tested in enumerate(names):
            csv_rows.append(
                ["a", fitted, tested, "matrix", float(values[i, j])]
            )
    strip = {
        "reader": own,
        "words": [table[t]["text_word"] for t in names],
        "characters": [table[t]["text_char"] for t in names],
        "random": [table[t]["random"] for t in names],
    }
    for series, found in strip.items():
        for t, value in zip(names, found, strict=True):
            csv_rows.append(["b", t, "", series, value])

    with paper.paper_style():
        fig = _rows.new_figure(HEIGHT)
        grid = fig.add_gridspec(1, 2, width_ratios=[2.6, 1.0])
        ax_a = fig.add_subplot(grid[0])
        ax_b = fig.add_subplot(grid[1], sharey=ax_a)
        image = axes.heatmap(ax_a, values, vmin=0.5, vmax=1.0)
        ax_a.set_aspect("auto")
        ax_a.set_xticks(range(len(names)))
        ax_a.set_xticklabels([str(k + 1) for k in range(len(names))])
        ax_a.set_yticks(range(len(names)))
        ax_a.set_yticklabels(
            [f"{k + 1} {_name(t)}" for k, t in enumerate(names)]
        )
        ax_a.tick_params(length=0)
        ax_a.set_xlabel("tested on question type")
        bar = fig.colorbar(
            image, ax=ax_a, location="top", shrink=0.55, aspect=30, pad=0.02
        )
        bar.set_label("AUROC")
        bar.ax.tick_params(labelsize=6.5)
        for series, marker, color in SERIES:
            ys = list(range(len(names)))
            ax_b.plot(
                strip[series],
                ys,
                ls="none",
                marker=marker,
                color=color,
                markersize=_rows.MARKER_SIZE,
                markeredgewidth=_rows.EDGE_WIDTH,
                zorder=3,
            )
        ax_b.set_xlim(0.5, 1.02)
        ax_b.xaxis.set_major_locator(ticker.FixedLocator([0.5, 0.75, 1.0]))
        ax_b.xaxis.set_major_formatter(axes.minus_formatter())
        ax_b.set_xlabel("AUROC on own type")
        ax_b.tick_params(axis="y", length=0, labelleft=False)
        ax_b.spines["left"].set_visible(False)
        ax_b.grid(axis="y", color=paper.RANDOM_BAND, lw=0.4, zorder=0)
        ax_a.set_ylim(len(names) - 0.5, -0.5)
        paper.panel_letter(ax_a, "a")
        paper.panel_letter(ax_b, "b")
        legend = fig.legend(
            handles=[
                _rows.series_handle(label, marker, color)
                for label, marker, color in SERIES
            ],
            loc="outside lower center",
            ncol=4,
            handlelength=1.0,
            columnspacing=1.2,
            fontsize=7,
        )
        legend.get_frame().set_edgecolor("none")

    count = sum(close)
    claim = (
        "A classifier on the words or characters alone comes within "
        f"{SLACK:.3f} of the reader on {count} of {len(names)} question "
        "types."
    )
    numbers = {"close": float(count), "types": float(len(names))}
    return FigureOutput(
        fig,
        claim,
        DETAIL.format(model=paper.display_name(model)),
        numbers,
        _group.csv_text(HEADER, csv_rows),
        {},
    )
