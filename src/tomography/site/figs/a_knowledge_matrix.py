"""Appendix figure: Transfer of the knowledge reader between question types.

Panel (a) is the AUROC of a reader fitted on one question type and tested on
another, for the largest base checkpoint. Panel (b) gives, for each type, the
reader on its own type minus the best of the classifiers on the words and on
the characters of the questions, against a band of 0.020; a narrow column next
to it gives the AUROC of a random direction.
"""

from typing import Any

import matplotlib as mpl
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

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
CMAP = mpl.colormaps[paper.SEQUENTIAL_CMAP]
NUMBER_BELOW = 0.8
NUMBER_FONT = paper.MIN_TEXT
DETAIL = (
    "(a) AUROC of a logistic reader fitted on the question type in the row "
    "and tested on the type in the column, for {model} at its validation "
    "layer; the diagonal is outlined in white, and a number is written only "
    "on the diagonal and in cells below {below}. (b) The AUROC of the reader "
    "on its own type minus that of the best text classifier (the better of "
    "the one on the words and the one on the characters of the questions); "
    "the band marks a difference of {slack:.3f} either way, a filled marker "
    "is a reader ahead of the band ({ahead} of {types} types) and an open "
    "one is a reader inside it or behind. (c) The AUROC of a random "
    "direction on the same type, as a grey tick. A high value in a column "
    "of (a) does not show that the direction reads knowledge where a text "
    "classifier comes this close."
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
    best = [
        max(table[t]["text_word"] or 0.0, table[t]["text_char"] or 0.0)
        for t in names
    ]
    strip = {
        "reader": own,
        "words": [table[t]["text_word"] for t in names],
        "characters": [table[t]["text_char"] for t in names],
        "random": [table[t]["random"] for t in names],
        "difference": [o - b for o, b in zip(own, best, strict=True)],
    }
    for series, found in strip.items():
        for t, value in zip(names, found, strict=True):
            csv_rows.append(["b", t, "", series, value])
    ahead = sum(d > SLACK for d in strip["difference"])

    with paper.paper_style():
        fig = _rows.new_figure(HEIGHT)
        grid = fig.add_gridspec(1, 3, width_ratios=[3.0, 0.9, 0.5])
        ax_a = fig.add_subplot(grid[0])
        ax_b = fig.add_subplot(grid[1], sharey=ax_a)
        ax_c = fig.add_subplot(grid[2], sharey=ax_a)
        image = ax_a.imshow(
            values, cmap=CMAP, vmin=0.5, vmax=1.0, interpolation="nearest"
        )
        ax_a.set_aspect("auto")
        for i in range(len(names)):
            ax_a.add_patch(
                Rectangle(
                    (i - 0.5, i - 0.5),
                    1,
                    1,
                    fill=False,
                    ec="white",
                    lw=1.2,
                    joinstyle="miter",
                    zorder=3,
                )
            )
        for i in range(len(names)):
            for j in range(len(names)):
                if i != j and values[i, j] >= NUMBER_BELOW:
                    continue
                ax_a.text(
                    j,
                    i,
                    f"{values[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=NUMBER_FONT,
                    color=paper.tile_ink(CMAP((values[i, j] - 0.5) / 0.5)),
                    gid=paper.TILE_LABEL,
                    zorder=4,
                )
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
        ax_b.add_patch(
            Rectangle(
                (-SLACK, -0.5),
                2 * SLACK,
                len(names),
                fc=paper.RANDOM_BAND,
                ec="none",
                alpha=0.6,
                gid="band",
                zorder=1,
            )
        )
        ax_b.axvline(0.0, color=paper.INK, lw=0.5, zorder=2)
        for i, d in enumerate(strip["difference"]):
            ax_b.plot(
                [d],
                [i],
                ls="none",
                marker=paper.SHAPES["direction"],
                color=paper.INK,
                markerfacecolor=paper.INK if d > SLACK else "white",
                markeredgewidth=_rows.EDGE_WIDTH,
                markersize=_rows.MARKER_SIZE,
                gid="difference",
                zorder=4,
            )
        ax_b.set_xlim(-0.05, 0.25)
        ax_b.xaxis.set_major_locator(ticker.FixedLocator([0.0, 0.1, 0.2]))
        ax_b.xaxis.set_major_formatter(axes.minus_formatter())
        ax_b.set_xlabel("reader minus\nbest text")
        ax_b.tick_params(axis="y", length=0, labelleft=False)
        ax_b.spines["left"].set_visible(False)
        ax_b.grid(axis="y", color=paper.RANDOM_BAND, lw=0.4, zorder=0)
        for i, value in enumerate(strip["random"]):
            ax_c.plot(
                [value - 0.03, value + 0.03],
                [i, i],
                color=paper.RANDOM_GREY,
                lw=paper.DATA_WIDTH,
                solid_capstyle="butt",
                gid="random",
                zorder=3,
            )
        ax_c.set_xlim(0.5, 0.85)
        ax_c.xaxis.set_major_locator(ticker.FixedLocator([0.5, 0.8]))
        ax_c.xaxis.set_major_formatter(axes.minus_formatter())
        ax_c.set_xlabel("random\nAUROC")
        ax_c.tick_params(axis="y", length=0, labelleft=False)
        ax_c.spines["left"].set_visible(False)
        ax_c.grid(axis="y", color=paper.RANDOM_BAND, lw=0.4, zorder=0)
        ax_a.set_ylim(len(names) - 0.5, -0.5)
        paper.panel_title(
            ax_a, "a", f"reader transfer, {paper.display_name(model)}"
        )
        paper.panel_title(ax_b, "b", "vs text")
        paper.panel_title(ax_c, "c", "")
        legend = fig.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    ls="none",
                    marker=paper.SHAPES["direction"],
                    color=paper.INK,
                    markerfacecolor=paper.INK,
                    markersize=_rows.MARKER_SIZE,
                    label="reader ahead of the band",
                ),
                Line2D(
                    [],
                    [],
                    ls="none",
                    marker=paper.SHAPES["direction"],
                    color=paper.INK,
                    markerfacecolor="white",
                    markersize=_rows.MARKER_SIZE,
                    label="inside the band or behind",
                ),
                Line2D(
                    [],
                    [],
                    color=paper.RANDOM_GREY,
                    lw=paper.DATA_WIDTH,
                    label="random direction",
                ),
            ],
            loc="outside lower center",
            ncol=3,
            handlelength=1.4,
            columnspacing=1.2,
            fontsize=7,
        )
        legend.get_frame().set_edgecolor("none")
        _rows.freeze_layout(fig)

    count = sum(close)
    claim = (
        "A classifier on the words or characters alone comes within "
        f"{SLACK:.3f} of the reader or beats it on {count} of "
        f"{len(names)} question types."
    )
    numbers = {
        "close": float(count),
        "types": float(len(names)),
        "ahead": float(ahead),
    }
    return FigureOutput(
        fig,
        claim,
        DETAIL.format(
            model=paper.display_name(model),
            below=f"{NUMBER_BELOW:g}",
            slack=SLACK,
            ahead=ahead,
            types=len(names),
        ),
        numbers,
        _group.csv_text(HEADER, csv_rows),
        {},
    )
