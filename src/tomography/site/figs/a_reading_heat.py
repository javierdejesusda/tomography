"""Appendix figure: the margin of every reading at every layer, as heat maps.

One map per concept with a row per checkpoint: the reader's accuracy minus
that of a random direction of the same run at every layer, with the earliest
layer above 0.1 marked and the layer read circled. The curves by role are in
Figure 4; the maps keep the ten checkpoints in one view.
"""

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D

from tomography.site import reading
from tomography.site import store
from tomography.site.figs.reading import CONCEPTS
from tomography.site.figs.reading import MARGIN
from tomography.site.figs.reading import Cell
from tomography.site.figs.reading import cells_of
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import axes as shared
from tomography.viz import paper

HEIGHT = 3.2
VMAX = 0.6
HALO = 1.6
TICK = 0.8
TEXT = 6.5
TITLE_X = 0.14


def _heat(
    ax: Axes, cells: dict[str, Cell], rows: dict[str, float]
) -> AxesImage:
    image = None
    for model, cell in cells.items():
        y = rows[model]
        image = ax.imshow(
            cell.curve.selectivity[None, :],
            extent=(0.0, 1.0, y + 0.5, y - 0.5),
            aspect="auto",
            cmap="cividis",
            vmin=0.0,
            vmax=VMAX,
            interpolation="nearest",
        )
        count = cell.curve.n_layers
        first = reading.first_depth_above(cell.curve, MARGIN)
        if first is not None:
            x = first - 0.5 / count
            ax.plot([x, x], [y - 0.45, y + 0.45], color="white", lw=HALO)
            ax.plot([x, x], [y - 0.45, y + 0.45], color=paper.INK, lw=TICK)
        ax.plot(
            [(cell.at.layer - 0.5) / count],
            [y],
            ls="none",
            marker=paper.MEASURE_MARKERS["direction"],
            ms=2.6,
            mfc="white",
            mec=paper.INK,
            mew=0.6,
        )
    assert image is not None
    return image


def _key(fig: Figure) -> None:
    handles = [
        Line2D([], [], color=paper.INK, lw=TICK + 0.6),
        Line2D(
            [],
            [],
            ls="none",
            marker=paper.MEASURE_MARKERS["direction"],
            ms=2.6,
            mfc="white",
            mec=paper.INK,
            mew=0.6,
        ),
    ]
    fig.legend(
        handles,
        ["onset (earliest layer above 0.1)", "layer read"],
        loc="outside upper right",
        ncol=2,
        frameon=False,
        fontsize=TEXT,
        columnspacing=1.0,
        handlelength=1.2,
    )


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    found = {c: cells_of(manifest, c, exp) for c, _, exp in CONCEPTS}
    models = sorted(
        {m for cells in found.values() for m in cells}, key=store.model_rank
    )
    rows = shared.model_rows(models)
    with paper.paper_style():
        fig = Figure(figsize=(paper.FULL_WIDTH, HEIGHT), layout="constrained")
        axes = fig.subplots(1, len(CONCEPTS), sharey=True)
        image = None
        for col, (ax, (concept, label, _)) in enumerate(
            zip(axes, CONCEPTS, strict=True)
        ):
            image = _heat(ax, found[concept], rows)
            ax.set_ylim(max(rows.values()) + 0.5, -0.5)
            ax.set_xlim(0.0, 1.0)
            ax.set_title(label, fontsize=paper.TITLE, loc="left", x=TITLE_X)
            ax.set_xticks([0.0, 0.5, 1.0], ["0", "0.5", "1"])
            ax.set_xlabel("relative depth")
            ax.tick_params(axis="y", length=0)
            paper.panel_letter(ax, "abcd"[col])
        axes[0].set_yticks(
            list(rows.values()), [paper.display_name(m) for m in rows]
        )
        assert image is not None
        bar = fig.colorbar(
            image,
            ax=axes,
            orientation="horizontal",
            location="top",
            fraction=0.06,
            pad=0.04,
            shrink=0.5,
        )
        bar.set_ticks([0.0, 0.2, 0.4, 0.6])
        bar.ax.tick_params(labelsize=TEXT)
        bar.set_label("reader minus random accuracy", fontsize=TEXT)
        bar.outline.set_linewidth(0.4)
        _key(fig)
    claim = (
        "Reader accuracy minus random-direction accuracy at every layer, "
        "one heat map per concept and one row per checkpoint."
    )
    detail = (
        "The colour of a cell is the margin of the reader over the random "
        "direction of the same run (colour bar, capped at 0.6); the black "
        "tick marks the earliest layer where the margin exceeds 0.1 and the "
        "circle the layer read. The same curves, by role and with the "
        "random band, are in [[fig:reading]]."
    )
    return FigureOutput(fig, claim, detail, {}, _csv(found))


def _csv(found: dict[str, dict[str, Cell]]) -> str:
    lines = ["concept,model,depth,margin"]
    for concept, _, _ in CONCEPTS:
        for model, cell in found[concept].items():
            for depth, value in zip(
                cell.curve.depth, cell.curve.selectivity, strict=True
            ):
                lines.append(f"{concept},{model},{depth:.10g},{value:.10g}")
    return "\n".join(lines) + "\n"
