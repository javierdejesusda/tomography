"""Appendix figure: where in the network a concept is read.

A grid of layer curves with a column per concept (sentiment, utility,
morality, truth) and a row per role (GPT-2, Qwen3 base, Qwen3 post-trained),
never pooled. Each panel draws, for every checkpoint of the role, the reader's
accuracy minus that of a random direction of the same run against relative
depth, with a dot at the layer read and a tick on the x axis at the onset (the
earliest layer above 0.1); the grey band is the stored interval of the random
direction over seeds, drawn around the zero line. Sentiment curves are the
report-split curves of the finder comparison.
"""

import re

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np

from tomography.site import reading
from tomography.site import stats
from tomography.site import store
from tomography.site.figs.reading import CONCEPTS
from tomography.site.figs.reading import CURVE_LABEL_Y
from tomography.site.figs.reading import CURVE_TICKS
from tomography.site.figs.reading import CURVE_YLIM
from tomography.site.figs.reading import DOT
from tomography.site.figs.reading import GPT2_LABELS
from tomography.site.figs.reading import GRID
from tomography.site.figs.reading import LINE
from tomography.site.figs.reading import MARGIN
from tomography.site.figs.reading import ROLE_LABEL
from tomography.site.figs.reading import ROLE_ROWS
from tomography.site.figs.reading import TEXT
from tomography.site.figs.reading import TICK_HEIGHT
from tomography.site.figs.reading import Cell
from tomography.site.figs.reading import cells_of
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import axes as shared
from tomography.viz import paper

HEIGHT = 4.6
WIDTH_RATIOS = [1.0, 1.0, 1.0, 1.0, 0.34]


def _end_label(model: str) -> str:
    if model in GPT2_LABELS:
        return GPT2_LABELS[model]
    match = re.match(r"qwen3-(\d+(?:\.\d+)?)b", model)
    assert match is not None, model
    return f"{match.group(1)}B"


def _band(cells: dict[str, Cell]) -> tuple[np.ndarray, np.ndarray]:
    low, high = [], []
    for cell in cells.values():
        curve = cell.curve
        assert curve.random_lo is not None
        assert curve.random_hi is not None
        low.append(np.interp(GRID, curve.depth, curve.random_lo - curve.random))
        high.append(
            np.interp(GRID, curve.depth, curve.random_hi - curve.random)
        )
    return np.min(low, axis=0), np.max(high, axis=0)


def _curve_panel(ax: Axes, cells: dict[str, Cell]) -> None:
    low, high = _band(cells)
    band = ax.fill_between(
        GRID, low, high, color=paper.RANDOM_BAND, lw=0, zorder=1.5
    )
    band.set_gid("random_band")
    zero = ax.axhline(0.0, color=paper.INK, lw=0.5, zorder=2)
    zero.set_gid("zero")
    onsets, colours = [], []
    for model in sorted(cells, key=store.model_rank):
        cell = cells[model]
        colour = shared.model_style(model).color
        margin = cell.curve.selectivity
        (line,) = ax.plot(
            cell.curve.depth, margin, color=colour, lw=LINE, zorder=3
        )
        line.set_gid(f"curve:{model}")
        index = cell.curve.layer - 1
        (dot,) = ax.plot(
            [cell.curve.depth[index]],
            [margin[index]],
            ls="none",
            marker=paper.MEASURE_MARKERS["direction"],
            ms=DOT,
            color=colour,
            mec=paper.INK,
            mew=0.5,
            zorder=4,
            clip_on=False,
        )
        dot.set_gid(f"read:{model}")
        onset = reading.first_depth_above(cell.curve, MARGIN)
        if onset is not None:
            onsets.append(onset)
            colours.append(colour)
    floor = CURVE_YLIM[0]
    ticks = ax.vlines(
        onsets,
        floor,
        floor + TICK_HEIGHT * (CURVE_YLIM[1] - floor),
        colors=colours,
        lw=1.4,
        zorder=4,
    )
    ticks.set_gid("onset")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(*CURVE_YLIM)
    ax.set_xticks([0.0, 0.5, 1.0], ["0", "0.5", "1"])
    ax.set_yticks(CURVE_TICKS, ["0", "0.2", "0.4"])


def _label_column(ax: Axes, models: list[str]) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    count = len(models)
    for i, model in enumerate(sorted(models, key=store.model_rank)):
        y = 0.82 - i * (0.64 / max(count - 1, 1))
        ax.plot(
            [0.0, 0.28],
            [y, y],
            color=shared.model_style(model).color,
            lw=LINE + 0.4,
            transform=ax.transAxes,
            solid_capstyle="butt",
        )
        text = ax.text(
            0.34,
            y,
            _end_label(model),
            transform=ax.transAxes,
            fontsize=TEXT,
            va="center",
            ha="left",
            color=paper.INK,
        )
        text.set_gid("end_label")


def _numbers(found: dict[str, dict[str, Cell]]) -> dict[str, float]:
    numbers: dict[str, float] = {}
    for concept, _, _ in CONCEPTS:
        onsets: dict[str, list[float]] = {r: [] for r in stats.ROLES}
        for model, cell in found[concept].items():
            depth = reading.first_depth_above(cell.curve, MARGIN)
            if depth is not None:
                onsets[stats.role(model)].append(depth)
        for role, depths in onsets.items():
            if depths:
                numbers[f"onset_{concept}_{role}_min"] = min(depths)
                numbers[f"onset_{concept}_{role}_max"] = max(depths)
    numbers["truth_n"] = next(iter(found["truth"].values())).at.n
    return numbers


def _missing(found: dict[str, dict[str, Cell]]) -> dict[str, stats.RoleCount]:
    """Counts the checkpoints with no onset for some concept, per role."""
    lacking: dict[str, bool] = {}
    for cells in found.values():
        for model, cell in cells.items():
            none = reading.first_depth_above(cell.curve, MARGIN) is None
            lacking[model] = lacking.get(model, False) or none
    return stats.per_role(lacking)


def _missing_clause(counts: dict[str, stats.RoleCount]) -> str:
    """Words the cells without an onset, or nothing when there are none.

    Args:
        counts: Role to its count of checkpoints that lack an onset.

    Returns:
        A closing clause with the per-role counts, or an empty string.
    """
    lacking = {r: c for r, c in counts.items() if c.hits}
    if not lacking:
        return ""
    return (
        " For some concept the margin never reaches 0.1 on "
        + stats.per_role_text(lacking)
        + "."
    )


def _claim(
    numbers: dict[str, float],
    names: dict[str, str],
    missing: dict[str, stats.RoleCount],
) -> str:
    parts = []
    for concept in names:
        spans = [
            f"{numbers[f'onset_{concept}_{r}_min']:.3f} to "
            f"{numbers[f'onset_{concept}_{r}_max']:.3f} "
            f"on {stats.ROLE_NAMES[r]}"
            for r in stats.ROLES
            if f"onset_{concept}_{r}_min" in numbers
        ]
        parts.append(f"{concept} {stats.join_and(spans)}")
    return (
        "A reader's margin over a random direction reaches 0.1 at a relative "
        "depth of " + "; ".join(parts) + "." + _missing_clause(missing)
    )


def _csv(found: dict[str, dict[str, Cell]]) -> str:
    lines = ["panel,concept,model,x,y,lo,hi,comparison"]
    for concept, _, _ in CONCEPTS:
        for model, cell in found[concept].items():
            for depth, value in zip(
                cell.curve.depth, cell.curve.selectivity, strict=True
            ):
                lines.append(
                    f"depth,{concept},{model},{depth:.10g},{value:.10g},,,"
                )
            index = cell.curve.layer - 1
            lines.append(
                f"read,{concept},{model},{cell.curve.depth[index]:.10g},"
                f"{cell.curve.selectivity[index]:.10g},,,"
            )
    return "\n".join(lines) + "\n"


def _detail(numbers: dict[str, float]) -> str:
    n = int(numbers["truth_n"])
    return (
        "Reader accuracy minus random-direction accuracy at every layer, one "
        "line per checkpoint in the panel of its role (rows: GPT-2, Qwen3 "
        "base, Qwen3 post-trained; darker lines are larger checkpoints, "
        "named in the right column of each row). The dot marks the layer "
        "read, chosen on a validation split (for sentiment the layer of the "
        "finder comparison); the tick on the x axis marks the onset, the "
        "earliest layer at which the margin curve crosses 0.1, which is not "
        f"an estimate with an interval; for truth the curve rests on {n} "
        "truth pairs. The grey band is the stored 95 percent interval over "
        "seeds of the random direction, drawn around the zero line and "
        "spanning the checkpoints of the panel. Sentiment curves are the "
        "report-split accuracy of the logistic regression of the finder "
        "comparison. The accuracy at the layer read against the checkpoint "
        "size is in [[fig:reading]]."
    )


def _key(fig: Figure) -> None:
    ink = paper.INK
    handles = [
        Line2D(
            [],
            [],
            ls="none",
            marker=paper.MEASURE_MARKERS["direction"],
            ms=DOT,
            mfc=ink,
            mec=ink,
        ),
        Line2D([], [], ls="none", marker="|", ms=6, mew=1.4, color=ink),
        Line2D([], [], color=paper.RANDOM_BAND, lw=5, solid_capstyle="butt"),
    ]
    fig.legend(
        handles,
        [
            "layer read",
            "onset (earliest layer above 0.1)",
            "random direction, 95 percent interval",
        ],
        loc="outside upper center",
        ncol=2,
        frameon=False,
        fontsize=TEXT,
        columnspacing=1.0,
        handletextpad=0.4,
    )


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    found = {c: cells_of(manifest, c, exp) for c, _, exp in CONCEPTS}
    names = {c: label for c, label, _ in CONCEPTS}
    numbers = _numbers(found)
    with paper.paper_style():
        fig = Figure(figsize=(paper.FULL_WIDTH, HEIGHT), layout="constrained")
        grid = fig.add_gridspec(3, 5, width_ratios=WIDTH_RATIOS)
        for r, (role, role_name) in enumerate(ROLE_ROWS):
            members: list[str] = []
            for col, (concept, label, _) in enumerate(CONCEPTS):
                ax = fig.add_subplot(grid[r, col])
                cells = {
                    m: c
                    for m, c in found[concept].items()
                    if stats.role(m) == role
                }
                members = list(cells)
                ax.set_gid(f"curve_{role}_{concept}")
                _curve_panel(ax, cells)
                if r == 0:
                    ax.set_title(
                        label, fontsize=paper.TITLE, loc="left", x=0.14
                    )
                    paper.panel_letter(ax, "abcd"[col])
                if r < len(ROLE_ROWS) - 1:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel("relative depth")
                if col == 0:
                    ax.set_ylabel(role_name, fontsize=ROLE_LABEL)
                else:
                    ax.tick_params(labelleft=False)
            label_ax = fig.add_subplot(grid[r, 4])
            label_ax.set_gid(f"labels_{role}")
            _label_column(label_ax, members)
        shared_label = fig.supylabel(
            "reader minus random accuracy", y=CURVE_LABEL_Y, fontsize=ROLE_LABEL
        )
        shared_label.set_gid("curve_ylabel")
        _key(fig)
    return FigureOutput(
        fig,
        _claim(numbers, names, _missing(found)),
        _detail(numbers),
        numbers,
        _csv(found),
    )
