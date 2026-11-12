"""Appendix figure: The knowledge readers on their questions, beside controls.

One column per role shows, for every contrast of the knowledge readers, how
the test questions of its two sides spread along the all-pairs reader at the
layer of the reader. The last panel gives the AUROC behind each cell beside
the two controls of the reader: its random directions and the same reader
refitted on shuffled labels.
"""

from collections.abc import Mapping
import dataclasses
import math
from typing import Any

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
import numpy as np
from numpy.typing import NDArray

from tomography.site import runfiles
from tomography.site import stats
from tomography.site import views
from tomography.site.figs import _group
from tomography.site.figs import _rows
from tomography.site.figs import a_knowledge_geometry
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import axes
from tomography.viz import paper

SHOWN = a_knowledge_geometry.SHOWN
BLOCKS = {"entity": "entity", "question": "question", "unknown": "unknown type"}
LIMIT = 3.0
BINS = 18
EDGES = np.linspace(-LIMIT, LIMIT, BINS + 1)
HALF = 0.42
HEIGHT = 6.2
BLOCK_GAP = 0.9
BLOCK_LABEL = 0.64
LANE = 0.25
BAND = 0.17
FOREST_WIDTH = 1.2
FOREST_START = 0.4
FOREST_END = 1.02
FOREST_STEP = 0.02
SEPARATOR = 0.6
READER_SIZE = 3.4
HALO = 1.2
RANDOM_SIZE = 3.0
SHUFFLED_SIZE = 4.6
THIN_EDGE = 0.7
TICK_EDGE = 1.0
LEGEND_BAND = 4.0
HEADER = ["checkpoint", "panel", "contrast", "series", "low", "high", "value"]
DETAIL = (
    "Panels (a) to (c) are one checkpoint per role and each row is a drawn "
    "contrast of the knowledge readers, in three blocks: the entity "
    "contrasts, the question contrasts and the contrasts between two kinds "
    "of unknown question. The views run, which reads the questions of the "
    "knowledge reading run again to recompute its readers and draw their "
    "hidden states, draws two entity contrasts only; "
    "the other entity contrasts, {undrawn}, are scored and not drawn, and "
    "[[table:knowledge_projections]] gives their readers, the lowest "
    "of the three checkpoints at {hidden}. In the row names, synthal is "
    "SyntHal [[cite:lu2026relational]], a published set of questions about "
    "fictional entities, generated is the set of fictional names generated "
    "for this study, and kuq is one category of KUQ "
    "[[cite:amayuelas2024knowledge]], a published set of questions that "
    "have no known answer. A cell is the distribution of the test questions "
    "of the contrast along its all-pairs reader, the reader the reading run "
    "fitted on every training pair, at the layer of the reader: "
    "the projection of each question on the reader as a z-score over the "
    "questions of the cell, in {bins} bins that end {limit} standard "
    "deviations from the mean on either side, with the values beyond in the "
    "end bins. In the entity and question blocks the upper side is the "
    "known questions and the lower side the unknown questions the row "
    "names; in the unknown type block both sides are unknown questions and "
    "the upper side is the group named first. Each side is scaled to its "
    "own share and holds at most {cap} questions, the first by a hash of "
    "their identifier. Panel (d) gives the AUROC behind each cell, the "
    "three checkpoints from top to bottom in a row: the circle is the "
    "all-pairs reader, brown for GPT-2 and blue for Qwen3, and the other "
    "marks are its two controls. The grey band spans the random directions "
    "of the {draws} draws, with a cross at their mean; a random direction "
    "is oriented on the training labels like the reader, so it can score "
    "far above 0.5 on a small contrast. The tick is the mean of the {draws} "
    "shuffled-label refits, each the reader refitted with its labels "
    "shuffled inside each stratum; single refits scatter over most of the "
    "axis and their lowest and highest are in the data file. The axis "
    "starts at {left}, and a band that reaches its left edge continues "
    "below it. A reader inside its grey band does not read the contrast "
    "better than a direction drawn at random. {exceptions} The dotted line "
    "is chance, and an empty row is a contrast the checkpoint has no reader "
    "for. The data file holds the bins and the values of {with_run}."
)


@dataclasses.dataclass(frozen=True)
class Cell:
    """One contrast of one checkpoint.

    Attributes:
        panel: The block of the contrast, a key of `BLOCKS`.
        first: Share of the first side of the contrast in each bin.
        second: Share of its second side in each bin.
        values: The AUROC of the reader and of its controls.
    """

    panel: str
    first: NDArray[np.float64]
    second: NDArray[np.float64]
    values: views.Contrast


def _shares(z: NDArray[np.float64]) -> NDArray[np.float64]:
    found, _ = np.histogram(z, bins=EDGES)
    return np.asarray(found / found.sum(), dtype=np.float64)


def _cells(found: views.Views) -> dict[str, Cell]:
    """Bins the displayed questions of every contrast a checkpoint draws.

    Args:
        found: The views of the checkpoint.

    Returns:
        Contrast to its cell, block by block in the order of `BLOCKS`.
    """
    sidecar = runfiles.data(found.run, "projections.json")
    stored = runfiles.arrays(found.run, "projections.npz")
    values = {contrast.name: contrast for contrast in found.contrasts}
    cells: dict[str, Cell] = {}
    for panel in BLOCKS:
        for name in sidecar["panels"].get(panel, {"types": []})["types"]:
            scores = np.asarray(stored[f"projection_{name}"], dtype=np.float64)
            split = len(sidecar["uids"][name]["positive"])
            z = np.clip((scores - scores.mean()) / scores.std(), -LIMIT, LIMIT)
            cells[name] = Cell(
                panel, _shares(z[:split]), _shares(z[split:]), values[name]
            )
    return cells


def _positions(shown: Mapping[str, Mapping[str, Cell]]) -> dict[str, float]:
    """Places the contrasts of the shown checkpoints in rows.

    Args:
        shown: Checkpoint to its cells, see `_cells`.

    Returns:
        Contrast to row position, one unit apart inside a block and
        `BLOCK_GAP` further apart between two blocks.
    """
    positions: dict[str, float] = {}
    place = 0.0
    for panel in BLOCKS:
        names = dict.fromkeys(
            name
            for cells in shown.values()
            for name, cell in cells.items()
            if cell.panel == panel
        )
        for name in names:
            positions[name] = place
            place += 1.0
        place += BLOCK_GAP
    return positions


def _histograms(
    ax: Axes, cells: Mapping[str, Cell], positions: Mapping[str, float]
) -> None:
    """Draws the two sides of every contrast of a checkpoint back to back.

    Args:
        ax: The panel; rows run down the y axis.
        cells: The cells of the checkpoint, see `_cells`.
        positions: Contrast to row position.
    """
    for name, cell in cells.items():
        base = positions[name]
        scale = HALF / max(cell.first.max(), cell.second.max())
        upper = "unknown" if cell.panel == "unknown" else "known"
        ax.stairs(
            base - cell.first * scale,
            EDGES,
            baseline=base,
            fill=True,
            color=paper.ENC[upper],
            gid=f"first:{name}",
        )
        ax.stairs(
            base + cell.second * scale,
            EDGES,
            baseline=base,
            fill=True,
            color=paper.ENC["unknown"],
            gid=f"second:{name}",
        )
        ax.plot([-LIMIT, LIMIT], [base, base], color=paper.WHITE, lw=SEPARATOR)
    ax.set_xlim(-LIMIT - 0.2, LIMIT + 0.2)
    ax.set_xticks([-2.0, 0.0, 2.0])
    ax.xaxis.set_major_formatter(axes.minus_formatter())
    ax.set_xlabel(
        "projection on\nthe reader (z-score\nwithin the cell)",
        fontsize=paper.MIN_TEXT,
    )


def _left_edge(shown: Mapping[str, Mapping[str, Cell]]) -> float:
    """Gives where the axis of the last panel starts.

    Args:
        shown: Checkpoint to its cells.

    Returns:
        `FOREST_START`, or a lower multiple of `FOREST_STEP` when the mean of
        a control of a drawn cell lies below it, so every mean stays on the
        axis.
    """
    lowest = min(
        min(cell.values.random.mean, cell.values.shuffled.mean)
        for cells in shown.values()
        for cell in cells.values()
    )
    steps = math.floor((lowest - FOREST_STEP / 2) / FOREST_STEP)
    return min(FOREST_START, round(steps * FOREST_STEP, 2))


def _forest(
    ax: Axes,
    shown: Mapping[str, Mapping[str, Cell]],
    positions: Mapping[str, float],
    left: float,
) -> None:
    """Draws the AUROC of every reader beside its two controls.

    The reader sits above its controls on a white halo, so a circle next to
    the end of a band does not read as inside it.

    Args:
        ax: The panel; it shares the rows of the histograms.
        shown: Checkpoint to its cells, in the order of the lanes.
        positions: Contrast to row position.
        left: Where the axis starts, see `_left_edge`.
    """
    axes.ref_line(ax, 0.5, "x")
    for place, (model, cells) in enumerate(shown.items()):
        offset = (place - (len(shown) - 1) / 2) * LANE
        color = axes.model_style(model).color
        for name, cell in cells.items():
            lane = positions[name] + offset
            key = f"{model}:{name}"
            found = cell.values
            ax.add_patch(
                Rectangle(
                    (found.random.low, lane - BAND / 2),
                    found.random.high - found.random.low,
                    BAND,
                    fc=paper.RANDOM_BAND,
                    ec="none",
                    lw=0,
                    gid=f"random:{key}",
                    zorder=1.5,
                )
            )
            ax.plot(
                [found.random.mean],
                [lane],
                gid=f"random mean:{key}",
                zorder=2,
                **_random_mark(),
            )
            ax.plot(
                [found.shuffled.mean],
                [lane],
                gid=f"shuffled:{key}",
                zorder=2.5,
                **_shuffled_mark(),
            )
            ax.plot(
                [found.published],
                [lane],
                gid=f"halo:{key}",
                zorder=2.9,
                **_halo_mark(),
            )
            ax.plot(
                [found.published],
                [lane],
                gid=f"reader:{key}",
                zorder=3,
                **_reader_mark(model, color),
            )
    ax.set_xlim(left, FOREST_END)
    axes.unit_axis(ax, "x")
    ax.set_xlabel("AUROC", fontsize=paper.MIN_TEXT)


def _reader_mark(model: str, color: str) -> dict[str, Any]:
    return {
        "ls": "none",
        "marker": paper.MEASURE_MARKERS["direction"],
        "color": color,
        **_rows.fill(model, color),
        "markersize": READER_SIZE,
        "markeredgewidth": THIN_EDGE,
    }


def _halo_mark() -> dict[str, Any]:
    return {
        "ls": "none",
        "marker": paper.MEASURE_MARKERS["direction"],
        "color": paper.WHITE,
        "markerfacecolor": paper.WHITE,
        "markeredgecolor": paper.WHITE,
        "markeredgewidth": 0.0,
        "markersize": READER_SIZE + 2 * HALO,
    }


def _random_mark() -> dict[str, Any]:
    return {
        "ls": "none",
        "marker": paper.MEASURE_MARKERS["random"],
        "color": paper.RANDOM_GREY,
        "markersize": RANDOM_SIZE,
        "markeredgewidth": THIN_EDGE,
    }


def _shuffled_mark() -> dict[str, Any]:
    return {
        "ls": "none",
        "marker": "|",
        "color": paper.ENC["shuffled"],
        "markersize": SHUFFLED_SIZE,
        "markeredgewidth": TICK_EDGE,
    }


def _legend_handles() -> list[Any]:
    """Builds the keys of the two classes, the reader and its controls."""
    readers = tuple(
        Line2D([], [], **_reader_mark(model, axes.model_style(model).color))
        for model in SHOWN
    )
    band = {
        **_random_mark(),
        "ls": "-",
        "lw": LEGEND_BAND,
        "solid_capstyle": "butt",
        "color": paper.RANDOM_BAND,
        "markeredgecolor": paper.RANDOM_GREY,
    }
    return [
        Patch(fc=paper.ENC["known"], ec="none", lw=0),
        Patch(fc=paper.ENC["unknown"], ec="none", lw=0),
        readers,
        Line2D([], [], **band),
        Line2D([], [], **_shuffled_mark()),
    ]


def _csv_rows(model: str, cells: Mapping[str, Cell]) -> list[list[Any]]:
    """Lists the bins and the values of a checkpoint.

    Args:
        model: Checkpoint key.
        cells: The cells of the checkpoint, see `_cells`.

    Returns:
        For each contrast, one row per bin and side with its share, then the
        reader and each control with its mean and its lowest and highest
        draw.
    """
    found: list[list[Any]] = []
    for name, cell in cells.items():
        head = [model, cell.panel, name]
        for series, shares in (
            ("first side", cell.first),
            ("second side", cell.second),
        ):
            for low, high, share in zip(
                EDGES[:-1], EDGES[1:], shares, strict=True
            ):
                found.append(
                    [*head, series, f"{low:.6g}", f"{high:.6g}", f"{share:.6f}"]
                )
        values = cell.values
        found.append([*head, "reader", None, None, f"{values.published:.6f}"])
        for series, spread in (
            ("random", values.random),
            ("shuffled labels", values.shuffled),
        ):
            found.append(
                [
                    *head,
                    series,
                    f"{spread.low:.6f}",
                    f"{spread.high:.6f}",
                    f"{spread.mean:.6f}",
                ]
            )
    return found


def _above(cells: Mapping[str, Cell]) -> int:
    """Counts the readers above every one of their random directions."""
    return sum(
        cell.values.published > cell.values.random.high
        for cell in cells.values()
    )


def _label(name: str) -> str:
    return name.replace("_", " ")


def _exceptions(
    every: Mapping[str, Mapping[str, Cell]], misses: Mapping[str, list[str]]
) -> str:
    """Says where a random direction is above the reader.

    Args:
        every: Checkpoint to its cells, for every checkpoint with a run.
        misses: Checkpoint to its drawn contrasts whose reader is not above
            every random direction; no checkpoint without one.

    Returns:
        One sentence naming each such checkpoint and contrast with the
        highest random direction and the reader.
    """
    scope = "Over every checkpoint in the data file, "
    if not misses:
        return f"{scope}no reader is inside its band."
    parts = []
    for model, names in misses.items():
        listed = stats.join_and(
            [
                f"{_label(name)} "
                f"({every[model][name].values.random.high:.3f} against "
                f"{every[model][name].values.published:.3f})"
                for name in names
            ]
        )
        drawn = "" if model in SHOWN else ", which is not drawn"
        parts.append(f"{paper.display_name(model)}{drawn}, on {listed}")
    return (
        f"{scope}the highest random direction is above the reader only on "
        f"{'; and on '.join(parts)}."
    )


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    every: dict[str, dict[str, Cell]] = {}
    caps = set()
    counted = set()
    undrawn: dict[str, None] = {}
    hidden: list[float] = []
    csv_rows: list[list[Any]] = []
    for model in manifest.models(views.EXPERIMENT):
        found = views.load(manifest, model)
        if found is None:
            continue
        every[model] = _cells(found)
        caps.add(runfiles.data(found.run, "projections.json")["max_per_side"])
        counted.add(found.draws)
        left_out = [c for c in found.contrasts if c.name not in every[model]]
        undrawn.update(dict.fromkeys(c.name for c in left_out))
        if model in SHOWN:
            hidden += [c.published for c in left_out]
        csv_rows += _csv_rows(model, every[model])
    (cap,) = caps
    (draws,) = counted
    shown = {model: every[model] for model in SHOWN}
    positions = _positions(shown)
    left = _left_edge(shown)
    misses = {
        model: [
            name
            for name, cell in cells.items()
            if cell.values.published <= cell.values.random.high
        ]
        for model, cells in every.items()
    }
    misses = {model: names for model, names in misses.items() if names}

    with paper.paper_style():
        fig = _rows.new_figure(HEIGHT)
        grid = fig.subplots(
            1,
            len(SHOWN) + 1,
            sharey=True,
            width_ratios=[1.0] * len(SHOWN) + [FOREST_WIDTH],
        )
        for ax, letter, model in zip(grid[:-1], "abc", SHOWN, strict=True):
            _histograms(ax, shown[model], positions)
            paper.panel_title(
                ax, letter, paper.display_name(model), size=paper.MIN_TEXT
            )
        _forest(grid[-1], shown, positions, left)
        paper.panel_title(
            grid[-1], "d", "reader and controls", size=paper.MIN_TEXT
        )
        first = grid[0]
        first.set_ylim(max(positions.values()) + 0.7, -1.0)
        first.set_yticks(list(positions.values()))
        first.set_yticklabels([_label(name) for name in positions])
        for ax in grid:
            ax.tick_params(axis="y", length=0)
            ax.spines["left"].set_visible(False)
        pad = mpl.rcParams["ytick.major.pad"]
        for panel, label in BLOCKS.items():
            top = min(
                positions[name]
                for cells in shown.values()
                for name, cell in cells.items()
                if cell.panel == panel
            )
            first.annotate(
                label,
                xy=(0.0, top - BLOCK_LABEL),
                xycoords=first.get_yaxis_transform(),
                xytext=(-pad, 0.0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=paper.MIN_TEXT,
                color=paper.ENC["secondary_text"],
                gid="block",
            )
        legend = fig.legend(
            handles=_legend_handles(),
            labels=[
                "known questions",
                "unknown questions",
                "all-pairs reader (brown: GPT-2; blue: Qwen3; open: "
                "post-trained)",
                "random directions: range and mean",
                "shuffled-label refits: mean",
            ],
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)},
            loc="outside lower center",
            ncol=2,
            handlelength=2.4,
            columnspacing=1.6,
            fontsize=paper.MIN_TEXT,
        )
        legend.get_frame().set_edgecolor("none")
        _rows.freeze_layout(fig)

    readers = [
        c.values.published for cells in shown.values() for c in cells.values()
    ]
    randoms = [
        c.values.random.high for cells in shown.values() for c in cells.values()
    ]
    numbers = {
        "reader_min": min(readers),
        "random_max": max(randoms),
        "bins": float(BINS),
        "limit": LIMIT,
        "per_side": float(cap),
        "draws": float(draws),
        "forest_left": left,
        "undrawn_reader_min": min(hidden),
    }
    parts = []
    for model, cells in shown.items():
        role = stats.role(model)
        above = _above(cells)
        numbers[f"above_{role}"] = float(above)
        numbers[f"contrasts_{role}"] = float(len(cells))
        parts.append(
            f"{above} of {len(cells)} drawn contrasts on "
            f"{paper.display_name(model)}"
        )
    counts = {
        "with_run": stats.per_role(
            {model: model in every for model in manifest.checkpoints()}
        ),
        "above_every_draw": stats.per_role(
            {
                model: _above(cells) == len(cells)
                for model, cells in every.items()
            }
        ),
    }
    labels = {
        "undrawn": stats.join_and([_label(name) for name in undrawn]),
        "exceptions": stats.join_and(
            [paper.display_name(model) for model in misses]
        ),
        "exception_contrasts": stats.join_and(
            [
                _label(name)
                for name in dict.fromkeys(
                    name for names in misses.values() for name in names
                )
            ]
        ),
    }
    claim = (
        f"The all-pairs reader has a higher AUROC than each of its {draws} "
        f"random directions on {stats.join_and(parts)}; the lowest drawn "
        f"reader is at {min(readers):.2f} and the highest single random "
        f"direction at {max(randoms):.2f}."
    )
    return FigureOutput(
        fig,
        claim,
        DETAIL.format(
            bins=BINS,
            limit=f"{LIMIT:g}",
            cap=cap,
            draws=draws,
            left=f"{left:.2f}",
            undrawn=labels["undrawn"],
            hidden=f"{min(hidden):.3f}",
            exceptions=_exceptions(every, misses),
            with_run=stats.per_role_text(counts["with_run"]),
        ),
        numbers,
        _group.csv_text(HEADER, csv_rows),
        counts,
        labels,
    )
