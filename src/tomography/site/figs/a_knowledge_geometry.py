"""Appendix figure: The hidden states behind the knowledge readers.

One row per role shows the test questions of the two entity contrasts on two
components: of the hidden states at layer 1, at the layer of the reader and
at the last layer, and of the words of the questions alone. Layer 1 and the
words are the controls of the other two views. The number in a panel is the
folded AUROC of known against unknown questions along the better of its two
axes. Known questions, unknown questions about real entities and questions
about fictional entities each take their own marker.
"""

from collections.abc import Sequence
import dataclasses
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
from matplotlib.legend_handler import HandlerTuple
import numpy as np
from numpy.typing import NDArray

from tomography.reading.metrics import auroc
from tomography.site import runfiles
from tomography.site import stats
from tomography.site.figs import _group
from tomography.site.figs import _rows
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import paper

EXPERIMENT = "v1"
PANEL = "entity"
SHOWN = ("gpt2-xl", "qwen3-1.7b-base", "qwen3-1.7b")
VIEWS = ("layer 1", "reader layer", "last layer", "words only")
WORDS = VIEWS[-1]
FICTIONAL = "fictional_generated"
MARKS = ("fictional", "unknown", "known")
MAX_DRAWN = 40
HEIGHT = 4.7
HEADER = [
    "checkpoint",
    "view",
    "layer",
    "series",
    "type",
    "x",
    "y",
    "drawn",
    "auroc",
]
SIZES = dict(zip(("unknown", "known"), paper.DENSE_MARKER_RANGE, strict=True))
SIZES["fictional"] = SIZES["known"] * _rows.STAR_SIZE / _rows.MARKER_SIZE
LABELS = {
    "known": "known (filled: GPT-2 and base; open: post-trained)",
    "unknown": "unknown, real entity",
    "fictional": "unknown, fictional entity",
}
FILLED_EDGE = 0.3
OPEN_EDGE = 0.6
HEADROOM = 0.22
DETAIL = (
    "Each row is one checkpoint, one per role: {names}. Columns (a) to (c) "
    "place test questions of the two entity contrasts on the two leading "
    "principal components of the hidden states at the last token of the "
    "few-shot prompt: layer 1, an early control; the layer the knowledge "
    "reader was chosen at; and the last layer. Layer 0 is not drawn, "
    "because every prompt ends in the same token. Column (d) is a second "
    "control with no model in it: the same questions placed by the two "
    "leading components of the TF-IDF weights of their words (latent "
    "semantic analysis). The stored questions of a row are test questions "
    "only, at most {side} per side of each contrast, and the components of "
    "a panel are fitted on them, so no training question enters a view. A "
    "known question (triangle) is a PopQA question labelled known for the "
    "checkpoint; an unknown question is a PopQA question about a real "
    "entity labelled unknown (circle) or the question of a generated "
    "fictional entity (star). Each panel draws at most {cap} questions of "
    "each of these three groups, the first by a hash of their identifier, "
    "and its number is the AUROC of known against unknown along whichever "
    "of its two axes separates them better, computed on every stored "
    "question ({low} to {high} per checkpoint) and folded, so it is 0.5 or "
    "more whichever class lies higher on the axis. Axis scales differ "
    "between panels and carry no unit. A separation that the words alone "
    "also show does not need the hidden states. The data file holds every "
    "stored question of these views for {with_views}."
)


@dataclasses.dataclass(frozen=True)
class Group:
    """The stored questions of one side of the panel.

    Attributes:
        series: `known` or `unknown`.
        name: `known`, or the contrast whose unknown side this is.
        rows: Row of each question in the coordinates of the panel, in the
            stored order, which is by ascending hash of the identifier.
    """

    series: str
    name: str
    rows: tuple[int, ...]

    @property
    def mark(self) -> str:
        """The entry of `MARKS` the questions are drawn with."""
        return "fictional" if self.name == FICTIONAL else self.series


@dataclasses.dataclass(frozen=True)
class View:
    """One 2D view of the questions of a panel.

    Attributes:
        name: An entry of `VIEWS`.
        layer: The layer of the hidden states, None for the words.
        points: Coordinates of shape (n, 2).
        separation: AUROC of known against unknown along the better axis.
    """

    name: str
    layer: int | None
    points: NDArray[Any]
    separation: float


def _groups(run: Path) -> list[Group]:
    """Reads which stored questions are known and which unknown.

    Args:
        run: A run directory of the experiment.

    Returns:
        The known questions, then the unknown side of each contrast.
    """
    sidecar = runfiles.data(run, "projections.json")
    panel = sidecar["panels"][PANEL]
    row = {uid: i for i, uid in enumerate(panel["uids"])}
    known: dict[str, None] = {}
    for name in panel["types"]:
        known.update(dict.fromkeys(sidecar["uids"][name]["positive"]))
    found = [Group("known", "known", tuple(row[uid] for uid in known))]
    for name in panel["types"]:
        unknown = sidecar["uids"][name]["negative"]
        found.append(Group("unknown", name, tuple(row[uid] for uid in unknown)))
    return found


def _separation(points: NDArray[Any], groups: Sequence[Group]) -> float:
    """Scores how far one axis of a view separates the two classes.

    Args:
        points: Coordinates of shape (n, 2).
        groups: The known and unknown questions, see `_groups`.

    Returns:
        The AUROC of known against unknown along the better of the two axes,
        whichever class lies higher on it.
    """
    rows = [i for group in groups for i in group.rows]
    labels = np.array(
        [int(group.series == "unknown") for group in groups for _ in group.rows]
    )
    areas = auroc(np.asarray(points[rows], dtype=np.float64).T, labels)
    return float(np.maximum(areas, 1.0 - areas).max())


def _view_name(position: int, layer: int, chosen: int) -> str:
    """Names a view of the hidden states.

    Args:
        position: Place of the view among the stored ones, in layer order.
        layer: Its layer.
        chosen: The layer of the reader.

    Returns:
        An entry of `VIEWS`. The layer of the reader is named so even when
        it is the first or the last one.
    """
    if layer == chosen:
        return VIEWS[1]
    return VIEWS[0] if position == 0 else VIEWS[2]


def _views(run: Path, groups: Sequence[Group]) -> list[View]:
    """Reads the stored views of a run.

    Args:
        run: A run directory of the experiment.
        groups: The known and unknown questions, see `_groups`.

    Returns:
        The views of the hidden states in layer order, then the words.
    """
    chosen = runfiles.summary(run)["layer"]
    pca = runfiles.arrays(run, "pca2d.npz")
    found = []
    for v, layer in enumerate(pca["view_layers"].tolist()):
        name = _view_name(v, layer, chosen)
        points = pca[f"coordinates_{PANEL}"][v]
        found.append(View(name, layer, points, _separation(points, groups)))
    words = runfiles.arrays(run, "text2d.npz")[f"coordinates_{PANEL}"]
    found.append(View(WORDS, None, words, _separation(words, groups)))
    return found


def _hue(mark: str) -> str:
    """Gives the hue of a group: one for known, one for both unknown kinds.

    Args:
        mark: An entry of `MARKS`.

    Returns:
        The colour.
    """
    return str(paper.ENC["known" if mark == "known" else "unknown"])


def _marker(model: str, mark: str) -> dict[str, Any]:
    """Gives the marker of a group on the row of a checkpoint.

    Args:
        model: Checkpoint key.
        mark: An entry of `MARKS`.

    Returns:
        Keyword arguments for `Axes.plot`: the hue of the class and the
        shape of the group, filled with a thin white edge for GPT-2 and base
        checkpoints and open for post-trained ones.
    """
    hue = _hue(mark)
    post = stats.role(model) == "post"
    return {
        "ls": "none",
        "marker": paper.QUESTION_SET_MARKERS[mark],
        "color": hue,
        "markerfacecolor": "white" if post else hue,
        "markeredgecolor": hue if post else "white",
        "markeredgewidth": OPEN_EDGE if post else FILLED_EDGE,
        "markersize": SIZES[mark],
    }


def _panel(ax: Axes, model: str, view: View, groups: Sequence[Group]) -> None:
    """Draws one view of one checkpoint.

    Args:
        ax: The axes.
        model: Checkpoint key.
        view: The view to draw.
        groups: The known and unknown questions, see `_groups`.
    """
    for mark in MARKS:
        rows = [i for g in groups if g.mark == mark for i in g.rows[:MAX_DRAWN]]
        ax.plot(
            view.points[rows, 0],
            view.points[rows, 1],
            gid=mark,
            **_marker(model, mark),
        )
    ax.margins(0.08)
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + HEADROOM * (high - low))
    ax.set_xticks([])
    ax.set_yticks([])
    if view.layer is not None:
        ax.text(
            0.04,
            0.97,
            f"layer {view.layer}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=paper.MIN_TEXT,
            gid="layer",
        )
    ax.text(
        0.96,
        0.97,
        f"AUROC {view.separation:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=paper.MIN_TEXT,
        gid="auroc",
    )


def _csv_rows(
    model: str, views: Sequence[View], groups: Sequence[Group]
) -> list[list[Any]]:
    """Lists the stored questions and the separation of every view.

    Args:
        model: Checkpoint key.
        views: The views of the checkpoint.
        groups: The known and unknown questions, see `_groups`.

    Returns:
        One row per question and view, then one row with the AUROC of the
        view.
    """
    found: list[list[Any]] = []
    for view in views:
        head = [model, view.name, view.layer]
        for group in groups:
            for place, row in enumerate(group.rows):
                x, y = (f"{float(v):.4g}" for v in view.points[row])
                drawn = int(place < MAX_DRAWN)
                found.append(
                    [*head, group.series, group.name, x, y, drawn, None]
                )
        area = f"{view.separation:.6f}"
        found.append([*head, "separation", None, None, None, None, area])
    return found


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    groups: dict[str, list[Group]] = {}
    views: dict[str, dict[str, View]] = {}
    csv_rows: list[list[Any]] = []
    per_side: set[int] = set()
    for model in manifest.models(EXPERIMENT):
        run = manifest.run(EXPERIMENT, model)
        if run is None:
            continue
        sidecar = runfiles.data(run, "projections.json")
        if PANEL not in sidecar["panels"]:
            continue
        per_side.add(int(sidecar["max_per_side"]))
        groups[model] = _groups(run)
        found = _views(run, groups[model])
        views[model] = {view.name: view for view in found}
        csv_rows += _csv_rows(model, found, groups[model])

    with paper.paper_style():
        fig = _rows.new_figure(HEIGHT)
        grid = fig.subplots(len(SHOWN), len(VIEWS))
        for r, model in enumerate(SHOWN):
            for c, name in enumerate(VIEWS):
                _panel(grid[r][c], model, views[model][name], groups[model])
            grid[r][0].set_ylabel(f"{paper.display_name(model)}\ncomponent 2")
        for c, name in enumerate(VIEWS):
            paper.panel_title(grid[0][c], "abcd"[c], name, size=paper.LABEL)
            grid[-1][c].set_xlabel("component 1")
        handles = [
            tuple(
                _rows.series_handle(
                    "", paper.QUESTION_SET_MARKERS[mark], _hue(mark), filled
                )
                for filled in (True, False)
            )
            for mark in LABELS
        ]
        legend = fig.legend(
            handles=handles,
            labels=list(LABELS.values()),
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
            loc="outside lower center",
            ncol=len(LABELS),
            handlelength=2.4,
            columnspacing=1.2,
            fontsize=paper.MIN_TEXT,
        )
        legend.get_frame().set_edgecolor("none")
        _rows.freeze_layout(fig)

    def span(name: str) -> tuple[float, float]:
        found = [views[model][name].separation for model in SHOWN]
        return min(found), max(found)

    early, reader, last, words = (span(name) for name in VIEWS)
    held = sum(
        views[model][VIEWS[2]].separation >= views[model][VIEWS[1]].separation
        for model in SHOWN
    )
    if len(per_side) != 1:
        raise ValueError(f"the runs keep {sorted(per_side)} questions per side")
    (side,) = per_side
    stored = [sum(len(g.rows) for g in groups[model]) for model in SHOWN]
    counts = {
        "with_views": stats.per_role(
            {model: model in views for model in manifest.checkpoints()}
        ),
        "reader_above_words": stats.per_role(
            {
                model: found[VIEWS[1]].separation > found[WORDS].separation
                for model, found in views.items()
            }
        ),
    }
    claim = (
        f"On {held} of the {len(SHOWN)} checkpoints shown, the better of the "
        "two leading principal components of the hidden states separates "
        "known from unknown entity questions at least as well at the last "
        "layer as at the layer of the reader, with an AUROC of "
        f"{last[0]:.2f} to {last[1]:.2f} against {reader[0]:.2f} to "
        f"{reader[1]:.2f}; layer 1 gives {early[0]:.2f} to {early[1]:.2f} "
        "and the better of two components of the words alone "
        f"{words[0]:.2f} to {words[1]:.2f}."
    )
    numbers = {
        "reader_min": reader[0],
        "reader_max": reader[1],
        "early_min": early[0],
        "early_max": early[1],
        "last_min": last[0],
        "last_max": last[1],
        "words_min": words[0],
        "words_max": words[1],
        "per_group": float(MAX_DRAWN),
        "per_side": float(side),
        "last_at_least_reader": float(held),
        "stored_min": float(min(stored)),
        "stored_max": float(max(stored)),
    }
    return FigureOutput(
        fig,
        claim,
        DETAIL.format(
            names=stats.join_and([paper.display_name(m) for m in SHOWN]),
            cap=MAX_DRAWN,
            side=side,
            low=min(stored),
            high=max(stored),
            with_views=stats.per_role_text(counts["with_views"]),
        ),
        numbers,
        _group.csv_text(HEADER, csv_rows),
        counts,
    )
