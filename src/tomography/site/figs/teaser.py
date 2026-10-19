"""Figure 1: the scorecard of what a direction reads and steers.

One dot plot with a row per test. Each marker is a checkpoint, placed at the
margin of a fitted direction over the baseline named in its row; lanes
separate GPT-2, Qwen3 base and Qwen3 post-trained checkpoints, and a column
on the right counts the checkpoints whose interval lies above zero.
"""

import dataclasses
from typing import Any

from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

from tomography.site import finders as loader
from tomography.site import params
from tomography.site import reading
from tomography.site import runfiles
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.site.registry import RoleCount
from tomography.viz import axes as shared
from tomography.viz import paper

HEIGHT = 3.7
XLIM = (-0.35, 0.55)
LANE = {"gpt2": -0.27, "base": 0.0, "post": 0.27}
BLOCK_GAP = 0.6
HEADER = 0.75
MARKER = 3.5
TEXT = 7


@dataclasses.dataclass(frozen=True)
class Cell:
    """One checkpoint on one row.

    Attributes:
        margin: Direction minus baseline.
        lo: Lower end of the 95 percent interval of the margin.
        hi: Upper end of the 95 percent interval of the margin.
        baseline: The baseline value the margin is measured from.
        n: Items the margin is read from, or None when the interval is
            stored.
    """

    margin: float
    lo: float
    hi: float
    baseline: float
    n: int | None


@dataclasses.dataclass(frozen=True)
class Row:
    """One test of the scorecard.

    Attributes:
        key: Name used in the numbers and counts.
        label: Row label.
        block: Index of the block the row belongs to.
        exp: Experiment whose checkpoints the row covers.
    """

    key: str
    label: str
    block: int
    exp: str


ROWS = (
    Row("utility", "Utility", 0, "r2"),
    Row("morality", "Morality", 0, "r3"),
    Row("truth", "Truth", 0, "r7"),
    Row("neg_cities", "Negated cities", 0, "r7"),
    Row("honesty_transfer", "Honesty, transfer", 0, "r5"),
    Row("paper_finder", "Paper finder, sentiment", 1, "p4"),
    Row("truthfulqa", "TruthfulQA, against words", 1, "r6"),
    Row("knowledge", "Knowledge, against popularity", 1, "p9b"),
    Row("knowledge_confidence", "Knowledge, added to confidence", 1, "p9c"),
    Row("steering", "Steering against random", 2, "m1"),
)
BLOCKS = (
    "against a random direction",
    "against a stronger baseline",
    "steering",
)


def _from_scores(
    accuracy: float, baseline: float, n: int, n_baseline: int | None = None
) -> Cell:
    margin, lo, hi = stats.diff_interval(accuracy, baseline, n, n_baseline)
    return Cell(margin, lo, hi, baseline, n)


def _read(manifest: Manifest, exp: str, model: str) -> Cell | None:
    found = reading.load(manifest, exp, model)
    if found is None:
        return None
    n = params.eval_pairs(manifest, exp, model)
    return _from_scores(found.accuracy, found.random_at_layer, n)


def _neg_cities(manifest: Manifest, model: str) -> Cell | None:
    path = manifest.run("r7", model)
    if path is None:
        return None
    summary = runfiles.summary(path)
    block = summary["sets"]["neg_cities"]
    series = runfiles.data(path, "curves_neg_cities.json")["series"]
    mean = runfiles.arrays(path, "curves_neg_cities.npz")["mean"]
    random = float(mean[series.index("random")][int(summary["val_layer"]) - 1])
    return _from_scores(
        float(block["accuracy_at_val_layer"]),
        random,
        int(block["n_eval_pairs"]),
    )


def _paper_finder(manifest: Manifest, model: str) -> Cell | None:
    cells = loader.load(manifest, model)
    if cells is None:
        return None
    rows = cells["sentiment"].rows
    return _from_scores(
        rows["pca_diff"].report,
        rows["random"].report,
        params.REPORT_PAIRS["sentiment"],
    )


def _truthfulqa(manifest: Manifest, model: str) -> Cell | None:
    path = manifest.run("r6", model)
    if path is None:
        return None
    summary = runfiles.summary(path)
    reader = float(summary["lat"]["truthfulqa"]["validation-layer"]["mc1"])
    words = float(summary["text_baselines"]["text_word"]["mc1"])
    return _from_scores(reader, words, params.eval_pairs(manifest, "r6", model))


def _knowledge(manifest: Manifest, model: str) -> Cell | None:
    path = manifest.run("p9b", model)
    if path is None:
        return None
    found = [
        r
        for r in runfiles.rows(path, "table.json")
        if r["type"] == "obscure_real"
    ]
    if not found:
        return None
    row = found[0]
    best = max(
        float(row["popularity_within"]),
        float(row["text_word_within"]),
        float(row["text_char_within"]),
    )
    n_pos, n_neg = int(row["n_positive_test"]), int(row["n_negative_test"])
    margin, lo, hi = stats.auc_diff_interval(
        float(row["logistic_within"]), best, n_pos, n_neg
    )
    return Cell(margin, lo, hi, best, n_pos + n_neg)


def _confidence(manifest: Manifest, model: str) -> Cell | None:
    path = manifest.run("p9c", model)
    if path is None:
        return None
    table = [
        r
        for r in runfiles.rows(path, "table_incremental.json")
        if r["dataset"] == "popqa"
    ]
    added = [r for r in table if r["features"] == "confidence+lat"]
    alone = [r for r in table if r["features"] == "confidence"]
    if not added or not alone:
        return None
    return Cell(
        float(added[0]["diff_vs_confidence"]),
        float(added[0]["lo"]),
        float(added[0]["hi"]),
        float(alone[0]["auroc"]),
        None,
    )


def _steering(manifest: Manifest, model: str) -> Cell | None:
    path = manifest.run("m1", model)
    if path is None:
        return None
    table = [
        r
        for r in runfiles.rows(path, "table.json")
        if r["task"] == "manipulation" and r["setting"] == "0.3"
    ]
    (logistic,) = [r for r in table if r["method"] == "logistic"]
    (random,) = [r for r in table if r["method"] == "random"]
    return _from_scores(
        float(logistic["value"]),
        float(random["value"]),
        int(logistic["n"]),
        int(random["n"]),
    )


def _cell(manifest: Manifest, row: Row, model: str) -> Cell | None:
    if row.key in ("utility", "morality", "truth", "honesty_transfer"):
        return _read(manifest, row.exp, model)
    getter: Any = {
        "neg_cities": _neg_cities,
        "paper_finder": _paper_finder,
        "truthfulqa": _truthfulqa,
        "knowledge": _knowledge,
        "knowledge_confidence": _confidence,
        "steering": _steering,
    }[row.key]
    cell: Cell | None = getter(manifest, model)
    return cell


def _collect(manifest: Manifest) -> dict[str, dict[str, Cell]]:
    found: dict[str, dict[str, Cell]] = {}
    for row in ROWS:
        cells = {
            m: c
            for m in manifest.models(row.exp)
            if (c := _cell(manifest, row, m)) is not None
        }
        if not cells:
            raise FileNotFoundError(f"no results for the {row.key} row")
        found[row.key] = cells
    return found


def _counts(
    cells: dict[str, Cell],
) -> dict[str, RoleCount]:
    return stats.per_role({m: c.lo > 0 for m, c in cells.items()})


def _positions() -> dict[str, float]:
    place: dict[str, float] = {}
    y = 0.0
    block = -1
    for row in ROWS:
        if row.block != block:
            block = row.block
            y += HEADER if block == 0 else BLOCK_GAP + HEADER
        place[row.key] = y
        y += 1.0
    return place


def _markers(ax: Any, row_y: float, model: str, cell: Cell) -> None:
    style = shared.model_style(model)
    y = row_y + LANE[stats.role(model)]
    ax.plot(
        [cell.lo, cell.hi],
        [y, y],
        color=style.color,
        lw=paper.WHISKER_WIDTH,
        solid_capstyle="butt",
        zorder=3,
    )
    ax.plot(
        [cell.margin],
        [y],
        ls="none",
        marker="o",
        ms=MARKER,
        mfc=style.color if style.filled else "white",
        mec=style.color,
        mew=0.8,
        zorder=4,
    )


def _legend(fig: Figure) -> None:
    def dots(ramp: tuple[str, ...]) -> tuple[Line2D, ...]:
        return tuple(
            Line2D([], [], ls="none", marker="o", ms=MARKER, color=c)
            for c in ramp
        )

    black = paper.ENC["direction"]
    handles: list[Any] = [
        dots(paper.ENC["gpt2"]),
        dots(paper.ENC["qwen3"]),
        Line2D([], [], ls="none", marker="o", ms=MARKER, color=black),
        Line2D(
            [], [], ls="none", marker="o", ms=MARKER, mfc="white", mec=black
        ),
    ]
    fig.legend(
        handles,
        [
            "GPT-2 small to XL",
            "Qwen3 0.6B to 4B",
            "base or GPT-2",
            "post-trained",
        ],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.1)},
        loc="outside upper center",
        ncol=4,
        frameon=False,
        fontsize=TEXT,
        columnspacing=1.2,
        handletextpad=0.4,
    )


def _draw(fig: Figure, found: dict[str, dict[str, Cell]]) -> None:
    ax = fig.subplots()
    place = _positions()
    for row in ROWS:
        for model, cell in found[row.key].items():
            _markers(ax, place[row.key], model, cell)
        counts = _counts(found[row.key])
        text = " ".join(
            f"{counts[r].hits}/{counts[r].total}" for r in stats.ROLES
        )
        ax.text(
            1.02,
            place[row.key],
            text,
            transform=ax.get_yaxis_transform(),
            fontsize=TEXT,
            va="center",
            ha="left",
            color=paper.INK,
            clip_on=False,
        )
    ax.text(
        1.02,
        -0.2,
        "interval above 0",
        transform=ax.get_yaxis_transform(),
        fontsize=TEXT,
        va="center",
        ha="left",
        color=paper.ENC["secondary_text"],
        clip_on=False,
    )
    ax.text(
        1.02,
        0.3,
        "GPT-2 / base / post",
        transform=ax.get_yaxis_transform(),
        fontsize=TEXT,
        va="center",
        ha="left",
        color=paper.ENC["secondary_text"],
        clip_on=False,
    )
    firsts: dict[int, float] = {}
    for row in ROWS:
        firsts.setdefault(row.block, place[row.key])
    for block, y in firsts.items():
        ax.text(
            0.0,
            y - 0.75,
            BLOCKS[block],
            transform=ax.get_yaxis_transform(),
            fontsize=TEXT,
            style="italic",
            va="center",
            ha="left",
            color=paper.ENC["secondary_text"],
        )
    ax.set_yticks([place[r.key] for r in ROWS], [r.label for r in ROWS])
    ax.set_ylim(max(place.values()) + 0.7, -1.3)
    ax.set_xlim(*XLIM)
    ax.set_xticks([-0.2, 0.0, 0.2, 0.4])
    ax.xaxis.set_major_formatter(shared.minus_formatter())
    ax.set_xlabel("margin of the direction over its baseline")
    shared.ref_line(ax, 0.0, "x")
    ax.tick_params(axis="y", length=0, labelsize=TEXT)
    ax.spines[["top", "right", "left"]].set_visible(False)
    _legend(fig)


def _numbers(
    found: dict[str, dict[str, Cell]],
) -> tuple[dict[str, float], dict[str, dict[str, RoleCount]]]:
    numbers: dict[str, float] = {}
    counts: dict[str, dict[str, RoleCount]] = {}
    for row in ROWS:
        cells = found[row.key]
        counts[row.key] = _counts(cells)
        for role in stats.ROLES:
            members = [c for m, c in cells.items() if stats.role(m) == role]
            numbers[f"{row.key}_{role}_above"] = counts[row.key][role].hits
            numbers[f"{row.key}_{role}_n"] = counts[row.key][role].total
            numbers[f"{row.key}_{role}_below"] = sum(c.hi < 0 for c in members)
        margins = [c.margin for c in cells.values()]
        numbers[f"{row.key}_min"] = min(margins)
        numbers[f"{row.key}_max"] = max(margins)
    numbers["knowledge_confidence_n"] = len(found["knowledge_confidence"])
    return numbers, counts


def _reads(counts: dict[str, RoleCount], concept: str) -> str:
    total = sum(c.total for c in counts.values())
    if all(c.hits == c.total for c in counts.values()):
        return f"{concept} on all {total} checkpoints"
    return f"{concept} on {stats.per_role_text(counts)}"


def _claim(counts: dict[str, dict[str, RoleCount]]) -> str:
    utility, morality = counts["utility"], counts["morality"]
    every = all(
        c.hits == c.total for r in (utility, morality) for c in r.values()
    )
    if every:
        total = sum(c.total for c in utility.values())
        reads = (
            "A direction reads utility and morality better than a random "
            f"direction on all {total} checkpoints"
        )
    else:
        reads = (
            f"A direction reads {_reads(utility, 'utility')} and "
            f"{_reads(morality, 'morality')}"
        )
    reads += f", and {_reads(counts['truth'], 'truth')}."
    second = (
        "Against a stronger baseline: the paper's own finder "
        "beats a random direction for sentiment on "
        f"{stats.per_role_text(counts['paper_finder'])}, "
    )
    if all(c.hits == 0 for c in counts["truthfulqa"].values()):
        second += (
            "the TruthfulQA reader is never above a classifier that sees "
            "only the words with an interval excluding zero, "
        )
    else:
        second += (
            "the TruthfulQA reader is above a classifier that sees only the "
            "words with an interval excluding zero on "
            f"{stats.per_role_text(counts['truthfulqa'])}, "
        )
    second += (
        "and a direction for questions the model can answer is above the "
        "model's own confidence, with an interval excluding zero, on "
        f"{stats.per_role_text(counts['knowledge_confidence'])}."
    )
    third = (
        "Steering along a logistic direction beats random directions of the "
        f"same norm on {stats.per_role_text(counts['steering'])}."
    )
    return (
        "Where a direction reads and steers a concept, and where it does "
        f"not. {reads} {second} {third}"
    )


DETAIL = (
    "Each marker is one checkpoint, placed at the margin of a fitted "
    "direction over the baseline named in its row; lanes separate GPT-2, "
    "Qwen3 base and Qwen3 post-trained checkpoints, and the right column "
    "counts the checkpoints whose 95 percent interval lies above zero. "
    "Baselines: rows 1 to 5, the random direction of the same run at the "
    "same layer; row 6, the random direction of the finder comparison; "
    "row 7, a word classifier; row 8, the strongest of popularity and two "
    "text classifiers; row 9, the model's answer probability alone; row 10, "
    "three random directions (win rate at relative coefficient 0.3). "
    "Intervals cover the sampling of the test items and treat the two arms "
    "as independent, which makes them conservative; rows 8 and 9 use the "
    "Hanley and McNeil approximation and the stored paired bootstrap."
)


def _csv(found: dict[str, dict[str, Cell]]) -> str:
    lines = ["row,model,role,value,lo,hi,baseline,n"]
    for row in ROWS:
        for model, c in found[row.key].items():
            n = "" if c.n is None else str(c.n)
            lines.append(
                f"{row.key},{model},{stats.role(model)},{c.margin:.10g},"
                f"{c.lo:.10g},{c.hi:.10g},{c.baseline:.10g},{n}"
            )
    return "\n".join(lines) + "\n"


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    found = _collect(manifest)
    numbers, counts = _numbers(found)
    with paper.paper_style():
        fig = Figure(figsize=(paper.FULL_WIDTH, HEIGHT), layout="constrained")
        _draw(fig, found)
    return FigureOutput(
        fig, _claim(counts), DETAIL, numbers, _csv(found), counts
    )
