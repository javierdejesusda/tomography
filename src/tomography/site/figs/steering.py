"""Figure 7: Sentiment steering and fluency.

Three columns (GPT-2, Qwen3 base, Qwen3 post-trained) and three rows share
one relative-coefficient axis. Row a draws the judged sentiment of each
checkpoint with the range over the three steering vectors and the score under
an instruction; rows b and c draw the two inputs of the fluency rule, the
neutral-text perplexity ratio and the distinct-1 of the completions, each
against its limit. A marker is filled where a checkpoint keeps the rule at
that coefficient and open where it does not.
"""

from collections.abc import Sequence
import itertools
import json
from typing import Any

from matplotlib import ticker
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np

from tomography.site import runfiles
from tomography.site import stats
from tomography.site.figs import _group
from tomography.site.figs import _rows
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import axes
from tomography.viz import paper

MAX_RATIO = 1.5
MIN_DISTINCT = 0.15
HEIGHT = 5.0
COEFFS = (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3)
X_LIMIT = 0.33
X_PAD = 4.5
RATIO_LIMITS = (1.0, 100.0)
DISTINCT_LIMIT = 0.35
BAND_ALPHA = 0.15
LABEL_GAP = 0.12
LABEL_FONT = 7
ROLE_TITLES = {
    "gpt2": "GPT-2",
    "base": "Qwen3 base",
    "post": "Qwen3 post-trained",
}
GPT2_LABELS = {
    "gpt2": "small",
    "gpt2-medium": "medium",
    "gpt2-large": "large",
    "gpt2-xl": "XL",
}
HEADER = ["panel", "model", "x", "series", "value", "lo", "hi"]
DETAIL = (
    "(a) Mean judged sentiment of the completions (positive-class "
    "probability of a sentiment classifier) against the relative "
    "coefficient, one line per checkpoint (lighter for the smaller "
    "checkpoint, named at the line end), with the range over the 3 "
    "steering vectors (shaded band) and the score under a positive or "
    "negative instruction instead of steering (dotted horizontal lines). "
    "A marker is filled where the fluency rule holds (ratio at most "
    "{max_ratio} and distinct-1 at least {min_distinct}) and open where it "
    "does not, and a segment that touches an open marker is dashed. At "
    "relative coefficients {minus}0.1 and +0.1 the mean score moves "
    "{moves}. A checkpoint counts as moving when all 3 steering vectors "
    "move the score past the unsteered value at both signs, on the primary "
    "judge and on a second judge (another sentiment classifier): this holds "
    "on {moves_count}, and the two judges agree on {agree}. (b) "
    "Perplexity of {n_neutral} held-out WikiText-103 passages under the "
    "steered model, as a ratio to the unsteered model on a log axis, with "
    "its limit of {max_ratio}. (c) Distinct-1 of the completions, which "
    "fails the rule below {min_distinct} (the second limit). The rule holds "
    "at both signs of 0.1 on {held}, and at 0.3 or {minus}0.3 on {fails}. "
    "This run has no random-direction arm; the finder comparison steers with "
    "three random directions of the same norm ([[fig:teaser]], last row). "
    "The fluency rule uses the perplexity of neutral text and the distinct-1 "
    "of the completions, not the perplexity of the completions, which "
    "[[table:steering]] gives together with the ratios at {minus}0.1 and "
    "+0.1."
)


def _table(manifest: Manifest, model: str) -> list[dict[str, Any]]:
    run = manifest.run("c1", model)
    assert run is not None
    return runfiles.rows(run, "table.json")


def _neutral_passages(manifest: Manifest, models: Sequence[str]) -> int:
    sizes = set()
    for model in models:
        run = manifest.run("c1", model)
        assert run is not None
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        sizes.add(int(config["n_neutral"]))
    if len(sizes) != 1:
        raise ValueError(f"runs disagree on n_neutral: {sorted(sizes)}")
    return sizes.pop()


class _Model:
    """The per-coefficient values of one checkpoint."""

    def __init__(self, key: str, rows: list[dict[str, Any]]) -> None:
        (none,) = [r for r in rows if r["arm"] == "none"]
        self.key = key
        self.score: dict[float, float] = {0.0: float(none["score"])}
        self.low: dict[float, float] = {0.0: float(none["score"])}
        self.high: dict[float, float] = {0.0: float(none["score"])}
        self.ratio: dict[float, float] = {0.0: 1.0}
        self.distinct: dict[float, float] = {0.0: float(none["distinct_1"])}
        for coeff in COEFFS:
            if coeff == 0.0:
                continue
            cell = [
                r for r in rows if r["arm"] == "linear" and r["coeff"] == coeff
            ]
            scores = [float(r["score"]) for r in cell]
            self.score[coeff] = float(np.mean(scores))
            self.low[coeff] = min(scores)
            self.high[coeff] = max(scores)
            perplexity = float(np.mean([r["perplexity"] for r in cell]))
            self.ratio[coeff] = perplexity / float(none["perplexity"])
            self.distinct[coeff] = float(
                np.mean([r["distinct_1"] for r in cell])
            )
        self.vectors = [r for r in rows if r["arm"] == "linear"]
        self.unsteered = none
        self.prompting = {
            arm: float(next(r for r in rows if r["arm"] == arm)["score"])
            for arm in ("prompting+1", "prompting-1")
        }

    def moves(self, column: str) -> bool:
        """Tells whether every steering vector moves a judge's score.

        Args:
            column: The judge's score column.

        Returns:
            Whether, for every vector, the score is above the unsteered one
            at +0.1 and below it at -0.1.
        """
        base = float(self.unsteered[column])
        return all(
            sign * (float(r[column]) - base) > 0
            for r in self.vectors
            for coeff, sign in ((0.1, 1), (-0.1, -1))
            if r["coeff"] == coeff
        )

    def fluent(self, coeff: float) -> bool:
        return (
            self.ratio[coeff] <= MAX_RATIO
            and self.distinct[coeff] >= MIN_DISTINCT
        )


def end_label(model: str) -> str:
    """Names a checkpoint at the end of its line.

    Args:
        model: Checkpoint key such as `gpt2-xl` or `qwen3-4b-base`.

    Returns:
        The size: a word for GPT-2 and the parameter count for Qwen3.
    """
    if model.startswith("gpt2"):
        return GPT2_LABELS[model]
    return model.removeprefix("qwen3-").removesuffix("-base").upper()


def _spread(ys: list[float], gap: float, ceiling: float) -> list[float]:
    """Moves label heights apart so that neighbours keep a minimum gap.

    Args:
        ys: Wanted heights, in any order.
        gap: Minimum distance between two heights.
        ceiling: The highest height a label may take.

    Returns:
        The heights, in the order of the input: pushed up where they are
        closer than `gap`, then pulled back down from the ceiling.
    """
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    placed = [ys[i] for i in order]
    for k in range(1, len(placed)):
        placed[k] = max(placed[k], placed[k - 1] + gap)
    placed[-1] = min(placed[-1], ceiling)
    for k in range(len(placed) - 2, -1, -1):
        placed[k] = min(placed[k], placed[k + 1] - gap)
    out = [0.0] * len(ys)
    for k, i in enumerate(order):
        out[i] = placed[k]
    return out


def _series(
    ax: Any, model: _Model, values: dict[float, float], top: bool = False
) -> None:
    color = axes.model_style(model.key).color
    for a, b in itertools.pairwise(COEFFS):
        solid = model.fluent(a) and model.fluent(b)
        ax.plot(
            [a, b],
            [values[a], values[b]],
            color=color,
            ls="-" if solid else paper.LINESTYLES["post_trained"],
            lw=paper.DATA_WIDTH,
            gid="segment",
            zorder=3,
        )
    for coeff in COEFFS:
        ax.plot(
            [coeff],
            [values[coeff]],
            ls="none",
            marker=paper.SHAPES["direction"],
            color=color,
            markerfacecolor=color if model.fluent(coeff) else "white",
            markeredgecolor=color,
            markeredgewidth=paper.MARKER_EDGE,
            markersize=paper.MARKER_SIZE,
            gid="point",
            zorder=4,
        )
    if top:
        ax.fill_between(
            list(COEFFS),
            [model.low[c] for c in COEFFS],
            [model.high[c] for c in COEFFS],
            color=color,
            alpha=BAND_ALPHA,
            lw=0,
            zorder=2,
        )
        for score in model.prompting.values():
            line = ax.axhline(
                score, color=color, ls=":", lw=0.8, gid="instruction", zorder=2
            )
            line.set_alpha(0.9)


def _end_labels(ax: Any, members: list[_Model]) -> None:
    ends = [m.score[COEFFS[-1]] for m in members]
    for model, y in zip(members, _spread(ends, LABEL_GAP, 0.97), strict=True):
        ax.plot(
            [1.015, 1.075],
            [y, y],
            transform=ax.get_yaxis_transform(),
            color=axes.model_style(model.key).color,
            lw=paper.DATA_WIDTH,
            clip_on=False,
            gid="label-key",
        )
        ax.text(
            1.095,
            y,
            end_label(model.key),
            transform=ax.get_yaxis_transform(),
            color=paper.INK,
            fontsize=LABEL_FONT,
            va="center",
            ha="left",
        )


def _limit(ax: Any, value: float) -> None:
    ax.axhline(
        value,
        color=paper.ENC["reference"],
        lw=0.5,
        gid="limit",
        zorder=1,
    )
    ax.text(
        1.015,
        value,
        f"limit {value:g}",
        transform=ax.get_yaxis_transform(),
        color=paper.ENC["secondary_text"],
        fontsize=LABEL_FONT,
        va="center",
        ha="left",
    )


def _style_axes(grid: Any) -> None:
    for ax in grid.flat:
        ax.set_xlim(-X_LIMIT, X_LIMIT)
        ax.xaxis.set_major_locator(ticker.FixedLocator([-0.3, 0.0, 0.3]))
        ax.xaxis.set_major_formatter(axes.minus_formatter())
        ax.tick_params(axis="x", pad=X_PAD)
    for ax in grid[0]:
        ax.set_ylim(0.0, 1.0)
        axes.unit_axis(ax, "y")
    for ax in grid[1]:
        ax.set_yscale("log")
        ax.set_ylim(*RATIO_LIMITS)
        ax.yaxis.set_major_locator(ticker.FixedLocator([1, 10, 100]))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_locator(ticker.NullLocator())
    for ax in grid[2]:
        ax.set_ylim(0.0, DISTINCT_LIMIT)
        ax.yaxis.set_major_locator(ticker.FixedLocator([0.0, 0.1, 0.2, 0.3]))
        ax.yaxis.set_major_formatter(axes.minus_formatter())


def _key(fig: Figure) -> None:
    color = paper.INK
    handles = [
        Line2D(
            [],
            [],
            color=color,
            marker=paper.SHAPES["direction"],
            markerfacecolor=color,
            markersize=paper.MARKER_SIZE,
            lw=paper.DATA_WIDTH,
            label="fluent",
        ),
        Line2D(
            [],
            [],
            color=color,
            marker=paper.SHAPES["direction"],
            markerfacecolor="white",
            markersize=paper.MARKER_SIZE,
            ls=paper.LINESTYLES["post_trained"],
            lw=paper.DATA_WIDTH,
            label="not fluent",
        ),
        Line2D([], [], color=color, ls=":", lw=0.8, label="instruction score"),
    ]
    legend = fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=3,
        handlelength=2.2,
        columnspacing=1.6,
        fontsize=LABEL_FONT,
    )
    legend.get_frame().set_edgecolor("none")


def _csv(by_key: dict[str, _Model], models: list[str]) -> str:
    rows: list[list[Any]] = []
    for key in models:
        model = by_key[key]
        for c in COEFFS:
            rows.append(
                [
                    "a",
                    key,
                    c,
                    "score",
                    model.score[c],
                    model.low[c],
                    model.high[c],
                ]
            )
        for arm, value in model.prompting.items():
            rows.append(["a", key, None, arm, value, None, None])
        for c in COEFFS:
            rows.append(["b", key, c, "ppl_ratio", model.ratio[c], None, None])
            rows.append(
                ["c", key, c, "distinct_1", model.distinct[c], None, None]
            )
            rows.append(
                ["a", key, c, "fluent", float(model.fluent(c)), None, None]
            )
    return _group.csv_text(HEADER, rows)


def _hits(flags: dict[str, bool]) -> str:
    counts = stats.per_role(flags)
    if not any(c.hits for c in counts.values()):
        return f"none of the {len(flags)} checkpoints"
    return stats.per_role_text(counts)


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    models = manifest.models("c1")
    by_key = {m: _Model(m, _table(manifest, m)) for m in models}
    by_role = {
        r: [m for m in models if stats.role(m) == r] for r in stats.ROLES
    }
    n_neutral = _neutral_passages(manifest, models)

    def mean(coeff: float) -> dict[str, float]:
        return {
            r: float(np.mean([by_key[m].score[coeff] for m in by_role[r]]))
            for r in stats.ROLES
        }

    unsteered, minus, plus = mean(0.0), mean(-0.1), mean(0.1)
    flags01 = {
        m: by_key[m].fluent(0.1) and by_key[m].fluent(-0.1) for m in models
    }
    flags03 = {
        m: by_key[m].fluent(0.3) or by_key[m].fluent(-0.3) for m in models
    }
    primary = {m: by_key[m].moves("score") for m in models}
    check = {m: by_key[m].moves("score_check") for m in models}
    counts = {
        "fluent01": stats.per_role(flags01),
        "fluent03": stats.per_role(flags03),
        "moves01": stats.per_role({m: primary[m] and check[m] for m in models}),
        "moves01_primary": stats.per_role(primary),
        "moves01_check": stats.per_role(check),
        "judges_agree": stats.per_role(
            {m: primary[m] == check[m] for m in models}
        ),
    }

    with paper.paper_style():
        fig = _rows.new_figure(HEIGHT)
        grid = fig.subplots(3, 3, sharex=True, sharey="row")
        _style_axes(grid)
        for col, role in enumerate(stats.ROLES):
            members = [by_key[m] for m in by_role[role]]
            for model in members:
                _series(grid[0, col], model, model.score, top=True)
                _series(grid[1, col], model, model.ratio)
                _series(grid[2, col], model, model.distinct)
            grid[0, col].set_title(ROLE_TITLES[role], loc="center")
            _end_labels(grid[0, col], members)
            count = counts["fluent01"][role]
            grid[0, col].text(
                0.97,
                0.04,
                f"fluent at both\nsigns of 0.1:\n{count.hits} of {count.total}",
                transform=grid[0, col].transAxes,
                fontsize=LABEL_FONT,
                color=paper.INK,
                va="bottom",
                ha="right",
            )
            _limit(grid[1, col], MAX_RATIO)
            _limit(grid[2, col], MIN_DISTINCT)
            grid[2, col].set_xlabel("relative coefficient")
        grid[0, 0].set_ylabel("judged sentiment")
        grid[1, 0].set_ylabel("neutral-text\nperplexity ratio")
        grid[2, 0].set_ylabel("distinct-1")
        for row, letter in enumerate("abc"):
            paper.panel_letter(grid[row, 0], letter)
        _key(fig)
        _rows.freeze_layout(fig)

    numbers: dict[str, float] = {"n_neutral": float(n_neutral)}
    for role in stats.ROLES:
        numbers[f"unsteered_{role}"] = unsteered[role]
        numbers[f"minus_{role}"] = minus[role]
        numbers[f"plus_{role}"] = plus[role]
        numbers[f"n_{role}"] = float(len(by_role[role]))
        for name in counts:
            numbers[f"{name}_{role}"] = float(counts[name][role].hits)
    moves = _rows.join_and(
        [
            f"from {_rows.number(unsteered[r])} to {_rows.number(minus[r])} "
            f"and {_rows.number(plus[r])} {where}"
            for r, where in zip(
                stats.ROLES,
                ("on GPT-2", "on Qwen3 base", "on post-trained checkpoints"),
                strict=True,
            )
        ]
    )
    detail = DETAIL.format(
        minus=axes.MINUS,
        max_ratio=MAX_RATIO,
        min_distinct=MIN_DISTINCT,
        moves=moves,
        moves_count=stats.per_role_text(counts["moves01"]),
        agree=stats.per_role_text(counts["judges_agree"]),
        n_neutral=n_neutral,
        held=stats.per_role_text(counts["fluent01"]),
        fails=_hits(flags03),
    )
    claim = (
        "Steering moves the judged sentiment, less on post-trained checkpoints."
    )
    return FigureOutput(
        fig, claim, detail, numbers, _csv(by_key, models), counts
    )
