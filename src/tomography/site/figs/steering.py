"""Figure 7: Sentiment steering and fluency.

Three dose panels (GPT-2, Qwen3 base, Qwen3 post-trained) draw the judged
sentiment score of each checkpoint against the relative coefficient of the
C1 run, with the range over the three steering vectors and the prompting
scores in a strip. A tile matrix next to them carries the fluency rule of
every checkpoint and coefficient.
"""

from collections.abc import Sequence
import json
from typing import Any

import matplotlib as mpl
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
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
HEIGHT = 3.6
COEFFS = (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3)
TILE_FONT = 6.5
TILE_LIMIT = 32.0
COLORBAR_TICKS = (1, 1.5, 4, 10, 30)
ROLE_TITLES = {
    "gpt2": "GPT-2",
    "base": "Qwen3 base",
    "post": "Qwen3 post-trained",
}
HEADER = ["panel", "model", "x", "series", "value", "lo", "hi"]
CMAP = LinearSegmentedColormap.from_list(
    "perplexity",
    [mpl.colormaps["cividis_r"](v) for v in np.linspace(0, 0.5, 64)],
)
DETAIL = (
    "(a to c) Mean judged sentiment of the completions (positive-class "
    "probability of a sentiment classifier) against the relative "
    "coefficient, one line per checkpoint, with the range over the 3 "
    "steering vectors (band) and the score under a positive or negative "
    "instruction instead of steering (triangles, right strip). At relative "
    "coefficients {minus}0.1 and +0.1 the mean score moves {moves}. (d) "
    "Perplexity of {n_neutral} held-out WikiText-103 passages under the "
    "steered model, as a ratio to the unsteered model (colour and number); "
    "an asterisk marks a distinct-1 of the completions below 0.15, and "
    "framed cells keep the fluency rule (ratio at most 1.5 and distinct-1 "
    "at least 0.15). The rule holds at both signs of 0.1 on {held}, and at "
    "0.3 or {minus}0.3 on {fails}. This run has no random-direction arm; "
    "the finder comparison steers with three random directions of the same "
    "norm (Figure 1, last row). Table 6 adds the perplexity of the "
    "completions themselves."
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
        self.prompting = {
            arm: float(next(r for r in rows if r["arm"] == arm)["score"])
            for arm in ("prompting+1", "prompting-1")
        }

    def fluent(self, coeff: float) -> bool:
        return (
            self.ratio[coeff] <= MAX_RATIO
            and self.distinct[coeff] >= MIN_DISTINCT
        )


def _linestyle(model: str) -> Any:
    if stats.role(model) == "post":
        return paper.LINESTYLES["post_trained"]
    return "-"


def _dose_panel(ax: Any, strip: Any, members: list[_Model], role: str) -> None:
    ax.set_title(ROLE_TITLES[role], loc="left")
    for k, model in enumerate(members):
        style = axes.model_style(model.key)
        xs = list(COEFFS)
        ax.fill_between(
            xs,
            [model.low[c] for c in xs],
            [model.high[c] for c in xs],
            color=style.color,
            alpha=0.15,
            lw=0,
            zorder=2,
        )
        ax.plot(
            xs,
            [model.score[c] for c in xs],
            color=style.color,
            ls=_linestyle(model.key),
            lw=paper.DATA_WIDTH,
            marker=style.marker,
            zorder=3,
            **_rows.fill(model.key, style.color),
        )
        x = np.linspace(0.2, 0.8, max(len(members), 2))[k]
        for arm, marker in (("prompting+1", "^"), ("prompting-1", "v")):
            strip.plot(
                [x],
                [model.prompting[arm]],
                marker=marker,
                ls="none",
                color=style.color,
                zorder=3,
                **_rows.fill(model.key, style.color),
            )
    ax.set_ylim(0.0, 1.0)
    axes.unit_axis(ax, "y")
    ax.set_xlim(-0.33, 0.33)
    ax.xaxis.set_major_locator(ticker.FixedLocator([-0.3, 0.0, 0.3]))
    ax.xaxis.set_major_formatter(_rows.tick_formatter())
    strip.set_xlim(0.0, 1.0)
    strip.set_xticks([])
    strip.tick_params(left=False, labelleft=False)
    strip.spines["left"].set_visible(False)


def _tile_text(model: _Model, coeff: float) -> str:
    ratio = model.ratio[coeff]
    text = f"{ratio:.2f}" if ratio < 10 else f"{ratio:.0f}"
    return text + ("*" if model.distinct[coeff] < MIN_DISTINCT else "")


def _matrix(
    ax: Any, count_ax: Any, by_key: dict[str, _Model], models: list[str]
) -> Any:
    rows = axes.model_rows(models)
    norm = LogNorm(vmin=1.0, vmax=TILE_LIMIT, clip=True)
    mesh = None
    for key in models:
        model = by_key[key]
        values = np.array([[model.ratio[c] for c in COEFFS]])
        mesh = ax.pcolormesh(
            np.arange(len(COEFFS) + 1) - 0.5,
            [rows[key] - 0.5, rows[key] + 0.5],
            values,
            cmap=CMAP,
            norm=norm,
            zorder=1,
        )
        for col, coeff in enumerate(COEFFS):
            ax.text(
                col,
                rows[key],
                _tile_text(model, coeff),
                ha="center",
                va="center",
                fontsize=TILE_FONT,
                color=paper.INK,
                zorder=3,
            )
            if model.fluent(coeff):
                ax.add_patch(
                    Rectangle(
                        (col - 0.5, rows[key] - 0.5),
                        1,
                        1,
                        fill=False,
                        ec=paper.INK,
                        lw=0.9,
                        zorder=4,
                    )
                )
    ax.set_xlim(-0.5, len(COEFFS) - 0.5)
    ax.set_ylim(max(rows.values()) + 0.5, -0.5)
    ax.set_yticks([rows[k] for k in models])
    ax.set_yticklabels([paper.display_name(k) for k in models])
    ax.set_xticks(range(len(COEFFS)))
    ax.set_xticklabels([_rows.tick_text(c) for c in COEFFS])
    ax.tick_params(length=0, labelsize=TILE_FONT)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    count_ax.set_xlim(-0.5, len(COEFFS) - 0.5)
    count_ax.set_ylim(2.5, -0.5)
    count_ax.set_yticks(range(3))
    count_ax.set_yticklabels([ROLE_TITLES[r] for r in stats.ROLES])
    count_ax.set_xticks([])
    count_ax.tick_params(length=0, labelsize=TILE_FONT)
    for spine in count_ax.spines.values():
        spine.set_visible(False)
    count_ax.set_xlabel("relative coefficient")
    for r, role in enumerate(stats.ROLES):
        members = [by_key[k] for k in models if stats.role(k) == role]
        for col, coeff in enumerate(COEFFS):
            hits = sum(m.fluent(coeff) for m in members)
            count_ax.text(
                col,
                r,
                f"{hits}/{len(members)}",
                ha="center",
                va="center",
                fontsize=TILE_FONT,
                color=paper.INK,
            )
    return mesh


def _legend(fig: Figure, models: list[str]) -> None:
    handles = []
    for key in models:
        style = axes.model_style(key)
        handles.append(
            Line2D(
                [],
                [],
                color=style.color,
                ls=_linestyle(key),
                lw=paper.DATA_WIDTH,
                marker=style.marker,
                label=paper.display_name(key),
                **_rows.fill(key, style.color),
            )
        )
    legend = fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=5,
        handlelength=1.8,
        columnspacing=1.0,
        fontsize=7,
    )
    legend.get_frame().set_edgecolor("none")


def _csv(by_key: dict[str, _Model], models: list[str]) -> str:
    letter = {"gpt2": "a", "base": "b", "post": "c"}
    rows: list[list[Any]] = []
    for key in models:
        model = by_key[key]
        panel = letter[stats.role(key)]
        for c in COEFFS:
            rows.append(
                [
                    panel,
                    key,
                    c,
                    "score",
                    model.score[c],
                    model.low[c],
                    model.high[c],
                ]
            )
        for arm, value in model.prompting.items():
            rows.append([panel, key, None, arm, value, None, None])
        for c in COEFFS:
            rows.append(["d", key, c, "ppl_ratio", model.ratio[c], None, None])
            rows.append(
                ["d", key, c, "distinct_1", model.distinct[c], None, None]
            )
            rows.append(
                ["d", key, c, "fluent", float(model.fluent(c)), None, None]
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

    with paper.paper_style():
        fig = _rows.new_figure(HEIGHT)
        outer = fig.add_gridspec(1, 2, width_ratios=[1.55, 2.45])
        left = outer[0].subgridspec(3, 2, width_ratios=[1.0, 0.25], wspace=0.04)
        right = outer[1].subgridspec(
            2, 2, height_ratios=[10, 1.6], width_ratios=[1.0, 0.05]
        )
        first = None
        for i, role in enumerate(stats.ROLES):
            ax = fig.add_subplot(left[i, 0], sharex=first)
            first = first or ax
            strip = fig.add_subplot(left[i, 1], sharey=ax)
            _dose_panel(ax, strip, [by_key[m] for m in by_role[role]], role)
            paper.panel_letter(ax, "abc"[i])
            if i < 2:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("relative coefficient")
            if i == 1:
                ax.set_ylabel("judged sentiment")
        matrix_ax = fig.add_subplot(right[0, 0])
        count_ax = fig.add_subplot(right[1, 0])
        mesh = _matrix(matrix_ax, count_ax, by_key, models)
        paper.panel_letter(matrix_ax, "d")
        cax = fig.add_subplot(right[0, 1])
        bar = fig.colorbar(mesh, cax=cax)
        bar.set_ticks(list(COLORBAR_TICKS))
        bar.set_ticklabels([f"{t:g}" for t in COLORBAR_TICKS])
        bar.ax.minorticks_off()
        bar.ax.tick_params(labelsize=TILE_FONT, length=2)
        bar.set_label(
            "neutral-text perplexity, ratio to unsteered", fontsize=TILE_FONT
        )
        bar.outline.set_visible(False)
        _legend(fig, models)

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
    counts = {
        "fluent01": stats.per_role(flags01),
        "fluent03": stats.per_role(flags03),
    }
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
        moves=moves,
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
