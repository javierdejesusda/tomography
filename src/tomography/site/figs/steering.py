"""Figure 7: Sentiment steering and fluency.

Each checkpoint draws its seed-mean judged sentiment score, perplexity ratio
to the unsteered text and distinct-1 against the relative coefficient of the
C1 run, with the unsteered row at coefficient 0 and the prompting arms beside
the score panels.
"""

from collections import defaultdict
from typing import Any

from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np

from tomography.site import runfiles
from tomography.site.figs import _group
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import axes
from tomography.viz import paper

MAX_RATIO = 1.5
MIN_DISTINCT = 0.15
LEFT = 0.62
PANEL_WIDTH = 1.26
PANEL_HEIGHT = 1.1
PITCH = 1.95
PROMPT_X = 0.345
PROMPT_STEP = 0.011
HEADER = ["panel", "model", "x", "series", "value", "lo", "hi"]
DETAIL = " ".join(
    [
        "Sentiment steering of each checkpoint with the linear direction at",
        "relative coefficients from -0.3 to 0.3; each point is the mean over",
        "the three readers, and the unsteered text sits at coefficient 0.",
        "(a, d) Judged sentiment score; the triangles at the right edge are",
        "the score of prompting for positive (up) and negative (down)",
        "sentiment. (b, e) Perplexity as a ratio to the unsteered text, on a",
        "log axis, with a line at 1.5. (c, f) Distinct-1, with a line at",
        "0.15. A point is hollow when it breaks the descriptive fluency rule",
        "(perplexity ratio at most 1.5 and distinct-1 at least 0.15).",
        "C1 has no random-direction arm, so the baselines are the unsteered",
        "text and prompting. The rows are GPT-2 (top) and Qwen3 (bottom); a",
        "dashed line is a post-trained model.",
    ]
)


def _tables(manifest: Manifest) -> dict[str, list[dict[str, Any]]]:
    found = {}
    for model in manifest.models("c1"):
        run = manifest.run("c1", model)
        assert run is not None
        found[model] = runfiles.rows(run, "table.json")
    return found


def _mean(rows: list[dict[str, Any]], coeff: float, key: str) -> float:
    cell = [
        r[key] for r in rows if r["arm"] == "linear" and r["coeff"] == coeff
    ]
    return float(np.mean(cell))


def _reference(rows: list[dict[str, Any]]) -> dict[str, Any]:
    (row,) = [r for r in rows if r["arm"] == "none"]
    return row


def _curves(rows: list[dict[str, Any]]) -> dict[str, dict[float, float]]:
    ref = _reference(rows)
    coeffs = sorted({r["coeff"] for r in rows if r["arm"] == "linear"})
    curves: dict[str, dict[float, float]] = defaultdict(dict)
    curves["score"][0.0] = float(ref["score"])
    curves["ppl_ratio"][0.0] = 1.0
    curves["distinct_1"][0.0] = float(ref["distinct_1"])
    for coeff in coeffs:
        curves["score"][coeff] = _mean(rows, coeff, "score")
        ratio = _mean(rows, coeff, "perplexity") / ref["perplexity"]
        curves["ppl_ratio"][coeff] = ratio
        curves["distinct_1"][coeff] = _mean(rows, coeff, "distinct_1")
    return curves


def _is_fluent(ratio: float, distinct: float) -> bool:
    return ratio <= MAX_RATIO and distinct >= MIN_DISTINCT


def _place(fig: Figure, left: float, top: float) -> Axes:
    w, h = fig.get_size_inches()
    return fig.add_axes(
        (
            left / w,
            1 - (top + PANEL_HEIGHT) / h,
            PANEL_WIDTH / w,
            PANEL_HEIGHT / h,
        )
    )


def _draw_model(
    ax_set: list[Axes],
    model: str,
    curves: dict[str, dict[float, float]],
    letters: list[str],
    csv_rows: list[list[Any]],
) -> None:
    style = axes.model_style(model)
    xs = sorted(curves["score"])
    fluent = {
        x: x == 0.0
        or _is_fluent(curves["ppl_ratio"][x], curves["distinct_1"][x])
        for x in xs
    }
    for ax, letter, series in zip(
        ax_set, letters, ("score", "ppl_ratio", "distinct_1"), strict=True
    ):
        values = [curves[series][x] for x in xs]
        ax.plot(
            xs,
            values,
            color=style.color,
            ls=style.linestyle,
            lw=1.0,
            zorder=3,
        )
        for x, value in zip(xs, values, strict=True):
            ax.plot(
                [x],
                [value],
                marker="o",
                ms=2.6,
                mfc=style.color if fluent[x] else "white",
                mec=style.color,
                mew=0.6,
                ls="none",
                zorder=4,
            )
            csv_rows.append([letter, model, x, series, value, None, None])


def _row_legend(
    fig: Figure, models: list[str], top: float, extra: bool
) -> None:
    handles: list[Line2D] = []
    for model in models:
        style = axes.model_style(model)
        handles.append(
            Line2D(
                [],
                [],
                color=style.color,
                ls=style.linestyle,
                lw=1.0,
                label=paper.display_name(model),
            )
        )
    if extra:
        handles.append(
            Line2D(
                [],
                [],
                ls="none",
                marker="o",
                ms=2.6,
                mfc="white",
                mec=paper.INK,
                mew=0.6,
                label="not fluent",
            )
        )
        handles.append(
            Line2D(
                [],
                [],
                ls="none",
                marker="^",
                ms=3.0,
                mfc=paper.INK,
                mec=paper.INK,
                label="prompting",
            )
        )
    height = fig.get_size_inches()[1]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1 - top / height),
        ncol=4 if extra else 3,
        handlelength=1.6,
        columnspacing=1.0,
        labelspacing=0.2,
        borderaxespad=0.0,
    )


def _finish(ax_set: list[Axes], bottom: bool) -> None:
    score, ratio, distinct = ax_set
    score.set_ylim(0.0, 1.0)
    score.set_yticks([0.0, 0.5, 1.0])
    score.set_ylabel("sentiment score")
    ratio.set_yscale("log")
    ratio.set_ylim(0.8, 130)
    ratio.set_yticks([1, 10, 100])
    ratio.set_yticklabels(["1", "10", "100"])
    ratio.yaxis.set_minor_locator(ticker.NullLocator())
    ratio.axhline(
        MAX_RATIO, color=paper.CHANCE, lw=0.5, ls=(0, (3, 2)), zorder=1
    )
    ratio.set_ylabel("perplexity ratio")
    distinct.set_ylim(0.0, 0.32)
    distinct.set_yticks([0.0, 0.1, 0.2, 0.3])
    distinct.axhline(
        MIN_DISTINCT, color=paper.CHANCE, lw=0.5, ls=(0, (3, 2)), zorder=1
    )
    distinct.set_ylabel("distinct-1")
    for ax in ax_set:
        ax.set_xlim(-0.33, 0.42 if ax is score else 0.33)
        ax.set_xticks([-0.3, 0.0, 0.3])
        ax.set_xticklabels(["-0.3", "0", "0.3"])
        if bottom:
            ax.set_xlabel("relative coefficient")


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    tables = _tables(manifest)
    curves = {m: _curves(t) for m, t in tables.items()}
    rows_of = {
        "GPT-2": [m for m in tables if m.startswith("gpt2")],
        "Qwen3": [m for m in tables if not m.startswith("gpt2")],
    }
    csv_rows: list[list[Any]] = []
    height = 4.3
    with paper.paper_style():
        fig = Figure(figsize=(paper.FULL_WIDTH, height))
        letters = iter("abcdef")
        for r, models in enumerate(rows_of.values()):
            top = 0.85 + r * PITCH
            ax_set = [
                _place(fig, LEFT + c * (PANEL_WIDTH + 0.5), top)
                for c in range(3)
            ]
            names = [next(letters) for _ in range(3)]
            for model in models:
                _draw_model(ax_set, model, curves[model], names, csv_rows)
            for k, model in enumerate(models):
                style = axes.model_style(model)
                for arm, marker in (("prompting+1", "^"), ("prompting-1", "v")):
                    (row,) = [x for x in tables[model] if x["arm"] == arm]
                    x = PROMPT_X + k * PROMPT_STEP
                    ax_set[0].plot(
                        [x],
                        [row["score"]],
                        marker=marker,
                        ms=2.8,
                        color=style.color,
                        mfc=style.color if style.filled else "white",
                        ls="none",
                        zorder=4,
                    )
                    csv_rows.append(
                        [
                            names[0],
                            model,
                            None,
                            arm,
                            float(row["score"]),
                            None,
                            None,
                        ]
                    )
            _finish(ax_set, bottom=r == len(rows_of) - 1)
            for ax, name in zip(ax_set, names, strict=True):
                paper.panel_letter(ax, name)
            _group.clip_ticks(ax_set[0])
            _group.clip_ticks(ax_set[2])
            _row_legend(fig, models, top - 0.72, extra=r == 0)

    fluent = {0.1: 0, 0.3: 0}
    for model in tables:
        for size in fluent:
            if all(
                _is_fluent(
                    curves[model]["ppl_ratio"][s],
                    curves[model]["distinct_1"][s],
                )
                for s in (size, -size)
            ):
                fluent[size] += 1
    numbers = {
        "models": float(len(tables)),
        "score_unsteered": float(
            np.mean([_reference(t)["score"] for t in tables.values()])
        ),
        "score_minus": float(
            np.mean([c["score"][-0.1] for c in curves.values()])
        ),
        "score_plus": float(
            np.mean([c["score"][0.1] for c in curves.values()])
        ),
        "fluent_at_0.1": float(fluent[0.1]),
        "fluent_at_0.3": float(fluent[0.3]),
    }
    n = len(tables)
    claim = (
        "Across the "
        f"{n} checkpoints the mean judged sentiment score is "
        f"{numbers['score_unsteered']:.3f} unsteered, "
        f"{numbers['score_minus']:.3f} at relative coefficient -0.1 and "
        f"{numbers['score_plus']:.3f} at +0.1, and {fluent[0.1]} of {n} keep "
        "fluent text at both signs of 0.1 (perplexity within 1.5 times the "
        "unsteered text and distinct-1 at least 0.15) against "
        f"{fluent[0.3]} of {n} at both signs of 0.3."
    )
    return FigureOutput(
        fig, claim, DETAIL, numbers, _group.csv_text(HEADER, csv_rows)
    )
