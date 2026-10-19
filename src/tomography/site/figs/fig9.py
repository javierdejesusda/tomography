"""Figure 9: Where reading does not beat text or noise.

Panel (a) reads TruthfulQA MC1 of the validation-layer reader of R6 against
the text-only classifiers and the log-probability row, panel (b) pairs the
honesty transfer of R5 with the seed agreement of A1, and panel (c) gives the
out-of-distribution accuracy of every model on the R7 evaluation sets.
"""

from typing import Any

from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from tomography.site import params
from tomography.site import store as store_module
from tomography.site.figs import _group
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import axes
from tomography.viz import paper

NOT_SETS = ("val", "test")
NEGATED = ("neg_cities", "neg_sp_en_trans")
TEXT = ("text_word", "text_char")
LEFT = 0.62
HEADER = ["panel", "model", "x", "series", "value", "lo", "hi"]
HEAT = LinearSegmentedColormap.from_list(
    "chance_centred", ["#7fb2d9", "#ffffff", "#e8a06b"]
)
SET_NAMES = {
    "cities": "cities",
    "neg_cities": "neg. cities",
    "sp_en_trans": "sp-en",
    "neg_sp_en_trans": "neg. sp-en",
    "larger_than": "larger than",
    "smaller_than": "smaller than",
    "cities_cities_conj": "cities conj.",
    "cities_cities_disj": "cities disj.",
    "common_claim_true_false": "common claims",
}
DETAIL = " ".join(
    [
        "(a) MC1 on TruthfulQA of the logistic reader at the layer chosen on",
        "validation (dots, with 95 percent bootstrap intervals that resample",
        "the 15 readers), the standard log-probability row (crosses), and the",
        "two text-only classifiers (grey band, the range of their 95 percent",
        "Wilson intervals over 173 questions, identical for every model).",
        "The interval kinds differ, so their widths are not comparable. The",
        "text rows are stored by the runs that have them. (b) Honesty",
        "transfer accuracy of R5 against the agreement of the reader",
        "directions across seeds at the same layer, one dot per model;",
        "exploratory, 59 test pairs, and the transfer layer is chosen on 53",
        "validation pairs, so it is noisy; the dashed line is chance.",
        "(c) Accuracy of the reader on each R7 evaluation set at the layer",
        "chosen on the fact validation split, no refit; the color is centred",
        "at chance, blue below it and orange above it, and the number is the",
        "accuracy. Negated sets are the ones named neg. Filled markers are",
        "GPT-2 and base models, open markers are post-trained models.",
    ]
)


def _table(
    store: store_module.Store, manifest: Manifest, exp: str, model: str
) -> list[dict[str, Any]]:
    run = manifest.run(exp, model)
    assert run is not None
    return store.rows(run, "table.json")


def _r6(
    manifest: Manifest, store: store_module.Store
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, Any]]]:
    found: dict[str, dict[str, dict[str, Any]]] = {}
    text: dict[str, dict[str, Any]] = {}
    for model in manifest.models("r6"):
        rows = _table(store, manifest, "r6", model)
        found[model] = {
            "reader": next(
                r
                for r in rows
                if r["method"] == "lat"
                and r["layer_rule"] == "validation-layer"
                and r["source"] == "truthfulqa"
            ),
            "standard": next(r for r in rows if r["method"] == "standard"),
        }
        for name in TEXT:
            hit = [r for r in rows if r["method"] == name]
            if not hit:
                continue
            if name in text and text[name]["mc1"] != hit[0]["mc1"]:
                raise ValueError(f"the {name} row differs between models")
            text[name] = hit[0]
    return found, text


def _short(name: str) -> str:
    return SET_NAMES.get(name, name.replace("_", " "))


def _place(
    fig: Figure, left: float, top: float, width: float, height: float
) -> Axes:
    w, h = fig.get_size_inches()
    return fig.add_axes(
        (left / w, 1 - (top + height) / h, width / w, height / h)
    )


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    store = store_module.Store(manifest.root)
    r6, text = _r6(manifest, store)
    r5_models = manifest.models("r5")
    accuracy: dict[str, float] = {}
    agreement: dict[str, float] = {}
    for model in r5_models:
        run5, run_a1 = manifest.run("r5", model), manifest.run("a1", model)
        if run5 is None or run_a1 is None:
            continue
        accuracy[model] = float(store.summary(run5)["test_accuracy"])
        agreement[model] = float(
            store.summary(run_a1)["agreement_at_layer"]["r5"]
        )
    r7_models = manifest.models("r7")
    r7: dict[str, dict[str, float]] = {}
    for model in r7_models:
        run = manifest.run("r7", model)
        assert run is not None
        sets = store.summary(run)["sets"]
        r7[model] = {
            n: float(v["accuracy_at_val_layer"])
            for n, v in sets.items()
            if n not in NOT_SETS
        }
    names = list(next(iter(r7.values())))
    low = min(t["lo"] for t in text.values())
    high = max(t["hi"] for t in text.values())
    csv_rows: list[list[Any]] = []

    with paper.paper_style():
        fig = Figure(figsize=(paper.FULL_WIDTH, 5.75))
        ax_a = _place(fig, LEFT, 0.65, 2.1, 1.4)
        counts = [params.param_count(m) for m in r6]
        axes.size_axis(ax_a, counts)
        ax_a.xaxis.set_major_locator(ticker.FixedLocator([3e8, 1e9, 3e9]))
        ax_a.axhspan(low, high, color=paper.RANDOM_BAND, lw=0, zorder=1)
        reader = {m: float(p["reader"]["mc1"]) for m, p in r6.items()}
        lo = {m: float(p["reader"]["lo"]) for m, p in r6.items()}
        hi = {m: float(p["reader"]["hi"]) for m, p in r6.items()}
        _group.draw_series(ax_a, reader, paper.SERIES[0], lo=lo, hi=hi)
        for model, parts in r6.items():
            ax_a.plot(
                [params.param_count(model)],
                [parts["standard"]["mc1"]],
                marker="x",
                ms=3.4,
                mew=0.8,
                color=paper.NEUTRALS[0],
                ls="none",
                zorder=3,
            )
            x = params.param_count(model)
            csv_rows.append(
                ["a", model, x, "reader", reader[model], lo[model], hi[model]]
            )
            csv_rows.append(
                [
                    "a",
                    model,
                    x,
                    "standard",
                    parts["standard"]["mc1"],
                    None,
                    None,
                ]
            )
        for name in TEXT:
            csv_rows.append(
                [
                    "a",
                    "all models",
                    None,
                    name,
                    text[name]["mc1"],
                    text[name]["lo"],
                    text[name]["hi"],
                ]
            )
        ax_a.set_ylim(0.1, 0.85)
        ax_a.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax_a.set_ylabel("MC1 on TruthfulQA")
        ax_a.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    ls="none",
                    marker="o",
                    ms=3.2,
                    color=paper.SERIES[0],
                    label="reader",
                ),
                Line2D(
                    [],
                    [],
                    ls="none",
                    marker="x",
                    ms=3.4,
                    mew=0.8,
                    color=paper.NEUTRALS[0],
                    label="log-prob.",
                ),
                Patch(color=paper.RANDOM_BAND, label="text-only"),
            ],
            loc="lower left",
            bbox_to_anchor=(0.0, 1.03),
            ncol=3,
            borderaxespad=0.0,
            columnspacing=0.8,
            handletextpad=0.3,
            handlelength=1.2,
        )
        paper.panel_letter(ax_a, "a")

        ax_b = _place(fig, 3.45, 0.65, 1.85, 1.4)
        for model in accuracy:
            style = axes.model_style(model)
            ax_b.plot(
                [agreement[model]],
                [accuracy[model]],
                marker=style.marker,
                ms=3.6,
                mfc=style.color if style.filled else "white",
                mec=style.color,
                mew=0.8,
                ls="none",
                zorder=3,
            )
            csv_rows.append(
                [
                    "b",
                    model,
                    agreement[model],
                    "honesty_transfer",
                    accuracy[model],
                    None,
                    None,
                ]
            )
        ax_b.axhline(0.5, color=paper.CHANCE, lw=0.5, ls=(0, (3, 2)), zorder=1)
        ax_b.set_xlim(0.97, 0.99)
        ax_b.set_xticks([0.975, 0.98, 0.985])
        ax_b.set_xticklabels(["0.975", "0.98", "0.985"])
        ax_b.set_ylim(0.3, 0.9)
        ax_b.set_yticks([0.4, 0.6, 0.8])
        ax_b.set_xlabel("seed agreement")
        ax_b.set_ylabel("honesty transfer")
        paper.panel_letter(ax_b, "b")

        ax_c = _place(fig, 1.3, 2.75, 4.0, 0.19 * len(r7))
        grid = [[r7[m][n] for n in names] for m in r7]
        ax_c.imshow(grid, cmap=HEAT, vmin=0.0, vmax=1.0, aspect="auto")
        for i, model in enumerate(r7):
            for j, name in enumerate(names):
                ax_c.text(
                    j,
                    i,
                    f"{grid[i][j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                )
                csv_rows.append(
                    ["c", model, name, "accuracy", grid[i][j], None, None]
                )
        ax_c.set_xticks(range(len(names)))
        ax_c.set_xticklabels(
            [_short(n) for n in names], rotation=90, fontsize=7
        )
        ax_c.set_yticks(range(len(r7)))
        ax_c.set_yticklabels([paper.display_name(m) for m in r7], fontsize=7)
        for spine in ax_c.spines.values():
            spine.set_visible(False)
        ax_c.tick_params(length=0)
        paper.panel_letter(ax_c, "c")
        fig.legend(
            handles=_group.role_handles(),
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, 0.0),
            handlelength=2.0,
            columnspacing=1.5,
        )

    inside = sum(low <= float(p["reader"]["mc1"]) <= high for p in r6.values())
    below = sum(r7[m][n] < 0.5 for m in r7 for n in NEGATED if n in r7[m])
    cells = sum(1 for m in r7 for n in NEGATED if n in r7[m])
    numbers = {
        "models": float(len(r6)),
        "inside_text_interval": float(inside),
        "negated_cells": float(cells),
        "negated_below_chance": float(below),
        "honesty_min": min(accuracy.values()),
        "honesty_max": max(accuracy.values()),
        "agreement_min": min(agreement.values()),
    }
    claim = (
        "On TruthfulQA the validation-layer reader reaches an MC1 inside the "
        f"interval of the text-only classifiers on {inside} of {len(r6)} "
        f"models; {below} of {cells} model and negated-set cells fall below "
        "chance accuracy; and honesty transfer accuracy spans "
        f"{numbers['honesty_min']:.3f} to {numbers['honesty_max']:.3f} across "
        "models while the seed agreement stays at "
        f"{numbers['agreement_min']:.3f} or more."
    )
    return FigureOutput(
        fig, claim, DETAIL, numbers, _group.csv_text(HEADER, csv_rows)
    )
