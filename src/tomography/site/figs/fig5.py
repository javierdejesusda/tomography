"""Figure 5: A knowledge direction against popularity and confidence.

Panel (a) reads the within-relation AUROC of the known-versus-unknown reader
of P9b, panel (b) the within-relation AUROC for predicting errors of P9c, and
panel (c) the risk-coverage curves of P9c for the largest Qwen3 base model.
"""

import json
from typing import Any

from matplotlib import ticker
from matplotlib.figure import Figure
import numpy as np

from tomography.site import params
from tomography.site import store as store_module
from tomography.site.figs import _group
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import axes
from tomography.viz import paper

COLORS = {
    "reader": paper.SERIES[0],
    "lat": paper.SERIES[0],
    "popularity": paper.SERIES[1],
    "text_word": paper.SERIES[2],
    "probe": paper.SERIES[2],
    "text_char": paper.SERIES[3],
    "answer_logprob": paper.NEUTRALS[0],
}
NAMES = {
    "reader": "reader",
    "lat": "lat",
    "popularity": "popularity",
    "text_word": "text words",
    "text_char": "text chars",
    "probe": "probe",
    "answer_logprob": "answer log-p",
}
PANEL_A = (
    ("reader", "logistic_within"),
    ("popularity", "popularity_within"),
    ("text_word", "text_word_within"),
    ("text_char", "text_char_within"),
)
PANEL_B = ("answer_logprob", "probe", "lat", "popularity")
RISK_SERIES = ("answer_logprob", "lat", "probe")
HEADER = ["panel", "model", "x", "series", "value", "lo", "hi"]
DETAIL = " ".join(
    [
        "(a) Within-relation AUROC of the logistic reader of known against",
        "unknown PopQA questions, with popularity of the subject and two",
        "text-only classifiers as baselines, on the questions each model",
        "labels (the sets differ between models; GPT-2 small has no",
        "known-versus-unknown contrast). The stored interval resamples the",
        "five reader draws, not the test questions, so it is not drawn.",
        "(b) Within-relation AUROC for predicting wrong PopQA answers with",
        "the label-free direction (lat), the supervised probe, the answer",
        "log-probability and popularity; whiskers are 95 percent bootstrap",
        "intervals that resample the test questions; GPT-2 small has no lat.",
        "(c) Risk-coverage on PopQA for {model}: error rate among the",
        "questions answered when only the most confident share is answered;",
        "random order is the error rate at full coverage. Parameter counts",
        "include embeddings. Lines join only checkpoints of one family and",
        "role: filled markers and solid lines are GPT-2 and base models,",
        "open markers and dashed lines are post-trained models.",
    ]
)


def _table(
    store: store_module.Store, manifest: Manifest, exp: str, model: str
) -> list[dict[str, Any]]:
    run = manifest.run(exp, model)
    assert run is not None
    return store.rows(run, "table.json")


def _reader_rows(
    manifest: Manifest, store: store_module.Store
) -> dict[str, dict[str, Any]]:
    found = {}
    for model in manifest.models("p9b"):
        rows = [
            r
            for r in _table(store, manifest, "p9b", model)
            if r["type"] == "obscure_real"
        ]
        if rows:
            found[model] = rows[0]
    return found


def _error_rows(
    manifest: Manifest, store: store_module.Store
) -> dict[str, dict[str, dict[str, Any]]]:
    found = {}
    for model in manifest.models("p9c"):
        rows = _table(store, manifest, "p9c", model)
        found[model] = {
            r["predictor"]: r for r in rows if r["dataset"] == "popqa"
        }
    return found


def _risk_curves(
    manifest: Manifest, model: str
) -> tuple[list[float], dict[str, list[float]]]:
    run = manifest.run("p9c", model)
    assert run is not None
    curves: dict[str, list[float]] = {}
    coverage: list[float] = []
    for stem in ("riskcov_popqa_outputs", "riskcov_popqa_controls"):
        names = json.loads((run / f"{stem}.json").read_text(encoding="utf-8"))
        names = names["data"]["series"]
        with np.load(run / f"{stem}.npz") as arrays:
            coverage = [float(v) for v in arrays["x"]]
            for name in (*RISK_SERIES, "popularity"):
                if name in names and name not in curves:
                    row = names.index(name)
                    curves[name] = [float(v) for v in arrays["mean"][row]]
    return coverage, curves


def _largest_base(models: list[str]) -> str:
    base = [m for m in models if _group.role(m) == "Qwen3 base"]
    return max(base, key=params.param_count)


def _values(rows: dict[str, dict[str, Any]], column: str) -> dict[str, float]:
    return {
        m: float(r[column]) for m, r in rows.items() if r[column] is not None
    }


def _style_size_axis(ax: Any, models: list[str], ylabel: str) -> None:
    axes.size_axis(ax, [params.param_count(m) for m in models])
    ax.xaxis.set_major_locator(ticker.FixedLocator([3e8, 1e9, 3e9]))
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.5, color=paper.CHANCE, lw=0.5, ls=(0, (3, 2)), zorder=1)
    ax.set_ylabel(ylabel)


def _legend(ax: Any, columns: int = 2) -> None:
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.03),
        ncol=columns,
        handlelength=1.3,
        columnspacing=0.6,
        fontsize=7,
        borderaxespad=0.0,
        labelspacing=0.2,
    )


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    store = store_module.Store(manifest.root)
    reader = _reader_rows(manifest, store)
    errors = _error_rows(manifest, store)
    risk_model = _largest_base(list(errors))
    coverage, curves = _risk_curves(manifest, risk_model)
    csv_rows: list[list[Any]] = []

    with paper.paper_style():
        fig = Figure(figsize=(paper.FULL_WIDTH, 3.1))
        fig.subplots_adjust(left=0.09, right=0.965, bottom=0.25, top=0.76)
        grid = fig.subplots(1, 3, gridspec_kw={"wspace": 0.5})
        ax_a, ax_b, ax_c = grid

        _style_size_axis(ax_a, list(reader), "AUROC, known vs unknown")
        for series, column in PANEL_A:
            values = _values(reader, column)
            _group.draw_series(
                ax_a, values, COLORS[series], label=NAMES[series]
            )
            for model, value in values.items():
                csv_rows.append(
                    [
                        "a",
                        model,
                        params.param_count(model),
                        series,
                        value,
                        None,
                        None,
                    ]
                )
        _legend(ax_a)

        _style_size_axis(ax_b, list(errors), "AUROC, error prediction")
        for series in PANEL_B:
            have = {m: v[series] for m, v in errors.items() if series in v}
            values = _values(have, "auroc_within")
            lo = _values(have, "lo_within")
            hi = _values(have, "hi_within")
            _group.draw_series(
                ax_b,
                values,
                COLORS[series],
                lo=lo,
                hi=hi,
                label=NAMES[series],
            )
            for model, value in values.items():
                csv_rows.append(
                    [
                        "b",
                        model,
                        params.param_count(model),
                        series,
                        value,
                        lo[model],
                        hi[model],
                    ]
                )
        _legend(ax_b)

        style = axes.model_style(risk_model)
        base_rate = float(
            errors[risk_model]["answer_logprob"]["n_error"]
        ) / float(errors[risk_model]["answer_logprob"]["n"])
        for name in (*RISK_SERIES, "popularity"):
            ax_c.plot(
                coverage,
                curves[name],
                color=COLORS[name],
                ls=style.linestyle,
                lw=1.0,
                label=NAMES[name],
            )
            for x, value in zip(coverage, curves[name], strict=True):
                csv_rows.append(["c", risk_model, x, name, value, None, None])
        ax_c.axhline(
            base_rate,
            color=paper.CHANCE,
            lw=0.6,
            ls=(0, (3, 2)),
        )
        ax_c.annotate(
            "random order",
            xy=(2, base_rate),
            xytext=(0, 2),
            textcoords="offset points",
            fontsize=7,
            va="bottom",
        )
        for x in coverage[:1]:
            csv_rows.append(
                ["c", risk_model, x, "random_order", base_rate, None, None]
            )
        ax_c.set_xlim(0, 100)
        ax_c.set_ylim(0.0, 1.0)
        ax_c.set_xlabel("coverage (%)")
        ax_c.set_ylabel("error rate among answered")
        _legend(ax_c, 1)

        for ax, letter in zip(grid, "abc", strict=True):
            paper.panel_letter(ax, letter)
        fig.legend(
            handles=_group.role_handles(),
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, 0.0),
            handlelength=2.0,
            columnspacing=1.5,
        )

    above_pop = sum(
        r["logistic_within"] > r["popularity_within"] for r in reader.values()
    )
    above_text = sum(
        r["logistic_within"] > max(r["text_word_within"], r["text_char_within"])
        for r in reader.values()
    )
    with_lat = {m: v for m, v in errors.items() if "lat" in v}
    gaps = [
        v["answer_logprob"]["auroc_within"] - v["lat"]["auroc_within"]
        for v in with_lat.values()
    ]
    numbers = {
        "models_reader": len(reader),
        "reader_above_popularity": above_pop,
        "reader_above_text": above_text,
        "models_lat": len(with_lat),
        "lat_below_logprob": sum(g > 0 for g in gaps),
        "gap_min": min(gaps),
        "gap_max": max(gaps),
    }
    claim = (
        "A direction fitted on known against unknown questions ranks known "
        f"above unknown questions better than popularity on "
        f"{above_pop} of {len(reader)} models and better than both text "
        f"baselines on {above_text}, but as an error predictor the "
        "label-free direction stays below the answer log-probability of the "
        f"model on {numbers['lat_below_logprob']} of {len(with_lat)} models, "
        f"by {min(gaps):.3f} to {max(gaps):.3f} AUROC."
    )
    return FigureOutput(
        fig,
        claim,
        DETAIL.format(model=paper.display_name(risk_model)),
        {k: float(v) for k, v in numbers.items()},
        _group.csv_text(HEADER, csv_rows),
    )
