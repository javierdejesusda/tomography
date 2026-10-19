"""Figure 8: A knowledge direction against popularity and confidence.

Panel (a) reads the within-relation AUROC of the known-against-unknown reader
of P9b with its baselines, panel (b) the paired change in the AUROC for
predicting wrong answers of P9c when the direction or a supervised probe is
added to the model's own confidence. Both panels share the checkpoint rows.
"""

from typing import Any

from matplotlib import ticker

from tomography.site import runfiles
from tomography.site import stats
from tomography.site.figs import _group
from tomography.site.figs import _rows
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import axes
from tomography.viz import paper

HEIGHT = 3.4
Z = stats.Z
HEADER = ["panel", "model", "series", "value", "lo", "hi"]
PANEL_A = (
    ("direction", "logistic_within", "o", paper.INK, -0.3),
    ("popularity", "popularity_within", "s", paper.ENC["popularity"], -0.1),
    ("words", "text_word_within", "^", paper.ENC["text"], 0.1),
    ("characters", "text_char_within", "v", paper.ENC["text"], 0.3),
)
PANEL_B = (
    ("direction", "confidence+lat", "o", -0.2),
    ("probe", "confidence+probe", "D", 0.2),
)
NO_CONTRAST = "no known-against-unknown contrast"
CLAIM = (
    "A known-against-unknown direction ranks questions but adds nothing to "
    "the model's confidence."
)
DETAIL = (
    "(a) AUROC for separating PopQA questions the model answers correctly "
    "from those it does not, within a relation: the knowledge direction "
    "(black), the popularity of the subject (orange, monthly Wikipedia page "
    "views) and classifiers on the words and on the characters of the "
    "question (green), with approximate 95 percent intervals (Hanley and "
    "McNeil) for the direction and popularity. The direction is above "
    "popularity, the strongest baseline, on {point} by point estimate, and "
    "the interval of the difference lies above zero on {interval}. (b) "
    "Change in the AUROC for predicting wrong answers when the direction "
    "(circle) or a supervised probe (diamond) is added to the model's own "
    "confidence, with paired 95 percent bootstrap intervals over the test "
    "questions: the direction changes it by {dmin} to {dmax} with an "
    "interval covering zero on {cover}, while the probe adds {pmin} to "
    "{pmax} with an interval above zero on {above}."
)


def _contrast(manifest: Manifest) -> dict[str, dict[str, Any]]:
    found = {}
    for model in manifest.models("p9b"):
        run = manifest.run("p9b", model)
        assert run is not None
        rows = [
            r
            for r in runfiles.rows(run, "table.json")
            if r["type"] == "obscure_real"
        ]
        if rows:
            found[model] = rows[0]
    return found


def _increments(manifest: Manifest) -> dict[str, dict[str, dict[str, Any]]]:
    found = {}
    for model in manifest.models("p9c"):
        run = manifest.run("p9c", model)
        assert run is not None
        found[model] = {
            r["features"]: r
            for r in runfiles.rows(run, "table_incremental.json")
            if r["dataset"] == "popqa"
        }
    return found


def _interval(row: dict[str, Any], column: str) -> tuple[float, float]:
    value = float(row[column])
    half = Z * stats.hanley_mcneil_se(
        value, int(row["n_positive_test"]), int(row["n_negative_test"])
    )
    return max(0.0, value - half), min(1.0, value + half)


def _difference(row: dict[str, Any]) -> tuple[float, float, float]:
    return stats.auc_diff_interval(
        float(row["logistic_within"]),
        float(row["popularity_within"]),
        int(row["n_positive_test"]),
        int(row["n_negative_test"]),
    )


def _range(values: list[float]) -> tuple[str, str]:
    return _rows.number(min(values)), _rows.number(max(values))


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    models = manifest.models("p9b")
    contrast = _contrast(manifest)
    increments = _increments(manifest)
    rows = axes.model_rows(models)
    csv_rows: list[list[Any]] = []

    with paper.paper_style():
        fig = _rows.new_figure(HEIGHT)
        grid = fig.add_gridspec(1, 2, width_ratios=[2.9, 2.3])
        ax_a = fig.add_subplot(grid[0, 0])
        ax_b = fig.add_subplot(grid[0, 1], sharey=ax_a)
        for model, row in contrast.items():
            for series, column, marker, color, dodge in PANEL_A:
                whisker = (
                    _interval(row, column)
                    if series in ("direction", "popularity")
                    else None
                )
                _rows.dot(
                    ax_a,
                    model,
                    rows[model] + dodge,
                    float(row[column]),
                    marker,
                    color,
                    whisker,
                )
                csv_rows.append(
                    [
                        "a",
                        model,
                        series,
                        row[column],
                        None if whisker is None else whisker[0],
                        None if whisker is None else whisker[1],
                    ]
                )
        for model in models:
            if model not in contrast:
                ax_a.text(
                    0.47,
                    rows[model],
                    NO_CONTRAST,
                    color=paper.ENC["secondary_text"],
                    fontsize=6.5,
                    va="center",
                    ha="left",
                )
        for model, found in increments.items():
            for series, feature, marker, dodge in PANEL_B:
                if feature not in found:
                    continue
                cell = found[feature]
                _rows.dot(
                    ax_b,
                    model,
                    rows[model] + dodge,
                    float(cell["diff_vs_confidence"]),
                    marker,
                    paper.INK,
                    (float(cell["lo"]), float(cell["hi"])),
                )
                csv_rows.append(
                    [
                        "b",
                        model,
                        series,
                        cell["diff_vs_confidence"],
                        cell["lo"],
                        cell["hi"],
                    ]
                )
        _rows.row_axis(ax_a, rows, labels=True)
        _rows.row_axis(ax_b, rows, labels=False)
        ax_a.set_xlim(0.45, 1.0)
        axes.unit_axis(ax_a, "x")
        axes.ref_line(ax_a, 0.5, "x")
        ax_a.set_xlabel("AUROC, known against unknown questions")
        ax_b.set_xlim(-0.035, 0.085)
        ax_b.xaxis.set_major_locator(
            ticker.FixedLocator([-0.02, 0.0, 0.02, 0.04, 0.06, 0.08])
        )
        ax_b.xaxis.set_major_formatter(_rows.tick_formatter())
        axes.ref_line(ax_b, 0.0, "x")
        ax_b.set_xlabel("change in AUROC for\npredicting wrong answers")
        paper.panel_letter(ax_a, "a")
        paper.panel_letter(ax_b, "b")
        legend = fig.legend(
            handles=[
                _rows.series_handle("knowledge direction", "o", paper.INK),
                _rows.series_handle("popularity", "s", paper.ENC["popularity"]),
                _rows.series_handle("words", "^", paper.ENC["text"]),
                _rows.series_handle("characters", "v", paper.ENC["text"]),
                _rows.series_handle("supervised probe", "D", paper.INK),
                _rows.series_handle("GPT-2 and base", "o", paper.INK),
                _rows.series_handle(
                    "post-trained", "o", paper.INK, filled=False
                ),
            ],
            loc="outside upper center",
            ncol=4,
            handlelength=1.0,
            columnspacing=1.2,
            fontsize=7,
        )
        legend.get_frame().set_edgecolor("none")

    point = stats.per_role(
        {
            m: float(r["logistic_within"]) > float(r["popularity_within"])
            for m, r in contrast.items()
        }
    )
    interval = stats.per_role(
        {m: _difference(r)[1] > 0 for m, r in contrast.items()}
    )
    lat = {
        m: f["confidence+lat"]
        for m, f in increments.items()
        if "confidence+lat" in f
    }
    probe = {m: f["confidence+probe"] for m, f in increments.items()}
    cover = stats.per_role(
        {m: float(r["lo"]) <= 0.0 <= float(r["hi"]) for m, r in lat.items()}
    )
    above = stats.per_role({m: float(r["lo"]) > 0.0 for m, r in probe.items()})
    counts = {
        "above_pop_point": point,
        "above_pop": interval,
        "dir_cover": cover,
        "probe_above": above,
    }
    lat_gain = [float(r["diff_vs_confidence"]) for r in lat.values()]
    probe_gain = [float(r["diff_vs_confidence"]) for r in probe.values()]
    numbers: dict[str, float] = {
        "dir_inc_min": min(lat_gain),
        "dir_inc_max": max(lat_gain),
        "dir_inc_n": float(len(lat)),
        "dir_inc_cover_all": float(sum(c.hits for c in cover.values())),
        "probe_inc_min": min(probe_gain),
        "probe_inc_max": max(probe_gain),
    }
    for role in stats.ROLES:
        numbers[f"above_pop_point_{role}"] = float(point[role].hits)
        numbers[f"above_pop_{role}"] = float(interval[role].hits)
        numbers[f"probe_above_{role}"] = float(above[role].hits)
    dmin, dmax = _range(lat_gain)
    pmin, pmax = _range(probe_gain)
    detail = DETAIL.format(
        point=_rows.all_or_per_role(point),
        interval=stats.per_role_text(interval),
        dmin=dmin,
        dmax=dmax,
        cover=_rows.all_or_per_role(cover),
        pmin=pmin,
        pmax=pmax,
        above=_rows.all_or_per_role(above),
    )
    return FigureOutput(
        fig, CLAIM, detail, numbers, _group.csv_text(HEADER, csv_rows), counts
    )
