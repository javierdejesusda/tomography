"""Figure 8: A knowledge direction against popularity and confidence.

Panel (a) reads the within-relation AUROC of the known-against-unknown reader
of P9b with its baselines and the model's own answer probability, panel (b)
the paired difference between the direction and popularity, panel (c) the
paired change in the AUROC for predicting wrong answers of P9c when the
direction or a supervised probe is added to the model's own confidence. All
panels share the checkpoint rows, which sit in three role blocks.
"""

from typing import Any

from matplotlib import ticker
from matplotlib.lines import Line2D

from tomography.site import runfiles
from tomography.site import stats
from tomography.site.figs import _group
from tomography.site.figs import _rows
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import axes
from tomography.viz import paper

HEIGHT = 4.2
MARKERS = paper.MEASURE_MARKERS
Z = stats.Z
HEADER = ["panel", "model", "series", "value", "lo", "hi"]
BLOCK_GAP = 0.6
HEAD_Y = -0.95
TOP = -1.4
GROUP_X = -0.3
NARROW = 0.035
WIDE = 0.08
RANGE_WIDTH = 2.2
GROUPS = {
    "GPT-2": "GPT-2",
    "Qwen3 base": "Qwen3\nbase",
    "Qwen3 post-trained": "Qwen3\npost-trained",
}
DODGE = {"direction": -0.3, "popularity": -0.1, "own": 0.1, "text": 0.3}
SEPARATOR_WIDTH = 0.5
CLAIM = (
    "A known-against-unknown direction ranks questions but adds at most a "
    "small amount to the model's confidence."
)
DETAIL = (
    "(a) Within-relation AUROC for separating PopQA questions the model "
    "answers correctly from those it does not, with the checkpoints in "
    "three role blocks (filled markers for GPT-2 and Qwen3 base, open "
    "markers for post-trained): the knowledge direction (circle), the "
    "popularity of the subject (square, monthly Wikipedia page views), both "
    "with approximate 95 percent intervals (Hanley and McNeil), the range of "
    "the two text classifiers (green bar, from the classifier on the words "
    "to the one on the characters of the question, without intervals), and "
    "the model's own probability of its first answer token (pentagon), "
    "which is {own_min} to {own_max} against {dir_min} to {dir_max} for the "
    "direction. The model's own confidence in (c) stacks five output "
    "signals, of which this is one, and only the pooled level of the stack "
    "is stored, so the single signal stands for it here. No random "
    "direction is drawn in (a) because its within-relation level is not "
    "stored. (b) The difference between the two AUROCs, direction minus "
    "popularity, the strongest baseline, with the conservative interval "
    "from the two Hanley and McNeil standard errors: the direction is above "
    "popularity on {point} by point estimate, and the interval of the "
    "difference lies above zero on {interval}. (c) Change in the AUROC for "
    "predicting wrong answers when the direction (narrow axis) or a "
    "supervised probe (wide axis) is added to the model's own confidence, "
    "with paired 95 percent bootstrap intervals over the test questions: "
    "the direction changes it by {dmin} to {dmax} with an interval covering "
    "zero on {cover}, while the probe adds {pmin} to {pmax} with an "
    "interval above zero on {above}. The largest upper bound of the "
    "direction's interval is +{hi_max} ({hi_model}). The probe is fitted on "
    "the correct and wrong labels of the training questions; the direction "
    "is not. GPT-2 small has no known-against-unknown contrast, so it has no "
    "row; its probe adds {g_gain} [{g_lo}, {g_hi}]."
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


def _own_probability(manifest: Manifest) -> dict[str, dict[str, Any]]:
    found = {}
    for model in manifest.models("p9c"):
        run = manifest.run("p9c", model)
        assert run is not None
        (row,) = [
            r
            for r in runfiles.rows(run, "table.json")
            if r["dataset"] == "popqa" and r["predictor"] == "first_token_prob"
        ]
        found[model] = row
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


def row_label(model: str) -> str:
    """Gives the short row label of a checkpoint inside its role block.

    Args:
        model: Checkpoint key.

    Returns:
        The size name: `medium`, `XL`, `0.6B` and so on.
    """
    name = paper.display_name(model)
    if model.startswith("gpt2"):
        return name.removeprefix("GPT-2 ")
    return name.split("-")[1]


def block_rows(models: list[str]) -> tuple[dict[str, float], list[float]]:
    """Places checkpoints in role blocks, one row each.

    Args:
        models: Checkpoint keys.

    Returns:
        The row of every checkpoint, and the positions of the thin
        separators between blocks.
    """
    rows: dict[str, float] = {}
    separators: list[float] = []
    position = 0.0
    for members in _group.lines_by_role(models).values():
        if rows:
            separators.append(position - 0.5 - BLOCK_GAP / 2)
        for model in members:
            rows[model] = position
            position += 1.0
        position += BLOCK_GAP
    return rows, separators


def _unit_step(low: float, high: float) -> list[float]:
    step = 0.2 if high - low > 0.5 else 0.1
    first = -(-low // step) * step
    values = []
    value = first
    while value <= high + 1e-9:
        values.append(round(value, 10))
        value += step
    return values


def draw_group_labels(ax: Any, rows: dict[str, float]) -> None:
    """Writes the role of each block of rows, rotated, left of the labels.

    Args:
        ax: The leftmost panel, whose rows are checkpoints.
        rows: Checkpoint key to row position, from `block_rows`.
    """
    by_role = _group.lines_by_role(list(rows))
    for role, members in by_role.items():
        centre = sum(rows[m] for m in members) / len(members)
        ax.text(
            GROUP_X,
            centre,
            GROUPS[role],
            transform=ax.get_yaxis_transform(),
            color=paper.ENC["secondary_text"],
            fontsize=paper.MIN_TEXT,
            ha="center",
            va="center",
            rotation=90,
            linespacing=0.95,
            clip_on=False,
        )


def _row_setup(
    panels: list[Any],
    rows: dict[str, float],
    separators: list[float],
) -> None:
    for position, ax in enumerate(panels):
        _rows.row_axis(ax, rows, labels=position == 0)
        ax.set_ylim(max(rows.values()) + 0.6, TOP)
        for y in separators:
            ax.axhline(
                y,
                color=paper.RANDOM_BAND,
                lw=SEPARATOR_WIDTH,
                zorder=0,
            )
    first = -0.5
    low, high = panels[0].get_ylim()
    for ax in panels:
        for line in ax.get_lines():
            if line.get_color() == paper.ENC["reference"]:
                line.set_ydata([0.0, (first - low) / (high - low)])
    panels[0].set_yticklabels([row_label(m) for m in rows])
    draw_group_labels(panels[0], rows)


def _head(ax: Any, x: float, text: str) -> None:
    ax.text(
        x,
        HEAD_Y,
        text,
        transform=ax.get_yaxis_transform(),
        color=paper.ENC["secondary_text"],
        fontsize=paper.MIN_TEXT,
        ha="center",
        va="center",
        linespacing=0.95,
    )


def _draw_levels(
    ax: Any,
    contrast: dict[str, dict[str, Any]],
    own: dict[str, dict[str, Any]],
    rows: dict[str, float],
    csv_rows: list[list[Any]],
) -> None:
    for model, row in contrast.items():
        y = rows[model]
        for series, column, marker, color in (
            ("direction", "logistic_within", MARKERS["direction"], paper.INK),
            (
                "popularity",
                "popularity_within",
                MARKERS["popularity"],
                paper.ENC["popularity"],
            ),
        ):
            whisker = _interval(row, column)
            _rows.dot(
                ax,
                model,
                y + DODGE[series],
                float(row[column]),
                marker,
                color,
                whisker,
            )
            csv_rows.append(
                ["a", model, series, row[column], whisker[0], whisker[1]]
            )
        text = [float(row["text_word_within"]), float(row["text_char_within"])]
        ax.plot(
            [min(text), max(text)],
            [y + DODGE["text"]] * 2,
            color=paper.TEXT_GREEN,
            lw=RANGE_WIDTH,
            solid_capstyle="projecting",
            zorder=2,
        )
        for series, column in (
            ("words", "text_word_within"),
            ("characters", "text_char_within"),
        ):
            csv_rows.append(["a", model, series, row[column], None, None])
        cell = own[model]
        _rows.dot(
            ax,
            model,
            y + DODGE["own"],
            float(cell["auroc_within"]),
            MARKERS["answer_probability"],
            paper.ENC["own_probability"],
            (float(cell["lo_within"]), float(cell["hi_within"])),
        )
        csv_rows.append(
            [
                "a",
                model,
                "answer_probability",
                cell["auroc_within"],
                cell["lo_within"],
                cell["hi_within"],
            ]
        )
    ax.set_xlim(0.45, 1.0)
    axes.unit_axis(ax, "x")
    axes.ref_line(ax, 0.5, "x")
    ax.set_xlabel("AUROC, known against\nunknown (within relation)")


def _draw_difference(
    ax: Any,
    contrast: dict[str, dict[str, Any]],
    rows: dict[str, float],
    csv_rows: list[list[Any]],
) -> None:
    found = {m: _difference(r) for m, r in contrast.items()}
    for model, (diff, lo, hi) in found.items():
        _rows.dot(
            ax,
            model,
            rows[model],
            diff,
            MARKERS["direction"],
            paper.INK,
            (lo, hi),
        )
        csv_rows.append(["b", model, "difference", diff, lo, hi])
    low = min(lo for _, lo, _ in found.values()) - 0.03
    high = max(hi for _, _, hi in found.values()) + 0.03
    ax.set_xlim(low, high)
    ax.xaxis.set_major_locator(ticker.FixedLocator(_unit_step(low, high)))
    ax.xaxis.set_major_formatter(axes.minus_formatter())
    axes.ref_line(ax, 0.0, "x")
    ax.set_xlabel("direction minus\npopularity")
    _head(ax, 0.5, "difference")


def _draw_increments(
    narrow: Any,
    wide: Any,
    increments: dict[str, dict[str, dict[str, Any]]],
    rows: dict[str, float],
    csv_rows: list[list[Any]],
) -> None:
    for model, found in increments.items():
        for series, feature, marker, ax in (
            ("direction", "confidence+lat", MARKERS["direction"], narrow),
            ("probe", "confidence+probe", MARKERS["probe"], wide),
        ):
            if feature not in found:
                continue
            cell = found[feature]
            csv_rows.append(
                [
                    "c",
                    model,
                    series,
                    cell["diff_vs_confidence"],
                    cell["lo"],
                    cell["hi"],
                ]
            )
            if model not in rows:
                continue
            _rows.dot(
                ax,
                model,
                rows[model],
                float(cell["diff_vs_confidence"]),
                marker,
                paper.INK,
                (float(cell["lo"]), float(cell["hi"])),
            )
    narrow.set_xlim(-NARROW, NARROW)
    narrow.xaxis.set_major_locator(ticker.FixedLocator([-0.03, 0.0, 0.03]))
    narrow.xaxis.set_major_formatter(axes.minus_formatter())
    axes.ref_line(narrow, 0.0, "x")
    wide.set_xlim(0.0, WIDE)
    wide.xaxis.set_major_locator(ticker.FixedLocator([0.0, 0.04, 0.08]))
    wide.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    narrow.set_xlabel("change in\nAUROC")
    wide.set_xlabel("change in\nAUROC")
    _head(narrow, 0.5, "direction\nadded")
    _head(wide, 0.5, "probe\nadded")


def _legend(fig: Any) -> None:
    legend = fig.legend(
        handles=[
            _rows.series_handle(
                "knowledge direction", MARKERS["direction"], paper.INK
            ),
            _rows.series_handle(
                "popularity", MARKERS["popularity"], paper.ENC["popularity"]
            ),
            Line2D(
                [],
                [],
                color=paper.TEXT_GREEN,
                lw=RANGE_WIDTH,
                solid_capstyle="projecting",
                label="text classifiers",
            ),
            _rows.series_handle(
                "own answer probability",
                MARKERS["answer_probability"],
                paper.ENC["own_probability"],
            ),
        ],
        loc="outside upper center",
        ncol=4,
        handlelength=1.2,
        columnspacing=1.8,
        fontsize=7,
    )
    legend.get_frame().set_edgecolor("none")


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    contrast = _contrast(manifest)
    increments = _increments(manifest)
    own = _own_probability(manifest)
    rows, separators = block_rows(list(contrast))
    csv_rows: list[list[Any]] = []

    with paper.paper_style():
        fig = _rows.new_figure(HEIGHT)
        grid = fig.add_gridspec(1, 4, width_ratios=[1.9, 1.0, 0.85, 0.85])
        ax_a = fig.add_subplot(grid[0, 0])
        panels = [ax_a] + [
            fig.add_subplot(grid[0, i], sharey=ax_a) for i in range(1, 4)
        ]
        _draw_levels(ax_a, contrast, own, rows, csv_rows)
        _draw_difference(panels[1], contrast, rows, csv_rows)
        _draw_increments(panels[2], panels[3], increments, rows, csv_rows)
        _row_setup(panels, rows, separators)
        for ax, letter in ((ax_a, "a"), (panels[1], "b"), (panels[2], "c")):
            paper.panel_letter(ax, letter)
        _legend(fig)

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
    widest = max(lat, key=lambda m: float(lat[m]["hi"]))
    probe_gain = [float(r["diff_vs_confidence"]) for r in probe.values()]
    small = probe["gpt2"]
    numbers: dict[str, float] = {
        "dir_inc_min": min(lat_gain),
        "dir_inc_max": max(lat_gain),
        "dir_inc_n": float(len(lat)),
        "dir_inc_hi_max": float(lat[widest]["hi"]),
        "dir_inc_cover_all": float(sum(c.hits for c in cover.values())),
        "probe_inc_min": min(probe_gain),
        "probe_inc_max": max(probe_gain),
        "gpt2_small_probe_gain": float(small["diff_vs_confidence"]),
    }
    for role in stats.ROLES:
        numbers[f"above_pop_point_{role}"] = float(point[role].hits)
        numbers[f"above_pop_{role}"] = float(interval[role].hits)
        numbers[f"probe_above_{role}"] = float(above[role].hits)
    dmin, dmax = _range(lat_gain)
    pmin, pmax = _range(probe_gain)
    own_min, own_max = _range([float(own[m]["auroc_within"]) for m in contrast])
    dir_min, dir_max = _range(
        [float(r["logistic_within"]) for r in contrast.values()]
    )
    detail = DETAIL.format(
        own_min=own_min,
        own_max=own_max,
        dir_min=dir_min,
        dir_max=dir_max,
        point=stats.per_role_text(point),
        interval=stats.per_role_text(interval),
        dmin=dmin,
        dmax=dmax,
        cover=stats.per_role_text(cover),
        pmin=pmin,
        pmax=pmax,
        hi_max=f"{float(lat[widest]['hi']):.3f}",
        hi_model=paper.display_name(widest),
        above=stats.per_role_text(above),
        g_gain=f"{float(small['diff_vs_confidence']):.3f}",
        g_lo=f"{float(small['lo']):.3f}",
        g_hi=f"{float(small['hi']):.3f}",
    )
    return FigureOutput(
        fig, CLAIM, detail, numbers, _group.csv_text(HEADER, csv_rows), counts
    )
