"""Figure 8: A knowledge direction against popularity and confidence.

Panel (a) reads the within-relation AUROC of the known-against-unknown reader
of P9b with its controls and the model's own answer probability, panel (b)
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
    "A known-against-unknown direction ranks questions above the strongest "
    "of popularity and the two text classifiers, with an interval excluding "
    "zero, on {interval}, and adds at most {hi_gpt2} (GPT-2), {hi_base} "
    "(Qwen3 base) and {hi_post} (post-trained) to the confidence of the "
    "model at its interval's upper end."
)
DETAIL = (
    "(a) Within-relation AUROC on PopQA (open markers: post-trained): "
    "direction (circle), popularity (square), range of the two text "
    "classifiers (green bar) and its first answer token probability "
    "(pentagon). GPT-2 {gpt2}; Qwen3 base {base}; Qwen3 post-trained "
    "{post}. (b) Difference in AUROC, direction minus the strongest of "
    "popularity and the two text classifiers, conservative interval. (c) "
    "Change in AUROC for wrong answers when the direction, or a "
    "logistic-regression probe fitted on the correct and wrong labels, is "
    "added to the baseline, model confidence; the interval covers zero on "
    "{cover}. See [[sec:appendix_knowledge]]."
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


def _strongest(row: dict[str, Any]) -> float:
    """Gives the AUROC of the strongest of popularity and the text readers."""
    return max(
        float(row[c])
        for c in ("popularity_within", "text_word_within", "text_char_within")
    )


def _difference(row: dict[str, Any]) -> tuple[float, float, float]:
    return stats.auc_diff_interval(
        float(row["logistic_within"]),
        _strongest(row),
        int(row["n_positive_test"]),
        int(row["n_negative_test"]),
    )


def _range(values: list[float]) -> tuple[str, str]:
    return _rows.number(min(values)), _rows.number(max(values))


def _role_text(
    members: list[str],
    contrast: dict[str, dict[str, Any]],
    own: dict[str, dict[str, Any]],
) -> str:
    """Writes the AUROC ranges of one role beside their controls.

    Args:
        members: The checkpoints of the role that have a contrast.
        contrast: The contrast rows by checkpoint.
        own: The model's own first-token probability rows by checkpoint.

    Returns:
        The direction range against the popularity, text-classifier and
        own-probability ranges.
    """

    def span(values: list[float]) -> str:
        low, high = _range(values)
        return f"{low} to {high}"

    return (
        f"{span([float(contrast[m]['logistic_within']) for m in members])} "
        "against "
        f"{span([float(contrast[m]['popularity_within']) for m in members])}"
        ", "
        + span(
            [
                float(contrast[m][c])
                for m in members
                for c in ("text_word_within", "text_char_within")
            ]
        )
        + " and "
        + span([float(own[m]["auroc_within"]) for m in members])
    )


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


def draw_group_labels(
    ax: Any, rows: dict[str, float], x: float = GROUP_X
) -> None:
    """Writes the role of each block of rows, rotated, left of the labels.

    Args:
        ax: The leftmost panel, whose rows are checkpoints.
        rows: Checkpoint key to row position, from `block_rows`.
        x: Horizontal position in axes fractions.
    """
    by_role = _group.lines_by_role(list(rows))
    for role, members in by_role.items():
        centre = sum(rows[m] for m in members) / len(members)
        ax.text(
            x,
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
    ax.set_xlabel("direction minus\nstrongest control")
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
        _rows.freeze_layout(fig)

    point = stats.per_role(
        {
            m: float(r["logistic_within"]) > _strongest(r)
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
        "above_best_point": point,
        "above_best": interval,
        "dir_cover": cover,
        "probe_above": above,
    }
    numbers: dict[str, float] = {
        "gpt2_small_probe_gain": float(probe["gpt2"]["diff_vs_confidence"]),
    }
    for role in stats.ROLES:
        in_role = [m for m in lat if stats.role(m) == role]
        gains = [float(lat[m]["diff_vs_confidence"]) for m in in_role]
        probe_gains = [
            float(r["diff_vs_confidence"])
            for m, r in probe.items()
            if stats.role(m) == role
        ]
        numbers[f"dir_inc_min_{role}"] = min(gains)
        numbers[f"dir_inc_max_{role}"] = max(gains)
        numbers[f"dir_inc_hi_max_{role}"] = max(
            float(lat[m]["hi"]) for m in in_role
        )
        numbers[f"probe_inc_min_{role}"] = min(probe_gains)
        numbers[f"probe_inc_max_{role}"] = max(probe_gains)
        numbers[f"above_best_point_{role}"] = float(point[role].hits)
        numbers[f"above_best_{role}"] = float(interval[role].hits)
        numbers[f"probe_above_{role}"] = float(above[role].hits)
    by_role = {
        role: _role_text(
            [m for m in contrast if stats.role(m) == role], contrast, own
        )
        for role in stats.ROLES
    }
    claim = CLAIM.format(
        interval=stats.per_role_text(interval),
        **{
            f"hi_{role}": _rows.number(numbers[f"dir_inc_hi_max_{role}"], 3)
            for role in stats.ROLES
        },
    )
    detail = DETAIL.format(cover=stats.per_role_text(cover), **by_role)
    return FigureOutput(
        fig, claim, detail, numbers, _group.csv_text(HEADER, csv_rows), counts
    )
