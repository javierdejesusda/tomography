"""Figure 8: Approximate answers and range readers.

Panel (a) reads the greedy few-shot behavior of P10a on range items, exact
items and the people diagnostic; panels (b) and (c) read the held-out
duration cell and the people position of the concept-template reader of
P10b, fitted on height and wingspan.
"""

import csv
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from tomography.site.figs import _group
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import axes
from tomography.viz import paper

CELL = ("free", "fewshot", "range_last", "greedy")
TRAIN = "height+wingspan"
TEMPLATE = "concept"
LEFT = 0.62
WIDTH = 4.78
HEADER = ["panel", "model", "x", "series", "value", "lo", "hi"]
OFFSETS = {0: -0.22, 1: 0.0, 2: 0.22}
RANGE_COLOR = paper.SERIES[0]
EXACT_COLOR = paper.SERIES[1]
PEOPLE_COLOR = paper.SERIES[2]
RANDOM_COLOR = paper.NEUTRALS[1]
DETAIL = " ".join(
    [
        "(a) Share of greedy few-shot answers written as a range, in the",
        "free condition with the range example last, on the sourced range",
        "items, on the exact items and on the people's heights (a",
        "diagnostic, not a clean contrast); whiskers are 95 percent Wilson",
        "intervals, which resample the items (from table.csv for the range",
        "and exact items, from diagnostic.csv for people). GPT-2 small",
        "copies the format of the last example on every item.",
        "(b) AUROC of separating range from exact items on held-out",
        "durations for the logistic reader of the concept template fitted on",
        "height and wingspan, a random direction at its own layer, and the",
        "text-word classifier; reader and random whiskers resample the",
        "items, and an AUROC of 1.000 has a zero-width interval. (c)",
        "Position of the 120 people's heights between the test buildings (0)",
        "and the test breeds (1) on the reader axis, for the same reader and",
        "a random direction. Models are in size order; the shaded half is",
        "GPT-2; filled markers are GPT-2 and base models, open markers are",
        "post-trained models.",
    ]
)


def _cell(text: str) -> float | str | None:
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return text


def _read(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [
            {k: _cell(v) for k, v in row.items()}
            for row in csv.DictReader(handle)
        ]


def _behavior(manifest: Manifest) -> dict[str, dict[str, dict[str, Any]]]:
    found = {}
    for model in manifest.models("p10a"):
        run = manifest.run("p10a", model)
        assert run is not None
        table = {
            r["kind"]: r
            for r in _read(run / "table.csv")
            if r["family"] == "all"
            and (r["condition"], r["format"], r["shot_order"], r["decoding"])
            == CELL
        }
        (diag,) = [
            r
            for r in _read(run / "diagnostic.csv")
            if (r["condition"], r["format"], r["shot_order"], r["decoding"])
            == CELL
        ]
        found[model] = {
            "range_items": {
                "value": table["range"]["range_rate"],
                "lo": table["range"]["range_lo"],
                "hi": table["range"]["range_hi"],
            },
            "exact_items": {
                "value": table["exact"]["range_rate"],
                "lo": table["exact"]["range_lo"],
                "hi": table["exact"]["range_hi"],
            },
            "people": {
                "value": diag["people_rate"],
                "lo": diag["people_lo"],
                "hi": diag["people_hi"],
            },
        }
    return found


def _reader(manifest: Manifest) -> dict[str, dict[str, dict[str, Any]]]:
    found = {}
    for model in manifest.models("p10b"):
        run = manifest.run("p10b", model)
        assert run is not None
        cell = {
            r["method"]: r
            for r in _read(run / "transfer.csv")
            if r["template"] == TEMPLATE
            and r["train_set"] == TRAIN
            and r["test_family"] == "duration"
        }
        (text,) = [
            r
            for r in _read(run / "baselines.csv")
            if r["baseline"] == "text_word"
            and r["train_set"] == TRAIN
            and r["test_family"] == "duration"
        ]
        people = {
            r["method"]: r
            for r in _read(run / "people.csv")
            if r["source"] == "reader"
            and r["template"] == TEMPLATE
            and r["train_set"] == TRAIN
        }
        found[model] = {
            "reader": {
                "value": cell["logistic"]["auroc"],
                "lo": cell["logistic"]["lo"],
                "hi": cell["logistic"]["hi"],
            },
            "random": {
                "value": cell["random"]["auroc"],
                "lo": cell["random"]["lo"],
                "hi": cell["random"]["hi"],
            },
            "text_word": {"value": text["auroc"], "lo": None, "hi": None},
            "position_reader": {
                "value": people["logistic"]["position"],
                "lo": None,
                "hi": None,
            },
            "position_random": {
                "value": people["random"]["position"],
                "lo": None,
                "hi": None,
            },
        }
    return found


def _positions(models: list[str]) -> dict[str, float]:
    last_gpt2 = max(
        (i for i, m in enumerate(models) if m.startswith("gpt2")), default=-1
    )
    return {
        m: i + (0.8 if i > last_gpt2 >= 0 else 0.0)
        for i, m in enumerate(models)
    }


def _draw(
    ax: Axes,
    letter: str,
    series: dict[str, tuple[str, str, int]],
    data: dict[str, dict[str, dict[str, Any]]],
    pos: dict[str, float],
    csv_rows: list[list[Any]],
) -> None:
    for name, (color, marker_kind, slot) in series.items():
        for model, parts in data.items():
            cell = parts[name]
            style = axes.model_style(model)
            x = pos[model] + OFFSETS[slot]
            if cell["lo"] is not None and cell["hi"] is not None:
                ax.plot(
                    [x, x],
                    [cell["lo"], cell["hi"]],
                    color=color,
                    lw=0.6,
                    zorder=2,
                )
            if marker_kind == "x":
                ax.plot(
                    [x],
                    [cell["value"]],
                    marker="x",
                    ms=3.2,
                    mew=0.8,
                    color=color,
                    ls="none",
                    zorder=3,
                )
            else:
                ax.plot(
                    [x],
                    [cell["value"]],
                    marker=style.marker,
                    ms=3.2,
                    mfc=color if style.filled else "white",
                    mec=color,
                    mew=0.8,
                    ls="none",
                    zorder=3,
                )
            csv_rows.append(
                [
                    letter,
                    model,
                    pos[model],
                    name,
                    cell["value"],
                    cell["lo"],
                    cell["hi"],
                ]
            )


def _frame(ax: Axes, pos: dict[str, float], ylabel: str) -> None:
    models = list(pos)
    gpt2 = [pos[m] for m in models if m.startswith("gpt2")]
    if gpt2:
        ax.axvspan(-0.5, max(gpt2) + 0.5, color="#f3f3f3", lw=0, zorder=0)
    ax.set_xlim(-0.5, max(pos.values()) + 0.5)
    ax.set_xticks([pos[m] for m in models])
    ax.set_xticklabels(
        [paper.display_name(m) for m in models],
        rotation=90,
        fontsize=7,
    )
    ax.set_ylabel(ylabel)


def _place(
    fig: Figure, left: float, top: float, width: float, height: float
) -> Axes:
    w, h = fig.get_size_inches()
    return fig.add_axes(
        (left / w, 1 - (top + height) / h, width / w, height / h)
    )


def _dot(color: str, label: str, marker: str = "o") -> Line2D:
    return Line2D(
        [],
        [],
        ls="none",
        marker=marker,
        ms=3.2,
        mfc=color,
        mec=color,
        mew=0.8,
        label=label,
    )


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    behavior = _behavior(manifest)
    reader = _reader(manifest)
    pos_a = _positions(list(behavior))
    pos_b = _positions(list(reader))
    csv_rows: list[list[Any]] = []

    with paper.paper_style():
        fig = Figure(figsize=(paper.FULL_WIDTH, 5.55))
        ax_a = _place(fig, LEFT, 0.5, WIDTH, 1.2)
        _draw(
            ax_a,
            "a",
            {
                "range_items": (RANGE_COLOR, "o", 0),
                "exact_items": (EXACT_COLOR, "o", 1),
                "people": (PEOPLE_COLOR, "o", 2),
            },
            behavior,
            pos_a,
            csv_rows,
        )
        _frame(ax_a, pos_a, "share written as a range")
        ax_a.set_ylim(0.0, 1.05)
        ax_a.set_yticks([0.0, 0.5, 1.0])
        ax_a.legend(
            handles=[
                _dot(RANGE_COLOR, "range items"),
                _dot(EXACT_COLOR, "exact items"),
                _dot(PEOPLE_COLOR, "people"),
            ],
            loc="lower left",
            bbox_to_anchor=(0.0, 1.03),
            ncol=3,
            borderaxespad=0.0,
            columnspacing=1.0,
            handletextpad=0.3,
        )
        paper.panel_letter(ax_a, "a")

        half = 1.95
        ax_b = _place(fig, LEFT, 3.1, half, 1.15)
        _draw(
            ax_b,
            "b",
            {
                "reader": (RANGE_COLOR, "o", 0),
                "random": (RANDOM_COLOR, "x", 1),
                "text_word": (PEOPLE_COLOR, "o", 2),
            },
            reader,
            pos_b,
            csv_rows,
        )
        _frame(ax_b, pos_b, "AUROC, held-out duration")
        ax_b.set_ylim(0.0, 1.05)
        ax_b.set_yticks([0.0, 0.5, 1.0])
        ax_b.axhline(0.5, color=paper.CHANCE, lw=0.5, ls=(0, (3, 2)), zorder=1)
        ax_b.legend(
            handles=[
                _dot(RANGE_COLOR, "reader"),
                _dot(RANDOM_COLOR, "random", "x"),
                _dot(PEOPLE_COLOR, "text"),
            ],
            loc="lower left",
            bbox_to_anchor=(0.0, 1.03),
            ncol=3,
            borderaxespad=0.0,
            columnspacing=0.8,
            handletextpad=0.3,
        )
        paper.panel_letter(ax_b, "b")

        ax_c = _place(fig, LEFT + half + 0.8, 3.1, half, 1.15)
        _draw(
            ax_c,
            "c",
            {
                "position_reader": (RANGE_COLOR, "o", 0),
                "position_random": (RANDOM_COLOR, "x", 1),
            },
            reader,
            pos_b,
            csv_rows,
        )
        _frame(ax_c, pos_b, "people position")
        ax_c.set_ylim(0.0, 1.6)
        ax_c.set_yticks([0.0, 0.5, 1.0, 1.5])
        for level in (0.0, 1.0):
            ax_c.axhline(
                level, color=paper.CHANCE, lw=0.5, ls=(0, (3, 2)), zorder=1
            )
        ax_c.legend(
            handles=[
                _dot(RANGE_COLOR, "reader"),
                _dot(RANDOM_COLOR, "random", "x"),
            ],
            loc="lower left",
            bbox_to_anchor=(0.0, 1.03),
            ncol=2,
            borderaxespad=0.0,
            columnspacing=0.8,
            handletextpad=0.3,
        )
        paper.panel_letter(ax_c, "c")
        fig.legend(
            handles=_group.role_handles(),
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, 0.0),
            handlelength=2.0,
            columnspacing=1.5,
        )

    qwen = [m for m in behavior if not m.startswith("gpt2")]
    pct = {
        k: [round(100 * behavior[m][k]["value"]) for m in qwen]
        for k in ("range_items", "exact_items", "people")
    }
    text = next(iter(reader.values()))["text_word"]["value"]
    numbers = {
        "qwen_models": float(len(qwen)),
        "range_min": float(min(pct["range_items"])),
        "range_max": float(max(pct["range_items"])),
        "exact_max": float(max(pct["exact_items"])),
        "people_max": float(max(pct["people"])),
        "models": float(len(reader)),
        "reader_above_text": float(
            sum(
                r["reader"]["value"] > r["text_word"]["value"]
                for r in reader.values()
            )
        ),
        "text_word": float(text),
        "random_high": float(
            sum(r["random"]["value"] >= 0.9 for r in reader.values())
        ),
    }
    claim = (
        f"The {len(qwen)} Qwen3 checkpoints write a range on "
        f"{round(numbers['range_min'])}% to {round(numbers['range_max'])}% "
        "of the sourced range items, on at most "
        f"{round(numbers['exact_max'])}% of the exact items and on "
        f"{round(numbers['people_max'])}% of people's heights; on held-out "
        "durations the logistic reader beats the text-word AUROC of "
        f"{text:.3f} on {round(numbers['reader_above_text'])} of "
        f"{len(reader)} checkpoints, but a random direction reaches an AUROC "
        f"of at least 0.9 on {round(numbers['random_high'])}."
    )
    return FigureOutput(
        fig, claim, DETAIL, numbers, _group.csv_text(HEADER, csv_rows)
    )
