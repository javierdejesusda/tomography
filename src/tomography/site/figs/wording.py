"""Figure 5: wording moves a reader, and distant directions mostly still read.

Panel (a) is a strip per checkpoint: the accuracy of the sentiment reader
fitted on the paper's wording when it is scored on every other wording, with
the paper wording and the range of a random direction. Panel (b) is a scatter
of every direction fitted on another wording by its cosine with the paper
direction and its accuracy on the paper wording.
"""

from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from tomography.site import runfiles
from tomography.site import stats
from tomography.site.figs import _rows as rowkit
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.site.registry import RoleCount
from tomography.viz import axes as shared
from tomography.viz import paper

HEIGHT = 2.8
TASK = "sentiment"
DROPPED = "no_trailing_space"
MARKER = paper.VARIANT_MARKERS
DODGE = {"frame": -0.17, "wording": 0.0, "concept": 0.17}
KIND = {
    "frame": "frame change",
    "wording": "wording change",
    "concept": "concept swap",
}
SIZE = 3.0
YLIM = (0.45, 1.0)
TEXT = 6.5


def _rows(manifest: Manifest, model: str) -> tuple[list[dict[str, Any]], int]:
    path = manifest.run("p1", model)
    if path is None:
        raise FileNotFoundError(f"results/p1/{model} has no logistic run")
    table = [
        r
        for r in runfiles.rows(path, "table.json")
        if r["task"] == TASK and r["variant"] != DROPPED
    ]
    summary = runfiles.data(path, "summary.json")
    return table, int(summary["n_eval_pairs"][TASK]["report"])


def _size(marker: str, size: float) -> float:
    return rowkit.STAR_SIZE if marker == paper.CATEGORY_MARKERS[0] else size


def _marker(
    ax: Axes,
    x: float,
    y: float,
    family: str,
    model: str,
    *,
    color: str | None = None,
    size: float = SIZE,
    edge: str | None = None,
) -> None:
    style = shared.model_style(model)
    face = color or style.color
    ax.plot(
        [x],
        [y],
        ls="none",
        marker=MARKER[family],
        ms=_size(MARKER[family], size),
        mfc=face if style.filled else "white",
        mec=edge or face,
        mew=0.8,
        zorder=4,
    )


def _strip(
    ax: Axes,
    rows: dict[str, float],
    data: dict[str, list[dict[str, Any]]],
    noise: dict[str, float],
) -> None:
    shared.ref_line(ax, 0.5, "x")
    for model, y in rows.items():
        table = data[model]
        random = [r["random"] for r in table]
        ax.add_patch(
            Rectangle(
                (min(random), y - 0.25),
                max(random) - min(random),
                0.5,
                fc=paper.RANDOM_BAND,
                ec="none",
                zorder=1.5,
            )
        )
        for r in table:
            if r["variant"] == "paper":
                ax.plot(
                    [r["from_paper"]] * 2,
                    [y - 0.4, y + 0.4],
                    color=paper.ENC["paper"],
                    lw=1.4,
                    zorder=5,
                )
            else:
                _marker(
                    ax,
                    r["from_paper"],
                    y + DODGE[r["family"]],
                    r["family"],
                    model,
                    color=paper.ENC["direction"],
                )
        values = [r["from_paper"] for r in table]
        ax.text(
            1.02,
            y,
            f"{max(values) - min(values):.3f}",
            transform=ax.get_yaxis_transform(),
            fontsize=TEXT,
            va="center",
            color=paper.INK,
        )
        ax.text(
            1.2,
            y,
            f"{noise[model]:.3f}",
            transform=ax.get_yaxis_transform(),
            fontsize=TEXT,
            va="center",
            color=paper.ENC["secondary_text"],
        )
    for x, text, color in (
        (1.02, "span", paper.INK),
        (1.2, "noise", paper.ENC["secondary_text"]),
    ):
        ax.text(
            x,
            -0.9,
            text,
            transform=ax.get_yaxis_transform(),
            fontsize=TEXT,
            va="center",
            color=color,
        )
    ax.set_xlim(*YLIM)
    shared.unit_axis(ax, "x")
    ax.set_xlabel("accuracy on the wording scored")
    ax.set_yticks(list(rows.values()), [paper.display_name(m) for m in rows])
    ax.set_ylim(max(rows.values()) + 0.7, -1.3)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def _scatter(ax: Axes, data: dict[str, list[dict[str, Any]]]) -> None:
    shared.ref_line(ax, 0.5, "y")
    labelled = False
    for model, table in data.items():
        style = shared.model_style(model)
        paper_row = next(r for r in table if r["variant"] == "paper")
        for r in table:
            if r["variant"] == "paper":
                continue
            bare = r["variant"] == "bare"
            _marker(
                ax,
                r["cosine_paper"],
                r["to_paper"],
                r["family"],
                model,
                size=SIZE * 1.4 if bare else SIZE,
                edge=paper.INK if bare else None,
            )
            if bare and not labelled:
                ax.annotate(
                    "bare",
                    (r["cosine_paper"], r["to_paper"]),
                    xytext=(5, -7),
                    textcoords="offset points",
                    fontsize=TEXT,
                    color=paper.INK,
                )
                labelled = True
        ax.plot(
            [paper_row["stability"]] * 2,
            [YLIM[0], YLIM[0] + 0.03],
            color=style.color,
            lw=1.2,
            clip_on=False,
            zorder=5,
        )
        ax.plot(
            [0.0, 0.03],
            [paper_row["random"]] * 2,
            color=paper.ENC["random_marker"],
            lw=1.2,
            clip_on=False,
            zorder=5,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(*YLIM)
    shared.unit_axis(ax, "x")
    shared.unit_axis(ax, "y")
    ax.set_xlabel("cosine with paper direction", fontsize=7)
    ax.set_ylabel("accuracy on the paper wording")


def _legend(fig: Figure) -> None:
    handles: list[Any] = [
        Line2D(
            [],
            [],
            ls="none",
            marker=MARKER[f],
            ms=_size(MARKER[f], SIZE),
            color=paper.ENC["direction"],
        )
        for f in MARKER
    ]
    handles.append(Line2D([], [], color=paper.ENC["paper"], lw=1.4))
    handles.append(
        Line2D([], [], color=paper.RANDOM_BAND, lw=5, solid_capstyle="butt")
    )
    handles.append(
        rowkit.series_handle("GPT-2 and base", paper.ROLE_KEY_MARKER, paper.INK)
    )
    handles.append(
        rowkit.series_handle(
            "post-trained", paper.ROLE_KEY_MARKER, paper.INK, filled=False
        )
    )
    fig.legend(
        handles,
        [
            *KIND.values(),
            "paper wording",
            "random direction",
            "GPT-2 and base",
            "post-trained",
        ],
        loc="outside upper center",
        ncol=4,
        frameon=False,
        fontsize=TEXT,
        columnspacing=1.2,
        handletextpad=0.4,
    )


def _numbers(
    data: dict[str, list[dict[str, Any]]], noise: dict[str, float]
) -> dict[str, float]:
    numbers: dict[str, float] = {}
    for role in stats.ROLES:
        spans = [
            max(r["from_paper"] for r in t) - min(r["from_paper"] for r in t)
            for m, t in data.items()
            if stats.role(m) == role
        ]
        if spans:
            numbers[f"span_{role}_min"] = min(spans)
            numbers[f"span_{role}_max"] = max(spans)
    numbers["noise_max"] = max(noise.values())
    bare = {
        m: next(r for r in t if r["variant"] == "bare") for m, t in data.items()
    }
    base = {
        m: next(r for r in t if r["variant"] == "paper")
        for m, t in data.items()
    }
    for name, values in (
        ("bare_cos", [b["cosine_paper"] for b in bare.values()]),
        ("bare_to_paper", [b["to_paper"] for b in bare.values()]),
        ("bare_random", [b["random"] for b in base.values()]),
    ):
        numbers[f"{name}_min"] = min(values)
        numbers[f"{name}_max"] = max(values)
    counts = _counts(data)
    for role, count in counts.items():
        numbers[f"bare_above_{role}"] = count.hits
        numbers[f"bare_n_{role}"] = count.total
    numbers["n_checkpoints"] = len(data)
    return numbers


def _span_text(numbers: dict[str, float]) -> str:
    parts = [
        f"{numbers[f'span_{r}_min']:.3f} to {numbers[f'span_{r}_max']:.3f} "
        f"on {stats.ROLE_NAMES[r]}"
        for r in stats.ROLES
        if f"span_{r}_min" in numbers
    ]
    if len(parts) > 1:
        parts = [", ".join(parts[:-1]) + " and " + parts[-1]]
    return parts[0] + " checkpoints"


def _counts(data: dict[str, list[dict[str, Any]]]) -> dict[str, RoleCount]:
    flags = {}
    for model, table in data.items():
        bare = next(r for r in table if r["variant"] == "bare")
        base = next(r for r in table if r["variant"] == "paper")
        flags[model] = bare["to_paper"] > base["random"]
    return stats.per_role(flags)


def _below(
    data: dict[str, list[dict[str, Any]]],
) -> list[tuple[str, float, float]]:
    found = []
    for model, table in data.items():
        bare = next(r for r in table if r["variant"] == "bare")
        base = next(r for r in table if r["variant"] == "paper")
        if bare["to_paper"] <= base["random"]:
            found.append((model, bare["to_paper"], base["random"]))
    return found


def _claim(
    numbers: dict[str, float],
    counts: dict[str, RoleCount],
    below: list[tuple[str, float, float]],
) -> str:
    exceptions = "".join(
        f"; on {paper.display_name(m)} the bare reader reads {read:.3f} "
        f"against {rand:.3f} for its random direction"
        for m, read, rand in below
    )
    return (
        "Wording moves a reader, and a direction at a large angle from the "
        "paper direction mostly still reads the paper wording above a "
        "random direction. A direction fitted on the bare wording has cosine "
        f"{numbers['bare_cos_min']:.3f} to {numbers['bare_cos_max']:.3f} "
        "with the paper direction. It reads the paper wording above a "
        "random direction by point estimate on "
        f"{stats.per_role_text(counts)}{exceptions}; "
        f"across checkpoints it reads at {numbers['bare_to_paper_min']:.3f} "
        f"to {numbers['bare_to_paper_max']:.3f} and the random direction at "
        f"{numbers['bare_random_min']:.3f} to "
        f"{numbers['bare_random_max']:.3f}."
    )


def _detail(numbers: dict[str, float]) -> str:
    return (
        "(a) Accuracy of the sentiment reader fitted on the paper's wording "
        "when it is scored on each other wording (shape: kind of change), "
        "the paper wording (purple tick) and the range of a random "
        "direction over the wordings (grey); the span between the best and "
        f"the worst wording is {_span_text(numbers)}, against at most "
        f"{numbers['noise_max']:.3f} expected from test-pair noise alone "
        "(right columns). (b) Each direction fitted on another wording, by "
        "its cosine with the paper direction and its accuracy on the paper "
        "wording; ticks on the x axis mark how well the paper direction "
        "agrees with itself across seeds, and grey ticks on the left the "
        "random direction on the paper wording; marker colour darkens "
        "with size "
        "within a family, filled for GPT-2 and Qwen3 base checkpoints and "
        "open for post-trained ones. Matrices for every pair of "
        "wordings are in Figure A4 and the wordings in Table A6."
    )


def _csv(
    data: dict[str, list[dict[str, Any]]],
) -> str:
    lines = [
        "model,role,variant,kind,family,from_paper,to_paper,cosine_paper,"
        "random,stability"
    ]
    for model, table in data.items():
        for r in table:
            stability = r["stability"] if r["variant"] == "paper" else ""
            lines.append(
                f"{model},{stats.role(model)},{r['variant']},{r['kind']},"
                f"{r['family']},{r['from_paper']:.10g},{r['to_paper']:.10g},"
                f"{r['cosine_paper']:.10g},{r['random']:.10g},{stability}"
            )
    return "\n".join(lines) + "\n"


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    models = manifest.models("p1")
    loaded = {m: _rows(manifest, m) for m in models}
    data = {m: t for m, (t, _) in loaded.items()}
    noise = {}
    for m, (table, n) in loaded.items():
        p = sum(r["from_paper"] for r in table) / len(table)
        noise[m] = stats.expected_range(len(table)) * (p * (1 - p) / n) ** 0.5
    rows = shared.model_rows(models)
    numbers = _numbers(data, noise)
    counts = _counts(data)
    with paper.paper_style():
        fig = Figure(figsize=(paper.FULL_WIDTH, HEIGHT), layout="constrained")
        left, right = fig.subplots(
            1, 2, gridspec_kw={"width_ratios": [3.0, 2.3]}
        )
        _strip(left, rows, data, noise)
        _scatter(right, data)
        paper.panel_letter(left, "a")
        paper.panel_letter(right, "b")
        _legend(fig)
    return FigureOutput(
        fig,
        _claim(numbers, counts, _below(data)),
        _detail(numbers),
        numbers,
        _csv(data),
        {"bare_above": counts},
    )
