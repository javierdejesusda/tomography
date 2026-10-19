"""Figure 5: wording moves a reader, and distant directions mostly still read.

Panel (a) has a row per checkpoint: the accuracy of the sentiment reader
fitted on the paper's wording when it is scored on each of the other 15
wordings, as identical circles coloured by three wording families, with the
paper wording as a larger diamond (its rank among the 16 beside it), the
median of the 16, a grey band whose full width is the noise expected from test
pairs alone, and the random direction as a thin tick. Panel (b) has a facet per
role: every direction fitted on another wording by its cosine with the paper
direction and its accuracy on the paper wording.
"""

from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from tomography.site import runfiles
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.site.registry import RoleCount
from tomography.viz import axes as shared
from tomography.viz import paper

HEIGHT = 6.6
TASK = "sentiment"
DROPPED = "no_trailing_space"
FAMILIES = ("frame", "wording", "concept")
LANE = {"frame": -0.34, "wording": 0.0, "concept": 0.34}
LANE_NAMES = {
    "frame": "frame",
    "wording": "paraphrase, scaffold",
    "concept": "concept swap",
}
LANE_X = 0.86
CIRCLE = 3.0
DIAMOND = 5.2
YLIM = (0.45, 1.0)
XLIM = (0.45, 1.16)
SPLIT = [2.7, 1.75]
TEXT = paper.MIN_TEXT
HEAD = 0.95
NAME_GAP = 0.012
ROW_RATIOS = [2.7, 1.0]
COLUMNS = (
    ("paper", 0.0),
    ("noise", 0.2),
    ("span", 0.4),
    ("rank\nof 16", 0.6),
    ("above\nnoise", 0.8),
)
TITLES = {
    "gpt2": "GPT-2",
    "base": "Qwen3 base",
    "post": "Qwen3 post-trained",
}


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


def _name(row: dict[str, Any]) -> str:
    """Names a wording as the table of wordings prints it."""
    base = str(row["kind"]).replace("_", " ")
    name = row["variant"]
    if "_" not in name or name == row["kind"]:
        return base
    return f"{base} {int(name.rsplit('_', 1)[1]) + 1}"


def _rank(table: list[dict[str, Any]]) -> int:
    values = sorted((r["from_paper"] for r in table), reverse=True)
    (row,) = [r for r in table if r["variant"] == "paper"]
    return values.index(row["from_paper"]) + 1


def _tick(
    ax: Axes, name: str, model: str, x: float, y: float, half: float, **kw: Any
) -> None:
    (tick,) = ax.plot([x, x], [y - half, y + half], zorder=3, **kw)
    tick.set_gid(f"{name}:{model}")


def _strip_row(
    ax: Axes, y: float, model: str, table: list[dict[str, Any]], noise: float
) -> None:
    (paper_row,) = [r for r in table if r["variant"] == "paper"]
    centre = paper_row["from_paper"]
    values = sorted(r["from_paper"] for r in table)
    median = (values[7] + values[8]) / 2
    band = Rectangle(
        (centre - noise / 2, y - 0.4),
        noise,
        0.8,
        fc=paper.RANDOM_BAND,
        ec="none",
        zorder=1.5,
    )
    band.set_gid(f"noise:{model}")
    ax.add_patch(band)
    _tick(
        ax,
        "random",
        model,
        paper_row["random"],
        y + 0.32,
        0.14,
        color=paper.ENC["random_marker"],
        lw=1.0,
    )
    _tick(ax, "median", model, median, y, 0.42, color=paper.INK, lw=1.0)
    others = sorted(
        (r for r in table if r["variant"] != "paper"),
        key=lambda r: r["from_paper"],
    )
    for family in FAMILIES:
        members = [r for r in others if r["family"] == family]
        (line,) = ax.plot(
            [r["from_paper"] for r in members],
            [y + LANE[family]] * len(members),
            ls="none",
            marker=paper.MEASURE_MARKERS["direction"],
            ms=CIRCLE,
            mfc=paper.INK,
            mec=paper.INK,
            mew=0.3,
            zorder=4,
        )
        line.set_gid(f"words:{model}:{family}")
    (diamond,) = ax.plot(
        [centre],
        [y],
        ls="none",
        marker=paper.MEASURE_MARKERS["paper_finder"],
        ms=DIAMOND,
        mfc=paper.ENC["paper"],
        mec=paper.INK,
        mew=0.6,
        zorder=6,
    )
    diamond.set_gid(f"paper:{model}")
    for pick, align, sign in (
        (others[-1], "left", 1),
        (others[0], "right", -1),
    ):
        x = pick["from_paper"]
        if abs(x - centre) <= noise / 2:
            continue
        named = ax.text(
            x + sign * NAME_GAP,
            y,
            _name(pick),
            fontsize=TEXT,
            ha=align,
            va="center",
            color=paper.ENC["secondary_text"],
        )
        named.set_gid("named")


def _cell(ax: Axes, gid: str, column: int, y: float, text: str) -> None:
    label = ax.text(
        COLUMNS[column][1],
        y,
        text,
        transform=ax.get_yaxis_transform(),
        fontsize=TEXT,
        va="center",
        color=paper.INK,
    )
    label.set_gid(gid)


def _strip(
    ax: Axes,
    side: Axes,
    rows: dict[str, float],
    data: dict[str, list[dict[str, Any]]],
    noise: dict[str, float],
) -> None:
    for model, y in rows.items():
        table = data[model]
        _strip_row(ax, y, model, table, noise[model])
        (paper_row,) = [r for r in table if r["variant"] == "paper"]
        values = [r["from_paper"] for r in table]
        span = max(values) - min(values)
        _cell(side, "paper_acc", 0, y, f"{paper_row['from_paper']:.3f}")
        _cell(side, "noise_value", 1, y, f"{noise[model]:.3f}")
        _cell(side, "span", 2, y, f"{span:.3f}")
        _cell(side, "rank", 3, y, str(_rank(table)))
        if span > noise[model]:
            _cell(side, "above_noise", 4, y, "yes")
    first = min(rows.values())
    first_qwen = min(y for m, y in rows.items() if m.startswith("qwen3"))
    for text, y in (
        ("GPT-2", first - HEAD),
        ("Qwen3, base then post-trained", first_qwen - HEAD),
    ):
        head = ax.text(
            XLIM[0] + 0.005,
            y,
            text,
            fontsize=TEXT,
            style="italic",
            va="center",
            color=paper.ENC["secondary_text"],
        )
        head.set_gid("block")
    for family in FAMILIES:
        lane = ax.text(
            LANE_X,
            first + LANE[family],
            LANE_NAMES[family],
            fontsize=TEXT,
            va="center",
            color=paper.ENC["secondary_text"],
        )
        lane.set_gid("lane")
    for name, x in COLUMNS:
        head = side.text(
            x,
            first - HEAD,
            name,
            transform=side.get_yaxis_transform(),
            fontsize=TEXT,
            va="center",
            color=paper.ENC["secondary_text"],
        )
        head.set_gid("column")
    ax.set_xlim(*XLIM)
    ax.set_xticks([0.5, 0.75, 1.0], ["0.5", "0.75", "1"])
    ax.spines["bottom"].set_bounds(0.5, 1.0)
    ax.set_xlabel("accuracy of the reader fitted on the paper wording")
    ax.set_yticks(list(rows.values()), [paper.display_name(m) for m in rows])
    ax.set_ylim(max(rows.values()) + 0.7, first - HEAD - 0.4)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    side.tick_params(labelleft=False, left=False)
    side.axis("off")


def _facet(ax: Axes, models: list[str], data: dict[str, Any]) -> None:
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(*YLIM)
    (chance,) = ax.plot(
        ax.get_xlim(),
        [0.5, 0.5],
        color=paper.ENC["reference"],
        ls=paper.LINESTYLES["reference"],
        lw=paper.REFERENCE_WIDTH,
        zorder=1,
    )
    chance.set_gid("chance")
    for model in models:
        style = shared.model_style(model)
        table = data[model]
        (paper_row,) = [r for r in table if r["variant"] == "paper"]
        others = [r for r in table if r["variant"] != "paper"]
        (random_line,) = ax.plot(
            ax.get_xlim(),
            [paper_row["random"]] * 2,
            color=paper.ENC["random_marker"],
            lw=0.6,
            zorder=2,
        )
        random_line.set_gid(f"random_line:{model}")
        (line,) = ax.plot(
            [r["cosine_paper"] for r in others],
            [r["to_paper"] for r in others],
            ls="none",
            marker=paper.MEASURE_MARKERS["direction"],
            ms=CIRCLE,
            mfc=style.color,
            mec=style.color,
            mew=0.4,
            zorder=4,
        )
        line.set_gid(f"cosine:{model}")
        (diamond,) = ax.plot(
            [1.0],
            [paper_row["from_paper"]],
            ls="none",
            marker=paper.MEASURE_MARKERS["paper_finder"],
            ms=DIAMOND,
            mfc=style.color,
            mec=paper.INK,
            mew=0.6,
            zorder=6,
            clip_on=False,
        )
        diamond.set_gid(f"paperdir:{model}")
        ax.plot(
            [paper_row["stability"]] * 2,
            [YLIM[0], YLIM[0] + 0.03],
            color=style.color,
            lw=1.2,
            clip_on=False,
            zorder=5,
        )
    ax.set_xlim(0.0, 1.05)
    ax.set_xticks([0.0, 0.5, 1.0], ["0", "0.5", "1"])
    ax.set_yticks([0.5, 0.75, 1.0], ["0.5", "0.75", "1"])


def _legend(fig: Figure) -> None:
    ink = paper.INK
    grey = paper.ENC["random_marker"]
    handles: list[Any] = [
        Line2D(
            [],
            [],
            ls="none",
            marker=paper.MEASURE_MARKERS["direction"],
            ms=CIRCLE,
            mfc=ink,
            mec=ink,
            mew=0.3,
        ),
        Line2D(
            [],
            [],
            ls="none",
            marker=paper.MEASURE_MARKERS["paper_finder"],
            ms=DIAMOND,
            mfc=paper.ENC["paper"],
            mec=ink,
            mew=0.6,
        ),
        Line2D([], [], ls="none", marker="|", ms=6, mew=1.0, color=ink),
        Line2D([], [], ls="none", marker="|", ms=6, mew=1.0, color=grey),
        Line2D([], [], color=paper.RANDOM_BAND, lw=5, solid_capstyle="butt"),
        Line2D(
            [],
            [],
            ls="none",
            marker="|",
            ms=6,
            mew=1.6,
            color=paper.ENC["qwen3"][1],
        ),
    ]
    fig.legend(
        handles,
        [
            "15 other wordings",
            "paper wording, paper direction",
            "median of the 16 wordings",
            "random direction",
            "test-pair noise",
            "agreement across seeds, one tick per checkpoint (b)",
        ],
        loc="outside upper center",
        ncol=2,
        frameon=False,
        fontsize=TEXT,
        columnspacing=1.0,
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
    for role, count in _span_counts(data, noise).items():
        numbers[f"span_above_noise_{role}"] = count.hits
        numbers[f"span_above_noise_n_{role}"] = count.total
    return numbers


def _span_counts(
    data: dict[str, list[dict[str, Any]]], noise: dict[str, float]
) -> dict[str, RoleCount]:
    flags = {}
    for model, table in data.items():
        values = [r["from_paper"] for r in table]
        flags[model] = max(values) - min(values) > noise[model]
    return stats.per_role(flags)


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


def _rank_numbers(data: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    ranks = [_rank(table) for table in data.values()]
    return {"paper_rank_min": min(ranks), "paper_rank_max": max(ranks)}


def _claim(
    numbers: dict[str, float],
    counts: dict[str, RoleCount],
    below: list[tuple[str, float, float]],
    spans: dict[str, RoleCount],
) -> str:
    exceptions = "".join(
        f"; on {paper.display_name(m)} the bare reader reads {read:.3f} "
        f"against {rand:.3f} for its random direction"
        for m, read, rand in below
    )
    return (
        "Wording moves a reader by more than the noise of the test pairs on "
        f"{stats.per_role_text(spans)} (a point comparison of the span "
        "between the best and the worst wording with the noise expected "
        "from test pairs alone), and a direction at a large angle from the "
        "paper direction mostly still reads the paper wording above a "
        "random direction. The paper wording ranks "
        f"{int(numbers['paper_rank_min'])} to "
        f"{int(numbers['paper_rank_max'])} of the 16 wordings by accuracy. "
        "A direction fitted on the bare wording has cosine "
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
        "(a) One row per checkpoint, base and post-trained checkpoints "
        "kept apart. Accuracy of the sentiment reader fitted on the paper's "
        "wording when it is scored on each of the 15 other wordings: the 15 "
        "circles of a row sit in three lanes by kind of change (frame "
        "change, paraphrase or scaffold, concept swap, top to bottom, named "
        "in the first row); the diamond is the paper wording; the black "
        "tick is the median of the 16 wordings; the thin grey tick is the "
        "random direction; the best and the worst wording are named on the "
        "row unless they fall inside the grey band, with names as in "
        "[[table:wordings]]. The band is centred on the paper wording and "
        "its full width is the noise expected from test pairs alone; it is "
        "conservative, because the wordings share the same test pairs, and "
        "it excludes fitting noise. The right columns give the paper "
        "accuracy, the noise, the span between the best and the worst "
        "wording, the rank of the paper wording among the 16 and a mark "
        f"where the span exceeds the noise; the span is {_span_text(numbers)}"
        f", against at most {numbers['noise_max']:.3f} expected from "
        "test-pair noise alone. (b) For each role, each direction fitted on "
        "another wording, by its cosine with the paper direction and its "
        "accuracy on the paper wording; the diamond at cosine 1 is the paper "
        "direction itself, ticks on the x axis mark how well the paper "
        "direction agrees with itself across seeds, and thin grey lines the "
        "random direction on the paper wording; colour darkens with size "
        "within a family. Matrices for every pair of wordings are in "
        "[[fig:a_wording]]."
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
    numbers.update(_rank_numbers(data))
    counts = _counts(data)
    spans = _span_counts(data, noise)
    with paper.paper_style():
        fig = Figure(figsize=(paper.FULL_WIDTH, HEIGHT), layout="constrained")
        grid = fig.add_gridspec(2, 1, height_ratios=ROW_RATIOS)
        top = grid[0].subgridspec(1, 2, width_ratios=SPLIT)
        bottom = grid[1].subgridspec(1, 3)
        strip = fig.add_subplot(top[0, 0])
        strip.set_gid("wordings")
        side = fig.add_subplot(top[0, 1], sharey=strip)
        side.set_gid("table")
        _strip(strip, side, rows, data, noise)
        paper.panel_letter(strip, "a")
        facets: list[Axes] = []
        for col, role in enumerate(stats.ROLES):
            ax = fig.add_subplot(
                bottom[0, col], sharey=facets[0] if facets else None
            )
            ax.set_gid(f"cosine_{role}")
            mine = [m for m in models if stats.role(m) == role]
            _facet(ax, mine, data)
            ax.set_title(TITLES[role], fontsize=8, loc="center")
            if role == "base":
                ax.set_xlabel("cosine with the paper direction")
            if facets:
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel("accuracy on the paper wording")
                paper.panel_letter(ax, "b")
            facets.append(ax)
        _legend(fig)
    return FigureOutput(
        fig,
        _claim(numbers, counts, _below(data), spans),
        _detail(numbers),
        numbers,
        _csv(data),
        {"bare_above": counts, "span_above_noise": spans},
    )
