"""Figure 9: Dose response of pushing along the knowledge direction.

Two forests give the slope of the outcome per unit relative coefficient for
every run, in its own unit: the entropy shift for GPT-2 and base checkpoints
and the abstention rate for post-trained ones. Each row shows the direction on
unknown and on known questions against the three random directions. Three
dose panels show the stored curves of three runs.
"""

from typing import Any

from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
import numpy as np

from tomography.site import runfiles
from tomography.site import stats
from tomography.site.figs import _group
from tomography.site.figs import _rows
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import axes
from tomography.viz import paper

HEIGHT = 4.8
BLOCK_GAP = 0.9
TAG_KEY = (
    "above n of 3: random directions whose gap interval lies above zero; "
    "smallest one-sided p is 1/4"
)
LABEL_FONT = 7
HEADER = ["panel", "run", "x", "series", "value", "lo", "hi"]
ARMS = ("random0", "random1", "random2")
OFFSETS = {
    "direction_known": -0.30,
    "direction_unknown": -0.18,
    "random0": 0.04,
    "random1": 0.16,
    "random2": 0.28,
}
EXEMPLARS = (
    ("gpt2-xl", "fewshot"),
    ("qwen3-4b-base", "fewshot"),
    ("qwen3-4b", "fewshot"),
)
CLAIM = (
    "Adding the knowledge direction moves unknown questions more than each "
    "of 3 random directions of the same norm on {distinguishable}, yet by "
    "the largest shift of the mean dose curve (c to e), which the slopes in "
    "(a, b) do not show, known questions move at least as much as unknown "
    "ones on {held}."
)
DETAIL = (
    "(a) Slope of the entropy shift per unit relative coefficient on "
    "unknown questions (circle) and known questions (left triangle) of "
    "GPT-2 and base checkpoints, against 3 random directions (grey "
    "crosses), with 95 percent intervals that resample the questions. (b) "
    "The same for the abstention rate of post-trained checkpoints "
    "(few-shot and concept prompt); outcomes are never pooled. "
    '"above n of 3" counts the random directions beaten. (c to e) Dose '
    "curves of GPT-2 XL, Qwen3-4B-Base and Qwen3-4B (few-shot) against "
    "the mean of the 3 random directions on unknown questions with its "
    "interval (grey band) and the direction's interval (tint). Rows run by "
    "size; GPT-2 small has no run. All runs: [[fig:a_dose_runs]]; by "
    "question type: [[fig:a_dose_types]]."
)
APPENDIX_DETAIL = (
    "The direction moves unknown questions more than known questions on "
    "[[runs:fig:knowledge_steering.selective]], by the gap between the "
    "absolute mean slopes over the coefficient grid, whatever the sign. A "
    "direction counts as above a random direction when the 95 percent "
    "interval of that gap lies above zero; with 3 random directions a "
    "direction passes by chance with probability at most 1/4. By the "
    "largest shift of the mean dose curve, which a slope does not show, "
    "known questions move at least as much as unknown ones on "
    "[[runs:fig:knowledge_steering.known_ge_unknown]]."
)


class Run:
    """One P9d run and what the figures read from it."""

    def __init__(self, manifest: Manifest, model: str, template: str) -> None:
        """Opens a run.

        Args:
            manifest: The run resolver.
            model: Checkpoint key.
            template: `fewshot` or `concept`.
        """
        path = manifest.run("p9d", model, template=template)
        assert path is not None
        self.model = model
        self.template = template
        self.path = path
        self.head: dict[str, Any] = runfiles.summary(path)["headline"]
        self.entropy = self.head["metric"] == "entropy"
        self.dose = runfiles.rows(path, "table_dose.json")

    @property
    def label(self) -> str:
        """Checkpoint key and template."""
        return f"{self.model} {self.template}"

    @property
    def name(self) -> str:
        """Display name; the template only where it distinguishes runs."""
        if self.entropy:
            return paper.display_name(self.model)
        return paper.display_name(self.label)

    @property
    def block(self) -> str:
        """The first row label: the role group, or the checkpoint."""
        role = stats.role(self.model)
        if role == "gpt2":
            return "GPT-2"
        if role == "base":
            return "Qwen3 base"
        return paper.display_name(self.model)

    @property
    def tick(self) -> str:
        """The second row label: the checkpoint size or the prompt."""
        role = stats.role(self.model)
        name = paper.display_name(self.model)
        if role == "gpt2":
            return name.removeprefix("GPT-2 ")
        if role == "base":
            return name.removeprefix("Qwen3-").removesuffix("-Base")
        return "few-shot" if self.template == "fewshot" else self.template

    @property
    def beats(self) -> int:
        """How many of the 3 random directions the direction is above."""
        return sum(v[1] > 0 for v in self.head["abs_gaps"].values())

    def slope(self, side: str, arm: str) -> dict[str, Any]:
        """The stored slope row of the headline type and metric.

        Args:
            side: `known` or `unknown`.
            arm: `lat0` or a random arm.

        Returns:
            The row with `slope`, `lo` and `hi`.
        """
        (row,) = [
            r
            for r in self.dose
            if r["type"] == self.head["group"]
            and r["metric"] == self.head["metric"]
            and r["side"] == side
            and r["arm"] == arm
            and r["vs"] == ""
        ]
        return row

    def curves(self) -> tuple[list[str], dict[str, np.ndarray]]:
        """The stored dose curves of the headline type.

        Returns:
            The series names and the arrays `x`, `mean`, `lo`, `hi`.
        """
        stem = (
            "curves_obscure_real_entropy"
            if self.entropy
            else "curves_obscure_real"
        )
        names = runfiles.data(self.path, f"{stem}.json")["series"]
        return names, runfiles.arrays(self.path, f"{stem}.npz")

    def largest_shift(self, series: str) -> float:
        """Largest move of a mean curve away from its value at no push.

        Args:
            series: A series name of the stored curves.

        Returns:
            The largest absolute difference.
        """
        names, arrays = self.curves()
        mean = arrays["mean"][names.index(series)]
        origin = float(mean[int(np.argmin(np.abs(arrays["x"])))])
        return float(np.max(np.abs(mean - origin)))


def load_runs(manifest: Manifest) -> list[Run]:
    """Reads every P9d run, few-shot before concept for each checkpoint.

    Args:
        manifest: The run resolver.

    Returns:
        The runs in checkpoint order.
    """
    found: list[Run] = []
    for model in manifest.models("p9d"):
        templates = sorted(
            {str(r.config["template"]) for r in manifest.runs("p9d", model)},
            key=lambda t: ("fewshot", "concept").index(t),
        )
        found.extend(Run(manifest, model, t) for t in templates)
    return found


def direction_keys() -> tuple[list[Any], list[str]]:
    """Builds the key entries of the direction on unknown and known questions.

    Each entry is a filled and an open marker, so one entry carries the role.

    Returns:
        The handle pairs and their labels.
    """

    def pair(marker: str) -> tuple[Line2D, Line2D]:
        return (
            _rows.series_handle("", marker, paper.INK),
            _rows.series_handle("", marker, paper.INK, filled=False),
        )

    handles = [pair("o"), pair(paper.QUESTION_SET_MARKERS["known"])]
    labels = [
        "direction, unknown questions (filled: GPT-2 and base; open: "
        "post-trained)",
        "direction, known questions, same fill",
    ]
    return handles, labels


def _layout(runs: list[Run]) -> tuple[list[Run], list[float]]:
    """Keeps the rows in checkpoint order within each block and places them.

    Args:
        runs: The runs of one forest in checkpoint order.

    Returns:
        The runs in drawing order and their row positions.
    """
    blocks: dict[str, list[Run]] = {}
    for run in runs:
        blocks.setdefault(run.block, []).append(run)
    ordered: list[Run] = []
    positions: list[float] = []
    for rows in blocks.values():
        for i, run in enumerate(rows):
            if not positions:
                positions.append(0.0)
            else:
                positions.append(
                    positions[-1] + 1.0 + (0.0 if i else BLOCK_GAP)
                )
            ordered.append(run)
    return ordered, positions


def _forest(
    ax: Axes, runs: list[Run], letter: str, csv_rows: list[list[Any]]
) -> None:
    runs, positions = _layout(runs)
    low, high = 0.0, 0.0
    seen: list[str] = []
    for run, base in zip(runs, positions, strict=True):
        if run.block not in seen:
            seen.append(run.block)
            ax.text(
                0.01,
                base - 0.75,
                run.block,
                transform=ax.get_yaxis_transform(),
                color=paper.ENC["secondary_text"],
                fontsize=LABEL_FONT,
                va="center",
                ha="left",
                gid="block",
            )
        ax.text(
            1.02,
            base + OFFSETS["direction_unknown"],
            f"above {run.beats} of 3",
            transform=ax.get_yaxis_transform(),
            color=paper.ENC["secondary_text"],
            fontsize=LABEL_FONT,
            va="center",
            ha="left",
            clip_on=False,
            gid="above",
        )
        marks = {
            "direction_unknown": (
                run.head["lat_slope"],
                run.head["lat_lo"],
                run.head["lat_hi"],
            ),
        }
        known = run.slope("known", "lat0")
        marks["direction_known"] = (known["slope"], known["lo"], known["hi"])
        for arm in ARMS:
            row = run.slope("unknown", arm)
            marks[arm] = (row["slope"], row["lo"], row["hi"])
        for series, (value, lo, hi) in marks.items():
            y = base + OFFSETS[series]
            if series.startswith("random"):
                ax.plot(
                    [lo, hi],
                    [y, y],
                    color=paper.ENC["random_marker"],
                    lw=paper.WHISKER_WIDTH,
                    zorder=2,
                )
                ax.plot(
                    [value],
                    [y],
                    marker="x",
                    ls="none",
                    color=paper.ENC["random_marker"],
                    markersize=_rows.MARKER_SIZE,
                    markeredgewidth=0.8,
                    zorder=3,
                )
            else:
                _rows.dot(
                    ax,
                    run.model,
                    y,
                    float(value),
                    paper.QUESTION_SET_MARKERS[
                        "unknown" if series == "direction_unknown" else "known"
                    ],
                    paper.INK,
                    (float(lo), float(hi)),
                )
            low, high = min(low, float(lo)), max(high, float(hi))
            csv_rows.append([letter, run.label, None, series, value, lo, hi])
    margin = 0.06 * (high - low)
    ax.set_xlim(low - margin, high + margin)
    ax.set_ylim(positions[-1] + 0.9, -1.1)
    ax.set_yticks(positions)
    ax.set_yticklabels([r.tick for r in runs])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    axes.ref_line(ax, 0.0, "x")
    ax.xaxis.set_major_formatter(axes.minus_formatter())


def draw_curves(
    ax: Axes,
    model: str,
    label: str,
    curves: tuple[list[str], dict[str, np.ndarray]],
    letter: str,
    csv_rows: list[list[Any]],
    tint: str | None = None,
) -> None:
    """Draws known, unknown and pooled random dose curves on a panel.

    Args:
        ax: The panel.
        model: Checkpoint key, which fixes the fill and the line style.
        label: Run label written to the data rows.
        curves: Series names and the stored arrays `x`, `mean`, `lo`, `hi`.
        letter: Panel letter written to the data rows.
        csv_rows: Receives one row per curve point.
        tint: Colour of the direction intervals; they are a pale ink when None.
    """
    names, arrays = curves
    x = arrays["x"]
    dashed = stats.role(model) == "post"
    style = paper.LINESTYLES["post_trained"] if dashed else "-"
    index = {name: names.index(name) for name in names}
    random = index["random: unknown"]
    ax.fill_between(
        x,
        arrays["lo"][random],
        arrays["hi"][random],
        color=paper.RANDOM_BAND,
        lw=0,
        zorder=1,
    )
    ax.plot(
        x,
        arrays["mean"][random],
        color=paper.ENC["random_marker"],
        lw=paper.DATA_WIDTH,
        zorder=2,
    )
    for series, marker in (
        ("lat: known", paper.QUESTION_SET_MARKERS["known"]),
        ("lat: unknown", paper.QUESTION_SET_MARKERS["unknown"]),
    ):
        k = index[series]
        ax.fill_between(
            x,
            arrays["lo"][k],
            arrays["hi"][k],
            color=tint or paper.INK,
            alpha=0.2 if tint else 0.15,
            lw=0,
            zorder=2,
        )
        ax.plot(
            x,
            arrays["mean"][k],
            color=paper.INK,
            ls=style,
            lw=paper.DATA_WIDTH,
            marker=marker,
            zorder=3,
            **{
                **_rows.fill(model, paper.INK),
                "markersize": _rows.marker_size(marker),
            },
        )
    for series in ("lat: known", "lat: unknown", "random: unknown"):
        k = index[series]
        for j, value in enumerate(arrays["mean"][k]):
            csv_rows.append(
                [
                    letter,
                    label,
                    float(x[j]),
                    series,
                    float(value),
                    float(arrays["lo"][k][j]),
                    float(arrays["hi"][k][j]),
                ]
            )
    ax.set_xlim(-0.44, 0.44)
    ax.xaxis.set_major_locator(ticker.FixedLocator([-0.4, 0.0, 0.4]))
    ax.xaxis.set_major_formatter(axes.minus_formatter())
    ax.yaxis.set_major_formatter(axes.minus_formatter())
    ax.set_xlabel("relative coefficient")


def draw_dose(
    ax: Axes,
    run: Run,
    letter: str,
    csv_rows: list[list[Any]],
    tint: str | None = None,
) -> None:
    """Draws the stored dose curves of one run on a panel.

    Args:
        ax: The panel.
        run: The run.
        letter: Panel letter written to the data rows.
        csv_rows: Receives one row per curve point.
        tint: Colour of the direction intervals; a pale ink when None.
    """
    draw_curves(ax, run.model, run.label, run.curves(), letter, csv_rows, tint)
    ax.set_ylabel("entropy shift" if run.entropy else "abstention rate")
    if run.entropy:
        axes.ref_line(ax, 0.0)


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    runs = load_runs(manifest)
    by_label = {r.label: r for r in runs}
    entropy = [r for r in runs if r.entropy]
    abstention = [r for r in runs if not r.entropy]
    csv_rows: list[list[Any]] = []
    held = [
        r
        for r in runs
        if r.largest_shift("lat: known") >= r.largest_shift("lat: unknown")
    ]

    with paper.paper_style():
        fig = _rows.new_figure(HEIGHT)
        top, bottom = fig.subfigures(
            2, 1, squeeze=False, height_ratios=[2.3, 1.3], hspace=0.0
        )[:, 0]
        ax_a, ax_b = top.subplots(1, 2, squeeze=False)[0]
        doses = bottom.subplots(1, 3, squeeze=False)[0]
        _forest(ax_a, entropy, "a", csv_rows)
        _forest(ax_b, abstention, "b", csv_rows)
        ax_a.set_xlabel("entropy shift per unit\nrelative coefficient")
        ax_b.set_xlabel("abstention rate per unit\nrelative coefficient")
        for ax, letter, key in zip(doses, "cde", EXEMPLARS, strict=True):
            run = by_label[" ".join(key)]
            draw_dose(
                ax, run, letter, csv_rows, axes.model_style(run.model).color
            )
            message = (
                "known moves at least as much"
                if run in held
                else "known moves less"
            )
            paper.panel_title(
                ax, letter, f"{run.name}\n{message}", paper.MIN_TEXT
            )
        low = min(ax.get_ylim()[0] for ax in doses[:2])
        high = max(ax.get_ylim()[1] for ax in doses[:2])
        for ax in doses[:2]:
            ax.set_ylim(low, high)
        doses[1].set_ylabel("")
        doses[1].tick_params(labelleft=False)
        paper.panel_letter(ax_a, "a")
        paper.panel_letter(ax_b, "b")
        keys = direction_keys()
        legend = top.legend(
            handles=[
                *keys[0],
                _rows.series_handle(
                    "random direction", "x", paper.ENC["random_marker"]
                ),
                Line2D(
                    [],
                    [],
                    ls="none",
                ),
            ],
            labels=[*keys[1], "random direction", TAG_KEY],
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0.1)},
            loc="outside upper center",
            ncol=1,
            handlelength=1.6,
            columnspacing=1.2,
            fontsize=paper.TICK,
        )
        legend.get_frame().set_edgecolor("none")

    distinguishable = _rows.run_counts(
        {(r.model, r.label): bool(r.head["distinguishable"]) for r in runs}
    )
    selective = _rows.run_counts(
        {(r.model, r.label): bool(r.head["selective"]) for r in runs}
    )
    known_ge_unknown = _rows.run_counts(
        {(r.model, r.label): r in held for r in runs}
    )
    numbers: dict[str, float] = {}
    for role in stats.ROLES:
        numbers[f"distinguishable_{role}"] = float(distinguishable[role].hits)
        numbers[f"selective_{role}"] = float(selective[role].hits)
    claim = CLAIM.format(
        distinguishable=stats.runs_text(distinguishable),
        held=stats.runs_text(known_ge_unknown),
    )
    return FigureOutput(
        fig,
        claim,
        DETAIL,
        numbers,
        _group.csv_text(HEADER, csv_rows),
        {
            "distinguishable": distinguishable,
            "selective": selective,
            "known_ge_unknown": known_ge_unknown,
        },
    )
