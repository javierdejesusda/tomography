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
import numpy as np

from tomography.site import runfiles
from tomography.site import stats
from tomography.site.figs import _group
from tomography.site.figs import _rows
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput
from tomography.viz import axes
from tomography.viz import paper

HEIGHT = 3.9
HEADER = ["panel", "run", "x", "series", "value", "lo", "hi"]
ARMS = ("random0", "random1", "random2")
OFFSETS = {
    "direction_known": -0.28,
    "direction_unknown": 0.0,
    "random0": 0.22,
    "random1": 0.38,
    "random2": 0.54,
}
EXEMPLARS = (
    ("gpt2-xl", "fewshot"),
    ("qwen3-4b-base", "fewshot"),
    ("qwen3-4b", "fewshot"),
)
CLAIM = (
    "Adding the knowledge direction moves unknown questions more than "
    "random directions do on GPT-2 and base checkpoints, and not "
    "consistently on post-trained ones."
)
DETAIL = (
    "(a) Slope of the next-token entropy shift per unit relative "
    "coefficient on unknown questions (circle) and known questions "
    "(triangle), against 3 random directions of the same norm on unknown "
    "questions (grey), with 95 percent intervals that resample the "
    "questions. (b) The same for the abstention rate of post-trained "
    "checkpoints, with a few-shot and a concept prompt. The direction moves "
    "unknown questions more than each of the 3 random directions on "
    "{distinguishable}, and more than known questions on {selective}, by "
    "the gap between the absolute mean slopes over the coefficient grid, "
    "whatever the sign; with 3 random directions a direction passes by "
    "chance with probability at most 1/4. The outcome is the entropy shift "
    "for GPT-2 and base checkpoints and the abstention rate for "
    "post-trained ones. (c to e) Dose curves "
    "of GPT-2 XL, Qwen3-4B-Base and Qwen3-4B (few-shot prompt) on unknown "
    "and known questions, against the mean over the 3 random directions on "
    "unknown questions with its 95 percent interval that resamples the "
    "questions (grey band); "
    "across all runs the largest shift of the mean curve is as large on "
    "known as on unknown questions for {held}, which a slope does not show. "
    "Every run is in Figure A6, and the post-trained runs by question type "
    "in Figure A7."
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


def _positions(runs: list[Run]) -> list[float]:
    found: list[float] = []
    for i, run in enumerate(runs):
        if i == 0:
            found.append(0.0)
            continue
        step = 1.0
        previous = runs[i - 1]
        if previous.model.startswith("gpt2") and not run.model.startswith(
            "gpt2"
        ):
            step += axes.GAP_AFTER_GPT2
        elif previous.model != run.model:
            step += axes.GAP_BETWEEN_PAIRS
        found.append(found[-1] + step)
    return found


def _forest(
    ax: Axes, runs: list[Run], letter: str, csv_rows: list[list[Any]]
) -> None:
    positions = _positions(runs)
    low, high = 0.0, 0.0
    for run, base in zip(runs, positions, strict=True):
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
                    "o" if series == "direction_unknown" else "^",
                    paper.INK,
                    (float(lo), float(hi)),
                )
            low, high = min(low, float(lo)), max(high, float(hi))
            csv_rows.append([letter, run.label, None, series, value, lo, hi])
    margin = 0.06 * (high - low)
    ax.set_xlim(low - margin, high + margin)
    ax.set_ylim(positions[-1] + 0.9, -0.6)
    ax.set_yticks(positions)
    ax.set_yticklabels([r.name for r in runs])
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
) -> None:
    """Draws known, unknown and pooled random dose curves on a panel.

    Args:
        ax: The panel.
        model: Checkpoint key, which fixes the fill and the line style.
        label: Run label written to the data rows.
        curves: Series names and the stored arrays `x`, `mean`, `lo`, `hi`.
        letter: Panel letter written to the data rows.
        csv_rows: Receives one row per curve point.
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
    for series, marker in (("lat: known", "^"), ("lat: unknown", "o")):
        k = index[series]
        ax.fill_between(
            x,
            arrays["lo"][k],
            arrays["hi"][k],
            color=paper.INK,
            alpha=0.15,
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
            **_rows.fill(model, paper.INK),
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
    ax: Axes, run: Run, letter: str, csv_rows: list[list[Any]]
) -> None:
    """Draws the stored dose curves of one run on a panel.

    Args:
        ax: The panel.
        run: The run.
        letter: Panel letter written to the data rows.
        csv_rows: Receives one row per curve point.
    """
    draw_curves(ax, run.model, run.label, run.curves(), letter, csv_rows)
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
            draw_dose(ax, by_label[" ".join(key)], letter, csv_rows)
            paper.panel_title(ax, letter, by_label[" ".join(key)].name)
        paper.panel_letter(ax_a, "a")
        paper.panel_letter(ax_b, "b")
        legend = top.legend(
            handles=[
                _rows.series_handle(
                    "direction, unknown questions", "o", paper.INK
                ),
                _rows.series_handle(
                    "direction, known questions", "^", paper.INK
                ),
                _rows.series_handle(
                    "random direction", "x", paper.ENC["random_marker"]
                ),
                _rows.series_handle("GPT-2 and base", "h", paper.INK),
                _rows.series_handle(
                    "post-trained", "h", paper.INK, filled=False
                ),
            ],
            loc="outside upper center",
            ncol=3,
            handlelength=1.0,
            columnspacing=1.2,
            fontsize=7,
        )
        legend.get_frame().set_edgecolor("none")

    distinguishable = _rows.run_counts(
        {(r.model, r.label): bool(r.head["distinguishable"]) for r in runs}
    )
    selective = _rows.run_counts(
        {(r.model, r.label): bool(r.head["selective"]) for r in runs}
    )
    held = [
        r
        for r in runs
        if r.largest_shift("lat: known") >= r.largest_shift("lat: unknown")
    ]
    known_ge_unknown = _rows.run_counts(
        {(r.model, r.label): r in held for r in runs}
    )
    numbers: dict[str, float] = {}
    for role in stats.ROLES:
        numbers[f"distinguishable_{role}"] = float(distinguishable[role].hits)
        numbers[f"selective_{role}"] = float(selective[role].hits)
    detail = DETAIL.format(
        distinguishable=_rows.runs_text(distinguishable),
        selective=_rows.runs_text(selective),
        held=_rows.join_and([r.name for r in held]),
    )
    return FigureOutput(
        fig,
        CLAIM,
        detail,
        numbers,
        _group.csv_text(HEADER, csv_rows),
        {
            "distinguishable": distinguishable,
            "selective": selective,
            "known_ge_unknown": known_ge_unknown,
        },
    )
