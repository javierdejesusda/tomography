import itertools
from pathlib import Path
import runpy
from typing import Any

import matplotlib
import matplotlib.colors
import matplotlib.text
import numpy as np
import pytest

from tomography.site import registry
from tomography.site import stats
from tomography.site.figs import a_wording
from tomography.site.figs import wording
from tomography.site.manifest import Manifest
from tomography.viz import checks
from tomography.viz import paper
from tomography.viz.axes import MINUS

_RESULTS = runpy.run_path(str(Path(__file__).with_name("_results.py")))
data = _RESULTS["data"]
models = _RESULTS["models"]
run = _RESULTS["run"]

RESULTS = Path(__file__).resolve().parents[1] / "results"
MODELS = models("p1")
ROLES = ("gpt2", "base", "post")
DROPPED = "no_trailing_space"
EM = chr(0x2014)
EN = chr(0x2013)


def _role(model: str) -> str:
    if not model.startswith("qwen3"):
        return "gpt2"
    return "base" if model.endswith("-base") else "post"


def _names() -> list[str]:
    labels = data(run("p1", MODELS[0]), "transfer_sentiment.json")["rows"]
    return [v for v in labels if v != DROPPED]


def _margin(model: str) -> np.ndarray:
    path = run("p1", model)
    labels = data(path, "transfer_sentiment.json")["rows"]
    with np.load(path / "transfer_sentiment.npz") as z:
        values = np.asarray(z["values"], dtype=float)
    keep = [labels.index(n) for n in _names()]
    table = data(path, "table.json")
    random = {
        r["variant"]: r["random"] for r in table if r["task"] == "sentiment"
    }
    scored = np.array([random[n] for n in _names()])
    return values[np.ix_(keep, keep)] - scored[None, :]


ORDER = [
    "qwen3-0.6b-base",
    "qwen3-0.6b",
    "qwen3-1.7b-base",
    "qwen3-1.7b",
    "qwen3-4b-base",
    "qwen3-4b",
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
]
KINDS = {
    "paper": "paper",
    "paraphrase": "paraphrase",
    "scaffold": "scaffold",
    "concept_swap": "concept swap",
}


def _noise(model: str) -> float:
    return wording.noise(Manifest(RESULTS), model)


def _kind_labels() -> list[str]:
    table = data(run("p1", MODELS[0]), "table.json")
    kind = {r["variant"]: r["kind"] for r in table if r["task"] == "sentiment"}
    return [KINDS.get(kind[n], "frame") for n in _names()]


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure("a_wording", Manifest(RESULTS))


def _matrices(output: registry.FigureOutput) -> list:
    return [ax for ax in output.figure.axes if ax.images]


def test_one_matrix_per_checkpoint_within_the_appendix_budget(
    output: registry.FigureOutput,
) -> None:
    axes = _matrices(output)
    assert len(axes) == len(MODELS)
    assert len(output.figure.axes) == len(MODELS) + 1
    width, height = output.figure.get_size_inches()
    assert width == pytest.approx(paper.FULL_WIDTH)
    assert height <= 9.5 + 1e-9


def test_every_cell_is_the_transfer_accuracy_minus_the_scored_random(
    output: registry.FigureOutput,
) -> None:
    for ax, model in zip(_matrices(output), ORDER, strict=True):
        array = ax.images[0].get_array()
        shown = np.ma.getdata(array).astype(float)
        assert shown.shape == (len(_names()),) * 2
        assert shown == pytest.approx(_margin(model), abs=1e-9)
        assert ax.images[0].get_clim() == (-0.15, 0.45)


def test_the_minima_and_the_below_count_are_recomputed(
    output: registry.FigureOutput,
) -> None:
    n = output.numbers
    minima = {m: float(_margin(m).min()) for m in MODELS}
    for model, value in minima.items():
        assert n[f"min_{model}"] == pytest.approx(value)
    for role in ROLES:
        group = [v for m, v in minima.items() if _role(m) == role]
        assert n[f"min_{role}_lo"] == pytest.approx(min(group))
        assert n[f"min_{role}_hi"] == pytest.approx(max(group))
    assert n["n_below"] == sum(int((_margin(m) <= 0).sum()) for m in MODELS)


def test_the_caption_quotes_the_minima_or_the_cells_at_or_below_zero(
    output: registry.FigureOutput,
) -> None:
    text = f"{output.claim} {output.detail}"
    n = output.numbers
    if n["n_below"] == 0:
        assert "Every cell is above its random direction" in text
        for role in ROLES:
            assert (
                f"{n[f'min_{role}_lo']:.3f} to {n[f'min_{role}_hi']:.3f}"
                in text
            )
    else:
        assert f"at or below zero in {int(n['n_below'])} cells" in text
    assert "[[table:wordings]]" in text
    assert "purple frame" in text
    assert "stippled" in text
    for bad in (EM, EN, "_"):
        assert bad not in text


def test_both_caption_branches_follow_the_data() -> None:
    names = ["gpt2", "qwen3-0.6b-base", "qwen3-0.6b"]
    above = {m: np.full((3, 3), 0.2) + i / 100 for i, m in enumerate(names)}
    numbers, sentence = a_wording.summarize(above)
    assert numbers["n_below"] == 0
    assert "Every cell is above its random direction" in sentence
    assert "0.200 to 0.200 on GPT-2" in sentence
    assert "0.210 to 0.210 on Qwen3 base" in sentence
    assert "0.220 to 0.220 on post-trained" in sentence
    below = {m: v.copy() for m, v in above.items()}
    below["qwen3-0.6b"][0, 1] = -0.01
    below["qwen3-0.6b"][2, 2] = 0.0
    numbers, sentence = a_wording.summarize(below)
    assert numbers["n_below"] == 2
    assert numbers["min_qwen3-0.6b"] == pytest.approx(-0.01)
    assert "at or below zero in 2 cells" in sentence
    assert (
        stats.per_role_text(
            stats.per_role({m: bool((v <= 0).any()) for m, v in below.items()})
        )
        in sentence
    )
    assert "Every cell" not in sentence


def test_the_guards_are_green(output: registry.FigureOutput) -> None:
    assert checks.guard_problems(output.figure, appendix=True) == []
    assert checks.problems(output.figure) == []


def test_the_csv_holds_every_cell(output: registry.FigureOutput) -> None:
    lines = output.csv.strip().splitlines()
    head = lines[0].split(",")
    assert head == ["model", "role", "fit", "score", "margin"]
    assert len(lines) - 1 == len(MODELS) * len(_names()) ** 2
    first = dict(zip(head, lines[1].split(","), strict=True))
    assert first["model"] == ORDER[0]
    assert float(first["margin"]) == pytest.approx(
        _margin(ORDER[0])[0, 0], abs=1e-6
    )


def _titles(output: registry.FigureOutput) -> list[str]:
    return [
        ax.get_title() or ax.get_title(loc="left") for ax in _matrices(output)
    ]


def test_panels_pair_base_and_post_trained_then_the_gpt2_sizes(
    output: registry.FigureOutput,
) -> None:
    axes = _matrices(output)
    assert _titles(output) == [paper.display_name(m) for m in ORDER]
    tops = [round(ax.get_position().y1, 3) for ax in axes]
    lefts = [round(ax.get_position().x0, 3) for ax in axes]
    for i in range(0, len(ORDER), 2):
        assert tops[i] == tops[i + 1]
        assert lefts[i] < lefts[i + 1]


def test_the_map_is_cividis_unreversed_with_the_diagonal_masked_grey(
    output: registry.FigureOutput,
) -> None:
    for ax in _matrices(output):
        image = ax.images[0]
        assert image.cmap.name == "cividis"
        mask = np.ma.getmaskarray(image.get_array())
        assert np.array_equal(mask, np.eye(mask.shape[0], dtype=bool))
        bad = image.cmap.get_bad()
        assert matplotlib.colors.to_hex(bad) == paper.ENC["random_band"]


def test_cells_within_the_noise_are_stippled(
    output: registry.FigureOutput,
) -> None:
    total = 0
    for ax, model in zip(_matrices(output), ORDER, strict=True):
        margin = _margin(model)
        off = ~np.eye(margin.shape[0], dtype=bool)
        low = (margin <= _noise(model)) & off
        total += int(low.sum())
        patches = [p for p in ax.patches if p.get_gid() == "stipple"]
        assert len(patches) == int(low.sum())
        assert all(p.get_hatch() for p in patches)
        assert not [t for t in ax.texts if t.get_gid() == paper.TILE_LABEL]
    assert output.numbers["n_below_noise"] == total


def test_every_cell_at_or_below_zero_carries_one_dot(
    output: registry.FigureOutput,
) -> None:
    total = 0
    for ax, model in zip(_matrices(output), ORDER, strict=True):
        margin = _margin(model)
        off = ~np.eye(margin.shape[0], dtype=bool)
        zero = np.argwhere((margin <= 0.0) & off)
        total += len(zero)
        dots = [ln for ln in ax.get_lines() if ln.get_gid() == "zero_dot"]
        found = sorted(
            (round(float(ln.get_ydata()[0])), round(float(ln.get_xdata()[0])))
            for ln in dots
        )
        assert found == sorted((int(i), int(j)) for i, j in zero)
        for ln in dots:
            assert ln.get_marker() == "o"
            assert ln.get_markerfacecolor() == "white"
            assert ln.get_markeredgecolor() == paper.INK
    assert output.numbers["n_below"] == total
    low = min(_margin(m)[~np.eye(16, dtype=bool)].min() for m in ORDER)
    assert output.numbers["min_margin"] == pytest.approx(low)
    text = output.detail
    shown = chr(0x2212) + f"{abs(low):.2f}" if low < 0 else f"{low:.2f}"
    assert f"the {total} cells at or below zero (all on" in text
    assert f"lowest {shown}" in text
    assert "carry a dot" in text
    assert "annotated with their margin" not in text


def test_the_paper_row_and_column_are_framed_like_the_paper_wording(
    output: registry.FigureOutput,
) -> None:
    k = len(_names())
    index = _names().index("paper")
    for ax in _matrices(output):
        frames = [
            p for p in ax.patches if (p.get_gid() or "").startswith("paper_")
        ]
        assert sorted(p.get_gid() for p in frames) == ["paper_col", "paper_row"]
        for p in frames:
            assert not p.get_fill()
            assert matplotlib.colors.to_hex(p.get_edgecolor()) == (
                paper.ENC["paper"].lower()
            )
        row = next(p for p in frames if p.get_gid() == "paper_row")
        assert (row.get_x(), row.get_y()) == (-0.5, index - 0.5)
        assert (row.get_width(), row.get_height()) == (k, 1)
        col = next(p for p in frames if p.get_gid() == "paper_col")
        assert (col.get_x(), col.get_y()) == (index - 0.5, -0.5)
        assert (col.get_width(), col.get_height()) == (1, k)


def test_the_axes_have_thin_kind_separators_and_short_kind_labels(
    output: registry.FigureOutput,
) -> None:
    labels = _kind_labels()
    runs = [
        (label, labels.index(label), len(labels) - labels[::-1].index(label))
        for label in dict.fromkeys(labels)
    ]
    bounds = [end for _, _, end in runs[:-1]]
    for position, ax in enumerate(_matrices(output)):
        seps = [ln for ln in ax.get_lines() if ln.get_gid() == "separator"]
        assert len(seps) == 2 * len(bounds)
        kinds = [t.get_text() for t in ax.texts if t.get_gid() == "kind"]
        if position % 2 == 0:
            expected = [lab for lab, a, b in runs if b - a >= 2]
            assert kinds == expected
        else:
            assert kinds == []


def test_one_shared_colour_bar_below_the_grid_marks_the_noise(
    output: registry.FigureOutput,
) -> None:
    fig = output.figure
    fig.draw_without_rendering()
    matrices = _matrices(output)
    (bar,) = [ax for ax in fig.axes if not ax.images]
    assert bar.get_position().y1 <= min(ax.get_position().y0 for ax in matrices)
    ticks = [ln for ln in bar.get_lines() if ln.get_gid() == "noise_tick"]
    edges = []
    for role in ROLES:
        mine = [_noise(m) for m in MODELS if _role(m) == role]
        edges += [min(mine), max(mine)]
        assert output.numbers[f"noise_{role}_min"] == pytest.approx(min(mine))
        assert output.numbers[f"noise_{role}_max"] == pytest.approx(max(mine))
    assert sorted(float(t.get_xdata()[0]) for t in ticks) == pytest.approx(
        sorted(edges)
    )
    assert "noise_min" not in output.numbers
    assert "noise_max" not in output.numbers
    assert fig.get_size_inches()[1] <= 9.5 + 1e-9


def test_the_text_is_at_least_the_glossary_floor_and_never_bold(
    output: registry.FigureOutput,
) -> None:
    for t in output.figure.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text():
            assert t.get_fontsize() >= paper.MIN_TEXT, t.get_text()
            assert t.get_fontweight() in ("normal", 400), t.get_text()


def test_index_one_is_labelled_on_both_axes_of_every_matrix(
    output: registry.FigureOutput,
) -> None:
    seen = {"x": 0, "y": 0}
    for ax in _matrices(output):
        assert ax.get_xticks()[0] == 0
        assert ax.get_yticks()[0] == 0
        for name, labels in (
            ("x", ax.get_xticklabels()),
            ("y", ax.get_yticklabels()),
        ):
            if labels:
                seen[name] += 1
                assert labels[0].get_text() == "1"
    assert all(seen.values())


def test_the_caption_gives_the_noise_range_per_role(
    output: registry.FigureOutput,
) -> None:
    text = output.detail
    for role in ROLES:
        mine = [_noise(m) for m in MODELS if _role(m) == role]
        assert f"{min(mine):.3f} to {max(mine):.3f}" in text


def test_the_cells_at_or_below_zero_are_named_by_checkpoint(
    output: registry.FigureOutput,
) -> None:
    text = output.detail
    below = {
        m: int((_margin(m) <= 0).sum())
        for m in MODELS
        if (_margin(m) <= 0).any()
    }
    assert below
    for model in below:
        assert paper.display_name(model) in text
        low = float(_margin(model).min())
        shown = (MINUS if low < 0 else "") + f"{abs(low):.2f}"
        assert shown in text
    assert "lowest" in text
    assert f"the {int(output.numbers['n_below'])} cells" in text


def _bar(output: registry.FigureOutput) -> Any:
    output.figure.draw_without_rendering()
    (bar,) = [ax for ax in output.figure.axes if not ax.images]
    return bar


def _extent(artist: Any, output: registry.FigureOutput) -> Any:
    return artist.get_window_extent(output.figure.canvas.get_renderer())


def test_each_role_has_its_own_labelled_row_of_noise_ticks(
    output: registry.FigureOutput,
) -> None:
    bar = _bar(output)
    labels = [t for t in bar.texts if t.get_gid() == "noise_role"]
    assert [t.get_text() for t in labels] == [
        stats.ROLE_NAMES[r] for r in ROLES
    ]
    bar_box = _extent(bar, output)
    boxes = [_extent(t, output) for t in labels]
    for text, box in zip(labels, boxes, strict=True):
        assert text.get_fontsize() >= paper.MIN_TEXT
        assert box.x0 >= bar_box.x1
    for first, second in itertools.pairwise(boxes):
        assert first.y0 >= second.y1 or second.y0 >= first.y1
    ticks = [ln for ln in bar.get_lines() if ln.get_gid() == "noise_tick"]
    assert len(ticks) == 2 * len(ROLES)
    for role, label in zip(ROLES, boxes, strict=True):
        mine = [_noise(m) for m in MODELS if _role(m) == role]
        for tick in ticks:
            if float(tick.get_xdata()[0]) in (
                pytest.approx(min(mine)),
                pytest.approx(max(mine)),
            ):
                box = _extent(tick, output)
                centre = (box.y0 + box.y1) / 2
                assert label.y0 <= centre <= label.y1


def test_the_caption_says_which_noise_edges_coincide(
    output: registry.FigureOutput,
) -> None:
    edges = []
    for role in ROLES:
        mine = [_noise(m) for m in MODELS if _role(m) == role]
        edges += [
            (role, "min", round(min(mine), 3)),
            (role, "max", round(max(mine), 3)),
        ]
    shared = {
        value
        for _, _, value in edges
        if len({r for r, _, v in edges if v == value}) > 1
    }
    assert shared
    for value in shared:
        assert f"coincide at {value:.3f}" in output.detail


def test_the_heatmaps_are_titled_fitted_and_scored_wording(
    output: registry.FigureOutput,
) -> None:
    fig = output.figure
    fig.draw_without_rendering()
    matrices = _matrices(output)
    sup = fig._supylabel
    assert sup is not None
    assert sup.get_text() == "fitted wording"
    assert sup.get_fontsize() >= paper.MIN_TEXT
    box = _extent(sup, output)
    others = [
        _extent(t, output)
        for ax in matrices
        for t in ax.texts
        if t.get_gid() == "kind"
    ]
    for ax in matrices:
        others.append(ax.get_window_extent())
    for other in others:
        assert (
            box.x1 <= other.x0
            or other.x1 <= box.x0
            or box.y1 <= other.y0
            or other.y1 <= box.y0
        )
    bottom = matrices[-2:]
    for ax in bottom:
        label = ax.xaxis.label
        assert label.get_text() == "scored wording"
        assert label.get_fontsize() >= paper.MIN_TEXT
    for ax in matrices[:-2]:
        assert ax.xaxis.label.get_text() == ""


def test_the_zero_clause_is_data_driven() -> None:
    above = {"gpt2": np.ones((3, 3))}
    clause = a_wording._zero_clause(above)
    assert clause == "no cell is at or below zero; "
    one = {"gpt2": np.ones((3, 3)), "qwen3-4b": np.ones((3, 3))}
    one["qwen3-4b"][0, 1] = -0.1
    text = a_wording._zero_clause(one)
    assert "the 1 cell at or below zero" in text
    assert "carries a dot; " in text
    assert "0 cells" not in text
    assert "  " not in text
    many = {"qwen3-4b": np.zeros((2, 2))}
    assert a_wording._zero_clause(many).startswith(
        "the 4 cells at or below zero (all on "
    )
    assert "  " not in a_wording._zero_clause(many)


def test_the_current_caption_keeps_its_zero_sentence(
    output: registry.FigureOutput,
) -> None:
    n = int(output.numbers["n_below"])
    assert n > 0
    assert f"the {n} cells at or below zero (all on " in output.detail
    assert ") carry a dot; " in output.detail
