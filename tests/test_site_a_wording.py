from pathlib import Path
import runpy

import numpy as np
import pytest

from tomography.site import registry
from tomography.site import stats
from tomography.site.figs import a_wording
from tomography.site.manifest import Manifest
from tomography.viz import checks
from tomography.viz import paper

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
    assert height <= 8.0 + 1e-9


def test_every_cell_is_the_transfer_accuracy_minus_the_scored_random(
    output: registry.FigureOutput,
) -> None:
    for ax, model in zip(_matrices(output), MODELS, strict=True):
        shown = np.asarray(ax.images[0].get_array(), dtype=float)
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
    assert "Table A6" in text
    assert "purple" in text
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


def test_no_text_sits_inside_a_matrix_and_nothing_has_an_underscore(
    output: registry.FigureOutput,
) -> None:
    for ax in _matrices(output):
        assert all(t.get_text().startswith("$") for t in ax.texts)
    assert checks.text_problems(output.figure) == []


def test_the_diagonal_has_white_dots_and_the_paper_tick_is_purple(
    output: registry.FigureOutput,
) -> None:
    k = len(_names())
    paper_index = _names().index("paper")
    for ax in _matrices(output):
        dots = [
            ln
            for ln in ax.get_lines()
            if len(ln.get_xdata()) == k and ln.get_markerfacecolor() == "white"
        ]
        assert len(dots) == 1
        assert list(dots[0].get_xdata()) == list(range(k))
        assert list(dots[0].get_ydata()) == list(range(k))
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == [str(i + 1) for i in range(k)]
        ticks = ax.xaxis.get_major_ticks()
        for i, tick in enumerate(ticks):
            colour = tick.tick1line.get_markeredgecolor()
            assert (colour == paper.ENC["paper"]) == (i == paper_index)


def test_the_guards_are_green(output: registry.FigureOutput) -> None:
    assert checks.guard_problems(output.figure, appendix=True) == []
    assert checks.problems(output.figure) == []


def test_the_csv_holds_every_cell(output: registry.FigureOutput) -> None:
    lines = output.csv.strip().splitlines()
    head = lines[0].split(",")
    assert head == ["model", "role", "fit", "score", "margin"]
    assert len(lines) - 1 == len(MODELS) * len(_names()) ** 2
    first = dict(zip(head, lines[1].split(","), strict=True))
    assert first["model"] == MODELS[0]
    assert float(first["margin"]) == pytest.approx(
        _margin(MODELS[0])[0, 0], abs=1e-6
    )


def test_the_colour_scale_is_centred_on_zero_and_shows_negative_margins(
    output: registry.FigureOutput,
) -> None:
    lowest = min(float(_margin(m).min()) for m in MODELS)
    assert lowest < 0
    for ax in _matrices(output):
        norm = ax.images[0].norm
        assert norm.vcenter == 0
        assert norm.vmin < lowest
    (bar,) = [ax for ax in output.figure.axes if not ax.images]
    ticks = [float(t) for t in bar.get_xticks()]
    assert min(ticks) < 0
    assert 0.0 in ticks


def test_the_colour_bar_fills_the_empty_slots_and_the_figure_is_compact(
    output: registry.FigureOutput,
) -> None:
    fig = output.figure
    fig.draw_without_rendering()
    matrices = _matrices(output)
    (bar,) = [ax for ax in fig.axes if not ax.images]
    lowest_row = min(ax.get_position().y0 for ax in matrices)
    assert bar.get_position().y1 <= max(ax.get_position().y1 for ax in matrices)
    assert bar.get_position().y0 >= lowest_row - 1e-6
    assert fig.get_size_inches()[1] <= 7.5
