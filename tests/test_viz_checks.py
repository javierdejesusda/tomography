from matplotlib.figure import Figure

from tomography.viz import checks
from tomography.viz import paper


def _clean() -> Figure:
    with paper.paper_style():
        fig = Figure(figsize=(3, 2), layout="constrained")
        ax = fig.subplots()
        ax.plot([0, 1], [0.4, 0.6], marker="o")
        ax.set_xlabel("relative depth")
    return fig


def test_a_clean_figure_has_no_problems() -> None:
    assert checks.problems(_clean()) == []


def test_overlapping_text_is_reported() -> None:
    fig = _clean()
    ax = fig.axes[0]
    ax.text(0.5, 0.5, "alpha", transform=ax.transAxes)
    ax.text(0.5, 0.5, "alpha", transform=ax.transAxes)
    found = checks.problems(fig)
    assert any("overlap" in item for item in found)


def test_text_outside_the_figure_is_reported() -> None:
    fig = _clean()
    fig.text(0.99, 0.5, "far too long to fit", ha="left")
    found = checks.problems(fig)
    assert any("outside" in item for item in found)


def test_a_marker_on_a_tick_label_is_reported() -> None:
    fig = _clean()
    ax = fig.axes[0]
    ax.set_xlim(0, 1)
    ax.set_yticks([0.4, 0.5, 0.6])
    ax.plot([-0.04], [0.5], marker="o", ms=8, clip_on=False)
    found = checks.problems(fig)
    assert any("marker" in item for item in found)
