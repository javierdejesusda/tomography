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


def _plain() -> Figure:
    with paper.paper_style():
        fig = Figure(figsize=(3, 2), layout="constrained")
        ax = fig.subplots()
        ax.plot([0, 1], [0.4, 0.6], color=paper.ENC["direction"])
        ax.set_xlabel("relative depth")
    return fig


def test_a_figure_with_dictionary_colours_has_no_palette_problem() -> None:
    fig = _plain()
    ax = fig.axes[0]
    ax.plot([0, 1], [0.5, 0.5], color=paper.ENC["gpt2"][1], marker="o")
    ax.fill_between([0, 1], 0.4, 0.6, color=paper.ENC["random_band"])
    ax.text(0.5, 0.5, "note", color=paper.ENC["secondary_text"])
    ax.scatter([0.2], [0.2], facecolor="white", edgecolor=paper.ENC["qwen3"][0])
    assert checks.palette_problems(fig) == []


def test_an_off_dictionary_line_colour_is_a_palette_problem() -> None:
    fig = _plain()
    fig.axes[0].plot([0, 1], [0.1, 0.2], color="#ff00ff")
    found = checks.palette_problems(fig)
    assert any("#ff00ff" in item for item in found)


def test_an_off_dictionary_patch_colour_is_a_palette_problem() -> None:
    fig = _plain()
    fig.axes[0].bar([0.5], [0.2], color="#123456")
    assert any("#123456" in i for i in checks.palette_problems(fig))


def test_the_secondary_grey_is_allowed_for_text_only() -> None:
    fig = _plain()
    fig.axes[0].plot([0, 1], [0.1, 0.2], color="#4d4d4d")
    assert any("#4d4d4d" in i for i in checks.palette_problems(fig))


def test_an_off_dictionary_text_colour_is_a_palette_problem() -> None:
    fig = _plain()
    fig.axes[0].text(0.5, 0.5, "red", color="#cc0000")
    assert any("#cc0000" in i for i in checks.palette_problems(fig))


def test_a_heatmap_is_exempt_from_the_palette_rule() -> None:
    fig = _plain()
    fig.axes[0].imshow([[0.1, 0.9], [0.5, 0.2]], cmap="cividis")
    assert checks.palette_problems(fig) == []


def test_main_figures_need_6_5_pt_and_appendix_figures_6_pt() -> None:
    fig = _plain()
    fig.axes[0].text(0.5, 0.5, "small", fontsize=6.2)
    assert any("small" in i for i in checks.font_problems(fig))
    assert checks.font_problems(fig, appendix=True) == []
    fig.axes[0].text(0.2, 0.2, "tiny", fontsize=5.5)
    assert any("tiny" in i for i in checks.font_problems(fig, appendix=True))
    assert checks.MIN_FONT_MAIN == 6.5
    assert checks.MIN_FONT_APPENDIX == 6.0


def test_a_plain_paper_figure_has_no_font_problem() -> None:
    assert checks.font_problems(_plain()) == []


def test_an_underscore_outside_mathtext_is_a_text_problem() -> None:
    fig = _plain()
    fig.axes[0].text(0.5, 0.5, "no_template")
    found = checks.text_problems(fig)
    assert any("no_template" in item for item in found)


def test_an_underscore_inside_mathtext_is_allowed() -> None:
    fig = _plain()
    fig.axes[0].text(0.5, 0.5, r"$x_1$ and $h_\ell$")
    assert checks.text_problems(fig) == []


def test_an_underscore_in_a_tick_label_is_a_text_problem() -> None:
    fig = _plain()
    ax = fig.axes[0]
    ax.set_xticks([0, 1], ["no_template", "paper"])
    assert any("no_template" in i for i in checks.text_problems(fig))


def test_guard_problems_runs_the_three_guards() -> None:
    fig = _plain()
    fig.axes[0].plot([0, 1], [0.1, 0.2], color="#ff00ff")
    fig.axes[0].text(0.5, 0.5, "a_b", fontsize=5)
    found = checks.guard_problems(fig)
    assert any("#ff00ff" in i for i in found)
    assert any("a_b" in i for i in found)
    assert any("font" in i for i in found)
