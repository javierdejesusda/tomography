from matplotlib.figure import Figure
import pytest

from tomography.site.figs import _group


def test_role_separates_gpt2_qwen3_base_and_post_trained() -> None:
    assert _group.role("gpt2-xl") == "GPT-2"
    assert _group.role("qwen3-4b-base") == "Qwen3 base"
    assert _group.role("qwen3-4b") == "Qwen3 post-trained"


def test_lines_by_role_never_mix_families_or_roles() -> None:
    models = [
        "gpt2",
        "gpt2-xl",
        "qwen3-0.6b-base",
        "qwen3-0.6b",
        "qwen3-4b-base",
        "qwen3-4b",
    ]
    assert _group.lines_by_role(models) == {
        "GPT-2": ["gpt2", "gpt2-xl"],
        "Qwen3 base": ["qwen3-0.6b-base", "qwen3-4b-base"],
        "Qwen3 post-trained": ["qwen3-0.6b", "qwen3-4b"],
    }


def test_lines_by_role_sorts_by_parameter_count_and_skips_empty_roles() -> None:
    found = _group.lines_by_role(["gpt2-xl", "gpt2", "gpt2-medium"])
    assert found == {"GPT-2": ["gpt2", "gpt2-medium", "gpt2-xl"]}


def test_wilson_matches_the_closed_form() -> None:
    lo, hi = _group.wilson(0, 120)
    assert lo == 0.0
    assert hi == pytest.approx(0.031019166418703472)
    lo, hi = _group.wilson(59, 120)
    assert lo < 59 / 120 < hi


def test_wilson_rejects_an_empty_sample() -> None:
    with pytest.raises(ValueError, match="positive"):
        _group.wilson(0, 0)


def test_csv_text_writes_the_header_and_rows() -> None:
    text = _group.csv_text(["a", "b"], [[1, 0.5], [2, None]])
    assert text == "a,b\n1,0.5\n2,\n"


def test_draw_series_connects_only_within_a_role() -> None:
    fig = Figure()
    ax = fig.subplots()
    values = {
        "gpt2": 0.5,
        "gpt2-xl": 0.6,
        "qwen3-0.6b-base": 0.7,
        "qwen3-4b-base": 0.8,
        "qwen3-0.6b": 0.65,
        "qwen3-4b": 0.75,
    }
    _group.draw_series(ax, values, "#0072B2")
    lines = [line for line in ax.get_lines()]
    assert len(lines) == 3
    assert sorted(len(line.get_xdata()) for line in lines) == [2, 2, 2]
    styles = {line.get_linestyle() for line in lines}
    assert styles == {"-", "--"}


def test_draw_series_leaves_post_trained_markers_open() -> None:
    fig = Figure()
    ax = fig.subplots()
    _group.draw_series(ax, {"qwen3-0.6b": 0.5, "qwen3-4b": 0.6}, "#0072B2")
    assert ax.get_lines()[0].get_markerfacecolor() == "white"


def test_role_handles_name_the_three_roles() -> None:
    labels = [h.get_label() for h in _group.role_handles()]
    assert labels == ["GPT-2", "Qwen3 base", "Qwen3 post-trained"]


def test_clip_ticks_keeps_only_ticks_inside_the_view() -> None:
    fig = Figure()
    ax = fig.subplots()
    ax.plot([0.0, 1.0], [-2.4, 3.9])
    _group.clip_ticks(ax)
    low, high = ax.get_ylim()
    assert all(low <= t <= high for t in ax.get_yticks())
    assert len(ax.get_yticks()) > 1
    left, right = ax.get_xlim()
    assert all(left <= t <= right for t in ax.get_xticks())
