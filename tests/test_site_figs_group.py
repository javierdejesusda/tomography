from matplotlib.figure import Figure

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
