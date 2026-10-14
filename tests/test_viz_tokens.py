"""Tests for the token scan primitive."""

from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.image import AxesImage
from matplotlib.text import Text
import numpy as np
import pytest

from tomography.viz import style
from tomography.viz import tokens

TOKENS = ["The", " cat", " sat", " on", " it"]


def make_scores(n_layers: int = 4, n_tokens: int = 5) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(0.0, 1.0, (n_layers, n_tokens))


def scan_image(ax: Axes) -> AxesImage:
    (image,) = ax.images
    return image


def test_scan_puts_layer_zero_at_the_bottom() -> None:
    scores = make_scores()

    fig = tokens.plot_scan(scores, np.arange(4), tokens=TOKENS)

    image = scan_image(fig.axes[0])
    assert image.origin == "lower"
    np.testing.assert_array_equal(np.ma.getdata(image.get_array()), scores)


def test_color_range_is_symmetric_around_zero() -> None:
    scores = make_scores()

    fig = tokens.plot_scan(scores, np.arange(4), tokens=TOKENS)

    norm = scan_image(fig.axes[0]).norm
    assert norm.vmax == pytest.approx(np.abs(scores).max())
    assert norm.vmin == pytest.approx(-np.abs(scores).max())


def test_zero_maps_to_the_gray_midpoint() -> None:
    fig = tokens.plot_scan(make_scores(), np.arange(4), tokens=TOKENS)

    image = scan_image(fig.axes[0])

    assert to_rgb(image.cmap(image.norm(0.0))) == pytest.approx(
        to_rgb(style.NEUTRAL), abs=2 / 255
    )


def test_an_explicit_vmax_fixes_the_color_range() -> None:
    fig = tokens.plot_scan(make_scores(), np.arange(4), tokens=TOKENS, vmax=5.0)

    norm = scan_image(fig.axes[0]).norm
    assert (norm.vmin, norm.vmax) == (-5.0, 5.0)


def test_mask_first_blanks_the_first_token() -> None:
    scores = make_scores()
    scores[:, 0] = 100.0

    fig = tokens.plot_scan(scores, np.arange(4), tokens=TOKENS, mask_first=True)

    image = scan_image(fig.axes[0])
    mask = np.ma.getmaskarray(image.get_array())
    assert mask[:, 0].all()
    assert not mask[:, 1:].any()
    assert image.norm.vmax == pytest.approx(np.abs(scores[:, 1:]).max())


def test_nothing_is_masked_by_default() -> None:
    fig = tokens.plot_scan(make_scores(), np.arange(4), tokens=TOKENS)

    mask = np.ma.getmaskarray(scan_image(fig.axes[0]).get_array())

    assert not mask.any()


def test_token_ticks_show_every_token_with_escaped_whitespace() -> None:
    names = ["a", " b", "\n", "\t"]

    fig = tokens.plot_scan(make_scores(n_tokens=4), np.arange(4), tokens=names)

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        "a",
        " b",
        "\\n",
        "\\t",
    ]


def test_layer_ticks_are_the_layer_indices() -> None:
    fig = tokens.plot_scan(
        make_scores(n_layers=3), np.array([0, 6, 12]), tokens=TOKENS
    )

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["0", "6", "12"]
    assert ax.get_ylabel() == "Layer"


def test_many_layers_get_a_thinned_tick_set_that_keeps_both_ends() -> None:
    layers = np.arange(49)

    fig = tokens.plot_scan(make_scores(n_layers=49), layers, tokens=TOKENS)

    ax = fig.axes[0]
    ticks = [int(t.get_text()) for t in ax.get_yticklabels()]
    assert len(ticks) <= 10
    assert ticks[0] == 0
    assert ticks[-1] == 48


def test_thinned_layer_ticks_keep_a_constant_stride() -> None:
    fig = tokens.plot_scan(
        make_scores(n_layers=13), np.arange(13), tokens=TOKENS
    )

    ticks = [int(t.get_text()) for t in fig.axes[0].get_yticklabels()]

    assert ticks == [0, 2, 4, 6, 8, 10, 12]


def test_a_colorbar_is_added_by_default_and_can_be_dropped() -> None:
    with_bar = tokens.plot_scan(make_scores(), np.arange(4), tokens=TOKENS)
    without = tokens.plot_scan(
        make_scores(), np.arange(4), tokens=TOKENS, colorbar=False
    )

    assert len(with_bar.axes) == 2
    assert with_bar.axes[1].get_ylabel() == "Reading score"
    assert len(without.axes) == 1


def test_plot_scan_draws_on_a_given_axes() -> None:
    fig, axes = style.new_figure(1, 2)

    returned = tokens.plot_scan(
        make_scores(), np.arange(4), tokens=TOKENS, ax=axes[0, 1]
    )

    assert returned is fig
    assert axes[0, 1].images
    assert not axes[0, 0].images


@pytest.mark.parametrize(
    ("scores", "layers", "names", "message"),
    [
        (np.zeros(5), np.arange(1), TOKENS, "2-D"),
        (np.zeros((4, 5)), np.arange(3), TOKENS, "layers"),
        (np.zeros((4, 5)), np.arange(4), TOKENS[:3], "tokens"),
    ],
    ids=["not-2d", "layers-mismatch", "tokens-mismatch"],
)
def test_plot_scan_rejects_mismatched_inputs(
    scores: np.ndarray, layers: np.ndarray, names: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        tokens.plot_scan(scores, layers, tokens=names)


def test_plot_scan_rejects_a_non_positive_vmax() -> None:
    with pytest.raises(ValueError, match="vmax"):
        tokens.plot_scan(make_scores(), np.arange(4), tokens=TOKENS, vmax=0.0)


def test_plot_scan_rejects_masking_the_only_token() -> None:
    with pytest.raises(ValueError, match="mask_first"):
        tokens.plot_scan(
            make_scores(n_tokens=1),
            np.arange(4),
            tokens=["x"],
            mask_first=True,
        )


def test_scan_figure_saves_in_all_formats(tmp_path: Path) -> None:
    fig = tokens.plot_scan(make_scores(), np.arange(4), tokens=TOKENS)

    paths = style.save_figure(fig, tmp_path, "scan")

    assert [path.suffix for path in paths] == [".pdf", ".png", ".svg"]
    assert all(path.stat().st_size > 0 for path in paths)


def strip_texts(ax: Axes) -> list[Text]:
    return list(ax.texts)


def fill_of(text: Text) -> tuple[float, ...]:
    patch = text.get_bbox_patch()
    assert patch is not None
    return tuple(patch.get_facecolor())


def test_token_strip_writes_one_text_per_token() -> None:
    fig = tokens.plot_token_strip(np.zeros(5), TOKENS)

    shown = [text.get_text() for text in strip_texts(fig.axes[0])]

    assert shown == ["The", "cat", "sat", "on", "it"]


def test_token_strip_fills_each_token_by_its_score() -> None:
    cmap = style.diverging_cmap()

    fig = tokens.plot_token_strip(
        np.array([-1.0, 0.0, 1.0]), ["a", "b", "c"], vmax=1.0
    )

    fills = [fill_of(text) for text in strip_texts(fig.axes[0])]
    for fill, position in zip(fills, (0.0, 0.5, 1.0), strict=True):
        assert fill == pytest.approx(cmap(position), abs=2 / 255)


def test_token_strip_text_stays_readable_on_every_fill() -> None:
    fig = tokens.plot_token_strip(
        np.array([-1.0, 0.0, 1.0]), ["a", "b", "c"], vmax=1.0
    )

    inks = [text.get_color() for text in strip_texts(fig.axes[0])]

    assert inks == [style.SURFACE, style.INK, style.SURFACE]


def test_token_strip_leaves_the_masked_first_token_unfilled() -> None:
    scores = np.array([9.0, 0.5, -0.5])

    fig = tokens.plot_token_strip(scores, ["<bos>", "a", "b"], mask_first=True)

    first, second, _ = strip_texts(fig.axes[0])
    assert fill_of(first)[3] == 0.0
    assert fill_of(second)[3] == 1.0
    assert first.get_color() == style.INK_SECONDARY


def test_token_strip_color_range_ignores_the_masked_token() -> None:
    cmap = style.diverging_cmap()
    scores = np.array([100.0, 1.0, -1.0])

    fig = tokens.plot_token_strip(scores, ["<bos>", "a", "b"], mask_first=True)

    _, second, third = strip_texts(fig.axes[0])
    assert fill_of(second) == pytest.approx(cmap(1.0), abs=2 / 255)
    assert fill_of(third) == pytest.approx(cmap(0.0), abs=2 / 255)


def test_token_strip_wraps_inside_the_axes_in_reading_order() -> None:
    words = [f" word{i}" for i in range(40)]

    fig = tokens.plot_token_strip(np.zeros(40), words, width=3.5)

    fig.canvas.draw()
    ax = fig.axes[0]
    boxes = [text.get_window_extent() for text in strip_texts(ax)]
    limit = ax.get_window_extent().x1 + 1
    assert all(box.x1 <= limit for box in boxes)
    rows = sorted({round(box.y0) for box in boxes}, reverse=True)
    assert len(rows) > 1
    previous = None
    for box in boxes:
        if previous is not None and round(box.y0) == round(previous.y0):
            assert box.x0 >= previous.x1 - 1
        else:
            assert previous is None or box.y0 < previous.y0
        previous = box


def test_token_strip_figure_grows_with_the_number_of_lines() -> None:
    short = tokens.plot_token_strip(np.zeros(3), ["a", " b", " c"])
    long = tokens.plot_token_strip(
        np.zeros(120), [f" word{i}" for i in range(120)]
    )

    assert long.get_size_inches()[1] > short.get_size_inches()[1]


def test_token_strip_draws_on_a_given_axes() -> None:
    fig, axes = style.new_figure(2, 1)

    returned = tokens.plot_token_strip(
        np.zeros(3), ["a", "b", "c"], ax=axes[1, 0]
    )

    assert returned is fig
    assert len(axes[1, 0].texts) == 3
    assert not axes[0, 0].texts
    assert not axes[1, 0].axison


def test_token_strip_rejects_mismatched_or_unusable_inputs() -> None:
    with pytest.raises(ValueError, match="tokens"):
        tokens.plot_token_strip(np.zeros(3), ["a", "b"])
    with pytest.raises(ValueError, match="1-D"):
        tokens.plot_token_strip(np.zeros((2, 2)), ["a", "b"])
    with pytest.raises(ValueError, match="vmax"):
        tokens.plot_token_strip(np.zeros(2), ["a", "b"], vmax=-1.0)
    with pytest.raises(ValueError, match="mask_first"):
        tokens.plot_token_strip(np.zeros(1), ["a"], mask_first=True)


def test_token_strip_saves_in_all_formats(tmp_path: Path) -> None:
    fig = tokens.plot_token_strip(np.linspace(-1, 1, 5), TOKENS)

    paths = style.save_figure(fig, tmp_path, "strip")

    assert [path.suffix for path in paths] == [".pdf", ".png", ".svg"]
    assert all(path.stat().st_size > 0 for path in paths)
