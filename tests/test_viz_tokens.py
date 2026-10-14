"""Tests for the token scan primitive."""

from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.image import AxesImage
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
