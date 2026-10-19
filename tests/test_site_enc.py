import itertools
from pathlib import Path
import runpy

import pytest

from tomography.viz import paper

_COLOR = runpy.run_path(str(Path(__file__).with_name("_color.py")))
de2000 = _COLOR["de2000"]

NON_FAMILY = ("text", "paper", "popularity", "own_probability")


def test_the_family_ramps_have_the_dictionary_hex_values() -> None:
    assert paper.size_ramp("gpt2", 4) == [
        "#eb6800",
        "#ac4c00",
        "#6d3000",
        "#2e1400",
    ]
    assert paper.size_ramp("qwen3", 3) == ["#0083cc", "#005280", "#002133"]


def test_enc_holds_one_value_per_meaning() -> None:
    assert paper.ENC["gpt2"] == tuple(paper.size_ramp("gpt2", 4))
    assert paper.ENC["qwen3"] == tuple(paper.size_ramp("qwen3", 3))
    assert paper.ENC["direction"] == "#1a1a1a"
    assert paper.ENC["random_band"] == "#d9d9d9"
    assert paper.ENC["random_marker"] == "#888888"
    assert paper.ENC["reference"] == "#7f7f7f"
    assert paper.ENC["text"] == "#009E73"
    assert paper.ENC["paper"] == "#CC79A7"
    assert paper.ENC["popularity"] == "#404040"
    assert paper.ENC["own_probability"] == "#6a6a6a"
    assert paper.ENC["unused"] == "#F0E442"
    assert paper.ENC["secondary_text"] == "#4d4d4d"


def test_line_styles_have_one_meaning_each() -> None:
    assert paper.LINESTYLES["solid"] == "-"
    assert paper.LINESTYLES["post_trained"] == (0, (3, 2))
    assert paper.LINESTYLES["reference"] == (0, (1, 1.5))
    assert paper.LINESTYLES["trend"] == (0, (4, 1.5, 1, 1.5))
    assert paper.REFERENCE_WIDTH == 0.5
    assert paper.TREND_WIDTH == 0.6


@pytest.mark.parametrize("family", ["gpt2", "qwen3"])
def test_every_ramp_step_is_far_from_every_other_hue(family: str) -> None:
    for step, other in itertools.product(paper.ENC[family], NON_FAMILY):
        assert de2000(step, paper.ENC[other]) >= 15.0, (step, other)


@pytest.mark.parametrize("family", ["gpt2", "qwen3"])
def test_adjacent_ramp_steps_differ_by_at_least_ten(family: str) -> None:
    ramp = paper.ENC[family]
    for a, b in itertools.pairwise(ramp):
        assert de2000(a, b) >= 10.0, (a, b)


def test_the_ciede2000_helper_matches_a_published_pair() -> None:
    assert de2000("#ff0000", "#ff0000") == pytest.approx(0.0)
    assert de2000("#000000", "#ffffff") == pytest.approx(100.0, abs=0.01)


def test_text_colours_are_the_ink_or_the_secondary_grey_only() -> None:
    for key in ("text.color", "axes.labelcolor", "xtick.color", "ytick.color"):
        assert paper.RC[key] in (paper.INK, paper.ENC["secondary_text"])
