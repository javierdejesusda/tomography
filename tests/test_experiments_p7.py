"""Tests for the P7 extraction template versus steering experiment."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.analysis import lens_logits
from tomography.analysis import pole_separation
from tomography.experiments.a2_logit_lens import lens_params
from tomography.experiments.a2_logit_lens import pole_ids
from tomography.experiments.p7_extraction_steering import CHAT_INSTRUCTION
from tomography.experiments.p7_extraction_steering import FORMATS
from tomography.experiments.p7_extraction_steering import STORY_PREFIX
from tomography.experiments.p7_extraction_steering import TEMPLATES
from tomography.experiments.p7_extraction_steering import extraction_variants
from tomography.experiments.p7_extraction_steering import lens_separation
from tomography.experiments.p7_extraction_steering import safe_slope
from tomography.experiments.p7_extraction_steering import shift_prompts
from tomography.experiments.p7_extraction_steering import spearman
from tomography.experiments.steering import dose_slope
from tomography.experiments.variants import READING_TASKS
from tomography.experiments.variants import template_variants
from tomography.model import LoadedModel
from tomography.reading.reader import RepReader
from tomography.templates import ChatFormat

PLAIN = ChatFormat("plain")


def _template_chat() -> ChatFormat:
    def render(messages: list[dict[str, str]]) -> str:
        return f"<u>{messages[0]['content']}</u><a>"

    return ChatFormat("template", render_fn=render)


def test_the_default_templates_are_six_names_below_the_series_cap() -> None:
    assert TEMPLATES == (
        "paper",
        "no_template",
        "no_chat_tags",
        "neutral_frame",
        "scaffold_0",
        "concept_swap_0",
    )


@pytest.mark.parametrize(
    "chat", [PLAIN, _template_chat()], ids=["plain", "template"]
)
def test_the_default_names_are_the_variants_of_the_sentiment_task(
    chat: ChatFormat,
) -> None:
    variants, _ = template_variants(READING_TASKS["sentiment"], chat)
    by_name = {v.name: v for v in variants}
    picked = extraction_variants(chat, TEMPLATES)
    assert picked == tuple(by_name[name] for name in TEMPLATES)
    assert all(v.position is None and v.evaluable for v in picked)


def test_the_variants_keep_the_order_of_the_names() -> None:
    picked = extraction_variants(PLAIN, ("scaffold_0", "paper", "no_template"))
    assert [v.name for v in picked] == ["scaffold_0", "paper", "no_template"]


def test_an_indexed_kind_is_picked_by_its_index() -> None:
    picked = extraction_variants(PLAIN, ("paper", "concept_swap_2"))
    assert [v.name for v in picked] == ["paper", "concept_swap_2"]


def test_framing_swap_raises_with_the_skip_reason() -> None:
    with pytest.raises(
        ValueError,
        match=r"'framing_swap'.*task 'sentiment' has no function framing",
    ):
        extraction_variants(PLAIN, ("paper", "framing_swap"))


def test_an_index_past_the_last_variant_raises() -> None:
    with pytest.raises(
        ValueError,
        match=r"'scaffold_9'.*variants\.py has no variant of that name",
    ):
        extraction_variants(PLAIN, ("paper", "scaffold_9"))


@pytest.mark.parametrize("name", ["shuffle", "shuffle_0", "no_trailing_0"])
def test_a_name_of_an_unknown_kind_raises(name: str) -> None:
    with pytest.raises(ValueError, match=f"{name!r}.*not a variant kind"):
        extraction_variants(PLAIN, ("paper", name))


def test_names_without_paper_raise() -> None:
    with pytest.raises(ValueError, match="kinds must include 'paper'"):
        extraction_variants(PLAIN, ("no_template", "scaffold_0"))


FRAGMENTS = ["the movie", "a film"]


def test_the_formats_are_raw_chat_and_story() -> None:
    assert FORMATS == ("raw", "chat", "story")
    assert CHAT_INSTRUCTION == "Continue the text."
    assert STORY_PREFIX == "Here is a short story: "


def test_raw_prompts_are_the_fragments() -> None:
    assert shift_prompts(FRAGMENTS, "raw", PLAIN) == FRAGMENTS


def test_chat_prompts_start_the_answer_with_the_fragment() -> None:
    assert shift_prompts(FRAGMENTS, "chat", PLAIN) == [
        "USER: Continue the text. ASSISTANT: the movie",
        "USER: Continue the text. ASSISTANT: a film",
    ]
    assert shift_prompts(FRAGMENTS, "chat", _template_chat()) == [
        "<u>Continue the text.</u><a>the movie",
        "<u>Continue the text.</u><a>a film",
    ]


def test_story_prompts_put_the_prefix_before_the_fragment() -> None:
    assert shift_prompts(FRAGMENTS, "story", PLAIN) == [
        "Here is a short story: the movie",
        "Here is a short story: a film",
    ]


def test_an_unknown_format_raises() -> None:
    with pytest.raises(ValueError, match="unknown format 'poem'"):
        shift_prompts(FRAGMENTS, "poem", PLAIN)


def test_spearman_of_a_monotone_pair_is_one() -> None:
    assert spearman([1.0, 2.0, 3.0, 4.0], [1.0, 8.0, 9.0, 50.0]) == (
        pytest.approx(1.0)
    )
    assert spearman([1.0, 2.0, 3.0], [9.0, 5.0, 1.0]) == pytest.approx(-1.0)


def test_spearman_averages_the_ranks_of_ties() -> None:
    value = spearman([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 2.0, 3.0])
    assert value == pytest.approx(np.sqrt(0.9))


def test_spearman_drops_the_pairs_with_a_non_finite_side() -> None:
    x = [1.0, 2.0, 3.0, 4.0, np.nan, 5.0]
    y = [1.0, 2.0, 3.0, 4.0, 9.0, np.inf]
    assert spearman(x, y) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("x", "y"),
    [
        ([1.0], [2.0]),
        ([1.0, 2.0], [3.0, 4.0]),
        ([1.0, 2.0, np.nan], [3.0, 4.0, 5.0]),
        ([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]),
        ([1.0, 2.0, 3.0], [4.0, 4.0, 4.0]),
    ],
    ids=[
        "one-pair",
        "two-pairs",
        "two-finite-pairs",
        "constant-x",
        "constant-y",
    ],
)
def test_spearman_is_nan_without_three_pairs_or_with_a_constant_side(
    x: list[float], y: list[float]
) -> None:
    assert np.isnan(spearman(x, y))


def test_spearman_of_vectors_of_different_length_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        spearman([1.0, 2.0, 3.0], [1.0, 2.0])


COEFFS = (-0.3, -0.15, 0.0, 0.15, 0.3)


def test_safe_slope_is_the_dose_slope() -> None:
    values = np.array([0.1, 0.25, 0.4, 0.5, 0.9])
    assert safe_slope(COEFFS, values) == dose_slope(COEFFS, values)


def test_safe_slope_drops_the_non_finite_points() -> None:
    values = np.array([0.1, np.nan, 0.4, np.nan, 0.9])
    assert safe_slope(COEFFS, values) == dose_slope(COEFFS, values)
    assert safe_slope(COEFFS, values) == pytest.approx(
        (0.9 - 0.1) / (0.3 - -0.3)
    )


def test_safe_slope_is_nan_below_two_finite_points() -> None:
    lonely = np.array([np.nan, np.nan, 0.4, np.nan, np.nan])
    assert np.isnan(safe_slope(COEFFS, lonely))
    assert np.isnan(safe_slope(COEFFS, np.full(5, np.nan)))


def test_safe_slope_keeps_the_length_error() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        safe_slope(COEFFS, np.zeros(3))


WORDS = ((" good", " very"), (" dog", " text"))


def _reader(directions: NDArray[np.float32], sign: float = 1.0) -> RepReader:
    layers = directions.shape[0]
    return RepReader(
        "test",
        tuple(range(1, layers + 1)),
        directions,
        np.zeros_like(directions),
        np.full(layers, sign, dtype=np.float32),
    )


def _random_reader(
    loaded: LoadedModel, seed: int, sign: float = 1.0
) -> RepReader:
    rng = np.random.default_rng(seed)
    unit = rng.standard_normal((3, loaded.d_model)).astype(np.float32)
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    return _reader(unit, sign)


def test_lens_separation_is_one_row_per_reader_and_one_column_per_layer(
    tiny_arch: LoadedModel,
) -> None:
    readers = [_random_reader(tiny_arch, 0), _random_reader(tiny_arch, 1)]
    values = lens_separation(tiny_arch, readers, WORDS)
    assert values.shape == (2, 3)
    unembedding, scale, center = lens_params(tiny_arch)
    positive, _ = pole_ids(tiny_arch, WORDS[0])
    negative, _ = pole_ids(tiny_arch, WORDS[1])
    for row, reader in zip(values, readers, strict=True):
        logits = lens_logits(
            reader.directions, unembedding, scale=scale, center=center
        )
        np.testing.assert_allclose(
            row, pole_separation(logits, positive, negative)
        )


def test_lens_separation_reads_the_oriented_direction(
    tiny_arch: LoadedModel,
) -> None:
    up = lens_separation(tiny_arch, [_random_reader(tiny_arch, 0)], WORDS)
    down = lens_separation(
        tiny_arch, [_random_reader(tiny_arch, 0, sign=-1.0)], WORDS
    )
    np.testing.assert_allclose(down, -up)


def test_lens_separation_is_positive_for_a_direction_toward_the_positive_pole(
    tiny_arch: LoadedModel,
) -> None:
    unembedding, _, _ = lens_params(tiny_arch)
    (good,), (dog,) = (pole_ids(tiny_arch, [w])[0] for w in (" good", " dog"))
    planted = (unembedding[good] - unembedding[dog])[None, :]
    words = ((" good",), (" dog",))
    toward = lens_separation(tiny_arch, [_reader(planted)], words)
    away = lens_separation(tiny_arch, [_reader(-planted)], words)
    assert toward[0, 0] > 1.0
    np.testing.assert_allclose(away, -toward)


def test_lens_separation_without_readers_is_an_empty_matrix(
    tiny_arch: LoadedModel,
) -> None:
    values = lens_separation(tiny_arch, [], WORDS)
    assert values.ndim == 2
    assert values.shape[0] == 0


@pytest.mark.parametrize(
    "words",
    [((" zzzqqq",), (" dog",)), ((" good",), (" zzzqqq",))],
    ids=["no-positive", "no-negative"],
)
def test_lens_separation_needs_a_single_token_word_per_pole(
    tiny_arch: LoadedModel, words: Any
) -> None:
    with pytest.raises(ValueError, match="no pole word is a single token"):
        lens_separation(tiny_arch, [_random_reader(tiny_arch, 0)], words)
