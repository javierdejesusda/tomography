"""Tests for the C1 sentiment steering experiment."""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.experiments.c1_sentiment_steering import Config
from tomography.experiments.c1_sentiment_steering import default_sweep_layers
from tomography.experiments.c1_sentiment_steering import dose_with_none
from tomography.experiments.c1_sentiment_steering import draw_prompts
from tomography.experiments.c1_sentiment_steering import prompt_fragments
from tomography.experiments.c1_sentiment_steering import slope_layer
from tomography.experiments.c1_sentiment_steering import window_around
from tomography.experiments.steering import Arm
from tomography.experiments.steering import GenSettings
from tomography.experiments.steering import dose_response
from tomography.experiments.steering import plain_arm
from tomography.experiments.steering import reader_arm
from tomography.experiments.steering import run_trial
from tomography.model import LoadedModel

PROMPTS = ["The quick brown fox", "The lazy dog", "Honest answers are"]
SETTINGS = GenSettings(
    max_new_tokens=4,
    temperature=0.8,
    top_p=0.95,
    num_samples=2,
    seed=3,
    bos="none",
    batch_size=2,
)


@dataclasses.dataclass(frozen=True)
class StubSource:
    """A fitted-reader stand-in: unit directions and positive signs."""

    layers: tuple[int, ...]
    directions: NDArray[np.float32]
    signs: NDArray[np.float32]


def _source(loaded: LoadedModel) -> StubSource:
    rng = np.random.default_rng(0)
    directions = rng.standard_normal((2, loaded.d_model)).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return StubSource(
        layers=(1, 2), directions=directions, signs=np.ones(2, np.float32)
    )


def _score(texts: Sequence[str]) -> NDArray[np.float32]:
    vowels = [sum(ch in "aeiou" for ch in t) / max(len(t), 1) for t in texts]
    spaces = [t.count(" ") / max(len(t), 1) for t in texts]
    return np.stack([vowels, spaces], axis=1).astype(np.float32)


@pytest.fixture
def full_length(
    tiny_arch: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> LoadedModel:
    """A tiny model whose rows never end early on the shared pad/eos id."""
    monkeypatch.setattr(tiny_arch.model.generation_config, "eos_token_id", None)
    return tiny_arch


@pytest.mark.parametrize(
    ("n_layers", "count", "expected"),
    [
        (12, 6, (1, 3, 5, 8, 10, 12)),
        (28, 6, (1, 6, 12, 17, 23, 28)),
        (3, 6, (1, 2, 3)),
        (12, 4, (1, 5, 8, 12)),
        (12, 1, (1,)),
    ],
    ids=["gpt2", "qwen3", "tiny", "four", "one"],
)
def test_default_sweep_layers_spaces_the_depth(
    n_layers: int, count: int, expected: tuple[int, ...]
) -> None:
    assert default_sweep_layers(n_layers, count) == expected


def test_default_sweep_layers_defaults_to_six() -> None:
    assert default_sweep_layers(12) == default_sweep_layers(12, 6)


def test_default_sweep_layers_rejects_a_count_below_one() -> None:
    with pytest.raises(ValueError, match="count"):
        default_sweep_layers(12, 0)


@pytest.mark.parametrize(
    ("layer", "width", "expected"),
    [
        (6, 3, (5, 6, 7)),
        (1, 3, (1, 2, 3)),
        (12, 3, (10, 11, 12)),
        (2, 3, (1, 2, 3)),
        (6, 4, (4, 5, 6, 7)),
        (6, 2, (5, 6)),
        (1, 2, (1, 2)),
        (12, 2, (11, 12)),
        (6, 1, (6,)),
        (6, 12, tuple(range(1, 13))),
        (1, 12, tuple(range(1, 13))),
    ],
    ids=[
        "centered",
        "low-edge",
        "high-edge",
        "near-low-edge",
        "even-four",
        "even-two",
        "even-low-edge",
        "even-high-edge",
        "single",
        "whole-depth",
        "whole-depth-at-edge",
    ],
)
def test_window_around_is_centered_and_stays_inside_the_depth(
    layer: int, width: int, expected: tuple[int, ...]
) -> None:
    assert window_around(layer, width, 12) == expected


@pytest.mark.parametrize(
    ("layer", "width"),
    [(6, 0), (6, 13), (0, 3), (13, 3)],
    ids=["width-zero", "width-too-wide", "layer-zero", "layer-too-deep"],
)
def test_window_around_rejects_values_outside_the_depth(
    layer: int, width: int
) -> None:
    with pytest.raises(ValueError, match=r"width|layer"):
        window_around(layer, width, 12)


COEFFS = (-0.2, 0.0, 0.2)


def _ramp(slope: float, readers: int = 2) -> NDArray[np.float64]:
    return np.tile(np.array(COEFFS) * slope + 0.5, (readers, 1))


def test_slope_layer_picks_the_layer_with_the_largest_mean_slope() -> None:
    per_seed = np.stack([_ramp(0.1), _ramp(2.0), _ramp(-3.0)])
    assert slope_layer(per_seed, COEFFS, (3, 6, 9)) == 6


def test_slope_layer_averages_the_readers() -> None:
    steep_and_flat = np.stack([_ramp(4.0, 1)[0], _ramp(0.0, 1)[0]])
    steady = _ramp(1.5)
    assert slope_layer(np.stack([steep_and_flat, steady]), COEFFS, (3, 6)) == 3


def test_slope_layer_skips_a_reader_with_fewer_than_two_finite_points() -> None:
    lonely = np.array([0.5, np.nan, np.nan])
    first = np.stack([_ramp(2.0, 1)[0], lonely])
    assert slope_layer(np.stack([first, _ramp(1.5)]), COEFFS, (3, 6)) == 3


def test_slope_layer_never_picks_a_layer_without_a_finite_slope() -> None:
    blank = np.full((2, 3), np.nan)
    per_seed = np.stack([blank, _ramp(-1.0)])
    assert slope_layer(per_seed, COEFFS, (3, 6)) == 6


def test_slope_layer_raises_when_no_layer_has_a_slope() -> None:
    with pytest.raises(ValueError, match="slope"):
        slope_layer(np.full((2, 2, 3), np.nan), COEFFS, (3, 6))


def test_slope_layer_breaks_ties_toward_the_first_layer() -> None:
    per_seed = np.stack([_ramp(1.0), _ramp(1.0), _ramp(0.5)])
    assert slope_layer(per_seed, COEFFS, (3, 6, 9)) == 3


def test_prompt_fragments_cut_the_first_words_and_drop_repeats() -> None:
    texts = [
        "The film is great fun",
        "The film is a bore",
        "  Spaced   out   words here ",
        "A tiny one",
    ]
    assert prompt_fragments(texts, 3) == [
        "The film is",
        "Spaced out words",
        "A tiny one",
    ]


def test_prompt_fragments_drop_empty_texts_and_keep_short_ones() -> None:
    assert prompt_fragments(["", "   ", "Hi", "Hi there you all"], 2) == [
        "Hi",
        "Hi there",
    ]


def test_prompt_fragments_reject_fewer_than_one_word() -> None:
    with pytest.raises(ValueError, match="words"):
        prompt_fragments(["a b c"], 0)


POOL = [f"prompt {i}" for i in range(20)]


def test_draw_prompts_returns_disjoint_lists_from_one_permutation() -> None:
    first, second = draw_prompts(POOL, 5, sets=2, seed=4)
    order = np.random.default_rng(4).permutation(len(POOL))
    assert first == [POOL[i] for i in order[:5]]
    assert second == [POOL[i] for i in order[5:10]]
    assert not set(first) & set(second)


def test_draw_prompts_is_deterministic_and_follows_the_seed() -> None:
    assert draw_prompts(POOL, 5, sets=2, seed=4) == draw_prompts(
        POOL, 5, sets=2, seed=4
    )
    assert draw_prompts(POOL, 5, sets=2, seed=4) != draw_prompts(
        POOL, 5, sets=2, seed=5
    )


def test_draw_prompts_can_use_the_whole_pool() -> None:
    lists = draw_prompts(POOL, 5, sets=4, seed=0)
    assert sorted(p for chosen in lists for p in chosen) == sorted(POOL)


@pytest.mark.parametrize(
    ("n", "sets"),
    [(0, 2), (5, 0), (11, 2)],
    ids=["no-prompts", "no-sets", "short-pool"],
)
def test_draw_prompts_rejects_a_bad_size_or_a_short_pool(
    n: int, sets: int
) -> None:
    with pytest.raises(ValueError, match=r"n |sets|pool"):
        draw_prompts(POOL, n, sets=sets, seed=0)


def _arms(
    loaded: LoadedModel, calls: list[float]
) -> tuple[Callable[[float], Arm], Arm]:
    source = _source(loaded)

    def make(coeff: float) -> Arm:
        calls.append(coeff)
        return reader_arm(
            loaded,
            SETTINGS,
            source,
            layers=[1, 2],
            coeff=coeff,
            norms={1: 5.0, 2: 5.0},
        )

    return make, plain_arm(loaded, SETTINGS)


def test_dose_with_none_equals_dose_response_bit_for_bit(
    full_length: LoadedModel,
) -> None:
    calls: list[float] = []
    make, plain = _arms(full_length, calls)
    coeffs = (-0.3, 0.0, 0.3)
    none = run_trial(plain, PROMPTS, _score, columns=2)

    shared = dose_with_none(make, coeffs, PROMPTS, _score, none, columns=2)
    full = dose_response(make, coeffs, PROMPTS, _score, columns=2)

    assert shared.coeffs == full.coeffs == coeffs
    assert shared.trials[1] is none
    for got, expected in zip(shared.trials, full.trials, strict=True):
        assert got.completions == expected.completions
        np.testing.assert_array_equal(got.scores, expected.scores)
        np.testing.assert_array_equal(got.lengths, expected.lengths)
    assert calls[:2] == [-0.3, 0.3]


def test_dose_with_none_never_builds_the_arm_of_coefficient_zero(
    full_length: LoadedModel,
) -> None:
    calls: list[float] = []
    make, plain = _arms(full_length, calls)
    none = run_trial(plain, PROMPTS, _score, columns=2)

    dose = dose_with_none(
        make, (0.0, 0.1, 0.2), PROMPTS, _score, none, columns=2
    )

    assert calls == [0.1, 0.2]
    assert dose.coeffs == (0.0, 0.1, 0.2)
    assert dose.trials[0] is none
    assert [t.name for t in dose.trials[1:]] == ["0.1", "0.2"]


@pytest.mark.parametrize(
    ("coeffs", "prompts", "message"),
    [
        ((-0.1, 0.1), PROMPTS, "0.0"),
        ((0.0,), PROMPTS, "nonzero"),
        ((-0.1, 0.0, 0.1), PROMPTS[:2], "prompts"),
    ],
    ids=["no-zero", "only-zero", "other-prompts"],
)
def test_dose_with_none_rejects_a_grid_or_prompts_that_do_not_fit(
    full_length: LoadedModel,
    coeffs: tuple[float, ...],
    prompts: list[str],
    message: str,
) -> None:
    calls: list[float] = []
    make, plain = _arms(full_length, calls)
    none = run_trial(plain, PROMPTS, _score, columns=2)

    with pytest.raises(ValueError, match=message):
        dose_with_none(make, coeffs, prompts, _score, none, columns=2)
    assert calls == []


def test_config_defaults() -> None:
    config = Config()
    assert config.experiment == "c1"
    assert config.model == "gpt2"
    assert config.method == "logistic"
    assert config.judge == "sentiment-primary"
    assert config.judge_check == "sentiment-secondary"
    assert config.n_seeds == 10
    assert config.n_steer_seeds == 3
    assert config.n_pairs == 64
    assert config.n_eval_pairs == 256
    assert config.min_words == 3
    assert config.sweep_layers is None
    assert config.window == 3
    assert config.ops == ("linear", "piecewise")
    assert config.truncate_words == 3
    assert len(config.prompt_positive) == len(config.prompt_negative) == 2
    assert config.n_prompts == 100
    assert config.mode == "relative"


def test_config_round_trips_through_a_dict_with_its_tuple_fields() -> None:
    config = Config(
        sweep_layers=(2, 4),
        ops=("piecewise",),
        coeffs=(-0.1, 0.0, 0.1),
        prompt_positive=("Be happy. ",),
        prompt_negative=("Be sad. ", "Be angry. "),
    )
    assert Config.from_dict(config.to_dict()) == config
