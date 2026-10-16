"""Tests for the P7 extraction template versus steering experiment."""

from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Sequence
import contextlib
import dataclasses
from functools import partial
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any
from typing import NamedTuple

from huggingface_hub import snapshot_download
import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.analysis import lens_logits
from tomography.analysis import pole_separation
from tomography.cli import load_experiment
from tomography.control.controllers import mean_residual_norms
from tomography.datasets.records import Example
from tomography.datasets.sst2 import load_sst2
from tomography.datasets.wikitext import load_wikitext
from tomography.experiments import p7_extraction_steering as p7
from tomography.experiments.a2_logit_lens import lens_params
from tomography.experiments.a2_logit_lens import pole_ids
from tomography.experiments.aggregate import read_table
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.c1_sentiment_steering import draw_prompts
from tomography.experiments.c1_sentiment_steering import prompt_fragments
from tomography.experiments.c1_sentiment_steering import slope_layer
from tomography.experiments.effects import read_trials
from tomography.experiments.effects import split_held_out
from tomography.experiments.p7_extraction_steering import CHAT_INSTRUCTION
from tomography.experiments.p7_extraction_steering import FORMATS
from tomography.experiments.p7_extraction_steering import MAX_SERIES
from tomography.experiments.p7_extraction_steering import STORY_PREFIX
from tomography.experiments.p7_extraction_steering import TEMPLATES
from tomography.experiments.p7_extraction_steering import Config
from tomography.experiments.p7_extraction_steering import execute
from tomography.experiments.p7_extraction_steering import extraction_variants
from tomography.experiments.p7_extraction_steering import lens_separation
from tomography.experiments.p7_extraction_steering import plot
from tomography.experiments.p7_extraction_steering import run
from tomography.experiments.p7_extraction_steering import safe_slope
from tomography.experiments.p7_extraction_steering import shift_prompts
from tomography.experiments.p7_extraction_steering import spearman
from tomography.experiments.r1_sentiment import select_examples
from tomography.experiments.steering import Trial
from tomography.experiments.steering import dose_slope
from tomography.experiments.sweep import SERIES
from tomography.experiments.sweep import chat_format_for
from tomography.experiments.variants import READING_TASKS
from tomography.experiments.variants import task_set
from tomography.experiments.variants import template_variants
from tomography.judges import JUDGES
from tomography.judges import load_judge
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.reading.metrics import wilson_ci
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


def test_a_unit_is_readable_from_the_threshold_and_inverted_below_zero() -> (
    None
):
    read = np.array([[0.9, 0.6, 0.59, 0.9]])
    slope = np.array([[-1.0, -0.1, -2.0, 0.3]])
    readable, inverted = p7._inversion(read, slope, 0.6)
    assert readable.tolist() == [[True, True, False, True]]
    assert inverted.tolist() == [[True, True, False, False]]


def test_a_flat_or_missing_slope_is_not_inverted() -> None:
    read = np.full((1, 3), 0.9)
    slope = np.array([[0.0, np.nan, 1e-9]])
    readable, inverted = p7._inversion(read, slope, 0.6)
    assert readable.all()
    assert not inverted.any()


def test_the_inversion_summary_counts_the_units_with_a_wilson_interval() -> (
    None
):
    readable = np.array([[True] * 8, [False] * 8])
    inverted = np.zeros_like(readable)
    inverted[0, :2] = True
    summary = p7._inversion_summary(readable, inverted)
    assert summary == {
        "readable": 8,
        "inverted": 2,
        "rate": 0.25,
        "ci": list(wilson_ci(2, 8)),
    }


def test_the_inversion_summary_has_no_rate_when_nothing_is_readable() -> None:
    none = np.zeros((2, 3), dtype=bool)
    assert p7._inversion_summary(none, none) == {
        "readable": 0,
        "inverted": 0,
        "rate": None,
        "ci": None,
    }


def test_judge_agreement_is_the_share_of_units_with_one_sign_of_slope() -> None:
    first = np.array([1.0, -1.0, 0.5, np.nan, 2.0])
    second = np.array([2.0, 3.0, 0.1, 1.0, np.nan])
    assert p7._sign_agreement(first, second) == pytest.approx(2 / 3)
    assert p7._sign_agreement(first, first) == 1.0


def test_judge_agreement_is_nan_without_a_unit_with_both_slopes() -> None:
    blank = np.full(3, np.nan)
    assert np.isnan(p7._sign_agreement(blank, np.ones(3)))


def test_the_finite_mean_skips_non_finite_values_without_a_warning() -> None:
    values = np.array([[1.0, 3.0, np.nan], [np.nan, np.inf, np.nan]])
    result = p7._finite_mean(values, axis=1)
    assert result[0] == 2.0
    assert np.isnan(result[1])


def test_the_lens_layer_is_the_sweep_layer_of_the_largest_mean_separation() -> (
    None
):
    lens = np.array([[0.1, 0.3], [0.9, 0.7], [0.3, 0.3]])
    slope = np.array([[0.0, 0.2], [2.0, 1.0], [1.0, 1.0]])
    layer, rho = p7._lens_summary(lens, slope, (2, 5, 8))
    assert layer == 5
    assert rho == pytest.approx(1.0)


def test_the_lens_layer_of_a_tie_is_the_first_and_rho_needs_three_layers() -> (
    None
):
    lens = np.array([[0.5], [0.5]])
    slope = np.array([[0.2], [0.4]])
    layer, rho = p7._lens_summary(lens, slope, (2, 5))
    assert layer == 2
    assert np.isnan(rho)


def test_the_config_defaults_and_round_trip() -> None:
    config = Config()
    assert (config.experiment, config.model, config.method) == (
        "p7",
        "gpt2",
        "logistic",
    )
    assert (config.judge, config.judge_check) == (
        "sentiment-primary",
        "sentiment-secondary",
    )
    assert (config.templates, config.formats) == (TEMPLATES, FORMATS)
    assert (config.n_seeds, config.n_steer_seeds, config.n_pairs) == (10, 2, 64)
    assert (config.n_eval_pairs, config.min_words, config.n_prompts) == (
        256,
        3,
        50,
    )
    assert config.coeffs == (-0.3, -0.15, 0.0, 0.15, 0.3)
    assert config.sweep_layers is None
    assert (config.read_threshold, config.truncate_words) == (0.6, 3)
    assert MAX_SERIES == 8
    custom = Config(
        templates=("paper", "scaffold_1"),
        formats=("raw", "story"),
        sweep_layers=(2, 5),
        read_threshold=0.7,
    )
    assert Config.from_dict(custom.to_dict()) == custom


SENTENCE_WORDS = tuple(
    [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "eta",
        "theta",
        "iota",
        "kappa",
        "lambda",
        "mu",
        "nu",
        "xi",
        "omicron",
        "pi",
        "rho",
        "sigma",
        "tau",
        "upsilon",
    ]
)
SWEEP = (1, 2, 3)
TEMPLATE_NAMES = ("paper", "no_template")
FORMAT_NAMES = ("raw", "chat")
READERS = (0, 1)
DOSE_COEFFS = (-0.2, 0.0, 0.2)
NONZERO = tuple(c for c in DOSE_COEFFS if c != 0.0)


def _sst2(n: int = 80) -> list[Example]:
    rng = np.random.default_rng(2)
    return [
        Example(
            uid=f"s{i}",
            text=" ".join(rng.choice(SENTENCE_WORDS, 6)),
            label=i % 2,
            split="train" if (i // 2) % 4 else "val",
        )
        for i in range(n)
    ]


def _texts(n: int = 60) -> list[str]:
    return [
        " ".join(np.random.default_rng(i).choice(SENTENCE_WORDS, 12))
        for i in range(n)
    ]


def _score(texts: Any) -> NDArray[np.float32]:
    vowels = [sum(ch in "aeiou" for ch in t) / max(len(t), 1) for t in texts]
    spaces = [t.count(" ") / max(len(t), 1) for t in texts]
    return np.stack([vowels, spaces], axis=1).astype(np.float32)


def _config(**overrides: Any) -> Config:
    base = Config(
        model="tiny",
        templates=TEMPLATE_NAMES,
        formats=FORMAT_NAMES,
        n_seeds=3,
        n_steer_seeds=2,
        n_pairs=8,
        n_eval_pairs=8,
        sweep_layers=SWEEP,
        coeffs=DOSE_COEFFS,
        n_prompts=3,
        n_calibration=10,
        n_neutral=8,
        max_new_tokens=4,
        batch_size=8,
        n_boot=50,
    )
    return dataclasses.replace(base, **overrides)


def _execute(loaded: LoadedModel, tmp_path: Path, **overrides: Any) -> RunDir:
    config = _config(**overrides)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, loaded, _sst2(), _texts(), _score, WORDS)
    return run_dir


def _spy_reading_sweep(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    calls: list[Any] = []
    real = p7.reading_sweep

    def spy(loaded: Any, train_sets: Any, eval_sets: Any, **kwargs: Any) -> Any:
        result = real(loaded, train_sets, eval_sets, **kwargs)
        calls.append(
            SimpleNamespace(
                train=train_sets, evals=eval_sets, kwargs=kwargs, result=result
            )
        )
        return result

    monkeypatch.setattr(p7, "reading_sweep", spy)
    return calls


def _forbid_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    def reached(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the model was reached")

    monkeypatch.setattr(p7, "reading_sweep", reached)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"templates": ()}, "templates"),
        (
            {"templates": ("paper", *(f"paraphrase_{i}" for i in range(8)))},
            "templates",
        ),
        ({"templates": ("paper", "paper")}, "templates must be distinct"),
        ({"templates": ("paper", "shuffle")}, "not a variant kind"),
        ({"templates": ("paper", "framing_swap")}, "no function framing"),
        ({"templates": ("paper", "scaffold_9")}, "no variant of that name"),
        ({"templates": ("no_template",)}, "kinds must include 'paper'"),
        ({"formats": ()}, "formats"),
        ({"formats": ("raw", "raw")}, "formats"),
        ({"formats": ("raw", "poem")}, "unknown formats"),
        ({"n_steer_seeds": 0}, "n_steer_seeds"),
        ({"n_steer_seeds": 4}, "n_steer_seeds"),
        ({"coeffs": (-0.2, 0.2)}, "coeffs"),
        ({"coeffs": (0.0,)}, "coeffs"),
        ({"sweep_layers": ()}, "sweep_layers"),
        ({"sweep_layers": (0, 2)}, "sweep_layers"),
        ({"sweep_layers": (1, 4)}, "sweep_layers"),
        ({"sweep_layers": (1, 1)}, "sweep_layers"),
        ({"sweep_layers": (1,) * 9}, "sweep_layers"),
        ({"read_threshold": 0.49}, "read_threshold"),
        ({"read_threshold": 1.0}, "read_threshold"),
        ({"n_prompts": 100}, "pool"),
        ({"n_calibration": 0}, "n_calibration"),
        ({"temperature": None}, "needs a temperature"),
        ({"n_pairs": 31}, "n_pairs is 31 but at most 30 pairs exist"),
        ({"n_eval_pairs": 11}, "n_eval_pairs is 11 but at most 10 pairs"),
        ({"n_pairs": 0}, "n_pairs must be at least 1, got 0"),
    ],
    ids=[
        "no-template",
        "nine-templates",
        "template-repeated",
        "unknown-kind",
        "framing-swap",
        "index-past-the-last",
        "no-paper",
        "no-format",
        "format-repeated",
        "unknown-format",
        "steer-seeds-zero",
        "steer-seeds-above-seeds",
        "no-zero",
        "only-zero",
        "no-sweep-layer",
        "sweep-layer-zero",
        "sweep-layer-too-deep",
        "sweep-layer-repeated",
        "nine-sweep-layers",
        "threshold-below-chance",
        "threshold-one",
        "prompt-pool-too-small",
        "no-calibration-text",
        "sampling-without-a-temperature",
        "train-pairs-above-the-split",
        "eval-pairs-above-the-split",
        "no-train-pairs",
    ],
)
def test_an_invalid_config_raises_before_the_model_runs(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
    message: str,
) -> None:
    _forbid_reading(monkeypatch)
    with pytest.raises(ValueError, match=message):
        _execute(tiny_arch, tmp_path, **overrides)


def test_pole_words_without_a_single_token_raise_before_the_model_runs(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_reading(monkeypatch)
    config = _config()
    run_dir = RunDir.create(config, out_root=tmp_path)
    words = ((" good",), (" zzzqqq",))
    with pytest.raises(ValueError, match="no pole word is a single token"):
        execute(config, run_dir, tiny_arch, _sst2(), _texts(), _score, words)


def test_too_few_held_out_texts_raise_before_the_model_runs(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_reading(monkeypatch)
    config = _config()
    run_dir = RunDir.create(config, out_root=tmp_path)
    with pytest.raises(ValueError, match="need 18 texts"):
        execute(config, run_dir, tiny_arch, _sst2(), _texts(5), _score, WORDS)


def test_every_template_reads_the_same_seeds_and_items(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _spy_reading_sweep(monkeypatch)
    config = _config()
    _execute(tiny_arch, tmp_path)
    assert len(calls) == 2
    task = READING_TASKS["sentiment"]
    train = select_examples(_sst2(), split="train", min_words=3)
    val = select_examples(_sst2(), split="val", min_words=3)
    variants = extraction_variants(PLAIN, config.templates)
    for call, variant in zip(calls, variants, strict=True):
        expected = [
            task_set(
                task,
                variant,
                train,
                n_pairs=8,
                seed=derive_seed(config.seed, "train", i),
            )
            for i in range(config.n_seeds)
        ]
        assert [s.texts for s in call.train] == [s.texts for s in expected]
        assert list(call.evals) == ["val"]
        item = task_set(
            task, variant, val, n_pairs=8, seed=derive_seed(config.seed, "eval")
        )
        assert call.evals["val"].texts == item.texts
        assert call.kwargs == {
            "method": "logistic",
            "seed": config.seed,
            "bos": None,
            "batch_size": 8,
        }
    assert calls[0].evals["val"].uids == calls[1].evals["val"].uids
    assert calls[0].train[0].texts != calls[1].train[0].texts


def test_the_reading_curves_hold_the_reader_accuracy_of_each_template(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _spy_reading_sweep(monkeypatch)
    run_dir = _execute(tiny_arch, tmp_path)
    curves = run_dir.read_arrays("curves_read")
    sidecar = run_dir.read_json("curves_read")
    assert isinstance(sidecar, dict)
    assert sidecar["series"] == ["paper", "no_template"]
    assert sidecar["chance"] == 0.5
    assert sidecar["n_layers"] == 3
    assert curves["x"].tolist() == [1.0, 2.0, 3.0]
    assert curves["per_seed"].shape == (2, 3, 3)
    reader = SERIES.index("reader")
    for t, call in enumerate(calls):
        np.testing.assert_array_equal(
            curves["per_seed"][t], call.result.per_seed("val")[reader]
        )


def test_the_lens_curves_hold_the_separation_of_the_steering_readers(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _spy_reading_sweep(monkeypatch)
    run_dir = _execute(tiny_arch, tmp_path)
    curves = run_dir.read_arrays("curves_lens")
    sidecar = run_dir.read_json("curves_lens")
    assert isinstance(sidecar, dict)
    assert sidecar["series"] == ["paper", "no_template"]
    assert sidecar["chance"] == 0.0
    assert curves["per_seed"].shape == (2, 2, 3)
    for t, call in enumerate(calls):
        np.testing.assert_allclose(
            curves["per_seed"][t],
            lens_separation(tiny_arch, call.result.readers[:2], WORDS),
        )


READ = {
    "paper": ((0.9, 0.7, 0.4), (0.5, 0.65, 0.3), (0.1, 0.1, 0.1)),
    "no_template": ((0.5, 0.6, 0.8), (0.7, 0.55, 0.9), (0.2, 0.2, 0.2)),
}


@contextlib.contextmanager
def _planted_reading() -> Iterator[None]:
    """Replaces the reader accuracy of each template by the `READ` values.

    `READ[template]` holds one row per seed and one column per layer. The
    third seed does not steer.
    """
    real = p7.reading_sweep
    planted = iter(READ.values())

    def sweep(loaded: Any, train: Any, evals: Any, **kwargs: Any) -> Any:
        result = real(loaded, train, evals, **kwargs)
        accuracy = result.accuracy.copy()
        accuracy[:, SERIES.index("reader")] = np.array(next(planted))
        return dataclasses.replace(result, accuracy=accuracy)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(p7, "reading_sweep", sweep)
        yield


class Finished(NamedTuple):
    config: Config
    run_dir: RunDir
    loaded: LoadedModel


_FINISHED: dict[str, Finished] = {}


@pytest.fixture
def finished(
    tiny_arch: LoadedModel, tmp_path_factory: pytest.TempPathFactory
) -> Finished:
    """One tiny run per architecture, with planted reading accuracies."""
    key = type(tiny_arch.model).__name__
    if key not in _FINISHED:
        config = _config()
        run_dir = RunDir.create(config, out_root=tmp_path_factory.mktemp("p7"))
        with _planted_reading():
            execute(
                config, run_dir, tiny_arch, _sst2(), _texts(), _score, WORDS
            )
        _FINISHED[key] = Finished(config, run_dir, tiny_arch)
    return _FINISHED[key]


def _trials(run_dir: RunDir, stem: str) -> dict[str, Trial]:
    return {t.name: t for t in read_trials(run_dir, stem)}


def _dose(
    trials: dict[str, Trial], prefix: str, column: int
) -> NDArray[np.float64]:
    """Rebuilds a dose response, with the unsteered trial at coefficient 0."""
    return np.array(
        [
            (
                trials["none"] if c == 0.0 else trials[f"{prefix}/{c:g}"]
            ).mean_score()[column]
            for c in DOSE_COEFFS
        ]
    )


def test_the_sweep_trials_are_named_after_template_layer_and_reader(
    finished: Finished,
) -> None:
    trials = read_trials(finished.run_dir, "trials_sweep")
    assert [t.name for t in trials] == [
        "none",
        *(
            f"{template}/layer{layer}/{reader}/{c:g}"
            for template in TEMPLATE_NAMES
            for layer in SWEEP
            for reader in READERS
            for c in NONZERO
        ),
    ]
    assert trials[0].scores.shape == (3, 2, 2)


@pytest.mark.parametrize("fmt", FORMAT_NAMES)
def test_the_transfer_trials_are_named_after_template_and_reader(
    finished: Finished, fmt: str
) -> None:
    trials = read_trials(finished.run_dir, f"trials_transfer_{fmt}")
    assert [t.name for t in trials] == [
        "none",
        *(
            f"{template}/{reader}/{c:g}"
            for template in TEMPLATE_NAMES
            for reader in READERS
            for c in NONZERO
        ),
    ]


def test_selection_and_report_prompts_are_disjoint_draws(
    finished: Finished,
) -> None:
    val = select_examples(_sst2(), split="val", min_words=3)
    pool = prompt_fragments([e.text for e in val], 3)
    select, report = draw_prompts(
        pool, 3, sets=2, seed=derive_seed(finished.config.seed, "prompts")
    )
    assert not set(select) & set(report)
    sweep = read_trials(finished.run_dir, "trials_sweep")
    assert {t.prompts for t in sweep} == {tuple(select)}
    chat = chat_format_for(finished.loaded)
    for fmt in FORMAT_NAMES:
        trials = read_trials(finished.run_dir, f"trials_transfer_{fmt}")
        expected = tuple(shift_prompts(report, fmt, chat))
        assert {t.prompts for t in trials} == {expected}


def test_each_prompt_list_has_one_unsteered_trial(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arms: list[str] = []
    real = p7.run_trial

    def spy(arm: Any, *args: Any, **kwargs: Any) -> Any:
        arms.append(arm.name)
        return real(arm, *args, **kwargs)

    monkeypatch.setattr(p7, "run_trial", spy)
    run_dir = _execute(tiny_arch, tmp_path)
    assert arms == ["none"] * 3
    for stem in ("trials_sweep", "trials_transfer_raw", "trials_transfer_chat"):
        names = [t.name for t in read_trials(run_dir, stem)]
        assert names.count("none") == 1


@pytest.mark.parametrize(
    ("stem", "column"), [("curves_slope", 0), ("curves_slope_check", 1)]
)
def test_the_slope_curves_are_the_dose_slopes_of_the_sweep_trials(
    finished: Finished, stem: str, column: int
) -> None:
    arrays = finished.run_dir.read_arrays(stem)
    sidecar = finished.run_dir.read_json(stem)
    assert isinstance(sidecar, dict)
    assert sidecar["series"] == list(TEMPLATE_NAMES)
    assert sidecar["chance"] == 0.0
    assert sidecar["xlabel"] == "Layer"
    assert sidecar["n_layers"] == 3
    assert ("primary", "check")[column] in sidecar["ylabel"]
    assert arrays["x"].tolist() == [1.0, 2.0, 3.0]
    trials = _trials(finished.run_dir, "trials_sweep")
    expected = np.array(
        [
            [
                [
                    dose_slope(
                        DOSE_COEFFS,
                        _dose(trials, f"{template}/layer{layer}/{r}", column),
                    )
                    for layer in SWEEP
                ]
                for r in READERS
            ]
            for template in TEMPLATE_NAMES
        ]
    )
    assert arrays["per_seed"].shape == (2, 2, 3)
    np.testing.assert_allclose(arrays["per_seed"], expected, atol=1e-6)


def _spy_slope_layer(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    calls: list[Any] = []
    real = p7.slope_layer

    def spy(per_seed: Any, coeffs: Any, layers: Any) -> int:
        layer = real(per_seed, coeffs, layers)
        calls.append(
            SimpleNamespace(per_seed=per_seed, layers=layers, out=layer)
        )
        return layer

    monkeypatch.setattr(p7, "slope_layer", spy)
    return calls


def _spy_reader_arm(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    real = p7.reader_arm

    def spy(
        loaded: LoadedModel, settings: Any, reader: Any, **kwargs: Any
    ) -> Any:
        calls.append({"reader": reader, **kwargs})
        return real(loaded, settings, reader, **kwargs)

    monkeypatch.setattr(p7, "reader_arm", spy)
    return calls


def test_the_steering_layer_of_a_template_is_where_the_primary_slope_peaks(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layers = _spy_slope_layer(monkeypatch)
    arms = _spy_reader_arm(monkeypatch)
    run_dir = _execute(tiny_arch, tmp_path)
    trials = _trials(run_dir, "trials_sweep")
    assert len(layers) == 2
    for template, call in zip(TEMPLATE_NAMES, layers, strict=True):
        assert call.layers == SWEEP
        primary = np.array(
            [
                [
                    _dose(trials, f"{template}/layer{layer}/{r}", 0)
                    for r in READERS
                ]
                for layer in SWEEP
            ]
        )
        np.testing.assert_allclose(call.per_seed, primary)
    chosen = [call.out for call in layers]
    expected = [
        [chosen[t]]
        for _ in FORMAT_NAMES
        for t in (0, 1)
        for _ in READERS
        for _ in NONZERO
    ]
    assert [c["layers"] for c in arms[24:]] == expected


def test_each_template_steers_the_transfer_at_its_own_layer(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    planted = iter((1, 3))
    monkeypatch.setattr(p7, "slope_layer", lambda *args: next(planted))
    arms = _spy_reader_arm(monkeypatch)
    run_dir = _execute(tiny_arch, tmp_path)
    expected = [
        [(1, 3)[t]]
        for _ in FORMAT_NAMES
        for t in (0, 1)
        for _ in READERS
        for _ in NONZERO
    ]
    assert [c["layers"] for c in arms[24:]] == expected
    assert [row["steer_layer"] for row in _table(run_dir)] == [1, 3]


def test_the_arms_use_the_sweep_layers_and_the_first_readers(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arms = _spy_reader_arm(monkeypatch)
    reads = _spy_reading_sweep(monkeypatch)
    _execute(tiny_arch, tmp_path)
    sweep = arms[:24]
    seen: list[tuple[tuple[int, ...], int]] = []
    for call in sweep:
        key = (tuple(call["layers"]), id(call["reader"]))
        if key not in seen:
            seen.append(key)
    first = {t: [id(r) for r in reads[t].result.readers[:2]] for t in (0, 1)}
    assert seen == [
        ((layer,), first[t][r])
        for t in (0, 1)
        for layer in SWEEP
        for r in READERS
    ]
    assert all(c["coeff"] != 0.0 for c in arms)
    steering = {id(r) for call in reads for r in call.result.readers[:2]}
    assert {id(c["reader"]) for c in arms} == steering
    held = split_held_out(_texts(), n_calibration=10, n_neutral=8, seed=0)
    norms = mean_residual_norms(
        tiny_arch, held.calibration, [1, 2, 3], batch_size=8
    )
    assert all(c["norms"] == norms for c in arms)


def test_the_steering_mode_mask_and_norm_follow_the_config(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arms = _spy_reader_arm(monkeypatch)
    _execute(
        tiny_arch,
        tmp_path,
        mode="absolute",
        mask="all",
        preserve_norm=True,
        op="piecewise",
        templates=("paper",),
        formats=("raw",),
        n_steer_seeds=1,
        sweep_layers=(2,),
    )
    assert arms
    assert {c["mode"] for c in arms} == {"absolute"}
    assert {c["mask"] for c in arms} == {"all"}
    assert {c["preserve_norm"] for c in arms} == {True}
    assert {c["op"] for c in arms} == {"piecewise"}


@pytest.mark.parametrize("fmt", FORMAT_NAMES)
def test_the_transfer_matrix_is_the_mean_primary_slope_of_the_transfer_trials(
    finished: Finished, fmt: str
) -> None:
    arrays = finished.run_dir.read_arrays("matrix_transfer")
    sidecar = finished.run_dir.read_json("matrix_transfer")
    assert isinstance(sidecar, dict)
    assert sidecar["rows"] == list(TEMPLATE_NAMES)
    assert sidecar["cols"] == list(FORMAT_NAMES)
    assert arrays["values"].shape == (2, 2)
    trials = _trials(finished.run_dir, f"trials_transfer_{fmt}")
    for t, template in enumerate(TEMPLATE_NAMES):
        slopes = [
            dose_slope(DOSE_COEFFS, _dose(trials, f"{template}/{r}", 0))
            for r in READERS
        ]
        assert arrays["values"][t, FORMAT_NAMES.index(fmt)] == pytest.approx(
            np.mean(slopes), abs=1e-6
        )


_table = partial(read_table, stem="table")


def _summary(run_dir: RunDir) -> dict[str, Any]:
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    return summary


def _units(run_dir: RunDir, sweep: tuple[int, ...] = SWEEP) -> SimpleNamespace:
    """The (template, sweep layer, reader) units, rebuilt from the files."""
    slope = run_dir.read_arrays("curves_slope")["per_seed"]
    check = run_dir.read_arrays("curves_slope_check")["per_seed"]
    lens = run_dir.read_arrays("curves_lens")["per_seed"]
    read = np.array(
        [
            [[READ[template][r][layer - 1] for r in READERS] for layer in sweep]
            for template in TEMPLATE_NAMES
        ]
    )
    return SimpleNamespace(
        read=read,
        slope=slope.transpose(0, 2, 1),
        check=check.transpose(0, 2, 1),
        lens=lens.transpose(0, 2, 1),
    )


def test_execute_writes_twenty_files(finished: Finished) -> None:
    names = sorted(p.name for p in finished.run_dir.path.iterdir())
    stems = (
        "curves_read",
        "curves_lens",
        "curves_slope",
        "curves_slope_check",
        "matrix_transfer",
        "trials_sweep",
        "trials_transfer_raw",
        "trials_transfer_chat",
    )
    assert names == sorted(
        [
            "config.json",
            "meta.json",
            "summary.json",
            "table.json",
            *(f"{stem}.{ext}" for stem in stems for ext in ("npz", "json")),
        ]
    )


def test_the_table_has_one_row_per_template_and_the_documented_columns(
    finished: Finished,
) -> None:
    table = finished.run_dir.read_json("table")
    assert isinstance(table, dict)
    assert table["columns"] == [
        "template",
        "read_layer",
        "read_depth",
        "read_accuracy",
        "steer_layer",
        "steer_depth",
        "steer_slope",
        "lens_layer",
        "lens_rho",
        "n_readable",
        "n_inverted",
    ]
    assert [row["template"] for row in _table(finished.run_dir)] == list(
        TEMPLATE_NAMES
    )


def test_the_table_reads_each_template_at_its_best_validation_layer(
    finished: Finished,
) -> None:
    rows = _table(finished.run_dir)
    assert [row["read_layer"] for row in rows] == [1, 3]
    assert [row["read_depth"] for row in rows] == pytest.approx([1 / 3, 1.0])
    assert [row["read_accuracy"] for row in rows] == pytest.approx(
        [0.5, 1.9 / 3]
    )


def test_the_table_steers_each_template_at_its_steepest_sweep_layer(
    finished: Finished,
) -> None:
    trials = _trials(finished.run_dir, "trials_sweep")
    slopes = finished.run_dir.read_arrays("curves_slope")["per_seed"]
    for t, (template, row) in enumerate(
        zip(TEMPLATE_NAMES, _table(finished.run_dir), strict=True)
    ):
        primary = np.array(
            [
                [
                    _dose(trials, f"{template}/layer{layer}/{r}", 0)
                    for r in READERS
                ]
                for layer in SWEEP
            ]
        )
        layer = slope_layer(primary, DOSE_COEFFS, SWEEP)
        assert row["steer_layer"] == layer
        assert row["steer_depth"] == pytest.approx(layer / 3)
        assert row["steer_slope"] == pytest.approx(
            slopes[t, :, SWEEP.index(layer)].mean()
        )


def test_the_table_places_the_lens_peak_and_its_rank_correlation(
    finished: Finished,
) -> None:
    units = _units(finished.run_dir)
    for t, row in enumerate(_table(finished.run_dir)):
        lens = units.lens[t].mean(axis=1)
        assert row["lens_layer"] == SWEEP[int(np.argmax(lens))]
        assert row["lens_rho"] == pytest.approx(
            spearman(lens, units.slope[t].mean(axis=1)), nan_ok=True
        )


def test_the_table_counts_the_readable_and_the_inverted_units(
    finished: Finished,
) -> None:
    units = _units(finished.run_dir)
    readable = units.read >= 0.6
    inverted = readable & (units.slope < 0)
    rows = _table(finished.run_dir)
    assert [row["n_readable"] for row in rows] == [3, 4]
    assert [row["n_inverted"] for row in rows] == inverted.sum(
        axis=(1, 2)
    ).tolist()


def test_the_summary_lists_the_run(finished: Finished) -> None:
    summary = _summary(finished.run_dir)
    assert list(summary) == [
        "templates",
        "formats",
        "sweep_layers",
        "n_prompts",
        "inversion",
        "read_steer_rho",
        "lens_steer_rho",
        "lens_agreement",
        "layer_gap",
        "judge_sign_agreement",
    ]
    assert summary["templates"] == list(TEMPLATE_NAMES)
    assert summary["formats"] == list(FORMAT_NAMES)
    assert summary["sweep_layers"] == list(SWEEP)
    assert summary["n_prompts"] == 3


def test_the_inversion_is_the_readable_units_with_a_negative_slope(
    finished: Finished,
) -> None:
    units = _units(finished.run_dir)
    readable = units.read >= 0.6
    inverted = int((readable & (units.slope < 0)).sum())
    inversion = _summary(finished.run_dir)["inversion"]
    assert inversion == {
        "readable": 7,
        "inverted": inverted,
        "rate": inverted / 7,
        "ci": list(wilson_ci(inverted, 7)),
    }
    rows = _table(finished.run_dir)
    assert sum(row["n_readable"] for row in rows) == inversion["readable"]
    assert sum(row["n_inverted"] for row in rows) == inversion["inverted"]


def test_the_correlations_pool_every_unit(finished: Finished) -> None:
    units = _units(finished.run_dir)
    summary = _summary(finished.run_dir)
    assert summary["read_steer_rho"] == pytest.approx(
        spearman(units.read.ravel(), units.slope.ravel()), nan_ok=True
    )
    assert summary["lens_steer_rho"] == pytest.approx(
        spearman(units.lens.ravel(), units.slope.ravel()), nan_ok=True
    )


def test_the_agreements_and_the_layer_gap_follow_the_table(
    finished: Finished,
) -> None:
    rows = _table(finished.run_dir)
    summary = _summary(finished.run_dir)
    assert summary["lens_agreement"] == pytest.approx(
        np.mean([row["lens_layer"] == row["steer_layer"] for row in rows])
    )
    assert summary["layer_gap"] == pytest.approx(
        np.mean([abs(row["read_depth"] - row["steer_depth"]) for row in rows])
    )
    units = _units(finished.run_dir)
    same = np.sign(units.slope) == np.sign(units.check)
    assert summary["judge_sign_agreement"] == pytest.approx(same.mean())


def test_nothing_readable_leaves_the_inversion_rate_empty(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    with _planted_reading():
        run_dir = _execute(
            tiny_arch,
            tmp_path,
            read_threshold=0.99,
            formats=("raw",),
            sweep_layers=(2,),
            n_steer_seeds=1,
        )
    assert _summary(run_dir)["inversion"] == {
        "readable": 0,
        "inverted": 0,
        "rate": None,
        "ci": None,
    }
    assert [row["n_readable"] for row in _table(run_dir)] == [0, 0]


def test_the_cli_discovers_the_experiment() -> None:
    experiment = load_experiment("p7")
    assert experiment.config_cls is Config
    assert experiment.run is run
    assert experiment.plot is plot


class StubJudge:
    def __init__(self, column: int) -> None:
        self.column = column

    def score(self, texts: Any) -> NDArray[np.float32]:
        return np.ascontiguousarray(_score(texts)[:, self.column])


def test_run_scores_with_both_judges_in_two_columns(
    finished: Finished, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: dict[str, list[str]] = {"model": [], "judge": []}
    executed: list[tuple[Any, ...]] = []

    def load_model(key: str) -> LoadedModel:
        asked["model"].append(key)
        return finished.loaded

    def load_judge(key: str) -> StubJudge:
        asked["judge"].append(key)
        return StubJudge(0 if key == "sentiment-primary" else 1)

    monkeypatch.setattr(p7, "load_model", load_model)
    monkeypatch.setattr(p7, "load_judge", load_judge)
    monkeypatch.setattr(p7, "load_sst2", _sst2)
    monkeypatch.setattr(
        p7, "load_wikitext", lambda: [SimpleNamespace(text=t) for t in _texts()]
    )
    monkeypatch.setattr(p7, "execute", lambda *args: executed.append(args))
    run_dir = RunDir.create(finished.config, out_root=tmp_path)
    run(finished.config, run_dir)

    assert asked == {
        "model": ["tiny"],
        "judge": ["sentiment-primary", "sentiment-secondary"],
    }
    [(config, target, loaded, sst2, texts, score)] = executed
    assert config is finished.config
    assert target is run_dir
    assert loaded is finished.loaded
    assert sst2 == _sst2()
    assert texts == _texts()
    np.testing.assert_array_equal(score(texts), _score(texts))


PLOT_STEMS = (
    "curves_read",
    "curves_lens",
    "curves_slope",
    "curves_slope_check",
    "matrix_transfer",
    "read_vs_steer",
)


def test_plot_draws_the_four_curves_the_matrix_and_the_scatter(
    finished: Finished, tmp_path: Path
) -> None:
    paths = plot(finished.run_dir, tmp_path / "figs")
    assert [p.name for p in paths] == [
        f"{stem}.{ext}" for stem in PLOT_STEMS for ext in ("pdf", "png", "svg")
    ]
    assert all(p.stat().st_size > 0 for p in paths)


def test_the_transfer_matrix_is_drawn_on_a_diverging_scale(
    finished: Finished, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[dict[str, Any]] = []
    real = p7.plot_matrix_file

    def spy(*args: Any, **kwargs: Any) -> Any:
        asked.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(p7, "plot_matrix_file", spy)
    plot(finished.run_dir, tmp_path)
    assert len(asked) == 1
    assert asked[0]["stem"] == "matrix_transfer"
    assert asked[0]["kind"] == "diverging"


def _scatter(
    run_dir: RunDir, out_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> Any:
    """Plots a run and returns the axes of the read versus steer figure."""
    saved: dict[str, Any] = {}
    real = p7.save_figure

    def spy(fig: Any, out_dir: Any, name: str, **kwargs: Any) -> Any:
        saved[name] = fig
        return real(fig, out_dir, name, **kwargs)

    monkeypatch.setattr(p7, "save_figure", spy)
    plot(run_dir, out_dir)
    return saved["read_vs_steer"].axes[0]


def _assert_points(ax: Any, units: SimpleNamespace) -> None:
    lines = {line.get_label(): line for line in ax.lines}
    for t, template in enumerate(TEMPLATE_NAMES):
        x, y = lines[template].get_data()
        np.testing.assert_allclose(x, units.read[t].T.ravel())
        np.testing.assert_allclose(y, units.slope[t].T.ravel())


def test_the_scatter_puts_each_unit_at_its_reading_accuracy_and_slope(
    finished: Finished, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ax = _scatter(finished.run_dir, tmp_path, monkeypatch)
    _assert_points(ax, _units(finished.run_dir))
    lines = {line.get_label(): line for line in ax.lines}
    colors = {lines[template].get_color() for template in TEMPLATE_NAMES}
    assert len(colors) == len(TEMPLATE_NAMES)
    assert lines["_zero_slope"].get_ydata()[0] == 0.0
    assert lines["_threshold"].get_xdata()[0] == finished.config.read_threshold
    assert ax.get_xlabel() == "Reading accuracy"
    assert ax.get_ylabel() == "Dose slope (primary judge)"


def test_each_unit_reads_at_its_own_sweep_layer(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sweep = (2, 3)
    with _planted_reading():
        run_dir = _execute(
            tiny_arch, tmp_path, sweep_layers=sweep, formats=("raw",)
        )
    units = _units(run_dir, sweep)
    assert [row["n_readable"] for row in _table(run_dir)] == [2, 3]
    assert _summary(run_dir)["inversion"]["readable"] == 5
    _assert_points(_scatter(run_dir, tmp_path / "figs", monkeypatch), units)


def _cached[T](load: Callable[[], T]) -> T:
    try:
        return load()
    except FileNotFoundError as error:
        pytest.skip(f"cache missing: {error}")


def _cached_judge(key: str) -> Any:
    spec = JUDGES[key]
    _cached(
        lambda: snapshot_download(
            spec.repo_id, revision=spec.revision, local_files_only=True
        )
    )
    return load_judge(key)


@pytest.mark.slow
def test_p7_on_gpt2_prints_the_slopes_under_both_judges(tmp_path: Path) -> None:
    sst2 = _cached(lambda: load_sst2(offline=True))
    texts = [e.text for e in _cached(lambda: load_wikitext(offline=True))]
    first = _cached_judge("sentiment-primary")
    second = _cached_judge("sentiment-secondary")

    def score(batch: Sequence[str]) -> NDArray[np.float32]:
        return np.column_stack([first.score(batch), second.score(batch)])

    config = Config(
        templates=("paper", "no_template"),
        formats=("raw",),
        n_seeds=3,
        n_steer_seeds=1,
        n_pairs=32,
        n_eval_pairs=128,
        sweep_layers=(4, 8),
        coeffs=(-0.2, 0.0, 0.2),
        n_prompts=10,
        num_samples=1,
    )
    run_dir = RunDir.create(config, out_root=tmp_path)
    started = time.perf_counter()
    execute(config, run_dir, load_model("gpt2"), sst2, texts, score)
    seconds = time.perf_counter() - started

    generations = sum(
        int(trial.scores.shape[0] * trial.scores.shape[1])
        for stem in ("trials_sweep", "trials_transfer_raw")
        for trial in read_trials(run_dir, stem)
    )
    print(f"{seconds / generations:.3f} s per generation, whole run")
    print(_summary(run_dir))
    for row in _table(run_dir):
        print(row)
    assert _summary(run_dir)["sweep_layers"] == [4, 8]
