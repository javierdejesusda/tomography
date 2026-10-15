"""Tests for the P7 extraction template versus steering experiment."""

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.analysis import lens_logits
from tomography.analysis import pole_separation
from tomography.datasets.records import Example
from tomography.experiments import p7_extraction_steering as p7
from tomography.experiments.a2_logit_lens import lens_params
from tomography.experiments.a2_logit_lens import pole_ids
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.p7_extraction_steering import CHAT_INSTRUCTION
from tomography.experiments.p7_extraction_steering import FORMATS
from tomography.experiments.p7_extraction_steering import MAX_SERIES
from tomography.experiments.p7_extraction_steering import STORY_PREFIX
from tomography.experiments.p7_extraction_steering import TEMPLATES
from tomography.experiments.p7_extraction_steering import Config
from tomography.experiments.p7_extraction_steering import execute
from tomography.experiments.p7_extraction_steering import extraction_variants
from tomography.experiments.p7_extraction_steering import lens_separation
from tomography.experiments.p7_extraction_steering import safe_slope
from tomography.experiments.p7_extraction_steering import shift_prompts
from tomography.experiments.p7_extraction_steering import spearman
from tomography.experiments.r1_sentiment import select_examples
from tomography.experiments.steering import dose_slope
from tomography.experiments.sweep import SERIES
from tomography.experiments.variants import READING_TASKS
from tomography.experiments.variants import task_set
from tomography.experiments.variants import template_variants
from tomography.model import LoadedModel
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
NONZERO = (-0.2, 0.2)


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
        templates=("paper", "no_template"),
        formats=("raw", "chat"),
        n_seeds=3,
        n_steer_seeds=2,
        n_pairs=8,
        n_eval_pairs=8,
        sweep_layers=SWEEP,
        coeffs=(-0.2, 0.0, 0.2),
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
