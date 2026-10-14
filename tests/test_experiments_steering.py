from collections.abc import Sequence
import dataclasses
import json
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.control.controllers import ReadingVectorController
from tomography.control.controllers import mean_residual_norms
from tomography.control.generate import Completion
from tomography.control.generate import generate
from tomography.experiments.steering import Arm
from tomography.experiments.steering import DoseResponse
from tomography.experiments.steering import GenSettings
from tomography.experiments.steering import SteeringConfig
from tomography.experiments.steering import Trial
from tomography.experiments.steering import dose_response
from tomography.experiments.steering import dose_slope
from tomography.experiments.steering import edit_arm
from tomography.experiments.steering import plain_arm
from tomography.experiments.steering import reader_arm
from tomography.experiments.steering import run_trial
from tomography.experiments.steering import scoring_bos
from tomography.judges import perplexity
from tomography.judges import sequence_logprob
from tomography.model import LoadedModel


def completion(text: str, n_tokens: int | None = None) -> Completion:
    count = len(text.split()) if n_tokens is None else n_tokens
    return Completion(
        text=text, token_ids=tuple(range(1, count + 1)), finished=False
    )


def canned_arm(rows: list[list[Completion]], name: str = "canned") -> Arm:
    return Arm(name=name, generate=lambda prompts: rows)


class RecordingScorer:
    """Scores a text by its length and remembers what it was asked."""

    def __init__(self, columns: int | None = None) -> None:
        self.columns = columns
        self.calls: list[list[str]] = []

    def __call__(self, texts: Sequence[str]) -> NDArray[np.float64]:
        self.calls.append(list(texts))
        lengths = np.array([float(len(t)) for t in texts])
        if self.columns is None:
            return lengths
        return np.stack(
            [lengths * (k + 1) for k in range(self.columns)], axis=1
        )


def test_gen_settings_defaults_match_the_contract() -> None:
    settings = GenSettings()

    assert settings.max_new_tokens == 40
    assert settings.temperature == 0.8
    assert settings.top_p == 0.95
    assert settings.top_k == 0
    assert settings.num_samples == 2
    assert settings.seed == 0
    assert settings.batch_size == 16
    assert settings.bos is None
    assert settings.score_max_length == 128


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_new_tokens": 0},
        {"num_samples": 0},
        {"batch_size": 0},
        {"score_max_length": 1},
        {"temperature": None, "num_samples": 2},
        {"top_p": 0.0},
        {"top_p": 1.5},
        {"top_k": -1},
        {"temperature": 0.0},
        {"temperature": -0.5},
        {"temperature": float("nan")},
        {"temperature": float("inf")},
    ],
    ids=[
        "max_new_tokens_zero",
        "num_samples_zero",
        "batch_size_zero",
        "score_max_length_one",
        "samples_without_temperature",
        "top_p_zero",
        "top_p_above_one",
        "top_k_negative",
        "temperature_zero",
        "temperature_negative",
        "temperature_nan",
        "temperature_infinite",
    ],
)
def test_gen_settings_rejects_invalid_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r"must be|needs a temperature"):
        GenSettings(**kwargs)


def test_gen_settings_accepts_greedy_with_one_sample() -> None:
    settings = GenSettings(temperature=None, num_samples=1)

    assert settings.temperature is None


def test_from_config_copies_the_generation_fields() -> None:
    config = SteeringConfig(
        experiment="x",
        model="gpt2",
        seed=7,
        max_new_tokens=9,
        temperature=0.5,
        top_p=0.9,
        num_samples=3,
        batch_size=4,
        bos="none",
    )

    settings = GenSettings.from_config(config)

    assert settings == GenSettings(
        max_new_tokens=9,
        temperature=0.5,
        top_p=0.9,
        num_samples=3,
        seed=7,
        batch_size=4,
        bos="none",
    )


def test_steering_config_defaults() -> None:
    config = SteeringConfig(experiment="x", model="gpt2")

    assert config.layers is None
    assert config.coeffs == (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3)
    assert config.mode == "relative"
    assert config.mask == "response"
    assert config.op == "linear"


def test_from_dict_turns_json_integers_into_floats() -> None:
    data = json.loads(
        json.dumps({"experiment": "x", "model": "gpt2", "coeffs": [-1, 0, 2]})
    )

    config = SteeringConfig.from_dict(data)

    assert config.coeffs == (-1.0, 0.0, 2.0)
    assert all(type(c) is float for c in config.coeffs)


def test_from_dict_keeps_layers_none_and_round_trips() -> None:
    config = SteeringConfig(experiment="x", model="gpt2", layers=None)
    other = SteeringConfig(experiment="x", model="gpt2", layers=(6, 7))

    assert SteeringConfig.from_dict(config.to_dict()).layers is None
    assert SteeringConfig.from_dict(other.to_dict()) == other
    assert dataclasses.replace(config, layers=(1,)).layers == (1,)


def test_run_trial_flattens_prompt_major_and_scores_each_completion() -> None:
    rows = [
        [completion("aa"), completion("bbb")],
        [completion("c"), completion("dddd")],
    ]
    scorer = RecordingScorer()

    trial = run_trial(canned_arm(rows), ["p0", "p1"], scorer)

    assert scorer.calls == [["aa", "bbb", "c", "dddd"]]
    assert trial.name == "canned"
    assert trial.prompts == ("p0", "p1")
    assert trial.completions == (("aa", "bbb"), ("c", "dddd"))
    assert trial.scores.dtype == np.float32
    assert trial.scores.tolist() == [[2.0, 3.0], [1.0, 4.0]]
    assert trial.lengths.dtype == np.int64
    assert trial.lengths.tolist() == [[1, 1], [1, 1]]


def test_run_trial_name_override() -> None:
    rows = [[completion("a")]]

    trial = run_trial(canned_arm(rows), ["p"], RecordingScorer(), name="x")

    assert trial.name == "x"


def test_empty_and_whitespace_completions_score_nan_and_skip_the_scorer() -> (
    None
):
    rows = [
        [completion("", 0), completion("hello")],
        [completion("   ", 2), completion("ok")],
    ]
    scorer = RecordingScorer()

    trial = run_trial(canned_arm(rows), ["p0", "p1"], scorer)

    assert scorer.calls == [["hello", "ok"]]
    assert np.isnan(trial.scores[0, 0])
    assert np.isnan(trial.scores[1, 0])
    assert trial.scores[0, 1] == 5.0
    assert trial.scores[1, 1] == 2.0
    assert trial.lengths.tolist() == [[0, 1], [2, 1]]
    assert trial.empty_rate == 0.5


def test_columns_give_a_three_dimensional_score_array() -> None:
    rows = [
        [completion("ab"), completion("")],
        [completion("abc"), completion("d")],
    ]
    scorer = RecordingScorer(columns=3)

    trial = run_trial(canned_arm(rows), ["p0", "p1"], scorer, columns=3)

    assert trial.scores.shape == (2, 2, 3)
    assert trial.scores[0, 0].tolist() == [2.0, 4.0, 6.0]
    assert np.isnan(trial.scores[0, 1]).all()
    assert trial.scores[1, 0].tolist() == [3.0, 6.0, 9.0]


def test_all_empty_trial_is_nan_without_a_warning() -> None:
    rows = [[completion("", 0)], [completion(" ", 0)]]
    scorer = RecordingScorer(columns=2)

    trial = run_trial(canned_arm(rows), ["p0", "p1"], scorer, columns=2)

    assert scorer.calls == []
    assert trial.scores.shape == (2, 1, 2)
    assert np.isnan(trial.scores).all()
    assert trial.empty_rate == 1.0
    assert np.isnan(trial.per_prompt()).all()
    assert trial.per_prompt().shape == (2, 2)
    assert np.isnan(trial.mean_score()).all()
    assert trial.mean_score().shape == (2,)


def test_statistics_are_nan_aware() -> None:
    scores = np.array([[1.0, np.nan], [2.0, 4.0], [np.nan, np.nan]], np.float32)
    trial = Trial(
        name="t",
        prompts=("a", "b", "c"),
        completions=(("x", ""), ("y", "z"), ("", "")),
        lengths=np.array([[1, 0], [1, 1], [0, 0]], np.int64),
        scores=scores,
    )

    per_prompt = trial.per_prompt()
    mean = trial.mean_score()

    assert per_prompt[:2].tolist() == [1.0, 3.0]
    assert np.isnan(per_prompt[2])
    assert mean.shape == ()
    assert float(mean) == pytest.approx(7.0 / 3.0)
    assert trial.empty_rate == 0.5


def test_statistics_with_columns_reduce_each_column() -> None:
    scores = np.array(
        [[[1.0, 10.0], [3.0, 30.0]], [[np.nan, np.nan], [5.0, 50.0]]],
        np.float32,
    )
    trial = Trial(
        name="t",
        prompts=("a", "b"),
        completions=(("x", "y"), ("", "z")),
        lengths=np.array([[1, 1], [0, 1]], np.int64),
        scores=scores,
    )

    assert trial.per_prompt().tolist() == [[2.0, 20.0], [5.0, 50.0]]
    assert trial.mean_score().tolist() == [3.0, 30.0]


def test_run_trial_rejects_a_row_count_mismatch() -> None:
    arm = canned_arm([[completion("a")]])

    with pytest.raises(ValueError, match="rows"):
        run_trial(arm, ["p0", "p1"], RecordingScorer())


def test_run_trial_rejects_rows_of_unequal_length() -> None:
    arm = canned_arm([[completion("a"), completion("b")], [completion("c")]])

    with pytest.raises(ValueError, match="same number"):
        run_trial(arm, ["p0", "p1"], RecordingScorer())


@pytest.mark.parametrize(
    ("prompts", "rows"),
    [([], []), (["p"], [[]])],
    ids=["no_prompts", "no_samples"],
)
def test_run_trial_rejects_no_completions(
    prompts: list[str], rows: list[list[Completion]]
) -> None:
    with pytest.raises(ValueError, match="no completions"):
        run_trial(canned_arm(rows), prompts, RecordingScorer())


@pytest.mark.parametrize(
    ("columns", "scorer_columns"),
    [(None, 2), (3, None), (3, 2)],
    ids=["unexpected_columns", "missing_columns", "wrong_columns"],
)
def test_run_trial_rejects_a_wrong_scorer_shape(
    columns: int | None, scorer_columns: int | None
) -> None:
    rows = [[completion("a"), completion("b")]]

    with pytest.raises(ValueError, match="scorer"):
        run_trial(
            canned_arm(rows),
            ["p"],
            RecordingScorer(columns=scorer_columns),
            columns=columns,
        )


def linear_arm_factory(slope: float = 2.0, intercept: float = 1.0) -> Any:
    """Builds arms whose completion text encodes `intercept + slope * coeff`."""

    def make(coeff: float) -> Arm:
        value = intercept + slope * coeff
        rows = [[completion(f"{value}"), completion(f"{value}")]] * 3
        return canned_arm(rows, name="ignored")

    return make


def parse_scorer(texts: Sequence[str]) -> NDArray[np.float64]:
    return np.array([float(t) for t in texts])


def test_dose_response_runs_one_trial_per_coefficient_in_order() -> None:
    coeffs = [-0.5, 0.0, 0.25, 1.0]

    dose = dose_response(
        linear_arm_factory(), coeffs, ["a", "b", "c"], parse_scorer
    )

    assert isinstance(dose, DoseResponse)
    assert dose.coeffs == (-0.5, 0.0, 0.25, 1.0)
    assert [t.name for t in dose.trials] == ["-0.5", "0", "0.25", "1"]
    assert all(t.prompts == ("a", "b", "c") for t in dose.trials)
    np.testing.assert_allclose(dose.curve(), [0.0, 1.0, 1.5, 3.0])


def test_dose_slope_recovers_the_planted_slope() -> None:
    coeffs = [-0.3, -0.1, 0.0, 0.1, 0.3]
    dose = dose_response(
        linear_arm_factory(slope=2.5), coeffs, ["a", "b", "c"], parse_scorer
    )

    assert dose_slope(dose.coeffs, dose.curve()) == pytest.approx(2.5)


def test_dose_slope_returns_a_python_float() -> None:
    slope = dose_slope([0.0, 1.0], np.array([0.0, 3.0]))

    assert type(slope) is float
    assert slope == pytest.approx(3.0)


def test_dose_slope_drops_nan_values() -> None:
    values = np.array([1.0, np.nan, 5.0, 7.0])

    assert dose_slope([0.0, 0.5, 2.0, 3.0], values) == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("coeffs", "values"),
    [
        ([0.0], [1.0]),
        ([0.0, 1.0], [1.0, np.nan]),
        ([1.0, 1.0], [1.0, 2.0]),
        ([], []),
    ],
    ids=["one_point", "one_finite", "one_distinct", "empty"],
)
def test_dose_slope_needs_two_distinct_finite_points(
    coeffs: list[float], values: list[float]
) -> None:
    with pytest.raises(ValueError, match="two distinct"):
        dose_slope(coeffs, np.array(values))


def test_dose_slope_rejects_a_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length"):
        dose_slope([0.0, 1.0], np.array([1.0, 2.0, 3.0]))


@pytest.mark.parametrize(
    "coeffs", [[], np.array([])], ids=["list", "numpy_array"]
)
def test_dose_response_rejects_empty_coefficients(
    coeffs: Sequence[float],
) -> None:
    with pytest.raises(ValueError, match="coeffs"):
        dose_response(linear_arm_factory(), coeffs, ["a"], parse_scorer)


def test_dose_response_accepts_a_numpy_grid() -> None:
    coeffs = np.linspace(-1.0, 1.0, 5)

    dose = dose_response(
        linear_arm_factory(), coeffs, ["a", "b", "c"], parse_scorer
    )

    assert dose.coeffs == (-1.0, -0.5, 0.0, 0.5, 1.0)
    assert [t.name for t in dose.trials] == ["-1", "-0.5", "0", "0.5", "1"]
    np.testing.assert_allclose(dose.curve(), [-1.0, 0.0, 1.0, 2.0, 3.0])


def test_dose_curve_selects_a_column() -> None:
    def make(coeff: float) -> Arm:
        return canned_arm([[completion(f"{coeff}")]])

    def two_columns(texts: Sequence[str]) -> NDArray[np.float64]:
        values = parse_scorer(texts)
        return np.stack([values, -values], axis=1)

    dose = dose_response(make, [1.0, 2.0], ["a"], two_columns, columns=2)

    np.testing.assert_allclose(dose.curve(0), [1.0, 2.0])
    np.testing.assert_allclose(dose.curve(column=1), [-1.0, -2.0])


def test_dose_curve_validates_the_column() -> None:
    def make(coeff: float) -> Arm:
        return canned_arm([[completion("1.0")]])

    flat = dose_response(make, [0.0], ["a"], parse_scorer)
    wide = dose_response(
        make,
        [0.0],
        ["a"],
        lambda texts: np.ones((len(texts), 2)),
        columns=2,
    )

    with pytest.raises(ValueError, match="column"):
        flat.curve(0)
    with pytest.raises(ValueError, match="column"):
        wide.curve()
    with pytest.raises(ValueError, match="column"):
        wide.curve(2)
    with pytest.raises(ValueError, match="column"):
        wide.curve(-1)


PROMPTS = ["The quick brown fox", "The lazy dog", "Honest answers are"]
CONTINUATIONS = [" jumps over it", " sleeps all day", " true and good"]
NEUTRAL = [
    "A short text, a longer text, and another text with more words in it.",
    "The model reads the stream of the residual after every block.",
]
LAYERS = (1, 2)
SETTINGS = GenSettings(
    max_new_tokens=5,
    temperature=0.8,
    top_p=0.95,
    top_k=5,
    num_samples=2,
    seed=3,
    bos="none",
    batch_size=2,
    score_max_length=8,
)


@dataclasses.dataclass(frozen=True)
class StubSource:
    """A fitted-reader stand-in: random unit directions, positive signs."""

    layers: tuple[int, ...]
    directions: NDArray[np.float32]
    signs: NDArray[np.float32]


def stub_source(loaded: LoadedModel) -> StubSource:
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(len(LAYERS), loaded.d_model)).astype(np.float32)
    unit = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    return StubSource(
        layers=LAYERS, directions=unit, signs=np.ones(len(LAYERS), np.float32)
    )


def make_controller(
    loaded: LoadedModel,
    coeff: float,
    *,
    mask: Any = "response",
    norms: dict[int, float] | None = None,
) -> ReadingVectorController:
    return ReadingVectorController.from_reader(
        stub_source(loaded),
        LAYERS,
        coeff=coeff,
        mode="relative",
        norms=norms or {layer: 5.0 for layer in LAYERS},
        mask=mask,
        device=loaded.device,
    )


def hooks_left(loaded: LoadedModel) -> int:
    return sum(
        len(module._forward_hooks) + len(module._forward_pre_hooks)
        for module in loaded.model.modules()
    )


@pytest.fixture
def full_length(
    tiny_arch: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> LoadedModel:
    """A tiny model whose rows never end early on the shared pad/eos id."""
    monkeypatch.setattr(tiny_arch.model.generation_config, "eos_token_id", None)
    return tiny_arch


def test_scoring_bos_follows_the_default_bos(tiny_arch: LoadedModel) -> None:
    prepending = dataclasses.replace(tiny_arch, default_bos="prepend", bos_id=5)
    plain = dataclasses.replace(tiny_arch, default_bos="none")

    assert scoring_bos(prepending) == 5
    assert scoring_bos(plain) is None


def test_plain_arm_equals_a_direct_generate_call(
    full_length: LoadedModel,
) -> None:
    arm = plain_arm(full_length, SETTINGS)

    expected = generate(
        full_length,
        PROMPTS,
        max_new_tokens=5,
        temperature=0.8,
        top_p=0.95,
        top_k=5,
        num_samples=2,
        seed=3,
        bos="none",
        batch_size=2,
    )

    assert arm.name == "none"
    assert arm.generate(PROMPTS) == expected
    assert plain_arm(full_length, SETTINGS, name="base").name == "base"


def test_a_greedy_config_generates_with_its_defaults(
    full_length: LoadedModel,
) -> None:
    config = SteeringConfig(
        experiment="x",
        model="gpt2",
        temperature=None,
        num_samples=1,
        max_new_tokens=4,
        batch_size=2,
    )
    settings = GenSettings.from_config(config)
    plain = plain_arm(full_length, settings)
    steered = edit_arm(
        full_length, settings, make_controller(full_length, 0.0), name="z"
    )

    expected = generate(
        full_length,
        PROMPTS,
        max_new_tokens=4,
        temperature=None,
        num_samples=1,
        batch_size=2,
    )

    assert plain.generate(PROMPTS) == expected
    assert steered.generate(PROMPTS) == expected


def test_plain_arm_scores_like_the_judges(tiny_arch: LoadedModel) -> None:
    arm = plain_arm(tiny_arch, SETTINGS)
    assert arm.logprobs is not None
    assert arm.perplexity is not None
    bos = scoring_bos(tiny_arch)

    expected_logprobs = sequence_logprob(
        tiny_arch.model,
        tiny_arch.tokenizer,
        PROMPTS,
        CONTINUATIONS,
        batch_size=2,
        bos_id=bos,
    )
    expected_perplexity = perplexity(
        tiny_arch.model,
        tiny_arch.tokenizer,
        NEUTRAL,
        max_length=8,
        batch_size=2,
        bos_id=bos,
    )

    assert np.array_equal(
        arm.logprobs(PROMPTS, CONTINUATIONS), expected_logprobs
    )
    assert arm.perplexity(NEUTRAL) == expected_perplexity


def test_coefficient_zero_edit_arm_is_bit_identical_to_the_plain_arm(
    full_length: LoadedModel,
) -> None:
    plain = plain_arm(full_length, SETTINGS)
    zero = edit_arm(
        full_length, SETTINGS, make_controller(full_length, 0.0), name="zero"
    )
    assert plain.logprobs is not None
    assert zero.logprobs is not None
    assert plain.perplexity is not None
    assert zero.perplexity is not None

    assert zero.name == "zero"
    assert zero.generate(PROMPTS) == plain.generate(PROMPTS)
    assert np.array_equal(
        zero.logprobs(PROMPTS, CONTINUATIONS),
        plain.logprobs(PROMPTS, CONTINUATIONS),
    )
    assert zero.perplexity(NEUTRAL) == plain.perplexity(NEUTRAL)


def test_a_steered_arm_changes_the_output_and_leaves_no_hook(
    full_length: LoadedModel,
) -> None:
    plain = plain_arm(full_length, SETTINGS)
    steered = edit_arm(
        full_length, SETTINGS, make_controller(full_length, 0.5), name="s"
    )
    assert plain.logprobs is not None
    assert steered.logprobs is not None
    assert plain.perplexity is not None
    assert steered.perplexity is not None
    before_logprobs = plain.logprobs(PROMPTS, CONTINUATIONS)
    before_perplexity = plain.perplexity(NEUTRAL)

    steered_logprobs = steered.logprobs(PROMPTS, CONTINUATIONS)
    steered_perplexity = steered.perplexity(NEUTRAL)
    steered_text = steered.generate(PROMPTS)

    assert not np.array_equal(steered_logprobs, before_logprobs)
    assert steered_perplexity != before_perplexity
    assert steered_text != plain.generate(PROMPTS)
    assert hooks_left(full_length) == 0
    assert np.array_equal(
        plain.logprobs(PROMPTS, CONTINUATIONS), before_logprobs
    )
    assert plain.perplexity(NEUTRAL) == before_perplexity


def test_response_and_all_masks_score_identically(
    tiny_arch: LoadedModel,
) -> None:
    response = edit_arm(
        tiny_arch, SETTINGS, make_controller(tiny_arch, 0.5), name="r"
    )
    everywhere = edit_arm(
        tiny_arch,
        SETTINGS,
        make_controller(tiny_arch, 0.5, mask="all"),
        name="a",
    )
    assert response.logprobs is not None
    assert everywhere.logprobs is not None
    assert response.perplexity is not None
    assert everywhere.perplexity is not None

    assert np.array_equal(
        response.logprobs(PROMPTS, CONTINUATIONS),
        everywhere.logprobs(PROMPTS, CONTINUATIONS),
    )
    assert response.perplexity(NEUTRAL) == everywhere.perplexity(NEUTRAL)


def test_prefix_is_prepended_to_prompts_and_not_to_the_completion(
    full_length: LoadedModel,
) -> None:
    prefix = "Be kind. "
    arm = plain_arm(full_length, SETTINGS, prefix=prefix)

    expected = generate(
        full_length,
        [prefix + p for p in PROMPTS],
        max_new_tokens=5,
        temperature=0.8,
        top_p=0.95,
        top_k=5,
        num_samples=2,
        seed=3,
        bos="none",
        batch_size=2,
    )

    assert arm.generate(PROMPTS) == expected
    assert arm.generate(PROMPTS) != plain_arm(full_length, SETTINGS).generate(
        PROMPTS
    )


def test_prefix_arm_scores_given_the_prefixed_prompt_and_has_no_perplexity(
    tiny_arch: LoadedModel,
) -> None:
    prefix = "Be kind. "
    arm = plain_arm(tiny_arch, SETTINGS, prefix=prefix)
    steered = edit_arm(
        tiny_arch,
        SETTINGS,
        make_controller(tiny_arch, 0.5),
        name="s",
        prefix=prefix,
    )
    assert arm.logprobs is not None
    assert steered.logprobs is not None

    expected = sequence_logprob(
        tiny_arch.model,
        tiny_arch.tokenizer,
        [prefix + p for p in PROMPTS],
        CONTINUATIONS,
        batch_size=2,
        bos_id=scoring_bos(tiny_arch),
    )

    assert np.array_equal(arm.logprobs(PROMPTS, CONTINUATIONS), expected)
    assert steered.logprobs(PROMPTS, CONTINUATIONS).shape == (3,)
    assert arm.perplexity is None
    assert steered.perplexity is None


def test_reader_arm_equals_edit_arm_of_the_same_controller(
    full_length: LoadedModel,
) -> None:
    norms = mean_residual_norms(full_length, NEUTRAL, list(LAYERS))
    by_reader = reader_arm(
        full_length,
        SETTINGS,
        stub_source(full_length),
        layers=LAYERS,
        coeff=0.25,
        norms=norms,
    )
    by_hand = edit_arm(
        full_length,
        SETTINGS,
        make_controller(full_length, 0.25, norms=norms),
        name="x",
    )
    assert by_reader.logprobs is not None
    assert by_hand.logprobs is not None
    assert by_reader.perplexity is not None
    assert by_hand.perplexity is not None

    assert by_reader.name == "0.25"
    assert by_reader.generate(PROMPTS) == by_hand.generate(PROMPTS)
    assert np.array_equal(
        by_reader.logprobs(PROMPTS, CONTINUATIONS),
        by_hand.logprobs(PROMPTS, CONTINUATIONS),
    )
    assert by_reader.perplexity(NEUTRAL) == by_hand.perplexity(NEUTRAL)


def test_reader_arm_passes_its_options_to_the_controller(
    tiny_arch: LoadedModel,
) -> None:
    by_reader = reader_arm(
        tiny_arch,
        SETTINGS,
        stub_source(tiny_arch),
        layers=LAYERS,
        coeff=1.0,
        op="projection",
        mode="absolute",
        mask="all",
        preserve_norm=True,
        name="abs",
    )
    by_hand = edit_arm(
        tiny_arch,
        SETTINGS,
        ReadingVectorController.from_reader(
            stub_source(tiny_arch),
            LAYERS,
            coeff=1.0,
            op="projection",
            mode="absolute",
            mask="all",
            preserve_norm=True,
            device=tiny_arch.device,
        ),
        name="abs",
    )
    plain = plain_arm(tiny_arch, SETTINGS)
    assert by_reader.logprobs is not None
    assert by_hand.logprobs is not None
    assert plain.logprobs is not None

    assert by_reader.name == "abs"
    assert np.array_equal(
        by_reader.logprobs(PROMPTS, CONTINUATIONS),
        by_hand.logprobs(PROMPTS, CONTINUATIONS),
    )
    assert not np.array_equal(
        by_reader.logprobs(PROMPTS, CONTINUATIONS),
        plain.logprobs(PROMPTS, CONTINUATIONS),
    )


def test_reader_arm_relative_mode_needs_norms(tiny_arch: LoadedModel) -> None:
    with pytest.raises(ValueError, match="norm"):
        reader_arm(
            tiny_arch,
            SETTINGS,
            stub_source(tiny_arch),
            layers=LAYERS,
            coeff=0.1,
        )


def test_coefficient_zero_reproduces_the_plain_trial(
    full_length: LoadedModel,
) -> None:
    norms = mean_residual_norms(full_length, NEUTRAL, list(LAYERS))
    plain = run_trial(
        plain_arm(full_length, SETTINGS), PROMPTS, parse_scorer_len
    )
    zero = run_trial(
        reader_arm(
            full_length,
            SETTINGS,
            stub_source(full_length),
            layers=LAYERS,
            coeff=0.0,
            norms=norms,
        ),
        PROMPTS,
        parse_scorer_len,
    )

    assert zero.completions == plain.completions
    assert np.array_equal(zero.scores, plain.scores, equal_nan=True)
    assert np.array_equal(zero.lengths, plain.lengths)


def parse_scorer_len(texts: Sequence[str]) -> NDArray[np.float64]:
    return np.array([float(len(t)) for t in texts])
