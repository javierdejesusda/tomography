import dataclasses
import io
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pytest
import torch

from tomography.control.controllers import mean_residual_norms
from tomography.datasets.wikitext import load_wikitext
from tomography.experiments.base import RunDir
from tomography.experiments.effects import HeldOut
from tomography.experiments.effects import SideEffects
from tomography.experiments.effects import distinct_n
from tomography.experiments.effects import fluency
from tomography.experiments.effects import read_trials
from tomography.experiments.effects import side_effects
from tomography.experiments.effects import split_held_out
from tomography.experiments.effects import write_dose_curves
from tomography.experiments.effects import write_trials
from tomography.experiments.steering import Arm
from tomography.experiments.steering import GenSettings
from tomography.experiments.steering import SteeringConfig
from tomography.experiments.steering import Trial
from tomography.experiments.steering import plain_arm
from tomography.experiments.steering import reader_arm
from tomography.experiments.steering import run_trial
from tomography.experiments.steering import scoring_bos
from tomography.judges import sequence_logprob
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.viz.reading import curves_from_arrays
from tomography.viz.reading import plot_curves
from tomography.viz.style import style_context

PROMPTS = ("The quick brown fox", "The lazy dog", "Honest answers are")
NEUTRAL = [
    "A short text, a longer text, and another text with more words in it.",
    "The model reads the stream of the residual after every block.",
]
SETTINGS = GenSettings(
    max_new_tokens=5,
    temperature=0.8,
    top_p=0.95,
    num_samples=2,
    seed=3,
    batch_size=2,
    score_max_length=32,
)


@dataclasses.dataclass(frozen=True)
class StubSource:
    """A fitted-reader stand-in: unit directions and positive signs."""

    layers: tuple[int, ...]
    directions: NDArray[np.float32]
    signs: NDArray[np.float32]


def make_trial(
    completions: tuple[tuple[str, ...], ...],
    lengths: list[list[int]],
    scores: NDArray[np.float32] | None = None,
    name: str = "t",
) -> Trial:
    prompts = PROMPTS[: len(completions)]
    return Trial(
        name=name,
        prompts=prompts,
        completions=completions,
        lengths=np.array(lengths, dtype=np.int64),
        scores=(
            np.zeros(np.shape(lengths), np.float32)
            if scores is None
            else scores
        ),
    )


def test_distinct_n_pools_over_all_completions() -> None:
    sequences = [[1, 2, 3], [1, 2, 3]]

    assert distinct_n(sequences, 1) == 0.5
    assert distinct_n(sequences, 2) == 0.5
    assert distinct_n([[1, 2, 3, 4]], 3) == 1.0


def test_distinct_n_counts_no_ngram_across_two_completions() -> None:
    assert distinct_n([[1, 2], [2, 1]], 2) == 1.0
    assert distinct_n([[1, 1], [1, 1]], 2) == 0.5


@pytest.mark.parametrize(
    ("sequences", "n"),
    [([], 1), ([[]], 1), ([[1]], 2), ([[1], [2]], 2)],
    ids=["no_sequence", "empty_sequence", "too_short", "all_too_short"],
)
def test_distinct_n_is_nan_without_ngrams(
    sequences: list[list[int]], n: int
) -> None:
    assert np.isnan(distinct_n(sequences, n))


def test_distinct_n_rejects_n_below_one() -> None:
    with pytest.raises(ValueError, match="n must be"):
        distinct_n([[1, 2]], 0)


TEXTS = [f"text number {i:03d} " + "x" * 50 for i in range(30)]


def test_split_held_out_returns_disjoint_slices_of_the_right_size() -> None:
    held = split_held_out(TEXTS, n_calibration=5, n_neutral=7, seed=0)

    assert isinstance(held, HeldOut)
    assert len(held.calibration) == 5
    assert len(held.neutral) == 7
    assert not set(held.calibration) & set(held.neutral)
    assert set(held.calibration) | set(held.neutral) <= set(TEXTS)


def test_split_held_out_is_deterministic_and_seed_dependent() -> None:
    first = split_held_out(TEXTS, n_calibration=5, n_neutral=7, seed=0)
    again = split_held_out(TEXTS, n_calibration=5, n_neutral=7, seed=0)
    other = split_held_out(TEXTS, n_calibration=5, n_neutral=7, seed=1)

    assert first == again
    assert first != other


def test_split_held_out_matches_the_documented_permutation() -> None:
    held = split_held_out(TEXTS, n_calibration=4, n_neutral=3, seed=9)

    order = np.random.default_rng(9).permutation(len(TEXTS))

    assert held.calibration == tuple(TEXTS[i] for i in order[:4])
    assert held.neutral == tuple(TEXTS[i] for i in order[4:7])


def test_split_held_out_drops_short_texts() -> None:
    texts = ["= Heading =", *TEXTS, "tiny"]

    held = split_held_out(texts, n_calibration=15, n_neutral=15, seed=0)

    assert set(held.calibration) | set(held.neutral) == set(TEXTS)


def test_split_held_out_min_chars_is_inclusive() -> None:
    texts = [f"{'a' * 40}{i}" for i in range(5)] + ["b" * 40, "c" * 39]

    held = split_held_out(texts, n_calibration=3, n_neutral=3, seed=0)

    kept = set(held.calibration) | set(held.neutral)
    assert "b" * 40 in kept
    assert "c" * 39 not in kept
    assert len(kept) == 6


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_calibration": 0, "n_neutral": 5},
        {"n_calibration": 5, "n_neutral": 0},
        {"n_calibration": 20, "n_neutral": 20},
    ],
    ids=["no_calibration", "no_neutral", "too_few_texts"],
)
def test_split_held_out_rejects_bad_sizes(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match=r"n_calibration|n_neutral|texts"):
        split_held_out(TEXTS, seed=0, **kwargs)


def test_fluency_equals_the_manual_per_token_logprob(
    tiny_arch: LoadedModel,
) -> None:
    trial = make_trial(
        ((" jumps over it", " is"), (" sleeps all day", " ok")),
        [[3, 1], [3, 1]],
    )
    prompts = [PROMPTS[0], PROMPTS[0], PROMPTS[1], PROMPTS[1]]
    texts = [" jumps over it", " is", " sleeps all day", " ok"]
    logprobs = sequence_logprob(
        tiny_arch.model,
        tiny_arch.tokenizer,
        prompts,
        texts,
        batch_size=16,
        bos_id=scoring_bos(tiny_arch),
    )
    counts = [
        len(tiny_arch.tokenizer(t, add_special_tokens=False)["input_ids"])
        for t in texts
    ]

    expected = float(np.mean(logprobs / np.array(counts)))

    assert fluency(tiny_arch, trial) == pytest.approx(expected, rel=1e-6)
    assert fluency(tiny_arch, trial, batch_size=1) == pytest.approx(
        expected, rel=1e-5
    )


def test_fluency_skips_empty_completions(tiny_arch: LoadedModel) -> None:
    full = make_trial(((" jumps over it",), (" sleeps all day",)), [[3], [3]])
    holes = make_trial(
        ((" jumps over it", ""), (" sleeps all day", "  ")),
        [[3, 0], [3, 2]],
    )

    assert fluency(tiny_arch, holes) == pytest.approx(
        fluency(tiny_arch, full), rel=1e-6
    )


def test_fluency_is_nan_when_every_completion_is_empty(
    tiny_arch: LoadedModel,
) -> None:
    trial = make_trial((("",), (" ",)), [[0], [1]])

    assert np.isnan(fluency(tiny_arch, trial))


def test_side_effects_fills_every_field(tiny_arch: LoadedModel) -> None:
    arm = plain_arm(tiny_arch, SETTINGS)
    trial = make_trial(
        ((" jumps over it", ""), (" sleeps all day", " over it")),
        [[3, 0], [3, 2]],
    )

    result = side_effects(tiny_arch, arm, trial, NEUTRAL)

    assert isinstance(result, SideEffects)
    assert arm.perplexity is not None
    assert result.perplexity == arm.perplexity(NEUTRAL)
    assert result.fluency == fluency(tiny_arch, trial)
    assert result.mean_length == pytest.approx(2.0)
    assert result.empty_rate == 0.25
    ids = [
        tiny_arch.tokenizer(t, add_special_tokens=False)["input_ids"]
        for t in (" jumps over it", " sleeps all day", " over it")
    ]
    assert result.distinct_1 == distinct_n(ids, 1)
    assert result.distinct_2 == distinct_n(ids, 2)


def test_side_effects_perplexity_is_none_for_an_arm_without_one(
    tiny_arch: LoadedModel,
) -> None:
    arm = Arm(name="stub", generate=lambda prompts: [])
    trial = make_trial(((" jumps over it",),), [[3]])

    result = side_effects(tiny_arch, arm, trial, NEUTRAL)

    assert result.perplexity is None
    assert np.isfinite(result.fluency)


def test_side_effects_with_a_prefix_arm_has_no_perplexity(
    tiny_arch: LoadedModel,
) -> None:
    arm = plain_arm(tiny_arch, SETTINGS, prefix="Be kind. ")
    trial = run_trial(
        arm, list(PROMPTS), lambda texts: np.zeros(len(texts), np.float32)
    )

    result = side_effects(tiny_arch, arm, trial, NEUTRAL)

    assert result.perplexity is None


@pytest.fixture
def run_dir(tmp_path: Path) -> RunDir:
    config = SteeringConfig(experiment="x", model="gpt2")
    return RunDir.create(config, out_root=tmp_path)


def assert_same_trial(read: Trial, written: Trial) -> None:
    assert read.name == written.name
    assert read.prompts == written.prompts
    assert read.completions == written.completions
    assert read.lengths.dtype == np.int64
    assert np.array_equal(read.lengths, written.lengths)
    assert read.scores.dtype == np.float32
    assert np.array_equal(read.scores, written.scores, equal_nan=True)


def test_trials_round_trip_with_nan(run_dir: RunDir) -> None:
    first = make_trial(
        (("a b", ""), ("c", "d e f")),
        [[2, 0], [1, 3]],
        np.array([[0.5, np.nan], [1.5, 2.5]], np.float32),
        name="0",
    )
    second = make_trial(
        (("g", "h"), ("", "i")),
        [[1, 1], [0, 1]],
        np.array([[3.0, 4.0], [np.nan, 5.0]], np.float32),
        name="0.5",
    )

    write_trials(run_dir, "trials", [first, second])
    read = read_trials(run_dir, "trials")

    assert len(read) == 2
    assert_same_trial(read[0], first)
    assert_same_trial(read[1], second)
    arrays = run_dir.read_arrays("trials")
    assert arrays["scores"].shape == (2, 2, 2)
    assert arrays["lengths"].shape == (2, 2, 2)


def test_trials_with_columns_round_trip(run_dir: RunDir) -> None:
    scores = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    scores[0, 1] = np.nan
    trial = make_trial(
        (("a", ""), ("b", "c")), [[1, 0], [1, 1]], scores, name="emo"
    )

    write_trials(run_dir, "trials", [trial])
    (read,) = read_trials(run_dir, "trials")

    assert read.scores.shape == (2, 2, 3)
    assert_same_trial(read, trial)


def test_write_trials_stores_names_prompts_and_completions(
    run_dir: RunDir,
) -> None:
    trial = make_trial((("a", "b"), ("c", "d")), [[1, 1], [1, 1]], name="n")

    write_trials(run_dir, "gallery", [trial])

    payload = run_dir.read_json("gallery")
    assert payload == {
        "names": ["n"],
        "prompts": list(PROMPTS[:2]),
        "completions": [[["a", "b"], ["c", "d"]]],
    }


def test_write_trials_rejects_no_trials(run_dir: RunDir) -> None:
    with pytest.raises(ValueError, match="no trials"):
        write_trials(run_dir, "trials", [])


def test_write_trials_rejects_different_prompts(run_dir: RunDir) -> None:
    first = make_trial((("a",), ("b",)), [[1], [1]])
    other = dataclasses.replace(first, prompts=("x", "y"))

    with pytest.raises(ValueError, match="prompts"):
        write_trials(run_dir, "trials", [first, other])


def test_write_trials_rejects_different_shapes(run_dir: RunDir) -> None:
    first = make_trial((("a",), ("b",)), [[1], [1]])
    wide = make_trial((("a", "b"), ("c", "d")), [[1, 1], [1, 1]])

    with pytest.raises(ValueError, match="shape"):
        write_trials(run_dir, "trials", [first, wide])


def curves_input() -> NDArray[np.float64]:
    rng = np.random.default_rng(0)
    return rng.normal(size=(2, 4, 3))


def test_dose_curves_round_trip_through_read_arrays(run_dir: RunDir) -> None:
    per_seed = curves_input()

    write_dose_curves(
        run_dir,
        per_seed,
        [-0.5, 0.0, 0.25],
        series=["a", "b"],
        n_boot=200,
    )
    arrays = run_dir.read_arrays("curves")
    sidecar = run_dir.read_json("curves")

    assert sorted(arrays) == ["hi", "lo", "mean", "per_seed", "x"]
    assert arrays["x"].dtype == np.float64
    assert arrays["x"].tolist() == [-0.5, 0.0, 0.25]
    assert np.allclose(arrays["mean"], per_seed.mean(axis=1))
    assert np.array_equal(arrays["per_seed"], per_seed)
    assert arrays["lo"].shape == (2, 3)
    assert (arrays["lo"] <= arrays["mean"] + 1e-12).all()
    assert (arrays["mean"] <= arrays["hi"] + 1e-12).all()
    assert sidecar == {
        "series": ["a", "b"],
        "xlabel": "Coefficient",
        "ylabel": "Score",
        "chance": None,
        "n_layers": None,
    }


def test_dose_curves_custom_labels_stem_and_chance(run_dir: RunDir) -> None:
    write_dose_curves(
        run_dir,
        curves_input(),
        [0.0, 1.0, 2.0],
        series=["a", "b"],
        stem="dose",
        xlabel="Relative coefficient",
        ylabel="Positive rate",
        chance=0.5,
        n_boot=50,
    )

    sidecar = run_dir.read_json("dose")

    assert sidecar == {
        "series": ["a", "b"],
        "xlabel": "Relative coefficient",
        "ylabel": "Positive rate",
        "chance": 0.5,
        "n_layers": None,
    }


def test_dose_curves_are_drawn_by_plot_curves_on_a_float_axis(
    run_dir: RunDir,
) -> None:
    write_dose_curves(
        run_dir,
        curves_input(),
        [-0.5, 0.0, 0.25],
        series=["a", "b"],
        chance=0.0,
        n_boot=100,
    )
    arrays = run_dir.read_arrays("curves")
    sidecar = run_dir.read_json("curves")
    assert isinstance(sidecar, dict)

    with style_context():
        figure = plot_curves(
            curves_from_arrays(
                arrays["x"], arrays["mean"], arrays["lo"], arrays["hi"]
            ),
            series=sidecar["series"],
            per_seed=list(arrays["per_seed"]),
            xlabel=sidecar["xlabel"],
            ylabel=sidecar["ylabel"],
            chance=sidecar["chance"],
            n_layers=sidecar["n_layers"],
        )
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png")

    assert buffer.getvalue().startswith(b"\x89PNG")


@pytest.mark.parametrize(
    ("per_seed", "coeffs", "series", "message"),
    [
        (np.zeros((2, 3)), [0.0, 1.0, 2.0], ["a", "b"], "shape"),
        (np.zeros((2, 4, 3)), [0.0, 1.0, 2.0], ["a"], "series"),
        (np.zeros((2, 4, 3)), [0.0, 1.0], ["a", "b"], "coeffs"),
    ],
    ids=["not_3d", "series_mismatch", "coeffs_mismatch"],
)
def test_dose_curves_reject_shape_errors(
    run_dir: RunDir,
    per_seed: NDArray[np.float64],
    coeffs: list[float],
    series: list[str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        write_dose_curves(run_dir, per_seed, coeffs, series=series)


@pytest.mark.slow
def test_neutral_perplexity_ratio_under_a_random_vector_pilot() -> None:
    loaded = load_model("gpt2", dtype=torch.float32)
    texts = [example.text for example in load_wikitext()]
    held = split_held_out(texts, n_calibration=200, n_neutral=200, seed=0)
    layer = 6
    norms = mean_residual_norms(
        loaded, held.calibration, [layer], batch_size=16
    )
    generator = torch.Generator().manual_seed(0)
    direction = torch.randn(loaded.d_model, generator=generator)
    direction = (direction / direction.norm()).unsqueeze(0).numpy()
    reader = StubSource(
        layers=(layer,),
        directions=direction.astype(np.float32),
        signs=np.ones(1, np.float32),
    )
    settings = GenSettings()
    baseline = plain_arm(loaded, settings).perplexity
    assert baseline is not None
    base_value = baseline(held.neutral)

    ratios: dict[float, float] = {}
    for coeff in (0.0, 0.05, 0.1, 0.2, 0.4):
        arm = reader_arm(
            loaded,
            settings,
            reader,
            layers=[layer],
            coeff=coeff,
            norms=norms,
        )
        assert arm.perplexity is not None
        ratios[coeff] = arm.perplexity(held.neutral) / base_value
        print(f"coeff {coeff:g}: perplexity ratio {ratios[coeff]:.4f}")

    assert all(np.isfinite(r) and r > 0 for r in ratios.values())
    assert ratios[0.0] == pytest.approx(1.0)
