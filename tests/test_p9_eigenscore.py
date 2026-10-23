"""Tests of EigenScore (`tomography.p9.eigenscore`)."""

from collections.abc import Sequence
import math

import numpy as np
import pytest

from tomography.model import LoadedModel
from tomography.p9.answer_probe import answer_activations
from tomography.p9.eigenscore import eigen_layer
from tomography.p9.eigenscore import eigenscore
from tomography.p9.eigenscore import eigenscores

PROMPTS = ["Q: who wrote it?\nA:", "Q: where?\nA:", "Q: when?\nA:"]
SAMPLES = [
    ("Paris", "Rome", "a vehicle"),
    ("the north", "the north", "an animal in the north"),
    ("1990", "", "1991"),
]
SHOTS = "Q: what is a cat?\nA: an animal\n\nQ: what is a car?\nA: a vehicle\n\n"


def _embed(
    loaded: LoadedModel, prompt: str, samples: Sequence[str], layer: int
) -> np.ndarray:
    """The (K, d) answer embeddings of one prompt's samples."""
    return answer_activations(
        loaded, [prompt] * len(samples), list(samples), layers=[layer]
    )[0]


def test_eigenscore_matches_a_hand_computed_covariance() -> None:
    # Rows centred by their own mean: (-1, 0, 1) and (1, 0, -1). Divided by
    # d - 1 = 2 the covariance is [[1, -1], [-1, 1]], eigenvalues 0 and 2.
    z = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])

    score = eigenscore(z, alpha=1e-3)

    assert score == pytest.approx((math.log(1e-3) + math.log(2 + 1e-3)) / 2)


def test_eigenscore_of_orthogonal_rows() -> None:
    # Centred rows (1, -1, 0, 0) and (0, 0, 1, -1): covariance 2/3 I.
    z = np.array([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]])

    assert eigenscore(z, alpha=0.5) == pytest.approx(math.log(2 / 3 + 0.5))


def _rank_one(row: np.ndarray, k: int, alpha: float = 1e-3) -> float:
    """The score of k copies of one row: one eigenvalue k * var, k - 1 zero."""
    variance = float(np.var(row, ddof=1))
    return ((k - 1) * math.log(alpha) + math.log(k * variance + alpha)) / k


def test_identical_samples_leave_one_eigenvalue_above_alpha() -> None:
    row = np.array([0.3, -1.2, 4.0, 2.2])

    score = eigenscore(np.tile(row, (5, 1)), alpha=1e-3)

    assert score == pytest.approx(_rank_one(row, 5))


def test_a_constant_shift_of_one_sample_does_not_change_the_score() -> None:
    rng = np.random.default_rng(0)
    z = rng.standard_normal((5, 16))
    shifted = z.copy()
    shifted[2] += 7.0

    assert eigenscore(shifted) == pytest.approx(eigenscore(z))


def test_more_divergent_samples_score_higher() -> None:
    rng = np.random.default_rng(1)
    centre = rng.standard_normal(32)
    noise = rng.standard_normal((5, 32))

    close = eigenscore(centre + 0.01 * noise)
    far = eigenscore(centre + 1.0 * noise)

    assert far > close


def test_the_default_alpha_is_the_reference_value() -> None:
    z = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])

    assert eigenscore(z) == eigenscore(z, alpha=1e-3)


@pytest.mark.parametrize(
    ("z", "alpha", "message"),
    [
        (np.ones((1, 4)), 1e-3, "at least 2 samples"),
        (np.ones((3, 1)), 1e-3, "at least 2 features"),
        (np.ones(4), 1e-3, "shape"),
        (np.array([[1.0, np.nan], [0.0, 1.0]]), 1e-3, "finite"),
        (np.ones((2, 3)), 0.0, "alpha must be positive"),
    ],
)
def test_eigenscore_rejects_bad_input(
    z: np.ndarray, alpha: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        eigenscore(z, alpha=alpha)


def test_the_layer_is_half_the_hidden_states_as_in_the_reference(
    tiny_arch: LoadedModel,
) -> None:
    # The reference takes int(len(hidden_states) / 2) over L + 1 states.
    assert tiny_arch.n_layers == 3
    assert eigen_layer(tiny_arch) == 2


def test_scores_use_the_answer_activations_at_the_middle_layer(
    tiny_arch: LoadedModel,
) -> None:
    scores = eigenscores(tiny_arch, PROMPTS[:2], SAMPLES[:2])

    layer = eigen_layer(tiny_arch)
    for i in range(2):
        z = _embed(tiny_arch, PROMPTS[i], SAMPLES[i], layer)
        assert scores[i] == pytest.approx(eigenscore(z), rel=1e-4, abs=1e-4)


def test_an_explicit_layer_and_alpha_are_used(tiny_arch: LoadedModel) -> None:
    scores = eigenscores(
        tiny_arch, PROMPTS[:1], SAMPLES[:1], layer=0, alpha=0.1
    )

    z = _embed(tiny_arch, PROMPTS[0], SAMPLES[0], 0)
    assert scores[0] == pytest.approx(
        eigenscore(z, alpha=0.1), rel=1e-4, abs=1e-4
    )


def test_empty_samples_are_dropped(tiny_arch: LoadedModel) -> None:
    scores = eigenscores(tiny_arch, PROMPTS[2:], SAMPLES[2:])

    kept = eigenscores(tiny_arch, PROMPTS[2:], [("1990", "1991")])
    assert scores[0] == pytest.approx(kept[0], rel=1e-4, abs=1e-4)


def test_fewer_than_two_answers_give_nan(tiny_arch: LoadedModel) -> None:
    scores = eigenscores(
        tiny_arch, PROMPTS, [("Paris", ""), ("", ""), ("a", "b")]
    )

    assert np.isnan(scores[:2]).all()
    assert np.isfinite(scores[2])


def test_identical_answers_give_the_rank_one_score(
    tiny_arch: LoadedModel,
) -> None:
    scores = eigenscores(tiny_arch, PROMPTS[:1], [("Paris",) * 4])

    z = _embed(tiny_arch, PROMPTS[0], ["Paris"], eigen_layer(tiny_arch))[0]
    assert scores[0] == pytest.approx(_rank_one(z, 4), rel=1e-4, abs=1e-4)


def test_scores_do_not_depend_on_the_batch_size(tiny_arch: LoadedModel) -> None:
    single = eigenscores(tiny_arch, PROMPTS, SAMPLES, batch_size=1)
    batched = eigenscores(tiny_arch, PROMPTS, SAMPLES, batch_size=8)

    np.testing.assert_allclose(single, batched, rtol=1e-4, atol=1e-4)


def test_the_shared_prefix_matches_the_default_path(
    tiny_arch: LoadedModel,
) -> None:
    prompts = [SHOTS + p for p in PROMPTS]

    default = eigenscores(tiny_arch, prompts, SAMPLES)
    shared = eigenscores(tiny_arch, prompts, SAMPLES, share_prefix=True)

    np.testing.assert_allclose(shared, default, rtol=1e-4, atol=1e-4)


def test_scores_reject_unequal_lengths(tiny_arch: LoadedModel) -> None:
    with pytest.raises(ValueError, match="one sample set per prompt"):
        eigenscores(tiny_arch, PROMPTS, SAMPLES[:2])
