from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
import pytest
from scipy.special import expit
import torch

from tomography.reading import ccs
from tomography.reading.ccs import CCSProbe
from tomography.reading.ccs import fit_ccs
from tomography.reading.reader import RepReader


class ContrastPairs(NamedTuple):
    h: NDArray[np.float32]
    pairs: NDArray[np.int64]
    labels: NDArray[np.int64]
    direction: NDArray[np.float64]


def make_contrast_pairs(
    n_pairs: int,
    d: int = 16,
    *,
    n_layers: int = 1,
    seed: int = 0,
    signal: float = 4.0,
    offset: float = 3.0,
) -> ContrastPairs:
    """Builds statement and negation pairs that differ in truth.

    Row 2i is a statement and row 2i + 1 its negation; exactly one of them
    is true. Truth adds `signal` along a planted unit direction, and the
    two sets are shifted by `offset` in opposite directions on every
    feature, which the per-set centring removes.
    """
    rng = np.random.default_rng(seed)
    planted = rng.standard_normal(d)
    planted /= np.linalg.norm(planted)
    statement_true = rng.random(n_pairs) < 0.5
    base = rng.standard_normal((n_layers, n_pairs, d))
    truth = np.where(statement_true, 1.0, -1.0)[None, :, None]
    statement = base + signal * truth * planted + offset
    negation = base - signal * truth * planted - offset
    statement += 0.3 * rng.standard_normal(statement.shape)
    negation += 0.3 * rng.standard_normal(negation.shape)
    h = np.empty((n_layers, 2 * n_pairs, d))
    h[:, 0::2] = statement
    h[:, 1::2] = negation
    labels = np.empty(2 * n_pairs, dtype=np.int64)
    labels[0::2] = statement_true
    labels[1::2] = ~statement_true
    pairs = np.arange(2 * n_pairs, dtype=np.int64).reshape(n_pairs, 2)
    return ContrastPairs(h.astype(np.float32), pairs, labels, planted)


def abs_cosine(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    return float(abs(a64 @ b64) / (np.linalg.norm(a64) * np.linalg.norm(b64)))


def member_probabilities(
    probe: CCSProbe, h: NDArray[np.float32], pairs: NDArray[np.int64], k: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """p(x+) and p(x-) of layer k, each set centred on its training mean."""
    rows = h[k].astype(np.float64)
    first = expit(
        (rows[pairs[:, 0]] - probe.first_means[k]) @ probe.weights[k]
        + probe.biases[k]
    )
    second = expit(
        (rows[pairs[:, 1]] - probe.second_means[k]) @ probe.weights[k]
        + probe.biases[k]
    )
    return first, second


def ccs_loss(
    probe: CCSProbe, h: NDArray[np.float32], pairs: NDArray[np.int64]
) -> NDArray[np.float64]:
    """The CCS loss of each layer, from its definition."""
    losses = []
    for k in range(h.shape[0]):
        first, second = member_probabilities(probe, h, pairs, k)
        consistency = (first - (1.0 - second)) ** 2
        confidence = np.minimum(first, second) ** 2
        losses.append(float(np.mean(consistency + confidence)))
    return np.array(losses)


FAST = {"restarts": 2, "epochs": 500}


def test_ccs_recovers_the_planted_truth_direction() -> None:
    data = make_contrast_pairs(400, seed=1, signal=2.0)

    probe = fit_ccs(data.h, [5], pairs=data.pairs, labels=data.labels, **FAST)

    assert abs_cosine(probe.reader.directions[0], data.direction) > 0.95
    assert abs_cosine(probe.weights[0], data.direction) > 0.95


def test_ccs_probabilities_are_consistent_on_contrast_pairs() -> None:
    data = make_contrast_pairs(180, seed=2)
    train = data.pairs[:120]
    held_out = data.pairs[120:]

    probe = fit_ccs(data.h, [0], pairs=train, labels=data.labels, **FAST)

    first, second = member_probabilities(probe, data.h, train, 0)
    assert np.mean(np.abs(first + second - 1.0)) < 0.1
    assert probe.losses[0] < 0.02
    probability = probe.pair_probabilities(data.h, held_out)[0]
    first_true = data.labels[held_out[:, 0]] == 1
    assert ((probability > 0.5) == first_true).mean() > 0.95


def test_ccs_pair_probability_is_the_averaged_contrast_prediction() -> None:
    probe = CCSProbe(
        reader=RepReader.from_arrays(
            "ccs", [0], np.array([[1.0, 0.0]]), np.zeros((1, 2))
        ),
        weights=np.array([[2.0, 0.0]]),
        biases=np.array([0.5]),
        first_means=np.array([[1.0, 1.0]]),
        second_means=np.array([[-1.0, 0.0]]),
        losses=np.array([0.0]),
    )
    h = np.array([[[2.0, 5.0], [0.0, 3.0]]], dtype=np.float32)
    pairs = np.array([[0, 1]], dtype=np.int64)

    probability = probe.pair_probabilities(h, pairs)

    first = expit(2.0 * (2.0 - 1.0) + 0.5)
    second = expit(2.0 * (0.0 + 1.0) + 0.5)
    expected = 0.5 * (first + 1.0 - second)
    assert probability.shape == (1, 1)
    assert probability.dtype == np.float32
    assert probability[0, 0] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("offset", [0.0, 3.0])
def test_ccs_is_oriented_on_the_training_labels(offset: float) -> None:
    data = make_contrast_pairs(100, seed=4, n_layers=2, offset=offset)

    probe = fit_ccs(
        data.h, [0, 1], pairs=data.pairs, labels=data.labels, **FAST
    )
    flipped = fit_ccs(
        data.h, [0, 1], pairs=data.pairs, labels=1 - data.labels, **FAST
    )

    first_true = data.labels[data.pairs[:, 0]] == 1
    probability = probe.pair_probabilities(data.h, data.pairs)
    assert ((probability > 0.5) == first_true).mean() > 0.95
    np.testing.assert_array_equal(flipped.reader.signs, -probe.reader.signs)
    np.testing.assert_allclose(
        flipped.pair_probabilities(data.h, data.pairs),
        1.0 - probability,
        atol=1e-6,
    )


def test_ccs_reader_compares_pairs_when_the_sets_share_a_mean() -> None:
    data = make_contrast_pairs(100, seed=8, offset=0.0)

    probe = fit_ccs(data.h, [0], pairs=data.pairs, labels=data.labels, **FAST)

    scores = probe.reader.score(data.h)[0]
    first_true = data.labels[data.pairs[:, 0]] == 1
    first_higher = scores[data.pairs[:, 0]] > scores[data.pairs[:, 1]]
    assert (first_higher == first_true).mean() > 0.95


def test_ccs_reader_centres_on_the_mean_of_both_sets() -> None:
    data = make_contrast_pairs(50, seed=5)

    probe = fit_ccs(data.h, [0], pairs=data.pairs, labels=data.labels, **FAST)

    rows = data.h[0].astype(np.float64)
    np.testing.assert_allclose(
        probe.first_means[0], rows[data.pairs[:, 0]].mean(axis=0), atol=1e-5
    )
    np.testing.assert_allclose(
        probe.second_means[0], rows[data.pairs[:, 1]].mean(axis=0), atol=1e-5
    )
    np.testing.assert_allclose(
        probe.reader.means[0],
        0.5 * (probe.first_means[0] + probe.second_means[0]),
        atol=1e-5,
    )
    assert probe.reader.method == "ccs"
    assert probe.reader.layers == (0,)


def test_ccs_is_deterministic_for_a_seed() -> None:
    data = make_contrast_pairs(60, seed=6, n_layers=2)

    first = fit_ccs(
        data.h, [0, 1], pairs=data.pairs, labels=data.labels, seed=3, **FAST
    )
    second = fit_ccs(
        data.h, [0, 1], pairs=data.pairs, labels=data.labels, seed=3, **FAST
    )

    np.testing.assert_array_equal(first.weights, second.weights)
    np.testing.assert_array_equal(first.biases, second.biases)
    np.testing.assert_array_equal(first.losses, second.losses)
    np.testing.assert_array_equal(
        first.reader.directions, second.reader.directions
    )


def test_more_restarts_never_raise_the_unsupervised_loss() -> None:
    data = make_contrast_pairs(40, d=24, seed=7, signal=0.3)
    rows = data.h[0].astype(np.float64)
    first = rows[data.pairs[:, 0]] - rows[data.pairs[:, 0]].mean(axis=0)
    second = rows[data.pairs[:, 1]] - rows[data.pairs[:, 1]].mean(axis=0)

    losses = [
        ccs._fit_layer(first, second, restarts=r, epochs=200, lr=0.01, seed=4)[
            2
        ]
        for r in (1, 5)
    ]

    assert losses[1] <= losses[0]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"h": np.zeros((4, 3))}, r"shape \(K, n, d\)"),
        ({"layers": [0, 1]}, "layers"),
        ({"pairs": np.array([[0, 1, 2]])}, r"shape \(m, 2\)"),
        ({"pairs": np.array([[0, 1]])}, "at least 2 pairs"),
        ({"pairs": np.array([[0, 1], [2, 99]])}, "index the rows"),
        ({"pairs": np.array([[0, 1], [1, 2]])}, "both a first and a second"),
        ({"labels": np.array([0, 1, 2, 0, 1, 0, 1, 0])}, "labels"),
        ({"restarts": 0}, "restarts"),
        ({"epochs": 0}, "epochs"),
        ({"lr": 0.0}, "lr"),
        ({"h": np.full((1, 8, 3), np.inf)}, "finite"),
        ({"pairs": np.array([[0.0, 1.0], [2.0, 3.0]])}, "integer"),
        ({"h": np.ones((1, 8, 3))}, "variance"),
    ],
    ids=[
        "flat-h",
        "layers",
        "pair-shape",
        "single-pair",
        "pair-range",
        "shared-member",
        "labels",
        "restarts",
        "epochs",
        "lr",
        "non-finite",
        "float-pairs",
        "no-variance",
    ],
)
def test_ccs_rejects_malformed_inputs(
    change: dict[str, object], message: str
) -> None:
    data = make_contrast_pairs(4, d=3)
    arguments: dict[str, object] = {
        "h": data.h,
        "layers": [0],
        "pairs": data.pairs,
        "labels": data.labels,
        "restarts": 1,
        "epochs": 1,
        "lr": 0.01,
    }
    arguments.update(change)

    with pytest.raises(ValueError, match=message):
        fit_ccs(**arguments)  # type: ignore[arg-type]


def test_pair_probabilities_reject_activations_of_another_width() -> None:
    data = make_contrast_pairs(10, d=3)
    probe = fit_ccs(
        data.h, [0], pairs=data.pairs, labels=data.labels, restarts=1, epochs=1
    )

    with pytest.raises(ValueError, match="width 3"):
        probe.pair_probabilities(np.zeros((1, 4, 5)), data.pairs[:2])


@pytest.mark.parametrize("flip", [False, True])
def test_ccs_losses_belong_to_the_oriented_probe(flip: bool) -> None:
    data = make_contrast_pairs(60, seed=9, n_layers=2, signal=1.0)
    labels = 1 - data.labels if flip else data.labels

    probe = fit_ccs(data.h, [0, 1], pairs=data.pairs, labels=labels, **FAST)

    np.testing.assert_allclose(
        probe.losses, ccs_loss(probe, data.h, data.pairs), atol=1e-9
    )


def test_ccs_checks_both_label_classes_before_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = make_contrast_pairs(10, d=3)

    def never(*args: object, **kwargs: object) -> None:
        raise AssertionError("trained before validating the labels")

    monkeypatch.setattr(ccs, "_fit_layer", never)

    with pytest.raises(ValueError, match="both classes"):
        fit_ccs(
            data.h,
            [0],
            pairs=data.pairs,
            labels=np.ones_like(data.labels),
            **FAST,
        )


def test_pair_probabilities_saturate_without_overflow() -> None:
    probe = CCSProbe(
        reader=RepReader.from_arrays(
            "ccs", [0], np.array([[1.0]]), np.zeros((1, 1))
        ),
        weights=np.array([[1000.0]]),
        biases=np.array([0.0]),
        first_means=np.zeros((1, 1)),
        second_means=np.zeros((1, 1)),
        losses=np.array([0.0]),
    )
    h = np.array([[[-5.0], [5.0]]], dtype=np.float32)

    probability = probe.pair_probabilities(h, np.array([[0, 1]]))

    assert probability[0, 0] == 0.0


def test_ccs_trains_on_rows_of_average_norm_sqrt_d(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = make_contrast_pairs(30, d=9, seed=10)
    seen: list[tuple[torch.Tensor, torch.Tensor]] = []
    real = ccs._ccs_loss

    def spy(
        first: torch.Tensor,
        second: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        seen.append((first, second))
        return real(first, second, weight, bias)

    monkeypatch.setattr(ccs, "_ccs_loss", spy)

    fit_ccs(
        100.0 * data.h,
        [0],
        pairs=data.pairs,
        labels=data.labels,
        restarts=1,
        epochs=1,
    )

    first, second = seen[0]
    rows = torch.cat([first, second]).numpy()
    assert np.linalg.norm(rows, axis=1).mean() == pytest.approx(3.0)
    np.testing.assert_allclose(first.numpy().mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(second.numpy().mean(axis=0), 0.0, atol=1e-12)


def test_ccs_needs_both_classes_among_the_pair_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(12)
    h = rng.standard_normal((1, 6, 3)).astype(np.float32)
    pairs = np.array([[0, 1], [2, 3]], dtype=np.int64)
    labels = np.array([1, 1, 1, 1, 0, 0], dtype=np.int64)

    def never(*args: object, **kwargs: object) -> None:
        raise AssertionError("trained before validating the labels")

    monkeypatch.setattr(ccs, "_fit_layer", never)

    with pytest.raises(ValueError, match=r"pair members.*both classes"):
        fit_ccs(h, [0], pairs=pairs, labels=labels, **FAST)
