import numpy as np
from numpy.typing import NDArray
import pytest
from scipy.special import expit
from sklearn.covariance import ledoit_wolf_shrinkage
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from tomography.reading.directions import FloatArray
from tomography.reading.directions import fit_direction
from tomography.reading.probes import Reader
from tomography.reading.probes import fit_lda
from tomography.reading.probes import fit_mlp
from tomography.reading.reader import RepReader
from tomography.reading.reader import _layer_seed
from tomography.reading.reader import orient


def correlated_classes(
    n: int, d: int, seed: int
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Two layers of two classes under correlated noise and an offset.

    The noise is mixed across features, so the LDA direction differs from
    the difference of the class means. Rows 2i and 2i + 1 have labels 0
    and 1.
    """
    rng = np.random.default_rng(seed)
    labels = (np.arange(n) % 2).astype(np.int64)
    noise = rng.standard_normal((2, n, d)) @ rng.standard_normal((d, d))
    shift = labels[:, None] * rng.standard_normal((2, 1, d))
    return (noise + shift + 5.0).astype(np.float32), labels


def unit(vector: NDArray[np.floating]) -> NDArray[np.float64]:
    as64 = vector.astype(np.float64)
    return as64 / np.linalg.norm(as64)


def class_centered(
    h: NDArray[np.float32], labels: NDArray[np.int64]
) -> NDArray[np.float64]:
    rows = h.astype(np.float64)
    for value in (0, 1):
        rows[:, labels == value] -= rows[:, labels == value].mean(axis=1)[
            :, None
        ]
    return rows


def test_rep_reader_satisfies_the_reader_protocol() -> None:
    reader = RepReader.from_arrays(
        "hand", [4, 9], np.eye(2), np.zeros((2, 2)), np.array([1.0, -1.0])
    )
    h = np.arange(12, dtype=np.float32).reshape(2, 3, 2)

    scorer: Reader[FloatArray] = reader
    scores: NDArray[np.float32] = scorer.score(h)

    assert isinstance(reader, Reader)
    assert scorer.method == "hand"
    assert scorer.layers == (4, 9)
    assert scores.dtype == np.float32
    np.testing.assert_array_equal(scores, [[0, 2, 4], [-7, -9, -11]])


def test_lda_reader_is_the_oriented_shrunk_mass_mean_direction() -> None:
    h, labels = correlated_classes(n=60, d=8, seed=5)
    pairs = np.arange(60, dtype=np.int64).reshape(30, 2)

    reader = fit_lda(h, [3, 7], labels=labels, pairs=pairs, shrinkage=0.3)

    mass_mean = fit_lda(h, [3, 7], labels=labels, pairs=pairs, shrinkage=1.0)
    assert isinstance(reader, RepReader)
    assert reader.method == "lda"
    assert reader.layers == (3, 7)
    for name in ("directions", "means", "signs"):
        assert getattr(reader, name).dtype == np.float32
    np.testing.assert_allclose(
        reader.means, h.astype(np.float64).mean(axis=1), atol=1e-5
    )
    np.testing.assert_array_equal(
        reader.signs, orient(reader, h, labels=labels, pairs=pairs).signs
    )
    for k in range(2):
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.3)
        coef = lda.fit(h[k].astype(np.float64), labels).coef_[0]
        oriented = reader.signs[k] * reader.directions[k]
        np.testing.assert_allclose(oriented, unit(coef), atol=1e-5)
        gap = fit_direction("mean_diff", h[k], labels=labels).direction
        assert abs(unit(gap) @ unit(oriented)) < 0.99
        np.testing.assert_allclose(
            mass_mean.signs[k] * mass_mean.directions[k],
            unit(h[k, labels == 1].mean(0) - h[k, labels == 0].mean(0)),
            atol=1e-5,
        )


def test_lda_default_shrinkage_is_the_ledoit_wolf_weight() -> None:
    h, labels = correlated_classes(n=20, d=50, seed=6)

    reader = fit_lda(h, [0, 1], labels=labels)

    centered = class_centered(h, labels)
    for k in range(2):
        weight = ledoit_wolf_shrinkage(centered[k], assume_centered=True)
        assert 0.0 < weight < 1.0
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=weight)
        coef = lda.fit(h[k].astype(np.float64), labels).coef_[0]
        np.testing.assert_allclose(
            reader.signs[k] * reader.directions[k], unit(coef), atol=1e-5
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"shrinkage": 0.0}, "shrinkage"),
        ({"shrinkage": 1.5}, "shrinkage"),
        ({"labels": np.zeros(20, dtype=np.int64)}, "both classes"),
        ({"labels": np.zeros(19, dtype=np.int64)}, "labels"),
        ({"h": np.zeros((20, 4), dtype=np.float32)}, "shape"),
        ({"layers": [0]}, "layers"),
    ],
    ids=[
        "no-shrinkage",
        "over-shrinkage",
        "one-class",
        "label-count",
        "two-dimensional",
        "layer-count",
    ],
)
def test_lda_rejects_inconsistent_inputs(
    change: dict[str, object], message: str
) -> None:
    h, labels = correlated_classes(n=20, d=4, seed=7)
    arguments: dict[str, object] = {
        "h": h,
        "layers": [0, 1],
        "labels": labels,
        **change,
    }

    with pytest.raises(ValueError, match=message):
        fit_lda(**arguments)  # type: ignore[arg-type]


def standardized(
    h: NDArray[np.float32], reference: NDArray[np.float32]
) -> NDArray[np.float64]:
    """Scales each feature of `h` by the mean and deviation of `reference`."""
    rows = reference.astype(np.float64)
    return (h - rows.mean(axis=1, keepdims=True)) / rows.std(
        axis=1, keepdims=True
    )


def test_mlp_reader_scores_each_layer_with_its_own_seeded_mlp() -> None:
    h, labels = correlated_classes(n=60, d=8, seed=8)
    test, _ = correlated_classes(n=20, d=8, seed=9)

    reader = fit_mlp(
        h, [3, 7], labels=labels, hidden_sizes=(16, 8), alpha=0.1, seed=4
    )

    scores = reader.score(test)
    again = fit_mlp(
        h, [3, 7], labels=labels, hidden_sizes=(16, 8), alpha=0.1, seed=4
    )
    other = fit_mlp(
        h, [3, 7], labels=labels, hidden_sizes=(16, 8), alpha=0.1, seed=5
    )
    assert isinstance(reader, Reader)
    assert reader.method == "mlp"
    assert reader.layers == (3, 7)
    assert not hasattr(reader, "directions")
    assert scores.dtype == np.float32
    assert scores.shape == (2, 20)
    np.testing.assert_array_equal(scores, again.score(test))
    assert not np.allclose(scores, other.score(test))
    inputs = standardized(test, h)
    for k, model in enumerate(reader.models):
        assert model.hidden_layer_sizes == (16, 8)
        assert model.alpha == 0.1
        assert model.random_state == _layer_seed(4, k)
        probability = model.predict_proba(inputs[k])[:, 1]
        np.testing.assert_allclose(expit(scores[k]), probability, rtol=1e-5)
    with pytest.raises(ValueError, match="shape"):
        reader.score(test[:1])


def test_joint_mlp_reader_scores_the_concatenated_layers() -> None:
    h, labels = correlated_classes(n=60, d=8, seed=8)
    test, _ = correlated_classes(n=20, d=8, seed=9)

    reader = fit_mlp(
        h,
        [3, 7],
        labels=labels,
        hidden_sizes=(16,),
        alpha=0.1,
        seed=4,
        joint=True,
    )

    scores = reader.score(test)
    (model,) = reader.models
    inputs = np.concatenate(list(standardized(test, h)), axis=1)
    assert isinstance(reader, Reader)
    assert reader.method == "mlp_joint"
    assert reader.layers == (3, 7)
    assert scores.dtype == np.float32
    assert scores.shape == (1, 20)
    assert model.n_features_in_ == 16
    assert model.random_state == 4
    np.testing.assert_allclose(
        expit(scores[0]), model.predict_proba(inputs)[:, 1], rtol=1e-5
    )
