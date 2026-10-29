from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import replace
from functools import partial
import tracemalloc

import numpy as np
from numpy.typing import NDArray
import pytest
from scipy.special import expit
from sklearn.covariance import ledoit_wolf_shrinkage
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from tomography.activations import Activations
from tomography.reading.directions import FloatArray
from tomography.reading.directions import fit_direction
from tomography.reading.metrics import auroc
from tomography.reading.probes import AttentionReader
from tomography.reading.probes import CellMaxReader
from tomography.reading.probes import Reader
from tomography.reading.probes import fit_attention
from tomography.reading.probes import fit_cell_max
from tomography.reading.probes import fit_lda
from tomography.reading.probes import fit_mlp
from tomography.reading.probes import fit_ttpd
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
        members = labels == value
        rows[:, members] -= rows[:, members].mean(axis=1, keepdims=True)
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


@pytest.mark.parametrize("shrinkage", [None, 0.5], ids=["default", "given"])
def test_lda_rejects_rows_without_within_class_variance(
    shrinkage: float | None,
) -> None:
    labels = (np.arange(10) % 2).astype(np.int64)
    h = np.tile(np.eye(2, 4)[labels], (2, 1, 1)).astype(np.float32)

    with pytest.raises(ValueError, match="no within-class variance"):
        fit_lda(h, [0, 1], labels=labels, shrinkage=shrinkage)


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

    fit = partial(
        fit_mlp, h, [3, 7], labels=labels, hidden_sizes=(16, 8), alpha=0.1
    )

    reader = fit(seed=4)

    scores = reader.score(test)
    again = fit(seed=4)
    other = fit(seed=5)
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


def test_mlp_reader_scores_one_layer_at_a_time() -> None:
    h, labels = correlated_classes(n=40, d=32, seed=8)
    reader = fit_mlp(
        np.repeat(h, 16, axis=0),
        list(range(32)),
        labels=labels,
        hidden_sizes=(4,),
        alpha=1.0,
    )
    test = np.random.default_rng(9).normal(size=(32, 2000, 32))
    test = test.astype(np.float32)

    tracemalloc.start()
    try:
        scores = reader.score(test)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert scores.shape == (32, 2000)
    # A float64 copy of the whole stack alone takes twice its bytes.
    assert peak < test.nbytes / 2


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


def token_rows(
    data: FloatArray,
    counts: Sequence[int] | NDArray[np.int64],
    layers: Sequence[int],
) -> Activations:
    """Wraps rows as `response` activations, `counts[i]` for stimulus i."""
    rank = np.concatenate([np.arange(count) for count in counts])
    return Activations(
        data=data.astype(np.float32),
        layers=tuple(layers),
        position="response",
        row_stimulus=np.repeat(np.arange(len(counts)), counts),
        row_rank=rank,
        row_token=rank,
        row_token_id=np.zeros_like(rank),
        n_stimuli=len(counts),
    )


def mean_pooled(acts: Activations) -> NDArray[np.float64]:
    """The mean of each stimulus's rows, of shape (K, n_stimuli, d)."""
    return np.stack(
        [
            acts.data[:, acts.row_stimulus == i].astype(np.float64).mean(1)
            for i in range(acts.n_stimuli)
        ],
        axis=1,
    )


def reference_weights(
    acts: Activations, reader: AttentionReader
) -> NDArray[np.float64]:
    """Softmax weight of every cell, computed stimulus by stimulus."""
    rows = acts.data.astype(np.float64)
    logits = np.einsum("krd,kd->kr", rows, reader.queries)
    logits += reader.offsets[:, None]
    weights = np.zeros_like(logits)
    for i in range(acts.n_stimuli):
        cells = acts.row_stimulus == i
        block = logits[:, cells]
        if reader.joint:
            raw = np.exp(block - block.max())
            weights[:, cells] = raw / raw.sum()
        else:
            raw = np.exp(block - block.max(axis=1, keepdims=True))
            weights[:, cells] = raw / raw.sum(axis=1, keepdims=True)
    return weights


def cell_values(
    acts: Activations, reader: AttentionReader | CellMaxReader
) -> NDArray[np.float64]:
    """The value of every cell, of shape (K, n_rows)."""
    rows = acts.data.astype(np.float64)
    values: NDArray[np.float64] = np.einsum("krd,kd->kr", rows, reader.values)
    return values + reader.biases[:, None]


def reference_scores(
    acts: Activations, reader: AttentionReader
) -> NDArray[np.float64]:
    """The softmax-weighted values of each stimulus, one at a time."""
    weighted = reference_weights(acts, reader) * cell_values(acts, reader)
    if reader.joint:
        weighted = weighted.sum(axis=0, keepdims=True)
    return np.stack(
        [
            weighted[:, acts.row_stimulus == i].sum(axis=1)
            for i in range(acts.n_stimuli)
        ],
        axis=1,
    )


def hand_reader(
    rng: np.random.Generator, queries: NDArray[np.float64]
) -> AttentionReader:
    """A per-layer reader on layers 2 and 5 with random value heads."""
    return AttentionReader(
        layers=(2, 5),
        joint=False,
        queries=queries,
        offsets=np.array([0.5, -1.0]),
        values=rng.standard_normal((2, 6)),
        biases=rng.standard_normal(2),
        signs=np.ones(2, dtype=np.float32),
    )


def test_attention_reader_with_a_zero_query_scores_the_mean_pooled_tokens() -> (
    None
):
    rng = np.random.default_rng(11)
    data = 4.0 * rng.standard_normal((2, 15, 6))
    acts = token_rows(data, [3, 1, 4, 2, 5], [2, 5])
    reader = hand_reader(rng, np.zeros((2, 6)))

    scores = reader.score(acts)
    joint = replace(reader, joint=True, offsets=np.zeros(2)).score(acts)

    linear = np.einsum("knd,kd->kn", mean_pooled(acts), reader.values)
    linear += reader.biases[:, None]
    assert scores.dtype == np.float32
    assert scores.shape == (2, 5)
    np.testing.assert_allclose(scores, linear, rtol=1e-6, atol=1e-6)
    assert joint.dtype == np.float32
    assert joint.shape == (1, 5)
    np.testing.assert_allclose(
        joint, linear.mean(axis=0, keepdims=True), rtol=1e-6, atol=1e-6
    )


def test_attention_reader_softmaxes_over_the_cells_of_each_stimulus() -> None:
    rng = np.random.default_rng(12)
    acts = token_rows(rng.standard_normal((2, 15, 6)), [3, 1, 4, 2, 5], [2, 5])
    reader = hand_reader(rng, rng.standard_normal((2, 6)))
    joint = replace(reader, joint=True)

    scores = reader.score(acts)
    joint_scores = joint.score(acts)

    scorer: Reader[Activations] = reader
    moved = np.array([3.0, -2.0])
    assert isinstance(reader, Reader)
    assert scorer.method == "attention"
    assert joint.method == "attention_joint"
    assert scorer.layers == (2, 5)
    np.testing.assert_allclose(
        scores, reference_scores(acts, reader), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        joint_scores, reference_scores(acts, joint), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        replace(reader, offsets=moved).score(acts), scores, rtol=1e-6
    )
    assert not np.allclose(
        replace(joint, offsets=moved).score(acts), joint_scores
    )
    with pytest.raises(ValueError, match="layers"):
        reader.score(replace(acts, layers=(2, 6)))
    with pytest.raises(ValueError, match="width"):
        reader.score(replace(acts, data=acts.data[:, :, :5]))


def scaled_logistic(
    rows: NDArray[np.float64], labels: NDArray[np.int64], c: float
) -> tuple[NDArray[np.float64], float]:
    """scikit-learn's logistic fit on rows scaled as the cell fits scale them.

    The rows are centered and divided by their root mean square norm.
    Returns the weights and the bias in the units of `rows`.
    """
    center = rows.mean(axis=0)
    scale = np.linalg.norm(rows - center) / np.sqrt(len(rows))
    model = LogisticRegression(C=c, tol=1e-10, max_iter=10000)
    model.fit((rows - center) / scale, labels)
    weights = model.coef_[0] / scale
    return weights, float(model.intercept_[0] - weights @ center)


def test_attention_reader_on_one_token_per_stimulus_is_the_logistic_fit() -> (
    None
):
    h, labels = correlated_classes(n=60, d=8, seed=13)
    acts = token_rows(h, [1] * 60, [3, 7])

    reader = fit_attention(acts, labels=labels, c=0.5, tol=1e-10)

    assert reader.method == "attention"
    assert reader.layers == (3, 7)
    np.testing.assert_allclose(reader.queries, 0.0, atol=1e-12)
    for k in range(2):
        rows = h[k].astype(np.float64)
        weights, bias = scaled_logistic(rows, labels, c=0.5)
        np.testing.assert_allclose(reader.values[k], weights, rtol=1e-5)
        np.testing.assert_allclose(reader.biases[k], bias, rtol=1e-5)
        np.testing.assert_allclose(
            reader.score(acts)[k], rows @ weights + bias, rtol=1e-5, atol=1e-5
        )


def planted_tokens(
    n: int, seed: int, n_layers: int = 1
) -> tuple[Activations, NDArray[np.int64], NDArray[np.int64]]:
    """Five tokens per stimulus, of which one carries the label.

    On layer 0, feature 0 marks one token of each stimulus and feature 1
    holds the label there. Feature 1 is loud noise on every other token and
    on every other layer, so the mean of a stimulus's tokens hides the
    label. Returns the activations, the labels and the planted rows.
    """
    rng = np.random.default_rng(seed)
    tokens = 5
    labels = (np.arange(n) % 2).astype(np.int64)
    data = rng.standard_normal((n_layers, n * tokens, 4))
    data[:, :, 1] *= 3.0
    planted = np.arange(n) * tokens + rng.integers(tokens, size=n)
    data[0, planted, 0] += 4.0
    data[0, planted, 1] = 2.0 * (2 * labels - 1) + rng.standard_normal(n)
    acts = token_rows(data, [tokens] * n, range(n_layers))
    return acts, labels, planted


def test_attention_reader_reads_a_planted_token_that_the_mean_hides() -> None:
    acts, labels, _ = planted_tokens(n=200, seed=14)
    test, test_labels, test_planted = planted_tokens(n=200, seed=15)

    reader = fit_attention(acts, labels=labels)

    again = fit_attention(acts, labels=labels)
    baseline = LogisticRegression().fit(mean_pooled(acts)[0], labels)
    mean_scores = baseline.decision_function(mean_pooled(test)[0])
    weights = reference_weights(test, reader)[0, test_planted]
    scores = reader.score(test)
    for name in ("queries", "offsets", "values", "biases", "signs"):
        np.testing.assert_array_equal(
            getattr(reader, name), getattr(again, name)
        )
    assert scores.shape == (1, 200)
    assert weights.mean() > 0.6
    assert auroc(scores[0], test_labels) > 0.95
    assert auroc(mean_scores, test_labels) < 0.8


def test_joint_attention_reader_pools_the_cells_of_all_layers() -> None:
    acts, labels, _ = planted_tokens(n=200, seed=16, n_layers=2)
    test, test_labels, planted = planted_tokens(n=200, seed=17, n_layers=2)

    reader = fit_attention(acts, labels=labels, joint=True)

    weights = reference_weights(test, reader)
    scores = reader.score(test)
    assert reader.method == "attention_joint"
    assert reader.layers == (0, 1)
    assert scores.dtype == np.float32
    assert scores.shape == (1, 200)
    assert weights[0, planted].mean() > 0.6
    assert auroc(scores[0], test_labels) > 0.95


def moved(
    rows: Activations, scale: NDArray[np.float64], shift: NDArray[np.float64]
) -> Activations:
    """The rows with each layer scaled by `scale[k]` and shifted by `shift[k]`.

    Both have shape (K, 1, 1) or (K, 1, d).
    """
    data = rows.data.astype(np.float64) * scale + shift
    return replace(rows, data=data.astype(np.float32))


def test_attention_fit_ignores_the_location_and_scale_of_each_layer() -> None:
    acts, labels, _ = planted_tokens(n=200, seed=16, n_layers=2)
    test, _, _ = planted_tokens(n=200, seed=17, n_layers=2)
    shift = 10.0 * np.random.default_rng(22).standard_normal((2, 1, 4))
    scale = np.array([0.5, 20.0])[:, None, None]
    move = partial(moved, scale=scale, shift=shift)

    reader = fit_attention(acts, labels=labels, joint=True, tol=1e-8)
    other = fit_attention(move(acts), labels=labels, joint=True, tol=1e-8)

    np.testing.assert_allclose(
        other.score(move(test)), reader.score(test), rtol=1e-3, atol=1e-3
    )


def test_attention_reader_exports_its_oriented_value_heads() -> None:
    acts, labels, _ = planted_tokens(n=60, seed=18, n_layers=2)
    pairs = np.arange(60).reshape(30, 2)

    reader = fit_attention(acts, labels=labels, pairs=pairs)
    exported = reader.rep_reader()

    expected = orient(
        exported,
        acts.data,
        labels=acts.row_values(labels),
        pairs=acts.row_groups(pairs),
    )
    norms = np.linalg.norm(reader.values, axis=1)
    assert isinstance(exported, RepReader)
    assert exported.method == "attention"
    assert exported.layers == (0, 1)
    np.testing.assert_array_equal(reader.signs, expected.signs)
    np.testing.assert_array_equal(exported.signs, expected.signs)
    np.testing.assert_allclose(
        exported.directions, reader.values / norms[:, None], atol=1e-6
    )
    np.testing.assert_allclose(
        exported.raw_score(acts.data) * norms[:, None],
        cell_values(acts, reader),
        rtol=1e-4,
        atol=1e-4,
    )


def rows_of(counts: Sequence[int], seed: int) -> Activations:
    """Random rows on layers 0 and 1, `counts[i]` for stimulus i."""
    data = np.random.default_rng(seed).standard_normal((2, sum(counts), 4))
    return token_rows(data, counts, [0, 1])


def filled_rows(value: float) -> Activations:
    """Two rows per stimulus for six stimuli, every entry equal to `value`."""
    return replace(rows_of([2] * 6, seed=20), data=np.full((2, 12, 4), value))


def constant_layer() -> Activations:
    """Two rows per stimulus for six stimuli; every entry of layer 1 is 1."""
    acts = rows_of([2] * 6, seed=49)
    data = acts.data.copy()
    data[1] = 1.0
    return replace(acts, data=data)


CELL_FITS = pytest.mark.parametrize(
    "fit", [fit_attention, fit_cell_max], ids=["attention", "cellmax"]
)


@CELL_FITS
@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"labels": np.zeros(6, dtype=np.int64)}, "both classes"),
        ({"labels": np.zeros(5, dtype=np.int64)}, "labels"),
        ({"c": 0.0}, "c must be positive"),
        ({"c": -1.0}, "c must be positive"),
        ({"acts": rows_of([2, 0, 3, 1, 2, 2], seed=19)}, "stimulus 1"),
        ({"acts": filled_rows(1.0)}, "variance"),
        ({"acts": constant_layer()}, "layer 1 have no variance"),
        ({"acts": filled_rows(np.nan)}, "finite"),
    ],
    ids=[
        "one-class",
        "label-count",
        "no-penalty",
        "negative-c",
        "no-rows",
        "constant",
        "constant-layer",
        "not-finite",
    ],
)
def test_cell_fits_reject_inconsistent_inputs(
    fit: Callable[..., object], change: dict[str, object], message: str
) -> None:
    arguments: dict[str, object] = {
        "acts": rows_of([2, 1, 3, 1, 2, 2], seed=21),
        "labels": np.array([0, 1, 0, 1, 0, 1]),
        **change,
    }

    with pytest.raises(ValueError, match=message):
        fit(**arguments)


@CELL_FITS
def test_cell_fits_warn_at_their_caller_when_they_stop_early(
    fit: Callable[..., object],
) -> None:
    acts, labels, _ = planted_tokens(n=60, seed=23)

    with pytest.warns(ConvergenceWarning, match="did not converge") as caught:
        fit(acts, labels=labels, max_iter=1)

    assert [record.filename for record in caught] == [__file__]


def reference_max(
    acts: Activations, reader: CellMaxReader
) -> NDArray[np.float64]:
    """The highest cell value of each stimulus, one stimulus at a time."""
    values = cell_values(acts, reader)
    return np.array(
        [values[:, acts.row_stimulus == i].max() for i in range(acts.n_stimuli)]
    )


def mean_pooled_logistic(
    acts: Activations, labels: NDArray[np.int64], test: Activations
) -> NDArray[np.float64]:
    """Test scores of a logistic fit on the mean rows of each stimulus.

    The means of all layers sit side by side.
    """
    rows = np.concatenate(list(mean_pooled(acts)), axis=1)
    model = LogisticRegression().fit(rows, labels)
    test_rows = np.concatenate(list(mean_pooled(test)), axis=1)
    scores: NDArray[np.float64] = model.decision_function(test_rows)
    return scores


def planted_cells(
    n: int, seed: int, tokens: tuple[int, int] = (30, 60)
) -> tuple[Activations, NDArray[np.int64], NDArray[np.int64]]:
    """Three layers of stimuli in which one cell carries label 1.

    Each stimulus has a token count drawn from `range(*tokens)`. Each label
    1 stimulus has +3 on feature 0 at one random token of layer 1, over
    noise of deviation 0.6 everywhere. The cell stands out, but the mean of
    the stimulus's tokens dilutes it. Returns the activations, the labels
    and the planted row of every stimulus (unused for label 0).
    """
    rng = np.random.default_rng(seed)
    labels = (np.arange(n) % 2).astype(np.int64)
    counts = rng.integers(*tokens, size=n)
    planted = np.cumsum(counts) - counts + rng.integers(counts)
    data = 0.6 * rng.standard_normal((3, counts.sum(), 8))
    data[1, planted[labels == 1], 0] += 3.0
    return token_rows(data, counts, [0, 1, 2]), labels, planted


def hand_max_reader(rng: np.random.Generator) -> CellMaxReader:
    """A reader on layers 2 and 5 with random value heads."""
    return CellMaxReader(
        layers=(2, 5),
        values=rng.standard_normal((2, 6)),
        biases=np.array([0.5, -1.0]),
        signs=np.ones(2, dtype=np.float32),
    )


def test_cell_max_reader_scores_each_stimulus_by_its_highest_cell() -> None:
    rng = np.random.default_rng(41)
    data = rng.standard_normal((2, 15, 6))
    acts = token_rows(data, [3, 1, 4, 2, 5], [2, 5])
    reader = hand_max_reader(rng)

    scores = reader.score(acts)

    scorer: Reader[Activations] = reader
    assert isinstance(reader, Reader)
    assert scorer.method == "cellmax"
    assert scorer.layers == (2, 5)
    assert scores.dtype == np.float32
    assert scores.shape == (1, 5)
    np.testing.assert_allclose(
        scores[0], reference_max(acts, reader), rtol=1e-6, atol=1e-6
    )
    with pytest.raises(ValueError, match="layers"):
        reader.score(replace(acts, layers=(2, 6)))
    with pytest.raises(ValueError, match="width"):
        reader.score(replace(acts, data=acts.data[:, :, :5]))
    with pytest.raises(ValueError, match="stimulus 1"):
        reader.score(token_rows(data, [3, 0, 5, 2, 5], [2, 5]))


def test_cell_max_reader_on_one_cell_per_stimulus_is_the_logistic_fit() -> None:
    h, labels = correlated_classes(n=60, d=8, seed=13)
    acts = token_rows(h[:1], [1] * 60, [3])

    reader = fit_cell_max(acts, labels=labels, c=0.5, tol=1e-10)

    rows = h[0].astype(np.float64)
    weights, bias = scaled_logistic(rows, labels, c=0.5)
    assert reader.method == "cellmax"
    assert reader.layers == (3,)
    np.testing.assert_allclose(
        reader.score(acts)[0], rows @ weights + bias, rtol=1e-4, atol=1e-4
    )


def test_cell_max_reader_finds_a_planted_cell_that_the_mean_hides() -> None:
    acts, labels, _ = planted_cells(n=200, seed=42)
    test, test_labels, test_planted = planted_cells(n=200, seed=43)

    reader = fit_cell_max(acts, labels=labels)

    again = fit_cell_max(acts, labels=labels)
    mean_scores = mean_pooled_logistic(acts, labels, test)
    scores = reader.score(test)
    values = cell_values(test, reader)
    hits = []
    for i in np.flatnonzero(test_labels == 1):
        rows = np.flatnonzero(test.row_stimulus == i)
        layer, rank = np.unravel_index(
            np.argmax(values[:, rows]), (3, len(rows))
        )
        hits.append(layer == 1 and rows[rank] == test_planted[i])
    for name in ("values", "biases", "signs"):
        np.testing.assert_array_equal(
            getattr(reader, name), getattr(again, name)
        )
    assert scores.shape == (1, 200)
    assert auroc(mean_scores, test_labels) < 0.75
    assert auroc(scores[0], test_labels) >= 0.95
    assert np.argmax(np.linalg.norm(reader.values, axis=1)) == 1
    assert np.mean(hits) >= 0.9


def test_joint_attention_reader_reads_the_planted_cell_too() -> None:
    acts, labels, _ = planted_cells(n=200, seed=42)
    test, test_labels, _ = planted_cells(n=200, seed=43)

    reader = fit_attention(acts, labels=labels, joint=True)

    assert auroc(reader.score(test)[0], test_labels) >= 0.95


def cancelling_cells(
    n: int, seed: int
) -> tuple[Activations, NDArray[np.int64]]:
    """5 to 9 tokens per stimulus; label 1 adds +4 and -4 on two of them.

    Both cells are on layer 0, feature 0, so the mean of a label 1
    stimulus's tokens has the distribution of a label 0 one.
    """
    rng = np.random.default_rng(seed)
    labels = (np.arange(n) % 2).astype(np.int64)
    counts = rng.integers(5, 10, size=n)
    starts = np.cumsum(counts) - counts
    first = rng.integers(counts)
    second = (first + 1 + rng.integers(counts - 1)) % counts
    data = rng.standard_normal((2, counts.sum(), 4))
    data[0, (starts + first)[labels == 1], 0] += 4.0
    data[0, (starts + second)[labels == 1], 0] -= 4.0
    return token_rows(data, counts, [0, 1]), labels


def test_cell_max_reader_reads_cells_that_cancel_in_the_mean() -> None:
    acts, labels = cancelling_cells(n=200, seed=44)
    test, test_labels = cancelling_cells(n=200, seed=45)

    reader = fit_cell_max(acts, labels=labels)

    mean_scores = mean_pooled_logistic(acts, labels, test)
    assert abs(auroc(mean_scores, test_labels) - 0.5) < 0.1
    assert auroc(reader.score(test)[0], test_labels) >= 0.95


def two_layer_cells(n: int, seed: int) -> tuple[Activations, NDArray[np.int64]]:
    """10 to 19 tokens per stimulus; label 1 adds +3 on one cell of a layer.

    The planted cell of a label 1 stimulus is on feature 0 of layer 0 or of
    layer 1, at random. Feature 0 is quieter on layer 0 (deviation 0.3 vs
    1), so the two layers need different thresholds.
    """
    rng = np.random.default_rng(seed)
    labels = (np.arange(n) % 2).astype(np.int64)
    counts = rng.integers(10, 20, size=n)
    planted = np.cumsum(counts) - counts + rng.integers(counts)
    layer = rng.integers(2, size=n)
    data = rng.standard_normal((2, counts.sum(), 4))
    data[0, :, 0] *= 0.3
    positive = labels == 1
    data[layer[positive], planted[positive], 0] += 3.0
    return token_rows(data, counts, [0, 1]), labels


def test_cell_max_fit_gives_each_layer_its_own_free_bias() -> None:
    acts, labels = two_layer_cells(n=200, seed=60)

    reader = fit_cell_max(acts, labels=labels, tol=1e-8)

    # The loss's gradient for the unpenalized bias of a layer is the mean
    # residual of the stimuli whose highest cell lies on that layer.
    values = cell_values(acts, reader)
    winners = np.array(
        [
            np.argmax(values[:, acts.row_stimulus == i].max(axis=1))
            for i in range(acts.n_stimuli)
        ]
    )
    residuals = expit(reference_max(acts, reader)) - labels
    for layer in (0, 1):
        assert (winners == layer).sum() > 50
        assert abs(residuals[winners == layer].sum()) / len(labels) < 2e-3


def stimuli_of(
    acts: Activations,
    chosen: Sequence[int],
    rng: np.random.Generator | None = None,
) -> Activations:
    """The rows of the chosen stimuli in that order, shuffled given `rng`."""
    blocks = [np.flatnonzero(acts.row_stimulus == i) for i in chosen]
    if rng is not None:
        blocks = [rng.permutation(block) for block in blocks]
    return token_rows(
        acts.data[:, np.concatenate(blocks)],
        [len(block) for block in blocks],
        acts.layers,
    )


def test_cell_max_scores_ignore_the_other_stimuli_and_the_row_order() -> None:
    acts, labels, _ = planted_cells(n=60, seed=46)
    reader = fit_cell_max(acts, labels=labels)
    counts = np.bincount(acts.row_stimulus)
    target = int(np.argsort(counts)[30])
    longer, shorter = int(np.argmax(counts)), int(np.argmin(counts))

    full = reader.score(acts)[0, target]
    alone = reader.score(stimuli_of(acts, [target]))[0, 0]
    among = reader.score(stimuli_of(acts, [longer, target, shorter]))[0, 1]
    permuted = reader.score(
        stimuli_of(acts, [shorter, target], np.random.default_rng(47))
    )[0, 1]

    assert counts[shorter] < counts[target] < counts[longer]
    for score in (alone, among, permuted):
        np.testing.assert_allclose(score, full, rtol=1e-6, atol=1e-6)


def test_cell_max_fit_standardizes_on_the_training_rows_only() -> None:
    acts, labels, _ = planted_cells(n=200, seed=42)
    test, _, _ = planted_cells(n=200, seed=43)
    shift = 10.0 * np.random.default_rng(48).standard_normal((3, 1, 8))
    scale = np.array([0.5, 20.0, 3.0])[:, None, None]
    moved_test = moved(test, scale, shift)

    reader = fit_cell_max(acts, labels=labels)
    other = fit_cell_max(moved(acts, scale, shift), labels=labels)

    scores = other.score(moved_test)
    np.testing.assert_allclose(scores, reader.score(test), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(
        scores[0], reference_max(moved_test, other), rtol=1e-6, atol=1e-6
    )


def test_cell_max_reader_exports_its_oriented_value_heads() -> None:
    acts, labels, _ = planted_cells(n=60, seed=53, tokens=(40, 41))
    pairs = np.arange(60).reshape(30, 2)

    reader = fit_cell_max(acts, labels=labels, pairs=pairs)
    exported = reader.rep_reader()

    expected = orient(
        exported,
        acts.data,
        labels=acts.row_values(labels),
        pairs=acts.row_groups(pairs),
    )
    norms = np.linalg.norm(reader.values, axis=1)
    means = exported.means.astype(np.float64)
    assert isinstance(exported, RepReader)
    assert exported.method == "cellmax"
    assert exported.layers == (0, 1, 2)
    np.testing.assert_array_equal(reader.signs, expected.signs)
    np.testing.assert_array_equal(exported.signs, expected.signs)
    np.testing.assert_allclose(
        exported.directions, reader.values / norms[:, None], atol=1e-6
    )
    np.testing.assert_allclose(
        np.einsum("kd,kd->k", means, reader.values) + reader.biases,
        0.0,
        atol=1e-4,
    )


def truth_and_polarity(
    n: int, d: int, seed: int
) -> tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.int64]]:
    """Two layers where negation flips a polarity-dependent truth axis.

    A statement with truth t (+1 or -1) and polarity p (+1 affirmative, -1
    negated) sits at `mu + t g + t p q` plus noise, the model of the
    reference paper; rows alternate labels 0 and 1 and the second half is
    negated.
    """
    rng = np.random.default_rng(seed)
    labels = (np.arange(n) % 2).astype(np.int64)
    polarities = np.where(np.arange(n) < n // 2, 1, -1).astype(np.int64)
    truth = 2.0 * labels - 1.0
    general, polar = rng.standard_normal((2, 2, 1, d))
    noise = 1.5 * rng.standard_normal((2, n, d))
    rows = (
        3.0
        + truth[:, None] * general
        + (truth * polarities)[:, None] * polar
        + noise
    )
    return rows.astype(np.float32), labels, polarities


def test_ttpd_without_polarities_reads_the_least_squares_truth_direction() -> (
    None
):
    h, labels = correlated_classes(n=60, d=8, seed=31)
    pairs = np.arange(60, dtype=np.int64).reshape(30, 2)

    reader = fit_ttpd(h, [3, 7], labels=labels, pairs=pairs)

    assert isinstance(reader, RepReader)
    assert reader.method == "ttpd"
    assert reader.layers == (3, 7)
    truth = (2.0 * labels - 1.0)[:, None]
    for k in range(2):
        rows = h[k].astype(np.float64)
        centered = rows - rows.mean(axis=0)
        expected = np.linalg.lstsq(truth, centered, rcond=None)[0][0]
        oriented = reader.signs[k] * reader.directions[k]
        np.testing.assert_allclose(oriented, unit(expected), atol=1e-6)
        np.testing.assert_allclose(
            reader.means[k], rows.mean(axis=0), rtol=1e-6
        )
    scores = reader.score(h)
    assert (scores[:, labels == 1].mean(axis=1) > 0).all()


def test_ttpd_with_polarities_scores_the_logistic_fit_on_two_projections() -> (
    None
):
    h, labels, polarities = truth_and_polarity(n=200, d=6, seed=33)

    reader = fit_ttpd(h, [2, 5], labels=labels, polarities=polarities)

    truth = 2.0 * labels - 1.0
    design = np.stack([truth, truth * polarities], axis=1)
    for k in range(2):
        rows = h[k].astype(np.float64)
        centered = rows - rows.mean(axis=0)
        general = np.linalg.lstsq(design, centered, rcond=None)[0][0]
        polar = LogisticRegression(C=np.inf).fit(rows, polarities == 1).coef_[0]
        projected = np.stack([rows @ general, rows @ polar], axis=1)
        final = LogisticRegression(C=np.inf).fit(projected, labels)
        weights = final.coef_[0]
        direction = weights[0] * general + weights[1] * polar
        decision = final.decision_function(projected)
        np.testing.assert_allclose(
            reader.score(h)[k] * np.linalg.norm(direction),
            decision,
            rtol=1e-4,
            atol=1e-4,
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"polarities": np.array([1, -1, 1])}, "one polarity per row"),
        ({"polarities": np.array([1, 0, 1, -1, 1, -1])}, r"-1 or \+1"),
        ({"polarities": np.ones(6, dtype=np.int64)}, "both polarities"),
        ({"labels": np.zeros(6, dtype=np.int64)}, "both classes"),
        ({"layers": [1]}, "layers has 1 entries"),
    ],
)
def test_fit_ttpd_rejects_inconsistent_inputs(
    change: dict[str, object], message: str
) -> None:
    arguments: dict[str, object] = {
        "h": np.random.default_rng(37).standard_normal((2, 6, 3)),
        "layers": [0, 1],
        "labels": np.array([0, 1, 0, 1, 0, 1]),
        **change,
    }

    with pytest.raises(ValueError, match=message):
        fit_ttpd(**arguments)  # type: ignore[arg-type]
