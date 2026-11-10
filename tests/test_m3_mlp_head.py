"""Tests for the MLP head of the M3 cell readers."""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
import json
from pathlib import Path
from typing import Any
from typing import cast
import warnings

import numpy as np
from numpy.typing import NDArray
import pytest
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
import torch

from tomography.activations import Activations
from tomography.datasets.records import Example
from tomography.experiments import m2_readers
from tomography.experiments import m3_matrix
from tomography.experiments import p2_positions
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.model import LoadedModel
from tomography.reading import probes
from tomography.reading.metrics import auroc
from tomography.reading.probes import AttentionMLPReader
from tomography.reading.probes import CellMaxMLPReader
from tomography.reading.probes import Reader
from tomography.reading.probes import RestartedFitWarning
from tomography.reading.probes import StalledFitWarning
from tomography.reading.probes import fit_attention
from tomography.reading.probes import fit_attention_mlp
from tomography.reading.probes import fit_cell_max
from tomography.reading.probes import fit_cell_max_mlp

MLP_FITS = pytest.mark.parametrize(
    "fit", [fit_attention_mlp, fit_cell_max_mlp], ids=["cellattn", "cellmax"]
)


def token_rows(
    data: NDArray[np.float64],
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


def planted(n: int, seed: int) -> tuple[Activations, NDArray[np.int64]]:
    """Two layers of short stimuli; label 1 has +3 on one cell of layer 1."""
    rng = np.random.default_rng(seed)
    labels = (np.arange(n) % 2).astype(np.int64)
    counts = rng.integers(3, 7, size=n)
    rows = np.cumsum(counts) - counts + rng.integers(counts)
    data = 0.6 * rng.standard_normal((2, counts.sum(), 4))
    data[1, rows[labels == 1], 0] += 3.0
    return token_rows(data, counts, [3, 7]), labels


def either_sign(n: int, seed: int) -> tuple[Activations, NDArray[np.int64]]:
    """Label 1 stimuli hold one cell at +4 or -4 on feature 0 of layer 1.

    The sign is drawn per stimulus, so no linear head ranks both kinds of
    planted cells above the noise cells, while |x| is one ReLU pair away.
    """
    rng = np.random.default_rng(seed)
    labels = (np.arange(n) % 2).astype(np.int64)
    counts = rng.integers(3, 6, size=n)
    rows = np.cumsum(counts) - counts + rng.integers(counts)
    data = 0.5 * rng.standard_normal((2, counts.sum(), 4))
    signs = rng.choice([-1.0, 1.0], size=n)
    data[1, rows[labels == 1], 0] += 4.0 * signs[labels == 1]
    return token_rows(data, counts, [0, 1]), labels


def same_cells_everywhere() -> tuple[Activations, NDArray[np.int64]]:
    """Six stimuli that hold the same five cells, which no head tells apart."""
    cells = np.random.default_rng(31).standard_normal((2, 5, 4))
    return token_rows(np.tile(cells, (1, 6, 1)), [5] * 6, [0, 1]), np.array(
        [0, 1] * 3
    )


def scaled_rows(
    acts: Activations,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The rows as the cell fits scale them, and the layers' scales."""
    rows = acts.data.astype(np.float64)
    rows -= rows.mean(axis=1, keepdims=True)
    scales = np.sqrt(np.square(rows).sum(axis=(1, 2)) / rows.shape[1])
    return rows / scales[:, None, None], scales


def hidden_units(
    acts: Activations, reader: AttentionMLPReader | CellMaxMLPReader
) -> NDArray[np.float64]:
    """The ReLU units of every cell, shape (K, n_rows, H)."""
    rows = acts.data.astype(np.float64)
    pre = np.einsum("krd,khd->krh", rows, reader.hidden)
    return np.maximum(pre + reader.hidden_biases[:, None, :], 0.0)


def reference_values(
    acts: Activations, reader: AttentionMLPReader | CellMaxMLPReader
) -> NDArray[np.float64]:
    """The value of every cell, shape (K, n_rows)."""
    values = np.einsum("krh,kh->kr", hidden_units(acts, reader), reader.outputs)
    return values + reader.biases[:, None]


def attention_weights(
    acts: Activations, reader: AttentionMLPReader
) -> NDArray[np.float64]:
    """The softmax weight of every cell over all cells of its stimulus."""
    rows = acts.data.astype(np.float64)
    logits = np.einsum("krd,kd->kr", rows, reader.queries)
    logits += reader.offsets[:, None]
    weights = np.zeros_like(logits)
    for i in range(acts.n_stimuli):
        cells = acts.row_stimulus == i
        raw = np.exp(logits[:, cells] - logits[:, cells].max())
        weights[:, cells] = raw / raw.sum()
    return weights


def max_weights(
    acts: Activations, reader: CellMaxMLPReader
) -> NDArray[np.float64]:
    """1 on the highest cell of every stimulus, 0 elsewhere."""
    values = reference_values(acts, reader)
    weights = np.zeros_like(values)
    for i in range(acts.n_stimuli):
        cells = np.flatnonzero(acts.row_stimulus == i)
        k, r = np.unravel_index(
            np.argmax(values[:, cells]), values[:, cells].shape
        )
        weights[k, cells[r]] = 1.0
    return weights


def hand_mlp(
    rng: np.random.Generator, hidden: int = 3
) -> tuple[AttentionMLPReader, CellMaxMLPReader]:
    """Readers on layers 2 and 5 of width 6 with random parameters."""
    common: dict[str, Any] = {
        "layers": (2, 5),
        "hidden": rng.standard_normal((2, hidden, 6)),
        "hidden_biases": rng.standard_normal((2, hidden)),
        "outputs": rng.standard_normal((2, hidden)),
    }
    attention = AttentionMLPReader(
        queries=rng.standard_normal((2, 6)),
        offsets=np.array([0.5, -1.0]),
        biases=np.full(2, 0.25),
        **common,
    )
    return attention, CellMaxMLPReader(biases=np.array([0.3, -0.2]), **common)


def test_mlp_readers_score_cells_through_one_relu_layer_per_layer() -> None:
    rng = np.random.default_rng(12)
    acts = token_rows(rng.standard_normal((2, 15, 6)), [3, 1, 4, 2, 5], [2, 5])
    attention, cellmax = hand_mlp(rng)

    values = reference_values(acts, attention)
    weighted = attention_weights(acts, attention) * values
    attention_scores = [
        weighted[:, acts.row_stimulus == i].sum() for i in range(5)
    ]
    max_values = reference_values(acts, cellmax)
    max_scores = [max_values[:, acts.row_stimulus == i].max() for i in range(5)]
    for reader, expected, method in (
        (attention, attention_scores, "attention_joint_mlp"),
        (cellmax, max_scores, "cellmax_mlp"),
    ):
        scorer: Reader[Activations] = reader
        assert isinstance(reader, Reader)
        assert scorer.method == method
        assert scorer.layers == (2, 5)
        scores = reader.score(acts)
        assert scores.dtype == np.float32
        assert scores.shape == (1, 5)
        np.testing.assert_allclose(scores[0], expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            reader.cell_values(acts), reference_values(acts, reader), rtol=1e-12
        )


def test_mlp_readers_refuse_cells_of_other_layers_or_width() -> None:
    rng = np.random.default_rng(13)
    for reader in hand_mlp(rng):
        with pytest.raises(ValueError, match="layers"):
            reader.score(token_rows(np.ones((2, 3, 6)), [3], [2, 4]))
        with pytest.raises(ValueError, match="width 6"):
            reader.score(token_rows(np.ones((2, 3, 5)), [3], [2, 5]))


def test_mlp_readers_have_no_direction_to_export() -> None:
    for reader in hand_mlp(np.random.default_rng(14)):
        with pytest.raises(
            ValueError, match="an MLP head has no single direction per layer"
        ):
            reader.rep_reader()


def quiet(fit: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit(*args, **kwargs)


@MLP_FITS
def test_mlp_fits_read_cells_that_no_linear_head_can(
    fit: Callable[..., Any],
) -> None:
    acts, labels = either_sign(80, seed=1)
    test, test_labels = either_sign(200, seed=2)
    linear = (
        fit_attention(acts, labels=labels, c=10.0, joint=True, tol=1e-10)
        if fit is fit_attention_mlp
        else quiet(fit_cell_max, acts, labels=labels, c=10.0, tol=1e-10)
    )

    reader = quiet(
        fit, acts, labels=labels, hidden=4, seed=0, c=10.0, tol=1e-10
    )

    area = float(auroc(reader.score(test)[0], test_labels))
    linear_area = float(auroc(linear.score(test)[0], test_labels))
    assert area >= 0.95
    assert area > linear_area + 0.1


def penalized_loss(
    acts: Activations,
    labels: NDArray[np.int64],
    reader: AttentionMLPReader | CellMaxMLPReader,
    c: float,
) -> float:
    """The loss the MLP fits minimize, computed from a fitted reader.

    The mean logistic loss of the scores plus 1 / (2 c n) times the squared
    norms of the queries, the hidden weights and the output weights, all
    against the scaled rows; the biases are not penalized. The output
    weights are the same in scaled and activation units, the queries and
    hidden weights are the activation ones times the layer's scale.
    """
    _, scales = scaled_rows(acts)
    values = reference_values(acts, reader)
    if isinstance(reader, AttentionMLPReader):
        weighted = attention_weights(acts, reader) * values
        scores = np.array(
            [
                weighted[:, acts.row_stimulus == i].sum()
                for i in range(len(labels))
            ]
        )
        penalty = np.square(reader.queries * scales[:, None]).sum()
    else:
        scores = np.array(
            [
                values[:, acts.row_stimulus == i].max()
                for i in range(len(labels))
            ]
        )
        penalty = 0.0
    penalty += np.square(reader.hidden * scales[:, None, None]).sum()
    penalty += np.square(reader.outputs).sum()
    loss = np.mean(np.logaddexp(0.0, scores) - labels * scores)
    return float(loss + penalty / (2.0 * c * len(labels)))


@MLP_FITS
@pytest.mark.parametrize("c", [1.0, 30.0])
def test_mlp_fits_minimize_the_logistic_loss_with_scikit_learn_c(
    fit: Callable[..., Any], c: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    acts, labels = either_sign(60, seed=3)
    runs = lbfgs_spy(monkeypatch)

    reader = quiet(fit, acts, labels=labels, hidden=4, seed=1, c=c, tol=1e-10)

    # The kept run's loss is the loss of the reader under the penalty on
    # every weight and no bias, and it is far below log 2.
    (_, result), *_ = runs
    assert len(runs) == 1
    assert np.abs(reader.biases).max() > 0.01
    assert result.fun == pytest.approx(
        penalized_loss(acts, labels, reader, c), rel=1e-9
    )
    assert result.fun < np.log(2.0) - 0.1


def lbfgs_spy(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[NDArray[np.float64], OptimizeResult]]:
    """Records the start and the result of every L-BFGS run of the probes."""
    runs: list[tuple[NDArray[np.float64], OptimizeResult]] = []

    def recording(*args: Any, **kwargs: Any) -> OptimizeResult:
        result = minimize(*args, **kwargs)
        runs.append((np.array(args[1]), result))
        return result

    monkeypatch.setattr(probes, "minimize", recording)
    return runs


@MLP_FITS
def test_mlp_fits_start_from_small_seeded_hidden_weights(
    fit: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    acts, labels = planted(20, seed=4)
    runs = lbfgs_spy(monkeypatch)
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()

    first = quiet(fit, acts, labels=labels, hidden=3, seed=7, c=1.0)
    n_first = len(runs)
    again = quiet(fit, acts, labels=labels, hidden=3, seed=7, c=1.0)
    other = quiet(fit, acts, labels=labels, hidden=3, seed=8, c=1.0)

    # The hidden weights start as a standard normal draw of the seed over
    # the square root of the width, so each unit's row has norm near 1 on
    # the scaled rows; every other parameter starts at zero.
    drawn = np.random.default_rng(7).standard_normal((2, 3, 4)) / 2.0
    start = runs[0][0]
    assert np.count_nonzero(start) == drawn.size
    np.testing.assert_array_equal(
        np.sort(start[start != 0]), np.sort(drawn.ravel())
    )
    np.testing.assert_array_equal(runs[n_first][0], start)
    for name in ("hidden", "hidden_biases", "outputs", "biases"):
        np.testing.assert_array_equal(
            getattr(again, name), getattr(first, name)
        )
    assert not np.array_equal(other.hidden, first.hidden)
    # No global random state is drawn from.
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)


@MLP_FITS
@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"labels": np.zeros(6, dtype=np.int64)}, "both classes"),
        ({"c": 0.0}, "c must be positive"),
        ({"hidden": 0}, "hidden must be at least"),
        (
            {
                "acts": token_rows(
                    np.random.default_rng(0).standard_normal((2, 10, 4)),
                    [2] * 5 + [0],
                    [0, 1],
                )
            },
            "stimulus 5",
        ),
        (
            {"acts": token_rows(np.full((2, 12, 4), np.nan), [2] * 6, [0, 1])},
            "finite",
        ),
        (
            {"acts": token_rows(np.ones((2, 12, 4)), [2] * 6, [0, 1])},
            "variance",
        ),
    ],
    ids=[
        "one-class",
        "no-penalty",
        "no-unit",
        "no-rows",
        "not-finite",
        "constant",
    ],
)
def test_mlp_fits_reject_inconsistent_inputs(
    fit: Callable[..., Any], change: dict[str, Any], message: str
) -> None:
    acts, labels = planted(6, seed=5)
    arguments: dict[str, Any] = {
        "acts": acts,
        "labels": labels,
        "hidden": 3,
        "seed": 0,
        **change,
    }

    with pytest.raises(ValueError, match=message):
        fit(**arguments)


def test_the_cell_max_mlp_fit_needs_two_units_and_attention_one() -> None:
    acts, labels = planted(6, seed=5)

    # A stalled cell max fit restarts from a linear head held by two units.
    with pytest.raises(ValueError, match=r"^hidden must be at least 2, got 1$"):
        fit_cell_max_mlp(acts, labels=labels, hidden=1, seed=0)
    with pytest.raises(ValueError, match=r"^hidden must be at least 1, got 0$"):
        fit_attention_mlp(acts, labels=labels, hidden=0, seed=0)
    reader = quiet(fit_attention_mlp, acts, labels=labels, hidden=1, seed=0)
    assert reader.hidden.shape == (2, 1, 4)


@MLP_FITS
def test_mlp_fits_warn_at_their_caller_when_they_stop_early(
    fit: Callable[..., Any],
) -> None:
    acts, labels = planted(30, seed=6)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit(acts, labels=labels, hidden=3, seed=0, max_iter=1)

    # One step does not beat the all-zero reader either, so the fit also
    # warns that it stalled (an attention fit) or restarted (a cell max fit).
    stopped = [r for r in caught if r.category is ConvergenceWarning]
    assert len(stopped) == 1
    assert "did not converge" in str(stopped[0].message)
    assert [record.filename for record in caught] == [__file__] * len(caught)


@MLP_FITS
def test_mlp_fits_warn_at_their_caller_when_they_never_beat_zero(
    fit: Callable[..., Any],
) -> None:
    acts, labels = same_cells_everywhere()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reader = fit(acts, labels=labels, hidden=3, seed=0, tol=1e-10)

    # The fit shrinks its seeded start back to the all-zero reader, which
    # scores every stimulus the same, so it stalls; a cell max fit runs
    # again from the mean pooling solution and stalls there too.
    assert [record.category for record in caught] == [StalledFitWarning]
    assert [record.filename for record in caught] == [__file__]
    assert "its loss at zero" in str(caught[0].message)
    scores = reader.score(acts)[0]
    assert np.ptp(scores) < 1e-6


def test_a_stalled_cell_max_mlp_fit_restarts_from_the_mean_pooled_linear_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acts, labels = planted(20, seed=9)
    starts: list[NDArray[np.float64]] = []

    def forced(fun: Any, x0: NDArray[np.float64], **kwargs: Any) -> Any:
        # The first run is made to stall; the mean pooling fit runs for
        # real; the refit returns its start, so the reader is that start.
        starts.append(np.array(x0))
        if len(starts) == 3:
            return OptimizeResult(x=x0, fun=-1.0, status=0, message="")
        result = minimize(fun, x0, **kwargs)
        if len(starts) == 1:
            result.fun = 10.0
        return result

    monkeypatch.setattr(probes, "minimize", forced)

    with pytest.warns(RestartedFitWarning, match="stalled") as caught:
        reader = fit_cell_max_mlp(
            acts, labels=labels, hidden=3, seed=2, c=0.5, tol=1e-10
        )

    assert [record.filename for record in caught] == [__file__]
    assert len(starts) == 3
    # The mean pooling fit is scikit-learn's logistic fit of the stimulus
    # means of the scaled cells of both layers side by side, over the number
    # of layers; the restart puts its head of each layer v on two units,
    # relu(v . z) - relu(-v . z) = v . z, so every cell takes the value its
    # layer adds to the mean pooled score.
    rows, _ = scaled_rows(acts)
    means = np.stack(
        [
            rows[:, acts.row_stimulus == i].mean(axis=1)
            for i in range(acts.n_stimuli)
        ],
        axis=1,
    )
    model = LogisticRegression(C=0.5, tol=1e-10, max_iter=10000)
    model.fit(np.concatenate(list(means), axis=1) / 2, labels)
    heads = model.coef_[0].reshape(2, 4) / 2
    expected = np.einsum("krd,kd->kr", rows, heads) + model.intercept_[0] / 2
    np.testing.assert_allclose(
        reader.cell_values(acts), expected, rtol=1e-4, atol=1e-6
    )


def test_a_cell_max_mlp_fit_that_learns_runs_lbfgs_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acts, labels = planted(40, seed=10)
    runs = lbfgs_spy(monkeypatch)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_cell_max_mlp(acts, labels=labels, hidden=3, seed=0, c=10.0)

    assert caught == []
    assert len(runs) == 1


def test_the_linear_cell_fits_keep_their_scores() -> None:
    # Scores computed by the cell fits before the MLP head existed (main
    # 63828ba0): the linear path must not move.
    acts, labels = planted(12, seed=2026)
    test, _ = planted(8, seed=11)
    expected = {
        "attention": [
            -1.055376648902893,
            2.1013376712799072,
            -0.9808539748191833,
            1.475362777709961,
            -0.788522481918335,
            -0.23856164515018463,
            -0.8148956298828125,
            1.5965946912765503,
        ],
        "cellmax": [
            -1.9259394407272339,
            2.1218855381011963,
            -1.4739514589309692,
            1.677702784538269,
            -0.9634101986885071,
            0.14271126687526703,
            -1.498447299003601,
            1.5497024059295654,
        ],
    }

    attention = quiet(
        fit_attention, acts, labels=labels, c=1.0, joint=True, tol=1e-10
    )
    cellmax = quiet(fit_cell_max, acts, labels=labels, c=1.0, tol=1e-10)

    np.testing.assert_allclose(
        attention.score(test)[0], expected["attention"], rtol=1e-6
    )
    np.testing.assert_allclose(
        cellmax.score(test)[0], expected["cellmax"], rtol=1e-6
    )


RESULTS = Path(__file__).resolve().parents[1] / "results"
DEFAULT_HASH = "0df973a1c1e0"
SHORT_DECOY_LINE = "Hint: {noun} is {{question}}.\n"


def test_the_linear_head_is_the_default_and_stays_out_of_the_config() -> None:
    config = m3_matrix.Config()
    data = config.to_dict()
    mlp = dataclasses.replace(config, cell_head="mlp")
    mlp_data = mlp.to_dict()

    assert m3_matrix.CELL_HEADS == ("linear", "mlp")
    assert config.cell_head == "linear"
    assert config.cell_hidden == 8
    # The linear head leaves both fields out of config.json and the hash,
    # so every linear run keeps the hash it had before they existed.
    assert "cell_head" not in data
    assert "cell_hidden" not in data
    assert config.config_hash() == DEFAULT_HASH
    assert m3_matrix.Config.from_dict(data) == config
    assert mlp_data["cell_head"] == "mlp"
    assert mlp_data["cell_hidden"] == 8
    assert mlp.config_hash() != DEFAULT_HASH
    assert m3_matrix.Config.from_dict(mlp_data) == mlp
    wider = dataclasses.replace(mlp, cell_hidden=16)
    assert wider.config_hash() != mlp.config_hash()


def test_every_committed_m3_run_keeps_its_hash_and_its_config() -> None:
    paths = sorted(RESULTS.glob("m3/*/*/config.json"))

    assert len(paths) >= 20
    for path in paths:
        stored = json.loads(path.read_text(encoding="utf-8"))
        config = m3_matrix.Config.from_dict(stored)
        assert config.cell_head == "linear"
        assert config.config_hash() == path.parent.name
        assert config.to_dict() == stored


def _refused(config: m3_matrix.Config, tmp_path: Path) -> None:
    m3_matrix.execute(
        config, RunDir.create(config, out_root=tmp_path), cast(Any, None), {}
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"cell_head": "deep"},
            r"^unknown cell head 'deep', expected one of \('linear', 'mlp'\)$",
        ),
        (
            {"cell_head": "mlp", "cell_hidden": 1},
            r"^cell_hidden must be at least 2 for the MLP head of "
            r"\('cellattn', 'cellmax'\), got 1$",
        ),
        (
            {"cell_head": "mlp", "cell_hidden": 0, "poolings": ("cellattn",)},
            r"^cell_hidden must be at least 1 for the MLP head of "
            r"\('cellattn',\), got 0$",
        ),
    ],
)
def test_a_bad_head_is_refused_before_any_model_call(
    overrides: dict[str, Any], message: str, tmp_path: Path
) -> None:
    config = dataclasses.replace(m3_matrix.Config(model="tiny"), **overrides)

    with pytest.raises(ValueError, match=message):
        _refused(config, tmp_path)


def test_cell_weights_follow_the_mlp_heads() -> None:
    rng = np.random.default_rng(15)
    acts = token_rows(rng.standard_normal((2, 15, 6)), [3, 1, 4, 2, 5], [2, 5])
    attention, cellmax = hand_mlp(rng)

    np.testing.assert_allclose(
        m3_matrix.cell_weights(attention, acts),
        attention_weights(acts, attention),
    )
    np.testing.assert_array_equal(
        m3_matrix.cell_weights(cellmax, acts), max_weights(acts, cellmax)
    )
    for reader in (attention, cellmax):
        weights = m3_matrix.cell_weights(reader, acts)
        weighted = weights * reader.cell_values(acts)
        totals = [weighted[:, acts.row_stimulus == i].sum() for i in range(5)]
        np.testing.assert_allclose(totals, reader.score(acts)[0], rtol=1e-6)


def _utility_rows(pairs: int = 8) -> list[Example]:
    words = {
        1: ("sun", "gift", "joy", "fun"),
        0: ("rain", "loss", "pain", "fear"),
    }
    return [
        Example(
            f"ethics-{split}-{i}:{suffix}",
            f"a {words[label][i % 4]} day {i}",
            label,
            split,
            {"pair": f"ethics-{split}-{i}"},
        )
        for split in ("train", "val", "test")
        for i in range(pairs)
        for suffix, label in (("a", 1), ("b", 0))
    ]


def _tiny_config(**overrides: Any) -> m3_matrix.Config:
    base = m3_matrix.Config(
        model="tiny",
        tasks=("utility",),
        n_seeds=2,
        n_pairs=4,
        n_eval_pairs=3,
        top_k=2,
        layers=(1, 2, 3),
        cell_layers=(1, 3),
        upper_layers=(3,),
        cell_sets=("prompt",),
        poolings=("cellattn", "cellmax"),
        logistic_c=(0.1, 10.0),
        n_boot=10,
        cell_head="mlp",
        cell_hidden=2,
    )
    return dataclasses.replace(base, **overrides)


@dataclasses.dataclass(frozen=True)
class _MLPRun:
    """A finished MLP-head run and the fits it made.

    Attributes:
        config: The config of the run.
        run_dir: The finished run.
        fits: Per fit, the pooling and the fit's keyword arguments.
        splits: The items the run rendered for its task.
    """

    config: m3_matrix.Config
    run_dir: RunDir
    fits: list[tuple[str, dict[str, Any]]]
    splits: Any


def _run(
    config: m3_matrix.Config,
    loaded: LoadedModel,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _MLPRun:
    fits: list[tuple[str, dict[str, Any]]] = []
    splits: list[Any] = []
    prepare = m2_readers._prepare_task

    def capturing(*args: Any, **kwargs: Any) -> Any:
        prepared = prepare(*args, **kwargs)
        splits.append(prepared[0])
        return prepared

    def recording(pooling: str, fit: Callable[..., Any]) -> Callable[..., Any]:
        def fitting(acts: Activations, **kwargs: Any) -> Any:
            fits.append((pooling, kwargs))
            return fit(acts, **kwargs)

        return fitting

    def linear(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("an MLP-head run fitted a linear head")

    monkeypatch.setattr(p2_positions, "DECOY_LINE", SHORT_DECOY_LINE)
    monkeypatch.setattr(m2_readers, "_prepare_task", capturing)
    monkeypatch.setattr(
        m3_matrix,
        "fit_attention_mlp",
        recording("cellattn", fit_attention_mlp),
    )
    monkeypatch.setattr(
        m3_matrix, "fit_cell_max_mlp", recording("cellmax", fit_cell_max_mlp)
    )
    monkeypatch.setattr(m3_matrix, "fit_attention", linear)
    monkeypatch.setattr(m3_matrix, "fit_cell_max", linear)
    run_dir = RunDir.create(config, out_root=root)
    with warnings.catch_warnings():
        for category in m3_matrix._COUNTED:
            warnings.simplefilter("ignore", category)
        m3_matrix.execute(
            config, run_dir, loaded, {"ethics_utility": _utility_rows()}
        )
    return _MLPRun(config, run_dir, fits, splits[0])


@pytest.fixture(scope="module")
def mlp_runs() -> dict[str, _MLPRun]:
    return {}


@pytest.fixture
def mlp_run(
    mlp_runs: dict[str, _MLPRun],
    tiny_gpt2: LoadedModel,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> _MLPRun:
    if "run" not in mlp_runs:
        mlp_runs["run"] = _run(
            _tiny_config(),
            tiny_gpt2,
            tmp_path_factory.mktemp("m3mlp"),
            monkeypatch,
        )
    return mlp_runs["run"]


def _rows(run_dir: RunDir, name: str) -> list[dict[str, Any]]:
    payload = cast(dict[str, Any], run_dir.read_json(name))
    return [
        dict(zip(payload["columns"], row, strict=True))
        for row in payload["rows"]
    ]


MLP_CELLS = ("cellattnmlp_prompt", "cellmaxmlp_prompt")
SERIES = ("reader", "shuffled")


def test_mlp_readers_are_named_by_their_head_in_every_file(
    mlp_run: _MLPRun,
) -> None:
    run_dir = mlp_run.run_dir
    joint = {
        (r["scorer"], r["readout"], r["series"])
        for r in _rows(run_dir, "table")
        if r["layer_rule"] == "joint"
    }
    summary = cast(dict[str, Any], run_dir.read_json("summary"))
    compared = {
        r["readout"]
        for r in _rows(run_dir, "comparisons")
        if r["layer_rule"] == "joint"
    }
    diagnosed = [
        r for r in _rows(run_dir, "diagnostics") if r["readout"] in MLP_CELLS
    ]

    # The random control keeps its linear heads, and its name.
    assert joint == {
        *(("logistic", cell, s) for cell in MLP_CELLS for s in SERIES),
        ("random", "cellmax_prompt", "reader"),
    }
    assert set(summary["fits"]["utility"]) == {
        f"logistic/{cell}/joint/{s}" for cell in MLP_CELLS for s in SERIES
    }
    assert set(MLP_CELLS) <= compared
    assert {(r["readout"], r["series"]) for r in diagnosed} == {
        (cell, s) for cell in MLP_CELLS for s in SERIES
    }
    for row in diagnosed:
        shares = [row[segment] for segment in m3_matrix.SEGMENTS]
        assert sum(shares) == pytest.approx(1.0)
        assert row["prefix"] == 0.0


def test_mlp_fits_share_one_start_per_draw_and_reader(mlp_run: _MLPRun) -> None:
    config = mlp_run.config
    n_grid = len(config.logistic_c)
    seeds: dict[tuple[str, int], set[int]] = {}
    for j, (pooling, kwargs) in enumerate(mlp_run.fits):
        assert kwargs["hidden"] == config.cell_hidden
        assert kwargs["tol"] == config.cell_tol
        assert kwargs["c"] == config.logistic_c[j % n_grid]
        # Per draw: each pooling fits its reader series, then its shuffled.
        draw = j // (len(config.poolings) * len(SERIES) * n_grid)
        seeds.setdefault((pooling, draw), set()).add(kwargs["seed"])

    assert len(mlp_run.fits) == config.n_seeds * 2 * 2 * n_grid
    # One start per draw and reader: the shuffled control and every grid
    # value start where the reader does.
    assert seeds == {
        (pooling, i): {
            derive_seed(
                config.seed, "cell_head", "utility", f"{pooling}mlp_prompt", i
            )
        }
        for pooling in config.poolings
        for i in range(config.n_seeds)
    }


def _mlp_reader(heads: Any, key: str, i: int, layers: tuple[int, ...]) -> Any:
    fields = {
        name: heads[f"{key}_{name}"][i]
        for name in ("hidden", "hidden_biases", "outputs", "biases")
    }
    if f"{key}_queries" in heads:
        return AttentionMLPReader(
            layers=layers,
            queries=heads[f"{key}_queries"][i],
            offsets=heads[f"{key}_offsets"][i],
            **fields,
        )
    return CellMaxMLPReader(layers=layers, **fields)


def test_the_kept_mlp_heads_reproduce_the_select_scores(
    mlp_run: _MLPRun, tiny_gpt2: LoadedModel
) -> None:
    config, run_dir = mlp_run.config, mlp_run.run_dir
    heads = run_dir.read_arrays("cellheads_utility")
    sidecar = cast(dict[str, Any], run_dir.read_json("cellheads_utility"))
    scores = run_dir.read_arrays("scores_utility")
    entries = cast(dict[str, Any], run_dir.read_json("scores_utility"))
    cells, segments = m3_matrix.prompt_cells(
        tiny_gpt2,
        mlp_run.splits.select_items,
        config.cell_layers,
        bos=None,
        batch_size=config.batch_size,
        decoy=False,
    )
    acts = m3_matrix.select_cells(cells, segments, "prompt", config.cell_layers)
    mlp = [
        key
        for key, entry in sidecar["entries"].items()
        if "mlp_" in entry["entry"]
    ]

    assert sidecar["axes"]["hidden"] == ["seed", "layer", "unit", "feature"]
    assert sidecar["axes"]["outputs"] == ["seed", "layer", "unit"]
    assert len(mlp) == len(MLP_CELLS) * len(SERIES)
    for key in mlp:
        e = entries["entries"].index(sidecar["entries"][key]["entry"])
        assert heads[f"{key}_hidden"].shape == (
            config.n_seeds,
            len(config.cell_layers),
            config.cell_hidden,
            acts.data.shape[2],
        )
        assert f"{key}_signs" not in heads
        for i in range(config.n_seeds):
            reader = _mlp_reader(heads, key, i, acts.layers)
            np.testing.assert_array_equal(
                reader.score(acts)[0], scores["select"][e, i]
            )


def test_a_rerun_of_an_mlp_head_run_writes_the_same_files(
    mlp_run: _MLPRun,
    tiny_gpt2: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    again = _run(mlp_run.config, tiny_gpt2, tmp_path, monkeypatch)

    for name in ("table", "comparisons", "diagnostics"):
        assert _rows(again.run_dir, name) == _rows(mlp_run.run_dir, name)
    first = mlp_run.run_dir.read_arrays("cellheads_utility")
    second = again.run_dir.read_arrays("cellheads_utility")
    assert set(first) == set(second)
    for key in first:
        np.testing.assert_array_equal(second[key], first[key])
