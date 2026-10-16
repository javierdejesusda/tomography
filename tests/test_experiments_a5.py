"""Tests for the A5 decomposition of reading vectors into SAE features."""

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.analysis import decoder_cosines
from tomography.analysis import dictionary_directions
from tomography.analysis import lens_logits
from tomography.analysis import sparse_fit
from tomography.analysis import top_tokens
from tomography.cli import load_experiment
from tomography.experiments import a5_sae_features
from tomography.experiments.a2_logit_lens import lens_params
from tomography.experiments.a5_sae_features import Config
from tomography.experiments.a5_sae_features import execute
from tomography.experiments.a5_sae_features import plot
from tomography.experiments.a5_sae_features import run
from tomography.experiments.aggregate import ReaderSet
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.model import LoadedModel
from tomography.sae import SAES
from tomography.sae import Decoders
from tomography.sae import SaeSpec

LAYERS = (1, 2, 3)
SPEC = SaeSpec(
    key="tiny-sae",
    repo_id="owner/tiny-sae",
    revision="0123456789abcdef0123456789abcdef01234567",
    model="tiny",
    n_layers=3,
    d_model=32,
    width=20,
    k=4,
    license="test",
)
N_RANDOM = 40


@pytest.fixture(autouse=True)
def tiny_sae(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(a5_sae_features, "SAES", {**SAES, SPEC.key: SPEC})


def _columns(n: int, scale: float = 1.0) -> NDArray[np.float32]:
    """The decoder of SAE layer n: orthonormal columns, optionally scaled."""
    raw = np.random.default_rng(100 + n).standard_normal((32, 20))
    q, _ = np.linalg.qr(raw)
    return (q * scale).astype(np.float32)


def _decoders(calls: list[int] | None = None, scale: float = 1.0) -> Decoders:
    def decoders(n: int) -> NDArray[np.float32]:
        if calls is not None:
            calls.append(n)
        return _columns(n, scale)

    return decoders


def _unit(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def _planted(n: int, layers: tuple[int, ...] = LAYERS) -> NDArray[np.float64]:
    """Identical seeds, each layer 0.8 of feature 3 minus 0.6 of feature 7."""
    rows = []
    for layer in layers:
        decoder = _columns(layer - 1).astype(np.float64)
        rows.append(0.8 * decoder[:, 3] - 0.6 * decoder[:, 7])
    return np.broadcast_to(np.stack(rows), (n, len(layers), 32)).copy()


def _random(n: int, seed: int) -> NDArray[np.float64]:
    shape = (n, len(LAYERS), 32)
    return _unit(np.random.default_rng(seed).normal(size=shape))


def _readers(loaded: LoadedModel) -> dict[str, ReaderSet]:
    vectors = {
        "r1": _planted(4),
        "r7": _random(6, 1),
        "r4:anger": _random(5, 2),
    }
    return {
        label: ReaderSet(LAYERS, v, loaded.n_layers)
        for label, v in vectors.items()
    }


def _config(**overrides: Any) -> Config:
    base = Config(
        model="tiny",
        sae=SPEC.key,
        n_boot=20,
        n_random=N_RANDOM,
        top_k=2,
        n_atoms=2,
    )
    return dataclasses.replace(base, **overrides)


def _run_dir(tmp_path: Path, config: Config | None = None) -> RunDir:
    return RunDir.create(config or _config(), out_root=tmp_path)


def _json(run_dir: RunDir, name: str) -> dict[str, Any]:
    payload = run_dir.read_json(name)
    assert isinstance(payload, dict)
    return payload


def _executed(
    tiny_qwen3: LoadedModel,
    tmp_path: Path,
    config: Config | None = None,
    readers: dict[str, ReaderSet] | None = None,
    decoders: Decoders | None = None,
    spec: SaeSpec = SPEC,
) -> RunDir:
    config = config or _config()
    run_dir = _run_dir(tmp_path, config)
    execute(
        config,
        run_dir,
        tiny_qwen3,
        readers or _readers(tiny_qwen3),
        spec,
        decoders or _decoders(),
    )
    return run_dir


def _null(layer: int, seed: int = 0) -> NDArray[np.float64]:
    """Replays the matched null of the contract at a reading layer."""
    rng = np.random.default_rng(derive_seed(seed, "dictionary", layer))
    return dictionary_directions(_columns(layer - 1), N_RANDOM, rng)


def _null_explained(layer: int, n_atoms: int = 2) -> NDArray[np.float64]:
    """Replays the explained fraction of the null at the last atom count."""
    fits = sparse_fit(_null(layer), _columns(layer - 1), n_atoms)[0]
    return fits[:, -1]


def _rows(run_dir: RunDir, concept: str, layer: int) -> list[list[Any]]:
    rows = _json(run_dir, "table")["rows"]
    return [r for r in rows if r[:2] == [concept, layer]]


def test_config_defaults_and_round_trip() -> None:
    config = Config()
    assert config.experiment == "a5"
    assert config.model == "qwen3-1.7b-base"
    assert config.sources == ("r1", "r2", "r3", "r5", "r7")
    assert config.extra_runs == ()
    assert config.sae == "qwen-scope-1.7b-base-l0-50"
    assert config.sae in SAES
    assert config.layers is None
    assert config.top_k == 10
    assert config.n_atoms == 50
    assert config.n_random == 64
    assert config.n_boot == 2000
    assert Config.from_dict(config.to_dict()) == config
    chosen = dataclasses.replace(config, layers=(4, 8))
    assert Config.from_dict(chosen.to_dict()) == chosen


def test_the_cli_discovers_the_experiment() -> None:
    experiment = load_experiment("a5")
    assert experiment.config_cls is Config
    assert experiment.run is run
    assert experiment.plot is plot


def test_execute_finds_the_planted_features_and_their_omp_ranks(
    tiny_qwen3: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _executed(tiny_qwen3, tmp_path)
    table = _json(run_dir, "table")
    assert table["columns"] == [
        "concept",
        "layer",
        "rank",
        "feature",
        "cos",
        "null_q95",
        "omp_rank",
        "tokens",
    ]
    assert table["n_layers"] == tiny_qwen3.n_layers
    assert len(table["rows"]) == 3 * 3 * 2
    assert [r[0] for r in table["rows"]][::6] == ["r1", "r7", "r4:anger"]
    for layer in LAYERS:
        first, second = _rows(run_dir, "r1", layer)
        assert first[2:5] == [1, 3, pytest.approx(0.8, abs=1e-5)]
        assert second[2:5] == [2, 7, pytest.approx(-0.6, abs=1e-5)]
        assert [first[6], second[6]] == [1, 2]
        assert all(len(r[7]) == 5 for r in (first, second))
        assert all(isinstance(t, str) for r in (first, second) for t in r[7])


def test_the_null_quantile_is_the_matched_null_at_each_rank(
    tiny_qwen3: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _executed(tiny_qwen3, tmp_path)
    for layer in LAYERS:
        cosines = decoder_cosines(_null(layer), _columns(layer - 1))
        ranked = -np.sort(-np.abs(cosines), axis=1)[:, :2]
        expected = np.quantile(ranked, 0.95, axis=0)
        for concept in ("r1", "r7", "r4:anger"):
            got = [r[5] for r in _rows(run_dir, concept, layer)]
            np.testing.assert_allclose(got, expected, atol=1e-6)
    assert expected[0] > expected[1] > 0.0


def test_omp_rank_is_zero_for_a_feature_outside_the_fit(
    tiny_qwen3: LoadedModel, tmp_path: Path
) -> None:
    config = _config(top_k=3, n_atoms=1)
    run_dir = _executed(tiny_qwen3, tmp_path, config)
    for layer in LAYERS:
        ranks = [r[6] for r in _rows(run_dir, "r1", layer)]
        assert ranks == [1, 0, 0]


def test_the_explained_curve_is_the_excess_over_the_replayed_null(
    tiny_qwen3: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _executed(tiny_qwen3, tmp_path)
    sidecar = _json(run_dir, "curves_explained")
    assert sidecar == {
        "series": ["r1", "r7", "r4:anger"],
        "xlabel": "Layer",
        "ylabel": "Explained fraction above the matched null",
        "chance": 0.0,
        "n_layers": tiny_qwen3.n_layers,
    }
    arrays = run_dir.read_arrays("curves_explained")
    assert arrays["x"].tolist() == [1.0, 2.0, 3.0]
    assert arrays["per_seed"].shape == (3, 4, 3)
    assert (arrays["lo"] <= arrays["mean"]).all()
    assert (arrays["mean"] <= arrays["hi"]).all()
    for k, layer in enumerate(LAYERS):
        null_mean = _null_explained(layer).mean()
        np.testing.assert_allclose(
            arrays["per_seed"][0, :, k], 1.0 - null_mean, atol=1e-6
        )
        assert arrays["per_seed"][0, :, k].min() > 0.4


def test_stability_is_one_for_identical_seeds_and_lower_for_the_null(
    tiny_qwen3: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _executed(tiny_qwen3, tmp_path)
    sidecar = _json(run_dir, "matrix_stability")
    assert sidecar["rows"] == ["r1", "r7", "r4:anger", "null"]
    assert sidecar["cols"] == ["1", "2", "3"]
    values = run_dir.read_arrays("matrix_stability")["values"]
    assert values.shape == (4, 3)
    np.testing.assert_allclose(values[0], 1.0)
    assert values[3].max() < 0.5
    assert ((values >= 0.0) & (values <= 1.0)).all()


def test_the_explained_file_holds_the_mean_fits_and_the_null_means(
    tiny_qwen3: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _executed(tiny_qwen3, tmp_path, _config(n_atoms=3))
    arrays = run_dir.read_arrays("explained")
    assert set(arrays) == {"reading", "null", "layers"}
    assert arrays["layers"].tolist() == [1, 2, 3]
    assert arrays["reading"].shape == (3, 3, 3)
    assert arrays["null"].shape == (3, 3)
    np.testing.assert_allclose(arrays["reading"][0, :, 1:], 1.0, atol=1e-6)
    for k, layer in enumerate(LAYERS):
        expected = sparse_fit(_null(layer), _columns(layer - 1), 3)[0]
        np.testing.assert_allclose(
            arrays["null"][k], expected.mean(axis=0), atol=1e-6
        )
    assert (np.diff(arrays["reading"], axis=2) >= -1e-12).all()


def test_the_summary_reports_the_decomposition_and_its_nulls(
    tiny_qwen3: LoadedModel, tmp_path: Path
) -> None:
    readers = _readers(tiny_qwen3)
    readers["r1"] = dataclasses.replace(readers["r1"], config_hash="abc123")
    run_dir = _executed(tiny_qwen3, tmp_path, readers=readers)
    summary = _json(run_dir, "summary")
    assert set(summary) == {
        "concepts",
        "layers",
        "skipped_layers",
        "n_seeds",
        "top_k",
        "n_atoms",
        "sae",
        "transfer",
        "decoder_norm_deviation",
        "null",
        "best_layer",
        "source_runs",
    }
    assert summary["concepts"] == ["r1", "r7", "r4:anger"]
    assert summary["layers"] == [1, 2, 3]
    assert summary["skipped_layers"] == []
    assert (summary["n_seeds"], summary["top_k"], summary["n_atoms"]) == (
        4,
        2,
        2,
    )
    assert summary["sae"] == {
        "key": SPEC.key,
        "repo_id": SPEC.repo_id,
        "revision": SPEC.revision,
        "k": SPEC.k,
        "width": SPEC.width,
        "model": SPEC.model,
    }
    assert summary["transfer"] is False
    assert summary["decoder_norm_deviation"] < 1e-6
    assert summary["source_runs"] == {
        "r1": "abc123",
        "r7": None,
        "r4:anger": None,
    }
    mean = run_dir.read_arrays("curves_explained")["mean"]
    for i, label in enumerate(summary["concepts"]):
        assert summary["best_layer"][label] == LAYERS[int(mean[i].argmax())]


def test_the_summary_nulls_replay_the_matched_and_the_isotropic_draws(
    tiny_qwen3: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _executed(tiny_qwen3, tmp_path)
    null = _json(run_dir, "summary")["null"]
    assert set(null) == {"dictionary", "isotropic"}
    assert set(null["dictionary"]) == {
        "explained_mean",
        "explained_std",
        "top1_q95",
    }
    assert set(null["isotropic"]) == {"top1_q95"}
    for k, layer in enumerate(LAYERS):
        decoder = _columns(layer - 1)
        explained = _null_explained(layer)
        top1 = np.abs(decoder_cosines(_null(layer), decoder)).max(axis=1)
        assert null["dictionary"]["explained_mean"][k] == pytest.approx(
            explained.mean(), abs=1e-6
        )
        assert null["dictionary"]["explained_std"][k] == pytest.approx(
            explained.std(ddof=1), abs=1e-6
        )
        assert null["dictionary"]["top1_q95"][k] == pytest.approx(
            np.quantile(top1, 0.95), abs=1e-6
        )
        draws = np.random.default_rng(
            derive_seed(0, "isotropic", layer)
        ).standard_normal((N_RANDOM, 32))
        top1_iso = np.abs(decoder_cosines(draws, decoder)).max(axis=1)
        assert null["isotropic"]["top1_q95"][k] == pytest.approx(
            np.quantile(top1_iso, 0.95), abs=1e-6
        )


def test_the_null_is_reproducible_for_one_seed_and_differs_for_another(
    tiny_qwen3: LoadedModel, tmp_path: Path
) -> None:
    runs = {
        name: _executed(tiny_qwen3, tmp_path / name, _config(seed=seed))
        for name, seed in (("a", 3), ("b", 3), ("c", 4))
    }
    null = {k: v.read_arrays("explained")["null"] for k, v in runs.items()}
    np.testing.assert_array_equal(null["a"], null["b"])
    assert np.abs(null["a"] - null["c"]).max() > 1e-3
    for name in ("a", "b", "c"):
        np.testing.assert_allclose(
            runs[name].read_arrays("explained")["reading"][0, :, 1], 1.0
        )


def test_a_decoder_with_unnormalized_columns_changes_nothing_but_the_report(
    tiny_qwen3: LoadedModel, tmp_path: Path
) -> None:
    plain = _executed(tiny_qwen3, tmp_path / "plain")
    scaled = _executed(
        tiny_qwen3, tmp_path / "scaled", decoders=_decoders(scale=1.25)
    )
    assert _json(scaled, "summary")["decoder_norm_deviation"] == pytest.approx(
        0.25, abs=1e-5
    )
    for name in ("explained", "matrix_stability"):
        for key, values in plain.read_arrays(name).items():
            np.testing.assert_allclose(
                scaled.read_arrays(name)[key], values, atol=1e-5
            )
    rows = _json(plain, "table")["rows"]
    other = _json(scaled, "table")["rows"]
    assert [r[:4] for r in other] == [r[:4] for r in rows]
    np.testing.assert_allclose(
        [r[4] for r in other], [r[4] for r in rows], atol=1e-5
    )


def test_the_layers_map_to_the_autoencoder_layer_below(
    tiny_qwen3: LoadedModel, tmp_path: Path
) -> None:
    calls: list[int] = []
    _executed(tiny_qwen3, tmp_path, decoders=_decoders(calls))
    assert calls == [0, 1, 2]


def test_layer_zero_has_no_autoencoder_and_is_skipped_and_listed(
    tiny_qwen3: LoadedModel, tmp_path: Path
) -> None:
    layers = (0, *LAYERS)
    shape = (4, len(layers), 32)
    vectors = _unit(np.random.default_rng(0).normal(size=shape))
    vectors[:, 1:] = _planted(4)
    readers = {"r1": ReaderSet(layers, vectors, tiny_qwen3.n_layers)}
    calls: list[int] = []
    run_dir = _executed(
        tiny_qwen3, tmp_path, readers=readers, decoders=_decoders(calls)
    )
    assert calls == [0, 1, 2]
    summary = _json(run_dir, "summary")
    assert summary["layers"] == [1, 2, 3]
    assert summary["skipped_layers"] == [0]
    assert run_dir.read_arrays("curves_explained")["x"].tolist() == [1, 2, 3]
    assert _json(run_dir, "matrix_stability")["cols"] == ["1", "2", "3"]


def test_a_subset_of_layers_loads_only_their_autoencoders(
    tiny_qwen3: LoadedModel, tmp_path: Path
) -> None:
    calls: list[int] = []
    config = _config(layers=(2,))
    run_dir = _executed(tiny_qwen3, tmp_path, config, decoders=_decoders(calls))
    assert calls == [1]
    assert _json(run_dir, "summary")["layers"] == [2]
    assert run_dir.read_arrays("explained")["reading"].shape == (3, 1, 2)
    assert {r[1] for r in _json(run_dir, "table")["rows"]} == {2}


def _unfit(name: str, loaded: LoadedModel) -> tuple[Config, Any, str]:
    readers = _readers(loaded)
    n_layers = loaded.n_layers
    cases: dict[str, tuple[Config, Any, str]] = {
        "top-k": (_config(top_k=0), readers, "top_k"),
        "top-k-width": (_config(top_k=21), readers, "top_k"),
        "n-atoms": (_config(n_atoms=0), readers, "n_atoms"),
        "n-atoms-width": (_config(n_atoms=33), readers, "n_atoms"),
        "n-atoms-features": (_config(n_atoms=21), readers, "n_atoms"),
        "n-random": (_config(n_random=1), readers, "n_random"),
        "n-boot": (_config(n_boot=0), readers, "n_boot"),
        "seed": (_config(seed=-1), readers, "seed"),
        "sae": (_config(sae="nope"), readers, "sae"),
        "no-layers": (_config(layers=()), readers, "layers"),
        "repeated-layers": (_config(layers=(2, 2)), readers, "layers"),
        "layer-zero": (_config(layers=(0,)), readers, "layer 0"),
        "layer-too-deep": (_config(layers=(4,)), readers, "layer 4"),
        "layer-not-read": (
            _config(layers=(2,)),
            {"r1": ReaderSet((1, 3), _planted(4, (1, 3)), n_layers)},
            "layer 2",
        ),
        "width": (
            _config(),
            {"r1": ReaderSet(LAYERS, _random(4, 0)[..., :16], n_layers)},
            "width",
        ),
        "depth": (
            _config(),
            {"r1": ReaderSet(LAYERS, _random(4, 0), n_layers + 1)},
            "depth",
        ),
        "one-seed": (
            _config(),
            {"r1": ReaderSet(LAYERS, _random(1, 0), n_layers)},
            "at least 2 seeds",
        ),
        "layers-differ": (
            _config(),
            {
                "r1": readers["r1"],
                "r7": ReaderSet((1, 2, 4), _random(6, 1), n_layers),
            },
            "layers",
        ),
        "nine": (
            _config(),
            {f"c{i}": readers["r1"] for i in range(9)},
            "at most 8",
        ),
        "no-sets": (_config(), {}, "no reader sets"),
        "only-layer-zero": (
            _config(),
            {"r1": ReaderSet((0,), _random(4, 0)[:, :1], n_layers)},
            "no layer of the source runs",
        ),
    }
    return cases[name]


UNFIT = [
    "top-k",
    "top-k-width",
    "n-atoms",
    "n-atoms-width",
    "n-atoms-features",
    "n-random",
    "n-boot",
    "seed",
    "sae",
    "no-layers",
    "repeated-layers",
    "layer-zero",
    "layer-too-deep",
    "layer-not-read",
    "width",
    "depth",
    "one-seed",
    "layers-differ",
    "nine",
    "no-sets",
    "only-layer-zero",
]


@pytest.mark.parametrize("name", UNFIT)
def test_execute_refuses_unusable_input_before_any_work(
    tiny_qwen3: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    def started(*args: Any) -> None:
        raise AssertionError("the model was read before the checks")

    monkeypatch.setattr(a5_sae_features, "lens_params", started)
    calls: list[int] = []
    config, readers, match = _unfit(name, tiny_qwen3)
    run_dir = _run_dir(tmp_path, config)
    with pytest.raises(ValueError, match=match):
        execute(config, run_dir, tiny_qwen3, readers, SPEC, _decoders(calls))
    assert calls == []
    assert not list(run_dir.path.glob("*.npz"))
    assert not (run_dir.path / "summary.json").exists()
    assert not (run_dir.path / "table.json").exists()


def test_execute_refuses_a_spec_that_is_not_the_configured_release(
    tiny_qwen3: LoadedModel, tmp_path: Path
) -> None:
    other = dataclasses.replace(SPEC, key="another-release")
    config = _config()
    run_dir = _run_dir(tmp_path, config)
    calls: list[int] = []
    with pytest.raises(ValueError, match="another-release"):
        execute(
            config,
            run_dir,
            tiny_qwen3,
            _readers(tiny_qwen3),
            other,
            _decoders(calls),
        )
    assert calls == []
    assert not (run_dir.path / "summary.json").exists()


def test_execute_refuses_more_atoms_than_the_stream_has_dimensions(
    tiny_qwen3: LoadedModel, tmp_path: Path
) -> None:
    wide = dataclasses.replace(SPEC, width=40)
    config = _config(n_atoms=33)
    run_dir = _run_dir(tmp_path, config)
    calls: list[int] = []
    with pytest.raises(ValueError, match="n_atoms"):
        execute(
            config,
            run_dir,
            tiny_qwen3,
            _readers(tiny_qwen3),
            wide,
            _decoders(calls),
        )
    assert calls == []


def test_transfer_is_flagged_when_the_model_is_not_the_one_the_sae_saw(
    tiny_qwen3: LoadedModel, tmp_path: Path
) -> None:
    other = dataclasses.replace(SPEC, model="trained-on-something-else")
    run_dir = _executed(tiny_qwen3, tmp_path, spec=other)
    summary = _json(run_dir, "summary")
    assert summary["transfer"] is True
    assert summary["sae"]["model"] == "trained-on-something-else"


def test_the_first_seeds_of_the_longer_sets_are_used(
    tiny_qwen3: LoadedModel, tmp_path: Path
) -> None:
    readers = _readers(tiny_qwen3)
    run_dir = _executed(tiny_qwen3, tmp_path, readers=readers)
    assert _json(run_dir, "summary")["n_seeds"] == 4
    per_seed = run_dir.read_arrays("curves_explained")["per_seed"]
    assert per_seed.shape == (3, 4, 3)
    longer = readers["r7"].vectors
    assert longer.shape[0] == 6
    for k, layer in enumerate(LAYERS):
        null_mean = _null_explained(layer).mean()
        fit = sparse_fit(longer[:4, k], _columns(layer - 1), 2)[0][:, 1]
        np.testing.assert_allclose(
            per_seed[1, :, k], fit - null_mean, atol=1e-6
        )


def _planted_on_a_token(
    loaded: LoadedModel, token: int, sign: float
) -> tuple[ReaderSet, Decoders, NDArray[np.float64]]:
    """A reader along feature 3, whose column is the lens row of `token`."""
    unembedding, scale, _ = lens_params(loaded)
    assert scale is not None
    direction = unembedding[token].astype(np.float64) / scale
    direction /= np.linalg.norm(direction)
    decoder = _columns(0)
    decoder[:, 3] = direction.astype(np.float32)
    vectors = np.broadcast_to(sign * direction, (3, 1, 32)).copy()
    reader = ReaderSet((1,), vectors, loaded.n_layers)
    return reader, lambda n: decoder.copy(), direction


def _lens_tokens(
    loaded: LoadedModel, direction: NDArray[np.float64]
) -> list[str]:
    unembedding, scale, center = lens_params(loaded)
    logits = lens_logits(direction, unembedding, scale=scale, center=center)
    top, _ = top_tokens(logits, 5)
    return [loaded.tokenizer.decode([int(i)]) for i in top]


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["positive", "negative"])
def test_the_tokens_of_a_feature_are_the_lens_of_its_signed_column(
    tiny_qwen3: LoadedModel, tmp_path: Path, sign: float
) -> None:
    token = int(
        tiny_qwen3.tokenizer.encode(" good", add_special_tokens=False)[0]
    )
    reader, decoders, column = _planted_on_a_token(tiny_qwen3, token, sign)
    config = _config(top_k=1, n_atoms=1, layers=(1,))
    run_dir = _executed(
        tiny_qwen3,
        tmp_path,
        config,
        readers={"r1": reader},
        decoders=decoders,
    )
    (row,) = _rows(run_dir, "r1", 1)
    assert row[3] == 3
    assert row[4] == pytest.approx(sign, abs=1e-5)
    assert row[7] == _lens_tokens(tiny_qwen3, sign * column)
    promoted = tiny_qwen3.tokenizer.decode([token])
    assert (promoted in row[7]) is (sign > 0)


def _stub_run(
    monkeypatch: pytest.MonkeyPatch,
    loaded: LoadedModel,
    readers: dict[str, ReaderSet],
    calls: list[tuple[str, Any]],
) -> None:
    """Replaces what `run` reaches out to and records each call in order."""

    def fake_load_model(key: str) -> LoadedModel:
        calls.append(("model", key))
        return loaded

    def fake_gather(*args: Any) -> dict[str, ReaderSet]:
        calls.append(("gather", args))
        return readers

    def fake_load_decoder(spec: SaeSpec, layer: int) -> NDArray[np.float32]:
        calls.append(("decoder", (spec, layer)))
        return _columns(layer)

    def fake_execute(*args: Any) -> None:
        calls.append(("execute", args))

    monkeypatch.setattr(a5_sae_features, "load_model", fake_load_model)
    monkeypatch.setattr(a5_sae_features, "gather_readers", fake_gather)
    monkeypatch.setattr(a5_sae_features, "load_decoder", fake_load_decoder)
    monkeypatch.setattr(a5_sae_features, "execute", fake_execute)


def test_run_gathers_the_readers_checks_them_then_loads_the_model(
    tiny_qwen3: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readers = _readers(tiny_qwen3)
    calls: list[tuple[str, Any]] = []
    _stub_run(monkeypatch, tiny_qwen3, readers, calls)
    config = _config(sources=("r1", "r7"), extra_runs=("somewhere",))
    run_dir = _run_dir(tmp_path, config)
    run(config, run_dir)
    assert [name for name, _ in calls] == ["gather", "model", "execute"]
    gathered, model, executed = (args for _, args in calls)
    assert gathered == (tmp_path, ("r1", "r7"), "tiny", ("somewhere",))
    assert model == "tiny"
    assert executed[:5] == (config, run_dir, tiny_qwen3, readers, SPEC)
    executed[5](2)
    assert calls[-1] == ("decoder", (SPEC, 2))


@pytest.mark.parametrize(
    "name",
    [
        "top-k",
        "n-atoms",
        "n-random",
        "n-boot",
        "seed",
        "sae",
        "no-layers",
        "repeated-layers",
    ],
)
def test_run_refuses_a_bad_config_before_gathering_or_loading_anything(
    tiny_qwen3: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    calls: list[tuple[str, Any]] = []
    _stub_run(monkeypatch, tiny_qwen3, _readers(tiny_qwen3), calls)
    config, _, match = _unfit(name, tiny_qwen3)
    with pytest.raises(ValueError, match=match):
        run(config, _run_dir(tmp_path, config))
    assert calls == []


@pytest.mark.parametrize(
    "name",
    [
        "width",
        "depth",
        "one-seed",
        "layers-differ",
        "nine",
        "layer-zero",
        "layer-too-deep",
        "layer-not-read",
        "top-k-width",
        "n-atoms-width",
        "n-atoms-features",
        "only-layer-zero",
    ],
)
def test_run_refuses_unfit_readers_before_loading_the_model(
    tiny_qwen3: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    config, readers, match = _unfit(name, tiny_qwen3)
    calls: list[tuple[str, Any]] = []
    _stub_run(monkeypatch, tiny_qwen3, readers, calls)
    with pytest.raises(ValueError, match=match):
        run(config, _run_dir(tmp_path, config))
    assert [step for step, _ in calls] == ["gather"]


def test_plot_draws_the_curves_and_the_stability_matrix(
    tiny_qwen3: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _executed(tiny_qwen3, tmp_path)
    paths = plot(run_dir, tmp_path / "figs")
    assert sorted(p.name for p in paths) == [
        "curves_explained.pdf",
        "curves_explained.png",
        "curves_explained.svg",
        "matrix_stability.pdf",
        "matrix_stability.png",
        "matrix_stability.svg",
    ]
    assert all(p.stat().st_size > 0 for p in paths)
