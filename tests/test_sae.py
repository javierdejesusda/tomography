"""Tests for the pinned Qwen-Scope sparse autoencoders."""

from collections.abc import Callable
import dataclasses
import json
from pathlib import Path
import pickle
from typing import Any

from huggingface_hub import hf_hub_download
import numpy as np
import pytest
import torch

from tomography import sae
from tomography.datasets.sources import cache_dir
from tomography.hooks import capture_residuals
from tomography.model import MODELS
from tomography.model import LoadedModel
from tomography.sae import SAES
from tomography.sae import SaeSpec
from tomography.sae import load_decoder
from tomography.sae import sae_layer

REVISION = "0123456789abcdef0123456789abcdef01234567"
PINS = {
    "qwen-scope-1.7b-base-l0-50": (
        "Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50",
        "ce1a79d9c5163932d65c417380e53230e1086370",
        50,
    ),
    "qwen-scope-1.7b-base-l0-100": (
        "Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_100",
        "2756a49850d1d0288a18e561783deae0d810791e",
        100,
    ),
}


def _spec(**overrides: object) -> SaeSpec:
    fields: dict[str, object] = {
        "key": "tiny-sae",
        "repo_id": "owner/tiny-sae",
        "revision": REVISION,
        "model": "tiny",
        "n_layers": 3,
        "d_model": 32,
        "width": 20,
        "k": 4,
        "license": "test",
    }
    return SaeSpec(**{**fields, **overrides})  # type: ignore[arg-type]


def test_the_registry_holds_the_two_pinned_qwen_scope_releases() -> None:
    assert set(SAES) == set(PINS)
    for key, (repo_id, revision, k) in PINS.items():
        spec = SAES[key]
        assert spec.key == key
        assert (spec.repo_id, spec.revision, spec.k) == (repo_id, revision, k)
        assert spec.model == "qwen3-1.7b-base"
        assert spec.model in MODELS
        assert (spec.n_layers, spec.d_model, spec.width) == (28, 2048, 32768)
        assert "Apache-2.0" in spec.license
    with pytest.raises(TypeError):
        SAES["other"] = _spec()  # type: ignore[index]


@pytest.mark.parametrize(
    "revision",
    [
        "",
        "main",
        REVISION[:39],
        REVISION + "0",
        REVISION.upper(),
        "g" + REVISION[1:],
    ],
    ids=["empty", "branch", "short", "long", "upper", "not-hex"],
)
def test_a_spec_needs_a_full_commit_sha(revision: str) -> None:
    with pytest.raises(ValueError, match="revision"):
        _spec(revision=revision)


def test_a_spec_with_a_full_commit_sha_is_valid() -> None:
    assert _spec().revision == REVISION
    assert dataclasses.replace(_spec(), k=8).k == 8


def test_our_layer_l_is_the_sae_layer_l_minus_one() -> None:
    spec = SAES["qwen-scope-1.7b-base-l0-50"]
    assert sae_layer(1, spec) == 0
    assert sae_layer(14, spec) == 13
    assert sae_layer(28, spec) == 27


@pytest.mark.parametrize("layer", [-1, 0, 29, 100])
def test_sae_layer_refuses_a_layer_without_an_autoencoder(layer: int) -> None:
    spec = SAES["qwen-scope-1.7b-base-l0-50"]
    with pytest.raises(ValueError, match="layer"):
        sae_layer(layer, spec)


def test_the_sae_layer_is_the_block_output_that_our_layer_captures(
    tiny_qwen3: LoadedModel,
) -> None:
    """Hooks `model.model.layers[n]` as the Qwen-Scope README does."""
    spec = _spec(n_layers=tiny_qwen3.n_layers)
    seen: dict[int, torch.Tensor] = {}

    def keep(n: int) -> Callable[[Any, Any, Any], None]:
        def hook(module: Any, args: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            seen[n] = hidden.detach()

        return hook

    handles = [
        tiny_qwen3.model.model.layers[n].register_forward_hook(keep(n))
        for n in range(spec.n_layers)
    ]
    ids = tiny_qwen3.encode(["The quick brown fox jumps"]).input_ids
    with capture_residuals(tiny_qwen3, [1, 2, 3]) as captured, torch.no_grad():
        tiny_qwen3.model(input_ids=ids)
    for handle in handles:
        handle.remove()
    for layer in (1, 2, 3):
        assert torch.equal(captured[layer], seen[sae_layer(layer, spec)])
    assert not torch.equal(seen[0], seen[1])


class _Hub:
    """Stands in for `hf_hub_download` and serves one saved checkpoint."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return str(self.path)


def _checkpoint(spec: SaeSpec, **overrides: torch.Tensor) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(0)
    state = {
        "W_enc": torch.randn(spec.width, spec.d_model, generator=generator),
        "W_dec": torch.randn(spec.d_model, spec.width, generator=generator),
        "b_enc": torch.randn(spec.width, generator=generator),
        "b_dec": torch.randn(spec.d_model, generator=generator),
    }
    return {**state, **overrides}


def _served(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    state: Any,
) -> _Hub:
    path = tmp_path / "layer.sae.pt"
    torch.save(state, path)
    hub = _Hub(path)
    monkeypatch.setattr(sae, "hf_hub_download", hub)
    monkeypatch.setenv("TOMOGRAPHY_CACHE", str(tmp_path / "cache"))
    return hub


def test_load_decoder_downloads_the_layer_file_at_the_pinned_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = _spec(n_layers=10)
    hub = _served(monkeypatch, tmp_path, _checkpoint(spec))
    load_decoder(spec, 7)
    assert hub.calls == [
        {
            "repo_id": "owner/tiny-sae",
            "filename": "layer7.sae.pt",
            "revision": REVISION,
            "cache_dir": cache_dir() / "hub",
        }
    ]
    assert cache_dir() == tmp_path / "cache"


def test_load_decoder_returns_the_saved_decoder_bit_for_bit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = _spec()
    state = _checkpoint(spec)
    _served(monkeypatch, tmp_path, state)
    decoder = load_decoder(spec, 1)
    assert isinstance(decoder, np.ndarray)
    assert decoder.dtype == np.float32
    assert decoder.shape == (spec.d_model, spec.width)
    assert decoder.flags["C_CONTIGUOUS"]
    assert decoder.flags["WRITEABLE"]
    assert decoder.flags["OWNDATA"]
    np.testing.assert_array_equal(decoder, state["W_dec"].numpy())


def test_load_decoder_maps_the_file_and_refuses_pickled_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = _spec()
    _served(monkeypatch, tmp_path, _checkpoint(spec))
    calls: list[dict[str, Any]] = []
    real = torch.load

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(sae.torch, "load", spy)
    load_decoder(spec, 0)
    assert calls == [
        {"map_location": "cpu", "weights_only": True, "mmap": True}
    ]


class _Payload:
    """An arbitrary object a pickled checkpoint could smuggle in."""


def test_load_decoder_refuses_a_checkpoint_that_pickles_an_object(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = _spec()
    state = {**_checkpoint(spec), "extra": _Payload()}
    _served(monkeypatch, tmp_path, state)
    with pytest.raises(pickle.UnpicklingError):
        load_decoder(spec, 0)


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("float16", "float32"),
        ("float64", "float32"),
        ("transposed", "shape"),
        ("narrow", "shape"),
        ("nan", "finite"),
        ("inf", "finite"),
        ("missing", "W_dec"),
        ("not-a-tensor", "W_dec"),
        ("not-a-dict", "W_dec"),
    ],
)
def test_load_decoder_refuses_a_file_that_is_not_a_decoder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str, match: str
) -> None:
    spec = _spec()
    good = _checkpoint(spec)["W_dec"]
    bad = good.clone()
    bad[0, 0] = float("nan") if case == "nan" else float("inf")
    states: dict[str, Any] = {
        "float16": _checkpoint(spec, W_dec=good.half()),
        "float64": _checkpoint(spec, W_dec=good.double()),
        "transposed": _checkpoint(spec, W_dec=good.T.contiguous()),
        "narrow": _checkpoint(spec, W_dec=good[:, :-1].contiguous()),
        "nan": _checkpoint(spec, W_dec=bad),
        "inf": _checkpoint(spec, W_dec=bad),
        "missing": {k: v for k, v in _checkpoint(spec).items() if k != "W_dec"},
        "not-a-tensor": {"W_dec": None},
        "not-a-dict": [1, 2],
    }
    _served(monkeypatch, tmp_path, states[case])
    with pytest.raises(ValueError, match=match):
        load_decoder(spec, 0)


@pytest.mark.parametrize("layer", [-1, 3, 28])
def test_load_decoder_refuses_a_layer_before_any_download(
    monkeypatch: pytest.MonkeyPatch, layer: int
) -> None:
    def downloaded(**kwargs: Any) -> str:
        raise AssertionError("a file was requested")

    monkeypatch.setattr(sae, "hf_hub_download", downloaded)
    with pytest.raises(ValueError, match="layer"):
        load_decoder(_spec(), layer)


@pytest.mark.network
def test_the_real_layer_0_decoder_matches_the_pin() -> None:
    spec = SAES["qwen-scope-1.7b-base-l0-50"]
    decoder = load_decoder(spec, 0)
    assert decoder.shape == (spec.d_model, spec.width)
    assert decoder.dtype == np.float32
    assert np.isfinite(decoder).all()
    norms = np.linalg.norm(decoder, axis=0)
    assert np.abs(norms - 1.0).max() < 1e-4
    config_file = hf_hub_download(
        repo_id=spec.repo_id,
        filename="config.json",
        revision=spec.revision,
        cache_dir=cache_dir() / "hub",
    )
    config = json.loads(Path(config_file).read_text(encoding="utf-8"))
    assert config["d_model"] == spec.d_model
    assert config["d_sae"] == spec.width
    assert config["k"] == spec.k
    assert config["num_layers"] == spec.n_layers
    assert config["hook_point"] == "resid_post"
    assert MODELS[spec.model].repo_id == config["base_model"]
