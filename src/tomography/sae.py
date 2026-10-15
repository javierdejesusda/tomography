"""Qwen-Scope sparse autoencoders of the residual stream, read by revision.

A5 reads only the decoder of each autoencoder: a column of `W_dec` is a unit
direction of the residual stream. Weights are fetched from the Hub at a pinned
commit into the project cache, one layer file at a time, and are never copied
into a run.
"""

from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
import re
from types import MappingProxyType

from huggingface_hub import hf_hub_download
import numpy as np
from numpy.typing import NDArray
import torch

from tomography.datasets.sources import cache_dir

Decoders = Callable[[int], NDArray[np.float32]]

_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class SaeSpec:
    """One pinned release of per-layer sparse autoencoders.

    Attributes:
        key: Registry key.
        repo_id: Hub repository id.
        revision: Full 40-hex commit SHA.
        model: Key of `MODELS` the autoencoders were trained on.
        n_layers: Number of autoencoder layers, one per decoder block.
        d_model: Width of the residual stream.
        width: Number of features per layer.
        k: Number of active features of the TopK activation.
        license: License terms of the weights.

    Raises:
        ValueError: If `revision` is not a full lowercase 40-hex commit SHA.
    """

    key: str
    repo_id: str
    revision: str
    model: str
    n_layers: int
    d_model: int
    width: int
    k: int
    license: str

    def __post_init__(self) -> None:
        """Validates that the release is pinned to a commit."""
        if not _COMMIT_SHA.fullmatch(self.revision):
            raise ValueError(
                f"{self.key}: revision must be a full 40-character lowercase "
                f"commit SHA, got {self.revision!r}"
            )


_LICENSE = (
    "other (qwen); Apache-2.0 LICENSE file; README caution on non-scientific "
    "interference"
)


def _release(k: int, revision: str) -> SaeSpec:
    return SaeSpec(
        key=f"qwen-scope-1.7b-base-l0-{k}",
        repo_id=f"Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_{k}",
        revision=revision,
        model="qwen3-1.7b-base",
        n_layers=28,
        d_model=2048,
        width=32768,
        k=k,
        license=_LICENSE,
    )


SAES: Mapping[str, SaeSpec] = MappingProxyType(
    {
        spec.key: spec
        for spec in (
            _release(50, "ce1a79d9c5163932d65c417380e53230e1086370"),
            _release(100, "2756a49850d1d0288a18e561783deae0d810791e"),
        )
    }
)


def sae_layer(layer: int, spec: SaeSpec) -> int:
    """Maps a layer of the project convention to an autoencoder layer.

    Our layer `l >= 1` is the raw output of block `l`, and the Qwen-Scope
    autoencoder `n` is trained on the output of `model.model.layers[n]`, which
    is block `n + 1`. So `l` is read by autoencoder `l - 1`, and layer 0 (the
    embeddings) has none.

    Args:
        layer: A layer in 1..`spec.n_layers`.
        spec: The autoencoder release.

    Returns:
        The autoencoder layer, in 0..`spec.n_layers - 1`.

    Raises:
        ValueError: If `layer` is outside 1..`spec.n_layers`.
    """
    if not 1 <= layer <= spec.n_layers:
        raise ValueError(
            f"layer {layer} has no autoencoder in {spec.key}, expected "
            f"1..{spec.n_layers}"
        )
    return layer - 1


def load_decoder(spec: SaeSpec, layer: int) -> NDArray[np.float32]:
    """Loads the decoder of one autoencoder layer from the project cache.

    The layer file is downloaded at the pinned revision when it is not cached
    and memory-mapped, so only `W_dec` is read from it.

    Args:
        spec: The autoencoder release.
        layer: The autoencoder layer, in 0..`spec.n_layers - 1`.

    Returns:
        `W_dec` as a contiguous float32 array of shape (d_model, width); a
        column is the residual-stream direction of one feature.

    Raises:
        ValueError: If `layer` is out of range (checked before any
            download), or the file has no `W_dec` tensor, or it is not
            float32, not of shape (d_model, width) or not finite.
    """
    if not 0 <= layer < spec.n_layers:
        raise ValueError(
            f"layer {layer} is outside 0..{spec.n_layers - 1} for {spec.key}"
        )
    path = hf_hub_download(
        repo_id=spec.repo_id,
        filename=f"layer{layer}.sae.pt",
        revision=spec.revision,
        cache_dir=cache_dir() / "hub",
    )
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    weight = state.get("W_dec") if isinstance(state, dict) else None
    label = f"{spec.key} layer {layer}"
    if not isinstance(weight, torch.Tensor):
        raise ValueError(f"{label} holds no W_dec tensor")
    if weight.dtype is not torch.float32:
        raise ValueError(f"{label}: W_dec must be float32, got {weight.dtype}")
    shape = (spec.d_model, spec.width)
    if tuple(weight.shape) != shape:
        raise ValueError(
            f"{label}: W_dec must have shape {shape}, got {tuple(weight.shape)}"
        )
    if not bool(torch.isfinite(weight).all()):
        raise ValueError(f"{label}: W_dec must be finite")
    return weight.numpy().copy()
