"""Model registry, loading and the tokenizer policy shared by all readers."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import torch
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerBase

BosMode = Literal["prepend", "none"]


class UnknownCheckpointError(ValueError):
    """Raised when a checkpoint key is not in `MODELS`."""


class UnsupportedArchitectureError(ValueError):
    """Raised when a model's `config.model_type` is not in `ARCHS`."""


@dataclass(frozen=True)
class ArchSpec:
    """Where one architecture keeps the parts that hooks and LoRA touch.

    Specs are keyed by `config.model_type`. Paths are dotted attribute paths
    from the `*ForCausalLM` module.

    Attributes:
        model_type: The Hugging Face `config.model_type` this spec covers.
        blocks_path: Path of the `ModuleList` of decoder blocks.
        final_norm_path: Path of the norm applied after the last block.
        embed_paths: Paths of the embedding modules feeding block 1.
        lora_targets: Module names LoRA adapters attach to.
        lora_fan_in_fan_out: Whether the LoRA targets store weights
            transposed (GPT-2 `Conv1D`).
        default_bos: BOS policy for plain-format stimuli.
    """

    model_type: str
    blocks_path: str
    final_norm_path: str
    embed_paths: tuple[str, ...]
    lora_targets: tuple[str, ...]
    lora_fan_in_fan_out: bool
    default_bos: BosMode


@dataclass(frozen=True)
class CheckpointSpec:
    """One pinned Hugging Face checkpoint.

    Attributes:
        key: Registry key, for example `gpt2-medium`.
        repo_id: Hub repository id.
        revision: Full 40-hex commit SHA.
        arch: Key into `ARCHS`.
        post_trained: Whether the checkpoint is instruction tuned.
        license: License name for the README table.
        dtype: Default `torch` dtype name, for example `float32`.
        chat_format: `template` iff the checkpoint is post-trained.
    """

    key: str
    repo_id: str
    revision: str
    arch: str
    post_trained: bool
    license: str
    dtype: str
    chat_format: Literal["plain", "template"]


ARCHS: Mapping[str, ArchSpec] = MappingProxyType(
    {
        "gpt2": ArchSpec(
            model_type="gpt2",
            blocks_path="transformer.h",
            final_norm_path="transformer.ln_f",
            embed_paths=("transformer.wte", "transformer.wpe"),
            lora_targets=("c_attn",),
            lora_fan_in_fan_out=True,
            default_bos="prepend",
        ),
    }
)


def _gpt2(key: str, revision: str, dtype: str) -> CheckpointSpec:
    return CheckpointSpec(
        key=key,
        repo_id=f"openai-community/{key}",
        revision=revision,
        arch="gpt2",
        post_trained=False,
        license="Modified MIT",
        dtype=dtype,
        chat_format="plain",
    )


MODELS: Mapping[str, CheckpointSpec] = MappingProxyType(
    {
        spec.key: spec
        for spec in (
            _gpt2(
                "gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e", "float32"
            ),
            _gpt2(
                "gpt2-medium",
                "6dcaa7a952f72f9298047fd5137cd6e4f05f41da",
                "float32",
            ),
            _gpt2(
                "gpt2-large",
                "32b71b12589c2f8d625668d2335a01cac3249519",
                "float32",
            ),
            _gpt2(
                "gpt2-xl",
                "15ea56dee5df4983c59b2538573817e1667135e2",
                "float16",
            ),
        )
    }
)


@dataclass(frozen=True)
class LoadedModel:
    """A causal LM with its tokenizer and the modules hooks attach to.

    Attributes:
        model: The `*ForCausalLM` module, in eval mode.
        tokenizer: A fast tokenizer (character offsets are required).
        spec: The architecture spec of `model`.
        checkpoint: The pinned checkpoint, or None for test models.
        n_layers: Number of decoder blocks (L).
        d_model: Residual stream width.
        device: Device holding the weights.
        dtype: Weight dtype.
        bos_id: Id prepended by `encode(bos="prepend")`, if any.
        pad_id: Id used to pad batches.
        decoder: The base model without the LM head.
        blocks: The decoder blocks.
        final_norm: The norm applied after the last block.
        default_bos: BOS policy used when `encode` gets `bos=None`.
    """

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    spec: ArchSpec
    checkpoint: CheckpointSpec | None
    n_layers: int
    d_model: int
    device: torch.device
    dtype: torch.dtype
    bos_id: int | None
    pad_id: int
    decoder: torch.nn.Module
    blocks: torch.nn.ModuleList
    final_norm: torch.nn.Module
    default_bos: BosMode

    def unembedding(self) -> torch.Tensor:
        """Returns the output embedding matrix for the logit lens.

        Returns:
            A detached `(vocab, d_model)` tensor.

        Raises:
            ValueError: If the model has no output embedding.
        """
        head = self.model.get_output_embeddings()  # type: ignore[no-untyped-call]
        if head is None:
            raise ValueError("The model has no output embedding")
        weight: torch.Tensor = head.weight
        return weight.detach()


def _module_at(root: torch.nn.Module, path: str) -> torch.nn.Module:
    module = root
    for name in path.split("."):
        module = getattr(module, name)
    return module


def _pad_id(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    bos_id: int | None,
) -> int:
    candidates = (
        tokenizer.pad_token_id,
        model.config.pad_token_id,
        tokenizer.eos_token_id,
        bos_id,
    )
    for candidate in candidates:
        if candidate is not None:
            return int(candidate)
    raise ValueError("The tokenizer and config define no pad, eos or bos id")


def wrap_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    checkpoint: CheckpointSpec | None = None,
) -> LoadedModel:
    """Wraps a causal LM and precomputes the modules hooks need.

    Args:
        model: A `*ForCausalLM` module of a registered architecture.
        tokenizer: A fast tokenizer matching `model`.
        checkpoint: The pinned checkpoint `model` was loaded from, if any.

    Returns:
        The wrapped model; `model` is put in eval mode.

    Raises:
        UnsupportedArchitectureError: If `config.model_type` is not in
            `ARCHS`.
        ValueError: If the tokenizer is not fast, the block path is not a
            `ModuleList`, or no pad id can be derived.
    """
    model_type = model.config.model_type
    spec = ARCHS.get(model_type)
    if spec is None:
        raise UnsupportedArchitectureError(
            f"Unsupported architecture {model_type!r}; "
            f"registered: {sorted(ARCHS)}"
        )
    if not tokenizer.is_fast:
        raise ValueError(
            "A fast tokenizer is required: encode needs character offsets"
        )
    model.eval()  # type: ignore[no-untyped-call]
    blocks = _module_at(model, spec.blocks_path)
    if not isinstance(blocks, torch.nn.ModuleList):
        raise ValueError(f"{spec.blocks_path!r} is not a ModuleList")
    bos_id = model.config.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.bos_token_id
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        spec=spec,
        checkpoint=checkpoint,
        n_layers=len(blocks),
        d_model=model.config.hidden_size,
        device=model.device,
        dtype=model.dtype,
        bos_id=bos_id,
        pad_id=_pad_id(model, tokenizer, bos_id),
        decoder=_module_at(model, spec.blocks_path.rpartition(".")[0]),
        blocks=blocks,
        final_norm=_module_at(model, spec.final_norm_path),
        default_bos=spec.default_bos,
    )
