"""Model registry, loading and the tokenizer policy shared by all readers."""

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
import os
from types import MappingProxyType
from typing import Literal

import huggingface_hub.constants
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
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
        "qwen3": ArchSpec(
            model_type="qwen3",
            blocks_path="model.layers",
            final_norm_path="model.norm",
            embed_paths=("model.embed_tokens",),
            lora_targets=("q_proj", "v_proj"),
            lora_fan_in_fan_out=False,
            # The BOS probe in tests/test_models_slow.py: without BOS the first
            # token is the sink; a prepended separator takes the role over.
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


def _qwen3(
    key: str, repo_id: str, revision: str, post_trained: bool
) -> CheckpointSpec:
    return CheckpointSpec(
        key=key,
        repo_id=repo_id,
        revision=revision,
        arch="qwen3",
        post_trained=post_trained,
        license="Apache-2.0",
        dtype="bfloat16",
        chat_format="template" if post_trained else "plain",
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
            _qwen3(
                "qwen3-1.7b-base",
                "Qwen/Qwen3-1.7B-Base",
                "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
                post_trained=False,
            ),
            _qwen3(
                "qwen3-1.7b",
                "Qwen/Qwen3-1.7B",
                "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
                post_trained=True,
            ),
            _qwen3(
                "qwen3-0.6b-base",
                "Qwen/Qwen3-0.6B-Base",
                "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
                post_trained=False,
            ),
            _qwen3(
                "qwen3-0.6b",
                "Qwen/Qwen3-0.6B",
                "c1899de289a04d12100db370d81485cdf75e47ca",
                post_trained=True,
            ),
        )
    }
)


@dataclass(frozen=True, eq=False)
class EncodedBatch:
    """A tokenized batch with explicit positions and character offsets.

    Attributes:
        input_ids: `(batch, seq)` int64 CPU token ids.
        attention_mask: `(batch, seq)` int64, 1 for every real token
            including BOS.
        position_ids: `(batch, seq)` int64, the mask cumulative sum minus
            one, clamped at 0. HF derives no positions from padding.
        offsets: Per row and token, the `(start, end)` character range;
            `(0, 0)` for BOS and padding.
        lengths: Real token count per row, BOS included.
        has_bos: Whether BOS was prepended to every row.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    offsets: tuple[tuple[tuple[int, int], ...], ...]
    lengths: tuple[int, ...]
    has_bos: bool


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

    def encode(
        self,
        texts: Sequence[str],
        *,
        bos: BosMode | None = None,
        padding_side: Literal["left", "right"] = "right",
    ) -> EncodedBatch:
        """Tokenizes texts with the BOS, padding and offset policy.

        Special-token strings in the text stay single tokens and nothing is
        truncated. Collection, generation and LoRRA all encode here so they
        share one policy.

        Args:
            texts: The texts to encode.
            bos: `prepend` puts `bos_id` before every row, `none` adds
                nothing, None uses `default_bos`.
            padding_side: Side padded with `pad_id`.

        Returns:
            The encoded batch.

        Raises:
            ValueError: If `texts` is empty, a text has no tokens, the
                padding side is unknown, or BOS is requested without an id.
        """
        if padding_side not in ("left", "right"):
            raise ValueError(
                f"padding_side must be 'left' or 'right', got {padding_side!r}"
            )
        if not texts:
            raise ValueError("texts is empty")
        mode = self.default_bos if bos is None else bos
        if mode == "prepend" and self.bos_id is None:
            raise ValueError(
                "bos='prepend' needs a bos id and this model has none"
            )
        encoding = self.tokenizer(
            list(texts),
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            padding=False,
        )
        prefix_ids = [self.bos_id] if mode == "prepend" else []
        prefix_offsets = [(0, 0)] if mode == "prepend" else []
        rows_ids = [prefix_ids + ids for ids in encoding["input_ids"]]
        rows_offsets = [
            prefix_offsets + [(int(a), int(b)) for a, b in offsets]
            for offsets in encoding["offset_mapping"]
        ]
        for index, ids in enumerate(rows_ids):
            if not ids:
                raise ValueError(f"text {index} has no tokens")
        lengths = tuple(len(ids) for ids in rows_ids)
        width = max(lengths)
        ids_out, mask_out, offsets_out = [], [], []
        for ids, offsets in zip(rows_ids, rows_offsets, strict=True):
            pad = width - len(ids)
            left, right = (pad, 0) if padding_side == "left" else (0, pad)
            ids_out.append([self.pad_id] * left + ids + [self.pad_id] * right)
            mask_out.append([0] * left + [1] * len(ids) + [0] * right)
            offsets_out.append(
                tuple([(0, 0)] * left + offsets + [(0, 0)] * right)
            )
        attention_mask = torch.tensor(mask_out, dtype=torch.int64)
        return EncodedBatch(
            input_ids=torch.tensor(ids_out, dtype=torch.int64),
            attention_mask=attention_mask,
            position_ids=(attention_mask.cumsum(dim=-1) - 1).clamp(min=0),
            offsets=tuple(offsets_out),
            lengths=lengths,
            has_bos=mode == "prepend",
        )


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
        The wrapped model; `model` is put in eval mode. Its `default_bos`
        is `none` for template-chat checkpoints (the chat template carries
        its own special tokens) and the architecture's policy otherwise.

    Raises:
        UnsupportedArchitectureError: If `config.model_type` is not in
            `ARCHS`.
        ValueError: If the checkpoint belongs to another architecture, the
            tokenizer is not fast, the block path is not a `ModuleList`, or
            no pad id can be derived.
    """
    model_type = model.config.model_type
    spec = ARCHS.get(model_type)
    if spec is None:
        raise UnsupportedArchitectureError(
            f"Unsupported architecture {model_type!r}; "
            f"registered: {sorted(ARCHS)}"
        )
    if checkpoint is not None and checkpoint.arch != spec.model_type:
        raise ValueError(
            f"Checkpoint {checkpoint.key!r} has arch {checkpoint.arch!r} "
            f"but the model is {spec.model_type!r}"
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
    templated = checkpoint is not None and checkpoint.chat_format == "template"
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
        default_bos="none" if templated else spec.default_bos,
    )


def load_model(
    key: str,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> LoadedModel:
    """Loads a pinned checkpoint and wraps it.

    The revision is always the pinned commit SHA. Loading silences the
    Windows symlink warning and disables transformers' safetensors
    auto-conversion, which could otherwise open pull requests on `.bin`
    repositories.

    Args:
        key: A key of `MODELS`.
        device: Target device; defaults to CUDA when available, else CPU.
        dtype: Weight dtype; defaults to the checkpoint's `dtype`.

    Returns:
        The wrapped model in eval mode.

    Raises:
        UnknownCheckpointError: If `key` is not in `MODELS`.
        UnsupportedArchitectureError: If the checkpoint's architecture is
            not registered.
    """
    checkpoint = MODELS.get(key)
    if checkpoint is None:
        raise UnknownCheckpointError(
            f"Unknown checkpoint {key!r}; known: {sorted(MODELS)}"
        )
    huggingface_hub.constants.HF_HUB_DISABLE_SYMLINKS_WARNING = True
    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
    if dtype is None:
        dtype = getattr(torch, checkpoint.dtype)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint.repo_id, revision=checkpoint.revision, dtype=dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint.repo_id, revision=checkpoint.revision
    )
    model = model.to(device)  # type: ignore[arg-type]
    return wrap_model(model, tokenizer, checkpoint=checkpoint)
