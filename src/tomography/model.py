"""Model registry: architecture specs and pinned checkpoints."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

BosMode = Literal["prepend", "none"]


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
