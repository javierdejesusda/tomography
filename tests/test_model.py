import re

import pytest

from tomography.model import ARCHS
from tomography.model import MODELS
from tomography.model import ArchSpec

GPT2_PINS = {
    "gpt2": (
        "openai-community/gpt2",
        "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "float32",
    ),
    "gpt2-medium": (
        "openai-community/gpt2-medium",
        "6dcaa7a952f72f9298047fd5137cd6e4f05f41da",
        "float32",
    ),
    "gpt2-large": (
        "openai-community/gpt2-large",
        "32b71b12589c2f8d625668d2335a01cac3249519",
        "float32",
    ),
    "gpt2-xl": (
        "openai-community/gpt2-xl",
        "15ea56dee5df4983c59b2538573817e1667135e2",
        "float16",
    ),
}


def test_gpt2_arch_spec_declares_the_hf_layout() -> None:
    assert ARCHS["gpt2"] == ArchSpec(
        model_type="gpt2",
        blocks_path="transformer.h",
        final_norm_path="transformer.ln_f",
        embed_paths=("transformer.wte", "transformer.wpe"),
        lora_targets=("c_attn",),
        lora_fan_in_fan_out=True,
        default_bos="prepend",
    )


def test_arch_registry_is_keyed_by_model_type() -> None:
    assert all(key == spec.model_type for key, spec in ARCHS.items())


@pytest.mark.parametrize("key", sorted(GPT2_PINS))
def test_gpt2_checkpoints_are_pinned_to_full_commit_shas(key: str) -> None:
    repo_id, revision, dtype = GPT2_PINS[key]

    checkpoint = MODELS[key]

    assert (checkpoint.repo_id, checkpoint.revision, checkpoint.dtype) == (
        repo_id,
        revision,
        dtype,
    )
    assert checkpoint.arch == "gpt2"
    assert checkpoint.chat_format == "plain"
    assert not checkpoint.post_trained


def test_every_checkpoint_is_consistent_with_the_registries() -> None:
    for key, checkpoint in MODELS.items():
        assert checkpoint.key == key
        assert checkpoint.arch in ARCHS
        assert re.fullmatch(r"[0-9a-f]{40}", checkpoint.revision)
        assert (checkpoint.chat_format == "template") == (
            checkpoint.post_trained
        )
        assert checkpoint.license
