import re
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig
from transformers import LlamaForCausalLM

from tomography.model import ARCHS
from tomography.model import MODELS
from tomography.model import ArchSpec
from tomography.model import LoadedModel
from tomography.model import UnsupportedArchitectureError
from tomography.model import wrap_model

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


def test_wrap_model_precomputes_the_structural_fields(
    tiny_gpt2: LoadedModel,
) -> None:
    lm = tiny_gpt2

    assert lm.spec is ARCHS["gpt2"]
    assert lm.checkpoint is None
    assert (lm.n_layers, lm.d_model) == (3, 32)
    assert lm.decoder is lm.model.transformer
    assert lm.blocks is lm.model.transformer.h
    assert lm.final_norm is lm.model.transformer.ln_f
    assert lm.device == torch.device("cpu")
    assert lm.dtype == torch.float32
    assert lm.default_bos == "prepend"
    assert not lm.model.training


def test_wrap_model_reads_special_token_ids(tiny_gpt2: LoadedModel) -> None:
    eos_id = tiny_gpt2.tokenizer.eos_token_id

    assert tiny_gpt2.bos_id == eos_id
    assert tiny_gpt2.pad_id == eos_id


def test_unembedding_is_the_vocab_by_d_model_head_matrix(
    tiny_gpt2: LoadedModel,
) -> None:
    unembedding = tiny_gpt2.unembedding()

    assert unembedding.shape == (len(tiny_gpt2.tokenizer), 32)
    assert torch.equal(unembedding, tiny_gpt2.model.lm_head.weight)


def test_wrap_model_rejects_an_unregistered_architecture(
    tiny_gpt2: LoadedModel,
) -> None:
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
    )

    with pytest.raises(UnsupportedArchitectureError, match="llama"):
        wrap_model(LlamaForCausalLM(config), tiny_gpt2.tokenizer)


def test_wrap_model_requires_a_fast_tokenizer(tiny_gpt2: LoadedModel) -> None:
    slow = SimpleNamespace(is_fast=False)

    with pytest.raises(ValueError, match="fast tokenizer"):
        wrap_model(tiny_gpt2.model, slow)  # type: ignore[arg-type]
