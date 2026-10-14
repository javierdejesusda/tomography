import dataclasses
import itertools
import os
import re
from types import SimpleNamespace
from typing import Literal

import huggingface_hub
import pytest
import torch
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import GPT2Config
from transformers import GPT2LMHeadModel
from transformers import LlamaConfig
from transformers import LlamaForCausalLM
from transformers import Qwen3Config
from transformers import Qwen3ForCausalLM

import tomography.model
from tomography.model import ARCHS
from tomography.model import MODELS
from tomography.model import ArchSpec
from tomography.model import LoadedModel
from tomography.model import UnknownCheckpointError
from tomography.model import UnsupportedArchitectureError
from tomography.model import load_model
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


def test_encode_prepends_bos_by_default_for_gpt2(
    tiny_gpt2: LoadedModel,
) -> None:
    batch = tiny_gpt2.encode(["The fox"])

    assert batch.has_bos
    assert batch.input_ids[0, 0].item() == tiny_gpt2.bos_id
    assert batch.offsets[0][0] == (0, 0)
    assert batch.lengths == (batch.input_ids.shape[1],)
    assert batch.attention_mask.tolist() == [[1] * batch.lengths[0]]


def test_encode_without_bos_matches_the_plain_tokenization(
    tiny_gpt2: LoadedModel,
) -> None:
    text = "The quick brown fox"
    expected = tiny_gpt2.tokenizer(text, add_special_tokens=False)

    batch = tiny_gpt2.encode([text], bos="none")

    assert not batch.has_bos
    assert batch.input_ids[0].tolist() == expected["input_ids"]


def test_encode_offsets_keep_the_leading_space_of_each_token(
    tiny_gpt2: LoadedModel,
) -> None:
    text = " The quick brown fox"

    batch = tiny_gpt2.encode([text], bos="none")

    offsets = batch.offsets[0]
    assert offsets[0][0] == 0
    assert offsets[-1][1] == len(text)
    assert all(a[1] == b[0] for a, b in itertools.pairwise(offsets))


def test_encode_pads_on_the_right_with_zero_offsets(
    tiny_gpt2: LoadedModel,
) -> None:
    batch = tiny_gpt2.encode(["The quick brown fox", "The"], bos="none")

    short, long = batch.lengths[1], batch.lengths[0]
    assert short < long
    assert batch.input_ids.shape == (2, long)
    assert batch.input_ids[1, short:].tolist() == [tiny_gpt2.pad_id] * (
        long - short
    )
    assert batch.attention_mask[1].tolist() == [1] * short + [0] * (
        long - short
    )
    assert batch.offsets[1][short:] == ((0, 0),) * (long - short)
    assert len(batch.offsets[1]) == long


def test_encode_pads_on_the_left_with_explicit_position_ids(
    tiny_gpt2: LoadedModel,
) -> None:
    batch = tiny_gpt2.encode(
        ["The quick brown fox", "The"], bos="none", padding_side="left"
    )

    short, long = batch.lengths[1], batch.lengths[0]
    pad = long - short
    assert batch.attention_mask[1].tolist() == [0] * pad + [1] * short
    assert batch.offsets[1][:pad] == ((0, 0),) * pad
    assert batch.position_ids[1].tolist() == [0] * pad + list(range(short))
    assert batch.position_ids[0].tolist() == list(range(long))


def test_encode_position_ids_are_the_clamped_mask_cumsum(
    tiny_gpt2: LoadedModel,
) -> None:
    batch = tiny_gpt2.encode(["The quick brown fox", "The"])

    expected = (batch.attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
    assert torch.equal(batch.position_ids, expected)
    assert batch.input_ids.dtype == torch.int64
    assert batch.attention_mask.dtype == torch.int64
    assert batch.position_ids.dtype == torch.int64


def test_encode_keeps_special_token_strings_as_single_tokens(
    tiny_gpt2: LoadedModel,
) -> None:
    batch = tiny_gpt2.encode(["fox<|endoftext|>dog"], bos="none")

    assert batch.input_ids[0].tolist().count(tiny_gpt2.bos_id) == 1


def test_encode_never_truncates(
    tiny_gpt2: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tiny_gpt2.tokenizer, "model_max_length", 4)

    batch = tiny_gpt2.encode(["The quick brown fox jumps over the lazy dog"])

    assert batch.lengths[0] > 4


def test_encode_prepend_without_a_bos_id_raises(
    tiny_gpt2: LoadedModel,
) -> None:
    lm = dataclasses.replace(tiny_gpt2, bos_id=None)

    with pytest.raises(ValueError, match="bos"):
        lm.encode(["The fox"], bos="prepend")
    with pytest.raises(ValueError, match="bos"):
        lm.encode(["The fox"])


def test_encode_rejects_an_empty_row_and_an_empty_batch(
    tiny_gpt2: LoadedModel,
) -> None:
    with pytest.raises(ValueError, match="text 1"):
        tiny_gpt2.encode(["The fox", ""], bos="none")
    with pytest.raises(ValueError, match="empty"):
        tiny_gpt2.encode([])


def test_encode_rejects_an_unknown_padding_side(
    tiny_gpt2: LoadedModel,
) -> None:
    with pytest.raises(ValueError, match="padding_side"):
        tiny_gpt2.encode(["The fox"], padding_side="middle")  # type: ignore[arg-type]


def _position_limit(lm: LoadedModel) -> int:
    return int(lm.model.config.max_position_embeddings)


def _text_of_tokens(lm: LoadedModel, count: int) -> str:
    """Builds a text that tokenizes to exactly `count` tokens."""
    for words in range(1, count + 1):
        for suffix in ("", " fox"):
            text = " ".join(["The"] * words) + suffix
            ids = lm.tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) == count:
                return text
    raise AssertionError(f"cannot build a text of {count} tokens")


def test_encode_accepts_a_row_of_exactly_the_position_limit(
    tiny_gpt2: LoadedModel,
) -> None:
    limit = _position_limit(tiny_gpt2)

    batch = tiny_gpt2.encode([_text_of_tokens(tiny_gpt2, limit)], bos="none")

    assert batch.lengths == (limit,)


@pytest.mark.parametrize(
    "padding_side", ["right", "left"], ids=["right", "left"]
)
def test_encode_rejects_a_row_over_the_position_limit(
    tiny_gpt2: LoadedModel, padding_side: Literal["left", "right"]
) -> None:
    limit = _position_limit(tiny_gpt2)
    text = _text_of_tokens(tiny_gpt2, limit + 1)

    with pytest.raises(
        ValueError,
        match=rf"text 0 has {limit + 1} tokens, more than the {limit} "
        "positions of this model",
    ):
        tiny_gpt2.encode([text], bos="none", padding_side=padding_side)


def test_encode_counts_the_bos_toward_the_position_limit(
    tiny_gpt2: LoadedModel,
) -> None:
    limit = _position_limit(tiny_gpt2)
    text = _text_of_tokens(tiny_gpt2, limit)

    with pytest.raises(ValueError, match=rf"text 0 has {limit + 1} tokens"):
        tiny_gpt2.encode([text])
    assert tiny_gpt2.encode([text], bos="none").lengths == (limit,)


def test_encode_reports_the_first_offending_row_of_a_batch(
    tiny_gpt2: LoadedModel,
) -> None:
    too_long = _text_of_tokens(tiny_gpt2, _position_limit(tiny_gpt2) + 1)

    with pytest.raises(ValueError, match=r"text 1 "):
        tiny_gpt2.encode(["The fox", too_long, too_long], bos="none")


def test_encode_skips_the_check_without_max_position_embeddings(
    tiny_gpt2: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tiny_gpt2.model, "config", SimpleNamespace())
    text = _text_of_tokens(tiny_gpt2, 200)

    batch = tiny_gpt2.encode([text], bos="none")

    assert batch.lengths == (200,)


class FakeHub:
    """Stands in for `from_pretrained` and records how it was called."""

    def __init__(self, loaded: LoadedModel) -> None:
        self.loaded = loaded
        self.model_calls: list[tuple[str, dict[str, object]]] = []
        self.tokenizer_calls: list[tuple[str, dict[str, object]]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hub = self

        class FakeModel:
            @staticmethod
            def from_pretrained(repo_id: str, **kwargs: object) -> object:
                hub.model_calls.append((repo_id, kwargs))
                return hub.loaded.model

        class FakeTokenizer:
            @staticmethod
            def from_pretrained(repo_id: str, **kwargs: object) -> object:
                hub.tokenizer_calls.append((repo_id, kwargs))
                return hub.loaded.tokenizer

        monkeypatch.setattr(tomography.model, "AutoModelForCausalLM", FakeModel)
        monkeypatch.setattr(tomography.model, "AutoTokenizer", FakeTokenizer)


def test_load_model_rejects_an_unknown_key() -> None:
    with pytest.raises(UnknownCheckpointError, match="nope"):
        load_model("nope")


def test_load_model_pins_the_revision_and_default_dtype(
    tiny_gpt2: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = FakeHub(tiny_gpt2)
    hub.install(monkeypatch)

    loaded = load_model("gpt2", device="cpu")

    checkpoint = MODELS["gpt2"]
    assert hub.model_calls == [
        (
            checkpoint.repo_id,
            {"revision": checkpoint.revision, "dtype": torch.float32},
        )
    ]
    assert hub.tokenizer_calls == [
        (checkpoint.repo_id, {"revision": checkpoint.revision})
    ]
    assert loaded.checkpoint is checkpoint
    assert loaded.device == torch.device("cpu")


def test_load_model_honors_a_dtype_override(
    tiny_gpt2: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = FakeHub(tiny_gpt2)
    hub.install(monkeypatch)

    load_model("gpt2-xl", device="cpu", dtype=torch.bfloat16)

    assert hub.model_calls[0][1]["dtype"] is torch.bfloat16


def test_load_model_defaults_to_cpu_without_cuda(
    tiny_gpt2: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeHub(tiny_gpt2).install(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert load_model("gpt2").device == torch.device("cpu")


def test_load_model_silences_the_symlink_warning_and_conversion(
    tiny_gpt2: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeHub(tiny_gpt2).install(monkeypatch)
    monkeypatch.setattr(
        huggingface_hub.constants, "HF_HUB_DISABLE_SYMLINKS_WARNING", False
    )
    monkeypatch.setenv("DISABLE_SAFETENSORS_CONVERSION", "placeholder")
    monkeypatch.delenv("DISABLE_SAFETENSORS_CONVERSION")

    load_model("gpt2", device="cpu")

    assert huggingface_hub.constants.HF_HUB_DISABLE_SYMLINKS_WARNING is True
    assert os.environ["DISABLE_SAFETENSORS_CONVERSION"] == "1"


def test_load_model_keeps_an_explicit_conversion_setting(
    tiny_gpt2: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeHub(tiny_gpt2).install(monkeypatch)
    monkeypatch.setenv("DISABLE_SAFETENSORS_CONVERSION", "0")

    load_model("gpt2", device="cpu")

    assert os.environ["DISABLE_SAFETENSORS_CONVERSION"] == "0"


@pytest.mark.network
def test_real_gpt2_tokenizer_offsets_keep_the_leading_space() -> None:
    checkpoint = MODELS["gpt2"]
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint.repo_id, revision=checkpoint.revision
    )
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_layer=3,
        n_embd=32,
        n_head=4,
        n_positions=64,
        bos_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    lm = wrap_model(GPT2LMHeadModel(config), tokenizer)
    text = "Is it good? The movie is"

    batch = lm.encode([text])

    assert lm.bos_id == 50256
    assert batch.input_ids[0, 0].item() == 50256
    start = text.index(" The")
    assert (start, start + len(" The")) in batch.offsets[0]


@pytest.mark.network
def test_load_model_loads_the_pinned_gpt2_on_cpu() -> None:
    loaded = load_model("gpt2", device="cpu")

    assert loaded.checkpoint is MODELS["gpt2"]
    assert (loaded.n_layers, loaded.d_model) == (12, 768)
    assert loaded.dtype == torch.float32
    assert loaded.bos_id == loaded.pad_id == 50256
    assert loaded.default_bos == "prepend"
    assert loaded.unembedding().shape == (50257, 768)


QWEN3_PINS = {
    "qwen3-1.7b-base": (
        "Qwen/Qwen3-1.7B-Base",
        "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
        False,
    ),
    "qwen3-1.7b": (
        "Qwen/Qwen3-1.7B",
        "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        True,
    ),
    "qwen3-0.6b-base": (
        "Qwen/Qwen3-0.6B-Base",
        "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        False,
    ),
    "qwen3-0.6b": (
        "Qwen/Qwen3-0.6B",
        "c1899de289a04d12100db370d81485cdf75e47ca",
        True,
    ),
}


def test_qwen3_arch_spec_declares_the_hf_layout() -> None:
    assert ARCHS["qwen3"] == ArchSpec(
        model_type="qwen3",
        blocks_path="model.layers",
        final_norm_path="model.norm",
        embed_paths=("model.embed_tokens",),
        lora_targets=("q_proj", "v_proj"),
        lora_fan_in_fan_out=False,
        default_bos="prepend",
    )


@pytest.mark.parametrize("key", sorted(QWEN3_PINS))
def test_qwen3_checkpoints_are_pinned_to_full_commit_shas(key: str) -> None:
    repo_id, revision, post_trained = QWEN3_PINS[key]

    checkpoint = MODELS[key]

    assert (checkpoint.repo_id, checkpoint.revision) == (repo_id, revision)
    assert checkpoint.arch == "qwen3"
    assert checkpoint.dtype == "bfloat16"
    assert checkpoint.license == "Apache-2.0"
    assert checkpoint.post_trained is post_trained


def test_tiny_qwen3_follows_the_qwen3_layout_without_a_bos_token(
    tiny_qwen3: LoadedModel,
) -> None:
    lm = tiny_qwen3
    end_id = lm.tokenizer.convert_tokens_to_ids("<|endoftext|>")

    assert lm.spec is ARCHS["qwen3"]
    assert lm.checkpoint is None
    assert (lm.n_layers, lm.d_model) == (3, 32)
    assert lm.decoder is lm.model.model
    assert lm.blocks is lm.model.model.layers
    assert lm.final_norm is lm.model.model.norm
    assert lm.tokenizer.bos_token is None
    assert lm.bos_id == lm.pad_id == end_id
    assert lm.default_bos == "prepend"
    assert lm.encode(["The fox"]).has_bos
    assert lm.encode(["The fox"], bos="prepend").input_ids[0, 0] == end_id


def test_tiny_qwen3_chat_template_renders_the_non_thinking_prompt(
    tiny_qwen3: LoadedModel,
) -> None:
    prompt = tiny_qwen3.tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    assert prompt == (
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def test_tiny_qwen3_keeps_chat_special_tokens_whole(
    tiny_qwen3: LoadedModel,
) -> None:
    lm = tiny_qwen3
    start = lm.tokenizer.convert_tokens_to_ids("<|im_start|>")
    end = lm.tokenizer.convert_tokens_to_ids("<|im_end|>")

    ids = lm.encode(["<|im_start|>user\nhi<|im_end|>"]).input_ids[0].tolist()

    assert ids.count(start) == 1
    assert ids.count(end) == 1


def test_template_chat_checkpoints_never_prepend_bos_by_default(
    tiny_gpt2: LoadedModel,
) -> None:
    checkpoint = dataclasses.replace(
        MODELS["gpt2"], post_trained=True, chat_format="template"
    )

    lm = wrap_model(tiny_gpt2.model, tiny_gpt2.tokenizer, checkpoint=checkpoint)

    assert tiny_gpt2.spec.default_bos == "prepend"
    assert lm.default_bos == "none"
    assert not lm.encode(["The fox"]).has_bos
    assert lm.encode(["The fox"], bos="prepend").has_bos


def test_plain_checkpoints_keep_the_architecture_bos_policy(
    tiny_gpt2: LoadedModel,
) -> None:
    lm = wrap_model(
        tiny_gpt2.model, tiny_gpt2.tokenizer, checkpoint=MODELS["gpt2"]
    )

    assert lm.default_bos == "prepend"


def test_wrap_model_rejects_a_checkpoint_of_another_architecture(
    tiny_gpt2: LoadedModel,
) -> None:
    with pytest.raises(ValueError, match="arch"):
        wrap_model(
            tiny_gpt2.model,
            tiny_gpt2.tokenizer,
            checkpoint=MODELS["qwen3-0.6b-base"],
        )


def test_load_model_pins_every_checkpoint_of_the_architecture(
    tiny_arch: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = FakeHub(tiny_arch)
    hub.install(monkeypatch)
    keys = [
        key
        for key, checkpoint in MODELS.items()
        if checkpoint.arch == tiny_arch.spec.model_type
    ]

    for key in keys:
        loaded = load_model(key, device="cpu")

        checkpoint = MODELS[key]
        assert hub.model_calls[-1] == (
            checkpoint.repo_id,
            {
                "revision": checkpoint.revision,
                "dtype": getattr(torch, checkpoint.dtype),
            },
        )
        assert hub.tokenizer_calls[-1] == (
            checkpoint.repo_id,
            {"revision": checkpoint.revision},
        )
        assert loaded.checkpoint is checkpoint
        expected_bos = (
            "none"
            if checkpoint.chat_format == "template"
            else tiny_arch.spec.default_bos
        )
        assert loaded.default_bos == expected_bos
    assert len(hub.model_calls) == len(keys) >= 1


@pytest.mark.network
@pytest.mark.parametrize("key", ["qwen3-0.6b-base", "qwen3-0.6b"])
def test_real_qwen3_tokenizer_and_config_match_the_registry(key: str) -> None:
    checkpoint = MODELS[key]
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint.repo_id, revision=checkpoint.revision
    )
    config = AutoConfig.from_pretrained(
        checkpoint.repo_id, revision=checkpoint.revision
    )
    tiny = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=len(tokenizer),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
        )
    )
    lm = wrap_model(tiny, tokenizer, checkpoint=checkpoint)
    text = "Is it good? The movie is"

    batch = lm.encode([text])

    assert tokenizer.bos_token is None
    assert lm.bos_id == lm.pad_id == 151643
    expected = "none" if checkpoint.chat_format == "template" else "prepend"
    assert lm.default_bos == expected
    assert batch.has_bos == (expected == "prepend")
    assert (batch.input_ids[0, 0].item() == 151643) == (expected == "prepend")
    start = text.index(" The")
    assert (start, start + len(" The")) in batch.offsets[0]
    if checkpoint.chat_format == "template":
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        assert prompt == (
            "<|im_start|>user\nhi<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
