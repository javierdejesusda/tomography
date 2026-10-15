"""Tests for LoRRA: the batch layout, the loss, the adapter and its training."""

from collections.abc import Sequence
import copy
import dataclasses
import gc
import math
import re
import time
from typing import Any

import numpy as np
from peft import PeftModel
from peft.tuners.lora import LoraLayer
import pytest
import torch

from tomography import lorra
from tomography.datasets.alpaca import load_alpaca
from tomography.experiments.sweep import chat_format_for
from tomography.hooks import capture_residuals
from tomography.lorra import LorraAdapter
from tomography.lorra import LorraBatch
from tomography.lorra import LorraSettings
from tomography.lorra import LorraTrace
from tomography.lorra import Pair
from tomography.lorra import attach_adapter
from tomography.lorra import build_batch
from tomography.lorra import default_layers
from tomography.lorra import fit_rows
from tomography.lorra import heldout_losses
from tomography.lorra import layer_losses
from tomography.lorra import lorra_loss
from tomography.lorra import step_loss
from tomography.lorra import train_lorra
from tomography.model import BosMode
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.templates import ChatFormat

PAIRS: list[Pair] = [
    ("Is the movie good?", "Yes, the movie is very good."),
    ("Is the movie bad?", "Honest answers are true."),
    ("Pick a number.", "Pick a number between 0 and 9: 0123456789."),
    ("Is it a short text?", "A short text, a longer text, and another text."),
    ("Is the dog lazy?", "The quick brown fox jumps over the lazy dog."),
    ("Does she feel happy?", "She felt happy, then sad, then angry."),
]
LONG_INSTRUCTION = " ".join(
    ["The quick brown fox jumps over the lazy dog."] * 2
)

BAD_SETTINGS: dict[str, dict[str, Any]] = {
    "empty-layers": {"layers": ()},
    "repeated-layer": {"layers": (2, 2)},
    "decreasing-layers": {"layers": (3, 2)},
    "layer-zero": {"layers": (0, 1)},
    "rank": {"rank": 0},
    "lora-alpha": {"lora_alpha": 0},
    "batch-size": {"batch_size": 0},
    "steps": {"steps": 0},
    "prompt-window": {"max_prompt_tokens": 0},
    "response-window": {"max_response_tokens": 0},
    "dropout-one": {"dropout": 1.0},
    "dropout-negative": {"dropout": -0.1},
    "lr-zero": {"lr": 0.0},
    "lr-negative": {"lr": -1e-3},
    "lr-nan": {"lr": float("nan")},
    "loss-form": {"loss": "l1"},
}


def test_settings_defaults() -> None:
    settings = LorraSettings(layers=(1, 2))
    assert settings.layers == (1, 2)
    assert settings.alpha == 5.0
    assert settings.pos_control == "Give a truthful answer."
    assert settings.neg_control == "Give an untruthful answer."
    assert settings.rank == 8
    assert settings.lora_alpha == 16
    assert settings.dropout == 0.05
    assert settings.lr == 3e-4
    assert settings.batch_size == 16
    assert settings.steps == 70
    assert settings.max_prompt_tokens == 256
    assert settings.max_response_tokens == 64
    assert settings.loss == "mean_norm"
    assert settings.gradient_checkpointing is False
    assert settings.seed == 0


def test_settings_are_frozen() -> None:
    settings = LorraSettings(layers=(1,))
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.alpha = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides", list(BAD_SETTINGS.values()), ids=list(BAD_SETTINGS)
)
def test_settings_reject_bad_values(overrides: dict[str, Any]) -> None:
    kwargs: dict[str, Any] = {"layers": (1, 2), **overrides}
    with pytest.raises(ValueError, match=r"\w"):
        LorraSettings(**kwargs)


def test_settings_accept_the_sum_squared_form_and_zero_dropout() -> None:
    settings = LorraSettings(layers=(1,), loss="sum_squared", dropout=0.0)
    assert settings.loss == "sum_squared"
    assert settings.dropout == 0.0


@pytest.mark.parametrize(
    ("n_layers", "expected"),
    [
        (32, (10, 12, 14, 16, 18, 20)),
        (28, (9, 11, 13, 15, 17)),
        (12, (4, 6, 8)),
        (3, (1,)),
        (1, (1,)),
    ],
    ids=["reference-7b", "qwen3-1.7b", "gpt2-small", "tiny", "single-block"],
)
def test_default_layers(n_layers: int, expected: tuple[int, ...]) -> None:
    assert default_layers(n_layers) == expected


@pytest.mark.parametrize("n_layers", [0, -4], ids=["zero", "negative"])
def test_default_layers_reject_a_model_without_blocks(n_layers: int) -> None:
    with pytest.raises(ValueError, match="n_layers"):
        default_layers(n_layers)


def _settings(**overrides: Any) -> LorraSettings:
    base = LorraSettings(
        layers=(1, 2),
        batch_size=2,
        steps=3,
        max_prompt_tokens=48,
        max_response_tokens=8,
    )
    return dataclasses.replace(base, **overrides)


def _ids(loaded: LoadedModel, text: str) -> list[int]:
    return list(loaded.tokenizer(text, add_special_tokens=False)["input_ids"])


def _prompt_ids(
    loaded: LoadedModel,
    chat: ChatFormat,
    user: str,
    bos: BosMode | None = None,
) -> list[int]:
    mode = loaded.default_bos if bos is None else bos
    prefix = [loaded.bos_id] if mode == "prepend" else []
    return [*prefix, *_ids(loaded, chat.render_turn(user)[0])]


def _row(
    loaded: LoadedModel,
    chat: ChatFormat,
    settings: LorraSettings,
    user: str,
    response: str,
    bos: BosMode | None = None,
) -> tuple[list[int], list[int]]:
    """Builds the expected `(input_ids, attention_mask)` of one row by hand."""
    prompt = _prompt_ids(loaded, chat, user, bos)
    answer = _ids(loaded, response)[: settings.max_response_tokens]
    left = settings.max_prompt_tokens - len(prompt)
    right = settings.max_response_tokens - len(answer)
    ids = [loaded.pad_id] * left + prompt + answer + [loaded.pad_id] * right
    mask = [0] * left + [1] * (len(prompt) + len(answer)) + [0] * right
    return ids, mask


def test_fit_rows_keeps_everything_that_fits(tiny_arch: LoadedModel) -> None:
    kept, dropped = fit_rows(tiny_arch, PAIRS, ChatFormat("plain"), _settings())
    assert kept == PAIRS
    assert dropped == 0


def test_fit_rows_of_no_pairs_is_empty(tiny_arch: LoadedModel) -> None:
    kept, dropped = fit_rows(tiny_arch, [], ChatFormat("plain"), _settings())
    assert (kept, dropped) == ([], 0)


def test_fit_rows_drops_a_pair_whose_control_prompt_does_not_fit(
    tiny_arch: LoadedModel,
) -> None:
    chat = ChatFormat("plain")
    longer: Pair = (LONG_INSTRUCTION, "Yes, the movie is very good.")
    plain_length = len(_prompt_ids(tiny_arch, chat, LONG_INSTRUCTION))
    narrow = _settings(max_prompt_tokens=plain_length)
    assert len(_prompt_ids(tiny_arch, chat, PAIRS[0][0])) < plain_length
    kept, dropped = fit_rows(tiny_arch, [PAIRS[0], longer], chat, narrow)
    assert kept == [PAIRS[0]]
    assert dropped == 1


def test_fit_rows_window_is_inclusive(tiny_arch: LoadedModel) -> None:
    chat = ChatFormat("plain")
    negative = f"{PAIRS[0][0]} {_settings().neg_control}"
    need = len(_prompt_ids(tiny_arch, chat, negative))
    exact = _settings(max_prompt_tokens=need)
    below = _settings(max_prompt_tokens=need - 1)
    assert fit_rows(tiny_arch, [PAIRS[0]], chat, exact) == ([PAIRS[0]], 0)
    assert fit_rows(tiny_arch, [PAIRS[0]], chat, below) == ([], 1)


def test_fit_rows_counts_the_bos_in_the_window(tiny_arch: LoadedModel) -> None:
    chat = ChatFormat("plain")
    negative = f"{PAIRS[0][0]} {_settings().neg_control}"
    need = len(_prompt_ids(tiny_arch, chat, negative, "none"))
    narrow = _settings(max_prompt_tokens=need)
    kept, _ = fit_rows(tiny_arch, [PAIRS[0]], chat, narrow, bos="none")
    assert kept == [PAIRS[0]]
    kept, dropped = fit_rows(tiny_arch, [PAIRS[0]], chat, narrow, bos="prepend")
    assert (kept, dropped) == ([], 1)


def test_fit_rows_drops_an_empty_response_and_keeps_order(
    tiny_arch: LoadedModel,
) -> None:
    pairs = [PAIRS[2], (PAIRS[0][0], ""), PAIRS[0], (PAIRS[1][0], " ")]
    kept, dropped = fit_rows(tiny_arch, pairs, ChatFormat("plain"), _settings())
    assert kept == [PAIRS[2], PAIRS[0], (PAIRS[1][0], " ")]
    assert dropped == 1


def test_batch_layout_matches_the_spec_row_by_row(
    tiny_arch: LoadedModel,
) -> None:
    chat = ChatFormat("plain")
    settings = _settings()
    pairs = PAIRS[:2]
    batch = build_batch(tiny_arch, pairs, chat, settings)
    users = [
        [instruction for instruction, _ in pairs],
        [f"{i} {settings.pos_control}" for i, _ in pairs],
        [f"{i} {settings.neg_control}" for i, _ in pairs],
    ]
    expected = [
        _row(tiny_arch, chat, settings, user, response)
        for stream in users
        for user, (_, response) in zip(stream, pairs, strict=True)
    ]
    assert batch.input_ids.tolist() == [ids for ids, _ in expected]
    assert batch.attention_mask.tolist() == [mask for _, mask in expected]


def test_batch_shapes_dtypes_and_window(tiny_arch: LoadedModel) -> None:
    settings = _settings()
    batch = build_batch(tiny_arch, PAIRS[:2], ChatFormat("plain"), settings)
    width = settings.max_prompt_tokens + settings.max_response_tokens
    for tensor in (batch.input_ids, batch.attention_mask, batch.position_ids):
        assert tensor.shape == (6, width)
        assert tensor.dtype == torch.int64
    assert batch.window_start == settings.max_prompt_tokens - 1
    assert batch.response_mask.shape == (2, settings.max_response_tokens + 1)
    assert batch.response_mask.dtype == torch.float32


def test_batch_streams_share_the_response_columns(
    tiny_arch: LoadedModel,
) -> None:
    settings = _settings()
    batch = build_batch(tiny_arch, PAIRS[:3], ChatFormat("plain"), settings)
    window = slice(
        settings.max_prompt_tokens,
        settings.max_prompt_tokens + settings.max_response_tokens,
    )
    responses = batch.input_ids[:, window]
    assert torch.equal(responses[0:3], responses[3:6])
    assert torch.equal(responses[0:3], responses[6:9])
    columns = batch.attention_mask[:, window]
    assert torch.equal(columns[0:3], columns[3:6])
    assert torch.equal(columns[0:3], columns[6:9])


def test_batch_prompts_differ_only_in_the_controls(
    tiny_arch: LoadedModel,
) -> None:
    chat = ChatFormat("plain")
    settings = _settings()
    instruction = PAIRS[0][0]
    batch = build_batch(tiny_arch, [PAIRS[0]], chat, settings)
    prompt_window = slice(0, settings.max_prompt_tokens)
    expected = [
        chat.render_turn(instruction)[0],
        chat.render_turn(f"{instruction} Give a truthful answer.")[0],
        chat.render_turn(f"{instruction} Give an untruthful answer.")[0],
    ]
    for row, text in enumerate(expected):
        real = batch.attention_mask[row, prompt_window].bool()
        tokens = batch.input_ids[row, prompt_window][real].tolist()
        assert tokens[0] == tiny_arch.bos_id
        assert tiny_arch.tokenizer.decode(tokens[1:]) == text


def test_batch_pads_prompts_on_the_left_and_responses_on_the_right(
    tiny_arch: LoadedModel,
) -> None:
    settings = _settings()
    pairs = [PAIRS[0], (PAIRS[1][0], "Hi")]
    batch = build_batch(tiny_arch, pairs, ChatFormat("plain"), settings)
    p = settings.max_prompt_tokens
    n_response = len(_ids(tiny_arch, "Hi"))
    assert batch.attention_mask[1, p - 1].item() == 1
    assert batch.attention_mask[1, 0].item() == 0
    assert batch.attention_mask[1, p + n_response - 1].item() == 1
    assert batch.attention_mask[1, p + n_response :].sum().item() == 0
    assert (batch.input_ids[1, p + n_response :] == tiny_arch.pad_id).all()


def test_batch_position_ids_count_real_tokens(tiny_arch: LoadedModel) -> None:
    settings = _settings()
    batch = build_batch(tiny_arch, PAIRS[:2], ChatFormat("plain"), settings)
    mask = batch.attention_mask
    expected = (mask.cumsum(dim=-1) - 1).clamp(min=0)
    assert torch.equal(batch.position_ids, expected)
    first = int(mask[0].argmax())
    assert batch.position_ids[0, first].item() == 0
    assert batch.position_ids[0, first + 3].item() == 3
    assert batch.position_ids[0, :first].abs().sum().item() == 0


def test_batch_cuts_the_response_at_the_window(tiny_arch: LoadedModel) -> None:
    settings = _settings(max_response_tokens=4)
    batch = build_batch(tiny_arch, [PAIRS[2]], ChatFormat("plain"), settings)
    p = settings.max_prompt_tokens
    full = _ids(tiny_arch, PAIRS[2][1])
    assert len(full) > 4
    assert batch.input_ids.shape == (3, p + 4)
    assert batch.input_ids[0, p:].tolist() == full[:4]
    assert batch.attention_mask[0, p:].tolist() == [1, 1, 1, 1]
    assert batch.response_mask.tolist() == [[1.0] * 5]


def test_batch_response_mask_marks_the_last_prompt_column_and_real_tokens(
    tiny_arch: LoadedModel,
) -> None:
    settings = _settings()
    pairs = [PAIRS[1], (PAIRS[0][0], "Hi")]
    batch = build_batch(tiny_arch, pairs, ChatFormat("plain"), settings)
    r = settings.max_response_tokens
    first = len(_ids(tiny_arch, PAIRS[1][1]))
    second = len(_ids(tiny_arch, "Hi"))
    assert second < first < r
    assert batch.response_mask[0].tolist() == (
        [1.0] * (1 + first) + [0.0] * (r - first)
    )
    assert batch.response_mask[1].tolist() == (
        [1.0] * (1 + second) + [0.0] * (r - second)
    )


def test_batch_prepends_bos_by_default_and_not_when_asked(
    tiny_arch: LoadedModel,
) -> None:
    chat = ChatFormat("plain")
    settings = _settings()
    assert tiny_arch.default_bos == "prepend"
    p = settings.max_prompt_tokens
    plain_prompt = _ids(tiny_arch, chat.render_turn(PAIRS[0][0])[0])
    default = build_batch(tiny_arch, [PAIRS[0]], chat, settings)
    start = p - (len(plain_prompt) + 1)
    assert default.attention_mask[0, start].item() == 1
    assert default.attention_mask[0, start - 1].item() == 0
    assert default.input_ids[0, start].item() == tiny_arch.bos_id
    assert default.input_ids[0, start + 1 : p].tolist() == plain_prompt
    bare = build_batch(tiny_arch, [PAIRS[0]], chat, settings, bos="none")
    start = p - len(plain_prompt)
    assert bare.attention_mask[0, start - 1].item() == 0
    assert bare.input_ids[0, start:p].tolist() == plain_prompt


def test_batch_with_a_template_chat_carries_its_own_special_tokens(
    tiny_qwen3: LoadedModel,
) -> None:
    chat = ChatFormat.from_tokenizer(tiny_qwen3.tokenizer)
    settings = _settings()
    batch = build_batch(tiny_qwen3, PAIRS[:1], chat, settings, bos="none")
    expected, _ = _row(
        tiny_qwen3, chat, settings, PAIRS[0][0], PAIRS[0][1], "none"
    )
    assert batch.input_ids[0].tolist() == expected
    opener = tiny_qwen3.tokenizer.convert_tokens_to_ids("<|im_start|>")
    assert opener in batch.input_ids[0].tolist()


def test_batch_rejects_no_pairs(tiny_arch: LoadedModel) -> None:
    with pytest.raises(ValueError, match="no pairs"):
        build_batch(tiny_arch, [], ChatFormat("plain"), _settings())


def test_batch_rejects_a_response_without_tokens(
    tiny_arch: LoadedModel,
) -> None:
    with pytest.raises(ValueError, match="response"):
        build_batch(
            tiny_arch, [(PAIRS[0][0], "")], ChatFormat("plain"), _settings()
        )


def test_batch_rejects_a_prompt_longer_than_the_window(
    tiny_arch: LoadedModel,
) -> None:
    narrow = _settings(max_prompt_tokens=20)
    with pytest.raises(ValueError, match="prompt"):
        build_batch(tiny_arch, PAIRS[:1], ChatFormat("plain"), narrow)


def test_batch_rejects_bos_prepend_without_a_bos_id(
    tiny_arch: LoadedModel,
) -> None:
    headless = dataclasses.replace(tiny_arch, bos_id=None)
    with pytest.raises(ValueError, match="bos"):
        build_batch(
            headless, PAIRS[:1], ChatFormat("plain"), _settings(), bos="prepend"
        )


def test_batch_is_a_frozen_value() -> None:
    batch = LorraBatch(
        input_ids=torch.zeros(3, 4, dtype=torch.int64),
        attention_mask=torch.ones(3, 4, dtype=torch.int64),
        position_ids=torch.zeros(3, 4, dtype=torch.int64),
        window_start=1,
        response_mask=torch.ones(1, 3),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        batch.window_start = 2  # type: ignore[misc]


N_PAIRS, PROMPT, RESPONSE, DIM = 2, 3, 2, 4


def _loss_batch() -> LorraBatch:
    """A batch of two pairs where the second has one padded response slot."""
    width = PROMPT + RESPONSE
    return LorraBatch(
        input_ids=torch.zeros(3 * N_PAIRS, width, dtype=torch.int64),
        attention_mask=torch.ones(3 * N_PAIRS, width, dtype=torch.int64),
        position_ids=torch.zeros(3 * N_PAIRS, width, dtype=torch.int64),
        window_start=PROMPT - 1,
        response_mask=torch.tensor(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]], dtype=torch.float32
        ),
    )


def _random_residuals(
    seed: int, rows: int, dtype: torch.dtype = torch.float32
) -> dict[int, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        layer: torch.randn(
            rows, PROMPT + RESPONSE, DIM, generator=generator
        ).to(dtype)
        for layer in (1, 2)
    }


def _constant_residuals(
    orig: float, pos: float, neg: float
) -> dict[int, torch.Tensor]:
    """Frozen residuals that are constant within each of the three streams."""
    column = torch.tensor([orig, pos, neg]).repeat_interleave(N_PAIRS)
    stream = column[:, None, None].expand(-1, PROMPT + RESPONSE, DIM)
    return {1: stream.clone(), 2: stream.clone()}


def _target(
    frozen: torch.Tensor, batch: LorraBatch, alpha: float
) -> torch.Tensor:
    """The target of the spec, spelled out on the full-width tensor."""
    n = N_PAIRS
    full = frozen[:n] + alpha * (frozen[n : 2 * n] - frozen[2 * n :])
    mask = torch.zeros(n, PROMPT + RESPONSE, 1)
    mask[:, batch.window_start :, 0] = batch.response_mask
    return full * mask


@pytest.mark.parametrize("form", ["mean_norm", "sum_squared"])
def test_layer_losses_are_zero_at_the_target(form: str) -> None:
    batch = _loss_batch()
    settings = _settings(loss=form)
    frozen = _random_residuals(0, 3 * N_PAIRS)
    current = {
        layer: _target(frozen[layer], batch, settings.alpha) for layer in frozen
    }
    losses = layer_losses(current, frozen, batch, settings)
    assert sorted(losses) == [1, 2]
    for value in losses.values():
        assert value.shape == ()
        assert value.item() == pytest.approx(0.0, abs=1e-6)
    assert lorra_loss(current, frozen, batch, settings).item() == pytest.approx(
        0.0, abs=1e-6
    )


def test_mean_norm_averages_over_every_slot_with_pads_counting_zero() -> None:
    batch = _loss_batch()
    settings = _settings(loss="mean_norm")
    frozen = _constant_residuals(0.0, 0.0, 0.0)
    shift = torch.tensor([3.0, 4.0, 0.0, 0.0])
    current = {layer: frozen[layer][:N_PAIRS] + shift for layer in frozen}
    losses = layer_losses(current, frozen, batch, settings)
    # Five real slots of norm 5 over the six slots of the mask.
    assert losses[1].item() == pytest.approx(25.0 / 6.0)
    assert losses[2].item() == pytest.approx(25.0 / 6.0)


def test_sum_squared_sums_tokens_and_averages_pairs() -> None:
    batch = _loss_batch()
    settings = _settings(loss="sum_squared")
    frozen = _constant_residuals(0.0, 0.0, 0.0)
    shift = torch.tensor([3.0, 4.0, 0.0, 0.0])
    current = {layer: frozen[layer][:N_PAIRS] + shift for layer in frozen}
    losses = layer_losses(current, frozen, batch, settings)
    # Squared norm 25 on 3 slots (first pair) and on 2 slots (second pair).
    assert losses[1].item() == pytest.approx((75.0 + 50.0) / 2.0)


def test_alpha_scales_the_contrast_in_the_target() -> None:
    batch = _loss_batch()
    frozen = _constant_residuals(0.0, 1.0, -1.0)
    current = {
        layer: torch.zeros(N_PAIRS, PROMPT + RESPONSE, DIM) for layer in (1, 2)
    }
    for alpha in (0.0, 1.0, 5.0):
        settings = _settings(alpha=alpha, loss="mean_norm")
        losses = layer_losses(current, frozen, batch, settings)
        # The target is alpha * (pos - neg) = 2 alpha on each of 4 features.
        expected = 5 * (2.0 * alpha * 2.0) / 6.0
        assert losses[1].item() == pytest.approx(expected, abs=1e-5)


def test_positions_before_the_window_do_not_matter() -> None:
    batch = _loss_batch()
    settings = _settings()
    frozen = _random_residuals(1, 3 * N_PAIRS)
    current = _random_residuals(2, N_PAIRS)
    before = lorra_loss(current, frozen, batch, settings)
    for tensor in current.values():
        tensor[:, : batch.window_start] += 100.0
    for tensor in frozen.values():
        tensor[:, : batch.window_start] += 100.0
    assert lorra_loss(current, frozen, batch, settings).item() == pytest.approx(
        before.item()
    )


def test_lorra_loss_is_the_sum_of_the_layer_losses() -> None:
    batch = _loss_batch()
    settings = _settings()
    frozen = _random_residuals(3, 3 * N_PAIRS)
    current = _random_residuals(4, N_PAIRS)
    losses = layer_losses(current, frozen, batch, settings)
    assert losses[1].item() != pytest.approx(losses[2].item())
    total = lorra_loss(current, frozen, batch, settings)
    assert total.shape == ()
    assert total.item() == pytest.approx(losses[1].item() + losses[2].item())


def test_only_the_target_layers_are_scored() -> None:
    batch = _loss_batch()
    settings = _settings(layers=(2,))
    frozen = _random_residuals(5, 3 * N_PAIRS)
    current = _random_residuals(6, N_PAIRS)
    assert sorted(layer_losses(current, frozen, batch, settings)) == [2]


@pytest.mark.parametrize("form", ["mean_norm", "sum_squared"])
def test_gradients_are_finite_with_an_exactly_zero_difference(
    form: str,
) -> None:
    batch = _loss_batch()
    settings = _settings(loss=form)
    frozen = _random_residuals(7, 3 * N_PAIRS)
    current = {
        layer: _target(frozen[layer], batch, settings.alpha).requires_grad_()
        for layer in frozen
    }
    lorra_loss(current, frozen, batch, settings).backward()
    for tensor in current.values():
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum().item() == 0.0


@pytest.mark.parametrize("form", ["mean_norm", "sum_squared"])
def test_gradients_are_finite_and_zero_at_the_pad_slot(form: str) -> None:
    batch = _loss_batch()
    settings = _settings(loss=form)
    frozen = _random_residuals(8, 3 * N_PAIRS)
    current = {
        layer: tensor.requires_grad_()
        for layer, tensor in _random_residuals(9, N_PAIRS).items()
    }
    lorra_loss(current, frozen, batch, settings).backward()
    for tensor in current.values():
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum().item() > 0.0
        assert tensor.grad[1, -1].abs().sum().item() == 0.0
        assert tensor.grad[:, : batch.window_start].abs().sum().item() == 0.0


def test_losses_are_float32_for_half_precision_residuals() -> None:
    batch = _loss_batch()
    settings = _settings()
    frozen = _random_residuals(10, 3 * N_PAIRS, torch.bfloat16)
    current = _random_residuals(11, N_PAIRS, torch.bfloat16)
    for value in layer_losses(current, frozen, batch, settings).values():
        assert value.dtype == torch.float32
        assert torch.isfinite(value)


def test_layer_losses_reject_a_missing_layer() -> None:
    batch = _loss_batch()
    settings = _settings()
    frozen = _random_residuals(12, 3 * N_PAIRS)
    current = _random_residuals(13, N_PAIRS)
    del current[2]
    with pytest.raises(ValueError, match="layer 2"):
        layer_losses(current, frozen, batch, settings)
    del frozen[1]
    with pytest.raises(ValueError, match="layer 1"):
        layer_losses(_random_residuals(13, N_PAIRS), frozen, batch, settings)


def test_layer_losses_reject_rows_that_do_not_fit_the_batch() -> None:
    batch = _loss_batch()
    settings = _settings()
    with pytest.raises(ValueError, match="rows"):
        layer_losses(
            _random_residuals(14, N_PAIRS + 1),
            _random_residuals(15, 3 * N_PAIRS),
            batch,
            settings,
        )
    with pytest.raises(ValueError, match="rows"):
        layer_losses(
            _random_residuals(14, N_PAIRS),
            _random_residuals(15, N_PAIRS),
            batch,
            settings,
        )


FORWARD_IDS = torch.tensor([[5, 6, 7, 8, 9, 10, 11, 12]])
LORA_BLOCK = re.compile(r"\.(?:h|layers)\.(\d+)\.")
TRAINABLE = {"gpt2": 2048, "qwen3": 1792}


def _capture(loaded: LoadedModel) -> dict[int, torch.Tensor]:
    """The residual stream at layers 0..L for a fixed forward pass."""
    layers = list(range(loaded.n_layers + 1))
    with torch.no_grad(), capture_residuals(loaded, layers) as captured:
        loaded.model(FORWARD_IDS)
    return dict(captured)


def _logits(loaded: LoadedModel) -> torch.Tensor:
    with torch.no_grad():
        return loaded.model(FORWARD_IDS).logits.detach().clone()


def _randomize(adapter: LorraAdapter) -> None:
    """Gives the adapter a nonzero effect by filling `lora_B`."""
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, parameter in adapter.peft_model.named_parameters():
            if "lora_B" in name:
                noise = torch.randn(parameter.shape, generator=generator)
                parameter.copy_(0.5 * noise)


def _lora_names(loaded: LoadedModel) -> list[str]:
    return [
        name
        for name, module in loaded.model.named_modules()
        if isinstance(module, LoraLayer)
    ]


def _adapted_blocks(loaded: LoadedModel) -> set[int]:
    found = set()
    for name in _lora_names(loaded):
        match = LORA_BLOCK.search(name)
        assert match is not None
        found.add(int(match.group(1)))
    return found


def test_attach_adapts_the_blocks_below_the_deepest_target_layer(
    tiny_arch: LoadedModel,
) -> None:
    adapter = attach_adapter(tiny_arch, _settings(layers=(1,)))
    assert adapter.blocks == (0,)
    assert _adapted_blocks(tiny_arch) == {0}


def test_attach_covers_every_block_for_the_last_layer(
    tiny_arch: LoadedModel,
) -> None:
    settings = _settings(layers=(1, 3))
    adapter = attach_adapter(tiny_arch, settings)
    assert adapter.blocks == (0, 1, 2)
    assert adapter.settings is settings
    assert isinstance(adapter.peft_model, PeftModel)
    assert _adapted_blocks(tiny_arch) == {0, 1, 2}


def test_attach_targets_the_architecture_modules(
    tiny_arch: LoadedModel,
) -> None:
    attach_adapter(tiny_arch, _settings())
    leaves = {name.rpartition(".")[2] for name in _lora_names(tiny_arch)}
    assert leaves == set(tiny_arch.spec.lora_targets)


def test_attach_counts_the_trainable_parameters(tiny_arch: LoadedModel) -> None:
    adapter = attach_adapter(tiny_arch, _settings(layers=(1, 2)))
    assert adapter.n_trainable() == TRAINABLE[tiny_arch.spec.model_type]
    assert sum(p.numel() for p in adapter.trainable()) == adapter.n_trainable()


def test_only_the_adapter_is_trainable(tiny_arch: LoadedModel) -> None:
    adapter = attach_adapter(tiny_arch, _settings())
    trainable = {id(p) for p in adapter.trainable()}
    for name, parameter in tiny_arch.model.named_parameters():
        assert parameter.requires_grad == (id(parameter) in trainable)
        assert parameter.requires_grad == ("lora_" in name)


def test_a_fresh_adapter_is_bit_identical_to_the_base(
    tiny_arch: LoadedModel,
) -> None:
    before = _capture(tiny_arch)
    adapter = attach_adapter(tiny_arch, _settings())
    after = _capture(tiny_arch)
    assert sorted(after) == sorted(before)
    for layer in before:
        assert torch.equal(after[layer], before[layer])
    with adapter.disabled():
        inside = _capture(tiny_arch)
    for layer in before:
        assert torch.equal(inside[layer], before[layer])


def _lora_a(adapter: LorraAdapter) -> list[torch.Tensor]:
    return [
        parameter.detach().clone()
        for name, parameter in adapter.peft_model.named_parameters()
        if "lora_A" in name
    ]


def test_attach_initializes_the_adapter_from_the_seed(
    tiny_arch: LoadedModel,
) -> None:
    twins = [copy.deepcopy(tiny_arch) for _ in range(3)]
    first, second, other = (
        attach_adapter(twins[0], _settings(seed=3)),
        attach_adapter(twins[1], _settings(seed=3)),
        attach_adapter(twins[2], _settings(seed=4)),
    )
    for a, b in zip(_lora_a(first), _lora_a(second), strict=True):
        assert torch.equal(a, b)
    assert any(
        not torch.equal(a, b)
        for a, b in zip(_lora_a(first), _lora_a(other), strict=True)
    )


def test_attach_leaves_the_global_random_state_alone(
    tiny_arch: LoadedModel,
) -> None:
    torch.manual_seed(321)
    expected = torch.rand(3)
    torch.manual_seed(321)
    attach_adapter(tiny_arch, _settings())
    assert torch.equal(torch.rand(3), expected)


def test_attach_leaves_the_model_in_eval_mode(tiny_arch: LoadedModel) -> None:
    attach_adapter(tiny_arch, _settings())
    assert not tiny_arch.model.training
    assert not any(module.training for module in tiny_arch.model.modules())


def test_the_adapter_is_float32_on_a_bf16_base(tiny_arch: LoadedModel) -> None:
    tiny_arch.model.to(torch.bfloat16)
    before = _capture(tiny_arch)
    attach_adapter(tiny_arch, _settings())
    for name, parameter in tiny_arch.model.named_parameters():
        expected = torch.float32 if "lora_" in name else torch.bfloat16
        assert parameter.dtype == expected
    after = _capture(tiny_arch)
    for layer in before:
        assert after[layer].dtype == torch.bfloat16
        assert torch.equal(after[layer], before[layer])


def test_attach_rejects_a_layer_above_the_model_depth(
    tiny_arch: LoadedModel,
) -> None:
    with pytest.raises(ValueError, match="layer 4"):
        attach_adapter(tiny_arch, _settings(layers=(1, 4)))
    assert _lora_names(tiny_arch) == []


def test_checkpointing_is_refused_for_a_base_with_active_dropout(
    tiny_gpt2: LoadedModel,
) -> None:
    with pytest.raises(ValueError, match="dropout"):
        attach_adapter(tiny_gpt2, _settings(gradient_checkpointing=True))
    assert _lora_names(tiny_gpt2) == []


def test_checkpointing_is_enabled_for_qwen3(
    tiny_qwen3: LoadedModel,
) -> None:
    assert not tiny_qwen3.model.is_gradient_checkpointing
    attach_adapter(tiny_qwen3, _settings(gradient_checkpointing=True))
    assert tiny_qwen3.model.is_gradient_checkpointing
    assert not tiny_qwen3.model.training


def test_checkpointing_stays_off_by_default(tiny_arch: LoadedModel) -> None:
    attach_adapter(tiny_arch, _settings())
    assert not tiny_arch.model.is_gradient_checkpointing


def test_disabled_runs_the_base_model_and_restores_the_adapter(
    tiny_arch: LoadedModel,
) -> None:
    base = _logits(tiny_arch)
    adapter = attach_adapter(tiny_arch, _settings())
    _randomize(adapter)
    adapted = _logits(tiny_arch)
    assert not torch.equal(adapted, base)
    with adapter.disabled():
        assert torch.equal(_logits(tiny_arch), base)
    assert torch.equal(_logits(tiny_arch), adapted)


def test_disabled_restores_the_adapter_after_an_exception(
    tiny_arch: LoadedModel,
) -> None:
    adapter = attach_adapter(tiny_arch, _settings())
    _randomize(adapter)
    adapted = _logits(tiny_arch)
    with pytest.raises(RuntimeError, match="boom"), adapter.disabled():
        raise RuntimeError("boom")
    assert torch.equal(_logits(tiny_arch), adapted)


def test_gated_off_is_the_base_and_on_is_the_adapter(
    tiny_arch: LoadedModel,
) -> None:
    base = _logits(tiny_arch)
    adapter = attach_adapter(tiny_arch, _settings())
    _randomize(adapter)
    adapted = _logits(tiny_arch)

    def run() -> torch.Tensor:
        return _logits(tiny_arch)

    assert torch.equal(adapter.gated(run, enabled=False)(), base)
    assert torch.equal(adapter.gated(run, enabled=True)(), adapted)
    assert torch.equal(run(), adapted)


def test_gated_passes_arguments_and_keeps_the_name(
    tiny_arch: LoadedModel,
) -> None:
    adapter = attach_adapter(tiny_arch, _settings())

    def add(a: int, *, b: int) -> int:
        return a + b

    for enabled in (True, False):
        wrapped = adapter.gated(add, enabled=enabled)
        assert wrapped(2, b=3) == 5
        assert wrapped.__name__ == "add"


def test_gated_restores_the_model_state_on_exceptions(
    tiny_arch: LoadedModel,
) -> None:
    adapter = attach_adapter(tiny_arch, _settings())
    _randomize(adapter)
    adapted = _logits(tiny_arch)

    def fail() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        adapter.gated(fail, enabled=False)()
    assert torch.equal(_logits(tiny_arch), adapted)


def test_training_mode_drops_only_the_adapter_dropout(
    tiny_arch: LoadedModel,
) -> None:
    adapter = attach_adapter(tiny_arch, _settings(dropout=0.5))
    with adapter.training():
        for name, module in tiny_arch.model.named_modules():
            assert module.training == ("lora_dropout" in name), name
    assert not any(module.training for module in tiny_arch.model.modules())


def test_training_mode_does_not_perturb_the_base(
    tiny_arch: LoadedModel,
) -> None:
    base = _logits(tiny_arch)
    adapter = attach_adapter(tiny_arch, _settings(dropout=0.5))
    with adapter.training():
        assert torch.equal(_logits(tiny_arch), base)


def test_training_mode_applies_the_adapter_dropout(
    tiny_arch: LoadedModel,
) -> None:
    adapter = attach_adapter(tiny_arch, _settings(dropout=0.5))
    _randomize(adapter)
    with adapter.training():
        first, second = _logits(tiny_arch), _logits(tiny_arch)
    assert not torch.equal(first, second)
    assert torch.equal(_logits(tiny_arch), _logits(tiny_arch))


def test_training_mode_is_the_full_train_mode_with_checkpointing(
    tiny_qwen3: LoadedModel,
) -> None:
    adapter = attach_adapter(tiny_qwen3, _settings(gradient_checkpointing=True))
    with adapter.training():
        assert tiny_qwen3.model.training
    assert not any(module.training for module in tiny_qwen3.model.modules())


def test_training_mode_returns_to_eval_after_an_exception(
    tiny_arch: LoadedModel,
) -> None:
    adapter = attach_adapter(tiny_arch, _settings())
    with pytest.raises(RuntimeError, match="boom"), adapter.training():
        raise RuntimeError("boom")
    assert not any(module.training for module in tiny_arch.model.modules())


EIGHT_PAIRS: list[Pair] = [
    *PAIRS,
    ("Is the model right?", "The model reads the stream of the residual."),
    ("Are the answers true?", "Honest answers are true, and others false."),
]


REPEATED_PAIR: list[Pair] = [PAIRS[0]] * 8


def _train_settings(**overrides: Any) -> LorraSettings:
    defaults = {"lr": 3e-2, "steps": 60, "batch_size": 8}
    return _settings(**{**defaults, **overrides})


def _twin(loaded: LoadedModel) -> LoadedModel:
    """Copies a model before an adapter is attached to it."""
    return copy.deepcopy(loaded)


def _base_weights(loaded: LoadedModel) -> dict[str, torch.Tensor]:
    return {
        name.replace(".base_layer", ""): parameter.detach().clone()
        for name, parameter in loaded.model.named_parameters()
        if "lora_" not in name
    }


def _adapter_state(adapter: LorraAdapter) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in adapter.trainable()]


def test_step_loss_has_a_graph_that_reaches_only_the_adapter(
    tiny_arch: LoadedModel,
) -> None:
    settings = _settings()
    adapter = attach_adapter(tiny_arch, settings)
    batch = build_batch(tiny_arch, PAIRS[:2], ChatFormat("plain"), settings)
    loss = step_loss(tiny_arch, adapter, batch)
    assert loss.shape == ()
    assert loss.requires_grad
    assert loss.item() > 0.0
    loss.backward()
    for name, parameter in tiny_arch.model.named_parameters():
        if "lora_B" in name:
            assert parameter.grad is not None
            assert parameter.grad.abs().sum().item() > 0.0
        elif "lora_" not in name:
            assert parameter.grad is None


def test_step_loss_scores_the_adapter_against_the_frozen_base(
    tiny_arch: LoadedModel,
) -> None:
    chat = ChatFormat("plain")
    settings = _settings(dropout=0.0)
    adapter = attach_adapter(tiny_arch, settings)
    _randomize(adapter)
    batch = build_batch(tiny_arch, PAIRS[:2], chat, settings)
    kwargs = {
        "attention_mask": batch.attention_mask,
        "position_ids": batch.position_ids,
        "use_cache": False,
    }
    layers = list(settings.layers)
    with (
        torch.no_grad(),
        adapter.disabled(),
        capture_residuals(tiny_arch, layers) as frozen,
    ):
        tiny_arch.model(batch.input_ids, **kwargs)
    with torch.no_grad(), capture_residuals(tiny_arch, layers) as current:
        tiny_arch.model(
            batch.input_ids[:2],
            attention_mask=batch.attention_mask[:2],
            position_ids=batch.position_ids[:2],
            use_cache=False,
        )
    expected = lorra_loss(current, frozen, batch, settings)
    assert step_loss(tiny_arch, adapter, batch).item() == pytest.approx(
        expected.item(), rel=1e-5
    )


def test_step_loss_runs_a_frozen_pass_over_all_rows_then_an_adapted_pass(
    tiny_arch: LoadedModel,
) -> None:
    settings = _settings()
    adapter = attach_adapter(tiny_arch, settings)
    batch = build_batch(tiny_arch, PAIRS[:2], ChatFormat("plain"), settings)
    calls: list[tuple[int, bool, bool]] = []
    target = next(
        module
        for module in tiny_arch.model.modules()
        if isinstance(module, LoraLayer)
    )

    def record(module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        hidden = args[0]
        calls.append(
            (hidden.shape[0], torch.is_grad_enabled(), target.disable_adapters)
        )

    handle = tiny_arch.blocks[0].register_forward_pre_hook(record)
    try:
        step_loss(tiny_arch, adapter, batch)
    finally:
        handle.remove()
    assert calls == [(6, False, True), (2, True, False)]


def test_train_lorra_lowers_the_loss(tiny_arch: LoadedModel) -> None:
    settings = _train_settings()
    adapter = attach_adapter(tiny_arch, settings)
    trace = train_lorra(tiny_arch, adapter, REPEATED_PAIR, ChatFormat("plain"))
    assert len(trace.losses) == settings.steps
    assert all(math.isfinite(loss) for loss in trace.losses)
    assert trace.losses[-1] <= 0.8 * trace.losses[0]
    assert trace.peak_memory_mb is None
    assert not any(module.training for module in tiny_arch.model.modules())


def test_train_lorra_lowers_the_sum_squared_loss(
    tiny_arch: LoadedModel,
) -> None:
    settings = _train_settings(loss="sum_squared", steps=30)
    adapter = attach_adapter(tiny_arch, settings)
    trace = train_lorra(tiny_arch, adapter, REPEATED_PAIR, ChatFormat("plain"))
    assert all(math.isfinite(loss) for loss in trace.losses)
    assert trace.losses[-1] < trace.losses[0]


def test_train_lorra_is_deterministic_under_the_seed(
    tiny_arch: LoadedModel,
) -> None:
    twin = _twin(tiny_arch)
    settings = _train_settings(steps=6, dropout=0.3)
    runs = []
    for loaded in (tiny_arch, twin):
        adapter = attach_adapter(loaded, settings)
        trace = train_lorra(loaded, adapter, EIGHT_PAIRS, ChatFormat("plain"))
        runs.append((trace.losses, _adapter_state(adapter)))
    assert runs[0][0] == runs[1][0]
    for a, b in zip(runs[0][1], runs[1][1], strict=True):
        assert torch.equal(a, b)


def test_train_lorra_depends_on_the_seed(tiny_arch: LoadedModel) -> None:
    twin = _twin(tiny_arch)
    traces = []
    for loaded, seed in ((tiny_arch, 0), (twin, 1)):
        settings = _train_settings(steps=4, seed=seed)
        adapter = attach_adapter(loaded, settings)
        traces.append(
            train_lorra(loaded, adapter, EIGHT_PAIRS, ChatFormat("plain"))
        )
    assert traces[0].losses != traces[1].losses


def test_train_lorra_dropout_does_not_depend_on_the_global_seed(
    tiny_arch: LoadedModel,
) -> None:
    twin = _twin(tiny_arch)
    settings = _train_settings(steps=6, dropout=0.3)
    runs = []
    for loaded, global_seed in ((tiny_arch, 11), (twin, 22)):
        adapter = attach_adapter(loaded, settings)
        torch.manual_seed(global_seed)
        runs.append(
            train_lorra(loaded, adapter, EIGHT_PAIRS, ChatFormat("plain"))
        )
    assert runs[0].losses == runs[1].losses


def test_train_lorra_applies_no_weight_decay(tiny_arch: LoadedModel) -> None:
    adapter = attach_adapter(tiny_arch, _train_settings(steps=1))
    before = _lora_a(adapter)
    train_lorra(tiny_arch, adapter, EIGHT_PAIRS, ChatFormat("plain"))
    for a, b in zip(before, _lora_a(adapter), strict=True):
        assert torch.equal(a, b)


def test_train_lorra_leaves_the_callers_random_state_alone(
    tiny_arch: LoadedModel,
) -> None:
    settings = _train_settings(steps=3)
    adapter = attach_adapter(tiny_arch, settings)
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    train_lorra(tiny_arch, adapter, EIGHT_PAIRS, ChatFormat("plain"))
    assert torch.equal(torch.rand(3), expected)


def test_train_lorra_seeds_no_cuda_generator_for_a_cpu_model(
    tiny_arch: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded: list[int] = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", seeded.append)
    monkeypatch.setattr(torch.cuda, "manual_seed", seeded.append)
    adapter = attach_adapter(tiny_arch, _train_settings(steps=2))
    train_lorra(tiny_arch, adapter, EIGHT_PAIRS, ChatFormat("plain"))
    assert seeded == []


def test_train_lorra_never_touches_the_base_weights(
    tiny_arch: LoadedModel,
) -> None:
    settings = _train_settings(steps=5)
    before = _base_weights(tiny_arch)
    adapter = attach_adapter(tiny_arch, settings)
    initial = _adapter_state(adapter)
    train_lorra(tiny_arch, adapter, EIGHT_PAIRS, ChatFormat("plain"))
    after = _base_weights(tiny_arch)
    assert after.keys() == before.keys()
    for name, weight in before.items():
        assert torch.equal(after[name], weight), name
    assert any(
        not torch.equal(a, b)
        for a, b in zip(initial, _adapter_state(adapter), strict=True)
    )


def test_train_lorra_reports_every_step(tiny_arch: LoadedModel) -> None:
    settings = _train_settings(steps=4)
    adapter = attach_adapter(tiny_arch, settings)
    seen: list[tuple[int, float]] = []
    trace = train_lorra(
        tiny_arch,
        adapter,
        EIGHT_PAIRS,
        ChatFormat("plain"),
        on_step=lambda step, loss: seen.append((step, loss)),
    )
    assert seen == list(enumerate(trace.losses))


def test_train_lorra_draws_batches_from_epochs_of_a_permutation(
    tiny_arch: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(batch_size=4, steps=3, seed=7)
    adapter = attach_adapter(tiny_arch, settings)
    drawn: list[list[Pair]] = []

    def spy(
        loaded: LoadedModel,
        pairs: Sequence[Pair],
        chat: ChatFormat,
        run: LorraSettings,
        *,
        bos: BosMode | None = None,
    ) -> LorraBatch:
        drawn.append(list(pairs))
        return build_batch(loaded, pairs, chat, run, bos=bos)

    monkeypatch.setattr(lorra, "build_batch", spy)
    train_lorra(tiny_arch, adapter, PAIRS, ChatFormat("plain"))
    rng = np.random.default_rng(7)
    order = [*rng.permutation(6).tolist(), *rng.permutation(6).tolist()]
    expected = [
        [PAIRS[i] for i in order[start : start + 4]] for start in (0, 4, 8)
    ]
    assert drawn == expected


def test_train_lorra_needs_a_full_batch(tiny_arch: LoadedModel) -> None:
    settings = _settings(batch_size=4)
    adapter = attach_adapter(tiny_arch, settings)
    with pytest.raises(ValueError, match="batch"):
        train_lorra(tiny_arch, adapter, PAIRS[:3], ChatFormat("plain"))


def _no_bos_window(loaded: LoadedModel, chat: ChatFormat) -> int:
    """The longest prompt stream of `REPEATED_PAIR` without a BOS token."""
    wide = _settings()
    batch = build_batch(loaded, REPEATED_PAIR[:1], chat, wide, bos="none")
    prompts = batch.attention_mask[:, : wide.max_prompt_tokens]
    return int(prompts.sum(dim=1).max().item())


def test_train_lorra_passes_the_bos_policy_to_every_batch(
    tiny_gpt2: LoadedModel,
) -> None:
    chat = ChatFormat("plain")
    settings = _train_settings(
        steps=2, max_prompt_tokens=_no_bos_window(tiny_gpt2, chat)
    )
    adapter = attach_adapter(tiny_gpt2, settings)
    with pytest.raises(ValueError, match="prompt"):
        train_lorra(tiny_gpt2, adapter, REPEATED_PAIR, chat)
    trace = train_lorra(tiny_gpt2, adapter, REPEATED_PAIR, chat, bos="none")
    assert all(math.isfinite(loss) for loss in trace.losses)


def test_train_lorra_gives_finite_losses_on_a_bf16_base(
    tiny_arch: LoadedModel,
) -> None:
    tiny_arch.model.to(torch.bfloat16)
    settings = _train_settings(steps=3)
    adapter = attach_adapter(tiny_arch, settings)
    trace = train_lorra(tiny_arch, adapter, EIGHT_PAIRS, ChatFormat("plain"))
    assert all(math.isfinite(loss) for loss in trace.losses)
    assert all(
        p.dtype == torch.float32 and torch.isfinite(p).all()
        for p in adapter.trainable()
    )


def test_checkpointing_gives_the_same_gradients_and_recomputes_two_blocks(
    tiny_qwen3: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    twin = _twin(tiny_qwen3)
    layer_class = type(tiny_qwen3.blocks[0])
    original = layer_class.forward
    calls = [0]

    def counting(self: torch.nn.Module, *args: Any, **kwargs: Any) -> Any:
        calls[0] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(layer_class, "forward", counting)
    grads: list[list[torch.Tensor]] = []
    counts: list[int] = []
    for loaded, checkpointing in ((tiny_qwen3, False), (twin, True)):
        settings = _settings(dropout=0.0, gradient_checkpointing=checkpointing)
        adapter = attach_adapter(loaded, settings)
        _randomize(adapter)
        batch = build_batch(loaded, PAIRS[:2], ChatFormat("plain"), settings)
        calls[0] = 0
        step_loss(loaded, adapter, batch).backward()
        counts.append(calls[0])
        grads.append(
            [p.grad.clone() for p in adapter.trainable() if p.grad is not None]
        )
    assert counts == [6, 8]
    assert len(grads[0]) == len(grads[1]) > 0
    for plain, checkpointed in zip(grads[0], grads[1], strict=True):
        assert torch.allclose(plain, checkpointed, atol=1e-5)
        assert plain.abs().sum().item() > 0.0


def test_heldout_losses_has_one_float_per_target_layer(
    tiny_arch: LoadedModel,
) -> None:
    adapter = attach_adapter(tiny_arch, _settings(layers=(1, 3)))
    losses = heldout_losses(tiny_arch, adapter, PAIRS, ChatFormat("plain"))
    assert sorted(losses) == [1, 3]
    assert all(isinstance(v, float) and v > 0.0 for v in losses.values())


def test_heldout_loss_before_training_is_the_loss_of_the_base_model(
    tiny_arch: LoadedModel,
) -> None:
    chat = ChatFormat("plain")
    settings = _settings(batch_size=3)
    batch = build_batch(tiny_arch, PAIRS[:3], chat, settings)
    with torch.no_grad(), capture_residuals(tiny_arch, [1, 2]) as base:
        tiny_arch.model(
            batch.input_ids,
            attention_mask=batch.attention_mask,
            position_ids=batch.position_ids,
            use_cache=False,
        )
    frozen = dict(base)
    current = {layer: tensor[:3] for layer, tensor in frozen.items()}
    expected = layer_losses(current, frozen, batch, settings)
    adapter = attach_adapter(tiny_arch, settings)
    got = heldout_losses(tiny_arch, adapter, PAIRS[:3], chat)
    for layer, value in expected.items():
        assert got[layer] == pytest.approx(value.item(), rel=1e-5)


def test_heldout_loss_falls_after_training(tiny_arch: LoadedModel) -> None:
    chat = ChatFormat("plain")
    settings = _train_settings(steps=40)
    adapter = attach_adapter(tiny_arch, settings)
    before = heldout_losses(tiny_arch, adapter, REPEATED_PAIR[:2], chat)
    train_lorra(tiny_arch, adapter, REPEATED_PAIR, chat)
    after = heldout_losses(tiny_arch, adapter, REPEATED_PAIR[:2], chat)
    assert sorted(after) == sorted(before)
    assert sum(after.values()) < sum(before.values())


def test_heldout_losses_use_no_dropout_and_no_gradients(
    tiny_arch: LoadedModel,
) -> None:
    adapter = attach_adapter(tiny_arch, _settings(dropout=0.5))
    _randomize(adapter)
    first = heldout_losses(tiny_arch, adapter, PAIRS, ChatFormat("plain"))
    second = heldout_losses(tiny_arch, adapter, PAIRS, ChatFormat("plain"))
    assert first == second
    assert all(parameter.grad is None for parameter in adapter.trainable())
    assert not any(module.training for module in tiny_arch.model.modules())


def _hand_layer_losses(
    loaded: LoadedModel, adapter: LorraAdapter, pairs: Sequence[Pair]
) -> dict[int, float]:
    """Scores one batch by hand: a frozen pass, an adapted pass, the losses."""
    settings = adapter.settings
    batch = build_batch(loaded, pairs, ChatFormat("plain"), settings)
    n = len(pairs)
    layers = list(settings.layers)
    with (
        torch.no_grad(),
        adapter.disabled(),
        capture_residuals(loaded, layers) as frozen,
    ):
        loaded.model(
            batch.input_ids,
            attention_mask=batch.attention_mask,
            position_ids=batch.position_ids,
            use_cache=False,
        )
    with torch.no_grad(), capture_residuals(loaded, layers) as current:
        loaded.model(
            batch.input_ids[:n],
            attention_mask=batch.attention_mask[:n],
            position_ids=batch.position_ids[:n],
            use_cache=False,
        )
    losses = layer_losses(current, frozen, batch, settings)
    return {layer: loss.item() for layer, loss in losses.items()}


def test_heldout_losses_average_over_batches_of_the_batch_size(
    tiny_arch: LoadedModel,
) -> None:
    adapter = attach_adapter(tiny_arch, _settings(batch_size=2))
    _randomize(adapter)
    pairs = PAIRS[:5]
    per_batch = [
        _hand_layer_losses(tiny_arch, adapter, pairs[start : start + 2])
        for start in (0, 2, 4)
    ]
    got = heldout_losses(tiny_arch, adapter, pairs, ChatFormat("plain"))
    assert sorted(got) == [1, 2]
    for layer, value in got.items():
        expected = sum(losses[layer] for losses in per_batch) / 3
        assert value == pytest.approx(expected, rel=1e-5)


def test_heldout_losses_run_every_forward_without_gradients(
    tiny_arch: LoadedModel,
) -> None:
    adapter = attach_adapter(tiny_arch, _settings(batch_size=2))
    _randomize(adapter)
    grad_modes: list[bool] = []

    def record(module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        grad_modes.append(torch.is_grad_enabled())

    handle = tiny_arch.blocks[0].register_forward_pre_hook(record)
    try:
        heldout_losses(tiny_arch, adapter, PAIRS[:3], ChatFormat("plain"))
    finally:
        handle.remove()
    assert grad_modes == [False] * 4


def test_heldout_losses_pass_the_bos_policy_to_every_batch(
    tiny_gpt2: LoadedModel,
) -> None:
    chat = ChatFormat("plain")
    settings = _settings(
        batch_size=2, max_prompt_tokens=_no_bos_window(tiny_gpt2, chat)
    )
    adapter = attach_adapter(tiny_gpt2, settings)
    with pytest.raises(ValueError, match="prompt"):
        heldout_losses(tiny_gpt2, adapter, REPEATED_PAIR[:3], chat)
    losses = heldout_losses(
        tiny_gpt2, adapter, REPEATED_PAIR[:3], chat, bos="none"
    )
    assert all(math.isfinite(loss) for loss in losses.values())


def test_heldout_losses_keep_the_adapter_enabled(
    tiny_arch: LoadedModel,
) -> None:
    adapter = attach_adapter(tiny_arch, _settings())
    _randomize(adapter)
    adapted = _logits(tiny_arch)
    heldout_losses(tiny_arch, adapter, PAIRS, ChatFormat("plain"))
    assert torch.equal(_logits(tiny_arch), adapted)


def test_heldout_losses_reject_no_pairs(tiny_arch: LoadedModel) -> None:
    adapter = attach_adapter(tiny_arch, _settings())
    with pytest.raises(ValueError, match="no pairs"):
        heldout_losses(tiny_arch, adapter, [], ChatFormat("plain"))


def test_state_is_the_float32_lora_weights_by_name(
    tiny_arch: LoadedModel,
) -> None:
    adapter = attach_adapter(tiny_arch, _settings(layers=(1, 2)))
    state = adapter.state()
    assert len(state) == 2 * len(_lora_names(tiny_arch))
    assert all(
        key.endswith(("lora_A.weight", "lora_B.weight")) for key in state
    )
    assert all(
        isinstance(array, np.ndarray) and array.dtype == np.float32
        for array in state.values()
    )
    assert sum(array.size for array in state.values()) == adapter.n_trainable()


def test_state_is_float32_on_a_bf16_base(tiny_arch: LoadedModel) -> None:
    tiny_arch.model.to(torch.bfloat16)
    adapter = attach_adapter(tiny_arch, _settings())
    assert all(array.dtype == np.float32 for array in adapter.state().values())


def test_state_is_a_copy(tiny_arch: LoadedModel) -> None:
    adapter = attach_adapter(tiny_arch, _settings())
    _randomize(adapter)
    state = adapter.state()
    snapshot = {key: array.copy() for key, array in state.items()}
    for array in state.values():
        array += 1.0
    again = adapter.state()
    for key, array in again.items():
        np.testing.assert_array_equal(array, snapshot[key])
    with torch.no_grad():
        for parameter in adapter.trainable():
            parameter.add_(1.0)
    for key, array in adapter.state().items():
        assert not np.array_equal(array, snapshot[key])


def test_load_state_round_trips_into_a_fresh_model(
    tiny_arch: LoadedModel,
) -> None:
    fresh = _twin(tiny_arch)
    chat = ChatFormat("plain")
    settings = _train_settings(steps=5, seed=2)
    trained = attach_adapter(tiny_arch, settings)
    train_lorra(tiny_arch, trained, EIGHT_PAIRS, chat)
    state = trained.state()
    other = attach_adapter(fresh, dataclasses.replace(settings, seed=9))
    assert not torch.equal(_logits(fresh), _logits(tiny_arch))
    other.load_state(state)
    assert torch.equal(_logits(fresh), _logits(tiny_arch))
    for key, array in other.state().items():
        np.testing.assert_array_equal(array, state[key])


def test_load_state_casts_to_the_parameter_dtype(
    tiny_arch: LoadedModel,
) -> None:
    adapter = attach_adapter(tiny_arch, _settings())
    wide = {
        key: np.full(array.shape, 0.25, dtype=np.float64)
        for key, array in adapter.state().items()
    }
    adapter.load_state(wide)
    assert all(p.dtype == torch.float32 for p in adapter.trainable())
    assert all(bool((p == 0.25).all().item()) for p in adapter.trainable())


def test_load_state_leaves_the_base_weights_alone(
    tiny_arch: LoadedModel,
) -> None:
    adapter = attach_adapter(tiny_arch, _settings())
    before = _base_weights(tiny_arch)
    adapter.load_state(
        {key: array + 1.0 for key, array in adapter.state().items()}
    )
    after = _base_weights(tiny_arch)
    for name, weight in before.items():
        assert torch.equal(after[name], weight), name


def test_load_state_rejects_a_missing_key(tiny_arch: LoadedModel) -> None:
    adapter = attach_adapter(tiny_arch, _settings())
    state = adapter.state()
    dropped = sorted(state)[0]
    del state[dropped]
    with pytest.raises(ValueError, match="lora_A") as caught:
        adapter.load_state(state)
    assert dropped in str(caught.value)


def test_load_state_rejects_a_foreign_key(tiny_arch: LoadedModel) -> None:
    adapter = attach_adapter(tiny_arch, _settings())
    state = adapter.state()
    state["base_model.model.foreign.lora_A.weight"] = np.zeros(
        (8, 4), dtype=np.float32
    )
    with pytest.raises(ValueError, match="foreign"):
        adapter.load_state(state)


def test_load_state_rejects_a_state_of_a_deeper_adapter(
    tiny_arch: LoadedModel,
) -> None:
    deep = attach_adapter(_twin(tiny_arch), _settings(layers=(1, 3)))
    shallow = attach_adapter(tiny_arch, _settings(layers=(1, 2)))
    with pytest.raises(ValueError, match="keys"):
        shallow.load_state(deep.state())


def test_load_state_rejects_a_shape_mismatch(tiny_arch: LoadedModel) -> None:
    adapter = attach_adapter(tiny_arch, _settings())
    state = adapter.state()
    key = sorted(state)[0]
    state[key] = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="shape") as caught:
        adapter.load_state(state)
    assert key in str(caught.value)


def test_load_state_does_not_partially_apply_a_bad_state(
    tiny_arch: LoadedModel,
) -> None:
    adapter = attach_adapter(tiny_arch, _settings())
    before = adapter.state()
    state = {key: array + 1.0 for key, array in before.items()}
    state[sorted(state)[-1]] = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        adapter.load_state(state)
    for key, array in adapter.state().items():
        np.testing.assert_array_equal(array, before[key])


def test_state_round_trip_never_looks_up_the_base_model(
    tiny_arch: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    tiny_arch.model.name_or_path = "org/repo"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    adapter = attach_adapter(tiny_arch, _settings())
    adapter.load_state(adapter.state())


def _alpaca_pairs(n: int) -> list[Pair]:
    train = [e for e in load_alpaca(offline=True) if e.split == "train"]
    return [(str(e.meta["instruction"]), e.text) for e in train[:n]]


def _load_post_trained_qwen3() -> LoadedModel:
    """Loads Qwen3-1.7B, or Qwen3-0.6B when the allocation fails."""
    try:
        return load_model("qwen3-1.7b")
    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        return load_model("qwen3-0.6b")


def _train_and_time(
    loaded: LoadedModel,
    adapter: LorraAdapter,
    pairs: Sequence[Pair],
    chat: ChatFormat,
) -> tuple[LorraTrace, float]:
    """Trains and returns the trace with the seconds per step."""
    cuda = loaded.device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(loaded.device)
    started = time.perf_counter()
    trace = train_lorra(loaded, adapter, pairs, chat)
    if cuda:
        torch.cuda.synchronize(loaded.device)
    return trace, (time.perf_counter() - started) / adapter.settings.steps


def test_train_and_time_reports_seconds_per_step(
    tiny_arch: LoadedModel,
) -> None:
    settings = _settings(steps=2, batch_size=2)
    adapter = attach_adapter(tiny_arch, settings)
    trace, seconds = _train_and_time(
        tiny_arch, adapter, PAIRS, ChatFormat("plain")
    )
    assert len(trace.losses) == 2
    assert 0.0 < seconds < 60.0


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize(
    "checkpointing", [False, True], ids=["plain", "checkpointed"]
)
def test_qwen3_training_memory_with_and_without_checkpointing(
    checkpointing: bool,
) -> None:
    loaded = _load_post_trained_qwen3()
    try:
        chat = chat_format_for(loaded)
        settings = LorraSettings(
            layers=default_layers(loaded.n_layers),
            batch_size=4,
            steps=6,
            gradient_checkpointing=checkpointing,
        )
        pairs, dropped = fit_rows(loaded, _alpaca_pairs(20), chat, settings)
        assert len(pairs) >= settings.batch_size
        adapter = attach_adapter(loaded, settings)
        trace, seconds = _train_and_time(loaded, adapter, pairs, chat)
        key = loaded.checkpoint.key if loaded.checkpoint else "unknown"
        print(
            f"{key} checkpointing={checkpointing} "
            f"peak_memory_mb={trace.peak_memory_mb:.0f} "
            f"seconds_per_step={seconds:.2f} "
            f"trainable={adapter.n_trainable()} "
            f"pairs={len(pairs)} dropped={dropped} "
            f"losses={[round(loss, 1) for loss in trace.losses]}"
        )
        assert all(math.isfinite(loss) for loss in trace.losses)
        assert trace.peak_memory_mb is not None
        assert trace.peak_memory_mb > 0.0
    finally:
        del loaded
        gc.collect()
        torch.cuda.empty_cache()
