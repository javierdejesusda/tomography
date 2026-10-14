"""Integration tests on real checkpoints; run with `pytest -m slow`."""

from typing import Literal

import pytest
import torch

from tomography.hooks import capture_residuals
from tomography.model import LoadedModel
from tomography.model import load_model

pytestmark = pytest.mark.slow

TEXTS = [
    "USER: Is the movie good? ASSISTANT: Yes, it is very good.",
    "The quick brown fox jumps over the lazy dog.",
    "Honest.",
]


@pytest.fixture(scope="module")
def gpt2() -> LoadedModel:
    return load_model("gpt2")


def forward_with_capture(
    lm: LoadedModel, texts: list[str], padding_side: Literal["left", "right"]
) -> tuple[dict[int, torch.Tensor], tuple[torch.Tensor, ...]]:
    encoded = lm.encode(texts, padding_side=padding_side)
    layers = list(range(lm.n_layers + 1))
    with capture_residuals(lm, layers) as captured, torch.inference_mode():
        output = lm.model(
            input_ids=encoded.input_ids.to(lm.device),
            attention_mask=encoded.attention_mask.to(lm.device),
            position_ids=encoded.position_ids.to(lm.device),
            output_hidden_states=True,
        )
    return dict(captured), tuple(output.hidden_states)


def test_gpt2_capture_equals_hf_hidden_states_below_the_top(
    gpt2: LoadedModel,
) -> None:
    captured, hidden_states = forward_with_capture(gpt2, TEXTS, "right")

    for layer in range(gpt2.n_layers):
        assert torch.equal(captured[layer], hidden_states[layer])


def test_gpt2_final_norm_of_the_top_capture_is_the_last_hidden_state(
    gpt2: LoadedModel,
) -> None:
    captured, hidden_states = forward_with_capture(gpt2, TEXTS, "right")

    with torch.inference_mode():
        normed = gpt2.final_norm(captured[gpt2.n_layers])

    assert torch.equal(normed, hidden_states[gpt2.n_layers])


def test_gpt2_left_padding_with_explicit_positions_matches_unpadded(
    gpt2: LoadedModel,
) -> None:
    padded, _ = forward_with_capture(gpt2, TEXTS, "left")
    width = max(gpt2.encode(TEXTS).lengths)

    for row, text in enumerate(TEXTS):
        alone, _ = forward_with_capture(gpt2, [text], "right")
        length = gpt2.encode([text]).lengths[0]
        pad = width - length
        for layer in (1, gpt2.n_layers // 2, gpt2.n_layers):
            assert torch.allclose(
                padded[layer][row, pad:],
                alone[layer][0, :length],
                rtol=1e-4,
                atol=1e-3,
            )
