import dataclasses

import pytest
import torch

from tomography.hooks import capture_residuals
from tomography.model import LoadedModel

TEXTS = ["The quick brown fox jumps", "The lazy dog"]


class TupleBlock(torch.nn.Module):
    """A block that returns a tuple, like pre-v5 decoder layers."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, int]:
        return hidden_states * self.scale, 0


class ToyDecoder(torch.nn.Module):
    """Calls its blocks with keyword arguments, like a custom decoder."""

    def __init__(self, n_blocks: int) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(TupleBlock() for _ in range(n_blocks))

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(hidden_states=x)[0]
        return x


def toy_model(tiny: LoadedModel, n_blocks: int = 2) -> LoadedModel:
    decoder = ToyDecoder(n_blocks)
    return dataclasses.replace(
        tiny, decoder=decoder, blocks=decoder.blocks, n_layers=n_blocks
    )


def batch(lm: LoadedModel) -> dict[str, torch.Tensor]:
    encoded = lm.encode(TEXTS)
    return {
        "input_ids": encoded.input_ids,
        "attention_mask": encoded.attention_mask,
    }


def hook_counts(lm: LoadedModel) -> int:
    modules = [lm.decoder, *lm.blocks]
    return sum(
        len(m._forward_hooks) + len(m._forward_pre_hooks) for m in modules
    )


def test_capture_equals_hf_hidden_states_below_the_top(
    tiny_arch: LoadedModel,
) -> None:
    lm = tiny_arch
    layers = list(range(lm.n_layers))

    with capture_residuals(lm, layers) as captured, torch.no_grad():
        output = lm.model(**batch(lm), output_hidden_states=True)

    for layer in layers:
        assert torch.equal(captured[layer], output.hidden_states[layer])


def test_final_norm_of_the_top_capture_is_the_last_hidden_state(
    tiny_arch: LoadedModel,
) -> None:
    lm = tiny_arch

    with capture_residuals(lm, [lm.n_layers]) as captured, torch.no_grad():
        output = lm.model(**batch(lm), output_hidden_states=True)

        normed = lm.final_norm(captured[lm.n_layers])

    assert torch.equal(normed, output.hidden_states[lm.n_layers])


def test_capture_shapes_and_dtype(tiny_arch: LoadedModel) -> None:
    lm = tiny_arch
    inputs = batch(lm)

    with capture_residuals(lm, [0, 2]) as captured, torch.no_grad():
        lm.model(**inputs)

    seq = inputs["input_ids"].shape
    for layer in (0, 2):
        assert captured[layer].shape == (*seq, lm.d_model)
        assert captured[layer].dtype == lm.dtype
        assert captured[layer].device == lm.device


def test_capture_keeps_the_most_recent_forward(tiny_gpt2: LoadedModel) -> None:
    lm = tiny_gpt2
    first = lm.encode(["The quick brown fox"])
    second = lm.encode(["A short text"])

    with capture_residuals(lm, [1]) as captured, torch.no_grad():
        lm.model(input_ids=first.input_ids)
        lm.model(input_ids=second.input_ids)
    with capture_residuals(lm, [1]) as alone, torch.no_grad():
        lm.model(input_ids=second.input_ids)

    assert torch.equal(captured[1], alone[1])


def test_capture_detaches_unless_asked_not_to(tiny_gpt2: LoadedModel) -> None:
    lm = tiny_gpt2
    inputs = batch(lm)

    with capture_residuals(lm, [2]) as detached:
        lm.model(**inputs)
    with capture_residuals(lm, [2], detach=False) as attached:
        lm.model(**inputs)

    assert not detached[2].requires_grad
    assert attached[2].requires_grad
    attached[2].sum().backward()


@pytest.mark.parametrize("bad_layer", [-1, 4], ids=["negative", "past-top"])
def test_capture_rejects_layers_outside_zero_to_l(
    tiny_gpt2: LoadedModel, bad_layer: int
) -> None:
    with (
        pytest.raises(ValueError, match="layer"),
        capture_residuals(tiny_gpt2, [1, bad_layer]),
    ):
        pass

    assert hook_counts(tiny_gpt2) == 0


def test_capture_removes_its_hooks_even_when_the_body_raises(
    tiny_gpt2: LoadedModel,
) -> None:
    def body() -> None:
        with capture_residuals(tiny_gpt2, [0, 1, 3]):
            assert hook_counts(tiny_gpt2) == 3
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        body()

    assert hook_counts(tiny_gpt2) == 0


def test_nested_captures_both_record(tiny_gpt2: LoadedModel) -> None:
    lm = tiny_gpt2
    inputs = batch(lm)

    with (
        capture_residuals(lm, [1]) as outer,
        capture_residuals(lm, [1, 2]) as inner,
        torch.no_grad(),
    ):
        lm.model(**inputs)

    assert torch.equal(outer[1], inner[1])
    assert set(inner) == {1, 2}


def test_capture_accepts_tuple_block_outputs_and_keyword_inputs(
    tiny_gpt2: LoadedModel,
) -> None:
    lm = toy_model(tiny_gpt2)
    x = torch.ones(1, 3, 4)

    with capture_residuals(lm, [0, 1, 2]) as captured, torch.no_grad():
        lm.decoder(x)

    assert torch.equal(captured[0], x)
    assert torch.equal(captured[1], x * 2)
    assert torch.equal(captured[2], x * 4)
