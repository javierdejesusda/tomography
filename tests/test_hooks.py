import dataclasses

import pytest
import torch

from tomography.hooks import ForwardInfo
from tomography.hooks import ResidualEdit
from tomography.hooks import capture_residuals
from tomography.hooks import edit_residuals
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


def test_identity_edit_leaves_the_logits_unchanged(
    tiny_arch: LoadedModel,
) -> None:
    lm = tiny_arch
    inputs = batch(lm)
    edits = {layer: lambda h, info: h for layer in range(lm.n_layers + 1)}
    with torch.no_grad():
        expected = lm.model(**inputs).logits

        with edit_residuals(lm, edits):
            actual = lm.model(**inputs).logits

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("layer", [0, 1, 3], ids=["embedding", "mid", "top"])
def test_additive_edit_shifts_the_stream_at_that_layer(
    tiny_arch: LoadedModel, layer: int
) -> None:
    lm = tiny_arch
    inputs = batch(lm)
    vector = torch.randn(lm.d_model)
    with capture_residuals(lm, [layer]) as clean, torch.no_grad():
        lm.model(**inputs)

    with (
        torch.no_grad(),
        edit_residuals(lm, {layer: lambda h, info: h + vector}),
        capture_residuals(lm, [layer]) as edited,
    ):
        lm.model(**inputs)

    assert torch.allclose(edited[layer], clean[layer] + vector)


def test_edit_at_the_top_steers_the_logits_through_the_final_norm(
    tiny_arch: LoadedModel,
) -> None:
    lm = tiny_arch
    inputs = batch(lm)
    vector = torch.randn(lm.d_model)
    top = lm.n_layers
    with capture_residuals(lm, [top]) as clean, torch.no_grad():
        lm.model(**inputs)
    expected = lm.model.get_output_embeddings()(
        lm.final_norm(clean[top] + vector)
    )

    with torch.no_grad(), edit_residuals(lm, {top: lambda h, info: h + vector}):
        logits = lm.model(**inputs).logits

    assert torch.allclose(logits, expected, atol=1e-5)


def test_a_capture_inside_an_edit_sees_the_edited_stream(
    tiny_gpt2: LoadedModel,
) -> None:
    lm = tiny_gpt2
    inputs = batch(lm)

    def shift(h: torch.Tensor, info: ForwardInfo) -> torch.Tensor:
        return h + 1

    with capture_residuals(lm, [2, 3]) as clean, torch.no_grad():
        lm.model(**inputs)

    with (
        torch.no_grad(),
        edit_residuals(lm, {2: shift}),
        capture_residuals(lm, [2, 3]) as inside,
    ):
        lm.model(**inputs)
    with (
        torch.no_grad(),
        capture_residuals(lm, [2]) as outside,
        edit_residuals(lm, {2: shift}),
    ):
        lm.model(**inputs)

    assert torch.allclose(inside[2], clean[2] + 1)
    assert not torch.allclose(inside[3], clean[3])
    assert torch.equal(outside[2], clean[2])


def test_nested_edits_apply_in_registration_order(
    tiny_gpt2: LoadedModel,
) -> None:
    lm = toy_model(tiny_gpt2)
    x = torch.ones(1, 3, 4)

    with (
        edit_residuals(lm, {1: lambda h, info: h * 3}),
        edit_residuals(lm, {1: lambda h, info: h + 1}),
    ):
        out = lm.decoder(x)

    assert torch.equal(out, ((x * 2) * 3 + 1) * 2)


def test_edit_works_on_tuple_block_outputs_and_keyword_inputs(
    tiny_gpt2: LoadedModel,
) -> None:
    lm = toy_model(tiny_gpt2)
    x = torch.ones(1, 3, 4)

    with edit_residuals(
        lm, {0: lambda h, info: h + 1, 2: lambda h, info: h - 1}
    ):
        out = lm.decoder(x)

    assert torch.equal(out, (((x + 1) * 2) * 2) - 1)


def test_edit_receives_the_mask_and_no_positions_in_a_plain_forward(
    tiny_gpt2: LoadedModel,
) -> None:
    lm = tiny_gpt2
    inputs = batch(lm)
    seen: list[ForwardInfo] = []

    def edit(h: torch.Tensor, info: ForwardInfo) -> torch.Tensor:
        seen.append(info)
        return h

    with torch.no_grad(), edit_residuals(lm, {1: edit}):
        lm.model(**inputs)

    assert torch.equal(seen[0].attention_mask, inputs["attention_mask"])
    assert seen[0].position_ids is None


def test_edit_receives_explicit_positions_when_the_caller_passes_them(
    tiny_gpt2: LoadedModel,
) -> None:
    lm = tiny_gpt2
    encoded = lm.encode(TEXTS, padding_side="left")
    seen: list[ForwardInfo] = []

    def edit(h: torch.Tensor, info: ForwardInfo) -> torch.Tensor:
        seen.append(info)
        return h

    with torch.no_grad(), edit_residuals(lm, {1: edit}):
        lm.model(
            input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask,
            position_ids=encoded.position_ids,
        )

    assert torch.equal(seen[0].position_ids, encoded.position_ids)


def test_generate_gives_the_edit_a_growing_mask_and_explicit_positions(
    tiny_arch: LoadedModel,
) -> None:
    lm = tiny_arch
    encoded = lm.encode(TEXTS, padding_side="left")
    seen: list[ForwardInfo] = []

    def edit(h: torch.Tensor, info: ForwardInfo) -> torch.Tensor:
        seen.append(info)
        return h

    with torch.no_grad(), edit_residuals(lm, {1: edit}):
        lm.model.generate(
            input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask,
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=lm.pad_id,
        )

    width = encoded.input_ids.shape[1]
    assert [i.attention_mask.shape[1] for i in seen] == [width, width + 1]
    assert [i.position_ids.shape[1] for i in seen] == [width, 1]


def test_edit_must_keep_shape_dtype_and_device(tiny_gpt2: LoadedModel) -> None:
    lm = tiny_gpt2
    inputs = batch(lm)

    def run(edit: ResidualEdit) -> None:
        with edit_residuals(lm, {1: edit}):
            lm.model(**inputs)

    with pytest.raises(ValueError, match="shape"):
        run(lambda h, info: h[:, :-1])
    with pytest.raises(ValueError, match="dtype"):
        run(lambda h, info: h.to(torch.float64))

    assert hook_counts(lm) == 0


def test_edit_removes_its_hooks_even_when_the_body_raises(
    tiny_gpt2: LoadedModel,
) -> None:
    keep = {0: lambda h, info: h, 2: lambda h, info: h}

    def body() -> None:
        with edit_residuals(tiny_gpt2, keep):
            assert hook_counts(tiny_gpt2) == 3
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        body()

    assert hook_counts(tiny_gpt2) == 0


def test_edit_rejects_layers_outside_zero_to_l(tiny_gpt2: LoadedModel) -> None:
    with (
        pytest.raises(ValueError, match="layer"),
        edit_residuals(tiny_gpt2, {4: lambda h, info: h}),
    ):
        pass

    assert hook_counts(tiny_gpt2) == 0
