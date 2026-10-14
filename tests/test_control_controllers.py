from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import pytest
import torch

from tomography.control import controllers
from tomography.control.controllers import ReadingVectorController
from tomography.control.operators import Operator
from tomography.control.operators import apply_operator
from tomography.hooks import ForwardInfo
from tomography.hooks import ResidualEdit
from tomography.hooks import capture_residuals
from tomography.hooks import edit_residuals
from tomography.model import EncodedBatch
from tomography.model import LoadedModel
from tomography.model import wrap_model

NO_INFO = ForwardInfo(None, None)
TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "The lazy dog.",
    "A short text, a longer text, and another text with more words in it.",
]


@dataclass(frozen=True)
class FakeSource:
    layers: tuple[int, ...]
    directions: NDArray[np.float32]
    signs: NDArray[np.float32]


def make_source(
    layers: tuple[int, ...] = (1, 2), d_model: int = 32, seed: int = 0
) -> FakeSource:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(len(layers), d_model)).astype(np.float32)
    directions = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    signs = np.array([(-1.0) ** k for k in range(len(layers))], np.float32)
    return FakeSource(layers, directions, signs.astype(np.float32))


def edit_on_zeros(
    controller: ReadingVectorController, layer: int, d_model: int = 32
) -> torch.Tensor:
    edit = controller.edits()[layer]
    return edit(torch.zeros(1, 1, d_model), NO_INFO)[0, 0]


def test_absolute_coefficient_adds_the_oriented_vector() -> None:
    source = make_source()
    controller = ReadingVectorController.from_reader(source, coeff=2.0)

    for k, layer in enumerate(source.layers):
        expected = 2.0 * source.signs[k] * source.directions[k]
        assert torch.allclose(
            edit_on_zeros(controller, layer), torch.from_numpy(expected)
        )


def test_relative_coefficient_scales_by_the_layer_norm() -> None:
    source = make_source()
    controller = ReadingVectorController.from_reader(
        source, coeff=0.5, mode="relative", norms={1: 10.0, 2: 4.0, 7: 1.0}
    )

    for k, layer in enumerate(source.layers):
        scale = 0.5 * {1: 10.0, 2: 4.0}[layer]
        expected = scale * source.signs[k] * source.directions[k]
        assert torch.allclose(
            edit_on_zeros(controller, layer), torch.from_numpy(expected)
        )


def test_layer_subset_builds_only_the_wanted_layers() -> None:
    source = make_source(layers=(1, 2, 3))

    controller = ReadingVectorController.from_reader(source, [3, 1], coeff=1.0)

    assert sorted(controller.edits()) == [1, 3]
    assert torch.allclose(
        edit_on_zeros(controller, 3),
        torch.from_numpy(source.signs[2] * source.directions[2]),
    )


def test_layers_none_takes_every_reader_layer() -> None:
    source = make_source(layers=(0, 2, 3))

    controller = ReadingVectorController.from_reader(source, coeff=1.0)

    assert sorted(controller.edits()) == [0, 2, 3]


def test_missing_layer_is_rejected_naming_the_layer() -> None:
    with pytest.raises(ValueError, match="layer 5"):
        ReadingVectorController.from_reader(make_source(), [1, 5], coeff=1.0)


def test_relative_with_projection_is_rejected() -> None:
    with pytest.raises(ValueError, match="projection"):
        ReadingVectorController.from_reader(
            make_source(),
            coeff=1.0,
            op="projection",
            mode="relative",
            norms={1: 1.0, 2: 1.0},
        )


def source_for(lm: LoadedModel, layers: tuple[int, ...] = (1, 2)) -> FakeSource:
    return make_source(layers, lm.d_model, seed=3)


@dataclass(frozen=True)
class Run:
    logits: torch.Tensor
    before: dict[int, torch.Tensor]
    after: dict[int, torch.Tensor]


def run(
    lm: LoadedModel,
    encoded: EncodedBatch,
    edits: dict[int, ResidualEdit] | None = None,
    capture: tuple[int, ...] = (),
) -> Run:
    """Runs one forward, capturing the streams before and after the edits.

    The first capture is opened outside `edit_residuals` and sees the
    unedited stream; the second is opened inside and sees the edited one.
    """
    kwargs = {
        "input_ids": encoded.input_ids,
        "attention_mask": encoded.attention_mask,
        "position_ids": encoded.position_ids,
    }
    with (
        torch.inference_mode(),
        capture_residuals(lm, capture) as before,
        edit_residuals(lm, edits or {}),
        capture_residuals(lm, capture) as after,
    ):
        logits = lm.model(**kwargs).logits
    return Run(logits, dict(before), dict(after))


def test_coefficient_zero_leaves_logits_bit_identical(
    tiny_arch: LoadedModel,
) -> None:
    encoded = tiny_arch.encode(TEXTS)
    controller = ReadingVectorController.from_reader(
        source_for(tiny_arch), coeff=0.0
    )

    plain = run(tiny_arch, encoded)
    edited = run(tiny_arch, encoded, controller(encoded))

    assert torch.equal(plain.logits, edited.logits)


def real_token_mask(encoded: EncodedBatch) -> torch.Tensor:
    return encoded.attention_mask > 0


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["plus", "minus"])
def test_linear_edit_adds_the_vector_at_real_tokens_only(
    tiny_arch: LoadedModel, sign: float
) -> None:
    encoded = tiny_arch.encode(TEXTS)
    source = source_for(tiny_arch)
    controller = ReadingVectorController.from_reader(source, coeff=sign * 1.5)
    layer = 2

    seen = run(tiny_arch, encoded, controller(encoded), (layer,))
    vector = controller.vectors[layer]

    real = real_token_mask(encoded)
    delta = seen.after[layer] - seen.before[layer]
    assert not real.all()
    assert torch.allclose(
        delta[real], (sign * 1.5) * vector.expand_as(delta[real]), atol=1e-5
    )
    assert torch.equal(delta[~real], torch.zeros_like(delta[~real]))


def test_projection_at_coefficient_one_removes_the_component(
    tiny_arch: LoadedModel,
) -> None:
    encoded = tiny_arch.encode(TEXTS)
    controller = ReadingVectorController.from_reader(
        source_for(tiny_arch), coeff=1.0, op="projection"
    )
    layer = 2

    seen = run(tiny_arch, encoded, controller(encoded), (layer,))

    real = real_token_mask(encoded)
    residual = (seen.after[layer] @ controller.vectors[layer]).abs()
    bound = 2e-7 * seen.after[layer].norm(dim=-1)
    assert (residual <= bound)[real].all()


def test_piecewise_follows_the_sign_of_the_dot_product(
    tiny_arch: LoadedModel,
) -> None:
    encoded = tiny_arch.encode(TEXTS)
    controller = ReadingVectorController.from_reader(
        source_for(tiny_arch), coeff=0.7, op="piecewise"
    )
    layer = 2
    vector = controller.vectors[layer]

    seen = run(tiny_arch, encoded, controller(encoded), (layer,))

    real = real_token_mask(encoded)
    signs = torch.where(seen.before[layer] @ vector >= 0, 1.0, -1.0)
    delta = seen.after[layer] - seen.before[layer]
    expected = 0.7 * signs.unsqueeze(-1) * vector
    assert (signs[real] < 0).any()
    assert (signs[real] > 0).any()
    assert torch.allclose(delta[real], expected[real], atol=1e-5)


def assert_response_only(
    lm: LoadedModel,
    encoded: EncodedBatch,
    prompt_lengths: list[int],
    coeff: float = 1.5,
) -> None:
    controller = ReadingVectorController.from_reader(
        source_for(lm), coeff=coeff, mask="response"
    )
    layer = 2

    seen = run(lm, encoded, controller.edits(prompt_lengths), (layer,))

    delta = seen.after[layer] - seen.before[layer]
    expected = coeff * controller.vectors[layer]
    for row, length in enumerate(encoded.lengths):
        real = (encoded.attention_mask[row] > 0).nonzero().flatten()
        first_edited = prompt_lengths[row] - 1
        for index, column in enumerate(real.tolist()):
            if index < first_edited:
                assert torch.equal(
                    seen.after[layer][row, column],
                    seen.before[layer][row, column],
                )
            else:
                assert torch.allclose(delta[row, column], expected, atol=1e-5)
        pads = (encoded.attention_mask[row] == 0).nonzero().flatten()
        assert torch.equal(delta[row, pads], torch.zeros_like(delta[row, pads]))
        assert length > first_edited


@pytest.mark.parametrize("side", ["right", "left"])
def test_response_mask_edits_from_the_last_prompt_token_per_row(
    tiny_arch: LoadedModel, side: str
) -> None:
    encoded = tiny_arch.encode(TEXTS, padding_side=side)  # type: ignore[arg-type]
    assert len(set(encoded.lengths)) == len(TEXTS)

    assert_response_only(tiny_arch, encoded, [3, 2, 6])


def test_response_mask_accepts_one_prompt_length_for_all_rows(
    tiny_arch: LoadedModel,
) -> None:
    encoded = tiny_arch.encode(TEXTS)

    assert_response_only(tiny_arch, encoded, [3, 3, 3])
    controller = ReadingVectorController.from_reader(
        source_for(tiny_arch), coeff=1.5, mask="response"
    )
    as_list = run(tiny_arch, encoded, controller.edits([3, 3, 3]))
    as_int = run(tiny_arch, encoded, controller.edits(3))

    assert torch.equal(as_list.logits, as_int.logits)


def test_edit_mask_is_built_once_per_forward_and_shared_by_layers(
    tiny_arch: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[ForwardInfo] = []
    real_builder = controllers._build_edit_mask

    def spy(info: ForwardInfo, *args: object) -> torch.Tensor | None:
        calls.append(info)
        return real_builder(info, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(controllers, "_build_edit_mask", spy)
    encoded = tiny_arch.encode(TEXTS)
    controller = ReadingVectorController.from_reader(
        source_for(tiny_arch, (0, 1, 2, 3)), coeff=1.0, mask="response"
    )
    edits = controller.edits(encoded.lengths)

    run(tiny_arch, encoded, edits)
    assert len(calls) == 1
    run(tiny_arch, encoded, edits)
    assert len(calls) == 2
    assert calls[0] is not calls[1]


def hand_info(
    attention_mask: list[list[int]] | None,
    position_ids: list[list[int]] | None,
) -> ForwardInfo:
    return ForwardInfo(
        None if attention_mask is None else torch.tensor(attention_mask),
        None if position_ids is None else torch.tensor(position_ids),
    )


@pytest.mark.parametrize(
    ("info", "expected"),
    [
        (
            hand_info([[0, 1, 1, 1]], [[0, 0, 1, 2]]),
            [False, False, True, True],
        ),
        (hand_info([[0, 1, 1, 1]], None), [False, False, True, True]),
        (hand_info(None, None), [False, True, True, True]),
        (hand_info([[1, 1, 1, 1, 1]], None), [True]),
    ],
    ids=["position_ids", "attention_mask", "arange", "decode_step"],
)
def test_token_index_rule_branches(
    info: ForwardInfo, expected: list[bool]
) -> None:
    controller = ReadingVectorController(
        {1: torch.ones(4)}, 1.0, mask="response"
    )
    h = torch.zeros(1, len(expected), 4)

    edited = controller.edits(2)[1](h, info)

    assert (edited[0, :, 0] != 0).tolist() == expected


def generate_ids(
    lm: LoadedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    steps: int,
) -> torch.Tensor:
    """Greedy cached generation of exactly `steps` new tokens."""
    generate: Callable[..., torch.Tensor] = lm.model.generate  # type: ignore[assignment]
    with torch.inference_mode():
        ids = generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=steps,
            min_new_tokens=steps,
            pad_token_id=lm.pad_id,
        )
    return ids[:, input_ids.shape[1] :]


def greedy_without_cache(
    lm: LoadedModel,
    encoded: EncodedBatch,
    edits: dict[int, ResidualEdit],
    steps: int,
) -> torch.Tensor:
    ids = encoded.input_ids
    mask = encoded.attention_mask
    for _ in range(steps):
        positions = (mask.cumsum(dim=-1) - 1).clamp(min=0)
        with torch.inference_mode(), edit_residuals(lm, edits):
            logits = lm.model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=positions,
                use_cache=False,
            ).logits
        ids = torch.cat([ids, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
    return ids[:, encoded.input_ids.shape[1] :]


@pytest.mark.parametrize(
    "batched", [True, False], ids=["left_padded", "single"]
)
def test_cached_generate_equals_a_no_cache_loop_under_the_same_edits(
    tiny_arch: LoadedModel, batched: bool
) -> None:
    texts = TEXTS if batched else TEXTS[:1]
    encoded = tiny_arch.encode(texts, padding_side="left")
    controller = ReadingVectorController.from_reader(
        source_for(tiny_arch), coeff=6.0, mask="response"
    )
    edits = controller.edits(encoded.lengths)
    mask = encoded.attention_mask if batched else None
    steps = 6

    with edit_residuals(tiny_arch, edits):
        generated = generate_ids(tiny_arch, encoded.input_ids, mask, steps)
    plain = generate_ids(tiny_arch, encoded.input_ids, mask, steps)
    reference = greedy_without_cache(tiny_arch, encoded, edits, steps)

    assert torch.equal(generated, reference)
    assert not torch.equal(generated, plain)


@pytest.mark.parametrize("op", ["linear", "piecewise", "projection"], ids=str)
def test_bfloat16_model_keeps_its_dtype_and_edits_in_float32(
    tiny_arch: LoadedModel, op: Operator
) -> None:
    lm = wrap_model(
        tiny_arch.model.to(torch.bfloat16),  # type: ignore[arg-type]
        tiny_arch.tokenizer,
    )
    encoded = lm.encode(TEXTS)
    coeff = 1.0 if op == "projection" else 2.0
    controller = ReadingVectorController.from_reader(
        source_for(lm, (2,)), coeff=coeff, op=op
    )
    layer = 2
    vector = controller.vectors[layer]

    seen = run(lm, encoded, controller(encoded), (layer,))

    real = real_token_mask(encoded)
    before, after = seen.before[layer], seen.after[layer]
    assert seen.logits.dtype == torch.bfloat16
    assert after.dtype == torch.bfloat16
    reference = apply_operator(before.float(), vector, coeff, op, mask=real)
    assert torch.equal(after, reference.to(torch.bfloat16))
    if op == "projection":
        bound = 2e-7 * reference.norm(dim=-1)
        assert ((reference @ vector).abs() <= bound)[real].all()
        out = after.float()
        scale = 2.0**-8 * (out * vector).abs().sum(dim=-1)
        assert ((out @ vector).abs() <= scale)[real].all()


def unit(d_model: int = 4) -> torch.Tensor:
    return torch.eye(d_model)[0]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"vectors": {1: torch.zeros(4)}}, "zero"),
        ({"vectors": {1: torch.tensor([1.0, float("nan"), 0, 0])}}, "finite"),
        ({"vectors": {1: torch.tensor([1.0, float("inf"), 0, 0])}}, "finite"),
        ({"vectors": {1: torch.eye(4)}}, "1-D"),
        ({"vectors": {1: torch.tensor(1.0)}}, "1-D"),
        ({"vectors": {}}, "empty"),
        ({"vectors": {1: unit(4), 2: unit(5)}}, "different lengths"),
        ({"coeff": float("nan")}, "finite"),
        ({"coeff": float("inf")}, "finite"),
        ({"mode": "relative"}, "norm"),
        ({"mode": "relative", "norms": {2: 1.0}}, "layer 1"),
        ({"mode": "relative", "norms": {1: 0.0}}, "layer 1"),
        ({"mode": "relative", "norms": {1: float("nan")}}, "layer 1"),
        (
            {"mode": "relative", "norms": {1: 1.0}, "op": "projection"},
            "projection",
        ),
    ],
    ids=[
        "zero_vector",
        "nan_vector",
        "inf_vector",
        "matrix_vector",
        "scalar_vector",
        "empty_vectors",
        "different_lengths",
        "nan_coeff",
        "inf_coeff",
        "relative_without_norms",
        "relative_missing_layer_norm",
        "relative_zero_norm",
        "relative_nan_norm",
        "relative_projection",
    ],
)
def test_construction_rejects_bad_arguments(
    kwargs: dict[str, object], match: str
) -> None:
    arguments: dict[str, object] = {"vectors": {1: unit()}, "coeff": 1.0}
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=match):
        ReadingVectorController(**arguments)  # type: ignore[arg-type]


def test_from_reader_rejects_empty_layers() -> None:
    with pytest.raises(ValueError, match="layers is empty"):
        ReadingVectorController.from_reader(make_source(), [], coeff=1.0)


def test_zero_reading_vector_from_a_reader_is_rejected() -> None:
    source = make_source()
    zeroed = FakeSource(
        source.layers, np.zeros_like(source.directions), source.signs
    )

    with pytest.raises(ValueError, match="zero"):
        ReadingVectorController.from_reader(zeroed, coeff=1.0)


def test_relative_mode_ignores_extra_norm_layers() -> None:
    controller = ReadingVectorController(
        {1: unit()}, 2.0, mode="relative", norms={1: 3.0, 9: -1.0}
    )

    assert edit_on_zeros(controller, 1, 4).tolist() == [6.0, 0, 0, 0]


def test_response_mask_without_prompt_lengths_is_rejected() -> None:
    controller = ReadingVectorController({1: unit()}, 1.0, mask="response")

    with pytest.raises(ValueError, match="prompt_lengths"):
        controller.edits()


@pytest.mark.parametrize("lengths", [0, -1, [2, 0], []], ids=str)
def test_prompt_lengths_below_one_are_rejected(
    lengths: int | list[int],
) -> None:
    controller = ReadingVectorController({1: unit()}, 1.0, mask="response")

    with pytest.raises(ValueError, match="at least 1"):
        controller.edits(lengths)


def test_all_mask_ignores_prompt_lengths() -> None:
    controller = ReadingVectorController({1: unit()}, 1.0)

    assert sorted(controller.edits([0, -5])) == [1]


@pytest.mark.parametrize("count", [1, 2, 4], ids=["one", "two", "four"])
def test_prompt_lengths_that_do_not_match_the_rows_fail_at_the_forward(
    tiny_arch: LoadedModel, count: int
) -> None:
    encoded = tiny_arch.encode(TEXTS)
    controller = ReadingVectorController.from_reader(
        source_for(tiny_arch), coeff=1.0, mask="response"
    )
    edits = controller.edits([3] * count)

    with pytest.raises(ValueError, match="prompt lengths"):
        run(tiny_arch, encoded, edits)
