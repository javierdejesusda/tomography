from typing import Any
from typing import get_args

import pytest
import torch

from tomography.control.operators import Operator
from tomography.control.operators import apply_operator

OPS: list[Operator] = list(get_args(Operator))
FP32 = pytest.param(torch.float32, id="fp32")
BF16 = pytest.param(torch.bfloat16, id="bf16")
FP16 = pytest.param(torch.float16, id="fp16")
FP64 = pytest.param(torch.float64, id="fp64")


def _randn(
    *shape: int, dtype: torch.dtype = torch.float32, seed: int = 0
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator).to(dtype)


def _unit(vector: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(vector, dim=0)


def _assert_identical(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _reference(
    h: torch.Tensor, v: torch.Tensor, coeff: float, op: Operator
) -> torch.Tensor:
    """Per-vector float64 loop implementation of the three operators."""
    flat = h.reshape(-1, h.shape[-1]).double()
    v64 = v.double()
    rows = []
    for row in flat:
        dot = float(row @ v64)
        if op == "linear":
            rows.append(row + coeff * v64)
        elif op == "piecewise":
            sign = 1.0 if dot >= 0 else -1.0
            rows.append(row + coeff * sign * v64)
        else:
            rows.append(row - coeff * (dot / float(v64 @ v64)) * v64)
    return torch.stack(rows).reshape(h.shape)


def test_linear_is_the_default_operator() -> None:
    h = _randn(4, 8, seed=1)
    v = _randn(8, seed=2)

    out = apply_operator(h, v, -2.0)

    torch.testing.assert_close(out, h - 2.0 * v)


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"mask": torch.ones(2, 3, dtype=torch.bool)},
        {"preserve_norm": True},
    ],
    ids=["plain", "masked", "preserve-norm"],
)
def test_zero_coefficient_returns_the_input_object(
    op: Operator, options: dict[str, Any]
) -> None:
    h = _randn(2, 3, 8, seed=1)
    v = _randn(8, seed=2)

    out = apply_operator(h, v, 0.0, op, **options)

    assert out is h


@pytest.mark.parametrize("dtype", [FP32, BF16, FP16, FP64])
def test_output_keeps_shape_dtype_and_device(dtype: torch.dtype) -> None:
    h = _randn(2, 5, 8, dtype=dtype, seed=1)
    v = _randn(8, seed=2)

    out = apply_operator(h, v, 1.5)

    assert out.shape == h.shape
    assert out.dtype == dtype
    assert out.device == h.device


@pytest.mark.parametrize("dtype", [BF16, FP16])
@pytest.mark.parametrize("preserve_norm", [False, True], ids=["plain", "norm"])
@pytest.mark.parametrize("op", OPS)
def test_low_precision_math_runs_in_float32_then_casts_back(
    op: Operator, preserve_norm: bool, dtype: torch.dtype
) -> None:
    h = _randn(2, 5, 16, dtype=dtype, seed=1) * 4
    v = _randn(16, dtype=dtype, seed=2)
    mask = torch.tensor([[True, False, True, True, False]] * 2)

    out = apply_operator(h, v, 1.3, op, mask=mask, preserve_norm=preserve_norm)

    exact = apply_operator(
        h.float(), v.float(), 1.3, op, mask=mask, preserve_norm=preserve_norm
    )
    _assert_identical(out, exact.to(dtype))


@pytest.mark.parametrize("op", OPS)
def test_float64_input_is_not_downcast(op: Operator) -> None:
    h = _randn(2, 5, 16, dtype=torch.float64, seed=1)
    v = _randn(16, dtype=torch.float64, seed=2)

    out = apply_operator(h, v, 0.1, op)

    expected = _reference(h, v, 0.1, op)
    torch.testing.assert_close(out, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("coeff", [-1.5, 0.4, 1.0, 3.0])
@pytest.mark.parametrize("dtype", [FP32, BF16])
def test_matches_the_per_vector_reference(
    op: Operator, coeff: float, dtype: torch.dtype
) -> None:
    h = _randn(2, 5, 16, dtype=dtype, seed=1)
    v = _randn(16, dtype=dtype, seed=2) * 3

    out = apply_operator(h, v, coeff, op)

    expected = _reference(h, v, coeff, op).to(dtype)
    torch.testing.assert_close(out, expected)


def test_piecewise_follows_the_sign_of_the_projection() -> None:
    v = torch.tensor([2.0, 0.0, 0.0, 0.0])
    h = torch.tensor(
        [
            [1.0, 5.0, 0.0, 0.0],
            [-1.0, 5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0],
        ]
    )

    out = apply_operator(h, v, 0.5, "piecewise")

    expected = torch.tensor(
        [
            [2.0, 5.0, 0.0, 0.0],
            [-2.0, 5.0, 0.0, 0.0],
            [1.0, 5.0, 0.0, 0.0],
        ]
    )
    _assert_identical(out, expected)


def test_piecewise_moves_each_vector_along_the_sign_of_its_projection() -> None:
    h = _randn(64, 16, seed=1) * 10
    v = _randn(16, seed=2)
    sign = torch.sign(h.double() @ v.double()).float()
    assert bool((sign > 0).any())
    assert bool((sign < 0).any())

    out = apply_operator(h, v, 0.5, "piecewise")

    torch.testing.assert_close(out - h, 0.5 * sign.unsqueeze(-1) * v)


@pytest.mark.parametrize(
    ("d_model", "norm"),
    [(64, 1.0), (64, 100.0), (768, 100.0)],
    ids=["d64-norm1", "d64-norm100", "d768-norm100"],
)
def test_projection_with_unit_coefficient_leaves_no_component_in_fp32(
    d_model: int, norm: float
) -> None:
    h = _randn(4, 6, d_model, seed=1)
    h = h / h.norm(dim=-1, keepdim=True) * norm
    v = _unit(_randn(d_model, seed=2))

    out = apply_operator(h, v, 1.0, "projection")

    assert float((out.double() @ v.double()).abs().max()) < 1e-5


def test_projection_with_unit_coefficient_stays_at_bf16_resolution() -> None:
    h = _randn(4, 6, 64, dtype=torch.bfloat16, seed=1)
    v = _unit(_randn(64, seed=2)).to(torch.bfloat16)

    out = apply_operator(h, v, 1.0, "projection")

    exact = apply_operator(h.float(), v.float(), 1.0, "projection")
    assert float((exact.double() @ v.double()).abs().max()) < 1e-5
    bound = (out.float().abs() * v.float().abs()).sum(-1) * 2**-8
    assert bool(((out.float() @ v.float()).abs() <= bound).all())


def test_projection_ignores_the_norm_of_the_direction() -> None:
    h = _randn(3, 16, seed=1)
    v = _randn(16, seed=2)

    out = apply_operator(h, v, 0.7, "projection")
    scaled = apply_operator(h, v * 10, 0.7, "projection")

    torch.testing.assert_close(out, scaled)


@pytest.mark.parametrize("op", OPS)
def test_zero_direction_leaves_the_vectors_unchanged(op: Operator) -> None:
    h = _randn(3, 16, seed=1)

    out = apply_operator(h, torch.zeros(16), 2.0, op)

    _assert_identical(out, h)


def test_unknown_operator_is_rejected() -> None:
    h = _randn(2, 8, seed=1)
    v = _randn(8, seed=2)

    with pytest.raises(ValueError, match="operator"):
        apply_operator(h, v, 1.0, "cubic")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "v_shape", [(7,), (2, 8), ()], ids=["wrong-size", "matrix", "scalar"]
)
def test_vector_must_match_the_hidden_size(v_shape: tuple[int, ...]) -> None:
    h = _randn(2, 8, seed=1)
    v = _randn(*v_shape, seed=2)

    with pytest.raises(ValueError, match="vector"):
        apply_operator(h, v, 1.0)


def test_non_floating_hidden_states_are_rejected() -> None:
    h = torch.ones(2, 8, dtype=torch.int64)
    v = _randn(8, seed=2)

    with pytest.raises(ValueError, match="floating"):
        apply_operator(h, v, 1.0)


def test_hidden_states_need_a_trailing_axis() -> None:
    with pytest.raises(ValueError, match="trailing"):
        apply_operator(torch.tensor(1.0), torch.tensor(1.0), 1.0)


@pytest.mark.parametrize("coeff", [float("nan"), float("inf")])
def test_non_finite_coefficient_is_rejected(coeff: float) -> None:
    h = _randn(2, 8, seed=1)
    v = _randn(8, seed=2)

    with pytest.raises(ValueError, match="finite"):
        apply_operator(h, v, coeff)


@pytest.mark.parametrize("dtype", [FP32, BF16])
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize(
    ("h_shape", "mask_values"),
    [
        pytest.param(
            (2, 4, 8),
            [[True, False, True, False], [False, False, True, True]],
            id="full",
        ),
        pytest.param((2, 4, 8), [True, False, False, True], id="over-batch"),
        pytest.param((2, 4, 8), [[True], [False]], id="over-sequence"),
        pytest.param((8,), True, id="scalar"),
    ],
)
def test_mask_edits_only_the_selected_vectors(
    h_shape: tuple[int, ...],
    mask_values: object,
    op: Operator,
    dtype: torch.dtype,
) -> None:
    h = _randn(*h_shape, dtype=dtype, seed=1)
    v = _randn(8, dtype=dtype, seed=2)
    mask = torch.tensor(mask_values)

    out = apply_operator(h, v, 1.3, op, mask=mask)

    full = apply_operator(h, v, 1.3, op)
    selected = mask.expand(h.shape[:-1])
    _assert_identical(out[selected], full[selected])
    _assert_identical(out[~selected], h[~selected])


def test_mask_must_be_boolean() -> None:
    h = _randn(2, 4, 8, seed=1)
    v = _randn(8, seed=2)

    with pytest.raises(ValueError, match="bool"):
        apply_operator(h, v, 1.0, mask=torch.ones(2, 4))


@pytest.mark.parametrize(
    "mask_shape", [(3,), (4, 2), (5, 2, 4)], ids=["short", "swapped", "larger"]
)
def test_mask_must_broadcast_to_the_leading_shape(
    mask_shape: tuple[int, ...],
) -> None:
    h = _randn(2, 4, 8, seed=1)
    v = _randn(8, seed=2)

    with pytest.raises(ValueError, match="mask"):
        apply_operator(h, v, 1.0, mask=torch.ones(mask_shape, dtype=torch.bool))


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("coeff", [-2.0, 0.5, 3.0])
def test_preserve_norm_rescales_each_edited_vector_to_its_prior_norm(
    op: Operator, coeff: float
) -> None:
    h = _randn(2, 5, 16, seed=1) * 4
    v = _randn(16, seed=2)

    out = apply_operator(h, v, coeff, op, preserve_norm=True)

    edited = apply_operator(h, v, coeff, op)
    rescaled = edited * (h.norm(dim=-1) / edited.norm(dim=-1)).unsqueeze(-1)
    torch.testing.assert_close(out, rescaled)
    torch.testing.assert_close(out.norm(dim=-1), h.norm(dim=-1))


def test_preserve_norm_leaves_masked_out_positions_untouched() -> None:
    h = _randn(2, 4, 8, seed=1) * 3
    v = _randn(8, seed=2)
    mask = torch.tensor([[True, False, True, False]] * 2)

    out = apply_operator(h, v, 2.0, mask=mask, preserve_norm=True)

    _assert_identical(out[~mask], h[~mask])
    torch.testing.assert_close(out.norm(dim=-1), h.norm(dim=-1))
    assert not torch.equal(out[mask], h[mask])


def test_preserve_norm_keeps_a_fully_projected_vector_at_zero() -> None:
    v = torch.tensor([1.0, 0.0, 0.0, 0.0])
    h = torch.tensor([[2.0, 0.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0]])

    out = apply_operator(h, v, 1.0, "projection", preserve_norm=True)

    _assert_identical(out[0], torch.zeros(4))
    torch.testing.assert_close(out[1].norm(), h[1].norm())


def test_preserve_norm_of_a_zero_vector_stays_zero() -> None:
    v = _randn(8, seed=2)

    out = apply_operator(torch.zeros(3, 8), v, 1.0, preserve_norm=True)

    _assert_identical(out, torch.zeros(3, 8))


@pytest.mark.parametrize("op", OPS)
def test_edits_have_no_data_dependent_control_flow(op: Operator) -> None:
    h = torch.empty(2, 3, 8, device="meta")
    v = torch.empty(8, device="meta")
    mask = torch.empty(2, 3, dtype=torch.bool, device="meta")

    out = apply_operator(h, v, 1.0, op, mask=mask, preserve_norm=True)

    assert out.device == h.device
    assert out.shape == h.shape


@pytest.mark.parametrize("op", OPS)
def test_gradients_stay_finite_when_the_edit_zeroes_a_vector(
    op: Operator,
) -> None:
    v = torch.tensor([1.0, 0.0, 0.0, 0.0])
    h = torch.tensor(
        [[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 0.0, 0.0]],
        requires_grad=True,
    )

    out = apply_operator(h, v, 1.0, op, preserve_norm=True)
    (grad,) = torch.autograd.grad(out.sum(), h)

    assert bool(torch.isfinite(grad).all())
