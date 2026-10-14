import pytest
import torch

from tomography.control.operators import Operator
from tomography.control.operators import apply_operator

OPS: list[Operator] = ["linear", "piecewise", "projection"]
DTYPES = [torch.float32, torch.bfloat16, torch.float16, torch.float64]
DTYPE_IDS = ["fp32", "bf16", "fp16", "fp64"]


def _randn(
    *shape: int, dtype: torch.dtype = torch.float32, seed: int = 0
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator).to(dtype)


def test_linear_adds_the_scaled_vector() -> None:
    h = _randn(2, 3, 8, seed=1)
    v = _randn(8, seed=2)

    out = apply_operator(h, v, 0.5, "linear")

    torch.testing.assert_close(out, h + 0.5 * v)


def test_linear_is_the_default_operator() -> None:
    h = _randn(4, 8, seed=1)
    v = _randn(8, seed=2)

    out = apply_operator(h, v, -2.0)

    torch.testing.assert_close(out, h - 2.0 * v)


@pytest.mark.parametrize("op", OPS)
def test_zero_coefficient_returns_the_input_object(op: Operator) -> None:
    h = _randn(2, 3, 8, seed=1)
    v = _randn(8, seed=2)

    out = apply_operator(h, v, 0.0, op)

    assert out is h


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_output_keeps_shape_dtype_and_device(dtype: torch.dtype) -> None:
    h = _randn(2, 5, 8, dtype=dtype, seed=1)
    v = _randn(8, seed=2)

    out = apply_operator(h, v, 1.5)

    assert out.shape == h.shape
    assert out.dtype == dtype
    assert out.device == h.device


def test_bf16_math_runs_in_float32_then_casts_back() -> None:
    h = _randn(2, 3, 8, dtype=torch.bfloat16, seed=1)
    v = _randn(8, dtype=torch.bfloat16, seed=2)

    out = apply_operator(h, v, 0.3)

    expected = (h.float() + 0.3 * v.float()).to(torch.bfloat16)
    assert torch.equal(out, expected)


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


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("coeff", [-1.5, 0.4, 1.0, 3.0])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"]
)
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
    assert torch.equal(out, expected)


def test_piecewise_grows_the_component_along_v_for_both_signs() -> None:
    h = _randn(64, 16, seed=1) * 10
    v = _randn(16, seed=2)
    before = (h @ v).abs()
    assert bool((h @ v < 0).any())
    assert bool((h @ v > 0).any())

    out = apply_operator(h, v, 0.5, "piecewise")

    grown = (out @ v).abs() - before
    torch.testing.assert_close(
        grown, torch.full_like(grown, 0.5 * float(v @ v))
    )


def _unit(vector: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(vector, dim=0)


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

    assert float((out @ v).abs().max()) < 1e-5


def test_projection_with_unit_coefficient_stays_at_bf16_resolution() -> None:
    h = _randn(4, 6, 64, dtype=torch.bfloat16, seed=1)
    v = _unit(_randn(64, seed=2)).to(torch.bfloat16)

    out = apply_operator(h, v, 1.0, "projection")

    exact = apply_operator(h.float(), v.float(), 1.0, "projection")
    assert float((exact @ v.float()).abs().max()) < 1e-5
    assert torch.equal(out, exact.to(torch.bfloat16))
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

    assert torch.equal(out, h)


def test_float64_input_keeps_float64_precision() -> None:
    h = _randn(2, 3, 8, dtype=torch.float64, seed=1)
    v = _randn(8, dtype=torch.float64, seed=2)

    out = apply_operator(h, v, 1e-9)

    assert torch.equal(out, h + 1e-9 * v)


def test_unknown_operator_is_rejected() -> None:
    h = _randn(2, 8, seed=1)
    v = _randn(8, seed=2)

    with pytest.raises(ValueError, match="operator"):
        apply_operator(h, v, 1.0, "cubic")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("v_shape", "h_shape"),
    [((7,), (2, 8)), ((2, 8), (2, 8)), ((), (2, 8))],
    ids=["wrong-size", "matrix", "scalar"],
)
def test_vector_must_match_the_hidden_size(
    v_shape: tuple[int, ...], h_shape: tuple[int, ...]
) -> None:
    h = _randn(*h_shape, seed=1)
    v = _randn(*v_shape, seed=2)

    with pytest.raises(ValueError, match="vector"):
        apply_operator(h, v, 1.0)


def test_non_floating_hidden_states_are_rejected() -> None:
    h = torch.ones(2, 8, dtype=torch.int64)
    v = _randn(8, seed=2)

    with pytest.raises(ValueError, match="floating"):
        apply_operator(h, v, 1.0)


def test_non_finite_coefficient_is_rejected() -> None:
    h = _randn(2, 8, seed=1)
    v = _randn(8, seed=2)

    with pytest.raises(ValueError, match="finite"):
        apply_operator(h, v, float("nan"))


@pytest.mark.parametrize("op", OPS)
def test_mask_edits_only_the_selected_positions(op: Operator) -> None:
    h = _randn(2, 4, 8, seed=1)
    v = _randn(8, seed=2)
    mask = torch.tensor(
        [[True, False, True, False], [False, False, True, True]]
    )

    out = apply_operator(h, v, 1.3, op, mask=mask)

    full = apply_operator(h, v, 1.3, op)
    assert torch.equal(out[mask], full[mask])
    assert torch.equal(out[~mask], h[~mask])


def test_sequence_mask_broadcasts_over_the_batch() -> None:
    h = _randn(2, 4, 8, seed=1)
    v = _randn(8, seed=2)
    mask = torch.tensor([True, False, False, True])

    out = apply_operator(h, v, 1.3, mask=mask)

    edited = apply_operator(h, v, 1.3)
    assert torch.equal(out[:, [0, 3]], edited[:, [0, 3]])
    assert torch.equal(out[:, [1, 2]], h[:, [1, 2]])


def test_batch_mask_broadcasts_over_the_sequence() -> None:
    h = _randn(2, 4, 8, seed=1)
    v = _randn(8, seed=2)
    mask = torch.tensor([[True], [False]])

    out = apply_operator(h, v, 1.3, mask=mask)

    edited = apply_operator(h, v, 1.3)
    assert torch.equal(out[0], edited[0])
    assert torch.equal(out[1], h[1])


def test_scalar_mask_selects_a_single_vector() -> None:
    h = _randn(8, seed=1)
    v = _randn(8, seed=2)

    kept = apply_operator(h, v, 1.3, mask=torch.tensor(False))
    edited = apply_operator(h, v, 1.3, mask=torch.tensor(True))

    assert torch.equal(kept, h)
    assert torch.equal(edited, apply_operator(h, v, 1.3))


def test_mask_leaves_unselected_bf16_positions_bit_exact() -> None:
    h = _randn(2, 4, 8, dtype=torch.bfloat16, seed=1)
    v = _randn(8, dtype=torch.bfloat16, seed=2)
    mask = torch.tensor([[True, False, True, False]] * 2)

    out = apply_operator(h, v, 1.3, mask=mask)

    assert out.dtype == torch.bfloat16
    assert torch.equal(out[~mask], h[~mask])


def test_zero_coefficient_with_a_mask_still_returns_the_input_object() -> None:
    h = _randn(2, 4, 8, seed=1)
    v = _randn(8, seed=2)

    out = apply_operator(h, v, 0.0, mask=torch.ones(2, 4, dtype=torch.bool))

    assert out is h


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
def test_preserve_norm_restores_the_norm_of_every_vector(
    op: Operator, coeff: float
) -> None:
    h = _randn(2, 5, 16, seed=1) * 4
    v = _randn(16, seed=2)

    out = apply_operator(h, v, coeff, op, preserve_norm=True)

    torch.testing.assert_close(out.norm(dim=-1), h.norm(dim=-1))


@pytest.mark.parametrize("op", OPS)
def test_preserve_norm_only_rescales_the_edited_vector(op: Operator) -> None:
    h = _randn(2, 5, 16, seed=1) * 4
    v = _randn(16, seed=2)

    out = apply_operator(h, v, 1.5, op, preserve_norm=True)

    edited = apply_operator(h, v, 1.5, op)
    rescaled = edited * (h.norm(dim=-1) / edited.norm(dim=-1)).unsqueeze(-1)
    torch.testing.assert_close(out, rescaled)


def test_preserve_norm_leaves_masked_out_positions_untouched() -> None:
    h = _randn(2, 4, 8, seed=1) * 3
    v = _randn(8, seed=2)
    mask = torch.tensor([[True, False, True, False]] * 2)

    out = apply_operator(h, v, 2.0, mask=mask, preserve_norm=True)

    assert torch.equal(out[~mask], h[~mask])
    torch.testing.assert_close(out.norm(dim=-1), h.norm(dim=-1))
    assert not torch.equal(out[mask], h[mask])


def test_preserve_norm_computes_in_float32_for_bf16() -> None:
    h = _randn(2, 5, 16, dtype=torch.bfloat16, seed=1) * 4
    v = _randn(16, dtype=torch.bfloat16, seed=2)

    out = apply_operator(h, v, 2.0, preserve_norm=True)

    exact = apply_operator(h.float(), v.float(), 2.0, preserve_norm=True)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, exact.to(torch.bfloat16))


def test_preserve_norm_keeps_a_fully_projected_vector_at_zero() -> None:
    v = torch.tensor([1.0, 0.0, 0.0, 0.0])
    h = torch.tensor([[2.0, 0.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0]])

    out = apply_operator(h, v, 1.0, "projection", preserve_norm=True)

    assert torch.equal(out[0], torch.zeros(4))
    torch.testing.assert_close(out[1].norm(), h[1].norm())


def test_preserve_norm_of_a_zero_vector_stays_zero() -> None:
    v = _randn(8, seed=2)

    out = apply_operator(torch.zeros(3, 8), v, 1.0, preserve_norm=True)

    assert torch.equal(out, torch.zeros(3, 8))


def test_zero_coefficient_with_preserve_norm_returns_the_input_object() -> None:
    h = _randn(2, 8, seed=1)
    v = _randn(8, seed=2)

    assert apply_operator(h, v, 0.0, preserve_norm=True) is h


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
