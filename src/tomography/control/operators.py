"""Residual-stream edit operators for representation control.

The operators act on plain tensors and know nothing about models or hooks, so
a controller can wrap them in a residual edit at any layer.
"""

import math
from typing import Literal

import torch

Operator = Literal["linear", "piecewise", "projection"]

_OPERATORS = ("linear", "piecewise", "projection")


def apply_operator(
    h: torch.Tensor,
    v: torch.Tensor,
    coeff: float,
    op: Operator = "linear",
    *,
    mask: torch.Tensor | None = None,
    preserve_norm: bool = False,
) -> torch.Tensor:
    """Applies a control operator along a direction to residual vectors.

    With `c` the coefficient, the operators are:

    * `"linear"`: `h + c v`.
    * `"piecewise"`: `h + c s v` with `s = +1` if `h.v >= 0` else `-1`, so
      the component along `v` always grows in magnitude for `c > 0`.
    * `"projection"`: `h - c (h.v / |v|^2) v`, which removes the component
      along `v` at `c = 1` and is independent of the norm of `v`.

    The math runs in float32 (float64 for float64 input) and the result is
    cast back to the dtype of `h`; shape and device are preserved. A zero
    coefficient is an exact no-op and returns `h` itself.

    Args:
        h: Residual vectors with shape `(..., d_model)`.
        v: Direction with shape `(d_model,)`.
        coeff: Finite scalar coefficient.
        op: Which operator to apply.
        mask: Optional bool tensor broadcastable to `h.shape[:-1]`; only
            the vectors where it is True are edited, the others are returned
            bit for bit.
        preserve_norm: Whether to rescale each edited vector back to its
            norm before the edit. A vector the edit reduces to zero stays
            zero.

    Returns:
        The edited residual vectors, or `h` itself when `coeff == 0`.

    Raises:
        ValueError: If `op` is unknown, `h` is not a floating point tensor
            with a trailing axis, `v` is not a vector of size `h.shape[-1]`,
            `coeff` is not finite, `mask` is not a bool tensor broadcastable
            to `h.shape[:-1]`, or `op` is `"projection"` and `v` is zero.
    """
    if op not in _OPERATORS:
        raise ValueError(
            f"unknown operator {op!r}; expected one of {_OPERATORS}"
        )
    if not h.is_floating_point():
        raise ValueError(f"h must be floating point, got {h.dtype}")
    if h.ndim == 0:
        raise ValueError("h must have a trailing d_model axis")
    if v.shape != h.shape[-1:]:
        raise ValueError(
            f"vector must have shape {tuple(h.shape[-1:])}, "
            f"got {tuple(v.shape)}"
        )
    if not math.isfinite(coeff):
        raise ValueError(f"coefficient must be finite, got {coeff}")
    if mask is not None:
        _check_mask(mask, h.shape[:-1])
    if coeff == 0:
        return h
    compute_dtype = torch.promote_types(h.dtype, torch.float32)
    h_c = h.to(compute_dtype)
    v_c = v.to(device=h.device, dtype=compute_dtype)
    edited = h_c + _shift(h_c, v_c, coeff, op)
    if preserve_norm:
        prior = h_c.norm(dim=-1, keepdim=True)
        current = edited.norm(dim=-1, keepdim=True)
        edited = edited * torch.where(current > 0, prior / current, 1.0)
    if mask is not None:
        edited = torch.where(mask.to(h.device).unsqueeze(-1), edited, h_c)
    return edited.to(h.dtype)


def _shift(
    h: torch.Tensor, v: torch.Tensor, coeff: float, op: Operator
) -> torch.Tensor:
    """Returns the change that `op` adds to `h`, broadcastable to `h`."""
    if op == "linear":
        return coeff * v
    dots = h @ v
    if op == "piecewise":
        sign = torch.where(dots >= 0, 1.0, -1.0).to(h.dtype)
        return coeff * sign.unsqueeze(-1) * v
    norm_sq = v @ v
    if norm_sq == 0:
        raise ValueError("projection needs a nonzero direction, got zero")
    return -coeff * (dots / norm_sq).unsqueeze(-1) * v


def _check_mask(mask: torch.Tensor, lead_shape: torch.Size) -> None:
    if mask.dtype != torch.bool:
        raise ValueError(f"mask must be a bool tensor, got {mask.dtype}")
    try:
        mask.expand(lead_shape)
    except RuntimeError as error:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} cannot broadcast to "
            f"{tuple(lead_shape)}"
        ) from error
