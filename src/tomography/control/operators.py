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

    Returns:
        The edited residual vectors, or `h` itself when `coeff == 0`.

    Raises:
        ValueError: If `op` is unknown, `h` is not a floating point tensor
            with a trailing axis, `v` is not a vector of size `h.shape[-1]`,
            `coeff` is not finite, or `op` is `"projection"` and `v` is zero.
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
    if coeff == 0:
        return h
    compute_dtype = torch.promote_types(h.dtype, torch.float32)
    h_c = h.to(compute_dtype)
    v_c = v.to(device=h.device, dtype=compute_dtype)
    if op == "linear":
        delta = coeff * v_c
    elif op == "piecewise":
        sign = torch.where(h_c @ v_c >= 0, 1.0, -1.0).to(compute_dtype)
        delta = coeff * sign.unsqueeze(-1) * v_c
    else:
        norm_sq = v_c @ v_c
        if norm_sq == 0:
            raise ValueError("projection needs a nonzero direction, got zero")
        delta = -coeff * ((h_c @ v_c) / norm_sq).unsqueeze(-1) * v_c
    return (h_c + delta).to(h.dtype)
