"""Context managers that read or edit the residual stream of a model.

Layer l in 0..L follows the project convention: l = 0 is the embedding output
(the input of block 1), l >= 1 is the raw output of block l, and l = L is the
last block output before the final norm.
"""

from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
import contextlib
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.hooks import RemovableHandle

from tomography.model import LoadedModel

TensorFn = Callable[[torch.Tensor], torch.Tensor | None]


@dataclass(frozen=True, eq=False)
class ForwardInfo:
    """What the decoder was called with, handed to residual edits.

    Attributes:
        attention_mask: The `(batch, kv_len)` 2D mask including cached
            positions, or None when the caller passed none (HF drops an
            all-ones mask).
        position_ids: The positions the caller passed. None in a plain
            forward, where HF derives positions from the sequence length, so
            controllers derive them from the mask; explicit in `generate`.
    """

    attention_mask: torch.Tensor | None
    position_ids: torch.Tensor | None


ResidualEdit = Callable[[torch.Tensor, ForwardInfo], torch.Tensor]


def _check_layers(loaded: LoadedModel, layers: Sequence[int]) -> list[int]:
    for layer in layers:
        if not 0 <= layer <= loaded.n_layers:
            raise ValueError(
                f"layer {layer} is outside 0..{loaded.n_layers} "
                f"for a model with {loaded.n_layers} layers"
            )
    return list(dict.fromkeys(layers))


def _register(loaded: LoadedModel, layer: int, fn: TensorFn) -> RemovableHandle:
    """Hooks `fn` at the residual stream of `layer`.

    `fn` receives the stream tensor and returns None to keep it or a
    replacement of the same kind.
    """
    if layer == 0:

        def pre_hook(
            module: torch.nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
            hidden = args[0] if args else kwargs["hidden_states"]
            replacement = fn(hidden)
            if replacement is None:
                return None
            if args:
                return (replacement, *args[1:]), kwargs
            return args, {**kwargs, "hidden_states": replacement}

        return loaded.blocks[0].register_forward_pre_hook(
            pre_hook, with_kwargs=True
        )

    def post_hook(
        module: torch.nn.Module, args: tuple[Any, ...], output: Any
    ) -> Any:
        if isinstance(output, tuple):
            replacement = fn(output[0])
            return None if replacement is None else (replacement, *output[1:])
        return fn(output)

    return loaded.blocks[layer - 1].register_forward_hook(post_hook)


@contextlib.contextmanager
def capture_residuals(
    loaded: LoadedModel, layers: Sequence[int], *, detach: bool = True
) -> Iterator[Mapping[int, torch.Tensor]]:
    """Records the residual stream at the given layers during forwards.

    The yielded mapping fills as forwards run and keeps the tensors of the
    most recent forward. Hooks run in registration order, so a capture opened
    inside `edit_residuals` sees the edited stream. Hooks are removed on exit,
    including on exceptions.

    Args:
        loaded: The wrapped model.
        layers: Layer indices in 0..L.
        detach: Detach the tensors from autograd; pass False to keep the
            graph (LoRRA).

    Yields:
        A mapping from layer to a `(batch, seq, d_model)` tensor in the model
        dtype on the model device.

    Raises:
        ValueError: If a layer is outside 0..L.
    """
    wanted = _check_layers(loaded, layers)
    captured: dict[int, torch.Tensor] = {}

    def recorder(layer: int) -> TensorFn:
        def record(hidden: torch.Tensor) -> None:
            captured[layer] = hidden.detach() if detach else hidden

        return record

    handles: list[RemovableHandle] = []
    try:
        for layer in wanted:
            handles.append(_register(loaded, layer, recorder(layer)))
        yield captured
    finally:
        for handle in handles:
            handle.remove()


@contextlib.contextmanager
def edit_residuals(
    loaded: LoadedModel, edits: Mapping[int, ResidualEdit]
) -> Iterator[None]:
    """Replaces the residual stream at the given layers during forwards.

    Edits sit at the same hook points as `capture_residuals`: layer 0
    replaces the embedding output and layer L the last block output before
    the final norm, so it steers the logits. Hooks run in registration order,
    so nested contexts compose in the order they were opened. Hooks are
    removed on exit, including on exceptions.

    Args:
        loaded: The wrapped model.
        edits: Maps a layer in 0..L to `edit(hidden, info)`, which returns a
            tensor of the same shape, dtype and device as `hidden`.

    Yields:
        None.

    Raises:
        ValueError: If a layer is outside 0..L, or an edit changes the
            shape, dtype or device of the stream.
    """
    wanted = _check_layers(loaded, list(edits))
    info = [ForwardInfo(None, None)]

    def record_info(
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        info[0] = ForwardInfo(
            kwargs.get("attention_mask"), kwargs.get("position_ids")
        )

    def editor(layer: int) -> TensorFn:
        def edit(hidden: torch.Tensor) -> torch.Tensor:
            edited = edits[layer](hidden, info[0])
            if edited.shape != hidden.shape:
                raise ValueError(
                    f"edit at layer {layer} changed the shape "
                    f"{tuple(hidden.shape)} to {tuple(edited.shape)}"
                )
            if edited.dtype != hidden.dtype or edited.device != hidden.device:
                raise ValueError(
                    f"edit at layer {layer} changed the dtype or device "
                    f"({hidden.dtype} on {hidden.device} to "
                    f"{edited.dtype} on {edited.device})"
                )
            return edited

        return edit

    handles: list[RemovableHandle] = []
    try:
        handles.append(
            loaded.decoder.register_forward_pre_hook(
                record_info, with_kwargs=True
            )
        )
        for layer in wanted:
            handles.append(_register(loaded, layer, editor(layer)))
        yield
    finally:
        for handle in handles:
            handle.remove()
