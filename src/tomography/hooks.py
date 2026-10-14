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
from typing import Any

import torch
from torch.utils.hooks import RemovableHandle

from tomography.model import LoadedModel

TensorFn = Callable[[torch.Tensor], torch.Tensor | None]


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
