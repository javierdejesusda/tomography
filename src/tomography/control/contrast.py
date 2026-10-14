"""Contrast vector steering over a positive and a negative prompt stream.

The three streams of a triple run as rows of one batch, and a batch-aware
residual edit adds the shift between the positive and negative streams.
"""

from collections.abc import Sequence
import math

import torch

from tomography.hooks import ForwardInfo
from tomography.hooks import ResidualEdit
from tomography.hooks import edit_residuals
from tomography.model import BosMode
from tomography.model import LoadedModel


def contrast_windows(lengths: Sequence[int], window: int) -> torch.Tensor:
    """Returns the number of shifted trailing columns of each triple.

    Args:
        lengths: Real token counts of the layout `[main, positive,
            negative]`, `3 * n` entries.
        window: Maximum number of shifted columns.

    Returns:
        An `(n,)` int64 tensor with `min(window, len_main, len_pos,
        len_neg)` per triple.

    Raises:
        ValueError: If `lengths` is empty, its count is not a multiple of 3,
            a length is not positive, or `window` is below 1.
    """
    if not lengths or len(lengths) % 3:
        raise ValueError(
            f"lengths must hold 3 * n entries with n >= 1, got {len(lengths)}"
        )
    if min(lengths) < 1:
        raise ValueError(f"lengths must be positive, got {list(lengths)}")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    rows = torch.tensor(lengths, dtype=torch.int64).view(3, -1)
    return rows.amin(dim=0).clamp(max=window)


def contrast_edits(
    layers: Sequence[int], alpha: float, windows: torch.Tensor
) -> dict[int, ResidualEdit]:
    """Builds the residual edits of the contrast cascade.

    On a `(3n, T, d)` stream with rows `[main, positive, negative]`, the edit
    computes `alpha * (h_pos - h_neg)` in float32 over the last
    `windows[i]` columns of triple `i` and adds that same shift to all three
    streams, so later layers see the earlier shifts in every stream.

    Args:
        layers: Layers in 0..L that receive the edit.
        alpha: Shift coefficient; 0 makes every edit the identity.
        windows: `(n,)` shifted trailing columns per triple.

    Returns:
        A mapping from layer to a residual edit; the edit ignores `info`.
    """

    def edit(h: torch.Tensor, info: ForwardInfo) -> torch.Tensor:
        n = len(windows)
        if h.shape[0] != 3 * n:
            raise ValueError(
                f"expected {3 * n} rows for {n} triples, got {h.shape[0]}"
            )
        if alpha == 0:
            return h
        s = h.float()
        columns = torch.arange(h.shape[1], device=h.device)
        inside = columns >= h.shape[1] - windows.to(h.device)[:, None]
        shift = alpha * (s[n : 2 * n] - s[2 * n :])
        shift = shift * inside[:, :, None]
        return (s + shift.repeat(3, 1, 1)).to(h.dtype)

    return {layer: edit for layer in dict.fromkeys(layers)}


def _check_layers(loaded: LoadedModel, layers: Sequence[int]) -> list[int]:
    if not layers:
        raise ValueError("layers is empty")
    for layer in layers:
        if not 0 <= layer <= loaded.n_layers:
            raise ValueError(
                f"layer {layer} is outside 0..{loaded.n_layers} "
                f"for a model with {loaded.n_layers} layers"
            )
    return list(dict.fromkeys(layers))


def _check_triples(
    prompts: Sequence[str],
    positive: Sequence[str],
    negative: Sequence[str],
) -> None:
    if not len(prompts) == len(positive) == len(negative):
        raise ValueError(
            "prompts, positive and negative must have the same length, got "
            f"{len(prompts)}, {len(positive)} and {len(negative)}"
        )


def _check_alpha(alpha: float) -> None:
    if not math.isfinite(alpha):
        raise ValueError(f"alpha must be finite, got {alpha}")


def _head(loaded: LoadedModel) -> torch.nn.Module:
    head: torch.nn.Module | None = loaded.model.get_output_embeddings()  # type: ignore[no-untyped-call]
    if head is None:
        raise ValueError("The model has no output embedding")
    return head


def _forward(
    loaded: LoadedModel,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    layers: Sequence[int],
    alpha: float,
    windows: torch.Tensor,
) -> torch.Tensor:
    """Runs the decoder under the cascade and returns the final hidden states.

    The LM head is left to the caller so it only sees the rows and columns it
    needs.
    """
    input_ids, attention_mask, position_ids = inputs
    with (
        torch.inference_mode(),
        edit_residuals(loaded, contrast_edits(layers, alpha, windows)),
    ):
        output = loaded.decoder(
            input_ids=input_ids.to(loaded.device),
            attention_mask=attention_mask.to(loaded.device),
            position_ids=position_ids.to(loaded.device),
            use_cache=False,
        )
    hidden: torch.Tensor = output.last_hidden_state
    return hidden


def contrast_logits(
    loaded: LoadedModel,
    prompts: Sequence[str],
    positive: Sequence[str],
    negative: Sequence[str],
    *,
    layers: Sequence[int],
    alpha: float,
    window: int = 8,
    bos: BosMode | None = None,
) -> torch.Tensor:
    """Returns the next-token logits of the main prompts under the cascade.

    The three streams of every triple run as rows `[main, positive,
    negative]` of one left-padded batch, so the streams are aligned by
    suffix.

    Args:
        loaded: The wrapped model.
        prompts: The main prompts.
        positive: One positive text per prompt.
        negative: One negative text per prompt.
        layers: Layers in 0..L that receive the shift.
        alpha: Shift coefficient.
        window: Maximum number of shifted trailing columns.
        bos: BOS policy passed to `loaded.encode`.

    Returns:
        An `(n, vocab)` float32 CPU tensor.

    Raises:
        ValueError: If the lists differ in length or `prompts` is empty, a
            layer is outside 0..L or `layers` is empty, `window` is below 1,
            `alpha` is not finite, or the model has no output embedding.
    """
    _check_triples(prompts, positive, negative)
    if not prompts:
        raise ValueError("prompts is empty")
    wanted = _check_layers(loaded, layers)
    _check_alpha(alpha)
    head = _head(loaded)
    batch = loaded.encode(
        [*prompts, *positive, *negative], bos=bos, padding_side="left"
    )
    windows = contrast_windows(batch.lengths, window)
    inputs = (batch.input_ids, batch.attention_mask, batch.position_ids)
    hidden = _forward(loaded, inputs, wanted, alpha, windows)
    with torch.inference_mode():
        logits: torch.Tensor = head(hidden[: len(prompts), -1])
    return logits.float().cpu().clone()
