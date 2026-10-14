"""Batched text generation with optional residual-stream edits."""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import contextlib
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch

from tomography.hooks import ResidualEdit
from tomography.hooks import edit_residuals
from tomography.model import BosMode
from tomography.model import EncodedBatch
from tomography.model import LoadedModel

EditFactory = Callable[[EncodedBatch], Mapping[int, ResidualEdit]]


@dataclass(frozen=True)
class Completion:
    """One generated continuation.

    Attributes:
        text: `decode(token_ids, skip_special_tokens=True)`, not stripped.
        token_ids: Generated ids up to but excluding the first
            end-of-sequence id.
        finished: Whether an end-of-sequence id was generated within
            `max_new_tokens`.
    """

    text: str
    token_ids: tuple[int, ...]
    finished: bool


def _check_arguments(
    *,
    max_new_tokens: int,
    temperature: float | None,
    top_p: float,
    top_k: int,
    num_samples: int,
    batch_size: int,
) -> None:
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")
    if temperature is not None and not (
        math.isfinite(temperature) and temperature > 0
    ):
        raise ValueError(
            f"temperature must be finite and positive, got {temperature}"
        )
    if not 0 < top_p <= 1:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if top_k < 0:
        raise ValueError(f"top_k must be >= 0, got {top_k}")
    if temperature is None and (num_samples > 1 or top_p != 1 or top_k != 0):
        raise ValueError("num_samples > 1, top_p and top_k need a temperature")


def _eos_ids(loaded: LoadedModel) -> frozenset[int]:
    eos = loaded.model.generation_config.eos_token_id
    if eos is None:
        return frozenset()
    if isinstance(eos, int):
        return frozenset((eos,))
    return frozenset(eos)


def _cut(
    loaded: LoadedModel, ids: list[int], eos: frozenset[int]
) -> Completion:
    """Cuts a generated row before its first end-of-sequence id."""
    end = next((i for i, token in enumerate(ids) if token in eos), None)
    kept = ids if end is None else ids[:end]
    return Completion(
        text=str(loaded.tokenizer.decode(kept, skip_special_tokens=True)),
        token_ids=tuple(kept),
        finished=end is not None,
    )


def generate(
    loaded: LoadedModel,
    prompts: Sequence[str],
    *,
    max_new_tokens: int,
    temperature: float | None = None,
    top_p: float = 1.0,
    top_k: int = 0,
    num_samples: int = 1,
    seed: int = 0,
    edits: EditFactory | None = None,
    bos: BosMode | None = None,
    batch_size: int = 8,
) -> list[list[Completion]]:
    """Generates continuations of prompts in left-padded batches.

    Each prompt is repeated `num_samples` times (prompt-major) and chunks of
    `batch_size` prompts run one Hugging Face call each. Sampling is seeded
    per chunk inside `fork_rng`, so results depend on `(seed, batch_size,
    prompt order)` and the global RNG state is left unchanged. Generation
    uses the KV cache, so after the prefill an edit sees one token per row
    and must be position local.

    Args:
        loaded: The wrapped model.
        prompts: The prompts, in the order of the result.
        max_new_tokens: Maximum number of generated tokens per row.
        temperature: Sampling temperature; None means greedy decoding.
        top_p: Nucleus mass in (0, 1]; needs a temperature when not 1.
        top_k: Number of candidates kept; 0 disables it.
        num_samples: Completions per prompt; above 1 needs a temperature.
        seed: Base seed of the per-chunk sampling generators.
        edits: Factory called once per chunk with the encoded rows; its
            residual edits are active during that chunk's generation.
        bos: BOS policy passed to `loaded.encode`.
        batch_size: Prompts per Hugging Face call.

    Returns:
        One list per prompt with `num_samples` completions each.

    Raises:
        ValueError: If an argument is out of range, or a prompt plus
            `max_new_tokens` exceeds the position limit of the model.
    """
    _check_arguments(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_samples=num_samples,
        batch_size=batch_size,
    )
    if not prompts:
        return []
    chunks = [
        loaded.encode(
            [
                prompt
                for prompt in prompts[start : start + batch_size]
                for _ in range(num_samples)
            ],
            bos=bos,
            padding_side="left",
        )
        for start in range(0, len(prompts), batch_size)
    ]
    limit = getattr(loaded.model.config, "max_position_embeddings", None)
    if limit is not None:
        longest = max(max(batch.lengths) for batch in chunks)
        if longest + max_new_tokens > limit:
            raise ValueError(
                f"The longest prompt has {longest} tokens and "
                f"max_new_tokens is {max_new_tokens}, which exceeds "
                f"max_position_embeddings ({limit})"
            )
    eos = _eos_ids(loaded)
    sampling: dict[str, Any] = {"do_sample": False}
    if temperature is not None:
        sampling = {
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }
    devices = [loaded.device] if loaded.device.type == "cuda" else []
    result: list[list[Completion]] = []
    for k, batch in enumerate(chunks):
        residual_edits = {} if edits is None else edits(batch)
        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.random.fork_rng(devices=devices))
            stack.enter_context(torch.inference_mode())
            if residual_edits:
                stack.enter_context(edit_residuals(loaded, residual_edits))
            torch.manual_seed(
                int(np.random.SeedSequence([seed, k]).generate_state(1)[0])
            )
            output = loaded.model.generate(  # type: ignore[operator]
                input_ids=batch.input_ids.to(loaded.device),
                attention_mask=batch.attention_mask.to(loaded.device),
                max_new_tokens=max_new_tokens,
                pad_token_id=loaded.pad_id,
                **sampling,
            )
        generated = output[:, batch.input_ids.shape[1] :].tolist()
        rows = [_cut(loaded, ids, eos) for ids in generated]
        result.extend(
            rows[start : start + num_samples]
            for start in range(0, len(rows), num_samples)
        )
    return result
