"""Contrast vector steering over a positive and a negative prompt stream.

The three streams of a triple run as rows of one batch, and a batch-aware
residual edit adds the shift between the positive and negative streams.
"""

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Self

import numpy as np
from numpy.typing import NDArray
import torch

from tomography.control.generate import Completion
from tomography.hooks import ForwardInfo
from tomography.hooks import ResidualEdit
from tomography.hooks import capture_residuals
from tomography.hooks import edit_residuals
from tomography.model import BosMode
from tomography.model import EncodedBatch
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


def _left_padded(
    loaded: LoadedModel, rows: Sequence[Sequence[int]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Left pads id rows; positions follow the rule of `LoadedModel.encode`."""
    width = max(len(row) for row in rows)
    ids = torch.tensor(
        [[loaded.pad_id] * (width - len(row)) + list(row) for row in rows]
    )
    mask = torch.tensor(
        [[0] * (width - len(row)) + [1] * len(row) for row in rows]
    )
    return ids, mask, (mask.cumsum(dim=-1) - 1).clamp(min=0)


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


def _eos_ids(loaded: LoadedModel) -> frozenset[int]:
    eos = loaded.model.generation_config.eos_token_id
    if eos is None:
        return frozenset()
    if isinstance(eos, int):
        return frozenset((eos,))
    return frozenset(eos)


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


def _chunk_completions(
    loaded: LoadedModel,
    head: torch.nn.Module,
    streams: list[list[list[int]]],
    *,
    layers: Sequence[int],
    alpha: float,
    window: int,
    max_new_tokens: int,
) -> list[Completion]:
    """Decodes one chunk greedily; `streams` holds main, positive, negative."""
    eos = _eos_ids(loaded)
    count = len(streams[0])
    generated: list[list[int]] = [[] for _ in range(count)]
    finished = [False] * count
    active = list(range(count))
    for _ in range(max_new_tokens):
        if not active:
            break
        rows = [stream[i] for stream in streams for i in active]
        windows = contrast_windows([len(row) for row in rows], window)
        hidden = _forward(
            loaded, _left_padded(loaded, rows), layers, alpha, windows
        )
        with torch.inference_mode():
            chosen = head(hidden[: len(active), -1]).argmax(dim=-1).tolist()
        for i, token in zip(active, chosen, strict=True):
            if token in eos:
                finished[i] = True
                continue
            generated[i].append(token)
            for stream in streams:
                stream[i].append(token)
        active = [i for i in active if not finished[i]]
    return [
        Completion(
            text=str(loaded.tokenizer.decode(ids, skip_special_tokens=True)),
            token_ids=tuple(ids),
            finished=done,
        )
        for ids, done in zip(generated, finished, strict=True)
    ]


def contrast_generate(
    loaded: LoadedModel,
    prompts: Sequence[str],
    positive: Sequence[str],
    negative: Sequence[str],
    *,
    layers: Sequence[int],
    alpha: float,
    max_new_tokens: int,
    window: int = 8,
    bos: BosMode | None = None,
    batch_size: int = 8,
) -> list[Completion]:
    """Generates greedily under the contrast cascade.

    The KV cache cannot be used: the shifted window slides with the sequence,
    so every step re-runs the three streams in full. The chosen token is
    appended to all three streams of a row, so the suffix alignment follows
    the current lengths.

    Args:
        loaded: The wrapped model.
        prompts: The main prompts, in the order of the result.
        positive: One positive text per prompt.
        negative: One negative text per prompt.
        layers: Layers in 0..L that receive the shift.
        alpha: Shift coefficient.
        max_new_tokens: Maximum number of generated tokens per row.
        window: Maximum number of shifted trailing columns.
        bos: BOS policy passed to `loaded.encode`.
        batch_size: Prompts per chunk, each chunk holding `3 * batch_size`
            rows.

    Returns:
        One completion per prompt; a row stops at an end-of-sequence id of
        the generation config and the id is excluded.

    Raises:
        ValueError: If the lists differ in length, a layer is outside 0..L
            or `layers` is empty, `max_new_tokens`, `batch_size` or `window`
            is below 1, `alpha` is not finite, or the longest of the three
            streams plus `max_new_tokens` exceeds `max_position_embeddings`.
    """
    _check_triples(prompts, positive, negative)
    wanted = _check_layers(loaded, layers)
    _check_alpha(alpha)
    for name, value in (
        ("max_new_tokens", max_new_tokens),
        ("batch_size", batch_size),
        ("window", window),
    ):
        if value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}")
    if not prompts:
        return []
    head = _head(loaded)
    chunks = []
    for start in range(0, len(prompts), batch_size):
        stop = start + batch_size
        chunks.append(
            loaded.encode(
                [
                    *prompts[start:stop],
                    *positive[start:stop],
                    *negative[start:stop],
                ],
                bos=bos,
                padding_side="left",
            )
        )
    limit = getattr(loaded.model.config, "max_position_embeddings", None)
    longest = max(max(batch.lengths) for batch in chunks)
    if limit is not None and longest + max_new_tokens > limit:
        raise ValueError(
            f"The longest stream has {longest} tokens and max_new_tokens is "
            f"{max_new_tokens}, which exceeds max_position_embeddings "
            f"({limit})"
        )
    result: list[Completion] = []
    for batch in chunks:
        width = batch.input_ids.shape[1]
        rows = [
            ids[width - length :]
            for ids, length in zip(
                batch.input_ids.tolist(), batch.lengths, strict=True
            )
        ]
        count = len(rows) // 3
        result.extend(
            _chunk_completions(
                loaded,
                head,
                [rows[:count], rows[count : 2 * count], rows[2 * count :]],
                layers=wanted,
                alpha=alpha,
                window=window,
                max_new_tokens=max_new_tokens,
            )
        )
    return result


def _token_ids(loaded: LoadedModel, texts: Sequence[str]) -> list[list[int]]:
    ids: list[list[int]] = loaded.tokenizer(
        list(texts), add_special_tokens=False
    )["input_ids"]
    return ids


def contrast_logprobs(
    loaded: LoadedModel,
    prompts: Sequence[str],
    continuations: Sequence[str],
    positive: Sequence[str],
    negative: Sequence[str],
    *,
    layers: Sequence[int],
    alpha: float,
    bos: BosMode | None = None,
    batch_size: int = 8,
) -> NDArray[np.float32]:
    """Scores continuations of the main prompts under the contrast cascade.

    The rows of a triple are `bos + prompt + continuation`, `bos + positive
    + continuation` and `bos + negative + continuation`, each part
    tokenized separately. The shifted window is `len(continuation) + 1`
    columns: the continuation plus the last prompt column, whose output
    predicts the first continuation token.

    Args:
        loaded: The wrapped model.
        prompts: The main prompts.
        continuations: One continuation to score per prompt.
        positive: One positive text per prompt.
        negative: One negative text per prompt.
        layers: Layers in 0..L that receive the shift.
        alpha: Shift coefficient.
        bos: BOS policy; the prefix is `loaded.bos_id` iff it is `prepend`.
        batch_size: Triples per forward.

    Returns:
        An `(n,)` float32 array with the sum of the continuation token
        log-probabilities of the main rows.

    Raises:
        ValueError: If the lists differ in length, a continuation has no
            tokens, `batch_size` is below 1, a layer is outside 0..L or
            `layers` is empty, `alpha` is not finite, a stream has no token
            before its continuation, or BOS is requested without an id.
    """
    _check_triples(prompts, positive, negative)
    if len(continuations) != len(prompts):
        raise ValueError(
            "continuations and prompts must have the same length, got "
            f"{len(continuations)} and {len(prompts)}"
        )
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    wanted = _check_layers(loaded, layers)
    _check_alpha(alpha)
    mode = loaded.default_bos if bos is None else bos
    prefix: list[int] = []
    if mode == "prepend":
        if loaded.bos_id is None:
            raise ValueError(
                "bos='prepend' needs a bos id and this model has none"
            )
        prefix = [loaded.bos_id]
    if not prompts:
        return np.zeros(0, np.float32)
    tails = _token_ids(loaded, continuations)
    if not all(tails):
        raise ValueError("a continuation has no tokens")
    streams = [
        _token_ids(loaded, texts) for texts in (prompts, positive, negative)
    ]
    if not all(prefix or ids for texts in streams for ids in texts):
        raise ValueError(
            "a stream has no context token before the continuation"
        )
    head = _head(loaded)
    scores: list[float] = []
    for start in range(0, len(prompts), batch_size):
        part = slice(start, start + batch_size)
        chunk_tails = tails[part]
        rows = [
            [*prefix, *ids, *tail]
            for texts in streams
            for ids, tail in zip(texts[part], chunk_tails, strict=True)
        ]
        own = torch.tensor([len(tail) + 1 for tail in chunk_tails])
        windows = torch.minimum(
            contrast_windows([len(row) for row in rows], int(own.max())), own
        )
        inputs = _left_padded(loaded, rows)
        hidden = _forward(loaded, inputs, wanted, alpha, windows)
        width = hidden.shape[1]
        with torch.inference_mode():
            for i, tail in enumerate(chunk_tails):
                logits: torch.Tensor = head(
                    hidden[i, width - len(tail) - 1 : width - 1]
                )
                logp = torch.log_softmax(logits.float(), dim=-1)
                targets = torch.tensor(tail, device=logp.device)
                scores.append(
                    float(logp[torch.arange(len(tail)), targets].sum())
                )
    return np.asarray(scores, dtype=np.float32)


def _token_index(info: ForwardInfo, h: torch.Tensor) -> torch.Tensor:
    """Returns the `(batch, seq)` token index of every column of `h`."""
    rows, columns = h.shape[:2]
    if info.position_ids is not None:
        index = info.position_ids[:, -columns:]
    elif info.attention_mask is not None:
        positions = info.attention_mask.cumsum(dim=-1) - 1
        index = positions[:, -columns:].clamp(min=0)
    else:
        index = torch.arange(columns).expand(rows, columns)
    return index.to(h.device)


def _prompt_limit(
    prompt_lengths: int | Sequence[int], h: torch.Tensor
) -> torch.Tensor:
    rows = h.shape[0]
    if isinstance(prompt_lengths, int):
        return torch.full((rows,), prompt_lengths, device=h.device)
    if len(prompt_lengths) != rows:
        raise ValueError(
            f"prompt_lengths has {len(prompt_lengths)} entries for {rows} rows"
        )
    return torch.tensor(prompt_lengths, device=h.device)


@dataclass(frozen=True, eq=False)
class PromptDifference:
    """ActAdd-style controller built from the difference of two prompts.

    The difference of the residual streams of a positive and a negative
    prompt keeps its raw magnitude, and `coeff` multiplies it. The edit adds
    `coeff * vectors[layer][j]` at the tokens of token index `j < k`, so it is
    position local and works under the KV cache.

    Attributes:
        vectors: Maps a layer to a `(k, d_model)` float32 per-position
            difference.
        coeff: Multiplier of the raw difference.
    """

    vectors: Mapping[int, torch.Tensor]
    coeff: float

    def __post_init__(self) -> None:
        """Validates the vectors and the coefficient.

        Raises:
            ValueError: If `vectors` is empty, a vector is not 2-D, not
                finite or all zero, the widths differ, or `coeff` is not
                finite.
        """
        if not self.vectors:
            raise ValueError("vectors is empty")
        if not math.isfinite(self.coeff):
            raise ValueError(f"coeff must be finite, got {self.coeff}")
        for layer, vector in self.vectors.items():
            if vector.ndim != 2:
                raise ValueError(
                    f"vectors[{layer}] must be 2-D, got {vector.ndim}-D"
                )
            if not bool(torch.isfinite(vector).all()):
                raise ValueError(f"vectors[{layer}] is not finite")
            if not bool(vector.any()):
                raise ValueError(f"vectors[{layer}] is all zero")
        widths = {vector.shape[1] for vector in self.vectors.values()}
        if len(widths) > 1:
            raise ValueError(f"vectors have different widths: {sorted(widths)}")

    @classmethod
    def from_prompts(
        cls,
        loaded: LoadedModel,
        positive: str,
        negative: str,
        layers: Sequence[int],
        *,
        coeff: float,
        bos: BosMode | None = None,
    ) -> Self:
        """Builds the controller from a positive and a negative prompt.

        Args:
            loaded: The wrapped model.
            positive: The positive prompt.
            negative: The negative prompt.
            layers: Layers in 0..L that get a vector.
            coeff: Multiplier of the raw difference.
            bos: BOS policy passed to `loaded.encode`.

        Returns:
            The controller; each vector holds the float32 differences of the
            first `min(len_pos, len_neg)` positions. Under a BOS the
            position-0 difference is set to exactly zero.

        Raises:
            ValueError: If `layers` is empty or outside 0..L, or the prompts
                have identical token ids over the first `min(len_pos,
                len_neg)` positions (equal, or one a token prefix of the
                other), so the difference would be zero.
        """
        wanted = _check_layers(loaded, layers)
        batch = loaded.encode([positive, negative], bos=bos)
        k = min(batch.lengths)
        rows = batch.input_ids.tolist()
        if rows[0][:k] == rows[1][:k]:
            raise ValueError(
                f"the prompts have identical token ids over their first {k}"
                " positions, so the difference is zero"
            )
        with torch.inference_mode(), capture_residuals(loaded, wanted) as seen:
            loaded.decoder(
                input_ids=batch.input_ids.to(loaded.device),
                attention_mask=batch.attention_mask.to(loaded.device),
                position_ids=batch.position_ids.to(loaded.device),
                use_cache=False,
            )
        vectors: dict[int, torch.Tensor] = {}
        for layer in wanted:
            stream = seen[layer].float()
            vector = (stream[0, :k] - stream[1, :k]).cpu().clone()
            if batch.has_bos:
                vector[0] = 0
            vectors[layer] = vector
        return cls(vectors, coeff)

    def edits(
        self, prompt_lengths: int | Sequence[int] | None = None
    ) -> dict[int, ResidualEdit]:
        """Builds the residual edits.

        Args:
            prompt_lengths: Token count of each row's prompt, or one int for
                all rows; edited indices stay below it. None applies only
                the `j < k` rule.

        Returns:
            A mapping from layer to a residual edit.

        Raises:
            ValueError: If a prompt length is below 1. A sequence whose size
                differs from the row count raises when the edit runs.
        """
        values = (
            [prompt_lengths]
            if isinstance(prompt_lengths, int)
            else list(prompt_lengths or [])
        )
        if any(value < 1 for value in values):
            raise ValueError(
                f"prompt_lengths must be >= 1, got {prompt_lengths}"
            )

        def edit_for(vector: torch.Tensor) -> ResidualEdit:
            def edit(h: torch.Tensor, info: ForwardInfo) -> torch.Tensor:
                if self.coeff == 0:
                    return h
                index = _token_index(info, h)
                edited = index < vector.shape[0]
                mask = info.attention_mask
                if mask is not None:
                    edited = edited & (mask[:, -h.shape[1] :] > 0).to(h.device)
                if prompt_lengths is not None:
                    limit = _prompt_limit(prompt_lengths, h)
                    edited = edited & (index < limit[:, None])
                rows = vector.to(h.device)[index.clamp(max=vector.shape[0] - 1)]
                return (h.float() + self.coeff * rows * edited[..., None]).to(
                    h.dtype
                )

            return edit

        return {layer: edit_for(vec) for layer, vec in self.vectors.items()}

    def __call__(self, batch: EncodedBatch) -> dict[int, ResidualEdit]:
        """Returns `edits(batch.lengths)`, the `generate` edit factory."""
        return self.edits(batch.lengths)
