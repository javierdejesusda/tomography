"""LoRRA: low-rank representation adaptation (Algorithm 1 of the paper).

A LoRA adapter is trained so that the residual stream of a plain prompt moves
to where the frozen model puts it under a contrast pair,
`h + alpha * (h_pos - h_neg)`, and is then used with no hook at all. This
module is the library: batch layout, loss, adapter, training loop, held-out
loss and persistence. It evaluates nothing about honesty.

Layer l follows the project convention of `hooks.py`: l = 0 is the embedding
output and l >= 1 the output of block l - 1 (zero-based), so an adapter for
the target layers (l_1, ..., l_k) covers the blocks 0..l_k - 1.
"""

from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
import contextlib
from dataclasses import dataclass
from dataclasses import field
import functools
import itertools
import math
from typing import Literal
from typing import ParamSpec
from typing import TypeVar
from typing import get_args

import numpy as np
from numpy.typing import NDArray
from peft import LoraConfig
from peft import PeftModel
from peft import get_peft_model
from peft import get_peft_model_state_dict
from peft import set_peft_model_state_dict
from peft.tuners.lora import LoraLayer
import torch

from tomography.hooks import capture_residuals
from tomography.model import BosMode
from tomography.model import LoadedModel
from tomography.templates import ChatFormat

P = ParamSpec("P")
R = TypeVar("R")
LossForm = Literal["mean_norm", "sum_squared"]
Pair = tuple[str, str]


@dataclass(frozen=True)
class LorraSettings:
    """Everything that defines one LoRRA training run.

    The defaults follow the reference implementation (alpha 5, LoRA rank 8,
    lr 3e-4, batch 16, 70 steps, a 64-token response window).

    Attributes:
        layers: Target layers, strictly increasing, each in 1..L.
        alpha: Strength of the contrast term in the target.
        pos_control: Control text appended to the user turn of the positive
            stream.
        neg_control: Control text appended to the user turn of the negative
            stream.
        rank: LoRA rank.
        lora_alpha: LoRA scaling numerator (the scale is `lora_alpha / rank`).
        dropout: LoRA dropout, applied in training steps only.
        lr: AdamW learning rate, constant, with weight decay 0.
        batch_size: Pairs per step; a forward holds `3 * batch_size` rows.
        steps: Optimizer steps.
        max_prompt_tokens: Prompt window P; prompts are left padded to it.
        max_response_tokens: Response window R; responses are truncated to
            it and right padded.
        loss: `mean_norm` is the reference form (mean unsquared L2 norm per
            slot); `sum_squared` is the paper's (squared norm summed over
            tokens, averaged over the batch).
        gradient_checkpointing: Recompute block activations in backward.
        seed: Seed of the batch order and of the adapter dropout.
    """

    layers: tuple[int, ...]
    alpha: float = 5.0
    pos_control: str = "Give a truthful answer."
    neg_control: str = "Give an untruthful answer."
    rank: int = 8
    lora_alpha: int = 16
    dropout: float = 0.05
    lr: float = 3e-4
    batch_size: int = 16
    steps: int = 70
    max_prompt_tokens: int = 256
    max_response_tokens: int = 64
    loss: LossForm = "mean_norm"
    gradient_checkpointing: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        """Checks the layers, the counts and the optimizer arguments.

        Raises:
            ValueError: If `layers` is empty, not strictly increasing or has
                a layer below 1, `rank`, `lora_alpha`, `batch_size`, `steps`
                or a window is below 1, `dropout` is outside [0, 1), `lr` is
                not finite and positive, or `loss` is unknown.
        """
        if not self.layers:
            raise ValueError("layers is empty")
        if self.layers[0] < 1:
            raise ValueError(f"layers must be >= 1, got {self.layers}")
        if any(a >= b for a, b in itertools.pairwise(self.layers)):
            raise ValueError(f"layers must be increasing, got {self.layers}")
        for name in (
            "rank",
            "lora_alpha",
            "batch_size",
            "steps",
            "max_prompt_tokens",
            "max_response_tokens",
        ):
            if getattr(self, name) < 1:
                raise ValueError(
                    f"{name} must be >= 1, got {getattr(self, name)}"
                )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not (math.isfinite(self.lr) and self.lr > 0):
            raise ValueError(f"lr must be finite and positive, got {self.lr}")
        if self.loss not in get_args(LossForm):
            raise ValueError(
                f"unknown loss {self.loss!r}, expected one of "
                f"{get_args(LossForm)}"
            )


def default_layers(n_layers: int) -> tuple[int, ...]:
    """Returns the reference target layers scaled to a model depth.

    Every second layer from `round(10 n / 32)` to `round(20 n / 32)`: 10, 12,
    .., 20 for 32 layers (the reference 7B setting), 9, 11, .., 17 for the 28
    layers of Qwen3-1.7B and 4, 6, 8 for GPT-2 small.

    Args:
        n_layers: Number of decoder blocks (L).

    Returns:
        The layers, in 1..L, increasing.

    Raises:
        ValueError: If `n_layers` is below 1.
    """
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    first = max(round(10 * n_layers / 32), 1)
    last = max(round(20 * n_layers / 32), first)
    return tuple(range(first, last + 1, 2))


@dataclass(frozen=True, eq=False)
class LorraBatch:
    """One LoRRA batch: the original, positive and negative streams.

    Rows are ordered `[orig_0..orig_{B-1}, pos_0.., neg_0..]`. Every row is
    the prompt left padded to P columns followed by the response right padded
    to R columns, with the same response ids in the three rows of a triple,
    so the response columns line up across the streams and column `P - 1`
    holds the last real prompt token of every row.

    Attributes:
        input_ids: `(3B, P + R)` int64 token ids.
        attention_mask: `(3B, P + R)` int64, 1 on real tokens.
        position_ids: `(3B, P + R)` int64, the mask cumulative sum minus
            one, clamped at 0 (HF derives no positions from padding).
        window_start: `P - 1`, the first column the loss reads.
        response_mask: `(B, R + 1)` float32, 1 on the last prompt column and
            on the real response tokens.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    window_start: int
    response_mask: torch.Tensor


def _bos_prefix(loaded: LoadedModel, bos: BosMode | None) -> list[int]:
    mode = loaded.default_bos if bos is None else bos
    if mode != "prepend":
        return []
    if loaded.bos_id is None:
        raise ValueError("bos='prepend' needs a bos id and this model has none")
    return [loaded.bos_id]


def _tokenize(
    loaded: LoadedModel,
    pairs: Sequence[Pair],
    chat: ChatFormat,
    settings: LorraSettings,
    bos: BosMode | None,
) -> tuple[list[list[list[int]]], list[list[int]]]:
    """Tokenizes the three prompts and the response of every pair.

    Returns:
        Per pair, the original, positive and negative prompt ids (BOS
        included under the `prepend` policy) and the response ids.
    """
    prefix = _bos_prefix(loaded, bos)
    users = [
        user
        for instruction, _ in pairs
        for user in (
            instruction,
            f"{instruction} {settings.pos_control}",
            f"{instruction} {settings.neg_control}",
        )
    ]
    prompts = loaded.tokenizer(
        [chat.render_turn(user)[0] for user in users], add_special_tokens=False
    )["input_ids"]
    responses = loaded.tokenizer(
        [response for _, response in pairs], add_special_tokens=False
    )["input_ids"]
    triples = [
        [prefix + list(ids) for ids in prompts[3 * i : 3 * i + 3]]
        for i in range(len(pairs))
    ]
    return triples, [list(ids) for ids in responses]


def fit_rows(
    loaded: LoadedModel,
    pairs: Sequence[Pair],
    chat: ChatFormat,
    settings: LorraSettings,
    *,
    bos: BosMode | None = None,
) -> tuple[list[Pair], int]:
    """Keeps the pairs that `build_batch` can lay out.

    A pair stays when its response has at least one token and its three
    prompts (BOS included) each have between 1 and `max_prompt_tokens`
    tokens.

    Args:
        loaded: The wrapped model.
        pairs: `(instruction, response)` pairs.
        chat: How the user turn becomes a prompt.
        settings: The run settings.
        bos: BOS policy; None uses the model default.

    Returns:
        The pairs that fit, in their original order, and how many were
        dropped.

    Raises:
        ValueError: If BOS is requested and the model has no id.
    """
    if not pairs:
        return [], 0
    triples, responses = _tokenize(loaded, pairs, chat, settings, bos)
    window = settings.max_prompt_tokens
    kept = [
        pair
        for pair, triple, response in zip(
            pairs, triples, responses, strict=True
        )
        if response and all(0 < len(prompt) <= window for prompt in triple)
    ]
    return kept, len(pairs) - len(kept)


def build_batch(
    loaded: LoadedModel,
    pairs: Sequence[Pair],
    chat: ChatFormat,
    settings: LorraSettings,
    *,
    bos: BosMode | None = None,
) -> LorraBatch:
    """Lays out the original, positive and negative rows of a batch.

    The three prompts of a pair are the chat rendering of the instruction,
    of the instruction followed by `pos_control` and of the instruction
    followed by `neg_control`; the controls end the user turn. Prompts and
    responses are tokenized apart, without special tokens, so a template
    chat carries its own. The response is truncated to its first
    `max_response_tokens` tokens.

    Args:
        loaded: The wrapped model.
        pairs: `(instruction, response)` pairs; call `fit_rows` first.
        chat: How the user turn becomes a prompt.
        settings: The run settings.
        bos: BOS policy; None uses the model default.

    Returns:
        The batch, with `3 * len(pairs)` rows.

    Raises:
        ValueError: If there are no pairs, a response has no tokens, a prompt
            has no tokens or more than `max_prompt_tokens`, or BOS is
            requested and the model has no id.
    """
    if not pairs:
        raise ValueError("no pairs")
    triples, responses = _tokenize(loaded, pairs, chat, settings, bos)
    window, width = settings.max_prompt_tokens, settings.max_response_tokens
    ids: list[list[int]] = []
    mask: list[list[int]] = []
    for stream in range(3):
        for index, (triple, response) in enumerate(
            zip(triples, responses, strict=True)
        ):
            prompt = triple[stream]
            if not response:
                raise ValueError(f"the response of pair {index} has no tokens")
            if not 0 < len(prompt) <= window:
                raise ValueError(
                    f"the prompt of pair {index} has {len(prompt)} tokens, "
                    f"outside 1..{window}"
                )
            answer = response[:width]
            left, right = window - len(prompt), width - len(answer)
            ids.append(
                [loaded.pad_id] * left
                + prompt
                + answer
                + [loaded.pad_id] * right
            )
            mask.append(
                [0] * left + [1] * (len(prompt) + len(answer)) + [0] * right
            )
    attention_mask = torch.tensor(mask, dtype=torch.int64)
    lengths = [min(len(response), width) for response in responses]
    response_mask = torch.tensor(
        [[1.0] * (1 + n) + [0.0] * (width - n) for n in lengths],
        dtype=torch.float32,
    )
    return LorraBatch(
        input_ids=torch.tensor(ids, dtype=torch.int64),
        attention_mask=attention_mask,
        position_ids=(attention_mask.cumsum(dim=-1) - 1).clamp(min=0),
        window_start=window - 1,
        response_mask=response_mask,
    )


def layer_losses(
    current: Mapping[int, torch.Tensor],
    frozen: Mapping[int, torch.Tensor],
    batch: LorraBatch,
    settings: LorraSettings,
) -> dict[int, torch.Tensor]:
    """Scores the adapted residuals against the contrast targets.

    The loss reads the `R + 1` columns from `batch.window_start`: the last
    prompt token, whose state predicts the first response token, and the real
    response tokens. The target of a layer comes from the frozen streams,
    `(h_orig + alpha * (h_pos - h_neg)) * mask`, and the loss from
    `h * mask - target` over the original rows, all in float32. `mean_norm`
    averages the unsquared L2 norm over all `B * (R + 1)` slots, padded slots
    counting 0; `sum_squared` sums the squared norm over tokens and averages
    over the batch.

    Args:
        current: Layer to `(B, P + R, d)` residuals of the adapted model on
            the original rows, with autograd history.
        frozen: Layer to `(3B, P + R, d)` residuals of the frozen model on
            all rows, in the batch order.
        batch: The batch both were computed on.
        settings: The run settings; their layers are scored.

    Returns:
        Layer to the scalar loss.

    Raises:
        ValueError: If a target layer is missing from `current` or `frozen`,
            or their row counts or sequence length do not fit the batch.
    """
    mask = batch.response_mask
    n_pairs, width = mask.shape
    start = batch.window_start
    losses: dict[int, torch.Tensor] = {}
    for layer in settings.layers:
        for name, residuals in (("current", current), ("frozen", frozen)):
            if layer not in residuals:
                raise ValueError(f"layer {layer} is missing from {name}")
        now, then = current[layer], frozen[layer]
        if now.shape[0] != n_pairs or then.shape[0] != 3 * n_pairs:
            raise ValueError(
                f"layer {layer} has {now.shape[0]} adapted and "
                f"{then.shape[0]} frozen rows, expected {n_pairs} and "
                f"{3 * n_pairs}"
            )
        if now.shape[1] != start + width or then.shape[1] != start + width:
            raise ValueError(
                f"layer {layer} has {now.shape[1]} and {then.shape[1]} "
                f"positions, expected {start + width}"
            )
        slots = mask.to(now.device).unsqueeze(-1)
        h = now[:, start:].float()
        ref = then[:, start:].float()
        target = (
            ref[:n_pairs]
            + settings.alpha * (ref[n_pairs : 2 * n_pairs] - ref[2 * n_pairs :])
        ) * slots
        diff = h * slots - target
        if settings.loss == "mean_norm":
            losses[layer] = torch.norm(diff, dim=-1).nanmean()
        else:
            losses[layer] = diff.pow(2).sum(dim=(-1, -2)).mean()
    return losses


def lorra_loss(
    current: Mapping[int, torch.Tensor],
    frozen: Mapping[int, torch.Tensor],
    batch: LorraBatch,
    settings: LorraSettings,
) -> torch.Tensor:
    """Sums `layer_losses` over the target layers.

    Args:
        current: As in `layer_losses`.
        frozen: As in `layer_losses`.
        batch: As in `layer_losses`.
        settings: As in `layer_losses`.

    Returns:
        The scalar total loss.

    Raises:
        ValueError: As in `layer_losses`.
    """
    losses = layer_losses(current, frozen, batch, settings)
    return torch.stack(list(losses.values())).sum()


@dataclass(eq=False)
class LorraAdapter:
    """A LoRA adapter attached in place to a loaded model.

    PEFT injects the LoRA layers into the modules of `loaded.model`, so
    `loaded.model`, `loaded.blocks` and every hook keep working and the model
    now runs the adapter. `disabled` gives back the base model exactly.

    Attributes:
        peft_model: The PEFT wrapper of `loaded.model`.
        settings: The settings the adapter was attached with.
        blocks: The adapted block indices, `0..max(layers) - 1`.
    """

    peft_model: PeftModel
    settings: LorraSettings
    blocks: tuple[int, ...]
    _training_depth: int = field(default=0, init=False, repr=False)

    @contextlib.contextmanager
    def disabled(self) -> Iterator[None]:
        """Runs the model as the base model, restoring the adapter on exit.

        Yields:
            None.
        """
        with self.peft_model.disable_adapter():
            yield

    def gated(self, fn: Callable[P, R], *, enabled: bool) -> Callable[P, R]:
        """Wraps a callable so that it runs with the adapter on or off.

        Args:
            fn: Any callable that runs the model.
            enabled: True returns `fn` unchanged, for an adapter that is on;
                False runs every call inside `disabled`.

        Returns:
            The wrapped callable, with the signature of `fn`.
        """
        if enabled:
            return fn

        @functools.wraps(fn)
        def run(*args: P.args, **kwargs: P.kwargs) -> R:
            with self.disabled():
                return fn(*args, **kwargs)

        return run

    def trainable(self) -> list[torch.nn.Parameter]:
        """Returns the parameters that training updates (the LoRA weights)."""
        return [p for p in self.peft_model.parameters() if p.requires_grad]

    def n_trainable(self) -> int:
        """Returns the number of trainable scalars."""
        return sum(p.numel() for p in self.trainable())

    def state(self) -> dict[str, NDArray[np.float32]]:
        """Returns the adapter weights as float32 NumPy copies.

        Returns:
            PEFT's adapter state dict keyed by parameter name.
        """
        return {
            key: tensor.detach()
            .to(device="cpu", dtype=torch.float32, copy=True)
            .numpy()
            for key, tensor in get_peft_model_state_dict(
                self.peft_model, save_embedding_layers=False
            ).items()
        }

    def load_state(self, state: Mapping[str, NDArray[np.float32]]) -> None:
        """Loads adapter weights saved by `state`, all or nothing.

        Each array is cast to the dtype and device of its parameter.

        Args:
            state: A mapping with exactly the keys of `state()`.

        Raises:
            ValueError: If the keys differ from the keys of this adapter
                (the message names a few) or an array has the wrong shape.
        """
        current = get_peft_model_state_dict(
            self.peft_model, save_embedding_layers=False
        )
        missing = sorted(set(current) - set(state))
        unexpected = sorted(set(state) - set(current))
        if missing or unexpected:
            raise ValueError(
                f"the adapter state keys differ: missing {missing[:3]}, "
                f"unexpected {unexpected[:3]}"
            )
        for key, target in current.items():
            if tuple(state[key].shape) != tuple(target.shape):
                raise ValueError(
                    f"{key} has shape {tuple(state[key].shape)}, "
                    f"expected {tuple(target.shape)}"
                )
        set_peft_model_state_dict(
            self.peft_model,
            {
                key: torch.from_numpy(np.array(array, dtype=np.float32))
                for key, array in state.items()
            },
        )

    @contextlib.contextmanager
    def training(self) -> Iterator[None]:
        """Puts the model in the mode of a training step, then back to eval.

        The base model stays in eval mode and only the adapter dropout layers
        train, because the base dropout of GPT-2 would change the
        distribution of the targets. With gradient checkpointing the whole
        model trains, since HF checkpoints only in train mode.

        The block is re-entrant: the model returns to eval mode when the
        outermost block exits. A step keeps it open through its backward,
        because checkpointing recomputes the forward there and the dropout
        must behave as it did in the original forward.

        Yields:
            None.
        """
        self._training_depth += 1
        try:
            if self._training_depth == 1:
                if self.settings.gradient_checkpointing:
                    self.peft_model.train()
                else:
                    for module in self.peft_model.modules():
                        if isinstance(module, LoraLayer):
                            module.lora_dropout.train()
            yield
        finally:
            self._training_depth -= 1
            if self._training_depth == 0:
                self.peft_model.eval()


def attach_adapter(
    loaded: LoadedModel, settings: LorraSettings
) -> LorraAdapter:
    """Attaches a LoRA adapter to the blocks below the deepest target layer.

    A target layer `l` is the output of block `l - 1`, so the adapter covers
    the blocks `0..max(layers) - 1`. GPT-2 adapts the fused `c_attn` (query,
    key and value together) and Qwen3 adapts `q_proj` and `v_proj`. The
    model is wrapped in place and a fresh adapter has `lora_B = 0`, so the
    model is bit-identical to the base until training. The random `lora_A`
    initialization is drawn from `settings.seed` without touching the global
    random state. Attach a model once: a second call on the same object
    nests adapters. The model is left in eval mode and the adapter weights
    are float32 on any base dtype.

    Args:
        loaded: The wrapped model; it now runs the adapter.
        settings: The run settings.

    Returns:
        The adapter.

    Raises:
        ValueError: If a target layer is above the model depth, or gradient
            checkpointing is on for a base model with an active
            `nn.Dropout` (train mode would activate it).
    """
    top = settings.layers[-1]
    if top > loaded.n_layers:
        raise ValueError(
            f"layer {top} is above the {loaded.n_layers} layers of the model"
        )
    if settings.gradient_checkpointing and any(
        isinstance(module, torch.nn.Dropout) and module.p > 0
        for module in loaded.model.modules()
    ):
        raise ValueError(
            "gradient checkpointing runs the model in train mode, which "
            "would activate the dropout of this base model"
        )
    blocks = tuple(range(top))
    config = LoraConfig(
        r=settings.rank,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.dropout,
        bias="none",
        target_modules=list(loaded.spec.lora_targets),
        fan_in_fan_out=loaded.spec.lora_fan_in_fan_out,
        layers_to_transform=list(blocks),
    )
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(settings.seed)
        peft_model = get_peft_model(loaded.model, config)
    if settings.gradient_checkpointing:
        loaded.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    peft_model.eval()
    return LorraAdapter(peft_model, settings, blocks)


@dataclass(frozen=True)
class LorraTrace:
    """What a training run records.

    Attributes:
        losses: The summed loss of every step.
        peak_memory_mb: `torch.cuda.max_memory_allocated` over the loop in
            MiB, or None on CPU.
    """

    losses: tuple[float, ...]
    peak_memory_mb: float | None


def _residual_pair(
    loaded: LoadedModel,
    adapter: LorraAdapter,
    batch: LorraBatch,
    *,
    training: bool,
) -> tuple[Mapping[int, torch.Tensor], Mapping[int, torch.Tensor]]:
    """Runs the frozen pass over all rows, then the adapted pass on the first B.

    The passes go through the decoder, without the LM head, with
    `use_cache=False` and the explicit positions of the batch.

    Returns:
        The adapted residuals `(B, P + R, d)` and the frozen residuals
        `(3B, P + R, d)` at the target layers.
    """
    layers = adapter.settings.layers
    n_pairs = batch.response_mask.shape[0]
    inputs = {
        "input_ids": batch.input_ids.to(loaded.device),
        "attention_mask": batch.attention_mask.to(loaded.device),
        "position_ids": batch.position_ids.to(loaded.device),
    }
    with (
        torch.no_grad(),
        adapter.disabled(),
        capture_residuals(loaded, layers) as frozen,
    ):
        loaded.decoder(**inputs, use_cache=False)
    mode = adapter.training() if training else contextlib.nullcontext()
    with mode, capture_residuals(loaded, layers, detach=False) as current:
        loaded.decoder(
            **{name: tensor[:n_pairs] for name, tensor in inputs.items()},
            use_cache=False,
        )
    return current, frozen


def step_loss(
    loaded: LoadedModel, adapter: LorraAdapter, batch: LorraBatch
) -> torch.Tensor:
    """Computes the loss of one training step, keeping its graph.

    The frozen pass runs over the `3B` rows under `no_grad` and
    `adapter.disabled()`; the adapted pass runs over the `B` original rows in
    `adapter.training()` mode. With gradient checkpointing the backward must
    run in that mode too, so wrap the call and the backward in one
    `adapter.training()` block, as `train_lorra` does.

    Args:
        loaded: The wrapped model, with `adapter` attached.
        adapter: The adapter being trained.
        batch: The batch of `build_batch`.

    Returns:
        The scalar `lorra_loss`.
    """
    current, frozen = _residual_pair(loaded, adapter, batch, training=True)
    return lorra_loss(current, frozen, batch, adapter.settings)


def train_lorra(
    loaded: LoadedModel,
    adapter: LorraAdapter,
    pairs: Sequence[Pair],
    chat: ChatFormat,
    *,
    bos: BosMode | None = None,
    on_step: Callable[[int, float], None] | None = None,
) -> LorraTrace:
    """Trains the adapter with AdamW on batches of pairs.

    Batches come from epochs of `rng.permutation(len(pairs))` with
    `rng = np.random.default_rng(seed)`; an epoch that ends mid-batch carries
    over into the next one. The loop runs inside `torch.random.fork_rng`
    seeded with `seed`, so the dropout masks are reproducible and the
    caller's random state is untouched.

    Args:
        loaded: The wrapped model, with `adapter` attached.
        adapter: The adapter to train.
        pairs: `(instruction, response)` pairs that fit (see `fit_rows`).
        chat: How the user turn becomes a prompt.
        bos: BOS policy; None uses the model default.
        on_step: Called as `on_step(step, loss)` after every step.

    Returns:
        The loss of every step and the peak CUDA memory.

    Raises:
        ValueError: If there are fewer pairs than `batch_size`.
    """
    settings = adapter.settings
    size = settings.batch_size
    if len(pairs) < size:
        raise ValueError(
            f"{len(pairs)} pairs are fewer than the batch size {size}"
        )
    rng = np.random.default_rng(settings.seed)
    optimizer = torch.optim.AdamW(
        adapter.trainable(), lr=settings.lr, weight_decay=0.0
    )
    on_cuda = loaded.device.type == "cuda"
    devices = (
        [
            torch.cuda.current_device()
            if loaded.device.index is None
            else loaded.device.index
        ]
        if on_cuda
        else []
    )
    if on_cuda:
        torch.cuda.reset_peak_memory_stats(loaded.device)
    queue: list[int] = []
    losses: list[float] = []
    with torch.random.fork_rng(devices=devices):
        torch.default_generator.manual_seed(settings.seed)
        if on_cuda:
            with torch.cuda.device(devices[0]):
                torch.cuda.manual_seed(settings.seed)
        for step in range(settings.steps):
            while len(queue) < size:
                queue.extend(rng.permutation(len(pairs)).tolist())
            chosen, queue = queue[:size], queue[size:]
            batch = build_batch(
                loaded, [pairs[i] for i in chosen], chat, settings, bos=bos
            )
            optimizer.zero_grad(set_to_none=True)
            with adapter.training():
                loss = step_loss(loaded, adapter, batch)
                loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            losses.append(loss.item())
            if on_step is not None:
                on_step(step, losses[-1])
    peak = (
        torch.cuda.max_memory_allocated(loaded.device) / 2**20
        if on_cuda
        else None
    )
    return LorraTrace(losses=tuple(losses), peak_memory_mb=peak)


def heldout_losses(
    loaded: LoadedModel,
    adapter: LorraAdapter,
    pairs: Sequence[Pair],
    chat: ChatFormat,
    *,
    bos: BosMode | None = None,
) -> dict[int, float]:
    """Measures the per-layer loss of the adapted model on held-out pairs.

    Runs the two forwards of a step without dropout and without gradients,
    in batches of `batch_size`. A fresh adapter equals the base model, so a
    call before `train_lorra` gives the loss of the untrained model and the
    difference to a call after it is the effect of training.

    Args:
        loaded: The wrapped model, with `adapter` attached.
        adapter: The adapter to evaluate.
        pairs: `(instruction, response)` pairs that fit (see `fit_rows`).
        chat: How the user turn becomes a prompt.
        bos: BOS policy; None uses the model default.

    Returns:
        Layer to the mean over batches of that layer's loss; the last batch
        may be smaller.

    Raises:
        ValueError: If there are no pairs.
    """
    if not pairs:
        raise ValueError("no pairs")
    settings = adapter.settings
    totals = dict.fromkeys(settings.layers, 0.0)
    n_batches = 0
    with torch.no_grad():
        for start in range(0, len(pairs), settings.batch_size):
            batch = build_batch(
                loaded,
                pairs[start : start + settings.batch_size],
                chat,
                settings,
                bos=bos,
            )
            current, frozen = _residual_pair(
                loaded, adapter, batch, training=False
            )
            for layer, loss in layer_losses(
                current, frozen, batch, settings
            ).items():
                totals[layer] += loss.item()
            n_batches += 1
    return {layer: total / n_batches for layer, total in totals.items()}
