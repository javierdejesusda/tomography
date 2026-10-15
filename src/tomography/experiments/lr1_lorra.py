"""LR1: one LoRRA adapter trained on Alpaca, and how to bring it back.

The experiment trains a single adapter and records its loss curve and its
held-out loss. It evaluates nothing about honesty: `load_adapter` and
`gated_arm` bring the trained adapter back as one more `Arm`, so the honesty
evaluation treats it like every other method.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
import statistics
import time

import numpy as np

from tomography.datasets.alpaca import load_alpaca
from tomography.datasets.records import Example
from tomography.datasets.records import Split
from tomography.experiments.base import ExperimentConfig
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.effects import write_dose_curves
from tomography.experiments.steering import Arm
from tomography.experiments.sweep import chat_format_for
from tomography.experiments.sweep import plot_curves_file
from tomography.lorra import LorraAdapter
from tomography.lorra import LorraSettings
from tomography.lorra import LossForm
from tomography.lorra import Pair
from tomography.lorra import attach_adapter
from tomography.lorra import default_layers
from tomography.lorra import fit_rows
from tomography.lorra import heldout_losses
from tomography.lorra import train_lorra
from tomography.model import BosMode
from tomography.model import LoadedModel
from tomography.model import load_model


@dataclass(frozen=True, kw_only=True)
class Config(ExperimentConfig):
    """LR1 settings: a LoRRA adapter on the post-trained Qwen3-1.7B.

    The training fields are those of `LorraSettings` and its defaults follow
    the reference implementation. `seed` seeds the adapter initialization,
    the batch order, the dropout and the draw of the pairs.

    Attributes:
        layers: Target layers; None means `default_layers` of the model.
        alpha: Strength of the contrast term in the target.
        pos_control: Control text of the positive stream.
        neg_control: Control text of the negative stream.
        rank: LoRA rank.
        lora_alpha: LoRA scaling numerator.
        dropout: LoRA dropout.
        lr: AdamW learning rate.
        batch_size: Pairs per step.
        steps: Optimizer steps.
        max_prompt_tokens: Prompt window.
        max_response_tokens: Response window.
        loss: `mean_norm` (reference) or `sum_squared` (paper).
        gradient_checkpointing: Recompute block activations in backward.
        n_train: Upper bound on the Alpaca `train` pairs drawn; the run reads
            `steps * batch_size` of them, so the bound is rarely reached.
        n_heldout: Upper bound on the Alpaca `val` pairs scored for the
            held-out loss.
        bos: BOS policy; None uses the model default.
    """

    experiment: str = "lr1"
    model: str = "qwen3-1.7b"
    layers: tuple[int, ...] | None = None
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
    n_train: int = 2000
    n_heldout: int = 64
    bos: BosMode | None = None

    def settings(self, n_layers: int) -> LorraSettings:
        """Builds the training settings for a model depth.

        Args:
            n_layers: Number of decoder blocks of the model.

        Returns:
            The settings; `layers` defaults to `default_layers(n_layers)`.

        Raises:
            ValueError: If a field is invalid for `LorraSettings`.
        """
        return LorraSettings(
            layers=(
                default_layers(n_layers) if self.layers is None else self.layers
            ),
            alpha=self.alpha,
            pos_control=self.pos_control,
            neg_control=self.neg_control,
            rank=self.rank,
            lora_alpha=self.lora_alpha,
            dropout=self.dropout,
            lr=self.lr,
            batch_size=self.batch_size,
            steps=self.steps,
            max_prompt_tokens=self.max_prompt_tokens,
            max_response_tokens=self.max_response_tokens,
            loss=self.loss,
            gradient_checkpointing=self.gradient_checkpointing,
            seed=self.seed,
        )


def _draw(
    examples: Sequence[Example], split: Split, bound: int, seed: int
) -> list[Pair]:
    """Draws up to `bound` `(instruction, response)` pairs of one split."""
    rows = [example for example in examples if example.split == split]
    order = np.random.default_rng(seed).permutation(len(rows))[:bound]
    return [
        (str(rows[i].meta["instruction"]), rows[i].text) for i in order.tolist()
    ]


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    examples: Sequence[Example],
) -> None:
    """Trains one adapter and writes its curve, table and summary.

    The pairs are Alpaca rows (`text` is the response, `meta["instruction"]`
    the prompt): `n_train` are drawn from the `train` split and `n_heldout`
    from the `val` split, so the two never overlap. Pairs that do not fit
    the windows are dropped and counted. The adapter is attached to `loaded`,
    so the experiment runs once per loaded model. Writes `adapter.npz`,
    `curves` (one series, the training loss by step), `table.json` (the
    held-out loss before and after training for each target layer) and
    `summary.json` (including `seconds`, the wall time of the training loop).

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model; it ends up running the adapter.
        examples: Alpaca examples with `train` and `val` splits.

    Raises:
        ValueError: If fewer than `batch_size` training pairs or no held-out
            pair is left after dropping, before the adapter is attached.
    """
    settings = config.settings(loaded.n_layers)
    chat = chat_format_for(loaded)
    train, dropped_train = fit_rows(
        loaded,
        _draw(
            examples,
            "train",
            config.n_train,
            derive_seed(config.seed, "lr1", "train"),
        ),
        chat,
        settings,
        bos=config.bos,
    )
    heldout, dropped_heldout = fit_rows(
        loaded,
        _draw(
            examples,
            "val",
            config.n_heldout,
            derive_seed(config.seed, "lr1", "heldout"),
        ),
        chat,
        settings,
        bos=config.bos,
    )
    if len(train) < settings.batch_size:
        raise ValueError(
            f"{len(train)} training pairs are fewer than the batch size "
            f"{settings.batch_size}"
        )
    if not heldout:
        raise ValueError("no held-out pair is left after dropping")
    adapter = attach_adapter(loaded, settings)
    before = heldout_losses(loaded, adapter, heldout, chat, bos=config.bos)
    started = time.perf_counter()
    trace = train_lorra(loaded, adapter, train, chat, bos=config.bos)
    seconds = time.perf_counter() - started
    after = heldout_losses(loaded, adapter, heldout, chat, bos=config.bos)
    run_dir.write_arrays("adapter", **adapter.state())
    write_dose_curves(
        run_dir,
        np.asarray(trace.losses, dtype=np.float64)[None, None, :],
        list(range(len(trace.losses))),
        series=["train loss"],
        xlabel="Step",
        ylabel="Loss",
    )
    run_dir.write_json(
        "table",
        [
            {
                "layer": layer,
                "heldout_before": before[layer],
                "heldout_after": after[layer],
                "reduction": (
                    None
                    if before[layer] == 0.0
                    else 1 - after[layer] / before[layer]
                ),
            }
            for layer in settings.layers
        ],
    )
    run_dir.write_json(
        "summary",
        {
            "layers": list(settings.layers),
            "blocks": list(adapter.blocks),
            "n_trainable": adapter.n_trainable(),
            "n_train": len(train),
            "n_heldout": len(heldout),
            "dropped_train": dropped_train,
            "dropped_heldout": dropped_heldout,
            "steps": settings.steps,
            "first_loss": statistics.fmean(trace.losses[:10]),
            "last_loss": statistics.fmean(trace.losses[-10:]),
            "peak_memory_mb": trace.peak_memory_mb,
            "seconds": seconds,
        },
    )


def run(config: Config, run_dir: RunDir) -> None:
    """Loads the model and Alpaca, then runs `execute`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
    """
    execute(config, run_dir, load_model(config.model), load_alpaca())


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws the training loss curve.

    Args:
        run_dir: A finished LR1 run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths.
    """
    return plot_curves_file(run_dir, out_dir)


def load_adapter(loaded: LoadedModel, run_path: str | Path) -> LorraAdapter:
    """Brings a trained adapter back onto a fresh model.

    The settings are rebuilt from the run's own config, the adapter is
    attached to `loaded` and the weights of `adapter.npz` are loaded into it.
    Gating covers the callables of an arm only: anything that reads the
    model to build a controller (reader activations, mean residual norms,
    `PromptDifference.from_prompts`) must run before this call or inside
    `adapter.disabled()`.

    Args:
        loaded: A freshly loaded model of the architecture the adapter was
            trained on; it now runs the adapter.
        run_path: A finished LR1 run directory.

    Returns:
        The adapter.

    Raises:
        ValueError: If the run is not finished, or the weights do not fit
            the model (a different architecture gives different keys or
            shapes); `loaded` then runs no adapter.
    """
    run_dir = RunDir.open(run_path)
    if run_dir.meta["finished_at"] is None:
        raise ValueError(f"{run_dir.path} is not a finished run")
    config = Config.from_dict(run_dir.config)
    state = run_dir.read_arrays("adapter")
    adapter = attach_adapter(loaded, config.settings(loaded.n_layers))
    try:
        adapter.load_state(state)
    except Exception:
        adapter.peft_model.unload()
        raise
    return adapter


def gated_arm(adapter: LorraAdapter, arm: Arm, *, enabled: bool) -> Arm:
    """Gates the callables of an arm on the adapter.

    With `enabled=False` the arm is bit-identical to a model that never had
    an adapter, so the comparison needs no second model in memory.

    Args:
        adapter: The adapter attached to the model the arm runs.
        arm: The arm to gate.
        enabled: Whether its generation, log-probabilities and perplexity
            run with the adapter on or off.

    Returns:
        A copy of the arm with `generate`, `logprobs` and `perplexity`
        gated (a missing callable stays missing) and the same name.
    """
    return replace(
        arm,
        generate=adapter.gated(arm.generate, enabled=enabled),
        logprobs=(
            None
            if arm.logprobs is None
            else adapter.gated(arm.logprobs, enabled=enabled)
        ),
        perplexity=(
            None
            if arm.perplexity is None
            else adapter.gated(arm.perplexity, enabled=enabled)
        ),
    )
