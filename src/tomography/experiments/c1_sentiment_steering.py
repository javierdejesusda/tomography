"""C1: steering sentiment on GPT-2 with a reading vector."""

from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tomography.control.operators import Operator
from tomography.experiments.steering import Arm
from tomography.experiments.steering import DoseResponse
from tomography.experiments.steering import Scorer
from tomography.experiments.steering import SteeringConfig
from tomography.experiments.steering import Trial
from tomography.experiments.steering import dose_response
from tomography.experiments.steering import dose_slope
from tomography.reading.directions import Method


@dataclass(frozen=True, kw_only=True)
class Config(SteeringConfig):
    """C1 settings: steering GPT-2 toward positive and negative sentiment.

    Readers are fitted as in R1 on SST-2 pairs. Every generation is scored
    by two judges at once, so the primary column drives every choice and the
    check column tests that the effect holds under both. The inherited
    `layers` and `op` are not used: C1 sweeps `sweep_layers` and compares
    `ops`.

    Attributes:
        method: The direction finder, as in R1.
        judge: Key of the primary judge, score column 0.
        judge_check: Key of the check judge, score column 1.
        n_seeds: Readers fitted on resampled training pairs.
        n_steer_seeds: The first readers, in fit order, that steer.
        n_pairs: Training pairs per reader.
        n_eval_pairs: Validation pairs; None means every pair.
        min_words: SST-2 phrases with fewer words are dropped.
        sweep_layers: Single layers of the dose-response sweep; None means
            `default_sweep_layers` of the model depth.
        window: Layers steered together at the headline setting, centered
            on the layer the sweep chooses.
        ops: Operators compared at the headline setting. The first one
            also gets the side-effect rows.
        truncate_words: Words kept of each validation sentence as a prompt.
        prompt_positive: Instructions of the positive prompting baseline.
        prompt_negative: Instructions of the negative prompting baseline.
    """

    experiment: str = "c1"
    model: str = "gpt2"
    method: Method = "logistic"
    judge: str = "sentiment-primary"
    judge_check: str = "sentiment-secondary"
    n_seeds: int = 10
    n_steer_seeds: int = 3
    n_pairs: int = 64
    n_eval_pairs: int | None = 256
    min_words: int = 3
    sweep_layers: tuple[int, ...] | None = None
    window: int = 3
    ops: tuple[Operator, ...] = ("linear", "piecewise")
    truncate_words: int = 3
    prompt_positive: tuple[str, ...] = (
        "Write a happy and positive text. ",
        "Write a very happy, joyful and enthusiastic text. ",
    )
    prompt_negative: tuple[str, ...] = (
        "Write a sad and negative text. ",
        "Write a very sad, angry and hateful text. ",
    )


def default_sweep_layers(n_layers: int, count: int = 6) -> tuple[int, ...]:
    """Spaces the sweep layers evenly over the depth.

    Args:
        n_layers: Depth of the model.
        count: Number of layers wanted; fewer come back for a shallow model
            or when the rounded positions repeat.

    Returns:
        Increasing layers in `1..n_layers`, from the first to the last.

    Raises:
        ValueError: If `count` is below 1.
    """
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")
    spaced = np.linspace(1, n_layers, min(count, n_layers)).round()
    return tuple(int(layer) for layer in np.unique(spaced))


def window_around(layer: int, width: int, n_layers: int) -> tuple[int, ...]:
    """Returns the consecutive layers centered on a layer.

    For an even width one more layer lies below `layer` than above. The
    window shifts to stay inside `1..n_layers`.

    Args:
        layer: The center layer.
        width: Number of layers in the window.
        n_layers: Depth of the model.

    Returns:
        The `width` consecutive layers.

    Raises:
        ValueError: If `width` or `layer` is outside `1..n_layers`.
    """
    if not 1 <= width <= n_layers:
        raise ValueError(f"width must be in 1..{n_layers}, got {width}")
    if not 1 <= layer <= n_layers:
        raise ValueError(f"layer must be in 1..{n_layers}, got {layer}")
    first = min(max(layer - width // 2, 1), n_layers - width + 1)
    return tuple(range(first, first + width))


def _mean_slopes(
    per_seed: NDArray[np.float64], coeffs: Sequence[float]
) -> NDArray[np.float64]:
    """Averages the dose slopes of the readers of each layer.

    A reader with fewer than two finite points has no slope and is skipped.
    A layer with no slope at all gets NaN.
    """
    means = np.full(per_seed.shape[0], np.nan)
    for at, readers in enumerate(per_seed):
        slopes = [
            dose_slope(coeffs, values)
            for values in readers
            if np.isfinite(values).sum() >= 2
        ]
        if slopes:
            means[at] = float(np.mean(slopes))
    return means


def slope_layer(
    per_seed: NDArray[np.float64],
    coeffs: Sequence[float],
    layers: Sequence[int],
) -> int:
    """Picks the layer where the dose response rises the most.

    Args:
        per_seed: Scores of shape (S, N, K): layer, reader, coefficient.
        coeffs: The K coefficients.
        layers: The layer of each of the S rows.

    Returns:
        The layer with the largest mean over readers of the dose slope. A
        reader with fewer than two finite points is skipped, a layer
        without a finite slope never wins and ties go to the first layer.

    Raises:
        ValueError: If no layer has a finite slope.
    """
    slopes = _mean_slopes(per_seed, coeffs)
    if not np.isfinite(slopes).any():
        raise ValueError("no layer has a finite dose slope")
    return int(layers[int(np.nanargmax(slopes))])


def prompt_fragments(texts: Sequence[str], words: int) -> list[str]:
    """Cuts texts to their first words, without repeats.

    Args:
        texts: The source texts.
        words: Number of whitespace-separated words to keep.

    Returns:
        The fragments, with empty ones dropped and every repeat after the
        first occurrence removed.

    Raises:
        ValueError: If `words` is below 1.
    """
    if words < 1:
        raise ValueError(f"words must be at least 1, got {words}")
    cut = (" ".join(text.split()[:words]) for text in texts)
    return list(dict.fromkeys(fragment for fragment in cut if fragment))


def draw_prompts(
    pool: Sequence[str], n: int, *, sets: int, seed: int
) -> list[list[str]]:
    """Draws disjoint prompt lists from one shuffle of a pool.

    Args:
        pool: The candidate prompts.
        n: Prompts per list.
        sets: Number of lists.
        seed: Seed of `np.random.default_rng` for the permutation.

    Returns:
        `sets` lists of `n` prompts; list k holds the k-th block of the
        permuted pool.

    Raises:
        ValueError: If `n` or `sets` is below 1, or the pool has fewer than
            `n * sets` prompts.
    """
    if n < 1 or sets < 1:
        raise ValueError(f"n and sets must be at least 1, got {n} and {sets}")
    if len(pool) < n * sets:
        raise ValueError(
            f"pool has {len(pool)} prompts, {sets} sets of {n} need {n * sets}"
        )
    order = np.random.default_rng(seed).permutation(len(pool))
    return [[pool[i] for i in order[k * n : (k + 1) * n]] for k in range(sets)]


def dose_with_none(
    make_arm: Callable[[float], Arm],
    coeffs: Sequence[float],
    prompts: Sequence[str],
    score: Scorer,
    none: Trial,
    *,
    columns: int | None = None,
) -> DoseResponse:
    """Runs a dose response and reuses the unsteered trial at coefficient 0.

    A reader arm at coefficient 0 is bit-identical to the plain arm, so the
    plain trial is generated once and stands in for every coefficient 0.

    Args:
        make_arm: Builds the arm of a nonzero coefficient.
        coeffs: The coefficients, in order; 0.0 must be one of them.
        prompts: The prompts shared by every trial.
        score: The scorer, as in `run_trial`.
        none: The plain arm's trial on the same prompts.
        columns: Number of scorer columns, as in `run_trial`.

    Returns:
        The trials, with `none` at each 0.0.

    Raises:
        ValueError: If `coeffs` has no 0.0 or no other value, or `none` was
            run on other prompts.
    """
    nonzero = [c for c in coeffs if c != 0.0]
    if len(nonzero) == len(coeffs):
        raise ValueError("coeffs must contain 0.0")
    if not nonzero:
        raise ValueError("coeffs must contain a nonzero value")
    if none.prompts != tuple(prompts):
        raise ValueError("none was run on other prompts than prompts")
    steered = iter(
        dose_response(make_arm, nonzero, prompts, score, columns=columns).trials
    )
    return DoseResponse(
        coeffs=tuple(float(c) for c in coeffs),
        trials=tuple(none if c == 0.0 else next(steered) for c in coeffs),
    )
