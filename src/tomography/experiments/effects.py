"""Side effects of steering, held-out text splits and result writers."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tomography.experiments.base import RunDir
from tomography.experiments.steering import Arm
from tomography.experiments.steering import Trial
from tomography.experiments.steering import is_empty_completion
from tomography.experiments.steering import scoring_bos
from tomography.judges import sequence_logprob
from tomography.model import LoadedModel
from tomography.reading.metrics import bootstrap_ci


@dataclass(frozen=True)
class SideEffects:
    """What steering costs besides the concept it moves.

    Attributes:
        perplexity: Perplexity of neutral text under the arm; None if the
            arm has none.
        fluency: Mean per-token log-probability of the completions under the
            unsteered model; NaN when every completion is empty.
        distinct_1: Distinct unigrams over all unigrams, pooled.
        distinct_2: Distinct bigrams over all bigrams, pooled.
        mean_length: Generated tokens per completion, empty ones included.
        empty_rate: Fraction of empty or whitespace completions.
    """

    perplexity: float | None
    fluency: float
    distinct_1: float
    distinct_2: float
    mean_length: float
    empty_rate: float


@dataclass(frozen=True)
class HeldOut:
    """Two disjoint slices of held-out text.

    Attributes:
        calibration: Texts that calibrate relative coefficients.
        neutral: Texts of the neutral perplexity.
    """

    calibration: tuple[str, ...]
    neutral: tuple[str, ...]


def distinct_n(token_ids: Sequence[Sequence[int]], n: int) -> float:
    """Computes the share of distinct n-grams, pooled over completions.

    No n-gram spans two completions.

    Args:
        token_ids: The token ids of each completion.
        n: The n-gram size.

    Returns:
        Distinct n-grams over all n-grams; NaN when no completion has n
        tokens.

    Raises:
        ValueError: If `n` is below 1.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    seen: set[tuple[int, ...]] = set()
    total = 0
    for ids in token_ids:
        for start in range(len(ids) - n + 1):
            seen.add(tuple(ids[start : start + n]))
            total += 1
    return len(seen) / total if total else float("nan")


def _non_empty(trial: Trial) -> list[tuple[str, str]]:
    """Returns the `(prompt, text)` of each non-empty completion."""
    return [
        (prompt, text)
        for prompt, texts, lengths in zip(
            trial.prompts, trial.completions, trial.lengths, strict=True
        )
        for text, length in zip(texts, lengths, strict=True)
        if not is_empty_completion(text, int(length))
    ]


def _token_ids(loaded: LoadedModel, texts: Sequence[str]) -> list[list[int]]:
    encoded = loaded.tokenizer(list(texts), add_special_tokens=False)
    return [list(ids) for ids in encoded["input_ids"]]


def fluency(
    loaded: LoadedModel, trial: Trial, *, batch_size: int = 16
) -> float:
    """Measures how likely the unsteered model finds the completions.

    The value is the mean over non-empty completions of the summed
    continuation log-probability given the original prompt (no prefix),
    divided by the token count of the completion text. Every arm is judged
    by this one yardstick, outside any hook.

    Args:
        loaded: The wrapped model.
        trial: The trial whose completions to score.
        batch_size: Sequences per forward pass.

    Returns:
        The mean per-token log-probability; NaN when every completion is
        empty.
    """
    pairs = _non_empty(trial)
    if not pairs:
        return float("nan")
    prompts = [prompt for prompt, _ in pairs]
    texts = [text for _, text in pairs]
    logprobs = sequence_logprob(
        loaded.model,
        loaded.tokenizer,
        prompts,
        texts,
        batch_size=batch_size,
        bos_id=scoring_bos(loaded),
    )
    counts = np.array([len(ids) for ids in _token_ids(loaded, texts)])
    return float(np.mean(logprobs / counts))


def side_effects(
    loaded: LoadedModel,
    arm: Arm,
    trial: Trial,
    neutral_texts: Sequence[str],
    *,
    batch_size: int = 16,
) -> SideEffects:
    """Collects the side effects of an arm.

    A trial holds texts and lengths only, so the distinct-n values are
    computed over the token ids that the model tokenizer gives each
    non-empty completion text.

    Args:
        loaded: The wrapped model.
        arm: The arm that produced the trial.
        trial: The judged trial of the arm.
        neutral_texts: Held-out text for the neutral perplexity.
        batch_size: Sequences per forward pass in `fluency`.

    Returns:
        The side effects.
    """
    ids = _token_ids(loaded, [text for _, text in _non_empty(trial)])
    return SideEffects(
        perplexity=(
            None if arm.perplexity is None else arm.perplexity(neutral_texts)
        ),
        fluency=fluency(loaded, trial, batch_size=batch_size),
        distinct_1=distinct_n(ids, 1),
        distinct_2=distinct_n(ids, 2),
        mean_length=float(trial.lengths.mean()),
        empty_rate=trial.empty_rate,
    )


def split_held_out(
    texts: Sequence[str],
    *,
    n_calibration: int,
    n_neutral: int,
    seed: int,
    min_chars: int = 40,
) -> HeldOut:
    """Splits held-out text into calibration and neutral slices.

    Texts shorter than `min_chars` characters (headings) are dropped, the
    rest are shuffled with `np.random.default_rng(seed)`, and the first
    `n_calibration` become the calibration slice and the next `n_neutral`
    the neutral slice.

    Args:
        texts: The candidate texts.
        n_calibration: Size of the calibration slice.
        n_neutral: Size of the neutral slice.
        seed: Seed of the shuffle.
        min_chars: Shortest text kept, in characters.

    Returns:
        The two disjoint slices.

    Raises:
        ValueError: If a slice size is below 1, or fewer than
            `n_calibration + n_neutral` texts remain after the filter.
    """
    if n_calibration < 1 or n_neutral < 1:
        raise ValueError(
            "n_calibration and n_neutral must be >= 1, got "
            f"{n_calibration} and {n_neutral}"
        )
    kept = [text for text in texts if len(text) >= min_chars]
    needed = n_calibration + n_neutral
    if len(kept) < needed:
        raise ValueError(
            f"need {needed} texts of at least {min_chars} characters, "
            f"got {len(kept)}"
        )
    order = np.random.default_rng(seed).permutation(len(kept))
    shuffled = [kept[i] for i in order]
    return HeldOut(
        calibration=tuple(shuffled[:n_calibration]),
        neutral=tuple(shuffled[n_calibration:needed]),
    )


def write_trials(run_dir: RunDir, stem: str, trials: Sequence[Trial]) -> None:
    """Writes judged trials: the audit trail of every judged number.

    `<stem>.npz` holds `scores` and `lengths`, stacked on a leading trial
    axis, and `<stem>.json` holds `names`, `prompts` and `completions`
    (trial, prompt, sample).

    Args:
        run_dir: The run to write into.
        stem: File stem of the pair.
        trials: The trials, on the same prompts and with the same shapes.

    Raises:
        ValueError: If there are no trials, or they differ in prompts or in
            array shapes.
    """
    if not trials:
        raise ValueError("no trials to write")
    first = trials[0]
    for trial in trials[1:]:
        if trial.prompts != first.prompts:
            raise ValueError(
                f"trial {trial.name!r} has different prompts than "
                f"{first.name!r}"
            )
        if (
            trial.scores.shape != first.scores.shape
            or trial.lengths.shape != first.lengths.shape
        ):
            raise ValueError(
                f"trial {trial.name!r} has scores shape {trial.scores.shape} "
                f"and lengths shape {trial.lengths.shape}, expected "
                f"{first.scores.shape} and {first.lengths.shape}"
            )
    run_dir.write_arrays(
        stem,
        scores=np.stack([trial.scores for trial in trials]),
        lengths=np.stack([trial.lengths for trial in trials]),
    )
    run_dir.write_json(
        stem,
        {
            "names": [trial.name for trial in trials],
            "prompts": list(first.prompts),
            "completions": [
                [list(row) for row in trial.completions] for trial in trials
            ],
        },
    )


def read_trials(run_dir: RunDir, stem: str) -> list[Trial]:
    """Reads the trials written by `write_trials`.

    Args:
        run_dir: The run to read from.
        stem: File stem given to `write_trials`.

    Returns:
        The trials, with NaN scores preserved.

    Raises:
        ValueError: If `<stem>.json` does not hold an object.
    """
    payload = run_dir.read_json(stem)
    if not isinstance(payload, dict):
        raise ValueError(
            f"{stem}.json must hold an object, got {type(payload).__name__}"
        )
    arrays = run_dir.read_arrays(stem)
    return [
        Trial(
            name=name,
            prompts=tuple(payload["prompts"]),
            completions=tuple(tuple(row) for row in payload["completions"][t]),
            lengths=arrays["lengths"][t].astype(np.int64),
            scores=arrays["scores"][t].astype(np.float32),
        )
        for t, name in enumerate(payload["names"])
    ]


def write_dose_curves(
    run_dir: RunDir,
    per_seed: NDArray[np.float64],
    coeffs: Sequence[float],
    *,
    series: Sequence[str],
    stem: str = "curves",
    xlabel: str = "Coefficient",
    ylabel: str = "Score",
    chance: float | None = None,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 0,
) -> None:
    """Writes the `curves` arrays and sidecar of a dose-response figure.

    The arrays hold `x (K,)`, `mean (S, K)`, `lo (S, K)`, `hi (S, K)` and
    `per_seed (S, N, K)`, with a percentile bootstrap band over the N units.

    Args:
        run_dir: The run to write into.
        per_seed: Values of shape (S, N, K): series, bootstrap unit (seed or
            reader), coefficient.
        coeffs: The x value of each of the K coefficients.
        series: The name of each series.
        stem: File stem of the pair (`<stem>.npz` and `<stem>.json`).
        xlabel: Label of the x axis.
        ylabel: Label of the y axis.
        chance: Level of the reference line, or None for no line.
        alpha: One minus the coverage of the band.
        n_boot: Number of bootstrap resamples.
        seed: Seed of the resampling.

    Raises:
        ValueError: If `per_seed` is not 3D or `series` or `coeffs` do not
            match its shape.
    """
    if per_seed.ndim != 3:
        raise ValueError(
            f"per_seed must have shape (S, N, K), got {per_seed.shape}"
        )
    if len(series) != per_seed.shape[0]:
        raise ValueError(
            f"got {len(series)} series names for {per_seed.shape[0]} series"
        )
    if len(coeffs) != per_seed.shape[2]:
        raise ValueError(
            f"got {len(coeffs)} coeffs for {per_seed.shape[2]} columns"
        )
    lo, hi = bootstrap_ci(
        np.moveaxis(per_seed, 1, 0), alpha=alpha, n_boot=n_boot, seed=seed
    )
    run_dir.write_arrays(
        stem,
        x=np.asarray(coeffs, dtype=np.float64),
        mean=per_seed.mean(axis=1),
        lo=lo,
        hi=hi,
        per_seed=per_seed,
    )
    run_dir.write_json(
        stem,
        {
            "series": list(series),
            "xlabel": xlabel,
            "ylabel": ylabel,
            "chance": chance,
            "n_layers": None,
        },
    )
