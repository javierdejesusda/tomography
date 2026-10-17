"""C2: steering emotions on GPT-2 with one reading vector per emotion.

An emotion reader is fitted one-vs-rest as in R4 and steers text generated
from short neutral prompts. The result is a matrix of the steered emotion by
the measured emotion, read with two judges at once.
"""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import cast

import numpy as np
from numpy.typing import NDArray

from tomography.control.controllers import mean_residual_norms
from tomography.datasets.emotions import EMOTIONS
from tomography.datasets.emotions import load_emotions
from tomography.datasets.records import Example
from tomography.datasets.stimulus import StimulusSet
from tomography.datasets.stimulus import build_stimulus_set
from tomography.datasets.wikitext import load_wikitext
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.c1_sentiment_steering import check_pair_count
from tomography.experiments.c1_sentiment_steering import dose_with_none
from tomography.experiments.c1_sentiment_steering import draw_prompts
from tomography.experiments.c1_sentiment_steering import prompt_fragments
from tomography.experiments.c1_sentiment_steering import window_around
from tomography.experiments.controls import max_abs
from tomography.experiments.controls import random_readers
from tomography.experiments.effects import SideEffects
from tomography.experiments.effects import effects_row
from tomography.experiments.effects import side_effects
from tomography.experiments.effects import split_held_out
from tomography.experiments.effects import write_dose_curves
from tomography.experiments.effects import write_trials
from tomography.experiments.steering import Arm
from tomography.experiments.steering import GenSettings
from tomography.experiments.steering import Scorer
from tomography.experiments.steering import SteeringConfig
from tomography.experiments.steering import Trial
from tomography.experiments.steering import plain_arm
from tomography.experiments.steering import reader_arm
from tomography.experiments.steering import run_trial
from tomography.experiments.sweep import SERIES
from tomography.experiments.sweep import SweepResult
from tomography.experiments.sweep import chat_format_for
from tomography.experiments.sweep import plot_curves_file
from tomography.experiments.sweep import reading_sweep
from tomography.judges import load_judge
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.reading.directions import Method
from tomography.reading.metrics import best_layer
from tomography.reading.reader import RepReader
from tomography.templates import PAPER_TEMPLATES
from tomography.templates import ChatFormat
from tomography.viz.geometry import plot_matrix
from tomography.viz.style import save_figure
from tomography.viz.style import style_context

_N = len(EMOTIONS)
_JUDGES = ("primary", "check")
_EFFECT_COLUMNS = (
    "arm",
    "coeff",
    "seed",
    "emotion",
    "perplexity",
    "fluency",
    "distinct_1",
    "distinct_2",
    "mean_length",
    "empty_rate",
)


@dataclass(frozen=True, kw_only=True)
class Config(SteeringConfig):
    """C2 settings: steering GPT-2 toward each emotion.

    One reader per emotion and seed is fitted one-vs-rest against all five
    other emotions, as in R4, whichever emotions are steered. Every
    generation is scored by two emotion judges at once: `score` columns
    `0..5` are the primary judge and `6..11` the check judge, in `EMOTIONS`
    order. The inherited `layers` is not used: the window follows the best
    reading layer.

    Attributes:
        method: The direction finder, as in R4.
        judge: Key of the primary judge.
        judge_check: Key of the check judge.
        emotions: The steered emotions, the rows of the matrix.
        n_seeds: Readers fitted per emotion on resampled training pairs.
        n_steer_seeds: The first readers, in fit order, that steer.
        n_pairs: Training pairs per reader.
        n_eval_pairs: Validation pairs; None means every pair.
        window: Layers steered together, centered on the best mean
            validation layer of the steered emotions.
        strength: The coefficient of the matrix; None means the largest of
            `coeffs`. It must be positive and one of `coeffs`.
        truncate_words: Words kept of each neutral text as a prompt.
        n_random: Random control directions, unit vectors drawn from the
            seed and norm-matched to the steering arm.
    """

    experiment: str = "c2"
    model: str = "gpt2"
    method: Method = "logistic"
    judge: str = "emotion-primary"
    judge_check: str = "emotion-secondary"
    emotions: tuple[str, ...] = EMOTIONS
    n_seeds: int = 10
    n_steer_seeds: int = 3
    n_pairs: int = 64
    n_eval_pairs: int | None = None
    window: int = 3
    strength: float | None = None
    truncate_words: int = 4
    n_random: int = 3


def emotion_stimuli(
    examples: Sequence[Example],
    emotion: str,
    chat: ChatFormat,
    *,
    n_pairs: int | None,
    seed: int,
) -> StimulusSet:
    """Pairs one emotion against the others, as R4 does.

    Args:
        examples: Emotion scenarios labeled with the index of their emotion
            in `EMOTIONS`. The negatives come from every other emotion in
            this list.
        emotion: The emotion that is label 1.
        chat: The chat format of the prompts.
        n_pairs: Number of pairs; None keeps every available pair.
        seed: Seed of the pairing.

    Returns:
        The stimulus set.

    Raises:
        ValueError: If `emotion` is not in `EMOTIONS`.
    """
    if emotion not in EMOTIONS:
        raise ValueError(f"unknown emotion {emotion!r}, expected {EMOTIONS}")
    return build_stimulus_set(
        examples,
        PAPER_TEMPLATES["emotion_concept"],
        chat,
        pairing="one_vs_rest",
        target_class=EMOTIONS.index(emotion),
        concept=emotion,
        n_pairs=n_pairs,
        seed=seed,
    )


def diagonal_stats(matrix: NDArray[np.float64]) -> tuple[float, float]:
    """Summarizes how much a steered-by-measured matrix peaks on its diagonal.

    Args:
        matrix: A square matrix whose row k and column k are the same
            emotion.

    Returns:
        The share of rows whose largest entry sits on the diagonal, and the
        mean diagonal entry minus the mean off-diagonal entry.

    Raises:
        ValueError: If `matrix` is not square or is smaller than 2 x 2.
    """
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] < 2
    ):
        raise ValueError(
            f"need a square matrix of at least 2 x 2, got {matrix.shape}"
        )
    diagonal = np.eye(matrix.shape[0], dtype=bool)
    hit_rate = float(
        np.mean(np.argmax(matrix, axis=1) == np.arange(len(matrix)))
    )
    return hit_rate, float(matrix[diagonal].mean() - matrix[~diagonal].mean())


def _check_emotions(emotions: tuple[str, ...]) -> None:
    """Checks that the steered emotions are at least two distinct names."""
    if (
        len(emotions) < 2
        or len(set(emotions)) != len(emotions)
        or not set(emotions) <= set(EMOTIONS)
    ):
        raise ValueError(
            f"emotions must be at least two distinct names of {EMOTIONS}, "
            f"got {emotions}"
        )


def _check_config(config: Config, n_layers: int) -> float:
    """Checks the settings that need no data and returns the strength."""
    _check_emotions(config.emotions)
    if 0.0 not in config.coeffs or all(c == 0.0 for c in config.coeffs):
        raise ValueError("coeffs must contain 0.0 and a nonzero value")
    strength = (
        max(config.coeffs) if config.strength is None else config.strength
    )
    if strength <= 0 or strength not in config.coeffs:
        raise ValueError(
            f"strength must be a positive value of coeffs {config.coeffs}, "
            f"got {strength}"
        )
    if config.n_random < 1:
        raise ValueError(f"n_random must be at least 1, got {config.n_random}")
    if not 1 <= config.n_steer_seeds <= config.n_seeds:
        raise ValueError(
            f"n_steer_seeds must be in 1..{config.n_seeds}, got "
            f"{config.n_steer_seeds}"
        )
    if not 1 <= config.window <= n_layers:
        raise ValueError(
            f"window must be in 1..{n_layers}, got {config.window}"
        )
    return strength


def _check_pair_counts(config: Config, scenarios: Sequence[Example]) -> None:
    """Checks that every steered emotion has the pairs the settings ask for."""
    _check_emotions(config.emotions)
    for split, name, wanted, allow_none in (
        ("train", "n_pairs", config.n_pairs, False),
        ("val", "n_eval_pairs", config.n_eval_pairs, True),
    ):
        labels = np.array([e.label for e in scenarios if e.split == split])
        for emotion in config.emotions:
            target = int(np.sum(labels == EMOTIONS.index(emotion)))
            check_pair_count(
                name,
                wanted,
                min(target, len(labels) - target),
                f"for {emotion!r} in the {split} split",
                allow_none=allow_none,
            )


def _fit_readers(
    config: Config, loaded: LoadedModel, scenarios: Sequence[Example]
) -> dict[str, SweepResult]:
    """Fits the readers of every steered emotion against all the others."""
    chat = chat_format_for(loaded)
    train = [e for e in scenarios if e.split == "train"]
    val = [e for e in scenarios if e.split == "val"]
    return {
        emotion: reading_sweep(
            loaded,
            [
                emotion_stimuli(
                    train,
                    emotion,
                    chat,
                    n_pairs=config.n_pairs,
                    seed=derive_seed(config.seed, "train", emotion, i),
                )
                for i in range(config.n_seeds)
            ],
            {
                "val": emotion_stimuli(
                    val,
                    emotion,
                    chat,
                    n_pairs=config.n_eval_pairs,
                    seed=derive_seed(config.seed, "eval", emotion),
                )
            },
            method=config.method,
            seed=derive_seed(config.seed, emotion),
            bos=config.bos,
            batch_size=config.batch_size,
        )
        for emotion in config.emotions
    }


def _write_matrix(
    run_dir: RunDir,
    stem: str,
    values: NDArray[np.float64],
    *,
    config: Config,
    title: str,
    layer: int,
) -> None:
    run_dir.write_arrays(stem, values=values)
    run_dir.write_json(
        stem,
        {
            "rows": list(config.emotions),
            "cols": list(EMOTIONS),
            "title": title,
            "layer": layer,
        },
    )


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    scenarios: Sequence[Example],
    texts: Sequence[str],
    score: Scorer,
) -> None:
    """Reads each emotion, steers with it and measures all six emotions.

    Readers are fitted on train scenarios and checked on val scenarios, one
    emotion against all the others even when only some are steered. The
    steering window is centered on the best mean validation layer of the
    steered emotions. Prompts are the first words of held-out neutral texts.
    The unsteered trial is generated once. The matrix is the change of every
    measured emotion at `strength` against it, averaged over readers.

    The random arms are `n_random` unit directions drawn from the seed and
    norm-matched to the steering arm: same window, relative coefficients,
    operator, mode and mask. They are run apart from the emotions and
    never enter the matrix.

    Writes `curves` and `curves_check` (own-emotion probability against the
    coefficient), `matrix` and `matrix_check`, `trials`, `trials_random`
    (trials named `random/<index>/<coefficient>`), `table.json`,
    `table_random.json` (the change of every measured emotion at `strength`
    under each random arm), `table_effects.json` (perplexity of neutral
    held-out text, completion fluency, distinct-1, distinct-2, mean length
    and empty rate of `none`, every emotion arm and every random arm, as in
    C1) and `summary.json`, whose
    `max_abs_random_change` is the largest absolute change of any measured
    emotion under any random arm and coefficient.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model.
        scenarios: Emotion scenarios with `train` and `val` splits,
            labeled with the emotion's index in `EMOTIONS`.
        texts: Held-out texts that set the relative coefficients and give
            the prompts.
        score: Maps texts to scores of shape (n, 12): the six emotions of
            the primary judge, then the six of the check judge.

    Raises:
        ValueError: If fewer than two distinct emotions are steered, `coeffs`
            has no 0.0 or no other value, `strength` is not a positive value
            of `coeffs`, `n_random` is below 1, `n_steer_seeds` is outside
            `1..n_seeds`, `window` does not fit the model, `n_pairs` or
            `n_eval_pairs` is below 1 or exceeds the pairs one steered
            emotion can form in its split (an emotion with no pair fails
            even when `n_eval_pairs` is None), the generation settings are
            invalid (see `GenSettings`), `n_calibration` or `n_neutral` is
            below 1, the usable (at least 40 characters) held-out texts are
            fewer than `n_calibration + n_neutral`, or the neutral ones give
            fewer than `n_prompts` distinct prompts.
    """
    n_layers = loaded.n_layers
    strength = _check_config(config, n_layers)
    _check_pair_counts(config, scenarios)
    settings = GenSettings.from_config(config)
    held = split_held_out(
        texts,
        n_calibration=config.n_calibration,
        n_neutral=config.n_neutral,
        seed=config.seed,
    )
    (prompts,) = draw_prompts(
        prompt_fragments(held.neutral, config.truncate_words),
        config.n_prompts,
        sets=1,
        seed=derive_seed(config.seed, "prompts"),
    )
    results = _fit_readers(config, loaded, scenarios)
    reader_at = SERIES.index("reader")
    reading_layer = best_layer(
        np.mean(
            [
                r.per_seed("val")[reader_at].mean(axis=0)
                for r in results.values()
            ],
            axis=0,
        ),
        next(iter(results.values())).layers,
    )
    window = window_around(reading_layer, config.window, n_layers)
    norms = mean_residual_norms(
        loaded,
        held.calibration,
        window,
        bos=config.bos,
        batch_size=config.batch_size,
    )

    def steering_arms(reader: RepReader) -> Callable[[float], Arm]:
        def make(coeff: float) -> Arm:
            return reader_arm(
                loaded,
                settings,
                reader,
                layers=window,
                coeff=coeff,
                norms=norms,
                op=config.op,
                mode=config.mode,
                mask=config.mask,
                preserve_norm=config.preserve_norm,
            )

        return make

    randoms = random_readers(
        window,
        next(iter(results.values())).readers[0].directions.shape[1],
        config.n_random,
        seed=config.seed,
    )
    columns = 2 * _N
    plain = plain_arm(loaded, settings)
    none = run_trial(plain, prompts, score, columns=columns)
    baseline = none.mean_score()

    def effects_of(arm: Arm, trial: Trial) -> SideEffects:
        return side_effects(
            loaded, arm, trial, held.neutral, batch_size=config.batch_size
        )

    effect_rows: list[list[object]] = [
        effects_row(effects_of(plain, none), "none", 0.0, None, None)
    ]
    at_strength = list(config.coeffs).index(strength)
    shape = (len(config.emotions), config.n_steer_seeds)
    own = np.full((len(_JUDGES), *shape, len(config.coeffs)), np.nan)
    change = np.full((len(_JUDGES), *shape, _N), np.nan)
    trials: list[Trial] = [none]
    for e, emotion in enumerate(config.emotions):
        own_column = EMOTIONS.index(emotion)
        readers = results[emotion].readers[: config.n_steer_seeds]
        for i, reader in enumerate(readers):
            dose = dose_with_none(
                steering_arms(reader),
                config.coeffs,
                prompts,
                score,
                none,
                columns=columns,
            )
            trials.extend(
                dataclasses.replace(trial, name=f"{emotion}/{i}/{coeff:g}")
                for coeff, trial in zip(dose.coeffs, dose.trials, strict=True)
                if coeff != 0.0
            )
            effect_rows.extend(
                effects_row(
                    effects_of(steering_arms(reader)(coeff), trial),
                    "reading",
                    coeff,
                    i,
                    emotion,
                )
                for coeff, trial in zip(dose.coeffs, dose.trials, strict=True)
                if coeff != 0.0
            )
            moved = dose.trials[at_strength].mean_score() - baseline
            for j in range(len(_JUDGES)):
                own[j, e, i] = dose.curve(j * _N + own_column)
                change[j, e, i] = moved[j * _N : (j + 1) * _N]

    random_trials: list[Trial] = []
    random_change = np.full(
        (config.n_random, len(config.coeffs), len(_JUDGES) * _N), np.nan
    )
    for r, reader in enumerate(randoms):
        dose = dose_with_none(
            steering_arms(reader),
            config.coeffs,
            prompts,
            score,
            none,
            columns=columns,
        )
        random_trials.extend(
            dataclasses.replace(trial, name=f"random/{r}/{coeff:g}")
            for coeff, trial in zip(dose.coeffs, dose.trials, strict=True)
            if coeff != 0.0
        )
        effect_rows.extend(
            effects_row(
                effects_of(steering_arms(reader)(coeff), trial),
                "random",
                coeff,
                r,
                None,
            )
            for coeff, trial in zip(dose.coeffs, dose.trials, strict=True)
            if coeff != 0.0
        )
        for c, trial in enumerate(dose.trials):
            random_change[r, c] = trial.mean_score() - baseline

    xlabel = (
        "Coefficient (relative)" if config.mode == "relative" else "Coefficient"
    )
    write_dose_curves(
        run_dir,
        own[0],
        config.coeffs,
        series=list(config.emotions),
        stem="curves",
        xlabel=xlabel,
        ylabel="Own-emotion probability (primary judge)",
        seed=config.seed,
    )
    write_dose_curves(
        run_dir,
        own[1],
        config.coeffs,
        series=list(config.emotions),
        stem="curves_check",
        xlabel=xlabel,
        ylabel="Own-emotion probability (check judge)",
        seed=config.seed,
    )
    matrices = change.mean(axis=2)
    steered = [EMOTIONS.index(emotion) for emotion in config.emotions]
    summary: dict[str, Any] = {
        "reading_layer": reading_layer,
        "window": list(window),
        "strength": strength,
        "n_prompts": config.n_prompts,
    }
    for j, stem in enumerate(("matrix", "matrix_check")):
        _write_matrix(
            run_dir,
            stem,
            matrices[j],
            config=config,
            title=(
                f"Change in measured emotion at strength {strength:g} "
                f"({_JUDGES[j]} judge)"
            ),
            layer=reading_layer,
        )
        hit_rate, specificity = diagonal_stats(matrices[j][:, steered])
        summary[f"{stem}_hit_rate"] = hit_rate
        summary[f"{stem}_specificity"] = specificity
    first, second = matrices[0].ravel(), matrices[1].ravel()
    summary["judge_correlation"] = (
        float(np.corrcoef(first, second)[0, 1])
        if first.std() > 0 and second.std() > 0
        else float("nan")
    )
    summary["max_abs_random_change"] = max_abs(random_change)
    write_trials(run_dir, "trials", trials)
    write_trials(run_dir, "trials_random", random_trials)
    run_dir.write_json(
        "table_effects",
        {"columns": list(_EFFECT_COLUMNS), "rows": effect_rows},
    )
    run_dir.write_json(
        "table_random",
        {
            "columns": [
                "steered",
                "random",
                "measured",
                "change",
                "change_check",
            ],
            "rows": [
                [
                    "random",
                    r,
                    measured,
                    float(random_change[r, at_strength, m]),
                    float(random_change[r, at_strength, _N + m]),
                ]
                for r in range(config.n_random)
                for m, measured in enumerate(EMOTIONS)
            ],
        },
    )
    run_dir.write_json(
        "table",
        {
            "columns": ["steered", "measured", "change", "change_check"],
            "rows": [
                [
                    emotion,
                    measured,
                    float(matrices[0][e, m]),
                    float(matrices[1][e, m]),
                ]
                for e, emotion in enumerate(config.emotions)
                for m, measured in enumerate(EMOTIONS)
            ],
        },
    )
    run_dir.write_json("summary", summary)


def run(config: Config, run_dir: RunDir) -> None:
    """Loads GPT-2, the emotions, WikiText and the two judges, then runs.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.

    Raises:
        ValueError: If the steered emotions are not at least two distinct
            names of `EMOTIONS`, or `n_pairs` or `n_eval_pairs` is below 1
            or exceeds the pairs one steered emotion can form in its split
            (an emotion with no pair fails even when `n_eval_pairs` is
            None). Both are raised before any judge or the model is loaded.
            An unknown judge or model name raises while it is loaded
            (`UnknownCheckpointError` is a `ValueError` too). Every other
            setting raises as in `execute`, once the judges and the model are
            loaded.
    """
    scenarios = load_emotions()
    _check_pair_counts(config, scenarios)
    texts = [example.text for example in load_wikitext()]
    first = load_judge(config.judge)
    second = load_judge(config.judge_check)

    def score(batch: Sequence[str]) -> NDArray[np.float32]:
        return np.concatenate([first.score(batch), second.score(batch)], axis=1)

    execute(config, run_dir, load_model(config.model), scenarios, texts, score)


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws the own-emotion curves and the two change matrices.

    Args:
        run_dir: A finished C2 run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths: `curves`, `curves_check`, `matrix`, then
        `matrix_check`.
    """
    paths = [
        path
        for stem in ("curves", "curves_check")
        for path in plot_curves_file(run_dir, out_dir, stem=stem)
    ]
    for stem in ("matrix", "matrix_check"):
        values = run_dir.read_arrays(stem)["values"]
        info = cast(dict[str, Any], run_dir.read_json(stem))
        with style_context():
            fig = plot_matrix(
                values,
                rows=info["rows"],
                cols=info["cols"],
                title=info["title"],
                kind="diverging",
                cbar_label="Change in probability",
            )
        paths += save_figure(fig, out_dir, stem)
    return paths
