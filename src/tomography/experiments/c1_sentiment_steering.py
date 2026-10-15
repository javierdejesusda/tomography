"""C1: steering sentiment on GPT-2 with a reading vector.

Readers are fitted on SST-2 as in R1 and steer text generated from short
neutral prompts. A layer sweep on one prompt list chooses where to steer, and
the operators, the prompting baseline and the side effects are reported on a
second list. Every completion is scored by two judges at once.
"""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tomography.control.controllers import mean_residual_norms
from tomography.control.operators import Operator
from tomography.datasets.records import Example
from tomography.datasets.sst2 import load_sst2
from tomography.datasets.stimulus import StimulusSet
from tomography.datasets.stimulus import build_stimulus_set
from tomography.datasets.wikitext import load_wikitext
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.effects import SideEffects
from tomography.experiments.effects import side_effects
from tomography.experiments.effects import split_held_out
from tomography.experiments.effects import write_dose_curves
from tomography.experiments.effects import write_trials
from tomography.experiments.r1_sentiment import select_examples
from tomography.experiments.steering import Arm
from tomography.experiments.steering import DoseResponse
from tomography.experiments.steering import GenSettings
from tomography.experiments.steering import Scorer
from tomography.experiments.steering import SteeringConfig
from tomography.experiments.steering import Trial
from tomography.experiments.steering import dose_response
from tomography.experiments.steering import dose_slope
from tomography.experiments.steering import plain_arm
from tomography.experiments.steering import reader_arm
from tomography.experiments.steering import run_trial
from tomography.experiments.sweep import SERIES
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

_COLUMNS = (
    "arm",
    "coeff",
    "seed",
    "score",
    "score_check",
    "perplexity",
    "fluency",
    "distinct_1",
    "distinct_2",
    "mean_length",
    "empty_rate",
)
_MAX_SERIES = 8


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


def judge_agreement(
    primary: NDArray[np.float64],
    check: NDArray[np.float64],
    coeffs: Sequence[float],
) -> float:
    """Measures how often the two judges see the dose move the same way.

    Args:
        primary: Primary-judge scores of shape (S, N, K): layer, reader,
            coefficient.
        check: Check-judge scores of the same shape.
        coeffs: The K coefficients.

    Returns:
        The share of layers whose mean dose slope has the same sign under
        both judges, among the layers with a finite slope under both; NaN
        when no layer has one.
    """
    first = _mean_slopes(primary, coeffs)
    second = _mean_slopes(check, coeffs)
    both = np.isfinite(first) & np.isfinite(second)
    if not both.any():
        return float("nan")
    return float(np.mean(np.sign(first[both]) == np.sign(second[both])))


def check_pair_count(
    name: str, wanted: int | None, available: int, where: str
) -> None:
    """Checks that a split can form the pairs a setting asks for.

    Args:
        name: The setting, as the message names it.
        wanted: Pairs asked for; None means every pair, which needs at least
            one.
        available: Pairs the split can form.
        where: Where the pairs are drawn from, as the message words it.

    Raises:
        ValueError: If `wanted` is below 1 or above `available`, or is None
            while no pair can be formed.
    """
    if wanted is not None and wanted < 1:
        raise ValueError(f"{name} must be at least 1 or None, got {wanted}")
    if (1 if wanted is None else wanted) > available:
        raise ValueError(
            f"{name} is {wanted} but at most {available} pairs exist {where}"
        )


def _check_pair_counts(
    config: Config, train: Sequence[Example], val: Sequence[Example]
) -> None:
    """Checks that each split forms the pairs the settings ask for."""
    for split, name, wanted, examples in (
        ("train", "n_pairs", config.n_pairs, train),
        ("val", "n_eval_pairs", config.n_eval_pairs, val),
    ):
        positive = sum(e.label == 1 for e in examples)
        check_pair_count(
            name,
            wanted,
            min(positive, len(examples) - positive),
            f"in the {split} split",
        )


def _stimuli(
    examples: Sequence[Example],
    chat_format: ChatFormat,
    n_pairs: int | None,
    seed: int,
) -> StimulusSet:
    return build_stimulus_set(
        examples,
        PAPER_TEMPLATES["concept"],
        chat_format,
        pairing="label_contrastive",
        concept="sentiment",
        n_pairs=n_pairs,
        seed=seed,
    )


def _check_config(config: Config, n_layers: int) -> tuple[int, ...]:
    """Checks the settings that need no data and returns the sweep layers."""
    if not 1 <= config.n_steer_seeds <= config.n_seeds:
        raise ValueError(
            f"n_steer_seeds must be in 1..{config.n_seeds}, got "
            f"{config.n_steer_seeds}"
        )
    if 0.0 not in config.coeffs or all(c == 0.0 for c in config.coeffs):
        raise ValueError("coeffs must contain 0.0 and a nonzero value")
    sweep = (
        default_sweep_layers(n_layers)
        if config.sweep_layers is None
        else config.sweep_layers
    )
    if not 1 <= len(sweep) <= _MAX_SERIES:
        raise ValueError(
            f"sweep_layers needs 1 to {_MAX_SERIES} layers, got {len(sweep)}"
        )
    if len(set(sweep)) != len(sweep) or not all(
        1 <= layer <= n_layers for layer in sweep
    ):
        raise ValueError(
            f"sweep_layers must be distinct layers in 1..{n_layers}, got "
            f"{sweep}"
        )
    if not config.ops or len(set(config.ops)) != len(config.ops):
        raise ValueError(
            f"ops must be non-empty and distinct, got {config.ops}"
        )
    if not 1 <= config.window <= n_layers:
        raise ValueError(
            f"window must be in 1..{n_layers}, got {config.window}"
        )
    return sweep


@dataclass(frozen=True)
class _Steering:
    """What every steered arm of a run shares."""

    config: Config
    loaded: LoadedModel
    settings: GenSettings
    norms: dict[int, float]
    score: Scorer
    plain: Arm

    def none(self, prompts: Sequence[str]) -> Trial:
        return run_trial(self.plain, prompts, self.score, columns=2)

    def arm(
        self,
        reader: RepReader,
        layers: Sequence[int],
        op: Operator,
        coeff: float,
    ) -> Arm:
        return reader_arm(
            self.loaded,
            self.settings,
            reader,
            layers=layers,
            coeff=coeff,
            norms=self.norms,
            op=op,
            mode=self.config.mode,
            mask=self.config.mask,
            preserve_norm=self.config.preserve_norm,
        )

    def dose(
        self,
        reader: RepReader,
        layers: Sequence[int],
        op: Operator,
        prompts: Sequence[str],
        none: Trial,
    ) -> DoseResponse:
        return dose_with_none(
            lambda coeff: self.arm(reader, layers, op, coeff),
            self.config.coeffs,
            prompts,
            self.score,
            none,
            columns=2,
        )

    def effects(
        self, arm: Arm, trial: Trial, neutral: Sequence[str]
    ) -> SideEffects:
        return side_effects(
            self.loaded,
            arm,
            trial,
            neutral,
            batch_size=self.config.batch_size,
        )


def _steered_trials(dose: DoseResponse, prefix: str) -> list[Trial]:
    """Names the nonzero-coefficient trials `<prefix>/<coefficient>`."""
    return [
        dataclasses.replace(trial, name=f"{prefix}/{coeff:g}")
        for coeff, trial in zip(dose.coeffs, dose.trials, strict=True)
        if coeff != 0.0
    ]


def _perplexity(effects: SideEffects) -> float:
    return float("nan") if effects.perplexity is None else effects.perplexity


def _row(
    arm: str,
    coeff: float | None,
    seed: int | None,
    trial: Trial,
    effects: SideEffects,
) -> list[object]:
    mean = trial.mean_score()
    return [
        arm,
        coeff,
        seed,
        float(mean[0]),
        float(mean[1]),
        effects.perplexity,
        effects.fluency,
        effects.distinct_1,
        effects.distinct_2,
        effects.mean_length,
        effects.empty_rate,
    ]


def _layer_sweep(
    steering: _Steering,
    readers: Sequence[RepReader],
    sweep: Sequence[int],
    prompts: Sequence[str],
    none: Trial,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[Trial]]:
    """Runs the dose response of every reader at every single layer."""
    shape = (len(sweep), len(readers), len(steering.config.coeffs))
    primary = np.full(shape, np.nan)
    check = np.full(shape, np.nan)
    trials = [none]
    for at, layer in enumerate(sweep):
        for i, reader in enumerate(readers):
            dose = steering.dose(reader, [layer], "linear", prompts, none)
            primary[at, i] = dose.curve(0)
            check[at, i] = dose.curve(1)
            trials.extend(_steered_trials(dose, f"layer{layer}/{i}"))
    return primary, check, trials


@dataclass(frozen=True, eq=False)
class _Headline:
    """The report-prompt results of the chosen window."""

    operators: NDArray[np.float64]
    perplexity: NDArray[np.float64]
    trials: list[Trial]
    rows: list[list[object]]


def _compare_operators(
    steering: _Steering,
    readers: Sequence[RepReader],
    window: Sequence[int],
    prompts: Sequence[str],
    none: Trial,
    neutral: Sequence[str],
) -> _Headline:
    """Runs every operator at the window and measures the first one's cost."""
    config = steering.config
    shape = (len(readers), len(config.coeffs))
    operators = np.full((len(config.ops), *shape), np.nan)
    perplexity = np.full((1, *shape), np.nan)
    plain_effects = steering.effects(steering.plain, none, neutral)
    trials = [none]
    rows = [_row("none", 0.0, None, none, plain_effects)]
    for at, op in enumerate(config.ops):
        for i, reader in enumerate(readers):
            dose = steering.dose(reader, window, op, prompts, none)
            operators[at, i] = dose.curve(0)
            trials.extend(_steered_trials(dose, f"{op}/{i}"))
            if at > 0:
                continue
            effects = [
                plain_effects
                if coeff == 0.0
                else steering.effects(
                    steering.arm(reader, window, op, coeff), trial, neutral
                )
                for coeff, trial in zip(dose.coeffs, dose.trials, strict=True)
            ]
            perplexity[0, i] = [_perplexity(e) for e in effects]
            rows.extend(
                _row(op, coeff, i, trial, e)
                for coeff, trial, e in zip(
                    dose.coeffs, dose.trials, effects, strict=True
                )
                if coeff != 0.0
            )
    return _Headline(operators, perplexity, trials, rows)


def _prompting_baseline(
    steering: _Steering, prompts: Sequence[str], neutral: Sequence[str]
) -> tuple[list[Trial], list[list[object]]]:
    """Runs the plain model with each instruction prepended to the prompt."""
    config = steering.config
    trials: list[Trial] = []
    rows: list[list[object]] = []
    for sign, instructions in (
        ("+", config.prompt_positive),
        ("-", config.prompt_negative),
    ):
        for level, prefix in enumerate(instructions, start=1):
            arm = plain_arm(
                steering.loaded,
                steering.settings,
                name=f"prompting{sign}{level}",
                prefix=prefix,
            )
            trial = run_trial(arm, prompts, steering.score, columns=2)
            trials.append(trial)
            rows.append(
                _row(
                    arm.name,
                    None,
                    None,
                    trial,
                    steering.effects(arm, trial, neutral),
                )
            )
    return trials, rows


def _write_curves(
    run_dir: RunDir,
    config: Config,
    per_seed: NDArray[np.float64],
    *,
    stem: str,
    series: Sequence[str],
    ylabel: str,
) -> None:
    write_dose_curves(
        run_dir,
        per_seed,
        config.coeffs,
        series=series,
        stem=stem,
        xlabel=(
            "Coefficient (relative)"
            if config.mode == "relative"
            else "Coefficient"
        ),
        ylabel=ylabel,
        seed=config.seed,
    )


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    sst2: Sequence[Example],
    texts: Sequence[str],
    score: Scorer,
) -> None:
    """Reads sentiment, picks the steering layer and reports the dose.

    Readers are fitted on SST-2 train pairs and checked on val pairs. The
    steering layer is not the reading layer: it is the sweep layer where the
    primary judge's score rises the most per unit of coefficient, measured
    on a first list of prompts. The window around it, the operators, the
    prompting baseline and the side effects are measured on a second list,
    so the headline is not scored on the prompts that chose the layer.
    Coefficient 0 is generated once per prompt list.

    Writes `curves_layers` and `curves_layers_check` (one series per sweep
    layer), `curves_operators` and `curves_perplexity`, `trials_sweep`
    (selection prompts) and `trials` (report prompts), `table.json` and
    `summary.json`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model.
        sst2: SST-2 examples with `train` and `val` splits.
        texts: Held-out texts that set the relative coefficients and give
            the neutral perplexity.
        score: Maps texts to scores of shape (n, 2): the primary judge, then
            the check judge.

    Raises:
        ValueError: If `coeffs` has no 0.0 or no other value, `n_steer_seeds`
            is outside `1..n_seeds`, the sweep layers or the window do not
            fit the model, the operators are empty or repeated, the
            generation settings are invalid (see `GenSettings`),
            `n_calibration` or `n_neutral` is below 1, the usable (at least
            40 characters) held-out texts are fewer than
            `n_calibration + n_neutral`, `n_pairs` or `n_eval_pairs` is below
            1 or exceeds the pairs its split can form (`n_eval_pairs` of None
            needs one), or the validation sentences give fewer than
            `2 * n_prompts` distinct prompts.
    """
    n_layers = loaded.n_layers
    sweep = _check_config(config, n_layers)
    settings = GenSettings.from_config(config)
    held = split_held_out(
        texts,
        n_calibration=config.n_calibration,
        n_neutral=config.n_neutral,
        seed=config.seed,
    )
    train = select_examples(sst2, split="train", min_words=config.min_words)
    val = select_examples(sst2, split="val", min_words=config.min_words)
    _check_pair_counts(config, train, val)
    select_prompts, report_prompts = draw_prompts(
        prompt_fragments([e.text for e in val], config.truncate_words),
        config.n_prompts,
        sets=2,
        seed=derive_seed(config.seed, "prompts"),
    )
    chat_format = chat_format_for(loaded)
    result = reading_sweep(
        loaded,
        [
            _stimuli(
                train,
                chat_format,
                config.n_pairs,
                derive_seed(config.seed, "train", i),
            )
            for i in range(config.n_seeds)
        ],
        {
            "val": _stimuli(
                val,
                chat_format,
                config.n_eval_pairs,
                derive_seed(config.seed, "eval"),
            )
        },
        method=config.method,
        seed=config.seed,
        bos=config.bos,
        batch_size=config.batch_size,
    )
    reading_layer = best_layer(
        result.per_seed("val")[SERIES.index("reader")].mean(axis=0),
        result.layers,
    )
    readers = result.readers[: config.n_steer_seeds]
    steering = _Steering(
        config=config,
        loaded=loaded,
        settings=settings,
        norms=mean_residual_norms(
            loaded,
            held.calibration,
            range(1, n_layers + 1),
            bos=config.bos,
            batch_size=config.batch_size,
        ),
        score=score,
        plain=plain_arm(loaded, settings),
    )

    primary, check, sweep_trials = _layer_sweep(
        steering, readers, sweep, select_prompts, steering.none(select_prompts)
    )
    layer_series = [f"layer {layer}" for layer in sweep]
    _write_curves(
        run_dir,
        config,
        primary,
        stem="curves_layers",
        series=layer_series,
        ylabel="Sentiment (primary judge)",
    )
    _write_curves(
        run_dir,
        config,
        check,
        stem="curves_layers_check",
        series=layer_series,
        ylabel="Sentiment (check judge)",
    )
    write_trials(run_dir, "trials_sweep", sweep_trials)
    steer_layer = slope_layer(primary, config.coeffs, sweep)
    window = window_around(steer_layer, config.window, n_layers)

    headline = _compare_operators(
        steering,
        readers,
        window,
        report_prompts,
        steering.none(report_prompts),
        held.neutral,
    )
    _write_curves(
        run_dir,
        config,
        headline.operators,
        stem="curves_operators",
        series=list(config.ops),
        ylabel="Sentiment (primary judge)",
    )
    _write_curves(
        run_dir,
        config,
        headline.perplexity,
        stem="curves_perplexity",
        series=["neutral perplexity"],
        ylabel="Perplexity",
    )
    prompting_trials, prompting_rows = _prompting_baseline(
        steering, report_prompts, held.neutral
    )
    write_trials(run_dir, "trials", [*headline.trials, *prompting_trials])
    run_dir.write_json(
        "table",
        {"columns": list(_COLUMNS), "rows": [*headline.rows, *prompting_rows]},
    )
    run_dir.write_json(
        "summary",
        {
            "reading_layer": reading_layer,
            "steer_layer": steer_layer,
            "window": list(window),
            "sweep_layers": list(sweep),
            "judge_agreement": judge_agreement(primary, check, config.coeffs),
            "n_prompts": config.n_prompts,
        },
    )


def run(config: Config, run_dir: RunDir) -> None:
    """Loads GPT-2, SST-2, WikiText and the two judges, then runs `execute`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
    """
    first = load_judge(config.judge)
    second = load_judge(config.judge_check)

    def score(texts: Sequence[str]) -> NDArray[np.float32]:
        return np.column_stack([first.score(texts), second.score(texts)])

    execute(
        config,
        run_dir,
        load_model(config.model),
        load_sst2(),
        [example.text for example in load_wikitext()],
        score,
    )


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws the layer, check, operator and perplexity curves.

    Args:
        run_dir: A finished C1 run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths of the four figures.
    """
    return [
        path
        for stem in (
            "curves_layers",
            "curves_layers_check",
            "curves_operators",
            "curves_perplexity",
        )
        for path in plot_curves_file(run_dir, out_dir, stem=stem)
    ]
