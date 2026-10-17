"""M1: six direction finders compared on correlation, steering and erasure.

Each finder reads one concept (utility, from ETHICS) and is then compared on
three tasks: correlation (does the direction read held-out pairs),
manipulation (does steering along it change the sentiment of the
continuation of a truncated scenario) and termination (does projecting it
out remove the concept). The `random` finder is the null of the comparison.
"""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tomography.activations import PositionStrategy
from tomography.control.controllers import mean_residual_norms
from tomography.datasets.ethics import load_ethics_utility
from tomography.datasets.records import Example
from tomography.datasets.records import Split
from tomography.datasets.stimulus import StimulusSet
from tomography.datasets.stimulus import build_stimulus_set
from tomography.datasets.wikitext import load_wikitext
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.controls import recording
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
from tomography.experiments.steering import dose_response
from tomography.experiments.steering import plain_arm
from tomography.experiments.steering import reader_arm
from tomography.experiments.steering import run_trial
from tomography.experiments.sweep import SERIES
from tomography.experiments.sweep import Prepared
from tomography.experiments.sweep import SweepResult
from tomography.experiments.sweep import chat_format_for
from tomography.experiments.sweep import plot_curves_file
from tomography.experiments.sweep import stimulus_activations
from tomography.experiments.sweep import sweep_from_activations
from tomography.experiments.sweep import write_curves
from tomography.judges import load_judge
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.reading.directions import Method
from tomography.reading.directions import Preprocess
from tomography.reading.metrics import best_layer
from tomography.reading.metrics import bootstrap_ci
from tomography.reading.metrics import group_accuracy
from tomography.reading.metrics import wilson_ci
from tomography.reading.reader import RepReader
from tomography.templates import PAPER_TEMPLATES
from tomography.templates import ChatFormat

METHODS: tuple[Method, ...] = (
    "pca_diff",
    "pca_raw",
    "mean_diff",
    "logistic",
    "kmeans",
    "random",
)

_CONTROL_DEPTHS = (0.4, 0.55, 0.7)
_ZERO_NORM = "zero norm"
_EVAL_SPLITS: tuple[Split, ...] = ("val", "test")
_READER = SERIES.index("reader")
_JUDGES = ("primary", "check")
_MANIPULATION_TASKS = ("manipulation", "manipulation_check")
_GAP_TASKS = ("termination_gap", "termination_gap_check")
_SUCCESS_CURVES = (
    ("curves_success", "Success rate"),
    ("curves_success_check", "Success rate (check judge)"),
)
_CURVE_STEMS = ("curves", "curves_val", *(stem for stem, _ in _SUCCESS_CURVES))
_TABLE_COLUMNS = ("method", "task", "setting", "value", "lo", "hi", "n")
_SEED_COLUMNS = (
    "method",
    "seed",
    "strength",
    "judge",
    "wins",
    "losses",
    "ties",
    "rate",
    "lo",
    "hi",
)
_EFFECT_COLUMNS = (
    "method",
    "seed",
    "coeff",
    "perplexity",
    "fluency",
    "distinct_1",
    "distinct_2",
    "mean_length",
    "empty_rate",
)
_NAN = float("nan")


@dataclass(frozen=True, kw_only=True)
class Config(SteeringConfig):
    """M1 settings: six direction finders on utility, read and steered.

    The shared steering fields apply, with two new defaults: 250 prompts
    and one sample per prompt. `layers` are the layers every method is
    steered and projected at; None means `default_control_layers`.

    Attributes:
        methods: The direction finders to compare.
        judge: Key of `judges.JUDGES` whose score is column 0 and drives
            the headline rows.
        judge_check: Key of the judge whose score is column 1, the check.
        n_seeds: Training sets, hence readers per method (correlation).
        n_steer_seeds: The first readers of each method that steer and are
            projected out.
        n_pairs: Training pairs per seed.
        n_eval_pairs: Evaluation pairs of the val and test splits; None
            means every pair.
        preprocess: Vector preparation of the finders.
        position: Token positions read.
        truncate_words: Words of a test scenario kept as a prompt.
        n_termination: Test pairs of the behavioral termination gap.
        fluency: When True, M1 measures the fluency and side effects of every
            steered trial (neutral perplexity and completion fluency) and
            writes `table_effects`. When False it skips those extra passes
            and writes no `table_effects`, so the run costs what it did
            without them.
    """

    experiment: str = "m1"
    model: str = "gpt2"
    methods: tuple[Method, ...] = METHODS
    judge: str = "sentiment-primary"
    judge_check: str = "sentiment-secondary"
    n_seeds: int = 10
    n_steer_seeds: int = 3
    n_pairs: int = 64
    n_eval_pairs: int | None = None
    preprocess: Preprocess = "center"
    position: PositionStrategy = "last"
    truncate_words: int = 6
    n_termination: int = 100
    n_prompts: int = 250
    num_samples: int = 1
    fluency: bool = True


@dataclass(frozen=True)
class Tally:
    """Wins, losses and ties of paired comparisons.

    Attributes:
        wins: Pairs whose first side is strictly higher.
        losses: Pairs whose first side is strictly lower.
        ties: Pairs with equal sides.
    """

    wins: int
    losses: int
    ties: int

    @property
    def decisive(self) -> int:
        """Pairs that are a win or a loss."""
        return self.wins + self.losses

    @property
    def rate(self) -> float:
        """Wins over decisive pairs; NaN when no pair is decisive."""
        return self.wins / self.decisive if self.decisive else float("nan")


def truncate_scenario(text: str, words: int) -> str:
    """Cuts a scenario to its first words.

    Args:
        text: The scenario.
        words: Number of whitespace-separated words to keep.

    Returns:
        The first `words` words joined by single spaces; a shorter text
        keeps all its words.

    Raises:
        ValueError: If `words` is below 1.
    """
    if words < 1:
        raise ValueError(f"words must be at least 1, got {words}")
    return " ".join(text.split()[:words])


def tally(
    positive: NDArray[np.float64], negative: NDArray[np.float64]
) -> Tally:
    """Counts, per prompt, whether the first score is above the second.

    Args:
        positive: One score per prompt under the positive coefficient.
        negative: One score per prompt under the negative coefficient.

    Returns:
        The wins, losses and ties. A prompt with a NaN score on either
        side (an empty completion) counts as none of them.

    Raises:
        ValueError: If the inputs are not two vectors of one length.
    """
    if positive.shape != negative.shape or positive.ndim != 1:
        raise ValueError(
            "need two vectors of one length, got shapes "
            f"{positive.shape} and {negative.shape}"
        )
    return Tally(
        wins=int((positive > negative).sum()),
        losses=int((positive < negative).sum()),
        ties=int((positive == negative).sum()),
    )


def paired_strengths(coeffs: Sequence[float]) -> tuple[float, ...]:
    """Returns the positive half of a symmetric coefficient grid.

    Args:
        coeffs: The coefficients, in any order.

    Returns:
        The positive values, sorted.

    Raises:
        ValueError: If `coeffs` lacks 0.0, has no positive value, or has a
            positive value without its negative or a negative value without
            its positive.
    """
    if 0.0 not in coeffs:
        raise ValueError("coeffs must contain 0.0")
    strengths = tuple(sorted(c for c in coeffs if c > 0))
    if not strengths:
        raise ValueError("coeffs must hold at least one positive strength")
    positive_side = {c for c in coeffs if c > 0}
    negative_side = {-c for c in coeffs if c < 0}
    if positive_side != negative_side:
        raise ValueError(
            "coeffs must hold every strength with its negative, got "
            f"{tuple(sorted(coeffs))}"
        )
    return strengths


def project_out(
    h: NDArray[np.float32], direction: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Removes the component of every vector along a direction.

    Args:
        h: Vectors of shape (..., d).
        direction: A vector of shape (d,); it need not be a unit vector.

    Returns:
        The vectors with no component along the direction, in float32. The
        arithmetic runs in float64.

    Raises:
        ValueError: If `direction` is not a nonzero vector of the width of
            `h`.
    """
    if direction.ndim != 1 or direction.shape[0] != h.shape[-1]:
        raise ValueError(
            f"direction must have shape ({h.shape[-1]},), got {direction.shape}"
        )
    unit = direction.astype(np.float64)
    norm = np.linalg.norm(unit)
    if not (np.isfinite(norm) and norm > 0):
        raise ValueError("direction must be finite and nonzero")
    unit /= norm
    wide = h.astype(np.float64)
    return (wide - (wide @ unit)[..., None] * unit).astype(np.float32)


def _probe_accuracy(
    train_h: NDArray[np.float32],
    train_labels: NDArray[np.int64],
    test_h: NDArray[np.float32],
    test_groups: NDArray[np.int64],
    test_labels: NDArray[np.int64],
    seed: int,
) -> float:
    try:
        probe = RepReader.fit(
            "logistic", train_h[None], [0], labels=train_labels, seed=seed
        )
    except ValueError as error:
        if _ZERO_NORM not in str(error):
            raise
        return 0.5
    scores = probe.score(test_h[None])
    return float(group_accuracy(scores, test_groups, test_labels)[0])


def erased_probe_accuracy(
    train_h: NDArray[np.float32],
    train_labels: NDArray[np.int64],
    test_h: NDArray[np.float32],
    test_groups: NDArray[np.int64],
    test_labels: NDArray[np.int64],
    direction: NDArray[np.float32],
    *,
    seed: int,
) -> tuple[float, float]:
    """Measures a fresh logistic probe before and after an erasure.

    The probe is fit on `train_h` and scores the pairs of `test_h`. The
    second probe is fit and scored after `direction` is projected out of
    both sets. A logistic fit that `RepReader.fit` rejects for a zero-norm
    direction (what projecting out the class-mean difference leaves) is a
    constant scorer, which counts as chance, 0.5.

    Args:
        train_h: Training activations of one layer, shape (n, d).
        train_labels: Binary labels of the training rows.
        test_h: Test activations of the same layer, shape (m, d).
        test_groups: Row groups of `test_h`, shape (g, k), each with one
            label-1 member.
        test_labels: Binary labels of the test rows.
        direction: The direction to project out, shape (d,).
        seed: Seed of the probe.

    Returns:
        The group accuracy of the probe before and after the erasure.

    Raises:
        ValueError: If the probe cannot be fit or scored for a reason other
            than a zero-norm direction, or `direction` is unusable.
    """
    before = _probe_accuracy(
        train_h, train_labels, test_h, test_groups, test_labels, seed
    )
    after = _probe_accuracy(
        project_out(train_h, direction),
        train_labels,
        project_out(test_h, direction),
        test_groups,
        test_labels,
        seed,
    )
    return before, after


def default_control_layers(n_layers: int) -> tuple[int, ...]:
    """Returns the layers every method is steered at.

    They sit at 40, 55 and 70 percent of the depth, so the layer is not a
    confound between methods.

    Args:
        n_layers: Number of transformer layers.

    Returns:
        The rounded layers, at least 1, without duplicates, sorted.
    """
    return tuple(sorted({max(round(n_layers * f), 1) for f in _CONTROL_DEPTHS}))


@dataclass(frozen=True)
class _Reading:
    """The readers of every method and the activations they were fit on."""

    train: list[Prepared]
    test: Prepared
    sweeps: dict[Method, SweepResult]
    rows: list[list[object]]


@dataclass(frozen=True)
class _Steering:
    """What every steered arm of the run shares."""

    config: Config
    loaded: LoadedModel
    settings: GenSettings
    score: Scorer
    control: tuple[int, ...]
    norms: dict[int, float]


@dataclass(frozen=True)
class _Manipulation:
    """The steered trials of every method and what they were scored to."""

    trials: list[Trial]
    rows: list[list[object]]
    seed_rows: list[list[object]]
    effect_rows: list[list[object]]
    success: NDArray[np.float64]
    agreement: float


@dataclass(frozen=True)
class _Termination:
    """The unsteered and projected trials on the full scenarios."""

    trials: list[Trial]
    rows: list[list[object]]


def _part(examples: Sequence[Example], split: Split) -> list[Example]:
    return [e for e in examples if e.split == split]


def _stimuli(
    examples: Sequence[Example],
    chat_format: ChatFormat,
    n_pairs: int | None,
    seed: int,
) -> StimulusSet:
    return build_stimulus_set(
        examples,
        PAPER_TEMPLATES["utility"],
        chat_format,
        pairing="by_group",
        group_ids=[str(e.meta["pair"]) for e in examples],
        n_pairs=n_pairs,
        seed=seed,
    )


def _gap_scenarios(
    test_rows: Sequence[Example], limit: int
) -> tuple[list[str], list[str]]:
    """Returns the pleasant and the unpleasant scenario of each test pair."""
    pairs: dict[str, dict[int, str]] = {}
    for example in test_rows:
        pairs.setdefault(str(example.meta["pair"]), {})[example.label] = (
            example.text
        )
    complete = [p for p in pairs.values() if 0 in p and 1 in p][:limit]
    if not complete:
        raise ValueError("no test pair has both scenarios")
    return [p[1] for p in complete], [p[0] for p in complete]


@dataclass(frozen=True)
class _Checked:
    """What `_check` settles before any forward."""

    strengths: tuple[float, ...]
    control: tuple[int, ...]
    pleasant: list[str]
    unpleasant: list[str]
    scenarios: list[str]


def _check_grid(config: Config) -> tuple[float, ...]:
    """Checks the methods and the coefficients; returns the strengths.

    Trial names and table keys are built from the method and the `:g` label
    of a coefficient, so both must be distinct.
    """
    strengths = paired_strengths(config.coeffs)
    if len(set(config.methods)) != len(config.methods):
        raise ValueError(f"methods must not repeat, got {config.methods}")
    if not all(math.isfinite(c) for c in config.coeffs):
        raise ValueError(f"coeffs must be finite, got {config.coeffs}")
    labels = [f"{c:g}" for c in config.coeffs if c != 0.0]
    if len(set(labels)) != len(labels):
        raise ValueError(
            "nonzero coeffs must differ at the six digits of their labels, "
            f"got {config.coeffs}"
        )
    return strengths


def _check(
    config: Config, loaded: LoadedModel, utility: Sequence[Example]
) -> _Checked:
    """Settles the grid, the layers, the prompts and the pairs."""
    strengths = _check_grid(config)
    if not 1 <= config.n_steer_seeds <= config.n_seeds:
        raise ValueError(
            f"n_steer_seeds must be in 1..n_seeds={config.n_seeds}, got "
            f"{config.n_steer_seeds}"
        )
    control = (
        default_control_layers(loaded.n_layers)
        if config.layers is None
        else tuple(config.layers)
    )
    outside = [layer for layer in control if not 1 <= layer <= loaded.n_layers]
    if outside:
        raise ValueError(
            f"control layers {outside} are outside 1..{loaded.n_layers}"
        )
    if len(set(control)) != len(control):
        raise ValueError(f"control layers must not repeat, got {control}")
    test_rows = _part(utility, "test")
    scenarios = list(
        dict.fromkeys(
            truncate_scenario(e.text, config.truncate_words) for e in test_rows
        )
    )
    if len(scenarios) < config.n_prompts:
        raise ValueError(
            f"n_prompts={config.n_prompts} exceeds the {len(scenarios)} "
            "distinct truncated test scenarios"
        )
    pleasant, unpleasant = _gap_scenarios(test_rows, config.n_termination)
    return _Checked(strengths, control, pleasant, unpleasant, scenarios)


def _read(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    utility: Sequence[Example],
) -> _Reading:
    """Fits every method on every training set and writes the curves."""
    chat_format = chat_format_for(loaded)
    layers = tuple(range(1, loaded.n_layers + 1))

    def collect(stimuli: StimulusSet) -> Prepared:
        act = stimulus_activations(
            loaded,
            stimuli,
            layers=layers,
            position=config.position,
            bos=config.bos,
            batch_size=config.batch_size,
        )
        return stimuli, act

    train = [
        collect(
            _stimuli(
                _part(utility, "train"),
                chat_format,
                config.n_pairs,
                derive_seed(config.seed, "train", i),
            )
        )
        for i in range(config.n_seeds)
    ]
    evals: dict[str, Prepared] = {
        name: collect(
            _stimuli(
                _part(utility, name),
                chat_format,
                config.n_eval_pairs,
                derive_seed(config.seed, "eval", name),
            )
        )
        for name in _EVAL_SPLITS
    }
    sweeps = {
        method: sweep_from_activations(
            train,
            evals,
            method=method,
            preprocess=config.preprocess,
            seed=derive_seed(config.seed, method),
        )
        for method in config.methods
    }
    val = np.stack([sweeps[m].per_seed("val")[_READER] for m in config.methods])
    test = np.stack(
        [sweeps[m].per_seed("test")[_READER] for m in config.methods]
    )
    for stem, accuracy in (("curves", test), ("curves_val", val)):
        write_curves(
            run_dir,
            accuracy,
            layers,
            series=config.methods,
            stem=stem,
            chance=0.5,
            n_layers=loaded.n_layers,
            seed=config.seed,
        )
    n_test_pairs = int(evals["test"][0].groups.shape[0])
    rows: list[list[object]] = []
    for m, method in enumerate(config.methods):
        best = best_layer(val[m].mean(axis=0), layers)
        at = layers.index(best)
        lo, hi = bootstrap_ci(test[m, :, at], seed=config.seed)
        rows.append(
            [
                method,
                "correlation",
                f"layer{best}",
                float(test[m, :, at].mean()),
                float(lo),
                float(hi),
                n_test_pairs,
            ]
        )
    return _Reading(train, evals["test"], sweeps, rows)


def _prompts(config: Config, scenarios: Sequence[str]) -> list[str]:
    """Picks `n_prompts` of the distinct truncated test scenarios."""
    order = np.random.default_rng(
        derive_seed(config.seed, "prompts")
    ).permutation(len(scenarios))
    return [scenarios[int(k)] for k in order[: config.n_prompts]]


def _rate_interval(result: Tally) -> tuple[float, float]:
    """Wilson interval of a success rate; NaN bounds with no decisive pair."""
    if not result.decisive:
        return _NAN, _NAN
    return wilson_ci(result.wins, result.decisive)


def _spread(values: Sequence[float]) -> tuple[float, float, float]:
    """Mean, minimum and maximum of the finite values; NaN with none."""
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return _NAN, _NAN, _NAN
    return float(np.mean(finite)), min(finite), max(finite)


def _agreement(
    primary: NDArray[np.float64], check: NDArray[np.float64]
) -> float:
    """Share of cells whose rates fall on the same side of one half.

    Only cells with a finite rate under both judges count, and a rate of
    exactly one half has sign 0, which agrees only with another one half.
    """
    usable = np.isfinite(primary) & np.isfinite(check)
    if not usable.any():
        return _NAN
    same = np.sign(primary[usable] - 0.5) == np.sign(check[usable] - 0.5)
    return float(same.mean())


def _arm_factory(ctx: _Steering, reader: RepReader) -> Callable[[float], Arm]:
    def make(coeff: float) -> Arm:
        return reader_arm(
            ctx.loaded,
            ctx.settings,
            reader,
            layers=ctx.control,
            coeff=coeff,
            norms=ctx.norms,
            op=ctx.config.op,
            mode=ctx.config.mode,
            mask=ctx.config.mask,
            preserve_norm=ctx.config.preserve_norm,
        )

    return make


def _manipulate(
    ctx: _Steering,
    sweeps: dict[Method, SweepResult],
    prompts: Sequence[str],
    strengths: tuple[float, ...],
    neutral: Sequence[str] | None,
) -> _Manipulation:
    """Steers with the first readers of each method and tallies successes.

    A prompt is a success for a strength and a judge when its score under
    the positive coefficient is strictly above its score under the negative
    one. The side effects of every steered trial are measured on the arm
    that produced it, with `neutral` as the held-out text of the perplexity;
    with `neutral` None nothing is measured and no effect row is made.
    """
    config = ctx.config
    methods = config.methods
    signed = [c for c in config.coeffs if c != 0.0]
    success = np.full(
        (len(_JUDGES), len(methods), config.n_steer_seeds, len(strengths)),
        np.nan,
    )
    counts = np.zeros((len(_JUDGES), len(methods), len(strengths), 3), np.int64)
    trials: list[Trial] = []
    seed_rows: list[list[object]] = []
    effect_rows: list[list[object]] = []
    for m, method in enumerate(methods):
        for i in range(config.n_steer_seeds):
            arms: dict[float, Arm] = {}
            dose = dose_response(
                recording(_arm_factory(ctx, sweeps[method].readers[i]), arms),
                signed,
                prompts,
                ctx.score,
                columns=len(_JUDGES),
            )
            by_coeff = dict(zip(signed, dose.trials, strict=True))
            if neutral is not None:
                effect_rows.extend(
                    effects_row(
                        side_effects(
                            ctx.loaded,
                            arms[coeff],
                            trial,
                            neutral,
                            batch_size=config.batch_size,
                        ),
                        method,
                        i,
                        coeff,
                    )
                    for coeff, trial in by_coeff.items()
                )
            trials.extend(
                dataclasses.replace(trial, name=f"{method}/{i}/{coeff:g}")
                for coeff, trial in by_coeff.items()
            )
            for k, strength in enumerate(strengths):
                up = by_coeff[strength].per_prompt()
                down = by_coeff[-strength].per_prompt()
                for j, judge in enumerate(_JUDGES):
                    result = tally(up[:, j], down[:, j])
                    counts[j, m, k] += (result.wins, result.losses, result.ties)
                    success[j, m, i, k] = result.rate
                    lo, hi = _rate_interval(result)
                    seed_rows.append(
                        [
                            method,
                            i,
                            strength,
                            judge,
                            result.wins,
                            result.losses,
                            result.ties,
                            result.rate,
                            lo,
                            hi,
                        ]
                    )
    rows: list[list[object]] = []
    pooled = np.full((len(_JUDGES), len(methods), len(strengths)), np.nan)
    for m, method in enumerate(methods):
        for j, task in enumerate(_MANIPULATION_TASKS):
            for k, strength in enumerate(strengths):
                result = Tally(*(int(c) for c in counts[j, m, k]))
                lo, hi = _rate_interval(result)
                pooled[j, m, k] = result.rate
                rows.append(
                    [
                        method,
                        task,
                        f"{strength:g}",
                        result.rate,
                        lo,
                        hi,
                        result.decisive,
                    ]
                )
    return _Manipulation(
        trials,
        rows,
        seed_rows,
        effect_rows,
        success,
        _agreement(pooled[0], pooled[1]),
    )


def _erasures(
    reading: _Reading,
    reader: RepReader,
    seed_index: int,
    layers: Sequence[int],
    seed: int,
) -> list[tuple[float, float]]:
    """Probe accuracy before and after the erasure at each layer."""
    train_set, train_act = reading.train[seed_index]
    test_set, test_act = reading.test
    train_labels = train_act.row_values(train_set.labels)
    test_groups = test_act.row_groups(test_set.groups)
    test_labels = test_act.row_values(test_set.labels)
    return [
        erased_probe_accuracy(
            train_act.data[train_act.layers.index(layer)],
            train_labels,
            test_act.data[test_act.layers.index(layer)],
            test_groups,
            test_labels,
            reader.directions[reader.layers.index(layer)],
            seed=derive_seed(seed, "probe", seed_index, layer),
        )
        for layer in layers
    ]


def _terminate(
    ctx: _Steering,
    reading: _Reading,
    pleasant: Sequence[str],
    unpleasant: Sequence[str],
) -> _Termination:
    """Measures the behavioral gap and the probe after each projection."""
    config = ctx.config
    prompts = [*pleasant, *unpleasant]
    n_pairs = len(pleasant)

    def gaps(trial: Trial) -> NDArray[np.float64]:
        per_prompt = trial.per_prompt()
        return per_prompt[:n_pairs] - per_prompt[n_pairs:]

    def judged(arm: Arm) -> Trial:
        return run_trial(arm, prompts, ctx.score, columns=len(_JUDGES))

    none = judged(plain_arm(ctx.loaded, ctx.settings, name="none"))
    trials = [none]
    rows: list[list[object]] = []
    none_gaps = gaps(none)
    for j, task in enumerate(_GAP_TASKS):
        finite = none_gaps[np.isfinite(none_gaps[:, j]), j]
        mean, lo, hi = _spread(finite.tolist())
        if finite.size:
            low, high = bootstrap_ci(finite, seed=config.seed)
            lo, hi = float(low), float(high)
        rows.append(["none", task, "none", mean, lo, hi, n_pairs])
    for method in config.methods:
        seed_gaps: list[NDArray[np.float64]] = []
        before: list[float] = []
        after: list[float] = []
        for i in range(config.n_steer_seeds):
            reader = reading.sweeps[method].readers[i]
            trial = judged(
                reader_arm(
                    ctx.loaded,
                    ctx.settings,
                    reader,
                    layers=ctx.control,
                    coeff=1.0,
                    op="projection",
                    mode="absolute",
                    mask="all",
                    name=f"{method}/{i}/projection",
                )
            )
            trials.append(trial)
            seed_gaps.append(
                np.array([_spread(g.tolist())[0] for g in gaps(trial).T])
            )
            for was, now in _erasures(
                reading, reader, i, ctx.control, config.seed
            ):
                before.append(was)
                after.append(now)
        for j, task in enumerate(_GAP_TASKS):
            mean, lo, hi = _spread([float(g[j]) for g in seed_gaps])
            rows.append([method, task, "projection", mean, lo, hi, n_pairs])
        for setting, values in (("before", before), ("after", after)):
            mean, lo, hi = _spread(values)
            rows.append(
                [
                    method,
                    "termination_probe",
                    setting,
                    mean,
                    lo,
                    hi,
                    len(values),
                ]
            )
    return _Termination(trials, rows)


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    utility: Sequence[Example],
    texts: Sequence[str],
    score: Scorer,
) -> None:
    """Reads utility with each finder, then steers and erases along it.

    Correlation: `config.n_seeds` training sets of `config.n_pairs` pairs
    fit every method on the stimuli of R2; the layer of each method is the
    best of the mean validation accuracy, reported on the test pairs.

    Manipulation: the first `config.n_steer_seeds` readers of each method
    steer the continuation of `config.n_prompts` test scenarios cut to
    `config.truncate_words` words (pairs often open alike, so the cuts are
    deduplicated before the draw), at the same control layers for every
    method (so depth is not a confound) and with coefficients relative to
    the mean residual norm of held-out text. A prompt is a success for a
    strength when its sentiment under `+s` is strictly above that under `-s`;
    both signs share the sampling seed, so they differ only in the
    steering. The prompts are shared by all seeds and methods, so a pooled
    Wilson interval is narrower than the truth and the per-seed rows are
    the honest spread.

    Termination has two operationalizations, since the protocol is not in the
    paper's text: (a) probe erasure, a fresh logistic probe fit on the
    stored activations of each control layer after projecting the method's
    direction out of them (no new forward; a zero-norm fit counts as chance),
    compared with the same probe before; (b) the behavioral gap, the score
    after the more pleasant scenario of a test pair minus after the less
    pleasant one, unsteered and under the joint projection at all control
    layers. The probe asks whether the concept is still linearly present
    at that layer, the gap whether removing the direction changes behavior,
    and a method can pass one and fail the other.

    Every generation is scored by both judges at once (`score` returns two
    columns, the primary judge first), so each headline row has a `_check`
    twin on the same completions.

    Writes `curves`, `curves_val`, `curves_success`, `curves_success_check`,
    `trials`, `trials_termination`, `table`, `table_seeds`, `table_effects`
    (only when `config.fluency`) and `summary`. `table_effects` holds the
    side effects of the unsteered arm and of every method, seed and
    coefficient trial: neutral perplexity of the steered model, completion
    fluency under the unsteered model, distinct-1 and distinct-2, mean length
    and empty rate.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model.
        utility: ETHICS utility examples with `train`, `val` and `test`
            splits and the row id in `meta["pair"]`.
        texts: Held-out text for the relative coefficients.
        score: Maps texts to scores of shape `(n, 2)`: the primary and the
            check judge.

    Raises:
        ValueError: If `config.coeffs` is not a symmetric grid with 0.0 of
            finite values whose nonzero labels are distinct, `config.methods`
            or the control layers repeat, `config.n_steer_seeds` is outside
            `1..config.n_seeds`, a control layer is outside `1..L`, fewer
            distinct truncated test scenarios than `config.n_prompts`
            remain, or no test pair has both scenarios (all checked before
            any forward), or an input is invalid.
    """
    checked = _check(config, loaded, utility)
    strengths, control = checked.strengths, checked.control
    pleasant, unpleasant = checked.pleasant, checked.unpleasant
    held = split_held_out(
        texts,
        n_calibration=config.n_calibration,
        n_neutral=config.n_neutral,
        seed=config.seed,
    )
    norms = mean_residual_norms(
        loaded,
        held.calibration,
        control,
        bos=config.bos,
        batch_size=config.batch_size,
    )
    reading = _read(config, run_dir, loaded, utility)
    ctx = _Steering(
        config, loaded, GenSettings.from_config(config), score, control, norms
    )
    prompts = _prompts(config, checked.scenarios)
    plain = plain_arm(loaded, ctx.settings, name="none")
    none = run_trial(plain, prompts, score, columns=len(_JUDGES))
    neutral = held.neutral if config.fluency else None
    none_effects = (
        None
        if neutral is None
        else side_effects(
            loaded, plain, none, neutral, batch_size=config.batch_size
        )
    )
    manipulation = _manipulate(ctx, reading.sweeps, prompts, strengths, neutral)
    termination = _terminate(ctx, reading, pleasant, unpleasant)

    for j, (stem, ylabel) in enumerate(_SUCCESS_CURVES):
        write_dose_curves(
            run_dir,
            manipulation.success[j],
            strengths,
            series=config.methods,
            stem=stem,
            xlabel="Steering strength (relative)",
            ylabel=ylabel,
            chance=0.5,
            seed=config.seed,
        )
    steered = [none, *manipulation.trials]
    write_trials(run_dir, "trials", steered)
    write_trials(run_dir, "trials_termination", termination.trials)
    run_dir.write_json(
        "table",
        {
            "columns": list(_TABLE_COLUMNS),
            "rows": [*reading.rows, *manipulation.rows, *termination.rows],
        },
    )
    run_dir.write_json(
        "table_seeds",
        {"columns": list(_SEED_COLUMNS), "rows": manipulation.seed_rows},
    )
    if none_effects is not None:
        run_dir.write_json(
            "table_effects",
            {
                "columns": list(_EFFECT_COLUMNS),
                "rows": [
                    effects_row(none_effects, "none", None, 0.0),
                    *manipulation.effect_rows,
                ],
            },
        )
    run_dir.write_json(
        "summary",
        {
            "control_layers": list(control),
            "strengths": list(strengths),
            "methods": list(config.methods),
            "n_prompts": len(prompts),
            "n_gap_pairs": len(pleasant),
            "none_score": [float(v) for v in none.mean_score()],
            "judge_agreement": manipulation.agreement,
            "empty_rate": float(
                np.mean([t.empty_rate for t in [*steered, *termination.trials]])
            ),
        },
    )


def run(config: Config, run_dir: RunDir) -> None:
    """Loads the model, ETHICS, WikiText and both judges, then runs `execute`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
    """
    primary = load_judge(config.judge)
    check = load_judge(config.judge_check)

    def score(texts: Sequence[str]) -> NDArray[np.float32]:
        return np.column_stack([primary.score(texts), check.score(texts)])

    execute(
        config,
        run_dir,
        load_model(config.model),
        load_ethics_utility(),
        [example.text for example in load_wikitext()],
        score,
    )


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws the correlation and success curves of a finished M1 run.

    Args:
        run_dir: A finished M1 run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths of the four figures.
    """
    return [
        path
        for stem in _CURVE_STEMS
        for path in plot_curves_file(run_dir, out_dir, stem=stem)
    ]
