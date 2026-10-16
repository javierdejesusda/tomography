"""P7: do directions read under different templates steer as well as they read?

Sentiment readers are extracted under the template variants of P1 and steer
text with C1's protocol, so the template changes nothing but the direction.
The study counts the units (a template, a layer and a reader) that read
sentiment well and steer it the wrong way, compares the layer that reads best
with the layer that steers best, and asks whether the logit lens predicts the
dose response.
"""

from collections.abc import Sequence
import dataclasses
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure
import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from tomography.analysis import lens_logits
from tomography.analysis import pole_separation
from tomography.control.controllers import mean_residual_norms
from tomography.datasets.records import Example
from tomography.datasets.sst2 import load_sst2
from tomography.datasets.wikitext import load_wikitext
from tomography.experiments.a2_logit_lens import POLE_WORDS
from tomography.experiments.a2_logit_lens import PoleWords
from tomography.experiments.a2_logit_lens import lens_params
from tomography.experiments.a2_logit_lens import pole_ids
from tomography.experiments.aggregate import read_curves
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.c1_sentiment_steering import check_pair_count
from tomography.experiments.c1_sentiment_steering import default_sweep_layers
from tomography.experiments.c1_sentiment_steering import dose_with_none
from tomography.experiments.c1_sentiment_steering import draw_prompts
from tomography.experiments.c1_sentiment_steering import prompt_fragments
from tomography.experiments.c1_sentiment_steering import slope_layer
from tomography.experiments.effects import split_held_out
from tomography.experiments.effects import write_trials
from tomography.experiments.matrices import plot_matrix_file
from tomography.experiments.matrices import write_matrix
from tomography.experiments.p3_size import spearman as rank_correlation
from tomography.experiments.r1_sentiment import select_examples
from tomography.experiments.steering import Arm
from tomography.experiments.steering import DoseResponse
from tomography.experiments.steering import GenSettings
from tomography.experiments.steering import Scorer
from tomography.experiments.steering import SteeringConfig
from tomography.experiments.steering import Trial
from tomography.experiments.steering import dose_slope
from tomography.experiments.steering import plain_arm
from tomography.experiments.steering import reader_arm
from tomography.experiments.steering import run_trial
from tomography.experiments.sweep import SERIES
from tomography.experiments.sweep import chat_format_for
from tomography.experiments.sweep import plot_curves_file
from tomography.experiments.sweep import reading_sweep
from tomography.experiments.sweep import write_curves
from tomography.experiments.variants import READING_TASKS
from tomography.experiments.variants import VARIANT_KINDS
from tomography.experiments.variants import Variant
from tomography.experiments.variants import VariantKind
from tomography.experiments.variants import pair_capacity
from tomography.experiments.variants import task_set
from tomography.experiments.variants import template_variants
from tomography.judges import load_judge
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.reading.directions import Method
from tomography.reading.metrics import best_layer
from tomography.reading.metrics import wilson_ci
from tomography.reading.reader import RepReader
from tomography.templates import ChatFormat
from tomography.viz.style import INK_MUTED
from tomography.viz.style import PX
from tomography.viz.style import TEXT_WIDTH
from tomography.viz.style import categorical_colors
from tomography.viz.style import new_figure
from tomography.viz.style import save_figure
from tomography.viz.style import style_context

TEMPLATES: tuple[str, ...] = (
    "paper",
    "no_template",
    "no_chat_tags",
    "neutral_frame",
    "scaffold_0",
    "concept_swap_0",
)
FORMATS: tuple[str, ...] = ("raw", "chat", "story")
CHAT_INSTRUCTION = "Continue the text."
STORY_PREFIX = "Here is a short story: "
MAX_SERIES = 8
_READER = SERIES.index("reader")
_COLUMNS = (
    "template",
    "read_layer",
    "read_depth",
    "read_accuracy",
    "steer_layer",
    "steer_depth",
    "steer_slope",
    "lens_layer",
    "lens_rho",
    "n_readable",
    "n_inverted",
)


@dataclass(frozen=True, kw_only=True)
class Config(SteeringConfig):
    """P7 settings: sentiment readers extracted under different templates.

    Every template is a variant of the sentiment task of P1, read with the
    same seeds and items, and steers text with C1's protocol: relative
    coefficients calibrated on held-out WikiText, C1's prompt fragments and
    both sentiment judges at once. The inherited `layers` is not used: P7
    steers the single layers of `sweep_layers`.

    Attributes:
        method: The direction finder, as in R1 and C1.
        judge: Key of the primary judge, score column 0.
        judge_check: Key of the check judge, score column 1.
        templates: Variant names of the sentiment task, at most
            `MAX_SERIES`, `paper` required. An indexed kind is named with
            its index, such as `scaffold_0`.
        formats: Prompt formats of the transfer, names from `FORMATS`.
        n_seeds: Readers fitted per template, for the reading curves.
        n_steer_seeds: The first readers, in fit order, that steer.
        n_pairs: Training pairs per reader.
        n_eval_pairs: Validation pairs; None means every pair.
        min_words: SST-2 phrases with fewer words are dropped.
        sweep_layers: Single layers of the sweep, at most `MAX_SERIES`;
            None means `default_sweep_layers` of the model depth.
        truncate_words: Words kept of each validation sentence as a prompt.
        read_threshold: A reader reads well at or above this validation
            accuracy, in [0.5, 1).
        n_boot: Number of bootstrap resamples of every curve band.
    """

    experiment: str = "p7"
    model: str = "gpt2"
    method: Method = "logistic"
    judge: str = "sentiment-primary"
    judge_check: str = "sentiment-secondary"
    templates: tuple[str, ...] = TEMPLATES
    formats: tuple[str, ...] = FORMATS
    n_seeds: int = 10
    n_steer_seeds: int = 2
    n_pairs: int = 64
    n_eval_pairs: int | None = 256
    min_words: int = 3
    sweep_layers: tuple[int, ...] | None = None
    coeffs: tuple[float, ...] = (-0.3, -0.15, 0.0, 0.15, 0.3)
    n_prompts: int = 50
    truncate_words: int = 3
    read_threshold: float = 0.6
    n_boot: int = 2000


def _kind_of(name: str) -> VariantKind:
    """Returns the kind of a variant name: the name itself or `<kind>_<i>`."""
    for kind in VARIANT_KINDS:
        if kind in (name, name.rpartition("_")[0]):
            return kind
    raise ValueError(
        f"{name!r} is not a variant kind or <kind>_<i> of one, expected a "
        f"kind of {VARIANT_KINDS}"
    )


def extraction_variants(
    chat: ChatFormat, names: Sequence[str]
) -> tuple[Variant, ...]:
    """Picks named template variants of the sentiment task.

    Args:
        chat: The chat format of the model.
        names: Variant names: a kind, or `<kind>_<i>` for an indexed kind,
            which `template_variants` expands to every index. `paper` is
            required.

    Returns:
        The named variants, in the order of `names`.

    Raises:
        ValueError: If a name is not a kind or `<kind>_<i>`, a name is not
            built for the sentiment task (the message carries the skip
            reason of its kind), or `names` lacks `paper`.
    """
    kinds = {name: _kind_of(name) for name in names}
    variants, skipped = template_variants(
        READING_TASKS["sentiment"],
        chat,
        kinds=tuple(k for k in VARIANT_KINDS if k in kinds.values()),
    )
    by_name = {v.name: v for v in variants}
    for name in names:
        if name not in by_name:
            reason = skipped.get(
                kinds[name], "variants.py has no variant of that name"
            )
            raise ValueError(
                f"{name!r} is not a variant of the sentiment task: {reason}"
            )
    return tuple(by_name[name] for name in names)


def shift_prompts(
    fragments: Sequence[str], fmt: str, chat: ChatFormat
) -> list[str]:
    """Renders the steering prompts in one of the transfer formats.

    Args:
        fragments: The prompt fragments, the first words of a text.
        fmt: A name from `FORMATS`: `raw` keeps the fragments, `chat` starts
            the answer to `CHAT_INSTRUCTION` with the fragment and `story`
            puts `STORY_PREFIX` before it.
        chat: The chat format of the `chat` prompts.

    Returns:
        One prompt per fragment.

    Raises:
        ValueError: If `fmt` is not in `FORMATS`.
    """
    if fmt == "raw":
        return list(fragments)
    if fmt == "chat":
        return [chat.render_turn(CHAT_INSTRUCTION, f)[0] for f in fragments]
    if fmt == "story":
        return [STORY_PREFIX + f for f in fragments]
    raise ValueError(f"unknown format {fmt!r}, expected one of {FORMATS}")


def spearman(x: ArrayLike, y: ArrayLike) -> float:
    """Computes the Spearman correlation of the finite pairs.

    Args:
        x: Values of one variable.
        y: Values of the other, one per entry of `x`.

    Returns:
        The Pearson correlation of the ranks, with the ranks of ties
        averaged. NaN for fewer than 3 finite pairs or a constant side.

    Raises:
        ValueError: If `x` and `y` differ in length.
    """
    first = np.asarray(x, dtype=np.float64)
    second = np.asarray(y, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError(
            "x and y must have the same length, got "
            f"{first.shape} and {second.shape}"
        )
    keep = np.isfinite(first) & np.isfinite(second)
    if keep.sum() < 3:
        return float("nan")
    return rank_correlation(first[keep], second[keep])


def safe_slope(coeffs: Sequence[float], values: NDArray[Any]) -> float:
    """Fits the dose slope, or NaN when too few finite points remain.

    Args:
        coeffs: The coefficients.
        values: One value per coefficient.

    Returns:
        `dose_slope` of the finite points; NaN when fewer than two distinct
        finite coefficients remain.

    Raises:
        ValueError: If `coeffs` and `values` differ in length.
    """
    x = np.asarray(coeffs, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if (
        x.shape == y.shape
        and np.unique(x[np.isfinite(x) & np.isfinite(y)]).size < 2
    ):
        return float("nan")
    return dose_slope(coeffs, y)


def _pole_tokens(
    loaded: LoadedModel, words: PoleWords
) -> tuple[list[int], list[int]]:
    """Returns the token ids of the pole words that are one token each."""
    positive, _ = pole_ids(loaded, words[0])
    negative, _ = pole_ids(loaded, words[1])
    if not positive or not negative:
        raise ValueError("no pole word is a single token for this tokenizer")
    return positive, negative


def lens_separation(
    loaded: LoadedModel, readers: Sequence[RepReader], words: PoleWords
) -> NDArray[np.float64]:
    """Measures how far the logit lens puts the poles apart for each reader.

    Args:
        loaded: The wrapped model; only its unembedding, final norm and
            tokenizer are used.
        readers: The readers, each with one direction per layer.
        words: The words of the positive and the negative pole, with their
            leading space. Those that are not one token are dropped.

    Returns:
        Array of shape (R, K): the pole separation (see `pole_separation`)
        of every reader's oriented direction (`directions * signs`) at every
        layer. An empty `readers` gives an array of shape (0, 0).

    Raises:
        ValueError: If no word of a pole is a single token.
    """
    positive, negative = _pole_tokens(loaded, words)
    unembedding, scale, center = lens_params(loaded)
    rows = [
        pole_separation(
            lens_logits(
                reader.directions * reader.signs[:, None],
                unembedding,
                scale=scale,
                center=center,
            ),
            positive,
            negative,
        )
        for reader in readers
    ]
    return np.stack(rows) if rows else np.empty((0, 0), dtype=np.float64)


def _finite_mean(values: NDArray[Any], axis: int) -> NDArray[np.float64]:
    """Averages the finite entries along `axis`; NaN where none is left."""
    finite = np.isfinite(values)
    count = finite.sum(axis=axis)
    total = np.where(finite, values, 0.0).sum(axis=axis)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _inversion(
    read: NDArray[np.float64], slope: NDArray[np.float64], threshold: float
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Marks the units that read well and the ones that steer the wrong way.

    Args:
        read: Reading accuracy of each unit.
        slope: Primary-judge dose slope of each unit, of the same shape.
        threshold: A unit is readable at or above this accuracy.

    Returns:
        The readable units, and the readable units with a negative slope. A
        unit with a NaN slope is never inverted.
    """
    readable = read >= threshold
    return readable, readable & (slope < 0.0)


def _inversion_summary(
    readable: NDArray[np.bool_], inverted: NDArray[np.bool_]
) -> dict[str, Any]:
    """Counts the units and gives the inversion rate with its interval."""
    n_readable = int(readable.sum())
    n_inverted = int(inverted.sum())
    summary: dict[str, Any] = {
        "readable": n_readable,
        "inverted": n_inverted,
        "rate": None,
        "ci": None,
    }
    if n_readable:
        summary["rate"] = n_inverted / n_readable
        summary["ci"] = list(wilson_ci(n_inverted, n_readable))
    return summary


def _sign_agreement(
    first: NDArray[np.float64], second: NDArray[np.float64]
) -> float:
    """Gives the share of units whose two slopes have the same sign.

    Returns:
        The share among the units with a finite slope under both judges; NaN
        when there is none.
    """
    both = np.isfinite(first) & np.isfinite(second)
    if not both.any():
        return float("nan")
    return float(np.mean(np.sign(first[both]) == np.sign(second[both])))


def _lens_summary(
    lens: NDArray[np.float64],
    slope: NDArray[np.float64],
    sweep: Sequence[int],
) -> tuple[int, float]:
    """Compares the lens separation with the dose slope over the sweep layers.

    Args:
        lens: Separation of shape (S, R): sweep layer, steering reader.
        slope: Primary-judge dose slope of the same shape.
        sweep: The S sweep layers.

    Returns:
        The sweep layer of the largest mean separation (the first on ties),
        and the Spearman correlation over the layers of the reader means of
        the separation and of the slope.
    """
    mean_lens = lens.mean(axis=1)
    layer = int(sweep[int(np.argmax(mean_lens))])
    return layer, spearman(mean_lens, _finite_mean(slope, axis=1))


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

    def dose(
        self,
        reader: RepReader,
        layer: int,
        prompts: Sequence[str],
        none: Trial,
    ) -> DoseResponse:
        config = self.config
        return dose_with_none(
            lambda coeff: reader_arm(
                self.loaded,
                self.settings,
                reader,
                layers=[layer],
                coeff=coeff,
                norms=self.norms,
                op=config.op,
                mode=config.mode,
                mask=config.mask,
                preserve_norm=config.preserve_norm,
            ),
            config.coeffs,
            prompts,
            self.score,
            none,
            columns=2,
        )


def _steered_trials(dose: DoseResponse, prefix: str) -> list[Trial]:
    """Names the nonzero-coefficient trials `<prefix>/<coefficient>`."""
    return [
        dataclasses.replace(trial, name=f"{prefix}/{coeff:g}")
        for coeff, trial in zip(dose.coeffs, dose.trials, strict=True)
        if coeff != 0.0
    ]


def _slopes(
    curves: NDArray[np.float64], coeffs: Sequence[float]
) -> NDArray[np.float64]:
    """Fits the dose slope along the last axis of some dose curves."""
    flat = curves.reshape(-1, curves.shape[-1])
    slopes = np.array([safe_slope(coeffs, row) for row in flat])
    return slopes.reshape(curves.shape[:-1])


def _check(config: Config, n_layers: int) -> tuple[int, ...]:
    """Checks the settings that need no data and returns the sweep layers."""
    if not 1 <= len(config.templates) <= MAX_SERIES:
        raise ValueError(
            f"templates needs 1 to {MAX_SERIES} names, got "
            f"{len(config.templates)}"
        )
    if len(set(config.templates)) != len(config.templates):
        raise ValueError(f"templates must be distinct, got {config.templates}")
    if not config.formats or len(set(config.formats)) != len(config.formats):
        raise ValueError(
            f"formats must be non-empty and distinct, got {config.formats}"
        )
    unknown = [f for f in config.formats if f not in FORMATS]
    if unknown:
        raise ValueError(
            f"unknown formats {unknown}, expected a subset of {FORMATS}"
        )
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
    if not 1 <= len(sweep) <= MAX_SERIES:
        raise ValueError(
            f"sweep_layers needs 1 to {MAX_SERIES} layers, got {len(sweep)}"
        )
    if len(set(sweep)) != len(sweep) or not all(
        1 <= layer <= n_layers for layer in sweep
    ):
        raise ValueError(
            f"sweep_layers must be distinct layers in 1..{n_layers}, got "
            f"{sweep}"
        )
    if not 0.5 <= config.read_threshold < 1.0:
        raise ValueError(
            f"read_threshold must be in [0.5, 1), got {config.read_threshold}"
        )
    return sweep


def _split_sst2(
    config: Config, sst2: Sequence[Example]
) -> tuple[list[Example], list[Example]]:
    """Selects the SST-2 train and val examples and checks their pairs."""
    task = READING_TASKS["sentiment"]
    train = select_examples(sst2, split="train", min_words=config.min_words)
    val = select_examples(sst2, split="val", min_words=config.min_words)
    for split, name, wanted, part, allow_none in (
        ("train", "n_pairs", config.n_pairs, train, False),
        ("val", "n_eval_pairs", config.n_eval_pairs, val, True),
    ):
        check_pair_count(
            name,
            wanted,
            pair_capacity(task, part),
            f"in the {split} split",
            allow_none=allow_none,
        )
    return train, val


def _read_templates(
    config: Config,
    loaded: LoadedModel,
    variants: Sequence[Variant],
    train: Sequence[Example],
    val: Sequence[Example],
) -> tuple[NDArray[np.float64], list[tuple[RepReader, ...]], tuple[int, ...]]:
    """Fits the readers of every template and scores them on the val pairs.

    Returns:
        The validation accuracy of shape (T, N, K): template, reader, layer;
        the first `n_steer_seeds` readers of each template; and the layers.
    """
    task = READING_TASKS["sentiment"]
    accuracy = []
    readers = []
    for variant in variants:
        result = reading_sweep(
            loaded,
            [
                task_set(
                    task,
                    variant,
                    train,
                    n_pairs=config.n_pairs,
                    seed=derive_seed(config.seed, "train", i),
                )
                for i in range(config.n_seeds)
            ],
            {
                "val": task_set(
                    task,
                    variant,
                    val,
                    n_pairs=config.n_eval_pairs,
                    seed=derive_seed(config.seed, "eval"),
                )
            },
            method=config.method,
            seed=config.seed,
            bos=config.bos,
            batch_size=config.batch_size,
        )
        accuracy.append(result.per_seed("val")[_READER])
        readers.append(result.readers[: config.n_steer_seeds])
        layers = result.layers
    return np.stack(accuracy), readers, layers


def _dose_sweep(
    steering: _Steering,
    run_dir: RunDir,
    names: Sequence[str],
    readers: Sequence[Sequence[RepReader]],
    sweep: Sequence[int],
    prompts: Sequence[str],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Runs the dose response of every reader at every sweep layer.

    Writes `trials_sweep`.

    Returns:
        The primary and the check scores, each of shape (T, S, R, K):
        template, sweep layer, reader, coefficient.
    """
    config = steering.config
    none = steering.none(prompts)
    trials = [none]
    shape = (len(names), len(sweep), config.n_steer_seeds, len(config.coeffs))
    primary = np.full(shape, np.nan)
    check = np.full(shape, np.nan)
    for t, name in enumerate(names):
        for s, layer in enumerate(sweep):
            for r, reader in enumerate(readers[t]):
                dose = steering.dose(reader, layer, prompts, none)
                primary[t, s, r] = dose.curve(0)
                check[t, s, r] = dose.curve(1)
                trials.extend(_steered_trials(dose, f"{name}/layer{layer}/{r}"))
    write_trials(run_dir, "trials_sweep", trials)
    return primary, check


def _prompt_shift(
    steering: _Steering,
    run_dir: RunDir,
    names: Sequence[str],
    readers: Sequence[Sequence[RepReader]],
    steer_layers: Sequence[int],
    prompts: Sequence[str],
    chat: ChatFormat,
) -> NDArray[np.float64]:
    """Steers the report prompts in each format at the steering layers.

    Writes `trials_transfer_<format>`.

    Returns:
        The primary slopes of shape (T, F, R): template, format, reader.
    """
    config = steering.config
    slopes = np.full(
        (len(names), len(config.formats), config.n_steer_seeds), np.nan
    )
    for f, fmt in enumerate(config.formats):
        shifted = shift_prompts(prompts, fmt, chat)
        none = steering.none(shifted)
        trials = [none]
        for t, name in enumerate(names):
            for r, reader in enumerate(readers[t]):
                dose = steering.dose(reader, steer_layers[t], shifted, none)
                slopes[t, f, r] = safe_slope(config.coeffs, dose.curve(0))
                trials.extend(_steered_trials(dose, f"{name}/{r}"))
        write_trials(run_dir, f"trials_transfer_{fmt}", trials)
    return slopes


@dataclass(frozen=True, eq=False)
class _Measured:
    """What a run measured, with the template on the first axis.

    Attributes:
        names: The templates.
        layers: The layers of the reading and lens curves.
        sweep: The sweep layers.
        reads: Validation accuracy of shape (T, N, K).
        lens: Lens separation of shape (T, R, K).
        slope: Primary-judge dose slope of shape (T, S, R): sweep layer and
            steering reader.
        check: Check-judge dose slope of the same shape.
        steer_layers: The steering layer of each template.
    """

    names: tuple[str, ...]
    layers: tuple[int, ...]
    sweep: tuple[int, ...]
    reads: NDArray[np.float64]
    lens: NDArray[np.float64]
    slope: NDArray[np.float64]
    check: NDArray[np.float64]
    steer_layers: list[int]


def _write_report(
    config: Config, run_dir: RunDir, n_layers: int, measured: _Measured
) -> None:
    """Writes `table.json` and `summary.json` from the measurements."""
    names, layers, sweep = measured.names, measured.layers, measured.sweep
    sweep_pos = [layers.index(layer) for layer in sweep]
    read_units = measured.reads[:, : config.n_steer_seeds, sweep_pos]
    read_units = read_units.transpose(0, 2, 1)
    lens_units = measured.lens[:, :, sweep_pos].transpose(0, 2, 1)
    slope = measured.slope
    readable, inverted = _inversion(read_units, slope, config.read_threshold)
    rows: list[list[Any]] = []
    depth_gaps: list[float] = []
    lens_hits: list[bool] = []
    for t, name in enumerate(names):
        read_curve = measured.reads[t].mean(axis=0)
        read_layer = best_layer(read_curve, layers)
        steer_layer = measured.steer_layers[t]
        steer_pos = sweep.index(steer_layer)
        lens_layer, lens_rho = _lens_summary(lens_units[t], slope[t], sweep)
        read_depth = read_layer / n_layers
        steer_depth = steer_layer / n_layers
        depth_gaps.append(abs(read_depth - steer_depth))
        lens_hits.append(lens_layer == steer_layer)
        rows.append(
            [
                name,
                read_layer,
                read_depth,
                float(read_curve[layers.index(read_layer)]),
                steer_layer,
                steer_depth,
                float(_finite_mean(slope[t, steer_pos], axis=0)),
                lens_layer,
                lens_rho,
                int(readable[t].sum()),
                int(inverted[t].sum()),
            ]
        )
    run_dir.write_json("table", {"columns": list(_COLUMNS), "rows": rows})
    run_dir.write_json(
        "summary",
        {
            "templates": list(names),
            "formats": list(config.formats),
            "sweep_layers": list(sweep),
            "n_prompts": config.n_prompts,
            "inversion": _inversion_summary(readable, inverted),
            "read_steer_rho": spearman(read_units.ravel(), slope.ravel()),
            "lens_steer_rho": spearman(lens_units.ravel(), slope.ravel()),
            "lens_agreement": float(np.mean(lens_hits)),
            "layer_gap": float(np.mean(depth_gaps)),
            "judge_sign_agreement": _sign_agreement(
                slope.ravel(), measured.check.ravel()
            ),
        },
    )


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    sst2: Sequence[Example],
    texts: Sequence[str],
    score: Scorer,
    words: PoleWords = POLE_WORDS["r1"],
) -> None:
    """Reads sentiment under every template.

    Every template of `config.templates` is a variant of the sentiment task.
    It fits `n_seeds` readers on SST-2 train pairs, with the seeds of C1, and
    scores them on the same SST-2 val pairs as every other template.

    The first `n_steer_seeds` readers of each template steer, as in C1: a
    dose response at every single layer of `sweep_layers` on a first list of
    prompts, with coefficient 0 generated once. The steering layer of a
    template is the sweep layer where the primary judge's score rises the
    most per unit of coefficient. The prompt-shift transfer then steers a
    second list of prompts, rendered in each of `formats`, at that layer.

    Writes `curves_read` (the validation accuracy of the readers of each
    template at every layer), `curves_lens` (the logit lens separation of
    the first `n_steer_seeds` readers at every layer), `curves_slope` and
    `curves_slope_check` (the dose slope of each steering reader at every
    sweep layer under the primary and the check judge), `matrix_transfer`
    (the mean primary slope of each template and format), `trials_sweep` and
    `trials_transfer_<format>`, `table.json` (one row per template) and
    `summary.json`. A unit is a template, a sweep layer and a steering
    reader. It is readable when its reader's validation accuracy at that
    layer is at least `read_threshold`, and inverted when it is readable and
    its primary slope is negative.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model.
        sst2: SST-2 examples with `train` and `val` splits.
        texts: Held-out texts that set the relative coefficients.
        score: Maps texts to scores of shape (n, 2): the primary judge, then
            the check judge.
        words: The words of the positive and the negative pole of the lens.

    Raises:
        ValueError: If `templates` are not 1 to `MAX_SERIES` distinct names
            that `extraction_variants` accepts, `formats` are empty, repeated
            or unknown, `n_steer_seeds` is outside `1..n_seeds`, `coeffs` has
            no 0.0 or no other value, `sweep_layers` are empty, repeated,
            more than `MAX_SERIES` or outside `1..L`, `read_threshold` is
            outside `[0.5, 1)`, the generation settings are invalid (see
            `GenSettings`), no word of a pole is a single token,
            `n_calibration` or `n_neutral` is below 1, the usable (at least
            40 characters) held-out texts are fewer than `n_calibration +
            n_neutral`, `n_pairs` or `n_eval_pairs` is below 1 or exceeds the
            pairs its split can form (`n_eval_pairs` of None needs one), or
            the validation sentences give fewer than `2 * n_prompts`
            distinct prompts.
    """
    n_layers = loaded.n_layers
    sweep = _check(config, n_layers)
    settings = GenSettings.from_config(config)
    _pole_tokens(loaded, words)
    chat = chat_format_for(loaded)
    variants = extraction_variants(chat, config.templates)
    held = split_held_out(
        texts,
        n_calibration=config.n_calibration,
        n_neutral=config.n_neutral,
        seed=config.seed,
    )
    train, val = _split_sst2(config, sst2)
    select_prompts, report_prompts = draw_prompts(
        prompt_fragments([e.text for e in val], config.truncate_words),
        config.n_prompts,
        sets=2,
        seed=derive_seed(config.seed, "prompts"),
    )
    names = tuple(v.name for v in variants)
    reads, readers, layers = _read_templates(
        config, loaded, variants, train, val
    )
    lens = lens_separation(
        loaded, [reader for row in readers for reader in row], words
    ).reshape(len(names), config.n_steer_seeds, -1)
    curves = partial(
        write_curves,
        run_dir,
        n_layers=n_layers,
        n_boot=config.n_boot,
        seed=config.seed,
    )
    curves(reads, layers, series=names, stem="curves_read", chance=0.5)
    curves(
        lens,
        layers,
        series=names,
        stem="curves_lens",
        ylabel="Lens separation (z)",
        chance=0.0,
    )
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
    primary, check = _dose_sweep(
        steering, run_dir, names, readers, sweep, select_prompts
    )
    slope = _slopes(primary, config.coeffs)
    check_slope = _slopes(check, config.coeffs)
    for stem, units, ylabel in (
        ("curves_slope", slope, "Dose slope (primary judge)"),
        ("curves_slope_check", check_slope, "Dose slope (check judge)"),
    ):
        curves(
            units.transpose(0, 2, 1),
            sweep,
            series=names,
            stem=stem,
            ylabel=ylabel,
            chance=0.0,
        )
    steer_layers = [
        slope_layer(primary[t], config.coeffs, sweep) for t in range(len(names))
    ]
    transfer = _prompt_shift(
        steering, run_dir, names, readers, steer_layers, report_prompts, chat
    )
    write_matrix(
        run_dir,
        "matrix_transfer",
        _finite_mean(transfer, axis=2),
        rows=names,
        cols=config.formats,
        title="Mean primary dose slope by steering prompt format",
    )
    _write_report(
        config,
        run_dir,
        n_layers,
        _Measured(
            names, layers, sweep, reads, lens, slope, check_slope, steer_layers
        ),
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


def _read_vs_steer(run_dir: RunDir) -> Figure:
    reads = read_curves(run_dir, "curves_read")
    slopes = read_curves(run_dir, "curves_slope")
    at = [int(np.flatnonzero(reads.x == layer)[0]) for layer in slopes.x]
    n_readers = slopes.per_seed.shape[1]
    with style_context():
        fig, axes = new_figure(width=TEXT_WIDTH, aspect=0.55)
        ax = axes[0, 0]
        for t, (name, color) in enumerate(
            zip(
                slopes.series,
                categorical_colors(len(slopes.series)),
                strict=True,
            )
        ):
            ax.plot(
                reads.per_seed[t, :n_readers][:, at].ravel(),
                slopes.per_seed[t].ravel(),
                "o",
                color=color,
                label=name,
            )
        ax.axhline(0.0, color=INK_MUTED, lw=PX, label="_zero_slope")
        ax.axvline(
            float(run_dir.config["read_threshold"]),
            color=INK_MUTED,
            lw=PX,
            label="_threshold",
        )
        ax.set_xlabel("Reading accuracy")
        ax.set_ylabel("Dose slope (primary judge)")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    return fig


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws the curves, the transfer matrix and the read versus steer scatter.

    Args:
        run_dir: A finished P7 run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths of `curves_read`, `curves_lens`,
        `curves_slope` and `curves_slope_check`, then of `matrix_transfer`
        on a diverging scale, then of `read_vs_steer`: each unit's reading
        accuracy against its primary dose slope, one color per template, with
        the zero slope and the readable threshold marked.
    """
    paths = [
        path
        for stem in (
            "curves_read",
            "curves_lens",
            "curves_slope",
            "curves_slope_check",
        )
        for path in plot_curves_file(run_dir, out_dir, stem=stem)
    ]
    paths += plot_matrix_file(
        run_dir, out_dir, stem="matrix_transfer", kind="diverging"
    )
    return paths + save_figure(
        _read_vs_steer(run_dir), out_dir, "read_vs_steer"
    )
