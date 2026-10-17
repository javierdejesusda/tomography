"""P9d: does the known versus unknown direction control what a model does?

The direction that separates known from unknown questions (P9b) is added to
the residual stream of the model while it answers, at a norm calibrated on
held-out questions, and the answer is reported as a finding whichever way it
falls. A post-trained checkpoint can abstain: the behavior is the rate of
abstention on unknown questions, of false refusals on known ones, the
coverage and the wrong answers among the answered. A base model does not
abstain, so the behavior is the shift of the answer probability and of the
entropy of the next token. Random directions of the same norm separate "this
direction" from "any edit of this size".

The study needs a known side, so it runs only for a model the P9a gate
admits, and only on contrasts of known against unknown questions.
"""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
import itertools
import math
from pathlib import Path
from typing import Any
from typing import Literal
from typing import Self
from typing import get_args

import numpy as np
from numpy.typing import NDArray

from tomography.control.controllers import CoeffMode
from tomography.control.controllers import ReadingVectorController
from tomography.control.controllers import ResponseMask
from tomography.control.controllers import mean_residual_norms
from tomography.control.operators import Operator
from tomography.experiments.base import RunDir
from tomography.experiments.controls import random_readers
from tomography.experiments.intervals import write_interval_curves
from tomography.experiments.p9a_labels import load_labeled
from tomography.experiments.steering import GenSettings
from tomography.experiments.steering import SteeringConfig
from tomography.experiments.steering import reader_arm
from tomography.experiments.sweep import chat_format_for
from tomography.experiments.sweep import plot_curves_file
from tomography.knowledge import DEFAULT_SHOTS
from tomography.knowledge import few_shot_prompt
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.p9.behavior import BehaviorRates
from tomography.p9.behavior import CompletionRecord
from tomography.p9.behavior import Outcome
from tomography.p9.behavior import abs_slope_gap_ci
from tomography.p9.behavior import behavior_rates
from tomography.p9.behavior import classify
from tomography.p9.behavior import item_slopes
from tomography.p9.behavior import select_coefficient
from tomography.p9.behavior import selectivity_gap
from tomography.p9.behavior import shift_ci
from tomography.p9.behavior import slope_ci
from tomography.p9.behavior import write_completion_check
from tomography.p9.pool import LabeledItem
from tomography.p9.reading import check_pool
from tomography.p9.reading import collect_item_activations
from tomography.p9.reading import fit_type_reader
from tomography.p9.reading import p9_template
from tomography.p9.reading import type_auroc
from tomography.p9.signals import first_answer_token
from tomography.p9.signals import next_token_stats
from tomography.p9.unknowns import UNKNOWN_TYPES
from tomography.p9.unknowns import TypePool
from tomography.p9.unknowns import type_pools
from tomography.reading.directions import Method
from tomography.reading.directions import Preprocess
from tomography.reading.metrics import best_layer
from tomography.reading.metrics import wilson_ci
from tomography.reading.reader import RepReader
from tomography.templates import ChatFormat

CALIBRATION = "calibration"
LAT = "lat"
RANDOM = "random"
STEERING_ARM = f"{LAT}0"
METRICS = ("entropy", "answer_logprob")
SIDES = ("known", "unknown")
XLABELS: Mapping[CoeffMode, str] = {
    "relative": "Relative coefficient",
    "absolute": "Coefficient",
}

SHIFT_COLUMNS = [
    "type",
    "side",
    "arm",
    "coeff",
    "metric",
    "shift",
    "lo",
    "hi",
]
RATE_COLUMNS = [
    "type",
    "arm",
    "coeff",
    "false_refusal",
    "correct_refusal",
    "coverage",
    "wrong_among_answered",
    "balanced",
]
EFFECT_COLUMNS = [
    "type",
    "side",
    "arm",
    "coeff",
    "n",
    "abstention_shift",
    "lo",
    "hi",
]
DOSE_COLUMNS = [
    "type",
    "side",
    "metric",
    "arm",
    "vs",
    "n",
    "slope",
    "lo",
    "hi",
    "span",
    "abs_gap",
    "abs_lo",
    "abs_hi",
]

SELECTIVITY_COLUMNS = [
    "type",
    "metric",
    "n_known",
    "n_unknown",
    "known_slope",
    "unknown_slope",
    "gap",
    "lo",
    "hi",
    "selective",
]

Cell = tuple[float, float, float]
CellSlopes = dict[tuple[str, str, str], dict[str, NDArray[np.float64]]]

Behavior = Literal["auto", "abstention", "shift"]
Mode = Literal["abstention", "shift"]


@dataclass(frozen=True, kw_only=True)
class Config(SteeringConfig):
    """P9d settings: the direction, the control and the behavior measured.

    The inherited fields keep their meaning: `coeffs` are relative to the
    mean residual norm, `layers` None steers the layer with the best
    validation AUROC of the direction, and `n_calibration` is the number of
    held-out train questions whose prompts set the norm.

    Attributes:
        labels_run: Path of a finished P9a run of the same model whose gate
            is open.
        template: The prompt the direction is read on, one of
            `tomography.p9.reading.TEMPLATE_NAMES`.
        n_shots: Demonstrations of the "fewshot" prompts; `run` requires
            the number the labels were made with, whose own config bounds
            it.
        direction_type: The knowledge contrast whose direction steers.
        eval_types: The knowledge contrasts the behavior is reported on.
            `post_cutoff` is not a default: its test side holds at most
            1.6 x (2026 - cutoff year) unknown awards, about 3 for a 2024
            cutoff, so it is always skipped for the Qwen3 models. Name it
            to score it for a model with an earlier cutoff.
        method: The direction finder.
        preprocess: Vector preparation, see `fit_direction`.
        n_pairs: Pairs the direction is fitted on.
        n_random: Random directions, norm-matched to the steering one.
        min_pairs: Fewest train items on each side of the direction contrast.
        min_eval: Fewest val items of the direction contrast, and fewest
            test items of an evaluation contrast, on each side.
        behavior: "abstention", "shift", or "auto", which is abstention iff
            the checkpoint is post-trained.
        instruction: The instruction of the chat prompt of abstention.
        n_hand_check: Rows of the completion check file.
        n_boot: Bootstrap resamples of every shift interval.
    """

    experiment: str = "p9d"
    model: str = "qwen3-1.7b"
    labels_run: str
    template: str = "fewshot"
    n_shots: int = 15
    direction_type: str = "obscure_real"
    eval_types: tuple[str, ...] = (
        "obscure_real",
        "fictional_generated",
        "unknowable_detail",
        "false_premise",
    )
    method: Method = "logistic"
    preprocess: Preprocess = "center"
    n_pairs: int = 64
    n_random: int = 3
    min_pairs: int = 16
    min_eval: int = 10
    behavior: Behavior = "auto"
    instruction: str = (
        "Answer the question in a few words. If you do not know the answer, "
        "say that you do not know."
    )
    n_hand_check: int = 100
    n_boot: int = 1000
    n_prompts: int = 100
    max_new_tokens: int = 24
    temperature: float | None = None
    top_p: float = 1.0
    num_samples: int = 1
    coeffs: tuple[float, ...] = (-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4)

    def __post_init__(self) -> None:
        """Checks the settings that would act silently or fail late.

        Raises:
            ValueError: If `coeffs` lacks 0.0 (the unsteered reference),
                holds a non-finite value or fewer than two distinct
                values, `n_random`, `n_prompts`, `n_pairs`, `min_pairs` or
                `min_eval` is below 1, `behavior` is not one of the three
                modes, `mode`, `op` or `mask` is not a known value, or a
                generation setting is invalid (see `GenSettings`). Every
                behavior is held to the same generation settings.
        """
        if 0.0 not in self.coeffs:
            raise ValueError(
                "coeffs must contain 0.0, the unsteered reference, "
                f"got {self.coeffs}"
            )
        if not all(math.isfinite(c) for c in self.coeffs):
            raise ValueError(f"coeffs must be finite, got {self.coeffs}")
        if len(set(self.coeffs)) < 2:
            raise ValueError(
                "coeffs need at least two distinct values for a dose slope, "
                f"got {self.coeffs}"
            )
        for name in (
            "n_random",
            "n_prompts",
            "n_pairs",
            "min_pairs",
            "min_eval",
        ):
            if getattr(self, name) < 1:
                raise ValueError(
                    f"{name} must be at least 1, got {getattr(self, name)}"
                )
        if self.behavior not in get_args(Behavior):
            raise ValueError(
                f"behavior must be one of {get_args(Behavior)}, "
                f"got {self.behavior!r}"
            )
        for name, kinds in (
            ("mode", CoeffMode),
            ("op", Operator),
            ("mask", ResponseMask),
        ):
            if getattr(self, name) not in get_args(kinds):
                raise ValueError(
                    f"{name} must be one of {get_args(kinds)}, "
                    f"got {getattr(self, name)!r}"
                )
        GenSettings.from_config(self)


@dataclass(frozen=True)
class Group:
    """The questions one set of rates is computed on.

    Attributes:
        name: "calibration" or the evaluation contrast.
        known: The known side, the first questions by uid.
        unknown: The unknown side, the first questions by uid.
    """

    name: str
    known: tuple[LabeledItem, ...]
    unknown: tuple[LabeledItem, ...]

    @property
    def sides(self) -> tuple[tuple[str, tuple[LabeledItem, ...]], ...]:
        """The known and the unknown side, each with its name."""
        return (("known", self.known), ("unknown", self.unknown))


@dataclass(frozen=True)
class Plan:
    """The contrasts a run reads and scores.

    Attributes:
        direction: The contrast whose direction steers.
        groups: The calibration group, then one group per usable evaluation
            contrast.
        skipped: Why an evaluation contrast is not scored, by name.
    """

    direction: TypePool
    groups: tuple[Group, ...]
    skipped: dict[str, str]

    @property
    def evaluation(self) -> tuple[Group, ...]:
        """The groups that are scored, after the calibration group."""
        return self.groups[1:]


def _first_by_uid(
    items: Sequence[LabeledItem], n: int
) -> tuple[LabeledItem, ...]:
    return tuple(sorted(items, key=lambda i: i.example.uid)[:n])


def plan_groups(
    config: Config, items: Sequence[LabeledItem], known_contrasts: bool
) -> Plan:
    """Checks the gate and the contrasts and picks the questions to run.

    The calibration group holds the validation questions of the direction
    contrast, the evaluation groups the test questions of each usable
    evaluation contrast, the first `n_prompts` by uid on each side (fewer
    when a pool is smaller).

    Args:
        config: The experiment settings.
        items: The labeled questions of the P9a run.
        known_contrasts: The P9a gate: whether the model has known
            questions left after the guess control.

    Returns:
        The plan.

    Raises:
        ValueError: If the gate is closed, the direction or an evaluation
            contrast is an unknown-type contrast, the direction contrast
            does not exist or fails the pool check, or no evaluation
            contrast is usable.
    """
    if not known_contrasts:
        raise ValueError(
            "the P9a gate known_contrasts is closed for this model: steering "
            "needs known questions to calibrate false refusals"
        )
    pools = type_pools(items)
    refused = [
        name
        for name in dict.fromkeys((config.direction_type, *config.eval_types))
        if name in UNKNOWN_TYPES
    ]
    if refused:
        raise ValueError(
            "P9 steering reads and scores known versus unknown contrasts "
            f"only, not {refused}"
        )
    direction = pools.get(config.direction_type)
    if direction is None:
        raise ValueError(
            f"direction_type {config.direction_type!r} is not a contrast of "
            f"the labels; available {list(pools)}"
        )
    reason = check_pool(
        direction, min_pairs=config.min_pairs, min_eval=config.min_eval
    )
    if reason is not None:
        raise ValueError(f"direction contrast {direction.name}: {reason}")
    groups = [
        Group(
            CALIBRATION,
            *(
                _first_by_uid(side, config.n_prompts)
                for side in direction.split("val")
            ),
        )
    ]
    skipped: dict[str, str] = {}
    for name in dict.fromkeys(config.eval_types):
        pool = pools.get(name)
        if pool is None:
            skipped[name] = "no such type"
            continue
        known, unknown = pool.split("test")
        if min(len(known), len(unknown)) < config.min_eval:
            skipped[name] = (
                f"fewer than {config.min_eval} test items on one side"
            )
            continue
        groups.append(
            Group(
                name,
                _first_by_uid(known, config.n_prompts),
                _first_by_uid(unknown, config.n_prompts),
            )
        )
    if len(groups) == 1:
        raise ValueError(f"no usable evaluation contrast: {skipped}")
    return Plan(direction, tuple(groups), skipped)


def resolve_mode(behavior: Behavior, loaded: LoadedModel) -> Mode:
    """Chooses the behavior a model is measured by.

    Args:
        behavior: "abstention", "shift" or "auto".
        loaded: The wrapped model.

    Returns:
        The behavior; "auto" is abstention iff the checkpoint is
        post-trained (the test models have none and count as base).
    """
    if behavior == "auto":
        checkpoint = loaded.checkpoint
        post_trained = checkpoint is not None and checkpoint.post_trained
        return "abstention" if post_trained else "shift"
    return behavior


def arm_names(n_random: int) -> list[str]:
    """Names the arms of a run: the steering direction, then the controls.

    Args:
        n_random: Number of random directions.

    Returns:
        "lat0", then "random0" to "random{n_random - 1}".
    """
    return [STEERING_ARM, *(f"{RANDOM}{r}" for r in range(n_random))]


@dataclass(frozen=True)
class _Steering:
    """The directions of a run and what they are scaled by.

    Attributes:
        layers: The steered layers.
        layer_source: "config" or "validation".
        val_auroc: The validation AUROC of the direction, per layer 1..L.
        norms: Mean residual norm of the steered layers.
        readers: The reader of each arm, by arm name.
    """

    layers: tuple[int, ...]
    layer_source: str
    val_auroc: list[float]
    norms: dict[int, float]
    readers: dict[str, RepReader]


@dataclass(frozen=True)
class _Items:
    """The distinct questions of a run, in a fixed order.

    Attributes:
        items: Each question once, whichever groups hold it.
        prompts: The prompt of each question.
        rows: The position of each uid in `items`.
    """

    items: list[LabeledItem]
    prompts: list[str]
    rows: dict[str, int]

    @classmethod
    def of(cls, plan: Plan, render: Callable[[LabeledItem], str]) -> Self:
        """Collects the questions of every group of a plan once.

        Args:
            plan: The groups.
            render: Maps a question to its prompt.

        Returns:
            The questions, in order of first appearance.
        """
        unique: dict[str, LabeledItem] = {}
        for group in plan.groups:
            for _, members in group.sides:
                for item in members:
                    unique.setdefault(item.example.uid, item)
        items = list(unique.values())
        return cls(
            items,
            [render(item) for item in items],
            {item.example.uid: r for r, item in enumerate(items)},
        )

    def index(self, members: Sequence[LabeledItem]) -> list[int]:
        """Returns the position of each member."""
        return [self.rows[item.example.uid] for item in members]


def _both(
    sides: tuple[list[LabeledItem], list[LabeledItem]],
) -> list[LabeledItem]:
    return [item for side in sides for item in side]


def _prompt_renderer(
    config: Config, loaded: LoadedModel, mode: Mode
) -> Callable[[LabeledItem], str]:
    if mode == "abstention":
        chat = ChatFormat.from_tokenizer(loaded.tokenizer)
        return lambda item: chat.render_turn(
            f"{config.instruction}\n\nQuestion: {item.example.question}"
        )[0]
    shots = DEFAULT_SHOTS[: config.n_shots]
    return lambda item: few_shot_prompt(item.example.question, shots)


def _steering(
    config: Config,
    loaded: LoadedModel,
    plan: Plan,
    render: Callable[[LabeledItem], str],
) -> _Steering:
    direction = plan.direction
    train = _both(direction.split("train"))
    acts = collect_item_activations(
        loaded,
        [*train, *_both(direction.split("val"))],
        p9_template(config.template, DEFAULT_SHOTS[: config.n_shots]),
        chat_format_for(loaded),
        layers=tuple(range(1, loaded.n_layers + 1)),
        bos=config.bos,
        batch_size=config.batch_size,
    )
    reader = fit_type_reader(
        acts,
        direction,
        method=config.method,
        preprocess=config.preprocess,
        n_pairs=config.n_pairs,
        seed=config.seed,
    )
    val = type_auroc(reader, acts, direction, "val")
    layers = (
        (best_layer(val, acts.layers),)
        if config.layers is None
        else config.layers
    )
    by_uid = sorted(train, key=lambda i: i.example.uid)
    norms = mean_residual_norms(
        loaded,
        [render(i) for i in by_uid[: config.n_calibration]],
        layers,
        bos=config.bos,
        batch_size=config.batch_size,
    )
    names = arm_names(config.n_random)
    randoms = random_readers(
        reader.layers,
        reader.directions.shape[1],
        config.n_random,
        seed=config.seed,
    )
    readers = dict(zip(names, [reader, *randoms], strict=True))
    return _Steering(
        layers=tuple(layers),
        layer_source="validation" if config.layers is None else "config",
        val_auroc=[float(v) for v in val],
        norms=norms,
        readers=readers,
    )


def _write_arm_curves(
    run_dir: RunDir,
    stem: str,
    cells: Mapping[tuple[str, str, float], Cell],
    *,
    n_random: int,
    coeffs: Sequence[float],
    xlabel: str,
    ylabel: str,
) -> None:
    """Writes the four series of one curves file; writes nothing if empty.

    The steering arm keeps its own interval. The random arms are drawn as
    the mean of their values with the minimum and maximum as the band. A
    series with a missing cell is left out.

    Args:
        run_dir: The run to write into.
        stem: File stem of the pair.
        cells: The value, lower and upper bound of each (side, arm,
            coefficient); a missing cell has no entry.
        n_random: Number of random arms.
        coeffs: The x values.
        xlabel: Label of the x axis.
        ylabel: Label of the y axis.
    """
    arms = arm_names(n_random)
    names: list[str] = []
    stacks: list[list[NDArray[np.float64]]] = [[], [], []]
    for kind, members in ((LAT, arms[:1]), (RANDOM, arms[1:])):
        for side in SIDES:
            grid = [[cells.get((side, a, c)) for c in coeffs] for a in members]
            found = [[cell for cell in row if cell is not None] for row in grid]
            if any(len(row) != len(coeffs) for row in found):
                continue
            table = np.array(found, dtype=np.float64)
            if kind == LAT:
                parts = [table[0, :, k] for k in range(3)]
            else:
                values = table[:, :, 0]
                parts = [
                    values.mean(axis=0),
                    values.min(axis=0),
                    values.max(axis=0),
                ]
            for stack, part in zip(stacks, parts, strict=True):
                stack.append(part)
            names.append(f"{kind}: {side}")
    if names:
        write_interval_curves(
            run_dir,
            coeffs,
            np.stack(stacks[0]),
            np.stack(stacks[1]),
            np.stack(stacks[2]),
            series=names,
            stem=stem,
            xlabel=xlabel,
            ylabel=ylabel,
        )


def _dose_rows(
    group: str,
    side: str,
    metric: str,
    coeffs: Sequence[float],
    values: Mapping[str, NDArray[np.float64]],
    *,
    n_boot: int,
    seed: int,
    slopes_out: dict[str, NDArray[np.float64]] | None = None,
) -> list[list[Any]]:
    """Fits the dose slope of every arm and its contrast with each random arm.

    Args:
        group: Name of the evaluation contrast.
        side: "known" or "unknown".
        metric: Name of the outcome.
        coeffs: The coefficients, in the row order of `values`.
        values: By arm, the outcome of each item at each coefficient, shape
            (len(coeffs), n_items), with the steering arm first.
        n_boot: Bootstrap resamples of the items.
        seed: Seed of the resampling.
        slopes_out: If given, receives the slope of every item under every
            arm, whether or not any row is returned.

    Returns:
        One row of `DOSE_COLUMNS` per arm, then one per random arm for the
        paired difference of the steering slope to the slope of that arm,
        which also carries the gap of the absolute mean slopes with its
        interval (None on the rows of one arm); none if no item has a finite
        steering slope. A row without a finite item has `n` 0 and None in
        every other cell, so a random arm without finite slopes does not
        stop the run.
    """
    slopes = {arm: item_slopes(coeffs, v) for arm, v in values.items()}
    if slopes_out is not None:
        slopes_out.update(slopes)
    if not np.isfinite(slopes[STEERING_ARM]).any():
        return []
    width = max(coeffs) - min(coeffs)
    random_arms = [arm for arm in slopes if arm != STEERING_ARM]
    pairs = [(arm, "") for arm in slopes] + [
        (STEERING_ARM, arm) for arm in random_arms
    ]
    rows: list[list[Any]] = []
    for arm, vs in pairs:
        reference = slopes[vs] if vs else None
        own = slopes[arm]
        used = own if reference is None else own - reference
        n = int(np.isfinite(used).sum())
        if n == 0:
            rows.append([group, side, metric, arm, vs, 0, *[None] * 7])
            continue
        slope, lo, hi = slope_ci(own, reference, n_boot=n_boot, seed=seed)
        gap: list[float | None] = [None, None, None]
        if reference is not None:
            gap = list(
                abs_slope_gap_ci(
                    slopes[arm], reference, n_boot=n_boot, seed=seed
                )
            )
        rows.append(
            [
                group,
                side,
                metric,
                arm,
                vs,
                n,
                slope,
                lo,
                hi,
                slope * width,
                *gap,
            ]
        )
    return rows


def _finite_steering_slopes(
    cell_slopes: CellSlopes, key: tuple[str, str, str]
) -> NDArray[np.float64]:
    """The finite steering slopes of one cell, empty if it has none."""
    own = cell_slopes.get(key, {}).get(STEERING_ARM, np.array([]))
    return own[np.isfinite(own)]


def _selectivity_rows(
    cell_slopes: CellSlopes, *, n_boot: int, seed: int
) -> list[list[Any]]:
    """Compares the steering slope on the unknown and the known items.

    Args:
        cell_slopes: By (type, side, metric), the slope of every item under
            every arm.
        n_boot: Bootstrap resamples of the items.
        seed: Seed of the resampling.

    Returns:
        One row of `SELECTIVITY_COLUMNS` per type and metric where either
        side has a finite steering slope. The row holds the counts and None
        elsewhere if a side has no finite slope.
    """
    rows: list[list[Any]] = []
    for group, metric in dict.fromkeys((g, m) for g, _, m in cell_slopes):
        known, unknown = (
            _finite_steering_slopes(cell_slopes, (group, side, metric))
            for side in ("known", "unknown")
        )
        if known.size == 0 and unknown.size == 0:
            continue
        counts = [group, metric, known.size, unknown.size]
        if known.size == 0 or unknown.size == 0:
            rows.append([*counts, *[None] * 6])
            continue
        gap, lo, hi = selectivity_gap(unknown, known, n_boot=n_boot, seed=seed)
        rows.append(
            [
                *counts,
                float(known.mean()),
                float(unknown.mean()),
                gap,
                lo,
                hi,
                lo > 0,
            ]
        )
    return rows


def _write_item_slopes(
    run_dir: RunDir, plan: Plan, cell_slopes: CellSlopes
) -> None:
    """Saves the slope of every item under every arm.

    Args:
        run_dir: Where `item_slopes.npz` is written.
        plan: The contrasts of the run, which give the item order.
        cell_slopes: By (type, side, metric), the slope of every item under
            every arm.
    """
    arrays: dict[str, NDArray[Any]] = {}
    for group in plan.evaluation:
        for side, members in group.sides:
            arrays[f"uids__{group.name}__{side}"] = np.array(
                [m.example.uid for m in members], dtype=str
            )
    for (name, side, metric), arms in cell_slopes.items():
        for arm, values in arms.items():
            arrays[f"slope__{name}__{side}__{metric}__{arm}"] = values
    run_dir.write_arrays("item_slopes", **arrays)


def _write_selectivity(
    run_dir: RunDir, plan: Plan, cell_slopes: CellSlopes, config: Config
) -> list[list[Any]]:
    """Writes the selectivity table and the item slopes.

    Args:
        run_dir: Where the files are written.
        plan: The contrasts of the run.
        cell_slopes: By (type, side, metric), the slope of every item under
            every arm.
        config: The run settings, for the bootstrap size and seed.

    Returns:
        The rows of `SELECTIVITY_COLUMNS`.
    """
    rows = _selectivity_rows(
        cell_slopes, n_boot=config.n_boot, seed=config.seed
    )
    run_dir.write_json(
        "table_selectivity", {"columns": SELECTIVITY_COLUMNS, "rows": rows}
    )
    _write_item_slopes(run_dir, plan, cell_slopes)
    return rows


def _with_selective(
    headline: dict[str, Any] | None, selectivity: Sequence[Sequence[Any]]
) -> dict[str, Any] | None:
    """Adds whether the headline contrast is selective.

    Args:
        headline: The result of `_headline`.
        selectivity: The rows of `SELECTIVITY_COLUMNS` of the whole run.

    Returns:
        The headline with `selective` from the row of its type and metric,
        None if that row is missing or empty; None if there is no headline.
    """
    if headline is None:
        return None
    selective = {(r[0], r[1]): r[9] for r in selectivity}
    return {
        **headline,
        "selective": selective.get((headline["group"], headline["metric"])),
    }


def _headline(
    dose: Sequence[Sequence[Any]], group: str, metric: str
) -> dict[str, Any] | None:
    """Reads the pre-registered claim from the dose rows.

    Args:
        dose: The rows of `DOSE_COLUMNS` of the whole run.
        group: The direction contrast.
        metric: The outcome of the headline.

    Returns:
        The steering slope on the unknown side with its interval, the
        largest absolute random slope, the paired difference to each random
        arm, the gap of the absolute mean slopes to each random arm and
        whether every one of those intervals lies above zero, which is
        false for an arm whose gap is missing; None if the run did not score
        that contrast.
    """
    rows = [r for r in dose if list(r[:3]) == [group, "unknown", metric]]
    own = {r[3]: r for r in rows if not r[4]}
    if STEERING_ARM not in own:
        return None
    differences = {r[4]: [r[6], r[7], r[8]] for r in rows if r[4]}
    abs_gaps = {r[4]: [r[10], r[11], r[12]] for r in rows if r[4]}
    lat = own[STEERING_ARM]
    return {
        "group": group,
        "side": "unknown",
        "metric": metric,
        "lat_slope": lat[6],
        "lat_lo": lat[7],
        "lat_hi": lat[8],
        "max_abs_random_slope": max(
            (
                abs(r[6])
                for arm, r in own.items()
                if arm != STEERING_ARM and r[6] is not None
            ),
            default=0.0,
        ),
        "differences": differences,
        "abs_gaps": abs_gaps,
        "distinguishable": all(
            lo is not None and lo > 0 for _, lo, _ in abs_gaps.values()
        ),
    }


def _summary(
    config: Config, mode: Mode, plan: Plan, steering: _Steering
) -> dict[str, Any]:
    return {
        "mode": mode,
        "template": config.template,
        "n_items": {
            g.name: {"known": len(g.known), "unknown": len(g.unknown)}
            for g in plan.groups
        },
        "layers": list(steering.layers),
        "layer_source": steering.layer_source,
        "direction": {
            "type": config.direction_type,
            "val_auroc": steering.val_auroc,
        },
        "coeffs": list(config.coeffs),
        "types": [g.name for g in plan.evaluation],
        "skipped": plan.skipped,
    }


def _edit_arguments(config: Config, steering: _Steering) -> dict[str, Any]:
    """The edit settings that a controller and a generation arm share."""
    return {
        "layers": steering.layers,
        "op": config.op,
        "mode": config.mode,
        "norms": steering.norms,
        "mask": config.mask,
        "preserve_norm": config.preserve_norm,
    }


def _controller(
    config: Config,
    loaded: LoadedModel,
    steering: _Steering,
    arm: str,
    coeff: float,
) -> ReadingVectorController:
    return ReadingVectorController.from_reader(
        steering.readers[arm],
        coeff=coeff,
        device=loaded.device,
        **_edit_arguments(config, steering),
    )


def _run_shift(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    plan: Plan,
    steering: _Steering,
    render: Callable[[LabeledItem], str],
) -> None:
    distinct = _Items.of(plan, render)
    answered = np.array([bool(i.example.answers) for i in distinct.items])
    targets = [
        first_answer_token(loaded, i.example.answers[0])
        if i.example.answers
        else 0
        for i in distinct.items
    ]

    def measure(
        controller: ReadingVectorController | None,
    ) -> dict[str, NDArray[np.float64]]:
        stats = next_token_stats(
            loaded,
            distinct.prompts,
            targets=targets,
            edits=controller,
            bos=config.bos,
            batch_size=config.batch_size,
        )
        assert stats.target_logprob is not None
        return {
            "entropy": stats.entropy,
            "answer_logprob": np.where(answered, stats.target_logprob, np.nan),
        }

    base = measure(None)
    steered = {
        (name, coeff): measure(
            _controller(config, loaded, steering, name, coeff)
        )
        for name in steering.readers
        for coeff in config.coeffs
    }
    rows: list[list[Any]] = []
    cells: dict[tuple[str, str], dict[tuple[str, str, float], Cell]] = {}
    for group in plan.groups:
        for side, members in group.sides:
            index = distinct.index(members)
            for name, coeff, metric in itertools.product(
                steering.readers, config.coeffs, METRICS
            ):
                before = base[metric][index]
                after = steered[(name, coeff)][metric][index]
                if not (np.isfinite(before) & np.isfinite(after)).any():
                    continue
                cell = shift_ci(
                    before, after, n_boot=config.n_boot, seed=config.seed
                )
                rows.append([group.name, side, name, coeff, metric, *cell])
                cells.setdefault((group.name, metric), {})[
                    (side, name, coeff)
                ] = cell
    dose: list[list[Any]] = []
    cell_slopes: CellSlopes = {}
    for group in plan.evaluation:
        for side, members in group.sides:
            index = distinct.index(members)
            for metric in METRICS:
                values = {
                    name: np.stack(
                        [
                            steered[(name, coeff)][metric] - base[metric]
                            for coeff in config.coeffs
                        ]
                    )[:, index]
                    for name in steering.readers
                }
                dose += _dose_rows(
                    group.name,
                    side,
                    metric,
                    config.coeffs,
                    values,
                    n_boot=config.n_boot,
                    seed=config.seed,
                    slopes_out=cell_slopes.setdefault(
                        (group.name, side, metric), {}
                    ),
                )
    for group in plan.evaluation:
        for metric in METRICS:
            _write_arm_curves(
                run_dir,
                f"curves_{group.name}_{metric}",
                cells.get((group.name, metric), {}),
                n_random=config.n_random,
                coeffs=config.coeffs,
                xlabel=XLABELS[config.mode],
                ylabel=f"Shift in {metric}",
            )
    run_dir.write_json(
        "table",
        {
            "columns": SHIFT_COLUMNS,
            "rows": rows,
        },
    )
    run_dir.write_json("table_dose", {"columns": DOSE_COLUMNS, "rows": dose})
    selectivity = _write_selectivity(run_dir, plan, cell_slopes, config)
    run_dir.write_json(
        "summary",
        {
            **_summary(config, "shift", plan, steering),
            "headline": _with_selective(
                _headline(dose, config.direction_type, "entropy"), selectivity
            ),
            "n_with_answer": {
                g.name: {
                    side: sum(bool(i.example.answers) for i in members)
                    for side, members in g.sides
                }
                for g in plan.groups
            },
        },
    )


def _generate_outcomes(
    config: Config,
    loaded: LoadedModel,
    steering: _Steering,
    distinct: _Items,
) -> tuple[dict[tuple[str, float], list[Outcome]], dict[float, list[str]]]:
    settings = GenSettings.from_config(config)
    outcomes: dict[tuple[str, float], list[Outcome]] = {}
    texts: dict[float, list[str]] = {}
    for name, reader in steering.readers.items():
        for coeff in config.coeffs:
            # A zero coefficient edits nothing, so every arm generates what
            # the steering arm does.
            if coeff == 0 and name != STEERING_ARM:
                outcomes[(name, coeff)] = outcomes[(STEERING_ARM, coeff)]
                continue
            arm = reader_arm(
                loaded,
                settings,
                reader,
                coeff=coeff,
                **_edit_arguments(config, steering),
            )
            first = [samples[0] for samples in arm.generate(distinct.prompts)]
            outcomes[(name, coeff)] = [
                classify(completion, item.example.answers)
                for completion, item in zip(first, distinct.items, strict=True)
            ]
            if name == STEERING_ARM:
                texts[coeff] = [completion.text for completion in first]
    return outcomes, texts


def _run_abstention(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    plan: Plan,
    steering: _Steering,
    render: Callable[[LabeledItem], str],
) -> None:
    distinct = _Items.of(plan, render)
    outcomes, texts = _generate_outcomes(config, loaded, steering, distinct)
    arms = arm_names(config.n_random)

    def side_outcomes(
        arm: str, coeff: float, members: Sequence[LabeledItem]
    ) -> list[Outcome]:
        return [outcomes[(arm, coeff)][r] for r in distinct.index(members)]

    rates: dict[tuple[str, str, float], BehaviorRates] = {}
    rows: list[list[Any]] = []
    for group, coeff, arm in itertools.product(
        plan.groups, config.coeffs, arms
    ):
        rate = behavior_rates(
            side_outcomes(arm, coeff, group.known),
            side_outcomes(arm, coeff, group.unknown),
        )
        rates[(group.name, arm, coeff)] = rate
        rows.append(
            [
                group.name,
                arm,
                coeff,
                rate.false_refusal,
                rate.correct_refusal,
                rate.coverage,
                rate.wrong_among_answered,
                rate.balanced,
            ]
        )
    chosen = select_coefficient(
        config.coeffs,
        [rates[(CALIBRATION, STEERING_ARM, c)] for c in config.coeffs],
    )

    indicator = {
        key: np.array([o.abstained for o in value], dtype=np.float64)
        for key, value in outcomes.items()
    }

    effects: list[list[Any]] = []
    dose: list[list[Any]] = []
    cell_slopes: CellSlopes = {}
    for group in plan.evaluation:
        cells: dict[tuple[str, str, float], Cell] = {}
        for side, members in group.sides:
            index = distinct.index(members)
            for arm, coeff in itertools.product(arms, config.coeffs):
                flags = indicator[(arm, coeff)][index]
                cells[(side, arm, coeff)] = (
                    float(flags.mean()),
                    *wilson_ci(int(flags.sum()), len(flags)),
                )
            for arm in arms:
                shift = shift_ci(
                    indicator[(arm, 0.0)][index],
                    indicator[(arm, chosen)][index],
                    n_boot=config.n_boot,
                    seed=config.seed,
                )
                effects.append(
                    [group.name, side, arm, chosen, len(index), *shift]
                )
            values = {
                arm: np.stack(
                    [indicator[(arm, coeff)] for coeff in config.coeffs]
                )[:, index]
                for arm in arms
            }
            dose += _dose_rows(
                group.name,
                side,
                "abstention",
                config.coeffs,
                values,
                n_boot=config.n_boot,
                seed=config.seed,
                slopes_out=cell_slopes.setdefault(
                    (group.name, side, "abstention"), {}
                ),
            )
        _write_arm_curves(
            run_dir,
            f"curves_{group.name}",
            cells,
            n_random=config.n_random,
            coeffs=config.coeffs,
            xlabel=XLABELS[config.mode],
            ylabel="Abstention rate",
        )
    records = [
        CompletionRecord(
            uid=item.example.uid,
            kind=item.example.kind,
            coeff=coeff,
            question=item.example.question,
            completion=text,
            abstained=outcome.abstained,
            correct=outcome.correct,
        )
        for coeff in config.coeffs
        for item, text, outcome in zip(
            distinct.items,
            texts[coeff],
            outcomes[(STEERING_ARM, coeff)],
            strict=True,
        )
    ]
    written = write_completion_check(
        run_dir.path / "hand_check.csv",
        records,
        n=config.n_hand_check,
        seed=config.seed,
    )
    run_dir.write_json(
        "table",
        {
            "columns": RATE_COLUMNS,
            "rows": rows,
        },
    )
    run_dir.write_json(
        "table_effect",
        {
            "columns": EFFECT_COLUMNS,
            "rows": effects,
        },
    )
    balanced = {
        arm: [rates[(CALIBRATION, arm, c)].balanced for c in config.coeffs]
        for arm in arms
    }
    run_dir.write_json("table_dose", {"columns": DOSE_COLUMNS, "rows": dose})
    selectivity = _write_selectivity(run_dir, plan, cell_slopes, config)
    run_dir.write_json(
        "summary",
        {
            **_summary(config, "abstention", plan, steering),
            "headline": _with_selective(
                _headline(dose, config.direction_type, "abstention"),
                selectivity,
            ),
            "chosen_coeff": chosen,
            "chosen_index": list(config.coeffs).index(chosen),
            "calibration_balanced_range": {
                arm: max(values) - min(values)
                for arm, values in balanced.items()
            },
            "hand_check_rows": written,
        },
    )


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    items: Sequence[LabeledItem],
    *,
    known_contrasts: bool,
) -> None:
    """Steers along the known versus unknown direction and writes the result.

    The direction is fitted on the train items of the direction contrast
    and added to the residual stream at the steered layers with the
    coefficients of the grid, once per arm: the direction itself and each
    random direction of the same size. Every distinct question is run once
    per arm and coefficient, whichever groups it belongs to.

    In abstention mode the first completion of each question is classified
    and the rates of each group, arm and coefficient are written to `table`.
    The headline is the dose slope: per evaluation group, side and arm,
    `table_dose` holds the mean over items of the least-squares slope of the
    abstention indicator on the coefficient, with an item bootstrap
    interval, and for each random arm the paired difference of the steering
    slope to that arm, and the gap between the absolute mean slopes. The
    `summary` headline reads the unknown side of the direction contrast:
    steering is distinguishable from any edit of this size iff, for every
    random arm, the lower bound of that gap is above zero, that is, steering
    moves the outcome more than the random arm does, whatever the sign.
    The coefficient chosen on the calibration group is secondary, because
    the calibration score can be flat over the grid
    (`calibration_balanced_range` shows how flat): `table_effect` holds,
    for each arm separately, the paired change of the abstention indicator
    against coefficient 0.0 at that coefficient. Also written are
    `curves_<type>` per evaluation contrast, `hand_check.csv` (completions
    of the steering arm for checking the abstention patterns by hand) and
    `summary`.

    In shift mode each group, side and arm get the paired shift of the
    entropy and of the answer log-probability of the next token against the
    unsteered model, and no coefficient is chosen. `table_dose` holds the
    same slopes and differences for both metrics on the evaluation groups
    (a side without finite values is left out) and the headline metric is
    the entropy. Writes `curves_<type>_entropy` and
    `curves_<type>_answer_logprob` per evaluation contrast (a series with a
    missing cell is left out, a file without series is not written),
    `table`, `table_dose` and `summary`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model.
        items: The labeled questions of the P9a run.
        known_contrasts: The P9a gate.

    Raises:
        ValueError: If the gate is closed, a contrast is refused or too
            small, abstention is asked of a tokenizer without a chat
            template, a steered layer is not a layer of the model, or the
            template is unknown.
    """
    plan = plan_groups(config, items, known_contrasts)
    mode = resolve_mode(config.behavior, loaded)
    if mode == "abstention" and not getattr(
        loaded.tokenizer, "chat_template", None
    ):
        raise ValueError(
            "abstention steering needs a tokenizer with a chat template"
        )
    if config.layers is not None and not all(
        1 <= layer <= loaded.n_layers for layer in config.layers
    ):
        raise ValueError(
            f"layers {config.layers} must lie in 1..{loaded.n_layers}"
        )
    render = _prompt_renderer(config, loaded, mode)
    steering = _steering(config, loaded, plan, render)
    run = _run_abstention if mode == "abstention" else _run_shift
    run(config, run_dir, loaded, plan, steering, render)


def run(config: Config, run_dir: RunDir) -> None:
    """Loads the labeled pool and the model, then runs `execute`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.

    Raises:
        ValueError: If `config.labels_run` is not a finished run, it labeled
            another model than `config.model`, or it was made with another
            number of shots than `config.n_shots`.
    """
    items, summary = load_labeled(config.labels_run, model=config.model)
    labeled_shots = summary["recipe"]["n_shots"]
    if labeled_shots != config.n_shots:
        raise ValueError(
            f"the labels run was made with n_shots {labeled_shots} but "
            f"config.n_shots is {config.n_shots}; the direction is read on "
            "the few-shot prompt the labels were made with"
        )
    execute(
        config,
        run_dir,
        load_model(config.model),
        items,
        known_contrasts=bool(summary["known_contrasts"]),
    )


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws every curves file of a P9d run.

    Args:
        run_dir: A finished P9d run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths of each `curves_*` stem, by stem.
    """
    paths: list[Path] = []
    for file in sorted(run_dir.path.glob("curves_*.json")):
        paths += plot_curves_file(run_dir, out_dir, stem=file.stem)
    return paths
