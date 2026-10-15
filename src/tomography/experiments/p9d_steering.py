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

from tomography.control.controllers import ReadingVectorController
from tomography.control.controllers import mean_residual_norms
from tomography.experiments.base import RunDir
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
from tomography.p9.behavior import behavior_rates
from tomography.p9.behavior import classify
from tomography.p9.behavior import select_coefficient
from tomography.p9.behavior import shift_ci
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
METRICS = ("entropy", "answer_logprob")
SIDES = ("known", "unknown")
XLABEL = "Relative coefficient"

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

Cell = tuple[float, float, float]

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
        n_shots: Demonstrations of the "fewshot" prompts.
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
    n_prompts: int = 50
    max_new_tokens: int = 24
    temperature: float | None = None
    top_p: float = 1.0
    num_samples: int = 1
    coeffs: tuple[float, ...] = (-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4)

    def __post_init__(self) -> None:
        """Checks the settings that would act silently or fail late.

        Raises:
            ValueError: If `n_shots` is outside 0 to the number of default
                demonstrations, `coeffs` lacks 0.0 (the unsteered reference)
                or holds a non-finite value, `n_random`, `n_prompts`,
                `n_pairs`, `min_pairs` or `min_eval` is below 1,
                `behavior` is not one of the three modes, or a generation
                setting is invalid (see `GenSettings`). Every behavior is
                held to the same generation settings.
        """
        if not 0 <= self.n_shots <= len(DEFAULT_SHOTS):
            raise ValueError(
                f"n_shots must be in [0, {len(DEFAULT_SHOTS)}], "
                f"got {self.n_shots}"
            )
        if 0.0 not in self.coeffs:
            raise ValueError(
                "coeffs must contain 0.0, the unsteered reference, "
                f"got {self.coeffs}"
            )
        if not all(math.isfinite(c) for c in self.coeffs):
            raise ValueError(f"coeffs must be finite, got {self.coeffs}")
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
    evaluation contrast, the first `n_prompts` by uid on each side.

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
    return [f"{LAT}0", *(f"{RANDOM}{r}" for r in range(n_random))]


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
    rng = np.random.default_rng(config.seed)
    zeros = np.zeros_like(reader.directions)
    names = arm_names(config.n_random)
    readers = {names[0]: reader}
    for name in names[1:]:
        readers[name] = RepReader.from_arrays(
            RANDOM,
            reader.layers,
            rng.standard_normal(reader.directions.shape),
            zeros,
        )
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
            xlabel=XLABEL,
            ylabel=ylabel,
        )


def _summary(
    config: Config, mode: Mode, plan: Plan, steering: _Steering
) -> dict[str, Any]:
    return {
        "mode": mode,
        "layers": list(steering.layers),
        "layer_source": steering.layer_source,
        "direction": {
            "type": config.direction_type,
            "val_auroc": steering.val_auroc,
        },
        "coeffs": list(config.coeffs),
        "types": [g.name for g in plan.groups[1:]],
        "skipped": plan.skipped,
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
        steering.layers,
        coeff=coeff,
        op=config.op,
        mode=config.mode,
        norms=steering.norms,
        mask=config.mask,
        preserve_norm=config.preserve_norm,
        device=loaded.device,
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
    for group in plan.groups[1:]:
        for metric in METRICS:
            _write_arm_curves(
                run_dir,
                f"curves_{group.name}_{metric}",
                cells.get((group.name, metric), {}),
                n_random=config.n_random,
                coeffs=config.coeffs,
                ylabel=f"Shift in {metric}",
            )
    run_dir.write_json(
        "table",
        {
            "columns": SHIFT_COLUMNS,
            "rows": rows,
        },
    )
    run_dir.write_json(
        "summary",
        {
            **_summary(config, "shift", plan, steering),
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
    steering_arm = arm_names(config.n_random)[0]
    outcomes: dict[tuple[str, float], list[Outcome]] = {}
    texts: dict[float, list[str]] = {}
    for name, reader in steering.readers.items():
        for coeff in config.coeffs:
            # A zero coefficient edits nothing, so every arm generates what
            # the steering arm does.
            if coeff == 0 and name != steering_arm:
                outcomes[(name, coeff)] = outcomes[(steering_arm, coeff)]
                continue
            arm = reader_arm(
                loaded,
                settings,
                reader,
                layers=steering.layers,
                coeff=coeff,
                op=config.op,
                mode=config.mode,
                norms=steering.norms,
                mask=config.mask,
                preserve_norm=config.preserve_norm,
            )
            first = [samples[0] for samples in arm.generate(distinct.prompts)]
            outcomes[(name, coeff)] = [
                classify(completion, item.example.answers)
                for completion, item in zip(first, distinct.items, strict=True)
            ]
            if name == steering_arm:
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
        config.coeffs, [rates[(CALIBRATION, arms[0], c)] for c in config.coeffs]
    )

    indicator = {
        key: np.array([o.abstained for o in value], dtype=np.float64)
        for key, value in outcomes.items()
    }

    def share(kind_arms: Sequence[str], coeff: float) -> NDArray[np.float64]:
        return np.mean([indicator[(a, coeff)] for a in kind_arms], axis=0)

    effects: list[list[Any]] = []
    for group in plan.groups[1:]:
        cells: dict[tuple[str, str, float], Cell] = {}
        for side, members in group.sides:
            index = distinct.index(members)
            for arm, coeff in itertools.product(arms, config.coeffs):
                flags = indicator[(arm, coeff)][index]
                cells[(side, arm, coeff)] = (
                    float(flags.mean()),
                    *wilson_ci(int(flags.sum()), len(flags)),
                )
            for kind, kind_arms in ((LAT, arms[:1]), (RANDOM, arms[1:])):
                shift = shift_ci(
                    share(kind_arms, 0.0)[index],
                    share(kind_arms, chosen)[index],
                    n_boot=config.n_boot,
                    seed=config.seed,
                )
                effects.append(
                    [group.name, side, kind, chosen, len(index), *shift]
                )
        _write_arm_curves(
            run_dir,
            f"curves_{group.name}",
            cells,
            n_random=config.n_random,
            coeffs=config.coeffs,
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
            outcomes[(arms[0], coeff)],
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
    run_dir.write_json(
        "summary",
        {
            **_summary(config, "abstention", plan, steering),
            "chosen_coeff": chosen,
            "chosen_index": list(config.coeffs).index(chosen),
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
    The coefficient is chosen on the calibration group only and reported on
    the evaluation groups: `table_effect` holds the paired change of the
    abstention indicator against coefficient 0.0 at that coefficient. Also
    written are `curves_<type>` per evaluation contrast, `hand_check.csv`
    (completions of the steering arm for checking the abstention patterns
    by hand) and `summary`.

    In shift mode each group, side and arm get the paired shift of the
    entropy and of the answer log-probability of the next token against the
    unsteered model, and no coefficient is chosen. Writes
    `curves_<type>_entropy` and `curves_<type>_answer_logprob` per
    evaluation contrast (a series with a missing cell is left out, a file
    without series is not written), `table` and `summary`.

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
            f"config.n_shots is {config.n_shots}; the steering prompt is "
            "the prompt the labels were made with"
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
