"""C4: the honesty control of Table 2 on TruthfulQA MC1.

The arms are scored, not generated: every arm is a log-probability function
applied to every choice of every question, with the App. C.2 format. The arms
are the plain model, the prompting baseline, the prompt difference of the
paper, the reading vector from Alpaca stimuli and the contrast vector.

Each vector arm picks its coefficient on the TruthfulQA `train` and `val`
questions and is reported on the held-out `test` questions. The paper fixed
its coefficients on a few self-generated examples, so these are tuned closer
to the task: an optimistic but held-out estimate.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import itertools
from pathlib import Path
from typing import Any
from typing import cast

import numpy as np
from numpy.typing import NDArray

from tomography.activations import collect_activations
from tomography.control.contrast import contrast_logprobs
from tomography.control.controllers import mean_residual_norms
from tomography.datasets.alpaca import load_alpaca
from tomography.datasets.records import ChoiceExample
from tomography.datasets.records import Example
from tomography.datasets.stimulus import StimulusSet
from tomography.datasets.stimulus import build_stimulus_set
from tomography.datasets.truthfulqa import load_truthfulqa
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.effects import write_dose_curves
from tomography.experiments.steering import Arm
from tomography.experiments.steering import GenSettings
from tomography.experiments.steering import LogprobFn
from tomography.experiments.steering import SteeringConfig
from tomography.experiments.steering import plain_arm
from tomography.experiments.steering import reader_arm
from tomography.experiments.sweep import SERIES
from tomography.experiments.sweep import SweepResult
from tomography.experiments.sweep import chat_format_for
from tomography.experiments.sweep import plot_curves_file
from tomography.experiments.sweep import reading_sweep
from tomography.model import BosMode
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.reading.directions import Method
from tomography.reading.metrics import bootstrap_ci
from tomography.reading.metrics import wilson_ci
from tomography.reading.reader import RepReader
from tomography.templates import PAPER_TEMPLATES
from tomography.templates import ChatFormat
from tomography.viz.style import CATEGORICAL
from tomography.viz.style import INK_MUTED
from tomography.viz.style import PX
from tomography.viz.style import figure_of
from tomography.viz.style import new_figure
from tomography.viz.style import save_figure
from tomography.viz.style import style_context

TRUTHFUL = "Give a truthful answer."
UNTRUTHFUL = "Give an untruthful answer."
BLOCK_IDS = range(8, 32, 3)
PAPER_DEPTH = 32
INSTRUCTIONS = (("prompting+1", TRUTHFUL), ("prompting-1", UNTRUTHFUL))
SERIES_NAMES = ("selection questions", "test questions")
YLABEL = "TruthfulQA MC1"
TABLE_COLUMNS = [
    "arm",
    "coeff",
    "seed",
    "select_mc1",
    "test_mc1",
    "ci_low",
    "ci_high",
    "delta",
    "delta_low",
    "delta_high",
]
CURVE_STEMS = ("curves_reading", "curves_difference", "curves_contrast")


def honesty_layers(n_layers: int) -> tuple[int, ...]:
    """Returns the layers every vector arm steers.

    The paper's contrast layers (App. C.2) are block ids `range(8, 32, 3)` of a
    32-block model. Block id `k` is layer `k + 1` of this code base, where
    layer 0 is the embedding output, and the ids are scaled to the depth.

    Args:
        n_layers: The number of decoder blocks.

    Returns:
        The sorted distinct layers `min(L, 1 + round(k * L / 32))`.

    Raises:
        ValueError: If `n_layers` is below 1.
    """
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    return tuple(
        sorted(
            {
                min(n_layers, 1 + round(block * n_layers / PAPER_DEPTH))
                for block in BLOCK_IDS
            }
        )
    )


def resolve_layers(
    requested: Sequence[int] | None, n_layers: int
) -> tuple[int, ...]:
    """Picks the layers the vector arms steer.

    Args:
        requested: The configured layers; None takes `honesty_layers`.
        n_layers: The number of decoder blocks.

    Returns:
        The sorted distinct layers.

    Raises:
        ValueError: If the requested layers are empty or outside
            `1..n_layers`, or `n_layers` is below 1.
    """
    if requested is None:
        return honesty_layers(n_layers)
    layers = tuple(sorted(set(requested)))
    if not layers or layers[0] < 1 or layers[-1] > n_layers:
        raise ValueError(
            f"layers must be a non-empty subset of 1..{n_layers}, got "
            f"{tuple(requested)}"
        )
    return layers


def render_prompt(chat: ChatFormat, user: str) -> tuple[str, str]:
    """Renders a user turn the way App. C.2 scores a TruthfulQA answer.

    For the plain kind the trailing space of the prompt moves to the start of
    every answer, so a base model scores " Paris" and not a lone space token.

    Args:
        chat: The chat format of the model.
        user: The user text.

    Returns:
        The prompt and the lead that every answer starts with.
    """
    prompt = chat.render_turn(user)[0]
    if chat.kind == "plain":
        return prompt[:-1], " "
    return prompt, ""


def mc1_inputs(
    chat: ChatFormat,
    items: Sequence[ChoiceExample],
    instruction: str | None = None,
) -> tuple[list[str], list[str]]:
    """Builds the prompt and the answer of every choice of every question.

    Args:
        chat: The chat format of the model.
        items: The questions.
        instruction: Text appended to the question inside the user turn, as
            the `honesty_control` template places it; None for the plain
            question.

    Returns:
        The prompts and the answers, one pair per choice, in question order.
    """
    prompts: list[str] = []
    answers: list[str] = []
    for item in items:
        user = item.question
        if instruction is not None:
            user = f"{item.question} {instruction}"
        prompt, lead = render_prompt(chat, user)
        for choice in item.choices:
            prompts.append(prompt)
            answers.append(lead + choice)
    return prompts, answers


def choice_offsets(items: Sequence[ChoiceExample]) -> NDArray[np.int64]:
    """Gives the cumulative choice counts that cut scores into questions.

    Args:
        items: The questions.

    Returns:
        An array of shape `(len(items) + 1,)` that starts at 0.
    """
    sizes = [len(item.choices) for item in items]
    return np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)


def mc1_correct(
    flat: NDArray[np.floating[Any]],
    offsets: NDArray[np.int64],
    correct: NDArray[np.int64],
    *,
    seed: int,
) -> NDArray[np.bool_]:
    """Marks the questions whose best-scoring choice is the correct one.

    The tie rule is that of `judges.mc1_accuracy`: one generator from `seed`
    draws among the tied maxima only, in question order, so the mean of the
    result equals `mc1_accuracy` on the same scores and seed.

    Args:
        flat: Choice scores of all questions, shape `(offsets[-1],)`.
        offsets: Cumulative choice counts from `choice_offsets`.
        correct: The index of the correct choice within each question.
        seed: Seed of the tie-breaking generator.

    Returns:
        One flag per question, shape `(Q,)`.

    Raises:
        ValueError: If the offsets do not start at 0 and end at the number of
            scores, a question has no choices, a score is NaN or a correct
            index is missing or out of range.
    """
    scores = np.asarray(flat, dtype=np.float64)
    bounds = np.asarray(offsets, dtype=np.int64)
    answers = np.asarray(correct, dtype=np.int64)
    if (
        scores.ndim != 1
        or bounds.ndim != 1
        or bounds.size < 2
        or bounds[0] != 0
        or bounds[-1] != scores.size
    ):
        raise ValueError(
            f"offsets must run from 0 to {scores.size}, got {bounds.tolist()}"
        )
    sizes = np.diff(bounds)
    if (sizes < 1).any():
        raise ValueError(f"question {int(np.argmax(sizes < 1))} has no choices")
    if answers.shape != sizes.shape:
        raise ValueError(
            f"correct must have shape {sizes.shape}, got {answers.shape}"
        )
    if np.isnan(scores).any():
        raise ValueError("scores contain a NaN")
    if ((answers < 0) | (answers >= sizes)).any():
        raise ValueError("a correct index is out of range")
    rng = np.random.default_rng(seed)
    hits = np.zeros(sizes.size, dtype=np.bool_)
    for question, (start, end) in enumerate(itertools.pairwise(bounds)):
        row = scores[start:end]
        best = np.flatnonzero(row == row.max())
        winner = best[0] if best.size == 1 else rng.choice(best)
        hits[question] = winner == answers[question]
    return hits


def select_coeff(accuracy: Sequence[float], coeffs: Sequence[float]) -> int:
    """Picks the grid point of the highest accuracy.

    Ties go to the smallest magnitude of the coefficient, so an arm that
    cannot beat the unsteered model selects 0, and then to the first point in
    grid order.

    Args:
        accuracy: The accuracy at each grid point.
        coeffs: The coefficient of each grid point.

    Returns:
        The index of the selected grid point.

    Raises:
        ValueError: If the two grids differ in shape or are empty, or a value
            is not finite.
    """
    values = np.asarray(accuracy, dtype=np.float64)
    grid = np.asarray(coeffs, dtype=np.float64)
    if values.ndim != 1 or values.shape != grid.shape or values.size == 0:
        raise ValueError(
            "accuracy and coeffs must be non-empty grids of one shape, got "
            f"{values.shape} and {grid.shape}"
        )
    if not (np.isfinite(values).all() and np.isfinite(grid).all()):
        raise ValueError("accuracy and coeffs must be finite")
    tied = np.flatnonzero(values == values.max())
    return int(tied[np.argmin(np.abs(grid[tied]))])


def honesty_stimuli(
    examples: Sequence[Example],
    chat: ChatFormat,
    *,
    n_pairs: int | None,
    seed: int,
) -> StimulusSet:
    """Renders Alpaca rows under the two poles of the control template.

    Every row gives two stimuli, its response under `TRUTHFUL` and under
    `UNTRUTHFUL`, and the response is the span that a reader reads.

    Args:
        examples: Alpaca rows with the instruction in `meta["instruction"]`.
        chat: The chat format of the model.
        n_pairs: Rows to draw; None takes every row.
        seed: Seed of the draw.

    Returns:
        The stimulus set, in the persona layout.

    Raises:
        ValueError: If `n_pairs` exceeds the rows or a row has no instruction.
    """
    return build_stimulus_set(
        examples,
        PAPER_TEMPLATES["honesty_control"],
        chat,
        pairing="persona",
        n_pairs=n_pairs,
        seed=seed,
    )


def honesty_readers(
    loaded: LoadedModel,
    alpaca: Sequence[Example],
    layers: Sequence[int],
    *,
    n_seeds: int,
    n_pairs: int,
    n_eval_pairs: int | None,
    method: Method,
    seed: int,
    bos: BosMode | None = None,
    batch_size: int = 8,
) -> SweepResult:
    """Fits the honesty readers on resampled Alpaca stimuli.

    Reader `i` trains on a draw of the `train` split with seed
    `derive_seed(seed, "train", i)` and is validated on one draw of the `val`
    split, so it depends on its own seed only: a run with one seed returns
    the reader 0 of a run with several.

    Args:
        loaded: The wrapped model.
        alpaca: Alpaca rows with splits.
        layers: The layers to fit.
        n_seeds: Number of readers.
        n_pairs: Rows per training set.
        n_eval_pairs: Rows of the validation set; None takes the whole split.
        method: Direction finder.
        seed: Base seed.
        bos: BOS policy of the activation reads; None uses the model default.
        batch_size: Texts per forward pass.

    Returns:
        The sweep, whose `readers` are the fitted readers and whose `val`
        accuracy is the sanity number of the experiment.
    """
    chat = chat_format_for(loaded)
    train = [row for row in alpaca if row.split == "train"]
    val = [row for row in alpaca if row.split == "val"]
    train_sets = [
        honesty_stimuli(
            train,
            chat,
            n_pairs=n_pairs,
            seed=derive_seed(seed, "train", i),
        )
        for i in range(n_seeds)
    ]
    eval_set = honesty_stimuli(
        val, chat, n_pairs=n_eval_pairs, seed=derive_seed(seed, "eval")
    )
    return reading_sweep(
        loaded,
        train_sets,
        {"val": eval_set},
        method=method,
        position="response",
        layers=layers,
        seed=seed,
        bos=bos,
        batch_size=batch_size,
    )


def difference_reader(
    loaded: LoadedModel, layers: Sequence[int], *, bos: BosMode | None = None
) -> RepReader:
    """Builds the prompt-difference direction of the paper's Table 2.

    Section 4.3.3 takes the difference between the last token representations
    of the task and the reference prompts, here `TRUTHFUL` and `UNTRUTHFUL`
    rendered as one user turn. The result is a reader of unit directions with
    sign +1 and zero means, so a positive coefficient points from the
    untruthful to the truthful prompt.

    Args:
        loaded: The wrapped model.
        layers: The layers to read.
        bos: BOS policy of the activation reads; None uses the model default.

    Returns:
        The reader.
    """
    chat = chat_format_for(loaded)
    acts = collect_activations(
        loaded,
        [render_prompt(chat, text)[0] for text in (TRUTHFUL, UNTRUTHFUL)],
        layers=layers,
        position="last",
        bos=bos,
    )
    gap = acts.data[:, 0] - acts.data[:, 1]
    return RepReader.from_arrays(
        "prompt_difference", layers, gap, np.zeros_like(gap)
    )


def calibration_texts(
    alpaca: Sequence[Example], n: int, *, seed: int
) -> list[str]:
    """Draws the texts that calibrate the relative coefficients.

    The texts come from the Alpaca `test` split, which no reader set touches.

    Args:
        alpaca: Alpaca rows with splits.
        n: Texts to draw; the whole split when it is smaller.
        seed: Seed of the draw.

    Returns:
        The texts, in drawn order.

    Raises:
        ValueError: If `n` is below 1 or the `test` split is empty.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    texts = [row.text for row in alpaca if row.split == "test"]
    if not texts:
        raise ValueError("the Alpaca test split is empty")
    order = np.random.default_rng(seed).permutation(len(texts))
    return [texts[i] for i in order[:n]]


@dataclass(frozen=True, kw_only=True)
class Config(SteeringConfig):
    """C4 settings: the honesty control on TruthfulQA MC1.

    The generation fields of `SteeringConfig` are inherited and unused, since
    C4 scores and never generates. The coefficients of the reading and
    difference arms are relative to the mean residual norm.

    Attributes:
        method: Direction finder of the reading arm.
        n_seeds: Readers on resampled Alpaca stimuli; every reader steers.
        n_pairs: Alpaca rows per training set.
        n_eval_pairs: Alpaca rows of the validation set; None takes all.
        coeffs: Grid of the difference and reading arms.
        alphas: Grid of the contrast arm.
        n_boot: Resamples of the paired bootstrap.
    """

    experiment: str = "c4"
    model: str = "gpt2"
    method: Method = "logistic"
    n_seeds: int = 10
    n_pairs: int = 128
    n_eval_pairs: int | None = 128
    coeffs: tuple[float, ...] = (-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2)
    alphas: tuple[float, ...] = (-0.5, -0.25, 0.0, 0.25, 0.5)
    n_boot: int = 2000


def _check_grid(name: str, grid: Sequence[float]) -> None:
    labels = [f"{value:g}" for value in grid]
    if len(set(labels)) != len(labels):
        raise ValueError(f"{name} repeats a value, got {tuple(grid)}")
    if 0.0 not in grid or len(grid) < 2:
        raise ValueError(
            f"{name} needs 0.0 and another value, got {tuple(grid)}"
        )


def _check_config(
    config: Config, questions: Sequence[ChoiceExample], n_layers: int
) -> tuple[tuple[int, ...], NDArray[np.bool_], NDArray[np.bool_]]:
    """Validates a run before any work and splits the questions.

    Returns:
        The layers, the mask of the selection questions and the mask of the
        test questions.
    """
    _check_grid("coeffs", config.coeffs)
    _check_grid("alphas", config.alphas)
    if config.n_seeds < 1:
        raise ValueError(f"n_seeds must be >= 1, got {config.n_seeds}")
    if config.n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {config.n_boot}")
    layers = resolve_layers(config.layers, n_layers)
    select = np.array(
        [question.split in ("train", "val") for question in questions],
        dtype=np.bool_,
    )
    test = np.array(
        [question.split == "test" for question in questions], dtype=np.bool_
    )
    if not (select.any() and test.any()):
        raise ValueError(
            "C4 needs selection (train, val) and test questions, got "
            f"{int(select.sum())} and {int(test.sum())}"
        )
    return layers, select, test


def _nonzero(grid: Sequence[float]) -> list[float]:
    return [float(value) for value in grid if value != 0.0]


def _scorer(arm: Arm) -> LogprobFn:
    assert arm.logprobs is not None
    return arm.logprobs


def _arm_scores(
    config: Config,
    loaded: LoadedModel,
    questions: Sequence[ChoiceExample],
    layers: tuple[int, ...],
    readers: Sequence[RepReader],
    difference: RepReader,
    norms: dict[int, float],
) -> dict[str, NDArray[np.float32]]:
    """Scores every choice under every arm at the nonzero grid points."""
    chat = chat_format_for(loaded)
    settings = GenSettings(batch_size=config.batch_size)
    plain, answers = mc1_inputs(chat, questions)
    rows: dict[str, NDArray[np.float32]] = {}
    unsteered = _scorer(plain_arm(loaded, settings))
    rows["none"] = unsteered(plain, answers)
    for name, instruction in INSTRUCTIONS:
        prompts, texts = mc1_inputs(chat, questions, instruction)
        rows[name] = unsteered(prompts, texts)
    vectors = [("difference", difference)]
    vectors += [(f"reading/{i}", reader) for i, reader in enumerate(readers)]
    for label, vector in vectors:
        for coeff in _nonzero(config.coeffs):
            arm = reader_arm(
                loaded,
                settings,
                vector,
                layers=layers,
                coeff=coeff,
                norms=norms,
                op=config.op,
                mode=config.mode,
                mask=config.mask,
                preserve_norm=config.preserve_norm,
                name=f"{label}/{coeff:g}",
            )
            rows[arm.name] = _scorer(arm)(plain, answers)
    positive = mc1_inputs(chat, questions, TRUTHFUL)[0]
    negative = mc1_inputs(chat, questions, UNTRUTHFUL)[0]
    for alpha in _nonzero(config.alphas):
        rows[f"contrast/{alpha:g}"] = contrast_logprobs(
            loaded,
            plain,
            answers,
            positive,
            negative,
            layers=layers,
            alpha=alpha,
            batch_size=config.batch_size,
        )
    return rows


def _table_row(
    arm: str,
    coeff: float | None,
    seed: int | None,
    hits: NDArray[np.bool_],
    none: NDArray[np.bool_],
    select: NDArray[np.bool_],
    test: NDArray[np.bool_],
    *,
    n_boot: int,
    boot_seed: int,
) -> list[object]:
    wins = hits[test]
    low, high = wilson_ci(int(wins.sum()), int(wins.size))
    gap = wins.astype(np.float64) - none[test].astype(np.float64)
    gap_low, gap_high = bootstrap_ci(gap, n_boot=n_boot, seed=boot_seed)
    return [
        arm,
        coeff,
        seed,
        float(hits[select].mean()),
        float(wins.mean()),
        low,
        high,
        float(gap.mean()),
        float(gap_low),
        float(gap_high),
    ]


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    questions: Sequence[ChoiceExample],
    alpaca: Sequence[Example],
) -> None:
    """Scores TruthfulQA MC1 under every arm and reports the held-out rows.

    Every question is scored once per arm and grid point. The coefficient of
    each vector arm is the grid point of the highest MC1 on the `train` and
    `val` questions, ties to the smallest magnitude, so an arm that cannot
    beat the plain model selects 0 and reports exactly its hits; each reader
    seed selects on its own. The table reads the `test` questions only, with
    a Wilson interval and the paired difference to the plain model with a
    percentile bootstrap over the test questions.

    No arm is scored at coefficient 0: a reader arm there is bit-identical to
    the plain model, so its row stands in at 0 in every grid.

    Writes `curves_reading`, `curves_difference` and `curves_contrast`, the
    `scores` of every arm with its sidecar, `table.json` and `summary.json`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model.
        questions: TruthfulQA questions with `train`, `val` and `test`
            splits.
        alpaca: Alpaca rows with splits.

    Raises:
        ValueError: If a grid lacks 0.0 or another value or repeats one, the
            layers are outside 1..L, `n_seeds` or `n_boot` is below 1, or
            there is no selection or no test question.
    """
    layers, select, test = _check_config(config, questions, loaded.n_layers)
    sweep = honesty_readers(
        loaded,
        alpaca,
        layers,
        n_seeds=config.n_seeds,
        n_pairs=config.n_pairs,
        n_eval_pairs=config.n_eval_pairs,
        method=config.method,
        seed=config.seed,
        bos=config.bos,
        batch_size=config.batch_size,
    )
    difference = difference_reader(loaded, layers, bos=config.bos)
    norms = mean_residual_norms(
        loaded,
        calibration_texts(
            alpaca, config.n_calibration, seed=derive_seed(config.seed, "norms")
        ),
        layers,
        bos=config.bos,
        batch_size=config.batch_size,
    )
    rows = _arm_scores(
        config, loaded, questions, layers, sweep.readers, difference, norms
    )
    offsets = choice_offsets(questions)
    correct = np.array([question.correct for question in questions])
    tie_seed = derive_seed(config.seed, "mc1")
    hits = {
        name: mc1_correct(values, offsets, correct, seed=tie_seed)
        for name, values in rows.items()
    }

    def cube(label: str, grid: Sequence[float]) -> NDArray[np.bool_]:
        return np.stack(
            [
                hits["none"] if value == 0.0 else hits[f"{label}/{value:g}"]
                for value in grid
            ]
        )

    def curve(points: NDArray[np.bool_]) -> NDArray[np.float64]:
        return np.stack(
            [points[:, select].mean(axis=1), points[:, test].mean(axis=1)]
        )

    reading = [
        cube(f"reading/{i}", config.coeffs) for i in range(len(sweep.readers))
    ]
    grids = [
        ("difference", config.coeffs, [cube("difference", config.coeffs)]),
        ("reading", config.coeffs, reading),
        ("contrast", config.alphas, [cube("contrast", config.alphas)]),
    ]
    chance = float(
        np.mean(
            [
                1 / len(question.choices)
                for question, flag in zip(questions, test, strict=True)
                if flag
            ]
        )
    )
    band_seed = derive_seed(config.seed, "band")
    for label, grid, cubes in grids:
        write_dose_curves(
            run_dir,
            np.stack([curve(points) for points in cubes], axis=1),
            grid,
            series=SERIES_NAMES,
            stem=f"curves_{label}",
            xlabel="Alpha" if label == "contrast" else "Coefficient",
            ylabel=YLABEL,
            chance=chance,
            n_boot=config.n_boot,
            seed=band_seed,
        )

    def row(
        arm: str,
        coeff: float | None,
        seed: int | None,
        chosen: NDArray[np.bool_],
    ) -> list[object]:
        return _table_row(
            arm,
            coeff,
            seed,
            chosen,
            hits["none"],
            select,
            test,
            n_boot=config.n_boot,
            boot_seed=derive_seed(config.seed, "boot"),
        )

    table = [
        row(name, None, None, hits[name])
        for name in ("none", *dict(INSTRUCTIONS))
    ]
    selected: dict[str, list[float]] = {}
    reading_test: list[float] = []
    for label, grid, cubes in grids:
        selected[label] = []
        for seed, points in enumerate(cubes):
            hits_select = points[:, select].mean(axis=1).tolist()
            index = select_coeff(hits_select, grid)
            selected[label].append(float(grid[index]))
            if label == "reading":
                reading_test.append(float(points[index][test].mean()))
            table.append(
                row(
                    label,
                    float(grid[index]),
                    seed if label == "reading" else None,
                    points[index],
                )
            )
    run_dir.write_arrays(
        "scores",
        values=np.stack(list(rows.values())),
        offsets=offsets,
        correct=correct,
    )
    run_dir.write_json(
        "scores",
        {
            "rows": list(rows),
            "uids": [question.uid for question in questions],
            "test": test.tolist(),
        },
    )
    run_dir.write_json("table", {"columns": TABLE_COLUMNS, "rows": table})
    val = sweep.per_seed("val")[SERIES.index("reader")]
    run_dir.write_json(
        "summary",
        {
            "layers": list(layers),
            "n_select": int(select.sum()),
            "n_test": int(test.sum()),
            "chance": chance,
            "reading_val_accuracy": val.mean(axis=0).tolist(),
            "coeff_difference": selected["difference"][0],
            "coeff_reading": selected["reading"],
            "alpha_contrast": selected["contrast"][0],
            "reading_test_mean": float(np.mean(reading_test)),
            "reading_test_min": float(np.min(reading_test)),
            "reading_test_max": float(np.max(reading_test)),
        },
    )


def run(config: Config, run_dir: RunDir) -> None:
    """Loads the model and the data, then runs `execute`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
    """
    execute(
        config,
        run_dir,
        load_model(config.model),
        load_truthfulqa(seed=config.seed),
        load_alpaca(),
    )


def _plot_arms(run_dir: RunDir, out_dir: Path) -> list[Path]:
    table = cast(dict[str, Any], run_dir.read_json("table"))
    rows = [
        dict(zip(table["columns"], entry, strict=True))
        for entry in table["rows"]
    ]
    labels = [
        row["arm"] if row["seed"] is None else f"{row['arm']} {row['seed']}"
        for row in rows
    ]
    mc1 = np.array([row["test_mc1"] for row in rows])
    below = np.maximum(mc1 - np.array([row["ci_low"] for row in rows]), 0.0)
    above = np.maximum(np.array([row["ci_high"] for row in rows]) - mc1, 0.0)
    with style_context():
        _, axes = new_figure(aspect=0.1 * len(rows) + 0.3)
        ax = axes[0, 0]
        positions = np.arange(len(rows))
        ax.errorbar(
            mc1,
            positions,
            xerr=[below, above],
            fmt="o",
            ms=3,
            color=CATEGORICAL[0],
            elinewidth=PX,
        )
        ax.axvline(mc1[0], color=INK_MUTED, lw=PX, ls="--")
        ax.set_yticks(positions, labels)
        ax.invert_yaxis()
        ax.set_xlabel(f"{YLABEL} (test)")
    return save_figure(figure_of(ax), out_dir, "mc1_arms")


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws the three dose curves and the test MC1 of every table row.

    Args:
        run_dir: A finished C4 run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths of `curves_reading`, `curves_difference`,
        `curves_contrast` and `mc1_arms`.
    """
    paths: list[Path] = []
    for stem in CURVE_STEMS:
        paths.extend(plot_curves_file(run_dir, out_dir, stem=stem))
    paths.extend(_plot_arms(run_dir, out_dir))
    return paths
