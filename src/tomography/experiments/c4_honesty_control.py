"""C4: the honesty control of Table 2 on TruthfulQA MC1.

The arms are scored, not generated: every arm is a log-probability function
applied to every choice of every question, with the App. C.2 format. This
module holds the helpers of that experiment: the layers, the prompt rule, the
MC1 bookkeeping and the readers of the vector arms.
"""

from collections.abc import Sequence
import itertools
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tomography.activations import collect_activations
from tomography.datasets.records import ChoiceExample
from tomography.datasets.records import Example
from tomography.datasets.stimulus import StimulusSet
from tomography.datasets.stimulus import build_stimulus_set
from tomography.experiments.base import derive_seed
from tomography.experiments.sweep import SweepResult
from tomography.experiments.sweep import chat_format_for
from tomography.experiments.sweep import reading_sweep
from tomography.model import BosMode
from tomography.model import LoadedModel
from tomography.reading.directions import Method
from tomography.reading.reader import RepReader
from tomography.templates import PAPER_TEMPLATES
from tomography.templates import ChatFormat

TRUTHFUL = "Give a truthful answer."
UNTRUTHFUL = "Give an untruthful answer."
BLOCK_IDS = range(8, 32, 3)
PAPER_DEPTH = 32


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
