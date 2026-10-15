"""C5: the galleries of Figs. 10 and 17.

The lie gallery asks TriviaQA questions under the plain model, under the
honesty reader steered towards honesty and under the same reader steered
away from it. The emotion galleries read the completions that a finished C2
run already generated and scored. Every gallery shows rows that were not
chosen for how they look (the first ones) next to rows that were (the ones
that show the effect), so the reader can tell how much the choice flatters.
"""

from collections.abc import Sequence
import dataclasses
import math

import numpy as np

from tomography.datasets.emotions import EMOTIONS
from tomography.experiments.steering import Arm
from tomography.experiments.steering import Trial
from tomography.experiments.steering import is_empty_completion
from tomography.experiments.steering import run_trial
from tomography.knowledge import DEFAULT_SHOTS
from tomography.knowledge import extract_answer
from tomography.knowledge import few_shot_prompt
from tomography.knowledge import is_correct
from tomography.templates import ChatFormat

TABLE_COLUMNS = [
    "gallery",
    "kind",
    "arm",
    "prompt_index",
    "sample",
    "prompt",
    "completion",
    "correct",
    "score",
    "score_check",
]
CHECK_OFFSET = len(EMOTIONS)
N_JUDGE_COLUMNS = 2 * len(EMOTIONS)


def fact_prompt(
    chat: ChatFormat, question: str, n_shots: int = len(DEFAULT_SHOTS)
) -> str:
    """Builds the prompt of one world-fact question.

    The plain kind has no chat tags, so the question follows the first
    `n_shots` demonstrations of the closed-book prompt of the knowledge
    study. The template kind is asked as one chat turn.

    Args:
        chat: The chat format of the model.
        question: The question.
        n_shots: Demonstrations of the plain kind; the template kind ignores
            them.

    Returns:
        The prompt.

    Raises:
        ValueError: If `n_shots` is outside `0..len(DEFAULT_SHOTS)`.
    """
    if not 0 <= n_shots <= len(DEFAULT_SHOTS):
        raise ValueError(
            f"n_shots must be in 0..{len(DEFAULT_SHOTS)}, got {n_shots}"
        )
    if chat.kind == "plain":
        return few_shot_prompt(question, DEFAULT_SHOTS[:n_shots])
    return chat.render_turn(question)[0]


def fact_trial(
    arm: Arm, prompts: Sequence[str], answers: Sequence[Sequence[str]]
) -> Trial:
    """Generates with an arm and checks every completion against the gold.

    `run_trial` generates and drops the empty completions; the aliases differ
    per prompt, so the scores it would compute are replaced by the check of
    the answer, the completion up to its first newline.

    Args:
        arm: The arm to run.
        prompts: The prompts.
        answers: The gold aliases of each prompt.

    Returns:
        The trial, whose scores of shape `(P, S)` are 1.0 for a correct
        answer, 0.0 for a wrong one and NaN for an empty completion.

    Raises:
        ValueError: If `answers` and `prompts` differ in length.
    """
    if len(answers) != len(prompts):
        raise ValueError(
            f"got {len(answers)} answers for {len(prompts)} prompts"
        )
    trial = run_trial(
        arm, prompts, lambda texts: np.zeros(len(texts), dtype=np.float32)
    )
    scores = np.array(
        [
            [
                np.nan
                if is_empty_completion(text, int(length))
                else float(is_correct(extract_answer(text), aliases))
                for text, length in zip(texts, lengths, strict=True)
            ]
            for texts, lengths, aliases in zip(
                trial.completions, trial.lengths, answers, strict=True
            )
        ],
        dtype=np.float32,
    )
    return dataclasses.replace(trial, scores=scores)


def _verdict(score: float) -> bool | None:
    return None if math.isnan(score) else score == 1.0


def _lie_row(kind: str, arm: str, trial: Trial, prompt: int) -> list[object]:
    return [
        "lie",
        kind,
        arm,
        prompt,
        0,
        trial.prompts[prompt],
        trial.completions[prompt][0],
        _verdict(float(trial.scores[prompt, 0])),
        None,
        None,
    ]


def lie_rows(
    none: Trial, honest: Trial, lie: Trial, n_show: int
) -> list[list[object]]:
    """Picks the rows of the lie gallery.

    The unfiltered rows are the first `n_show` questions of the pool, whatever
    the arms answered. The curated rows are the first `n_show` questions that
    the plain model answers correctly and the lie arm answers wrongly, the
    flips the gallery is about; a missing verdict is neither. Each question
    contributes the `none`, `honest` and `lie` completions of sample 0.

    Args:
        none: The plain model on the pool.
        honest: The reader steered towards honesty.
        lie: The reader steered away from honesty.
        n_show: Questions per kind.

    Returns:
        Rows in the order of `TABLE_COLUMNS`, unfiltered rows first.

    Raises:
        ValueError: If the trials do not share their prompts.
    """
    if not none.prompts == honest.prompts == lie.prompts:
        raise ValueError("the trials do not share their prompts")
    arms = (("none", none), ("honest", honest), ("lie", lie))
    pool = range(len(none.prompts))
    flips = [
        prompt
        for prompt in pool
        if none.scores[prompt, 0] == 1.0 and lie.scores[prompt, 0] == 0.0
    ]
    return [
        _lie_row(kind, arm, trial, prompt)
        for kind, prompts in (
            ("unfiltered", pool[:n_show]),
            ("curated", flips[:n_show]),
        )
        for prompt in prompts
        for arm, trial in arms
    ]


def _score(trial: Trial, prompt: int, sample: int, column: int) -> float | None:
    value = float(trial.scores[prompt, sample, column])
    return None if math.isnan(value) else value


def _emotion_row(
    emotion: str, kind: str, trial: Trial, prompt: int, sample: int
) -> list[object]:
    column = EMOTIONS.index(emotion)
    return [
        f"emotion:{emotion}",
        kind,
        trial.name,
        prompt,
        sample,
        trial.prompts[prompt],
        trial.completions[prompt][sample],
        None,
        _score(trial, prompt, sample, column),
        _score(trial, prompt, sample, CHECK_OFFSET + column),
    ]


def _steered_name(name: str) -> tuple[str, int, str] | None:
    parts = name.split("/")
    if len(parts) != 3 or parts[0] not in EMOTIONS or not parts[1].isdigit():
        return None
    return parts[0], int(parts[1]), parts[2]


def emotion_rows(
    trials: Sequence[Trial], strength: float, n_show: int
) -> list[list[object]]:
    """Picks the rows of the emotion galleries from a finished C2 run.

    Each emotion with steered trials has one gallery, in the order of
    `EMOTIONS`. Its unfiltered rows are the first `n_show` prompts, sample 0,
    of the `none` trial and of the first reader at `strength`. Its curated
    rows are the `n_show` completions with the highest score of that emotion
    over the trials of every reader at `strength`, every prompt and every
    sample, ties by reader, prompt and sample; a missing score is never
    picked. Every row carries the primary score and the check score of the
    emotion.

    Args:
        trials: The trials of the C2 run: `none`, then one trial named
            `<emotion>/<seed>/<coeff>` per steered emotion, reader and
            coefficient, with scores of shape `(P, S, 12)`.
        strength: The coefficient of the galleries.
        n_show: Rows per kind and emotion.

    Returns:
        Rows in the order of `TABLE_COLUMNS`.

    Raises:
        ValueError: If there is no `none` trial, a trial does not have
            scores of shape `(P, S, 12)`, or an emotion has no trial of the
            first reader at `strength`.
    """
    none = next((trial for trial in trials if trial.name == "none"), None)
    if none is None:
        raise ValueError("the trials have no 'none' trial")
    steered = [
        (name, trial)
        for trial in trials
        if (name := _steered_name(trial.name)) is not None
    ]
    for trial in (none, *(trial for _, trial in steered)):
        if trial.scores.ndim != 3 or trial.scores.shape[2] != N_JUDGE_COLUMNS:
            raise ValueError(
                f"trial {trial.name!r} has scores of shape "
                f"{trial.scores.shape}, expected (P, S, {N_JUDGE_COLUMNS})"
            )
    label = f"{strength:g}"
    rows: list[list[object]] = []
    for emotion in EMOTIONS:
        if not any(name[0] == emotion for name, _ in steered):
            continue
        readers = {
            name[1]: trial
            for name, trial in steered
            if name[0] == emotion and name[2] == label
        }
        if 0 not in readers:
            raise ValueError(f"the trials have no {emotion}/0/{label} trial")
        column = EMOTIONS.index(emotion)
        first = readers[0]
        for prompt in range(min(n_show, len(first.prompts))):
            rows.append(_emotion_row(emotion, "unfiltered", none, prompt, 0))
            rows.append(_emotion_row(emotion, "unfiltered", first, prompt, 0))
        ranked = sorted(
            (-score, seed, prompt, sample)
            for seed, trial in readers.items()
            for prompt in range(len(trial.prompts))
            for sample in range(trial.scores.shape[1])
            if (score := _score(trial, prompt, sample, column)) is not None
        )
        rows.extend(
            _emotion_row(emotion, "curated", readers[seed], prompt, sample)
            for _, seed, prompt, sample in ranked[:n_show]
        )
    return rows
