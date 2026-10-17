"""C5: the galleries of Figs. 10 and 17.

The lie gallery asks TriviaQA questions under the plain model, under the
honesty reader of C4 steered towards honesty and under the same reader
steered away from it. The emotion galleries read the completions that a
finished C2 run already generated and scored. Every gallery shows rows that
were not chosen for how they look (the first ones) next to rows that were
(the ones that show the effect), so the reader can tell how much the choice
flatters.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace
import math
from pathlib import Path
from typing import Any
from typing import cast

import numpy as np

from tomography.control.controllers import mean_residual_norms
from tomography.datasets.alpaca import load_alpaca
from tomography.datasets.emotions import EMOTIONS
from tomography.datasets.records import Example
from tomography.datasets.records import QAExample
from tomography.datasets.triviaqa import load_triviaqa
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.c4_honesty_control import calibration_texts
from tomography.experiments.c4_honesty_control import check_relative_op
from tomography.experiments.c4_honesty_control import honesty_readers
from tomography.experiments.c4_honesty_control import resolve_layers
from tomography.experiments.effects import SideEffects
from tomography.experiments.effects import read_trials
from tomography.experiments.effects import side_effects
from tomography.experiments.effects import write_trials
from tomography.experiments.steering import Arm
from tomography.experiments.steering import GenSettings
from tomography.experiments.steering import SteeringConfig
from tomography.experiments.steering import Trial
from tomography.experiments.steering import is_empty_completion
from tomography.experiments.steering import plain_arm
from tomography.experiments.steering import reader_arm
from tomography.experiments.steering import run_trial
from tomography.experiments.sweep import chat_format_for
from tomography.knowledge import DEFAULT_SHOTS
from tomography.knowledge import extract_answer
from tomography.knowledge import few_shot_prompt
from tomography.knowledge import is_correct
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.reading.directions import Method
from tomography.reading.metrics import wilson_ci
from tomography.reading.reader import RepReader
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
EFFECT_COLUMNS = [
    "arm",
    "coeff",
    "seed",
    "perplexity",
    "fluency",
    "distinct_1",
    "distinct_2",
    "mean_length",
    "empty_rate",
]
RANDOM = "random"
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
    return replace(trial, scores=scores)


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
    kinds: list[tuple[str, Sequence[int]]] = [
        ("unfiltered", pool[:n_show]),
        ("curated", flips[:n_show]),
    ]
    return [
        _lie_row(kind, arm, trial, prompt)
        for kind, prompts in kinds
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


@dataclass(frozen=True, kw_only=True)
class Config(SteeringConfig):
    """C5 settings: the galleries of Figs. 10 and 17.

    Attributes:
        method: Direction finder of the honesty reader.
        n_pairs: Alpaca rows of the reader's training set.
        n_eval_pairs: Alpaca rows of its validation set; None takes all.
        strength: Relative coefficient of the lie gallery: the honest arm
            steers with `+strength` and the lie arm with `-strength`.
        n_pool: TriviaQA questions answered under the three arms.
        n_show: Questions or completions per gallery and kind.
        n_shots: Demonstrations of the plain-kind prompt.
        c2_run: Path of a finished C2 run, or None for no emotion galleries.
        n_random: Random directions, unit vectors drawn from the seed and
            norm-matched to the steering arm, that steer at `+strength`
            and `-strength` as controls.
    """

    experiment: str = "c5"
    model: str = "gpt2"
    method: Method = "logistic"
    n_pairs: int = 128
    n_eval_pairs: int | None = 128
    strength: float = 0.1
    n_pool: int = 200
    n_show: int = 4
    n_shots: int = len(DEFAULT_SHOTS)
    num_samples: int = 1
    max_new_tokens: int = 24
    c2_run: str | None = None
    n_random: int = 3


def _check_config(
    config: Config, n_facts: int, n_layers: int
) -> tuple[int, ...]:
    """Validates a run before any work and returns the steered layers."""
    if not config.strength > 0:
        raise ValueError(f"strength must be positive, got {config.strength}")
    for name in ("n_random", "n_neutral"):
        if getattr(config, name) < 1:
            raise ValueError(
                f"{name} must be >= 1, got {getattr(config, name)}"
            )
    if config.n_show < 1:
        raise ValueError(f"n_show must be >= 1, got {config.n_show}")
    if not 1 <= config.n_pool <= n_facts:
        raise ValueError(f"n_pool must be in 1..{n_facts}, got {config.n_pool}")
    check_relative_op(config)
    return resolve_layers(config.layers, n_layers)


def _emotion_strength(emotion_run: RunDir) -> float:
    summary = emotion_run.read_json("summary")
    if not isinstance(summary, dict) or "strength" not in summary:
        raise ValueError("the emotion run has no 'strength' in its summary")
    return float(summary["strength"])


def _mean_score(trial: Trial) -> float | None:
    value = float(trial.mean_score())
    return None if math.isnan(value) else value


def _mean_over_arms(trials: Sequence[Trial]) -> float | None:
    """Averages the mean scores of several arms.

    Args:
        trials: The trials of the arms to average.

    Returns:
        The mean of the arm means, or None when every arm mean is missing.
    """
    means = [m for trial in trials if (m := _mean_score(trial)) is not None]
    return float(np.mean(means)) if means else None


def neutral_texts(
    alpaca: Sequence[Example], config: Config, *, seed: int
) -> list[str]:
    """Draws the neutral texts of the perplexity.

    They are the texts that follow the calibration texts in the draw of
    `calibration_texts`, so the two sets are disjoint and the calibration
    texts are the ones a run without controls draws.

    Args:
        alpaca: Alpaca rows with splits.
        config: The experiment settings.
        seed: Seed of the calibration draw.

    Returns:
        Up to `n_neutral` texts.

    Raises:
        ValueError: If the Alpaca `test` split has no text left after the
            calibration texts.
    """
    drawn = calibration_texts(
        alpaca, config.n_calibration + config.n_neutral, seed=seed
    )
    neutral = drawn[config.n_calibration :]
    if not neutral:
        raise ValueError(
            f"the Alpaca test split has no neutral text after the "
            f"{config.n_calibration} calibration texts"
        )
    return neutral


def _random_readers(reader: RepReader, config: Config) -> list[RepReader]:
    """Draws the random control readers.

    The directions are unit vectors drawn in order from one generator seeded
    with `config.seed`, with the layers and shape of `reader`.

    Args:
        reader: The steering reader whose layers and shape to match.
        config: The experiment settings.

    Returns:
        `config.n_random` readers.
    """
    rng = np.random.default_rng(config.seed)
    zeros = np.zeros_like(reader.directions)
    return [
        RepReader.from_arrays(
            RANDOM,
            reader.layers,
            rng.standard_normal(reader.directions.shape),
            zeros,
        )
        for _ in range(config.n_random)
    ]


def _effect_row(
    arm: str,
    coeff: float | None,
    seed: int | None,
    effects: SideEffects,
) -> list[object]:
    """Builds one row of the effects table.

    Args:
        arm: The arm family: "none", "reading" or "random".
        coeff: The relative coefficient, or None for the plain model.
        seed: The random direction index k for a random arm, 0 for the
            reading arm, None for the plain model.
        effects: The side effects of the arm.

    Returns:
        The row in the order of `EFFECT_COLUMNS`.
    """
    return [
        arm,
        coeff,
        seed,
        effects.perplexity,
        effects.fluency,
        effects.distinct_1,
        effects.distinct_2,
        effects.mean_length,
        effects.empty_rate,
    ]


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    alpaca: Sequence[Example],
    facts: Sequence[QAExample],
    emotion_run: RunDir | None = None,
) -> None:
    """Builds the lie gallery and reads the emotion galleries.

    The honesty reader is the reader 0 of C4 for the same seed and recipe.
    The pool is a seeded draw of the questions, answered by the plain model,
    by the reader steered towards honesty and by the reader steered away
    from it, all three with the same sampling seed. A flip is a question that
    the plain model answers correctly and the lie arm answers wrongly, both
    at sample 0.

    The emotion galleries need no model: they read the completions of a
    finished C2 run at the strength of that run.

    The random arms are `n_random` unit directions drawn from the seed and
    norm-matched to the steering arm: same layers, strengths, operator, mode,
    mask and norm handling. Their questions are the pool's and their trials
    stay out of the lie gallery.

    Writes `trials` (`none`, `reading/0/<s>`, `reading/0/<-s>`),
    `trials_random` (`random/<k>/<s>` and `random/<k>/<-s>` for each random
    direction `k`), `table.json`, `table_effects.json` (perplexity of
    neutral held-out text, unsteered completion fluency, distinct-n, length
    and empty rate of every arm) and `summary.json`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model.
        alpaca: Alpaca rows with splits.
        facts: TriviaQA questions with gold aliases.
        emotion_run: A finished C2 run, or None for no emotion galleries.

    Raises:
        ValueError: If `strength` is not positive, `n_show` is below 1,
            `n_pool` is outside `1..len(facts)`, `n_shots` is outside
            `0..15`, the layers are outside `1..L`, `n_calibration`,
            `n_neutral`, `n_random`, `num_samples`, `max_new_tokens` or
            `batch_size` is below 1, the projection operator has relative
            coefficients, the Alpaca `test` split is empty or has no text
            left after the calibration texts, or the emotion run is not
            usable.
    """
    layers = _check_config(config, len(facts), loaded.n_layers)
    settings = GenSettings.from_config(config)
    texts = calibration_texts(
        alpaca, config.n_calibration, seed=derive_seed(config.seed, "norms")
    )
    neutral = neutral_texts(
        alpaca, config, seed=derive_seed(config.seed, "norms")
    )
    chat = chat_format_for(loaded)
    order = np.random.default_rng(derive_seed(config.seed, "pool")).permutation(
        len(facts)
    )
    pool = [facts[i] for i in order[: config.n_pool]]
    prompts = [fact_prompt(chat, q.question, config.n_shots) for q in pool]
    answers = [q.answers for q in pool]
    emotion_table: list[list[object]] = []
    emotion_strength = None
    if emotion_run is not None:
        emotion_strength = _emotion_strength(emotion_run)
        emotion_table = emotion_rows(
            read_trials(emotion_run, "trials"),
            emotion_strength,
            config.n_show,
        )
    sweep = honesty_readers(
        loaded,
        alpaca,
        layers,
        n_seeds=1,
        n_pairs=config.n_pairs,
        n_eval_pairs=config.n_eval_pairs,
        method=config.method,
        seed=config.seed,
        bos=config.bos,
        batch_size=config.batch_size,
    )
    norms = mean_residual_norms(
        loaded, texts, layers, bos=config.bos, batch_size=config.batch_size
    )

    def steering_arm(
        reader: RepReader, label: str, k: int, coeff: float
    ) -> Arm:
        return reader_arm(
            loaded,
            settings,
            reader,
            layers=layers,
            coeff=coeff,
            norms=norms,
            op=config.op,
            mode=config.mode,
            mask=config.mask,
            preserve_norm=config.preserve_norm,
            name=f"{label}/{k}/{coeff:g}",
        )

    strengths = (config.strength, -config.strength)
    arms = [plain_arm(loaded, settings)] + [
        steering_arm(sweep.readers[0], "reading", 0, coeff)
        for coeff in strengths
    ]
    random_arms = [
        steering_arm(reader, RANDOM, k, coeff)
        for k, reader in enumerate(_random_readers(sweep.readers[0], config))
        for coeff in strengths
    ]
    none, honest, lie = (fact_trial(arm, prompts, answers) for arm in arms)
    random_trials = [fact_trial(arm, prompts, answers) for arm in random_arms]
    write_trials(run_dir, "trials", [none, honest, lie])
    write_trials(run_dir, "trials_random", random_trials)
    labels: list[tuple[str, float | None, int | None]] = [
        ("none", None, None),
        ("reading", strengths[0], 0),
        ("reading", strengths[1], 0),
        *(
            (RANDOM, coeff, k)
            for k in range(config.n_random)
            for coeff in strengths
        ),
    ]
    effect_rows = [
        _effect_row(
            label,
            coeff,
            seed,
            side_effects(
                loaded, arm, trial, neutral, batch_size=config.batch_size
            ),
        )
        for (label, coeff, seed), arm, trial in zip(
            labels,
            [*arms, *random_arms],
            [none, honest, lie, *random_trials],
            strict=True,
        )
    ]
    run_dir.write_json(
        "table_effects", {"columns": EFFECT_COLUMNS, "rows": effect_rows}
    )
    rows = lie_rows(none, honest, lie, config.n_show) + emotion_table
    run_dir.write_json("table", {"columns": TABLE_COLUMNS, "rows": rows})
    known = none.scores[:, 0] == 1.0
    n_known = int(known.sum())
    n_flipped = int((known & (lie.scores[:, 0] == 0.0)).sum())
    run_dir.write_json(
        "summary",
        {
            "layers": list(layers),
            "strength": config.strength,
            "n_pool": config.n_pool,
            "correct_none": _mean_score(none),
            "correct_honest": _mean_score(honest),
            "correct_lie": _mean_score(lie),
            "n_random": config.n_random,
            "correct_random_plus": _mean_over_arms(random_trials[0::2]),
            "correct_random_minus": _mean_over_arms(random_trials[1::2]),
            "n_known": n_known,
            "n_flipped": n_flipped,
            "flip_rate": n_flipped / n_known if n_known else None,
            "flip_ci": list(wilson_ci(n_flipped, n_known)) if n_known else None,
            "emotions": list(
                dict.fromkeys(
                    str(row[0]).removeprefix("emotion:")
                    for row in emotion_table
                )
            ),
            "emotion_strength": emotion_strength,
            "c2_run": config.c2_run,
        },
    )


def run(config: Config, run_dir: RunDir) -> None:
    """Loads the model, the data and the emotion run, then runs `execute`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
    """
    execute(
        config,
        run_dir,
        load_model(config.model),
        load_alpaca(),
        load_triviaqa(),
        None if config.c2_run is None else RunDir.open(config.c2_run),
    )


def _without_demonstrations(prompt: str) -> str:
    _, separator, question = prompt.rpartition("\n\nQ: ")
    return f"Q: {question}" if separator else prompt


def _cell(value: object, column: str) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3g}"
    text = str(value)
    if column == "prompt":
        text = _without_demonstrations(text)
    return " ".join(text.split()).replace("|", "\\|")


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Writes the galleries as one markdown file.

    A gallery is text, so the file holds a section per gallery and a table
    per section, not a figure. The few-shot demonstrations are cut from the
    prompts, a pipe is escaped and a line break becomes a space. A column
    that is empty in a whole gallery is left out.

    Args:
        run_dir: A finished C5 run.
        out_dir: Directory of the output.

    Returns:
        The path of `gallery.md`.
    """
    table = cast(dict[str, Any], run_dir.read_json("table"))
    columns: list[str] = table["columns"]
    galleries: dict[str, list[list[object]]] = {}
    for row in table["rows"]:
        galleries.setdefault(row[0], []).append(row)
    lines = ["# Galleries", ""]
    for gallery, rows in galleries.items():
        shown = [
            i
            for i in range(1, len(columns))
            if any(row[i] is not None for row in rows)
        ]
        lines += [f"## {gallery}", ""]
        lines.append("| " + " | ".join(columns[i] for i in shown) + " |")
        lines.append("| " + " | ".join("---" for _ in shown) + " |")
        lines += [
            "| " + " | ".join(_cell(row[i], columns[i]) for i in shown) + " |"
            for row in rows
        ]
        lines.append("")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "gallery.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return [path]
