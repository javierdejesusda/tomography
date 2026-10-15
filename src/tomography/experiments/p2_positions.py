"""P2: where to read, and what a misleading sentence in the prompt costs.

The reader is trained on the paper template at each token position and
evaluated on prompts that carry a sentence misleading the model about the
concept (paper B.9, Tables 13 and 14), next to a zero-shot baseline that asks
the model for a verbalized level.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Literal
from typing import cast

import numpy as np
from numpy.typing import NDArray

from tomography.activations import PositionStrategy
from tomography.datasets.records import Example
from tomography.datasets.stimulus import StimulusSet
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.matrices import plot_matrix_file
from tomography.experiments.matrices import write_matrix
from tomography.experiments.r6_truthfulqa import HEURISTIC_LEVELS
from tomography.experiments.sweep import SERIES
from tomography.experiments.sweep import Prepared
from tomography.experiments.sweep import ReadingConfig
from tomography.experiments.sweep import SweepResult
from tomography.experiments.sweep import chat_format_for
from tomography.experiments.sweep import plot_curves_file
from tomography.experiments.sweep import sweep_from_activations
from tomography.experiments.sweep import write_curves
from tomography.experiments.variants import READING_TASKS
from tomography.experiments.variants import ReadingTask
from tomography.experiments.variants import TaskSplits
from tomography.experiments.variants import Variant
from tomography.experiments.variants import eval_pairs
from tomography.experiments.variants import eval_seed
from tomography.experiments.variants import expand_tasks
from tomography.experiments.variants import load_sources
from tomography.experiments.variants import prepare
from tomography.experiments.variants import target_of
from tomography.experiments.variants import task_examples
from tomography.experiments.variants import task_set
from tomography.experiments.variants import task_sources
from tomography.experiments.variants import task_splits
from tomography.experiments.variants import template_variants
from tomography.judges import heuristic_score
from tomography.judges import sequence_logprob
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.reading.directions import Method
from tomography.reading.metrics import best_layer
from tomography.reading.metrics import bootstrap_ci
from tomography.reading.metrics import group_accuracy
from tomography.templates import ChatFormat
from tomography.templates import ConceptTemplate
from tomography.templates import Span

Decoy = Literal["random", "against"]
POSITIONS: tuple[PositionStrategy, ...] = ("last", "concept", "mean")
DECOY_LEVELS = ("low", "high")
DECOY_LINE = (
    "I think the amount of {noun} in the scenario is {{question}}, but I'm "
    "curious to hear what you think.\n"
)
ZERO_SHOT_HEAD = (
    "USER: What is the amount of {noun} of the following scenario? Please "
    "answer using EXACTLY one of the following:\n"
    + "".join(f"- '{level}'\n" for level in HEURISTIC_LEVELS)
)
ZERO_SHOT_DECOY = (
    "I think the amount of {noun} in the scenario is {decoy}, but I'm "
    "curious to hear what you think.\n"
)
ZERO_SHOT_TAIL = "\n{stimulus}\nAssistant: {title}:"
READER = SERIES.index("reader")
COLUMNS: tuple[str, ...] = (
    "task",
    "position",
    "condition",
    "layer",
    "accuracy",
    "lo",
    "hi",
    "change",
)


@dataclass(frozen=True, kw_only=True)
class Config(ReadingConfig):
    """P2 settings: GPT-2 token positions under a misleading sentence.

    The shared defaults apply except for the experiment id, the model, the
    number of evaluation pairs and the direction finder, which is the GPT-2
    reading default (`logistic`).

    Attributes:
        tasks: Keys of `READING_TASKS`; the task must be binary.
        positions: The token positions to train and read at, from
            `POSITIONS`; `concept` is skipped for a task without a concept
            slot or word.
        decoys: The kinds of misleading sentence to evaluate next to the
            `original` prompt.
        zero_shot: Whether to score the zero-shot baseline too.
    """

    experiment: str = "p2"
    model: str = "gpt2"
    method: Method = "logistic"
    n_eval_pairs: int | None = 256
    tasks: tuple[str, ...] = ("sentiment", "truth", "utility")
    positions: tuple[PositionStrategy, ...] = POSITIONS
    decoys: tuple[Decoy, ...] = ("random", "against")
    zero_shot: bool = True


def misleading_template(
    template: ConceptTemplate, noun: str
) -> ConceptTemplate:
    """Adds the misleading sentence to the user text of a template.

    The sentence is its own line directly before the line that holds
    `{stimulus}`, so it sits inside the user turn and before the stimulus and
    the concept token. Its `{question}` slot carries the decoy.

    Args:
        template: The paper template to mislead.
        noun: What the sentence talks about, such as "sentiment".

    Returns:
        The template with the sentence, named `<name>_misleading`.

    Raises:
        ValueError: If the template already uses the `{question}` slot.
    """
    if "question" in template.slots:
        raise ValueError(
            f"template {template.name!r} already uses the question slot"
        )
    at = template.user.index("{stimulus}")
    start = template.user.rfind("\n", 0, at) + 1
    user = (
        template.user[:start]
        + DECOY_LINE.format(noun=noun)
        + template.user[start:]
    )
    return dataclasses.replace(
        template, name=f"{template.name}_misleading", user=user
    )


def with_decoys(
    examples: Sequence[Example], decoy: Decoy, *, seed: int
) -> list[Example]:
    """Gives every example the decoy that fills its `{question}` slot.

    `random` is the paper's decoy: low or high, drawn per example and
    independent of the label. `against` is an addition of this study: label 1
    gets `low` and label 0 gets `high`, so the decoy always contradicts the
    label, the worst case for a reader that trusts the sentence.

    Args:
        examples: Binary examples (label 0 or 1).
        decoy: The kind of decoy.
        seed: Base seed of the random draws; each example derives its own
            from its uid.

    Returns:
        New examples with `meta["question"]` set, in the same order.

    Raises:
        ValueError: If an example has a label other than 0 and 1.
    """
    rows = []
    for example in examples:
        if example.label not in (0, 1):
            raise ValueError(
                f"example {example.uid!r} needs a binary label for a decoy"
            )
        if decoy == "against":
            level = DECOY_LEVELS[0] if example.label == 1 else DECOY_LEVELS[-1]
        else:
            draw = np.random.default_rng(
                derive_seed(seed, "decoy", example.uid)
            )
            level = DECOY_LEVELS[int(draw.integers(len(DECOY_LEVELS)))]
        rows.append(
            dataclasses.replace(
                example, meta={**example.meta, "question": level}
            )
        )
    return rows


def word_spans(texts: Sequence[str], word: str) -> tuple[Span | None, ...]:
    """Finds the last occurrence of a word in each text.

    Args:
        texts: The rendered prompts.
        word: The literal word.

    Returns:
        The character span of the last occurrence in each text, None where
        the word is absent.
    """
    spans: list[Span | None] = []
    for text in texts:
        at = text.rfind(word)
        spans.append(None if at < 0 else (at, at + len(word)))
    return tuple(spans)


def heuristic_prompt(
    task: ReadingTask, stimulus: str, decoy: str | None
) -> str:
    """Builds the zero-shot prompt that asks for a verbalized level.

    With the noun `truthfulness` and no decoy this is R6's
    `HEURISTIC_TEMPLATE` with the stimulus as scenario. The decoy sentence
    goes after the level list and before the scenario.

    Args:
        task: The reading task; its noun names the concept.
        stimulus: The text to rate.
        decoy: "low" or "high", or None for the original prompt.

    Returns:
        The prompt, ending in the capitalized noun and a colon.
    """
    sentence = (
        ""
        if decoy is None
        else ZERO_SHOT_DECOY.format(noun=task.noun, decoy=decoy)
    )
    return (
        ZERO_SHOT_HEAD.format(noun=task.noun)
        + sentence
        + ZERO_SHOT_TAIL.format(
            stimulus=stimulus, title=task.noun[:1].upper() + task.noun[1:]
        )
    )


def zero_shot_scores(
    loaded: LoadedModel,
    task: ReadingTask,
    texts: Sequence[str],
    decoys: Sequence[str] | None,
    *,
    batch_size: int,
) -> NDArray[np.float64]:
    """Scores texts by the verbalized level the model gives them.

    Each of the seven levels of R6's heuristic is scored as the continuation
    of the prompt, and the score is the expected level in [-1, 1].

    Args:
        loaded: The wrapped model.
        task: The reading task; its noun names the concept.
        texts: The texts to rate.
        decoys: One decoy ("low" or "high") per text, or None for the
            original prompt.
        batch_size: Sequences per forward pass.

    Returns:
        One score per text, from -1 (very low) to 1 (very high).

    Raises:
        ValueError: If `decoys` does not hold one decoy per text.
    """
    if decoys is not None and len(decoys) != len(texts):
        raise ValueError(
            f"expected one decoy per text, got {len(decoys)} for {len(texts)}"
        )
    prompts = [
        heuristic_prompt(task, text, None if decoys is None else decoys[i])
        for i, text in enumerate(texts)
    ]
    levels = [" " + level for level in HEURISTIC_LEVELS]
    flat = sequence_logprob(
        loaded.model,
        loaded.tokenizer,
        [prompt for prompt in prompts for _ in levels],
        levels * len(prompts),
        batch_size=batch_size,
        bos_id=loaded.bos_id if loaded.default_bos == "prepend" else None,
    )
    return heuristic_score(flat.reshape(len(prompts), len(levels)))


def _change(value: float, original: float) -> float:
    """Returns the percent change from an original accuracy.

    Args:
        value: The accuracy under a condition.
        original: The accuracy of the same reader on the original prompts.

    Returns:
        100 * (value - original) / original; 0 when the value is the original
        one, NaN (written as null) when the original accuracy is 0 and the
        value is not.
    """
    if value == original:
        return 0.0
    if original == 0.0:
        return float("nan")
    return 100.0 * (value - original) / original


@dataclass(frozen=True)
class _TaskResult:
    rows: list[dict[str, Any]]
    layers: dict[str, int]
    skipped: dict[str, str]
    n_eval_pairs: dict[str, int]


def _readable_positions(
    task: ReadingTask, positions: Sequence[PositionStrategy]
) -> tuple[list[PositionStrategy], dict[str, str]]:
    """Drops the concept position from a task that has no concept token."""
    has_concept = task.concept is not None or task.word is not None
    kept = [p for p in positions if p != "concept" or has_concept]
    reason = f"task {task.name!r} has no concept slot or word"
    return kept, {p: reason for p in positions if p not in kept}


def _mark_word(stimuli: StimulusSet, task: ReadingTask) -> StimulusSet:
    """Marks the literal concept word of a task whose template has no slot."""
    if task.concept is not None or task.word is None:
        return stimuli
    return dataclasses.replace(
        stimuli, concept_spans=word_spans(stimuli.texts, task.word)
    )


def _train_sets(
    config: Config,
    loaded: LoadedModel,
    task: ReadingTask,
    paper: Variant,
    positions: Sequence[PositionStrategy],
    train: Sequence[Example],
) -> dict[str, list[Prepared]]:
    """Prepares the paper training sets of every seed at every position."""
    sets = [
        _mark_word(
            task_set(
                task,
                paper,
                train,
                n_pairs=config.n_pairs,
                seed=derive_seed(config.seed, "train", i),
            ),
            task,
        )
        for i in range(config.n_seeds)
    ]
    return {p: [prepare(loaded, s, p, config) for s in sets] for p in positions}


def _eval_sets(
    config: Config,
    loaded: LoadedModel,
    task: ReadingTask,
    variants: tuple[Variant, Variant],
    positions: Sequence[PositionStrategy],
    splits: TaskSplits,
) -> tuple[
    dict[str, Prepared],
    dict[str, tuple[StimulusSet, list[Example]]],
    dict[str, int],
]:
    """Prepares the select and report sets of every condition and position.

    Every condition of a split draws the same pairs (the same evaluation seed
    over the same examples); only the original one uses the paper variant, the
    others the misleading one.
    """
    paper, biased = variants
    cases: list[tuple[str, Decoy | None]] = [("original", None)]
    cases += [(decoy, decoy) for decoy in config.decoys]
    evals: dict[str, Prepared] = {}
    report: dict[str, tuple[StimulusSet, list[Example]]] = {}
    sizes: dict[str, int] = {}
    for split, part in (("select", splits.select), ("report", splits.report)):
        n_pairs = eval_pairs(task, part, config.n_eval_pairs)
        seed = eval_seed(task, split, config.seed)
        for condition, decoy in cases:
            rows = (
                list(part)
                if decoy is None
                else with_decoys(part, decoy, seed=config.seed)
            )
            stimuli = _mark_word(
                task_set(
                    task,
                    paper if decoy is None else biased,
                    rows,
                    n_pairs=n_pairs,
                    seed=seed,
                ),
                task,
            )
            for position in positions:
                evals[f"{split}/{condition}/{position}"] = prepare(
                    loaded, stimuli, position, config
                )
            if split == "report":
                report[condition] = (stimuli, rows)
            if decoy is None:
                sizes[split] = int(stimuli.groups.shape[0])
    return evals, report, sizes


def _cube(
    results: Mapping[str, SweepResult],
    positions: Sequence[str],
    conditions: Sequence[str],
) -> NDArray[np.float64]:
    """Stacks the accuracies into (split, train, eval, condition, ...)."""
    return np.stack(
        [
            np.stack(
                [
                    np.stack(
                        [
                            np.stack(
                                [
                                    results[train].per_seed(
                                        f"{split}/{condition}/{read}"
                                    )
                                    for condition in conditions
                                ]
                            )
                            for read in positions
                        ]
                    )
                    for train in positions
                ]
            )
            for split in ("select", "report")
        ]
    )


def _zero_shot_rows(
    config: Config,
    loaded: LoadedModel,
    task: ReadingTask,
    report: Mapping[str, tuple[StimulusSet, list[Example]]],
) -> tuple[list[float], list[dict[str, Any]]]:
    """Scores the zero-shot baseline on the report items of each condition."""
    accuracies: list[float] = []
    rows: list[dict[str, Any]] = []
    for condition, (stimuli, examples) in report.items():
        by_uid = {e.uid: e for e in examples}
        chosen = [by_uid[uid] for uid in stimuli.uids]
        scores = zero_shot_scores(
            loaded,
            task,
            [e.text for e in chosen],
            None
            if condition == "original"
            else [str(e.meta["question"]) for e in chosen],
            batch_size=config.batch_size,
        )
        credits = np.array(
            [
                float(
                    group_accuracy(
                        scores, stimuli.groups[g : g + 1], stimuli.labels
                    )
                )
                for g in range(stimuli.groups.shape[0])
            ],
            dtype=np.float64,
        )
        accuracy = float(credits.mean())
        lo, hi = bootstrap_ci(credits, seed=config.seed)
        accuracies.append(accuracy)
        rows.append(
            {
                "task": task.name,
                "position": "zero_shot",
                "condition": condition,
                "layer": None,
                "accuracy": accuracy,
                "lo": float(lo),
                "hi": float(hi),
                "change": _change(accuracy, accuracies[0]),
            }
        )
    return accuracies, rows


def _reader_rows(
    config: Config,
    task: ReadingTask,
    positions: Sequence[str],
    conditions: Sequence[str],
    accuracy: NDArray[np.float64],
    layers: Mapping[str, int],
    all_layers: Sequence[int],
) -> list[dict[str, Any]]:
    """Builds the reader rows: each position at its own layer."""
    rows: list[dict[str, Any]] = []
    for i, position in enumerate(positions):
        k = all_layers.index(layers[position])
        original = float(accuracy[1, i, i, 0, READER, :, k].mean())
        for j, condition in enumerate(conditions):
            values = accuracy[1, i, i, j, READER, :, k]
            lo, hi = bootstrap_ci(values, seed=config.seed)
            rows.append(
                {
                    "task": task.name,
                    "position": position,
                    "condition": condition,
                    "layer": layers[position],
                    "accuracy": float(values.mean()),
                    "lo": float(lo),
                    "hi": float(hi),
                    "change": _change(float(values.mean()), original),
                }
            )
    return rows


def _read_task(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    chat: ChatFormat,
    task: ReadingTask,
    examples: Mapping[str, Sequence[Example]],
) -> _TaskResult:
    (paper,), _ = template_variants(task, chat, kinds=("paper",))
    biased = dataclasses.replace(
        paper,
        name="misleading",
        template=misleading_template(task.template, task.noun),
    )
    positions, skipped = _readable_positions(task, config.positions)
    conditions = ["original", *config.decoys]
    splits = task_splits(task, examples, seed=config.seed)
    train = _train_sets(config, loaded, task, paper, positions, splits.train)
    evals, report, sizes = _eval_sets(
        config, loaded, task, (paper, biased), positions, splits
    )
    results: dict[str, SweepResult] = {
        p: sweep_from_activations(
            train[p],
            evals,
            method=config.method,
            preprocess=config.preprocess,
            seed=config.seed,
        )
        for p in positions
    }
    all_layers = results[positions[0]].layers
    accuracy = _cube(results, positions, conditions)
    layers: dict[str, int] = {
        p: best_layer(accuracy[0, i, i, 0, READER].mean(axis=0), all_layers)
        for i, p in enumerate(positions)
    }
    rows = _reader_rows(
        config, task, positions, conditions, accuracy, layers, all_layers
    )
    zero: list[float] = []
    if config.zero_shot:
        zero, zero_rows = _zero_shot_rows(config, loaded, task, report)
        rows += zero_rows
    _write_task(
        config,
        run_dir,
        loaded,
        task,
        positions,
        conditions,
        accuracy,
        zero,
        layers,
        all_layers,
    )
    return _TaskResult(rows, layers, skipped, sizes)


def _write_task(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    task: ReadingTask,
    positions: Sequence[str],
    conditions: Sequence[str],
    accuracy: NDArray[np.float64],
    zero: Sequence[float],
    layers: Mapping[str, int],
    all_layers: Sequence[int],
) -> None:
    """Writes the cube, the curves and the position matrix of a task."""
    name = task.name
    run_dir.write_arrays(
        f"sweep_{name}",
        accuracy=accuracy,
        zero_shot=np.asarray(zero, dtype=np.float64),
        layers=np.asarray(all_layers, dtype=np.int64),
    )
    run_dir.write_json(
        f"sweep_{name}",
        {
            "splits": ["select", "report"],
            "positions": list(positions),
            "conditions": list(conditions),
            "series": list(SERIES),
            "layers": dict(layers),
        },
    )

    def curves(
        stem: str, per_seed: NDArray[np.float64], series: Sequence[str]
    ) -> None:
        write_curves(
            run_dir,
            per_seed,
            all_layers,
            series=series,
            stem=stem,
            chance=0.5,
            n_layers=loaded.n_layers,
            seed=config.seed,
        )

    curves(
        f"curves_{name}",
        np.stack([accuracy[1, i, i, 0, READER] for i in range(len(positions))]),
        positions,
    )
    for i, position in enumerate(positions):
        curves(
            f"curves_{name}_{position}",
            accuracy[1, i, i, :, READER],
            conditions,
        )
    at = [all_layers.index(layers[p]) for p in positions]
    write_matrix(
        run_dir,
        f"positions_{name}",
        np.stack(
            [
                accuracy[1, i, :, 0, READER, :, at[i]].mean(axis=-1)
                for i in range(len(positions))
            ]
        ),
        rows=list(positions),
        cols=list(positions),
        title=(
            f"{name}: train position (rows, at its layer) vs "
            "evaluation position"
        ),
    )


def _check(config: Config) -> tuple[str, ...]:
    """Checks what the config alone settles and expands its tasks."""
    tasks = expand_tasks(config.tasks)
    if not tasks:
        raise ValueError("p2 needs at least one task")
    if not config.positions:
        raise ValueError("p2 needs at least one position")
    for position in config.positions:
        if position not in POSITIONS:
            raise ValueError(
                f"unknown position {position!r}, expected one of "
                f"{list(POSITIONS)}"
            )
    for name in tasks:
        task = READING_TASKS[name]
        if task.pairing == "one_vs_rest":
            raise ValueError(f"p2 reads binary tasks only, not {name!r}")
        if not _readable_positions(task, config.positions)[0]:
            raise ValueError(
                f"task {name!r} has no position left to read: its concept "
                "position needs a concept slot or word"
            )
    return tasks


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    examples: Mapping[str, Sequence[Example]],
) -> None:
    """Reads every task at every position, original and under decoys.

    Per task the paper reader is trained at each position, on unbiased
    prompts, and scored at every position on the select and report items of
    the original prompts and of each decoy condition. Each position's layer is
    its best select layer on the original prompts, and every condition of that
    position is reported there. The zero-shot baseline rates the same report
    items by a verbalized level. Writes, per task, `sweep_<task>` (the cube of
    accuracies, the zero-shot accuracies and the layers), `curves_<task>` (the
    positions) and `curves_<task>_<position>` (the conditions), both on the
    report split, and `positions_<task>`; and `table.json` and `summary.json`
    for the whole run.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model.
        examples: The examples of each source, by source name.

    Raises:
        ValueError: If the config is refused (see `run`) or a source is
            missing (both checked before any model call).
    """
    tasks = _check(config)
    for name in tasks:
        task = READING_TASKS[name]
        task_examples(task, examples)
        task_examples(target_of(task), examples)
    chat = chat_format_for(loaded)
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "tasks": list(tasks),
        "layers": {},
        "skipped": {},
        "n_eval_pairs": {},
        "conditions": ["original", *config.decoys],
    }
    for name in tasks:
        result = _read_task(
            config, run_dir, loaded, chat, READING_TASKS[name], examples
        )
        rows.extend(result.rows)
        summary["layers"][name] = result.layers
        summary["skipped"][name] = result.skipped
        summary["n_eval_pairs"][name] = result.n_eval_pairs
    run_dir.write_json("table", rows)
    run_dir.write_json("summary", summary)


def run(config: Config, run_dir: RunDir) -> None:
    """Loads the model and the datasets of the tasks, then runs `execute`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.

    Raises:
        ValueError: If there is no task or no position, a task is unknown, a
            position is not in `POSITIONS`, a task is not binary or has no
            position left to read (all checked before the model is loaded).
    """
    _check(config)
    sources = task_sources(config.tasks)
    execute(config, run_dir, load_model(config.model), load_sources(sources))


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws the P2 figures of every task.

    Per task: the curves of the positions, the curves of the conditions at
    each position that was read, and the matrix of the train position by the
    evaluation position (accuracy in 0 to 1).

    Args:
        run_dir: A finished P2 run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths.
    """
    summary = cast(dict[str, Any], run_dir.read_json("summary"))
    paths: list[Path] = []
    for task in summary["tasks"]:
        sidecar = cast(dict[str, Any], run_dir.read_json(f"sweep_{task}"))
        paths += plot_curves_file(run_dir, out_dir, stem=f"curves_{task}")
        for position in sidecar["positions"]:
            paths += plot_curves_file(
                run_dir, out_dir, stem=f"curves_{task}_{position}"
            )
        paths += plot_matrix_file(
            run_dir, out_dir, stem=f"positions_{task}", vmin=0.0, vmax=1.0
        )
    return paths
