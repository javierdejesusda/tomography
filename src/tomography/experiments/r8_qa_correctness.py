"""R8: LAT against few-shot prompting on the RepE QA tasks.

Replicates the protocol of Figure 7 and Table 9 of the RepE paper on one
checkpoint, on OpenBookQA, CommonsenseQA, ARC-Easy, ARC-Challenge and RACE,
plus TruthfulQA MC1 (its Table 1), with the templates, splits and prompts of
`qa_tasks`.

Each seed draws the reference's training questions (seed `config.seed + i`,
so the default seeds are the reference's 0 to 9) and fits one LAT direction
per layer on them, from unsupervised pairs inside each question, oriented by
the reference sign rule. Three rules pick the layer each seed is tested at:
"reference" (the best accuracy on the labeled layer-selection questions),
"k-shot" (the best accuracy on the seed's own training questions, so no
example beyond the few-shot ones is used) and "test-oracle" (one layer for
every seed, the best mean test accuracy; an upper bound, not a protocol).
Ties go to the deeper layer, as the reference's scan from the last layer
does. On TruthfulQA the layer-selection questions are the 6 training
primers, as in the reference, so the "reference" and "k-shot" rules score
the same questions and pick the same layers there (unless `n_eval_pairs`
keeps fewer than 6); both are still reported.
"""

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any
from typing import cast
from typing import get_args

import numpy as np
from numpy.typing import NDArray
import torch

from tomography.datasets.arc import load_arc
from tomography.datasets.commonsenseqa import load_commonsenseqa
from tomography.datasets.openbookqa import load_openbookqa
from tomography.datasets.race import RacePassage
from tomography.datasets.race import load_race
from tomography.datasets.records import ChoiceExample
from tomography.datasets.stimulus import PairingMode
from tomography.datasets.stimulus import StimulusSet
from tomography.datasets.stimulus import build_stimulus_set
from tomography.datasets.truthfulqa import load_truthfulqa
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.sweep import SERIES
from tomography.experiments.sweep import Prepared
from tomography.experiments.sweep import ReadingConfig
from tomography.experiments.sweep import plot_curves_file
from tomography.experiments.sweep import stimulus_activations
from tomography.experiments.sweep import sweep_from_activations
from tomography.experiments.sweep import write_curves
from tomography.judges import continuation_logprob
from tomography.model import BosMode
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.qa_tasks import SHOTS
from tomography.qa_tasks import TEMPLATES
from tomography.qa_tasks import FewShotTask
from tomography.qa_tasks import HarnessPrompt
from tomography.qa_tasks import QaSplits
from tomography.qa_tasks import Task
from tomography.qa_tasks import choice_splits
from tomography.qa_tasks import fewshot_prompt
from tomography.qa_tasks import harness_credit
from tomography.qa_tasks import lat_item
from tomography.qa_tasks import race_item
from tomography.qa_tasks import race_prompt
from tomography.qa_tasks import race_splits
from tomography.qa_tasks import tqa_prompt
from tomography.reading.metrics import best_layer
from tomography.reading.metrics import bootstrap_ci
from tomography.reading.metrics import group_accuracy
from tomography.reading.metrics import wilson_ci
from tomography.reading.reader import RepReader
from tomography.templates import ChatFormat

TASKS: tuple[Task, ...] = get_args(Task)
LAYER_RULES = ("reference", "k-shot", "test-oracle")
# The tasks of Figure 7 and of Table 9's average, in the figure's order.
FIGURE7_TASKS: tuple[Task, ...] = (
    "race",
    "csqa",
    "obqa",
    "arc_challenge",
    "arc_easy",
)
FEW_SHOT_TASKS: tuple[FewShotTask, ...] = get_args(FewShotTask)
METRICS = ("acc", "acc_norm")
# The tasks whose only harness metric is acc; Figure 7's acc_norm average
# takes their acc.
ACC_ONLY_TASKS: tuple[Task, ...] = ("race", "tqa")
COLUMNS = ("task", "method", "rule", "metric", "seed", "layer", "accuracy")

QaRow = ChoiceExample | RacePassage


@dataclass(frozen=True, kw_only=True)
class Config(ReadingConfig):
    """R8 settings: the RepE QA protocol on one checkpoint.

    The shared defaults are the protocol's: 10 seeds, `pca_diff` on centered
    differences (`preprocess="unit"` is the App. C.1 variant) and the last
    token. `n_pairs` is not used: the training questions of a seed are the
    stimulus counts of Figure 7 (`qa_tasks.SHOTS`). `n_eval_pairs` keeps that
    many questions of each evaluation split, layer selection and test, for a
    smoke run; None keeps them all, as the protocol does. `bos` must be
    "none": the code templates are plain text for every checkpoint and the
    Qwen3 tokenizer adds no BOS, so base and post-trained checkpoints read
    the same tokens, and the few-shot scorer never prepends one, as the
    harness does not. `layers`, when set, must be strictly increasing.

    Attributes:
        tasks: The tasks to run, in run order.
        n_fewshot_seeds: How many of the seeds the few-shot baseline scores,
            the first ones; None means all `n_seeds`.
        race_batch_size: Texts per forward pass for the RACE stimuli, whose
            articles make prompts of up to about 1,110 tokens; the other
            tasks use `batch_size`.
    """

    experiment: str = "r8"
    model: str = "qwen3-0.6b-base"
    tasks: tuple[Task, ...] = TASKS
    n_fewshot_seeds: int | None = None
    race_batch_size: int = 1
    bos: BosMode | None = "none"


@dataclass(frozen=True, eq=False)
class _Lat:
    """The LAT accuracies of one task, per seed and layer.

    Attributes:
        layers: The layer of each column.
        train: The reader's accuracy on its seed's training questions,
            shape (N, K).
        select: Its accuracy on its seed's layer-selection questions, shape
            (N, K).
        test: The test accuracy of every `SERIES`, shape (3, N, K).
        n_select: The number of layer-selection questions of each seed.
        test_rows: The test rows read, in source order.
    """

    layers: tuple[int, ...]
    train: NDArray[np.float64]
    select: NDArray[np.float64]
    test: NDArray[np.float64]
    n_select: list[int]
    test_rows: list[QaRow]


def _check(config: Config) -> None:
    """Refuses a config that cannot run, before any model call."""
    if not config.tasks:
        raise ValueError("r8 needs at least one task")
    unknown = [task for task in config.tasks if task not in TASKS]
    if unknown:
        raise ValueError(f"unknown tasks {unknown}, expected from {TASKS}")
    repeated = sorted({t for t in config.tasks if config.tasks.count(t) > 1})
    if repeated:
        raise ValueError(f"tasks {repeated} are listed more than once")
    for field in ("n_seeds", "batch_size", "race_batch_size"):
        value = getattr(config, field)
        if value < 1:
            raise ValueError(f"{field} must be at least 1, got {value}")
    size = config.n_eval_pairs
    if size is not None and size < 1:
        raise ValueError(f"n_eval_pairs must be None or at least 1, got {size}")
    fewshot = config.n_fewshot_seeds
    if fewshot is not None and not 1 <= fewshot <= config.n_seeds:
        raise ValueError(
            f"n_fewshot_seeds must be None or between 1 and n_seeds="
            f"{config.n_seeds}, got {fewshot}"
        )
    if config.bos != "none":
        raise ValueError(
            f'bos must be "none", as the few-shot scorer reads no BOS, '
            f"got {config.bos!r}"
        )
    layers = config.layers
    if layers is not None and any(a >= b for a, b in pairwise(layers)):
        raise ValueError(f"layers must be strictly increasing, got {layers}")


def _question(row: QaRow) -> ChoiceExample:
    """The multiple-choice item of a row: a RACE passage's last question."""
    return row.last if isinstance(row, RacePassage) else row


def _splits(task: Task, rows: Sequence[QaRow], seed: int) -> QaSplits[QaRow]:
    """The reference's training, layer-selection and test rows of a seed."""
    if task == "race":
        return race_splits(cast(Sequence[RacePassage], rows), seed=seed)
    return choice_splits(task, cast(Sequence[ChoiceExample], rows), seed=seed)


def _stimuli(
    task: Task,
    rows: Sequence[QaRow],
    pairing: PairingMode,
    seed: int,
    n_pairs: int | None = None,
) -> StimulusSet:
    """Renders one stimulus per choice of each row with the code template."""
    if task == "race":
        items = [race_item(cast(RacePassage, row)) for row in rows]
    else:
        items = [lat_item(task, cast(ChoiceExample, row)) for row in rows]
    examples = [example for item in items for example in item.flatten()]
    return build_stimulus_set(
        examples,
        TEMPLATES[task],
        ChatFormat("none"),
        pairing=pairing,
        group_ids=[str(example.meta["group"]) for example in examples],
        n_pairs=n_pairs,
        seed=seed,
    )


def _group_questions(stimuli: StimulusSet) -> list[str]:
    """The question uid of each group of a stimulus set."""
    return [
        stimuli.uids[group[0]].rsplit(":", 1)[0] for group in stimuli.groups
    ]


def _accuracy(
    reader: RepReader, prepared: Prepared, keep: set[str] | None = None
) -> NDArray[np.float64]:
    """A reader's accuracy per layer on the groups of a stimulus set.

    Args:
        reader: The fitted reader.
        prepared: The stimulus set and its activations.
        keep: The question uids whose groups count; None keeps every group.

    Returns:
        The accuracy of each layer of the reader.
    """
    stimuli, act = prepared
    groups = stimuli.groups
    if keep is not None:
        groups = groups[[uid in keep for uid in _group_questions(stimuli)]]
    return group_accuracy(
        reader.score(act.data),
        act.row_groups(groups),
        act.row_values(stimuli.labels),
    )


def _lat(
    config: Config,
    loaded: LoadedModel,
    task: Task,
    splits: Sequence[QaSplits[QaRow]],
) -> _Lat:
    """Fits a reader per seed and scores it on every split of the task.

    The training questions of each seed are paired inside each question
    (`within_group`). The test and then the layer-selection questions are
    read once for all seeds; CommonsenseQA's layer-selection questions
    depend on the seed, so they are read for the union of the seeds and
    each seed counts its own.
    """
    layers = (
        tuple(range(1, loaded.n_layers + 1))
        if config.layers is None
        else config.layers
    )
    batch_size = config.race_batch_size if task == "race" else config.batch_size

    def read(stimuli: StimulusSet, *, train: bool) -> Prepared:
        position = config.position
        if not train and config.eval_position is not None:
            position = config.eval_position
        act = stimulus_activations(
            loaded,
            stimuli,
            layers=layers,
            position=position,
            bos=config.bos,
            batch_size=batch_size,
        )
        return stimuli, act

    def evaluation(rows: Sequence[QaRow], name: str) -> Prepared:
        size = config.n_eval_pairs
        stimuli = _stimuli(
            task,
            rows,
            "by_group",
            derive_seed(config.seed, "eval", task, name),
            None if size is None else min(size, len(rows)),
        )
        return read(stimuli, train=False)

    train = [
        read(
            _stimuli(
                task,
                split.train,
                "within_group",
                derive_seed(config.seed, "train", task, i),
            ),
            train=True,
        )
        for i, split in enumerate(splits)
    ]
    test = evaluation(splits[0].test, "test")
    result = sweep_from_activations(
        train,
        {"test": test},
        method=config.method,
        preprocess=config.preprocess,
        seed=derive_seed(config.seed, task),
    )
    train_accuracy = np.stack(
        [
            _accuracy(reader, prepared)
            for reader, prepared in zip(result.readers, train, strict=True)
        ]
    )
    tested = set(_group_questions(test[0]))
    # The layer-selection questions are read last, after the training and
    # test activations are released, so the host never holds all three.
    del train, test
    pool = {_question(row).uid: row for s in splits for row in s.layer}
    select = evaluation(list(pool.values()), "layer")
    selected = set(_group_questions(select[0]))
    kept = [
        {_question(row).uid for row in split.layer} & selected
        for split in splits
    ]
    return _Lat(
        layers=tuple(layers),
        train=train_accuracy,
        select=np.stack(
            [
                _accuracy(reader, select, keep)
                for reader, keep in zip(result.readers, kept, strict=True)
            ]
        ),
        test=result.per_seed("test"),
        n_select=[len(keep) for keep in kept],
        test_rows=[
            row for row in splits[0].test if _question(row).uid in tested
        ],
    )


def _deepest_best(acc: NDArray[np.float64], layers: Sequence[int]) -> int:
    """The layer of the highest accuracy, the deepest one on ties."""
    return best_layer(acc[::-1], layers[::-1])


def _rule_layers(lat: _Lat) -> dict[str, list[int]]:
    """The layer each rule tests each seed at."""
    oracle = _deepest_best(lat.test[0].mean(axis=0), lat.layers)
    return {
        "reference": [_deepest_best(acc, lat.layers) for acc in lat.select],
        "k-shot": [_deepest_best(acc, lat.layers) for acc in lat.train],
        "test-oracle": [oracle] * lat.test.shape[1],
    }


def _interval(
    values: NDArray[np.float64], n_questions: int, seed: int
) -> dict[str, Any]:
    """The mean of per-seed accuracies with its 95% interval.

    The interval resamples the seeds when there are several, else it is the
    Wilson interval of the questions of the single run.
    """
    mean = float(values.mean())
    if values.size > 1:
        lo, hi = (float(bound) for bound in bootstrap_ci(values, seed=seed))
        interval = "seeds"
    else:
        lo, hi = wilson_ci(mean * n_questions, n_questions)
        interval = "questions"
    return {"accuracy": mean, "lo": lo, "hi": hi, "interval": interval}


def _harness_prompt(
    task: Task, row: QaRow, shots: Sequence[QaRow]
) -> HarnessPrompt:
    """The harness prompt of a row; only the few-shot tasks read `shots`."""
    if isinstance(row, RacePassage):
        return race_prompt(row)
    if task == "tqa":
        return tqa_prompt(row)
    return fewshot_prompt(
        cast(FewShotTask, task), row, cast(Sequence[ChoiceExample], shots)
    )


def _few_shot(
    config: Config,
    loaded: LoadedModel,
    task: Task,
    splits: Sequence[QaSplits[QaRow]],
    test_rows: Sequence[QaRow],
) -> tuple[list[int | None], NDArray[np.float32]]:
    """Scores every choice of the test questions with the harness prompts.

    A few-shot task shows each of the first `n_fewshot_seeds` seeds its own
    training questions, the k examples its LAT reader was fitted on, so the
    k-shot context is the same for every test question of a seed. RACE and
    TruthfulQA are zero-shot and run once. One `continuation_logprob` call
    scores every (question, choice) pair of a run, so the shared context runs
    once per seed and task. Scores depend on the grouping of a call at
    float32 rounding level, because a call shares the longest token prefix
    of all its contexts; one call per seed and task fixes it.

    Returns:
        The seed of each run (None for a zero-shot task) and the summed
        log-likelihood of every continuation, shape (runs, choices), the
        choices of the test questions in order.
    """
    runs: list[tuple[int | None, Sequence[QaRow]]] = [(None, ())]
    if task in FEW_SHOT_TASKS:
        n = config.n_fewshot_seeds or config.n_seeds
        runs = [(config.seed + i, splits[i].train) for i in range(n)]
    loglik = []
    for _, shots in runs:
        prompts = [_harness_prompt(task, row, shots) for row in test_rows]
        loglik.append(
            continuation_logprob(
                loaded.model,
                loaded.tokenizer,
                [p.context for p in prompts for _ in p.continuations],
                [c for p in prompts for c in p.continuations],
                bos_id=None,
            )
        )
    return [seed for seed, _ in runs], np.stack(loglik)


def _credits(
    loglik: NDArray[np.float32], items: Sequence[ChoiceExample]
) -> NDArray[np.float64]:
    """The mean acc and acc_norm of each run, shape (runs, 2)."""
    ends = np.cumsum([len(item.choices) for item in items])
    return np.array(
        [
            np.mean(
                [
                    harness_credit(
                        run[end - len(item.choices) : end].tolist(), item
                    )
                    for item, end in zip(items, ends, strict=True)
                ],
                axis=0,
            )
            for run in loglik
        ]
    )


def _figure7(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Table 9's average row over the Figure 7 tasks of the run.

    The few-shot acc_norm average takes RACE's acc, its only harness metric.
    """
    tasks = [task for task in FIGURE7_TASKS if task in summary]

    def average(method: str, keys: Mapping[str, str]) -> float | None:
        values = [summary[t][method][keys[t]]["accuracy"] for t in tasks]
        return float(np.mean(values)) if values else None

    return {
        "tasks": tasks,
        "lat": {
            rule: average("lat", dict.fromkeys(tasks, rule))
            for rule in LAYER_RULES
        },
        "few_shot": {
            metric: average(
                "few_shot",
                {t: "acc" if t in ACC_ONLY_TASKS else metric for t in tasks},
            )
            for metric in METRICS
        },
    }


def _run_task(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    task: Task,
    rows: Sequence[QaRow],
) -> tuple[list[list[Any]], dict[str, Any]]:
    """Runs one task and writes its files, `task_<task>` last.

    The task's activations are released on return, before the next task is
    read.

    Returns:
        The task's table rows and its summary.
    """
    seeds = [config.seed + i for i in range(config.n_seeds)]
    splits = [_splits(task, rows, seed) for seed in seeds]
    lat = _lat(config, loaded, task, splits)
    test_rows = lat.test_rows
    items = [_question(row) for row in test_rows]
    n_test = len(items)
    chance = float(np.mean([1 / len(item.choices) for item in items]))
    write_curves(
        run_dir,
        lat.test,
        lat.layers,
        series=SERIES,
        stem=f"curves_{task}",
        ylabel="Test accuracy",
        chance=chance,
        n_layers=loaded.n_layers,
        seed=derive_seed(config.seed, "curves", task),
    )
    run_dir.write_arrays(
        f"lat_{task}",
        layers=np.asarray(lat.layers, dtype=np.int64),
        seeds=np.asarray(seeds, dtype=np.int64),
        train=lat.train,
        layer=lat.select,
    )
    table: list[list[Any]] = []
    lat_summary: dict[str, Any] = {}
    for rule, chosen in _rule_layers(lat).items():
        accuracy = np.array(
            [
                lat.test[0, i, lat.layers.index(layer)]
                for i, layer in enumerate(chosen)
            ]
        )
        table += [
            [task, "lat", rule, "acc", seed, layer, value]
            for seed, layer, value in zip(seeds, chosen, accuracy, strict=True)
        ]
        lat_summary[rule] = {
            **_interval(
                accuracy, n_test, derive_seed(config.seed, "ci", task, rule)
            ),
            "layers": chosen,
        }
    fewshot_seeds, loglik = _few_shot(config, loaded, task, splits, test_rows)
    run_dir.write_arrays(
        f"fewshot_{task}",
        loglik=loglik,
        n_choices=np.array([len(item.choices) for item in items]),
        correct=np.array([item.correct for item in items]),
        chars=np.array([len(c) for item in items for c in item.choices]),
    )
    credits = _credits(loglik, items)
    table += [
        [task, "few-shot", None, metric, seed, None, value]
        for seed, run in zip(fewshot_seeds, credits, strict=True)
        for metric, value in zip(METRICS, run, strict=True)
    ]
    few_shot: dict[str, Any] = {
        "seeds": [seed for seed in fewshot_seeds if seed is not None],
        "shots": SHOTS[task] if task in FEW_SHOT_TASKS else 0,
    }
    for column, metric in enumerate(METRICS):
        few_shot[metric] = _interval(
            credits[:, column],
            n_test,
            derive_seed(config.seed, "ci", task, metric),
        )
    summary = {
        "n_questions": {
            "train": len(splits[0].train),
            "layer": lat.n_select,
            "test": n_test,
        },
        "test_uids": [item.uid for item in items],
        "chance": chance,
        "lat": lat_summary,
        "few_shot": few_shot,
    }
    run_dir.write_json(f"task_{task}", {"rows": table, "summary": summary})
    return table, summary


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    data: Mapping[Task, Sequence[QaRow]],
) -> None:
    """Runs LAT and the few-shot baseline on every task.

    Each task writes its files as it ends, so an interrupted run keeps the
    tasks it finished: `curves_<task>` (the test accuracy of the reader and
    the random and shuffled controls per seed and layer), `lat_<task>` (the
    reader's accuracy per seed and layer on its training questions and on
    its layer-selection questions), `fewshot_<task>` (the log-likelihood of
    every test choice per few-shot seed, with the choice counts, the gold
    indices and the choice lengths in characters) and, last, `task_<task>`
    (its table rows and summary). The run then writes `table.json` (one row
    per task, method, layer rule, metric and seed) and `summary.json` (every
    task's summary, and in `figure7` the mean over the Figure 7 tasks of each
    LAT rule and few-shot metric, Table 9's average row).

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model, in float32 for a real run.
        data: The rows of every task: `ChoiceExample` items, or
            `RacePassage` passages for "race".

    Raises:
        ValueError: If the config is refused or a task has no data (both
            checked before any model call).
    """
    _check(config)
    missing = [task for task in config.tasks if task not in data]
    if missing:
        raise ValueError(f"no data for tasks {missing}")
    rows: list[list[Any]] = []
    summary: dict[str, Any] = {}
    for task in config.tasks:
        task_rows, summary[task] = _run_task(
            config, run_dir, loaded, task, data[task]
        )
        rows += task_rows
    run_dir.write_json("table", {"columns": list(COLUMNS), "rows": rows})
    run_dir.write_json(
        "summary", {"tasks": summary, "figure7": _figure7(summary)}
    )


def _load(task: Task) -> Sequence[QaRow]:
    """Loads the rows of one task from its pinned source."""
    if task == "obqa":
        return load_openbookqa()
    if task == "csqa":
        return load_commonsenseqa()
    if task == "arc_easy":
        return load_arc(config="ARC-Easy")
    if task == "arc_challenge":
        return load_arc()
    if task == "race":
        return load_race()
    return load_truthfulqa()


def run(config: Config, run_dir: RunDir) -> None:
    """Loads the datasets of the tasks and the model in float32, then runs.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.

    Raises:
        ValueError: If the config is refused (checked before anything is
            loaded).
    """
    _check(config)
    data = {task: _load(task) for task in config.tasks}
    execute(
        config, run_dir, load_model(config.model, dtype=torch.float32), data
    )


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws the test accuracy per layer of every task.

    Args:
        run_dir: A finished R8 run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths of every `curves_<task>` file.
    """
    summary = cast(dict[str, Any], run_dir.read_json("summary"))
    paths: list[Path] = []
    for task in summary["tasks"]:
        paths += plot_curves_file(run_dir, out_dir, stem=f"curves_{task}")
    return paths
