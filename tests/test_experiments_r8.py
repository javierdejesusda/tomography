"""Tests for R8, LAT against few-shot prompting on the RepE QA tasks."""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import importlib
from pathlib import Path
from typing import Any
from typing import cast
import weakref

import numpy as np
from numpy.typing import NDArray
import pytest
import torch

from tomography.activations import Activations
from tomography.cli import load_experiment
from tomography.datasets.race import RacePassage
from tomography.datasets.records import ChoiceExample
from tomography.datasets.records import Split
from tomography.datasets.stimulus import StimulusSet
from tomography.experiments import r8_qa_correctness
from tomography.experiments.base import RunDir
from tomography.experiments.r8_qa_correctness import FIGURE7_TASKS
from tomography.experiments.r8_qa_correctness import LAYER_RULES
from tomography.experiments.r8_qa_correctness import TASKS
from tomography.experiments.r8_qa_correctness import Config
from tomography.experiments.r8_qa_correctness import execute
from tomography.experiments.r8_qa_correctness import plot
from tomography.experiments.r8_qa_correctness import run
from tomography.judges import sequence_logprob
from tomography.model import LoadedModel
from tomography.qa_tasks import SHOTS
from tomography.qa_tasks import HarnessPrompt
from tomography.qa_tasks import Task
from tomography.qa_tasks import choice_splits
from tomography.qa_tasks import fewshot_prompt
from tomography.qa_tasks import race_prompt
from tomography.qa_tasks import tqa_prompt

DEFAULT_HASH = "68370987e9a5"
# The default config hash of every experiment that shares `ReadingConfig`
# with R8, before R8 existed.
READING_HASHES = {
    "m2_readers": "3c82fe936345",
    "p1_templates": "078b9d6482a2",
    "p2_positions": "ee099a704022",
    "p3_size": "676c31488397",
    "p4_pairing": "8c4a73d1e9db",
    "p5_persona_wording": "80b4ae80a6ca",
    "r1_sentiment": "bd4438c46cef",
    "r2_utility": "e1821ff197ce",
    "r3_morality": "a9f7b2a71c22",
    "r4_emotions": "4702fcb81ed3",
    "r5_honesty": "779f5b093803",
    "r6_truthfulqa": "47e8ec0cf46c",
    "r7_truth": "5908a78acf64",
}
# Choice words of the tiny tokenizer's corpus, so prompts stay short.
ANSWERS = ("yes", "no", "the dog", "a text", "the model")
Data = Mapping[Task, Sequence[ChoiceExample] | Sequence[RacePassage]]


def _items(
    task: str,
    counts: Mapping[Split, int],
    widths: Sequence[int] = (4,),
) -> list[ChoiceExample]:
    """Questions named "<task>-<split>-<i>", the width cycling `widths`."""
    items = []
    for split, count in counts.items():
        for i in range(count):
            width = widths[i % len(widths)]
            items.append(
                ChoiceExample(
                    uid=f"{task}-{split}-{i}",
                    question=f"Is the movie {i} good?",
                    choices=tuple(f"{ANSWERS[j]} {i}" for j in range(width)),
                    correct=i % width,
                    split=split,
                )
            )
    return items


def _passages(counts: Mapping[Split, int]) -> list[RacePassage]:
    return [
        RacePassage(
            article="The quick brown fox jumps.\nThe dog is lazy.",
            last=item,
            earlier=(("Who is lazy?", "the dog"),),
        )
        for item in _items("race", counts)
    ]


def _data() -> dict[Task, Sequence[ChoiceExample] | Sequence[RacePassage]]:
    return {
        "obqa": _items("obqa", {"train": 6, "val": 3, "test": 3}),
        "csqa": _items("csqa", {"train": 30, "val": 4}, widths=(5,)),
        "arc_easy": [
            *_items("arc_easy", {"train": 26}),
            *_items("arc_easy", {"val": 3, "test": 3}, widths=(4, 3, 5)),
        ],
        "arc_challenge": _items(
            "arc_challenge", {"train": 26, "val": 3, "test": 3}
        ),
        "race": _passages({"train": 4, "val": 2, "test": 2}),
        "tqa": _items("tqa", {"train": 2, "val": 1, "test": 1}, (2, 3)),
    }


def _config(**overrides: Any) -> Config:
    return Config(**{"model": "tiny", "n_seeds": 2, **overrides})


def _rows(run_dir: RunDir, name: str = "table") -> list[dict[str, Any]]:
    payload = cast(dict[str, Any], run_dir.read_json(name))
    return [
        dict(zip(payload["columns"], row, strict=True))
        for row in payload["rows"]
    ]


def _task_file(run_dir: RunDir, task: str) -> dict[str, Any]:
    return cast(dict[str, Any], run_dir.read_json(f"task_{task}"))


def _run(
    config: Config,
    loaded: LoadedModel,
    tmp_path: Path,
    data: Data | None = None,
) -> RunDir:
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, loaded, _data() if data is None else data)
    return run_dir


@pytest.fixture
def qwen3(tiny_qwen3: LoadedModel) -> LoadedModel:
    # The tiny Qwen3 has no position table (rotary embeddings), so lifting
    # its limit only lets the RACE and 25-shot prompts through the encoder.
    tiny_qwen3.model.config.max_position_embeddings = 4096
    return tiny_qwen3


@pytest.fixture(scope="module")
def runs() -> dict[str, RunDir]:
    return {}


@pytest.fixture
def finished(
    runs: dict[str, RunDir],
    qwen3: LoadedModel,
    tmp_path_factory: pytest.TempPathFactory,
) -> RunDir:
    if "run" not in runs:
        runs["run"] = _run(_config(), qwen3, tmp_path_factory.mktemp("r8"))
    return runs["run"]


def test_config_defaults_follow_the_protocol() -> None:
    config = Config()

    assert config.experiment == "r8"
    assert config.model == "qwen3-0.6b-base"
    assert config.tasks == (
        "obqa",
        "csqa",
        "arc_easy",
        "arc_challenge",
        "race",
        "tqa",
    )
    assert config.n_seeds == 10
    assert config.n_fewshot_seeds is None
    assert config.n_eval_pairs is None
    assert (config.method, config.preprocess) == ("pca_diff", "center")
    assert config.position == "last"
    assert config.batch_size == 8
    assert config.race_batch_size == 1
    assert config.bos == "none"
    assert Config.from_dict(config.to_dict()) == config


def test_the_default_config_hash_is_stable() -> None:
    assert Config().config_hash() == DEFAULT_HASH


@pytest.mark.parametrize("name", sorted(READING_HASHES))
def test_the_default_hashes_of_the_other_reading_experiments_hold(
    name: str,
) -> None:
    module = importlib.import_module(f"tomography.experiments.{name}")

    assert module.Config().config_hash() == READING_HASHES[name]


def test_r8_is_discovered_by_the_cli() -> None:
    experiment = load_experiment("r8")

    assert experiment.config_cls is Config


def test_a_config_rejects_an_unknown_task_name() -> None:
    data = Config().to_dict()
    data["tasks"] = ["obqa", "squad"]

    with pytest.raises(ValueError, match="tasks"):
        Config.from_dict(data)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"tasks": ()}, "at least one task"),
        ({"tasks": ("obqa", "obqa")}, "more than once"),
        ({"tasks": ("obqa", "squad")}, "unknown tasks"),
        ({"n_seeds": 0}, "n_seeds"),
        ({"n_fewshot_seeds": 0}, "n_fewshot_seeds"),
        ({"n_fewshot_seeds": 3}, "n_fewshot_seeds"),
        ({"batch_size": 0}, "batch_size"),
        ({"race_batch_size": 0}, "race_batch_size"),
        ({"n_eval_pairs": 0}, "n_eval_pairs"),
    ],
)
def test_a_bad_config_is_refused_before_any_model_call(
    overrides: dict[str, Any], message: str, tmp_path: Path
) -> None:
    config = _config(**overrides)

    with pytest.raises(ValueError, match=message):
        _run(config, cast(LoadedModel, None), tmp_path)


def test_a_task_without_data_is_refused_before_any_model_call(
    tmp_path: Path,
) -> None:
    data: Data = {t: rows for t, rows in _data().items() if t != "race"}

    with pytest.raises(ValueError, match="race"):
        _run(_config(), cast(LoadedModel, None), tmp_path, data)


def test_each_task_writes_its_curves_lat_arrays_and_rows(
    finished: RunDir,
) -> None:
    for task in TASKS:
        curves = finished.read_arrays(f"curves_{task}")
        lat = finished.read_arrays(f"lat_{task}")

        assert curves["per_seed"].shape == (3, 2, 3)
        assert lat["layers"].tolist() == [1, 2, 3]
        assert lat["train"].shape == lat["layer"].shape == (2, 3)
        assert _task_file(finished, task)["rows"]


def test_lat_has_a_row_per_task_layer_rule_and_seed(finished: RunDir) -> None:
    lat = [row for row in _rows(finished) if row["method"] == "lat"]

    assert [(r["task"], r["rule"], r["seed"]) for r in lat] == [
        (task, rule, seed)
        for task in TASKS
        for rule in LAYER_RULES
        for seed in (0, 1)
    ]
    assert all(row["metric"] == "acc" for row in lat)
    assert all(row["layer"] in (1, 2, 3) for row in lat)
    assert all(0.0 <= row["accuracy"] <= 1.0 for row in lat)


def test_the_test_curves_hold_the_accuracy_the_rules_read(
    finished: RunDir,
) -> None:
    for task in TASKS:
        test = finished.read_arrays(f"curves_{task}")["per_seed"][0]
        for row in _rows(finished):
            if row["task"] == task and row["method"] == "lat":
                assert row["accuracy"] == pytest.approx(
                    test[row["seed"], row["layer"] - 1]
                )


def test_the_csqa_layer_split_drops_each_seeds_training_questions(
    finished: RunDir,
) -> None:
    items = cast(list[ChoiceExample], _data()["csqa"])
    expected = [len(choice_splits("csqa", items, seed=s).layer) for s in (0, 1)]

    counts = _task_file(finished, "csqa")["summary"]["n_questions"]

    assert counts["layer"] == expected
    assert counts == {"train": SHOTS["csqa"], "layer": expected, "test": 4}


def test_tqa_keeps_both_rules_although_they_read_the_same_primers(
    finished: RunDir,
) -> None:
    rows = _rows(finished)
    lat = _task_file(finished, "tqa")["summary"]["lat"]

    def layers(rule: str) -> list[tuple[int, float]]:
        return [
            (row["layer"], row["accuracy"])
            for row in rows
            if (row["task"], row["method"], row["rule"]) == ("tqa", "lat", rule)
        ]

    assert len(layers("reference")) == len(layers("k-shot")) == 2
    assert layers("reference") == layers("k-shot")
    for key in ("accuracy", "layers"):
        assert lat["reference"][key] == lat["k-shot"][key]
    assert _task_file(finished, "tqa")["summary"]["n_questions"] == {
        "train": 6,
        "layer": [6, 6],
        "test": 4,
    }


def _planted(
    signal: Mapping[str, Sequence[float]],
    by_question: Mapping[str, Sequence[float]] | None = None,
) -> Callable[..., Activations]:
    """Stands in for `stimulus_activations` with planted activations.

    Every gold stimulus of a question in split `s` (read from its uid) is
    `signal[s][k]` times the first axis at layer k, or `by_question[q][k]`
    for a question q listed there; every other stimulus is zero, so equal
    members tie exactly.
    """
    by_question = by_question or {}

    def collect(
        loaded: LoadedModel,
        stimuli: StimulusSet,
        *,
        layers: Sequence[int],
        position: str,
        **kwargs: Any,
    ) -> Activations:
        n = len(stimuli.texts)
        data = np.zeros((len(layers), n, 4), dtype=np.float32)
        for row, uid in enumerate(stimuli.uids):
            question = uid.rsplit(":", 1)[0]
            if stimuli.labels[row] == 1:
                data[:, row, 0] = by_question.get(
                    question, signal[uid.split("-")[1]]
                )
        index = np.arange(n, dtype=np.int64)
        return Activations(
            data=data,
            layers=tuple(layers),
            position=cast(Any, position),
            row_stimulus=index,
            row_rank=np.zeros(n, dtype=np.int64),
            row_token=index,
            row_token_id=index,
            n_stimuli=n,
        )

    return collect


@pytest.fixture
def planted(
    runs: dict[str, RunDir],
    qwen3: LoadedModel,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> RunDir:
    # Training groups read at every layer; the layer split ties layers 1 and
    # 2; the test reads layer 1, inverts layer 2 and ties at layer 3.
    if "planted" not in runs:
        monkeypatch.setattr(
            r8_qa_correctness,
            "stimulus_activations",
            _planted(
                {
                    "train": (10.0, 10.0, 10.0),
                    "val": (10.0, 10.0, 0.0),
                    "test": (10.0, -10.0, 0.0),
                }
            ),
        )
        runs["planted"] = _run(
            _config(tasks=("obqa",)),
            qwen3,
            tmp_path_factory.mktemp("r8_planted"),
        )
    return runs["planted"]


@pytest.mark.parametrize(
    ("rule", "layer", "accuracy"),
    [
        # The layer split ties layers 1 and 2: the deeper one is taken.
        ("reference", 2, 0.0),
        # Every layer reads the k training groups: the deepest is taken.
        ("k-shot", 3, 0.25),
        ("test-oracle", 1, 1.0),
    ],
)
def test_the_layer_rules_pick_the_planted_layers(
    planted: RunDir, rule: str, layer: int, accuracy: float
) -> None:
    rows = [row for row in _rows(planted) if row["rule"] == rule]

    assert [(row["seed"], row["layer"]) for row in rows] == [
        (0, layer),
        (1, layer),
    ]
    assert [row["accuracy"] for row in rows] == [accuracy, accuracy]
    summary = _task_file(planted, "obqa")["summary"]["lat"][rule]
    assert summary["accuracy"] == accuracy
    assert summary["layers"] == [layer, layer]


def test_planted_accuracies_reach_the_curves_and_the_select_arrays(
    planted: RunDir,
) -> None:
    lat = planted.read_arrays("lat_obqa")
    test = planted.read_arrays("curves_obqa")["per_seed"][0]

    np.testing.assert_array_equal(lat["train"], [[1.0, 1.0, 1.0]] * 2)
    np.testing.assert_array_equal(lat["layer"], [[1.0, 1.0, 0.25]] * 2)
    np.testing.assert_array_equal(test, [[1.0, 0.0, 0.25]] * 2)


def test_each_seed_selects_its_layer_on_its_own_questions_only(
    qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # As CommonsenseQA's do, the layer-selection questions differ by seed:
    # seed 0 reads val 0 and 1, seed 1 val 1 and 2, and val 2 is inverted.
    splits = r8_qa_correctness.choice_splits

    def by_seed(task: Any, items: Any, *, seed: int) -> Any:
        found = splits(task, items, seed=seed)
        return dataclasses.replace(found, layer=found.layer[seed : seed + 2])

    monkeypatch.setattr(r8_qa_correctness, "choice_splits", by_seed)
    monkeypatch.setattr(
        r8_qa_correctness,
        "stimulus_activations",
        _planted(
            {"train": (10.0,) * 3, "val": (10.0,) * 3, "test": (10.0,) * 3},
            {"obqa-val-2": (-10.0,) * 3},
        ),
    )
    run_dir = _run(_config(tasks=("obqa",)), qwen3, tmp_path)

    lat = run_dir.read_arrays("lat_obqa")
    np.testing.assert_array_equal(lat["layer"], [[1.0] * 3, [0.5] * 3])
    counts = _task_file(run_dir, "obqa")["summary"]["n_questions"]
    assert counts["layer"] == [2, 2]


def test_a_run_interrupted_after_one_task_keeps_that_tasks_output(
    qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collect = r8_qa_correctness.stimulus_activations

    def failing(loaded: LoadedModel, stimuli: StimulusSet, **kw: Any) -> Any:
        if stimuli.uids[0].startswith("race"):
            raise RuntimeError("interrupted")
        return collect(loaded, stimuli, **kw)

    monkeypatch.setattr(r8_qa_correctness, "stimulus_activations", failing)
    config = _config(tasks=("obqa", "race"))
    run_dir = RunDir.create(config, out_root=tmp_path)

    with pytest.raises(RuntimeError, match="interrupted"):
        execute(config, run_dir, qwen3, _data())

    assert _task_file(run_dir, "obqa")["rows"]
    assert (run_dir.path / "curves_obqa.npz").is_file()
    assert (run_dir.path / "lat_obqa.npz").is_file()
    assert not list(run_dir.path.glob("*race*"))
    assert not (run_dir.path / "table.json").exists()


def test_a_task_releases_its_activations_before_the_next_is_read(
    qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    held: list[tuple[str, weakref.ref[Activations]]] = []
    collect = r8_qa_correctness.stimulus_activations

    def tracking(loaded: LoadedModel, stimuli: StimulusSet, **kw: Any) -> Any:
        task = stimuli.uids[0].split("-")[0]
        assert all(ref() is None for name, ref in held if name != task)
        act = collect(loaded, stimuli, **kw)
        held.append((task, weakref.ref(act)))
        return act

    monkeypatch.setattr(r8_qa_correctness, "stimulus_activations", tracking)
    _run(_config(tasks=("obqa", "race")), qwen3, tmp_path)

    assert {name for name, _ in held} == {"obqa", "race"}


def test_race_stimuli_are_read_at_the_race_batch_size(
    qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sizes: list[tuple[str, int]] = []
    collect = r8_qa_correctness.stimulus_activations

    def recording(loaded: LoadedModel, stimuli: StimulusSet, **kw: Any) -> Any:
        sizes.append((stimuli.uids[0].split("-")[0], kw["batch_size"]))
        return collect(loaded, stimuli, **kw)

    monkeypatch.setattr(r8_qa_correctness, "stimulus_activations", recording)
    config = _config(tasks=("obqa", "race"), batch_size=5, race_batch_size=3)
    _run(config, qwen3, tmp_path)

    assert set(sizes) == {("obqa", 5), ("race", 3)}


def test_n_eval_pairs_keeps_that_many_questions_per_evaluation_split(
    qwen3: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _run(
        _config(tasks=("obqa", "tqa"), n_eval_pairs=2), qwen3, tmp_path
    )

    for task in ("obqa", "tqa"):
        counts = _task_file(run_dir, task)["summary"]["n_questions"]
        assert counts["layer"] == [2, 2]
        assert counts["test"] == 2
        test = _task_file(run_dir, task)["summary"]["test_uids"]
        assert len(test) == 2
        n_choices = run_dir.read_arrays(f"fewshot_{task}")["n_choices"]
        assert n_choices.size == 2


FEW_SHOT_TASKS = ("obqa", "csqa", "arc_easy", "arc_challenge")
METRICS = ("acc", "acc_norm")


def _test_rows(run_dir: RunDir, task: Task) -> list[Any]:
    """The rows of a task's tested questions, in the run's order."""
    uids = _task_file(run_dir, task)["summary"]["test_uids"]
    rows = {
        (row.last if isinstance(row, RacePassage) else row).uid: row
        for row in _data()[task]
    }
    return [rows[uid] for uid in uids]


def _prompts(task: Task, rows: Sequence[Any], seed: int) -> list[HarnessPrompt]:
    """The harness prompts of a task's rows for one seed's shots."""
    if task == "race":
        return [race_prompt(row) for row in rows]
    if task == "tqa":
        return [tqa_prompt(row) for row in rows]
    items = cast(list[ChoiceExample], _data()[task])
    shots = choice_splits(task, items, seed=seed).train
    return [fewshot_prompt(task, row, shots) for row in rows]


def test_few_shot_has_a_row_per_task_seed_and_metric(finished: RunDir) -> None:
    rows = [row for row in _rows(finished) if row["method"] == "few-shot"]

    expected = []
    for task in TASKS:
        seeds = (0, 1) if task in FEW_SHOT_TASKS else (None,)
        expected += [(task, seed, m) for seed in seeds for m in METRICS]
    assert [(r["task"], r["seed"], r["metric"]) for r in rows] == expected
    assert all(r["rule"] is None and r["layer"] is None for r in rows)
    assert all(0.0 <= r["accuracy"] <= 1.0 for r in rows)


def test_few_shot_rows_come_from_the_stored_loglikelihoods(
    finished: RunDir,
) -> None:
    for task in TASKS:
        arrays = finished.read_arrays(f"fewshot_{task}")
        rows = _test_rows(finished, task)
        items = [r.last if isinstance(r, RacePassage) else r for r in rows]
        assert arrays["n_choices"].tolist() == [len(i.choices) for i in items]
        assert arrays["correct"].tolist() == [i.correct for i in items]
        assert arrays["chars"].tolist() == [
            len(choice) for item in items for choice in item.choices
        ]
        ends = np.cumsum(arrays["n_choices"])
        expected = []
        for loglik in arrays["loglik"]:
            wins: list[list[float]] = [[], []]
            for item, end in zip(items, ends, strict=True):
                span = slice(end - len(item.choices), end)
                for k, score in enumerate(
                    (loglik[span], loglik[span] / arrays["chars"][span])
                ):
                    top = np.flatnonzero(score == score.max())
                    wins[k].append(float(item.correct in top) / top.size)
            expected += [np.mean(wins[0]), np.mean(wins[1])]
        found = [
            row["accuracy"]
            for row in _rows(finished)
            if row["task"] == task and row["method"] == "few-shot"
        ]
        np.testing.assert_allclose(found, expected, atol=1e-12)


def test_shared_prefix_scores_equal_scoring_each_prompt_alone(
    finished: RunDir, qwen3: LoadedModel
) -> None:
    for task in TASKS:
        loglik = finished.read_arrays(f"fewshot_{task}")["loglik"]
        rows = _test_rows(finished, task)
        for index, seed in enumerate(
            (0, 1) if task in FEW_SHOT_TASKS else (0,)
        ):
            alone = np.concatenate(
                [
                    sequence_logprob(
                        qwen3.model,
                        qwen3.tokenizer,
                        [prompt.context] * len(prompt.continuations),
                        list(prompt.continuations),
                        batch_size=1,
                    )
                    for prompt in _prompts(task, rows, seed)
                ]
            )
            np.testing.assert_allclose(
                loglik[index], alone, rtol=1e-5, atol=1e-5
            )


def _spy_scorer(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[list[str], list[str], int | None]]:
    calls: list[tuple[list[str], list[str], int | None]] = []
    score = r8_qa_correctness.continuation_logprob

    def spy(
        model: Any,
        tokenizer: Any,
        prompts: Sequence[str],
        continuations: Sequence[str],
        **kw: Any,
    ) -> NDArray[np.float32]:
        calls.append((list(prompts), list(continuations), kw["bos_id"]))
        return score(model, tokenizer, prompts, continuations, **kw)

    monkeypatch.setattr(r8_qa_correctness, "continuation_logprob", spy)
    return calls


@pytest.mark.parametrize("n_fewshot_seeds", [None, 1])
def test_one_scorer_call_per_seed_and_task_with_that_seeds_shots(
    qwen3: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    n_fewshot_seeds: int | None,
) -> None:
    calls = _spy_scorer(monkeypatch)
    config = _config(
        tasks=("obqa", "race", "tqa"), n_fewshot_seeds=n_fewshot_seeds
    )
    run_dir = _run(config, qwen3, tmp_path)

    seeds = [0, 1] if n_fewshot_seeds is None else [0]
    expected = []
    for task in ("obqa", "race", "tqa"):
        rows = _test_rows(run_dir, task)
        for seed in seeds if task == "obqa" else [0]:
            prompts = _prompts(task, rows, seed)
            expected.append(
                (
                    [p.context for p in prompts for _ in p.continuations],
                    [c for p in prompts for c in p.continuations],
                    None,
                )
            )
    assert calls == expected
    summary = _task_file(run_dir, "obqa")["summary"]["few_shot"]
    assert summary["seeds"] == seeds
    assert summary["shots"] == SHOTS["obqa"]
    assert _task_file(run_dir, "race")["summary"]["few_shot"]["shots"] == 0


def test_planted_loglikelihoods_give_the_expected_acc_and_acc_norm(
    qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Gold choices always score highest, so acc is 1; per character only
    # the test question with the longest gold text wins, so acc_norm is 1/3.
    items = cast(list[ChoiceExample], _data()["obqa"])
    gold = {item.choices[item.correct] for item in items}

    def planted(
        model: Any, tokenizer: Any, prompts: Any, continuations: Any, **kw: Any
    ) -> NDArray[np.float32]:
        return np.array(
            [-2.0 if c.strip() in gold else -2.5 for c in continuations],
            dtype=np.float32,
        )

    monkeypatch.setattr(r8_qa_correctness, "continuation_logprob", planted)
    run_dir = _run(_config(tasks=("obqa",)), qwen3, tmp_path)

    rows = [row for row in _rows(run_dir) if row["method"] == "few-shot"]
    assert [(r["seed"], r["metric"]) for r in rows] == [
        (0, "acc"),
        (0, "acc_norm"),
        (1, "acc"),
        (1, "acc_norm"),
    ]
    assert [r["accuracy"] for r in rows] == pytest.approx([1.0, 1 / 3] * 2)
    summary = _task_file(run_dir, "obqa")["summary"]["few_shot"]
    assert summary["acc"]["accuracy"] == 1.0
    assert summary["acc_norm"]["accuracy"] == pytest.approx(1 / 3)
    assert summary["acc"]["interval"] == "seeds"


def test_a_zero_shot_task_takes_the_wilson_interval_of_its_questions(
    finished: RunDir,
) -> None:
    for task in ("race", "tqa"):
        summary = _task_file(finished, task)["summary"]
        acc = summary["few_shot"]["acc"]
        assert acc["interval"] == "questions"
        assert acc["lo"] <= acc["accuracy"] <= acc["hi"]
        assert summary["few_shot"]["seeds"] == []


def test_the_summary_averages_the_figure7_tasks_like_table9(
    finished: RunDir,
) -> None:
    summary = cast(dict[str, Any], finished.read_json("summary"))
    tasks = summary["tasks"]
    figure7 = summary["figure7"]

    assert FIGURE7_TASKS == (
        "race",
        "csqa",
        "obqa",
        "arc_challenge",
        "arc_easy",
    )
    assert figure7["tasks"] == list(FIGURE7_TASKS)
    for rule in LAYER_RULES:
        assert figure7["lat"][rule] == pytest.approx(
            np.mean([tasks[t]["lat"][rule]["accuracy"] for t in FIGURE7_TASKS])
        )
    for metric in METRICS:
        assert figure7["few_shot"][metric] == pytest.approx(
            np.mean(
                [
                    tasks[t]["few_shot"][metric]["accuracy"]
                    for t in FIGURE7_TASKS
                ]
            )
        )
    assert tasks == {
        task: _task_file(finished, task)["summary"] for task in TASKS
    }


def _loaders(
    monkeypatch: pytest.MonkeyPatch, asked: list[str]
) -> dict[str, Any]:
    """Stands in for the dataset loaders, recording what each was asked."""
    data = _data()

    def loader(name: str, rows: Any) -> Callable[..., Any]:
        def load(**kwargs: Any) -> Any:
            asked.append(f"{name}{kwargs}")
            return rows

        return load

    fakes = {
        "load_openbookqa": loader("obqa", data["obqa"]),
        "load_commonsenseqa": loader("csqa", data["csqa"]),
        "load_race": loader("race", data["race"]),
        "load_truthfulqa": loader("tqa", data["tqa"]),
    }

    def load_arc(**kwargs: Any) -> Any:
        asked.append(f"arc{kwargs}")
        easy = kwargs.get("config") == "ARC-Easy"
        return data["arc_easy" if easy else "arc_challenge"]

    fakes["load_arc"] = load_arc
    for name, fake in fakes.items():
        monkeypatch.setattr(r8_qa_correctness, name, fake)
    return fakes


def test_run_loads_the_model_in_float32_and_only_the_tasks_data(
    qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models: list[tuple[str, dict[str, Any]]] = []

    def load_model(key: str, **kwargs: Any) -> LoadedModel:
        models.append((key, kwargs))
        return qwen3

    asked: list[str] = []
    _loaders(monkeypatch, asked)
    monkeypatch.setattr(r8_qa_correctness, "load_model", load_model)
    config = _config(tasks=("arc_easy", "obqa", "arc_challenge"))
    run_dir = RunDir.create(config, out_root=tmp_path)
    run(config, run_dir)

    assert models == [("tiny", {"dtype": torch.float32})]
    assert asked == ["arc{'config': 'ARC-Easy'}", "obqa{}", "arc{}"]
    summary = cast(dict[str, Any], run_dir.read_json("summary"))
    assert list(summary["tasks"]) == ["arc_easy", "obqa", "arc_challenge"]


def test_run_refuses_a_bad_config_before_loading_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[str] = []
    _loaders(monkeypatch, asked)

    def load_model(key: str, **kwargs: Any) -> LoadedModel:
        raise AssertionError("the model must not load")

    monkeypatch.setattr(r8_qa_correctness, "load_model", load_model)
    config = _config(tasks=("obqa", "obqa"))

    with pytest.raises(ValueError, match="more than once"):
        run(config, RunDir.create(config, out_root=tmp_path))
    assert asked == []


def test_plot_draws_the_test_curves_of_every_task(
    finished: RunDir, tmp_path: Path
) -> None:
    paths = plot(finished, tmp_path / "figs")

    assert sorted(p.name for p in paths) == sorted(
        f"curves_{task}.{ext}"
        for task in TASKS
        for ext in ("pdf", "png", "svg")
    )
    assert all(p.stat().st_size > 0 for p in paths)
