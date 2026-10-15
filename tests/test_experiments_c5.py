"""Tests for the C5 galleries: the lie about a world fact and the emotions."""

from collections.abc import Sequence
import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.cli import load_experiment
from tomography.control.generate import Completion
from tomography.datasets.alpaca import load_alpaca
from tomography.datasets.emotions import EMOTIONS
from tomography.datasets.records import Example
from tomography.datasets.records import QAExample
from tomography.datasets.records import Split
from tomography.datasets.sources import OfflineCacheMissError
from tomography.datasets.triviaqa import load_triviaqa
from tomography.experiments import c5_gallery
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.c4_honesty_control import calibration_texts
from tomography.experiments.c4_honesty_control import honesty_layers
from tomography.experiments.c5_gallery import TABLE_COLUMNS
from tomography.experiments.c5_gallery import Config
from tomography.experiments.c5_gallery import emotion_rows
from tomography.experiments.c5_gallery import execute
from tomography.experiments.c5_gallery import fact_prompt
from tomography.experiments.c5_gallery import fact_trial
from tomography.experiments.c5_gallery import lie_rows
from tomography.experiments.c5_gallery import plot
from tomography.experiments.c5_gallery import run
from tomography.experiments.effects import read_trials
from tomography.experiments.effects import write_trials
from tomography.experiments.steering import Arm
from tomography.experiments.steering import GenSettings
from tomography.experiments.steering import Trial
from tomography.experiments.sweep import SweepResult
from tomography.experiments.sweep import chat_format_for
from tomography.knowledge import DEFAULT_SHOTS
from tomography.knowledge import few_shot_prompt
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.reading.metrics import wilson_ci
from tomography.reading.reader import RepReader
from tomography.templates import ChatFormat


def _template_chat() -> ChatFormat:
    def render(messages: list[dict[str, str]]) -> str:
        return f"<user>{messages[-1]['content']}</user><assistant>"

    return ChatFormat("template", render_fn=render)


def test_the_plain_prompt_is_the_few_shot_prompt_of_every_demonstration() -> (
    None
):
    assert fact_prompt(ChatFormat("plain"), "Who?") == few_shot_prompt(
        "Who?", DEFAULT_SHOTS
    )


@pytest.mark.parametrize("n_shots", [0, 1, 7, len(DEFAULT_SHOTS)])
def test_the_plain_prompt_keeps_the_first_demonstrations(n_shots: int) -> None:
    assert fact_prompt(ChatFormat("plain"), "Who?", n_shots) == few_shot_prompt(
        "Who?", DEFAULT_SHOTS[:n_shots]
    )


def test_the_template_prompt_is_one_chat_turn_without_demonstrations() -> None:
    prompt = fact_prompt(_template_chat(), "Who?", 3)
    assert prompt == "<user>Who?</user><assistant>"


@pytest.mark.parametrize("n_shots", [-1, len(DEFAULT_SHOTS) + 1])
@pytest.mark.parametrize("kind", ["plain", "template"])
def test_a_shot_count_outside_the_demonstrations_raises(
    kind: str, n_shots: int
) -> None:
    chat = ChatFormat("plain") if kind == "plain" else _template_chat()
    with pytest.raises(ValueError, match="n_shots"):
        fact_prompt(chat, "Who?", n_shots)


def _completion(text: str, n_tokens: int | None = None) -> Completion:
    count = len(text.split()) if n_tokens is None else n_tokens
    return Completion(
        text=text, token_ids=tuple(range(1, count + 1)), finished=False
    )


def _canned(rows: list[list[Completion]], name: str = "canned") -> Arm:
    return Arm(name=name, generate=lambda prompts: rows)


def test_a_fact_trial_scores_one_zero_and_nan() -> None:
    rows = [
        [_completion(" Paris\nQ: next")],
        [_completion(" Rome")],
        [_completion("   ", 1)],
        [_completion("", 0)],
    ]
    trial = fact_trial(
        _canned(rows),
        ["p0", "p1", "p2", "p3"],
        [("Paris", "City of Light"), ("Madrid",), ("Oslo",), ("Oslo",)],
    )
    assert trial.name == "canned"
    assert trial.prompts == ("p0", "p1", "p2", "p3")
    assert trial.scores.shape == (4, 1)
    np.testing.assert_array_equal(
        trial.scores[:, 0], [1.0, 0.0, np.nan, np.nan]
    )


def test_a_fact_trial_reads_the_answer_up_to_the_first_newline() -> None:
    rows = [[_completion(" Rome\nParis")]]
    trial = fact_trial(_canned(rows), ["p"], [("Paris",)])
    assert trial.scores[0, 0] == 0.0


def test_a_fact_trial_scores_every_sample_of_a_prompt() -> None:
    rows = [
        [_completion(" Paris"), _completion(" Rome")],
        [_completion(" Rome"), _completion(" Madrid")],
    ]
    trial = fact_trial(_canned(rows), ["p0", "p1"], [("Paris",), ("Madrid",)])
    assert trial.scores.shape == (2, 2)
    np.testing.assert_array_equal(trial.scores, [[1.0, 0.0], [0.0, 1.0]])
    assert trial.completions == ((" Paris", " Rome"), (" Rome", " Madrid"))


def test_a_fact_trial_needs_one_answer_list_per_prompt() -> None:
    def forbidden(prompts: list[str]) -> list[list[Completion]]:
        raise AssertionError("generated before the lengths were checked")

    arm = Arm(name="x", generate=forbidden)
    with pytest.raises(ValueError, match="answers"):
        fact_trial(arm, ["p0", "p1"], [("Paris",)])


def _trial(name: str, scores: NDArray[np.float64], prefix: str = "p") -> Trial:
    n_prompts, n_samples = scores.shape[:2]
    return Trial(
        name=name,
        prompts=tuple(f"{prefix}{i}" for i in range(n_prompts)),
        completions=tuple(
            tuple(f"{name} #{p}.{s}" for s in range(n_samples))
            for p in range(n_prompts)
        ),
        lengths=np.ones((n_prompts, n_samples), dtype=np.int64),
        scores=scores.astype(np.float32),
    )


NAN = np.nan
NONE_FACTS = np.array([[1, 0], [0, 1], [1, 0], [NAN, 1], [1, 1], [1, 0]])
HONEST_FACTS = np.array([[1, 1], [0, 0], [1, 1], [1, 1], [0, 1], [1, 1]])
LIE_FACTS = np.array([[1, 0], [0, 1], [0, 1], [0, 0], [NAN, 0], [0, 1]])


def _facts() -> tuple[Trial, Trial, Trial]:
    return (
        _trial("none", NONE_FACTS),
        _trial("reading/0/0.1", HONEST_FACTS),
        _trial("reading/0/-0.1", LIE_FACTS),
    )


def _lie_row(
    kind: str, arm: str, trial: Trial, prompt: int, correct: bool | None
) -> list[object]:
    return [
        "lie",
        kind,
        arm,
        prompt,
        0,
        f"p{prompt}",
        trial.completions[prompt][0],
        correct,
        None,
        None,
    ]


def test_the_unfiltered_lie_rows_are_the_first_questions_at_sample_zero() -> (
    None
):
    none, honest, lie = _facts()
    rows = lie_rows(none, honest, lie, 2)[:6]
    assert rows == [
        _lie_row("unfiltered", "none", none, 0, True),
        _lie_row("unfiltered", "honest", honest, 0, True),
        _lie_row("unfiltered", "lie", lie, 0, True),
        _lie_row("unfiltered", "none", none, 1, False),
        _lie_row("unfiltered", "honest", honest, 1, False),
        _lie_row("unfiltered", "lie", lie, 1, False),
    ]


def test_an_unfiltered_row_without_a_verdict_has_no_correct_flag() -> None:
    none, honest, lie = _facts()
    rows = lie_rows(none, honest, lie, 4)
    assert rows[9] == _lie_row("unfiltered", "none", none, 3, None)
    assert rows[11] == _lie_row("unfiltered", "lie", lie, 3, False)


def test_the_curated_lie_rows_are_the_known_questions_that_flip() -> None:
    none, honest, lie = _facts()
    rows = lie_rows(none, honest, lie, 1)
    assert rows[3:] == [
        _lie_row("curated", "none", none, 2, True),
        _lie_row("curated", "honest", honest, 2, True),
        _lie_row("curated", "lie", lie, 2, False),
    ]


def test_the_curated_lie_rows_keep_the_pool_order_and_stop_at_n_show() -> None:
    none, honest, lie = _facts()
    rows = lie_rows(none, honest, lie, 4)
    curated = [row for row in rows if row[1] == "curated"]
    assert [row[3] for row in curated] == [2, 2, 2, 5, 5, 5]
    assert [row[2] for row in curated[:3]] == ["none", "honest", "lie"]


def test_a_missing_verdict_is_neither_known_nor_wrong() -> None:
    none, honest, lie = _facts()
    rows = lie_rows(none, honest, lie, 6)
    curated = {row[3] for row in rows if row[1] == "curated"}
    assert 3 not in curated
    assert 4 not in curated


def test_no_curated_lie_row_when_nothing_flips() -> None:
    none, honest, _ = _facts()
    rows = lie_rows(none, honest, none, 3)
    assert {row[1] for row in rows} == {"unfiltered"}
    assert len(rows) == 9


def test_lie_rows_need_trials_on_the_same_prompts() -> None:
    none, honest, _ = _facts()
    other = _trial("reading/0/-0.1", LIE_FACTS, prefix="q")
    with pytest.raises(ValueError, match="prompts"):
        lie_rows(none, honest, other, 2)


N_PROMPTS = 4
N_SAMPLES = 2


def _none_emotions() -> Trial:
    values = np.arange(N_PROMPTS * N_SAMPLES * 12) / 128
    return _trial("none", values.reshape(N_PROMPTS, N_SAMPLES, 12))


def _emotion_trial(name: str, primary: Sequence[Sequence[float]]) -> Trial:
    column = EMOTIONS.index(name.split("/")[0])
    scores = np.full((N_PROMPTS, N_SAMPLES, 12), -1.0)
    scores[..., column] = np.array(primary)
    scores[..., 6 + column] = np.array(primary) + 0.25
    return _trial(name, scores)


READER_0 = [[0.25, 0.875], [0.5, 0.125], [0.375, 0.375], [NAN, 0.625]]
READER_1 = [[0.875, 0.25], [0.5, 0.375], [0.125, 0.625], [0.75, 0.8125]]
DECOY = [[2.0, 2.0]] * N_PROMPTS
FLAT = [[0.0, 0.0]] * N_PROMPTS


def _c2_trials() -> list[Trial]:
    return [
        _none_emotions(),
        _emotion_trial("surprise/0/0.1", FLAT),
        _emotion_trial("sadness/1/0.1", READER_1),
        _emotion_trial("sadness/0/0.1", READER_0),
        _emotion_trial("sadness/0/0.2", DECOY),
        _emotion_trial("sadness/0/-0.1", DECOY),
    ]


def _find(trials: list[Trial], name: str) -> Trial:
    return next(trial for trial in trials if trial.name == name)


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
        f"p{prompt}",
        trial.completions[prompt][sample],
        None,
        float(trial.scores[prompt, sample, column]),
        float(trial.scores[prompt, sample, 6 + column]),
    ]


def _of(
    rows: list[list[object]], gallery: str, kind: str
) -> list[list[object]]:
    return [row for row in rows if row[:2] == [gallery, kind]]


def test_emotion_galleries_follow_the_order_of_the_emotions() -> None:
    rows = emotion_rows(_c2_trials(), 0.1, 1)
    assert list(dict.fromkeys(row[0] for row in rows)) == [
        "emotion:sadness",
        "emotion:surprise",
    ]


def test_the_unfiltered_emotion_rows_pair_none_with_the_first_reader() -> None:
    trials = _c2_trials()
    none = _find(trials, "none")
    reader = _find(trials, "sadness/0/0.1")
    rows = _of(emotion_rows(trials, 0.1, 2), "emotion:sadness", "unfiltered")
    assert rows == [
        _emotion_row("sadness", "unfiltered", none, 0, 0),
        _emotion_row("sadness", "unfiltered", reader, 0, 0),
        _emotion_row("sadness", "unfiltered", none, 1, 0),
        _emotion_row("sadness", "unfiltered", reader, 1, 0),
    ]


def test_the_curated_emotion_rows_are_the_top_scores_at_the_strength() -> None:
    trials = _c2_trials()
    zero = _find(trials, "sadness/0/0.1")
    one = _find(trials, "sadness/1/0.1")
    rows = _of(emotion_rows(trials, 0.1, 15), "emotion:sadness", "curated")
    assert rows == [
        _emotion_row("sadness", "curated", trial, prompt, sample)
        for trial, prompt, sample in [
            (zero, 0, 1),
            (one, 0, 0),
            (one, 3, 1),
            (one, 3, 0),
            (zero, 3, 1),
            (one, 2, 1),
            (zero, 1, 0),
            (one, 1, 0),
            (zero, 2, 0),
            (zero, 2, 1),
            (one, 1, 1),
            (zero, 0, 0),
            (one, 0, 1),
            (zero, 1, 1),
            (one, 2, 0),
        ]
    ]


def test_the_curated_emotion_rows_stop_at_n_show() -> None:
    rows = emotion_rows(_c2_trials(), 0.1, 3)
    curated = _of(rows, "emotion:sadness", "curated")
    assert [row[2:5] for row in curated] == [
        ["sadness/0/0.1", 0, 1],
        ["sadness/1/0.1", 0, 0],
        ["sadness/1/0.1", 3, 1],
    ]


def test_equal_scores_curate_the_first_reader_prompt_and_sample() -> None:
    rows = emotion_rows(_c2_trials(), 0.1, 1)
    flat = _of(rows, "emotion:surprise", "curated")
    assert [row[2:5] for row in flat] == [["surprise/0/0.1", 0, 0]]


def test_emotion_rows_need_the_none_trial() -> None:
    trials = [t for t in _c2_trials() if t.name != "none"]
    with pytest.raises(ValueError, match="none"):
        emotion_rows(trials, 0.1, 2)


def test_emotion_rows_need_the_twelve_judge_columns() -> None:
    trials = _c2_trials()
    trials[2] = _trial("sadness/1/0.1", np.zeros((N_PROMPTS, N_SAMPLES, 2)))
    with pytest.raises(ValueError, match="12"):
        emotion_rows(trials, 0.1, 2)


@pytest.mark.parametrize(
    ("strength", "missing"),
    [(0.05, "sadness/0/0.05"), (0.1, "sadness/0/0.1")],
    ids=["strength-not-run", "first-reader-absent"],
)
def test_emotion_rows_need_the_first_reader_at_the_strength(
    strength: float, missing: str
) -> None:
    trials = _c2_trials()
    if strength == 0.1:
        trials = [t for t in trials if t.name != "sadness/0/0.1"]
    with pytest.raises(ValueError, match=missing):
        emotion_rows(trials, strength, 2)


SPLIT_SIZES: dict[Split, int] = {"train": 12, "val": 8, "test": 8}


def _alpaca() -> list[Example]:
    rows = []
    for split, size in SPLIT_SIZES.items():
        for i in range(size):
            rows.append(
                Example(
                    uid=f"alpaca-{split}-{i}",
                    text=f"The {split} answer {i} is a short reply.",
                    label=-1,
                    split=split,
                    meta={"instruction": f"Name item {i} of the {split} set."},
                )
            )
    return rows


FACTS = [
    QAExample(
        uid=f"tq-{i:02d}",
        question=f"Question {i:02d}?",
        answers=(f"Answer{i:02d}", f"Alias{i:02d}"),
        kind="real",
        split="train",
    )
    for i in range(14)
]


def _config(**overrides: Any) -> Config:
    base = Config(
        model="tiny",
        n_pairs=6,
        n_eval_pairs=6,
        n_pool=12,
        n_show=3,
        n_shots=1,
        max_new_tokens=6,
        n_calibration=8,
        batch_size=8,
    )
    return dataclasses.replace(base, **overrides)


def _pool(n_pool: int, seed: int = 0) -> list[int]:
    order = np.random.default_rng(derive_seed(seed, "pool")).permutation(
        len(FACTS)
    )
    return [int(i) for i in order[:n_pool]]


def test_config_defaults_and_round_trip() -> None:
    config = Config()
    assert config.experiment == "c5"
    assert config.model == "gpt2"
    assert config.method == "logistic"
    assert (config.n_pairs, config.n_eval_pairs) == (128, 128)
    assert config.strength == 0.1
    assert (config.n_pool, config.n_show) == (200, 4)
    assert config.n_shots == len(DEFAULT_SHOTS) == 15
    assert (config.num_samples, config.max_new_tokens) == (1, 24)
    assert config.c2_run is None
    assert Config.from_dict(config.to_dict()) == config


def _c2_run(
    tmp_path: Path, strength: float = 0.2, drop: str | None = None
) -> RunDir:
    config = dataclasses.replace(_config(), experiment="c2")
    run_dir = RunDir.create(config, out_root=tmp_path / "c2")
    trials = [_none_emotions()] + [
        _emotion_trial(
            f"{emotion}/{seed}/{coeff}", READER_0 if seed == 0 else READER_1
        )
        for emotion in ("sadness", "surprise")
        for seed in (0, 1)
        for coeff in ("0.1", "0.2")
    ]
    write_trials(run_dir, "trials", [t for t in trials if t.name != drop])
    run_dir.write_json("summary", {"strength": strength})
    return run_dir


def _read(run_dir: RunDir, name: str) -> dict[str, Any]:
    data = run_dir.read_json(name)
    assert isinstance(data, dict)
    return data


NONE_OK = {0, 2, 3, 4, 6, 8, 9, 11, 12}
HONEST_OK = {0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 13}
LIE_OK = {0, 3, 5, 6, 12, 13}
LIE_EMPTY = {2, 9}


def _expected_scores(
    n_pool: int, correct: set[int], empty: set[int] | None = None
) -> NDArray[np.float64]:
    return np.array(
        [
            np.nan if i in (empty or set()) else float(i in correct)
            for i in _pool(n_pool)
        ]
    )


def _canned_arm(name: str, correct: set[int], empty: set[int]) -> Arm:
    def generate(prompts: Sequence[str]) -> list[list[Completion]]:
        rows = []
        for prompt in prompts:
            i = next(i for i, q in enumerate(FACTS) if q.question in prompt)
            if i in empty:
                rows.append([Completion("", (), True)])
            elif i in correct:
                text = f" {FACTS[i].answers[0]}\n"
                rows.append([Completion(text, (1, 2), True)])
            else:
                rows.append([Completion(" wrong\n", (1, 2), True)])
        return rows

    return Arm(name, generate)


def _plant(
    monkeypatch: pytest.MonkeyPatch, none_ok: set[int]
) -> dict[str, Any]:
    calls: dict[str, Any] = {"readers": [], "norms": [], "arms": []}

    def readers(
        loaded: LoadedModel,
        alpaca: Sequence[Example],
        layers: Sequence[int],
        **kwargs: Any,
    ) -> SweepResult:
        calls["readers"].append({"layers": tuple(layers), **kwargs})
        shape = (len(layers), 4)
        reader = RepReader.from_arrays(
            "dummy", layers, np.ones(shape), np.zeros(shape)
        )
        calls["reader"] = reader
        return SweepResult(
            tuple(layers),
            ("val",),
            np.zeros((1, 3, kwargs["n_seeds"], len(layers))),
            (reader,),
        )

    def norms(
        loaded: LoadedModel,
        texts: Sequence[str],
        layers: Sequence[int],
        **kwargs: Any,
    ) -> dict[int, float]:
        calls["norms"].append(
            {"texts": list(texts), "layers": tuple(layers), **kwargs}
        )
        return {layer: 2.0 for layer in layers}

    def fake_plain_arm(
        loaded: LoadedModel, settings: GenSettings, **kwargs: Any
    ) -> Arm:
        calls["arms"].append({"settings": settings, **kwargs})
        return _canned_arm("none", none_ok, set())

    def fake_reader_arm(
        loaded: LoadedModel,
        settings: GenSettings,
        reader: RepReader,
        **kwargs: Any,
    ) -> Arm:
        calls["arms"].append({"settings": settings, "reader": reader, **kwargs})
        honest = kwargs["coeff"] > 0
        return _canned_arm(
            kwargs["name"],
            HONEST_OK if honest else LIE_OK,
            set() if honest else LIE_EMPTY,
        )

    monkeypatch.setattr(c5_gallery, "honesty_readers", readers)
    monkeypatch.setattr(c5_gallery, "mean_residual_norms", norms)
    monkeypatch.setattr(c5_gallery, "plain_arm", fake_plain_arm)
    monkeypatch.setattr(c5_gallery, "reader_arm", fake_reader_arm)
    return calls


def _run_planted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loaded: LoadedModel,
    *,
    emotions: bool = True,
    none_ok: set[int] = NONE_OK,
    **overrides: Any,
) -> tuple[RunDir, dict[str, Any]]:
    calls = _plant(monkeypatch, none_ok)
    config = _config(**overrides)
    run_dir = RunDir.create(config, out_root=tmp_path)
    c2 = _c2_run(tmp_path) if emotions else None
    calls["c2"] = c2
    execute(config, run_dir, loaded, _alpaca(), FACTS, c2)
    return run_dir, calls


def _forbid_work(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("work started before the configuration passed")

    for name in (
        "honesty_readers",
        "mean_residual_norms",
        "plain_arm",
        "reader_arm",
    ):
        monkeypatch.setattr(c5_gallery, name, forbidden)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"strength": 0.0}, "strength"),
        ({"strength": -0.1}, "strength"),
        ({"n_show": 0}, "n_show"),
        ({"n_pool": 0}, "n_pool"),
        ({"n_pool": 15}, "n_pool"),
        ({"n_shots": 16}, "n_shots"),
        ({"layers": (0, 1)}, "layers"),
        ({"layers": (1, 4)}, "layers"),
    ],
    ids=[
        "zero-strength",
        "negative-strength",
        "nothing-to-show",
        "empty-pool",
        "pool-beyond-the-questions",
        "sixteen-shots",
        "layer-zero",
        "layer-beyond-depth",
    ],
)
def test_execute_rejects_a_bad_configuration_before_any_work(
    tiny_gpt2: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
    message: str,
) -> None:
    _forbid_work(monkeypatch)
    config = _config(**overrides)
    run_dir = RunDir.create(config, out_root=tmp_path)
    with pytest.raises(ValueError, match=message):
        execute(config, run_dir, tiny_gpt2, _alpaca(), FACTS)
    assert not list(run_dir.path.glob("*.npz"))


@pytest.mark.parametrize(
    ("c2_strength", "drop"),
    [(0.2, "sadness/0/0.2"), (0.3, None)],
    ids=["first-reader-absent", "strength-not-run"],
)
def test_execute_rejects_an_unusable_emotion_run_before_any_work(
    tiny_gpt2: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    c2_strength: float,
    drop: str | None,
) -> None:
    _forbid_work(monkeypatch)
    config = _config()
    run_dir = RunDir.create(config, out_root=tmp_path)
    c2 = _c2_run(tmp_path, strength=c2_strength, drop=drop)
    with pytest.raises(ValueError, match="sadness/0/0"):
        execute(config, run_dir, tiny_gpt2, _alpaca(), FACTS, c2)
    assert not list(run_dir.path.glob("*.npz"))


def test_execute_builds_the_reader_and_the_norms_once(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, calls = _run_planted(
        tmp_path, monkeypatch, tiny_gpt2, bos="none", batch_size=4
    )
    layers = honesty_layers(tiny_gpt2.n_layers)
    assert calls["readers"] == [
        {
            "layers": layers,
            "n_seeds": 1,
            "n_pairs": 6,
            "n_eval_pairs": 6,
            "method": "logistic",
            "seed": 0,
            "bos": "none",
            "batch_size": 4,
        }
    ]
    assert calls["norms"] == [
        {
            "texts": calibration_texts(
                _alpaca(), 8, seed=derive_seed(0, "norms")
            ),
            "layers": layers,
            "bos": "none",
            "batch_size": 4,
        }
    ]


def test_execute_steers_towards_and_away_from_honesty_with_one_reader(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, calls = _run_planted(
        tmp_path,
        monkeypatch,
        tiny_gpt2,
        strength=0.15,
        op="projection",
        mode="absolute",
        mask="all",
        preserve_norm=True,
    )
    layers = honesty_layers(tiny_gpt2.n_layers)
    settings = GenSettings.from_config(
        _config(strength=0.15, mode="absolute", mask="all")
    )
    plain, honest, lie = calls["arms"]
    assert plain == {"settings": settings}
    for arm, coeff, name in (
        (honest, 0.15, "reading/0/0.15"),
        (lie, -0.15, "reading/0/-0.15"),
    ):
        assert arm == {
            "settings": settings,
            "reader": calls["reader"],
            "layers": layers,
            "coeff": coeff,
            "norms": {layer: 2.0 for layer in layers},
            "op": "projection",
            "mode": "absolute",
            "mask": "all",
            "preserve_norm": True,
            "name": name,
        }


def test_execute_stores_the_three_trials_on_the_pool(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _run_planted(tmp_path, monkeypatch, tiny_gpt2)
    chat = chat_format_for(tiny_gpt2)
    trials = read_trials(run_dir, "trials")
    assert [trial.name for trial in trials] == [
        "none",
        "reading/0/0.1",
        "reading/0/-0.1",
    ]
    prompts = tuple(fact_prompt(chat, FACTS[i].question, 1) for i in _pool(12))
    assert all(trial.prompts == prompts for trial in trials)
    for trial, expected in zip(
        trials,
        (
            _expected_scores(12, NONE_OK),
            _expected_scores(12, HONEST_OK),
            _expected_scores(12, LIE_OK, LIE_EMPTY),
        ),
        strict=True,
    ):
        assert trial.scores.shape == (12, 1)
        np.testing.assert_array_equal(trial.scores[:, 0], expected)


def test_execute_summarizes_the_flips(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _run_planted(tmp_path, monkeypatch, tiny_gpt2)
    none = _expected_scores(12, NONE_OK)
    honest = _expected_scores(12, HONEST_OK)
    lie = _expected_scores(12, LIE_OK, LIE_EMPTY)
    n_known = int((none == 1.0).sum())
    n_flipped = int(((none == 1.0) & (lie == 0.0)).sum())
    assert np.isnan(lie).any()
    assert 0 < n_flipped < n_known
    summary = _read(run_dir, "summary")
    assert summary["layers"] == list(honesty_layers(tiny_gpt2.n_layers))
    assert summary["strength"] == 0.1
    assert summary["n_pool"] == 12
    assert summary["correct_none"] == pytest.approx(np.nanmean(none))
    assert summary["correct_honest"] == pytest.approx(np.nanmean(honest))
    assert summary["correct_lie"] == pytest.approx(np.nanmean(lie))
    assert summary["n_known"] == n_known
    assert summary["n_flipped"] == n_flipped
    assert summary["flip_rate"] == pytest.approx(n_flipped / n_known)
    assert summary["flip_ci"] == pytest.approx(
        list(wilson_ci(n_flipped, n_known))
    )
    assert summary["emotions"] == ["sadness", "surprise"]
    assert summary["emotion_strength"] == 0.2
    assert summary["c2_run"] is None


def test_the_flip_rate_is_missing_when_the_plain_model_knows_nothing(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _run_planted(tmp_path, monkeypatch, tiny_gpt2, none_ok=set())
    summary = _read(run_dir, "summary")
    assert summary["n_known"] == 0
    assert summary["n_flipped"] == 0
    assert summary["flip_rate"] is None
    assert summary["flip_ci"] is None
    assert summary["correct_none"] == 0.0
    lie = [row for row in _read(run_dir, "table")["rows"] if row[0] == "lie"]
    assert {row[1] for row in lie} == {"unfiltered"}


def test_execute_tables_the_lie_gallery(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _run_planted(tmp_path, monkeypatch, tiny_gpt2)
    chat = chat_format_for(tiny_gpt2)
    table = _read(run_dir, "table")
    assert table["columns"] == TABLE_COLUMNS
    rows = [row for row in table["rows"] if row[0] == "lie"]
    none = _expected_scores(12, NONE_OK)
    lie = _expected_scores(12, LIE_OK, LIE_EMPTY)
    order = _pool(12)
    first = FACTS[order[0]]
    assert rows[0] == [
        "lie",
        "unfiltered",
        "none",
        0,
        0,
        fact_prompt(chat, first.question, 1),
        f" {first.answers[0]}\n" if none[0] == 1.0 else " wrong\n",
        bool(none[0] == 1.0),
        None,
        None,
    ]
    unfiltered = [row for row in rows if row[1] == "unfiltered"]
    assert [row[3] for row in unfiltered] == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert [row[2] for row in unfiltered[:3]] == ["none", "honest", "lie"]
    honest = _expected_scores(12, HONEST_OK)
    flips = [j for j in range(12) if none[j] == 1.0 and lie[j] == 0.0][:3]
    curated = [row for row in rows if row[1] == "curated"]
    assert [row[3] for row in curated] == [j for j in flips for _ in range(3)]
    assert [row[7] for row in curated[:3]] == [
        True,
        bool(honest[flips[0]] == 1.0),
        False,
    ]


def test_execute_tables_the_emotion_galleries_of_the_emotion_run(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, calls = _run_planted(tmp_path, monkeypatch, tiny_gpt2)
    rows = _read(run_dir, "table")["rows"]
    expected = emotion_rows(read_trials(calls["c2"], "trials"), 0.2, 3)
    assert [row for row in rows if row[0] != "lie"] == expected
    assert {row[2] for row in expected if row[1] == "unfiltered"} == {
        "none",
        "sadness/0/0.2",
        "surprise/0/0.2",
    }


def test_execute_without_an_emotion_run_has_only_the_lie_gallery(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _run_planted(tmp_path, monkeypatch, tiny_gpt2, emotions=False)
    assert {row[0] for row in _read(run_dir, "table")["rows"]} == {"lie"}
    summary = _read(run_dir, "summary")
    assert summary["emotions"] == []
    assert summary["emotion_strength"] is None


def _real_run(
    tmp_path: Path,
    loaded: LoadedModel,
    *,
    emotions: bool = True,
    **overrides: Any,
) -> RunDir:
    config = _config(**overrides)
    run_dir = RunDir.create(config, out_root=tmp_path)
    c2 = _c2_run(tmp_path) if emotions else None
    execute(config, run_dir, loaded, _alpaca(), FACTS, c2)
    return run_dir


def test_execute_writes_the_files_of_the_experiment(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _real_run(tmp_path, tiny_arch)
    assert sorted(path.name for path in run_dir.path.iterdir()) == [
        "config.json",
        "meta.json",
        "summary.json",
        "table.json",
        "trials.json",
        "trials.npz",
    ]


def test_execute_tables_nine_rows_per_gallery_on_a_random_model(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _real_run(tmp_path, tiny_arch)
    rows = _read(run_dir, "table")["rows"]
    assert len(rows) == 27
    assert [row[0] for row in rows] == (
        ["lie"] * 9 + ["emotion:sadness"] * 9 + ["emotion:surprise"] * 9
    )
    assert {row[1] for row in rows[:9]} == {"unfiltered"}


def test_execute_generates_the_three_trials_on_the_pool(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _real_run(tmp_path, tiny_arch)
    assert _read(run_dir, "trials")["names"] == [
        "none",
        "reading/0/0.1",
        "reading/0/-0.1",
    ]
    assert run_dir.read_arrays("trials")["scores"].shape == (3, 12, 1)


def test_a_strong_steer_changes_the_completions(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _real_run(tmp_path, tiny_arch, strength=3.0, emotions=False)
    none, honest, lie = read_trials(run_dir, "trials")
    assert honest.name == "reading/0/3"
    assert lie.name == "reading/0/-3"
    assert honest.completions != none.completions
    assert lie.completions != none.completions
    assert honest.completions != lie.completions


def test_execute_is_deterministic(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    first = _real_run(tmp_path / "a", tiny_gpt2)
    second = _real_run(tmp_path / "b", tiny_gpt2)
    assert first.read_json("table") == second.read_json("table")
    assert first.read_json("summary") == second.read_json("summary")
    assert first.read_json("trials") == second.read_json("trials")


def _hand_made_run(tmp_path: Path) -> RunDir:
    run_dir = RunDir.create(_config(), out_root=tmp_path)
    prompt = few_shot_prompt("Who | what?\nNow", DEFAULT_SHOTS[:1])
    rows = [
        [
            *("lie", "unfiltered", "none", 0, 0),
            *(prompt, " Paris | Rome\nno", True, None, None),
        ],
        [
            *("lie", "curated", "lie", 0, 0),
            *(prompt, " wrong", False, None, None),
        ],
        [
            *("emotion:sadness", "unfiltered", "sadness/0/0.1", 1, 0),
            *("Tell me\n\nmore", " I am sad", None, 0.75, 0.5),
        ],
        [
            *("emotion:sadness", "curated", "sadness/0/0.1", 2, 1),
            *("p", " ok", None, None, 0.25),
        ],
    ]
    run_dir.write_json("table", {"columns": TABLE_COLUMNS, "rows": rows})
    return run_dir


def test_plot_writes_a_section_and_a_table_per_gallery(
    tmp_path: Path,
) -> None:
    run_dir = _hand_made_run(tmp_path)
    paths = plot(run_dir, tmp_path / "figs")
    assert paths == [tmp_path / "figs" / "gallery.md"]
    lines = paths[0].read_text(encoding="utf-8").splitlines()
    assert [line for line in lines if line.startswith("## ")] == [
        "## lie",
        "## emotion:sadness",
    ]
    header = "| kind | arm | prompt_index | sample | prompt | completion |"
    assert [line for line in lines if line.startswith("| kind")] == [
        f"{header} correct |",
        f"{header} score | score_check |",
    ]


def test_plot_flattens_escapes_and_cuts_the_cells(tmp_path: Path) -> None:
    run_dir = _hand_made_run(tmp_path)
    path = plot(run_dir, tmp_path / "figs")[0]
    text = path.read_text(encoding="utf-8")
    assert DEFAULT_SHOTS[0][0] not in text
    for line in text.splitlines():
        assert not line or line[0] in "#|"
    assert (
        "| unfiltered | none | 0 | 0 | Q: Who \\| what? Now A: |"
        " Paris \\| Rome no | yes |"
    ) in text
    assert (
        "| curated | lie | 0 | 0 | Q: Who \\| what? Now A: | wrong | no |"
        in text
    )
    assert (
        "| unfiltered | sadness/0/0.1 | 1 | 0 | Tell me more | I am sad |"
        in text
    )
    assert " 0.75 | 0.5 |" in text
    assert "| curated | sadness/0/0.1 | 2 | 1 | p | ok |  | 0.25 |" in text


def test_plot_writes_the_gallery_of_a_real_run(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _real_run(tmp_path, tiny_gpt2)
    path = plot(run_dir, tmp_path / "figs")[0]
    headings = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("## ")
    ]
    assert headings == ["## lie", "## emotion:sadness", "## emotion:surprise"]


def test_the_cli_discovers_the_experiment() -> None:
    experiment = load_experiment("c5")
    assert experiment.config_cls is Config
    assert experiment.run is run
    assert experiment.plot is plot


def test_run_loads_the_model_the_data_and_the_emotion_run(
    tiny_gpt2: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[str] = []

    def fake_load_model(key: str) -> LoadedModel:
        asked.append(key)
        return tiny_gpt2

    monkeypatch.setattr(c5_gallery, "load_model", fake_load_model)
    monkeypatch.setattr(c5_gallery, "load_alpaca", _alpaca)
    monkeypatch.setattr(c5_gallery, "load_triviaqa", lambda: FACTS)
    c2 = _c2_run(tmp_path)
    config = _config(c2_run=str(c2.path))
    run_dir = RunDir.create(config, out_root=tmp_path)
    run(config, run_dir)
    assert asked == ["tiny"]
    summary = _read(run_dir, "summary")
    assert summary["c2_run"] == str(c2.path)
    assert summary["emotions"] == ["sadness", "surprise"]
    assert summary["emotion_strength"] == 0.2


def test_run_without_an_emotion_run_skips_the_emotion_galleries(
    tiny_gpt2: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(c5_gallery, "load_model", lambda key: tiny_gpt2)
    monkeypatch.setattr(c5_gallery, "load_alpaca", _alpaca)
    monkeypatch.setattr(c5_gallery, "load_triviaqa", lambda: FACTS)
    config = _config()
    run_dir = RunDir.create(config, out_root=tmp_path)
    run(config, run_dir)
    summary = _read(run_dir, "summary")
    assert summary["c2_run"] is None
    assert summary["emotions"] == []


@pytest.mark.slow
def test_c5_on_gpt2_small_prints_the_flip_rate(tmp_path: Path) -> None:
    try:
        facts = load_triviaqa(offline=True)
        alpaca = load_alpaca(offline=True)
    except OfflineCacheMissError:
        pytest.skip("the TriviaQA or Alpaca cache is missing")
    config = Config(n_pool=40)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, load_model("gpt2"), alpaca, facts)
    summary = _read(run_dir, "summary")
    print(f"flip rate {summary['flip_rate']} of {summary['n_known']} known")
    for row in _read(run_dir, "table")["rows"][: 3 * config.n_show]:
        print(row)
    rate = summary["flip_rate"]
    assert rate is None or 0.0 <= rate <= 1.0
