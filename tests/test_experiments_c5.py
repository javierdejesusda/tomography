"""Tests for the C5 galleries: the lie about a world fact and the emotions."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.control.generate import Completion
from tomography.datasets.emotions import EMOTIONS
from tomography.experiments.c5_gallery import emotion_rows
from tomography.experiments.c5_gallery import fact_prompt
from tomography.experiments.c5_gallery import fact_trial
from tomography.experiments.c5_gallery import lie_rows
from tomography.experiments.steering import Arm
from tomography.experiments.steering import Trial
from tomography.knowledge import DEFAULT_SHOTS
from tomography.knowledge import few_shot_prompt
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
