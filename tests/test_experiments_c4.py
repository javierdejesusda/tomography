"""Tests for the C4 honesty control on TruthfulQA MC1."""

from collections.abc import Mapping
from collections.abc import Sequence
import itertools
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.activations import collect_activations
from tomography.datasets.records import ChoiceExample
from tomography.datasets.records import Example
from tomography.datasets.records import Split
from tomography.datasets.stimulus import StimulusSet
from tomography.experiments import c4_honesty_control
from tomography.experiments.base import derive_seed
from tomography.experiments.c4_honesty_control import TRUTHFUL
from tomography.experiments.c4_honesty_control import UNTRUTHFUL
from tomography.experiments.c4_honesty_control import calibration_texts
from tomography.experiments.c4_honesty_control import choice_offsets
from tomography.experiments.c4_honesty_control import difference_reader
from tomography.experiments.c4_honesty_control import honesty_layers
from tomography.experiments.c4_honesty_control import honesty_readers
from tomography.experiments.c4_honesty_control import honesty_stimuli
from tomography.experiments.c4_honesty_control import mc1_correct
from tomography.experiments.c4_honesty_control import mc1_inputs
from tomography.experiments.c4_honesty_control import render_prompt
from tomography.experiments.c4_honesty_control import select_coeff
from tomography.experiments.sweep import SweepResult
from tomography.experiments.sweep import chat_format_for
from tomography.judges import mc1_accuracy
from tomography.model import BosMode
from tomography.model import LoadedModel
from tomography.templates import PAPER_TEMPLATES
from tomography.templates import ChatFormat
from tomography.templates import FunctionTemplate


@pytest.mark.parametrize(
    ("n_layers", "expected"),
    [
        (3, (2, 3)),
        (12, (4, 5, 6, 7, 9, 10, 11, 12)),
        (24, (7, 9, 11, 14, 16, 18, 21, 23)),
        (28, (8, 11, 13, 16, 19, 21, 24, 26)),
        (32, (9, 12, 15, 18, 21, 24, 27, 30)),
        (36, (10, 13, 17, 20, 23, 27, 30, 34)),
        (48, (13, 17, 22, 27, 31, 35, 40, 45)),
    ],
    ids=["l3", "gpt2", "l24", "qwen3-1.7b", "l32", "l36", "l48"],
)
def test_honesty_layers_scale_the_paper_blocks_to_the_depth(
    n_layers: int, expected: tuple[int, ...]
) -> None:
    assert honesty_layers(n_layers) == expected


@pytest.mark.parametrize("n_layers", range(1, 61))
def test_honesty_layers_stay_inside_the_decoder(n_layers: int) -> None:
    layers = honesty_layers(n_layers)
    assert layers == tuple(sorted(set(layers)))
    assert layers[0] >= 1
    assert layers[-1] <= n_layers


@pytest.mark.parametrize("n_layers", [0, -3])
def test_honesty_layers_reject_a_model_without_blocks(n_layers: int) -> None:
    with pytest.raises(ValueError, match="n_layers"):
        honesty_layers(n_layers)


def _items() -> list[ChoiceExample]:
    return [
        ChoiceExample(
            uid="a",
            question="Is it?",
            choices=("yes", "no"),
            correct=0,
            split="train",
        ),
        ChoiceExample(
            uid="b",
            question="Why?",
            choices=("x", "y", "z"),
            correct=2,
            split="test",
        ),
    ]


def _template_chat() -> ChatFormat:
    def render(messages: list[dict[str, str]]) -> str:
        return f"<user>{messages[-1]['content']}</user><assistant>"

    return ChatFormat("template", render_fn=render)


def test_the_instructions_are_the_poles_of_the_honesty_control_template() -> (
    None
):
    template = PAPER_TEMPLATES["honesty_control"]
    assert isinstance(template, FunctionTemplate)
    assert f"Give {template.pos_personas[0]} answer." == TRUTHFUL
    assert f"Give {template.neg_personas[0]} answer." == UNTRUTHFUL


def test_render_prompt_moves_the_plain_trailing_space_to_the_answer() -> None:
    assert render_prompt(ChatFormat("plain"), "Is it?") == (
        "USER: Is it? ASSISTANT:",
        " ",
    )


def test_render_prompt_keeps_a_template_prompt_whole() -> None:
    assert render_prompt(_template_chat(), "Is it?") == (
        "<user>Is it?</user><assistant>",
        "",
    )


def test_render_prompt_adds_nothing_without_tags() -> None:
    assert render_prompt(ChatFormat("none"), "Is it?") == ("Is it?", "")


def test_mc1_inputs_give_one_pair_per_choice_in_question_order() -> None:
    prompts, answers = mc1_inputs(ChatFormat("plain"), _items())
    assert (
        prompts
        == ["USER: Is it? ASSISTANT:"] * 2 + ["USER: Why? ASSISTANT:"] * 3
    )
    assert answers == [" yes", " no", " x", " y", " z"]


def test_mc1_inputs_put_the_instruction_inside_the_user_turn() -> None:
    prompts, answers = mc1_inputs(ChatFormat("plain"), _items(), TRUTHFUL)
    assert (
        prompts
        == ["USER: Is it? Give a truthful answer. ASSISTANT:"] * 2
        + ["USER: Why? Give a truthful answer. ASSISTANT:"] * 3
    )
    assert answers == [" yes", " no", " x", " y", " z"]


def test_mc1_inputs_of_a_template_chat_have_no_lead() -> None:
    prompts, answers = mc1_inputs(_template_chat(), _items(), UNTRUTHFUL)
    assert (
        prompts[0]
        == "<user>Is it? Give an untruthful answer.</user><assistant>"
    )
    assert (
        prompts[-1] == "<user>Why? Give an untruthful answer.</user><assistant>"
    )
    assert answers == ["yes", "no", "x", "y", "z"]


def test_choice_offsets_cut_the_flat_scores_into_questions() -> None:
    offsets = choice_offsets(_items())
    assert offsets.dtype == np.int64
    assert offsets.tolist() == [0, 2, 5]
    assert choice_offsets([]).tolist() == [0]


def _random_scores(
    kind: str,
) -> tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.int64]]:
    rng = np.random.default_rng(3)
    sizes = rng.integers(2, 6, size=40)
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    flat = rng.normal(size=int(offsets[-1])).astype(np.float32)
    if kind == "zeros":
        flat = np.zeros_like(flat)
    elif kind == "rounded":
        flat = np.round(flat)
    return flat, offsets, rng.integers(0, sizes).astype(np.int64)


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("kind", ["random", "zeros", "rounded"])
def test_mc1_correct_averages_to_the_mc1_accuracy_of_the_judges(
    kind: str, seed: int
) -> None:
    flat, offsets, correct = _random_scores(kind)
    hits = mc1_correct(flat, offsets, correct, seed=seed)
    split = [flat[a:b] for a, b in itertools.pairwise(offsets)]
    assert hits.dtype == np.bool_
    assert hits.shape == (40,)
    assert float(hits.mean()) == mc1_accuracy(split, correct, seed=seed)


def test_mc1_correct_marks_the_questions_whose_best_choice_is_right() -> None:
    flat = np.array([0.1, 0.9, 0.5, 0.2, 0.1], dtype=np.float32)
    offsets = np.array([0, 2, 5], dtype=np.int64)
    hits = mc1_correct(flat, offsets, np.array([1, 0]), seed=0)
    assert hits.tolist() == [True, True]
    hits = mc1_correct(flat, offsets, np.array([0, 1]), seed=0)
    assert hits.tolist() == [False, False]


def test_mc1_correct_breaks_ties_among_the_tied_maxima_only() -> None:
    flat = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    offsets = np.array([0, 3], dtype=np.int64)
    for seed in range(20):
        assert not mc1_correct(flat, offsets, np.array([2]), seed=seed)[0]


@pytest.mark.parametrize(
    ("flat", "offsets", "correct", "message"),
    [
        ([0.1, 0.2, 0.3], [0, 2, 4], [0, 0], "offsets"),
        ([0.1, 0.2, 0.3], [1, 3], [0], "offsets"),
        ([0.1, 0.2, 0.3], [0, 3, 3], [0, 0], "no choices"),
        ([0.1, np.nan, 0.3], [0, 3], [0], "NaN"),
        ([0.1, 0.2, 0.3], [0, 3], [3], "correct"),
        ([0.1, 0.2, 0.3], [0, 1, 3], [0], "correct"),
    ],
    ids=[
        "short-offsets",
        "offsets-start",
        "empty-question",
        "nan",
        "out-of-range",
        "wrong-length",
    ],
)
def test_mc1_correct_rejects_inconsistent_inputs(
    flat: Sequence[float],
    offsets: Sequence[int],
    correct: Sequence[int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        mc1_correct(
            np.asarray(flat, dtype=np.float32),
            np.asarray(offsets, dtype=np.int64),
            np.asarray(correct, dtype=np.int64),
            seed=0,
        )


def test_select_coeff_takes_the_best_grid_point() -> None:
    assert select_coeff([0.2, 0.5, 0.4], [-0.1, 0.0, 0.1]) == 1
    assert select_coeff([0.2, 0.3, 0.9], [-0.1, 0.0, 0.1]) == 2


def test_select_coeff_breaks_ties_to_the_smallest_magnitude() -> None:
    assert select_coeff([0.5, 0.4, 0.5], [-0.2, 0.0, 0.1]) == 2
    assert select_coeff([0.5, 0.5, 0.5], [-0.2, 0.0, 0.1]) == 1


def test_select_coeff_breaks_equal_magnitudes_to_the_first_in_grid_order() -> (
    None
):
    assert select_coeff([0.6, 0.1, 0.6], [-0.1, 0.0, 0.1]) == 0
    assert select_coeff([0.6, 0.1, 0.6], [0.1, 0.0, -0.1]) == 0


@pytest.mark.parametrize(
    ("accuracy", "coeffs"),
    [
        ([0.1, 0.2], [0.0, 0.1, 0.2]),
        ([], []),
        ([0.1, float("nan")], [0.0, 0.1]),
        ([0.1, float("inf")], [0.0, 0.1]),
        ([0.1, 0.2], [0.0, float("nan")]),
    ],
    ids=["shape", "empty", "nan-accuracy", "inf-accuracy", "nan-coeff"],
)
def test_select_coeff_rejects_a_bad_grid(
    accuracy: Sequence[float], coeffs: Sequence[float]
) -> None:
    with pytest.raises(ValueError, match=r"grids|finite"):
        select_coeff(accuracy, coeffs)


SPLIT_SIZES: dict[Split, int] = {"train": 40, "val": 16, "test": 16}


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


def test_honesty_stimuli_render_both_poles_of_every_row() -> None:
    rows = _alpaca()
    by_uid = {row.uid: row for row in rows}
    stimuli = honesty_stimuli(rows, ChatFormat("plain"), n_pairs=5, seed=0)
    assert len(stimuli.texts) == 10
    assert stimuli.labels.tolist() == [1, 0] * 5
    assert stimuli.uids[0::2] == stimuli.uids[1::2]
    for text, span, label, uid in zip(
        stimuli.texts,
        stimuli.response_spans,
        stimuli.labels,
        stimuli.uids,
        strict=True,
    ):
        row = by_uid[uid]
        pole = TRUTHFUL if label else UNTRUTHFUL
        assert text == (
            f"USER: {row.meta['instruction']} {pole} ASSISTANT: {row.text}"
        )
        assert span is not None
        assert text[span[0] : span[1]] == row.text


def test_honesty_stimuli_draw_the_rows_from_the_seed() -> None:
    rows = _alpaca()
    chat = ChatFormat("plain")
    first = honesty_stimuli(rows, chat, n_pairs=5, seed=1)
    again = honesty_stimuli(rows, chat, n_pairs=5, seed=1)
    other = honesty_stimuli(rows, chat, n_pairs=5, seed=2)
    assert first.texts == again.texts
    assert first.texts != other.texts


def test_honesty_stimuli_use_every_row_without_a_count() -> None:
    rows = _alpaca()[:7]
    stimuli = honesty_stimuli(rows, ChatFormat("plain"), n_pairs=None, seed=0)
    assert len(stimuli.texts) == 14


def test_honesty_stimuli_reject_more_pairs_than_rows() -> None:
    with pytest.raises(ValueError, match="n_pairs"):
        honesty_stimuli(_alpaca()[:3], ChatFormat("plain"), n_pairs=4, seed=0)


def _record_sweeps(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    real = c4_honesty_control.reading_sweep

    def spy(
        loaded: LoadedModel,
        train_sets: Sequence[StimulusSet],
        eval_sets: Mapping[str, StimulusSet],
        **kwargs: Any,
    ) -> SweepResult:
        calls.append(
            {
                "train_sets": list(train_sets),
                "eval_sets": dict(eval_sets),
                **kwargs,
            }
        )
        return real(loaded, train_sets, eval_sets, **kwargs)

    monkeypatch.setattr(c4_honesty_control, "reading_sweep", spy)
    return calls


def test_honesty_readers_train_on_train_rows_and_validate_on_val_rows(
    tiny_gpt2: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _record_sweeps(monkeypatch)
    rows = _alpaca()
    honesty_readers(
        tiny_gpt2,
        rows,
        (1, 3),
        n_seeds=2,
        n_pairs=6,
        n_eval_pairs=5,
        method="logistic",
        seed=3,
        bos="none",
        batch_size=4,
    )
    call = calls[0]
    chat = chat_format_for(tiny_gpt2)
    for split, stimuli in [
        *[("train", s) for s in call["train_sets"]],
        ("val", call["eval_sets"]["val"]),
    ]:
        allowed = {row.uid for row in rows if row.split == split}
        assert set(stimuli.uids) <= allowed
    train = [row for row in rows if row.split == "train"]
    val = [row for row in rows if row.split == "val"]
    assert [s.texts for s in call["train_sets"]] == [
        honesty_stimuli(
            train, chat, n_pairs=6, seed=derive_seed(3, "train", i)
        ).texts
        for i in range(2)
    ]
    assert list(call["eval_sets"]) == ["val"]
    assert (
        call["eval_sets"]["val"].texts
        == honesty_stimuli(
            val, chat, n_pairs=5, seed=derive_seed(3, "eval")
        ).texts
    )
    assert call["position"] == "response"
    assert call["layers"] == (1, 3)
    assert call["method"] == "logistic"
    assert call["bos"] == "none"
    assert call["batch_size"] == 4


def test_honesty_readers_fit_one_reader_per_seed_at_the_layers(
    tiny_arch: LoadedModel,
) -> None:
    result = honesty_readers(
        tiny_arch,
        _alpaca(),
        (1, 3),
        n_seeds=2,
        n_pairs=6,
        n_eval_pairs=None,
        method="logistic",
        seed=0,
    )
    assert result.layers == (1, 3)
    assert result.eval_names == ("val",)
    assert len(result.readers) == 2
    assert all(reader.layers == (1, 3) for reader in result.readers)
    assert result.per_seed("val").shape == (3, 2, 2)


def test_the_first_reader_does_not_depend_on_the_number_of_seeds(
    tiny_arch: LoadedModel,
) -> None:
    def readers(n_seeds: int) -> SweepResult:
        return honesty_readers(
            tiny_arch,
            _alpaca(),
            (1, 2, 3),
            n_seeds=n_seeds,
            n_pairs=6,
            n_eval_pairs=5,
            method="logistic",
            seed=4,
        )

    one, two = readers(1).readers[0], readers(2).readers[0]
    np.testing.assert_array_equal(one.directions, two.directions)
    np.testing.assert_array_equal(one.means, two.means)
    np.testing.assert_array_equal(one.signs, two.signs)


@pytest.mark.parametrize("bos", [None, "none"], ids=["default-bos", "no-bos"])
def test_difference_reader_points_from_the_untruthful_to_the_truthful_prompt(
    tiny_arch: LoadedModel, bos: BosMode | None
) -> None:
    layers = (1, 2, 3)
    chat = chat_format_for(tiny_arch)
    texts = [
        render_prompt(chat, TRUTHFUL)[0],
        render_prompt(chat, UNTRUTHFUL)[0],
    ]
    acts = collect_activations(
        tiny_arch, texts, layers=layers, position="last", bos=bos
    )
    gap = acts.data[:, 0] - acts.data[:, 1]
    reader = difference_reader(tiny_arch, layers, bos=bos)
    assert reader.method == "prompt_difference"
    assert reader.layers == layers
    np.testing.assert_allclose(
        reader.directions,
        gap / np.linalg.norm(gap, axis=1, keepdims=True),
        atol=1e-5,
    )
    np.testing.assert_array_equal(reader.signs, np.ones(3))
    np.testing.assert_array_equal(reader.means, np.zeros_like(gap))
    scores = reader.score(acts.data)
    assert (scores[:, 0] > scores[:, 1]).all()


def test_calibration_texts_draw_only_the_test_split() -> None:
    rows = _alpaca()
    allowed = {row.text for row in rows if row.split == "test"}
    texts = calibration_texts(rows, 5, seed=0)
    assert len(texts) == 5
    assert len(set(texts)) == 5
    assert set(texts) <= allowed


def test_calibration_texts_follow_the_seed() -> None:
    rows = _alpaca()
    assert calibration_texts(rows, 8, seed=1) == calibration_texts(
        rows, 8, seed=1
    )
    assert calibration_texts(rows, 8, seed=1) != calibration_texts(
        rows, 8, seed=2
    )


def test_calibration_texts_return_the_whole_split_when_it_is_short() -> None:
    rows = _alpaca()
    texts = calibration_texts(rows, 100, seed=0)
    assert sorted(texts) == sorted(r.text for r in rows if r.split == "test")


def test_calibration_texts_reject_a_count_below_one() -> None:
    with pytest.raises(ValueError, match="n must be"):
        calibration_texts(_alpaca(), 0, seed=0)


def test_calibration_texts_reject_a_missing_test_split() -> None:
    rows = [row for row in _alpaca() if row.split != "test"]
    with pytest.raises(ValueError, match="test split"):
        calibration_texts(rows, 5, seed=0)
