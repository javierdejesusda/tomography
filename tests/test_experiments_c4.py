"""Tests for the C4 honesty control on TruthfulQA MC1."""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import itertools
from pathlib import Path
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.activations import collect_activations
from tomography.cli import load_experiment
from tomography.control.contrast import contrast_logprobs
from tomography.control.controllers import mean_residual_norms
from tomography.datasets.alpaca import load_alpaca
from tomography.datasets.records import ChoiceExample
from tomography.datasets.records import Example
from tomography.datasets.records import Split
from tomography.datasets.sources import OfflineCacheMissError
from tomography.datasets.stimulus import StimulusSet
from tomography.datasets.truthfulqa import load_truthfulqa
from tomography.experiments import c4_honesty_control
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.c4_honesty_control import TRUTHFUL
from tomography.experiments.c4_honesty_control import UNTRUTHFUL
from tomography.experiments.c4_honesty_control import Config
from tomography.experiments.c4_honesty_control import calibration_texts
from tomography.experiments.c4_honesty_control import choice_offsets
from tomography.experiments.c4_honesty_control import difference_reader
from tomography.experiments.c4_honesty_control import execute
from tomography.experiments.c4_honesty_control import fitting_rows
from tomography.experiments.c4_honesty_control import honesty_layers
from tomography.experiments.c4_honesty_control import honesty_readers
from tomography.experiments.c4_honesty_control import honesty_stimuli
from tomography.experiments.c4_honesty_control import mc1_correct
from tomography.experiments.c4_honesty_control import mc1_inputs
from tomography.experiments.c4_honesty_control import plot
from tomography.experiments.c4_honesty_control import render_prompt
from tomography.experiments.c4_honesty_control import resolve_layers
from tomography.experiments.c4_honesty_control import run
from tomography.experiments.c4_honesty_control import select_coeff
from tomography.experiments.steering import Arm
from tomography.experiments.steering import GenSettings
from tomography.experiments.steering import LogprobFn
from tomography.experiments.steering import plain_arm
from tomography.experiments.steering import reader_arm
from tomography.experiments.sweep import SweepResult
from tomography.experiments.sweep import chat_format_for
from tomography.judges import mc1_accuracy
from tomography.model import BosMode
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.reading.metrics import bootstrap_ci
from tomography.reading.metrics import wilson_ci
from tomography.reading.reader import RepReader
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


def test_resolve_layers_default_to_the_honesty_layers() -> None:
    assert resolve_layers(None, 12) == honesty_layers(12)


def test_resolve_layers_sort_and_deduplicate_the_request() -> None:
    assert resolve_layers((9, 3, 9), 12) == (3, 9)


@pytest.mark.parametrize(
    "requested",
    [(), (0, 1), (1, 13)],
    ids=["empty", "layer-zero", "beyond-depth"],
)
def test_resolve_layers_reject_layers_outside_the_decoder(
    requested: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="layers"):
        resolve_layers(requested, 12)


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


def _long_row(split: Split) -> Example:
    return Example(
        uid=f"alpaca-{split}-long",
        text=" ".join(["word"] * 400),
        label=-1,
        split=split,
        meta={"instruction": "Name a long item."},
    )


def test_honesty_readers_drop_the_train_rows_beyond_the_model_positions(
    tiny_arch: LoadedModel,
) -> None:
    result = honesty_readers(
        tiny_arch,
        [*_alpaca(), _long_row("train")],
        (1, 2),
        n_seeds=2,
        n_pairs=SPLIT_SIZES["train"],
        n_eval_pairs=5,
        method="logistic",
        seed=0,
    )
    assert len(result.readers) == 2


def test_honesty_readers_drop_the_val_rows_beyond_the_model_positions(
    tiny_arch: LoadedModel,
) -> None:
    result = honesty_readers(
        tiny_arch,
        [*_alpaca(), _long_row("val")],
        (1, 2),
        n_seeds=1,
        n_pairs=6,
        n_eval_pairs=None,
        method="logistic",
        seed=0,
    )
    assert len(result.readers) == 1


def test_the_readers_do_not_see_the_rows_that_do_not_fit(
    tiny_arch: LoadedModel,
) -> None:
    def readers(rows: Sequence[Example]) -> SweepResult:
        return honesty_readers(
            tiny_arch,
            rows,
            (1, 2, 3),
            n_seeds=2,
            n_pairs=6,
            n_eval_pairs=5,
            method="logistic",
            seed=4,
        )

    clean = readers(_alpaca())
    long = readers([*_alpaca(), _long_row("train"), _long_row("val")])
    np.testing.assert_array_equal(clean.accuracy, long.accuracy)
    for a, b in zip(clean.readers, long.readers, strict=True):
        np.testing.assert_array_equal(a.directions, b.directions)
        np.testing.assert_array_equal(a.means, b.means)
        np.testing.assert_array_equal(a.signs, b.signs)


def test_a_row_that_fills_the_positions_is_kept_unless_a_bos_is_added(
    tiny_gpt2: LoadedModel,
) -> None:
    chat = chat_format_for(tiny_gpt2)
    rows = [row for row in _alpaca() if row.split == "train"][:3]
    pairs = honesty_stimuli(rows, chat, n_pairs=None, seed=0)
    ids = tiny_gpt2.tokenizer(list(pairs.texts), add_special_tokens=False)
    tiny_gpt2.model.config.max_position_embeddings = max(
        len(row) for row in ids["input_ids"]
    )
    assert fitting_rows(tiny_gpt2, rows, chat, bos="none") == rows
    assert len(fitting_rows(tiny_gpt2, rows, chat, bos="prepend")) < len(rows)


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


SPLIT_OFFSET: dict[Split, int] = {"train": 0, "val": 1, "test": 2}


def _questions() -> list[ChoiceExample]:
    counts: dict[Split, int] = {"train": 14, "val": 6, "test": 10}
    items = []
    for split, count in counts.items():
        for i in range(count):
            k = 2 + (i + SPLIT_OFFSET[split]) % 3
            items.append(
                ChoiceExample(
                    uid=f"q-{split}-{i:02d}",
                    question=f"{split} q{i:02d}?",
                    choices=tuple(f"c{j}" for j in range(k)),
                    correct=i % k,
                    split=split,
                )
            )
    return items


def _config(**overrides: Any) -> Config:
    base = Config(
        model="tiny",
        n_seeds=2,
        n_pairs=6,
        n_eval_pairs=6,
        coeffs=(-0.1, 0.0, 0.1),
        alphas=(-0.25, 0.0, 0.25),
        n_boot=50,
        n_calibration=8,
        batch_size=8,
    )
    return dataclasses.replace(base, **overrides)


def test_config_defaults_and_round_trip() -> None:
    config = Config()
    assert config.experiment == "c4"
    assert config.model == "gpt2"
    assert config.method == "logistic"
    assert config.n_seeds == 10
    assert (config.n_pairs, config.n_eval_pairs) == (128, 128)
    assert config.coeffs == (-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2)
    assert config.alphas == (-0.5, -0.25, 0.0, 0.25, 0.5)
    assert config.n_boot == 2000
    assert Config.from_dict(config.to_dict()) == config


def _forbid_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("work started before the configuration passed")

    for name in (
        "honesty_readers",
        "difference_reader",
        "mean_residual_norms",
        "plain_arm",
        "reader_arm",
        "contrast_logprobs",
    ):
        monkeypatch.setattr(c4_honesty_control, name, forbidden)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"coeffs": (0.1, 0.2)}, "coeffs"),
        ({"coeffs": (0.0,)}, "coeffs"),
        ({"coeffs": (-0.1, 0.0, 0.1, 0.1)}, "coeffs"),
        ({"coeffs": (-0.0, 0.0, 0.1)}, "coeffs"),
        ({"coeffs": (0.0, -0.0)}, "coeffs"),
        ({"alphas": (0.25, 0.5)}, "alphas"),
        ({"alphas": (-0.25, 0.0, 0.25, 0.25)}, "alphas"),
        ({"alphas": (0.25, 0.0, -0.0)}, "alphas"),
        ({"layers": (0, 1)}, "layers"),
        ({"layers": (1, 4)}, "layers"),
        ({"layers": ()}, "layers"),
        ({"n_seeds": 0}, "n_seeds"),
        ({"n_boot": 0}, "n_boot"),
    ],
    ids=[
        "coeffs-without-zero",
        "coeffs-only-zero",
        "coeffs-repeated",
        "coeffs-signed-zero-repeated",
        "coeffs-only-signed-zeros",
        "alphas-without-zero",
        "alphas-repeated",
        "alphas-signed-zero-repeated",
        "layer-zero",
        "layer-beyond-depth",
        "layers-empty",
        "no-seeds",
        "no-resamples",
    ],
)
def test_execute_rejects_a_bad_configuration_before_any_scoring(
    tiny_gpt2: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
    message: str,
) -> None:
    _forbid_scoring(monkeypatch)
    config = _config(**overrides)
    run_dir = RunDir.create(config, out_root=tmp_path)
    with pytest.raises(ValueError, match=message):
        execute(config, run_dir, tiny_gpt2, _questions(), _alpaca())
    assert not list(run_dir.path.glob("*.npz"))


@pytest.mark.parametrize("missing", ["test", "train+val"])
def test_execute_needs_selection_and_test_questions(
    tiny_gpt2: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    _forbid_scoring(monkeypatch)
    dropped = {"test"} if missing == "test" else {"train", "val"}
    questions = [q for q in _questions() if q.split not in dropped]
    config = _config()
    run_dir = RunDir.create(config, out_root=tmp_path)
    with pytest.raises(ValueError, match="questions"):
        execute(config, run_dir, tiny_gpt2, questions, _alpaca())


QUESTIONS = _questions()
SELECTION = [q for q in QUESTIONS if q.split != "test"]
TEST = [q for q in QUESTIONS if q.split == "test"]
PLAN: dict[str, tuple[int, int]] = {
    "none": (4, 3),
    "prompting+1": (0, 10),
    "prompting-1": (14, 0),
    "difference/-0.1": (2, 10),
    "difference/0.1": (6, 2),
    "reading/0/-0.1": (3, 9),
    "reading/0/0.1": (4, 9),
    "reading/1/-0.1": (8, 5),
    "reading/1/0.1": (8, 7),
    "contrast/-0.25": (3, 1),
    "contrast/0.25": (5, 6),
}
SCORE_ROWS = list(PLAN)
VAL_ACCURACY = np.array([[0.5, 0.6], [0.7, 0.9]])


def _rank(item: ChoiceExample) -> int:
    return int(item.uid.rsplit("-", 1)[1])


def _pattern(selection: int, test: int) -> Callable[[ChoiceExample], bool]:
    def hit(item: ChoiceExample) -> bool:
        return _rank(item) < (test if item.split == "test" else selection)

    return hit


def _scorer(
    pick: Callable[[str], Callable[[ChoiceExample], bool]],
) -> LogprobFn:
    def score(
        prompts: Sequence[str], continuations: Sequence[str]
    ) -> NDArray[np.float32]:
        values = np.zeros(len(prompts), dtype=np.float32)
        for n, (prompt, continuation) in enumerate(
            zip(prompts, continuations, strict=True)
        ):
            item = next(
                q
                for q in QUESTIONS
                if prompt.startswith(f"USER: {q.question} ")
            )
            chosen = item.correct
            if not pick(prompt)(item):
                chosen = (item.correct + 1) % len(item.choices)
            values[n] = float(int(continuation.strip()[1:]) == chosen)
        return values

    return score


def _plant(
    monkeypatch: pytest.MonkeyPatch,
    plan: Mapping[str, tuple[int, int]] = PLAN,
) -> dict[str, list[dict[str, Any]]]:
    calls: dict[str, list[dict[str, Any]]] = {
        "readers": [],
        "difference": [],
        "norms": [],
        "arms": [],
        "contrast": [],
    }

    def hit(name: str) -> Callable[[ChoiceExample], bool]:
        return _pattern(*plan[name])

    def plain(prompt: str) -> Callable[[ChoiceExample], bool]:
        if TRUTHFUL in prompt:
            return hit("prompting+1")
        return hit("prompting-1" if UNTRUTHFUL in prompt else "none")

    def dummy(layers: Sequence[int]) -> RepReader:
        shape = (len(layers), 4)
        return RepReader.from_arrays(
            "dummy", layers, np.ones(shape), np.zeros(shape)
        )

    def readers(
        loaded: LoadedModel,
        alpaca: Sequence[Example],
        layers: Sequence[int],
        **kwargs: Any,
    ) -> SweepResult:
        calls["readers"].append({"layers": tuple(layers), **kwargs})
        n_seeds = kwargs["n_seeds"]
        accuracy = np.zeros((1, 3, n_seeds, len(layers)))
        accuracy[0, 0] = VAL_ACCURACY[:n_seeds, : len(layers)]
        return SweepResult(
            tuple(layers),
            ("val",),
            accuracy,
            tuple(dummy(layers) for _ in range(n_seeds)),
        )

    def difference(
        loaded: LoadedModel, layers: Sequence[int], **kwargs: Any
    ) -> RepReader:
        calls["difference"].append({"layers": tuple(layers), **kwargs})
        return dummy(layers)

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
        return Arm("none", lambda prompts: [], _scorer(plain))

    def fake_reader_arm(
        loaded: LoadedModel,
        settings: GenSettings,
        reader: RepReader,
        **kwargs: Any,
    ) -> Arm:
        calls["arms"].append({"settings": settings, **kwargs})
        name = kwargs["name"]
        return Arm(name, lambda prompts: [], _scorer(lambda _: hit(name)))

    def fake_contrast(
        loaded: LoadedModel,
        prompts: Sequence[str],
        continuations: Sequence[str],
        positive: Sequence[str],
        negative: Sequence[str],
        **kwargs: Any,
    ) -> NDArray[np.float32]:
        calls["contrast"].append(
            {
                "positive": list(positive),
                "negative": list(negative),
                "prompts": list(prompts),
                "continuations": list(continuations),
                **kwargs,
            }
        )
        name = f"contrast/{kwargs['alpha']:g}"
        return _scorer(lambda _: hit(name))(prompts, continuations)

    monkeypatch.setattr(c4_honesty_control, "honesty_readers", readers)
    monkeypatch.setattr(c4_honesty_control, "difference_reader", difference)
    monkeypatch.setattr(c4_honesty_control, "mean_residual_norms", norms)
    monkeypatch.setattr(c4_honesty_control, "plain_arm", fake_plain_arm)
    monkeypatch.setattr(c4_honesty_control, "reader_arm", fake_reader_arm)
    monkeypatch.setattr(c4_honesty_control, "contrast_logprobs", fake_contrast)
    return calls


def _run_planted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loaded: LoadedModel,
    plan: Mapping[str, tuple[int, int]] = PLAN,
    **overrides: Any,
) -> tuple[RunDir, dict[str, list[dict[str, Any]]]]:
    calls = _plant(monkeypatch, plan)
    config = _config(**overrides)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, loaded, QUESTIONS, _alpaca())
    return run_dir, calls


def _read(run_dir: RunDir, name: str) -> dict[str, Any]:
    data = run_dir.read_json(name)
    assert isinstance(data, dict)
    return data


def _mean(
    hit: Callable[[ChoiceExample], bool], items: list[ChoiceExample]
) -> float:
    return float(np.mean([hit(q) for q in items]))


def _expected_stats(chosen: str) -> list[float]:
    hit, none = _pattern(*PLAN[chosen]), _pattern(*PLAN["none"])
    wins = np.array([hit(q) for q in TEST], dtype=np.float64)
    diff = wins - np.array([none(q) for q in TEST], dtype=np.float64)
    lo, hi = wilson_ci(int(wins.sum()), len(TEST))
    d_lo, d_hi = bootstrap_ci(diff, n_boot=50, seed=derive_seed(0, "boot"))
    return [
        _mean(hit, SELECTION),
        float(wins.mean()),
        lo,
        hi,
        float(diff.mean()),
        float(d_lo),
        float(d_hi),
    ]


CHOSEN = [
    "none",
    "prompting+1",
    "prompting-1",
    "difference/0.1",
    "none",
    "reading/1/-0.1",
    "contrast/0.25",
]


def test_execute_reports_the_table_of_the_planted_arms(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _run_planted(tmp_path, monkeypatch, tiny_gpt2)
    table = _read(run_dir, "table")
    assert table["columns"] == [
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
    assert [row[:3] for row in table["rows"]] == [
        ["none", None, None],
        ["prompting+1", None, None],
        ["prompting-1", None, None],
        ["difference", 0.1, None],
        ["reading", 0.0, 0],
        ["reading", -0.1, 1],
        ["contrast", 0.25, None],
    ]
    for row, chosen in zip(table["rows"], CHOSEN, strict=True):
        assert row[3:] == pytest.approx(_expected_stats(chosen))


def test_execute_reports_the_known_planted_numbers(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _run_planted(tmp_path, monkeypatch, tiny_gpt2)
    rows = _read(run_dir, "table")["rows"]
    assert [row[3] for row in rows] == pytest.approx(
        [0.4, 0.0, 1.0, 0.6, 0.4, 0.7, 0.5]
    )
    assert [row[4] for row in rows] == pytest.approx(
        [0.3, 1.0, 0.0, 0.2, 0.3, 0.5, 0.6]
    )
    assert [row[7] for row in rows] == pytest.approx(
        [0.0, 0.7, -0.3, -0.1, 0.0, 0.2, 0.3]
    )
    assert rows[0][7:] == [0.0, 0.0, 0.0]


def test_execute_selects_each_coefficient_on_the_selection_questions(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _run_planted(tmp_path, monkeypatch, tiny_gpt2)
    summary = _read(run_dir, "summary")
    assert summary["coeff_difference"] == 0.1
    assert summary["coeff_reading"] == [0.0, -0.1]
    assert summary["alpha_contrast"] == 0.25


def test_the_selection_does_not_read_the_test_questions(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    permuted = {
        name: (selection, (3 * test + 4) % 11)
        for name, (selection, test) in PLAN.items()
    }
    first, _ = _run_planted(tmp_path / "a", monkeypatch, tiny_gpt2)
    second, _ = _run_planted(tmp_path / "b", monkeypatch, tiny_gpt2, permuted)
    one, two = _read(first, "summary"), _read(second, "summary")
    for key in ("coeff_difference", "coeff_reading", "alpha_contrast"):
        assert one[key] == two[key]
    rows_one = _read(first, "table")["rows"]
    rows_two = _read(second, "table")["rows"]
    assert [r[:4] for r in rows_one] == [r[:4] for r in rows_two]
    assert [r[4] for r in rows_one] != [r[4] for r in rows_two]


def test_execute_writes_the_planted_dose_curves(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _run_planted(tmp_path, monkeypatch, tiny_gpt2)
    chance = float(np.mean([1 / len(q.choices) for q in TEST]))
    expected = {
        "curves_reading": (
            [-0.1, 0.0, 0.1],
            [
                [[0.3, 0.4, 0.4], [0.7, 0.4, 0.7]],
                [[0.9, 0.3, 0.9], [0.5, 0.3, 0.7]],
            ],
        ),
        "curves_difference": (
            [-0.1, 0.0, 0.1],
            [[[0.2, 0.4, 0.6]], [[1.0, 0.3, 0.2]]],
        ),
        "curves_contrast": (
            [-0.25, 0.0, 0.25],
            [[[0.3, 0.4, 0.5]], [[0.1, 0.3, 0.6]]],
        ),
    }
    for stem, (x, per_seed) in expected.items():
        arrays = run_dir.read_arrays(stem)
        np.testing.assert_allclose(arrays["x"], x)
        np.testing.assert_allclose(arrays["per_seed"], per_seed)
        np.testing.assert_allclose(arrays["mean"], np.mean(per_seed, axis=1))
        sidecar = _read(run_dir, stem)
        assert sidecar["series"] == ["selection questions", "test questions"]
        assert sidecar["ylabel"] == "TruthfulQA MC1"
        assert sidecar["chance"] == pytest.approx(chance)
    assert run_dir.read_arrays("curves_difference")["lo"].tolist() == (
        run_dir.read_arrays("curves_difference")["hi"].tolist()
    )


def test_execute_scores_no_arm_at_grid_point_zero(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, calls = _run_planted(tmp_path, monkeypatch, tiny_gpt2)
    assert sorted(call["coeff"] for call in calls["arms"]) == sorted(
        [-0.1, 0.1] * 3
    )
    assert sorted(call["alpha"] for call in calls["contrast"]) == [-0.25, 0.25]
    assert _read(run_dir, "scores")["rows"] == SCORE_ROWS
    assert run_dir.read_arrays("scores")["values"].shape[0] == 11


def test_execute_stores_the_scores_with_their_questions(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _run_planted(tmp_path, monkeypatch, tiny_gpt2)
    arrays = run_dir.read_arrays("scores")
    n_choices = sum(len(q.choices) for q in QUESTIONS)
    assert arrays["values"].dtype == np.float32
    assert arrays["values"].shape == (11, n_choices)
    assert arrays["offsets"].tolist() == choice_offsets(QUESTIONS).tolist()
    assert arrays["correct"].tolist() == [q.correct for q in QUESTIONS]
    sidecar = _read(run_dir, "scores")
    assert sidecar["uids"] == [q.uid for q in QUESTIONS]
    assert sidecar["test"] == [q.split == "test" for q in QUESTIONS]
    row = SCORE_ROWS.index("difference/0.1")
    hits = mc1_correct(
        arrays["values"][row],
        arrays["offsets"],
        arrays["correct"],
        seed=derive_seed(0, "mc1"),
    )
    expected = [_pattern(*PLAN["difference/0.1"])(q) for q in QUESTIONS]
    assert hits.tolist() == expected


def test_execute_builds_the_readers_the_vector_and_the_norms_once(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, calls = _run_planted(
        tmp_path, monkeypatch, tiny_gpt2, bos="none", batch_size=4
    )
    layers = honesty_layers(tiny_gpt2.n_layers)
    assert calls["readers"] == [
        {
            "layers": layers,
            "n_seeds": 2,
            "n_pairs": 6,
            "n_eval_pairs": 6,
            "method": "logistic",
            "seed": 0,
            "bos": "none",
            "batch_size": 4,
        }
    ]
    assert calls["difference"] == [{"layers": layers, "bos": "none"}]
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


def test_execute_steers_every_vector_arm_the_same_way(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, calls = _run_planted(
        tmp_path,
        monkeypatch,
        tiny_gpt2,
        op="piecewise",
        mode="absolute",
        mask="all",
        preserve_norm=True,
        batch_size=4,
    )
    layers = honesty_layers(tiny_gpt2.n_layers)
    assert len(calls["arms"]) == 6
    for call in calls["arms"]:
        assert call["layers"] == layers
        assert call["norms"] == {layer: 2.0 for layer in layers}
        assert call["op"] == "piecewise"
        assert call["mode"] == "absolute"
        assert call["mask"] == "all"
        assert call["preserve_norm"] is True
        assert call["settings"].batch_size == 4
    assert [call["name"] for call in calls["arms"]] == [
        "difference/-0.1",
        "difference/0.1",
        "reading/0/-0.1",
        "reading/0/0.1",
        "reading/1/-0.1",
        "reading/1/0.1",
    ]


def test_execute_contrasts_the_truthful_and_untruthful_prompts(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, calls = _run_planted(tmp_path, monkeypatch, tiny_gpt2, batch_size=4)
    chat = chat_format_for(tiny_gpt2)
    plain, answers = mc1_inputs(chat, QUESTIONS)
    layers = honesty_layers(tiny_gpt2.n_layers)
    for call in calls["contrast"]:
        assert call["prompts"] == plain
        assert call["continuations"] == answers
        assert call["positive"] == mc1_inputs(chat, QUESTIONS, TRUTHFUL)[0]
        assert call["negative"] == mc1_inputs(chat, QUESTIONS, UNTRUTHFUL)[0]
        assert call["layers"] == layers
        assert call["batch_size"] == 4
        assert call.get("bos") is None


def test_execute_summarizes_the_run(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _run_planted(tmp_path, monkeypatch, tiny_gpt2)
    summary = _read(run_dir, "summary")
    assert set(summary) == {
        "layers",
        "n_select",
        "n_test",
        "chance",
        "reading_val_accuracy",
        "coeff_difference",
        "coeff_reading",
        "alpha_contrast",
        "reading_test_mean",
        "reading_test_min",
        "reading_test_max",
    }
    assert summary["layers"] == list(honesty_layers(tiny_gpt2.n_layers))
    assert (summary["n_select"], summary["n_test"]) == (20, 10)
    assert summary["chance"] == pytest.approx(
        np.mean([1 / len(q.choices) for q in TEST])
    )
    assert summary["reading_val_accuracy"] == pytest.approx([0.6, 0.75])
    assert summary["reading_test_mean"] == pytest.approx(0.4)
    assert summary["reading_test_min"] == pytest.approx(0.3)
    assert summary["reading_test_max"] == pytest.approx(0.5)


def _real_run(tmp_path: Path, loaded: LoadedModel, **overrides: Any) -> RunDir:
    config = _config(**overrides)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, loaded, QUESTIONS, _alpaca())
    return run_dir


def test_execute_writes_the_files_of_the_experiment(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _real_run(tmp_path, tiny_arch)
    assert sorted(path.name for path in run_dir.path.iterdir()) == [
        "config.json",
        "curves_contrast.json",
        "curves_contrast.npz",
        "curves_difference.json",
        "curves_difference.npz",
        "curves_reading.json",
        "curves_reading.npz",
        "meta.json",
        "scores.json",
        "scores.npz",
        "summary.json",
        "table.json",
    ]


def test_execute_scores_the_eleven_arm_rows_on_every_choice(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _real_run(tmp_path, tiny_arch)
    n_choices = sum(len(q.choices) for q in QUESTIONS)
    values = run_dir.read_arrays("scores")["values"]
    assert values.shape == (11, n_choices)
    assert np.isfinite(values).all()
    assert _read(run_dir, "scores")["rows"] == [
        "none",
        "prompting+1",
        "prompting-1",
        "difference/-0.1",
        "difference/0.1",
        "reading/0/-0.1",
        "reading/0/0.1",
        "reading/1/-0.1",
        "reading/1/0.1",
        "contrast/-0.25",
        "contrast/0.25",
    ]
    rows = _read(run_dir, "table")["rows"]
    assert [row[:3] for row in rows][:3] == [
        ["none", None, None],
        ["prompting+1", None, None],
        ["prompting-1", None, None],
    ]
    assert [(row[0], row[2]) for row in rows[3:]] == [
        ("difference", None),
        ("reading", 0),
        ("reading", 1),
        ("contrast", None),
    ]


def test_the_table_agrees_with_the_stored_scores(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _real_run(tmp_path, tiny_arch)
    arrays = run_dir.read_arrays("scores")
    names = _read(run_dir, "scores")["rows"]
    test = np.array(_read(run_dir, "scores")["test"])
    rows = _read(run_dir, "table")["rows"]
    for row in rows[:3]:
        hits = mc1_correct(
            arrays["values"][names.index(row[0])],
            arrays["offsets"],
            arrays["correct"],
            seed=derive_seed(0, "mc1"),
        )
        assert row[3] == pytest.approx(hits[~test].mean())
        assert row[4] == pytest.approx(hits[test].mean())
        assert (row[5], row[6]) == wilson_ci(int(hits[test].sum()), 10)


def test_the_zero_point_of_every_curve_is_the_plain_model(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _real_run(tmp_path, tiny_arch)
    arrays = run_dir.read_arrays("scores")
    test = np.array(_read(run_dir, "scores")["test"])
    hits = mc1_correct(
        arrays["values"][0],
        arrays["offsets"],
        arrays["correct"],
        seed=derive_seed(0, "mc1"),
    )
    for stem in ("curves_reading", "curves_difference", "curves_contrast"):
        per_seed = run_dir.read_arrays(stem)["per_seed"]
        np.testing.assert_allclose(per_seed[0, :, 1], hits[~test].mean())
        np.testing.assert_allclose(per_seed[1, :, 1], hits[test].mean())


def _plain_inputs(
    loaded: LoadedModel,
) -> tuple[list[str], list[str], tuple[int, ...]]:
    prompts, answers = mc1_inputs(chat_format_for(loaded), QUESTIONS)
    return prompts, answers, honesty_layers(loaded.n_layers)


def test_a_zero_coefficient_reader_arm_scores_exactly_like_the_plain_arm(
    tiny_arch: LoadedModel,
) -> None:
    prompts, answers, layers = _plain_inputs(tiny_arch)
    settings = GenSettings(batch_size=8)
    norms = mean_residual_norms(
        tiny_arch, calibration_texts(_alpaca(), 8, seed=0), layers
    )
    arm = reader_arm(
        tiny_arch,
        settings,
        difference_reader(tiny_arch, layers),
        layers=layers,
        coeff=0.0,
        norms=norms,
    )
    plain = plain_arm(tiny_arch, settings)
    assert arm.logprobs is not None
    assert plain.logprobs is not None
    np.testing.assert_array_equal(
        arm.logprobs(prompts, answers), plain.logprobs(prompts, answers)
    )


def test_the_contrast_at_alpha_zero_matches_the_plain_scores(
    tiny_arch: LoadedModel,
) -> None:
    prompts, answers, layers = _plain_inputs(tiny_arch)
    chat = chat_format_for(tiny_arch)
    contrast = contrast_logprobs(
        tiny_arch,
        prompts,
        answers,
        mc1_inputs(chat, QUESTIONS, TRUTHFUL)[0],
        mc1_inputs(chat, QUESTIONS, UNTRUTHFUL)[0],
        layers=layers,
        alpha=0.0,
        batch_size=8,
    )
    plain = plain_arm(tiny_arch, GenSettings(batch_size=8)).logprobs
    assert plain is not None
    np.testing.assert_allclose(contrast, plain(prompts, answers), atol=1e-4)


def test_execute_is_deterministic(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    first = _real_run(tmp_path / "a", tiny_gpt2)
    second = _real_run(tmp_path / "b", tiny_gpt2)
    assert first.read_json("table") == second.read_json("table")
    assert first.read_json("summary") == second.read_json("summary")
    np.testing.assert_array_equal(
        first.read_arrays("scores")["values"],
        second.read_arrays("scores")["values"],
    )


def test_plot_writes_the_curves_and_the_arm_figure(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _real_run(tmp_path, tiny_gpt2)
    paths = plot(run_dir, tmp_path / "figs")
    assert sorted(path.name for path in paths) == [
        f"{stem}.{ext}"
        for stem in (
            "curves_contrast",
            "curves_difference",
            "curves_reading",
            "mc1_arms",
        )
        for ext in ("pdf", "png", "svg")
    ]
    assert all(path.stat().st_size > 0 for path in paths)


def test_the_cli_discovers_the_experiment() -> None:
    experiment = load_experiment("c4")
    assert experiment.config_cls is Config
    assert experiment.run is run
    assert experiment.plot is plot


def test_run_loads_the_model_and_the_data(
    tiny_gpt2: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[str] = []
    shuffles: list[int] = []

    def fake_load_model(key: str) -> LoadedModel:
        asked.append(key)
        return tiny_gpt2

    def fake_load_truthfulqa(*, seed: int = 0) -> list[ChoiceExample]:
        shuffles.append(seed)
        return QUESTIONS

    monkeypatch.setattr(c4_honesty_control, "load_model", fake_load_model)
    monkeypatch.setattr(
        c4_honesty_control, "load_truthfulqa", fake_load_truthfulqa
    )
    monkeypatch.setattr(c4_honesty_control, "load_alpaca", _alpaca)
    config = _config(seed=5)
    run_dir = RunDir.create(config, out_root=tmp_path)
    run(config, run_dir)
    assert asked == ["tiny"]
    assert shuffles == [5]
    assert (run_dir.path / "summary.json").is_file()


@pytest.mark.slow
def test_c4_on_gpt2_small_pilots_the_layers(tmp_path: Path) -> None:
    try:
        questions = load_truthfulqa(offline=True)
        alpaca = load_alpaca(offline=True)
    except OfflineCacheMissError:
        pytest.skip("the TruthfulQA or Alpaca cache is missing")
    config = Config(
        n_seeds=2, coeffs=(-0.1, 0.0, 0.1), alphas=(-0.25, 0.0, 0.25)
    )
    run_dir = RunDir.create(config, out_root=tmp_path)
    started = time.perf_counter()
    execute(config, run_dir, load_model("gpt2"), questions, alpaca)
    passes = 1 + 2 + 2 + 2 * 2 + 2 * 3
    print(f"{(time.perf_counter() - started) / passes:.1f} s per pass")
    table = _read(run_dir, "table")
    for row in table["rows"]:
        print(row)
    assert len(table["rows"]) == 8
    assert all(0.0 <= row[4] <= 1.0 for row in table["rows"])
