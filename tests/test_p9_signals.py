"""Tests for the per-question signals of the P9 reliability study."""

from collections.abc import Mapping
import dataclasses
import math
import time

import numpy as np
from numpy.typing import NDArray
import pytest
import torch

from tomography.datasets.records import MetaValue
from tomography.datasets.records import QAExample
from tomography.hooks import ResidualEdit
from tomography.judges import sequence_logprob
from tomography.knowledge import ItemLabel
from tomography.knowledge import few_shot_prompt
from tomography.model import EncodedBatch
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.p9.pool import LabeledItem
from tomography.p9.signals import NAME_PROMPT
from tomography.p9.signals import SIGNAL_NAMES
from tomography.p9.signals import answer_logprobs
from tomography.p9.signals import consistency
from tomography.p9.signals import first_answer_token
from tomography.p9.signals import item_signals
from tomography.p9.signals import name_logprobs
from tomography.p9.signals import next_token_stats
from tomography.p9.signals import p_true
from tomography.p9.signals import scoring_bos

PROMPTS = [
    "Q: who wrote it?\nA:",
    "Q: where?\nA:",
    "Q: which city is the capital of the country in the north?\nA:",
    "Q: when?\nA:",
    "Q: what?\nA:",
]
SHOTS = (("what is a cat?", "an animal"), ("what is a car?", "a vehicle"))


def _label(
    greedy: str, samples: tuple[str, ...] = (), uid: str = "x"
) -> ItemLabel:
    return ItemLabel(uid, "known", "behavior", greedy, True, samples, 0)


def _item(
    uid: str,
    question: str,
    greedy: str,
    meta: Mapping[str, MetaValue] | None = None,
    samples: tuple[str, ...] = (),
) -> LabeledItem:
    example = QAExample(
        uid=uid,
        question=question,
        answers=(greedy,),
        kind="real",
        split="train",
        meta=meta or {},
    )
    return LabeledItem(example, _label(greedy, samples, uid))


def _items() -> list[LabeledItem]:
    return [
        _item(
            "a",
            "who wrote it?",
            "Paris",
            {"popularity": 1000, "subject": "the cat"},
            ("Paris", "paris", "Rome"),
        ),
        _item("b", "where is the car?", "a vehicle", {"popularity": 0}),
        _item(
            "c",
            "when?",
            "",
            {"popularity": 25, "subject": "the car"},
            ("x",),
        ),
        _item("d", "what is it in the north?", "an animal"),
    ]


def _zero_edit(batch: EncodedBatch) -> Mapping[int, ResidualEdit]:
    return {1: lambda h, info: h + torch.zeros_like(h)}


def _shift_edit(batch: EncodedBatch) -> Mapping[int, ResidualEdit]:
    return {1: lambda h, info: h + torch.linspace(-3.0, 3.0, h.shape[-1])}


def _direct_stats(
    loaded: LoadedModel, prompt: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    batch = loaded.encode([prompt], padding_side="left")
    with torch.inference_mode():
        out = loaded.model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            position_ids=batch.position_ids,
            use_cache=False,
        )
    logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
    return (
        np.array([logp.exp().max().item()]),
        np.array([-(logp.exp() * logp).sum().item()]),
    )


def test_scoring_bos_follows_the_model_default(tiny_arch: LoadedModel) -> None:
    prepending = dataclasses.replace(tiny_arch, default_bos="prepend", bos_id=5)
    plain = dataclasses.replace(tiny_arch, default_bos="none")

    assert scoring_bos(prepending) == 5
    assert scoring_bos(plain) is None


def test_stats_match_a_direct_forward(tiny_arch: LoadedModel) -> None:
    stats = next_token_stats(tiny_arch, PROMPTS[:1])
    max_prob, entropy = _direct_stats(tiny_arch, PROMPTS[0])

    assert stats.target_logprob is None
    np.testing.assert_allclose(stats.max_prob, max_prob, atol=1e-6)
    np.testing.assert_allclose(stats.entropy, entropy, atol=1e-5)


def test_stats_are_probabilities_and_entropies(tiny_arch: LoadedModel) -> None:
    stats = next_token_stats(tiny_arch, PROMPTS)

    assert stats.max_prob.shape == stats.entropy.shape == (len(PROMPTS),)
    assert np.all((stats.max_prob > 0.0) & (stats.max_prob <= 1.0))
    assert np.all(stats.entropy >= 0.0)


def test_stats_do_not_depend_on_the_batch_size(tiny_arch: LoadedModel) -> None:
    single = next_token_stats(tiny_arch, PROMPTS, batch_size=1)
    batched = next_token_stats(tiny_arch, PROMPTS, batch_size=4)

    np.testing.assert_allclose(single.max_prob, batched.max_prob, atol=1e-5)
    np.testing.assert_allclose(single.entropy, batched.entropy, atol=1e-4)


def test_target_logprob_is_negative_and_below_the_top(
    tiny_arch: LoadedModel,
) -> None:
    targets = [first_answer_token(tiny_arch, w) for w in "abcde"]

    stats = next_token_stats(tiny_arch, PROMPTS, targets=targets)

    assert stats.target_logprob is not None
    assert np.all(stats.target_logprob < 0.0)
    assert np.all(stats.target_logprob <= np.log(stats.max_prob) + 1e-6)


def test_target_logprob_does_not_depend_on_the_batch_size(
    tiny_arch: LoadedModel,
) -> None:
    targets = [first_answer_token(tiny_arch, w) for w in "abcde"]

    single = next_token_stats(tiny_arch, PROMPTS, targets=targets, batch_size=1)
    batched = next_token_stats(
        tiny_arch, PROMPTS, targets=targets, batch_size=4
    )

    assert single.target_logprob is not None
    assert batched.target_logprob is not None
    np.testing.assert_allclose(
        single.target_logprob, batched.target_logprob, atol=1e-4
    )


def test_stats_reject_the_wrong_number_of_targets(
    tiny_arch: LoadedModel,
) -> None:
    with pytest.raises(ValueError, match="targets must have one id per prompt"):
        next_token_stats(tiny_arch, PROMPTS, targets=[1, 2])


def test_a_zero_edit_leaves_the_stats_bit_identical(
    tiny_arch: LoadedModel,
) -> None:
    plain = next_token_stats(tiny_arch, PROMPTS)
    edited = next_token_stats(tiny_arch, PROMPTS, edits=_zero_edit)

    np.testing.assert_array_equal(plain.max_prob, edited.max_prob)
    np.testing.assert_array_equal(plain.entropy, edited.entropy)


def test_a_shift_edit_changes_the_stats(tiny_arch: LoadedModel) -> None:
    plain = next_token_stats(tiny_arch, PROMPTS)
    shifted = next_token_stats(tiny_arch, PROMPTS, edits=_shift_edit)

    assert not np.allclose(plain.entropy, shifted.entropy)


def test_stats_follow_the_bos_argument(tiny_arch: LoadedModel) -> None:
    loaded = dataclasses.replace(tiny_arch, default_bos="prepend", bos_id=5)

    default = next_token_stats(loaded, PROMPTS)
    prepend = next_token_stats(loaded, PROMPTS, bos="prepend")
    plain = next_token_stats(loaded, PROMPTS, bos="none")

    np.testing.assert_array_equal(default.entropy, prepend.entropy)
    assert not np.allclose(default.entropy, plain.entropy)


def test_first_answer_token_is_the_first_token_of_the_spaced_answer(
    tiny_arch: LoadedModel,
) -> None:
    ids = tiny_arch.tokenizer(" the cat", add_special_tokens=False)["input_ids"]

    assert len(ids) > 1
    assert first_answer_token(tiny_arch, "the cat") == ids[0]


def test_first_answer_token_rejects_an_answer_without_tokens(
    tiny_arch: LoadedModel,
) -> None:
    class NoTokens:
        def __call__(self, text: str, **kwargs: object) -> dict[str, list[int]]:
            return {"input_ids": []}

    loaded = dataclasses.replace(tiny_arch, tokenizer=NoTokens())

    with pytest.raises(ValueError, match="has no tokens"):
        first_answer_token(loaded, "x")


def test_answer_logprobs_are_negative_sums_with_token_counts(
    tiny_arch: LoadedModel,
) -> None:
    answers = ["the cat", "a"]

    total, counts = answer_logprobs(tiny_arch, PROMPTS[:2], answers)

    assert total.shape == counts.shape == (2,)
    assert np.all(total < 0.0)
    expected = [
        len(tiny_arch.tokenizer(" " + a, add_special_tokens=False)["input_ids"])
        for a in answers
    ]
    assert counts.tolist() == expected
    direct = sequence_logprob(
        tiny_arch.model,
        tiny_arch.tokenizer,
        PROMPTS[:2],
        [" " + a for a in answers],
        batch_size=2,
        bos_id=scoring_bos(tiny_arch),
    )
    np.testing.assert_allclose(total, direct, atol=1e-4)


def test_answer_logprobs_give_an_empty_answer_nan_and_no_tokens(
    tiny_arch: LoadedModel,
) -> None:
    total, counts = answer_logprobs(
        tiny_arch, PROMPTS[:3], ["the cat", "", "a"]
    )

    assert math.isnan(total[1])
    assert counts[1] == 0
    assert np.all(np.isfinite(total[[0, 2]]))
    assert np.all(counts[[0, 2]] > 0)


def test_answer_logprobs_of_only_empty_answers_are_all_nan(
    tiny_arch: LoadedModel,
) -> None:
    total, counts = answer_logprobs(tiny_arch, PROMPTS[:2], ["", ""])

    assert np.all(np.isnan(total))
    assert counts.tolist() == [0, 0]


def test_name_logprobs_score_the_name_after_the_name_prompt(
    tiny_arch: LoadedModel,
) -> None:
    names = ["the cat", "a car"]

    total, counts = name_logprobs(tiny_arch, names)
    expected_total, expected_counts = answer_logprobs(
        tiny_arch, [NAME_PROMPT] * 2, names
    )

    np.testing.assert_array_equal(total, expected_total)
    np.testing.assert_array_equal(counts, expected_counts)


def test_p_true_is_a_probability_from_the_true_false_logits(
    tiny_arch: LoadedModel,
) -> None:
    questions = ["who wrote it?", "where?"]
    answers = ["the cat", "a car"]

    out = p_true(tiny_arch, questions, answers)

    assert out.shape == (2,)
    assert np.all((out > 0.0) & (out < 1.0))
    prompts = [
        f"Question: {q}\nProposed answer: {a}\n"
        "Is the proposed answer correct? Answer True or False.\nAnswer:"
        for q, a in zip(questions, answers, strict=True)
    ]
    kw = {"batch_size": 2, "bos_id": scoring_bos(tiny_arch)}
    yes = sequence_logprob(
        tiny_arch.model, tiny_arch.tokenizer, prompts, [" True"] * 2, **kw
    )
    no = sequence_logprob(
        tiny_arch.model, tiny_arch.tokenizer, prompts, [" False"] * 2, **kw
    )
    expected = 1.0 / (1.0 + np.exp(-(yes.astype(np.float64) - no)))
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_p_true_does_not_depend_on_the_batch_size(
    tiny_arch: LoadedModel,
) -> None:
    questions = ["who wrote it?", "where?", "when?"]
    answers = ["the cat", "a car", "an animal"]

    single = p_true(tiny_arch, questions, answers, batch_size=1)
    batched = p_true(tiny_arch, questions, answers, batch_size=3)

    np.testing.assert_allclose(single, batched, atol=1e-5)


def test_consistency_is_the_share_of_samples_equal_to_the_greedy_answer() -> (
    None
):
    label = _label("The Paris.", ("paris", "Paris!", "Rome"))

    assert consistency(label) == pytest.approx(2 / 3)


def test_consistency_is_nan_without_samples() -> None:
    assert math.isnan(consistency(_label("Paris")))


def test_item_signals_hold_the_nine_arrays(tiny_arch: LoadedModel) -> None:
    items = _items()

    signals = item_signals(tiny_arch, items, shots=SHOTS)

    assert tuple(signals) == SIGNAL_NAMES
    assert all(v.shape == (len(items),) for v in signals.values())


def test_item_signals_leave_nan_only_where_the_value_is_missing(
    tiny_arch: LoadedModel,
) -> None:
    signals = item_signals(tiny_arch, _items(), shots=SHOTS)

    assert np.isnan(signals["popularity"]).tolist() == [
        False,
        True,
        False,
        True,
    ]
    assert np.isnan(signals["name_logprob"]).tolist() == [
        False,
        True,
        False,
        True,
    ]
    assert np.isnan(signals["consistency"]).tolist() == [
        False,
        True,
        False,
        True,
    ]
    for name in ("answer_logprob", "mean_token_logprob", "p_true"):
        assert np.isnan(signals[name]).tolist() == [False, False, True, False]
    for name in ("first_token_prob", "neg_entropy", "length"):
        assert np.all(np.isfinite(signals[name]))


def test_item_signals_follow_their_definitions(tiny_arch: LoadedModel) -> None:
    items = _items()

    signals = item_signals(tiny_arch, items, shots=SHOTS)

    prompts = [few_shot_prompt(i.example.question, SHOTS) for i in items]
    stats = next_token_stats(tiny_arch, prompts)
    np.testing.assert_allclose(signals["first_token_prob"], stats.max_prob)
    np.testing.assert_allclose(signals["neg_entropy"], -stats.entropy)
    total, counts = answer_logprobs(
        tiny_arch, prompts, [i.label.greedy for i in items]
    )
    np.testing.assert_allclose(signals["answer_logprob"], total)
    np.testing.assert_allclose(
        signals["mean_token_logprob"][[0, 1, 3]],
        (total / np.maximum(counts, 1))[[0, 1, 3]],
    )
    assert signals["popularity"][0] == pytest.approx(3.0)
    assert signals["popularity"][2] == pytest.approx(math.log10(25))
    assert signals["consistency"][0] == pytest.approx(2 / 3)
    lengths = [
        len(
            tiny_arch.tokenizer(i.example.question, add_special_tokens=False)[
                "input_ids"
            ]
        )
        for i in items
    ]
    assert signals["length"].tolist() == lengths
    name_total, name_counts = name_logprobs(tiny_arch, ["the cat", "the car"])
    np.testing.assert_allclose(
        signals["name_logprob"][[0, 2]], name_total / name_counts
    )
    assert np.all(signals["neg_entropy"] <= 0.0)
    assert np.all(signals["answer_logprob"][[0, 1, 3]] < 0.0)


@pytest.mark.slow
def test_item_signals_on_gpt2_with_fifteen_shots() -> None:
    loaded = load_model("gpt2")
    items = [
        _item(
            f"q{i}",
            f"who is the author of book number {i}?",
            f"Author {i}",
            {"popularity": 10 + i, "subject": f"Book {i}"},
            (f"Author {i}",) * 3,
        )
        for i in range(16)
    ]

    start = time.perf_counter()
    signals = item_signals(loaded, items)
    seconds = (time.perf_counter() - start) / len(items)

    print(f"item_signals: {seconds:.3f} seconds per item")
    assert tuple(signals) == SIGNAL_NAMES
    for name, values in signals.items():
        assert values.shape == (16,)
        assert np.all(np.isfinite(values)), name
