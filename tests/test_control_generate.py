from typing import Any

import pytest

from tomography.control.generate import generate
from tomography.model import LoadedModel

PROMPTS = ["The quick brown fox", "The lazy dog", "Honest answers are"]


def test_empty_prompts_return_an_empty_list(tiny_arch: LoadedModel) -> None:
    assert generate(tiny_arch, [], max_new_tokens=3) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_new_tokens": 0},
        {"batch_size": 0},
        {"num_samples": 0},
        {"temperature": 0.0},
        {"temperature": -1.0},
        {"temperature": float("nan")},
        {"temperature": float("inf")},
        {"temperature": 1.0, "top_p": 0.0},
        {"temperature": 1.0, "top_p": 1.5},
        {"temperature": 1.0, "top_k": -1},
        {"num_samples": 2},
        {"top_p": 0.9},
        {"top_k": 5},
    ],
    ids=[
        "max_new_tokens_zero",
        "batch_size_zero",
        "num_samples_zero",
        "temperature_zero",
        "temperature_negative",
        "temperature_nan",
        "temperature_inf",
        "top_p_zero",
        "top_p_above_one",
        "top_k_negative",
        "num_samples_without_temperature",
        "top_p_without_temperature",
        "top_k_without_temperature",
    ],
)
def test_invalid_arguments_raise_value_error(
    tiny_arch: LoadedModel, kwargs: dict[str, Any]
) -> None:
    options: dict[str, Any] = {"max_new_tokens": 3, **kwargs}

    with pytest.raises(ValueError, match=r"must be|need a temperature"):
        generate(tiny_arch, PROMPTS, **options)


def test_invalid_arguments_raise_even_without_prompts(
    tiny_arch: LoadedModel,
) -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        generate(tiny_arch, [], max_new_tokens=0)


def test_greedy_returns_one_completion_per_prompt(
    tiny_arch: LoadedModel,
) -> None:
    tiny_arch.model.generation_config.eos_token_id = None

    result = generate(tiny_arch, PROMPTS, max_new_tokens=5)

    assert [len(row) for row in result] == [1, 1, 1]
    for (completion,) in result:
        assert len(completion.token_ids) == 5
        assert completion.text == tiny_arch.tokenizer.decode(
            completion.token_ids, skip_special_tokens=True
        )
        assert not completion.finished


def test_greedy_batched_equals_one_prompt_at_a_time(
    tiny_arch: LoadedModel,
) -> None:
    tiny_arch.model.generation_config.eos_token_id = None

    batched = generate(tiny_arch, PROMPTS, max_new_tokens=6, batch_size=8)
    alone = [
        generate(tiny_arch, [prompt], max_new_tokens=6)[0] for prompt in PROMPTS
    ]

    assert batched == alone


def test_greedy_does_not_depend_on_batch_size(tiny_arch: LoadedModel) -> None:
    tiny_arch.model.generation_config.eos_token_id = None

    results = [
        generate(tiny_arch, PROMPTS, max_new_tokens=6, batch_size=size)
        for size in (1, 2, 3, 8)
    ]

    assert all(result == results[0] for result in results)


def test_position_check_fires_before_any_generation(
    tiny_arch: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fail(*args: Any, **kwargs: Any) -> None:
        calls.append("generate")

    monkeypatch.setattr(tiny_arch.model, "generate", fail)
    limit = tiny_arch.model.config.max_position_embeddings

    with pytest.raises(ValueError, match="max_position_embeddings"):
        generate(tiny_arch, PROMPTS, max_new_tokens=limit)

    assert calls == []


def test_position_check_counts_the_longest_row_of_a_later_chunk(
    tiny_arch: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        tiny_arch.model, "generate", lambda *a, **k: calls.append("generate")
    )
    limit = tiny_arch.model.config.max_position_embeddings
    prompts = ["The lazy dog", "The lazy dog " * 40]

    with pytest.raises(ValueError, match="max_position_embeddings"):
        generate(tiny_arch, prompts, max_new_tokens=limit - 20, batch_size=1)

    assert calls == []


def first_new_id(ids: tuple[int, ...]) -> int:
    """Returns an id of `ids` and checks it does not occur earlier."""
    first = ids[2]
    assert first not in ids[:2]
    return first


@pytest.mark.parametrize("as_list", [False, True], ids=["int_eos", "list_eos"])
def test_completion_is_cut_before_the_end_of_sequence_id(
    tiny_arch: LoadedModel, as_list: bool
) -> None:
    config = tiny_arch.model.generation_config
    config.eos_token_id = None
    full = generate(tiny_arch, PROMPTS, max_new_tokens=8)
    eos = first_new_id(full[0][0].token_ids)
    config.eos_token_id = [eos, 399] if as_list else eos

    cut = generate(tiny_arch, PROMPTS, max_new_tokens=8)

    first = cut[0][0]
    assert first.token_ids == full[0][0].token_ids[:2]
    assert first.finished
    assert first.text == tiny_arch.tokenizer.decode(
        first.token_ids, skip_special_tokens=True
    )
    for row, expected in zip(cut[1:], full[1:], strict=True):
        ids = expected[0].token_ids
        if eos in ids:
            assert row[0].token_ids == ids[: ids.index(eos)]
            assert row[0].finished
        else:
            assert row[0].token_ids == ids
            assert not row[0].finished


def test_finished_is_false_when_no_end_of_sequence_id_appears(
    tiny_arch: LoadedModel,
) -> None:
    config = tiny_arch.model.generation_config
    config.eos_token_id = None
    full = generate(tiny_arch, PROMPTS, max_new_tokens=4)
    seen = {i for (completion,) in full for i in completion.token_ids}
    config.eos_token_id = next(i for i in range(399, 0, -1) if i not in seen)

    result = generate(tiny_arch, PROMPTS, max_new_tokens=4)

    assert result == full
    assert not any(completion.finished for (completion,) in result)
