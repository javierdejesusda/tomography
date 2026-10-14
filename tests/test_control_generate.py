from typing import Any

import pytest
import torch

from tomography.control.generate import generate
from tomography.hooks import ForwardInfo
from tomography.hooks import ResidualEdit
from tomography.model import EncodedBatch
from tomography.model import LoadedModel
from tomography.model import load_model

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
    prompts = ["The lazy dog", "The lazy dog " * 10]

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


def test_sampling_is_deterministic_per_seed_and_batch_size(
    tiny_arch: LoadedModel,
) -> None:
    tiny_arch.model.generation_config.eos_token_id = None
    options: dict[str, Any] = {"max_new_tokens": 6, "temperature": 1.0}

    first = generate(tiny_arch, PROMPTS, seed=3, batch_size=2, **options)
    again = generate(tiny_arch, PROMPTS, seed=3, batch_size=2, **options)
    other = generate(tiny_arch, PROMPTS, seed=4, batch_size=2, **options)

    assert first == again
    assert first != other


def test_sampling_leaves_the_global_rng_state_unchanged(
    tiny_arch: LoadedModel,
) -> None:
    tiny_arch.model.generation_config.eos_token_id = None
    torch.manual_seed(123)
    before = torch.get_rng_state()

    generate(tiny_arch, PROMPTS, max_new_tokens=4, temperature=1.0, seed=1)

    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("temperature", [None, 1.0], ids=["greedy", "sampling"])
def test_generate_never_calls_the_global_seed_functions(
    tiny_arch: LoadedModel,
    monkeypatch: pytest.MonkeyPatch,
    temperature: float | None,
) -> None:
    tiny_arch.model.generation_config.eos_token_id = None
    calls: list[str] = []

    def recorder(name: str) -> Any:
        def record(*args: Any, **kwargs: Any) -> None:
            calls.append(name)

        return record

    for owner, name in (
        (torch, "manual_seed"),
        (torch.cuda, "manual_seed_all"),
        (torch.cuda, "manual_seed"),
    ):
        monkeypatch.setattr(owner, name, recorder(f"{owner.__name__}.{name}"))

    generate(
        tiny_arch,
        PROMPTS,
        max_new_tokens=2,
        temperature=temperature,
        seed=1,
    )

    assert calls == []


def test_num_samples_returns_completions_in_prompt_major_order(
    tiny_arch: LoadedModel,
) -> None:
    tiny_arch.model.generation_config.eos_token_id = None
    greedy = generate(tiny_arch, PROMPTS, max_new_tokens=6)

    sampled = generate(
        tiny_arch,
        PROMPTS,
        max_new_tokens=6,
        temperature=1e-3,
        num_samples=3,
        batch_size=2,
    )

    assert [len(row) for row in sampled] == [3, 3, 3]
    for row, (expected,) in zip(sampled, greedy, strict=True):
        assert row == [expected] * 3


def hook_count(loaded: LoadedModel) -> int:
    modules = [loaded.decoder, *loaded.blocks]
    return sum(
        len(module._forward_hooks) + len(module._forward_pre_hooks)
        for module in modules
    )


def test_edits_factory_is_called_once_per_chunk_with_replicated_rows(
    tiny_arch: LoadedModel,
) -> None:
    tiny_arch.model.generation_config.eos_token_id = None
    batches: list[EncodedBatch] = []

    def factory(batch: EncodedBatch) -> dict[int, ResidualEdit]:
        batches.append(batch)
        return {}

    generate(
        tiny_arch,
        PROMPTS,
        max_new_tokens=3,
        temperature=1.0,
        num_samples=2,
        batch_size=2,
        edits=factory,
    )

    assert [len(batch.lengths) for batch in batches] == [4, 2]
    assert batches[0].lengths[0] == batches[0].lengths[1]
    assert batches[0].lengths[2] == batches[0].lengths[3]


def test_identity_edit_equals_no_edit(tiny_arch: LoadedModel) -> None:
    tiny_arch.model.generation_config.eos_token_id = None
    layers = range(tiny_arch.n_layers + 1)

    def identity(h: torch.Tensor, info: ForwardInfo) -> torch.Tensor:
        return h.clone()

    plain = generate(tiny_arch, PROMPTS, max_new_tokens=6)
    edited = generate(
        tiny_arch,
        PROMPTS,
        max_new_tokens=6,
        edits=lambda batch: {layer: identity for layer in layers},
    )

    assert edited == plain


def test_additive_last_layer_edit_changes_the_tokens(
    tiny_arch: LoadedModel,
) -> None:
    tiny_arch.model.generation_config.eos_token_id = None
    direction = torch.randn(
        tiny_arch.d_model, generator=torch.Generator().manual_seed(0)
    )

    def add(h: torch.Tensor, info: ForwardInfo) -> torch.Tensor:
        return h + 50.0 * direction.to(h)

    plain = generate(tiny_arch, PROMPTS, max_new_tokens=6)
    edited = generate(
        tiny_arch,
        PROMPTS,
        max_new_tokens=6,
        edits=lambda batch: {tiny_arch.n_layers: add},
    )

    assert edited != plain


def test_hooks_are_removed_after_generation(tiny_arch: LoadedModel) -> None:
    tiny_arch.model.generation_config.eos_token_id = None
    before = hook_count(tiny_arch)

    generate(
        tiny_arch,
        PROMPTS,
        max_new_tokens=2,
        edits=lambda batch: {1: lambda h, info: h},
    )

    assert hook_count(tiny_arch) == before


def test_hooks_are_removed_when_generation_raises(
    tiny_arch: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    tiny_arch.model.generation_config.eos_token_id = None
    before = hook_count(tiny_arch)
    expected = generate(tiny_arch, PROMPTS, max_new_tokens=3)

    def boom(*args: Any, **kwargs: Any) -> None:
        assert hook_count(tiny_arch) > before
        raise RuntimeError("boom")

    with monkeypatch.context() as patch:
        patch.setattr(tiny_arch.model, "generate", boom)
        with pytest.raises(RuntimeError, match="boom"):
            generate(
                tiny_arch,
                PROMPTS,
                max_new_tokens=3,
                edits=lambda batch: {1: lambda h, info: h},
            )

    assert hook_count(tiny_arch) == before
    assert generate(tiny_arch, PROMPTS, max_new_tokens=3) == expected


def test_position_check_fires_before_the_edits_factory(
    tiny_arch: LoadedModel,
) -> None:
    calls: list[EncodedBatch] = []

    def factory(batch: EncodedBatch) -> dict[int, ResidualEdit]:
        calls.append(batch)
        return {}

    limit = tiny_arch.model.config.max_position_embeddings

    with pytest.raises(ValueError, match="max_position_embeddings"):
        generate(tiny_arch, PROMPTS, max_new_tokens=limit, edits=factory)

    assert calls == []


@pytest.mark.parametrize(
    "option", [{"top_k": 1}, {"top_p": 1e-6}], ids=["top_k_one", "top_p_tiny"]
)
def test_top_k_and_top_p_restrict_sampling_to_the_greedy_token(
    tiny_arch: LoadedModel, option: dict[str, Any]
) -> None:
    tiny_arch.model.generation_config.eos_token_id = None
    greedy = generate(tiny_arch, PROMPTS, max_new_tokens=6)

    sampled = generate(
        tiny_arch,
        PROMPTS,
        max_new_tokens=6,
        temperature=10.0,
        num_samples=2,
        **option,
    )

    for row, (expected,) in zip(sampled, greedy, strict=True):
        assert row == [expected] * 2


def test_the_chunk_index_is_part_of_the_seed(tiny_arch: LoadedModel) -> None:
    tiny_arch.model.generation_config.eos_token_id = None

    result = generate(
        tiny_arch,
        ["The lazy dog"] * 2,
        max_new_tokens=6,
        temperature=1.0,
        batch_size=1,
    )

    assert result[0][0] != result[1][0]


def test_an_empty_edit_mapping_registers_no_hooks(
    tiny_arch: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    tiny_arch.model.generation_config.eos_token_id = None
    before = hook_count(tiny_arch)
    seen: list[int] = []
    original = tiny_arch.model.generate

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(hook_count(tiny_arch))
        return original(*args, **kwargs)

    monkeypatch.setattr(tiny_arch.model, "generate", spy)

    generate(tiny_arch, PROMPTS, max_new_tokens=2, edits=lambda batch: {})

    assert seen == [before]
    assert hook_count(tiny_arch) == before


@pytest.mark.slow
def test_gpt2_generation_plumbing() -> None:
    gpt2 = load_model("gpt2")

    greedy = generate(gpt2, PROMPTS, max_new_tokens=8)
    sampled = generate(
        gpt2,
        PROMPTS,
        max_new_tokens=8,
        temperature=0.8,
        top_p=0.9,
        top_k=40,
        num_samples=2,
        edits=lambda batch: {gpt2.n_layers // 2: lambda h, info: h},
    )

    assert [len(row) for row in greedy] == [1, 1, 1]
    assert [len(row) for row in sampled] == [2, 2, 2]
    for row in (*greedy, *sampled):
        for completion in row:
            assert len(completion.token_ids) <= 8
            assert isinstance(completion.text, str)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize(
    "device", ["cpu", "cuda"], ids=["cpu_model", "cuda_model"]
)
def test_gpt2_sampling_seeds_and_restores_the_global_rng_on_cuda(
    device: str,
) -> None:
    gpt2 = load_model("gpt2", device=device)
    torch.manual_seed(123)
    before = (torch.get_rng_state(), torch.cuda.get_rng_state())
    options: dict[str, Any] = {"max_new_tokens": 8, "temperature": 1.0}

    first = generate(gpt2, PROMPTS, seed=5, **options)
    again = generate(gpt2, PROMPTS, seed=5, **options)
    other = generate(gpt2, PROMPTS, seed=6, **options)
    generate(gpt2, PROMPTS, max_new_tokens=8)

    assert first == again
    assert first != other
    assert torch.equal(torch.get_rng_state(), before[0])
    assert torch.equal(torch.cuda.get_rng_state(), before[1])
