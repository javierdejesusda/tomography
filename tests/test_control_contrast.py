from collections.abc import Sequence
import time
from typing import Any

import numpy as np
import pytest
import torch

from tomography.control.contrast import PromptDifference
from tomography.control.contrast import contrast_edits
from tomography.control.contrast import contrast_generate
from tomography.control.contrast import contrast_logits
from tomography.control.contrast import contrast_logprobs
from tomography.control.contrast import contrast_windows
from tomography.control.generate import generate
from tomography.hooks import ForwardInfo
from tomography.hooks import capture_residuals
from tomography.hooks import edit_residuals
from tomography.model import BosMode
from tomography.model import LoadedModel
from tomography.model import load_model


def test_windows_take_the_minimum_over_the_triple() -> None:
    result = contrast_windows([5, 9, 4, 7, 3, 8], 6)

    assert result.dtype == torch.int64
    assert result.tolist() == [3, 6]


@pytest.mark.parametrize("window", [2, 4, 100], ids=["w2", "w4", "w100"])
def test_windows_clamp_to_the_window_and_the_real_lengths(window: int) -> None:
    lengths = [6, 3, 9, 5, 7, 2, 8, 4, 10]

    result = contrast_windows(lengths, window)

    expected = [
        min(window, lengths[i], lengths[3 + i], lengths[6 + i])
        for i in range(3)
    ]
    assert result.tolist() == expected


def test_a_short_prompt_never_exceeds_its_own_length() -> None:
    result = contrast_windows([2, 20, 9, 20, 7, 20], 8)

    assert result.tolist() == [2, 8]


@pytest.mark.parametrize(
    ("lengths", "window"),
    [
        ([], 4),
        ([3, 3], 4),
        ([3, 3, 3, 3], 4),
        ([3, 0, 3], 4),
        ([3, -1, 3], 4),
        ([3, 3, 3], 0),
    ],
    ids=[
        "empty",
        "two_rows",
        "four_rows",
        "zero_length",
        "negative_length",
        "window_zero",
    ],
)
def test_windows_reject_invalid_arguments(
    lengths: list[int], window: int
) -> None:
    with pytest.raises(ValueError, match=r"lengths|window"):
        contrast_windows(lengths, window)


INFO = ForwardInfo(None, None)


def stream(n: int = 2, t: int = 5, d: int = 4) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(3 * n, t, d, generator=generator)


def explicit_shift(
    h: torch.Tensor, alpha: float, windows: list[int]
) -> torch.Tensor:
    n, t = len(windows), h.shape[1]
    result = h.clone()
    for i, w in enumerate(windows):
        shift = alpha * (h[n + i, t - w :] - h[2 * n + i, t - w :])
        for row in (i, n + i, 2 * n + i):
            result[row, t - w :] += shift
    return result


def test_edits_add_the_shift_to_all_three_streams_in_the_window() -> None:
    h = stream()
    windows = [2, 3]

    edited = contrast_edits([1], 0.5, torch.tensor(windows))[1](h, INFO)

    torch.testing.assert_close(edited, explicit_shift(h, 0.5, windows))
    assert not torch.equal(edited[0, -2:], h[0, -2:])
    assert not torch.equal(edited[2, -2:], h[2, -2:])
    assert not torch.equal(edited[4, -2:], h[4, -2:])


def test_edits_leave_columns_before_the_window_unchanged() -> None:
    h = stream()

    edited = contrast_edits([1], 2.0, torch.tensor([2, 3]))[1](h, INFO)

    assert torch.equal(edited[[0, 2, 4], :3], h[[0, 2, 4], :3])
    assert torch.equal(edited[[1, 3, 5], :2], h[[1, 3, 5], :2])


def test_edits_with_alpha_zero_return_the_same_object() -> None:
    h = stream()

    assert contrast_edits([1], 0.0, torch.tensor([2, 3]))[1](h, INFO) is h


def test_edits_are_keyed_by_the_layers() -> None:
    edits = contrast_edits([3, 0, 3, 2], 1.0, torch.tensor([1, 1]))

    assert sorted(edits) == [0, 2, 3]


def test_edits_on_a_bf16_stream_equal_the_float32_result_cast() -> None:
    h = stream().to(torch.bfloat16)
    windows = [2, 3]

    edited = contrast_edits([1], 0.3, torch.tensor(windows))[1](h, INFO)

    expected = explicit_shift(h.float(), 0.3, windows).to(torch.bfloat16)
    assert edited.dtype == torch.bfloat16
    assert torch.equal(edited, expected)


def test_edits_reject_a_row_count_that_is_not_three_times_the_windows() -> None:
    edit = contrast_edits([1], 0.5, torch.tensor([2, 3, 1]))[1]

    with pytest.raises(ValueError, match="rows"):
        edit(stream(), INFO)


PROMPTS = ["The quick brown fox", "The lazy dog", "Honest answers are"]
POSITIVE = [
    "Honest answers are true and the quick brown fox jumps",
    "A short text, a longer text",
    "She felt happy",
]
NEGATIVE = [
    "Dishonest answers are false, and honest answers are true.",
    "Pick a number between 0 and 9: 0123456789.",
    "The model reads the stream of the residual",
]

MESSAGES = r"same length|empty|outside|must be"


def reference_streams(
    lm: LoadedModel, texts: Sequence[str], bos: BosMode
) -> list[list[int]]:
    prefix = [lm.bos_id] if bos == "prepend" else []
    ids = lm.tokenizer(list(texts), add_special_tokens=False)["input_ids"]
    return [prefix + row for row in ids]


def embed(lm: LoadedModel, ids: list[int]) -> torch.Tensor:
    tokens = torch.tensor([ids])
    if lm.spec.model_type == "gpt2":
        positions = torch.arange(len(ids))[None]
        return lm.model.transformer.wte(tokens) + lm.model.transformer.wpe(
            positions
        )
    return lm.model.model.embed_tokens(tokens)


def run_block(lm: LoadedModel, number: int, h: torch.Tensor) -> torch.Tensor:
    block = lm.blocks[number - 1]
    if lm.spec.model_type == "gpt2":
        out = block(h)
    else:
        pos = torch.arange(h.shape[1])[None]
        out = block(
            h,
            attention_mask=None,
            position_ids=pos,
            position_embeddings=lm.decoder.rotary_emb(h, pos),
        )
    return out[0] if isinstance(out, tuple) else out


def reference_hidden(
    lm: LoadedModel,
    main: list[int],
    pos: list[int],
    neg: list[int],
    layers: Sequence[int],
    alpha: float,
    window: int,
) -> torch.Tensor:
    w = min(window, len(main), len(pos), len(neg))
    streams = [embed(lm, ids) for ids in (main, pos, neg)]

    def cascade() -> None:
        shift = alpha * (streams[1][:, -w:] - streams[2][:, -w:])
        for stream in streams:
            stream[:, -w:] += shift

    if 0 in layers:
        cascade()
    for number in range(1, lm.n_layers + 1):
        streams = [run_block(lm, number, h) for h in streams]
        if number in layers:
            cascade()
    return lm.final_norm(streams[0])


def reference_logits(
    lm: LoadedModel,
    prompts: Sequence[str],
    positive: Sequence[str],
    negative: Sequence[str],
    layers: Sequence[int],
    alpha: float,
    window: int,
) -> torch.Tensor:
    bos: BosMode = lm.default_bos
    rows = []
    with torch.no_grad():
        for main, pos, neg in zip(
            reference_streams(lm, prompts, bos),
            reference_streams(lm, positive, bos),
            reference_streams(lm, negative, bos),
            strict=True,
        ):
            hidden = reference_hidden(lm, main, pos, neg, layers, alpha, window)
            rows.append(lm.model.get_output_embeddings()(hidden[0, -1]))
    return torch.stack(rows).float()


def close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    tol = 1e-4 * max(1.0, expected.abs().max().item())
    torch.testing.assert_close(actual, expected, atol=tol, rtol=0)


def hook_count(loaded: LoadedModel) -> int:
    modules = [loaded.decoder, *loaded.blocks]
    return sum(
        len(module._forward_hooks) + len(module._forward_pre_hooks)
        for module in modules
    )


def test_the_streams_of_each_triple_differ_in_length(
    tiny_arch: LoadedModel,
) -> None:
    for texts in zip(PROMPTS, POSITIVE, NEGATIVE, strict=True):
        ids = tiny_arch.tokenizer(list(texts), add_special_tokens=False)
        assert len({len(row) for row in ids["input_ids"]}) == 3


@pytest.mark.parametrize("alpha", [0.5, -0.7, 2.0], ids=["a0.5", "a-0.7", "a2"])
@pytest.mark.parametrize("window", [2, 4, 100], ids=["w2", "w4", "w100"])
def test_logits_equal_the_reference_block_loop(
    tiny_arch: LoadedModel, alpha: float, window: int
) -> None:
    layers = (1, 2, 3)

    actual = contrast_logits(
        tiny_arch,
        PROMPTS,
        POSITIVE,
        NEGATIVE,
        layers=layers,
        alpha=alpha,
        window=window,
    )

    expected = reference_logits(
        tiny_arch, PROMPTS, POSITIVE, NEGATIVE, layers, alpha, window
    )
    close(actual, expected)


def test_logits_equal_the_reference_with_the_embedding_layer(
    tiny_arch: LoadedModel,
) -> None:
    actual = contrast_logits(
        tiny_arch,
        PROMPTS,
        POSITIVE,
        NEGATIVE,
        layers=(0, 2),
        alpha=0.5,
        window=4,
    )

    expected = reference_logits(
        tiny_arch, PROMPTS, POSITIVE, NEGATIVE, (0, 2), 0.5, 4
    )
    close(actual, expected)


def test_logits_with_alpha_zero_equal_the_plain_forward(
    tiny_arch: LoadedModel,
) -> None:
    batch = tiny_arch.encode(PROMPTS, padding_side="left")
    with torch.no_grad():
        plain = tiny_arch.model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            position_ids=batch.position_ids,
        ).logits[:, -1]

    actual = contrast_logits(
        tiny_arch, PROMPTS, POSITIVE, NEGATIVE, layers=(1, 2), alpha=0.0
    )

    close(actual, plain.float())


def test_logits_with_alpha_one_differ_from_alpha_zero(
    tiny_arch: LoadedModel,
) -> None:
    options = {"layers": (1, 2)}

    off = contrast_logits(
        tiny_arch, PROMPTS, POSITIVE, NEGATIVE, alpha=0.0, **options
    )
    on = contrast_logits(
        tiny_arch, PROMPTS, POSITIVE, NEGATIVE, alpha=1.0, **options
    )

    assert not torch.allclose(on, off)


def test_logits_are_float32_on_the_cpu_with_one_row_per_prompt(
    tiny_arch: LoadedModel,
) -> None:
    result = contrast_logits(
        tiny_arch, PROMPTS, POSITIVE, NEGATIVE, layers=(1,), alpha=0.5
    )

    assert result.dtype == torch.float32
    assert result.device.type == "cpu"
    assert result.shape == (3, tiny_arch.model.config.vocab_size)


def test_logits_remove_their_hooks(tiny_arch: LoadedModel) -> None:
    before = hook_count(tiny_arch)

    contrast_logits(
        tiny_arch, PROMPTS, POSITIVE, NEGATIVE, layers=(0, 1, 3), alpha=0.5
    )

    assert hook_count(tiny_arch) == before


@pytest.mark.parametrize(
    "kwargs",
    [
        {"positive": POSITIVE[:2]},
        {"negative": NEGATIVE[:1]},
        {"prompts": [], "positive": [], "negative": []},
        {"layers": (4,)},
        {"layers": (-1,)},
        {"layers": ()},
        {"window": 0},
        {"alpha": float("nan")},
        {"alpha": float("inf")},
    ],
    ids=[
        "positive_shorter",
        "negative_shorter",
        "empty_prompts",
        "layer_above_range",
        "layer_negative",
        "empty_layers",
        "window_zero",
        "alpha_nan",
        "alpha_inf",
    ],
)
def test_logits_reject_invalid_arguments(
    tiny_arch: LoadedModel, kwargs: dict[str, Any]
) -> None:
    options: dict[str, Any] = {
        "prompts": PROMPTS,
        "positive": POSITIVE,
        "negative": NEGATIVE,
        "layers": (1,),
        "alpha": 0.5,
        **kwargs,
    }

    with pytest.raises(ValueError, match=MESSAGES):
        contrast_logits(tiny_arch, **options)


def reference_generate(
    lm: LoadedModel,
    prompts: Sequence[str],
    positive: Sequence[str],
    negative: Sequence[str],
    layers: Sequence[int],
    alpha: float,
    window: int,
    steps: int,
    *,
    grow: bool = True,
) -> list[list[int]]:
    bos: BosMode = lm.default_bos
    result = []
    for main, pos, neg in zip(
        reference_streams(lm, prompts, bos),
        reference_streams(lm, positive, bos),
        reference_streams(lm, negative, bos),
        strict=True,
    ):
        new: list[int] = []
        for _ in range(steps):
            with torch.no_grad():
                hidden = reference_hidden(
                    lm, main, pos, neg, layers, alpha, window
                )
                logits = lm.model.get_output_embeddings()(hidden[0, -1])
            token = int(logits.argmax())
            new.append(token)
            main.append(token)
            if grow:
                pos.append(token)
                neg.append(token)
        result.append(new)
    return result


GENERATE = {"layers": (1, 2, 3), "alpha": 2.0, "window": 4}


@pytest.mark.parametrize("window", [2, 4, 100], ids=["w2", "w4", "w100"])
def test_generate_equals_the_reference_greedy_loop(
    tiny_arch: LoadedModel, window: int
) -> None:
    tiny_arch.model.generation_config.eos_token_id = None
    options = {**GENERATE, "window": window}

    result = contrast_generate(
        tiny_arch, PROMPTS, POSITIVE, NEGATIVE, max_new_tokens=6, **options
    )

    expected = reference_generate(
        tiny_arch, PROMPTS, POSITIVE, NEGATIVE, steps=6, **options
    )
    assert [list(c.token_ids) for c in result] == expected
    assert not any(c.finished for c in result)


def test_generate_grows_the_positive_and_negative_streams(
    tiny_arch: LoadedModel,
) -> None:
    tiny_arch.model.generation_config.eos_token_id = None

    grown = reference_generate(
        tiny_arch, PROMPTS, POSITIVE, NEGATIVE, steps=6, **GENERATE
    )
    frozen = reference_generate(
        tiny_arch, PROMPTS, POSITIVE, NEGATIVE, steps=6, grow=False, **GENERATE
    )

    assert grown != frozen


def test_generate_cuts_at_the_end_of_sequence_id(
    tiny_arch: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tiny_arch.model.generation_config
    monkeypatch.setattr(config, "eos_token_id", None)
    full = contrast_generate(
        tiny_arch, PROMPTS, POSITIVE, NEGATIVE, max_new_tokens=6, **GENERATE
    )
    eos = full[0].token_ids[0]
    monkeypatch.setattr(config, "eos_token_id", eos)

    cut = contrast_generate(
        tiny_arch, PROMPTS, POSITIVE, NEGATIVE, max_new_tokens=6, **GENERATE
    )

    assert cut[0].token_ids == ()
    assert cut[0].finished
    for row, expected in zip(cut, full, strict=True):
        ids = expected.token_ids
        kept = ids[: ids.index(eos)] if eos in ids else ids
        assert row.token_ids == kept
        assert row.finished == (eos in ids)
        assert row.text == tiny_arch.tokenizer.decode(
            kept, skip_special_tokens=True
        )


def test_generate_accepts_a_list_of_end_of_sequence_ids(
    tiny_arch: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tiny_arch.model.generation_config
    monkeypatch.setattr(config, "eos_token_id", None)
    full = contrast_generate(
        tiny_arch, PROMPTS, POSITIVE, NEGATIVE, max_new_tokens=6, **GENERATE
    )
    monkeypatch.setattr(config, "eos_token_id", [full[1].token_ids[2], 399])

    cut = contrast_generate(
        tiny_arch, PROMPTS, POSITIVE, NEGATIVE, max_new_tokens=6, **GENERATE
    )

    assert cut[1].token_ids == full[1].token_ids[:2]
    assert cut[1].finished


def test_generate_text_is_the_decoded_ids(tiny_arch: LoadedModel) -> None:
    tiny_arch.model.generation_config.eos_token_id = None

    result = contrast_generate(
        tiny_arch, PROMPTS, POSITIVE, NEGATIVE, max_new_tokens=4, **GENERATE
    )

    for completion in result:
        assert completion.text == tiny_arch.tokenizer.decode(
            completion.token_ids, skip_special_tokens=True
        )


def test_generate_does_not_depend_on_batch_size(tiny_arch: LoadedModel) -> None:
    tiny_arch.model.generation_config.eos_token_id = None

    results = [
        contrast_generate(
            tiny_arch,
            PROMPTS,
            POSITIVE,
            NEGATIVE,
            max_new_tokens=5,
            batch_size=size,
            **GENERATE,
        )
        for size in (1, 2, 3)
    ]

    assert results[0] == results[1] == results[2]


def test_generate_stops_rows_independently(
    tiny_arch: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tiny_arch.model.generation_config
    monkeypatch.setattr(config, "eos_token_id", None)
    full = contrast_generate(
        tiny_arch, PROMPTS, POSITIVE, NEGATIVE, max_new_tokens=6, **GENERATE
    )
    monkeypatch.setattr(config, "eos_token_id", full[2].token_ids[1])

    cut = contrast_generate(
        tiny_arch, PROMPTS, POSITIVE, NEGATIVE, max_new_tokens=6, **GENERATE
    )
    alone = [
        contrast_generate(
            tiny_arch, [p], [q], [n], max_new_tokens=6, **GENERATE
        )[0]
        for p, q, n in zip(PROMPTS, POSITIVE, NEGATIVE, strict=True)
    ]

    assert cut == alone


def test_generate_position_limit_fires_before_any_forward(
    tiny_arch: LoadedModel,
) -> None:
    calls: list[int] = []
    handle = tiny_arch.decoder.register_forward_pre_hook(
        lambda module, args, kwargs: calls.append(1), with_kwargs=True
    )
    limit = tiny_arch.model.config.max_position_embeddings
    long = ["The lazy dog " * 20]
    max_new_tokens = limit - 20
    longest = max(tiny_arch.encode(long).lengths)
    assert longest <= limit < longest + max_new_tokens

    try:
        with pytest.raises(ValueError, match="max_position_embeddings"):
            contrast_generate(
                tiny_arch,
                PROMPTS[:1] + PROMPTS[:1],
                [*POSITIVE[:1], *long],
                [*NEGATIVE[:1], *NEGATIVE[:1]],
                max_new_tokens=max_new_tokens,
                batch_size=1,
                **GENERATE,
            )
    finally:
        handle.remove()

    assert calls == []


def test_generate_returns_an_empty_list_for_no_prompts(
    tiny_arch: LoadedModel,
) -> None:
    assert (
        contrast_generate(tiny_arch, [], [], [], max_new_tokens=3, **GENERATE)
        == []
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"positive": POSITIVE[:2]},
        {"negative": NEGATIVE[:1]},
        {"layers": ()},
        {"layers": (4,)},
        {"layers": (-1,)},
        {"max_new_tokens": 0},
        {"batch_size": 0},
        {"window": 0},
        {"alpha": float("nan")},
        {"alpha": float("inf")},
    ],
    ids=[
        "positive_shorter",
        "negative_shorter",
        "empty_layers",
        "layer_above_range",
        "layer_negative",
        "max_new_tokens_zero",
        "batch_size_zero",
        "window_zero",
        "alpha_nan",
        "alpha_inf",
    ],
)
def test_generate_rejects_invalid_arguments(
    tiny_arch: LoadedModel, kwargs: dict[str, Any]
) -> None:
    options: dict[str, Any] = {
        "prompts": PROMPTS,
        "positive": POSITIVE,
        "negative": NEGATIVE,
        "max_new_tokens": 3,
        **GENERATE,
        **kwargs,
    }

    with pytest.raises(ValueError, match=MESSAGES):
        contrast_generate(tiny_arch, **options)


CONTINUATIONS = [" jumps over", " is lazy and old", " true"]


def reference_logprob(
    lm: LoadedModel,
    prompt: str,
    continuation: str,
    positive: str,
    negative: str,
    layers: Sequence[int],
    alpha: float,
) -> float:
    prefix = [lm.bos_id] if lm.default_bos == "prepend" else []

    def ids(text: str) -> list[int]:
        return list(lm.tokenizer(text, add_special_tokens=False)["input_ids"])

    tail = ids(continuation)
    streams = [
        prefix + ids(text) + tail for text in (prompt, positive, negative)
    ]
    with torch.no_grad():
        hidden = reference_hidden(lm, *streams, layers, alpha, len(tail) + 1)
        logits = lm.model.get_output_embeddings()(hidden[0]).float()
    logp = torch.log_softmax(logits, dim=-1)
    total = len(streams[0])
    rows = torch.arange(total - len(tail) - 1, total - 1)
    return float(logp[rows, torch.tensor(tail)].sum())


def plain_logprob(lm: LoadedModel, prompt: str, continuation: str) -> float:
    prefix = [lm.bos_id] if lm.default_bos == "prepend" else []
    tail = lm.tokenizer(continuation, add_special_tokens=False)["input_ids"]
    head = lm.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([prefix + head + tail])
    with torch.no_grad():
        logits = lm.model(input_ids=ids).logits[0].float()
    logp = torch.log_softmax(logits, dim=-1)
    rows = torch.arange(ids.shape[1] - len(tail) - 1, ids.shape[1] - 1)
    return float(logp[rows, torch.tensor(tail)].sum())


@pytest.mark.parametrize("alpha", [0.5, -0.7], ids=["a0.5", "a-0.7"])
def test_logprobs_equal_the_reference_at_window_continuation_plus_one(
    tiny_arch: LoadedModel, alpha: float
) -> None:
    layers = (1, 2, 3)

    actual = contrast_logprobs(
        tiny_arch,
        PROMPTS,
        CONTINUATIONS,
        POSITIVE,
        NEGATIVE,
        layers=layers,
        alpha=alpha,
    )

    expected = [
        reference_logprob(tiny_arch, p, c, q, n, layers, alpha)
        for p, c, q, n in zip(
            PROMPTS, CONTINUATIONS, POSITIVE, NEGATIVE, strict=True
        )
    ]
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=0)


def test_logprobs_with_alpha_zero_equal_the_plain_log_probability(
    tiny_arch: LoadedModel,
) -> None:
    actual = contrast_logprobs(
        tiny_arch,
        PROMPTS,
        CONTINUATIONS,
        POSITIVE,
        NEGATIVE,
        layers=(1, 2),
        alpha=0.0,
    )

    expected = [
        plain_logprob(tiny_arch, p, c)
        for p, c in zip(PROMPTS, CONTINUATIONS, strict=True)
    ]
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)


def test_the_first_continuation_token_score_changes_with_alpha(
    tiny_arch: LoadedModel,
) -> None:
    tokenizer = tiny_arch.tokenizer
    one = tokenizer.decode(tokenizer(" jumps")["input_ids"][:1])
    assert len(tokenizer(one, add_special_tokens=False)["input_ids"]) == 1
    options = {
        "layers": (1, 2),
        "continuations": [one] * 3,
        "positive": POSITIVE,
        "negative": NEGATIVE,
    }

    off = contrast_logprobs(tiny_arch, PROMPTS, alpha=0.0, **options)
    on = contrast_logprobs(tiny_arch, PROMPTS, alpha=1.0, **options)

    assert not np.allclose(on, off, atol=1e-4)


def test_logprobs_do_not_depend_on_batch_size(tiny_arch: LoadedModel) -> None:
    results = [
        contrast_logprobs(
            tiny_arch,
            PROMPTS,
            CONTINUATIONS,
            POSITIVE,
            NEGATIVE,
            layers=(1, 3),
            alpha=0.5,
            batch_size=size,
        )
        for size in (1, 2, 3)
    ]

    np.testing.assert_allclose(results[1], results[0], atol=1e-4, rtol=0)
    np.testing.assert_allclose(results[2], results[0], atol=1e-4, rtol=0)


def test_logprobs_are_float32_with_one_score_per_prompt(
    tiny_arch: LoadedModel,
) -> None:
    result = contrast_logprobs(
        tiny_arch,
        PROMPTS,
        CONTINUATIONS,
        POSITIVE,
        NEGATIVE,
        layers=(1,),
        alpha=0.5,
    )

    assert result.dtype == np.float32
    assert result.shape == (3,)


def test_logprobs_of_no_prompts_are_an_empty_float32_array(
    tiny_arch: LoadedModel,
) -> None:
    result = contrast_logprobs(
        tiny_arch, [], [], [], [], layers=(1,), alpha=0.5
    )

    assert result.dtype == np.float32
    assert result.shape == (0,)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"continuations": CONTINUATIONS[:2]},
        {"positive": POSITIVE[:2]},
        {"negative": NEGATIVE[:1]},
        {"continuations": ["", " is", " true"]},
        {"batch_size": 0},
        {"layers": ()},
        {"layers": (4,)},
        {"alpha": float("nan")},
        {"bos": "none", "positive": ["", *POSITIVE[1:]]},
        {"bos": "none", "prompts": ["", *PROMPTS[1:]]},
        {"bos": "none", "negative": [*NEGATIVE[:2], ""]},
    ],
    ids=[
        "continuations_shorter",
        "positive_shorter",
        "negative_shorter",
        "continuation_without_tokens",
        "batch_size_zero",
        "empty_layers",
        "layer_above_range",
        "alpha_nan",
        "no_context_in_positive",
        "no_context_in_prompt",
        "no_context_in_negative",
    ],
)
def test_logprobs_reject_invalid_arguments(
    tiny_arch: LoadedModel, kwargs: dict[str, Any]
) -> None:
    options: dict[str, Any] = {
        "prompts": PROMPTS,
        "continuations": CONTINUATIONS,
        "positive": POSITIVE,
        "negative": NEGATIVE,
        "layers": (1,),
        "alpha": 0.5,
        **kwargs,
    }

    with pytest.raises(ValueError, match=MESSAGES + r"|context|tokens"):
        contrast_logprobs(tiny_arch, **options)


DIFF_POSITIVE = "Be truthful and honest in every answer today."
DIFF_NEGATIVE = "Never tell the truth, always lying."
DIFF_PROMPTS = ["The quick brown fox jumps over", "Hi", "Honest answers are"]


def captured_differences(
    lm: LoadedModel, layers: Sequence[int], bos: BosMode | None = None
) -> dict[int, torch.Tensor]:
    batch = lm.encode([DIFF_POSITIVE, DIFF_NEGATIVE], bos=bos)
    k = min(batch.lengths)
    with torch.no_grad(), capture_residuals(lm, layers) as captured:
        lm.model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            position_ids=batch.position_ids,
        )
    return {
        layer: (captured[layer][0, :k] - captured[layer][1, :k]).float()
        for layer in layers
    }


def test_from_prompts_vectors_equal_the_captured_differences(
    tiny_arch: LoadedModel,
) -> None:
    expected = captured_differences(tiny_arch, (1, 2))

    diff = PromptDifference.from_prompts(
        tiny_arch, DIFF_POSITIVE, DIFF_NEGATIVE, [1, 2], coeff=1.5
    )

    k = min(tiny_arch.encode([DIFF_POSITIVE, DIFF_NEGATIVE]).lengths)
    assert sorted(diff.vectors) == [1, 2]
    assert diff.coeff == 1.5
    for layer in (1, 2):
        vector = diff.vectors[layer]
        assert vector.dtype == torch.float32
        assert vector.shape == (k, tiny_arch.d_model)
        torch.testing.assert_close(vector[1:], expected[layer][1:])


def test_from_prompts_zeroes_the_position_zero_difference_under_bos(
    tiny_arch: LoadedModel,
) -> None:
    diff = PromptDifference.from_prompts(
        tiny_arch, DIFF_POSITIVE, DIFF_NEGATIVE, [1, 3], coeff=1.0
    )

    for vector in diff.vectors.values():
        assert torch.equal(vector[0], torch.zeros_like(vector[0]))
        assert vector[1:].abs().max() > 0


def test_from_prompts_keeps_the_position_zero_difference_without_bos(
    tiny_arch: LoadedModel,
) -> None:
    first = [
        row[0]
        for row in tiny_arch.tokenizer(
            [DIFF_POSITIVE, DIFF_NEGATIVE], add_special_tokens=False
        )["input_ids"]
    ]
    assert first[0] != first[1]

    diff = PromptDifference.from_prompts(
        tiny_arch, DIFF_POSITIVE, DIFF_NEGATIVE, [2], coeff=1.0, bos="none"
    )

    assert diff.vectors[2][0].abs().max() > 0


@pytest.mark.parametrize(
    ("positive", "layers"),
    [
        (DIFF_NEGATIVE, [1]),
        (DIFF_POSITIVE, []),
        (DIFF_POSITIVE, [4]),
        (DIFF_POSITIVE, [-1]),
    ],
    ids=["identical_prompts", "empty_layers", "layer_above", "layer_negative"],
)
def test_from_prompts_rejects_invalid_arguments(
    tiny_arch: LoadedModel, positive: str, layers: list[int]
) -> None:
    with pytest.raises(ValueError, match=r"identical|layers|layer"):
        PromptDifference.from_prompts(
            tiny_arch, positive, DIFF_NEGATIVE, layers, coeff=1.0
        )


@pytest.mark.parametrize(
    ("vectors", "coeff"),
    [
        ({}, 1.0),
        ({1: torch.ones(4)}, 1.0),
        ({1: torch.ones(2, 3, 4)}, 1.0),
        ({1: torch.full((3, 4), float("nan"))}, 1.0),
        ({1: torch.full((3, 4), float("inf"))}, 1.0),
        ({1: torch.zeros(3, 4)}, 1.0),
        ({1: torch.ones(3, 4), 2: torch.ones(3, 5)}, 1.0),
        ({1: torch.ones(3, 4)}, float("nan")),
        ({1: torch.ones(3, 4)}, float("inf")),
    ],
    ids=[
        "empty",
        "one_dimensional",
        "three_dimensional",
        "nan",
        "inf",
        "all_zero",
        "different_widths",
        "coeff_nan",
        "coeff_inf",
    ],
)
def test_construction_rejects_invalid_values(
    vectors: dict[int, torch.Tensor], coeff: float
) -> None:
    with pytest.raises(ValueError, match=r"vectors|coeff"):
        PromptDifference(vectors, coeff)


LEFT_PADDED = ([0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 2, 3])
FULL = ([1, 1, 1, 1, 1, 1], [0, 1, 2, 3, 4, 5])


def info_and_tables(
    source: str,
) -> tuple[ForwardInfo, list[list[int]], list[list[int]]]:
    masks = torch.tensor([LEFT_PADDED[0], FULL[0]])
    positions = torch.tensor([LEFT_PADDED[1], FULL[1]])
    if source == "positions":
        info = ForwardInfo(masks, positions)
        return info, positions.tolist(), masks.tolist()
    if source == "mask":
        info = ForwardInfo(masks, None)
        index = (masks.cumsum(-1) - 1).clamp(min=0)
        return info, index.tolist(), masks.tolist()
    columns = [list(range(6)), list(range(6))]
    return ForwardInfo(None, None), columns, [[1] * 6, [1] * 6]


def expected_edit(
    h: torch.Tensor,
    vectors: torch.Tensor,
    coeff: float,
    tables: tuple[list[list[int]], list[list[int]]],
    limits: list[int] | None = None,
) -> torch.Tensor:
    index, real = tables
    result = h.clone()
    for row in range(h.shape[0]):
        for col in range(h.shape[1]):
            j = index[row][col]
            capped = limits is not None and j >= limits[row]
            if real[row][col] and j < vectors.shape[0] and not capped:
                result[row, col] += coeff * vectors[j]
    return result


SOURCES = pytest.mark.parametrize(
    "source",
    ["positions", "mask", "neither"],
    ids=["positions", "mask", "none"],
)


def handmade() -> tuple[PromptDifference, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    vectors = torch.randn(3, 4, generator=generator)
    h = torch.randn(2, 6, 4, generator=generator)
    return PromptDifference({1: vectors}, 2.0), h


@SOURCES
def test_edit_adds_the_vector_at_the_first_token_indices_only(
    source: str,
) -> None:
    diff, h = handmade()
    info, index, real = info_and_tables(source)

    edited = diff.edits()[1](h, info)

    expected = expected_edit(h, diff.vectors[1], 2.0, (index, real))
    torch.testing.assert_close(edited, expected)
    assert not torch.equal(edited, h)


def test_edit_leaves_pad_columns_untouched() -> None:
    diff, h = handmade()
    info, _, _ = info_and_tables("positions")

    edited = diff.edits()[1](h, info)

    assert torch.equal(edited[0, :2], h[0, :2])
    assert torch.equal(edited[0, 5], h[0, 5])


@pytest.mark.parametrize(
    "lengths", [2, [1, 2]], ids=["int_for_all_rows", "per_row"]
)
def test_edit_caps_the_indices_below_the_prompt_length(
    lengths: int | list[int],
) -> None:
    diff, h = handmade()
    info, index, real = info_and_tables("positions")
    limits = [lengths] * 2 if isinstance(lengths, int) else lengths

    edited = diff.edits(lengths)[1](h, info)

    expected = expected_edit(h, diff.vectors[1], 2.0, (index, real), limits)
    torch.testing.assert_close(edited, expected)


def test_edit_rejects_a_prompt_length_sequence_of_the_wrong_size() -> None:
    diff, h = handmade()
    info, _, _ = info_and_tables("positions")

    with pytest.raises(ValueError, match=r"3.*2|2.*3"):
        diff.edits([2, 2, 2])[1](h, info)


@pytest.mark.parametrize(
    "lengths", [0, -1, [2, 0], [0, 3]], ids=["zero", "negative", "row0", "row1"]
)
def test_edits_reject_prompt_lengths_below_one(
    lengths: int | list[int],
) -> None:
    diff, _ = handmade()

    with pytest.raises(ValueError, match="prompt_lengths"):
        diff.edits(lengths)


def test_edit_with_a_zero_coefficient_returns_the_same_object() -> None:
    _, h = handmade()
    diff = PromptDifference({1: torch.ones(3, 4)}, 0.0)
    info, _, _ = info_and_tables("positions")

    assert diff.edits()[1](h, info) is h


def test_edit_on_a_bf16_stream_keeps_the_dtype() -> None:
    diff, h = handmade()
    info, index, real = info_and_tables("positions")

    edited = diff.edits()[1](h.to(torch.bfloat16), info)

    expected = expected_edit(
        h.to(torch.bfloat16).float(), diff.vectors[1], 2.0, (index, real)
    ).to(torch.bfloat16)
    assert edited.dtype == torch.bfloat16
    assert torch.equal(edited, expected)


def test_call_equals_edits_of_the_batch_lengths(tiny_arch: LoadedModel) -> None:
    diff = PromptDifference({1: torch.ones(12, 4)}, 2.0)
    batch = tiny_arch.encode(DIFF_PROMPTS, padding_side="left")
    width = 12
    h = torch.randn(3, width, 4)
    info = ForwardInfo(None, torch.arange(width).expand(3, width))

    called = diff(batch)
    direct = diff.edits(batch.lengths)

    assert called.keys() == direct.keys()
    assert torch.equal(called[1](h, info), direct[1](h, info))
    assert not torch.equal(called[1](h, info), diff.edits()[1](h, info))


def reference_steered_tokens(
    lm: LoadedModel, diff: PromptDifference, prompts: Sequence[str], steps: int
) -> list[tuple[int, ...]]:
    result = []
    for prompt in prompts:
        ids = lm.encode([prompt]).input_ids[0].tolist()
        fixed = len(ids)
        for _ in range(steps):
            tokens = torch.tensor([ids])
            mask = torch.ones_like(tokens)
            positions = (mask.cumsum(dim=-1) - 1).clamp(min=0)
            with torch.no_grad(), edit_residuals(lm, diff.edits(fixed)):
                logits = lm.model(
                    input_ids=tokens,
                    attention_mask=mask,
                    position_ids=positions,
                ).logits
            ids.append(int(logits[0, -1].argmax()))
        result.append(tuple(ids[fixed:]))
    return result


def test_cached_generate_equals_a_no_cache_loop_with_the_fixed_prompt_length(
    tiny_arch: LoadedModel,
) -> None:
    tiny_arch.model.generation_config.eos_token_id = None
    diff = PromptDifference.from_prompts(
        tiny_arch, DIFF_POSITIVE, DIFF_NEGATIVE, [1, 2], coeff=3.0
    )
    k = diff.vectors[1].shape[0]
    assert min(tiny_arch.encode([p]).lengths[0] for p in DIFF_PROMPTS) < k

    result = generate(tiny_arch, DIFF_PROMPTS, max_new_tokens=4, edits=diff)

    expected = reference_steered_tokens(tiny_arch, diff, DIFF_PROMPTS, 4)
    assert [row[0].token_ids for row in result] == expected


def test_steered_generation_differs_from_plain_generation(
    tiny_arch: LoadedModel,
) -> None:
    tiny_arch.model.generation_config.eos_token_id = None
    diff = PromptDifference.from_prompts(
        tiny_arch, DIFF_POSITIVE, DIFF_NEGATIVE, [1, 2], coeff=50.0
    )

    plain = generate(tiny_arch, DIFF_PROMPTS, max_new_tokens=4)
    steered = generate(tiny_arch, DIFF_PROMPTS, max_new_tokens=4, edits=diff)

    assert steered != plain


@pytest.mark.slow
def test_gpt2_contrast_logits_and_generation_speed() -> None:
    gpt2 = load_model("gpt2")
    sentence = "The quick brown fox jumps over the lazy dog. " * 8
    ids = gpt2.tokenizer(sentence, add_special_tokens=False)["input_ids"]
    prompts = [
        gpt2.tokenizer.decode(ids[start : start + 32])
        for start in range(0, 8 * 3, 3)
    ]
    positive = [p + " Give a truthful answer." for p in prompts]
    negative = [p + " Give an untruthful answer." for p in prompts]
    options = {"layers": [6, 9], "alpha": 0.25}

    batch = gpt2.encode(prompts[:3], padding_side="left")
    assert set(batch.lengths) == {33}
    with torch.no_grad():
        plain = (
            gpt2.model(
                input_ids=batch.input_ids.to(gpt2.device),
                attention_mask=batch.attention_mask.to(gpt2.device),
                position_ids=batch.position_ids.to(gpt2.device),
            )
            .logits[:, -1]
            .float()
            .cpu()
        )
    steered = contrast_logits(
        gpt2, prompts[:3], positive[:3], negative[:3], **options
    )
    start = time.perf_counter()
    completions = contrast_generate(
        gpt2, prompts, positive, negative, max_new_tokens=8, **options
    )
    seconds = time.perf_counter() - start

    assert torch.isfinite(steered).all()
    assert not torch.allclose(steered, plain)
    assert len(completions) == 8
    print(f"contrast_generate: {seconds / 8:.3f} s per generated token")


def decoder_row_counts(loaded: LoadedModel) -> tuple[list[int], Any]:
    rows: list[int] = []

    def record(module: Any, args: Any, kwargs: dict[str, Any]) -> None:
        rows.append(kwargs["input_ids"].shape[0])

    handle = loaded.decoder.register_forward_pre_hook(record, with_kwargs=True)
    return rows, handle


def test_generate_runs_chunks_of_three_times_batch_size_rows(
    tiny_arch: LoadedModel,
) -> None:
    tiny_arch.model.generation_config.eos_token_id = None
    rows, handle = decoder_row_counts(tiny_arch)

    try:
        contrast_generate(
            tiny_arch,
            PROMPTS,
            POSITIVE,
            NEGATIVE,
            max_new_tokens=2,
            batch_size=2,
            **GENERATE,
        )
    finally:
        handle.remove()

    assert set(rows) == {6, 3}


def test_logprobs_run_chunks_of_three_times_batch_size_rows(
    tiny_arch: LoadedModel,
) -> None:
    rows, handle = decoder_row_counts(tiny_arch)

    try:
        contrast_logprobs(
            tiny_arch,
            PROMPTS,
            CONTINUATIONS,
            POSITIVE,
            NEGATIVE,
            layers=(1, 3),
            alpha=0.5,
            batch_size=2,
        )
    finally:
        handle.remove()

    assert rows == [6, 3]


def test_logits_apply_the_lm_head_to_the_last_column_of_the_main_rows_only(
    tiny_arch: LoadedModel,
) -> None:
    shapes: list[tuple[int, ...]] = []

    def record(module: Any, args: Any) -> None:
        shapes.append(tuple(args[0].shape))

    head = tiny_arch.model.get_output_embeddings()
    handle = head.register_forward_pre_hook(record)
    try:
        contrast_logits(
            tiny_arch,
            PROMPTS,
            POSITIVE,
            NEGATIVE,
            layers=(1, 2),
            alpha=0.5,
        )
    finally:
        handle.remove()

    assert shapes == [(len(PROMPTS), tiny_arch.d_model)]
