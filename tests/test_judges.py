import math

import numpy as np
import pytest
import tokenizers
from tokenizers import decoders
from tokenizers import models
from tokenizers import pre_tokenizers
from tokenizers import trainers
import torch
from transformers import GPT2Config
from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast

from tomography import judges

BOS_TOKEN = "<|endoftext|>"
CORPUS = [
    "the cat sat on the mat",
    "a dog barked at the cat",
    "the quick brown fox jumps over the lazy dog",
    "she sells sea shells by the sea shore",
    "paris is the capital of france",
    "the movie was wonderful and moving",
    "the food was cold and the service was slow",
]


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizerFast:
    backend = tokenizers.Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=300,
        special_tokens=[BOS_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    backend.train_from_iterator(CORPUS * 4, trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token=BOS_TOKEN, eos_token=BOS_TOKEN
    )


@pytest.fixture(scope="module")
def bos_id(tokenizer: PreTrainedTokenizerFast) -> int:
    token_id = tokenizer.convert_tokens_to_ids(BOS_TOKEN)
    assert isinstance(token_id, int)
    return token_id


@pytest.fixture(scope="module")
def gpt2(tokenizer: PreTrainedTokenizerFast) -> GPT2LMHeadModel:
    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=2,
        initializer_range=0.4,
    )
    return GPT2LMHeadModel(config).eval()


def token_ids(tokenizer: PreTrainedTokenizerFast, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def reference_logprobs(model: GPT2LMHeadModel, ids: list[int]) -> list[float]:
    """Log-probs of ids[1:] given their prefixes, one unpadded sequence."""
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0].double()
    log_probs = torch.log_softmax(logits, dim=-1)
    return [log_probs[i, ids[i + 1]].item() for i in range(len(ids) - 1)]


TEXTS = [
    "the cat sat on the mat",
    "paris is the capital of france",
    "a dog barked",
    "the quick brown fox jumps over the lazy dog",
]


def test_perplexity_matches_the_reference_with_bos(
    gpt2: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizerFast,
    bos_id: int,
) -> None:
    ids = [bos_id, *token_ids(tokenizer, TEXTS[0])]
    expected = math.exp(-sum(reference_logprobs(gpt2, ids)) / (len(ids) - 1))

    got = judges.perplexity(
        gpt2, tokenizer, [TEXTS[0]], max_length=64, batch_size=1, bos_id=bos_id
    )

    assert got == pytest.approx(expected, rel=1e-4)


def test_perplexity_without_bos_does_not_score_the_first_token(
    gpt2: GPT2LMHeadModel, tokenizer: PreTrainedTokenizerFast
) -> None:
    ids = token_ids(tokenizer, TEXTS[0])
    expected = math.exp(-sum(reference_logprobs(gpt2, ids)) / (len(ids) - 1))

    got = judges.perplexity(
        gpt2, tokenizer, [TEXTS[0]], max_length=64, batch_size=1
    )

    assert got == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("batch_size", [1, 3, 8], ids=["b1", "b3", "b8"])
def test_perplexity_pools_tokens_across_texts_and_batches(
    gpt2: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizerFast,
    bos_id: int,
    batch_size: int,
) -> None:
    total_nll = 0.0
    total_tokens = 0
    for text in TEXTS:
        ids = [bos_id, *token_ids(tokenizer, text)]
        total_nll -= sum(reference_logprobs(gpt2, ids))
        total_tokens += len(ids) - 1
    expected = math.exp(total_nll / total_tokens)

    got = judges.perplexity(
        gpt2,
        tokenizer,
        TEXTS,
        max_length=64,
        batch_size=batch_size,
        bos_id=bos_id,
    )

    assert got == pytest.approx(expected, rel=1e-4)


def test_perplexity_truncates_to_max_length_counting_the_bos(
    gpt2: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizerFast,
    bos_id: int,
) -> None:
    text = TEXTS[3]
    ids = [bos_id, *token_ids(tokenizer, text)][:5]
    expected = math.exp(-sum(reference_logprobs(gpt2, ids)) / (len(ids) - 1))

    got = judges.perplexity(
        gpt2, tokenizer, [text], max_length=5, batch_size=2, bos_id=bos_id
    )

    assert got == pytest.approx(expected, rel=1e-4)


def test_perplexity_ignores_documents_with_nothing_to_predict(
    gpt2: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizerFast,
    bos_id: int,
) -> None:
    single = judges.perplexity(
        gpt2, tokenizer, TEXTS[:1], max_length=64, batch_size=2
    )

    with_empty = judges.perplexity(
        gpt2, tokenizer, ["", TEXTS[0]], max_length=64, batch_size=2
    )

    assert with_empty == pytest.approx(single, rel=1e-4)


@pytest.mark.parametrize(
    ("texts", "max_length", "batch_size", "message"),
    [
        ([], 8, 1, "texts must not be empty"),
        (["a b"], 1, 1, "max_length"),
        (["a b"], 8, 0, "batch_size"),
        ([""], 8, 1, "no text has a token to predict"),
    ],
    ids=["no-texts", "short-max-length", "bad-batch-size", "nothing-to-score"],
)
def test_perplexity_rejects_invalid_arguments(
    gpt2: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizerFast,
    texts: list[str],
    max_length: int,
    batch_size: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        judges.perplexity(
            gpt2,
            tokenizer,
            texts,
            max_length=max_length,
            batch_size=batch_size,
        )


def test_perplexity_passes_explicit_position_ids_for_padded_batches(
    gpt2: GPT2LMHeadModel, tokenizer: PreTrainedTokenizerFast
) -> None:
    seen: list[torch.Tensor] = []

    def record(
        _module: torch.nn.Module,
        _args: tuple[object, ...],
        kwargs: dict[str, torch.Tensor],
    ) -> None:
        seen.append(kwargs["position_ids"])
        expected = (kwargs["attention_mask"].cumsum(dim=-1) - 1).clamp(min=0)
        assert torch.equal(kwargs["position_ids"], expected)

    handle = gpt2.register_forward_pre_hook(record, with_kwargs=True)
    try:
        judges.perplexity(gpt2, tokenizer, TEXTS, max_length=64, batch_size=4)
    finally:
        handle.remove()

    assert len(seen) == 1
    assert seen[0].shape[0] == len(TEXTS)


def test_perplexity_runs_under_the_hooks_the_caller_has_active(
    gpt2: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizerFast,
    bos_id: int,
) -> None:
    def zero_block_output(
        _module: torch.nn.Module,
        _args: tuple[object, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros_like(output)

    plain = judges.perplexity(
        gpt2, tokenizer, TEXTS, max_length=64, batch_size=2, bos_id=bos_id
    )
    handle = gpt2.transformer.h[0].register_forward_hook(zero_block_output)
    try:
        hooked = judges.perplexity(
            gpt2, tokenizer, TEXTS, max_length=64, batch_size=2, bos_id=bos_id
        )
    finally:
        handle.remove()

    assert hooked != pytest.approx(plain, rel=1e-3)


def test_perplexity_leaves_the_model_gradients_untouched(
    gpt2: GPT2LMHeadModel, tokenizer: PreTrainedTokenizerFast
) -> None:
    judges.perplexity(gpt2, tokenizer, TEXTS, max_length=64, batch_size=2)

    assert all(parameter.grad is None for parameter in gpt2.parameters())


PROMPTS = [
    "paris is the capital of",
    "the cat sat on",
    "a dog",
    "the quick brown fox jumps over",
]
CONTINUATIONS = [" france", " the mat", " barked at the cat", " the lazy dog"]


def reference_continuation_logprob(
    model: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizerFast,
    prompt: str,
    continuation: str,
    bos_id: int | None,
) -> float:
    prefix = [] if bos_id is None else [bos_id]
    prompt_ids = prefix + token_ids(tokenizer, prompt)
    ids = prompt_ids + token_ids(tokenizer, continuation)
    return sum(reference_logprobs(model, ids)[len(prompt_ids) - 1 :])


@pytest.mark.parametrize("use_bos", [True, False], ids=["bos", "no-bos"])
def test_sequence_logprob_sums_the_continuation_tokens_only(
    gpt2: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizerFast,
    bos_id: int,
    use_bos: bool,
) -> None:
    bos = bos_id if use_bos else None
    expected = [
        reference_continuation_logprob(gpt2, tokenizer, p, c, bos)
        for p, c in zip(PROMPTS, CONTINUATIONS, strict=True)
    ]

    got = judges.sequence_logprob(
        gpt2, tokenizer, PROMPTS, CONTINUATIONS, batch_size=4, bos_id=bos
    )

    assert got.shape == (len(PROMPTS),)
    assert got.dtype == np.float32
    assert got == pytest.approx(expected, rel=1e-4)


def test_sequence_logprob_tokenizes_prompt_and_continuation_separately(
    gpt2: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizerFast,
    bos_id: int,
) -> None:
    prompt, continuation = "the ca", "t sat on the mat"
    joined = token_ids(tokenizer, prompt + continuation)
    separate = token_ids(tokenizer, prompt) + token_ids(tokenizer, continuation)
    assert joined != separate
    expected = reference_continuation_logprob(
        gpt2, tokenizer, prompt, continuation, bos_id
    )

    got = judges.sequence_logprob(
        gpt2, tokenizer, [prompt], [continuation], batch_size=1, bos_id=bos_id
    )

    assert got[0] == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("batch_size", [1, 2, 7], ids=["b1", "b2", "b7"])
def test_sequence_logprob_is_invariant_to_batching(
    gpt2: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizerFast,
    bos_id: int,
    batch_size: int,
) -> None:
    single = judges.sequence_logprob(
        gpt2, tokenizer, PROMPTS, CONTINUATIONS, batch_size=1, bos_id=bos_id
    )

    batched = judges.sequence_logprob(
        gpt2,
        tokenizer,
        PROMPTS,
        CONTINUATIONS,
        batch_size=batch_size,
        bos_id=bos_id,
    )

    assert batched == pytest.approx(single, rel=1e-4, abs=1e-3)


def test_sequence_logprob_scores_the_first_token_after_a_lone_bos(
    gpt2: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizerFast,
    bos_id: int,
) -> None:
    expected = reference_continuation_logprob(
        gpt2, tokenizer, "", "the cat", bos_id
    )

    got = judges.sequence_logprob(
        gpt2, tokenizer, [""], ["the cat"], batch_size=1, bos_id=bos_id
    )

    assert got[0] == pytest.approx(expected, rel=1e-4)


def test_sequence_logprob_of_nothing_is_an_empty_float32_array(
    gpt2: GPT2LMHeadModel, tokenizer: PreTrainedTokenizerFast
) -> None:
    got = judges.sequence_logprob(gpt2, tokenizer, [], [], batch_size=2)

    assert got.shape == (0,)
    assert got.dtype == np.float32


def test_sequence_logprob_returns_float32_for_a_bfloat16_model(
    tokenizer: PreTrainedTokenizerFast, bos_id: int
) -> None:
    torch.manual_seed(1)
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=2,
    )
    model = GPT2LMHeadModel(config).eval().to(torch.bfloat16)

    got = judges.sequence_logprob(
        model, tokenizer, PROMPTS, CONTINUATIONS, batch_size=2, bos_id=bos_id
    )

    assert got.dtype == np.float32
    assert np.isfinite(got).all()


@pytest.mark.parametrize(
    ("prompts", "continuations", "bos", "message"),
    [
        (["a"], ["b", "c"], True, "same length"),
        (["a b"], [""], True, "continuation 0"),
        (["a b", ""], [" c", " d"], False, "prompt 1"),
    ],
    ids=["length-mismatch", "empty-continuation", "empty-prompt-without-bos"],
)
def test_sequence_logprob_rejects_unscorable_inputs(
    gpt2: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizerFast,
    bos_id: int,
    prompts: list[str],
    continuations: list[str],
    bos: bool,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        judges.sequence_logprob(
            gpt2,
            tokenizer,
            prompts,
            continuations,
            batch_size=2,
            bos_id=bos_id if bos else None,
        )


def test_sequence_logprob_rejects_a_non_positive_batch_size(
    gpt2: GPT2LMHeadModel, tokenizer: PreTrainedTokenizerFast
) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        judges.sequence_logprob(gpt2, tokenizer, ["a b"], [" c"], batch_size=0)
