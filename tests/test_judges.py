import math
import os
import re
from types import SimpleNamespace

from huggingface_hub import constants as hub_constants
import numpy as np
import pytest
import tokenizers
from tokenizers import decoders
from tokenizers import models
from tokenizers import pre_tokenizers
from tokenizers import trainers
import torch
from transformers import BertConfig
from transformers import BertForSequenceClassification
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


def test_mc1_accuracy_counts_questions_whose_best_choice_is_correct() -> None:
    scores = [[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]

    assert judges.mc1_accuracy(scores, [1, 0, 0], seed=0) == pytest.approx(
        2 / 3
    )


def test_mc1_accuracy_accepts_ragged_rows_and_arrays() -> None:
    ragged = [np.array([0.1, 0.5, 0.2]), np.array([0.9, 0.1])]

    assert judges.mc1_accuracy(ragged, np.array([1, 1]), seed=0) == 0.5
    assert judges.mc1_accuracy(np.eye(3), [0, 1, 2], seed=0) == 1.0


def test_mc1_accuracy_only_breaks_ties_among_the_best_choices() -> None:
    scores = [[1.0, 1.0, 0.0]] * 50

    assert judges.mc1_accuracy(scores, [2] * 50, seed=3) == 0.0


def test_a_constant_scorer_is_at_chance_with_the_correct_index_at_zero() -> (
    None
):
    n_questions = 4000
    scores = np.zeros((n_questions, 4))

    accuracy = judges.mc1_accuracy(scores, [0] * n_questions, seed=0)

    assert accuracy == pytest.approx(0.25, abs=0.03)


def test_a_constant_scorer_is_at_chance_on_ragged_questions() -> None:
    sizes = [2, 4, 5] * 1000
    scores = [np.zeros(size) for size in sizes]
    chance = sum(1 / size for size in sizes) / len(sizes)

    accuracy = judges.mc1_accuracy(scores, [0] * len(sizes), seed=1)

    assert accuracy == pytest.approx(chance, abs=0.03)


def test_mc1_tie_breaks_are_reproducible_from_the_seed() -> None:
    scores = np.zeros((200, 4))
    correct = [0] * 200

    first = judges.mc1_accuracy(scores, correct, seed=5)
    again = judges.mc1_accuracy(scores, correct, seed=5)
    others = {judges.mc1_accuracy(scores, correct, seed=s) for s in range(6)}

    assert first == again
    assert len(others) > 1


@pytest.mark.parametrize(
    ("scores", "correct", "message"),
    [
        ([], [], "at least one question"),
        ([[0.1, 0.2]], [0, 1], "same length"),
        ([[0.1, 0.2]], [2], "correct"),
        ([[0.1, 0.2]], [-1], "correct"),
        ([[]], [0], "question 0"),
        ([[0.1, 0.2], [0.1, float("nan")]], [0, 0], "question 1"),
    ],
    ids=[
        "no-questions",
        "length-mismatch",
        "index-too-large",
        "negative-index",
        "empty-row",
        "nan-score",
    ],
)
def test_mc1_accuracy_rejects_invalid_inputs(
    scores: list[list[float]], correct: list[int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        judges.mc1_accuracy(scores, correct, seed=0)


def test_heuristic_score_maps_the_seven_levels_onto_minus_one_to_one() -> None:
    levels = np.linspace(-1.0, 1.0, 7)
    log_probs = np.array(
        [
            [-30.0, -30.0, -30.0, -30.0, -30.0, -30.0, 0.0],
            [0.0, -30.0, -30.0, -30.0, -30.0, -30.0, -30.0],
            [-2.0] * 7,
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0],
        ]
    )
    softmax = np.exp(log_probs[3]) / np.exp(log_probs[3]).sum()

    got = judges.heuristic_score(log_probs)

    assert got.shape == (4,)
    assert got[0] == pytest.approx(1.0, abs=1e-6)
    assert got[1] == pytest.approx(-1.0, abs=1e-6)
    assert got[2] == pytest.approx(0.0, abs=1e-9)
    assert got[3] == pytest.approx(float(softmax @ levels))


def test_heuristic_score_ignores_a_constant_shift_of_the_log_probs() -> None:
    log_probs = np.array([[-1.0, -2.0, -0.5, -4.0, -3.0, -2.5, -1.5]])

    assert judges.heuristic_score(log_probs - 7.0) == pytest.approx(
        judges.heuristic_score(log_probs)
    )


def test_heuristic_score_requires_seven_levels() -> None:
    with pytest.raises(ValueError, match="7 levels"):
        judges.heuristic_score(np.zeros((3, 6)))


SENTIMENT_LABELS = {0: "NEGATIVE", 1: "POSITIVE"}
HARTMANN_LABELS = dict(
    enumerate(
        ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    )
)
CARDIFF_LABELS = dict(
    enumerate(
        [
            "anger",
            "anticipation",
            "disgust",
            "fear",
            "joy",
            "love",
            "optimism",
            "pessimism",
            "sadness",
            "surprise",
            "trust",
        ]
    )
)
JUDGE_TEXTS = [
    "the movie was wonderful and moving",
    "the food was cold and the service was slow",
    "a dog barked at the cat",
    "paris is the capital of france",
    "she sells sea shells by the sea shore",
]


@pytest.fixture(scope="module")
def judge_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {"[PAD]": 0, "[UNK]": 1}
    for text in CORPUS:
        for word in text.split():
            vocab.setdefault(word, len(vocab))
    backend = tokenizers.Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]"
    )


def tiny_classifier(
    id2label: dict[int, str], vocab_size: int
) -> BertForSequenceClassification:
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        initializer_range=1.0,
        id2label=id2label,
    )
    return BertForSequenceClassification(config).eval()


def reference_logits(
    model: BertForSequenceClassification,
    tokenizer: PreTrainedTokenizerFast,
    texts: list[str],
) -> torch.Tensor:
    rows = []
    for text in texts:
        batch = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            rows.append(model(**batch).logits[0].double())
    return torch.stack(rows)


def make_judge(
    key: str,
    id2label: dict[int, str],
    tokenizer: PreTrainedTokenizerFast,
) -> judges.Judge:
    model = tiny_classifier(id2label, len(tokenizer))
    return judges.Judge(judges.JUDGES[key], model, tokenizer)


def test_emotions_keep_the_contract_order() -> None:
    assert judges.EMOTIONS == (
        "happiness",
        "sadness",
        "anger",
        "fear",
        "disgust",
        "surprise",
    )


def test_registry_has_a_primary_and_secondary_judge_per_attribute() -> None:
    pairs = {(spec.attribute, spec.role) for spec in judges.JUDGES.values()}

    assert pairs == {
        ("sentiment", "primary"),
        ("sentiment", "secondary"),
        ("emotion", "primary"),
        ("emotion", "secondary"),
    }
    for key, spec in judges.JUDGES.items():
        assert key == f"{spec.attribute}-{spec.role}"


def test_registry_pins_the_documented_checkpoints() -> None:
    pins = {
        key: (spec.repo_id, spec.revision, spec.head)
        for key, spec in judges.JUDGES.items()
    }

    assert pins == {
        "sentiment-primary": (
            "siebert/sentiment-roberta-large-english",
            "74cea614e245b0832c770ec9aa51bd58df965b9c",
            "softmax",
        ),
        "sentiment-secondary": (
            "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            "714eb0fa89d2f80546fda750413ed43d93601a13",
            "softmax",
        ),
        "emotion-primary": (
            "j-hartmann/emotion-english-distilroberta-base",
            "0e1cd914e3d46199ed785853e12b57304e04178b",
            "softmax",
        ),
        "emotion-secondary": (
            "cardiffnlp/twitter-roberta-base-emotion-latest",
            "415620c4fbc8bd82b82b9fd46642fcec6519d537",
            "sigmoid",
        ),
    }
    assert all(
        re.fullmatch(r"[0-9a-f]{40}", spec.revision)
        for spec in judges.JUDGES.values()
    )


def test_emotion_specs_read_one_model_label_per_emotion() -> None:
    for key in ("emotion-primary", "emotion-secondary"):
        labels = judges.JUDGES[key].labels
        assert len(labels) == len(judges.EMOTIONS)
        assert labels[0] == "joy"
        assert labels[1:] == judges.EMOTIONS[1:]


def test_sentiment_judge_returns_the_positive_probability(
    judge_tokenizer: PreTrainedTokenizerFast,
) -> None:
    judge = make_judge("sentiment-primary", SENTIMENT_LABELS, judge_tokenizer)
    logits = reference_logits(judge.model, judge_tokenizer, JUDGE_TEXTS)
    expected = torch.softmax(logits, dim=-1)[:, 1]

    got = judge.score(JUDGE_TEXTS, batch_size=2)

    assert got.shape == (len(JUDGE_TEXTS),)
    assert got.dtype == np.float32
    assert got == pytest.approx(expected.numpy(), abs=1e-5)


def test_sentiment_judge_finds_the_positive_label_by_name(
    judge_tokenizer: PreTrainedTokenizerFast,
) -> None:
    flipped = {0: "POSITIVE", 1: "NEGATIVE"}
    judge = make_judge("sentiment-secondary", flipped, judge_tokenizer)
    logits = reference_logits(judge.model, judge_tokenizer, JUDGE_TEXTS)
    expected = torch.softmax(logits, dim=-1)[:, 0]

    got = judge.score(JUDGE_TEXTS, batch_size=3)

    assert got == pytest.approx(expected.numpy(), abs=1e-5)


def test_softmax_emotion_judge_drops_neutral_and_follows_emotion_order(
    judge_tokenizer: PreTrainedTokenizerFast,
) -> None:
    judge = make_judge("emotion-primary", HARTMANN_LABELS, judge_tokenizer)
    logits = reference_logits(judge.model, judge_tokenizer, JUDGE_TEXTS)
    columns = [3, 5, 0, 2, 1, 6]
    expected = torch.softmax(logits, dim=-1)[:, columns]

    got = judge.score(JUDGE_TEXTS, batch_size=2)

    assert got.shape == (len(JUDGE_TEXTS), 6)
    assert got == pytest.approx(expected.numpy(), abs=1e-5)
    assert (got.sum(axis=1) < 1.0).all()


def test_sigmoid_emotion_judge_scores_each_emotion_independently(
    judge_tokenizer: PreTrainedTokenizerFast,
) -> None:
    judge = make_judge("emotion-secondary", CARDIFF_LABELS, judge_tokenizer)
    logits = reference_logits(judge.model, judge_tokenizer, JUDGE_TEXTS)
    columns = [4, 8, 0, 3, 2, 9]
    expected = torch.sigmoid(logits)[:, columns]

    got = judge.score(JUDGE_TEXTS, batch_size=4)

    assert got == pytest.approx(expected.numpy(), abs=1e-5)


def test_judge_scores_do_not_depend_on_the_batch_size(
    judge_tokenizer: PreTrainedTokenizerFast,
) -> None:
    judge = make_judge("emotion-primary", HARTMANN_LABELS, judge_tokenizer)

    one = judge.score(JUDGE_TEXTS, batch_size=1)
    many = judge.score(JUDGE_TEXTS, batch_size=5)

    assert many == pytest.approx(one, abs=1e-5)


def test_judge_truncates_to_max_length(
    judge_tokenizer: PreTrainedTokenizerFast,
) -> None:
    judge = make_judge("sentiment-primary", SENTIMENT_LABELS, judge_tokenizer)
    words = CORPUS[2].split()

    truncated = judge.score([" ".join(words)], max_length=4)
    prefix = judge.score([" ".join(words[:4])])

    assert truncated == pytest.approx(prefix, abs=1e-6)


@pytest.mark.parametrize(
    ("key", "id2label", "shape"),
    [
        ("sentiment-primary", SENTIMENT_LABELS, (0,)),
        ("emotion-primary", HARTMANN_LABELS, (0, 6)),
    ],
    ids=["sentiment", "emotion"],
)
def test_judge_scores_no_texts_as_an_empty_array(
    judge_tokenizer: PreTrainedTokenizerFast,
    key: str,
    id2label: dict[int, str],
    shape: tuple[int, ...],
) -> None:
    judge = make_judge(key, id2label, judge_tokenizer)

    got = judge.score([])

    assert got.shape == shape
    assert got.dtype == np.float32


def test_judge_rejects_a_model_without_the_expected_labels(
    judge_tokenizer: PreTrainedTokenizerFast,
) -> None:
    no_joy = {**HARTMANN_LABELS, 3: "happy"}

    with pytest.raises(ValueError, match="joy"):
        make_judge("emotion-primary", no_joy, judge_tokenizer)


def test_judge_rejects_a_non_positive_batch_size(
    judge_tokenizer: PreTrainedTokenizerFast,
) -> None:
    judge = make_judge("sentiment-primary", SENTIMENT_LABELS, judge_tokenizer)

    with pytest.raises(ValueError, match="batch_size"):
        judge.score(JUDGE_TEXTS, batch_size=0)


def test_load_judge_rejects_an_unknown_key_before_any_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("must not reach the Hub")

    monkeypatch.setattr(
        judges, "AutoTokenizer", SimpleNamespace(from_pretrained=forbidden)
    )

    with pytest.raises(ValueError, match="sentiment-primary"):
        judges.load_judge("nonsense")


def test_load_judge_pins_the_revision_and_disables_safetensors_conversion(
    monkeypatch: pytest.MonkeyPatch,
    judge_tokenizer: PreTrainedTokenizerFast,
) -> None:
    monkeypatch.setenv("DISABLE_SAFETENSORS_CONVERSION", "0")
    monkeypatch.delenv("DISABLE_SAFETENSORS_CONVERSION")
    monkeypatch.setattr(hub_constants, "HF_HUB_DISABLE_SYMLINKS_WARNING", False)
    seen: dict[str, object] = {}
    model = tiny_classifier(SENTIMENT_LABELS, len(judge_tokenizer))

    def fake_tokenizer(repo_id: str, **kwargs: str) -> object:
        seen["tokenizer"] = (repo_id, kwargs)
        return judge_tokenizer

    def fake_model(repo_id: str, **kwargs: str) -> object:
        seen["model"] = (repo_id, kwargs)
        seen["conversion"] = os.environ.get("DISABLE_SAFETENSORS_CONVERSION")
        seen["symlinks"] = hub_constants.HF_HUB_DISABLE_SYMLINKS_WARNING
        return model

    monkeypatch.setattr(
        judges, "AutoTokenizer", SimpleNamespace(from_pretrained=fake_tokenizer)
    )
    monkeypatch.setattr(
        judges,
        "AutoModelForSequenceClassification",
        SimpleNamespace(from_pretrained=fake_model),
    )

    judge = judges.load_judge("sentiment-primary", device="cpu")

    spec = judges.JUDGES["sentiment-primary"]
    expected = (spec.repo_id, {"revision": spec.revision})
    assert seen["tokenizer"] == expected
    assert seen["model"] == expected
    assert seen["conversion"] == "1"
    assert seen["symlinks"] is True
    assert judge.spec is spec
    assert judge.model.device == torch.device("cpu")
    assert not judge.model.training


def test_sentiment_accuracy_thresholds_the_positive_probability() -> None:
    p_positive = [0.9, 0.2, 0.6, 0.4]

    assert judges.sentiment_accuracy(p_positive, [1, 0, 0, 0]) == 0.75
    assert (
        judges.sentiment_accuracy(p_positive, [1, 0, 0, 0], threshold=0.7)
        == 1.0
    )


@pytest.mark.parametrize(
    ("p_positive", "labels", "message"),
    [
        ([0.5], [1, 0], "same length"),
        ([], [], "at least one"),
        ([0.5, 0.5], [1, 2], "labels"),
    ],
    ids=["length-mismatch", "empty", "non-binary-labels"],
)
def test_sentiment_accuracy_rejects_invalid_inputs(
    p_positive: list[float], labels: list[int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        judges.sentiment_accuracy(p_positive, labels)


def test_validate_sentiment_judge_scores_texts_against_labels(
    judge_tokenizer: PreTrainedTokenizerFast,
) -> None:
    judge = make_judge("sentiment-primary", SENTIMENT_LABELS, judge_tokenizer)
    labels = [1, 0, 1, 0, 1]
    expected = judges.sentiment_accuracy(judge.score(JUDGE_TEXTS), labels)

    got = judges.validate_sentiment_judge(
        judge, JUDGE_TEXTS, labels, batch_size=2
    )

    assert got == expected


def test_validate_sentiment_judge_rejects_an_emotion_judge(
    judge_tokenizer: PreTrainedTokenizerFast,
) -> None:
    judge = make_judge("emotion-primary", HARTMANN_LABELS, judge_tokenizer)

    with pytest.raises(ValueError, match="sentiment"):
        judges.validate_sentiment_judge(judge, JUDGE_TEXTS, [1, 0, 1, 0, 1])


def test_hard_labels_threshold_vectors_and_take_the_argmax_of_matrices() -> (
    None
):
    assert judges.hard_labels([0.2, 0.5, 0.9]).tolist() == [0, 1, 1]
    assert judges.hard_labels([[0.1, 0.7, 0.2], [0.5, 0.1, 0.4]]).tolist() == [
        1,
        0,
    ]


def test_agreement_is_perfect_for_identical_judges() -> None:
    p = [0.9, 0.1, 0.8, 0.2]

    got = judges.judge_agreement(p, p)

    assert got.rate == 1.0
    assert got.kappa == 1.0


def test_agreement_is_negative_for_opposite_judges() -> None:
    got = judges.judge_agreement([0.9, 0.1, 0.8, 0.2], [0.1, 0.9, 0.2, 0.8])

    assert got.rate == 0.0
    assert got.kappa == -1.0


def test_agreement_matches_cohens_kappa_by_hand() -> None:
    first = [1, 1, 1, 1, 0, 0, 0, 0, 1, 0]
    second = [1, 1, 1, 0, 0, 0, 0, 1, 1, 0]
    observed = 8 / 10
    expected_by_chance = (6 / 10) * (5 / 10) + (4 / 10) * (5 / 10)
    kappa = (observed - expected_by_chance) / (1 - expected_by_chance)

    got = judges.judge_agreement(first, second, threshold=0.5)

    assert got.rate == pytest.approx(observed)
    assert got.kappa == pytest.approx(kappa)


def test_agreement_on_emotion_probabilities_compares_the_argmax() -> None:
    first = [[0.7, 0.1, 0.2], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]]
    second = [[0.5, 0.3, 0.2], [0.4, 0.5, 0.1], [0.6, 0.3, 0.1]]

    got = judges.judge_agreement(first, second)

    assert got.rate == pytest.approx(2 / 3)


def test_agreement_is_defined_when_both_judges_use_a_single_class() -> None:
    got = judges.judge_agreement([0.9, 0.8], [0.7, 0.6])

    assert got.rate == 1.0
    assert got.kappa == 1.0


@pytest.mark.parametrize(
    ("first", "second", "message"),
    [
        ([0.5, 0.5], [0.5], "same shape"),
        ([], [], "at least one"),
    ],
    ids=["shape-mismatch", "empty"],
)
def test_agreement_rejects_invalid_inputs(
    first: list[float], second: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        judges.judge_agreement(first, second)
