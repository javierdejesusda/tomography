"""Judges and language-model scoring helpers.

The module holds the text judges (pinned sentiment and emotion classifiers
with validation and agreement helpers), the pinned entailment scorer for text
pairs, and the language-model helpers used for side-effect and multiple-choice
scoring: perplexity, sequence log-probability, MC1 accuracy and the verbalized
heuristic baseline.

Tokenization for the language-model helpers follows the policy of
`LoadedModel.encode`: no special tokens are added by the tokenizer, sequences
are right padded, position ids are explicit, and a BOS id is prepended only
when the caller passes one. The helpers take plain Hugging Face objects and run
under whatever hooks are active when they are called.
"""

from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
import math
import os
from typing import Literal

from huggingface_hub import constants as hub_constants
import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray
import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import DynamicCache
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerBase

EMOTIONS = ("happiness", "sadness", "anger", "fear", "disgust", "surprise")


def _tokenize(
    tokenizer: PreTrainedTokenizerBase, texts: Sequence[str]
) -> list[list[int]]:
    encoded = tokenizer(list(texts), add_special_tokens=False)
    return [list(ids) for ids in encoded["input_ids"]]


def _next_token_logprobs(
    model: PreTrainedModel,
    sequences: Sequence[Sequence[int]],
    batch_size: int,
) -> list[NDArray[np.float32]]:
    """Scores every token after the first of each sequence.

    Args:
        model: A causal language model in eval mode.
        sequences: Token id sequences, each with at least two tokens.
        batch_size: Sequences per forward pass.

    Returns:
        One float32 array per sequence, in the input order, of length
        `len(sequence) - 1`: entry j is the log-probability of token j + 1
        given tokens 0..j. Batches are formed longest first to limit padding.
    """
    results: list[NDArray[np.float32]] = [np.empty(0, dtype=np.float32)] * len(
        sequences
    )
    order = sorted(range(len(sequences)), key=lambda i: -len(sequences[i]))
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            chunk = [sequences[i] for i in indices]
            width = len(chunk[0])
            input_ids = torch.zeros((len(chunk), width), dtype=torch.long)
            mask = torch.zeros((len(chunk), width), dtype=torch.long)
            for row, sequence in enumerate(chunk):
                input_ids[row, : len(sequence)] = torch.tensor(
                    sequence, dtype=torch.long
                )
                mask[row, : len(sequence)] = 1
            position_ids = (mask.cumsum(dim=-1) - 1).clamp(min=0)
            logits = model(
                input_ids=input_ids.to(model.device),
                attention_mask=mask.to(model.device),
                position_ids=position_ids.to(model.device),
                use_cache=False,
            ).logits
            for row, sequence in enumerate(chunk):
                length = len(sequence)
                targets = input_ids[row, 1:length].to(model.device)
                picked = (
                    logits[row, : length - 1]
                    .float()
                    .log_softmax(-1)
                    .gather(-1, targets[:, None])
                    .squeeze(-1)
                )
                results[indices[row]] = picked.cpu().numpy()
            del logits
    return results


def perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    *,
    max_length: int,
    batch_size: int,
    bos_id: int | None = None,
) -> float:
    """Computes the token-level perplexity of a corpus.

    Each text is tokenized without special tokens, prefixed with `bos_id` when
    given, and truncated to `max_length` tokens (the BOS counts). Without a BOS
    the first token of a text has no context and is not scored. The result
    pools every scored token of every text, so long texts weigh more than short
    ones. Texts with nothing to predict are skipped.

    Args:
        model: A causal language model in eval mode.
        tokenizer: The tokenizer that matches the model.
        texts: The documents to score.
        max_length: Longest sequence fed to the model, BOS included.
        batch_size: Sequences per forward pass.
        bos_id: Token id prepended to every sequence, or None.

    Returns:
        The exponential of the mean negative log-likelihood per scored token.

    Raises:
        ValueError: If `texts` is empty, `max_length` is below 2,
            `batch_size` is below 1, or no text has a token to predict.
    """
    if not texts:
        raise ValueError("texts must not be empty")
    if max_length < 2:
        raise ValueError(f"max_length must be at least 2, got {max_length}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    prefix = [] if bos_id is None else [bos_id]
    limit = max_length - len(prefix)
    sequences = [prefix + ids[:limit] for ids in _tokenize(tokenizer, texts)]
    sequences = [sequence for sequence in sequences if len(sequence) >= 2]
    if not sequences:
        raise ValueError("no text has a token to predict")
    scored = _next_token_logprobs(model, sequences, batch_size)
    total = sum(float(values.sum(dtype=np.float64)) for values in scored)
    count = sum(values.size for values in scored)
    return math.exp(-total / count)


def sequence_logprob(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    continuations: Sequence[str],
    *,
    batch_size: int,
    bos_id: int | None = None,
) -> NDArray[np.float32]:
    """Scores continuations given prompts.

    Prompt and continuation are tokenized separately and joined, so the
    continuation keeps its leading space and the token boundary is the same for
    every pairing. The score is the float32 sum of the log-probabilities of the
    continuation tokens only, conditioned on `bos_id` (when given), the prompt
    and the earlier continuation tokens.

    Args:
        model: A causal language model in eval mode.
        tokenizer: The tokenizer that matches the model.
        prompts: One prompt per pair.
        continuations: One continuation per pair, same length as `prompts`.
        batch_size: Sequences per forward pass.
        bos_id: Token id prepended to every sequence, or None.

    Returns:
        A float32 array of shape `(len(prompts),)`.

    Raises:
        ValueError: If the inputs differ in length, `batch_size` is below 1, a
            continuation has no tokens, or a prompt has no tokens and no
            `bos_id` is given (the first continuation token would have no
            context).
    """
    if len(prompts) != len(continuations):
        raise ValueError("prompts and continuations must have the same length")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if not prompts:
        return np.zeros(0, dtype=np.float32)
    pairs = _tokenize_pairs(tokenizer, prompts, continuations, bos_id)
    scored = _next_token_logprobs(
        model,
        [context + continuation for context, continuation in pairs],
        batch_size,
    )
    return np.array(
        [
            values[len(context) - 1 :].sum(dtype=np.float32)
            for values, (context, _) in zip(scored, pairs, strict=True)
        ],
        dtype=np.float32,
    )


def _tokenize_pairs(
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    continuations: Sequence[str],
    bos_id: int | None,
) -> list[tuple[list[int], list[int]]]:
    """Tokenizes prompt and continuation pairs for scoring.

    Args:
        tokenizer: The tokenizer that matches the model.
        prompts: One prompt per pair.
        continuations: One continuation per pair, same length as `prompts`.
        bos_id: Token id that starts every context, or None.

    Returns:
        One `(context_ids, continuation_ids)` pair per input pair. The context
        is `bos_id` (when given) followed by the prompt tokens; prompt and
        continuation are tokenized separately.

    Raises:
        ValueError: If a continuation has no tokens, or a prompt has no tokens
            and no `bos_id` is given.
    """
    prefix = [] if bos_id is None else [bos_id]
    pairs: list[tuple[list[int], list[int]]] = []
    tokenized = zip(
        _tokenize(tokenizer, prompts),
        _tokenize(tokenizer, continuations),
        strict=True,
    )
    for index, (prompt_ids, continuation_ids) in enumerate(tokenized):
        if not continuation_ids:
            raise ValueError(f"continuation {index} has no tokens")
        if not prefix and not prompt_ids:
            raise ValueError(
                f"prompt {index} has no tokens and no bos_id was given"
            )
        pairs.append((prefix + prompt_ids, continuation_ids))
    return pairs


def continuation_logprob(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    continuations: Sequence[str],
    *,
    bos_id: int | None = None,
) -> NDArray[np.float32]:
    """Scores continuations given prompts, with logits only where needed.

    Pairs are tokenized and scored as in `sequence_logprob`: prompt and
    continuation are tokenized separately and joined after `bos_id` (when
    given), and the score is the float32 sum of the log-probabilities of the
    continuation tokens. Each sequence runs alone and the LM head sees only
    the positions that predict a continuation token, so the logits of a pair
    take `len(continuation) * vocab` floats however long the prompt is.

    Contexts (`bos_id` plus the prompt tokens) share a key-value cache: the
    longest token prefix common to all contexts runs once, each distinct
    context runs once for all of its continuations, and the cache is cropped
    back after every continuation, so a score does not depend on the other
    continuations of its context or on their order. Passing every choice of
    every question in one call shares a few-shot context across questions.
    Hooks see each sequence in chunks: the shared prefix, the rest of the
    context but its last token, then that token with the continuation.

    Scores match `sequence_logprob` up to float32 rounding, and at that level
    they also depend on where the shared prefix splits a context, hence on the
    other contexts of the call. For a prompt that does not end in whitespace
    and a continuation that starts with a space, such as `" " + choice`, the
    GPT-2 and Qwen3 pre-tokenizers break at the join, so the tokens equal
    those of the joint encoding that lm-evaluation-harness uses.

    Args:
        model: A causal language model in eval mode.
        tokenizer: The tokenizer that matches the model.
        prompts: One prompt per pair.
        continuations: One continuation per pair, same length as `prompts`.
        bos_id: Token id prepended to every sequence, or None.

    Returns:
        A float32 array of shape `(len(prompts),)`.

    Raises:
        ValueError: If the inputs differ in length, a continuation has no
            tokens, or a prompt has no tokens and no `bos_id` is given.
    """
    if len(prompts) != len(continuations):
        raise ValueError("prompts and continuations must have the same length")
    if not prompts:
        return np.zeros(0, dtype=np.float32)
    pairs = _tokenize_pairs(tokenizer, prompts, continuations, bos_id)
    groups: dict[tuple[int, ...], list[int]] = {}
    for index, pair in enumerate(pairs):
        groups.setdefault(tuple(pair[0]), []).append(index)
    contexts = list(groups)
    # The cache stops one token short of each context: every continuation pass
    # starts with the last context token, whose logits predict the first
    # continuation token, even when the whole context is shared.
    common = max(_shared_prefix_length(contexts) - 1, 0)
    scores = np.zeros(len(pairs), dtype=np.float32)
    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        _extend_cache(model, cache, contexts[0][:common])
        for context in contexts:
            _extend_cache(model, cache, context[common:-1])
            for index in groups[context]:
                continuation = pairs[index][1]
                logits = _cached_logits(
                    model,
                    cache,
                    [context[-1], *continuation[:-1]],
                    len(continuation),
                )
                targets = torch.tensor(continuation, device=logits.device)
                picked = (
                    logits[0]
                    .float()
                    .log_softmax(-1)
                    .gather(-1, targets[:, None])
                    .squeeze(-1)
                )
                scores[index] = picked.cpu().numpy().sum(dtype=np.float32)
                _truncate_cache(cache, len(context) - 1)
            _truncate_cache(cache, common)
    return scores


def _shared_prefix_length(rows: Sequence[Sequence[int]]) -> int:
    length = 0
    for column in zip(*rows, strict=False):
        if len(set(column)) > 1:
            break
        length += 1
    return length


def _cached_logits(
    model: PreTrainedModel,
    cache: DynamicCache,
    ids: Sequence[int],
    keep: int | torch.Tensor,
) -> torch.Tensor:
    """Runs tokens after the cached ones and extends the cache in place.

    Args:
        model: A causal language model in eval mode.
        cache: The cache of the tokens before `ids`, extended by `ids`.
        ids: Token ids to run, at the positions right after the cache.
        keep: Positions whose logits are computed, passed to the model as
            `logits_to_keep`: the last `keep` positions for an int, the
            listed positions for a tensor.

    Returns:
        The logits of the kept positions, shape `(1, kept, vocab)`.
    """
    start = cache.get_seq_length()
    end = start + len(ids)
    logits: torch.Tensor = model(
        input_ids=torch.tensor([ids], device=model.device),
        attention_mask=torch.ones(
            (1, end), dtype=torch.long, device=model.device
        ),
        position_ids=torch.arange(start, end, device=model.device)[None],
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=keep,
    ).logits
    return logits


def _extend_cache(
    model: PreTrainedModel, cache: DynamicCache, ids: Sequence[int]
) -> None:
    if ids:
        # An empty index keeps no position; logits_to_keep=0 would keep all.
        no_position = torch.zeros(0, dtype=torch.long, device=model.device)
        _cached_logits(model, cache, ids, no_position)


def _truncate_cache(cache: DynamicCache, length: int) -> None:
    extra = cache.get_seq_length() - length
    if extra > 0:
        cache.crop(-extra)


def mc1_accuracy(
    scores: Iterable[ArrayLike], correct: ArrayLike, *, seed: int
) -> float:
    """Computes multiple-choice accuracy with random tie-breaking.

    A question counts as correct when its highest-scoring choice is the correct
    one. Ties among the highest scores are broken uniformly at random from
    `seed`, so a scorer that ignores its input sits at chance even when the
    correct choice has a fixed position.

    Args:
        scores: One 1-D array of choice scores per question. Questions may have
            different numbers of choices.
        correct: The index of the correct choice for each question.
        seed: Seed of the tie-breaking generator.

    Returns:
        The share of questions answered correctly.

    Raises:
        ValueError: If there are no questions, `scores` and `correct` differ in
            length, a question has no choices or a NaN score, or a correct
            index is out of range.
    """
    rows = [np.asarray(row, dtype=np.float64) for row in scores]
    answers = np.asarray(correct, dtype=np.int64)
    if not rows:
        raise ValueError("mc1_accuracy needs at least one question")
    if answers.shape != (len(rows),):
        raise ValueError("scores and correct must have the same length")
    rng = np.random.default_rng(seed)
    hits = 0
    for index, (row, answer) in enumerate(zip(rows, answers, strict=True)):
        if row.ndim != 1 or row.size == 0:
            raise ValueError(f"question {index} has no choice scores")
        if np.isnan(row).any():
            raise ValueError(f"question {index} has a NaN score")
        if not 0 <= answer < row.size:
            raise ValueError(
                f"correct index {answer} of question {index} is out of range"
            )
        best = np.flatnonzero(row == row.max())
        winner = best[0] if best.size == 1 else rng.choice(best)
        hits += int(winner == answer)
    return hits / len(rows)


HEURISTIC_LEVELS = 7


def heuristic_score(level_logprobs: ArrayLike) -> NDArray[np.float64]:
    """Turns verbalized-level log-probabilities into a score in [-1, 1].

    The seven levels run from the most negative verbalization to the most
    positive one and map to evenly spaced values from -1 to 1. The score is the
    expectation of that value under the softmax of the log-probabilities, so
    log-probabilities that are not normalized over the seven levels are fine.

    Args:
        level_logprobs: Array of shape `(..., 7)`.

    Returns:
        A float64 array of shape `level_logprobs.shape[:-1]`.

    Raises:
        ValueError: If the last axis does not hold 7 levels.
    """
    values = np.asarray(level_logprobs, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != HEURISTIC_LEVELS:
        raise ValueError(
            f"expected {HEURISTIC_LEVELS} levels on the last axis, "
            f"got shape {values.shape}"
        )
    weights = np.exp(values - values.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    result: NDArray[np.float64] = weights @ np.linspace(
        -1.0, 1.0, HEURISTIC_LEVELS
    )
    return result


Attribute = Literal["sentiment", "emotion"]
Role = Literal["primary", "secondary"]
Head = Literal["softmax", "sigmoid"]


@dataclass(frozen=True)
class JudgeSpec:
    """One pinned classifier used to judge generated text.

    Attributes:
        attribute: What the judge measures.
        role: Headline judge (primary) or robustness check (secondary).
        repo_id: Hugging Face repository of the checkpoint.
        revision: Full commit SHA of the checkpoint.
        license: License declared by the checkpoint repository.
        head: How logits become probabilities: a softmax over all classes or
            an independent sigmoid per class.
        labels: Model labels to read, in output order. For sentiment the single
            positive label; for emotion one label per entry of `EMOTIONS`
            (happiness is read from the model label "joy").
    """

    attribute: Attribute
    role: Role
    repo_id: str
    revision: str
    license: str
    head: Head
    labels: tuple[str, ...]

    @property
    def key(self) -> str:
        """The registry key, `<attribute>-<role>`."""
        return f"{self.attribute}-{self.role}"


_EMOTION_LABELS = ("joy", *EMOTIONS[1:])

JUDGES: Mapping[str, JudgeSpec] = {
    spec.key: spec
    for spec in (
        JudgeSpec(
            attribute="sentiment",
            role="primary",
            repo_id="siebert/sentiment-roberta-large-english",
            revision="74cea614e245b0832c770ec9aa51bd58df965b9c",
            license="none declared",
            head="softmax",
            labels=("POSITIVE",),
        ),
        JudgeSpec(
            attribute="sentiment",
            role="secondary",
            repo_id="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            revision="714eb0fa89d2f80546fda750413ed43d93601a13",
            license="apache-2.0",
            head="softmax",
            labels=("POSITIVE",),
        ),
        JudgeSpec(
            attribute="emotion",
            role="primary",
            repo_id="j-hartmann/emotion-english-distilroberta-base",
            revision="0e1cd914e3d46199ed785853e12b57304e04178b",
            license="none declared",
            head="softmax",
            labels=_EMOTION_LABELS,
        ),
        JudgeSpec(
            attribute="emotion",
            role="secondary",
            repo_id="cardiffnlp/twitter-roberta-base-emotion-latest",
            revision="415620c4fbc8bd82b82b9fd46642fcec6519d537",
            license="mit",
            head="sigmoid",
            labels=_EMOTION_LABELS,
        ),
    )
}


def _label_columns(model: PreTrainedModel, labels: Sequence[str]) -> list[int]:
    id2label = model.config.id2label or {}
    by_name = {name.lower(): int(index) for index, name in id2label.items()}
    missing = [label for label in labels if label.lower() not in by_name]
    if missing:
        raise ValueError(
            f"model has no label {missing}; its labels are {sorted(by_name)}"
        )
    return [by_name[label.lower()] for label in labels]


def _class_probabilities(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    text_pairs: Sequence[str] | None,
    *,
    head: Head,
    labels: Sequence[str],
    batch_size: int,
    max_length: int,
) -> NDArray[np.float32]:
    """Runs a sequence classifier in batches and reads some of its classes.

    Args:
        model: A sequence classification model in eval mode.
        tokenizer: The tokenizer that matches the model.
        texts: The texts, or the first text of each pair.
        text_pairs: The second text of each pair, or None to score single
            texts. It has the length of `texts`.
        head: How logits become probabilities.
        labels: Model labels to return, in output order.
        batch_size: Texts per forward pass.
        max_length: Longest tokenized input, special tokens included.

    Returns:
        Probabilities of shape `(len(texts), len(labels))`, float32.

    Raises:
        ValueError: If `batch_size` is below 1, `text_pairs` differs in length
            from `texts`, or the model lacks one of `labels`.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if text_pairs is not None and len(text_pairs) != len(texts):
        raise ValueError(
            "texts and text_pairs must have the same length, got "
            f"{len(texts)} and {len(text_pairs)}"
        )
    columns = _label_columns(model, labels)
    chunks: list[NDArray[np.float32]] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            stop = start + batch_size
            batch = tokenizer(
                list(texts[start:stop]),
                None if text_pairs is None else list(text_pairs[start:stop]),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(model.device)
            logits = model(**batch).logits.float()
            if head == "softmax":
                probabilities = logits.softmax(dim=-1)
            else:
                probabilities = logits.sigmoid()
            chunks.append(probabilities[:, columns].cpu().numpy())
    if chunks:
        return np.concatenate(chunks)
    return np.zeros((0, len(columns)), dtype=np.float32)


@dataclass(frozen=True, eq=False)
class Judge:
    """A sequence classifier that scores texts for one attribute.

    Attributes:
        spec: The pinned checkpoint description.
        model: A sequence classification model in eval mode.
        tokenizer: The tokenizer that matches the model.
    """

    spec: JudgeSpec
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase

    def __post_init__(self) -> None:
        """Checks that the model exposes every label the spec reads.

        Raises:
            ValueError: If a label of the spec is missing from the model.
        """
        _label_columns(self.model, self.spec.labels)

    def score(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> NDArray[np.float32]:
        """Scores texts with the classifier.

        Texts are tokenized with the classifier's own special tokens, padded
        per batch and truncated to `max_length`. For emotion the probabilities
        of the six emotions are read as the model reports them: a neutral class
        keeps its mass out of the six columns, so rows can sum to less than one.

        Args:
            texts: The texts to score.
            batch_size: Texts per forward pass.
            max_length: Longest tokenized text, special tokens included.

        Returns:
            For sentiment, the probability of the positive class with shape
            `(n,)`. For emotion, probabilities in `EMOTIONS` order with shape
            `(n, 6)`. Always float32.

        Raises:
            ValueError: If `batch_size` is below 1.
        """
        scores = _class_probabilities(
            self.model,
            self.tokenizer,
            texts,
            None,
            head=self.spec.head,
            labels=self.spec.labels,
            batch_size=batch_size,
            max_length=max_length,
        )
        if self.spec.attribute == "sentiment":
            return np.ascontiguousarray(scores[:, 0])
        return scores


def load_judge(key: str, *, device: str | torch.device | None = None) -> Judge:
    """Loads a pinned judge from the Hugging Face Hub.

    The revision is the pinned commit SHA. Automatic safetensors conversion is
    disabled before loading because two judges ship only `.bin` weights and
    transformers would otherwise try to open conversion pull requests on them.

    Args:
        key: A key of `JUDGES`, such as `sentiment-primary`.
        device: Where to place the model. None picks CUDA when available,
            else the CPU.

    Returns:
        The judge, with its model in eval mode.

    Raises:
        ValueError: If `key` is not a registered judge.
    """
    spec = JUDGES.get(key)
    if spec is None:
        raise ValueError(f"unknown judge {key!r}; choose from {sorted(JUDGES)}")
    model, tokenizer = _load_classifier(spec.repo_id, spec.revision, device)
    return Judge(spec, model, tokenizer)


def _load_classifier(
    repo_id: str, revision: str, device: str | torch.device | None
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Loads a pinned sequence classifier and its tokenizer in eval mode."""
    hub_constants.HF_HUB_DISABLE_SYMLINKS_WARNING = True
    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
    tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        repo_id, revision=revision
    )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval(), tokenizer


NLI_LABELS = ("contradiction", "neutral", "entailment")


@dataclass(frozen=True)
class NliSpec:
    """One pinned natural language inference checkpoint.

    Attributes:
        repo_id: Hugging Face repository of the checkpoint.
        revision: Full commit SHA of the checkpoint.
        license: License declared by the checkpoint repository.
    """

    repo_id: str
    revision: str
    license: str


# DeBERTa-large MNLI is the entailment model of Kuhn et al. (2023), whose code
# loads it, and the one the text of Farquhar et al. (2024) names. The
# reference code of Farquhar et al. runs microsoft/deberta-v2-xlarge-mnli
# instead, whose tokenizer needs sentencepiece, which this stack does not
# install. Both checkpoints decide entailment the same way: the most likely of
# the three classes, with no threshold.
NLI = NliSpec(
    repo_id="microsoft/deberta-large-mnli",
    revision="7296194b9009373def4f7c5dad292651e4b5cf4e",
    license="mit",
)


@dataclass(frozen=True, eq=False)
class NliScorer:
    """A sequence classifier that scores a premise against a hypothesis.

    Attributes:
        spec: The pinned checkpoint description.
        model: A sequence classification model in eval mode whose labels
            include `NLI_LABELS`.
        tokenizer: The tokenizer that matches the model.
    """

    spec: NliSpec
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase

    def __post_init__(self) -> None:
        """Checks that the model exposes the three inference classes.

        Raises:
            ValueError: If a class of `NLI_LABELS` is missing from the model.
        """
        _label_columns(self.model, NLI_LABELS)

    def probabilities(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
        *,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> NDArray[np.float32]:
        """Scores each premise against its hypothesis.

        Each pair is tokenized as a text pair with the classifier's own
        special tokens, padded per batch and truncated to `max_length`. The
        classes are found by name in the checkpoint's `id2label`, so the
        columns follow `NLI_LABELS` whatever order the checkpoint uses.

        Args:
            premises: The premises.
            hypotheses: The hypotheses, one per premise.
            batch_size: Pairs per forward pass.
            max_length: Longest tokenized pair, special tokens included.

        Returns:
            Class probabilities of shape `(n, 3)` in `NLI_LABELS` order
            (contradiction, neutral, entailment). Always float32.

        Raises:
            ValueError: If `premises` and `hypotheses` differ in length, or
                `batch_size` is below 1.
        """
        return _class_probabilities(
            self.model,
            self.tokenizer,
            premises,
            hypotheses,
            head="softmax",
            labels=NLI_LABELS,
            batch_size=batch_size,
            max_length=max_length,
        )

    def entails(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
        *,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> NDArray[np.bool_]:
        """Tells whether each premise entails its hypothesis.

        A premise entails a hypothesis when entailment is the most likely of
        the three classes, with no probability threshold, as in the reference
        code of semantic entropy.

        Args:
            premises: The premises.
            hypotheses: The hypotheses, one per premise.
            batch_size: Pairs per forward pass.
            max_length: Longest tokenized pair, special tokens included.

        Returns:
            A boolean array of shape `(n,)`.

        Raises:
            ValueError: If `premises` and `hypotheses` differ in length, or
                `batch_size` is below 1.
        """
        probabilities = self.probabilities(
            premises, hypotheses, batch_size=batch_size, max_length=max_length
        )
        entailment = NLI_LABELS.index("entailment")
        verdicts: NDArray[np.bool_] = probabilities.argmax(axis=1) == entailment
        return verdicts


def load_nli(*, device: str | torch.device | None = None) -> NliScorer:
    """Loads the pinned entailment model from the Hugging Face Hub.

    The revision is the pinned commit SHA, and automatic safetensors
    conversion is disabled as in `load_judge`.

    Args:
        device: Where to place the model. None picks CUDA when available,
            else the CPU.

    Returns:
        The scorer, with its model in eval mode.
    """
    model, tokenizer = _load_classifier(NLI.repo_id, NLI.revision, device)
    return NliScorer(NLI, model, tokenizer)


def hard_labels(
    probabilities: ArrayLike, *, threshold: float = 0.5
) -> NDArray[np.int64]:
    """Turns judge probabilities into class labels.

    Args:
        probabilities: Positive-class probabilities of shape `(n,)`, or class
            probabilities of shape `(n, classes)`.
        threshold: Probability at or above which a positive-class probability
            counts as the positive class. Ignored for 2-D input.

    Returns:
        Labels of shape `(n,)`: a thresholded 0 or 1 for 1-D input, the
        first-maximum class index for 2-D input.

    Raises:
        ValueError: If the input is neither 1-D nor 2-D.
    """
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim == 1:
        return (values >= threshold).astype(np.int64)
    if values.ndim == 2:
        return values.argmax(axis=-1).astype(np.int64)
    raise ValueError(f"expected 1-D or 2-D probabilities, got {values.ndim}-D")


def sentiment_accuracy(
    p_positive: ArrayLike, labels: ArrayLike, *, threshold: float = 0.5
) -> float:
    """Computes the accuracy of positive-class probabilities.

    Args:
        p_positive: Probability of the positive class per example.
        labels: Gold labels, 1 for positive and 0 for negative.
        threshold: Probability at or above which a text is called positive.

    Returns:
        The share of examples whose thresholded prediction equals the label.

    Raises:
        ValueError: If the inputs differ in length, are empty, or the labels
            are not 0 or 1.
    """
    scores = np.asarray(p_positive, dtype=np.float64)
    gold = np.asarray(labels, dtype=np.int64)
    if scores.ndim != 1 or scores.shape != gold.shape:
        raise ValueError("p_positive and labels must have the same length")
    if scores.size == 0:
        raise ValueError("need at least one example")
    if not np.isin(gold, (0, 1)).all():
        raise ValueError("labels must be 0 or 1")
    return float((hard_labels(scores, threshold=threshold) == gold).mean())


def validate_sentiment_judge(
    judge: Judge,
    texts: Sequence[str],
    labels: ArrayLike,
    *,
    batch_size: int = 32,
) -> float:
    """Measures a sentiment judge against labeled texts.

    Args:
        judge: A sentiment judge.
        texts: Labeled texts, for example the SST-2 validation split.
        labels: Gold labels, 1 for positive and 0 for negative.
        batch_size: Texts per forward pass.

    Returns:
        The accuracy of the judge at the 0.5 threshold.

    Raises:
        ValueError: If `judge` is not a sentiment judge, or the inputs are
            invalid (see `sentiment_accuracy`).
    """
    if judge.spec.attribute != "sentiment":
        raise ValueError(
            "validate_sentiment_judge needs a sentiment judge, "
            f"got {judge.spec.key}"
        )
    return sentiment_accuracy(judge.score(texts, batch_size=batch_size), labels)


@dataclass(frozen=True)
class Agreement:
    """Agreement between two judges on the same texts.

    Attributes:
        rate: Share of texts given the same label by both judges.
        kappa: Cohen's kappa of the two label sequences. It is 1.0 when both
            judges use a single identical class, where chance agreement is 1.
    """

    rate: float
    kappa: float


def judge_agreement(
    first: ArrayLike, second: ArrayLike, *, threshold: float = 0.5
) -> Agreement:
    """Measures how often two judges give the same label.

    Args:
        first: Probabilities of the first judge, `(n,)` or `(n, classes)`.
        second: Probabilities of the second judge, same shape as `first`.
        threshold: Positive-class threshold for 1-D probabilities.

    Returns:
        The agreement rate and Cohen's kappa of the hard labels (see
        `hard_labels`).

    Raises:
        ValueError: If the inputs differ in shape or are empty.
    """
    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    if first_values.shape != second_values.shape:
        raise ValueError("first and second must have the same shape")
    if first_values.size == 0:
        raise ValueError("need at least one example")
    a = hard_labels(first_values, threshold=threshold)
    b = hard_labels(second_values, threshold=threshold)
    n_classes = max(int(a.max()), int(b.max()), 1) + 1
    counts = np.bincount(a * n_classes + b, minlength=n_classes**2)
    confusion = counts.reshape(n_classes, n_classes).astype(np.float64)
    observed = float(np.trace(confusion)) / a.size
    chance = float(confusion.sum(axis=1) @ confusion.sum(axis=0)) / a.size**2
    kappa = 1.0 if chance >= 1.0 else (observed - chance) / (1.0 - chance)
    return Agreement(rate=observed, kappa=kappa)
