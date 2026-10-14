"""Judges and language-model scoring helpers.

Tokenization for the language-model helpers follows the policy of
`LoadedModel.encode`: no special tokens are added by the tokenizer, sequences
are right padded, position ids are explicit, and a BOS id is prepended only
when the caller passes one. The helpers take plain Hugging Face objects and run
under whatever hooks are active when they are called.
"""

from collections.abc import Iterable
from collections.abc import Sequence
import math

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray
import torch
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerBase


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
        One float32 array per sequence, of length `len(sequence) - 1`: entry j
        is the log-probability of token j + 1 given tokens 0..j.
    """
    results: list[NDArray[np.float32]] = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start : start + batch_size]
            width = max(len(sequence) for sequence in chunk)
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
            log_probs = logits[:, :-1].float().log_softmax(dim=-1)
            targets = input_ids[:, 1:].to(model.device)
            picked = log_probs.gather(-1, targets[..., None]).squeeze(-1)
            picked_np = picked.cpu().numpy()
            for row, sequence in enumerate(chunk):
                results.append(picked_np[row, : len(sequence) - 1])
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
    prefix = [] if bos_id is None else [bos_id]
    sequences: list[list[int]] = []
    starts: list[int] = []
    pairs = zip(
        _tokenize(tokenizer, prompts),
        _tokenize(tokenizer, continuations),
        strict=True,
    )
    for index, (prompt_ids, continuation_ids) in enumerate(pairs):
        if not continuation_ids:
            raise ValueError(f"continuation {index} has no tokens")
        if not prefix and not prompt_ids:
            raise ValueError(
                f"prompt {index} has no tokens and no bos_id was given"
            )
        sequences.append(prefix + prompt_ids + continuation_ids)
        starts.append(len(prefix) + len(prompt_ids) - 1)
    scored = _next_token_logprobs(model, sequences, batch_size)
    return np.array(
        [
            values[start:].sum(dtype=np.float32)
            for values, start in zip(scored, starts, strict=True)
        ],
        dtype=np.float32,
    )


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
