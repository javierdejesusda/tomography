"""Per-question signals that predict whether the model answers correctly.

Every signal is oriented so that a higher value means the answer is more
likely to be correct. The first five read the model's own output, `p_true`
asks the model to grade its answer, and the last three are properties of the
question that need no model output: how popular its subject is, how long it
is, and how likely the model finds the subject's name.
"""

from collections.abc import Sequence
import contextlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import torch

from tomography.control.generate import EditFactory
from tomography.hooks import edit_residuals
from tomography.judges import sequence_logprob
from tomography.knowledge import DEFAULT_SHOTS
from tomography.knowledge import ItemLabel
from tomography.knowledge import few_shot_prompt
from tomography.knowledge import normalize_answer
from tomography.model import BosMode
from tomography.model import LoadedModel
from tomography.p9.pool import LabeledItem

NAME_PROMPT = "Name:"
SIGNAL_NAMES: tuple[str, ...] = (
    "first_token_prob",
    "neg_entropy",
    "answer_logprob",
    "mean_token_logprob",
    "consistency",
    "p_true",
    "popularity",
    "length",
    "name_logprob",
)


@dataclass(frozen=True)
class TokenStats:
    """Statistics of the next-token distribution after each prompt.

    Attributes:
        max_prob: Probability of the most likely token, shape (n,).
        entropy: Entropy of the distribution in nats, shape (n,).
        target_logprob: Log-probability of the requested token, shape (n,),
            or None when no targets were given.
    """

    max_prob: NDArray[np.float64]
    entropy: NDArray[np.float64]
    target_logprob: NDArray[np.float64] | None


def scoring_bos(loaded: LoadedModel) -> int | None:
    """Returns the BOS id that scoring prepends, if any.

    Args:
        loaded: The wrapped model.

    Returns:
        `loaded.bos_id` iff the model default is to prepend a BOS.
    """
    return loaded.bos_id if loaded.default_bos == "prepend" else None


def next_token_stats(
    loaded: LoadedModel,
    prompts: Sequence[str],
    *,
    targets: Sequence[int] | None = None,
    edits: EditFactory | None = None,
    bos: BosMode | None = None,
    batch_size: int = 8,
) -> TokenStats:
    """Reads the next-token distribution after each prompt.

    Args:
        loaded: The wrapped model.
        prompts: The prompts, left-padded in batches.
        targets: One token id per prompt whose log-probability is also
            returned, or None.
        edits: Builds the residual edits of a batch, or None for none.
        bos: BOS policy of the encoding; None uses the model default.
        batch_size: Prompts per forward pass.

    Returns:
        The statistics, one entry per prompt.

    Raises:
        ValueError: If `targets` does not hold one id per prompt.
    """
    if targets is not None and len(targets) != len(prompts):
        raise ValueError("targets must have one id per prompt")
    max_prob: list[float] = []
    entropy: list[float] = []
    target_logprob: list[float] = []
    for start in range(0, len(prompts), batch_size):
        batch = loaded.encode(
            prompts[start : start + batch_size], bos=bos, padding_side="left"
        )
        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.inference_mode())
            if edits is not None:
                stack.enter_context(edit_residuals(loaded, edits(batch)))
            out = loaded.model(
                input_ids=batch.input_ids.to(loaded.device),
                attention_mask=batch.attention_mask.to(loaded.device),
                position_ids=batch.position_ids.to(loaded.device),
                use_cache=False,
            )
        logp = torch.log_softmax(out.logits[:, -1].float(), dim=-1)
        p = logp.exp()
        max_prob.extend(p.max(dim=-1).values.cpu().tolist())
        entropy.extend((-(p * logp).sum(dim=-1)).cpu().tolist())
        if targets is not None:
            ids = torch.tensor(
                list(targets[start : start + batch_size]), device=logp.device
            )
            target_logprob.extend(logp.gather(1, ids[:, None])[:, 0].tolist())
    return TokenStats(
        np.array(max_prob),
        np.array(entropy),
        np.array(target_logprob) if targets is not None else None,
    )


def first_answer_token(loaded: LoadedModel, answer: str) -> int:
    """Returns the first token of the answer as it follows a prompt.

    Args:
        loaded: The wrapped model.
        answer: The answer text, without its leading space.

    Returns:
        The id of the first token of `" " + answer`.

    Raises:
        ValueError: If the answer has no tokens.
    """
    ids = loaded.tokenizer(" " + answer, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"answer {answer!r} has no tokens")
    return int(ids[0])


def answer_logprobs(
    loaded: LoadedModel,
    prompts: Sequence[str],
    answers: Sequence[str],
    *,
    batch_size: int = 8,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Scores the answers after the prompts.

    Args:
        loaded: The wrapped model.
        prompts: One prompt per answer.
        answers: The answers, scored as `" " + answer`.
        batch_size: Sequences per forward pass.

    Returns:
        The summed log-probability and the token count of each answer; NaN
        and 0 for an empty answer.
    """
    keep = [i for i, a in enumerate(answers) if a]
    total = np.full(len(prompts), np.nan)
    counts = np.zeros(len(prompts), dtype=np.int64)
    if keep:
        continuations = [" " + answers[i] for i in keep]
        total[keep] = sequence_logprob(
            loaded.model,
            loaded.tokenizer,
            [prompts[i] for i in keep],
            continuations,
            batch_size=batch_size,
            bos_id=scoring_bos(loaded),
        )
        counts[keep] = [
            len(loaded.tokenizer(c, add_special_tokens=False)["input_ids"])
            for c in continuations
        ]
    return total, counts


def p_true(
    loaded: LoadedModel,
    questions: Sequence[str],
    answers: Sequence[str],
    *,
    batch_size: int = 8,
) -> NDArray[np.float64]:
    """The model's own probability that a proposed answer is correct.

    Args:
        loaded: The wrapped model.
        questions: The questions.
        answers: One proposed answer per question.
        batch_size: Sequences per forward pass.

    Returns:
        The sigmoid of the log-probability of " True" minus that of " False"
        after a grading prompt, strictly inside (0, 1).
    """
    prompts = [
        f"Question: {q}\nProposed answer: {a}\n"
        "Is the proposed answer correct? Answer True or False.\nAnswer:"
        for q, a in zip(questions, answers, strict=True)
    ]
    bos_id = scoring_bos(loaded)
    yes = sequence_logprob(
        loaded.model,
        loaded.tokenizer,
        prompts,
        [" True"] * len(prompts),
        batch_size=batch_size,
        bos_id=bos_id,
    )
    no = sequence_logprob(
        loaded.model,
        loaded.tokenizer,
        prompts,
        [" False"] * len(prompts),
        batch_size=batch_size,
        bos_id=bos_id,
    )
    return 1.0 / (1.0 + np.exp(-(yes.astype(np.float64) - no)))


def name_logprobs(
    loaded: LoadedModel, names: Sequence[str], *, batch_size: int = 8
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Scores names after a bare name prompt.

    Args:
        loaded: The wrapped model.
        names: The names, scored as `" " + name`.
        batch_size: Sequences per forward pass.

    Returns:
        The summed log-probability and the token count of each name.
    """
    return answer_logprobs(
        loaded, [NAME_PROMPT] * len(names), list(names), batch_size=batch_size
    )


def consistency(label: ItemLabel) -> float:
    """The share of sampled answers that equal the greedy answer.

    Args:
        label: The label of one question.

    Returns:
        The share after answer normalization, NaN without samples.
    """
    if not label.samples:
        return float("nan")
    target = normalize_answer(label.greedy)
    return float(
        np.mean([normalize_answer(s) == target for s in label.samples])
    )


def item_signals(
    loaded: LoadedModel,
    items: Sequence[LabeledItem],
    *,
    shots: Sequence[tuple[str, str]] = DEFAULT_SHOTS,
    batch_size: int = 8,
) -> dict[str, NDArray[np.float64]]:
    """Computes the signals of every question.

    The prompt is the few-shot prompt the labels were made with, scored with
    the default BOS policy, and the answer is the greedy answer of the label.

    Args:
        loaded: The wrapped model.
        items: The labeled questions.
        shots: Demonstrations of the prompt.
        batch_size: Sequences per forward pass.

    Returns:
        One array of shape (n,) per name in `SIGNAL_NAMES`. A signal is NaN
        where it is undefined: the answer-based ones for an empty greedy
        answer, `consistency` without samples, `popularity` without a
        positive count and `name_logprob` without a subject.
    """
    n = len(items)
    prompts = [few_shot_prompt(i.example.question, shots) for i in items]
    greedy = [i.label.greedy for i in items]
    stats = next_token_stats(loaded, prompts, batch_size=batch_size)
    total, counts = answer_logprobs(
        loaded, prompts, greedy, batch_size=batch_size
    )
    mean_token = np.where(counts > 0, total / np.maximum(counts, 1), np.nan)
    graded = np.full(n, np.nan)
    answered = [k for k, g in enumerate(greedy) if g]
    if answered:
        graded[answered] = p_true(
            loaded,
            [items[k].example.question for k in answered],
            [greedy[k] for k in answered],
            batch_size=batch_size,
        )
    popularity = np.array(
        [
            np.log10(p)
            if isinstance(p := i.example.meta.get("popularity"), int) and p > 0
            else np.nan
            for i in items
        ]
    )
    length = np.array(
        [
            len(
                loaded.tokenizer(i.example.question, add_special_tokens=False)[
                    "input_ids"
                ]
            )
            for i in items
        ],
        dtype=np.float64,
    )
    name = np.full(n, np.nan)
    named = [
        k
        for k, i in enumerate(items)
        if isinstance(i.example.meta.get("subject"), str)
    ]
    if named:
        name_total, name_counts = name_logprobs(
            loaded,
            [str(items[k].example.meta["subject"]) for k in named],
            batch_size=batch_size,
        )
        name[named] = name_total / np.maximum(name_counts, 1)
    return {
        "first_token_prob": stats.max_prob,
        "neg_entropy": -stats.entropy,
        "answer_logprob": total,
        "mean_token_logprob": mean_token,
        "consistency": np.array([consistency(i.label) for i in items]),
        "p_true": graded,
        "popularity": popularity,
        "length": length,
        "name_logprob": name,
    }
