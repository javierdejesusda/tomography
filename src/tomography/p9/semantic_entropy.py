"""Discrete semantic entropy of sampled answers (Farquhar et al., 2024).

Sampled answers to one question are grouped into clusters of equal meaning,
and the uncertainty is the entropy of the share of samples per cluster. The
module follows the released code of the paper (jlko/semantic_uncertainty at
commit a8d9aa8c) where the paper and the code differ; each such choice is
named in the docstring of the function that makes it.

Only the discrete variant is implemented: it needs the cluster of every sample
and nothing else. The likelihood-weighted variant needs the token
log-probabilities of every sample, which P9a does not store, so it is out of
scope.

The module does not load or depend on any model. `semantic_entropy` takes the
entailment model as a callable, and `tomography.judges.NliScorer.entails` is
such a callable. To judge the pairs of many questions in one batch, list them
with `entailment_pairs`, run the model once over all of them, and give each
question's verdicts to `semantic_clusters`. `NliScorer` pins DeBERTa-large
MNLI, the model of Kuhn et al. (2023) and the one the text of Farquhar et al.
names; the reference code of Farquhar et al. runs DeBERTa-v2-xlarge MNLI
instead (see `judges.NLI`). The decision rule, the most likely of the three
classes, is the same.
"""

from collections import Counter
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import itertools
import math

import numpy as np
from numpy.typing import ArrayLike

Entailer = Callable[[Sequence[str], Sequence[str]], ArrayLike]
Verdicts = Mapping[tuple[str, str], bool]


def cluster_entropy(cluster_ids: Sequence[int]) -> float:
    """Computes the entropy of the share of samples per cluster.

    This is the discrete semantic entropy of the reference code
    (`cluster_assignment_entropy`): `p_c = count_c / M` for M samples and
    `H = -sum_c p_c ln p_c`. The logarithm is natural, so the unit is nats and
    the maximum is `ln M`, reached when every sample is its own cluster.

    Args:
        cluster_ids: The cluster of every sample. Only which samples share a
            value matters, not the values.

    Returns:
        The entropy in nats: 0.0 when all samples share one cluster.

    Raises:
        ValueError: If there are no samples.
    """
    if not cluster_ids:
        raise ValueError("need at least one sample")
    total = len(cluster_ids)
    entropy = 0.0
    for count in Counter(cluster_ids).values():
        share = count / total
        entropy -= share * math.log(share)
    return entropy


def entailment_pairs(
    question: str, answers: Sequence[str]
) -> list[tuple[str, str]]:
    """Lists the text pairs the entailment model must judge for one question.

    The model reads `"{question} {answer}"` for each answer, as the reference
    code does with its DeBERTa entailer, and each answer is stripped of outer
    whitespace, as the reference does when it generates them. The pairs are
    every ordered pair of distinct non-empty answers, so both directions of
    each pair appear once. Identical answers and empty answers never become a
    pair (see `semantic_clusters`).

    Args:
        question: The question the answers respond to.
        answers: The sampled answers, in sampling order.

    Returns:
        (premise, hypothesis) texts. Collect the pairs of many questions into
        one batch for the model, then pass each question its verdicts.
    """
    distinct = dict.fromkeys(
        stripped for answer in answers if (stripped := answer.strip())
    )
    texts = [f"{question} {answer}" for answer in distinct]
    return list(itertools.permutations(texts, 2))


def _equivalent(
    question: str, first: str, second: str, verdicts: Verdicts
) -> bool:
    """Tells whether two stripped answers have the same meaning."""
    if first == second:
        return True
    if not first or not second:
        return False
    premise = f"{question} {first}"
    hypothesis = f"{question} {second}"
    return verdicts[(premise, hypothesis)] and verdicts[(hypothesis, premise)]


def semantic_clusters(
    question: str, answers: Sequence[str], verdicts: Verdicts
) -> list[int]:
    """Groups sampled answers by bidirectional entailment.

    The rule and the order follow `get_semantic_ids` of the reference code
    with its default `strict_entailment=True`. Two answers are equivalent iff
    each entails the other. Samples are walked in order: a sample without a
    cluster opens the next one and is compared with every later sample, and
    each equivalent one joins it. As in the reference, a later sample is
    compared even when an earlier cluster has claimed it, so a cluster opened
    later takes it over if it is equivalent to the new head. The result
    depends on the order of `answers` whenever the entailment relation is not
    transitive.

    Two cases never need a verdict, which departs from the reference (it
    sends every pair through the model): identical answers are equivalent,
    and an empty answer (nothing but whitespace) is equivalent only to
    another empty answer. An empty sample says nothing, so it is not given
    the meaning of any answer.

    Args:
        question: The question the answers respond to.
        answers: The sampled answers, in sampling order.
        verdicts: Whether the model found that the first text entails the
            second, for every pair of `entailment_pairs(question, answers)`.

    Returns:
        The cluster id of every answer. Ids count up from 0 in order of the
        first answer of each cluster.

    Raises:
        ValueError: If there are no answers.
        KeyError: If `verdicts` lacks a pair that the walk needs.
    """
    if not answers:
        raise ValueError("need at least one sample")
    stripped = [answer.strip() for answer in answers]
    cluster_ids = [-1] * len(stripped)
    next_id = 0
    for head, head_answer in enumerate(stripped):
        if cluster_ids[head] != -1:
            continue
        cluster_ids[head] = next_id
        for other in range(head + 1, len(stripped)):
            if _equivalent(question, head_answer, stripped[other], verdicts):
                cluster_ids[other] = next_id
        next_id += 1
    return cluster_ids


def semantic_entropy(
    question: str, answers: Sequence[str], entails: Entailer
) -> float:
    """Computes the discrete semantic entropy of the answers to a question.

    The model is called once, for all the pairs of the question. To batch
    several questions, call `entailment_pairs` for each, judge all the pairs
    together, and give each question's `semantic_clusters` the verdicts.

    Args:
        question: The question the answers respond to.
        answers: The sampled answers, in sampling order.
        entails: Called as `entails(premises, hypotheses)` with two equally
            long lists of texts; returns one boolean per pair: whether the
            premise entails the hypothesis. It is not called when no pair
            needs a verdict.

    Returns:
        The entropy in nats of the share of answers per cluster of equal
        meaning (see `cluster_entropy` and `semantic_clusters`).

    Raises:
        ValueError: If there are no answers, or `entails` does not return
            one verdict per pair.
    """
    pairs = entailment_pairs(question, answers)
    verdicts: dict[tuple[str, str], bool] = {}
    if pairs:
        flags = np.asarray(
            entails(
                [premise for premise, _ in pairs], [hyp for _, hyp in pairs]
            ),
            dtype=bool,
        )
        if flags.shape != (len(pairs),):
            raise ValueError(
                "entails must return one verdict per pair: "
                f"expected {len(pairs)}, got shape {flags.shape}"
            )
        verdicts = dict(zip(pairs, flags.tolist(), strict=True))
    return cluster_entropy(semantic_clusters(question, answers, verdicts))
