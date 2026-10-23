"""The exact-answer-token probe (Orgad et al., 2025).

A logistic-regression probe reads whether the model's own answer is correct
from the residual stream at the last token of that answer, at a layer chosen
on validation items. The reference code locates a short "exact answer" span
inside a long generation; P9 answers are already short (cut at the first
newline), so the whole stored answer is the span, no extraction model runs
and no item is dropped. The reference reads the MLP output of a layer; this
module reads the residual stream, as every other P9 reader does. The
classifier is the P9c `probe` (`RepReader.fit` with the `logistic` finder),
fitted on the train items and oriented so that a higher score means a more
likely correct answer.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tomography.activations import Span
from tomography.activations import collect_activations
from tomography.model import BosMode
from tomography.model import LoadedModel
from tomography.reading.metrics import auroc
from tomography.reading.metrics import best_layer
from tomography.reading.reader import RepReader


def answer_texts(
    prompts: Sequence[str], answers: Sequence[str]
) -> tuple[list[str], list[Span]]:
    """Appends each answer to its prompt the way the model emitted it.

    The prompts end in a bare "A:", so the answer follows a space, as in
    `tomography.p9.signals.first_answer_token`.

    Args:
        prompts: The prompts.
        answers: One answer per prompt, without its leading space.

    Returns:
        The texts `prompt + " " + answer` and the character span of each
        answer inside its text.

    Raises:
        ValueError: If there is not one answer per prompt.
    """
    if len(prompts) != len(answers):
        raise ValueError("need one answer per prompt")
    texts = [f"{p} {a}" for p, a in zip(prompts, answers, strict=True)]
    spans = [(len(p) + 1, len(t)) for p, t in zip(prompts, texts, strict=True)]
    return texts, spans


def answer_activations(
    loaded: LoadedModel,
    prompts: Sequence[str],
    answers: Sequence[str],
    *,
    layers: Sequence[int],
    bos: BosMode | None = None,
    batch_size: int = 8,
    share_prefix: bool = False,
) -> NDArray[np.float32]:
    """Reads the residual stream at the last token of each answer.

    Args:
        loaded: The wrapped model.
        prompts: The prompts the answers were generated after.
        answers: One answer per prompt, without its leading space.
        layers: Layer indices in 0..L.
        bos: BOS policy; None uses the model's default.
        batch_size: Texts per forward pass.
        share_prefix: Run the token prefix common to all texts once, such as
            the demonstrations of a few-shot prompt, see
            `tomography.activations.collect_activations`.

    Returns:
        Array of shape (K, n, d_model), K = `len(layers)`; the rows of an
        empty answer are NaN.

    Raises:
        ValueError: If there is not one answer per prompt, or the arguments
            of `collect_activations` are inconsistent.
    """
    texts, spans = answer_texts(prompts, answers)
    out = np.full(
        (len(layers), len(texts), loaded.d_model), np.nan, dtype=np.float32
    )
    keep = [i for i, a in enumerate(answers) if a]
    if keep:
        acts = collect_activations(
            loaded,
            [texts[i] for i in keep],
            layers=layers,
            position="concept",
            concept_spans=[spans[i] for i in keep],
            bos=bos,
            batch_size=batch_size,
            share_prefix=share_prefix,
        )
        out[:, keep] = acts.data
    return out


@dataclass(frozen=True, eq=False)
class AnswerProbe:
    """An exact-answer-token probe and its scores.

    Attributes:
        layer: The layer chosen on the validation items.
        val_auroc: Validation AUROC of the probe at every candidate layer,
            in the order of the layers given to `fit_answer_probe`.
        scores: Score of every item at `layer`, higher for a more likely
            correct answer; NaN for an item without activations.
    """

    layer: int
    val_auroc: NDArray[np.float64]
    scores: NDArray[np.float64]


def fit_answer_probe(
    h: NDArray[np.float32],
    layers: Sequence[int],
    correct: NDArray[np.int64],
    *,
    train: NDArray[np.bool_],
    val: NDArray[np.bool_],
) -> AnswerProbe:
    """Fits the probe on train items and chooses its layer on validation.

    One probe per layer is fitted on the train items, the layer with the
    highest validation AUROC is kept (the first one on ties) and every item
    is scored there. Items whose activations hold NaN (empty answers) are
    left out of the fit and the validation AUROC and are scored NaN.

    Args:
        h: Activations of shape (K, n, d), such as `answer_activations`
            returns.
        layers: Layer index of each slice of `h`.
        correct: Whether each answer is correct (1) or not (0), shape (n,).
        train: Mask of the train items, shape (n,).
        val: Mask of the validation items, shape (n,), disjoint from
            `train`.

    Returns:
        The probe and its scores.

    Raises:
        ValueError: If the shapes disagree, the masks overlap, or the usable
            train or validation items do not hold both outcomes.
    """
    if h.ndim != 3:
        raise ValueError(f"h must have shape (K, n, d), got {h.shape}")
    if h.shape[0] != len(layers):
        raise ValueError(
            f"layers has {len(layers)} entries for {h.shape[0]} layers"
        )
    n = h.shape[1]
    if correct.shape != (n,):
        raise ValueError(f"need one label per item ({n}), got {correct.shape}")
    if train.shape != (n,) or val.shape != (n,):
        raise ValueError(f"train and val need one entry per item ({n})")
    if (train & val).any():
        raise ValueError("train and val overlap")
    usable = ~np.isnan(h).any(axis=(0, 2))
    fit_rows = train & usable
    val_rows = val & usable
    reader = RepReader.fit(
        "logistic", h[:, fit_rows], layers, labels=correct[fit_rows]
    )
    val_auroc = auroc(
        reader.score(h[:, val_rows]).astype(np.float64), correct[val_rows]
    )
    layer = best_layer(val_auroc, layers)
    k = list(layers).index(layer)
    probe = RepReader.fit(
        "logistic", h[k : k + 1, fit_rows], (layer,), labels=correct[fit_rows]
    )
    scores = np.full(n, np.nan)
    scores[usable] = probe.score(h[k : k + 1, usable])[0]
    return AnswerProbe(layer, val_auroc, scores)
