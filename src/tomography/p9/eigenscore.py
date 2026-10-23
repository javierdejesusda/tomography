"""EigenScore of sampled answers (INSIDE, Chen et al., 2024).

The sampled answers to one question are embedded by the residual stream at
the last token of each answer in the middle layer, and the score is the mean
log eigenvalue of the regularized covariance of those embeddings across
samples. A higher score means more divergent answers, so a more likely
hallucination; it is not oriented like `tomography.p9.signals`.

The module follows the released code (D2I-ai/eigenscore at commit ea8062a4,
`getEigenIndicator_v0`) where it differs from the paper: each embedding is
centred over its own features and the covariance is divided by d - 1, which
the paper's Eq. 4 does not do. The logarithm is natural, as in the paper; the
code takes log10, which only rescales the score. Feature clipping (paper
Section 3.2) is not in the code and is not applied. The embedding is read by
running the prompt, a space and the stored sample, so the token is the last
token of the sample text rather than the last one before a stop token.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from tomography.model import BosMode
from tomography.model import LoadedModel
from tomography.p9.answer_probe import answer_activations

ALPHA = 1e-3


def eigenscore(
    embeddings: NDArray[np.floating], *, alpha: float = ALPHA
) -> float:
    """Computes the EigenScore of the embeddings of K sampled answers.

    With each row centred over its d features, `C = Z Z^T / (d - 1)` is the
    K x K covariance of the samples and the score is the mean of
    `ln(lambda_i)` over the eigenvalues of `C + alpha I`. Identical samples
    are not at `ln(alpha)`: centring each row over its own features leaves
    one eigenvalue `K * var(z)`, so the score is
    `((K - 1) ln(alpha) + ln(K * var(z) + alpha)) / K`.

    Args:
        embeddings: Array of shape (K, d), one row per sample.
        alpha: The ridge added to the covariance; 1e-3 as in the reference.

    Returns:
        The score in nats.

    Raises:
        ValueError: If the array is not 2D, has fewer than 2 samples or 2
            features, holds a non-finite value, or `alpha` is not positive.
    """
    z = np.asarray(embeddings, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"embeddings must have shape (K, d), got {z.shape}")
    if z.shape[0] < 2:
        raise ValueError(f"need at least 2 samples, got {z.shape[0]}")
    if z.shape[1] < 2:
        raise ValueError(f"need at least 2 features, got {z.shape[1]}")
    if not np.isfinite(z).all():
        raise ValueError("embeddings must be finite")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    centred = z - z.mean(axis=1, keepdims=True)
    covariance = centred @ centred.T / (z.shape[1] - 1)
    eigenvalues = np.linalg.eigvalsh(covariance + alpha * np.eye(z.shape[0]))
    return float(np.mean(np.log(eigenvalues)))


def eigen_layer(loaded: LoadedModel) -> int:
    """Returns the middle layer, `int(L / 2)` as in the reference.

    Layer l is the residual stream after l blocks, which is
    `hidden_states[l]` of the Hugging Face models the reference reads.

    Args:
        loaded: The wrapped model.

    Returns:
        The layer index.
    """
    return loaded.n_layers // 2


def eigenscores(
    loaded: LoadedModel,
    prompts: Sequence[str],
    samples: Sequence[Sequence[str]],
    *,
    layer: int | None = None,
    alpha: float = ALPHA,
    bos: BosMode | None = None,
    batch_size: int = 8,
    share_prefix: bool = False,
) -> NDArray[np.float64]:
    """Scores the sampled answers of each prompt.

    All samples of all prompts are read in one pass, so `share_prefix` saves
    the demonstrations of a few-shot prompt once for the whole set. Empty
    samples have no token to read and are dropped.

    Args:
        loaded: The wrapped model.
        prompts: The prompts the samples were drawn after.
        samples: The sampled answers of each prompt, without their leading
            space.
        layer: The layer to read; None uses `eigen_layer`.
        alpha: The ridge of `eigenscore`.
        bos: BOS policy; None uses the model's default.
        batch_size: Texts per forward pass.
        share_prefix: Run the token prefix common to all texts once, see
            `tomography.activations.collect_activations`.

    Returns:
        The score of each prompt; NaN when fewer than 2 samples are not
        empty.

    Raises:
        ValueError: If there is not one sample set per prompt.
    """
    if len(samples) != len(prompts):
        raise ValueError("need one sample set per prompt")
    read = eigen_layer(loaded) if layer is None else layer
    owner = [i for i, group in enumerate(samples) for _ in group]
    h = answer_activations(
        loaded,
        [prompts[i] for i in owner],
        [s for group in samples for s in group],
        layers=[read],
        bos=bos,
        batch_size=batch_size,
        share_prefix=share_prefix,
    )[0]
    owners = np.array(owner, dtype=np.int64)
    usable = ~np.isnan(h).any(axis=1)
    scores = np.full(len(prompts), np.nan)
    for i in range(len(prompts)):
        rows = h[(owners == i) & usable]
        if len(rows) >= 2:
            scores[i] = eigenscore(rows, alpha=alpha)
    return scores
