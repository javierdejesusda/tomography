"""P7: do directions read under different templates steer as well as they read?

Sentiment readers are extracted under the template variants of P1 and steer
text with C1's protocol, so the template changes nothing but the direction.
The study counts the readers that read sentiment well and steer it the wrong
way, compares the layer that reads best with the layer that steers best, and
asks whether the logit lens predicts the dose response.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from tomography.analysis import lens_logits
from tomography.analysis import pole_separation
from tomography.experiments.a2_logit_lens import PoleWords
from tomography.experiments.a2_logit_lens import lens_params
from tomography.experiments.a2_logit_lens import pole_ids
from tomography.experiments.p3_size import spearman as rank_correlation
from tomography.experiments.steering import dose_slope
from tomography.experiments.variants import READING_TASKS
from tomography.experiments.variants import VARIANT_KINDS
from tomography.experiments.variants import Variant
from tomography.experiments.variants import VariantKind
from tomography.experiments.variants import template_variants
from tomography.model import LoadedModel
from tomography.reading.metrics import wilson_ci
from tomography.reading.reader import RepReader
from tomography.templates import ChatFormat

TEMPLATES: tuple[str, ...] = (
    "paper",
    "no_template",
    "no_chat_tags",
    "neutral_frame",
    "scaffold_0",
    "concept_swap_0",
)
FORMATS: tuple[str, ...] = ("raw", "chat", "story")
CHAT_INSTRUCTION = "Continue the text."
STORY_PREFIX = "Here is a short story: "


def _kind_of(name: str) -> VariantKind:
    """Returns the kind of a variant name: the name itself or `<kind>_<i>`."""
    for kind in VARIANT_KINDS:
        if kind in (name, name.rpartition("_")[0]):
            return kind
    raise ValueError(
        f"{name!r} is not a variant kind or <kind>_<i> of one, expected a "
        f"kind of {VARIANT_KINDS}"
    )


def extraction_variants(
    chat: ChatFormat, names: Sequence[str]
) -> tuple[Variant, ...]:
    """Picks named template variants of the sentiment task.

    Args:
        chat: The chat format of the model.
        names: Variant names: a kind, or `<kind>_<i>` for an indexed kind,
            which `template_variants` expands to every index. `paper` is
            required.

    Returns:
        The named variants, in the order of `names`.

    Raises:
        ValueError: If a name is not a kind or `<kind>_<i>`, a name is not
            built for the sentiment task (the message carries the skip
            reason of its kind), or `names` lacks `paper`.
    """
    kinds = {name: _kind_of(name) for name in names}
    variants, skipped = template_variants(
        READING_TASKS["sentiment"],
        chat,
        kinds=tuple(k for k in VARIANT_KINDS if k in kinds.values()),
    )
    by_name = {v.name: v for v in variants}
    for name in names:
        if name not in by_name:
            reason = skipped.get(
                kinds[name], "variants.py has no variant of that name"
            )
            raise ValueError(
                f"{name!r} is not a variant of the sentiment task: {reason}"
            )
    return tuple(by_name[name] for name in names)


def shift_prompts(
    fragments: Sequence[str], fmt: str, chat: ChatFormat
) -> list[str]:
    """Renders the steering prompts in one of the transfer formats.

    Args:
        fragments: The prompt fragments, the first words of a text.
        fmt: A name from `FORMATS`: `raw` keeps the fragments, `chat` starts
            the answer to `CHAT_INSTRUCTION` with the fragment and `story`
            puts `STORY_PREFIX` before it.
        chat: The chat format of the `chat` prompts.

    Returns:
        One prompt per fragment.

    Raises:
        ValueError: If `fmt` is not in `FORMATS`.
    """
    if fmt == "raw":
        return list(fragments)
    if fmt == "chat":
        return [chat.render_turn(CHAT_INSTRUCTION, f)[0] for f in fragments]
    if fmt == "story":
        return [STORY_PREFIX + f for f in fragments]
    raise ValueError(f"unknown format {fmt!r}, expected one of {FORMATS}")


def spearman(x: ArrayLike, y: ArrayLike) -> float:
    """Computes the Spearman correlation of the finite pairs.

    Args:
        x: Values of one variable.
        y: Values of the other, one per entry of `x`.

    Returns:
        The Pearson correlation of the ranks, with the ranks of ties
        averaged. NaN for fewer than 3 finite pairs or a constant side.

    Raises:
        ValueError: If `x` and `y` differ in length.
    """
    first = np.asarray(x, dtype=np.float64)
    second = np.asarray(y, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError(
            "x and y must have the same length, got "
            f"{first.shape} and {second.shape}"
        )
    keep = np.isfinite(first) & np.isfinite(second)
    if keep.sum() < 3:
        return float("nan")
    return rank_correlation(first[keep], second[keep])


def safe_slope(coeffs: Sequence[float], values: NDArray[Any]) -> float:
    """Fits the dose slope, or NaN when too few finite points remain.

    Args:
        coeffs: The coefficients.
        values: One value per coefficient.

    Returns:
        `dose_slope` of the finite points; NaN when fewer than two distinct
        finite coefficients remain.

    Raises:
        ValueError: If `coeffs` and `values` differ in length.
    """
    x = np.asarray(coeffs, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if (
        x.shape == y.shape
        and np.unique(x[np.isfinite(x) & np.isfinite(y)]).size < 2
    ):
        return float("nan")
    return dose_slope(coeffs, y)


def lens_separation(
    loaded: LoadedModel, readers: Sequence[RepReader], words: PoleWords
) -> NDArray[np.float64]:
    """Measures how far the logit lens puts the poles apart for each reader.

    Args:
        loaded: The wrapped model; only its unembedding, final norm and
            tokenizer are used.
        readers: The readers, each with one direction per layer.
        words: The words of the positive and the negative pole, with their
            leading space. Those that are not one token are dropped.

    Returns:
        Array of shape (R, K): the pole separation (see `pole_separation`)
        of every reader's oriented direction (`directions * signs`) at every
        layer. An empty `readers` gives an array of shape (0, 0).

    Raises:
        ValueError: If no word of a pole is a single token.
    """
    positive, _ = pole_ids(loaded, words[0])
    negative, _ = pole_ids(loaded, words[1])
    if not positive or not negative:
        raise ValueError("no pole word is a single token for this tokenizer")
    unembedding, scale, center = lens_params(loaded)
    rows = [
        pole_separation(
            lens_logits(
                reader.directions * reader.signs[:, None],
                unembedding,
                scale=scale,
                center=center,
            ),
            positive,
            negative,
        )
        for reader in readers
    ]
    return np.stack(rows) if rows else np.empty((0, 0), dtype=np.float64)


def _finite_mean(values: NDArray[Any], axis: int) -> NDArray[np.float64]:
    """Averages the finite entries along `axis`; NaN where none is left."""
    finite = np.isfinite(values)
    count = finite.sum(axis=axis)
    total = np.where(finite, values, 0.0).sum(axis=axis)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _inversion(
    read: NDArray[np.float64], slope: NDArray[np.float64], threshold: float
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Marks the units that read well and the ones that steer the wrong way.

    Args:
        read: Reading accuracy of each unit.
        slope: Primary-judge dose slope of each unit, of the same shape.
        threshold: A unit is readable at or above this accuracy.

    Returns:
        The readable units, and the readable units with a negative slope. A
        unit without a finite slope is never inverted.
    """
    readable = read >= threshold
    finite = np.isfinite(slope)
    negative = np.where(finite, slope, 0.0) < 0.0
    return readable, readable & negative


def _inversion_summary(
    readable: NDArray[np.bool_], inverted: NDArray[np.bool_]
) -> dict[str, Any]:
    """Counts the units and gives the inversion rate with its interval."""
    n_readable = int(readable.sum())
    n_inverted = int(inverted.sum())
    summary: dict[str, Any] = {
        "readable": n_readable,
        "inverted": n_inverted,
        "rate": None,
        "ci": None,
    }
    if n_readable:
        summary["rate"] = n_inverted / n_readable
        summary["ci"] = list(wilson_ci(n_inverted, n_readable))
    return summary


def _sign_agreement(
    first: NDArray[np.float64], second: NDArray[np.float64]
) -> float:
    """Gives the share of units whose two slopes have the same sign.

    Returns:
        The share among the units with a finite slope under both judges; NaN
        when there is none.
    """
    both = np.isfinite(first) & np.isfinite(second)
    if not both.any():
        return float("nan")
    return float(np.mean(np.sign(first[both]) == np.sign(second[both])))


def _lens_summary(
    lens: NDArray[np.float64],
    slope: NDArray[np.float64],
    sweep: Sequence[int],
) -> tuple[int, float]:
    """Compares the lens separation with the dose slope over the sweep layers.

    Args:
        lens: Separation of shape (S, R): sweep layer, steering reader.
        slope: Primary-judge dose slope of the same shape.
        sweep: The S sweep layers.

    Returns:
        The sweep layer of the largest mean separation (the first on ties),
        and the Spearman correlation over the layers of the reader means of
        the separation and of the slope.
    """
    mean_lens = lens.mean(axis=1)
    layer = int(sweep[int(np.argmax(mean_lens))])
    return layer, spearman(mean_lens, _finite_mean(slope, axis=1))
