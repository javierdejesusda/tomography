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
from numpy.typing import NDArray

from tomography.experiments.p3_size import spearman as rank_correlation
from tomography.experiments.steering import dose_slope
from tomography.experiments.variants import READING_TASKS
from tomography.experiments.variants import VARIANT_KINDS
from tomography.experiments.variants import Variant
from tomography.experiments.variants import VariantKind
from tomography.experiments.variants import template_variants
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


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
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
