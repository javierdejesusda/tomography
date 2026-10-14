"""Seeded pairing strategies that turn stimuli into reading pairs.

A pair is a row ``(i, j)`` of an int64 ``(m, 2)`` array of stimulus indices;
the reading difference is ``H[i] - H[j]``. Concept pairs randomize the order
inside each pair so that centering the differences keeps the signal; persona
pairs alternate it deterministically.
"""

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int64]


def random_pairs(n: int, *, seed: int) -> IntArray:
    """Matches ``n`` stimuli at random into ``n // 2`` disjoint pairs.

    Args:
        n: Number of stimuli, which must be even.
        seed: Seed of the generator.

    Returns:
        Int64 array of shape ``(n // 2, 2)`` that uses every index once, in
        random pair order and random order inside each pair.

    Raises:
        ValueError: If ``n`` is negative or odd.
    """
    if n < 0 or n % 2:
        raise ValueError(f"n must be a non-negative even number, got {n}")
    rng = np.random.default_rng(seed)
    return rng.permutation(n).astype(np.int64).reshape(-1, 2)


def label_contrastive_pairs(
    labels: IntArray,
    *,
    seed: int,
    n_pairs: int | None = None,
    strata: IntArray | None = None,
) -> IntArray:
    """Pairs each label-1 stimulus with a label-0 stimulus.

    Args:
        labels: Binary labels of shape ``(n,)``.
        seed: Seed of the generator.
        n_pairs: Number of pairs to return. ``None`` keeps every match the
            label counts allow, which is ``min(#positive, #negative)`` per
            stratum.
        strata: Optional int64 codes of shape ``(n,)``. A pair never crosses
            two strata, so matched members share surface form. ``n_pairs``
            caps the total over all strata.

    Returns:
        Int64 array of shape ``(m, 2)``, one stimulus per label in a random
        order inside each pair, in random pair order, without repeats.

    Raises:
        ValueError: If the labels are not binary, ``strata`` does not match
            ``labels``, no pair can be formed, or ``n_pairs`` is not between
            1 and the number of available pairs.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be a 1-D binary array of 0 and 1")
    if strata is None:
        codes = np.zeros(labels.shape, dtype=np.int64)
    else:
        codes = np.asarray(strata)
        if codes.shape != labels.shape:
            raise ValueError(
                f"strata shape {codes.shape} must match labels {labels.shape}"
            )
    rng = np.random.default_rng(seed)
    matches = [np.empty((0, 2), dtype=np.int64)]
    for code in np.unique(codes):
        members = np.flatnonzero(codes == code)
        positive = rng.permutation(members[labels[members] == 1])
        negative = rng.permutation(members[labels[members] == 0])
        size = min(len(positive), len(negative))
        matches.append(np.stack([positive[:size], negative[:size]], axis=1))
    return _draw(np.concatenate(matches), n_pairs, rng)


def one_vs_rest_pairs(
    classes: IntArray,
    target: int,
    *,
    seed: int,
    n_pairs: int | None = None,
) -> IntArray:
    """Pairs members of class ``target`` with random members of other classes.

    Args:
        classes: Integer class ids of shape ``(n,)``.
        target: The class that forms one side of every pair.
        seed: Seed of the generator.
        n_pairs: Number of pairs to return. ``None`` keeps as many as
            ``min(#target, #others)``.

    Returns:
        Int64 array of shape ``(m, 2)``: one target member and one member of
        another class, in a random order inside each pair, without repeats.

    Raises:
        ValueError: If ``target`` or the other classes have no member, or
            ``n_pairs`` is not between 1 and the number of available pairs.
    """
    classes = np.asarray(classes)
    members = np.flatnonzero(classes == target)
    others = np.flatnonzero(classes != target)
    if len(members) == 0:
        raise ValueError(f"target class {target} has no member")
    if len(others) == 0:
        raise ValueError(f"no member outside target class {target}")
    rng = np.random.default_rng(seed)
    size = min(len(members), len(others))
    matches = np.stack(
        [rng.permutation(members)[:size], rng.permutation(others)[:size]],
        axis=1,
    )
    return _draw(matches.astype(np.int64), n_pairs, rng)


def persona_pairs(n_stimuli: int) -> IntArray:
    """Pairs the two personas of each statement and alternates the order.

    The stimuli are laid out as ``[pos_0, neg_0, pos_1, neg_1, ...]``. Pair
    ``i`` is ``(2i, 2i + 1)`` for even ``i`` and ``(2i + 1, 2i)`` for odd
    ``i``, so the sign of the difference alternates as in the paper.

    Args:
        n_stimuli: Number of stimuli, which must be even.

    Returns:
        Int64 array of shape ``(n_stimuli // 2, 2)``.

    Raises:
        ValueError: If ``n_stimuli`` is negative or odd.
    """
    if n_stimuli < 0 or n_stimuli % 2:
        raise ValueError(
            f"n_stimuli must be a non-negative even number, got {n_stimuli}"
        )
    index = np.arange(n_stimuli // 2, dtype=np.int64)
    odd = index % 2
    return np.stack([2 * index + odd, 2 * index + 1 - odd], axis=1)


def _draw(
    matches: IntArray, n_pairs: int | None, rng: np.random.Generator
) -> IntArray:
    """Shuffles matched pairs, keeps ``n_pairs`` and flips each at random."""
    if len(matches) == 0:
        raise ValueError("no pair can be formed: a label or class is missing")
    if n_pairs is None:
        n_pairs = len(matches)
    if not 1 <= n_pairs <= len(matches):
        raise ValueError(
            f"n_pairs must be between 1 and {len(matches)}, got {n_pairs}"
        )
    pairs = matches[rng.permutation(len(matches))[:n_pairs]]
    flip = rng.random(n_pairs) < 0.5
    pairs[flip] = pairs[flip][:, ::-1]
    return pairs
