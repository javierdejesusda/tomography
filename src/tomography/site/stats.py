"""Interval and counting helpers shared by the figures and tables.

All functions are pure. Intervals are 95 percent (z = 1.96) and, where two
arms are scored on the same items, deliberately conservative: the two
variances are added as if the arms were independent.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import math

import numpy as np
from scipy import integrate
from scipy import stats as scipy_stats

Z = 1.959963984540054
ROLES = ("gpt2", "base", "post")
ROLE_NAMES = {
    "gpt2": "GPT-2",
    "base": "Qwen3 base",
    "post": "post-trained",
}


@dataclasses.dataclass(frozen=True)
class RoleCount:
    """A count over the checkpoints of one role.

    Attributes:
        hits: Checkpoints that satisfy the statement.
        total: Checkpoints of the role that have the measurement.
    """

    hits: int
    total: int


def wilson(p: float, n: int) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Args:
        p: Observed proportion.
        n: Number of items it is read from.

    Returns:
        The lower and upper limits.
    """
    denom = 1 + Z**2 / n
    centre = (p + Z**2 / (2 * n)) / denom
    half = Z * math.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def diff_interval(
    p1: float, p2: float, n1: int, n2: int | None = None
) -> tuple[float, float, float]:
    """Conservative interval for a difference of two accuracies.

    Args:
        p1: First accuracy.
        p2: Second accuracy.
        n1: Items behind the first accuracy.
        n2: Items behind the second; defaults to `n1`.

    Returns:
        The difference `p1 - p2` with its lower and upper limits.
    """
    n2 = n1 if n2 is None else n2
    half = Z * math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    d = p1 - p2
    return d, d - half, d + half


def hanley_mcneil_se(a: float, n_pos: int, n_neg: int) -> float:
    """Standard error of an AUROC by Hanley and McNeil.

    Args:
        a: The AUROC.
        n_pos: Positive items.
        n_neg: Negative items.

    Returns:
        The standard error.
    """
    q1 = a / (2 - a)
    q2 = 2 * a**2 / (1 + a)
    var = (
        a * (1 - a) + (n_pos - 1) * (q1 - a**2) + (n_neg - 1) * (q2 - a**2)
    ) / (n_pos * n_neg)
    return math.sqrt(var)


def auc_diff_interval(
    a1: float, a2: float, n_pos: int, n_neg: int
) -> tuple[float, float, float]:
    """Conservative interval for a difference of two AUROCs.

    Args:
        a1: First AUROC.
        a2: Second AUROC.
        n_pos: Positive items.
        n_neg: Negative items.

    Returns:
        The difference `a1 - a2` with its lower and upper limits.
    """
    se1 = hanley_mcneil_se(a1, n_pos, n_neg)
    se2 = hanley_mcneil_se(a2, n_pos, n_neg)
    half = Z * math.sqrt(se1**2 + se2**2)
    d = a1 - a2
    return d, d - half, d + half


def ols_slope(
    x: Sequence[float], y: Sequence[float]
) -> tuple[float, float, float]:
    """Least-squares slope with a t interval.

    Args:
        x: Predictor values.
        y: Response values.

    Returns:
        The slope with its lower and upper 95 percent limits.
    """
    fit = scipy_stats.linregress(x, y)
    t = float(scipy_stats.t.ppf(0.975, len(x) - 2))
    half = t * float(fit.stderr)
    slope = float(fit.slope)
    return slope, slope - half, slope + half


def expected_range(k: int) -> float:
    """Expected range of `k` independent standard normal draws.

    Args:
        k: Number of draws.

    Returns:
        E[max - min].
    """

    def tail(x: float) -> float:
        cdf = float(scipy_stats.norm.cdf(x))
        return 1.0 - cdf**k - (1.0 - cdf) ** k

    value, _ = integrate.quad(tail, -np.inf, np.inf)
    return float(value)


def role(model: str) -> str:
    """Names the role of a checkpoint.

    Args:
        model: Checkpoint key.

    Returns:
        `gpt2`, `base` or `post`.
    """
    if not model.startswith("qwen3"):
        return "gpt2"
    return "base" if model.endswith("-base") else "post"


def per_role(flags: Mapping[str, bool]) -> dict[str, RoleCount]:
    """Counts a per-checkpoint statement by role.

    Args:
        flags: Checkpoint key to whether the statement holds.

    Returns:
        Role to its count, for every role in `ROLES`.
    """
    counts = {r: [0, 0] for r in ROLES}
    for model, flag in flags.items():
        cell = counts[role(model)]
        cell[0] += int(flag)
        cell[1] += 1
    return {r: RoleCount(h, t) for r, (h, t) in counts.items()}


def per_role_text(counts: Mapping[str, RoleCount]) -> str:
    """Writes a per-role count as a caption phrase.

    Args:
        counts: Role to its count.

    Returns:
        Text such as `4 of 4 GPT-2, 3 of 3 Qwen3 base and 2 of 3
        post-trained checkpoints`; roles without checkpoints are omitted.
    """
    parts = [
        f"{counts[r].hits} of {counts[r].total} {ROLE_NAMES[r]}"
        for r in ROLES
        if r in counts and counts[r].total
    ]
    if len(parts) <= 1:
        text = "".join(parts)
    else:
        text = ", ".join(parts[:-1]) + " and " + parts[-1]
    return f"{text} checkpoints"
