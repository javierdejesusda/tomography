import math

import numpy as np
import pytest
from scipy import stats as scipy_stats

from tomography.site import stats

Z = 1.959963984540054


def test_wilson_matches_the_closed_form_for_half_of_a_hundred() -> None:
    lo, hi = stats.wilson(0.5, 100)
    assert lo == pytest.approx(0.4038, abs=1e-4)
    assert hi == pytest.approx(0.5962, abs=1e-4)


def test_wilson_with_no_successes_starts_at_zero_and_stays_inside() -> None:
    lo, hi = stats.wilson(0.0, 10)
    assert lo == pytest.approx(0.0, abs=1e-12)
    assert hi == pytest.approx(Z**2 / (10 + Z**2))
    assert 0.0 <= lo <= hi <= 1.0


def test_wilson_is_the_textbook_formula() -> None:
    p, n = 0.83, 214
    centre = (p + Z**2 / (2 * n)) / (1 + Z**2 / n)
    half = Z * math.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / (1 + Z**2 / n)
    lo, hi = stats.wilson(p, n)
    assert (lo, hi) == pytest.approx((centre - half, centre + half))


def test_diff_interval_with_one_size_uses_it_for_both_arms() -> None:
    d, lo, hi = stats.diff_interval(0.8, 0.6, 100)
    half = Z * math.sqrt(0.8 * 0.2 / 100 + 0.6 * 0.4 / 100)
    assert d == pytest.approx(0.2)
    assert (lo, hi) == pytest.approx((0.2 - half, 0.2 + half))


def test_diff_interval_with_two_sizes() -> None:
    d, lo, hi = stats.diff_interval(0.7, 0.6, 50, 200)
    half = Z * math.sqrt(0.7 * 0.3 / 50 + 0.6 * 0.4 / 200)
    assert (d, lo, hi) == pytest.approx((0.1, 0.1 - half, 0.1 + half))


def test_hanley_mcneil_matches_a_hand_value() -> None:
    se = stats.hanley_mcneil_se(0.5, 50, 50)
    assert se == pytest.approx(math.sqrt((0.25 + 2 * 49 / 12) / 2500))


def test_auc_diff_interval_adds_the_two_variances() -> None:
    d, lo, hi = stats.auc_diff_interval(0.8, 0.6, 40, 60)
    se1 = stats.hanley_mcneil_se(0.8, 40, 60)
    se2 = stats.hanley_mcneil_se(0.6, 40, 60)
    half = Z * math.sqrt(se1**2 + se2**2)
    assert (d, lo, hi) == pytest.approx((0.2, 0.2 - half, 0.2 + half))


def test_ols_slope_matches_scipy_with_a_t_interval() -> None:
    x = np.array([8.1, 8.55, 9.1, 9.19])
    y = np.array([0.77, 0.79, 0.84, 0.82])
    slope, lo, hi = stats.ols_slope(x, y)
    fit = scipy_stats.linregress(x, y)
    t = scipy_stats.t.ppf(0.975, len(x) - 2)
    assert slope == pytest.approx(fit.slope)
    assert lo == pytest.approx(fit.slope - t * fit.stderr)
    assert hi == pytest.approx(fit.slope + t * fit.stderr)


def test_expected_range_of_two_and_sixteen_normals() -> None:
    assert stats.expected_range(2) == pytest.approx(2 / math.sqrt(math.pi))
    assert stats.expected_range(16) == pytest.approx(3.532, abs=1e-3)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt2", "gpt2"),
        ("gpt2-xl", "gpt2"),
        ("qwen3-0.6b-base", "base"),
        ("qwen3-4b", "post"),
    ],
)
def test_role_of_a_checkpoint(model: str, expected: str) -> None:
    assert stats.role(model) == expected


def test_per_role_counts_hits_and_totals() -> None:
    flags = {
        "gpt2": True,
        "gpt2-medium": False,
        "qwen3-0.6b-base": True,
        "qwen3-1.7b-base": True,
        "qwen3-0.6b": False,
    }
    assert stats.per_role(flags) == {
        "gpt2": stats.RoleCount(1, 2),
        "base": stats.RoleCount(2, 2),
        "post": stats.RoleCount(0, 1),
    }


def test_per_role_text_names_every_role_with_checkpoints() -> None:
    counts = {
        "gpt2": stats.RoleCount(4, 4),
        "base": stats.RoleCount(3, 3),
        "post": stats.RoleCount(2, 3),
    }
    assert stats.per_role_text(counts) == (
        "4 of 4 GPT-2, 3 of 3 Qwen3 base and 2 of 3 post-trained checkpoints"
    )


def test_per_role_text_omits_roles_without_checkpoints() -> None:
    counts = {
        "gpt2": stats.RoleCount(1, 4),
        "base": stats.RoleCount(0, 0),
        "post": stats.RoleCount(2, 3),
    }
    assert stats.per_role_text(counts) == (
        "1 of 4 GPT-2 and 2 of 3 post-trained checkpoints"
    )


def test_join_and_writes_a_list_phrase() -> None:
    assert stats.join_and([]) == ""
    assert stats.join_and(["a"]) == "a"
    assert stats.join_and(["a", "b"]) == "a and b"
    assert stats.join_and(["a", "b", "c"]) == "a, b and c"


def test_runs_text_names_the_roles_and_skips_empty_ones() -> None:
    counts = {
        "gpt2": stats.RoleCount(3, 3),
        "base": stats.RoleCount(0, 0),
        "post": stats.RoleCount(3, 6),
    }
    assert stats.runs_text(counts) == (
        "3 of 3 GPT-2 runs and 3 of 6 post-trained runs"
    )
