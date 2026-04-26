"""Statistical significance tests for paired between-system comparison.

Used for RQ3 head-to-head, given two systems evaluated on the same query
set, decide whether the difference in their headline metric is real or
random.

Two tests are reported. Paired randomization is the primary, no normality
assumption, recommended for IR by Smucker, Allan & Carterette (CIKM 2007).
Paired t-test is the secondary, matches the conventional matkul-style
expectation. When the two converge the conclusion is robust, when they
diverge the randomization result is the trusted one because hit/miss is
Bernoulli, not Gaussian.

Effect size is Cohen's d for paired samples with Sawilowsky 2009 labels.
P-values without effect size are easy to misread, both are reported.

Pure Python plus optional scipy. The randomization test never needs scipy.
The t-test prefers scipy.stats but falls back to a normal approximation
that is acceptable for the thesis sample size (n >= 50).
"""

from __future__ import annotations

import math
import random


# ----------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------

DEFAULT_RANDOMIZATION_B = 10000
DEFAULT_SEED = 42


# ----------------------------------------------------------------------
# Paired randomization test (Smucker, Allan & Carterette CIKM 2007)
# ----------------------------------------------------------------------

def paired_randomization(
    a: list[float],
    b: list[float],
    *,
    B: int = DEFAULT_RANDOMIZATION_B,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Two-sided paired randomization test for the mean of differences.

    For each query i compute d_i = a_i - b_i. Null hypothesis, the sign of
    d_i is exchangeable, equivalent to "system labels are interchangeable".
    Resample sign assignments B times, recompute the mean each time, count
    how often the absolute permuted mean is at least as extreme as the
    observed absolute mean. Add-one smoothing follows Phipson & Smyth (2010).
    """
    if len(a) != len(b):
        raise ValueError(f"paired arrays must be same length, got {len(a)} and {len(b)}")
    n = len(a)
    if n < 2:
        return {
            "method": "paired-randomization",
            "n": n, "B": B, "seed": seed,
            "mean_diff": 0.0, "p_value": 1.0,
            "note": "n < 2, test undefined",
        }

    diffs = [float(ai) - float(bi) for ai, bi in zip(a, b)]
    observed = sum(diffs) / n
    abs_observed = abs(observed)

    rng = random.Random(seed)
    extreme_count = 0
    for _ in range(B):
        permuted_sum = 0.0
        for d in diffs:
            sign = 1.0 if rng.random() < 0.5 else -1.0
            permuted_sum += sign * d
        if abs(permuted_sum / n) >= abs_observed:
            extreme_count += 1

    # Phipson & Smyth (2010) unbiased Monte Carlo p-value, never zero.
    p_value = (extreme_count + 1) / (B + 1)
    return {
        "method": "paired-randomization",
        "n": n,
        "B": B,
        "seed": seed,
        "mean_diff": observed,
        "p_value": p_value,
    }


# ----------------------------------------------------------------------
# Paired t-test (matkul-conventional secondary test)
# ----------------------------------------------------------------------

def paired_t_test(a: list[float], b: list[float]) -> dict:
    """Two-sided paired t-test on the mean of differences.

    Reported alongside paired randomization as a sanity cross-check. For
    binary hit/miss the normality assumption fails, so the randomization
    result is the trusted one when the two diverge.
    """
    if len(a) != len(b):
        raise ValueError(f"paired arrays must be same length, got {len(a)} and {len(b)}")
    n = len(a)
    if n < 2:
        return {
            "method": "paired-t-test",
            "n": n, "p_value": 1.0, "t_stat": 0.0, "df": 0,
            "mean_diff": 0.0, "std_err": 0.0,
            "note": "n < 2, test undefined",
        }

    diffs = [float(ai) - float(bi) for ai, bi in zip(a, b)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    df = n - 1
    if var_d == 0.0:
        return {
            "method": "paired-t-test",
            "n": n, "df": df,
            "t_stat": 0.0, "mean_diff": mean_d, "std_err": 0.0,
            "p_value": 1.0,
            "note": "zero variance, all differences identical",
        }

    se = math.sqrt(var_d / n)
    t_stat = mean_d / se
    p_value = _two_sided_p_t(t_stat, df)
    return {
        "method": "paired-t-test",
        "n": n,
        "df": df,
        "t_stat": t_stat,
        "mean_diff": mean_d,
        "std_err": se,
        "p_value": p_value,
    }


def _two_sided_p_t(t_stat: float, df: int) -> float:
    """Two-sided p-value for a t-statistic. Uses scipy if available."""
    try:
        from scipy.stats import t as scipy_t
        return float(2.0 * scipy_t.sf(abs(t_stat), df))
    except ImportError:
        # Normal approximation, fine for thesis-scale n.
        return 2.0 * (1.0 - _normal_cdf(abs(t_stat)))


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ----------------------------------------------------------------------
# Effect size, Cohen's d for paired samples + Sawilowsky 2009 label
# ----------------------------------------------------------------------

def cohens_d_paired(a: list[float], b: list[float]) -> dict:
    """Cohen's d for paired samples, mean of differences over their SD."""
    if len(a) != len(b):
        raise ValueError(f"paired arrays must be same length, got {len(a)} and {len(b)}")
    n = len(a)
    if n < 2:
        return {"d": 0.0, "label": "n/a", "n": n, "note": "n < 2"}
    diffs = [float(ai) - float(bi) for ai, bi in zip(a, b)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    sd = math.sqrt(var_d) if var_d > 0 else 0.0
    if sd == 0.0:
        return {"d": 0.0, "label": "zero-variance", "n": n}
    d = mean_d / sd
    return {"d": d, "label": sawilowsky_label(d), "n": n}


def sawilowsky_label(d: float) -> str:
    """Sawilowsky (2009) descriptive label for Cohen's d magnitude."""
    abs_d = abs(d)
    if abs_d < 0.01:
        return "trivial"
    if abs_d < 0.20:
        return "very small"
    if abs_d < 0.50:
        return "small"
    if abs_d < 0.80:
        return "medium"
    if abs_d < 1.20:
        return "large"
    if abs_d < 2.0:
        return "very large"
    return "huge"


# ----------------------------------------------------------------------
# Convenience, run the whole headline suite at once
# ----------------------------------------------------------------------

def compare_paired(
    a: list[float],
    b: list[float],
    *,
    B: int = DEFAULT_RANDOMIZATION_B,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Bundle paired randomization, paired t-test, and Cohen's d into one dict.

    Use this as the headline output for one (system_a, system_b, metric).
    """
    rand = paired_randomization(a, b, B=B, seed=seed)
    t = paired_t_test(a, b)
    eff = cohens_d_paired(a, b)
    convergent = (
        rand.get("p_value", 1.0) < 0.05
    ) == (
        t.get("p_value", 1.0) < 0.05
    )
    return {
        "n": rand.get("n", len(a)),
        "mean_diff": rand.get("mean_diff", 0.0),
        "paired_randomization": rand,
        "paired_t_test": t,
        "cohens_d": eff,
        "tests_converge": convergent,
    }


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------

def run_self_test() -> None:
    """Sanity tests for each function. Called from the eval CLI on demand."""
    def assert_close(actual: float, expected: float, tol: float = 1e-6) -> None:
        if abs(actual - expected) > tol:
            raise AssertionError(f"expected {expected}, got {actual}")

    # Identical inputs -> p approx 1, mean_diff 0
    a = [1.0, 0.0, 1.0, 1.0, 0.0] * 20  # n=100
    res = compare_paired(a, a, B=2000, seed=1)
    assert_close(res["mean_diff"], 0.0)
    if res["paired_randomization"]["p_value"] < 0.5:
        raise AssertionError(
            f"identical inputs should give large p, got {res['paired_randomization']['p_value']}"
        )

    # Obvious difference -> small p
    b_better = [1.0] * 100
    b_worse = [0.0] * 100
    res = compare_paired(b_better, b_worse, B=2000, seed=1)
    if res["paired_randomization"]["p_value"] >= 0.01:
        raise AssertionError(
            f"obvious diff should give small p, got {res['paired_randomization']['p_value']}"
        )
    if res["cohens_d"]["d"] != 0 and res["cohens_d"]["label"] != "zero-variance":
        # All-1 vs all-0 has zero variance of differences (all diffs = 1).
        # That is the documented edge-case path.
        raise AssertionError(f"unexpected cohens_d branch, got {res['cohens_d']}")

    # Mild difference, n=200, mean diff 0.05
    rng = random.Random(7)
    s_a = [1.0 if rng.random() < 0.55 else 0.0 for _ in range(200)]
    s_b = [1.0 if rng.random() < 0.50 else 0.0 for _ in range(200)]
    res = compare_paired(s_a, s_b, B=5000, seed=1)
    # Don't assert p threshold (random noise), just sanity-check structure
    expected_keys = {"paired_randomization", "paired_t_test", "cohens_d", "tests_converge"}
    if not expected_keys.issubset(res.keys()):
        raise AssertionError(f"compare_paired missing keys, got {res.keys()}")

    # Sawilowsky label boundaries
    if sawilowsky_label(0.0) != "trivial":
        raise AssertionError("sawilowsky_label(0.0) wrong")
    if sawilowsky_label(0.30) != "small":
        raise AssertionError("sawilowsky_label(0.30) wrong")
    if sawilowsky_label(1.5) != "very large":
        raise AssertionError("sawilowsky_label(1.5) wrong")
    if sawilowsky_label(-2.5) != "huge":
        raise AssertionError("sawilowsky_label should use abs value")
