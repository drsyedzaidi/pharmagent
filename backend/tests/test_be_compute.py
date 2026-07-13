"""Average-BE compute validated against closed-form / hand-computed values.

GMR = exp(mean of ln(test/ref)). For a paired design the CI is built from the
sample SD of the per-subject log ratios with a Student-t critical value at
df = n - 1. These tests pin the point estimate and verdict to numbers computed
independently of the implementation.
"""
import math

import numpy as np
from scipy import stats

from app.compute.bioequivalence import assess_bioequivalence, be_one_parameter

# ---------------------------------------------------------------------------
# Paired (crossover)
# ---------------------------------------------------------------------------

def test_identical_vectors_gmr_100_and_be():
    vals = [100.0, 120.0, 80.0, 95.0, 110.0, 130.0]
    r = be_one_parameter(vals, list(vals), paired=True)
    # ln(t/r) == 0 for every pair -> GMR == 100, CI degenerate at 100.
    assert math.isclose(r["gmr_pct"], 100.0, abs_tol=1e-6)
    assert math.isclose(r["ci_lower_pct"], 100.0, abs_tol=1e-6)
    assert math.isclose(r["ci_upper_pct"], 100.0, abs_tol=1e-6)
    assert r["ci_lower_pct"] <= 100.0 <= r["ci_upper_pct"]
    assert r["within_limits"] is True
    assert math.isclose(r["cv_intra_pct"], 0.0, abs_tol=1e-9)
    assert r["n_test"] == 6 and r["n_ref"] == 6


def test_paired_gmr_matches_hand_computed():
    # Four subjects with chosen ratios; GMR is the geometric mean of ratios.
    test = [110.0, 90.0, 105.0, 100.0]
    ref = [100.0, 100.0, 100.0, 100.0]
    ratios = [t / r for t, r in zip(test, ref)]  # 1.10, 0.90, 1.05, 1.00

    log_ratios = [math.log(x) for x in ratios]
    mean_d = sum(log_ratios) / len(log_ratios)
    expected_gmr = 100.0 * math.exp(mean_d)

    n = len(ratios)
    sd_d = float(np.std(np.array(log_ratios), ddof=1))
    se = sd_d / math.sqrt(n)
    tcrit = float(stats.t.ppf(1.0 - 0.10 / 2.0, n - 1))
    expected_lo = 100.0 * math.exp(mean_d - tcrit * se)
    expected_hi = 100.0 * math.exp(mean_d + tcrit * se)
    expected_cv = 100.0 * math.sqrt(math.exp(sd_d * sd_d) - 1.0)

    r = be_one_parameter(test, ref, paired=True, alpha=0.10)
    # Module rounds outputs to 4 dp; compare against the rounded expectation.
    assert math.isclose(r["gmr_pct"], round(expected_gmr, 4), abs_tol=5e-5)
    assert math.isclose(r["ci_lower_pct"], round(expected_lo, 4), abs_tol=5e-5)
    assert math.isclose(r["ci_upper_pct"], round(expected_hi, 4), abs_tol=5e-5)
    assert math.isclose(r["cv_intra_pct"], round(expected_cv, 4), abs_tol=5e-5)


def test_paired_clearly_outside_limits_not_be():
    # Test ~ 2x reference -> GMR ~ 200%, far above 125.
    test = [200.0, 210.0, 190.0, 205.0]
    ref = [100.0, 100.0, 100.0, 100.0]
    r = be_one_parameter(test, ref, paired=True)
    assert r["gmr_pct"] > 125.0
    assert r["within_limits"] is False


def test_paired_drops_nonpositive_and_none():
    test = [100.0, -5.0, None, 120.0]
    ref = [100.0, 100.0, 100.0, 100.0]
    r = be_one_parameter(test, ref, paired=True)
    # Only pairs (100,100) and (120,100) survive.
    assert r["n_test"] == 2
    expected_gmr = 100.0 * math.exp((math.log(1.0) + math.log(1.2)) / 2.0)
    assert math.isclose(r["gmr_pct"], round(expected_gmr, 4), abs_tol=5e-5)


def test_paired_too_few_returns_nulls():
    r = be_one_parameter([100.0], [100.0], paired=True)
    assert r["gmr_pct"] is None
    assert r["ci_lower_pct"] is None
    assert r["ci_upper_pct"] is None
    assert r["within_limits"] is False
    assert r["cv_intra_pct"] is None
    assert r["limits"] == [80.0, 125.0]


# ---------------------------------------------------------------------------
# Parallel (two-group)
# ---------------------------------------------------------------------------

def test_parallel_identical_groups_gmr_100():
    g = [90.0, 100.0, 110.0, 105.0, 95.0]
    r = be_one_parameter(g, list(g), paired=False)
    assert math.isclose(r["gmr_pct"], 100.0, abs_tol=1e-6)
    assert r["within_limits"] is True
    assert r["cv_intra_pct"] is None  # not defined for parallel


def test_parallel_gmr_and_ci_hand_computed():
    test = [105.0, 110.0, 100.0, 115.0]
    ref = [100.0, 95.0, 105.0, 98.0]
    lt = np.log(np.array(test))
    lr = np.log(np.array(ref))
    n1, n2 = lt.size, lr.size
    df = n1 + n2 - 2
    sp = math.sqrt(((n1 - 1) * lt.var(ddof=1) + (n2 - 1) * lr.var(ddof=1)) / df)
    se = sp * math.sqrt(1.0 / n1 + 1.0 / n2)
    diff = float(lt.mean() - lr.mean())
    tcrit = float(stats.t.ppf(1.0 - 0.10 / 2.0, df))

    expected_gmr = 100.0 * math.exp(diff)
    expected_lo = 100.0 * math.exp(diff - tcrit * se)
    expected_hi = 100.0 * math.exp(diff + tcrit * se)

    r = be_one_parameter(test, ref, paired=False, alpha=0.10)
    assert math.isclose(r["gmr_pct"], round(expected_gmr, 4), abs_tol=5e-5)
    assert math.isclose(r["ci_lower_pct"], round(expected_lo, 4), abs_tol=5e-5)
    assert math.isclose(r["ci_upper_pct"], round(expected_hi, 4), abs_tol=5e-5)


def test_parallel_one_group_too_small_returns_nulls():
    r = be_one_parameter([100.0], [90.0, 100.0, 110.0], paired=False)
    assert r["gmr_pct"] is None
    assert r["within_limits"] is False


# ---------------------------------------------------------------------------
# assess_bioequivalence orchestration
# ---------------------------------------------------------------------------

def test_assess_all_be_true():
    test = {"Cmax": [100.0, 105.0, 95.0, 102.0],
            "AUC": [100.0, 98.0, 101.0, 99.0]}
    ref = {"Cmax": [100.0, 105.0, 95.0, 102.0],
           "AUC": [100.0, 98.0, 101.0, 99.0]}
    out = assess_bioequivalence(test, ref, paired=True)
    assert out["design"] == "crossover (paired)"
    assert out["limits"] == [80.0, 125.0]
    assert set(out["parameters"]) == {"Cmax", "AUC"}
    assert out["bioequivalent"] is True


def test_assess_one_param_fails_blocks_be():
    test = {"Cmax": [100.0, 105.0, 95.0, 102.0],   # equivalent
            "AUC": [200.0, 210.0, 190.0, 205.0]}    # not equivalent
    ref = {"Cmax": [100.0, 105.0, 95.0, 102.0],
           "AUC": [100.0, 100.0, 100.0, 100.0]}
    out = assess_bioequivalence(test, ref, paired=True)
    assert out["parameters"]["Cmax"]["within_limits"] is True
    assert out["parameters"]["AUC"]["within_limits"] is False
    assert out["bioequivalent"] is False


def test_assess_only_common_params_assessed():
    test = {"Cmax": [100.0, 105.0], "Tmax": [1.0, 2.0]}
    ref = {"Cmax": [100.0, 105.0], "AUC": [100.0, 105.0]}
    out = assess_bioequivalence(test, ref, paired=True)
    assert set(out["parameters"]) == {"Cmax"}


def test_assess_parallel_design_label():
    out = assess_bioequivalence(
        {"Cmax": [100.0, 105.0]}, {"Cmax": [100.0, 105.0]}, paired=False)
    assert out["design"] == "parallel"
