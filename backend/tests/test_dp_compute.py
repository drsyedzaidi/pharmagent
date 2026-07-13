"""Tests for app.compute.dose_proportionality — closed-form validation.

For value = k * dose**p, the power model ln(value) = ln(k) + p*ln(dose) is an
exact linear relationship, so OLS recovers slope == p and intercept == ln(k)
with no residual error. These are hand-verifiable against the analytic slope.
"""
from __future__ import annotations

import math

import pytest

from app.compute.dose_proportionality import (
    assess_dose_proportionality,
    power_model,
)

DOSES = [50.0, 100.0, 200.0, 400.0]


def test_perfectly_proportional_slope_one():
    # value = 3 * dose  -> slope exactly 1.0, intercept ln(3)
    k = 3.0
    values = [k * d for d in DOSES]
    res = power_model(DOSES, values)

    assert res["n"] == 4
    assert res["slope"] == pytest.approx(1.0, abs=1e-6)
    assert res["intercept"] == pytest.approx(math.log(k), abs=1e-6)
    assert res["r_squared"] == pytest.approx(1.0, abs=1e-9)
    # zero residual -> CI collapses onto the slope
    assert res["slope_ci_lower"] == pytest.approx(1.0, abs=1e-6)
    assert res["slope_ci_upper"] == pytest.approx(1.0, abs=1e-6)
    assert res["dose_ratio"] == pytest.approx(8.0, abs=1e-9)
    assert res["proportional"] is True


def test_sub_proportional_slope_half():
    # value = 2 * dose**0.5 -> slope exactly 0.5
    k = 2.0
    values = [k * d ** 0.5 for d in DOSES]
    res = power_model(DOSES, values)

    assert res["slope"] == pytest.approx(0.5, abs=1e-6)
    assert res["intercept"] == pytest.approx(math.log(k), abs=1e-6)
    assert res["r_squared"] == pytest.approx(1.0, abs=1e-9)
    # 0.5 is below the critical lower bound -> not proportional
    assert res["proportional"] is False


def test_super_proportional_slope_one_point_five():
    # value = 0.5 * dose**1.5 -> slope exactly 1.5
    k = 0.5
    values = [k * d ** 1.5 for d in DOSES]
    res = power_model(DOSES, values)

    assert res["slope"] == pytest.approx(1.5, abs=1e-6)
    assert res["intercept"] == pytest.approx(math.log(k), abs=1e-6)
    assert res["proportional"] is False


def test_critical_region_closed_form():
    # dose_ratio = 8, theta = [0.8, 1.25]
    # crit_lower = 1 + ln(0.8)/ln(8), crit_upper = 1 + ln(1.25)/ln(8)
    values = [3.0 * d for d in DOSES]
    res = power_model(DOSES, values)
    ln_r = math.log(8.0)
    expected_lower = 1.0 + math.log(0.8) / ln_r
    expected_upper = 1.0 + math.log(1.25) / ln_r
    assert res["critical_region"][0] == pytest.approx(expected_lower, abs=1e-6)
    assert res["critical_region"][1] == pytest.approx(expected_upper, abs=1e-6)


def test_se_and_ci_closed_form_three_points():
    # Hand-computed OLS on (x, y): x = ln(dose), y given below.
    # doses 1, 10, 100 -> x = 0, ln10, 2*ln10
    doses = [1.0, 10.0, 100.0]
    # choose y with a known residual to exercise se_slope/CI math
    # y = [0.0, 1.0, 2.5] -> xbar = 2*ln10/3
    values = [math.exp(0.0), math.exp(1.0), math.exp(2.5)]
    res = power_model(doses, values, alpha=0.10)

    x = [math.log(d) for d in doses]
    y = [0.0, 1.0, 2.5]
    xbar = sum(x) / 3
    ybar = sum(y) / 3
    sxx = sum((xi - xbar) ** 2 for xi in x)
    sxy = sum((xi - xbar) * (yi - ybar) for xi, yi in zip(x, y))
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    ssr = sum((yi - (intercept + slope * xi)) ** 2 for xi, yi in zip(x, y))
    se_slope = math.sqrt((ssr / (3 - 2)) / sxx)
    from scipy import stats

    tcrit = stats.t.ppf(1 - 0.10 / 2, 3 - 2)
    assert res["slope"] == pytest.approx(slope, abs=1e-6)
    assert res["intercept"] == pytest.approx(intercept, abs=1e-6)
    assert res["slope_ci_lower"] == pytest.approx(slope - tcrit * se_slope, abs=1e-6)
    assert res["slope_ci_upper"] == pytest.approx(slope + tcrit * se_slope, abs=1e-6)


def test_insufficient_data_returns_note():
    res = power_model([100.0, 100.0], [5.0, 6.0])
    assert res["slope"] is None
    assert res["proportional"] is None
    assert "note" in res

    res2 = power_model([100.0, 200.0], [5.0, 6.0])  # n=2 < 3
    assert res2["slope"] is None
    assert "note" in res2


def test_drops_nonpositive_and_none():
    doses = [50.0, 100.0, -200.0, 400.0, None]
    values = [150.0, 300.0, 600.0, None, 1000.0]
    # only (50,150),(100,300) survive -> n=2 -> insufficient
    res = power_model(doses, values)
    assert res["n"] == 2
    assert res["slope"] is None


def test_assess_aggregates_all_params():
    prop_vals = [3.0 * d for d in DOSES]          # slope 1.0 -> proportional
    sub_vals = [2.0 * d ** 0.5 for d in DOSES]    # slope 0.5 -> not proportional
    out = assess_dose_proportionality(
        DOSES, {"AUC": prop_vals, "Cmax": sub_vals}
    )
    assert out["alpha"] == 0.10
    assert out["theta"] == [0.8, 1.25]
    assert out["parameters"]["AUC"]["proportional"] is True
    assert out["parameters"]["Cmax"]["proportional"] is False
    assert out["proportional"] is False


def test_assess_all_proportional_true():
    a = [3.0 * d for d in DOSES]
    b = [7.0 * d for d in DOSES]
    out = assess_dose_proportionality(DOSES, {"AUC": a, "Cmax": b})
    assert out["proportional"] is True
