"""Tests for app.compute.poppk — two-stage (STS) population PK summary.

All expected values are closed-form / hand-computed, not snapshot.
"""
import math

import pytest

from app.compute.poppk import (
    _geocv_pct,
    _geomean,
    covariate_effect,
    two_stage_summary,
)

# --- helper-level closed-form checks ---------------------------------------

def test_geomean_cube_root_of_product():
    # CL_F = [4, 5, 6.25]; product = 125; cube root = 5 exactly.
    assert _geomean([4.0, 5.0, 6.25]) == pytest.approx(5.0, abs=1e-12)


def test_geomean_drops_nonpositive_and_none():
    # Non-positive / None entries ignored; geomean of {2, 8} = sqrt(16) = 4.
    assert _geomean([2.0, None, -3.0, 0.0, 8.0]) == pytest.approx(4.0, abs=1e-12)


def test_geomean_returns_none_when_no_usable_values():
    assert _geomean([]) is None
    assert _geomean([0.0, -1.0, None]) is None


def test_geocv_pct_closed_form():
    # ln([4,5,6.25]) has sample variance (ddof=1) = 0.0497929...
    # geoCV% = 100*sqrt(exp(var)-1).
    vals = [4.0, 5.0, 6.25]
    logs = [math.log(v) for v in vals]
    mean = sum(logs) / len(logs)
    var = sum((x - mean) ** 2 for x in logs) / (len(logs) - 1)
    expected = 100.0 * math.sqrt(math.exp(var) - 1.0)
    assert _geocv_pct(vals) == pytest.approx(expected, abs=1e-9)
    # numeric anchor: ~22.5952 %
    assert _geocv_pct(vals) == pytest.approx(22.5952, abs=1e-3)


def test_geocv_pct_needs_two_values():
    assert _geocv_pct([5.0]) is None
    assert _geocv_pct([]) is None


# --- two_stage_summary ------------------------------------------------------

def test_two_stage_summary_typical_value_and_iiv():
    subjects = [
        {"subject": "S1", "CL_F": 4.0},
        {"subject": "S2", "CL_F": 5.0},
        {"subject": "S3", "CL_F": 6.25},
    ]
    out = two_stage_summary(subjects, keys=("CL_F",))

    assert out["method"] == "two-stage (STS)"
    assert out["n_subjects"] == 3
    assert set(out.keys()) == {"method", "n_subjects", "parameters"}

    cl = out["parameters"]["CL_F"]
    assert set(cl.keys()) == {"typical_value", "iiv_cv_pct", "median", "n"}
    assert cl["typical_value"] == pytest.approx(5.0, abs=1e-6)  # geomean
    assert cl["iiv_cv_pct"] == pytest.approx(22.5952, abs=1e-3)
    assert cl["median"] == pytest.approx(5.0, abs=1e-9)
    assert cl["n"] == 3


def test_two_stage_summary_omits_key_with_no_usable_values():
    subjects = [
        {"subject": "S1", "CL_F": 4.0, "ka": None},
        {"subject": "S2", "CL_F": 5.0, "ka": 0.0},
        {"subject": "S3", "CL_F": 6.25},  # ka missing entirely
    ]
    out = two_stage_summary(subjects, keys=("CL_F", "Vz_F", "ka"))
    # CL_F usable -> present; Vz_F absent everywhere; ka only None/0 -> omitted.
    assert "CL_F" in out["parameters"]
    assert "Vz_F" not in out["parameters"]
    assert "ka" not in out["parameters"]
    assert out["n_subjects"] == 3


def test_two_stage_summary_skips_nonpositive_in_n():
    subjects = [
        {"subject": "S1", "CL_F": 4.0},
        {"subject": "S2", "CL_F": -1.0},   # dropped
        {"subject": "S3", "CL_F": None},   # dropped
        {"subject": "S4", "CL_F": 6.0},
    ]
    out = two_stage_summary(subjects, keys=("CL_F",))
    cl = out["parameters"]["CL_F"]
    assert cl["n"] == 2
    assert cl["typical_value"] == pytest.approx(math.sqrt(24.0), abs=1e-6)


# --- covariate_effect -------------------------------------------------------

def test_covariate_effect_perfect_log_linear():
    # Construct ln(CL_F) = a + b*WT exactly => pearson_r == 1, slope == b.
    a, b = 0.5, 0.03
    weights = [50.0, 60.0, 70.0, 80.0, 90.0]
    subjects = []
    cov = {}
    for i, wt in enumerate(weights):
        sid = f"S{i}"
        cl = math.exp(a + b * wt)
        subjects.append({"subject": sid, "CL_F": cl})
        cov[sid] = wt

    out = covariate_effect(subjects, cov, param_key="CL_F")
    assert set(out.keys()) == {"param", "n", "slope", "pearson_r", "r_squared"}
    assert out["param"] == "CL_F"
    assert out["n"] == 5
    assert out["slope"] == pytest.approx(b, abs=1e-6)        # known slope
    assert out["pearson_r"] == pytest.approx(1.0, abs=1e-6)
    assert out["r_squared"] == pytest.approx(1.0, abs=1e-6)


def test_covariate_effect_negative_relationship():
    # ln(CL_F) decreasing in covariate -> r ~ -1, slope negative.
    a, b = 1.0, -0.02
    cov_vals = [20.0, 40.0, 60.0, 80.0]
    subjects = []
    cov = {}
    for i, c in enumerate(cov_vals):
        sid = f"P{i}"
        subjects.append({"subject": sid, "CL_F": math.exp(a + b * c)})
        cov[sid] = c
    out = covariate_effect(subjects, cov, param_key="CL_F")
    assert out["slope"] == pytest.approx(b, abs=1e-6)
    assert out["pearson_r"] == pytest.approx(-1.0, abs=1e-6)
    assert out["r_squared"] == pytest.approx(1.0, abs=1e-6)


def test_covariate_effect_insufficient_points_returns_none():
    subjects = [
        {"subject": "S1", "CL_F": 5.0},
        {"subject": "S2", "CL_F": 6.0},
    ]
    cov = {"S1": 70.0, "S2": 80.0}
    out = covariate_effect(subjects, cov, param_key="CL_F")
    assert out["n"] == 2
    assert out["slope"] is None
    assert out["pearson_r"] is None
    assert out["r_squared"] is None


def test_covariate_effect_only_counts_matched_positive_pairs():
    subjects = [
        {"subject": "S1", "CL_F": 5.0},
        {"subject": "S2", "CL_F": None},     # dropped (no param)
        {"subject": "S3", "CL_F": 6.0},
        {"subject": "S4", "CL_F": 7.0},      # no covariate -> dropped
    ]
    cov = {"S1": 70.0, "S2": 75.0, "S3": 80.0}  # S4 absent
    out = covariate_effect(subjects, cov, param_key="CL_F")
    assert out["n"] == 2  # only S1, S3 are fully paired & positive
    assert out["slope"] is None
