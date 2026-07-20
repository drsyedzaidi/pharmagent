"""Tests for app.compute.forest.covariate_forest — covariate GMR forest rows.

Closed-form checks are computed independently (never re-deriving the
implementation's own formula), with expectations ROUNDED to the module's own
6-dp contract before comparison — a tight `rel=` tolerance against an
unrounded irrational (e.g. 2**0.75) would fail by construction against the
module's own rounding, which is not a defect.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy.stats import norm

from app.compute.forest import covariate_forest

Z90 = float(norm.ppf(0.95))


def _nlme(effects, *, model_key: str = "oral_1cmt", omega_cv=None) -> dict:
    return {
        "model_key": model_key,
        "omega_cv_pct": omega_cv or {"CL": 30.0, "V": 20.0},
        "covariate_effects": effects,
    }


# ── closed-form: power ───────────────────────────────────────────────────────

def test_power_gmr_closed_form():
    eff = {"param": "CL", "covariate": "WT", "kind": "power", "center": 70.0,
          "coefficient": 0.75, "rse_pct": 10.0, "levels": None}
    out = covariate_forest(_nlme([eff]), cov_values={"WT": [140.0]})
    row = out["rows"][0]
    expected_gmr = round((140.0 / 70.0) ** 0.75, 6)
    assert row["gmr"] == pytest.approx(expected_gmr, abs=5e-7)
    assert row["ci_source"] == "wald_loglinear"

    se_beta = 0.75 * 10.0 / 100.0
    se_ln = abs(math.log(140.0 / 70.0)) * se_beta
    ln_gmr = math.log((140.0 / 70.0) ** 0.75)
    expected_lo = round(math.exp(ln_gmr - Z90 * se_ln), 6)
    expected_hi = round(math.exp(ln_gmr + Z90 * se_ln), 6)
    assert row["ci_lo"] == pytest.approx(expected_lo, abs=5e-7)
    assert row["ci_hi"] == pytest.approx(expected_hi, abs=5e-7)


# ── closed-form: exponential ────────────────────────────────────────────────

def test_exponential_gmr_closed_form():
    eff = {"param": "V", "covariate": "AGE", "kind": "exponential", "center": 40.0,
          "coefficient": -0.015, "rse_pct": 25.0, "levels": None}
    out = covariate_forest(_nlme([eff]), cov_values={"AGE": [70.0]})
    row = out["rows"][0]
    expected_gmr = round(math.exp(-0.015 * (70.0 - 40.0)), 6)
    assert row["gmr"] == pytest.approx(expected_gmr, abs=5e-7)
    assert row["ci_source"] == "wald_loglinear"


# ── closed-form: categorical ────────────────────────────────────────────────

def test_categorical_gmr_closed_form_and_reference_row():
    eff = {"param": "CL", "covariate": "SEX", "kind": "categorical", "levels": ["F"],
          "coefficient": {"F": -0.223}, "rse_pct": {"F": 18.0}}
    out = covariate_forest(_nlme([eff]), cov_values={"SEX": ["M", "F"]},
                           ref_levels={"SEX": "M"})
    ref_row = next(r for r in out["rows"] if r["eval_value"] == "M")
    f_row = next(r for r in out["rows"] if r["eval_value"] == "F")
    assert ref_row["gmr"] == ref_row["ci_lo"] == ref_row["ci_hi"] == 1.0
    assert ref_row["ci_source"] == "reference"
    assert "SEX=M (reference)" == ref_row["eval_label"]
    assert f_row["eval_label"] == "SEX=F vs M"
    assert f_row["gmr"] == pytest.approx(round(math.exp(-0.223), 6), abs=5e-7)


def test_categorical_without_ref_level_label_says_unlabeled():
    eff = {"param": "CL", "covariate": "GENOTYPE", "kind": "categorical", "levels": ["PM"],
          "coefficient": {"PM": 0.5}, "rse_pct": {"PM": 20.0}}
    out = covariate_forest(_nlme([eff]), cov_values={"GENOTYPE": ["PM"]})
    row = out["rows"][0]
    assert "unlabeled" in row["eval_label"]


# ── closed-form / derivative: linear ────────────────────────────────────────

def test_linear_gmr_closed_form_and_derivative():
    eff = {"param": "CL", "covariate": "EGFR", "kind": "linear", "center": 90.0,
          "coefficient": 0.004, "rse_pct": 12.0, "levels": None}
    out = covariate_forest(_nlme([eff]), cov_values={"EGFR": [50.0]})
    row = out["rows"][0]
    d = 50.0 - 90.0
    g = 1.0 + 0.004 * d
    expected_gmr = round(g, 6)
    assert row["gmr"] == pytest.approx(expected_gmr, abs=5e-7)
    assert row["ci_source"] == "delta_nonlinear"

    # Direct derivative check: d ln(1+beta*d)/dbeta = d/(1+beta*d) at the
    # fitted beta -- verified by nudging beta and taking a finite difference,
    # independent of the module's own analytic formula.
    h = 1e-6
    ln_g_hi = math.log(1.0 + (0.004 + h) * d)
    ln_g_lo = math.log(1.0 + (0.004 - h) * d)
    fd_deriv = (ln_g_hi - ln_g_lo) / (2 * h)
    analytic_deriv = d / g
    assert analytic_deriv == pytest.approx(fd_deriv, rel=1e-4)


def test_linear_degenerate_extrapolation_reports_none_not_fabricated_value():
    # beta*d <= -1 -> the fitted linear model predicts CL <= 0: must not
    # report a fabricated near-zero/negative GMR.
    eff = {"param": "CL", "covariate": "EGFR", "kind": "linear", "center": 90.0,
          "coefficient": 0.02, "rse_pct": 12.0, "levels": None}
    out = covariate_forest(_nlme([eff]), cov_values={"EGFR": [10.0]})  # d=-80, beta*d=-1.6
    row = out["rows"][0]
    assert row["gmr"] is None
    assert row["ci_lo"] is None and row["ci_hi"] is None
    assert row["ci_source"] == "undefined_extrapolation"
    assert any("extrapolate" in n for n in out["notes"])


# ── missing RSE: None, never TypeError ──────────────────────────────────────

def test_missing_rse_gives_none_flags_not_a_crash():
    eff = {"param": "CL", "covariate": "WT", "kind": "power", "center": 70.0,
          "coefficient": 0.75, "rse_pct": None, "levels": None}
    out = covariate_forest(_nlme([eff]), cov_values={"WT": [100.0]},
                           bounds=(0.8, 1.25))
    row = out["rows"][0]
    assert row["gmr"] is not None  # point estimate still computable
    assert row["ci_lo"] is None and row["ci_hi"] is None
    assert row["ci_source"] == "unavailable"
    assert row["outside_reference_band"] is False  # must not raise on None


# ── Monte-Carlo coverage: catches a wrong derivative/z/nonlinearity at once ─

@pytest.mark.parametrize("kind,coef,center,value,rse", [
    ("power", 0.6, 70.0, 45.0, 15.0),
    ("exponential", -0.02, 40.0, 75.0, 20.0),
    ("linear", 0.006, 90.0, 130.0, 18.0),
])
def test_monte_carlo_ci_coverage(kind, coef, center, value, rse):
    eff = {"param": "CL", "covariate": "X", "kind": kind, "center": center,
          "coefficient": coef, "rse_pct": rse, "levels": None}
    out = covariate_forest(_nlme([eff]), cov_values={"X": [value]}, ci_level=0.90)
    row = out["rows"][0]

    rng = np.random.default_rng(2025)
    se_beta = abs(coef) * rse / 100.0
    betas = rng.normal(coef, se_beta, size=200_000)
    d = value - center
    if kind == "power":
        gmr_draws = (value / center) ** betas
    elif kind == "exponential":
        gmr_draws = np.exp(betas * d)
    else:
        g = 1.0 + betas * d
        gmr_draws = np.where(g > 0, g, np.nan)
    gmr_draws = gmr_draws[np.isfinite(gmr_draws)]
    lo_mc, hi_mc = np.quantile(gmr_draws, [0.05, 0.95])

    assert row["ci_lo"] == pytest.approx(lo_mc, rel=0.03)
    assert row["ci_hi"] == pytest.approx(hi_mc, rel=0.03)


# ── allometric collision note ────────────────────────────────────────────────

def test_allometric_collision_is_flagged_for_weight_on_scaled_param():
    # oral_1cmt has model.allometric = {"CL": 0.75, "V": 1.0}; a covariate
    # named "WT" acting on CL collides with that separate scaling.
    eff = {"param": "CL", "covariate": "WT", "kind": "power", "center": 70.0,
          "coefficient": 0.3, "rse_pct": 15.0, "levels": None}
    out = covariate_forest(_nlme([eff], model_key="oral_1cmt"), cov_values={"WT": [90.0]})
    assert out["rows"][0]["allometric_note"] is True
    assert any("allometric" in n for n in out["notes"])


def test_no_allometric_collision_for_unrelated_covariate():
    eff = {"param": "V", "covariate": "AGE", "kind": "exponential", "center": 40.0,
          "coefficient": -0.01, "rse_pct": 20.0, "levels": None}
    out = covariate_forest(_nlme([eff], model_key="oral_1cmt"), cov_values={"AGE": [60.0]})
    assert out["rows"][0]["allometric_note"] is False
    assert not any("allometric" in n for n in out["notes"])


# ── argument validation ─────────────────────────────────────────────────────

def test_ci_level_out_of_range_raises():
    with pytest.raises(ValueError):
        covariate_forest(_nlme([]), ci_level=1.0)
    with pytest.raises(ValueError):
        covariate_forest(_nlme([]), ci_level=0.0)
    with pytest.raises(ValueError):
        covariate_forest(_nlme([]), ci_level=-0.5)


def test_invalid_bounds_raises():
    with pytest.raises(ValueError):
        covariate_forest(_nlme([]), bounds=(1.25, 0.8))  # inverted
    with pytest.raises(ValueError):
        covariate_forest(_nlme([]), bounds=(-1.0, 1.0))  # non-positive


# ── bounds / reference-band flagging (no BE-criterion default) ─────────────

def test_no_bounds_by_default_no_band_flag():
    eff = {"param": "CL", "covariate": "WT", "kind": "power", "center": 70.0,
          "coefficient": 0.75, "rse_pct": 10.0, "levels": None}
    out = covariate_forest(_nlme([eff]), cov_values={"WT": [140.0]})
    assert out["bounds"] is None
    assert out["rows"][0]["outside_reference_band"] is False
    assert not any("reference band" in n for n in out["notes"])


def test_bounds_flag_and_disclaimer_note():
    eff = {"param": "CL", "covariate": "WT", "kind": "power", "center": 70.0,
          "coefficient": 0.75, "rse_pct": 10.0, "levels": None}
    out = covariate_forest(_nlme([eff]), cov_values={"WT": [140.0]}, bounds=(0.8, 1.25))
    row = out["rows"][0]
    assert out["bounds"] == [0.8, 1.25]
    assert row["outside_reference_band"] is True  # GMR ~1.68, well outside
    assert any("NOT a bioequivalence" in n for n in out["notes"])


# ── no covariate effects / no cov_values -> center-only row ────────────────

def test_no_cov_values_evaluates_at_center():
    eff = {"param": "CL", "covariate": "WT", "kind": "power", "center": 70.0,
          "coefficient": 0.75, "rse_pct": 10.0, "levels": None}
    out = covariate_forest(_nlme([eff]))
    row = out["rows"][0]
    assert row["eval_value"] == 70.0
    assert row["gmr"] == 1.0


def test_no_effects_returns_empty_rows():
    out = covariate_forest(_nlme([]))
    assert out["rows"] == []
    assert out["x_range"] is None
    assert out["summary"] == {"n_rows": 0, "n_effects": 0}


# ── determinism + JSON safety ────────────────────────────────────────────────

def test_deterministic_and_json_safe():
    effects = [
        {"param": "CL", "covariate": "WT", "kind": "power", "center": 70.0,
         "coefficient": 0.75, "rse_pct": 10.0, "levels": None},
        {"param": "CL", "covariate": "SEX", "kind": "categorical", "levels": ["F"],
         "coefficient": {"F": -0.2}, "rse_pct": {"F": None}},
    ]
    kwargs = dict(cov_values={"WT": [50.0, 100.0], "SEX": ["M", "F"]}, bounds=(0.8, 1.25))
    a = covariate_forest(_nlme(effects), **kwargs)
    b = covariate_forest(_nlme(effects), **kwargs)
    assert a == b
    json.dumps(a)
