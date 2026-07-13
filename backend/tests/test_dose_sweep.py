"""Tests for app.compute.dose_sweep — analytic PK-property validation.

These tests check the simulator-backed exposure sweep against KNOWN
pharmacokinetic properties, not merely that it runs:

  - LINEAR PK (1-cmt oral): the ODE is linear in the dose, so doubling the
    dose exactly doubles cmax and auc_tau (dose proportionality).
  - MICHAELIS-MENTEN elimination (1-cmt IV MM): elimination saturates as
    concentration rises, so dose-normalized exposure (auc_tau / dose) must
    INCREASE with dose (more-than-dose-proportional kinetics).

Plus structural invariants: finite metrics, profile count, equal-length grids.
"""
from __future__ import annotations

import math

import pytest

from app.compute.dose_sweep import dose_sweep

# ─────────────────────────── LINEAR PK ──────────────────────────────────────

def test_linear_pk_is_dose_proportional():
    # oral_1cmt is a linear ODE system -> exposure scales exactly with dose.
    doses = [50.0, 100.0]
    out = dose_sweep(
        "oral_1cmt",
        {"CL": 5.0, "V": 50.0, "KA": 1.0},
        doses,
        tau=24.0,
        n_doses=1,
        tmax=24.0,
        n_points=200,
    )

    assert out["model_key"] == "oral_1cmt"
    assert out["label"] == "1-cmt oral (linear)"
    assert len(out["profiles"]) == 2

    low, high = out["profiles"]
    assert low["dose"] == 50.0
    assert high["dose"] == 100.0

    # Doubling dose doubles cmax and auc_tau (ratio ~2.0 within 1e-3).
    cmax_ratio = high["cmax"] / low["cmax"]
    auc_ratio = high["auc_tau"] / low["auc_tau"]
    assert cmax_ratio == pytest.approx(2.0, abs=1e-3)
    assert auc_ratio == pytest.approx(2.0, abs=1e-3)


def test_linear_pk_triple_dose_triples_exposure():
    doses = [10.0, 30.0]
    out = dose_sweep(
        "oral_1cmt",
        {"CL": 5.0, "V": 50.0, "KA": 1.0},
        doses,
        tau=24.0,
        n_doses=1,
        tmax=24.0,
        n_points=200,
    )
    low, high = out["profiles"]
    assert high["auc_tau"] / low["auc_tau"] == pytest.approx(3.0, abs=1e-3)
    assert high["cmax"] / low["cmax"] == pytest.approx(3.0, abs=1e-3)
    # cavg is auc_tau / tau, so it scales identically.
    assert high["cavg"] / low["cavg"] == pytest.approx(3.0, abs=1e-3)


# ─────────────────────── MICHAELIS-MENTEN PK ────────────────────────────────

def test_michaelis_menten_more_than_dose_proportional():
    # Saturable elimination -> dose-normalized auc_tau increases with dose.
    doses = [50.0, 200.0, 800.0]
    out = dose_sweep(
        "iv_1cmt_mm",
        {"VMAX": 100.0, "KM": 5.0, "V": 50.0},
        doses,
        tau=24.0,
        n_doses=1,
        tmax=24.0,
        n_points=200,
    )
    assert out["model_key"] == "iv_1cmt_mm"
    assert len(out["profiles"]) == 3

    norm = [p["auc_tau"] / p["dose"] for p in out["profiles"]]
    # strictly increasing dose-normalized exposure
    assert norm[1] > norm[0]
    assert norm[2] > norm[1]


def test_michaelis_menten_super_proportional_auc_ratio():
    # AUC ratio should exceed the dose ratio for saturable elimination.
    doses = [50.0, 500.0]
    out = dose_sweep(
        "iv_1cmt_mm",
        {"VMAX": 100.0, "KM": 5.0, "V": 50.0},
        doses,
        tau=24.0,
        n_doses=1,
        tmax=24.0,
        n_points=200,
    )
    low, high = out["profiles"]
    dose_ratio = high["dose"] / low["dose"]  # 10.0
    auc_ratio = high["auc_tau"] / low["auc_tau"]
    assert auc_ratio > dose_ratio


# ─────────────────────────── INVARIANTS ─────────────────────────────────────

def test_metrics_finite_and_shapes_consistent():
    doses = [25.0, 50.0, 100.0]
    out = dose_sweep(
        "oral_1cmt",
        {"CL": 5.0, "V": 50.0, "KA": 1.0},
        doses,
        tau=12.0,
        n_doses=3,
        tmax=36.0,
        n_points=160,
    )
    assert len(out["profiles"]) == len(doses)
    for prof, d in zip(out["profiles"], doses):
        assert prof["dose"] == d
        assert len(prof["times"]) == len(prof["cp"])
        for key in ("cmax", "auc_tau", "cavg", "ctrough"):
            assert math.isfinite(prof[key])
        # exposure metrics are non-negative for a non-negative concentration
        assert prof["cmax"] >= 0.0
        assert prof["auc_tau"] >= 0.0


def test_return_envelope_keys_and_metadata():
    out = dose_sweep(
        "oral_1cmt",
        {"CL": 5.0, "V": 50.0, "KA": 1.0},
        [100.0],
        tau=24.0,
        n_doses=1,
        tmax=24.0,
    )
    assert set(out.keys()) == {
        "model_key", "label", "tau", "n_doses", "tmax", "profiles"
    }
    assert out["tau"] == 24.0
    assert out["n_doses"] == 1
    assert out["tmax"] == 24.0
    prof = out["profiles"][0]
    assert set(prof.keys()) == {
        "dose", "times", "cp", "cmax", "auc_tau", "cavg", "ctrough"
    }


def test_pkpd_profile_includes_eff():
    # PK/PD model -> each profile carries an 'eff' series aligned to times.
    out = dose_sweep(
        "pkpd_direct_emax",
        {"CL": 5.0, "V": 50.0, "KA": 1.0, "E0": 10.0, "EMAX": 100.0, "EC50": 2.0},
        [100.0, 200.0],
        tau=24.0,
        n_doses=1,
        tmax=24.0,
    )
    assert out["label"] == "Direct Emax PD"
    for prof in out["profiles"]:
        assert "eff" in prof
        assert len(prof["eff"]) == len(prof["times"])


def test_cavg_equals_auc_over_tau():
    out = dose_sweep(
        "oral_1cmt",
        {"CL": 5.0, "V": 50.0, "KA": 1.0},
        [100.0],
        tau=24.0,
        n_doses=1,
        tmax=24.0,
    )
    prof = out["profiles"][0]
    assert prof["cavg"] == pytest.approx(prof["auc_tau"] / 24.0, abs=1e-6)
