"""NONMEM / mrgsolve control-stream export from a fitted NLME result."""
from __future__ import annotations

from app.compute.nmexport import build_mrgsolve, build_nonmem


def _fit(model_key="oral_1cmt", theta=None, iiv=None, cov=None):
    return {
        "model_key": model_key, "label": model_key,
        "theta": theta or {"CL": 5.0, "V": 50.0, "KA": 1.0},
        "omega_cv_pct": {"CL": 30.0, "V": 20.0},
        "sigma": {"prop": 0.1, "add": None},
        "iiv_params": iiv or ["CL", "V"], "error_model": "proportional",
        "covariate_effects": cov or [],
    }


def test_nonmem_oral_1cmt_advan2_trans2():
    ctl = build_nonmem(_fit())
    assert "$SUBROUTINE ADVAN2 TRANS2" in ctl
    assert "TVCL = THETA(1)" in ctl and "CL = TVCL * EXP(ETA(1))" in ctl
    assert "S2 = V" in ctl
    assert "$THETA" in ctl and "$OMEGA" in ctl and "$SIGMA" in ctl
    assert "$ESTIMATION METHOD=1 INTERACTION" in ctl
    # proportional sigma variance = 0.1**2
    assert "0.01" in ctl


def test_nonmem_covariate_written_as_power_theta():
    ctl = build_nonmem(_fit(cov=[{"param": "CL", "covariate": "CRCL",
                                  "kind": "power", "center": 100.0, "coefficient": 0.75}]))
    assert "(CRCL/100)**THETA(4)" in ctl
    assert "CRCL" in ctl.splitlines()[1]      # listed in $INPUT


def test_nonmem_iv_2cmt_advan3_trans4():
    ctl = build_nonmem(_fit("iv_2cmt", theta={"CL": 5.0, "VC": 30.0, "Q": 10.0, "VP": 80.0},
                            iiv=["CL"]))
    assert "$SUBROUTINE ADVAN3 TRANS4" in ctl
    assert "V1" in ctl and "V2" in ctl and "S1 = V1" in ctl


def test_mrgsolve_oral_1cmt_has_ode_and_blocks():
    cpp = build_mrgsolve(_fit())
    assert "$ODE" in cpp and "dxdt_DEPOT = -KA*DEPOT" in cpp
    assert "$PARAM" in cpp and "$OMEGA" in cpp and "$SIGMA" in cpp
    assert "double CL = TVCL" in cpp and "exp(ECL)" in cpp


def test_mrgsolve_lag_on_dosing_compartment():
    """oral_1cmt_lag doses into DEPOT, so the absorption lag must be ALAG_DEPOT
    (ALAG_CENT would attach it to a compartment that receives no dose)."""
    cpp = build_mrgsolve(_fit("oral_1cmt_lag",
                              theta={"CL": 5.0, "V": 50.0, "KA": 1.0, "ALAG": 0.3}))
    assert "ALAG_DEPOT = ALAG" in cpp
    assert "ALAG_CENT" not in cpp


def test_nonmem_lag_uses_alag1_on_depot():
    """NONMEM ADVAN2 depot is compartment 1 -> ALAG1 (unchanged, correct)."""
    ctl = build_nonmem(_fit("oral_1cmt_lag",
                            theta={"CL": 5.0, "V": 50.0, "KA": 1.0, "ALAG": 0.3}))
    assert "ALAG1 = THETA(4)" in ctl


def test_unsupported_model_returns_none():
    assert build_nonmem(_fit("oral_1cmt_transit")) is None
    assert build_mrgsolve(_fit("iv_1cmt_mm")) is None
