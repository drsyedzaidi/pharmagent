"""NCA compute validated against an analytic mono-exponential profile.

For C(t) = C0*exp(-ke*t): AUCinf = C0/ke, lambda_z = ke, t1/2 = ln2/ke.
The log-down trapezoid is exact for exponential decline, so results should
match analytic values to tight tolerance.
"""
import math

import numpy as np

from app.compute.nca import Profile, nca_subject, run_nca

C0, KE, DOSE = 100.0, 0.2, 1000.0
TIMES = np.array([0, 0.5, 1, 2, 4, 6, 8, 12, 18, 24], dtype=float)


def _mono_profile():
    conc = C0 * np.exp(-KE * TIMES)
    return Profile(subject="S1", time=TIMES, conc=conc, dose=DOSE)


def test_lambda_z_and_thalf_exact():
    r = nca_subject(_mono_profile())
    assert abs(r["lambda_z"] - KE) < 1e-6
    assert abs(r["t_half"] - math.log(2) / KE) < 1e-4
    assert r["lambda_z_r2_adj"] > 0.999


def test_aucinf_matches_analytic():
    r = nca_subject(_mono_profile())
    assert math.isclose(r["AUC_inf"], C0 / KE, rel_tol=1e-3)  # 500


def test_cmax_tmax():
    r = nca_subject(_mono_profile())
    assert r["Cmax"] == 100.0
    assert r["Tmax"] == 0.0


def test_clearance_and_volume():
    r = nca_subject(_mono_profile())
    assert math.isclose(r["CL_F"], DOSE / (C0 / KE), rel_tol=1e-3)   # 2.0
    assert math.isclose(r["Vz_F"], (DOSE / (C0 / KE)) / KE, rel_tol=1e-3)  # 10.0


def test_pct_extrap_small():
    r = nca_subject(_mono_profile())
    assert 0 < r["pct_AUC_extrap"] < 2.0


def test_run_nca_groups_by_dose():
    records = [{"ID": "S1", "TIME": t, "DV": C0 * math.exp(-KE * t)} for t in TIMES if t > 0]
    out = run_nca(records, id_col="ID", time_col="TIME", dv_col="DV",
                  dose_by_subject={"S1": DOSE})
    assert out["nca_summary"]["n_subjects"] == 1
    assert out["nca_summary"]["by_dose"][0]["dose"] == DOSE
