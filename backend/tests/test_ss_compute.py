"""Steady-state compute: closed-form limits, superposition, NCA, fitting, extraction."""
import math

import numpy as np

from app.compute.compartmental import (
    conc_1cmt_oral,
    conc_1cmt_oral_ss,
    conc_2cmt_oral,
    conc_2cmt_oral_ss,
    fit_one_subject,
)
from app.compute.dosing import SSProfile, extract_ss_intervals, is_multiple_dose
from app.compute.nca_ss import nca_ss_subject, run_nca_ss

DOSE, KA, CL, V, TAU = 5000.0, 1.0, 5.0, 50.0, 24.0
T = np.array([0.25, 0.5, 1, 2, 4, 6, 8, 12, 24], dtype=float)


def test_ss_reduces_to_single_dose_as_tau_large():
    """As tau -> inf there is no accumulation; SS curve == single-dose curve."""
    sd = conc_1cmt_oral(T, DOSE, KA, CL, V)
    ss = conc_1cmt_oral_ss(T, DOSE, KA, CL, V, tau=1e6)
    assert np.allclose(sd, ss, rtol=1e-9, atol=1e-9)


def test_ss_1cmt_matches_superposition():
    """SS conc must equal the sum of many prior single doses (superposition)."""
    ss = conc_1cmt_oral_ss(T, DOSE, KA, CL, V, tau=TAU)
    # superpose 200 doses given every tau; evaluate at time-after-last-dose T
    n = 200
    sup = np.zeros_like(T)
    for k in range(n):
        sup += conc_1cmt_oral(T + k * TAU, DOSE, KA, CL, V)
    assert np.allclose(ss, sup, rtol=1e-6, atol=1e-6)


def test_ss_2cmt_matches_superposition():
    Q, V2 = 8.0, 80.0
    ss = conc_2cmt_oral_ss(T, DOSE, KA, CL, V, Q, V2, tau=TAU)
    n = 400
    sup = np.zeros_like(T)
    for k in range(n):
        sup += conc_2cmt_oral(T + k * TAU, DOSE, KA, CL, V, Q, V2)
    assert np.allclose(ss, sup, rtol=1e-5, atol=1e-5)


def test_ss_2cmt_reduces_to_single_dose():
    Q, V2 = 8.0, 80.0
    sd = conc_2cmt_oral(T, DOSE, KA, CL, V, Q, V2)
    ss = conc_2cmt_oral_ss(T, DOSE, KA, CL, V, Q, V2, tau=1e6)
    assert np.allclose(sd, ss, rtol=1e-9, atol=1e-9)


def test_ss_fit_recovers_parameters():
    """Fit the SS 1-cmt model to a simulated SS profile -> recover CL, V."""
    conc = conc_1cmt_oral_ss(T, DOSE, KA, CL, V, tau=TAU)
    r = fit_one_subject(T, conc, DOSE, model="1cmt_ss", tau=TAU)
    assert r["converged"] is True
    assert math.isclose(r["params"]["CL"], CL, rel_tol=0.02)
    assert math.isclose(r["params"]["V"], V, rel_tol=0.05)


def test_ss_nca_clf_equals_dose_over_auctau():
    conc = conc_1cmt_oral_ss(T, DOSE, KA, CL, V, tau=TAU)
    r = nca_ss_subject("S1", list(T), list(conc), DOSE, TAU)
    # CL_F == Dose/AUC_tau (allow 6-dp rounding of both reported values)
    assert math.isclose(r["CL_F"], DOSE / r["AUC_tau"], rel_tol=1e-4)
    # CL/F from a true 1-cmt SS profile should recover the true CL closely
    assert math.isclose(r["CL_F"], CL, rel_tol=0.05)
    assert r["steady_state"] is True
    assert r["accumulation_ratio"] is not None and r["accumulation_ratio"] > 1.0


def test_extract_multiple_dose():
    # one subject: dose at 0 with ADDL=6, II=24 -> doses 0..144, plus dose at 168
    recs = []
    recs.append({"ID": "A", "T": 0, "DV": ".", "AMT": 100, "II": 24, "ADDL": 6})
    recs.append({"ID": "A", "T": 168, "DV": ".", "AMT": 100, "II": 24, "ADDL": 0})
    for t in [168.5, 169, 170, 172, 176, 180, 192]:
        recs.append({"ID": "A", "T": t, "DV": 10.0, "AMT": ".", "II": ".", "ADDL": "."})
    assert is_multiple_dose(recs, time_col="T", amt_col="AMT", ii_col="II",
                            addl_col="ADDL", id_col="ID") is True
    ss = extract_ss_intervals(recs, id_col="ID", time_col="T", dv_col="DV",
                              amt_col="AMT", ii_col="II", addl_col="ADDL")
    p = ss["A"]
    assert p.tau == 24.0 and p.dose == 100.0 and p.n_doses == 8
    assert p.tad[0] == 0.0  # trough anchored (no t=0 sample)
    assert max(p.tad) <= 24.0 + 1e-6


def test_run_nca_ss_groups_by_dose():
    profiles = {}
    for sid, dose in [("A", 5000.0), ("B", 2500.0)]:
        conc = conc_1cmt_oral_ss(T, dose, KA, CL, V, tau=TAU)
        profiles[sid] = SSProfile(subject=sid, tad=tuple(T), conc=tuple(conc),
                                  dose=dose, tau=TAU, n_doses=8, c0_source="measured")
    out = run_nca_ss(profiles)
    assert out["steady_state"] is True
    assert out["nca_summary"]["n_subjects"] == 2
    assert len(out["nca_summary"]["by_dose"]) == 2
