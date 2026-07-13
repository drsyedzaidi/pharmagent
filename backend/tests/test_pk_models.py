"""PK model library: simulator vs analytic/closed-form, fit recovery, scaling."""
import math

import numpy as np

from app.compute.compartmental import conc_1cmt_oral
from app.compute.pk_fit import compare_models, fit_pk_dataset, fit_subject_model, fit_subject_pkpd
from app.compute.pk_models import PK_KEYS, PKPD_KEYS, REGISTRY, get_model, list_models
from app.compute.pk_simulate import scale_params, simulate, simulate_timecourse

T = np.array([0.25, 0.5, 1, 2, 4, 6, 8, 12, 18, 24], dtype=float)
DOSE = 100.0


def test_registry_has_18_models():
    assert len(REGISTRY) == 18
    assert len(PK_KEYS) == 10 and len(PKPD_KEYS) == 8
    assert len(list_models()) == 18


def test_iv_1cmt_matches_analytic_monoexp():
    CL, V = 5.0, 50.0
    sim = simulate(get_model("iv_1cmt"), {"CL": CL, "V": V},
                   [{"time": 0, "amt": DOSE}], T, wt=70.0)
    analytic = (DOSE / V) * np.exp(-(CL / V) * T)
    assert np.allclose(sim["cp"], analytic, rtol=1e-4, atol=1e-6)


def test_oral_1cmt_matches_closed_form():
    CL, V, KA = 5.0, 50.0, 1.0
    sim = simulate(get_model("oral_1cmt"), {"CL": CL, "V": V, "KA": KA},
                   [{"time": 0, "amt": DOSE}], T, wt=70.0)
    closed = conc_1cmt_oral(T, DOSE, KA, CL, V)
    assert np.allclose(sim["cp"], closed, rtol=1e-3, atol=1e-5)


def test_allometric_scaling():
    m = get_model("oral_1cmt")
    p = scale_params(m, {"CL": 5.0, "V": 50.0, "KA": 1.0}, wt=35.0)
    assert math.isclose(p["CL"], 5.0 * (35.0 / 70.0) ** 0.75, rel_tol=1e-9)
    assert math.isclose(p["V"], 50.0 * (35.0 / 70.0) ** 1.0, rel_tol=1e-9)
    assert p["KA"] == 1.0  # KA not scaled


def test_fit_recovers_oral_1cmt():
    CL, V, KA = 4.0, 40.0, 1.2
    conc = simulate(get_model("oral_1cmt"), {"CL": CL, "V": V, "KA": KA},
                    [{"time": 0, "amt": DOSE}], T, wt=70.0)["cp"]
    f = fit_subject_model("oral_1cmt", [{"time": 0, "amt": DOSE}], T, conc, wt=70.0)
    assert f["converged"]
    assert math.isclose(f["params"]["CL"], CL, rel_tol=0.03)
    assert math.isclose(f["params"]["V"], V, rel_tol=0.05)


def test_fit_recovers_iv_2cmt():
    truth = {"CL": 5.0, "VC": 20.0, "Q": 8.0, "VP": 60.0}
    conc = simulate(get_model("iv_2cmt"), truth, [{"time": 0, "amt": DOSE}], T, wt=70.0)["cp"]
    f = fit_subject_model("iv_2cmt", [{"time": 0, "amt": DOSE}], T, conc, wt=70.0)
    assert f["converged"]
    assert math.isclose(f["params"]["CL"], truth["CL"], rel_tol=0.05)


def test_michaelis_menten_is_saturable():
    """MM elimination: doubling dose more-than-doubles exposure (AUC/dose rises)."""
    m = get_model("iv_1cmt_mm")
    p = {"VMAX": 50.0, "KM": 5.0, "V": 50.0}
    lo = simulate(m, p, [{"time": 0, "amt": 50.0}], T, wt=70.0)["cp"]
    hi = simulate(m, p, [{"time": 0, "amt": 200.0}], T, wt=70.0)["cp"]
    auc_lo = np.trapezoid(lo, T) / 50.0
    auc_hi = np.trapezoid(hi, T) / 200.0
    assert auc_hi > auc_lo * 1.1  # nonlinear: dose-normalized AUC increases


def test_transit_delays_peak_vs_oral_1cmt():
    base = simulate(get_model("oral_1cmt"), {"CL": 5, "V": 50, "KA": 1.0},
                    [{"time": 0, "amt": DOSE}], T, wt=70.0)["cp"]
    tr = simulate(get_model("oral_1cmt_transit"), {"CL": 5, "V": 50, "MTT": 2.0},
                  [{"time": 0, "amt": DOSE}], T, wt=70.0)["cp"]
    assert T[int(np.argmax(tr))] >= T[int(np.argmax(base))]


def test_pkpd_emax_effect_bounded():
    m = get_model("pkpd_direct_emax")
    out = simulate(m, dict(m.defaults), [{"time": 0, "amt": DOSE}], T, wt=70.0)
    assert "eff" in out
    e = out["eff"]
    assert np.all(e >= m.defaults["E0"] - 1e-6)
    assert np.all(e <= m.defaults["E0"] + m.defaults["EMAX"] + 1e-6)


def test_idr_starts_at_baseline():
    m = get_model("pkpd_idr1_inhib_kin")
    out = simulate(m, dict(m.defaults), [{"time": 0, "amt": DOSE}], np.array([0.0, 1, 4, 12]), wt=70.0)
    baseline = m.defaults["KIN"] / m.defaults["KOUT"]
    assert math.isclose(out["eff"][0], baseline, rel_tol=1e-3)


def test_pkpd_dual_endpoint_fit_recovers():
    """Fit a direct-Emax PK/PD model jointly to cp + effect; recover PD params."""
    m = get_model("pkpd_direct_emax")
    truth = {"CL": 5.0, "V": 50.0, "KA": 1.0, "E0": 10.0, "EMAX": 80.0, "EC50": 3.0}
    pk_t = np.array([0.5, 1, 2, 4, 8, 12, 24], dtype=float)
    pd_t = np.array([0.5, 1, 2, 4, 8, 12, 24], dtype=float)
    sim = simulate(m, truth, [{"time": 0, "amt": 200.0}], pk_t, wt=70.0)
    cp, eff = sim["cp"], sim["eff"]
    f = fit_subject_pkpd("pkpd_direct_emax", [{"time": 0, "amt": 200.0}],
                         pk_t, cp, pd_t, eff, wt=70.0)
    assert f["converged"]
    assert math.isclose(f["params"]["EC50"], truth["EC50"], rel_tol=0.10)
    assert math.isclose(f["params"]["EMAX"], truth["EMAX"], rel_tol=0.10)
    assert math.isclose(f["params"]["E0"], truth["E0"], rel_tol=0.10)
    assert f["n_pk_obs"] == 7 and f["n_pd_obs"] == 7


def test_pkpd_dataset_fit_via_fit_pk_dataset():
    m = get_model("pkpd_direct_emax")
    truth = {"CL": 5.0, "V": 50.0, "KA": 1.0, "E0": 10.0, "EMAX": 80.0, "EC50": 3.0}
    t = np.array([0.5, 1, 2, 4, 8, 12, 24], dtype=float)
    sim = simulate(m, truth, [{"time": 0, "amt": 200.0}], t, wt=70.0)
    subjects = [{"subject": "S1", "doses": [{"time": 0, "amt": 200.0}],
                 "obs_t": t, "obs_c": sim["cp"], "pd_t": t, "pd_e": sim["eff"], "wt": 70.0}]
    res = fit_pk_dataset(subjects, model_key="pkpd_direct_emax")
    assert res["n_converged"] == 1
    assert "EMAX" in res["population"]["parameters"]


def test_simulate_timecourse_single_and_multidose():
    m = get_model("oral_1cmt")
    p = {"CL": 5.0, "V": 50.0, "KA": 1.0}
    # single dose
    sd = simulate_timecourse(m, p, dose=100, tau=24, n_doses=1, tmax=48, n_points=100)
    assert len(sd["times"]) == len(sd["cp"]) and sd["times"][0] == 0.0
    assert sd["cp"][0] == 0.0 and max(sd["cp"]) > 0  # oral starts at 0, rises
    # multiple dose -> accumulation: trough before 2nd dose < trough before 5th dose
    md = simulate_timecourse(m, p, dose=100, tau=12, n_doses=6, tmax=84, n_points=400)
    times = md["times"]; cp = md["cp"]
    def trough_before(dose_time):
        idx = max(i for i, t in enumerate(times) if t < dose_time)
        return cp[idx]
    assert trough_before(48) > trough_before(12)   # steady-state accumulation


def test_simulate_timecourse_pkpd_returns_effect():
    m = get_model("pkpd_direct_emax")
    out = simulate_timecourse(m, dict(m.defaults), dose=500, tau=24, n_doses=1, tmax=24, n_points=80)
    assert "eff" in out and len(out["eff"]) == len(out["times"])
    assert min(out["eff"]) >= m.defaults["E0"] - 1e-6


def test_compare_models_ranks_by_aic():
    truth = {"CL": 4.0, "V": 40.0, "KA": 1.0}
    conc = simulate(get_model("oral_1cmt"), truth, [{"time": 0, "amt": DOSE}], T, wt=70.0)["cp"]
    subjects = [{"subject": "S1", "doses": [{"time": 0, "amt": DOSE}],
                 "obs_t": T, "obs_c": conc, "wt": 70.0}]
    cmp = compare_models(subjects, ["oral_1cmt", "oral_2cmt"])
    assert cmp["best_model"] in ("oral_1cmt", "oral_2cmt")
    assert cmp["ranking"][0]["total_aic"] <= (cmp["ranking"][1]["total_aic"] or 1e9)
