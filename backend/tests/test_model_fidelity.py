"""Downstream paths must reproduce the FITTED model (review findings F-07/17/19/20).

The fitted model carries covariate effects, a (possibly correlated) Omega block,
BLQ flags and fixed WT allometry. VPC / diagnostics / clinical simulation /
CWRES / the NONMEM+mrgsolve exports used to ignore one or more of these.
"""
from __future__ import annotations

import math

import numpy as np

from app.compute.clinsim import clinical_trial_simulation
from app.compute.diagnostics import npde
from app.compute.nlme import posthoc_residuals
from app.compute.nmexport import build_mrgsolve, build_nonmem
from app.compute.pk_models import get_model
from app.compute.pk_simulate import scale_params, simulate
from app.compute.vpc import (
    exposure_predictive_check,
    make_param_sampler,
    obs_vs_pred,
    pcvpc,
    stratified_vpc,
)

OBS_T = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 48.0, 72.0]
LN2 = math.log(2.0)
SEX_CL = [{"param": "CL", "covariate": "SEX", "kind": "categorical",
           "levels": ["1"], "coefficient": {"1": LN2}, "center": None}]


def _subjects(n_per=8, blq_idx=(), dv_blq=1000.0):
    model = get_model("oral_1cmt")
    subs = []
    for sex in (0, 1):
        for i in range(n_per):
            cl = 5.0 * (2.0 if sex else 1.0)
            cp = simulate(model, {"CL": cl, "V": 50.0, "KA": 1.0},
                          [{"time": 0.0, "amt": 100.0}], OBS_T, wt=70.0)["cp"]
            c = [float(v) for v in cp]
            blq = [False] * len(c)
            for k in blq_idx:
                c[k], blq[k] = dv_blq, True
            subs.append({"subject": f"S{sex}_{i}", "doses": [{"time": 0.0, "amt": 100.0}],
                         "obs_t": list(OBS_T), "obs_c": c, "wt": 70.0,
                         "cov": {"SEX": sex}, "obs_blq": blq})
    return subs


TYP = {"CL": 5.0, "V": 50.0, "KA": 1.0}


# ───────────────────────── F-07: covariates / Omega / BLQ ────────────────────

def test_stratified_vpc_reflects_fitted_categorical_effect():
    subs = _subjects()
    kw = dict(stratify_by="SEX", correction="none", sigma_prop=0.0, n_sim=20, n_bins=4)
    flat = stratified_vpc("oral_1cmt", subs, TYP, {}, **kw)
    fit = stratified_vpc("oral_1cmt", subs, TYP, {}, covariate_effects=SEX_CL, **kw)

    def med(res, label):
        s = next(s for s in res["strata"] if s["label"] == label)
        return s["bins"][-1]["sim_p50"]

    assert math.isclose(med(flat, "0"), med(flat, "1"), rel_tol=1e-9)   # the bug
    assert med(fit, "0") / med(fit, "1") > 1.9                          # CL x2 -> conc >= 2x lower


def test_exposure_check_reflects_fitted_categorical_effect():
    subs = _subjects()
    res = exposure_predictive_check("oral_1cmt", subs, TYP, {}, group_by="SEX",
                                    covariate_effects=SEX_CL, n_sim=10)
    auc = {g["label"]: g["auc"]["sim_median"] for g in res["groups"]}
    assert 1.8 < auc["0"] / auc["1"] < 2.2


def test_obs_vs_pred_pred_includes_covariate_effect():
    subs = _subjects(n_per=2)
    ind = {s["subject"]: dict(TYP) for s in subs}
    plain = obs_vs_pred("oral_1cmt", subs, ind, TYP)
    cov = obs_vs_pred("oral_1cmt", subs, ind, TYP, covariate_effects=SEX_CL)
    assert plain["pred"] != cov["pred"]


def test_npde_accepts_covariates_and_omega():
    subs = _subjects(n_per=2)
    om = [[0.09, 0.0], [0.0, 0.04]]
    out = npde("oral_1cmt", subs, TYP, {"CL": 30.0, "V": 20.0}, sigma_prop=0.1, sigma_add=0.0,
               n_sim=50, covariate_effects=SEX_CL, omega_matrix=om, omega_params=["CL", "V"])
    assert out["summary"]["n"] > 0


def test_correlated_omega_changes_half_life_variance():
    model = get_model("iv_1cmt")
    typ = {"CL": 5.0, "V": 50.0}
    cv = {"CL": 53.0, "V": 53.0}          # marginal var ~ 0.25 each

    def half_life_sd(cov12):
        om = [[0.25, cov12], [cov12, 0.25]]
        draw = make_param_sampler(model, typ, cv, omega_matrix=om, omega_params=["CL", "V"])
        rng = np.random.default_rng(1)
        hl = [math.log(2) * p["V"] / p["CL"] for p in (draw(rng, {}) for _ in range(4000))]
        return float(np.std(np.log(hl)))

    assert half_life_sd(0.249) < 0.2 * half_life_sd(0.0)


def test_sampler_unchanged_without_covariates_or_omega():
    model = get_model("oral_1cmt")
    cv = {"CL": 30.0, "V": 20.0}
    draw = make_param_sampler(model, TYP, cv)
    r1, r2 = np.random.default_rng(7), np.random.default_rng(7)
    got = draw(r1, {})
    sd = {p: (math.sqrt(math.log1p((cv[p] / 100) ** 2)) if p in cv else 0.0) for p in model.params}
    ref = {p: TYP[p] * math.exp(r2.normal(0.0, sd[p]) if sd[p] > 0 else 0.0) for p in model.params}
    assert got == ref


def test_blq_rows_excluded_from_observed_side():
    subs = _subjects(blq_idx=(2,))          # huge DV flagged BLQ at 2 h
    pc = pcvpc("oral_1cmt", subs, TYP, {}, sigma_prop=0.1, n_sim=10, n_bins=4)
    assert max(b["obs_p95"] for b in pc["bins"] if b["obs_p95"] is not None) < 100.0
    ovp = obs_vs_pred("oral_1cmt", subs, {s["subject"]: dict(TYP) for s in subs}, TYP)
    assert max(ovp["observed"]) < 100.0
    ex = exposure_predictive_check("oral_1cmt", subs, TYP, {}, group_by="SEX", n_sim=5)
    assert all(g["auc"]["observed"] < 1000.0 for g in ex["groups"])


# ───────────────────────── F-19: clinical simulation ─────────────────────────

def test_clinsim_uses_correlated_omega():
    def spread(cov12):
        om = [[0.25, cov12], [cov12, 0.25]]
        res = clinical_trial_simulation(
            "iv_1cmt", theta={"CL": 5.0, "V": 50.0}, omega_cv_pct={"CL": 53.0, "V": 53.0},
            iiv_params=["CL", "V"], doses=[100.0], tau=48.0, n_doses=1, metric="ctrough",
            n_subjects=400, omega_matrix=om)
        d = res["doses"][0]
        return math.log(d["metric_p95"] / d["metric_p05"])

    assert spread(0.249) < 0.5 * spread(0.0)


# ───────────────────────── F-20: CWRES with block Omega ──────────────────────

def _cwres_inputs():
    subs = [{"subject": f"P{i}", "doses": [{"time": 0.0, "amt": 100.0}],
             "obs_t": [1.0, 4.0, 12.0, 24.0],
             "obs_c": [1.9 + 0.1 * i, 1.4, 0.8 + 0.05 * i, 0.3], "wt": 70.0}
            for i in range(3)]
    etas = {f"P{i}": {"CL": 0.3 * (i - 1), "V": -0.2 * (i - 1)} for i in range(3)}
    kw = dict(theta={"CL": 5.0, "V": 50.0}, omega2={"CL": 0.25, "V": 0.25},
              sigma_prop=0.1, sigma_add=0.0, iiv_params=["CL", "V"], etas=etas)
    return subs, kw


def test_cwres_uses_full_omega_block():
    subs, kw = _cwres_inputs()
    r0 = posthoc_residuals("iv_1cmt", subs, omega_matrix=[[0.25, 0.0], [0.0, 0.25]], **kw)
    r8 = posthoc_residuals("iv_1cmt", subs, omega_matrix=[[0.25, 0.2], [0.2, 0.25]], **kw)
    assert np.max(np.abs(np.array(r0["cwres"]) - np.array(r8["cwres"]))) > 1e-3


def test_cwres_diagonal_omega_matrix_matches_legacy():
    subs, kw = _cwres_inputs()
    legacy = posthoc_residuals("iv_1cmt", subs, **kw)
    diag = posthoc_residuals("iv_1cmt", subs, omega_matrix=[[0.25, 0.0], [0.0, 0.25]], **kw)
    np.testing.assert_allclose(legacy["cwres"], diag["cwres"], atol=1e-5)


# ───────────────────────── F-17: exports carry WT allometry ──────────────────

def _fit(model_key="iv_1cmt", theta=None):
    return {"model_key": model_key, "label": model_key,
            "theta": theta or {"CL": 5.0, "V": 50.0},
            "omega_cv_pct": {"CL": 30.0, "V": 20.0}, "sigma": {"prop": 0.1, "add": None},
            "iiv_params": ["CL", "V"], "error_model": "proportional", "covariate_effects": []}


def test_nonmem_export_has_weight_allometry():
    ctl = build_nonmem(_fit())
    assert "(WT/70)**0.75" in ctl and "(WT/70)**1" in ctl
    assert "WT" in next(ln for ln in ctl.splitlines() if ln.startswith("$INPUT")).split()


def test_mrgsolve_export_has_weight_allometry():
    cpp = build_mrgsolve(_fit())
    assert "pow(WT/70, 0.75)" in cpp and "pow(WT/70, 1)" in cpp
    assert "WT=70" in next(ln for ln in cpp.splitlines() if ln.startswith("$PARAM"))


def test_exported_allometry_matches_simulator():
    model = get_model("iv_1cmt")
    ctl = build_nonmem(_fit())
    for wt in (10.0, 70.0, 120.0):
        ref = scale_params(model, {"CL": 5.0, "V": 50.0}, wt)
        for sym in ("CL", "V"):
            line = next(ln for ln in ctl.splitlines() if ln.strip().startswith(f"TV{sym} = TV{sym} *"))
            rhs = line.split("*", 1)[1].strip()
            val = {"CL": 5.0, "V": 50.0}[sym] * eval(rhs, {}, {"WT": wt})   # noqa: S307 - own export
            assert math.isclose(val, ref[sym], rel_tol=1e-9)


def test_export_honours_fitted_wt_exponents():
    nl = {**_fit(), "wt_exponents": {"CL": 0.663, "V": 1.087}}
    assert "(WT/70)**0.663" in build_nonmem(nl)
    assert "pow(WT/70, 1.087)" in build_mrgsolve(nl)
