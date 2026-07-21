"""Modeler Agent tools: fit / compare the structural PK model library.

Builds per-subject dose schedules and observations from the loaded dataset
(single- or multiple-dose), then fits a chosen structural model — or compares
several and selects by AIC — with a two-stage population summary.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from app.compute.diagnostics import fit_residuals, npde
from app.compute.dose_sweep import dose_sweep
from app.compute.dosing import dose_events
from app.compute.pk_fit import compare_models, fit_pk_dataset
from app.compute.pk_models import PK_KEYS, REGISTRY, get_model, list_models
from app.compute.pk_simulate import simulate_timecourse
from app.compute.vpc import obs_vs_pred, pcvpc, vpc_band
from app.core.pharmstate import PharmState
from app.core.schema_extractor import detect_roles
from app.tools.base import Tool, ToolContext, ToolResult

_WT_NAMES = {"wt", "weight", "bw", "bwt", "bodyweight"}
# default candidate set for "compare" on oral data
_DEFAULT_ORAL = ["oral_1cmt", "oral_1cmt_lag", "oral_2cmt", "oral_1cmt_transit"]
_DEFAULT_IV = ["iv_1cmt", "iv_2cmt", "iv_3cmt"]


def _roles(df: pd.DataFrame, state: PharmState) -> dict[str, str]:
    if state.dataset_metadata and state.dataset_metadata.get("detected_roles"):
        return state.dataset_metadata["detected_roles"]
    return detect_roles(list(df.columns))


def _build_subjects(df: pd.DataFrame, roles: dict[str, str], *,
                    with_blq: bool = False) -> tuple[list[dict], bool, bool]:
    id_col = next((c for c, r in roles.items() if r == "ID"), None)
    time_col = next((c for c, r in roles.items() if r == "TIME"), None)
    dv_col = next((c for c, r in roles.items() if r == "DV"), None)
    amt_col = next((c for c, r in roles.items() if r == "AMT"), None)
    ii_col = next((c for c, r in roles.items() if r == "II"), None)
    addl_col = next((c for c, r in roles.items() if r == "ADDL"), None)
    dvid_col = next((c for c, r in roles.items() if r == "DVID"), None)
    pd_col = next((c for c, r in roles.items() if r == "PD"), None)
    # BLQ (M3) only when requested AND a CENS column exists, and only on the
    # single-endpoint path — keeps every other caller's behaviour unchanged.
    cens_col = next((c for c, r in roles.items() if r == "CENS"), None)
    do_blq = bool(with_blq and cens_col and not dvid_col)
    if not (id_col and time_col and dv_col and amt_col):
        raise ValueError("dataset needs ID/TIME/DV/AMT roles for PK model fitting")

    used = {c for c, r in roles.items()
            if r in {"ID", "TIME", "DV", "AMT", "EVID", "MDV", "CMT", "II", "ADDL", "DVID", "PD", "CENS"}}
    wt_col = next((c for c in df.columns if c not in used and c.strip().lower() in _WT_NAMES), None)
    # Candidate covariate columns: anything not a PK structural role (includes WT,
    # AGE, SEX, CRCL, ...). Baseline (first non-null) value per subject is taken.
    cov_cols = [c for c in df.columns if c not in used and c != id_col]

    dft = df.copy()
    dft[time_col] = pd.to_numeric(dft[time_col], errors="coerce")
    dft[dv_col] = pd.to_numeric(dft[dv_col], errors="coerce")
    if pd_col:
        dft[pd_col] = pd.to_numeric(dft[pd_col], errors="coerce")
    if dvid_col:
        dft[dvid_col] = pd.to_numeric(dft[dvid_col], errors="coerce")

    has_pd = bool(pd_col) or bool(dvid_col)
    subjects: list[dict] = []
    multi = False
    for sid, g in dft.groupby(id_col):
        rows = g.to_dict("records")
        doses = dose_events(rows, time_col=time_col, amt_col=amt_col,
                            ii_col=ii_col, addl_col=addl_col)
        if len(doses) > 1:
            multi = True
        # PK observations: positive concentrations at t>0. Oral first-order
        # absorption models give C(0)=0, so a measurable pre-/at-dose (t<=0)
        # sample is structurally unfittable and would distort the fit (inflating
        # residual error / collapsing IIV) — exclude it, matching the
        # compartmental fitter.
        if dvid_col:
            pk = g[g[dvid_col] == 1].dropna(subset=[time_col, dv_col])
            pk = pk[(pk[dv_col] > 0) & (pk[time_col] > 0)]
            pdo = g[g[dvid_col] == 2].dropna(subset=[time_col, dv_col])
        elif do_blq:
            pk = g.dropna(subset=[time_col, dv_col])
            cens = pd.to_numeric(pk[cens_col], errors="coerce").fillna(0) == 1
            pk = pk[(pk[time_col] > 0) & ((pk[dv_col] > 0) | cens)]
            pdo = g.iloc[0:0]
        else:
            pk = g.dropna(subset=[time_col, dv_col])
            pk = pk[(pk[dv_col] > 0) & (pk[time_col] > 0)]
            pdo = g.dropna(subset=[time_col, pd_col]) if pd_col else g.iloc[0:0]
        if pk.empty or not doses:
            continue
        wt = 70.0
        if wt_col is not None:
            w = pd.to_numeric(g[wt_col], errors="coerce").dropna()
            if len(w):
                wt = float(w.iloc[0])
        cov: dict[str, Any] = {}
        for c in cov_cols:
            s = g[c].dropna()
            if not len(s):
                continue
            v0 = s.iloc[0]
            num = pd.to_numeric(pd.Series([v0]), errors="coerce").iloc[0]
            cov[c] = float(num) if pd.notna(num) else str(v0)
        subj = {"subject": sid, "doses": doses,
                "obs_t": pk[time_col].to_numpy(float),
                "obs_c": pk[dv_col].to_numpy(float), "wt": wt, "cov": cov}
        if do_blq:
            blq = (pd.to_numeric(pk[cens_col], errors="coerce").fillna(0) == 1).to_numpy(bool)
            subj["obs_blq"] = blq.tolist()
            blq_dv = pk[dv_col].to_numpy(float)[blq]   # NONMEM: BLQ rows carry LLOQ in DV
            subj["lloq"] = float(np.median(blq_dv)) if blq_dv.size else None
        if has_pd and len(pdo):
            pd_dv = dv_col if dvid_col else pd_col
            subj["pd_t"] = pdo[time_col].to_numpy(float)
            subj["pd_e"] = pdo[pd_dv].to_numpy(float)
        subjects.append(subj)
    return subjects, multi, has_pd


def _trim_fit(res: dict[str, Any]) -> dict[str, Any]:
    """Compact, audit-safe view: drop nothing sensitive (params only already)."""
    return {
        "model_key": res["model_key"], "label": res["label"], "group": res["group"],
        "n_subjects": res["n_subjects"], "n_converged": res["n_converged"],
        "mean_aic": res["mean_aic"], "total_aic": res["total_aic"],
        "population": res["population"],
        "individual_fits": [{"subject": f["subject"], "converged": f["converged"],
                             "params": f.get("params"), "aic": f.get("aic"),
                             "r_squared": f.get("r_squared")}
                            for f in res["individual_fits"]],
    }


def fit_pk_model(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    dsid = args.get("dataset_id") or state.dataset_id
    df = ctx.dataset_store[dsid]
    roles = _roles(df, state)
    subjects, multi, has_pd = _build_subjects(df, roles)
    if not subjects:
        status = {"status": "no_subjects", "message": "No fittable subjects (need dose + concentration rows)."}
        return ToolResult(summary="PK model fit skipped: no fittable subjects.",
                          action="fit_pk_model(no_subjects)",
                          writes={"pk_model_results": status}, result=status)

    model_key = args.get("model_key")
    candidates = args.get("models")

    if model_key and not args.get("compare"):
        if model_key not in REGISTRY:
            raise ValueError(f"unknown model: {model_key}")
        if get_model(model_key).has_pd and not has_pd:
            status = {"status": "pd_required", "model_key": model_key,
                      "message": ("PK/PD models need a PD endpoint — add a DVID column "
                                  "(1=concentration, 2=effect) or an 'effect'/'response' column.")}
            return ToolResult(summary=f"{model_key} needs a PD endpoint; not fit.",
                              action=f"fit_pk_model({model_key}, pd_required)",
                              writes={"pk_model_results": status}, result=status)
        res = fit_pk_dataset(subjects, model_key=model_key)
        is_pkpd = get_model(model_key).has_pd
        payload = {"status": "ok", "mode": "fit", "multiple_dose": multi,
                   "is_pkpd": is_pkpd, **_trim_fit(res)}
        kind = "PK/PD (dual-endpoint)" if is_pkpd else "PK"
        return ToolResult(
            summary=(f"Fit {res['label']} [{kind}]: {res['n_converged']}/{res['n_subjects']} "
                     f"subjects converged (mean AIC {res['mean_aic']})."),
            action=f"fit_pk_model({model_key})",
            writes={"pk_model_results": payload}, result=payload)

    # compare mode — drop PK/PD candidates unless a PD endpoint is present
    if not candidates:
        candidates = list(args.get("models") or _DEFAULT_ORAL)
    candidates = [k for k in candidates if k in REGISTRY and (has_pd or not get_model(k).has_pd)]
    cmp = compare_models(subjects, candidates)
    payload = {"status": "ok", "mode": "compare", "multiple_dose": multi,
               "ranking": cmp["ranking"], "best_model": cmp["best_model"],
               "best": _trim_fit(cmp["best"]) if cmp["best"] else None}
    best_label = payload["best"]["label"] if payload["best"] else "none"
    return ToolResult(
        summary=(f"Compared {len(candidates)} PK models; best by AIC: {best_label}."),
        action=f"fit_pk_model(compare:{','.join(candidates)})",
        writes={"pk_model_results": payload}, result=payload)


def _fitted_population(state: PharmState) -> tuple[str | None, dict[str, float]]:
    """Recover (model_key, typical params) from the most recent fit, if any."""
    pm = state.pk_model_results or {}
    if pm.get("status") != "ok":
        return None, {}
    if pm.get("mode") == "fit":
        key = pm.get("model_key")
        pop = pm.get("population", {})
    else:  # compare -> use the best model
        best = pm.get("best") or {}
        key = best.get("model_key")
        pop = best.get("population", {})
    params = {k: v.get("typical_value") for k, v in (pop.get("parameters") or {}).items()
              if v.get("typical_value") is not None}
    return key, params


def simulate_pk_profile(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    fitted_key, fitted_params = _fitted_population(state)
    model_key = args.get("model_key") or fitted_key
    if not model_key or model_key not in REGISTRY:
        status = {"status": "no_model", "message": "Fit a PK model first, or pass a model_key to simulate."}
        return ToolResult(summary="Simulation skipped: no model selected.",
                          action="simulate_pk_profile(no_model)",
                          writes={"simulation_results": status}, result=status)
    model = get_model(model_key)

    # parameters: explicit override > fitted typical (same model) > model defaults
    params = dict(model.defaults)
    if model_key == fitted_key:
        params.update(fitted_params)
    if isinstance(args.get("params"), dict):
        params.update({k: float(v) for k, v in args["params"].items() if k in model.params})

    dose = float(args.get("dose", 100.0))
    tau = float(args.get("tau", 24.0))
    n_doses = int(args.get("n_doses", 1))
    wt = float(args.get("wt", 70.0))
    rate = float(args.get("rate", 0.0) or 0.0)
    tmax = float(args.get("tmax") or (max(tau * n_doses, 24.0) + tau))

    # Boundary validation: reject non-finite / non-physical inputs before building
    # dose lists and time grids (n_doses=1e9 -> OOM; inf/nan -> garbage output).
    for _name, _val in (("dose", dose), ("tau", tau), ("wt", wt), ("tmax", tmax)):
        if not math.isfinite(_val) or _val <= 0.0:
            raise ValueError(f"{_name} must be a finite positive number, got {_val!r}")
    if not math.isfinite(rate) or rate < 0.0:
        raise ValueError(f"rate must be a finite non-negative number, got {rate!r}")
    if not 1 <= n_doses <= 1000:
        raise ValueError(f"n_doses must be between 1 and 1000, got {n_doses}")

    tc = simulate_timecourse(model, params, dose=dose, tau=tau, n_doses=n_doses,
                             tmax=tmax, wt=wt, rate=rate)
    cmax = max(tc["cp"]) if tc["cp"] else None
    payload = {
        "status": "ok", "model_key": model_key, "label": model.label,
        "has_pd": model.has_pd, "from_fit": model_key == fitted_key and bool(fitted_params),
        "params": {k: round(float(params[k]), 6) for k in model.params},
        "regimen": {"dose": dose, "tau": tau, "n_doses": n_doses, "tmax": tmax, "wt": wt,
                    "rate": rate},
        "times": tc["times"], "cp": tc["cp"],
        **({"eff": tc["eff"]} if "eff" in tc else {}),
        "cmax": round(cmax, 4) if cmax is not None else None,
    }
    src = "fitted typical values" if payload["from_fit"] else "model defaults"
    return ToolResult(
        summary=(f"Simulated {model.label} — {n_doses}×{dose} q{tau}h to {tmax}h "
                 f"({src}); Cmax≈{payload['cmax']}."),
        action=f"simulate_pk_profile({model_key})",
        writes={"simulation_results": payload}, result={k: payload[k] for k in
                ("status", "model_key", "label", "regimen", "cmax", "params", "from_fit")})


def _last_fit(state: PharmState) -> tuple[str | None, list[dict], dict, dict]:
    """Return (model_key, individual_fits, population, typical_params) from the last fit."""
    pm = state.pk_model_results or {}
    if pm.get("status") != "ok":
        return None, [], {}, {}
    if pm.get("mode") == "fit":
        key, fits, pop = pm.get("model_key"), pm.get("individual_fits") or [], pm.get("population") or {}
    else:
        best = pm.get("best") or {}
        key, fits, pop = best.get("model_key"), best.get("individual_fits") or [], best.get("population") or {}
    typical = {k: v.get("typical_value") for k, v in (pop.get("parameters") or {}).items()
               if v.get("typical_value") is not None}
    return key, fits, pop, typical


def run_vpc(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    model_key, fits, pop, typical = _last_fit(state)
    if not model_key:
        status = {"status": "no_fit", "message": "Fit a PK model first to run a VPC / goodness-of-fit."}
        return ToolResult(summary="VPC skipped: no fitted model.", action="run_vpc(no_fit)",
                          writes={"vpc_results": status}, result=status)
    model = get_model(model_key)
    typical = {**model.defaults, **typical}
    df = ctx.dataset_store[state.dataset_id]
    subjects, _multi, _has_pd = _build_subjects(df, _roles(df, state))
    indiv = {f["subject"]: f["params"] for f in fits
             if f.get("converged") and f.get("params")}

    ovp = obs_vs_pred(model_key, subjects, indiv, typical)
    iiv = {k: v.get("iiv_cv_pct") for k, v in (pop.get("parameters") or {}).items()}

    # representative regimen for the VPC band: the most common dose group
    groups: dict[Any, list[dict]] = {}
    for s in subjects:
        d = s["doses"][0]["amt"] if s["doses"] else None
        groups.setdefault(d, []).append(s)
    band = {"times": [], "p05": [], "p50": [], "p95": []}
    vpc_dose, obs_t, obs_c = None, [], []
    if groups:
        vpc_dose, grp = max(groups.items(), key=lambda kv: len(kv[1]))
        rep = max(grp, key=lambda s: len(s["obs_t"]))
        tmax = float(max(rep["obs_t"])) if len(rep["obs_t"]) else 24.0
        band = vpc_band(model_key, typical, iiv, rep["doses"], tmax=tmax,
                        wt=float(rep.get("wt", 70.0)))
        for s in grp:
            obs_t += [float(t) for t in s["obs_t"]]
            obs_c += [float(c) for c in s["obs_c"]]

    # Prediction-corrected VPC over the whole dataset. Residual error: prefer a
    # fitted NLME sigma; otherwise use the log-scale GOF RMSE as a proportional
    # proxy so the simulated band reflects realistic scatter.
    nl = state.nlme_results if (state.nlme_results or {}).get("status") == "ok" else None
    sig = (nl or {}).get("sigma") or {}
    sigma_prop = float(sig.get("prop") or 0.0)
    sigma_add = float(sig.get("add") or 0.0)
    if sigma_prop <= 0.0 and sigma_add <= 0.0:
        sigma_prop = float(ovp["gof"].get("rmse_log_ipred") or 0.1) or 0.1
    pc = pcvpc(model_key, subjects, typical, iiv,
               sigma_prop=sigma_prop, sigma_add=sigma_add)

    payload = {"status": "ok", "model_key": model_key, "label": model.label,
               "gof": ovp["gof"], "obs_vs_pred": {k: ovp[k] for k in ("observed", "ipred", "pred")},
               "vpc": band, "vpc_dose": vpc_dose, "obs_t": obs_t, "obs_c": obs_c,
               "pcvpc": pc}
    g = ovp["gof"]
    return ToolResult(
        summary=(f"VPC / GOF for {model.label}: n={g['n']} obs, "
                 f"log-scale R²(IPRED)={g['r2_log_ipred']}; "
                 f"pcVPC over {pc.get('n_bins', 0)} time bins."),
        action=f"run_vpc({model_key})",
        writes={"vpc_results": payload},
        result={"status": "ok", "model_key": model_key, "gof": g, "vpc_dose": vpc_dose})


def run_nlme(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Population (mixed-effects) fit via FOCE-I or SAEM on a structural model."""
    from app.compute.nlme import population_fit  # lazy: heavy + optional dependency

    fitted_key, _f, _p, _t = _last_fit(state)
    model_key = args.get("model_key") or fitted_key
    if not model_key or model_key not in REGISTRY:
        status = {"status": "no_model",
                  "message": "Choose a structural model (fit one first, or pass model_key) for the NLME fit."}
        return ToolResult(summary="NLME skipped: no structural model selected.",
                          action="run_nlme(no_model)",
                          writes={"nlme_results": status}, result=status)
    if get_model(model_key).has_pd:
        status = {"status": "pd_unsupported", "message": "NLME currently supports PK models only."}
        return ToolResult(summary="NLME skipped: PK/PD models not supported.",
                          action="run_nlme(pd_unsupported)",
                          writes={"nlme_results": status}, result=status)

    df = ctx.dataset_store[state.dataset_id]
    subjects, _multi, _has_pd = _build_subjects(df, _roles(df, state), with_blq=True)
    if len(subjects) < 2:
        status = {"status": "insufficient", "message": "Need >=2 subjects for a population fit."}
        return ToolResult(summary="NLME skipped: too few subjects.", action="run_nlme(insufficient)",
                          writes={"nlme_results": status}, result=status)

    method = args.get("method", "focei")
    res = population_fit(model_key, subjects, method=method,
                         iiv_params=args.get("iiv_params"),
                         error_model=args.get("error_model", "proportional"),
                         covariate_model=args.get("covariate_model"))
    payload = {"status": "ok", **res}
    return ToolResult(
        summary=(f"{res['method']} fit of {res['label']}: OFV {res['ofv']}, "
                 f"{res['n_subjects']} subjects, IIV on {res['iiv_params']} "
                 f"({'converged' if res.get('converged') else 'did not converge'})."),
        action=f"run_nlme({method}, {model_key})",
        writes={"nlme_results": payload},
        result={"status": "ok", "method": res["method"], "theta": res["theta"],
                "omega_cv_pct": res["omega_cv_pct"], "sigma": res["sigma"], "ofv": res["ofv"]})


_MAX_AUTO_CANDIDATES = 6


def _covariate_candidates(subjects: list[dict], iiv_params: list[str]
                          ) -> list[dict[str, Any]]:
    """Auto-build SCM candidates: each usable covariate column x each IIV param.

    A covariate is continuous (power model) if numeric with >=3 distinct values,
    else categorical. Weight columns are skipped — the structural models already
    apply fixed allometric weight scaling, so testing WT again would double-count.
    Candidates are ordered param-major (all effects on the first IIV parameter —
    usually clearance — first) so that, under the count cap, the most relevant
    clearance covariates are screened before volume covariates.
    """
    cols: dict[str, set] = {}
    for s in subjects:
        for c, v in (s.get("cov") or {}).items():
            cols.setdefault(c, set()).add(v)
    cands: list[dict[str, Any]] = []
    for p in iiv_params:
        for c, vals in sorted(cols.items()):
            if c.strip().lower() in _WT_NAMES:
                continue
            nums = [v for v in vals if isinstance(v, (int, float))]
            is_cont = len(nums) == len(vals) and len(vals) >= 3
            if len(vals) < 2:                   # no variation -> not testable
                continue
            kind = "power" if is_cont else "categorical"
            cands.append({"param": p, "covariate": c, "kind": kind})
    return cands[:_MAX_AUTO_CANDIDATES]


def run_scm(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Stepwise covariate modeling (forward + backward) on a structural model."""
    from app.compute.nlme import scm  # lazy: heavy + optional dependency

    fitted_key, _f, _p, _t = _last_fit(state)
    model_key = args.get("model_key") or fitted_key
    if not model_key or model_key not in REGISTRY:
        status = {"status": "no_model",
                  "message": "Choose a structural model (fit one first, or pass model_key) before SCM."}
        return ToolResult(summary="SCM skipped: no structural model selected.",
                          action="run_scm(no_model)",
                          writes={"scm_results": status}, result=status)
    if get_model(model_key).has_pd:
        status = {"status": "pd_unsupported", "message": "SCM currently supports PK models only."}
        return ToolResult(summary="SCM skipped: PK/PD models not supported.",
                          action="run_scm(pd_unsupported)",
                          writes={"scm_results": status}, result=status)

    df = ctx.dataset_store[state.dataset_id]
    subjects, _multi, _has_pd = _build_subjects(df, _roles(df, state))
    if len(subjects) < 3:
        status = {"status": "insufficient", "message": "Need >=3 subjects for covariate modeling."}
        return ToolResult(summary="SCM skipped: too few subjects.", action="run_scm(insufficient)",
                          writes={"scm_results": status}, result=status)

    iiv = args.get("iiv_params") or ["CL", "V"]
    candidates = args.get("candidates") or _covariate_candidates(subjects, iiv)
    if not candidates:
        status = {"status": "no_covariates",
                  "message": ("No usable covariate columns found in the dataset "
                              "(need a varying non-PK column such as AGE, SEX, CRCL).")}
        return ToolResult(summary="SCM skipped: no covariate columns.", action="run_scm(no_covariates)",
                          writes={"scm_results": status}, result=status)

    res = scm(model_key, subjects, candidates=candidates, iiv_params=iiv,
              error_model=args.get("error_model", "proportional"),
              forward_p=float(args.get("forward_p", 0.05)),
              backward_p=float(args.get("backward_p", 0.01)),
              max_iter=int(args.get("max_iter", 12)))
    sel = ", ".join(f"{e['param']}~{e['covariate']}" for e in res.get("selected", [])) or "none"
    # Post-selection-inference caveat: stepwise search makes the retained effects'
    # SE/p-values optimistic (winner's curse). Attach it when anything was selected.
    caveat = ("Stepwise selection: the retained effects' standard errors and "
              "p-values are optimistic (post-selection inference); confirm on a "
              "pre-specified covariate set or validate by resampling."
              ) if res.get("selected") else None
    res["selection_caveat"] = caveat
    summary = (f"SCM on {res.get('label')}: tested {res.get('n_candidates')} candidate(s), "
               f"selected {sel}; OFV {res.get('base_ofv')} -> {res.get('final_ofv')}.")
    if caveat:
        summary += f" NOTE: {caveat}"
    return ToolResult(
        summary=summary,
        action=f"run_scm({model_key})",
        writes={"scm_results": res},
        result={"status": res.get("status"), "selected": res.get("selected"),
                "base_ofv": res.get("base_ofv"), "final_ofv": res.get("final_ofv"),
                "selection_caveat": caveat})


def forecast_map(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """MAP / empirical-Bayes forecast for a new patient from the fitted NLME model."""
    from app.compute.forecast import forecast  # lazy: heavy + optional dependency

    nl = state.nlme_results
    if not nl or nl.get("status") != "ok":
        status = {"status": "no_model",
                  "message": "Run a population (NLME) fit first — MAP forecasting needs it."}
        return ToolResult(summary="Forecast skipped: no NLME model.",
                          action="forecast_map(no_model)",
                          writes={"forecast_results": status}, result=status)

    res = forecast(
        nl, dose=float(args.get("dose", 100.0)), tau=float(args.get("tau", 24.0)),
        measured=args.get("measured") or [], wt=float(args.get("wt", 70.0)),
        cov=args.get("cov"), target=args.get("target"),
        target_metric=args.get("target_metric", "cmin"),
        tmax=args.get("tmax"))
    ind = res.get("individual_params", {})
    typ = res.get("typical_params", {})
    rec = res.get("recommendation")
    rec_str = ""
    if rec and rec.get("recommended_dose") is not None:
        rec_str = (f" Recommended dose for {rec['target_metric']}={rec['target']}: "
                   f"{rec['recommended_dose']}.")
    summ = (f"MAP forecast on {res.get('label')}: individual "
            + ", ".join(f"{k} {ind[k]}" for k in ind if k in typ)
            + " (vs typical " + ", ".join(f"{k} {typ[k]}" for k in typ) + ")."
            + rec_str) if res.get("status") == "ok" else res.get("message", "")
    return ToolResult(summary=summ, action="forecast_map",
                      writes={"forecast_results": res},
                      result={"status": res.get("status"),
                              "individual_params": ind,
                              "ss_individual": res.get("ss_individual"),
                              "recommendation": rec})


def run_diagnostics(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Residual diagnostics: legacy two-stage IWRES (always available once a
    structural model is fitted) plus a single-provenance CWRES/npd block built
    ENTIRELY from a converged NLME fit of the SAME structural model (theta,
    Omega, sigma, and stored EBEs all sourced from ``nlme_results`` — never
    mixed with the two-stage fit). CWRES/npd report ``{"status":
    "needs_nlme"}`` until ``run_nlme`` has converged; a figure combining rows
    from two different estimators is a reviewer-flaggable defect this
    single-provenance rule exists to prevent.
    """
    model_key, fits, pop, typical = _last_fit(state)
    if not model_key:
        status = {"status": "no_fit", "message": "Fit a PK model first to run residual diagnostics."}
        return ToolResult(summary="Diagnostics skipped: no fitted model.",
                          action="run_diagnostics(no_fit)",
                          writes={"diagnostics_results": status}, result=status)
    model = get_model(model_key)
    typical = {**model.defaults, **typical}
    df = ctx.dataset_store[state.dataset_id]
    # with_blq=True for parity with run_nlme: flags censored (BLQ) rows via
    # obs_blq/lloq so the residual diagnostics can drop them. Without it, BLQ
    # rows carry the LLOQ in DV, survive the c>0 mask, and are silently scored
    # as quantified observations (a structured positive-residual artefact).
    subjects, _multi, _has_pd = _build_subjects(df, _roles(df, state), with_blq=True)
    indiv = {f["subject"]: f["params"] for f in fits if f.get("converged") and f.get("params")}

    # Legacy two-stage IWRES (unweighted log residual; kept for the no-NLME
    # fallback and any other consumer of "residuals"). NOT part of the
    # single-provenance CWRES/npd block below.
    res = fit_residuals(model_key, subjects, indiv, typical)

    n_blq = sum(int(sum(1 for b in (s.get("obs_blq") or []) if b)) for s in subjects)
    nl = state.nlme_results if (state.nlme_results or {}).get("status") == "ok" else None
    if nl is not None and nl.get("model_key") != model_key:
        nl = None

    if nl is None:
        needs_nlme = {"status": "needs_nlme",
                      "message": (f"Needs a converged NLME fit of {model.label} (run_nlme) to "
                                  "supply theta/Omega/sigma; the two-stage fit alone cannot "
                                  "calibrate a single-provenance residual model.")}
        cwres_res: dict[str, Any] = dict(needs_nlme)
        npde_res: dict[str, Any] = dict(needs_nlme)
    else:
        from app.compute.nlme import cv_pct_to_omega2, posthoc_residuals  # lazy: heavy + optional dep

        theta_nl = nl.get("theta") or {}
        omega_cv = nl.get("omega_cv_pct") or {}
        omega2 = {p: cv_pct_to_omega2(cv) for p, cv in omega_cv.items()}
        sig = nl.get("sigma") or {}
        sigma_prop = float(sig.get("prop") or 0.0)
        sigma_add = float(sig.get("add") or 0.0)
        etas = {r["subject"]: r["eta"] for r in (nl.get("individual") or [])}
        interaction = bool(args.get("cwres_interaction", True))

        cwres_res = posthoc_residuals(
            model_key, subjects, theta=theta_nl, omega2=omega2,
            sigma_prop=sigma_prop, sigma_add=sigma_add,
            iiv_params=list(nl.get("iiv_params") or []),
            error_model=nl.get("error_model", "proportional"),
            covariate_effects=nl.get("covariate_effects"),
            etas=etas, interaction=interaction)

        if n_blq > 0:
            npde_res = {"status": "blq_unsupported", "n_blq": n_blq,
                        "message": ("Prediction-discrepancy diagnostics are not computed when the "
                                    "dataset has BLQ (censored) observations: widening the simulated "
                                    "cloud with residual error while BLQ rows are excluded on the "
                                    "observed side creates a spurious trend near the LLOQ.")}
        else:
            npde_res = npde(model_key, subjects, theta_nl, omega_cv,
                            sigma_prop=sigma_prop, sigma_add=sigma_add)

    payload = {"status": "ok", "model_key": model_key, "label": model.label,
               "residuals": res, "cwres": cwres_res, "npde": npde_res,
               "nlme_provenance": (nl.get("model_key") if nl else None)}
    npg = npde_res.get("summary") or {}
    cwg = cwres_res.get("summary") or {}
    cwres_bit = (f"CWRES mean={cwg.get('cwres_mean')}, sd={cwg.get('cwres_sd')}."
                if "cwres_mean" in cwg else f"CWRES unavailable ({cwres_res.get('status')}).")
    npd_bit = (f"npd mean={npg.get('mean')}, sd={npg.get('sd')} "
              f"({npg.get('pct_outside_1_96')}% outside ±1.96)."
              if "mean" in npg else f"npd unavailable ({npde_res.get('status')}).")
    return ToolResult(
        summary=(f"Residual diagnostics for {model.label}: two-stage IWRES n={res['summary']['n']}; "
                 f"{cwres_bit} {npd_bit}"),
        action=f"run_diagnostics({model_key})",
        writes={"diagnostics_results": payload},
        result={"status": "ok", "model_key": model_key,
                "npde_status": npde_res.get("status", "ok"), "npde_summary": npg,
                "cwres_status": cwres_res.get("status", "ok"), "cwres_summary": cwg,
                "iwres_summary": res["summary"]})


_MIN_N_FOR_COV_PERCENTILE = 5  # below this, a percentile is not trustworthy -> center-only row


def _num_or_none(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _cov_eval_points(subjects: list[dict], effects: list[dict], percentiles: list[float]
                     ) -> tuple[dict[str, list], dict[str, str], dict[str, dict]]:
    """Evaluation points + reference-level recovery for `covariate_forest`,
    from the CURRENT dataset's subjects (never from the fit result, which
    carries no raw covariate distribution).

    Categorical reference level mirrors `app.compute.nlme._build_cov_effects`
    exactly (`max(uniq, key=vals.count)`, over `str(subject.cov[covariate])`
    for subjects with a non-null value) — the fitted result never stores
    which level was chosen as reference, so this is the only way to recover
    a meaningful row label. Continuous percentiles are skipped (falling back
    to a center-only row inside `covariate_forest`) below a minimum sample
    size, since a percentile from a handful of subjects is not trustworthy.
    """
    cov_values: dict[str, list] = {}
    ref_levels: dict[str, str] = {}
    cov_stats: dict[str, dict] = {}
    for eff in effects:
        cov = eff["covariate"]
        if cov in cov_values:
            continue
        if eff["kind"] == "categorical":
            vals = [str(s.get("cov", {}).get(cov)) for s in subjects
                    if s.get("cov", {}).get(cov) is not None]
            if not vals:
                continue
            uniq = sorted(set(vals))
            ref_levels[cov] = max(uniq, key=vals.count)
            cov_values[cov] = uniq
            cov_stats[cov] = {"n_cov": len(vals), "levels": uniq}
        else:
            nums = [n for n in (_num_or_none(s.get("cov", {}).get(cov)) for s in subjects)
                    if n is not None]
            if len(nums) < _MIN_N_FOR_COV_PERCENTILE:
                continue
            lo, hi = np.percentile(nums, percentiles)
            cov_values[cov] = [float(lo), float(hi)]
            cov_stats[cov] = {"n_cov": len(nums), "cov_min": float(min(nums)), "cov_max": float(max(nums))}
    return cov_values, ref_levels, cov_stats


def run_covariate_forest(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Covariate GMR forest plot from a converged NLME or SCM covariate model.

    Single-provenance selection mirrors `run_diagnostics`: `source="auto"`
    (default) prefers a converged SCM result that actually selected covariate
    effects, else falls back to a plain NLME fit's covariate model; either can
    be forced via `source="nlme"`/`"scm"`. SCM's `final` sub-result carries no
    `status` key of its own (only the outer `scm_results` dict does) — it is
    explicitly wrapped with `{"status": "ok", **final}` here so an SCM-sourced
    forest does not silently report `no_fit`.
    """
    from app.compute.forest import covariate_forest

    source_arg = args.get("source", "auto")
    nl = state.nlme_results if (state.nlme_results or {}).get("status") == "ok" else None
    scm_outer = state.scm_results if (state.scm_results or {}).get("status") == "ok" else None
    scm_final = None
    if scm_outer is not None:
        final = scm_outer.get("final") or {}
        if final.get("covariate_effects"):
            scm_final = {"status": "ok", **final}

    if source_arg == "nlme":
        chosen, chosen_src = nl, "nlme"
    elif source_arg == "scm":
        chosen, chosen_src = scm_final, "scm"
    else:
        chosen, chosen_src = (scm_final, "scm") if scm_final is not None else (nl, "nlme")

    if not chosen or not chosen.get("covariate_effects"):
        status = {"status": "no_fit",
                  "message": ("Need a converged run_nlme fit, or run_scm with at least one "
                              "selected covariate effect, before a covariate forest plot can "
                              "be built.")}
        return ToolResult(summary="Covariate forest skipped: no covariate model available.",
                          action="run_covariate_forest(no_fit)",
                          writes={"forest_results": status}, result=status)

    model_key = chosen.get("model_key")
    df = ctx.dataset_store[state.dataset_id]
    subjects, _multi, _has_pd = _build_subjects(df, _roles(df, state))

    pct = args.get("percentiles")
    percentiles = [float(pct[0]), float(pct[1])] if isinstance(pct, (list, tuple)) and len(pct) == 2 \
        else [5.0, 95.0]
    cov_values, ref_levels, cov_stats = _cov_eval_points(
        subjects, chosen["covariate_effects"], percentiles)

    bnd = args.get("bounds")
    bounds = (float(bnd[0]), float(bnd[1])) if isinstance(bnd, (list, tuple)) and len(bnd) == 2 else None

    try:
        out = covariate_forest(chosen, cov_values=cov_values, ref_levels=ref_levels,
                               ci_level=float(args.get("ci_level", 0.90)), bounds=bounds)
    except ValueError as e:
        status = {"status": "invalid_args", "message": str(e)}
        return ToolResult(summary=f"Covariate forest skipped: {e}",
                          action="run_covariate_forest(invalid_args)",
                          writes={"forest_results": status}, result=status)

    notes = list(out["notes"])
    if chosen_src == "scm":
        caveat = (scm_outer or {}).get("selection_caveat")
        if caveat:
            notes.append(caveat)

    payload = {"status": "ok", "model_key": model_key, "label": chosen.get("label"),
               "source": chosen_src, "percentiles": percentiles, "rows": out["rows"],
               "x_range": out["x_range"], "bounds": out["bounds"], "ci_level": out["ci_level"],
               "notes": notes, "summary": out["summary"], "cov_stats": cov_stats}
    return ToolResult(
        summary=(f"Covariate forest for {chosen.get('label')} ({chosen_src}): "
                 f"{out['summary']['n_rows']} row(s) across {out['summary']['n_effects']} effect(s)."),
        action=f"run_covariate_forest({model_key})",
        writes={"forest_results": payload},
        result={"status": "ok", "model_key": model_key, "source": chosen_src,
                "n_rows": out["summary"]["n_rows"]})


def run_dose_sweep(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    fitted_key, _fits, _pop, typical = _last_fit(state)
    model_key = args.get("model_key") or fitted_key
    if not model_key or model_key not in REGISTRY:
        status = {"status": "no_model", "message": "Fit a model first, or pass a model_key."}
        return ToolResult(summary="Dose sweep skipped: no model.", action="run_dose_sweep(no_model)",
                          writes={"dose_sweep_results": status}, result=status)
    model = get_model(model_key)
    params = dict(model.defaults)
    if model_key == fitted_key:
        params.update(typical)
    if isinstance(args.get("params"), dict):
        params.update({k: float(v) for k, v in args["params"].items() if k in model.params})

    base = float(args.get("dose", 100.0))
    doses = [float(d) for d in (args.get("doses") or [base * 0.5, base, base * 2.0])]
    tau = float(args.get("tau", 24.0))
    n_doses = int(args.get("n_doses", 1))
    tmax = float(args.get("tmax") or (max(tau * n_doses, 24.0) + tau))

    out = dose_sweep(model_key, params, doses, tau=tau, n_doses=n_doses, tmax=tmax,
                     wt=float(args.get("wt", 70.0)))
    payload = {"status": "ok", **out}
    return ToolResult(
        summary=(f"Dose sweep ({model.label}): {len(doses)} levels {doses} "
                 f"q{tau}h ×{n_doses}; Cmax {[p['cmax'] for p in out['profiles']]}."),
        action=f"run_dose_sweep({model_key})",
        writes={"dose_sweep_results": payload},
        result={"status": "ok", "model_key": model_key, "doses": doses,
                "cmax": [p["cmax"] for p in out["profiles"]]})


def list_pk_models(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    models = list_models()
    return ToolResult(
        summary=f"PK model library: {len(models)} models "
                f"({len(PK_KEYS)} PK + {len(models) - len(PK_KEYS)} PK/PD).",
        action="list_pk_models",
        writes={}, result={"models": models})


TOOLS = [
    Tool("fit_pk_model",
         "Fit a structural PK model from the library to the dataset (single- or "
         "multiple-dose), or compare several models and select by AIC. Reports "
         "per-subject parameters and a two-stage population summary.",
         "modeler",
         {"type": "object",
          "properties": {"dataset_id": {"type": "string"},
                         "model_key": {"type": "string"},
                         "compare": {"type": "boolean"},
                         "models": {"type": "array", "items": {"type": "string"}}},
          "required": []},
         fit_pk_model),
    Tool("list_pk_models",
         "List the available structural PK / PK-PD models in the library.",
         "modeler",
         {"type": "object", "properties": {}, "required": []},
         list_pk_models),
    Tool("simulate_pk_profile",
         "Forward-simulate a dosing regimen (dose, interval, number of doses, "
         "duration) on a fitted or chosen PK/PK-PD model and return the predicted "
         "concentration (and effect) time-course for plotting.",
         "simulator",
         {"type": "object",
          "properties": {"model_key": {"type": "string"},
                         "dose": {"type": "number"}, "tau": {"type": "number"},
                         "n_doses": {"type": "integer"}, "tmax": {"type": "number"},
                         "wt": {"type": "number"}, "rate": {"type": "number"},
                         "params": {"type": "object"}},
          "required": []},
         simulate_pk_profile),
    Tool("run_vpc",
         "Goodness-of-fit and visual predictive check for the fitted PK model: "
         "observed-vs-predicted (individual and population) plus a 5/50/95 "
         "prediction band over the most common dosing regimen.",
         "modeler",
         {"type": "object", "properties": {}, "required": []},
         run_vpc),
    Tool("run_diagnostics",
         "Residual diagnostics for the fitted PK model: legacy two-stage IWRES "
         "(unweighted log residual, always available), plus a single-provenance "
         "CWRES (conditional weighted residuals, Hooker 2007) and CPRED, and "
         "simulation-based npd (Comets normalized prediction discrepancy). CWRES "
         "and npd require a converged run_nlme fit of the SAME structural model "
         "(theta/Omega/sigma/EBEs) and report status='needs_nlme' until then — "
         "never mixed with the two-stage fit. `cwres_interaction` (default true) "
         "selects FOCE-I-style (residual variance at the conditional mode) vs "
         "literal Hooker 2007 FOCE (at eta=0) weighting.",
         "modeler",
         {"type": "object",
          "properties": {"cwres_interaction": {"type": "boolean"}},
          "required": []},
         run_diagnostics),
    Tool("run_covariate_forest",
         "Covariate forest plot: geometric mean ratio (GMR) of a structural "
         "parameter at a covariate value vs the model's own reference, with a "
         "Wald confidence interval, from a converged run_nlme fit or run_scm "
         "covariate model (`source`='auto'|'nlme'|'scm', default auto: prefers "
         "SCM when it selected any effect). Continuous covariates are evaluated "
         "at `percentiles` (default 5th/95th) of the loaded dataset; categorical "
         "covariates at every observed level. `bounds` is an OPTIONAL "
         "user-supplied reference band — never defaulted, since an unjustified "
         "band (e.g. the bioequivalence 0.8-1.25 interval) would misrepresent "
         "clinical significance.",
         "modeler",
         {"type": "object",
          "properties": {
              "source": {"type": "string", "enum": ["auto", "nlme", "scm"]},
              "percentiles": {"type": "array", "items": {"type": "number"}},
              "ci_level": {"type": "number"},
              "bounds": {"type": "array", "items": {"type": "number"}},
          },
          "required": []},
         run_covariate_forest),
    Tool("run_nlme",
         "True population (mixed-effects) fit of a structural PK model by FOCE-I "
         "or SAEM: typical values (theta) with RSE%, between-subject variability "
         "(Omega/IIV CV%), residual error, OFV, condition number, eta-shrinkage, "
         "and optional covariate effects. method='focei_saem' starts FOCE-I from "
         "a short SAEM burn-in; method='auto' additionally probes for multiple "
         "optima and escalates to a multi-start search only when it finds them, "
         "returning the lowest-OFV fit. Prefer 'auto' on harder models (many "
         "parameters, several IIV terms, covariates), where a single start — cold "
         "OR seeded — can converge to the wrong optimum while reporting success.",
         "modeler",
         {"type": "object",
          "properties": {"method": {"type": "string",
                                    "enum": ["focei", "saem", "focei_saem",
                                             "auto"]},
                         "model_key": {"type": "string"},
                         "iiv_params": {"type": "array", "items": {"type": "string"}},
                         "error_model": {"type": "string"},
                         "covariate_model": {"type": "array", "items": {"type": "object"}}},
          "required": []},
         run_nlme),
    Tool("run_scm",
         "Stepwise covariate modeling (forward selection at p<0.05 then backward "
         "elimination at p<0.01) on a structural PK model using FOCE-I OFVs. "
         "Auto-builds candidates from dataset covariate columns x IIV parameters "
         "unless an explicit candidate list is given; reports the step path, the "
         "selected covariate effects with coefficients/RSE%, and OFV drop.",
         "modeler",
         {"type": "object",
          "properties": {"model_key": {"type": "string"},
                         "iiv_params": {"type": "array", "items": {"type": "string"}},
                         "error_model": {"type": "string"},
                         "candidates": {"type": "array", "items": {"type": "object"}},
                         "forward_p": {"type": "number"},
                         "backward_p": {"type": "number"},
                         "max_iter": {"type": "integer"}},
          "required": []},
         run_scm),
    Tool("forecast_map",
         "MAP / empirical-Bayes forecast: from the fitted population (NLME) model "
         "and a new patient's sparse measured levels, estimate their individual PK, "
         "forecast steady-state exposure (Cmin/Cmax/Cavg/AUC), and optionally "
         "recommend a dose to hit a target.",
         "modeler",
         {"type": "object",
          "properties": {"dose": {"type": "number"}, "tau": {"type": "number"},
                         "measured": {"type": "array", "items": {"type": "object"}},
                         "wt": {"type": "number"}, "cov": {"type": "object"},
                         "target": {"type": "number"},
                         "target_metric": {"type": "string"},
                         "tmax": {"type": "number"}},
          "required": []},
         forecast_map),
    Tool("run_dose_sweep",
         "Simulate the fitted/chosen model across several dose levels and compare "
         "the concentration profiles and exposure metrics (Cmax, AUC_tau, Cavg).",
         "simulator",
         {"type": "object",
          "properties": {"model_key": {"type": "string"},
                         "doses": {"type": "array", "items": {"type": "number"}},
                         "dose": {"type": "number"}, "tau": {"type": "number"},
                         "n_doses": {"type": "integer"}, "tmax": {"type": "number"},
                         "wt": {"type": "number"}, "params": {"type": "object"}},
          "required": []},
         run_dose_sweep),
]
