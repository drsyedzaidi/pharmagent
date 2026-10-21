"""Simulator Agent tools: forward PK/PD simulation, dose sweeps, clinical trial
simulation (PTA), exposure covariate forest, special populations, individual EBE
exposures, and pediatric dose-matching.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from app.compute.cohort import (
    _MIN_N_FOR_COV_PERCENTILE,
    _num_or_none,
    build_subjects,
    cov_eval_points,
)
from app.compute.dose_sweep import dose_sweep
from app.compute.pk_models import REGISTRY, get_model
from app.compute.pk_simulate import simulate_timecourse
from app.core.pharmstate import PharmState
from app.core.schema_extractor import detect_roles
from app.tools.base import Tool, ToolContext, ToolResult

# Covariate names commonly denoting renal function (auto-detected default stratum).
_RENAL_COV_NAMES = ("RF", "EGFR", "CRCL", "CLCR", "GFR")
_WT_COV_NAMES = ("WT", "WEIGHT", "BW", "BWT", "BODYWEIGHT")


def _roles(df: pd.DataFrame, state: PharmState) -> dict[str, str]:
    if state.dataset_metadata and state.dataset_metadata.get("detected_roles"):
        return state.dataset_metadata["detected_roles"]
    return detect_roles(list(df.columns))


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


def _covariate_rows(state: PharmState, ctx: ToolContext) -> tuple[list[dict], list[float]]:
    """Per-subject covariate dicts + weights from the loaded dataset, for
    resampling a virtual population. Empty when no dataset is bound."""
    if not state.dataset_id or state.dataset_id not in ctx.dataset_store:
        return [], []
    try:
        df = ctx.dataset_store[state.dataset_id]
        subjects, _m, _p = build_subjects(df, _roles(df, state))
    except (ValueError, KeyError):
        return [], []
    cov_rows = [dict(s.get("cov") or {}) for s in subjects]
    wt_rows = [float(s.get("wt", 70.0)) for s in subjects]
    return cov_rows, wt_rows


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


def run_clinsim(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Clinical trial simulation → probability of target attainment across doses."""
    from app.compute.clinsim import clinical_trial_simulation  # lazy: may import nlme

    model_key, _fits, pop, typical = _last_fit(state)
    if not model_key:
        status = {"status": "no_fit",
                  "message": "Fit a PK model first to simulate a virtual trial."}
        return ToolResult(summary="Clinical trial simulation skipped: no fitted model.",
                          action="run_clinsim(no_fit)",
                          writes={"clinsim_results": status}, result=status)
    model = get_model(model_key)
    typical = {**model.defaults, **typical}
    params = (pop.get("parameters") or {})
    iiv = {k: v.get("iiv_cv_pct") for k, v in params.items()}
    iiv_params = [k for k, v in iiv.items() if v]

    # Covariate effects only from a converged NLME fit of the SAME model.
    nl = state.nlme_results if (state.nlme_results or {}).get("status") == "ok" else None
    if nl is not None and nl.get("model_key") != model_key:
        nl = None
    cov_effects = (nl or {}).get("covariate_effects") if nl else None
    # Single provenance: if we adopt the NLME covariate model, take that fit's
    # theta AND between-subject variability too — never layer an NLME covariate
    # model onto two-stage typical values (a hybrid that is neither validated fit).
    if cov_effects and nl.get("theta"):
        typical = {**model.defaults, **nl["theta"]}
        if nl.get("omega_cv_pct"):
            iiv = nl["omega_cv_pct"]
            iiv_params = list(nl.get("iiv_params") or iiv_params)
    cov_rows, wt_rows = _covariate_rows(state, ctx)

    base = float(args.get("dose", 100.0))
    doses = [float(d) for d in (args.get("doses")
             or [base * f for f in (0.25, 0.5, 1.0, 2.0, 4.0)])]
    threshold = args.get("threshold")
    n_subjects = int(args.get("n_subjects", 500))

    # Parameter-uncertainty PTA band + sensitivity (Ex 3/4), opt-in. Requires a
    # matching NLME fit carrying theta + theta_rse_pct; use it as the SINGLE
    # provenance for the whole run so the point PTA and the band agree. Budget-
    # capped: subjects x draws x doses is bounded so an opt-in band stays ~1 min.
    param_draws = None
    param_note = ""
    if bool(args.get("param_uncertainty")):
        rse = (nl or {}).get("theta_rse_pct") if nl else None
        nl_theta = (nl or {}).get("theta") if nl else None
        if nl and rse and nl_theta:
            from app.compute.clinsim import sample_theta_draws
            typical = {**model.defaults, **nl_theta}
            iiv = nl.get("omega_cv_pct") or iiv
            iiv_params = list(nl.get("iiv_params") or iiv_params)
            # Budget the band so a synchronous opt-in call stays ~20-30s:
            # subjects x draws x doses is bounded to ~60k simulations.
            n_subjects = min(n_subjects, 200)
            n_draws = min(int(args.get("n_param_draws", 60)),
                          max(2, 60_000 // max(1, n_subjects * len(doses))))
            param_draws = sample_theta_draws(nl_theta, rse, n_draws) or None
            if param_draws is None:
                param_note = " (no structural RSE% available for the band)"
        else:
            param_note = " (parameter-uncertainty band needs a converged NLME fit with RSE%)"

    payload = clinical_trial_simulation(
        model_key, theta=typical, omega_cv_pct=iiv, iiv_params=iiv_params,
        doses=doses, tau=float(args.get("tau", 24.0)),
        n_doses=int(args.get("n_doses", 1)),
        metric=args.get("metric", "ctrough"),
        threshold=(None if threshold is None else float(threshold)),
        direction=args.get("direction", "above"),
        target_fraction=float(args.get("target_fraction", 0.9)),
        cov_rows=cov_rows if cov_effects else None, wt_rows=wt_rows,
        covariate_effects=cov_effects, n_subjects=n_subjects,
        param_draws=param_draws)

    if payload.get("status") != "ok":
        return ToolResult(summary=f"Clinical trial simulation: {payload.get('message', payload['status'])}.",
                          action=f"run_clinsim({payload['status']})",
                          writes={"clinsim_results": payload}, result=payload)

    rec = payload.get("recommended_dose")
    tgt = payload.get("target_fraction")
    note = (f" Recommended dose {rec:g} ({tgt:.0%} attainment)." if rec is not None
            else f" No dose reached the {tgt:.0%} target.")
    iiv_txt = " with IIV" if payload["with_iiv"] else ""
    pta_txt = ("" if payload["threshold"] is None
               else f" {payload['direction']} {format(payload['threshold'], 'g')}")
    band_txt = (f" +{payload['n_param_draws']} parameter draws (PTA band + sensitivity)"
                if payload.get("n_param_draws") else param_note)
    return ToolResult(
        summary=(f"Clinical trial simulation ({model.label}): {payload['n_subjects']} virtual "
                 f"subjects{iiv_txt} over {len(payload['doses'])} doses; "
                 f"PTA on {payload['metric']}{pta_txt}.{band_txt}" + note),
        action=f"run_clinsim({model_key})",
        writes={"clinsim_results": payload},
        result={"status": "ok", "model_key": model_key, "metric": payload["metric"],
                "recommended_dose": rec, "n_subjects": payload["n_subjects"]})


def run_exposure_forest(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Simulated exposure covariate forest — relative AUC/Cmax across covariate
    scenarios with parameter uncertainty (Week-12 forest-plots.R)."""
    from app.compute.clinsim import exposure_covariate_forest  # lazy: imports nlme

    # Single-provenance sourcing mirrors run_covariate_forest: prefer a converged
    # SCM final model that selected covariates, else a plain NLME covariate model.
    nl = state.nlme_results if (state.nlme_results or {}).get("status") == "ok" else None
    scm_outer = state.scm_results if (state.scm_results or {}).get("status") == "ok" else None
    scm_final = None
    if scm_outer is not None:
        final = scm_outer.get("final") or {}
        if final.get("covariate_effects"):
            scm_final = {"status": "ok", **final}
    chosen = scm_final if (scm_final is not None) else nl
    cov_effects = (chosen or {}).get("covariate_effects")
    if not chosen or not cov_effects:
        status = {"status": "no_covariate_model",
                  "message": ("Fit a population model with covariate effects "
                              "(run_nlme with a covariate_model, or run_scm) first.")}
        return ToolResult(summary="Exposure forest skipped: no covariate model.",
                          action="run_exposure_forest(no_covariate_model)",
                          writes={"exposure_forest_results": status}, result=status)

    model_key = chosen.get("model_key")
    model = get_model(model_key)
    theta = {**model.defaults, **(chosen.get("theta") or {})}
    df = ctx.dataset_store.get(state.dataset_id) if state.dataset_id else None
    subjects = []
    if df is not None:
        try:
            subjects, _m, _p = build_subjects(df, _roles(df, state))
        except (ValueError, KeyError):
            subjects = []

    pct = args.get("percentiles")
    percentiles = ([float(pct[0]), float(pct[1])]
                   if isinstance(pct, (list, tuple)) and len(pct) == 2 else [5.0, 95.0])
    cov_values, ref_levels, _stats = cov_eval_points(subjects, cov_effects, percentiles)

    # Reference covariates: fitted center (continuous) / reference level (categorical).
    reference_cov: dict[str, Any] = {}
    scenarios: list[dict] = []
    for eff in cov_effects:
        cov = eff["covariate"]
        if eff.get("kind") == "categorical":
            ref = ref_levels.get(cov)
            reference_cov[cov] = ref
            levels = [{"label": str(lv), "value": lv}
                      for lv in cov_values.get(cov, []) if str(lv) != str(ref)]
        else:
            reference_cov[cov] = float(eff.get("center") or 0.0)
            vals = cov_values.get(cov)
            if not vals:
                continue
            levels = [{"label": f"{percentiles[0]:g}th", "value": vals[0]},
                      {"label": f"{percentiles[1]:g}th", "value": vals[1]}]
        if levels:
            scenarios.append({"covariate": cov, "is_weight": False, "levels": levels})

    # WT scenario via allometric scaling (not part of the estimated covariate model).
    wt_rows = [float(s.get("wt", 70.0)) for s in subjects]
    ref_wt = float(np.median(wt_rows)) if wt_rows else 70.0
    if model.allometric and len(wt_rows) >= _MIN_N_FOR_COV_PERCENTILE:
        lo, hi = np.percentile(wt_rows, percentiles)
        scenarios.append({"covariate": "WT", "is_weight": True,
                          "levels": [{"label": f"{lo:.0f} kg", "value": float(lo)},
                                     {"label": f"{hi:.0f} kg", "value": float(hi)}]})

    if not scenarios:
        status = {"status": "no_scenarios",
                  "message": "no covariate levels available to simulate (need the source dataset)."}
        return ToolResult(summary="Exposure forest skipped: no scenarios.",
                          action="run_exposure_forest(no_scenarios)",
                          writes={"exposure_forest_results": status}, result=status)

    dose = float(args.get("dose", 100.0))
    payload = exposure_covariate_forest(
        model_key, theta=theta, covariate_effects=cov_effects, scenarios=scenarios,
        reference_cov=reference_cov, dose=dose, tau=float(args.get("tau", 24.0)),
        n_doses=int(args.get("n_doses", 7)), ref_wt=ref_wt,
        n_draws=int(args.get("n_draws", 500)))
    if payload.get("status") != "ok":
        return ToolResult(summary=f"Exposure forest: {payload.get('message', payload['status'])}.",
                          action=f"run_exposure_forest({payload['status']})",
                          writes={"exposure_forest_results": payload}, result=payload)
    n_out = len(payload["rows"])
    return ToolResult(
        summary=(f"Exposure covariate forest ({model.label}): relative AUC/Cmax over "
                 f"{n_out} covariate scenarios at {dose:g}, {payload['n_draws']} uncertainty "
                 f"draws; reference AUC {payload['reference']['auc']}."),
        action=f"run_exposure_forest({model_key})",
        writes={"exposure_forest_results": payload},
        result={"status": "ok", "model_key": model_key, "n_rows": n_out})


def run_special_population(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Special-population exposure simulation — steady-state exposure by renal (or
    other) covariate stratum vs a reference band (Week-13 renal-simulations.R)."""
    from app.compute.clinsim import (  # lazy: imports nlme
        reference_population,
        special_population_simulation,
    )

    model_key, _fits, pop, typical = _last_fit(state)
    if not model_key:
        status = {"status": "no_fit", "message": "Fit a PK model first to simulate special populations."}
        return ToolResult(summary="Special-population simulation skipped: no fitted model.",
                          action="run_special_population(no_fit)",
                          writes={"special_pop_results": status}, result=status)
    model = get_model(model_key)
    typical = {**model.defaults, **typical}
    params = (pop.get("parameters") or {})
    iiv = {k: v.get("iiv_cv_pct") for k, v in params.items()}
    iiv_params = [k for k, v in iiv.items() if v]

    # Covariate model: prefer SCM final, else NLME (single provenance, like the forest).
    nl = state.nlme_results if (state.nlme_results or {}).get("status") == "ok" else None
    scm_outer = state.scm_results if (state.scm_results or {}).get("status") == "ok" else None
    cov_effects = None
    # Single provenance: when we adopt a fit's covariate model, take that fit's
    # theta AND between-subject variability (omega), never mix with the two-stage IIV.
    if scm_outer and (scm_outer.get("final") or {}).get("covariate_effects"):
        final = scm_outer["final"]
        if final.get("model_key") == model_key:
            cov_effects = final.get("covariate_effects")
            typical = {**model.defaults, **(final.get("theta") or typical)}
            if final.get("omega_cv_pct"):
                iiv = final["omega_cv_pct"]
                iiv_params = list(final.get("iiv_params") or iiv_params)
    if cov_effects is None and nl and nl.get("model_key") == model_key:
        cov_effects = nl.get("covariate_effects")
        if cov_effects and nl.get("theta"):
            typical = {**model.defaults, **nl["theta"]}
            if nl.get("omega_cv_pct"):
                iiv = nl["omega_cv_pct"]
                iiv_params = list(nl.get("iiv_params") or iiv_params)

    # Population source: the analysis dataset (default) or a synthetic representative
    # adult population spanning renal categories (when the dataset lacks renal spread).
    source = args.get("source", "dataset")
    if source == "reference":
        cov_rows, wt_rows = reference_population(n=int(args.get("n_reference", 4000)))
    else:
        cov_rows, wt_rows = _covariate_rows(state, ctx)
    # Default stratum: a renal covariate present in the data, else the first covariate.
    cov_keys = {c for r in cov_rows for c in (r or {})}
    stratify_by = args.get("stratify_by")
    if not stratify_by:
        stratify_by = next((c for c in _RENAL_COV_NAMES if c in cov_keys), None)
    if not stratify_by:
        status = {"status": "no_covariate", "available": sorted(cov_keys),
                  "message": ("no renal covariate (EGFR/RF/CrCl) found; pass stratify_by "
                              f"as one of: {', '.join(sorted(cov_keys)) or '(none)'}")}
        return ToolResult(summary="Special-population simulation skipped: no stratifying covariate.",
                          action="run_special_population(no_covariate)",
                          writes={"special_pop_results": status}, result=status)

    base = float(args.get("dose", 100.0))
    doses = [float(d) for d in (args.get("doses")
             or [base * f for f in (0.5, 1.0, 2.0, 4.0)])]
    metrics = tuple(args.get("metrics") or ("auc_tau", "cmax"))
    payload = special_population_simulation(
        model_key, theta=typical, omega_cv_pct=iiv, iiv_params=iiv_params,
        cov_rows=cov_rows, wt_rows=wt_rows, stratify_by=stratify_by, doses=doses,
        tau=float(args.get("tau", 24.0)), n_doses=int(args.get("n_doses", 7)),
        covariate_effects=cov_effects, metrics=metrics,
        reference_stratum=args.get("reference_stratum", "Normal"),
        reference_dose=args.get("reference_dose"),
        n_per_stratum=int(args.get("n_per_stratum", 600)))
    # A covariate model that includes the stratifying covariate is what makes the
    # strata differ; flag when it is missing so a flat result is not misread.
    cov_in_model = bool(cov_effects) and any(
        (e.get("covariate") or "").upper() == stratify_by.upper() for e in (cov_effects or []))
    if payload.get("status") == "ok":
        payload["covariate_in_model"] = cov_in_model
        payload["population_source"] = source

    if payload.get("status") != "ok":
        return ToolResult(summary=f"Special-population simulation: {payload.get('message', payload['status'])}.",
                          action=f"run_special_population({payload['status']})",
                          writes={"special_pop_results": payload}, result=payload)
    note = "" if cov_in_model else (f" (no fitted {stratify_by} effect — strata differ only by "
                                    "allometric weight; run SCM/NLME with this covariate)")
    n_strata = len(payload["strata"])
    return ToolResult(
        summary=(f"Special-population simulation ({model.label}) by {stratify_by}: {n_strata} strata × "
                 f"{len(doses)} doses vs the {payload['reference_stratum']} reference band.{note}"),
        action=f"run_special_population({model_key})",
        writes={"special_pop_results": payload},
        result={"status": "ok", "model_key": model_key, "stratify_by": stratify_by,
                "n_strata": n_strata})


def run_individual_exposures(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Per-subject steady-state exposure (AUCss/Cmax,ss) from the fitted EBEs."""
    from app.compute.clinsim import individual_exposures  # lazy: imports nlme

    nl = state.nlme_results if (state.nlme_results or {}).get("status") == "ok" else None
    if not nl or not nl.get("individual"):
        status = {"status": "needs_nlme",
                  "message": ("Individual exposures need a converged run_nlme fit with per-subject "
                              "EBEs (eta values).")}
        return ToolResult(summary="Individual exposures skipped: no NLME EBEs.",
                          action="run_individual_exposures(needs_nlme)",
                          writes={"individual_exposures": status}, result=status)
    model_key = nl.get("model_key")
    model = get_model(model_key)
    theta = {**model.defaults, **(nl.get("theta") or {})}
    etas = {str(r["subject"]): (r.get("eta") or {}) for r in nl["individual"] if "subject" in r}

    df = ctx.dataset_store.get(state.dataset_id) if state.dataset_id else None
    if df is None:
        status = {"status": "no_dataset", "message": "the source dataset is not loaded."}
        return ToolResult(summary="Individual exposures skipped: no dataset.",
                          action="run_individual_exposures(no_dataset)",
                          writes={"individual_exposures": status}, result=status)
    subjects, _m, _p = build_subjects(df, _roles(df, state))
    subj_in = [{"subject": str(s["subject"]), "cov": s.get("cov") or {},
                "wt": float(s.get("wt", 70.0))} for s in subjects]

    group_key = args.get("group_by")
    if not group_key:
        cov_keys = {c for s in subj_in for c in s["cov"]}
        group_key = next((c for c in _RENAL_COV_NAMES if c in cov_keys), None)
    payload = individual_exposures(
        model_key, theta=theta, subjects=subj_in, etas=etas,
        dose=float(args.get("dose", 100.0)), tau=float(args.get("tau", 24.0)),
        n_doses=int(args.get("n_doses", 7)),
        covariate_effects=nl.get("covariate_effects"),
        iiv_params=list(nl.get("iiv_params") or []), group_key=group_key)
    if payload.get("status") != "ok":
        return ToolResult(summary=f"Individual exposures: {payload.get('message', payload['status'])}.",
                          action=f"run_individual_exposures({payload['status']})",
                          writes={"individual_exposures": payload}, result=payload)
    return ToolResult(
        summary=(f"Individual exposures ({model.label}): steady-state AUCss/Cmax,ss for "
                 f"{len(payload['subjects'])} subjects at {payload['dose']:g} q{payload['tau']}h"
                 + (f", grouped by {group_key}" if group_key else "") + "."),
        action=f"run_individual_exposures({model_key})",
        writes={"individual_exposures": payload},
        result={"status": "ok", "model_key": model_key, "n_subjects": len(payload["subjects"])})


def run_pediatric_simulation(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Pediatric dose-finding by exposure matching — steady-state exposure by age x
    weight stratum vs an adult reference band (Week-14 pediatric-simulations.R)."""
    from app.compute.clinsim import (  # lazy: imports nlme
        _cv_pct_to_omega2,
        individual_exposures,
        pediatric_reference_population,
        pediatric_simulation,
        reference_population,
    )

    model_key, _fits, pop, typical = _last_fit(state)
    if not model_key:
        status = {"status": "no_fit", "message": "Fit a PK model first to simulate pediatric doses."}
        return ToolResult(summary="Pediatric simulation skipped: no fitted model.",
                          action="run_pediatric_simulation(no_fit)",
                          writes={"pediatric_results": status}, result=status)
    model = get_model(model_key)
    typical = {**model.defaults, **typical}
    params = (pop.get("parameters") or {})
    iiv = {k: v.get("iiv_cv_pct") for k, v in params.items()}
    iiv_params = [k for k, v in iiv.items() if v]

    # Covariate model: SCM final else NLME (single provenance, like special-pop).
    nl = state.nlme_results if (state.nlme_results or {}).get("status") == "ok" else None
    scm_outer = state.scm_results if (state.scm_results or {}).get("status") == "ok" else None
    cov_effects = None
    # Single provenance: adopt the covariate model's fit theta AND omega together.
    if scm_outer and (scm_outer.get("final") or {}).get("covariate_effects"):
        final = scm_outer["final"]
        if final.get("model_key") == model_key:
            cov_effects = final.get("covariate_effects")
            typical = {**model.defaults, **(final.get("theta") or typical)}
            if final.get("omega_cv_pct"):
                iiv = final["omega_cv_pct"]
                iiv_params = list(final.get("iiv_params") or iiv_params)
    if cov_effects is None and nl and nl.get("model_key") == model_key:
        cov_effects = nl.get("covariate_effects")
        if cov_effects and nl.get("theta"):
            typical = {**model.defaults, **nl["theta"]}
            if nl.get("omega_cv_pct"):
                iiv = nl["omega_cv_pct"]
                iiv_params = list(nl.get("iiv_params") or iiv_params)

    # Model-estimated allometric exponents (opt-in): a WT-CL and WT-V exponent
    # (e.g. 0.663 / 1.087) replacing the built-in fixed 0.75 / 1.0. Applied to the
    # flow (exp<1) and volume (exp>=1) params via the model's allometric map.
    cl_e = _num_or_none(args.get("wt_exponent_cl"))
    v_e = _num_or_none(args.get("wt_exponent_v"))
    wt_exponents = None
    if cl_e is not None or v_e is not None:
        cl_e = 0.75 if cl_e is None else cl_e
        v_e = 1.0 if v_e is None else v_e
        wt_exponents = {p: (cl_e if expo < 1.0 else v_e) for p, expo in model.allometric.items()}
    # Weight always enters via allometry — the built-in fixed 0.75/1.0 on the default
    # path, or wt_exponents when supplied. A WT covariate in the fitted model would
    # double-count against BOTH, so strip it unconditionally before simulating.
    cov_effects = [e for e in (cov_effects or [])
                   if (e.get("covariate") or "").upper() not in _WT_COV_NAMES] or None

    tau = float(args.get("tau", 12.0))
    n_doses = int(args.get("n_doses", 14))
    ref_dose = float(args.get("reference_dose", 25.0))
    doses = [float(d) for d in (args.get("doses") or [5.0, 10.0, 15.0, 20.0, 25.0])]
    n_per = int(args.get("n_per_stratum", 1000))

    # Adult reference band: normal-renal adults at the label dose, one IIV draw
    # each (stochastic adult exposure distribution). Dataset adults if available,
    # else representative adults — mirrors the lab's normal-adult-exposures.
    adult_rows, adult_wt = _covariate_rows(state, ctx)
    adult_source = "dataset adults"
    if not adult_rows:
        adult_rows, adult_wt = reference_population(n=int(args.get("n_reference", 4000)))
        adult_source = "representative adults"
    renal_key = next((c for c in _RENAL_COV_NAMES for r in adult_rows if c in (r or {})), None)
    if renal_key:
        keep = [i for i, r in enumerate(adult_rows) if _num_or_none((r or {}).get(renal_key)) is None
                or _num_or_none((r or {}).get(renal_key)) >= 90.0]
        if keep:
            adult_rows = [adult_rows[i] for i in keep]
            adult_wt = [adult_wt[i] for i in keep]
    rng = np.random.default_rng(20250614)
    sds = {p: (_cv_pct_to_omega2(iiv.get(p)) ** 0.5) for p in iiv_params}
    adults = [{"subject": str(i), "cov": adult_rows[i], "wt": float(adult_wt[i])}
              for i in range(len(adult_rows))]
    etas = {str(i): {p: float(rng.normal(0.0, sds[p])) if sds[p] > 0 else 0.0 for p in iiv_params}
            for i in range(len(adult_rows))}
    adult_ie = individual_exposures(
        model_key, theta=typical, subjects=adults, etas=etas, covariate_effects=cov_effects,
        iiv_params=iiv_params, dose=ref_dose, tau=tau, n_doses=n_doses, wt_exponents=wt_exponents)
    ref_exp = {"auc_tau": [s["auc_ss"] for s in adult_ie.get("subjects", [])],
               "cmax": [s["cmax_ss"] for s in adult_ie.get("subjects", [])]}

    # Pediatric covariate source: representative pediatric population (default) or
    # the dataset (rare — analysis datasets are usually adults).
    source = args.get("source", "reference")
    if source == "dataset":
        ped_rows, ped_wt = _covariate_rows(state, ctx)
    else:
        ped_rows, ped_wt = pediatric_reference_population(n=int(args.get("n_pediatric", 6000)))

    payload = pediatric_simulation(
        model_key, theta=typical, omega_cv_pct=iiv, iiv_params=iiv_params, cov_rows=ped_rows,
        wt_rows=ped_wt, reference_exposures=ref_exp, doses=doses, tau=tau, n_doses=n_doses,
        covariate_effects=cov_effects, wt_exponents=wt_exponents, n_per_stratum=n_per)
    if payload.get("status") == "ok":
        payload["reference_dose"] = ref_dose
        payload["reference_source"] = adult_source
        payload["population_source"] = source
    if payload.get("status") != "ok":
        return ToolResult(summary=f"Pediatric simulation: {payload.get('message', payload['status'])}.",
                          action=f"run_pediatric_simulation({payload['status']})",
                          writes={"pediatric_results": payload}, result=payload)
    allo = "estimated" if wt_exponents else "fixed 0.75/1.0"
    return ToolResult(
        summary=(f"Pediatric simulation ({model.label}): {len(payload['strata'])} age x weight strata "
                 f"x {len(doses)} doses vs the adult range ({adult_source} at {ref_dose:g}); "
                 f"{allo} allometry."),
        action=f"run_pediatric_simulation({model_key})",
        writes={"pediatric_results": payload},
        result={"status": "ok", "model_key": model_key, "n_strata": len(payload["strata"])})


TOOLS = [
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
    Tool("run_clinsim",
         "Clinical trial simulation for dose selection: simulate a virtual population "
         "(covariates resampled from the dataset, between-subject variability from the "
         "fitted IIV) across a grid of doses and report the probability of target "
         "attainment — the fraction of subjects whose Cmax/AUC_tau/Cavg/Ctrough is "
         "above (efficacy) or below (safety) a clinical threshold — recommending the "
         "lowest efficacious / highest safe dose.",
         "simulator",
         {"type": "object",
          "properties": {"doses": {"type": "array", "items": {"type": "number"}},
                         "dose": {"type": "number"}, "tau": {"type": "number"},
                         "n_doses": {"type": "integer"},
                         "metric": {"type": "string",
                                    "enum": ["cmax", "auc_tau", "cavg", "ctrough"]},
                         "threshold": {"type": "number"},
                         "direction": {"type": "string", "enum": ["above", "below"]},
                         "target_fraction": {"type": "number"},
                         "n_subjects": {"type": "integer"},
                         "param_uncertainty": {"type": "boolean"},
                         "n_param_draws": {"type": "integer"}},
          "required": []},
         run_clinsim),
    Tool("run_exposure_forest",
         "Simulated exposure covariate forest: for a fitted covariate model, "
         "simulate a steady-state regimen for a reference patient and for each "
         "covariate at its dataset extremes, and report the relative AUC/Cmax vs "
         "the reference with a 95% interval from coefficient uncertainty — the "
         "labeling-style forest (shaded 0.8-1.25) that a parameter-ratio forest "
         "cannot give.",
         "simulator",
         {"type": "object",
          "properties": {"dose": {"type": "number"}, "tau": {"type": "number"},
                         "n_doses": {"type": "integer"},
                         "percentiles": {"type": "array", "items": {"type": "number"}},
                         "n_draws": {"type": "integer"}},
          "required": []},
         run_exposure_forest),
    Tool("run_special_population",
         "Special-population exposure simulation: stratify a virtual population by "
         "renal function (eGFR/RF bins Severe/Moderate/Mild/Normal) or another "
         "covariate, simulate steady-state exposure across a dose grid per stratum, "
         "and compare each stratum to the reference (normal) 90% band — the "
         "renal/hepatic/weight dose-adjustment question for a biologic.",
         "simulator",
         {"type": "object",
          "properties": {"stratify_by": {"type": "string"},
                         "doses": {"type": "array", "items": {"type": "number"}},
                         "dose": {"type": "number"}, "tau": {"type": "number"},
                         "n_doses": {"type": "integer"},
                         "metrics": {"type": "array", "items": {"type": "string"}},
                         "reference_stratum": {"type": "string"},
                         "reference_dose": {"type": "number"},
                         "n_per_stratum": {"type": "integer"},
                         "source": {"type": "string", "enum": ["dataset", "reference"]},
                         "n_reference": {"type": "integer"}},
          "required": []},
         run_special_population),
    Tool("run_individual_exposures",
         "Per-subject steady-state exposure (AUCss / Cmax,ss) computed from the "
         "fitted individual EBEs — the reference exposure table for a special-"
         "population comparison. Needs a converged run_nlme fit.",
         "simulator",
         {"type": "object",
          "properties": {"dose": {"type": "number"}, "tau": {"type": "number"},
                         "n_doses": {"type": "integer"}, "group_by": {"type": "string"}},
          "required": []},
         run_individual_exposures),
    Tool("run_pediatric_simulation",
         "Pediatric dose-finding by exposure matching: stratify a pediatric virtual "
         "population (ages 2-<18 y) by age x weight, simulate steady-state exposure "
         "across a dose grid, and report the % of each stratum whose AUCss/Cmax,ss "
         "lands inside the adult reference range — recommending the dose that best "
         "matches adult exposure. Supports model-estimated allometric exponents.",
         "simulator",
         {"type": "object",
          "properties": {"doses": {"type": "array", "items": {"type": "number"}},
                         "tau": {"type": "number"}, "n_doses": {"type": "integer"},
                         "reference_dose": {"type": "number"},
                         "n_per_stratum": {"type": "integer"},
                         "source": {"type": "string", "enum": ["dataset", "reference"]},
                         "n_pediatric": {"type": "integer"}, "n_reference": {"type": "integer"},
                         "wt_exponent_cl": {"type": "number"},
                         "wt_exponent_v": {"type": "number"}},
          "required": []},
         run_pediatric_simulation),
]
