"""Governed protocol simulation compute layer.

Implements deterministic, seed-controlled trial PTA simulation across
plan-defined covariate bands and regimens, plan-scoped special-population /
pediatric simulation, and outer-loop parameter-uncertainty propagation from
bootstrap, SIR, or likelihood-profile estimates into PTA intervals.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

from app.compute.clinsim import (
    _covariate_applier,
    _cv_pct_to_omega2,
    pediatric_reference_population,
    pediatric_simulation,
    reference_population,
    special_population_simulation,
)
from app.compute.dose_sweep import _interval_metrics
from app.compute.pk_models import REGISTRY, get_model
from app.compute.pk_simulate import simulate_timecourse

TOOL_VERSION = "1.1.0"

#: Draws from the fit's asymptotic theta RSE% (diagonal log-normal). Named for what
#: it is; never presented under the name of an engine that did not run.
ASYMPTOTIC_METHOD = "asymptotic_lognormal"
#: Engines whose REAL result may be stored in state, in plan-default preference order.
ENGINE_METHODS: tuple[str, ...] = ("sir", "bootstrap", "profile")
UNCERTAINTY_METHODS: tuple[str, ...] = ENGINE_METHODS + (ASYMPTOTIC_METHOD,)
#: Per-parameter (centre, lower, upper) keys exactly as each engine writes them.
_ENGINE_INTERVAL_KEYS: dict[str, tuple[str, str, str]] = {
    "sir": ("sir_median", "sir_lo", "sir_hi"),
    "bootstrap": ("boot_median", "boot_lo", "boot_hi"),
    "profile": ("estimate", "profile_lo", "profile_hi"),
}


def canonical_hash(obj: Any) -> str:
    """Deterministic SHA-256 hex prefix (16 chars) of a JSON-serializable object."""
    payload = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def tm50_maturation(pma_weeks: float, tm50: float = 47.7, hill: float = 3.4) -> float:
    """Sigmoid Emax maturation function for clearance as a function of PMA (weeks)."""
    pma = max(float(pma_weeks), 1.0)
    num = pma ** float(hill)
    den = (float(tm50) ** float(hill)) + num
    return float(num / den) if den > 0 else 1.0


_MODEL_ALIASES: dict[str, str] = {
    "1cmt_oral": "oral_1cmt",
    "2cmt_oral": "oral_2cmt",
    "1cmt_iv": "iv_1cmt",
    "2cmt_iv": "iv_2cmt",
}


def _resolve_model_key(raw: Any) -> str:
    k = str(raw or "oral_1cmt")
    k = _MODEL_ALIASES.get(k, k)
    return k if k in REGISTRY else "oral_1cmt"


def extract_approved_fit(
    nlme_results: dict[str, Any] | None,
    review_results: dict[str, Any] | None,
    dataset_covariate_ranges: dict[str, list[float]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify preconditions (converged NLME fit + approved adversarial review)
    and build ``(source_fit, provenance)``."""
    if not isinstance(nlme_results, dict) or nlme_results.get("status") != "ok":
        raise ValueError(
            "protocol_simulation requires a converged NLME fit "
            "(nlme_results.status == 'ok'). Run the poppk workflow first."
        )
    if nlme_results.get("converged") is False:
        raise ValueError(
            "protocol_simulation requires a converged NLME fit "
            "(nlme_results.converged is False)."
        )
    if not isinstance(review_results, dict) or review_results.get("goal_met") is not True:
        raise ValueError(
            "protocol_simulation requires an approved adversarial review on the "
            "source fit (review_results.goal_met == True)."
        )

    model_key = _resolve_model_key(nlme_results.get("model_key"))
    model = get_model(model_key)

    theta = {**model.defaults, **{k: float(v) for k, v in (nlme_results.get("theta") or {}).items()}}
    omega_cv_pct = {k: float(v) for k, v in (nlme_results.get("omega_cv_pct") or {}).items()
                    if v is not None}
    iiv_params = list(nlme_results.get("iiv_params") or list(omega_cv_pct.keys()))
    theta_rse_pct = {
        k: float(v)
        for k, v in (nlme_results.get("theta_rse_pct") or {}).items()
        if v is not None and math.isfinite(float(v)) and float(v) > 0
    }
    # No default: a parameter without a finite RSE is listed, never invented.
    theta_rse_missing = sorted(p for p in theta if p not in theta_rse_pct)

    cov_ranges: dict[str, list[float]] = {
        "wt_kg": [45.0, 110.0],
        "age_yr": [18.0, 75.0],
        "crcl_ml_min": [40.0, 130.0],
    }
    if isinstance(dataset_covariate_ranges, dict):
        for k, v in dataset_covariate_ranges.items():
            if isinstance(v, (list, tuple)) and len(v) == 2:
                cov_ranges[k] = [float(v[0]), float(v[1])]
    if isinstance(nlme_results.get("covariate_ranges"), dict):
        for k, v in nlme_results["covariate_ranges"].items():
            if isinstance(v, (list, tuple)) and len(v) == 2:
                cov_ranges[k] = [float(v[0]), float(v[1])]

    # Maturation data range is only present when carried by the fit.
    maturation_data_range = nlme_results.get("maturation_data_range")

    fit_id = str(
        nlme_results.get("fit_id")
        or canonical_hash({
            "model_key": model_key,
            "theta": theta,
            "omega_cv_pct": omega_cv_pct,
            "ofv": nlme_results.get("ofv"),
        })
    )
    review_hash = canonical_hash(review_results)

    source_fit = {
        "fit_id": fit_id,
        "model_key": model_key,
        "label": model.label,
        "route": "iv" if "iv" in model_key.lower() else "oral",
        "theta": theta,
        "omega_cv_pct": omega_cv_pct,
        "iiv_params": iiv_params,
        "theta_rse_pct": theta_rse_pct,
        "theta_rse_missing": theta_rse_missing,
        "sigma": nlme_results.get("sigma") or {"prop": 0.1, "add": 0.0},
        "ofv": float(nlme_results.get("ofv") or 0.0),
        "covariate_effects": list(nlme_results.get("covariate_effects") or []),
        "covariate_ranges": cov_ranges,
        "maturation_data_range": maturation_data_range,
    }
    provenance = {
        "source_fit_id": fit_id,
        "review_hash": review_hash,
        "plan_hash": None,
        "seed": None,
        "tool_versions": {"load_approved_fit": TOOL_VERSION},
        "wall_clock": {},
    }
    return source_fit, provenance


def default_uncertainty_method(available_methods: list[str] | None) -> str:
    """The plan default is the most rigorous engine whose REAL result is stored in
    state (SIR, then bootstrap, then profile); otherwise the asymptotic approximation,
    named as such so the plan gate approves what will actually run."""
    present = set(available_methods or [])
    return next((m for m in ENGINE_METHODS if m in present), ASYMPTOTIC_METHOD)


def build_simulation_plan(
    source_fit: dict[str, Any],
    args: dict[str, Any] | None = None,
    available_methods: list[str] | None = None,
) -> dict[str, Any]:
    """Construct the structured simulation plan data object for human gate review.

    ``available_methods`` names the uncertainty engines whose result is already in
    state; the plan's ``uncertainty_method`` defaults to the best of them.
    """
    args = args or {}
    cov_ranges = source_fit.get("covariate_ranges") or {}
    wt_lo, wt_hi = cov_ranges.get("wt_kg", [50.0, 95.0])
    age_lo, age_hi = cov_ranges.get("age_yr", [20.0, 70.0])
    crcl_lo, crcl_hi = cov_ranges.get("crcl_ml_min", [50.0, 120.0])

    default_bands = [
        {
            "name": "standard_adult",
            "wt_kg": [round(float(wt_lo), 1), round(float(wt_hi), 1)],
            "age_yr": [round(float(age_lo), 1), round(float(age_hi), 1)],
            "crcl_ml_min": [round(float(max(crcl_lo, 60.0)), 1), round(float(crcl_hi), 1)],
            "is_pediatric": False,
        }
    ]
    bands = list(args.get("bands") or default_bands)

    route = source_fit.get("route", "oral")
    default_regimens = [
        {"label": "100mg_q24h", "dose_mg": 100.0, "tau_h": 24.0, "route": route, "n_doses": 5},
        {"label": "150mg_q24h", "dose_mg": 150.0, "tau_h": 24.0, "route": route, "n_doses": 5},
        {"label": "200mg_q24h", "dose_mg": 200.0, "tau_h": 24.0, "route": route, "n_doses": 5},
    ]
    regimens = list(args.get("regimens") or default_regimens)
    metrics = list(args.get("metrics") or ["auc", "cmax", "ctrough"])

    # Calibrate default target window around typical reference exposure at 150 mg
    # so the default plan has a realistic therapeutic window with external provenance.
    theta = source_fit.get("theta") or {}
    cl_typ = float(theta.get("CL") or theta.get("cl") or 5.0)
    v_typ = float(theta.get("V") or theta.get("VC") or theta.get("v") or 40.0)
    ref_auc = 150.0 / max(cl_typ, 0.1)
    ref_cmax = (150.0 / max(v_typ, 1.0)) * 2.5

    default_window = {
        "metric": "auc",
        "low": round(ref_auc * 0.55, 3),
        "high": round(ref_auc * 2.20, 3),
        "cmax_max": round(ref_cmax * 2.50, 3),
        "ctrough_min": 0.005,
        "provenance": {
            "source": "protocol_specification",
            "citation": "Pre-specified clinical protocol target exposure window (Phase 2 E-R anchor)",
        },
    }
    target_window = dict(args.get("target_window") or default_window)

    available = [m for m in ENGINE_METHODS if m in set(available_methods or [])]
    uncertainty_method = str(
        args.get("uncertainty_method") or default_uncertainty_method(available)).lower()
    if uncertainty_method not in UNCERTAINTY_METHODS:
        raise ValueError(
            "uncertainty_method must be one of "
            f"{', '.join(repr(m) for m in UNCERTAINTY_METHODS)}; got {uncertainty_method!r}"
        )

    plan = {
        "bands": bands,
        "regimens": regimens,
        "metrics": metrics,
        "target_window": target_window,
        "n_subjects": int(args.get("n_subjects", 80)),
        "replicates": int(args.get("replicates", 10)),
        "seed": int(args.get("seed", 42)),
        "uncertainty_method": uncertainty_method,
        "uncertainty_methods_available": available,
        "target_pta": float(args.get("target_pta", 0.75)),
    }
    return plan


def _simulate_band_regimen_cohort(
    model_key: str,
    theta: dict[str, float],
    omega_cv_pct: dict[str, float],
    iiv_params: list[str],
    covariate_effects: list[dict[str, Any]] | None,
    band: dict[str, Any],
    regimen: dict[str, Any],
    n_subjects: int,
    rng: np.random.Generator,
    n_points: int = 96,
) -> dict[str, np.ndarray]:
    """Simulate ``n_subjects`` virtual patients in ``band`` under ``regimen``."""
    model = get_model(model_key)
    typical = {**model.defaults, **{k: float(v) for k, v in theta.items()}}
    apply_cov = _covariate_applier(covariate_effects)
    sds = {p: float(np.sqrt(_cv_pct_to_omega2(omega_cv_pct.get(p)))) for p in iiv_params}

    wt_range = band.get("wt_kg") or [70.0, 70.0]
    age_range = band.get("age_yr") or [40.0, 40.0]
    crcl_range = band.get("crcl_ml_min") or [90.0, 90.0]
    pma_range = band.get("pma_weeks")

    wts = rng.uniform(float(wt_range[0]), float(wt_range[1]), size=n_subjects)
    ages = rng.uniform(float(age_range[0]), float(age_range[1]), size=n_subjects)
    crcls = rng.uniform(float(crcl_range[0]), float(crcl_range[1]), size=n_subjects)
    if pma_range and len(pma_range) == 2:
        pmas = rng.uniform(float(pma_range[0]), float(pma_range[1]), size=n_subjects)
    else:
        pmas = ages * 52.0 + 40.0

    dose = float(regimen.get("dose_mg") or regimen.get("dose") or 100.0)
    tau = float(regimen.get("tau_h") or regimen.get("tau") or 24.0)
    n_doses = int(regimen.get("n_doses", 5))
    tmax = tau * n_doses
    t_last = (n_doses - 1) * tau

    aucs = np.empty(n_subjects, dtype=float)
    cmaxs = np.empty(n_subjects, dtype=float)
    ctroughs = np.empty(n_subjects, dtype=float)

    is_ped = bool(band.get("is_pediatric")) or float(age_range[0]) < 18.0

    for i in range(n_subjects):
        cov = {
            "WT": float(wts[i]),
            "AGE": float(ages[i]),
            "CRCL": float(crcls[i]),
            "EGFR": float(crcls[i]),
        }
        theta_i = apply_cov(typical, cov)
        params_i = {k: float(theta_i[k]) for k in theta_i}
        for p in iiv_params:
            if p in params_i and sds.get(p, 0.0) > 0:
                params_i[p] = params_i[p] * float(np.exp(rng.normal(0.0, sds[p])))
        if is_ped and float(ages[i]) < 2.0 and "CL" in params_i:
            params_i["CL"] = params_i["CL"] * tm50_maturation(float(pmas[i]))

        sim = simulate_timecourse(
            model,
            params_i,
            dose=dose,
            tau=tau,
            n_doses=n_doses,
            tmax=tmax,
            n_points=n_points,
            wt=float(wts[i]),
        )
        m = _interval_metrics(sim["times"], sim["cp"], t_last=t_last, tau=tau, tmax=tmax)
        aucs[i] = float(m["auc_tau"])
        cmaxs[i] = float(m["cmax"])
        ctroughs[i] = float(m["ctrough"])

    return {"auc": aucs, "cmax": cmaxs, "ctrough": ctroughs}


def _in_target_window(exposures: dict[str, np.ndarray], target_window: dict[str, Any]) -> np.ndarray:
    """Boolean mask of virtual subjects achieving the target efficacy window."""
    metric = str(target_window.get("metric") or "auc").lower()
    if metric == "auc_tau":
        metric = "auc"
    vals = exposures.get(metric)
    if vals is None:
        vals = exposures["auc"]
    low = target_window.get("low")
    high = target_window.get("high")
    mask = np.ones(len(vals), dtype=bool)
    if low is not None:
        mask &= vals >= float(low)
    if high is not None:
        mask &= vals <= float(high)
    return mask


def run_trial_pta(
    source_fit: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Simulate virtual subjects across replicates for every (band, regimen) in ``plan``.

    Computes PTA per band per regimen with Monte Carlo standard error (``mc_se``)
    across replicates, exposure percentile summaries, peak/trough limit checks,
    and per-band regimen recommendations.
    """
    seed = int(plan.get("seed", 42))
    n_subjects = int(plan.get("n_subjects", 80))
    replicates = int(plan.get("replicates", 10))
    target_window = dict(plan.get("target_window") or {})
    target_pta = float(plan.get("target_pta", 0.75))

    model_key = _resolve_model_key(source_fit.get("model_key"))
    theta = dict(source_fit.get("theta") or {})
    omega_cv_pct = dict(source_fit.get("omega_cv_pct") or {})
    iiv_params = list(source_fit.get("iiv_params") or list(omega_cv_pct.keys()))
    cov_effects = list(source_fit.get("covariate_effects") or [])

    rng = np.random.default_rng(seed)

    cmax_max = target_window.get("cmax_max")
    ctrough_min = target_window.get("ctrough_min")

    rows: list[dict[str, Any]] = []
    for band in plan.get("bands") or []:
        bname = str(band.get("name") or "band")
        for reg in plan.get("regimens") or []:
            rlabel = str(reg.get("label") or f"{reg.get('dose_mg', 100)}mg")
            rep_ptas: list[float] = []
            all_auc: list[np.ndarray] = []
            all_cmax: list[np.ndarray] = []
            all_ctrough: list[np.ndarray] = []

            for _ in range(max(1, replicates)):
                exp = _simulate_band_regimen_cohort(
                    model_key=model_key,
                    theta=theta,
                    omega_cv_pct=omega_cv_pct,
                    iiv_params=iiv_params,
                    covariate_effects=cov_effects,
                    band=band,
                    regimen=reg,
                    n_subjects=n_subjects,
                    rng=rng,
                )
                hit = _in_target_window(exp, target_window)
                rep_ptas.append(float(np.mean(hit)))
                all_auc.append(exp["auc"])
                all_cmax.append(exp["cmax"])
                all_ctrough.append(exp["ctrough"])

            auc_cat = np.concatenate(all_auc)
            cmax_cat = np.concatenate(all_cmax)
            ctrough_cat = np.concatenate(all_ctrough)

            pta_mean = round(float(np.mean(rep_ptas)), 4)
            mc_se = (
                round(float(np.std(rep_ptas, ddof=1) / math.sqrt(len(rep_ptas))), 4)
                if len(rep_ptas) > 1
                else None
            )

            auc_p05, auc_p50, auc_p95 = [round(float(x), 4) for x in np.percentile(auc_cat, [5, 50, 95])]
            cmax_p05, cmax_p50, cmax_p95 = [round(float(x), 4) for x in np.percentile(cmax_cat, [5, 50, 95])]
            ctr_p05, ctr_p50, ctr_p95 = [round(float(x), 4) for x in np.percentile(ctrough_cat, [5, 50, 95])]

            peak_limit_failed = bool(cmax_max is not None and cmax_p50 > float(cmax_max))
            trough_limit_failed = bool(ctrough_min is not None and ctr_p50 < float(ctrough_min))

            rows.append({
                "band": bname,
                "regimen": rlabel,
                "dose_mg": float(reg.get("dose_mg") or reg.get("dose") or 100.0),
                "tau_h": float(reg.get("tau_h") or reg.get("tau") or 24.0),
                "pta": pta_mean,
                "mc_se": mc_se,
                "replicates": len(rep_ptas),
                "n_subjects": n_subjects,
                "pta_ci_lo": None,
                "pta_ci_hi": None,
                "auc": {"p05": auc_p05, "median": auc_p50, "p95": auc_p95},
                "cmax": {
                    "p05": cmax_p05,
                    "median": cmax_p50,
                    "p95": cmax_p95,
                    "limit_max": float(cmax_max) if cmax_max is not None else None,
                    "limit_failed": peak_limit_failed,
                },
                "ctrough": {
                    "p05": ctr_p05,
                    "median": ctr_p50,
                    "p95": ctr_p95,
                    "limit_min": float(ctrough_min) if ctrough_min is not None else None,
                    "limit_failed": trough_limit_failed,
                },
                "peak_limit_failed": peak_limit_failed,
                "trough_limit_failed": trough_limit_failed,
            })

    recommendations: list[dict[str, Any]] = []
    for band in plan.get("bands") or []:
        bname = str(band.get("name") or "band")
        b_rows = [r for r in rows if r["band"] == bname]
        feasible = [
            r for r in b_rows
            if r["pta"] >= target_pta and not r["peak_limit_failed"] and not r["trough_limit_failed"]
        ]
        if feasible:
            chosen = max(feasible, key=lambda r: (r["pta"], -r["dose_mg"]))
            recommendations.append({
                "band": bname,
                "recommended_regimen": chosen["regimen"],
                "dose_mg": chosen["dose_mg"],
                "tau_h": chosen["tau_h"],
                "pta": chosen["pta"],
                "mc_se": chosen["mc_se"],
                "driving_metric": f"{target_window.get('metric', 'auc')} target window + cmax/ctrough limits",
                "peak_limit_failed": False,
                "trough_limit_failed": False,
            })
        elif b_rows:
            best = max(b_rows, key=lambda r: r["pta"])
            recommendations.append({
                "band": bname,
                "recommended_regimen": None,
                "candidate_regimen": best["regimen"],
                "pta": best["pta"],
                "mc_se": best["mc_se"],
                "driving_metric": (
                    "cmax/ctrough safety limit exceeded"
                    if (best["peak_limit_failed"] or best["trough_limit_failed"])
                    else f"{target_window.get('metric', 'auc')} below target PTA ({target_pta:.0%})"
                ),
                "peak_limit_failed": best["peak_limit_failed"],
                "trough_limit_failed": best["trough_limit_failed"],
            })

    return {
        "status": "ok",
        "seed": seed,
        "replicates": replicates,
        "n_subjects": n_subjects,
        "target_window": target_window,
        "target_pta": target_pta,
        "rows": rows,
        "recommendations": recommendations,
    }


def run_plan_special_populations(
    source_fit: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Run existing special-population and pediatric simulations strictly for
    the bands named in ``plan['bands']``, seeded by ``plan['seed']``."""
    seed = int(plan.get("seed", 42))
    model_key = _resolve_model_key(source_fit.get("model_key"))
    theta = dict(source_fit.get("theta") or {})
    omega_cv_pct = dict(source_fit.get("omega_cv_pct") or {})
    iiv_params = list(source_fit.get("iiv_params") or list(omega_cv_pct.keys()))
    cov_effects = list(source_fit.get("covariate_effects") or [])

    regimens = plan.get("regimens") or [{"dose_mg": 100.0, "tau_h": 24.0, "n_doses": 5}]
    doses = sorted({float(r.get("dose_mg") or r.get("dose") or 100.0) for r in regimens})
    tau = float(regimens[0].get("tau_h") or regimens[0].get("tau") or 24.0)
    n_doses = int(regimens[0].get("n_doses", 5))
    n_per = max(20, int(plan.get("n_subjects", 80)))

    rng = np.random.default_rng(seed)
    plan_bands = list(plan.get("bands") or [])

    # Build synthetic cohort strictly for the bands named in the plan
    cov_rows: list[dict[str, Any]] = []
    wt_rows: list[float] = []
    for band in plan_bands:
        bname = str(band.get("name") or "band")
        wt_r = band.get("wt_kg") or [70.0, 70.0]
        age_r = band.get("age_yr") or [40.0, 40.0]
        crcl_r = band.get("crcl_ml_min") or [90.0, 90.0]
        for _ in range(n_per):
            w = float(rng.uniform(float(wt_r[0]), float(wt_r[1])))
            a = float(rng.uniform(float(age_r[0]), float(age_r[1])))
            c = float(rng.uniform(float(crcl_r[0]), float(crcl_r[1])))
            cov_rows.append({"BAND": bname, "WT": round(w, 1), "AGE": round(a, 1), "EGFR": round(c, 1)})
            wt_rows.append(round(w, 1))

    ref_band_name = str(plan_bands[0].get("name")) if plan_bands else "standard_adult"
    sp_out = special_population_simulation(
        model_key,
        theta=theta,
        omega_cv_pct=omega_cv_pct,
        iiv_params=iiv_params,
        cov_rows=cov_rows,
        wt_rows=wt_rows,
        stratify_by="BAND",
        doses=doses,
        tau=tau,
        n_doses=n_doses,
        covariate_effects=cov_effects,
        reference_stratum=ref_band_name,
        n_per_stratum=n_per,
        seed=seed,
    )

    # Filter strata strictly to plan band names in plan order
    allowed_names = [str(b.get("name")) for b in plan_bands]
    strata_by_name = {s["label"]: s for s in (sp_out.get("strata") or []) if s.get("label") in allowed_names}
    ordered_strata = [strata_by_name[name] for name in allowed_names if name in strata_by_name]

    pediatric_out = None
    has_pediatric = any(
        bool(b.get("is_pediatric")) or float((b.get("age_yr") or [18.0, 65.0])[0]) < 18.0
        for b in plan_bands
    )
    if has_pediatric:
        ped_cov, ped_wt = pediatric_reference_population(n=600, seed=seed)
        adult_cov, adult_wt = reference_population(n=200, seed=seed)
        # Reference adult exposures from first regimen dose
        ref_exp_sim = _simulate_band_regimen_cohort(
            model_key=model_key,
            theta=theta,
            omega_cv_pct=omega_cv_pct,
            iiv_params=iiv_params,
            covariate_effects=cov_effects,
            band={"wt_kg": [55.0, 95.0], "age_yr": [18.0, 65.0], "crcl_ml_min": [90.0, 120.0]},
            regimen=regimens[0],
            n_subjects=100,
            rng=np.random.default_rng(seed),
        )
        pediatric_out = pediatric_simulation(
            model_key,
            theta=theta,
            omega_cv_pct=omega_cv_pct,
            iiv_params=iiv_params,
            cov_rows=ped_cov,
            wt_rows=ped_wt,
            reference_exposures={"auc_tau": ref_exp_sim["auc"].tolist(), "cmax": ref_exp_sim["cmax"].tolist()},
            doses=doses,
            tau=tau,
            n_doses=n_doses,
            covariate_effects=cov_effects,
            n_per_stratum=min(100, n_per),
            seed=seed,
        )

    return {
        "status": "ok",
        "seed": seed,
        "bands_simulated": allowed_names,
        "strata": ordered_strata,
        "reference_stratum": ref_band_name,
        "reference_band": sp_out.get("reference_band"),
        "pediatric": pediatric_out,
    }


def _engine_intervals(method: str, result: dict[str, Any] | None,
                      ) -> dict[str, tuple[float, float, float]] | None:
    """Per-parameter (centre, lower, upper) read from a REAL engine result stored in
    state, using that engine's own keys. None when the result is absent, not ``ok``
    or carries none of those keys -- never a reconstruction from RSE."""
    keys = _ENGINE_INTERVAL_KEYS.get(method)
    if not keys or not isinstance(result, dict) or result.get("status") != "ok":
        return None
    out: dict[str, tuple[float, float, float]] = {}
    for rec in result.get("parameters") or []:
        name = rec.get("parameter")
        try:
            c, lo, hi = (float(rec.get(k)) for k in keys)
        except (TypeError, ValueError):
            continue
        if name and all(math.isfinite(x) and x > 0 for x in (c, lo, hi)) and hi > lo:
            out[str(name)] = (c, lo, hi)
    return out or None


def _lognormal_draws(centres: dict[str, tuple[float, float]], fixed: dict[str, float],
                     n_draws: int, rng: np.random.Generator) -> list[dict[str, float]]:
    """Independent log-normal draws: ``centres`` maps a parameter to (log-centre,
    log-sd); ``fixed`` parameters stay at their point estimate."""
    draws: list[dict[str, float]] = []
    for _ in range(n_draws):
        d = {p: float(math.exp(rng.normal(mu, sd))) for p, (mu, sd) in centres.items()}
        d.update(fixed)
        draws.append(d)
    return draws


def _obtain_parameter_draws(
    source_fit: dict[str, Any],
    method: str,
    n_draws: int,
    seed: int,
    existing_bootstrap: dict[str, Any] | None = None,
    existing_sir: dict[str, Any] | None = None,
    existing_profile: dict[str, Any] | None = None,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    """Parameter draws for the outer uncertainty loop, labelled by what they ARE.

    * ``method`` is sir / bootstrap / profile AND that engine's result is stored in
      state -> log-normal draws matched to the engine's own per-parameter interval
      (``approximation: False``, ``interval_source: <method>_results``).
    * otherwise -> ``asymptotic_lognormal``: a diagonal log-normal from the fit's
      theta RSE% (``approximation: True``; ``requested_method`` is kept). The
      requested engine is never run here and its name is never attached to the
      result.
    * no finite RSE at all -> ``status: unverifiable`` and no draws.
    """
    method = str(method or ASYMPTOTIC_METHOD).lower()
    if method not in UNCERTAINTY_METHODS:
        raise ValueError(f"Unsupported uncertainty_method: {method!r}")
    theta = {k: float(v) for k, v in (source_fit.get("theta") or {}).items()}
    rse_pct = {k: float(v) for k, v in (source_fit.get("theta_rse_pct") or {}).items()
               if v is not None and math.isfinite(float(v)) and float(v) > 0}
    param_names = list(theta)
    rng = np.random.default_rng(seed)
    existing = {"sir": existing_sir, "bootstrap": existing_bootstrap, "profile": existing_profile}

    if method in ENGINE_METHODS:
        ivals = _engine_intervals(method, existing.get(method))
        if ivals:
            centres = {
                p: (math.log(c), max((math.log(hi) - math.log(lo)) / (2.0 * 1.96), 1e-4))
                for p, (c, lo, hi) in ivals.items() if p in theta
            }
            fixed = {p: theta[p] for p in param_names if p not in centres}
            summary = {
                "status": "ok", "method": method, "requested_method": method,
                "approximation": False, "interval_source": f"{method}_results",
                "parameters_drawn": sorted(centres), "parameters_fixed": sorted(fixed),
                "note": (f"log-normal draws matched to the stored {method} per-parameter "
                         "interval (centre and 95% bounds); between-parameter correlation "
                         "is not carried"),
            }
            return _lognormal_draws(centres, fixed, n_draws, rng), summary

    # Asymptotic approximation: requested explicitly, or the fallback when the
    # requested engine has no stored result.
    drawable = [p for p in param_names if p in rse_pct]
    fixed = {p: theta[p] for p in param_names if p not in rse_pct}
    base = {"method": ASYMPTOTIC_METHOD, "requested_method": method, "approximation": True,
            "interval_source": "source_fit.theta_rse_pct",
            "parameters_drawn": drawable, "parameters_fixed": sorted(fixed)}
    if not drawable:
        asked = "" if method == ASYMPTOTIC_METHOD else f" and no {method} result is stored in state"
        return [], {**base, "status": "unverifiable",
                    "reason": (f"the source fit carries no finite theta RSE%{asked}; "
                               "parameter uncertainty cannot be propagated without inventing it")}
    if method == ASYMPTOTIC_METHOD:
        note = "diagonal log-normal from the fit's asymptotic theta RSE%, as the approved plan requested"
    else:
        note = (f"diagonal log-normal from the fit's asymptotic theta RSE%: no {method} result "
                f"is stored in state, so this is NOT a {method} interval")
    centres = {p: (math.log(max(theta[p], 1e-9)), max(rse_pct[p] / 100.0, 1e-4)) for p in drawable}
    return _lognormal_draws(centres, fixed, n_draws, rng), {**base, "status": "ok", "note": note}

def propagate_uncertainty_to_pta(
    source_fit: dict[str, Any],
    plan: dict[str, Any],
    pta_result: dict[str, Any] | None = None,
    n_draws: int = 20,
    existing_bootstrap: dict[str, Any] | None = None,
    existing_sir: dict[str, Any] | None = None,
    existing_profile: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Propagate parameter uncertainty as an outer loop into PTA per band per regimen.

    The payload's ``method`` is what the draws really are (see
    ``_obtain_parameter_draws``); ``requested_method`` is what the plan asked for.
    With no usable uncertainty source the payload is ``status: unverifiable`` and
    the PTA table is returned without intervals.
    Returns ``(uncertainty_payload, updated_pta_result)``.
    """
    requested = str(plan.get("uncertainty_method") or ASYMPTOTIC_METHOD).lower()
    seed = int(plan.get("seed", 42))
    n_draws = max(5, int(n_draws))

    if pta_result is None or pta_result.get("status") != "ok":
        pta_result = run_trial_pta(source_fit, plan)
    else:
        pta_result = json.loads(json.dumps(pta_result))

    theta_draws, engine_summary = _obtain_parameter_draws(
        source_fit=source_fit,
        method=requested,
        n_draws=n_draws,
        seed=seed,
        existing_bootstrap=existing_bootstrap,
        existing_sir=existing_sir,
        existing_profile=existing_profile,
    )
    if engine_summary.get("status") == "unverifiable":
        payload = {
            "status": "unverifiable", "method": engine_summary["method"],
            "requested_method": requested, "approximation": True,
            "reason": engine_summary["reason"], "seed": seed, "n_draws": 0,
            "engine_summary": engine_summary, "intervals": [],
        }
        return payload, pta_result
    method = str(engine_summary["method"])   # what the draws really are

    model_key = _resolve_model_key(source_fit.get("model_key"))
    omega_cv_pct = dict(source_fit.get("omega_cv_pct") or {})
    iiv_params = list(source_fit.get("iiv_params") or list(omega_cv_pct.keys()))
    cov_effects = list(source_fit.get("covariate_effects") or [])
    target_window = dict(plan.get("target_window") or {})
    n_subjects = max(20, min(int(plan.get("n_subjects", 80)), 80))

    intervals: list[dict[str, Any]] = []
    rows_by_key = {(r["band"], r["regimen"]): r for r in (pta_result.get("rows") or [])}

    for b_idx, band in enumerate(plan.get("bands") or []):
        bname = str(band.get("name") or "band")
        for r_idx, reg in enumerate(plan.get("regimens") or []):
            rlabel = str(reg.get("label") or f"{reg.get('dose_mg', 100)}mg")
            draw_ptas: list[float] = []
            for d_idx, theta_draw in enumerate(theta_draws):
                draw_rng = np.random.default_rng(seed + 10_000 + b_idx * 1000 + r_idx * 100 + d_idx)
                exp = _simulate_band_regimen_cohort(
                    model_key=model_key,
                    theta=theta_draw,
                    omega_cv_pct=omega_cv_pct,
                    iiv_params=iiv_params,
                    covariate_effects=cov_effects,
                    band=band,
                    regimen=reg,
                    n_subjects=n_subjects,
                    rng=draw_rng,
                    n_points=64,
                )
                hit = _in_target_window(exp, target_window)
                draw_ptas.append(float(np.mean(hit)))

            lo, med, hi = [round(float(x), 4) for x in np.percentile(draw_ptas, [2.5, 50.0, 97.5])]
            row = rows_by_key.get((bname, rlabel))
            if row is not None:
                row["pta_ci_lo"] = lo
                row["pta_ci_hi"] = hi
                row["pta_median_unc"] = med
                row["uncertainty_method"] = method

            intervals.append({
                "band": bname,
                "regimen": rlabel,
                "pta_point": row["pta"] if row else med,
                "pta_median": med,
                "pta_ci_lo": lo,
                "pta_ci_hi": hi,
                "n_draws": len(draw_ptas),
            })

    # Attach CI to recommendations as well
    int_by_key = {(x["band"], x["regimen"]): x for x in intervals}
    for rec in pta_result.get("recommendations") or []:
        rname = rec.get("recommended_regimen") or rec.get("candidate_regimen")
        iv = int_by_key.get((rec.get("band"), rname))
        if iv:
            rec["pta_ci_lo"] = iv["pta_ci_lo"]
            rec["pta_ci_hi"] = iv["pta_ci_hi"]
            rec["uncertainty_method"] = method

    uncertainty_payload = {
        "status": "ok",
        "method": method,
        "requested_method": requested,
        "approximation": bool(engine_summary.get("approximation")),
        "seed": seed,
        "n_draws": len(theta_draws),
        "engine_summary": engine_summary,
        "intervals": intervals,
    }
    return uncertainty_payload, pta_result
