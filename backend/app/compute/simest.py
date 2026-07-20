"""Simulation-estimation: a bounded precision CHECK for a proposed single-arm
PK sampling design, evaluated against an already-fitted population model.

For each of ``n_rep`` replicates: simulate a trial under ``design`` from the
population model in ``nlme_result`` (typical values, Omega, residual error),
then re-fit the SAME structural model to the simulated data via the
CALLER-INJECTED ``fit_fn`` (this module never runs an estimator itself). The
per-replicate 95% CI is checked against the precision criterion from the
course's own Week 7 lecture slide (verbatim): "the study must be prospectively
powered to target a 95% CI within 60% and 140% of the geometric mean estimates
of clearance and volume of distribution ... with at least 80% power" -- i.e. a
SELF-REFERENTIAL precision check (each replicate's CI vs ITS OWN point
estimate), not a comparison to the simulating truth. [CITATION UNVERIFIED]:
this criterion is attributed to FDA pediatric study-design guidance by the
course lecture; the primary FDA source has not been independently fetched and
verified (see the project's fetch-don't-recall citation rule) -- do not
present it as a confirmed regulatory requirement without doing so.

Cost is real: a single FOCE-I fit on a ~12-subject cohort is roughly 40s (see
docs/WASM_BROWSER_NATIVE_SPEC.md), and `compute_uncertainty=True` (required
here, for the per-replicate CI) adds further OFV-Hessian passes. A 20-60
subject design several times that. Default n_rep=5 -> roughly 8-25 minutes;
the hard cap n_rep<=10 -> roughly 15-50 minutes. This is a BOUNDED CHECK, not
a publication-grade simulation-estimation study (which typically uses
200-1000 replicates) -- at n_rep=5 a reported pass-rate carries a binomial
standard error of roughly +/-22 percentage points, and `ci_validity` is
reported "unassessable" whenever the evaluable-replicate count is below 30
(true at every setting this cap allows).

SCOPE: covariate models (continuous or categorical) are NOT supported and are
REJECTED outright, not silently ignored -- see `run_simest`'s docstring.
Only structural-parameter precision is evaluated. "Pediatric" is deliberately
absent from this module's naming: the design object represents one fixed
sampling schedule and one dosing regimen for every simulated subject, with no
age/weight-cohort stratification, no per-cohort sample allocation, no
randomized sampling windows, and no dropout model -- see DESIGN_LIMITATIONS.

fit_fn is INJECTED so the expensive replicate loop is owned by the caller (a
background job, per the project's guardrail against submitting a real
NLME/SCM fit from an automated loop) and is trivially mockable: every test in
this module's test suite uses a fake fit_fn and never runs a real fit.
"""
from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np

from app.compute.nlme import cv_pct_to_omega2, simulate_replicate_subject

_ROUND_DP = 6
_MAX_REPLICATES = 10
_DEFAULT_N_REP = 5
_MAX_SECONDS = 30.0 * 60.0  # hard ceiling regardless of the caller's max_seconds
_MIN_R_FOR_ASSESSABLE_CI = 30  # below this, coverage/pass-rate CIs are unassessable
_Z95 = 1.9599639845400545  # norm.ppf(0.975)

CITATION_UNVERIFIED = (
    "[CITATION UNVERIFIED] The 95%-CI-within-60-140%-of-the-estimate, "
    "80%-power criterion is attributed to FDA pediatric study-design guidance "
    "by the course lecture this tool is based on. The primary FDA source has "
    "NOT been independently fetched and verified -- do not present this as a "
    "confirmed regulatory requirement.")

DESIGN_LIMITATIONS = [
    "Single sampling schedule (`design.obs_t`) applied identically to every "
    "simulated subject -- no randomized sampling windows or per-cohort schedules.",
    "Single dosing regimen (fixed absolute dose, or a fixed mg/kg dose scaled by "
    "each subject's simulated weight) -- no age/weight-cohort stratification, "
    "no per-cohort sample allocation, no dropout/missing-sample model.",
    "Covariate models (continuous or categorical) are not supported and are "
    "REJECTED outright -- only structural-parameter precision is evaluated.",
    "This is a bounded precision CHECK (up to 10 replicates), not a "
    "publication-grade simulation-estimation study (typically 200-1000 "
    "replicates); pass-rate estimates carry wide sampling error at this scale.",
]


def _passes_precision_criterion(theta_val: float, lo: float, hi: float) -> bool:
    """95% CI within 60%-140% of the replicate's OWN point estimate."""
    if theta_val <= 0.0:
        return False
    return (lo / theta_val) >= 0.6 and (hi / theta_val) <= 1.4


def _wilson_ci(k: int, n: int, z: float = _Z95) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1.0 - phat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _round(x: float | None, dp: int = _ROUND_DP) -> float | None:
    if x is None or not math.isfinite(x):
        return None
    return round(float(x), dp)


def _validate_design(design: dict[str, Any]) -> str | None:
    """Return an error message, or None if the design is well-formed."""
    n_subj = design.get("n_subjects")
    if not isinstance(n_subj, int) or n_subj < 2:
        return "design.n_subjects must be an integer >= 2"
    obs_t = design.get("obs_t")
    if not obs_t or not all(isinstance(t, (int, float)) and t > 0 for t in obs_t):
        return "design.obs_t must be a non-empty list of positive sampling times"
    dose, dose_per_kg = design.get("dose"), design.get("dose_per_kg")
    if (dose is None) == (dose_per_kg is None):
        return "design must specify exactly one of dose or dose_per_kg"
    if dose is not None and not (isinstance(dose, (int, float)) and dose > 0):
        return "design.dose must be a positive number"
    if dose_per_kg is not None and not (isinstance(dose_per_kg, (int, float)) and dose_per_kg > 0):
        return "design.dose_per_kg must be a positive number"
    n_doses = design.get("n_doses", 1)
    if not isinstance(n_doses, int) or n_doses < 1:
        return "design.n_doses must be an integer >= 1"
    if n_doses > 1:
        tau = design.get("tau")
        if not (isinstance(tau, (int, float)) and tau > 0):
            return "design.tau must be a positive number when n_doses > 1"
    return None


def run_simest(model_key: str, design: dict[str, Any], nlme_result: dict[str, Any], *,
               fit_fn: Callable[[list[dict], int], dict],
               n_rep: int = _DEFAULT_N_REP, params: tuple[str, ...] | None = None,
               ci_target_pct: float | None = None,
               seed: int = 20250614, max_seconds: float = _MAX_SECONDS,
               progress: Callable[[dict], None] | None = None) -> dict[str, Any]:
    """Simulation-estimation precision check (see module docstring).

    Parameters
    ----------
    model_key:
        Structural model to simulate and re-fit.
    design:
        ``{"n_subjects", "obs_t", "dose"|"dose_per_kg", "n_doses"=1, "tau",
        "wt_mean"=70.0, "wt_cv_pct"=20.0, "lloq"}``. Raises ``ValueError`` when
        malformed (a caller/config bug, not a runtime data condition).
    nlme_result:
        A converged NLME fit result carrying ``theta``, ``omega_cv_pct``,
        ``sigma``, ``iiv_params``, ``error_model``, ``covariate_effects``.
    fit_fn:
        ``(simulated_subjects, seed) -> fit_result_dict``, SAME shape as
        ``app.compute.nlme.population_fit``'s return (must carry ``status``,
        ``theta``, ``theta_rse_pct``, ``converged``). Owns the actual
        estimator call -- this module supplies data, never fits it.
    n_rep:
        Requested replicate count; clamped to ``[1, 10]`` (``_MAX_REPLICATES``).
    params:
        Structural parameters to evaluate; defaults to
        ``nlme_result["iiv_params"] ∩ nlme_result["theta"].keys()``. A name
        outside ``theta`` (e.g. a covariate coefficient) is dropped, not
        silently misread as a raw-scale RSE.
    ci_target_pct:
        Optional target percentage for ``criterion_met`` (e.g. 80.0, the
        course lecture's own target -- NOT hardcoded here, since presenting a
        specific number as authoritative without independent verification
        would repeat the citation problem this module already flags).
    max_seconds:
        Wall-clock budget for the WHOLE replicate loop; capped at
        ``_MAX_SECONDS`` regardless of what the caller passes.
    progress:
        Optional callback invoked after each completed replicate with
        ``{"replicate", "n_rep_planned", "elapsed_s"}`` -- for a background
        job to report status; never required for correctness.

    Returns
    -------
    A JSON-safe dict; never raises for a runtime data condition (missing
    NLME fit, covariate model present, zero evaluable replicates) -- those
    return ``{"status": ..., "message": ...}`` instead. See the module
    docstring for the honest cost estimate before calling this with a large
    design.
    """
    if (nlme_result or {}).get("status") != "ok":
        return {"status": "no_nlme",
                "message": "Needs a converged run_nlme fit to supply theta/Omega/sigma."}
    if nlme_result.get("covariate_effects"):
        return {"status": "covariates_unsupported",
                "message": ("Simulation-estimation does not support covariate models "
                            "(continuous or categorical): a categorical effect's "
                            "reference level is re-derived per replicate from the "
                            "modal simulated level (nlme.py, see _build_cov_effects), "
                            "which silently redefines what theta means across "
                            "replicates; a continuous effect held at its fitted center "
                            "for every simulated subject would make its coefficient "
                            "unidentifiable in the refit. Fit a covariate-free NLME "
                            "model to use this feature."),
                "design_limitations": list(DESIGN_LIMITATIONS)}

    err = _validate_design(design)
    if err is not None:
        raise ValueError(err)

    theta = dict(nlme_result.get("theta") or {})
    iiv_params_fit = list(nlme_result.get("iiv_params") or [])
    omega_cv = dict(nlme_result.get("omega_cv_pct") or {})
    sig = nlme_result.get("sigma") or {}
    sigma_prop = float(sig.get("prop") or 0.0)
    sigma_add = float(sig.get("add") or 0.0)
    error_model = nlme_result.get("error_model", "proportional")

    if params is None:
        resolved_params = tuple(p for p in iiv_params_fit if p in theta)
    else:
        resolved_params = tuple(p for p in params if p in theta)
    if not resolved_params:
        return {"status": "no_params",
                "message": (f"No usable structural parameters; available theta: "
                            f"{sorted(theta.keys())}.")}

    n_rep_requested = int(n_rep)
    n_rep_planned = max(1, min(n_rep_requested, _MAX_REPLICATES))
    max_seconds = min(float(max_seconds), _MAX_SECONDS)
    omega2 = {p: cv_pct_to_omega2(cv) for p, cv in omega_cv.items()}

    n_subj = int(design["n_subjects"])
    obs_t = list(design["obs_t"])
    n_doses = int(design.get("n_doses", 1))
    tau = float(design.get("tau") or 0.0)
    wt_mean = float(design.get("wt_mean", 70.0))
    wt_cv_pct = float(design.get("wt_cv_pct", 20.0))
    wt_sd_log = math.sqrt(cv_pct_to_omega2(wt_cv_pct)) if wt_cv_pct > 0 else 0.0
    lloq = design.get("lloq")
    dose, dose_per_kg = design.get("dose"), design.get("dose_per_kg")

    rng = np.random.default_rng(seed)
    t0 = time.monotonic()
    per_rep_seconds: list[float] = []
    excluded_reasons: Counter[str] = Counter()
    evaluated: list[dict[str, Any]] = []
    n_resampled_total = 0
    n_negative_draws_total = 0
    n_rep_completed = 0
    est_minutes_rough: float | None = None

    for r in range(n_rep_planned):
        if time.monotonic() - t0 > max_seconds:
            excluded_reasons["time_budget"] += (n_rep_planned - r)
            break
        rep_t0 = time.monotonic()

        sim_subjects: list[dict[str, Any]] = []
        for i in range(n_subj):
            wt_i = wt_mean * math.exp(rng.normal(0.0, wt_sd_log)) if wt_sd_log > 0 else wt_mean
            amt = float(dose_per_kg) * wt_i if dose_per_kg is not None else float(dose)
            doses = [{"time": k * tau, "amt": amt} for k in range(n_doses)]
            out = simulate_replicate_subject(
                model_key, theta, omega2, sigma_prop, sigma_add, iiv_params_fit,
                error_model, doses, obs_t, wt_i, rng, lloq=lloq)
            n_resampled_total += out["n_resampled"]
            n_negative_draws_total += out["n_negative_draws"]
            if not out["ok"]:
                continue
            subj: dict[str, Any] = {"subject": f"R{r}S{i}", "doses": doses,
                                    "obs_t": out["obs_t"], "obs_c": out["obs_c"], "wt": wt_i}
            if lloq is not None:
                subj["obs_blq"] = out["obs_blq"]
                subj["lloq"] = out["lloq"]
            sim_subjects.append(subj)

        if len(sim_subjects) < 2:
            excluded_reasons["too_few_simulated_subjects"] += 1
            continue

        try:
            fit_res = fit_fn(sim_subjects, seed + r + 1)
        except Exception:
            excluded_reasons["fit_exception"] += 1
            continue

        n_rep_completed += 1
        per_rep_seconds.append(time.monotonic() - rep_t0)
        if est_minutes_rough is None and per_rep_seconds:
            est_minutes_rough = round(per_rep_seconds[0] * n_rep_planned / 60.0, 2)
        if progress:
            progress({"replicate": r + 1, "n_rep_planned": n_rep_planned,
                      "elapsed_s": round(time.monotonic() - t0, 1)})

        if not fit_res or fit_res.get("status") != "ok":
            excluded_reasons["fit_not_ok"] += 1
            continue
        theta_hat = fit_res.get("theta") or {}
        if not all(p in theta_hat and math.isfinite(float(theta_hat[p])) for p in resolved_params):
            excluded_reasons["theta_missing"] += 1
            continue
        theta_hat_r = {p: float(theta_hat[p]) for p in resolved_params}

        rse = fit_res.get("theta_rse_pct") or {}
        ci_ok = all(p in rse and rse[p] is not None and math.isfinite(float(rse[p]))
                    for p in resolved_params)
        ci_r = None
        if ci_ok:
            ci_r = {}
            for p in resolved_params:
                se_log = float(rse[p]) / 100.0
                th = theta_hat_r[p]
                ci_r[p] = (th * math.exp(-_Z95 * se_log), th * math.exp(_Z95 * se_log), se_log)
        else:
            excluded_reasons["rse_unavailable"] += 1
        evaluated.append({"theta": theta_hat_r, "ci": ci_r})

    r_point = len(evaluated)
    ci_evaluated = [e for e in evaluated if e["ci"] is not None]
    r_ci = len(ci_evaluated)

    per_param: dict[str, Any] = {}
    for p in resolved_params:
        truth_p = float(theta[p])
        point_vals = np.array([e["theta"][p] for e in evaluated], dtype=float)
        gm = float(np.exp(np.mean(np.log(point_vals)))) if point_vals.size else None
        rel_bias_pct = 100.0 * (gm / truth_p - 1.0) if gm is not None else None
        rmse_pct = (100.0 * math.sqrt(float(np.mean((point_vals / truth_p - 1.0) ** 2)))
                   if point_vals.size else None)
        cv_across_pct = (100.0 * float(np.std(point_vals, ddof=1)) / float(np.mean(point_vals))
                         if point_vals.size >= 2 else None)

        n_pass = sum(1 for e in ci_evaluated
                    if _passes_precision_criterion(e["theta"][p], *e["ci"][p][:2]))
        pct_own_ci = 100.0 * n_pass / r_ci if r_ci else None
        pct_strict = 100.0 * n_pass / n_rep_planned
        wlo, whi = _wilson_ci(n_pass, r_ci) if r_ci else (None, None)
        mean_se_log = (float(np.mean([e["ci"][p][2] for e in ci_evaluated]))
                       if ci_evaluated else None)

        per_param[p] = {
            "truth": _round(truth_p),
            "gm_point_estimate": _round(gm),
            "rel_bias_pct": _round(rel_bias_pct, 4),
            "rmse_pct": _round(rmse_pct, 4),
            "cv_across_replicates_pct": _round(cv_across_pct, 4),
            "n_pass_precision_criterion": n_pass,
            "pct_within_60_140_of_own_estimate": _round(pct_own_ci, 4),
            "pct_within_60_140_strict": _round(pct_strict, 4),
            "coverage_wilson_ci_pct": [_round(wlo * 100 if wlo is not None else None, 4),
                                       _round(whi * 100 if whi is not None else None, 4)],
            "mean_theta_se_log": _round(mean_se_log),
        }

    n_pass_joint = sum(1 for e in ci_evaluated
                       if all(_passes_precision_criterion(e["theta"][p], *e["ci"][p][:2])
                             for p in resolved_params))
    pct_joint_strict = _round(100.0 * n_pass_joint / n_rep_planned, 4)
    pct_joint_own_ci = _round(100.0 * n_pass_joint / r_ci, 4) if r_ci else None
    criterion_met = (pct_joint_strict >= float(ci_target_pct)
                     if ci_target_pct is not None and pct_joint_strict is not None else None)

    # Rate of negative residual draws PER FINAL OBSERVATION (can exceed 1.0
    # under a pathologically mismatched error model, which is the point --
    # that is exactly what should trip the "partial" flag below).
    total_obs = max(1, n_subj * n_rep_completed * len(obs_t))
    negative_draw_rate = n_negative_draws_total / total_obs
    status = "ok"
    if r_point == 0:
        status = "not_evaluable"
    elif negative_draw_rate > 0.01:
        status = "partial"

    replicates_out = [
        {"theta": {p: _round(v) for p, v in e["theta"].items()},
        "ci": ({p: [_round(e["ci"][p][0]), _round(e["ci"][p][1])] for p in resolved_params}
              if e["ci"] is not None else None)}
        for e in evaluated
    ]

    return {
        "status": status,
        "model_key": model_key,
        "params": list(resolved_params),
        "rse_convention": "log_scale_se",
        "citation": CITATION_UNVERIFIED,
        "design_limitations": list(DESIGN_LIMITATIONS),
        "n_rep_requested": n_rep_requested,
        "n_rep_planned": n_rep_planned,
        "n_rep_completed": n_rep_completed,
        "n_point_evaluable": r_point,
        "n_ci_evaluable": r_ci,
        "n_excluded": sum(excluded_reasons.values()),
        "excluded_reasons": dict(excluded_reasons),
        "n_resampled_total": n_resampled_total,
        "n_negative_draws_total": n_negative_draws_total,
        "ci_validity": ("unassessable" if r_ci < _MIN_R_FOR_ASSESSABLE_CI else "assessable"),
        "criterion": {
            "pct_within_60_140_strict": pct_joint_strict,
            "pct_within_60_140_of_own_estimate": pct_joint_own_ci,
            "target_pct": (_round(ci_target_pct) if ci_target_pct is not None else None),
            "criterion_met": criterion_met,
        },
        "per_param": per_param,
        "replicates": replicates_out,
        "est_minutes_rough": est_minutes_rough,
        "elapsed_seconds": round(time.monotonic() - t0, 1),
    }
