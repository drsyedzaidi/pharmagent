"""QC Agent tools: independent diagnostic review of NCA and PopPK/NLME analyses.

A subset of the full 15-point NCA checklist (Phase 1) plus a dedicated
Population PK (NLME) acceptance checklist (modeled on AI-assisted-MIDD
``poppk_checks.R::acceptance_checks``). Each check yields a status
(PASS / WARN / FAIL); the overall verdict is a traffic light:
PASS, CONDITIONAL PASS, or FAIL.
"""
from __future__ import annotations

import math
from typing import Any

from app.compute.pk_models import REGISTRY
from app.core.pharmstate import PharmState
from app.tools.base import Tool, ToolContext, ToolResult

# NCA thresholds (named constants, not magic numbers).
MIN_SUBJECTS = 6
MIN_LAMBDAZ_POINTS = 3
MIN_LAMBDAZ_R2_ADJ = 0.80
MIN_LAMBDAZ_SPAN_RATIO = 2.0
MAX_PCT_EXTRAP = 20.0
MAX_MISSING_PCT = 20.0

# PopPK (NLME) acceptance thresholds (aligned with AI-assisted-MIDD poppk-estimation).
MAX_POPPK_RSE_PCT = 50.0
MAX_POPPK_SHRINKAGE_PCT = 30.0
MIN_POPPK_BSV_CV_PCT = 1.0
MAX_POPPK_BSV_CV_PCT = 100.0
MAX_POPPK_CONDITION_NUMBER = 1000.0
_THETA_REL_MIN = 1e-7
_THETA_REL_MAX = 1e7


def _check(name: str, status: str, detail: str) -> dict[str, Any]:
    return {"check": name, "status": status, "detail": detail}


def _is_finite_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def run_qc(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    params = state.nca_parameters or []
    quality = state.data_quality or {}
    checks: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    # 1. sample size
    n = len(params)
    checks.append(_check(
        "Sample size adequacy",
        "PASS" if n >= MIN_SUBJECTS else "WARN",
        f"{n} subjects (>= {MIN_SUBJECTS} recommended)"))
    if n < MIN_SUBJECTS:
        issues.append({"severity": "MEDIUM", "issue": f"only {n} subjects"})

    # 2. missing data
    miss = quality.get("total_missing_pct")
    if miss is not None:
        ok = miss <= MAX_MISSING_PCT
        checks.append(_check("Missing data", "PASS" if ok else "WARN",
                             f"{miss}% missing (<= {MAX_MISSING_PCT}%)"))
        if not ok:
            issues.append({"severity": "MEDIUM", "issue": f"{miss}% missing data"})

    # 3. lambda_z point count
    few = [p["subject"] for p in params if (p.get("lambda_z_n_points") or 0) < MIN_LAMBDAZ_POINTS]
    checks.append(_check("Lambda_z points >= 3", "PASS" if not few else "WARN",
                         f"{len(few)} subject(s) with < {MIN_LAMBDAZ_POINTS} points"))
    if few:
        issues.append({"severity": "MEDIUM", "issue": f"few lambda_z points: {few}"})

    # 4. lambda_z fit quality
    poor = [p["subject"] for p in params
            if p.get("lambda_z_r2_adj") is not None and p["lambda_z_r2_adj"] < MIN_LAMBDAZ_R2_ADJ]
    checks.append(_check("Lambda_z adj R^2 >= 0.80", "PASS" if not poor else "WARN",
                         f"{len(poor)} subject(s) below threshold"))
    if poor:
        issues.append({"severity": "LOW", "issue": f"poor terminal fit: {poor}"})

    # 5. lambda_z terminal window span ratio >= 2.0 half-lives
    short_span = [
        p["subject"] for p in params
        if p.get("lambda_z_span_ratio") is not None
        and float(p["lambda_z_span_ratio"]) < MIN_LAMBDAZ_SPAN_RATIO
    ]
    checks.append(_check(
        "Lambda_z span ratio >= 2.0",
        "PASS" if not short_span else "WARN",
        f"{len(short_span)} subject(s) with terminal window < {MIN_LAMBDAZ_SPAN_RATIO:.1f} half-lives",
    ))
    if short_span:
        issues.append({
            "severity": "LOW",
            "issue": f"short lambda_z span ratio (< {MIN_LAMBDAZ_SPAN_RATIO}): {short_span}",
        })

    # 6. AUC extrapolation
    hi = [p["subject"] for p in params
          if p.get("pct_AUC_extrap") is not None and p["pct_AUC_extrap"] > MAX_PCT_EXTRAP]
    checks.append(_check("AUC %extrap <= 20%", "PASS" if not hi else "WARN",
                         f"{len(hi)} subject(s) > {MAX_PCT_EXTRAP}%"))
    if hi:
        issues.append({"severity": "MEDIUM", "issue": f"high %extrap: {hi}"})

    # 7. parameter plausibility
    implausible = [p["subject"] for p in params
                   if (p.get("CL_F") is not None and p["CL_F"] <= 0)
                   or (p.get("Vz_F") is not None and p["Vz_F"] <= 0)]
    checks.append(_check("Parameter plausibility", "PASS" if not implausible else "FAIL",
                         f"{len(implausible)} subject(s) with non-physiological CL/V"))
    if implausible:
        issues.append({"severity": "HIGH", "issue": f"implausible params: {implausible}"})

    # 8. Tmax sanity (Tmax > 0 and not at last timepoint for all)
    bad_tmax = [p["subject"] for p in params
                if p.get("Tmax") is not None and p.get("Tlast") is not None
                and p["Tmax"] >= p["Tlast"]]
    checks.append(_check("Tmax before terminal phase", "PASS" if not bad_tmax else "WARN",
                         f"{len(bad_tmax)} subject(s) with late Tmax"))

    # verdict
    statuses = [c["status"] for c in checks]
    if "FAIL" in statuses:
        verdict = "FAIL"
    elif "WARN" in statuses:
        verdict = "CONDITIONAL PASS"
    else:
        verdict = "PASS"

    return ToolResult(
        summary=f"QC verdict: {verdict} "
                f"({statuses.count('PASS')} pass, {statuses.count('WARN')} warn, "
                f"{statuses.count('FAIL')} fail).",
        action="run_qc(nca)",
        writes={"qc_verdict": verdict, "qc_issues": issues, "qc_checklist": checks},
        result={"verdict": verdict, "checks": checks, "issues": issues},
    )


def evaluate_poppk_qc(
    nl: dict[str, Any],
    *,
    max_rse_pct: float = MAX_POPPK_RSE_PCT,
    max_shrinkage_pct: float = MAX_POPPK_SHRINKAGE_PCT,
    min_bsv_cv_pct: float = MIN_POPPK_BSV_CV_PCT,
    max_bsv_cv_pct: float = MAX_POPPK_BSV_CV_PCT,
    max_condition_number: float = MAX_POPPK_CONDITION_NUMBER,
) -> dict[str, Any]:
    """Evaluate a PopPK/NLME fit against hard-fail and soft-warn acceptance criteria.

    Modeled on ``AI-assisted-MIDD``'s ``poppk_checks.R::acceptance_checks``:
      * Blocking (FAIL): ``ofv_finite``, ``covariance_step``, ``structural_se_present``,
        ``theta_not_on_bound``, ``residual_error_positive``.
      * Caveat (WARN): ``structural_rse`` (> 50%), ``eta_shrinkage`` (> 30%),
        ``bsv_range`` (outside 1-100%), ``condition_number`` (> 1000).
    """
    checks: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    theta = dict(nl.get("theta") or {})
    theta_rse = dict(nl.get("theta_rse_pct") or {})
    omega_cv = dict(nl.get("omega_cv_pct") or {})
    shrinkage = dict(nl.get("shrinkage_pct") or {})
    sigma = dict(nl.get("sigma") or {})
    error_model = str(nl.get("error_model") or "proportional").lower()
    converged = bool(nl.get("converged"))
    ofv = nl.get("ofv")

    # 1. ofv_finite (FAIL if not converged or OFV not finite)
    ofv_ok = converged and _is_finite_num(ofv)
    ofv_str = f"OFV = {float(ofv):.3f}" if _is_finite_num(ofv) else f"OFV = {ofv!r}"
    if not converged:
        ofv_str += " (fit did not converge)"
    checks.append(_check("ofv_finite", "PASS" if ofv_ok else "FAIL", ofv_str))
    if not ofv_ok:
        issues.append({"severity": "HIGH", "issue": f"ofv_finite failed: {ofv_str}"})

    # 2. covariance_step (FAIL if no structural RSEs computed or covariance failed)
    has_any_se = bool(theta) and any(_is_finite_num(theta_rse.get(k)) for k in theta)
    cov_note = str(nl.get("cov_note") or "")
    cov_detail = "covariance available" if has_any_se else (cov_note or "covariance step failed")
    checks.append(_check("covariance_step", "PASS" if has_any_se else "FAIL", cov_detail))
    if not has_any_se:
        issues.append({"severity": "HIGH", "issue": f"covariance_step failed: {cov_detail}"})

    # 3. structural_se_present (FAIL if any structural THETA lacks SE / %RSE)
    se_missing = [k for k in theta if not _is_finite_num(theta_rse.get(k))]
    se_ok = bool(theta) and not se_missing
    se_detail = (
        f"SE missing: {', '.join(se_missing)}"
        if se_missing
        else ("all structural SEs present" if theta else "no structural parameters found")
    )
    checks.append(_check("structural_se_present", "PASS" if se_ok else "FAIL", se_detail))
    if not se_ok:
        issues.append({"severity": "HIGH", "issue": f"structural_se_present failed: {se_detail}"})

    # 4. theta_not_on_bound (FAIL if degenerate_params or non-positive/out-of-bounds THETA)
    model_key = str(nl.get("model_key") or "")
    defaults = dict(REGISTRY[model_key].defaults) if model_key in REGISTRY else {}
    on_bound: list[str] = list(nl.get("degenerate_params") or [])
    for k, val in theta.items():
        if k in on_bound:
            continue
        if not (_is_finite_num(val) and float(val) > 0.0):
            on_bound.append(k)
            continue
        ref = float(defaults.get(k, 0.0) or 0.0)
        if ref > 0.0 and not (_THETA_REL_MIN <= float(val) / ref <= _THETA_REL_MAX):
            on_bound.append(k)
    bound_ok = bool(theta) and not on_bound
    bound_detail = f"on bound: {', '.join(on_bound)}" if on_bound else "no THETA on a bound"
    checks.append(_check("theta_not_on_bound", "PASS" if bound_ok else "FAIL", bound_detail))
    if not bound_ok:
        issues.append({"severity": "HIGH", "issue": f"theta_not_on_bound failed: {bound_detail}"})

    # 5. structural_rse (WARN if any structural %RSE > max_rse_pct)
    high_rse = [
        f"{k} ({float(theta_rse[k]):.1f}%)"
        for k in theta
        if _is_finite_num(theta_rse.get(k)) and float(theta_rse[k]) > max_rse_pct
    ]
    rse_ok = not high_rse
    rse_detail = (
        f"%RSE > {max_rse_pct:g}: {', '.join(high_rse)}"
        if high_rse
        else f"all structural %RSE <= {max_rse_pct:g}"
    )
    checks.append(_check("structural_rse", "PASS" if rse_ok else "WARN", rse_detail))
    if not rse_ok:
        issues.append({"severity": "MEDIUM", "issue": rse_detail})

    # 6. eta_shrinkage (WARN if any ETA shrinkage > max_shrinkage_pct)
    if shrinkage:
        high_shr = [
            f"{k} ({float(v):.1f}%)"
            for k, v in shrinkage.items()
            if _is_finite_num(v) and float(v) > max_shrinkage_pct
        ]
        shr_ok = not high_shr
        shr_detail = (
            f"SD shrinkage > {max_shrinkage_pct:g}%: {', '.join(high_shr)}"
            if high_shr
            else f"all ETA SD shrinkage <= {max_shrinkage_pct:g}%"
        )
        checks.append(_check("eta_shrinkage", "PASS" if shr_ok else "WARN", shr_detail))
        if not shr_ok:
            issues.append({"severity": "MEDIUM", "issue": shr_detail})

    # 7. bsv_range (WARN if any omega CV% outside [min_bsv_cv_pct, max_bsv_cv_pct])
    if omega_cv:
        odd_bsv = [
            f"{k} ({float(v):.1f}%)"
            for k, v in omega_cv.items()
            if _is_finite_num(v) and (float(v) < min_bsv_cv_pct or float(v) > max_bsv_cv_pct)
        ]
        bsv_ok = not odd_bsv
        bsv_detail = (
            f"BSV CV% outside {min_bsv_cv_pct:g}-{max_bsv_cv_pct:g}%: {', '.join(odd_bsv)}"
            if odd_bsv
            else f"all BSV CV% within {min_bsv_cv_pct:g}-{max_bsv_cv_pct:g}%"
        )
        checks.append(_check("bsv_range", "PASS" if bsv_ok else "WARN", bsv_detail))
        if not bsv_ok:
            issues.append({"severity": "LOW", "issue": bsv_detail})

    # 8. condition_number (WARN if condition_number > max_condition_number)
    cond = nl.get("condition_number")
    if _is_finite_num(cond):
        cond_ok = float(cond) <= max_condition_number
        cond_detail = f"condition number = {float(cond):.1f} (<= {max_condition_number:g})"
        checks.append(_check("condition_number", "PASS" if cond_ok else "WARN", cond_detail))
        if not cond_ok:
            issues.append({
                "severity": "MEDIUM",
                "issue": f"high condition number: {float(cond):.1f} (> {max_condition_number:g})",
            })

    # 9. residual_error_positive (FAIL if active sigma components are missing or <= 0)
    if error_model == "proportional":
        active_err = ["prop"]
    elif error_model == "additive":
        active_err = ["add"]
    elif error_model == "combined":
        active_err = ["prop", "add"]
    else:
        active_err = [k for k, v in sigma.items() if v is not None] or ["prop"]

    err_ok = bool(active_err) and all(
        _is_finite_num(sigma.get(k)) and float(sigma[k]) > 0.0 for k in active_err
    )
    err_parts = [
        f"{k} = {float(sigma[k]):.3g}" if _is_finite_num(sigma.get(k)) else f"{k} = {sigma.get(k)!r}"
        for k in active_err
    ]
    err_detail = ", ".join(err_parts) if err_parts else "no residual error components"
    checks.append(_check("residual_error_positive", "PASS" if err_ok else "FAIL", err_detail))
    if not err_ok:
        issues.append({"severity": "HIGH", "issue": f"residual_error_positive failed: {err_detail}"})

    statuses = [c["status"] for c in checks]
    if "FAIL" in statuses:
        verdict = "FAIL"
    elif "WARN" in statuses:
        verdict = "CONDITIONAL PASS"
    else:
        verdict = "PASS"

    return {"verdict": verdict, "checks": checks, "issues": issues}


def run_poppk_qc(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Run PopPK (NLME) acceptance checks on ``state.nlme_results``."""
    nl = state.nlme_results
    if not nl or nl.get("status") != "ok":
        raise ValueError("run_poppk_qc requires a completed NLME fit (run_nlme) in state.nlme_results")

    max_rse = float(args.get("max_rse_pct", MAX_POPPK_RSE_PCT))
    max_shr = float(args.get("max_shrinkage_pct", MAX_POPPK_SHRINKAGE_PCT))
    out = evaluate_poppk_qc(nl, max_rse_pct=max_rse, max_shrinkage_pct=max_shr)
    verdict = out["verdict"]
    checks = out["checks"]
    issues = out["issues"]
    statuses = [c["status"] for c in checks]

    return ToolResult(
        summary=(
            f"PopPK QC verdict: {verdict} "
            f"({statuses.count('PASS')} pass, {statuses.count('WARN')} warn, "
            f"{statuses.count('FAIL')} fail)."
        ),
        action=f"run_poppk_qc({nl.get('model_key', 'nlme')})",
        writes={
            "poppk_qc_verdict": verdict,
            "poppk_qc_issues": issues,
            "poppk_qc_checklist": checks,
        },
        result={"verdict": verdict, "checks": checks, "issues": issues},
    )


TOOLS = [
    Tool("run_qc",
         "Independent QC of the NCA analysis against a diagnostic checklist; "
         "returns PASS / CONDITIONAL PASS / FAIL.",
         "qc",
         {"type": "object", "properties": {}, "required": []},
         run_qc),
    Tool("run_poppk_qc",
         "Independent PopPK (NLME) acceptance check against regulatory criteria "
         "(finite OFV, covariance step, structural SEs, bounds, %RSE <= 50%, "
         "ETA shrinkage <= 30%, BSV 1-100%, condition number <= 1000, positive sigma); "
         "returns PASS / CONDITIONAL PASS / FAIL.",
         "qc",
         {
             "type": "object",
             "properties": {
                 "max_rse_pct": {"type": "number"},
                 "max_shrinkage_pct": {"type": "number"},
             },
             "required": [],
         },
         run_poppk_qc),
]
