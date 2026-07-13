"""QC Agent tools: independent diagnostic review of an NCA analysis.

A subset of the full 15-point checklist (Phase 1). Each check yields a
status (PASS / WARN / FAIL); the overall verdict is a traffic light:
PASS, CONDITIONAL PASS, or FAIL.
"""
from __future__ import annotations

from typing import Any

from app.core.pharmstate import PharmState
from app.tools.base import Tool, ToolContext, ToolResult

# Thresholds (named constants, not magic numbers).
MIN_SUBJECTS = 6
MIN_LAMBDAZ_POINTS = 3
MIN_LAMBDAZ_R2_ADJ = 0.80
MAX_PCT_EXTRAP = 20.0
MAX_MISSING_PCT = 20.0


def _check(name: str, status: str, detail: str) -> dict[str, Any]:
    return {"check": name, "status": status, "detail": detail}


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

    # 5. AUC extrapolation
    hi = [p["subject"] for p in params
          if p.get("pct_AUC_extrap") is not None and p["pct_AUC_extrap"] > MAX_PCT_EXTRAP]
    checks.append(_check("AUC %extrap <= 20%", "PASS" if not hi else "WARN",
                         f"{len(hi)} subject(s) > {MAX_PCT_EXTRAP}%"))
    if hi:
        issues.append({"severity": "MEDIUM", "issue": f"high %extrap: {hi}"})

    # 6. parameter plausibility
    implausible = [p["subject"] for p in params
                   if (p.get("CL_F") is not None and p["CL_F"] <= 0)
                   or (p.get("Vz_F") is not None and p["Vz_F"] <= 0)]
    checks.append(_check("Parameter plausibility", "PASS" if not implausible else "FAIL",
                         f"{len(implausible)} subject(s) with non-physiological CL/V"))
    if implausible:
        issues.append({"severity": "HIGH", "issue": f"implausible params: {implausible}"})

    # 7. Tmax sanity (Tmax > 0 and not at last timepoint for all)
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


TOOLS = [
    Tool("run_qc",
         "Independent QC of the NCA analysis against a diagnostic checklist; "
         "returns PASS / CONDITIONAL PASS / FAIL.",
         "qc",
         {"type": "object", "properties": {}, "required": []},
         run_qc),
]
