"""QC Agent tools: independent diagnostic review of NCA and PopPK/NLME analyses.

A subset of the full 15-point NCA checklist (Phase 1) plus a dedicated
Population PK (NLME) acceptance checklist:
  * 8 upstream criteria modeled on ``AI-assisted-MIDD``
    (``poppk-estimation/scripts/poppk_checks.R::acceptance_checks``):
    ``ofv_finite``, ``covariance_step``, ``structural_se_present``,
    ``theta_not_on_bound``, ``structural_rse``, ``eta_shrinkage``,
    ``bsv_range``, and ``residual_error_positive``.
  * 1 PharmAgent extension: ``condition_number <= 1000`` (sourced from
    ``app.compute.nlme._COND_RED_FLAG``; tagged with
    ``source="pharmagent_extension"``).

Each check yields a status (PASS / WARN / FAIL); the overall verdict is a
traffic light: PASS, CONDITIONAL PASS, or FAIL. Flagged NCA terminal slopes
(``lambda_z``) are recorded in ``state.nca_flagged_subjects`` with an explicit
analyst accept/revise gate in ``state.lambda_z_review`` (matching the upstream
``AI-assisted-MIDD`` ``pk-nca`` workflow).
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

# PopPK (NLME) acceptance thresholds.
# Upstream AI-assisted-MIDD poppk_checks.R defaults:
MAX_POPPK_RSE_PCT = 50.0
MAX_POPPK_SHRINKAGE_PCT = 30.0
MIN_POPPK_BSV_CV_PCT = 1.0
MAX_POPPK_BSV_CV_PCT = 100.0
# PharmAgent extension (not an upstream AI-assisted-MIDD poppk_checks.R criterion):
MAX_POPPK_CONDITION_NUMBER = 1000.0

_THETA_REL_MIN = 1e-7
_THETA_REL_MAX = 1e7

_COV_FAIL_PHRASES = (
    "covariance step failed",
    "non-finite information matrix",
    "not positive",
    "indefinite",
    "invalid covariance diagonal",
    "near-singular",
    "standard errors unavailable",
    "standard errors unreliable",
)


def _check(
    name: str,
    status: str,
    detail: str,
    *,
    source: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {"check": name, "status": status, "detail": detail}
    if source is not None:
        out["source"] = source
    return out


def _is_finite_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _validate_threshold(
    name: str,
    value: Any,
    *,
    strictly_positive: bool = True,
    min_val: float | None = None,
    max_val: float | None = None,
) -> float:
    """Validate that a QC threshold is a finite float within its valid domain."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number; got {value!r}")
    try:
        fval = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number; got {value!r}") from exc
    if not math.isfinite(fval):
        raise ValueError(f"{name} must be finite (not NaN or infinity); got {value!r}")
    if strictly_positive and fval <= 0.0:
        raise ValueError(f"{name} must be > 0; got {fval}")
    if min_val is not None and fval < min_val:
        raise ValueError(f"{name} must be >= {min_val}; got {fval}")
    if max_val is not None and fval > max_val:
        raise ValueError(f"{name} must be <= {max_val}; got {fval}")
    return fval


def _collect_flagged_lambda_z_subjects(
    params: list[dict[str, Any]],
    *,
    min_span_ratio: float = MIN_LAMBDAZ_SPAN_RATIO,
    min_r2_adj: float = MIN_LAMBDAZ_R2_ADJ,
    min_points: int = MIN_LAMBDAZ_POINTS,
) -> list[dict[str, Any]]:
    """Collect subjects with missing or substandard terminal regression (lambda_z) fits."""
    flagged: list[dict[str, Any]] = []
    for p in params:
        reasons: list[str] = []
        n_pts = p.get("lambda_z_n_points") or 0
        r2 = p.get("lambda_z_r2_adj")
        span = p.get("lambda_z_span_ratio")
        if int(n_pts) < min_points:
            reasons.append(f"n_points < {min_points}")
        if not _is_finite_num(r2):
            reasons.append("r2_adj missing/NaN")
        elif float(r2) < min_r2_adj:
            reasons.append(f"r2_adj < {min_r2_adj:.2f}")
        if not _is_finite_num(span):
            reasons.append("span_ratio missing/NaN")
        elif float(span) < min_span_ratio:
            reasons.append(f"span_ratio < {min_span_ratio:.1f}")
        if reasons:
            flagged.append({
                "id": p.get("subject"),
                "span_ratio": float(span) if _is_finite_num(span) else None,
                "r2_adj": float(r2) if _is_finite_num(r2) else None,
                "n_points": int(n_pts),
                "reasons": reasons,
            })
    return flagged


def _build_lambda_z_review(
    flagged_subjects: list[dict[str, Any]],
    decision: str | None = None,
    notes: str = "",
    reviewed_subjects: list[Any] | None = None,
) -> dict[str, Any]:
    """Construct the analyst accept/revise review gate state for flagged lambda_z fits."""
    if not flagged_subjects and decision is None:
        return {
            "status": "not_required",
            "gate_open": False,
            "decision": None,
            "flagged_subjects": [],
            "notes": "",
        }
    if decision is not None:
        norm = str(decision).strip().lower()
        if norm not in {"accept", "revise"}:
            raise ValueError(
                f"lambda_z decision must be 'accept' or 'revise'; got {decision!r}"
            )
        target_ids = (
            list(reviewed_subjects)
            if reviewed_subjects is not None
            else [f["id"] for f in flagged_subjects]
        )
        if norm == "accept":
            return {
                "status": "accepted",
                "gate_open": False,
                "decision": "accept",
                "flagged_subjects": flagged_subjects,
                "reviewed_subjects": target_ids,
                "notes": notes or "Analyst accepted flagged lambda_z fits as caveats.",
            }
        return {
            "status": "revision_requested",
            "gate_open": True,
            "decision": "revise",
            "flagged_subjects": flagged_subjects,
            "reviewed_subjects": target_ids,
            "notes": notes or "Analyst requested revision of terminal regression windows.",
        }
    flagged_ids = [f["id"] for f in flagged_subjects]
    return {
        "status": "awaiting_analyst_decision",
        "gate_open": True,
        "decision": None,
        "flagged_subjects": flagged_subjects,
        "prompt": (
            f"Subjects with flagged lambda_z fits ({flagged_ids}): ask the analyst "
            "whether to accept the fits or revise them via "
            "review_lambda_z_fits(decision='accept'|'revise')."
        ),
        "notes": notes,
    }


def run_qc(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    params = state.nca_parameters or []
    quality = state.data_quality or {}
    checks: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    min_span_ratio = _validate_threshold(
        "min_span_ratio",
        args.get("min_span_ratio", MIN_LAMBDAZ_SPAN_RATIO),
        strictly_positive=True,
    )
    min_r2_adj = _validate_threshold(
        "min_r2_adj",
        args.get("min_r2_adj", MIN_LAMBDAZ_R2_ADJ),
        strictly_positive=True,
        max_val=1.0,
    )
    max_pct_extrap = _validate_threshold(
        "max_pct_extrap",
        args.get("max_pct_extrap", MAX_PCT_EXTRAP),
        strictly_positive=True,
        max_val=100.0,
    )
    max_missing_pct = _validate_threshold(
        "max_missing_pct",
        args.get("max_missing_pct", MAX_MISSING_PCT),
        strictly_positive=False,
        min_val=0.0,
        max_val=100.0,
    )
    min_points = int(
        _validate_threshold(
            "min_points",
            args.get("min_points", MIN_LAMBDAZ_POINTS),
            strictly_positive=True,
            min_val=2.0,
        )
    )

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
        ok = _is_finite_num(miss) and float(miss) <= max_missing_pct
        checks.append(_check("Missing data", "PASS" if ok else "WARN",
                             f"{miss}% missing (<= {max_missing_pct}%)"))
        if not ok:
            issues.append({"severity": "MEDIUM", "issue": f"{miss}% missing data"})

    # 3. lambda_z point count
    few = [p["subject"] for p in params if (p.get("lambda_z_n_points") or 0) < min_points]
    checks.append(_check("Lambda_z points >= 3", "PASS" if not few else "WARN",
                         f"{len(few)} subject(s) with < {min_points} points"))
    if few:
        issues.append({"severity": "MEDIUM", "issue": f"few lambda_z points: {few}"})

    # 4. lambda_z fit quality
    poor = [
        p["subject"] for p in params
        if p.get("lambda_z_r2_adj") is not None
        and (not _is_finite_num(p.get("lambda_z_r2_adj")) or float(p["lambda_z_r2_adj"]) < min_r2_adj)
    ]
    checks.append(_check("Lambda_z adj R^2 >= 0.80", "PASS" if not poor else "WARN",
                         f"{len(poor)} subject(s) below threshold"))
    if poor:
        issues.append({"severity": "LOW", "issue": f"poor terminal fit: {poor}"})

    # 5. lambda_z terminal window span ratio >= 2.0 half-lives
    # Missing (None) or non-finite (NaN/inf) span ratios count as WARN, never PASS.
    missing_span = [
        p["subject"] for p in params
        if not _is_finite_num(p.get("lambda_z_span_ratio"))
    ]
    short_span = [
        p["subject"] for p in params
        if _is_finite_num(p.get("lambda_z_span_ratio"))
        and float(p["lambda_z_span_ratio"]) < min_span_ratio
    ]
    bad_span = missing_span + short_span
    if missing_span and short_span:
        span_detail = (
            f"{len(short_span)} subject(s) with terminal window < {min_span_ratio:.1f} half-lives; "
            f"{len(missing_span)} subject(s) with missing/NaN span ratio"
        )
    elif missing_span:
        span_detail = (
            f"{len(missing_span)} subject(s) with missing/NaN span ratio "
            f"(>= {min_span_ratio:.1f} half-lives required)"
        )
    else:
        span_detail = (
            f"{len(short_span)} subject(s) with terminal window < {min_span_ratio:.1f} half-lives"
        )
    checks.append(_check(
        "Lambda_z span ratio >= 2.0",
        "PASS" if not bad_span else "WARN",
        span_detail,
    ))
    if short_span:
        issues.append({
            "severity": "LOW",
            "issue": f"short lambda_z span ratio (< {min_span_ratio}): {short_span}",
        })
    if missing_span:
        issues.append({
            "severity": "MEDIUM",
            "issue": f"missing or NaN lambda_z span ratio: {missing_span}",
        })

    # 6. AUC extrapolation
    hi = [
        p["subject"] for p in params
        if p.get("pct_AUC_extrap") is not None
        and (not _is_finite_num(p.get("pct_AUC_extrap")) or float(p["pct_AUC_extrap"]) > max_pct_extrap)
    ]
    checks.append(_check("AUC %extrap <= 20%", "PASS" if not hi else "WARN",
                         f"{len(hi)} subject(s) > {max_pct_extrap}%"))
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

    # Collect flagged lambda_z fits & construct analyst accept/revise review gate
    flagged_subjects = _collect_flagged_lambda_z_subjects(
        params,
        min_span_ratio=min_span_ratio,
        min_r2_adj=min_r2_adj,
        min_points=min_points,
    )
    lambda_z_review = _build_lambda_z_review(
        flagged_subjects,
        decision=args.get("lambda_z_decision"),
        notes=str(args.get("notes") or ""),
    )

    # verdict
    statuses = [c["status"] for c in checks]
    if "FAIL" in statuses:
        verdict = "FAIL"
    elif "WARN" in statuses:
        verdict = "CONDITIONAL PASS"
    else:
        verdict = "PASS"

    summary_msg = (
        f"QC verdict: {verdict} "
        f"({statuses.count('PASS')} pass, {statuses.count('WARN')} warn, "
        f"{statuses.count('FAIL')} fail)."
    )
    if lambda_z_review["status"] == "awaiting_analyst_decision":
        flagged_ids = [f["id"] for f in flagged_subjects]
        summary_msg += (
            f" Flagged lambda_z fits for subject(s) {flagged_ids}: "
            "analyst decision required (accept or revise via review_lambda_z_fits)."
        )

    return ToolResult(
        summary=summary_msg,
        action="run_qc(nca)",
        writes={
            "qc_verdict": verdict,
            "qc_issues": issues,
            "qc_checklist": checks,
            "nca_flagged_subjects": flagged_subjects,
            "lambda_z_review": lambda_z_review,
        },
        result={
            "verdict": verdict,
            "checks": checks,
            "issues": issues,
            "flagged_subjects": flagged_subjects,
            "lambda_z_review": lambda_z_review,
        },
    )


def review_lambda_z_fits(
    state: PharmState,
    ctx: ToolContext,
    args: dict[str, Any],
) -> ToolResult:
    """Explicit analyst accept/revise gate for flagged NCA lambda_z fits.

    Follows the upstream ``AI-assisted-MIDD`` ``pk-nca`` workflow: when subjects
    have ``lambda_z_span_ratio < 2.0`` (or missing/low-R² terminal fits), the
    analyst explicitly decides whether to accept the fits as caveats or request
    a terminal-window revision before finalizing reporting.
    """
    params = state.nca_parameters
    if not params:
        raise ValueError("review_lambda_z_fits requires NCA parameters (compute_nca) in state")

    decision = args.get("decision")
    if decision is None:
        raise ValueError("review_lambda_z_fits requires decision='accept' or decision='revise'")

    flagged = (
        list(state.nca_flagged_subjects)
        if state.nca_flagged_subjects is not None
        else _collect_flagged_lambda_z_subjects(params)
    )
    notes = str(args.get("notes") or "")
    subject_ids = args.get("subject_ids")
    review = _build_lambda_z_review(
        flagged,
        decision=str(decision),
        notes=notes,
        reviewed_subjects=subject_ids if isinstance(subject_ids, list) else None,
    )
    reviewed_ids = review.get("reviewed_subjects") or []
    return ToolResult(
        summary=(
            f"Recorded analyst lambda_z review decision: {review['decision'].upper()} "
            f"({review['status']}) for subject(s) {reviewed_ids}."
        ),
        action=f"review_lambda_z_fits({review['decision']})",
        writes={
            "nca_flagged_subjects": flagged,
            "lambda_z_review": review,
        },
        result={
            "lambda_z_review": review,
            "flagged_subjects": flagged,
        },
    )


def _parse_theta_bounds(
    raw_bounds: Any,
) -> dict[str, tuple[float | None, float | None]]:
    """Normalize optional declared estimation bounds ``{param: (lower, upper)}``."""
    if not isinstance(raw_bounds, dict):
        return {}
    parsed: dict[str, tuple[float | None, float | None]] = {}
    for param, b in raw_bounds.items():
        lo_raw: Any = None
        hi_raw: Any = None
        if isinstance(b, (list, tuple)) and len(b) == 2:
            lo_raw, hi_raw = b[0], b[1]
        elif isinstance(b, dict):
            lo_raw = b.get("lower", b.get("lo"))
            hi_raw = b.get("upper", b.get("hi"))
        else:
            raise ValueError(
                f"theta_bounds[{param!r}] must be a (lower, upper) pair or "
                f"{{'lower': ..., 'upper': ...}}; got {b!r}"
            )
        lo: float | None = None
        hi: float | None = None
        if lo_raw is not None:
            if isinstance(lo_raw, bool) or not isinstance(lo_raw, (int, float)):
                raise ValueError(f"theta_bounds[{param!r}] lower bound must be numeric")
            lo_f = float(lo_raw)
            if math.isnan(lo_f):
                raise ValueError(f"theta_bounds[{param!r}] lower bound cannot be NaN")
            lo = lo_f if math.isfinite(lo_f) else None
        if hi_raw is not None:
            if isinstance(hi_raw, bool) or not isinstance(hi_raw, (int, float)):
                raise ValueError(f"theta_bounds[{param!r}] upper bound must be numeric")
            hi_f = float(hi_raw)
            if math.isnan(hi_f):
                raise ValueError(f"theta_bounds[{param!r}] upper bound cannot be NaN")
            hi = hi_f if math.isfinite(hi_f) else None
        if lo is not None and hi is not None and lo >= hi:
            raise ValueError(
                f"theta_bounds[{param!r}] lower ({lo}) must be < upper ({hi})"
            )
        parsed[str(param)] = (lo, hi)
    return parsed


def _covariance_step_status(
    nl: dict[str, Any],
    theta: dict[str, Any],
    theta_rse: dict[str, Any],
) -> tuple[bool, str]:
    """Determine actual covariance-step status from ``nl`` diagnostics.

    Does not infer covariance success merely from a single finite structural RSE
    when ``cov_note`` reports a failed, non-positive, indefinite, or near-singular
    information matrix, or when ``covariance_step_ok`` is ``False``.
    """
    cov_note = str(nl.get("cov_note") or "").strip()
    low_note = cov_note.lower()
    note_failed = any(phrase in low_note for phrase in _COV_FAIL_PHRASES)

    explicit_ok = nl.get("covariance_step_ok")
    if isinstance(explicit_ok, bool):
        if not explicit_ok:
            return False, cov_note or "covariance step failed (covariance_step_ok=False)"
        if note_failed:
            return False, cov_note
        return True, cov_note or "covariance step succeeded"

    cov_status = str(nl.get("covariance_status") or "").strip().lower()
    if cov_status in {"failed", "error", "singular", "near_singular", "not_run", "none"}:
        return False, cov_note or f"covariance step {cov_status}"

    if note_failed:
        return False, cov_note

    if not theta:
        return False, "no structural parameters found"
    if not theta_rse:
        return False, cov_note or "covariance step not run or failed"

    # Without an explicit covariance_step_ok=True flag, a missing/suppressed
    # structural SE indicates a near-singular or incomplete covariance matrix.
    if not all(_is_finite_num(theta_rse.get(k)) for k in theta):
        return (
            False,
            cov_note or "covariance step incomplete (one or more structural SEs unavailable)",
        )

    return True, "covariance step succeeded" if not cov_note else f"covariance step succeeded ({cov_note})"


def evaluate_poppk_qc(
    nl: dict[str, Any],
    *,
    max_rse_pct: float = MAX_POPPK_RSE_PCT,
    max_shrinkage_pct: float = MAX_POPPK_SHRINKAGE_PCT,
    min_bsv_cv_pct: float = MIN_POPPK_BSV_CV_PCT,
    max_bsv_cv_pct: float = MAX_POPPK_BSV_CV_PCT,
    max_condition_number: float = MAX_POPPK_CONDITION_NUMBER,
    theta_bounds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a PopPK/NLME fit against hard-fail and soft-warn acceptance criteria.

    Provenance:
      * 8 checks modeled on ``AI-assisted-MIDD``
        (``poppk-estimation/scripts/poppk_checks.R::acceptance_checks``), tagged
        ``source="ai_assisted_midd"``:
        - Blocking (FAIL): ``ofv_finite``, ``covariance_step``,
          ``structural_se_present``, ``theta_not_on_bound``,
          ``residual_error_positive``.
        - Caveat (WARN): ``structural_rse`` (> 50%), ``eta_shrinkage`` (> 30%),
          ``bsv_range`` (outside 1-100%).
      * 1 PharmAgent extension, tagged ``source="pharmagent_extension"``:
        - Caveat (WARN): ``condition_number`` (> 1000, from
          ``app.compute.nlme._COND_RED_FLAG``).
    """
    max_rse_pct = _validate_threshold("max_rse_pct", max_rse_pct, strictly_positive=True)
    max_shrinkage_pct = _validate_threshold(
        "max_shrinkage_pct", max_shrinkage_pct, strictly_positive=True, max_val=100.0
    )
    min_bsv_cv_pct = _validate_threshold("min_bsv_cv_pct", min_bsv_cv_pct, strictly_positive=True)
    max_bsv_cv_pct = _validate_threshold("max_bsv_cv_pct", max_bsv_cv_pct, strictly_positive=True)
    if min_bsv_cv_pct >= max_bsv_cv_pct:
        raise ValueError(
            f"min_bsv_cv_pct ({min_bsv_cv_pct}) must be < max_bsv_cv_pct ({max_bsv_cv_pct})"
        )
    max_condition_number = _validate_threshold(
        "max_condition_number", max_condition_number, strictly_positive=True, min_val=1.0
    )

    declared_bounds = _parse_theta_bounds(
        theta_bounds if theta_bounds is not None else nl.get("theta_bounds")
    )

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
    checks.append(_check("ofv_finite", "PASS" if ofv_ok else "FAIL", ofv_str,
                         source="ai_assisted_midd"))
    if not ofv_ok:
        issues.append({"severity": "HIGH", "issue": f"ofv_finite failed: {ofv_str}"})

    # 2. covariance_step (FAIL if covariance step failed, near-singular, or not run)
    cov_ok, cov_detail = _covariance_step_status(nl, theta, theta_rse)
    checks.append(_check("covariance_step", "PASS" if cov_ok else "FAIL", cov_detail,
                         source="ai_assisted_midd"))
    if not cov_ok:
        issues.append({"severity": "HIGH", "issue": f"covariance_step failed: {cov_detail}"})

    # 3. structural_se_present (FAIL if any structural THETA lacks SE / %RSE)
    se_missing = [k for k in theta if not _is_finite_num(theta_rse.get(k))]
    se_ok = bool(theta) and not se_missing
    se_detail = (
        f"SE missing: {', '.join(se_missing)}"
        if se_missing
        else ("all structural SEs present" if theta else "no structural parameters found")
    )
    checks.append(_check("structural_se_present", "PASS" if se_ok else "FAIL", se_detail,
                         source="ai_assisted_midd"))
    if not se_ok:
        issues.append({"severity": "HIGH", "issue": f"structural_se_present failed: {se_detail}"})

    # 4. theta_not_on_bound
    # Checks proximity to declared estimation bounds (tol = 1e-4 * max(1, |est|))
    # when theta_bounds are supplied, AND checks default-box degeneration
    # (degenerate_params / [1e-7x, 1e7x] relative to model defaults).
    model_key = str(nl.get("model_key") or "")
    defaults = dict(REGISTRY[model_key].defaults) if model_key in REGISTRY else {}
    on_declared_bound: list[str] = []
    degenerate_thetas: list[str] = list(nl.get("degenerate_params") or [])
    has_finite_declared = any(
        lo is not None or hi is not None for lo, hi in declared_bounds.values()
    )

    for k, val in theta.items():
        if not _is_finite_num(val):
            if k not in degenerate_thetas:
                degenerate_thetas.append(k)
            continue
        fval = float(val)
        if k in declared_bounds:
            lo, hi = declared_bounds[k]
            tol = 1e-4 * max(1.0, abs(fval))
            at_lo = lo is not None and (fval <= lo or abs(fval - lo) < tol)
            at_hi = hi is not None and (fval >= hi or abs(fval - hi) < tol)
            if (at_lo or at_hi) and k not in on_declared_bound:
                on_declared_bound.append(k)
        if fval <= 0.0 and k not in degenerate_thetas and k not in on_declared_bound:
            degenerate_thetas.append(k)
            continue
        ref = float(defaults.get(k, 0.0) or 0.0)
        if (
            ref > 0.0
            and not (_THETA_REL_MIN <= fval / ref <= _THETA_REL_MAX)
            and k not in degenerate_thetas
        ):
            degenerate_thetas.append(k)

    bound_ok = bool(theta) and not on_declared_bound and not degenerate_thetas
    if on_declared_bound and degenerate_thetas:
        bound_detail = (
            f"on declared bound: {', '.join(on_declared_bound)}; "
            f"degenerate vs default box: {', '.join(degenerate_thetas)}"
        )
    elif on_declared_bound:
        bound_detail = f"on declared bound: {', '.join(on_declared_bound)}"
    elif degenerate_thetas:
        bound_detail = (
            f"degenerate relative to model defaults: {', '.join(degenerate_thetas)}"
        )
    elif not theta:
        bound_detail = "no structural parameters found"
    elif has_finite_declared:
        bound_detail = "no THETA on declared bound"
    else:
        bound_detail = (
            "unconstrained (no declared estimation bounds; no default-box degeneration)"
        )
    checks.append(_check("theta_not_on_bound", "PASS" if bound_ok else "FAIL", bound_detail,
                         source="ai_assisted_midd"))
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
    checks.append(_check("structural_rse", "PASS" if rse_ok else "WARN", rse_detail,
                         source="ai_assisted_midd"))
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
        checks.append(_check("eta_shrinkage", "PASS" if shr_ok else "WARN", shr_detail,
                             source="ai_assisted_midd"))
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
        checks.append(_check("bsv_range", "PASS" if bsv_ok else "WARN", bsv_detail,
                             source="ai_assisted_midd"))
        if not bsv_ok:
            issues.append({"severity": "LOW", "issue": bsv_detail})

    # 8. condition_number (WARN if condition_number > max_condition_number)
    # PharmAgent extension (from app.compute.nlme._COND_RED_FLAG = 1e3), not an
    # upstream AI-assisted-MIDD poppk_checks.R criterion.
    cond = nl.get("condition_number")
    if _is_finite_num(cond):
        cond_ok = float(cond) <= max_condition_number
        cond_detail = (
            f"condition number = {float(cond):.1f} "
            f"(<= {max_condition_number:g}; PharmAgent extension)"
        )
        checks.append(_check(
            "condition_number",
            "PASS" if cond_ok else "WARN",
            cond_detail,
            source="pharmagent_extension",
        ))
        if not cond_ok:
            issues.append({
                "severity": "MEDIUM",
                "issue": (
                    f"high condition number: {float(cond):.1f} "
                    f"(> {max_condition_number:g}; PharmAgent extension)"
                ),
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
    checks.append(_check("residual_error_positive", "PASS" if err_ok else "FAIL", err_detail,
                         source="ai_assisted_midd"))
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

    out = evaluate_poppk_qc(
        nl,
        max_rse_pct=args.get("max_rse_pct", MAX_POPPK_RSE_PCT),
        max_shrinkage_pct=args.get("max_shrinkage_pct", MAX_POPPK_SHRINKAGE_PCT),
        min_bsv_cv_pct=args.get("min_bsv_cv_pct", MIN_POPPK_BSV_CV_PCT),
        max_bsv_cv_pct=args.get("max_bsv_cv_pct", MAX_POPPK_BSV_CV_PCT),
        max_condition_number=args.get("max_condition_number", MAX_POPPK_CONDITION_NUMBER),
        theta_bounds=args.get("theta_bounds"),
    )
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
         "records flagged lambda_z subjects for analyst accept/revise review and "
         "returns PASS / CONDITIONAL PASS / FAIL.",
         "qc",
         {
             "type": "object",
             "properties": {
                 "min_span_ratio": {"type": "number"},
                 "min_r2_adj": {"type": "number"},
                 "min_points": {"type": "integer"},
                 "max_pct_extrap": {"type": "number"},
                 "max_missing_pct": {"type": "number"},
                 "lambda_z_decision": {"type": "string", "enum": ["accept", "revise"]},
                 "notes": {"type": "string"},
             },
             "required": [],
         },
         run_qc),
    Tool("review_lambda_z_fits",
         "Record the analyst's explicit accept/revise decision for subjects with "
         "flagged NCA terminal slope (lambda_z) fits (span ratio < 2.0, adj R^2 < 0.80, "
         "or < 3 terminal points).",
         "qc",
         {
             "type": "object",
             "properties": {
                 "decision": {"type": "string", "enum": ["accept", "revise"]},
                 "subject_ids": {"type": "array", "items": {}},
                 "notes": {"type": "string"},
             },
             "required": ["decision"],
         },
         review_lambda_z_fits),
    Tool("run_poppk_qc",
         "Independent PopPK (NLME) acceptance check against the 8 AI-assisted-MIDD "
         "criteria (finite OFV, covariance step, structural SEs, bounds/degeneration, "
         "%RSE <= 50%, ETA shrinkage <= 30%, BSV 1-100%, positive sigma) plus "
         "PharmAgent's condition number <= 1000 extension; "
         "returns PASS / CONDITIONAL PASS / FAIL.",
         "qc",
         {
             "type": "object",
             "properties": {
                 "max_rse_pct": {"type": "number"},
                 "max_shrinkage_pct": {"type": "number"},
                 "min_bsv_cv_pct": {"type": "number"},
                 "max_bsv_cv_pct": {"type": "number"},
                 "max_condition_number": {"type": "number"},
                 "theta_bounds": {"type": "object"},
             },
             "required": [],
         },
         run_poppk_qc),
]
