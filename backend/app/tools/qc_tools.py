"""QC Agent tools: independent diagnostic review of NCA and PopPK/NLME estimation.

A subset of the full 15-point NCA checklist (Phase 1) plus a dedicated
Population PK (NLME) numerical estimation & convergence QC checklist:
  * 8 upstream estimation-health criteria modeled on ``AI-assisted-MIDD``
    (``poppk-estimation/scripts/poppk_checks.R``):
    ``ofv_finite``, ``covariance_step``, ``structural_se_present``,
    ``theta_not_on_bound``, ``structural_rse``, ``eta_shrinkage``,
    ``bsv_range``, and ``residual_error_positive``.
  * 1 PharmAgent extension: ``condition_number <= 1000`` (sourced from
    ``app.compute.nlme._COND_RED_FLAG``; tagged with
    ``source="pharmagent_extension"``).

Scope & Regulatory Caveat:
  ``run_poppk_qc`` evaluates numerical estimation and parameter-uncertainty
  health only. It is a necessary prerequisite but does NOT establish model or
  regulatory acceptance, which additionally requires goodness-of-fit (GOF),
  residual diagnostics (IWRES/CWRES/NPDE), visual predictive checks (VPC/pcVPC),
  predictive qualification for the intended context of use, and human
  pharmacometrician review.

Provenance & Gate Enforcement:
  * ``run_poppk_qc`` binds its verdict to the canonical SHA-256 digest of
    ``state.nlme_results`` (``state.poppk_qc_nlme_digest``) so a subsequent
    NLME fit immediately invalidates a prior QC verdict.
  * Flagged NCA terminal slopes (``lambda_z``) are recorded in
    ``state.nca_flagged_subjects`` with a non-bypassable analyst accept/revise
    gate in ``state.lambda_z_review`` bound to the digest of
    ``state.nca_parameters``.
"""
from __future__ import annotations

import math
from typing import Any

from app.compute.pk_models import REGISTRY
from app.core.audit import hash_payload
from app.core.pharmstate import PharmState
from app.tools.base import Tool, ToolContext, ToolResult

# NCA thresholds (named constants, not magic numbers).
MIN_SUBJECTS = 6
MIN_LAMBDAZ_POINTS = 3
MIN_LAMBDAZ_R2_ADJ = 0.80
MIN_LAMBDAZ_SPAN_RATIO = 2.0
MAX_PCT_EXTRAP = 20.0
MAX_MISSING_PCT = 20.0

# PopPK (NLME) estimation-health thresholds.
# Upstream AI-assisted-MIDD poppk_checks.R defaults:
MAX_POPPK_RSE_PCT = 50.0
MAX_POPPK_SHRINKAGE_PCT = 30.0
MIN_POPPK_BSV_CV_PCT = 1.0
MAX_POPPK_BSV_CV_PCT = 100.0
# PharmAgent extension (not an upstream AI-assisted-MIDD poppk_checks.R criterion):
MAX_POPPK_CONDITION_NUMBER = 1000.0

_THETA_REL_MIN = 1e-7
_THETA_REL_MAX = 1e7
_BOUND_REL_TOL = 1e-4  # "on bound" = within this RELATIVE distance of the bound itself

# QC acceptance thresholds are fixed policy, not model-controllable arguments: a
# caller (LLM) that could loosen them could turn a WARN into a PASS. The tool entry
# points reject these keys; evaluate_poppk_qc keeps validated keyword overrides for
# programmatic (non-agent) use only.
_NCA_POLICY_ARGS = frozenset({
    "min_span_ratio", "min_r2_adj", "min_points", "max_pct_extrap", "max_missing_pct",
    "lambda_z_decision", "notes",
})
_POPPK_POLICY_ARGS = frozenset({
    "max_rse_pct", "max_shrinkage_pct", "min_bsv_cv_pct", "max_bsv_cv_pct",
    "max_condition_number",
})

POPPK_QC_SCOPE = "estimation_health_only"
POPPK_QC_QUALIFICATION_NOTE = (
    "Evaluates numerical estimation and covariance health only; does not establish "
    "model or regulatory acceptance (which requires GOF, CWRES/NPDE residual "
    "diagnostics, VPC/pcVPC predictive checks, context-of-use assessment, and "
    "human pharmacometrician review)."
)

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


def nlme_fit_digest(nl: dict[str, Any] | None) -> str | None:
    """Return canonical SHA-256 digest of ``nlme_results`` for QC provenance binding."""
    if not isinstance(nl, dict) or nl.get("status") != "ok":
        return None
    return hash_payload(nl)


def nca_parameters_digest(params: list[dict[str, Any]] | None) -> str | None:
    """Return canonical SHA-256 digest of ``nca_parameters`` for lambda_z gate binding."""
    if not isinstance(params, list) or not params:
        return None
    return hash_payload(params)


def is_poppk_qc_current(state: PharmState) -> bool:
    """True iff ``state.poppk_qc_verdict`` exists and matches ``state.nlme_results``."""
    digest = nlme_fit_digest(state.nlme_results)
    return bool(
        digest is not None
        and state.poppk_qc_verdict is not None
        and state.poppk_qc_nlme_digest == digest
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


def _reject_policy_overrides(tool: str, args: dict[str, Any], keys: frozenset[str]) -> None:
    """Refuse QC-policy overrides on a tool call (thresholds are fixed; decisions are human)."""
    bad = sorted(k for k in args if k in keys)
    if bad:
        hint = (" Analyst accept/revise decisions are recorded only through the "
                "review_lambda_z_fits control (a human action), never via run_qc."
                if {"lambda_z_decision", "notes"} & set(bad) else "")
        raise ValueError(
            f"{tool} does not accept {bad}: QC thresholds are fixed policy.{hint}")


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
        n_raw = p.get("lambda_z_n_points")
        n_pts = int(n_raw) if _is_finite_num(n_raw) else 0
        r2 = p.get("lambda_z_r2_adj")
        span = p.get("lambda_z_span_ratio")
        if n_pts < min_points:
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
                "n_points": n_pts,
                "reasons": reasons,
            })
    return flagged


def _sid(x: Any) -> str:
    return str(x)


def unaccepted_lambda_z_subjects(
    flagged: list[dict[str, Any]], review: dict[str, Any] | None
) -> list[Any]:
    """Flagged subject ids the analyst has NOT individually accepted."""
    accepted = {_sid(i) for i in (review or {}).get("accepted_subjects") or []}
    return [f["id"] for f in flagged if _sid(f["id"]) not in accepted]


def _build_lambda_z_review(
    flagged_subjects: list[dict[str, Any]],
    *,
    nca_digest: str | None = None,
    decision: str | None = None,
    notes: str = "",
    subject_ids: list[Any] | None = None,
    prior: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct the analyst accept/revise gate state for flagged lambda_z fits.

    Decisions are tracked PER SUBJECT: ``accept`` marks only the named subjects
    (default: all currently flagged); the gate is ``accepted`` only when every
    flagged subject is accepted and none is awaiting revision. Prior decisions
    carry over only while ``nca_digest`` is unchanged.
    """
    flagged_ids = [f["id"] for f in flagged_subjects]
    flagged_keys = {_sid(i) for i in flagged_ids}
    same_data = bool(prior) and prior.get("nca_digest") == nca_digest
    prior_accepted = [i for i in (prior or {}).get("accepted_subjects") or []
                      if same_data and _sid(i) in flagged_keys]
    prior_revise = [i for i in (prior or {}).get("revision_subjects") or []
                    if same_data and _sid(i) in flagged_keys]

    if not flagged_subjects:
        if decision is not None:
            raise ValueError("no flagged lambda_z fits to review")
        return {
            "status": "not_required", "gate_open": False, "decision": None,
            "flagged_subjects": [], "accepted_subjects": [], "revision_subjects": [],
            "nca_digest": nca_digest, "notes": "",
        }

    norm: str | None = None
    targets = list(flagged_ids)
    if decision is not None:
        norm = str(decision).strip().lower()
        if norm not in {"accept", "revise"}:
            raise ValueError(
                f"lambda_z decision must be 'accept' or 'revise'; got {decision!r}"
            )
        if subject_ids is not None:
            if not isinstance(subject_ids, list) or not subject_ids:
                raise ValueError("subject_ids must be a non-empty list of flagged subject ids")
            unknown = [i for i in subject_ids if _sid(i) not in flagged_keys]
            if unknown:
                raise ValueError(
                    f"subject_ids {unknown} are not flagged lambda_z subjects "
                    f"(flagged: {flagged_ids})")
            # Map back to the canonical flagged id (1 vs "1") and drop duplicates.
            canon = {_sid(i): i for i in flagged_ids}
            targets = list(dict.fromkeys(canon[_sid(i)] for i in subject_ids))

    tkeys = {_sid(i) for i in targets}
    if norm == "accept":
        accepted = prior_accepted + [i for i in targets if _sid(i) not in {_sid(a) for a in prior_accepted}]
        revise = [i for i in prior_revise if _sid(i) not in tkeys]
    elif norm == "revise":
        accepted = [i for i in prior_accepted if _sid(i) not in tkeys]
        revise = prior_revise + [i for i in targets if _sid(i) not in {_sid(r) for r in prior_revise}]
    else:
        accepted, revise = prior_accepted, prior_revise

    akeys = {_sid(i) for i in accepted}
    remaining = [i for i in flagged_ids if _sid(i) not in akeys]
    if revise:
        status = "revision_requested"
    elif remaining:
        status = "awaiting_analyst_decision"
    else:
        status = "accepted"
    review: dict[str, Any] = {
        "status": status,
        "gate_open": status != "accepted",
        "decision": norm,
        "flagged_subjects": flagged_subjects,
        "accepted_subjects": accepted,
        "revision_subjects": revise,
        "reviewed_subjects": targets if norm else [],
        "nca_digest": nca_digest,
        "notes": (notes or (prior or {}).get("notes", "")) if same_data else notes,
    }
    if status == "awaiting_analyst_decision":
        review["prompt"] = (
            f"Subjects with flagged lambda_z fits awaiting a decision ({remaining}): "
            "the analyst must accept or revise them via the lambda_z review control."
        )
    return review


def is_nca_qc_current(state: PharmState) -> bool:
    """True iff run_qc has run on exactly the current NCA parameters (or there are none)."""
    params = state.nca_parameters or []
    if not params:
        return True
    return bool(
        state.qc_verdict is not None
        and (state.lambda_z_review or {}).get("nca_digest") == nca_parameters_digest(params)
    )


def assert_deliverable_qc_gates(state: PharmState, deliverable: str = "report") -> None:
    """Refuse a QC-governed deliverable (report, Module 2.7.2, briefing memo) unless
    NCA QC is current and not FAIL, every flagged lambda_z fit is human-accepted for
    the current NCA parameters, and any successful NLME fit has a current, non-FAIL
    PopPK estimation QC verdict."""
    pre = f"Cannot generate {deliverable}"
    params = state.nca_parameters or []
    lz_rev = state.lambda_z_review or {}

    if params and state.qc_verdict == "FAIL" and is_nca_qc_current(state):
        raise ValueError(
            f"{pre}: NCA QC verdict is FAIL (blocking checks: "
            + ", ".join(c["check"] for c in (state.qc_checklist or []) if c.get("status") == "FAIL")
            + "). Correct the NCA inputs and re-run run_qc.")

    flagged = (_collect_flagged_lambda_z_subjects(params) if params
               else list(state.nca_flagged_subjects or []))
    if lz_rev.get("status") == "revision_requested":
        raise ValueError(
            f"{pre} while lambda_z_review status is 'revision_requested'; revise the terminal "
            "regression window, then have the analyst record an accept decision through the "
            "lambda_z review control.")
    if lz_rev.get("status") == "awaiting_analyst_decision" or (
        flagged and lz_rev.get("status") != "accepted"
    ):
        raise ValueError(
            f"{pre} while flagged lambda_z fits are awaiting analyst decision "
            f"(flagged: {[f['id'] for f in flagged]}). A human analyst must accept or revise "
            "them through the lambda_z review control after run_qc (sessions or NCA results "
            "created before span-ratio support also need run_qc + review).")
    if params and not is_nca_qc_current(state):
        if lz_rev.get("nca_digest"):
            raise ValueError(
                f"{pre}: NCA parameters changed after lambda_z review / QC; re-run run_qc "
                "and have the analyst re-review.")
        raise ValueError(
            f"{pre}: NCA QC (run_qc) has not been run for the current NCA parameters.")
    if flagged:
        pending = unaccepted_lambda_z_subjects(flagged, lz_rev)
        if pending:
            raise ValueError(
                f"{pre} while flagged lambda_z fits are awaiting analyst decision for "
                f"subject(s) {pending}.")

    nl = state.nlme_results
    if nl and nl.get("status") == "ok":
        if not state.poppk_qc_verdict:
            raise ValueError(
                f"{pre}: PopPK estimation QC (run_poppk_qc) has not been run for the "
                "current NLME fit.")
        if state.poppk_qc_nlme_digest != nlme_fit_digest(nl):
            raise ValueError(
                f"{pre}: PopPK estimation QC verdict is stale (computed for a different "
                "NLME fit); re-run run_poppk_qc.")
        if state.poppk_qc_verdict == "FAIL":
            raise ValueError(f"{pre}: PopPK estimation QC verdict is FAIL.")


def run_qc(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    params = state.nca_parameters or []
    quality = state.data_quality or {}
    checks: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    _reject_policy_overrides("run_qc", args, _NCA_POLICY_ARGS)
    min_span_ratio = MIN_LAMBDAZ_SPAN_RATIO
    min_r2_adj = MIN_LAMBDAZ_R2_ADJ
    max_pct_extrap = MAX_PCT_EXTRAP
    max_missing_pct = MAX_MISSING_PCT
    min_points = MIN_LAMBDAZ_POINTS

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
    few = [
        p["subject"] for p in params
        if not _is_finite_num(p.get("lambda_z_n_points"))
        or int(p["lambda_z_n_points"]) < min_points
    ]
    checks.append(_check("Lambda_z points >= 3", "PASS" if not few else "WARN",
                         f"{len(few)} subject(s) with < {min_points} points"))
    if few:
        issues.append({"severity": "MEDIUM", "issue": f"few lambda_z points: {few}"})

    # 4. lambda_z fit quality
    poor = [
        p["subject"] for p in params
        if not _is_finite_num(p.get("lambda_z_r2_adj"))
        or float(p["lambda_z_r2_adj"]) < min_r2_adj
    ]
    checks.append(_check("Lambda_z adj R^2 >= 0.80", "PASS" if not poor else "WARN",
                         f"{len(poor)} subject(s) below threshold or missing"))
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
    implausible = [
        p["subject"] for p in params
        if (p.get("CL_F") is not None and (not _is_finite_num(p["CL_F"]) or float(p["CL_F"]) <= 0))
        or (p.get("Vz_F") is not None and (not _is_finite_num(p["Vz_F"]) or float(p["Vz_F"]) <= 0))
    ]
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
    # Re-running QC never grants or clears an analyst decision: prior per-subject
    # decisions carry over only while the NCA parameters are unchanged.
    lambda_z_review = _build_lambda_z_review(
        flagged_subjects,
        nca_digest=nca_parameters_digest(params),
        prior=state.lambda_z_review,
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
    if lambda_z_review["status"] in {"awaiting_analyst_decision", "revision_requested"}:
        flagged_ids = [f["id"] for f in flagged_subjects]
        summary_msg += (
            f" Flagged lambda_z fits for subject(s) {flagged_ids}: "
            "analyst decision required (a human must accept or revise via the lambda_z review control)."
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
    """Record the HUMAN analyst's accept/revise decision for flagged NCA lambda_z fits.

    Registered ``human_only``: the registry refuses it on the agent/LLM path, so it
    runs only through the actor-attributed review endpoint (the audit entry records
    the authenticated actor). ``subject_ids`` limits the decision to named flagged
    subjects; omitted, it applies to all currently flagged subjects. The report gate
    opens only when every flagged subject is accepted for the current NCA digest.
    """
    params = state.nca_parameters
    if not params:
        raise ValueError("review_lambda_z_fits requires NCA parameters (compute_nca) in state")

    decision = args.get("decision")
    if decision is None:
        raise ValueError("review_lambda_z_fits requires decision='accept' or decision='revise'")

    flagged = _collect_flagged_lambda_z_subjects(params)
    review = _build_lambda_z_review(
        flagged,
        nca_digest=nca_parameters_digest(params),
        decision=str(decision),
        notes=str(args.get("notes") or ""),
        subject_ids=args.get("subject_ids"),
        prior=state.lambda_z_review,
    )
    reviewed_ids = review.get("reviewed_subjects") or []
    return ToolResult(
        summary=(
            f"Recorded analyst lambda_z review decision: {review['decision'].upper()} "
            f"for subject(s) {reviewed_ids}; gate is now '{review['status']}'."
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


def _parse_fit_theta_bounds(
    raw_bounds: Any,
) -> dict[str, tuple[float | None, float | None]]:
    """Normalize fit-recorded estimation bounds ``{param: (lower, upper)}``."""
    if not isinstance(raw_bounds, dict):
        raise ValueError(f"fit theta_bounds must be a dict; got {type(raw_bounds).__name__}")
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
    """Covariance-step status from what ``app.compute.nlme`` actually emits.

    nlme reports a failed / non-finite / non-positive-definite / near-singular
    information matrix through ``cov_note`` (matched in ``_COV_FAIL_PHRASES``) and
    returns empty standard errors. The step also fails when no standard error of
    any kind is available. Per-parameter completeness is judged separately by
    ``structural_se_present``.
    """
    cov_note = str(nl.get("cov_note") or "").strip()
    low_note = cov_note.lower()
    if any(phrase in low_note for phrase in _COV_FAIL_PHRASES):
        return False, cov_note
    if not theta:
        return False, "no structural parameters found"
    any_se = any(
        _is_finite_num(v)
        for group in (theta_rse, nl.get("omega_rse_pct") or {}, nl.get("sigma_rse_pct") or {})
        for v in group.values()
    )
    if not any_se:
        return False, cov_note or "covariance step not run or failed: no standard errors available"
    return True, "covariance step succeeded" if not cov_note else f"covariance step succeeded ({cov_note})"


def evaluate_poppk_qc(
    nl: dict[str, Any],
    *,
    max_rse_pct: float = MAX_POPPK_RSE_PCT,
    max_shrinkage_pct: float = MAX_POPPK_SHRINKAGE_PCT,
    min_bsv_cv_pct: float = MIN_POPPK_BSV_CV_PCT,
    max_bsv_cv_pct: float = MAX_POPPK_BSV_CV_PCT,
    max_condition_number: float = MAX_POPPK_CONDITION_NUMBER,
) -> dict[str, Any]:
    """Evaluate a PopPK/NLME fit against numerical estimation-health QC criteria.

    Scope:
      Checks numerical estimation and covariance health only
      (``scope="estimation_health_only"``). Does NOT establish regulatory or
      scientific model acceptance, which requires GOF, CWRES/NPDE residual
      diagnostics, VPC/pcVPC predictive checks, context-of-use evaluation, and
      human pharmacometrician review.

    Provenance:
      * 8 checks modeled on ``AI-assisted-MIDD``
        (``poppk-estimation/scripts/poppk_checks.R``), tagged
        ``source="ai_assisted_midd"``:
        - Blocking (FAIL): ``ofv_finite``, ``covariance_step``,
          ``structural_se_present``, ``theta_not_on_bound``,
          ``residual_error_positive``.
        - Caveat (WARN): ``structural_rse`` (> 50%), ``eta_shrinkage`` (> 30% or
          missing/NaN), ``bsv_range`` (outside 1-100% or missing/NaN).
      * 1 PharmAgent extension, tagged ``source="pharmagent_extension"``:
        - Caveat (WARN): ``condition_number`` (> 1000 or missing/NaN, from
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
    # Bounds must come from the actual fit configuration (nl["theta_bounds"] or
    # the fitted model's estimation box in REGISTRY[model_key].defaults) and
    # cover the entire fitted parameter set (all keys in theta).
    model_key = str(nl.get("model_key") or "")
    defaults = dict(REGISTRY[model_key].defaults) if model_key in REGISTRY else {}
    raw_fit_bounds = nl.get("theta_bounds")
    fit_bounds: dict[str, tuple[float | None, float | None]] = {}
    bound_cfg_err = ""
    has_explicit_fit_bounds = raw_fit_bounds is not None
    if has_explicit_fit_bounds:
        try:
            fit_bounds = _parse_fit_theta_bounds(raw_fit_bounds)
        except ValueError as exc:
            bound_cfg_err = str(exc)
    else:
        # Legacy fit without a recorded box: rebuild the optimiser's own box.
        from app.compute.nlme import theta_estimation_bounds  # lazy: heavy import
        fit_bounds = {
            k: (box[0], box[1])
            for k, box in theta_estimation_bounds(defaults, theta).items()
        }

    missing_bound_cfg = [
        k for k in theta
        if k not in fit_bounds
        or (fit_bounds[k][0] is None and fit_bounds[k][1] is None)
    ]
    on_bound: list[str] = []
    degenerate_thetas: list[str] = list(nl.get("degenerate_params") or [])

    for k, val in theta.items():
        if not _is_finite_num(val):
            if k not in degenerate_thetas:
                degenerate_thetas.append(k)
            continue
        fval = float(val)
        if k in fit_bounds:
            lo, hi = fit_bounds[k]
            # Relative to the BOUND, not an absolute floor: an absolute tolerance
            # flags legitimate small parameters (e.g. a 5e-5 /h rate constant).
            at_lo = lo is not None and (
                fval <= lo or abs(fval - lo) <= _BOUND_REL_TOL * abs(lo))
            at_hi = hi is not None and (
                fval >= hi or abs(fval - hi) <= _BOUND_REL_TOL * abs(hi))
            if (at_lo or at_hi) and k not in on_bound:
                on_bound.append(k)
        if fval <= 0.0 and k not in degenerate_thetas and k not in on_bound:
            degenerate_thetas.append(k)
            continue
        ref = float(defaults.get(k, 0.0) or 0.0)
        if (
            ref > 0.0
            and not (_THETA_REL_MIN <= fval / ref <= _THETA_REL_MAX)
            and k not in degenerate_thetas
        ):
            degenerate_thetas.append(k)

    bound_ok = (
        bool(theta)
        and not bound_cfg_err
        and not missing_bound_cfg
        and not on_bound
        and not degenerate_thetas
    )
    if bound_cfg_err:
        bound_detail = f"invalid fit theta_bounds: {bound_cfg_err}"
    elif missing_bound_cfg:
        bound_detail = (
            "incomplete theta_bounds in fit configuration "
            f"(missing finite bounds for: {', '.join(missing_bound_cfg)})"
        )
    elif on_bound and degenerate_thetas:
        bound_detail = (
            f"on bound: {', '.join(on_bound)}; "
            f"degenerate vs default box: {', '.join(degenerate_thetas)}"
        )
    elif on_bound:
        bound_detail = f"on bound: {', '.join(on_bound)}"
    elif degenerate_thetas:
        bound_detail = (
            f"degenerate relative to model defaults: {', '.join(degenerate_thetas)}"
        )
    elif not theta:
        bound_detail = "no structural parameters found"
    elif has_explicit_fit_bounds:
        bound_detail = "no THETA on declared fit bound (all fitted parameters covered)"
    else:
        bound_detail = (
            "no THETA on the model estimation box (all fitted parameters covered)"
        )
    checks.append(_check("theta_not_on_bound", "PASS" if bound_ok else "FAIL", bound_detail,
                         source="ai_assisted_midd"))
    if not bound_ok:
        issues.append({"severity": "HIGH", "issue": f"theta_not_on_bound failed: {bound_detail}"})

    # 5. structural_rse (WARN if any structural %RSE > max_rse_pct or missing)
    high_rse = [
        f"{k} ({float(theta_rse[k]):.1f}%)"
        for k in theta
        if _is_finite_num(theta_rse.get(k)) and float(theta_rse[k]) > max_rse_pct
    ]
    rse_ok = bool(theta) and not se_missing and not high_rse
    if high_rse:
        rse_detail = f"%RSE > {max_rse_pct:g}: {', '.join(high_rse)}"
    elif se_missing:
        rse_detail = f"%RSE unavailable for: {', '.join(se_missing)}"
    else:
        rse_detail = f"all structural %RSE <= {max_rse_pct:g}"
    checks.append(_check("structural_rse", "PASS" if rse_ok else "WARN", rse_detail,
                         source="ai_assisted_midd"))
    if not rse_ok:
        issues.append({"severity": "MEDIUM", "issue": rse_detail})

    # 6. eta_shrinkage (WARN if missing/NaN or any ETA shrinkage > max_shrinkage_pct)
    expected_iiv = list(
        dict.fromkeys(
            list(nl.get("iiv_params") or [])
            + list(omega_cv.keys())
            + list(shrinkage.keys())
        )
    )
    if not expected_iiv or not shrinkage:
        shr_ok = False
        shr_detail = "ETA shrinkage diagnostics missing or empty"
    else:
        bad_shr = [k for k in expected_iiv if not _is_finite_num(shrinkage.get(k))]
        high_shr = [
            f"{k} ({float(shrinkage[k]):.1f}%)"
            for k in expected_iiv
            if _is_finite_num(shrinkage.get(k)) and float(shrinkage[k]) > max_shrinkage_pct
        ]
        shr_ok = not bad_shr and not high_shr
        parts_shr: list[str] = []
        if high_shr:
            parts_shr.append(f"SD shrinkage > {max_shrinkage_pct:g}%: {', '.join(high_shr)}")
        if bad_shr:
            parts_shr.append(f"SD shrinkage missing/NaN: {', '.join(bad_shr)}")
        shr_detail = (
            "; ".join(parts_shr)
            if parts_shr
            else f"all ETA SD shrinkage <= {max_shrinkage_pct:g}%"
        )
    checks.append(_check("eta_shrinkage", "PASS" if shr_ok else "WARN", shr_detail,
                         source="ai_assisted_midd"))
    if not shr_ok:
        issues.append({"severity": "MEDIUM", "issue": shr_detail})

    # 7. bsv_range (WARN if missing/NaN or any omega CV% outside [min_bsv_cv_pct, max_bsv_cv_pct])
    if not expected_iiv or not omega_cv:
        bsv_ok = False
        bsv_detail = "BSV (omega CV%) diagnostics missing or empty"
    else:
        bad_bsv = [k for k in expected_iiv if not _is_finite_num(omega_cv.get(k))]
        odd_bsv = [
            f"{k} ({float(omega_cv[k]):.1f}%)"
            for k in expected_iiv
            if _is_finite_num(omega_cv.get(k))
            and (float(omega_cv[k]) < min_bsv_cv_pct or float(omega_cv[k]) > max_bsv_cv_pct)
        ]
        bsv_ok = not bad_bsv and not odd_bsv
        parts_bsv: list[str] = []
        if odd_bsv:
            parts_bsv.append(
                f"BSV CV% outside {min_bsv_cv_pct:g}-{max_bsv_cv_pct:g}%: {', '.join(odd_bsv)}"
            )
        if bad_bsv:
            parts_bsv.append(f"BSV CV% missing/NaN: {', '.join(bad_bsv)}")
        bsv_detail = (
            "; ".join(parts_bsv)
            if parts_bsv
            else f"all BSV CV% within {min_bsv_cv_pct:g}-{max_bsv_cv_pct:g}%"
        )
    checks.append(_check("bsv_range", "PASS" if bsv_ok else "WARN", bsv_detail,
                         source="ai_assisted_midd"))
    if not bsv_ok:
        issues.append({"severity": "LOW", "issue": bsv_detail})

    # 8. condition_number (WARN if missing/NaN/<=0 or > max_condition_number)
    # PharmAgent extension (from app.compute.nlme._COND_RED_FLAG = 1e3), not an
    # upstream AI-assisted-MIDD poppk_checks.R criterion.
    cond = nl.get("condition_number")
    if not _is_finite_num(cond) or float(cond) <= 0.0:
        cond_ok = False
        cond_detail = f"condition number missing or non-finite ({cond!r}; PharmAgent extension)"
        issues.append({
            "severity": "MEDIUM",
            "issue": f"condition number missing or non-finite: {cond!r} (PharmAgent extension)",
        })
    else:
        cond_ok = float(cond) <= max_condition_number
        cond_detail = (
            f"condition number = {float(cond):.1f} "
            f"(<= {max_condition_number:g}; PharmAgent extension)"
        )
        if not cond_ok:
            issues.append({
                "severity": "MEDIUM",
                "issue": (
                    f"high condition number: {float(cond):.1f} "
                    f"(> {max_condition_number:g}; PharmAgent extension)"
                ),
            })
    checks.append(_check(
        "condition_number",
        "PASS" if cond_ok else "WARN",
        cond_detail,
        source="pharmagent_extension",
    ))

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

    return {
        "verdict": verdict,
        "scope": POPPK_QC_SCOPE,
        "qualification_note": POPPK_QC_QUALIFICATION_NOTE,
        "checks": checks,
        "issues": issues,
    }


def run_poppk_qc(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Run PopPK (NLME) numerical estimation-health checks on ``state.nlme_results``."""
    nl = state.nlme_results
    if not nl or nl.get("status") != "ok":
        raise ValueError("run_poppk_qc requires a completed NLME fit (run_nlme) in state.nlme_results")

    _reject_policy_overrides("run_poppk_qc", args, _POPPK_POLICY_ARGS)
    out = evaluate_poppk_qc(nl)
    verdict = out["verdict"]
    checks = out["checks"]
    issues = out["issues"]
    digest = nlme_fit_digest(nl)
    statuses = [c["status"] for c in checks]

    return ToolResult(
        summary=(
            f"PopPK estimation QC verdict: {verdict} "
            f"({statuses.count('PASS')} pass, {statuses.count('WARN')} warn, "
            f"{statuses.count('FAIL')} fail; estimation health only — GOF, residual "
            "diagnostics, VPC, and human review still required for model qualification)."
        ),
        action=f"run_poppk_qc({nl.get('model_key', 'nlme')})",
        writes={
            "poppk_qc_verdict": verdict,
            "poppk_qc_issues": issues,
            "poppk_qc_checklist": checks,
            "poppk_qc_nlme_digest": digest,
        },
        result={
            "verdict": verdict,
            "scope": POPPK_QC_SCOPE,
            "qualification_note": POPPK_QC_QUALIFICATION_NOTE,
            "nlme_digest": digest,
            "model_key": nl.get("model_key"),
            "checks": checks,
            "issues": issues,
        },
    )


TOOLS = [
    Tool("run_qc",
         "Independent QC of the NCA analysis against a fixed diagnostic checklist; "
         "records flagged lambda_z subjects that need a human analyst's accept/revise "
         "review and returns PASS / CONDITIONAL PASS / FAIL. Thresholds are fixed policy.",
         "qc",
         {"type": "object", "properties": {}, "required": []},
         run_qc),
    Tool("review_lambda_z_fits",
         "HUMAN-ONLY: record the analyst's accept/revise decision for subjects with "
         "flagged NCA terminal slope (lambda_z) fits (span ratio < 2.0, adj R^2 < 0.80, "
         "or < 3 terminal points). Not callable by an agent; submitted via the review control.",
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
         review_lambda_z_fits,
         human_only=True),
    Tool("run_poppk_qc",
         "Independent PopPK (NLME) numerical estimation-health check against the 8 "
         "AI-assisted-MIDD estimation criteria (finite OFV, covariance step, structural "
         "SEs, fit bounds/degeneration, %RSE <= 50%, ETA shrinkage <= 30%, BSV 1-100%, "
         "positive sigma) plus PharmAgent's condition number <= 1000 extension. "
         "Thresholds are fixed policy. Evaluates estimation health only (does not replace "
         "GOF, residual diagnostics, VPC, or human model review); returns PASS / "
         "CONDITIONAL PASS / FAIL.",
         "qc",
         {"type": "object", "properties": {}, "required": []},
         run_poppk_qc),
]
