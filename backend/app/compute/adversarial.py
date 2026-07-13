"""Adversarial review engine — independent, deterministic refutation of results.

This is the "clean-context skeptic" surface: it does NOT trust the numbers the
analysis tools wrote. Where it can, it recomputes key quantities from the raw
data using *different, method-agnostic* arithmetic (e.g. Cmax = max observed
concentration, an AUC plausibility band from Cmax x Tlast) so a bug in the
primary code path — a units flip, a wrong scaling, a pooling error — shows up as
a discrepancy rather than being silently echoed back.

It is independent of the QC checklist (qc_tools): QC asks "does the analysis pass
our standard diagnostics"; the reviewer asks "can I break this result". Findings
carry a severity; the goal is checkable ("zero unresolved CRITICAL or HIGH").

Scientific decisions (drop a subject, accept a structural model) are never
auto-resolved here — those escalate to the pharmacometrician of record. The loop
driver (orchestrator) re-reviews until the goal is met.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

# ── severity + goal ─────────────────────────────────────────────────────────────
CRITICAL, HIGH, MEDIUM, LOW = "CRITICAL", "HIGH", "MEDIUM", "LOW"
_RANK = {CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3}
DEFAULT_GOAL = "zero unresolved CRITICAL or HIGH findings"

# ── physiological / numerical bounds (named, not magic) ─────────────────────────
MAX_APPARENT_CL_F = 1000.0   # L/h — apparent CL/F can exceed organ flow if F<1,
#                              but >1000 L/h is almost always a units/scaling error
AUC_BAND_LO = 0.10           # reported AUClast must lie within [LO, HI] x (Cmax*Tlast);
AUC_BAND_HI = 1.05           # a units flip (x1000) blows past HI, a zeroed curve below LO
CMAX_REL_TOL = 0.02          # reported Cmax must match max(observed) within 2%
CLF_REL_TOL = 0.05           # reported CL/F must match dose/AUCinf within 5%
COND_OVERPARAM = 1000.0      # NLME covariance condition number red flag
MAX_SHRINKAGE_PCT = 30.0     # eta-shrinkage above which EBEs are uninterpretable
MAX_IIV_CV_PCT = 150.0       # IIV CV% above which structural misspecification likely
MAX_THETA_RSE_PCT = 50.0     # structural parameter precision floor
AIC_TIE_DELTA = 2.0          # AIC gap below which two structural models are indistinct


def _finding(fid: str, severity: str, target: str, claim: str,
             evidence: str, action: str) -> dict[str, Any]:
    """A single refutation. `claim` is what the analysis asserts; `evidence` is the
    independent observation that challenges it; `action` is the suggested remedy."""
    return {"id": fid, "severity": severity, "target": target, "claim": claim,
            "evidence": evidence, "suggested_action": action, "resolved": False}


# ── state-internal refutations (no raw data needed) ─────────────────────────────
def _nca_internal(params: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in params:
        sid = p.get("subject")
        auc_l, auc_i = p.get("AUC_last"), p.get("AUC_inf")
        if auc_l is not None and auc_i is not None and auc_i < auc_l - 1e-9:
            out.append(_finding(
                f"nca-aucorder-{sid}", HIGH, f"subject {sid} AUC",
                f"AUCinf ({auc_i:.3g}) reported below AUClast ({auc_l:.3g})",
                "AUCinf = AUClast + Clast/lambda_z must be >= AUClast by construction",
                "re-examine the terminal extrapolation; a negative lambda_z or "
                "swapped columns produces this"))
        ext = p.get("pct_AUC_extrap")
        if ext is not None and not (0.0 <= ext <= 100.0):
            out.append(_finding(
                f"nca-extrap-{sid}", HIGH, f"subject {sid} %extrap",
                f"%AUC extrapolated = {ext:.3g}% is outside [0, 100]",
                "an out-of-range extrapolation fraction indicates a lambda_z error",
                "refit the terminal slope or exclude the subject"))
        clf = p.get("CL_F")
        if clf is not None and clf <= 0:
            out.append(_finding(
                f"nca-clf-sign-{sid}", CRITICAL, f"subject {sid} CL/F",
                f"CL/F = {clf:.3g} is non-physiological (<= 0)",
                "clearance must be strictly positive",
                "AUCinf is likely non-positive; investigate the concentration data"))
        elif clf is not None and clf > MAX_APPARENT_CL_F:
            out.append(_finding(
                f"nca-clf-mag-{sid}", HIGH, f"subject {sid} CL/F",
                f"CL/F = {clf:.4g} L/h exceeds {MAX_APPARENT_CL_F:.0f} L/h",
                "apparent CL/F this large usually means a dose/AUC unit mismatch",
                "verify dose units (mg) vs concentration units (e.g. ng/mL) and AUC scaling"))
        vzf = p.get("Vz_F")
        if vzf is not None and vzf <= 0:
            out.append(_finding(
                f"nca-vzf-{sid}", HIGH, f"subject {sid} Vz/F",
                f"Vz/F = {vzf:.3g} is non-physiological (<= 0)",
                "volume of distribution must be positive", "investigate lambda_z and AUCinf"))
        th = p.get("t_half")
        if th is not None and th <= 0:
            out.append(_finding(
                f"nca-thalf-{sid}", HIGH, f"subject {sid} t-half",
                f"t-half = {th:.3g} h is non-physiological (<= 0)",
                "half-life = ln2/lambda_z requires lambda_z > 0",
                "the terminal slope was estimated as flat or positive; refit"))
        # CL/F internal consistency: CL/F should equal dose / AUCinf
        dose = p.get("dose")
        if (clf is not None and clf > 0 and dose and auc_i and auc_i > 0):
            implied = dose / auc_i
            if abs(implied - clf) / clf > CLF_REL_TOL:
                out.append(_finding(
                    f"nca-clf-consist-{sid}", HIGH, f"subject {sid} CL/F consistency",
                    f"reported CL/F = {clf:.4g} but dose/AUCinf = {implied:.4g}",
                    f"the two disagree by {abs(implied - clf) / clf * 100:.0f}% — "
                    "CL/F must equal dose/AUCinf",
                    "a scaling factor was applied inconsistently between dose and AUC"))
    return out


def _nca_dose_monotonic(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Across dose groups, mean exposure should rise with dose (linear PK). A
    decrease is not proof of error but warrants a nonlinearity / data check."""
    rows = summary.get("by_dose") or []
    pairs = [(r.get("dose"), r.get("AUC_inf_geomean")) for r in rows
             if r.get("dose") is not None and r.get("AUC_inf_geomean") is not None]
    pairs.sort(key=lambda x: x[0])
    out: list[dict[str, Any]] = []
    for (d0, a0), (d1, a1) in zip(pairs, pairs[1:]):
        if a1 < a0 - 1e-9:
            out.append(_finding(
                f"nca-monotonic-{d0}-{d1}", MEDIUM, "dose-exposure",
                f"mean AUCinf falls from {a0:.3g} ({d0}) to {a1:.3g} ({d1}) as dose rises",
                "exposure decreasing with dose contradicts dose-proportional PK",
                "check for nonlinearity (saturable clearance) or a dose/exposure data error"))
    return out


def _nlme_refute(nl: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if nl.get("converged") is False:
        out.append(_finding(
            "nlme-converge", HIGH, "NLME convergence",
            "the population fit is reported as not converged",
            "estimates and standard errors from a non-converged fit are unreliable",
            "revisit initial estimates, simplify the model, or change the estimation method"))
    cond = nl.get("condition_number")
    if cond is not None and cond > COND_OVERPARAM:
        out.append(_finding(
            "nlme-cond", HIGH, "NLME conditioning",
            f"covariance condition number = {cond:.0f} (> {COND_OVERPARAM:.0f})",
            "high conditioning signals an over-parameterized / poorly identified model",
            "drop a redundant random effect or covariate; check parameter correlations"))
    for p, cv in (nl.get("omega_cv_pct") or {}).items():
        if cv is not None and cv > MAX_IIV_CV_PCT:
            out.append(_finding(
                f"nlme-iiv-{p}", MEDIUM, f"NLME IIV on {p}",
                f"IIV on {p} = {cv:.0f}% CV (> {MAX_IIV_CV_PCT:.0f}%)",
                "extreme inter-individual variability often reflects structural misspecification",
                "reconsider the structural model or whether IIV on this parameter is supported"))
    for p, sh in (nl.get("shrinkage_pct") or {}).items():
        if sh is not None and sh > MAX_SHRINKAGE_PCT:
            out.append(_finding(
                f"nlme-shrink-{p}", MEDIUM, f"NLME shrinkage on {p}",
                f"eta-shrinkage on {p} = {sh:.0f}% (> {MAX_SHRINKAGE_PCT:.0f}%)",
                "high shrinkage makes individual EBEs and EBE-based diagnostics uninformative",
                "do not interpret individual etas for this parameter; the design is sparse for it"))
    for p, rse in (nl.get("theta_rse_pct") or {}).items():
        if rse is not None and rse > MAX_THETA_RSE_PCT:
            out.append(_finding(
                f"nlme-rse-{p}", MEDIUM, f"NLME precision on {p}",
                f"{p} estimated with {rse:.0f}% RSE (> {MAX_THETA_RSE_PCT:.0f}%)",
                "a structural parameter this imprecise is barely identified by the data",
                "the data may not support this parameter; consider fixing or removing it"))
    return out


# ── raw-data recomputation (the genuine independence) ───────────────────────────
def _obs_by_subject(df: pd.DataFrame, roles: dict[str, str]) -> dict[Any, pd.DataFrame]:
    """Group observation rows (EVID==0 / MDV==0) by subject, time-sorted."""
    inv = {v: k for k, v in roles.items()}  # role -> column name
    idc, tc, dvc = inv.get("ID"), inv.get("TIME"), inv.get("DV")
    if not (idc and tc and dvc) or not all(c in df.columns for c in (idc, tc, dvc)):
        return {}
    work = df.copy()
    work[dvc] = pd.to_numeric(work[dvc], errors="coerce")
    work[tc] = pd.to_numeric(work[tc], errors="coerce")
    evc, mdvc = inv.get("EVID"), inv.get("MDV")
    if evc and evc in work.columns:
        work = work[pd.to_numeric(work[evc], errors="coerce").fillna(0) == 0]
    elif mdvc and mdvc in work.columns:
        work = work[pd.to_numeric(work[mdvc], errors="coerce").fillna(0) == 0]
    work = work.dropna(subset=[dvc, tc])
    out: dict[Any, pd.DataFrame] = {}
    for sid, g in work.groupby(idc):
        out[sid] = g.sort_values(tc)[[tc, dvc]].rename(columns={tc: "t", dvc: "c"})
    return out


def _nca_recompute(params: list[dict[str, Any]], obs: dict[Any, pd.DataFrame]
                   ) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_id = {str(p.get("subject")): p for p in params}
    for sid, g in obs.items():
        p = by_id.get(str(sid))
        if p is None or g.empty:
            continue
        c = g["c"].to_numpy()
        t = g["t"].to_numpy()
        obs_cmax = float(c.max())
        rep_cmax = p.get("Cmax")
        # Cmax: exact, method-independent
        if rep_cmax and obs_cmax > 0 and abs(rep_cmax - obs_cmax) / obs_cmax > CMAX_REL_TOL:
            out.append(_finding(
                f"recompute-cmax-{sid}", CRITICAL, f"subject {sid} Cmax",
                f"reported Cmax = {rep_cmax:.4g}",
                f"max observed concentration is {obs_cmax:.4g} "
                f"({abs(rep_cmax - obs_cmax) / obs_cmax * 100:.0f}% off) — "
                "an independent scan of the raw data disagrees",
                "a units conversion or column mapping differs between the analysis and the data"))
        # AUClast plausibility band from Cmax * Tlast (catches units flips)
        rep_auc = p.get("AUC_last")
        tlast = float(t.max())
        if rep_auc and obs_cmax > 0 and tlast > 0:
            ceiling = obs_cmax * tlast
            ratio = rep_auc / ceiling
            if not (AUC_BAND_LO <= ratio <= AUC_BAND_HI):
                out.append(_finding(
                    f"recompute-aucband-{sid}", CRITICAL, f"subject {sid} AUClast",
                    f"reported AUClast = {rep_auc:.4g}",
                    f"implausible vs Cmax*Tlast = {ceiling:.4g} (ratio {ratio:.2g}, "
                    f"expected {AUC_BAND_LO}-{AUC_BAND_HI}) — likely a units or scaling error",
                    "verify concentration/time/AUC units are mutually consistent"))
    return out


def _pkmodel_refute(pm: dict[str, Any]) -> list[dict[str, Any]]:
    """Refutations over a structural-model selection (fit_pk_model / compare mode)."""
    out: list[dict[str, Any]] = []
    is_compare = pm.get("mode") == "compare"
    best = (pm.get("best") or {}) if is_compare else pm
    n_sub = best.get("n_subjects")
    n_conv = best.get("n_converged")
    if n_sub and n_conv is not None and n_conv < n_sub:
        out.append(_finding(
            "pkmodel-converge", HIGH, "structural fit convergence",
            f"only {n_conv}/{n_sub} subjects converged for "
            f"{best.get('label', 'the selected model')}",
            "a model chosen as best while some subjects failed to converge is unreliable",
            "inspect the non-converged subjects; reconsider the winning structural model"))
    ranking = pm.get("ranking") or []
    if is_compare and len(ranking) >= 2:
        top, second = ranking[0].get("aic"), ranking[1].get("aic")
        if top is not None and second is not None and abs(second - top) < AIC_TIE_DELTA:
            out.append(_finding(
                "pkmodel-tie", MEDIUM, "structural model selection",
                f"best and runner-up AIC differ by {abs(second - top):.2g} "
                f"(< {AIC_TIE_DELTA:.0f})",
                "an AIC gap below ~2 does not statistically distinguish the two models",
                "report the ambiguity; the winning model is not clearly preferred"))
    return out


def _engine_refute(ec: dict[str, Any]) -> list[dict[str, Any]]:
    """Refutations over a cross-engine comparison (run_engine_comparison)."""
    out: list[dict[str, Any]] = []
    if not ec.get("winner"):
        out.append(_finding(
            "engine-nowinner", HIGH, "cross-engine selection",
            "the engine comparison produced no winning engine/model",
            "a comparison that selects no winner cannot confirm the model",
            "check engine availability and that at least one candidate fit converged"))
    # Count engines that actually produced a usable fit for the winning model —
    # not merely engines that were installed. An engine that skipped/failed the
    # model (e.g. nlmixr2 on an unsupported 2-cmt model) did NOT cross-confirm it.
    n_fit = len({r.get("engine") for r in (ec.get("prediction_ranking") or [])})
    if n_fit < 2:
        out.append(_finding(
            "engine-single", MEDIUM, "cross-engine confirmation",
            f"only {n_fit} engine produced a usable fit for the winning model",
            "a 'cross-engine' winner confirmed by a single engine is not cross-confirmed",
            "ensure a second engine supports and fits this model before "
            "claiming cross-engine confirmation"))
    return out


def _scm_refute(scm: dict[str, Any]) -> list[dict[str, Any]]:
    """Refutations over a stepwise covariate model (run_scm).

    Stepwise (forward/backward) selection is a data-driven search, so the
    retained effects' standard errors and p-values are optimistic — the classic
    post-selection-inference problem (Derksen & Keselman 1992; Harrell 2015). The
    finding is informational (MEDIUM, non-blocking): it does not condemn the model,
    it flags that the reported precision overstates certainty."""
    out: list[dict[str, Any]] = []
    selected = scm.get("selected") or []
    if selected:
        names = ", ".join(
            f"{e.get('param')}~{e.get('covariate')}" for e in selected)
        out.append(_finding(
            "scm-postselection", MEDIUM, "stepwise covariate selection",
            f"{len(selected)} covariate effect(s) ({names}) were chosen by a "
            "forward/backward stepwise search",
            "after data-driven stepwise selection the retained effects' standard "
            "errors and p-values are optimistic (winner's curse), and the search "
            "can retain noise covariates or drop confounders",
            "confirm on a pre-specified covariate set, or validate the selection "
            "by resampling (cross-validation / bootstrap); keep a predictive model "
            "distinct from a causal-effect claim"))
    return out


# ── public API ──────────────────────────────────────────────────────────────────
def review(state_dump: dict[str, Any], df: pd.DataFrame | None,
           roles: dict[str, str] | None, goal: str = DEFAULT_GOAL) -> dict[str, Any]:
    """Run all refutation checks over the current analysis state.

    `state_dump` is PharmState.model_dump(); `df` + `roles` are the raw dataset and
    its column-role map (optional — recomputation checks degrade gracefully without
    them, the state-internal checks still run).
    """
    findings: list[dict[str, Any]] = []
    params = state_dump.get("nca_parameters") or []
    if params:
        findings += _nca_internal(params)
        if df is not None and roles:
            findings += _nca_recompute(params, _obs_by_subject(df, roles))
    summary = state_dump.get("nca_summary") or {}
    if summary:
        findings += _nca_dose_monotonic(summary)
    nl = state_dump.get("nlme_results") or {}
    if nl.get("status") == "ok":
        findings += _nlme_refute(nl)
    pm = state_dump.get("pk_model_results") or {}
    if pm.get("status") == "ok":
        findings += _pkmodel_refute(pm)
    ec = state_dump.get("engine_comparison_results") or {}
    if ec.get("status") == "ok":
        findings += _engine_refute(ec)
    scm = state_dump.get("scm_results") or {}
    if scm.get("status") == "ok":
        findings += _scm_refute(scm)

    findings.sort(key=lambda f: _RANK.get(f["severity"], 9))
    counts = {sev: sum(1 for f in findings if f["severity"] == sev)
              for sev in (CRITICAL, HIGH, MEDIUM, LOW)}
    unresolved_blocking = [f for f in findings
                           if not f["resolved"] and f["severity"] in (CRITICAL, HIGH)]
    checked = {
        "nca": bool(params),
        "nca_recompute": bool(params and df is not None and roles),
        "nlme": nl.get("status") == "ok",
        "pkmodel": pm.get("status") == "ok",
        "engine": ec.get("status") == "ok",
        "scm": scm.get("status") == "ok",
    }
    # Fail closed: a review that had nothing to inspect must not report GOAL MET —
    # otherwise an empty/mismatched state silently rubber-stamps as passing.
    nothing_checked = not any(checked.values())
    goal_met = (len(unresolved_blocking) == 0) and not nothing_checked
    return {
        "goal": goal,
        "goal_met": goal_met,
        "findings": findings,
        "counts": counts,
        "n_findings": len(findings),
        "nothing_checked": nothing_checked,
        "checked": checked,
    }
