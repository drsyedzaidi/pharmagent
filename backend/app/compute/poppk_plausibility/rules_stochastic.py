"""Deterministic random-effects (IIV/Omega), residual-error, and prediction plausibility rules.

Implements rules PL-012 through PL-016 and PL-020.
"""
from __future__ import annotations

import math

import numpy as np

from app.compute.poppk_plausibility.rules_structural import make_finding
from app.compute.poppk_plausibility.schema import (
    FindingStatus,
    PlausibilityFinding,
    PopPKModelInput,
    RuleCatalog,
    SeverityLevel,
)


def eval_pl012_iiv_magnitude(
    catalog: RuleCatalog, model: PopPKModelInput
) -> list[PlausibilityFinding]:
    rule = catalog.by_id("PL-012")
    if not model.iiv:
        return [make_finding(rule, parameter="IIV", status="NOT_APPLICABLE")]
    out: list[PlausibilityFinding] = []
    for item in sorted(model.iiv, key=lambda x: x.parameter):
        if item.omega_var is not None and (not math.isfinite(item.omega_var) or item.omega_var < 0):
            out.append(
                make_finding(
                    rule, parameter=item.parameter,
                    status="UNRESOLVED", severity="WARN",
                    rationale="Variance is negative or non-finite (flagged under PL-013).",
                )
            )
            continue
        if item.cv_pct is not None and (not math.isfinite(item.cv_pct) or item.cv_pct < 0):
            out.append(
                make_finding(
                    rule, parameter=item.parameter,
                    status="UNRESOLVED", severity="WARN",
                    rationale="CV% is negative or non-finite (flagged under PL-013).",
                )
            )
            continue
        calc_cv: float | None = None
        if item.omega_var is not None:
            if item.distribution == "lognormal":
                calc_cv = 100.0 * math.sqrt(math.exp(min(item.omega_var, 50.0)) - 1.0)
            else:
                calc_cv = 100.0 * math.sqrt(item.omega_var)
        if item.cv_pct is not None and calc_cv is not None:
            if abs(calc_cv - item.cv_pct) / max(calc_cv, 1e-9) > 0.05:
                out.append(
                    make_finding(
                        rule, parameter=item.parameter,
                        status="FLAGGED", severity="HIGH",
                        observed_value=item.cv_pct, observed_unit="%CV",
                        normalized_value=calc_cv, normalized_unit="%CV",
                        expected_range=f"{calc_cv:.2f}% CV (+/-5%)",
                        rationale="Reported cv_pct is mathematically inconsistent with reported omega_var.",
                        override_allowed=False,
                    )
                )
                continue
        cv = item.cv_pct if item.cv_pct is not None else calc_cv
        if cv is None:
            out.append(make_finding(rule, parameter=item.parameter, status="UNRESOLVED", severity="WARN"))
            continue
        obs = item.omega_var if item.omega_var is not None else cv
        if cv > 200.0:
            st, sev, rng = "FLAGGED", "HIGH", "<= 100% CV (warn), <= 200% CV (high)"
        elif cv > 100.0:
            st, sev, rng = "FLAGGED", "WARN", "<= 100% CV"
        else:
            st, sev, rng = "PASS", "INFO", "<= 100% CV"
        out.append(
            make_finding(
                rule, parameter=item.parameter,
                status=st, severity=sev,
                observed_value=obs, normalized_value=cv,
                normalized_unit="%CV", expected_range=rng,
                override_allowed=True,
            )
        )
    return out


def eval_pl013_variance_nonnegative(
    catalog: RuleCatalog, model: PopPKModelInput
) -> list[PlausibilityFinding]:
    rule = catalog.by_id("PL-013")
    has_any = bool(model.iiv or model.omega_matrix or model.sigma_matrix or model.residual_error)
    if not has_any:
        return [make_finding(rule, parameter="variances", status="NOT_APPLICABLE")]
    flagged: list[PlausibilityFinding] = []
    for item in sorted(model.iiv, key=lambda x: x.parameter):
        for val in (item.omega_var, item.cv_pct):
            if val is not None and (not math.isfinite(val) or val < 0.0):
                flagged.append(
                    make_finding(
                        rule, parameter=item.parameter,
                        status="FLAGGED", severity="CRITICAL",
                        observed_value=str(val) if not math.isfinite(val) else val, expected_range=">= 0 (finite)",
                        override_allowed=False,
                    )
                )
                break
    for mat_name, mat in (("omega_matrix", model.omega_matrix), ("sigma_matrix", model.sigma_matrix)):
        if mat is not None:
            for i, row in enumerate(mat):
                if i < len(row):
                    diag = row[i]
                    if not math.isfinite(diag) or diag < 0.0:
                        flagged.append(
                            make_finding(
                                rule, parameter=f"{mat_name}[{i},{i}]",
                                status="FLAGGED", severity="CRITICAL",
                                observed_value=str(diag) if not math.isfinite(diag) else diag,
                                expected_range=">= 0 (finite)", override_allowed=False,
                            )
                        )
    if flagged:
        return flagged
    return [make_finding(rule, parameter="variances", status="PASS", expected_range=">= 0 (finite)")]


def eval_pl014_covariance_psd(
    catalog: RuleCatalog, model: PopPKModelInput
) -> list[PlausibilityFinding]:
    rule = catalog.by_id("PL-014")
    if model.omega_matrix is None and model.sigma_matrix is None:
        return [make_finding(rule, parameter="omega_matrix", status="NOT_APPLICABLE")]
    out: list[PlausibilityFinding] = []
    for mat_name, mat in (("omega_matrix", model.omega_matrix), ("sigma_matrix", model.sigma_matrix)):
        if mat is None:
            continue
        n = len(mat)
        if n == 0 or any(len(row) != n for row in mat):
            out.append(
                make_finding(
                    rule, parameter=mat_name,
                    status="FLAGGED", severity="CRITICAL",
                    rationale=f"{mat_name} is not a non-empty square matrix.", override_allowed=False,
                )
            )
            continue
        arr = np.asarray(mat, dtype=float)
        if not np.all(np.isfinite(arr)):
            out.append(
                make_finding(
                    rule, parameter=mat_name,
                    status="FLAGGED", severity="CRITICAL",
                    rationale=f"{mat_name} contains non-finite entries.", override_allowed=False,
                )
            )
            continue
        max_asym = float(np.max(np.abs(arr - arr.T)))
        if max_asym > 1e-8:
            out.append(
                make_finding(
                    rule, parameter=mat_name,
                    status="FLAGGED", severity="CRITICAL",
                    observed_value=max_asym, expected_range="symmetry |M - M^T| <= 1e-8",
                    rationale=f"{mat_name} is asymmetric (max asymmetry {max_asym:.3e}).", override_allowed=False,
                )
            )
            continue
        eigvals = np.linalg.eigvalsh(0.5 * (arr + arr.T))
        min_eig = float(np.min(eigvals))
        if min_eig < -1e-9 or np.any(np.diag(arr) < 0.0):
            out.append(
                make_finding(
                    rule, parameter=mat_name,
                    status="FLAGGED", severity="CRITICAL",
                    normalized_value=min_eig, expected_range="min_eigenvalue >= 0",
                    rationale=f"{mat_name} is not positive semidefinite (min eigenvalue = {min_eig:.4e}).",
                    override_allowed=False,
                )
            )
        else:
            out.append(
                make_finding(
                    rule, parameter=mat_name,
                    status="PASS", normalized_value=min_eig,
                    expected_range="min_eigenvalue >= 0",
                )
            )
    return out


def eval_pl015_correlations(
    catalog: RuleCatalog, model: PopPKModelInput
) -> list[PlausibilityFinding]:
    rule = catalog.by_id("PL-015")
    corr_items: list[tuple[str, float]] = sorted((k, float(v)) for k, v in model.correlations.items())
    if model.omega_matrix is not None and not model.correlations:
        n = len(model.omega_matrix)
        if n > 1 and all(len(r) == n for r in model.omega_matrix):
            for i in range(n):
                for j in range(i + 1, n):
                    vii = model.omega_matrix[i][i]
                    vjj = model.omega_matrix[j][j]
                    vij = model.omega_matrix[i][j]
                    if all(math.isfinite(x) for x in (vii, vjj, vij)) and vii > 0 and vjj > 0:
                        corr_items.append((f"omega[{i},{j}]", vij / math.sqrt(vii * vjj)))
    if not corr_items:
        return [make_finding(rule, parameter="correlations", status="NOT_APPLICABLE")]
    out: list[PlausibilityFinding] = []
    for key, r_val in corr_items:
        if not math.isfinite(r_val) or abs(r_val) > 1.0 + 1e-9:
            out.append(
                make_finding(
                    rule, parameter=key,
                    status="FLAGGED", severity="CRITICAL",
                    observed_value=str(r_val) if not math.isfinite(r_val) else r_val,
                    normalized_value=r_val if math.isfinite(r_val) else None, expected_range="[-1.0, 1.0]",
                    override_allowed=False,
                )
            )
        elif abs(r_val) >= 0.95:
            out.append(
                make_finding(
                    rule, parameter=key,
                    status="FLAGGED", severity="WARN",
                    observed_value=r_val, normalized_value=r_val,
                    expected_range="(-0.95, 0.95)",
                    rationale=f"Correlation {key} = {r_val:.3f} indicates near-collinear random effects.",
                    override_allowed=True,
                )
            )
        else:
            out.append(
                make_finding(
                    rule, parameter=key,
                    status="PASS", observed_value=r_val,
                    normalized_value=r_val, expected_range="[-1.0, 1.0]",
                )
            )
    return out


def eval_pl016_residual_error(
    catalog: RuleCatalog, model: PopPKModelInput
) -> list[PlausibilityFinding]:
    rule = catalog.by_id("PL-016")
    if not model.residual_error:
        return [make_finding(rule, parameter="residual_error", status="NOT_APPLICABLE")]
    out: list[PlausibilityFinding] = []
    positive_count = 0
    for idx, err in enumerate(model.residual_error):
        label = f"residual_{err.kind}_{idx}" if len(model.residual_error) > 1 else f"residual_{err.kind}"
        raw = err.sigma_var if err.sigma_var is not None else err.sd_or_cv
        if raw is None:
            out.append(make_finding(rule, parameter=label, status="UNRESOLVED", severity="WARN"))
            continue
        if not math.isfinite(raw) or raw < 0.0:
            out.append(
                make_finding(
                    rule, parameter=label,
                    status="FLAGGED", severity="CRITICAL",
                    observed_value=str(raw) if not math.isfinite(raw) else raw, expected_range="> 0 (finite)",
                    override_allowed=False,
                )
            )
            continue
        if raw > 0.0:
            positive_count += 1
        if err.kind in {"proportional", "exponential"}:
            is_pct = bool(err.unit and err.unit.strip() == "%")
            if err.sd_or_cv is not None:
                cv_pct = err.sd_or_cv if is_pct else err.sd_or_cv * 100.0
            else:
                cv_pct = math.sqrt(raw) if is_pct else math.sqrt(raw) * 100.0
            if cv_pct > 50.0:
                sev: SeverityLevel = "HIGH" if cv_pct > 100.0 else "WARN"
                out.append(
                    make_finding(
                        rule, parameter=label,
                        status="FLAGGED", severity=sev,
                        observed_value=raw, normalized_value=cv_pct,
                        normalized_unit="%CV", expected_range="<= 50% CV",
                        override_allowed=True,
                    )
                )
    if out:
        return out
    if positive_count == 0:
        return [
            make_finding(
                rule, parameter="residual_error",
                status="FLAGGED", severity="HIGH",
                observed_value=0.0, expected_range="> 0 across active error terms",
                rationale="All residual error terms are identically zero.", override_allowed=False,
            )
        ]
    return [make_finding(rule, parameter="residual_error", status="PASS", expected_range="finite, > 0")]


def eval_pl020_predictions(
    catalog: RuleCatalog, model: PopPKModelInput
) -> list[PlausibilityFinding]:
    rule = catalog.by_id("PL-020")
    if not model.predictions:
        return [make_finding(rule, parameter="predictions", status="NOT_APPLICABLE")]
    for idx, pt in enumerate(model.predictions):
        if not math.isfinite(pt.time):
            return [
                make_finding(
                    rule, parameter=f"predictions[{idx}].time",
                    status="FLAGGED", severity="CRITICAL",
                    observed_value=str(pt.time), expected_range="finite time",
                    override_allowed=False,
                )
            ]
        for field_name, val in (("pred", pt.pred), ("ipred", pt.ipred)):
            if val is None:
                continue
            if not math.isfinite(val) or val < 0.0:
                return [
                    make_finding(
                        rule, parameter=f"predictions[{idx}].{field_name}",
                        status="FLAGGED", severity="CRITICAL",
                        observed_value=str(val) if not math.isfinite(val) else val, observed_unit=pt.unit,
                        expected_range=">= 0 (finite)", override_allowed=False,
                    )
                ]
            if val > 1_000_000.0:
                st: FindingStatus = "FLAGGED"
                return [
                    make_finding(
                        rule, parameter=f"predictions[{idx}].{field_name}",
                        status=st, severity="HIGH",
                        observed_value=val, observed_unit=pt.unit,
                        expected_range="<= 1e6 mg/L", override_allowed=True,
                    )
                ]
    return [make_finding(rule, parameter="predictions", status="PASS", expected_range=">= 0 (finite)")]
