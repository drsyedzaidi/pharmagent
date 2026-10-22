"""Derived kinetics, allometry, mAb, micro-rate rules (PL-008..011, PL-017..019) and dispatcher."""
from __future__ import annotations

import math

from app.compute.poppk_plausibility.context import ResolvedPhysiologyContext
from app.compute.poppk_plausibility.rules_stochastic import (
    eval_pl012_iiv_magnitude,
    eval_pl013_variance_nonnegative,
    eval_pl014_covariance_psd,
    eval_pl015_correlations,
    eval_pl016_residual_error,
    eval_pl020_predictions,
)
from app.compute.poppk_plausibility.rules_structural import (
    by_canonical,
    eval_pl001_positive_finite,
    eval_pl002_units,
    eval_pl003_bioavailability,
    eval_pl004_systemic_cl,
    eval_pl005_renal_cl,
    eval_pl006_central_volume,
    eval_pl007_vss_consistency,
    make_finding,
    resolved_f_fraction,
    to_abs_flow,
    to_abs_vol_l,
    to_l_kg,
)
from app.compute.poppk_plausibility.schema import (
    FindingStatus,
    NormalizedParameter,
    PlausibilityFinding,
    PopPKModelInput,
    RuleCatalog,
    SeverityLevel,
)
from app.compute.poppk_plausibility.units import classify_parameter

__all__ = ["evaluate_all_rules", "make_finding"]


def _resolve_elimination_kinetics(
    params: tuple[NormalizedParameter, ...], phys: ResolvedPhysiologyContext
) -> tuple[float | None, float | None, float | None, bool]:
    """Return (elim_rate_per_h, thalf_terminal_h, thalf_eff_h, has_unresolvable_pair)."""
    cl = by_canonical(params, "CL") or by_canonical(params, "CL/F")
    vc = by_canonical(params, "Vc") or by_canonical(params, "Vc/F")
    if cl is not None and vc is not None:
        cl_l_h, vc_l = to_abs_flow(cl, phys), to_abs_vol_l(vc, phys)
        if cl_l_h is None or vc_l is None or cl.resolved_scale != vc.resolved_scale:
            return None, None, None, True
        k10 = cl_l_h / vc_l
        vp, q = by_canonical(params, "Vp"), by_canonical(params, "Q")
        vp_l = to_abs_vol_l(vp, phys) if vp else None
        q_l_h = to_abs_flow(q, phys) if q else None
        if (
            vp and q and vp_l is not None and q_l_h is not None
            and vp.resolved_scale == vc.resolved_scale and q.resolved_scale == cl.resolved_scale
        ):
            k12, k21 = q_l_h / vc_l, q_l_h / vp_l
            s = k10 + k12 + k21
            beta = 0.5 * (s - math.sqrt(max(0.0, s * s - 4.0 * k10 * k21)))
            if beta > 0.0:
                return beta, math.log(2.0) / beta, math.log(2.0) * (vc_l + vp_l) / cl_l_h, False
        th1 = math.log(2.0) / k10
        return k10, th1, th1, False

    ke_p = by_canonical(params, "ke") or by_canonical(params, "k10")
    if ke_p and ke_p.unit_status == "compatible" and ke_p.normalized_value is not None:
        if math.isfinite(ke_p.normalized_value) and ke_p.normalized_value > 0.0:
            th = math.log(2.0) / ke_p.normalized_value
            return ke_p.normalized_value, th, th, False
    return None, None, None, False


def eval_pl008_half_life(
    catalog: RuleCatalog, params: tuple[NormalizedParameter, ...], phys: ResolvedPhysiologyContext
) -> list[PlausibilityFinding]:
    rule = catalog.by_id("PL-008")
    min_h = rule.threshold_float("warn_min_hours", 0.01)
    max_h = rule.threshold_float("warn_max_hours", 5000.0)
    tol = rule.threshold_float("consistency_rel_tol", 0.10)
    thalf = by_canonical(params, "t_half")
    _, thalf_term, thalf_eff, unresolvable_pair = _resolve_elimination_kinetics(params, phys)

    if thalf is not None:
        if (
            thalf.unit_status != "compatible" or thalf.normalized_value is None
            or thalf.normalized_value <= 0 or not math.isfinite(thalf.normalized_value) or unresolvable_pair
        ):
            return [make_finding(rule, parameter=thalf.name, status="UNRESOLVED", severity="WARN")]
        obs_h = thalf.normalized_value
        if thalf_term is not None:
            err_term = abs(obs_h - thalf_term) / max(thalf_term, 1e-12)
            err_eff = abs(obs_h - thalf_eff) / max(thalf_eff, 1e-12) if thalf_eff is not None else err_term
            if min(err_term, err_eff) > tol:
                return [
                    make_finding(
                        rule, parameter=thalf.name, status="FLAGGED", severity="HIGH",
                        observed_value=thalf.observed_value, observed_unit=thalf.observed_unit,
                        normalized_value=obs_h, normalized_unit="h",
                        expected_range=f"{thalf_term:.3f} h (+/-{int(tol * 100)}%)", override_allowed=False,
                    )
                ]
        st: FindingStatus = "FLAGGED" if (obs_h < min_h or obs_h > max_h) else "PASS"
        sev: SeverityLevel = "WARN" if st == "FLAGGED" else "INFO"
        return [
            make_finding(
                rule, parameter=thalf.name, status=st, severity=sev,
                observed_value=thalf.observed_value, observed_unit=thalf.observed_unit,
                normalized_value=obs_h, normalized_unit="h", expected_range=f"[{min_h:g}, {max_h:g}] h",
            )
        ]
    if unresolvable_pair:
        return [make_finding(rule, parameter="t_half (derived)", status="UNRESOLVED", severity="WARN")]
    if thalf_term is None:
        return [make_finding(rule, parameter="t_half", status="NOT_APPLICABLE")]
    st = "FLAGGED" if (thalf_term < min_h or thalf_term > max_h) else "PASS"
    sev = "WARN" if st == "FLAGGED" else "INFO"
    return [
        make_finding(
            rule, parameter="t_half (derived)", status=st, severity=sev,
            normalized_value=thalf_term, normalized_unit="h", expected_range=f"[{min_h:g}, {max_h:g}] h",
        )
    ]


def eval_pl009_ka_flip_flop(
    catalog: RuleCatalog, model: PopPKModelInput,
    params: tuple[NormalizedParameter, ...], phys: ResolvedPhysiologyContext,
) -> list[PlausibilityFinding]:
    rule = catalog.by_id("PL-009")
    ka = by_canonical(params, "ka")
    if ka is None:
        return [make_finding(rule, parameter="ka", status="NOT_APPLICABLE")]
    elim_rate, _, _, unresolvable_pair = _resolve_elimination_kinetics(params, phys)
    if (
        ka.unit_status != "compatible" or ka.normalized_value is None
        or ka.normalized_value <= 0 or not math.isfinite(ka.normalized_value) or unresolvable_pair
    ):
        return [make_finding(rule, parameter=ka.name, status="UNRESOLVED", severity="WARN")]
    ka_val = ka.normalized_value
    is_mab = model.context.modality == "monoclonal_antibody"
    w_min = rule.threshold_float("mab_warn_min_ka_per_h" if is_mab else "warn_min_ka_per_h", 0.05)
    h_min = rule.threshold_float("mab_high_min_ka_per_h" if is_mab else "high_min_ka_per_h", 0.01)
    w_max = rule.threshold_float("warn_max_ka_per_h", 10.0)
    h_max = rule.threshold_float("high_max_ka_per_h", 20.0)
    rng_str = f"[{w_min:g}, {w_max:.1f}] 1/h"
    if ka_val < h_min or ka_val > h_max:
        return [
            make_finding(
                rule, parameter=ka.name, status="FLAGGED", severity="HIGH",
                observed_value=ka.observed_value, observed_unit=ka.observed_unit,
                normalized_value=ka_val, normalized_unit="1/h", expected_range=rng_str,
            )
        ]
    if elim_rate is not None and ka_val <= elim_rate:
        return [
            make_finding(
                rule, parameter=ka.name, status="FLAGGED", severity="WARN",
                observed_value=ka.observed_value, observed_unit=ka.observed_unit,
                normalized_value=ka_val, normalized_unit="1/h",
                expected_range=f"> ke ({elim_rate:.3f} 1/h)",
                rationale=(
                    f"Potential flip-flop kinetics detected: ka ({ka_val:.3f} 1/h) <= "
                    f"elimination rate constant ({elim_rate:.3f} 1/h)."
                ),
            )
        ]
    st: FindingStatus = "FLAGGED" if (ka_val < w_min or ka_val > w_max) else "PASS"
    sev: SeverityLevel = "WARN" if st == "FLAGGED" else "INFO"
    return [
        make_finding(
            rule, parameter=ka.name, status=st, severity=sev,
            observed_value=ka.observed_value, observed_unit=ka.observed_unit,
            normalized_value=ka_val, normalized_unit="1/h", expected_range=rng_str,
        )
    ]


def eval_pl010_and_pl011_allometry(
    catalog: RuleCatalog, model: PopPKModelInput
) -> list[PlausibilityFinding]:
    r_cl, r_v = catalog.by_id("PL-010"), catalog.by_id("PL-011")
    cl_exps: list[tuple[str, float]] = []
    v_exps: list[tuple[str, float]] = []
    findings: list[PlausibilityFinding] = []
    for k, val in sorted(model.allometric_exponents.items()):
        _, pclass, _ = classify_parameter(k)
        if pclass == "clearance":
            cl_exps.append((k, float(val)))
        elif pclass == "volume":
            v_exps.append((k, float(val)))
        else:
            findings.append(
                make_finding(
                    r_cl, parameter=k, status="UNRESOLVED", severity="WARN",
                    observed_value=str(val) if not math.isfinite(val) else float(val),
                    rationale=f"Unrecognized parameter '{k}' in allometric_exponents.",
                )
            )
    for rule, items, default_p, p_min, p_max, w_min, w_max in (
        (r_cl, cl_exps, "CL", 0.60, 0.90, 0.30, 1.20),
        (r_v, v_exps, "Vc", 0.80, 1.20, 0.50, 1.50),
    ):
        if not items:
            findings.append(make_finding(rule, parameter=default_p, status="NOT_APPLICABLE"))
            continue
        rng = f"[{p_min:.2f}, {p_max:.2f}]"
        for k, v in items:
            if not math.isfinite(v):
                findings.append(
                    make_finding(
                        rule, parameter=k, status="FLAGGED", severity="CRITICAL",
                        observed_value=str(v), override_allowed=False,
                    )
                )
            else:
                st: FindingStatus = "PASS" if p_min <= v <= p_max else "FLAGGED"
                sev: SeverityLevel = "INFO" if st == "PASS" else ("WARN" if w_min <= v <= w_max else "HIGH")
                findings.append(
                    make_finding(
                        rule, parameter=k, status=st, severity=sev,
                        observed_value=v, normalized_value=v, expected_range=rng,
                    )
                )
    return findings


def eval_pl017_and_pl018_mab(
    catalog: RuleCatalog, model: PopPKModelInput,
    params: tuple[NormalizedParameter, ...], phys: ResolvedPhysiologyContext,
) -> list[PlausibilityFinding]:
    r17, r18 = catalog.by_id("PL-017"), catalog.by_id("PL-018")
    if model.context.modality == "unknown":
        return [
            make_finding(r17, parameter="Vc", status="UNRESOLVED", severity="WARN"),
            make_finding(r18, parameter="CL", status="UNRESOLVED", severity="WARN"),
        ]
    if model.context.modality != "monoclonal_antibody":
        return [
            make_finding(r17, parameter="Vc", status="NOT_APPLICABLE"),
            make_finding(r18, parameter="CL", status="NOT_APPLICABLE"),
        ]
    out: list[PlausibilityFinding] = []
    f_val = resolved_f_fraction(params)
    vc = by_canonical(params, "Vc") or by_canonical(params, "Vc/F")
    vc_l_kg = to_l_kg(vc, phys) if vc else None
    if vc is not None and vc.resolved_scale == "apparent":
        vc_l_kg = (vc_l_kg * f_val) if (vc_l_kg is not None and f_val is not None) else None
    if vc is None or vc_l_kg is None:
        out.append(make_finding(r17, parameter=vc.name if vc else "Vc", status="UNRESOLVED", severity="WARN"))
    else:
        st17: FindingStatus = "PASS" if 0.03 <= vc_l_kg <= 0.10 else "FLAGGED"
        sev17: SeverityLevel = "INFO" if st17 == "PASS" else ("WARN" if 0.10 < vc_l_kg <= 0.15 else "HIGH")
        out.append(
            make_finding(
                r17, parameter=vc.name, status=st17, severity=sev17,
                observed_value=vc.observed_value, observed_unit=vc.observed_unit,
                normalized_value=vc_l_kg, normalized_unit="L/kg", expected_range="[0.03, 0.10] L/kg",
            )
        )
    cl = by_canonical(params, "CL") or by_canonical(params, "CL/F")
    cl_l_h = to_abs_flow(cl, phys) if cl else None
    if cl is not None and cl.resolved_scale == "apparent":
        cl_l_h = (cl_l_h * f_val) if (cl_l_h is not None and f_val is not None) else None
    if cl is None or cl_l_h is None or phys.flow_scale_factor is None:
        out.append(make_finding(r18, parameter=cl.name if cl else "CL", status="UNRESOLVED", severity="WARN"))
    else:
        cl_70 = cl_l_h / phys.flow_scale_factor
        tmdd_ok = model.context.has_tmdd_or_nonlinear_cl and cl_70 <= 1.0
        st18: FindingStatus = "PASS" if 0.003 <= cl_70 <= 0.05 else "FLAGGED"
        sev18: SeverityLevel = (
            "INFO" if st18 == "PASS"
            else ("WARN" if (0.05 < cl_70 <= 0.15 or (cl_70 > 0.05 and tmdd_ok)) else "HIGH")
        )
        out.append(
            make_finding(
                r18, parameter=cl.name, status=st18, severity=sev18,
                observed_value=cl.observed_value, observed_unit=cl.observed_unit,
                normalized_value=cl_70, normalized_unit="L/h (70kg eq)", expected_range="[0.003, 0.05] L/h",
            )
        )
    return out


def eval_pl019_micro_rates(
    catalog: RuleCatalog, params: tuple[NormalizedParameter, ...], phys: ResolvedPhysiologyContext
) -> list[PlausibilityFinding]:
    rule = catalog.by_id("PL-019")
    q, q2 = by_canonical(params, "Q"), by_canonical(params, "Q2")
    vc, vp, vp2 = by_canonical(params, "Vc"), by_canonical(params, "Vp"), by_canonical(params, "Vp2")
    cl = by_canonical(params, "CL")
    k12, k21 = by_canonical(params, "k12"), by_canonical(params, "k21")
    k13, k31 = by_canonical(params, "k13"), by_canonical(params, "k31")
    k10 = by_canonical(params, "k10") or by_canonical(params, "ke")
    active = [p for p in (q, q2, k12, k21, k13, k31, k10) if p is not None]
    if not active:
        return [make_finding(rule, parameter="Q", status="NOT_APPLICABLE")]
    for p in active:
        if (
            p.unit_status != "compatible" or p.normalized_value is None
            or not math.isfinite(p.normalized_value) or p.normalized_value <= 0
        ):
            return [make_finding(rule, parameter=p.name, status="UNRESOLVED", severity="WARN")]
    q_l_h = to_abs_flow(q, phys) if q else None
    q2_l_h = to_abs_flow(q2, phys) if q2 else None
    if (q is not None and (q_l_h is None or phys.cardiac_output_l_h is None)) or (
        q2 is not None and (q2_l_h is None or phys.cardiac_output_l_h is None)
    ):
        return [make_finding(rule, parameter=(q or q2).name, status="UNRESOLVED", severity="WARN")]
    vc_l, vp_l, vp2_l = to_abs_vol_l(vc, phys), to_abs_vol_l(vp, phys), to_abs_vol_l(vp2, phys)
    cl_l_h = to_abs_flow(cl, phys) if cl else None
    out: list[PlausibilityFinding] = []
    for k_p, num, den, label in (
        (k12, q_l_h, vc_l, "Q/Vc"), (k21, q_l_h, vp_l, "Q/Vp"),
        (k13, q2_l_h, vc_l, "Q2/Vc"), (k31, q2_l_h, vp2_l, "Q2/Vp2"),
        (k10, cl_l_h, vc_l, "CL/Vc"),
    ):
        if k_p is not None and k_p.normalized_value is not None and num is not None and den is not None:
            exp_k = num / den
            if abs(k_p.normalized_value - exp_k) / max(exp_k, 1e-12) > 0.02:
                out.append(
                    make_finding(
                        rule, parameter=k_p.name, status="FLAGGED", severity="HIGH",
                        observed_value=k_p.observed_value, observed_unit=k_p.observed_unit,
                        normalized_value=k_p.normalized_value, normalized_unit="1/h",
                        expected_range=f"{label} = {exp_k:.4f} 1/h (+/-2%)", override_allowed=False,
                    )
                )
    for qp, qlh in ((q, q_l_h), (q2, q2_l_h)):
        if qp is not None and qlh is not None and phys.cardiac_output_l_h is not None and qlh > phys.cardiac_output_l_h:
            out.append(
                make_finding(
                    rule, parameter=qp.name, status="FLAGGED", severity="HIGH",
                    normalized_value=qlh, normalized_unit="L/h",
                    expected_range=f"<= {phys.cardiac_output_l_h:.1f} L/h", override_allowed=True,
                )
            )
    return out or [
        make_finding(
            rule, parameter=q.name if q else "micro_rates",
            status="PASS", normalized_value=q_l_h, normalized_unit="L/h" if q_l_h else None,
        )
    ]


def evaluate_all_rules(
    catalog: RuleCatalog, model: PopPKModelInput,
    normalized_params: tuple[NormalizedParameter, ...], phys: ResolvedPhysiologyContext,
) -> tuple[PlausibilityFinding, ...]:
    """Run all 20 deterministic plausibility rules and return deterministically sorted findings."""
    findings: list[PlausibilityFinding] = []
    findings.extend(eval_pl001_positive_finite(catalog, normalized_params))
    findings.extend(eval_pl002_units(catalog, normalized_params))
    findings.extend(eval_pl003_bioavailability(catalog, normalized_params))
    findings.extend(eval_pl004_systemic_cl(catalog, normalized_params, phys))
    findings.extend(eval_pl005_renal_cl(catalog, model, normalized_params, phys))
    findings.extend(eval_pl006_central_volume(catalog, normalized_params, phys))
    findings.extend(eval_pl007_vss_consistency(catalog, normalized_params, phys))
    findings.extend(eval_pl008_half_life(catalog, normalized_params, phys))
    findings.extend(eval_pl009_ka_flip_flop(catalog, model, normalized_params, phys))
    findings.extend(eval_pl010_and_pl011_allometry(catalog, model))
    findings.extend(eval_pl012_iiv_magnitude(catalog, model))
    findings.extend(eval_pl013_variance_nonnegative(catalog, model))
    findings.extend(eval_pl014_covariance_psd(catalog, model))
    findings.extend(eval_pl015_correlations(catalog, model))
    findings.extend(eval_pl016_residual_error(catalog, model))
    findings.extend(eval_pl017_and_pl018_mab(catalog, model, normalized_params, phys))
    findings.extend(eval_pl019_micro_rates(catalog, normalized_params, phys))
    findings.extend(eval_pl020_predictions(catalog, model))
    return tuple(sorted(findings, key=lambda f: (f.rule_id, f.parameter, str(f.observed_value))))
