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


def eval_pl008_half_life(
    catalog: RuleCatalog, params: tuple[NormalizedParameter, ...], phys: ResolvedPhysiologyContext
) -> list[PlausibilityFinding]:
    rule = catalog.by_id("PL-008")
    thalf = by_canonical(params, "t_half")
    cl = by_canonical(params, "CL") or by_canonical(params, "CL/F")
    vc = by_canonical(params, "Vc") or by_canonical(params, "Vc/F")
    cl_l_h = to_abs_flow(cl, phys) if cl else None
    vc_l = to_abs_vol_l(vc, phys)
    same_scale = bool(cl and vc and cl.resolved_scale == vc.resolved_scale)
    calc_thalf: float | None = (
        (math.log(2.0) * vc_l / cl_l_h) if (cl_l_h and vc_l and same_scale) else None
    )
    if thalf is not None:
        if (
            thalf.unit_status != "valid"
            or thalf.normalized_value is None
            or thalf.normalized_value <= 0
            or not math.isfinite(thalf.normalized_value)
        ):
            return [make_finding(rule, parameter=thalf.name, status="UNRESOLVED", severity="WARN")]
        obs_h = thalf.normalized_value
        if calc_thalf is not None and abs(obs_h - calc_thalf) / max(calc_thalf, 1e-12) > 0.10:
            return [
                make_finding(
                    rule, parameter=thalf.name,
                    status="FLAGGED", severity="HIGH",
                    observed_value=thalf.observed_value, observed_unit=thalf.observed_unit,
                    normalized_value=obs_h, normalized_unit="h",
                    expected_range=f"{calc_thalf:.3f} h (+/-10%)",
                )
            ]
        st: FindingStatus = "FLAGGED" if (obs_h < 0.01 or obs_h > 5000.0) else "PASS"
        sev: SeverityLevel = "WARN" if st == "FLAGGED" else "INFO"
        return [
            make_finding(
                rule, parameter=thalf.name,
                status=st, severity=sev,
                observed_value=thalf.observed_value, observed_unit=thalf.observed_unit,
                normalized_value=obs_h, normalized_unit="h",
                expected_range="[0.01, 5000] h",
            )
        ]
    if calc_thalf is None:
        return [make_finding(rule, parameter="t_half", status="NOT_APPLICABLE")]
    st = "FLAGGED" if (calc_thalf < 0.01 or calc_thalf > 5000.0) else "PASS"
    sev = "WARN" if st == "FLAGGED" else "INFO"
    return [
        make_finding(
            rule, parameter="t_half (derived)",
            status=st, severity=sev,
            normalized_value=calc_thalf, normalized_unit="h",
            expected_range="[0.01, 5000] h",
        )
    ]


def eval_pl009_ka_flip_flop(
    catalog: RuleCatalog, params: tuple[NormalizedParameter, ...], phys: ResolvedPhysiologyContext
) -> list[PlausibilityFinding]:
    rule = catalog.by_id("PL-009")
    ka = by_canonical(params, "ka")
    if ka is None:
        return [make_finding(rule, parameter="ka", status="NOT_APPLICABLE")]
    if (
        ka.unit_status != "valid"
        or ka.normalized_value is None
        or ka.normalized_value <= 0
        or not math.isfinite(ka.normalized_value)
    ):
        return [make_finding(rule, parameter=ka.name, status="UNRESOLVED", severity="WARN")]
    ka_val = ka.normalized_value
    cl = by_canonical(params, "CL") or by_canonical(params, "CL/F")
    vc = by_canonical(params, "Vc") or by_canonical(params, "Vc/F")
    cl_l_h = to_abs_flow(cl, phys) if cl else None
    vc_l = to_abs_vol_l(vc, phys)
    same_scale = bool(cl and vc and cl.resolved_scale == vc.resolved_scale)
    ke = (cl_l_h / vc_l) if (cl_l_h and vc_l and same_scale) else None
    if ka_val < 0.01 or ka_val > 20.0:
        return [
            make_finding(
                rule, parameter=ka.name,
                status="FLAGGED", severity="HIGH",
                observed_value=ka.observed_value, observed_unit=ka.observed_unit,
                normalized_value=ka_val, normalized_unit="1/h",
                expected_range="[0.05, 10.0] 1/h",
            )
        ]
    if ke is not None and ka_val <= ke:
        return [
            make_finding(
                rule, parameter=ka.name,
                status="FLAGGED", severity="WARN",
                observed_value=ka.observed_value, observed_unit=ka.observed_unit,
                normalized_value=ka_val, normalized_unit="1/h",
                expected_range=f"> ke ({ke:.3f} 1/h)",
                rationale=(
                    f"Potential flip-flop kinetics detected: ka ({ka_val:.3f} 1/h) <= "
                    f"elimination rate constant ke ({ke:.3f} 1/h)."
                ),
            )
        ]
    st: FindingStatus = "FLAGGED" if (ka_val < 0.05 or ka_val > 10.0) else "PASS"
    sev: SeverityLevel = "WARN" if st == "FLAGGED" else "INFO"
    return [
        make_finding(
            rule, parameter=ka.name,
            status=st, severity=sev,
            observed_value=ka.observed_value, observed_unit=ka.observed_unit,
            normalized_value=ka_val, normalized_unit="1/h",
            expected_range="[0.05, 10.0] 1/h",
        )
    ]


def eval_pl010_and_pl011_allometry(
    catalog: RuleCatalog, model: PopPKModelInput
) -> list[PlausibilityFinding]:
    r_cl = catalog.by_id("PL-010")
    r_v = catalog.by_id("PL-011")
    cl_exps: list[tuple[str, float]] = []
    v_exps: list[tuple[str, float]] = []
    for k, val in sorted(model.allometric_exponents.items()):
        _, pclass, _ = classify_parameter(k)
        if pclass == "clearance":
            cl_exps.append((k, float(val)))
        elif pclass == "volume":
            v_exps.append((k, float(val)))
    findings: list[PlausibilityFinding] = []
    for rule, items, default_p, p_min, p_max, w_min, w_max in (
        (r_cl, cl_exps, "CL", 0.60, 0.90, 0.30, 1.20), (r_v, v_exps, "Vc", 0.80, 1.20, 0.50, 1.50),
    ):
        if not items:
            findings.append(make_finding(rule, parameter=default_p, status="NOT_APPLICABLE"))
            continue
        rng = f"[{p_min:.2f}, {p_max:.2f}]"
        for k, v in items:
            if not math.isfinite(v):
                findings.append(
                    make_finding(
                        rule, parameter=k,
                        status="FLAGGED", severity="CRITICAL",
                        observed_value=str(v), override_allowed=False,
                    )
                )
            elif p_min <= v <= p_max:
                findings.append(
                    make_finding(
                        rule, parameter=k,
                        status="PASS", observed_value=v,
                        normalized_value=v, expected_range=rng,
                    )
                )
            elif w_min <= v <= w_max:
                findings.append(
                    make_finding(
                        rule, parameter=k,
                        status="FLAGGED", severity="WARN",
                        observed_value=v, normalized_value=v,
                        expected_range=rng,
                    )
                )
            else:
                findings.append(
                    make_finding(
                        rule, parameter=k,
                        status="FLAGGED", severity="HIGH",
                        observed_value=v, normalized_value=v,
                        expected_range=rng,
                    )
                )
    return findings


def eval_pl017_and_pl018_mab(
    catalog: RuleCatalog, model: PopPKModelInput,
    params: tuple[NormalizedParameter, ...], phys: ResolvedPhysiologyContext,
) -> list[PlausibilityFinding]:
    r17 = catalog.by_id("PL-017")
    r18 = catalog.by_id("PL-018")
    if model.context.modality == "unknown":
        return [
            make_finding(
                r17, parameter="Vc",
                status="UNRESOLVED", severity="WARN",
                rationale="Modality is 'unknown'; cannot determine whether mAb central volume bounds apply.",
            ),
            make_finding(
                r18, parameter="CL",
                status="UNRESOLVED", severity="WARN",
                rationale="Modality is 'unknown'; cannot determine whether mAb linear clearance bounds apply.",
            ),
        ]
    if model.context.modality != "monoclonal_antibody":
        return [
            make_finding(r17, parameter="Vc", status="NOT_APPLICABLE"),
            make_finding(r18, parameter="CL", status="NOT_APPLICABLE"),
        ]
    out: list[PlausibilityFinding] = []
    vc = by_canonical(params, "Vc") or by_canonical(params, "Vc/F")
    vc_l_kg = to_l_kg(vc, phys) if vc else None
    if vc is None or vc_l_kg is None:
        out.append(make_finding(r17, parameter=vc.name if vc else "Vc", status="UNRESOLVED", severity="WARN"))
    else:
        if 0.03 <= vc_l_kg <= 0.10:
            st17, sev17 = "PASS", "INFO"
        elif 0.10 < vc_l_kg <= 0.15:
            st17, sev17 = "FLAGGED", "WARN"
        else:
            st17, sev17 = "FLAGGED", "HIGH"
        out.append(
            make_finding(
                r17, parameter=vc.name,
                status=st17, severity=sev17,
                observed_value=vc.observed_value, observed_unit=vc.observed_unit,
                normalized_value=vc_l_kg, normalized_unit="L/kg",
                expected_range="[0.03, 0.10] L/kg",
            )
        )

    cl = by_canonical(params, "CL") or by_canonical(params, "CL/F")
    cl_l_h = to_abs_flow(cl, phys) if cl else None
    if cl is None or cl_l_h is None or phys.flow_scale_factor is None:
        out.append(make_finding(r18, parameter=cl.name if cl else "CL", status="UNRESOLVED", severity="WARN"))
    else:
        cl_70 = cl_l_h / phys.flow_scale_factor
        if 0.003 <= cl_70 <= 0.05:
            st18, sev18 = "PASS", "INFO"
        elif 0.05 < cl_70 <= 0.15 or (cl_70 > 0.05 and model.context.has_tmdd_or_nonlinear_cl):
            st18, sev18 = "FLAGGED", "WARN"
        else:
            st18, sev18 = "FLAGGED", "HIGH"
        out.append(
            make_finding(
                r18, parameter=cl.name,
                status=st18, severity=sev18,
                observed_value=cl.observed_value, observed_unit=cl.observed_unit,
                normalized_value=cl_70, normalized_unit="L/h (70kg eq)",
                expected_range="[0.003, 0.05] L/h",
            )
        )
    return out


def eval_pl019_micro_rates(
    catalog: RuleCatalog, params: tuple[NormalizedParameter, ...], phys: ResolvedPhysiologyContext
) -> list[PlausibilityFinding]:
    rule = catalog.by_id("PL-019")
    q = by_canonical(params, "Q")
    vc = by_canonical(params, "Vc")
    vp = by_canonical(params, "Vp")
    cl = by_canonical(params, "CL")
    k12 = by_canonical(params, "k12")
    k21 = by_canonical(params, "k21")
    k10 = by_canonical(params, "k10") or by_canonical(params, "ke")
    if q is None and k12 is None and k21 is None and k10 is None:
        return [make_finding(rule, parameter="Q", status="NOT_APPLICABLE")]
    out: list[PlausibilityFinding] = []
    q_l_h = to_abs_flow(q, phys) if q else None
    vc_l = to_abs_vol_l(vc, phys)
    vp_l = to_abs_vol_l(vp, phys)
    cl_l_h = to_abs_flow(cl, phys) if cl else None

    for k_p, num, den, label in (
        (k12, q_l_h, vc_l, "Q/Vc"), (k21, q_l_h, vp_l, "Q/Vp"),
        (k10, cl_l_h, vc_l, "CL/Vc"),
    ):
        if k_p is not None and k_p.normalized_value is not None and num is not None and den is not None:
            exp_k = num / den
            if abs(k_p.normalized_value - exp_k) / max(exp_k, 1e-12) > 0.02:
                out.append(
                    make_finding(
                        rule, parameter=k_p.name,
                        status="FLAGGED", severity="HIGH",
                        observed_value=k_p.observed_value, observed_unit=k_p.observed_unit,
                        normalized_value=k_p.normalized_value, normalized_unit="1/h",
                        expected_range=f"{label} = {exp_k:.4f} 1/h (+/-2%)", override_allowed=False,
                    )
                )
    if q_l_h is not None and phys.cardiac_output_l_h is not None and q_l_h > phys.cardiac_output_l_h:
        out.append(
            make_finding(
                rule, parameter=q.name if q else "Q",
                status="FLAGGED", severity="HIGH",
                normalized_value=q_l_h, normalized_unit="L/h",
                expected_range=f"<= {phys.cardiac_output_l_h:.1f} L/h", override_allowed=True,
            )
        )
    if out:
        return out
    return [
        make_finding(
            rule, parameter=q.name if q else "micro_rates",
            status="PASS", normalized_value=q_l_h,
            normalized_unit="L/h" if q_l_h else None,
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
    findings.extend(eval_pl009_ka_flip_flop(catalog, normalized_params, phys))
    findings.extend(eval_pl010_and_pl011_allometry(catalog, model))
    findings.extend(eval_pl012_iiv_magnitude(catalog, model))
    findings.extend(eval_pl013_variance_nonnegative(catalog, model))
    findings.extend(eval_pl014_covariance_psd(catalog, model))
    findings.extend(eval_pl015_correlations(catalog, model))
    findings.extend(eval_pl016_residual_error(catalog, model))
    findings.extend(eval_pl017_and_pl018_mab(catalog, model, normalized_params, phys))
    findings.extend(eval_pl019_micro_rates(catalog, normalized_params, phys))
    findings.extend(eval_pl020_predictions(catalog, model))
    return tuple(sorted(findings, key=lambda f: (f.rule_id, f.parameter)))
