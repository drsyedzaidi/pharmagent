"""Deterministic domain, unit, bioavailability, clearance, and volume plausibility rules (PL-001..PL-007)."""
from __future__ import annotations

import math

from app.compute.poppk_plausibility.context import ResolvedPhysiologyContext
from app.compute.poppk_plausibility.schema import (
    FindingStatus,
    NormalizedParameter,
    PlausibilityFinding,
    PopPKModelInput,
    RuleCatalog,
    RuleDefinition,
    SeverityLevel,
)


def make_finding(
    rule: RuleDefinition,
    *,
    parameter: str,
    status: FindingStatus,
    severity: SeverityLevel | None = None,
    observed_value: float | str | None = None,
    observed_unit: str | None = None,
    normalized_value: float | None = None,
    normalized_unit: str | None = None,
    expected_range: str = "",
    rationale: str | None = None,
    recommendation: str | None = None,
    override_allowed: bool | None = None,
) -> PlausibilityFinding:
    eff_sev: SeverityLevel = (
        severity if severity is not None else ("INFO" if status != "FLAGGED" else rule.default_severity)
    )
    eff_override = rule.override_allowed if override_allowed is None else override_allowed
    return PlausibilityFinding(
        rule_id=rule.rule_id, rule_version=rule.version, category=rule.category,
        rule_type=rule.rule_type, approval_status=rule.approval_status,
        parameter=parameter, observed_value=observed_value, observed_unit=observed_unit,
        normalized_value=normalized_value, normalized_unit=normalized_unit,
        expected_range=expected_range, severity=eff_sev, status=status,
        scientific_rationale=rationale or rule.scientific_rationale,
        recommendation=recommendation or rule.recommendation,
        reference_basis=rule.reference_basis, override_allowed=eff_override,
    )


def by_canonical(params: tuple[NormalizedParameter, ...], name: str) -> NormalizedParameter | None:
    return next((p for p in params if p.canonical_name == name), None)


def resolved_f_fraction(params: tuple[NormalizedParameter, ...]) -> float | None:
    f_p = by_canonical(params, "F")
    if f_p and f_p.unit_status == "compatible" and f_p.normalized_value is not None:
        if math.isfinite(f_p.normalized_value) and 0.0 < f_p.normalized_value <= 1.0:
            return f_p.normalized_value
    return None


def _valid_pos_norm(p: NormalizedParameter | None) -> float | None:
    if p is None or p.unit_status != "compatible" or p.normalized_value is None:
        return None
    return p.normalized_value if (math.isfinite(p.normalized_value) and p.normalized_value > 0) else None


def to_abs_flow(p: NormalizedParameter, phys: ResolvedPhysiologyContext) -> float | None:
    val = _valid_pos_norm(p)
    if val is None:
        return None
    if p.weight_normalized:
        return (val * phys.weight_kg) if (phys.is_resolvable and phys.weight_kg is not None) else None
    return val


def to_l_kg(p: NormalizedParameter, phys: ResolvedPhysiologyContext) -> float | None:
    val = _valid_pos_norm(p)
    if val is None:
        return None
    if p.weight_normalized:
        return val
    return (val / phys.weight_kg) if (phys.is_resolvable and phys.weight_kg is not None) else None


def to_abs_vol_l(p: NormalizedParameter | None, phys: ResolvedPhysiologyContext) -> float | None:
    val = _valid_pos_norm(p)
    if val is None or p is None:
        return None
    if not p.weight_normalized:
        return val
    return (val * phys.weight_kg) if (phys.is_resolvable and phys.weight_kg is not None) else None


def eval_pl001_positive_finite(
    catalog: RuleCatalog, params: tuple[NormalizedParameter, ...]
) -> list[PlausibilityFinding]:
    rule = catalog.by_id("PL-001")
    if not params:
        return [make_finding(rule, parameter="all_parameters", status="NOT_APPLICABLE")]
    flagged = [
        make_finding(
            rule, parameter=p.name, status="FLAGGED", severity="CRITICAL",
            observed_value=p.observed_value, observed_unit=p.observed_unit,
            expected_range="> 0 (finite)", override_allowed=False,
        )
        for p in sorted(params, key=lambda x: (x.name, str(x.observed_value)))
        if isinstance(p.observed_value, str) or not math.isfinite(p.observed_value) or p.observed_value <= 0
    ]
    return flagged or [make_finding(rule, parameter="all_parameters", status="PASS", expected_range="> 0 (finite)")]


def eval_pl002_units(
    catalog: RuleCatalog, params: tuple[NormalizedParameter, ...]
) -> list[PlausibilityFinding]:
    rule = catalog.by_id("PL-002")
    if not params:
        return [make_finding(rule, parameter="all_parameters", status="NOT_APPLICABLE")]
    out: list[PlausibilityFinding] = []
    seen_canonical: dict[str, str] = {}
    for p in sorted(params, key=lambda x: (x.canonical_name, x.name, str(x.observed_value))):
        if p.param_class != "other":
            if p.canonical_name in seen_canonical:
                prev = seen_canonical[p.canonical_name]
                out.append(make_finding(
                    rule, parameter=p.name, status="FLAGGED", severity="CRITICAL",
                    observed_value=p.observed_value, observed_unit=p.observed_unit,
                    rationale=f"Duplicate canonical '{p.canonical_name}' ('{p.name}' vs '{prev}').",
                    override_allowed=False,
                ))
            else:
                seen_canonical[p.canonical_name] = p.name
        if p.unit_status == "missing":
            out.append(make_finding(
                rule, parameter=p.name, status="UNRESOLVED", severity="WARN",
                observed_value=p.observed_value, observed_unit=p.observed_unit,
                rationale=p.unit_message, override_allowed=False,
            ))
        elif p.unit_status in {"unknown", "incompatible"}:
            out.append(make_finding(
                rule, parameter=p.name, status="FLAGGED", severity="CRITICAL",
                observed_value=p.observed_value, observed_unit=p.observed_unit,
                rationale=p.unit_message, override_allowed=False,
            ))
    return out or [make_finding(rule, parameter="all_parameters", status="PASS", expected_range="compatible units")]


def eval_pl003_bioavailability(
    catalog: RuleCatalog, params: tuple[NormalizedParameter, ...]
) -> list[PlausibilityFinding]:
    rule = catalog.by_id("PL-003")
    f_param = by_canonical(params, "F")
    if f_param is None:
        return [make_finding(rule, parameter="F", status="NOT_APPLICABLE")]
    if isinstance(f_param.observed_value, str) or f_param.observed_value <= 0:
        return [make_finding(
            rule, parameter=f_param.name, status="FLAGGED", severity="CRITICAL",
            observed_value=f_param.observed_value, observed_unit=f_param.observed_unit,
            expected_range="(0, 1]", override_allowed=False,
        )]
    if f_param.unit_status != "compatible" or f_param.normalized_value is None:
        return [make_finding(rule, parameter=f_param.name, status="UNRESOLVED", severity="WARN")]
    val, max_f = f_param.normalized_value, rule.threshold_float("max_inclusive", 1.0)
    st: FindingStatus = "FLAGGED" if val > max_f else "PASS"
    sev: SeverityLevel = "HIGH" if val > max_f else "INFO"
    return [make_finding(
        rule, parameter=f_param.name, status=st, severity=sev,
        observed_value=f_param.observed_value, observed_unit=f_param.observed_unit,
        normalized_value=val, normalized_unit=f_param.normalized_unit,
        expected_range=f"(0, {max_f:g}]", override_allowed=True,
    )]


def eval_pl004_systemic_cl(
    catalog: RuleCatalog, model: PopPKModelInput,
    params: tuple[NormalizedParameter, ...], phys: ResolvedPhysiologyContext,
) -> list[PlausibilityFinding]:
    rule = catalog.by_id("PL-004")
    cl, clf, f_val = by_canonical(params, "CL"), by_canonical(params, "CL/F"), resolved_f_fraction(params)
    target, scale_factor = cl, 1.0
    if target is None:
        if clf is not None and f_val is not None:
            target, scale_factor = clf, f_val
        elif clf is not None:
            return [
                make_finding(
                    rule, parameter=clf.name, status="NOT_APPLICABLE",
                    observed_value=clf.observed_value, observed_unit=clf.observed_unit,
                    rationale="Apparent CL/F cannot be bounded by Q_H without known bioavailability F.",
                )
            ]
        else:
            return [make_finding(rule, parameter="CL", status="NOT_APPLICABLE")]
    if model.context.modality in {"monoclonal_antibody", "biologic_other"} or (
        model.context.elimination_pathway in {"renal", "catabolic", "extrahepatic"}
    ):
        return [make_finding(rule, parameter=target.name, status="NOT_APPLICABLE")]
    if (
        model.context.modality == "unknown" or model.context.elimination_pathway == "unknown"
        or target.resolved_scale == "ambiguous"
        or (model.context.clearance_matrix == "plasma" and phys.blood_to_plasma_ratio is None)
        or target.unit_status != "compatible" or not phys.is_resolvable
        or phys.hepatic_blood_flow_l_h is None or phys.cardiac_output_l_h is None
    ):
        return [
            make_finding(
                rule, parameter=target.name, status="UNRESOLVED", severity="WARN",
                observed_value=target.observed_value, observed_unit=target.observed_unit,
                rationale=phys.unresolved_reason or target.unit_message,
            )
        ]
    raw_flow = to_abs_flow(target, phys)
    if raw_flow is None:
        return [make_finding(rule, parameter=target.name, status="UNRESOLVED", severity="WARN")]
    rb = phys.blood_to_plasma_ratio if phys.blood_to_plasma_ratio is not None else 1.0
    cl_blood_l_h = (raw_flow * scale_factor) / rb
    qh, co = phys.hepatic_blood_flow_l_h, phys.cardiac_output_l_h
    rng = f"<= {qh:.2f} L/h (Q_H); cardiac output <= {co:.2f} L/h"
    st: FindingStatus = "FLAGGED" if cl_blood_l_h > qh else "PASS"
    sev: SeverityLevel = "HIGH" if cl_blood_l_h > co else ("WARN" if cl_blood_l_h > qh else "INFO")
    return [
        make_finding(
            rule, parameter=target.name, status=st, severity=sev,
            observed_value=target.observed_value, observed_unit=target.observed_unit,
            normalized_value=cl_blood_l_h, normalized_unit="L/h", expected_range=rng,
        )
    ]


def eval_pl005_renal_cl(
    catalog: RuleCatalog, model: PopPKModelInput,
    params: tuple[NormalizedParameter, ...], phys: ResolvedPhysiologyContext,
) -> list[PlausibilityFinding]:
    rule = catalog.by_id("PL-005")
    clr = by_canonical(params, "CL_R")
    if clr is None:
        return [make_finding(rule, parameter="CL_R", status="NOT_APPLICABLE")]
    if (
        clr.unit_status != "compatible" or clr.resolved_scale == "ambiguous"
        or not phys.is_resolvable or phys.renal_plasma_flow_l_h is None or phys.gfr_l_h is None
    ):
        return [
            make_finding(
                rule, parameter=clr.name, status="UNRESOLVED", severity="WARN",
                rationale=phys.unresolved_reason or clr.unit_message,
            )
        ]
    clr_l_h = to_abs_flow(clr, phys)
    if clr_l_h is None:
        return [make_finding(rule, parameter=clr.name, status="UNRESOLVED", severity="WARN")]
    cl = by_canonical(params, "CL")
    cl_l_h = to_abs_flow(cl, phys) if (cl and cl.resolved_scale == "systemic") else None
    if cl_l_h is not None and clr_l_h > cl_l_h:
        return [
            make_finding(
                rule, parameter=clr.name, status="FLAGGED", severity="HIGH",
                observed_value=clr.observed_value, observed_unit=clr.observed_unit,
                normalized_value=clr_l_h, normalized_unit="L/h",
                expected_range=f"<= total systemic CL ({cl_l_h:.2f} L/h)",
                rationale="Renal clearance CL_R cannot exceed total systemic clearance CL.",
                override_allowed=False,
            )
        ]
    qr = phys.renal_plasma_flow_l_h
    if clr_l_h > qr:
        return [
            make_finding(
                rule, parameter=clr.name, status="FLAGGED", severity="HIGH",
                observed_value=clr.observed_value, observed_unit=clr.observed_unit,
                normalized_value=clr_l_h, normalized_unit="L/h", expected_range=f"<= {qr:.2f} L/h (Q_R,plasma)",
            )
        ]
    fu = model.context.fraction_unbound_plasma
    if fu is not None and (not math.isfinite(fu) or fu <= 0.0 or fu > 1.0):
        return [make_finding(rule, parameter=clr.name, status="UNRESOLVED", severity="WARN")]
    filt = (fu * phys.gfr_l_h) if fu is not None else phys.gfr_l_h
    label = "fu*GFR without secretion" if fu is not None else "GFR without secretion"
    if clr_l_h > filt and not model.context.renal_secretion_known:
        return [
            make_finding(
                rule, parameter=clr.name, status="FLAGGED", severity="WARN",
                observed_value=clr.observed_value, observed_unit=clr.observed_unit,
                normalized_value=clr_l_h, normalized_unit="L/h", expected_range=f"<= {filt:.2f} L/h ({label})",
            )
        ]
    return [
        make_finding(
            rule, parameter=clr.name, status="PASS",
            observed_value=clr.observed_value, observed_unit=clr.observed_unit,
            normalized_value=clr_l_h, normalized_unit="L/h", expected_range=f"<= {qr:.2f} L/h",
        )
    ]


def eval_pl006_central_volume(
    catalog: RuleCatalog, params: tuple[NormalizedParameter, ...], phys: ResolvedPhysiologyContext
) -> list[PlausibilityFinding]:
    rule = catalog.by_id("PL-006")
    vc = by_canonical(params, "Vc") or by_canonical(params, "Vc/F")
    if vc is None:
        return [make_finding(rule, parameter="Vc", status="NOT_APPLICABLE")]
    raw_l_kg = (
        to_l_kg(vc, phys)
        if (vc.resolved_scale != "ambiguous" and vc.unit_status == "compatible" and phys.is_resolvable)
        else None
    )
    if raw_l_kg is None:
        return [
            make_finding(
                rule, parameter=vc.name, status="UNRESOLVED", severity="WARN",
                rationale=phys.unresolved_reason or vc.unit_message,
            )
        ]
    f_val = resolved_f_fraction(params)
    is_sys = vc.resolved_scale == "systemic" or f_val is not None
    vc_l_kg = raw_l_kg * f_val if (vc.resolved_scale == "apparent" and f_val is not None) else raw_l_kg
    min_vc = rule.threshold_float("min_vc_l_kg", phys.min_vc_l_kg)
    warn_vc = rule.threshold_float("warn_vc_l_kg", phys.warn_vc_l_kg)
    high_vc = rule.threshold_float("high_vc_l_kg", phys.high_vc_l_kg)
    rng = f"[{min_vc:.2f}, {warn_vc:.1f}] L/kg" if is_sys else f">= {min_vc:.2f} L/kg (apparent Vc/F)"
    st: FindingStatus = "PASS"
    sev: SeverityLevel = "INFO"
    if vc_l_kg < min_vc or (is_sys and vc_l_kg > high_vc):
        st, sev = "FLAGGED", "HIGH"
    elif is_sys and vc_l_kg > warn_vc:
        st, sev = "FLAGGED", "WARN"
    return [
        make_finding(
            rule, parameter=vc.name, status=st, severity=sev,
            observed_value=vc.observed_value, observed_unit=vc.observed_unit,
            normalized_value=vc_l_kg, normalized_unit="L/kg", expected_range=rng,
        )
    ]


def eval_pl007_vss_consistency(
    catalog: RuleCatalog, model: PopPKModelInput,
    params: tuple[NormalizedParameter, ...], phys: ResolvedPhysiologyContext,
) -> list[PlausibilityFinding]:
    rule = catalog.by_id("PL-007")
    vss, vc, vp, vp2 = (
        by_canonical(params, "Vss"), by_canonical(params, "Vc"),
        by_canonical(params, "Vp"), by_canonical(params, "Vp2"),
    )
    if vss is None and vp is None:
        return [make_finding(rule, parameter="Vss", status="NOT_APPLICABLE")]
    c_struct = model.context.compartmental_structure
    if c_struct == "unknown" or not phys.is_resolvable or any(
        p is not None and p.resolved_scale == "ambiguous" for p in (vss, vc, vp, vp2)
    ):
        return [make_finding(rule, parameter=vss.name if vss else "Vss", status="UNRESOLVED", severity="WARN")]
    is_mammillary = c_struct == "mammillary_central_elimination"
    if not is_mammillary and vss is None:
        return [make_finding(rule, parameter="Vss", status="NOT_APPLICABLE")]
    vc_l_kg, vp_l_kg = (to_l_kg(vc, phys) if vc else None), (to_l_kg(vp, phys) if vp else None)
    vp2_l_kg = to_l_kg(vp2, phys) if vp2 is not None else 0.0
    if (vp2 is not None and vp2_l_kg is None) or (vc and vc_l_kg is None) or (vp and vp_l_kg is None):
        return [make_finding(rule, parameter=vss.name if vss else "Vss", status="UNRESOLVED", severity="WARN")]
    comp_sum = (
        (vc_l_kg + vp_l_kg + (vp2_l_kg or 0.0))
        if (is_mammillary and vc_l_kg is not None and vp_l_kg is not None)
        else None
    )
    tol = rule.threshold_float("sum_rel_tol", 0.05)
    if vss is not None:
        vss_l_kg = to_l_kg(vss, phys)
        if vss_l_kg is None:
            return [make_finding(rule, parameter=vss.name, status="UNRESOLVED", severity="WARN")]
        if comp_sum is not None and abs(vss_l_kg - comp_sum) / max(comp_sum, 1e-12) > tol:
            return [make_finding(
                rule, parameter=vss.name, status="FLAGGED", severity="HIGH",
                observed_value=vss.observed_value, observed_unit=vss.observed_unit,
                normalized_value=vss_l_kg, normalized_unit="L/kg",
                expected_range=f"Vc + Vp = {comp_sum:.3f} L/kg (+/-{tol*100:.0f}%)",
                rationale="Reported Vss does not match mammillary sum Vc + sum(Vp).", override_allowed=False,
            )]
        if is_mammillary and comp_sum is None and vc_l_kg is not None and vss_l_kg < vc_l_kg * (1.0 - tol):
            return [make_finding(
                rule, parameter=vss.name, status="FLAGGED", severity="HIGH",
                observed_value=vss.observed_value, observed_unit=vss.observed_unit,
                normalized_value=vss_l_kg, normalized_unit="L/kg",
                expected_range=f">= Vc ({vc_l_kg:.3f} L/kg)", override_allowed=False,
            )]
        target_val, target_name = vss_l_kg, vss.name
    elif comp_sum is not None:
        target_val, target_name = comp_sum, "Vss (derived)"
    else:
        return [make_finding(rule, parameter="Vss", status="UNRESOLVED", severity="WARN")]
    min_vss = rule.threshold_float("min_vss_l_kg", phys.min_vss_l_kg)
    max_vss = rule.threshold_float("max_vss_l_kg", phys.max_vss_l_kg)
    st: FindingStatus = "FLAGGED" if (target_val < min_vss or target_val > max_vss) else "PASS"
    sev: SeverityLevel = "HIGH" if target_val < min_vss else ("WARN" if target_val > max_vss else "INFO")
    return [make_finding(
        rule, parameter=target_name, status=st, severity=sev,
        normalized_value=target_val, normalized_unit="L/kg", expected_range=f"[{min_vss:.2f}, {max_vss:.1f}] L/kg",
    )]
