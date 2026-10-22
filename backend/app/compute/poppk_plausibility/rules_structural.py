"""Deterministic domain, unit, bioavailability, clearance, and volume plausibility rules.

Implements rules PL-001 through PL-007.
"""
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
    rule: RuleDefinition, *, parameter: str, status: FindingStatus,
    severity: SeverityLevel | None = None, observed_value: float | str | None = None,
    observed_unit: str | None = None, normalized_value: float | None = None,
    normalized_unit: str | None = None, expected_range: str = "",
    rationale: str | None = None, recommendation: str | None = None, override_allowed: bool | None = None,
) -> PlausibilityFinding:
    eff_sev: SeverityLevel = (
        severity if severity is not None else ("INFO" if status != "FLAGGED" else rule.default_severity)
    )
    eff_override = rule.override_allowed if override_allowed is None else override_allowed
    return PlausibilityFinding(
        rule_id=rule.rule_id, rule_version=rule.version, category=rule.category, parameter=parameter,
        observed_value=observed_value, observed_unit=observed_unit,
        normalized_value=normalized_value, normalized_unit=normalized_unit,
        expected_range=expected_range, severity=eff_sev,
        status=status, scientific_rationale=rationale or rule.scientific_rationale,
        recommendation=recommendation or rule.recommendation, reference_basis=rule.reference_basis,
        override_allowed=eff_override,
    )


def by_canonical(params: tuple[NormalizedParameter, ...], name: str) -> NormalizedParameter | None:
    for p in params:
        if p.canonical_name == name:
            return p
    return None


def to_abs_flow(p: NormalizedParameter, phys: ResolvedPhysiologyContext) -> float | None:
    if (
        p.unit_status != "valid"
        or p.normalized_value is None
        or not math.isfinite(p.normalized_value)
        or p.normalized_value <= 0
    ):
        return None
    if p.weight_normalized:
        if not phys.is_resolvable or phys.weight_kg is None:
            return None
        return p.normalized_value * phys.weight_kg
    return p.normalized_value


def to_l_kg(p: NormalizedParameter, phys: ResolvedPhysiologyContext) -> float | None:
    if (
        p.unit_status != "valid"
        or p.normalized_value is None
        or not math.isfinite(p.normalized_value)
        or p.normalized_value <= 0
    ):
        return None
    if p.weight_normalized:
        return p.normalized_value
    if not phys.is_resolvable or phys.weight_kg is None:
        return None
    return p.normalized_value / phys.weight_kg


def to_abs_vol_l(p: NormalizedParameter | None, phys: ResolvedPhysiologyContext) -> float | None:
    if p is None:
        return None
    l_kg = to_l_kg(p, phys)
    if l_kg is not None and phys.weight_kg is not None:
        return l_kg * phys.weight_kg
    if not p.weight_normalized and p.unit_status == "valid" and p.normalized_value is not None:
        if math.isfinite(p.normalized_value) and p.normalized_value > 0:
            return p.normalized_value
    return None


def eval_pl001_positive_finite(
    catalog: RuleCatalog, params: tuple[NormalizedParameter, ...]
) -> list[PlausibilityFinding]:
    rule = catalog.by_id("PL-001")
    if not params:
        return [make_finding(rule, parameter="all_parameters", status="NOT_APPLICABLE")]
    flagged: list[PlausibilityFinding] = []
    for p in sorted(params, key=lambda x: x.name):
        if not math.isfinite(p.observed_value) or p.observed_value <= 0:
            obs: float | str = (
                str(p.observed_value) if not math.isfinite(p.observed_value) else p.observed_value
            )
            flagged.append(
                make_finding(
                    rule, parameter=p.name, status="FLAGGED", severity="CRITICAL",
                    observed_value=obs, observed_unit=p.observed_unit,
                    expected_range="> 0 (finite)", override_allowed=False,
                )
            )
    if flagged:
        return flagged
    return [make_finding(rule, parameter="all_parameters", status="PASS", expected_range="> 0 (finite)")]


def eval_pl002_units(
    catalog: RuleCatalog, params: tuple[NormalizedParameter, ...]
) -> list[PlausibilityFinding]:
    rule = catalog.by_id("PL-002")
    if not params:
        return [make_finding(rule, parameter="all_parameters", status="NOT_APPLICABLE")]
    out: list[PlausibilityFinding] = []
    for p in sorted(params, key=lambda x: x.name):
        if p.unit_status == "missing":
            out.append(
                make_finding(
                    rule, parameter=p.name, status="UNRESOLVED", severity="WARN",
                    observed_value=p.observed_value, observed_unit=p.observed_unit,
                    rationale=p.unit_message, override_allowed=False,
                )
            )
        elif p.unit_status in {"unknown", "incompatible"}:
            out.append(
                make_finding(
                    rule, parameter=p.name, status="FLAGGED", severity="CRITICAL",
                    observed_value=p.observed_value, observed_unit=p.observed_unit,
                    rationale=p.unit_message, override_allowed=False,
                )
            )
    if out:
        return out
    return [
        make_finding(
            rule, parameter="all_parameters", status="PASS", expected_range="compatible canonical units",
        )
    ]


def eval_pl003_bioavailability(
    catalog: RuleCatalog, params: tuple[NormalizedParameter, ...]
) -> list[PlausibilityFinding]:
    rule = catalog.by_id("PL-003")
    f_param = by_canonical(params, "F")
    if f_param is None:
        return [make_finding(rule, parameter="F", status="NOT_APPLICABLE")]
    if not math.isfinite(f_param.observed_value) or f_param.observed_value <= 0:
        obs: float | str = (
            str(f_param.observed_value)
            if not math.isfinite(f_param.observed_value)
            else f_param.observed_value
        )
        return [
            make_finding(
                rule, parameter=f_param.name, status="FLAGGED", severity="CRITICAL",
                observed_value=obs, observed_unit=f_param.observed_unit,
                expected_range="(0, 1]", override_allowed=False,
            )
        ]
    if f_param.unit_status != "valid" or f_param.normalized_value is None:
        return [make_finding(rule, parameter=f_param.name, status="UNRESOLVED", severity="WARN")]
    val = f_param.normalized_value
    status: FindingStatus = "FLAGGED" if val > 1.0 else "PASS"
    sev: SeverityLevel = "HIGH" if val > 1.0 else "INFO"
    return [
        make_finding(
            rule, parameter=f_param.name, status=status, severity=sev,
            observed_value=f_param.observed_value, observed_unit=f_param.observed_unit,
            normalized_value=val, normalized_unit=f_param.normalized_unit,
            expected_range="(0, 1]", override_allowed=True,
        )
    ]


def eval_pl004_systemic_cl(
    catalog: RuleCatalog, params: tuple[NormalizedParameter, ...], phys: ResolvedPhysiologyContext
) -> list[PlausibilityFinding]:
    rule = catalog.by_id("PL-004")
    cl = by_canonical(params, "CL")
    clf = by_canonical(params, "CL/F")
    if cl is None:
        if clf is not None:
            return [
                make_finding(
                    rule, parameter=clf.name, status="NOT_APPLICABLE", observed_value=clf.observed_value,
                    observed_unit=clf.observed_unit, rationale=(
                        "Apparent clearance (CL/F) cannot be bounded by hepatic blood flow "
                        "without known bioavailability F."
                    ),
                )
            ]
        return [make_finding(rule, parameter="CL", status="NOT_APPLICABLE")]
    if (
        cl.unit_status != "valid"
        or not phys.is_resolvable
        or phys.hepatic_blood_flow_l_h is None
        or phys.cardiac_output_l_h is None
    ):
        return [
            make_finding(
                rule, parameter=cl.name, status="UNRESOLVED", severity="WARN",
                observed_value=cl.observed_value, observed_unit=cl.observed_unit,
                rationale=phys.unresolved_reason or cl.unit_message,
            )
        ]
    cl_l_h = to_abs_flow(cl, phys)
    if cl_l_h is None:
        return [make_finding(rule, parameter=cl.name, status="UNRESOLVED", severity="WARN")]
    qh = phys.hepatic_blood_flow_l_h
    co = phys.cardiac_output_l_h
    rng = f"<= {qh:.2f} L/h (Q_H); cardiac output <= {co:.2f} L/h"
    if cl_l_h > co:
        sev, st = "HIGH", "FLAGGED"
    elif cl_l_h > qh:
        sev, st = "WARN", "FLAGGED"
    else:
        sev, st = "INFO", "PASS"
    return [
        make_finding(
            rule, parameter=cl.name, status=st, severity=sev,
            observed_value=cl.observed_value, observed_unit=cl.observed_unit,
            normalized_value=cl_l_h, normalized_unit="L/h", expected_range=rng,
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
        clr.unit_status != "valid"
        or not phys.is_resolvable
        or phys.renal_plasma_flow_l_h is None
        or phys.gfr_l_h is None
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
    if fu is not None and 0 < fu <= 1.0:
        filt = fu * phys.gfr_l_h
        if clr_l_h > filt and not model.context.renal_secretion_known:
            return [
                make_finding(
                    rule, parameter=clr.name, status="FLAGGED", severity="WARN",
                    observed_value=clr.observed_value, observed_unit=clr.observed_unit,
                    normalized_value=clr_l_h, normalized_unit="L/h",
                    expected_range=f"<= {filt:.2f} L/h (fu*GFR without secretion)",
                )
            ]
    return [
        make_finding(
            rule, parameter=clr.name, status="PASS", observed_value=clr.observed_value,
            observed_unit=clr.observed_unit, normalized_value=clr_l_h,
            normalized_unit="L/h", expected_range=f"<= {qr:.2f} L/h",
        )
    ]


def eval_pl006_central_volume(
    catalog: RuleCatalog, params: tuple[NormalizedParameter, ...], phys: ResolvedPhysiologyContext
) -> list[PlausibilityFinding]:
    rule = catalog.by_id("PL-006")
    vc = by_canonical(params, "Vc") or by_canonical(params, "Vc/F")
    if vc is None:
        return [make_finding(rule, parameter="Vc", status="NOT_APPLICABLE")]
    if vc.unit_status != "valid" or not phys.is_resolvable:
        return [
            make_finding(
                rule, parameter=vc.name, status="UNRESOLVED", severity="WARN",
                rationale=phys.unresolved_reason or vc.unit_message,
            )
        ]
    vc_l_kg = to_l_kg(vc, phys)
    if vc_l_kg is None:
        return [make_finding(rule, parameter=vc.name, status="UNRESOLVED", severity="WARN")]
    rng = f"[{phys.min_vc_l_kg:.2f}, {phys.warn_vc_l_kg:.1f}] L/kg"
    st: FindingStatus = "PASS"
    sev: SeverityLevel = "INFO"
    if vc_l_kg < phys.min_vc_l_kg:
        st, sev = "FLAGGED", "HIGH"
    elif vc.resolved_scale == "systemic":
        if vc_l_kg > phys.high_vc_l_kg:
            st, sev = "FLAGGED", "HIGH"
        elif vc_l_kg > phys.warn_vc_l_kg:
            st, sev = "FLAGGED", "WARN"
    return [
        make_finding(
            rule, parameter=vc.name, status=st, severity=sev,
            observed_value=vc.observed_value, observed_unit=vc.observed_unit,
            normalized_value=vc_l_kg, normalized_unit="L/kg", expected_range=rng,
        )
    ]


def eval_pl007_vss_consistency(
    catalog: RuleCatalog, params: tuple[NormalizedParameter, ...], phys: ResolvedPhysiologyContext
) -> list[PlausibilityFinding]:
    rule = catalog.by_id("PL-007")
    vss = by_canonical(params, "Vss")
    vc = by_canonical(params, "Vc")
    vp = by_canonical(params, "Vp")
    vp2 = by_canonical(params, "Vp2")
    if vss is None and vp is None:
        return [make_finding(rule, parameter="Vss", status="NOT_APPLICABLE")]
    if not phys.is_resolvable:
        return [
            make_finding(
                rule, parameter=vss.name if vss else "Vss", status="UNRESOLVED", severity="WARN",
                rationale=phys.unresolved_reason,
            )
        ]
    vc_l_kg = to_l_kg(vc, phys) if vc else None
    vp_l_kg = to_l_kg(vp, phys) if vp else None
    vp2_l_kg = to_l_kg(vp2, phys) if vp2 else 0.0
    comp_sum = (
        (vc_l_kg + vp_l_kg + (vp2_l_kg or 0.0))
        if (vc_l_kg is not None and vp_l_kg is not None)
        else None
    )
    if vss is not None:
        vss_l_kg = to_l_kg(vss, phys)
        if vss_l_kg is None:
            return [make_finding(rule, parameter=vss.name, status="UNRESOLVED", severity="WARN")]
        if comp_sum is not None and abs(vss_l_kg - comp_sum) / max(comp_sum, 1e-12) > 0.05:
            return [
                make_finding(
                    rule, parameter=vss.name, status="FLAGGED", severity="HIGH",
                    observed_value=vss.observed_value, observed_unit=vss.observed_unit,
                    normalized_value=vss_l_kg, normalized_unit="L/kg",
                    expected_range=f"Vc + Vp = {comp_sum:.3f} L/kg (+/-5%)",
                    rationale="Reported Vss does not match the sum of compartmental volumes Vc + sum(Vp).",
                )
            ]
        target_val, target_name = vss_l_kg, vss.name
    elif comp_sum is not None:
        target_val, target_name = comp_sum, "Vss (derived)"
    else:
        return [make_finding(rule, parameter="Vss", status="UNRESOLVED", severity="WARN")]
    rng = f"[{phys.min_vss_l_kg:.2f}, {phys.max_vss_l_kg:.1f}] L/kg"
    if target_val < phys.min_vss_l_kg:
        st, sev = "FLAGGED", "HIGH"
    elif target_val > phys.max_vss_l_kg:
        st, sev = "FLAGGED", "WARN"
    else:
        st, sev = "PASS", "INFO"
    return [
        make_finding(
            rule, parameter=target_name, status=st, severity=sev,
            normalized_value=target_val, normalized_unit="L/kg", expected_range=rng,
        )
    ]
