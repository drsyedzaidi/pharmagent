"""Physiological reference context resolution and body-weight scaling.

Treats pediatric, neonate, pregnancy, elderly, and organ-impairment adjustments as
UNRESOLVED unless both the required clinical covariate and an owner-reviewed
population adjustment rule are available.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from app.compute.poppk_plausibility.schema import PopPKContext, ReferencePhysiology


@dataclass(frozen=True)
class ResolvedPhysiologyContext:
    is_resolvable: bool
    unresolved_reason: str | None
    weight_kg: float | None
    flow_scale_factor: float | None
    volume_scale_factor: float | None
    hepatic_blood_flow_l_h: float | None
    cardiac_output_l_h: float | None
    gfr_l_h: float | None
    renal_plasma_flow_l_h: float | None
    plasma_volume_l: float | None
    ecf_volume_l: float | None
    tbw_volume_l: float | None
    blood_to_plasma_ratio: float | None = None
    min_vc_l_kg: float = 0.03
    warn_vc_l_kg: float = 20.0
    high_vc_l_kg: float = 50.0
    min_vss_l_kg: float = 0.04
    max_vss_l_kg: float = 100.0


def _unresolved(reason: str) -> ResolvedPhysiologyContext:
    return ResolvedPhysiologyContext(
        is_resolvable=False,
        unresolved_reason=reason,
        weight_kg=None,
        flow_scale_factor=None,
        volume_scale_factor=None,
        hepatic_blood_flow_l_h=None,
        cardiac_output_l_h=None,
        gfr_l_h=None,
        renal_plasma_flow_l_h=None,
        plasma_volume_l=None,
        ecf_volume_l=None,
        tbw_volume_l=None,
    )


def _check_covariate_requirements(ctx: PopPKContext) -> str | None:
    if ctx.age_years is not None and (not math.isfinite(ctx.age_years) or ctx.age_years < 0.0 or ctx.age_years > 130.0):
        return "Provided 'age_years' must be finite and within [0, 130] years."
    if ctx.gestational_age_weeks is not None and (
        not math.isfinite(ctx.gestational_age_weeks)
        or ctx.gestational_age_weeks <= 0.0
        or ctx.gestational_age_weeks > 45.0
    ):
        return "Provided 'gestational_age_weeks' must be finite and in (0, 45]."
    if ctx.egfr_ml_min_1_73m2 is not None and (
        not math.isfinite(ctx.egfr_ml_min_1_73m2) or ctx.egfr_ml_min_1_73m2 <= 0.0
    ):
        return "Provided 'egfr_ml_min_1_73m2' must be finite and strictly positive (> 0)."
    if ctx.blood_to_plasma_ratio is not None and (
        not math.isfinite(ctx.blood_to_plasma_ratio) or ctx.blood_to_plasma_ratio <= 0.0
    ):
        return "Provided 'blood_to_plasma_ratio' must be finite and strictly positive (> 0)."

    if ctx.population in {"pediatric", "neonate", "elderly"} and ctx.age_years is None:
        return f"Population '{ctx.population}' requires explicit 'age_years' to resolve physiology."
    if ctx.population == "pregnancy" and ctx.gestational_age_weeks is None:
        return "Pregnancy population requires explicit 'gestational_age_weeks' to resolve physiology."
    if ctx.population == "hepatic_impairment" and ctx.child_pugh is None:
        return "Hepatic impairment population requires explicit 'child_pugh' class ('A', 'B', or 'C')."
    if ctx.population == "renal_impairment" and ctx.egfr_ml_min_1_73m2 is None:
        return "Renal impairment population requires explicit 'egfr_ml_min_1_73m2' to resolve physiology."
    return None


def resolve_physiology_context(
    ctx: PopPKContext,
    ref_phys: ReferencePhysiology,
) -> ResolvedPhysiologyContext:
    """Resolve reference physiology for `ctx`, returning `is_resolvable=False` when context is incomplete."""
    if ctx.species.strip().lower() != "human":
        return _unresolved(f"Non-human species '{ctx.species}' requires species-specific reference physiology.")

    if ctx.population == "unknown":
        return _unresolved("Population is 'unknown'; cannot resolve physiological flow or volume bounds.")

    if ctx.reference_weight_kg is None or not math.isfinite(ctx.reference_weight_kg) or ctx.reference_weight_kg <= 0:
        return _unresolved(f"Population '{ctx.population}' is missing a valid positive 'reference_weight_kg'.")

    cov_err = _check_covariate_requirements(ctx)
    if cov_err is not None:
        return _unresolved(cov_err)

    if ctx.population != "adult":
        rule_status = ref_phys.special_population_rules_status.get(ctx.population, "pending_owner_review")
        if rule_status != "approved":
            return _unresolved(
                f"Special-population physiological adjustment rule for '{ctx.population}' is "
                f"{rule_status}; physiology-dependent bounds are left UNRESOLVED."
            )

    base_adult = ref_phys.adult_70kg
    pop_scale = ref_phys.population_scaling
    wt = float(ctx.reference_weight_kg)
    weight_ratio = wt / base_adult.weight_kg
    flow_scale = math.pow(weight_ratio, base_adult.allometric_flow_exponent)
    vol_scale = math.pow(weight_ratio, base_adult.allometric_volume_exponent)

    maturation_factor = 1.0
    if ctx.population == "neonate":
        maturation_factor = pop_scale.neonate_maturation_cl_factor
    elif ctx.population == "pediatric" and ctx.age_years is not None and ctx.age_years < 2.0:
        maturation_factor = 0.5 + 0.25 * max(0.0, ctx.age_years)

    elderly_factor = 1.0
    if ctx.population == "elderly" and ctx.age_years is not None:
        decades = max(1.0, (ctx.age_years - 65.0) / 10.0)
        elderly_factor = math.pow(pop_scale.elderly_flow_factor_per_decade_over_65, decades)

    hep_factor = 1.0
    if ctx.population == "hepatic_impairment" and ctx.child_pugh is not None:
        hep_factor = float(pop_scale.child_pugh_flow_factors[ctx.child_pugh])

    preg_flow_factor = 1.0
    preg_gfr_factor = 1.0
    preg_vol_factor = 1.0
    if ctx.population == "pregnancy" and ctx.gestational_age_weeks is not None:
        trimester_prog = min(max(ctx.gestational_age_weeks / 40.0, 0.0), 1.0)
        preg_flow_factor = 1.0 + (pop_scale.pregnancy_hepatic_flow_factor - 1.0) * trimester_prog
        preg_gfr_factor = 1.0 + (pop_scale.pregnancy_gfr_factor - 1.0) * trimester_prog
        preg_vol_factor = 1.0 + (pop_scale.pregnancy_plasma_volume_factor - 1.0) * trimester_prog

    combined_flow_factor = flow_scale * maturation_factor * elderly_factor
    qh = base_adult.hepatic_blood_flow_l_h * combined_flow_factor * hep_factor * preg_flow_factor
    co = base_adult.cardiac_output_l_h * flow_scale * elderly_factor * preg_flow_factor

    gfr_l_h: float | None
    qr_plasma: float | None
    if ctx.egfr_ml_min_1_73m2 is not None and ctx.egfr_ml_min_1_73m2 < 90.0:
        renal_status = ref_phys.special_population_rules_status.get("renal_impairment", "pending_owner_review")
        if renal_status != "approved":
            gfr_l_h = None
            qr_plasma = None
        else:
            renal_factor = min(float(ctx.egfr_ml_min_1_73m2) / 125.0, 1.2)
            gfr_l_h = base_adult.gfr_l_h * combined_flow_factor * renal_factor * preg_gfr_factor
            qr_plasma = base_adult.renal_plasma_flow_l_h * combined_flow_factor * renal_factor * preg_flow_factor
    else:
        gfr_l_h = base_adult.gfr_l_h * combined_flow_factor * preg_gfr_factor
        qr_plasma = base_adult.renal_plasma_flow_l_h * combined_flow_factor * preg_flow_factor

    rb = float(ctx.blood_to_plasma_ratio) if ctx.blood_to_plasma_ratio is not None else None
    return ResolvedPhysiologyContext(
        is_resolvable=True,
        unresolved_reason=None,
        weight_kg=wt,
        flow_scale_factor=flow_scale,
        volume_scale_factor=vol_scale,
        hepatic_blood_flow_l_h=qh,
        cardiac_output_l_h=co,
        gfr_l_h=gfr_l_h,
        renal_plasma_flow_l_h=qr_plasma,
        plasma_volume_l=base_adult.plasma_volume_l_kg * wt * preg_vol_factor,
        ecf_volume_l=base_adult.ecf_volume_l_kg * wt,
        tbw_volume_l=base_adult.tbw_volume_l_kg * wt,
        blood_to_plasma_ratio=rb,
    )
