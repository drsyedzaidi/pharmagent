"""Population, body-size, organ-function, and modality physiological context resolver.

Computes context-scaled physiological reference bounds and explicitly tracks
when context is incomplete (`UNRESOLVED`) so the engine never falls back to a
silent false `PASS`.
"""
from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict

from app.compute.poppk_plausibility.schema import PopPKContext, ReferencePhysiology


class ResolvedPhysiologyContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    is_resolvable: bool
    unresolved_reason: str = ""
    weight_kg: float | None = None
    flow_scale_factor: float | None = None
    hepatic_blood_flow_l_h: float | None = None
    cardiac_output_l_h: float | None = None
    gfr_l_h: float | None = None
    renal_plasma_flow_l_h: float | None = None
    min_vc_l_kg: float
    warn_vc_l_kg: float
    high_vc_l_kg: float
    min_vss_l_kg: float
    max_vss_l_kg: float
    plasma_volume_l: float | None = None
    ecf_volume_l: float | None = None
    tbw_volume_l: float | None = None


_CHILD_PUGH_QH_FACTOR: dict[str, float] = {
    "A": 0.85,
    "B": 0.70,
    "C": 0.50,
}


def resolve_physiology_context(
    ctx: PopPKContext,
    ref: ReferencePhysiology,
) -> ResolvedPhysiologyContext:
    """Resolve `PopPKContext` into scaled physiological bounds or an explicit unresolved state."""
    base_kwargs = {
        "min_vc_l_kg": ref.min_central_volume_l_kg,
        "warn_vc_l_kg": 20.0,
        "high_vc_l_kg": ref.max_central_volume_l_kg,
        "min_vss_l_kg": ref.min_central_volume_l_kg,
        "max_vss_l_kg": ref.max_vss_volume_l_kg,
    }

    if ctx.species.strip().lower() != "human":
        return ResolvedPhysiologyContext(
            is_resolvable=False,
            unresolved_reason=f"Species '{ctx.species}' does not have human physiological reference bounds.",
            **base_kwargs,
        )

    if ctx.population == "unknown":
        return ResolvedPhysiologyContext(
            is_resolvable=False,
            unresolved_reason="Population is 'unknown'; physiological flow and volume scaling cannot be resolved.",
            **base_kwargs,
        )

    if ctx.age_years is not None and (not math.isfinite(ctx.age_years) or not (0.0 <= ctx.age_years <= 130.0)):
        return ResolvedPhysiologyContext(
            is_resolvable=False,
            unresolved_reason="Provided 'age_years' must be a finite value in [0, 130] years.",
            **base_kwargs,
        )

    if ctx.population in {"pediatric", "neonatal"} and ctx.reference_weight_kg is None:
        return ResolvedPhysiologyContext(
            is_resolvable=False,
            unresolved_reason=(
                f"Population '{ctx.population}' requires an explicit 'reference_weight_kg' "
                "to scale physiological flows and volumes."
            ),
            **base_kwargs,
        )

    if ctx.population == "pregnancy":
        ga = ctx.gestational_age_weeks
        if ga is None or not math.isfinite(ga) or not (0.0 < ga <= 45.0):
            return ResolvedPhysiologyContext(
                is_resolvable=False,
                unresolved_reason=(
                    "Population 'pregnancy' requires a finite 'gestational_age_weeks' in (0, 45] "
                    "to resolve pregnancy-induced hemodynamic and volume changes."
                ),
                **base_kwargs,
            )
    elif ctx.gestational_age_weeks is not None:
        ga = ctx.gestational_age_weeks
        if not math.isfinite(ga) or not (0.0 < ga <= 45.0):
            return ResolvedPhysiologyContext(
                is_resolvable=False,
                unresolved_reason="Provided 'gestational_age_weeks' must be a finite value in (0, 45].",
                **base_kwargs,
            )

    if ctx.population == "hepatic_impairment" and ctx.child_pugh is None:
        return ResolvedPhysiologyContext(
            is_resolvable=False,
            unresolved_reason="Population 'hepatic_impairment' requires 'child_pugh' ('A', 'B', or 'C').",
            **base_kwargs,
        )

    rb = 1.0
    if ctx.blood_to_plasma_ratio is not None:
        if not math.isfinite(ctx.blood_to_plasma_ratio) or ctx.blood_to_plasma_ratio <= 0.0:
            return ResolvedPhysiologyContext(
                is_resolvable=False,
                unresolved_reason="Provided 'blood_to_plasma_ratio' must be finite and > 0.",
                **base_kwargs,
            )
        rb = float(ctx.blood_to_plasma_ratio)

    wt = ctx.reference_weight_kg
    if wt is None or not math.isfinite(wt) or wt <= 0:
        return ResolvedPhysiologyContext(
            is_resolvable=False,
            unresolved_reason="A positive finite 'reference_weight_kg' is required for physiological scaling.",
            **base_kwargs,
        )

    w_ratio = wt / ref.adult_reference_weight_kg
    flow_scale = w_ratio ** 0.75

    hepatic_factor = _CHILD_PUGH_QH_FACTOR.get(ctx.child_pugh, 1.0) if ctx.child_pugh else 1.0
    elderly_factor = 0.85 if ctx.population == "elderly" else 1.0
    preg_flow_factor = 1.0
    preg_vol_factor = 1.0
    if ctx.population == "pregnancy" and ctx.gestational_age_weeks is not None:
        ga_frac = min(max(float(ctx.gestational_age_weeks) / 40.0, 0.0), 1.0)
        preg_flow_factor = 1.0 + 0.35 * ga_frac
        preg_vol_factor = 1.0 + 0.40 * ga_frac

    qh = ref.adult_hepatic_blood_flow_l_h * flow_scale * hepatic_factor * elderly_factor * preg_flow_factor * rb
    co = ref.adult_cardiac_output_l_h * flow_scale * elderly_factor * preg_flow_factor * rb

    if ctx.egfr_ml_min_1_73m2 is not None:
        if not math.isfinite(ctx.egfr_ml_min_1_73m2) or ctx.egfr_ml_min_1_73m2 <= 0.0:
            gfr_l_h: float | None = None
            qr_plasma: float | None = None
        else:
            gfr_l_h = float(ctx.egfr_ml_min_1_73m2) * 0.06 * flow_scale
            renal_factor = min(float(ctx.egfr_ml_min_1_73m2) / 125.0, 1.2)
            qr_plasma = ref.adult_renal_plasma_flow_l_h * flow_scale * max(renal_factor, 0.25)
    elif ctx.population == "renal_impairment":
        gfr_l_h = None
        qr_plasma = None
    else:
        gfr_l_h = ref.adult_gfr_l_h * flow_scale * elderly_factor * preg_flow_factor
        qr_plasma = ref.adult_renal_plasma_flow_l_h * flow_scale * elderly_factor * preg_flow_factor

    return ResolvedPhysiologyContext(
        is_resolvable=True,
        weight_kg=wt,
        flow_scale_factor=flow_scale,
        hepatic_blood_flow_l_h=qh,
        cardiac_output_l_h=co,
        gfr_l_h=gfr_l_h,
        renal_plasma_flow_l_h=qr_plasma,
        plasma_volume_l=ref.plasma_volume_l_kg * wt * preg_vol_factor,
        ecf_volume_l=ref.ecf_volume_l_kg * wt * preg_vol_factor,
        tbw_volume_l=ref.tbw_volume_l_kg * wt * preg_vol_factor,
        **base_kwargs,
    )
