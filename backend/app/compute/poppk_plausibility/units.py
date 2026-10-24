"""Deterministic unit parsing, dimensional compatibility checking, and normalization.

Never guesses ambiguous units or silently coerces incompatible dimensions.
"""
from __future__ import annotations

import math
from typing import Literal

from app.compute.poppk_plausibility.schema import (
    NormalizedParameter,
    ParameterSpec,
    RouteKind,
    sanitize_json_float,
)


class UnitCompatibilityError(ValueError):
    """Raised when a parameter unit is missing, unknown, or dimensionally incompatible."""

    def __init__(self, kind: Literal["missing", "unknown", "incompatible"], message: str) -> None:
        super().__init__(message)
        self.kind = kind


# Canonical parameter name and physical dimension classification
_PARAM_CLASS_MAP: dict[str, tuple[str, str, bool]] = {
    # (canonical_name, param_class, is_apparent_by_name)
    "cl": ("CL", "clearance", False),
    "cl/f": ("CL/F", "clearance", True),
    "cl_f": ("CL/F", "clearance", True),
    "clr": ("CL_R", "clearance", False),
    "cl_r": ("CL_R", "clearance", False),
    "q": ("Q", "clearance", False),
    "q1": ("Q", "clearance", False),
    "q2": ("Q2", "clearance", False),
    "q3": ("Q3", "clearance", False),
    "q/f": ("Q/F", "clearance", True),
    "q_f": ("Q/F", "clearance", True),
    "q2/f": ("Q2/F", "clearance", True),
    "q2_f": ("Q2/F", "clearance", True),
    "vc": ("Vc", "volume", False),
    "v1": ("Vc", "volume", False),
    "v": ("Vc", "volume", False),
    "vc/f": ("Vc/F", "volume", True),
    "vc_f": ("Vc/F", "volume", True),
    "v/f": ("Vc/F", "volume", True),
    "v_f": ("Vc/F", "volume", True),
    "v1/f": ("Vc/F", "volume", True),
    "vp": ("Vp", "volume", False),
    "v2": ("Vp", "volume", False),
    "vp2": ("Vp2", "volume", False),
    "v3": ("Vp2", "volume", False),
    "vp/f": ("Vp/F", "volume", True),
    "v2/f": ("Vp/F", "volume", True),
    "vp2/f": ("Vp2/F", "volume", True),
    "v3/f": ("Vp2/F", "volume", True),
    "vss": ("Vss", "volume", False),
    "vss/f": ("Vss/F", "volume", True),
    "ka": ("ka", "rate_constant", False),
    "k_a": ("ka", "rate_constant", False),
    "ke": ("ke", "rate_constant", False),
    "k_e": ("ke", "rate_constant", False),
    "k10": ("k10", "rate_constant", False),
    "k12": ("k12", "rate_constant", False),
    "k21": ("k21", "rate_constant", False),
    "k13": ("k13", "rate_constant", False),
    "k31": ("k31", "rate_constant", False),
    "t_half": ("t_half", "half_life", False),
    "thalf": ("t_half", "half_life", False),
    "t1/2": ("t_half", "half_life", False),
    "t_half_beta": ("t_half", "half_life", False),
    "thalf_beta": ("t_half", "half_life", False),
    "t_half_terminal": ("t_half", "half_life", False),
    "t_half_eff": ("t_half_eff", "half_life", False),
    "thalf_eff": ("t_half_eff", "half_life", False),
    "effective_t_half": ("t_half_eff", "half_life", False),
    "t_half_effective": ("t_half_eff", "half_life", False),
    "f": ("F", "bioavailability", False),
    "f1": ("F", "bioavailability", False),
    "bioavailability": ("F", "bioavailability", False),
}

# Unit conversion table: raw_unit_lower -> (dimension_family, factor_to_canonical, canonical_unit, weight_normalized)
_UNIT_TABLE: dict[str, tuple[str, float, str, bool]] = {
    # Clearance / flow -> L/h or L/h/kg
    "l/h": ("clearance", 1.0, "L/h", False),
    "l/hr": ("clearance", 1.0, "L/h", False),
    "ml/min": ("clearance", 0.06, "L/h", False),
    "ml/h": ("clearance", 0.001, "L/h", False),
    "ml/hr": ("clearance", 0.001, "L/h", False),
    "l/day": ("clearance", 1.0 / 24.0, "L/h", False),
    "l/d": ("clearance", 1.0 / 24.0, "L/h", False),
    "ml/day": ("clearance", 0.001 / 24.0, "L/h", False),
    "ml/d": ("clearance", 0.001 / 24.0, "L/h", False),
    "l/min": ("clearance", 60.0, "L/h", False),
    "l/h/kg": ("clearance", 1.0, "L/h/kg", True),
    "l/hr/kg": ("clearance", 1.0, "L/h/kg", True),
    "ml/min/kg": ("clearance", 0.06, "L/h/kg", True),
    "ml/h/kg": ("clearance", 0.001, "L/h/kg", True),
    "ml/hr/kg": ("clearance", 0.001, "L/h/kg", True),
    "l/d/kg": ("clearance", 1.0 / 24.0, "L/h/kg", True),
    "l/day/kg": ("clearance", 1.0 / 24.0, "L/h/kg", True),
    "ml/d/kg": ("clearance", 0.001 / 24.0, "L/h/kg", True),
    "ml/day/kg": ("clearance", 0.001 / 24.0, "L/h/kg", True),
    # Volume -> L or L/kg
    "l": ("volume", 1.0, "L", False),
    "ml": ("volume", 0.001, "L", False),
    "dl": ("volume", 0.1, "L", False),
    "l/kg": ("volume", 1.0, "L/kg", True),
    "ml/kg": ("volume", 0.001, "L/kg", True),
    # Rate constant -> 1/h
    "1/h": ("rate_constant", 1.0, "1/h", False),
    "1/hr": ("rate_constant", 1.0, "1/h", False),
    "h^-1": ("rate_constant", 1.0, "1/h", False),
    "hr^-1": ("rate_constant", 1.0, "1/h", False),
    "/h": ("rate_constant", 1.0, "1/h", False),
    "1/min": ("rate_constant", 60.0, "1/h", False),
    "min^-1": ("rate_constant", 60.0, "1/h", False),
    "1/d": ("rate_constant", 1.0 / 24.0, "1/h", False),
    "1/day": ("rate_constant", 1.0 / 24.0, "1/h", False),
    "d^-1": ("rate_constant", 1.0 / 24.0, "1/h", False),
    "day^-1": ("rate_constant", 1.0 / 24.0, "1/h", False),
    # Time / half-life -> h
    "h": ("half_life", 1.0, "h", False),
    "hr": ("half_life", 1.0, "h", False),
    "hour": ("half_life", 1.0, "h", False),
    "hours": ("half_life", 1.0, "h", False),
    "min": ("half_life", 1.0 / 60.0, "h", False),
    "minutes": ("half_life", 1.0 / 60.0, "h", False),
    "d": ("half_life", 24.0, "h", False),
    "day": ("half_life", 24.0, "h", False),
    "days": ("half_life", 24.0, "h", False),
    # Bioavailability / dimensionless -> fraction
    "fraction": ("bioavailability", 1.0, "fraction", False),
    "unitless": ("bioavailability", 1.0, "fraction", False),
    "1": ("bioavailability", 1.0, "fraction", False),
    "-": ("bioavailability", 1.0, "fraction", False),
    "%": ("bioavailability", 0.01, "fraction", False),
    "percent": ("bioavailability", 0.01, "fraction", False),
    "pct": ("bioavailability", 0.01, "fraction", False),
    # Known non-PK-parameter physical units (used to report dimension mismatch rather than unknown unit)
    "mg": ("mass", 1.0, "mg", False),
    "g": ("mass", 1000.0, "mg", False),
    "ug": ("mass", 0.001, "mg", False),
    "kg": ("mass", 1e6, "mg", False),
    "g/l": ("concentration", 1000.0, "mg/L", False),
    "mg/l": ("concentration", 1.0, "mg/L", False),
    "ug/ml": ("concentration", 1.0, "mg/L", False),
    "mcg/ml": ("concentration", 1.0, "mg/L", False),
    "ng/ml": ("concentration", 0.001, "mg/L", False),
    "pg/ml": ("concentration", 1e-6, "mg/L", False),
}


def classify_parameter(param_name: str) -> tuple[str, str, bool]:
    """Return `(canonical_name, param_class, is_apparent_by_name)` for `param_name`."""
    key = param_name.strip().lower()
    if key in _PARAM_CLASS_MAP:
        return _PARAM_CLASS_MAP[key]
    return (param_name.strip(), "other", False)


def normalize_concentration_to_mg_l(value: float, unit: str | None) -> float | None:
    """Convert a concentration value to canonical `mg/L` when unit is recognized; else `None`."""
    if not math.isfinite(value) or unit is None:
        return None
    u_key = unit.strip().lower()
    if u_key in _UNIT_TABLE and _UNIT_TABLE[u_key][0] == "concentration":
        return float(value) * _UNIT_TABLE[u_key][1]
    return None


def normalize_parameter_unit(param_name: str, value: float, unit: str | None) -> tuple[float, str]:
    """Deterministically convert `(value, unit)` for `param_name` into its canonical unit.

    Raises `UnitCompatibilityError` if `unit` is missing, unknown, or incompatible.
    """
    _, param_class, _ = classify_parameter(param_name)
    if unit is None or not unit.strip():
        raise UnitCompatibilityError("missing", f"Parameter '{param_name}' is missing a required unit.")
    u_key = unit.strip().lower()
    if u_key not in _UNIT_TABLE:
        raise UnitCompatibilityError("unknown", f"Unit '{unit}' for parameter '{param_name}' is not recognized.")
    dim_family, factor, canonical_unit, _ = _UNIT_TABLE[u_key]
    if param_class != "other" and dim_family != param_class:
        raise UnitCompatibilityError(
            "incompatible",
            f"Unit '{unit}' (dimension '{dim_family}') is incompatible with parameter '{param_name}' "
            f"(expected dimension '{param_class}').",
        )
    norm_val = float(value) * factor if math.isfinite(value) else float(value)
    return norm_val, canonical_unit


def normalize_parameter_spec(
    spec: ParameterSpec, route: RouteKind = "iv", has_explicit_f: bool = False
) -> NormalizedParameter:
    """Convert a `ParameterSpec` into an immutable `NormalizedParameter` without raising."""
    canonical_name, param_class, apparent_by_name = classify_parameter(spec.name)
    resolved_scale: Literal["systemic", "apparent", "ambiguous", "not_applicable"]
    if param_class in {"clearance", "volume"}:
        if spec.scale == "apparent" or apparent_by_name:
            resolved_scale = "apparent"
            if not canonical_name.endswith("/F"):
                canonical_name = f"{canonical_name}/F"
        elif spec.scale == "systemic":
            resolved_scale = "systemic"
        elif route == "iv" or (route != "unknown" and has_explicit_f):
            resolved_scale = "systemic"
        else:
            resolved_scale = "ambiguous"
    else:
        resolved_scale = "not_applicable"

    obs_val: float | str = sanitize_json_float(spec.value)  # type: ignore[assignment]

    if spec.unit is None or not spec.unit.strip():
        return NormalizedParameter(
            name=spec.name,
            canonical_name=canonical_name,
            param_class=param_class,
            observed_value=obs_val,
            observed_unit=spec.unit,
            normalized_value=None,
            normalized_unit=None,
            weight_normalized=False,
            resolved_scale=resolved_scale,
            unit_status="missing",
            unit_message=f"Parameter '{spec.name}' is missing an explicit unit.",
        )

    u_key = spec.unit.strip().lower()
    if u_key not in _UNIT_TABLE:
        return NormalizedParameter(
            name=spec.name,
            canonical_name=canonical_name,
            param_class=param_class,
            observed_value=obs_val,
            observed_unit=spec.unit,
            normalized_value=None,
            normalized_unit=None,
            weight_normalized=False,
            resolved_scale=resolved_scale,
            unit_status="unknown",
            unit_message=f"Unit '{spec.unit}' for parameter '{spec.name}' is unknown.",
        )

    dim_family, factor, canonical_unit, weight_norm = _UNIT_TABLE[u_key]
    if param_class != "other" and dim_family != param_class:
        return NormalizedParameter(
            name=spec.name,
            canonical_name=canonical_name,
            param_class=param_class,
            observed_value=obs_val,
            observed_unit=spec.unit,
            normalized_value=None,
            normalized_unit=None,
            weight_normalized=False,
            resolved_scale=resolved_scale,
            unit_status="incompatible",
            unit_message=(
                f"Unit '{spec.unit}' (dimension '{dim_family}') is incompatible with "
                f"parameter '{spec.name}' (dimension '{param_class}')."
            ),
        )

    norm_val = float(spec.value) * factor if math.isfinite(spec.value) else None
    return NormalizedParameter(
        name=spec.name,
        canonical_name=canonical_name,
        param_class=param_class,
        observed_value=obs_val,
        observed_unit=spec.unit,
        normalized_value=norm_val,
        normalized_unit=canonical_unit,
        weight_normalized=weight_norm,
        resolved_scale=resolved_scale,
        unit_status="compatible",
        unit_message="Normalized to canonical unit.",
    )
