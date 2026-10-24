"""Pydantic schemas and deeply immutable contracts for PopPK physiological plausibility screening."""
from __future__ import annotations

import functools
import json
import math
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    field_validator,
    model_validator,
)

SeverityLevel = Literal["INFO", "WARN", "HIGH", "CRITICAL"]
FindingStatus = Literal["PASS", "FLAGGED", "UNRESOLVED", "NOT_APPLICABLE"]
OverrideStatus = Literal["none", "justification_required", "accepted_with_justification", "rejected"]
ScreeningStatus = Literal[
    "SCREENING_PASS",
    "REVIEW_ACCEPTED_WITH_JUSTIFICATION",
    "REQUIRES_JUSTIFICATION",
    "INCOMPLETE_CONTEXT",
    "CRITICAL_IMPLAUSIBILITY",
]

PopulationKind = Literal[
    "adult",
    "pediatric",
    "neonate",
    "elderly",
    "pregnancy",
    "renal_impairment",
    "hepatic_impairment",
    "unknown",
]
RouteKind = Literal["iv", "oral", "sc", "im", "inhalation", "unknown"]
ModalityKind = Literal["small_molecule", "monoclonal_antibody", "peptide", "biologic_other", "unknown"]
ScaleKind = Literal["systemic", "apparent", "auto"]
EliminationPathwayKind = Literal["hepatic", "renal", "mixed", "extrahepatic", "catabolic", "unknown"]
ClearanceMatrixKind = Literal["blood", "plasma"]
CompartmentalStructureKind = Literal[
    "mammillary_central_elimination", "non_mammillary", "peripheral_elimination", "unknown"
]
FormulationReleaseKind = Literal[
    "immediate_release", "extended_release", "depot_or_long_acting", "unknown"
]
RuleType = Literal["mathematical_identity", "physiological_reference", "modeling_heuristic"]
RuleApprovalStatus = Literal["pending_owner_review"]
RuleCategory = Literal[
    "structural_domain",
    "unit_compatibility",
    "bioavailability",
    "clearance_physiology",
    "volume_physiology",
    "derived_kinetics",
    "absorption_kinetics",
    "allometric_scaling",
    "random_effects",
    "residual_error",
    "biologics_mab",
    "micro_rate_consistency",
    "prediction_plausibility",
]


def _deep_freeze(v: Any) -> Any:
    if isinstance(v, Mapping):
        return MappingProxyType({k: _deep_freeze(val) for k, val in v.items()})
    if isinstance(v, (list, tuple)):
        return tuple(_deep_freeze(x) for x in v)
    return v


def _deep_thaw(v: Any) -> Any:
    if isinstance(v, Mapping):
        return {k: _deep_thaw(val) for k, val in v.items()}
    if isinstance(v, tuple):
        return [_deep_thaw(x) for x in v]
    return v


FrozenMap = Annotated[
    Mapping[str, Any],
    BeforeValidator(dict),
    AfterValidator(_deep_freeze),
    PlainSerializer(_deep_thaw, return_type=dict),
]
FrozenFloatMap = Annotated[
    Mapping[str, float],
    BeforeValidator(dict),
    AfterValidator(_deep_freeze),
    PlainSerializer(_deep_thaw, return_type=dict),
]
FrozenIntMap = Annotated[
    Mapping[str, int],
    BeforeValidator(dict),
    AfterValidator(_deep_freeze),
    PlainSerializer(_deep_thaw, return_type=dict),
]
FrozenStrMap = Annotated[
    Mapping[str, str],
    BeforeValidator(dict),
    AfterValidator(_deep_freeze),
    PlainSerializer(_deep_thaw, return_type=dict),
]


def sanitize_json_float(val: float | str | None) -> float | str | None:
    """Convert non-finite IEEE-754 floats (NaN, +/-Inf) to strings for RFC-8259 JSON safety."""
    if isinstance(val, float) and not math.isfinite(val):
        return str(val)
    return val


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AdultReferencePhysiology(_FrozenModel):
    weight_kg: float
    hepatic_blood_flow_l_h: float
    cardiac_output_l_h: float
    gfr_l_h: float
    renal_plasma_flow_l_h: float
    plasma_volume_l_kg: float
    blood_volume_l_kg: float
    ecf_volume_l_kg: float
    tbw_volume_l_kg: float
    allometric_flow_exponent: float
    allometric_volume_exponent: float


class PopulationScalingConfig(_FrozenModel):
    neonate_maturation_cl_factor: float
    elderly_flow_factor_per_decade_over_65: float
    child_pugh_flow_factors: FrozenFloatMap
    pregnancy_gfr_factor: float
    pregnancy_hepatic_flow_factor: float
    pregnancy_plasma_volume_factor: float


class ReferencePhysiology(_FrozenModel):
    adult_70kg: AdultReferencePhysiology
    special_population_rules_status: FrozenStrMap = Field(
        default_factory=lambda: MappingProxyType(
            {
                "pediatric": "pending_owner_review",
                "neonate": "pending_owner_review",
                "elderly": "pending_owner_review",
                "pregnancy": "pending_owner_review",
                "renal_impairment": "pending_owner_review",
                "hepatic_impairment": "pending_owner_review",
            }
        )
    )
    population_scaling: PopulationScalingConfig


class RuleDefinition(_FrozenModel):
    rule_id: str
    version: str
    category: RuleCategory
    rule_type: RuleType = "modeling_heuristic"
    approval_status: RuleApprovalStatus = "pending_owner_review"
    description: str
    scientific_rationale: str
    recommendation: str
    reference_basis: str
    default_severity: SeverityLevel
    override_allowed: bool
    applicable_modalities: tuple[str, ...]
    thresholds: FrozenMap = Field(default_factory=lambda: MappingProxyType({}))

    @property
    def title(self) -> str:
        return self.description

    def threshold_float(self, key: str, default: float) -> float:
        raw = self.thresholds.get(key, default)
        return float(raw)


class RuleCatalog(_FrozenModel):
    catalog_version: str
    reference_physiology: ReferencePhysiology
    rules: tuple[RuleDefinition, ...]

    def by_id(self, rule_id: str) -> RuleDefinition:
        for r in self.rules:
            if r.rule_id == rule_id:
                return r
        raise KeyError(f"Unknown plausibility rule_id: {rule_id!r}")


@functools.lru_cache(maxsize=1)
def _cached_catalog_json() -> str:
    catalog_path = Path(__file__).with_name("catalog.json")
    return catalog_path.read_text(encoding="utf-8")


def load_rule_catalog() -> RuleCatalog:
    """Load and validate a deeply immutable RuleCatalog instance from static JSON."""
    return RuleCatalog.model_validate(json.loads(_cached_catalog_json()))


class PopPKContext(_FrozenModel):
    species: str = "human"
    population: PopulationKind = "adult"
    reference_weight_kg: float | None = 70.0
    age_years: float | None = None
    gestational_age_weeks: float | None = None
    egfr_ml_min_1_73m2: float | None = None
    child_pugh: Literal["A", "B", "C"] | None = None
    route: RouteKind = "iv"
    modality: ModalityKind = "small_molecule"
    elimination_pathway: EliminationPathwayKind = "hepatic"
    clearance_matrix: ClearanceMatrixKind = "blood"
    compartmental_structure: CompartmentalStructureKind = "mammillary_central_elimination"
    formulation_release: FormulationReleaseKind = "immediate_release"
    fraction_unbound_plasma: float | None = None
    blood_to_plasma_ratio: float | None = None
    renal_secretion_known: bool = False
    has_tmdd_or_nonlinear_cl: bool = False


class ParameterSpec(_FrozenModel):
    name: str = Field(min_length=1, max_length=128)
    value: float
    unit: str | None = Field(default=None, max_length=64)
    scale: ScaleKind = "auto"
    fixed: bool = False
    lower_bound: float | None = None
    upper_bound: float | None = None


class NormalizedParameter(_FrozenModel):
    name: str
    canonical_name: str
    param_class: str
    observed_value: float | str
    observed_unit: str | None
    normalized_value: float | None
    normalized_unit: str | None
    weight_normalized: bool = False
    resolved_scale: Literal["systemic", "apparent", "ambiguous", "not_applicable"] = "not_applicable"
    unit_status: Literal["compatible", "missing", "incompatible", "unknown"] = "compatible"
    unit_message: str = ""

    @field_validator("observed_value", mode="before")
    @classmethod
    def _clean_obs(cls, v: Any) -> Any:
        return sanitize_json_float(v)

    @field_validator("normalized_value", mode="before")
    @classmethod
    def _clean_norm(cls, v: Any) -> Any:
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v


class IIVSpec(_FrozenModel):
    parameter: str = Field(min_length=1, max_length=128)
    omega_var: float | None = None
    cv_pct: float | None = None
    distribution: Literal["lognormal", "proportional", "normal", "logit"] = "lognormal"
    fixed: bool = False


class ResidualErrorSpec(_FrozenModel):
    kind: Literal["proportional", "additive", "combined", "exponential"] = "proportional"
    sigma_var: float | None = None
    sd_or_cv: float | None = None
    unit: str | None = Field(default=None, max_length=64)


class PredictionPoint(_FrozenModel):
    time: float
    pred: float | None = None
    ipred: float | None = None
    unit: str = Field(default="mg/L", max_length=64)


class JustificationRecord(_FrozenModel):
    rule_id: str = Field(min_length=1, max_length=32)
    parameter: str = Field(min_length=1, max_length=128)
    rationale: str = Field(min_length=1, max_length=2000)
    scientific_basis_or_citation: str = Field(min_length=1, max_length=2000)
    reviewer: str = Field(min_length=1, max_length=256)
    timestamp: str = Field(min_length=1, max_length=128)

    @field_validator(
        "rule_id",
        "parameter",
        "rationale",
        "scientific_basis_or_citation",
        "reviewer",
        "timestamp",
    )
    @classmethod
    def _must_not_be_whitespace_only(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Justification fields must not be blank or whitespace-only.")
        return cleaned


class PlausibilityFinding(_FrozenModel):
    rule_id: str
    rule_version: str
    category: RuleCategory
    rule_type: RuleType = "modeling_heuristic"
    approval_status: RuleApprovalStatus = "pending_owner_review"
    parameter: str
    observed_value: float | str | None = None
    observed_unit: str | None = None
    normalized_value: float | None = None
    normalized_unit: str | None = None
    expected_range: str = ""
    severity: SeverityLevel
    status: FindingStatus
    scientific_rationale: str
    recommendation: str
    reference_basis: str
    override_allowed: bool
    override_status: OverrideStatus = "none"
    justification: JustificationRecord | None = None
    override_note: str = ""

    @field_validator("observed_value", mode="before")
    @classmethod
    def _clean_obs_val(cls, v: Any) -> Any:
        return sanitize_json_float(v)

    @field_validator("normalized_value", mode="before")
    @classmethod
    def _clean_norm_val(cls, v: Any) -> Any:
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v


class PopPKModelInput(_FrozenModel):
    model_id: str = Field(default="unnamed_poppk_model", min_length=1, max_length=256)
    context: PopPKContext = Field(default_factory=PopPKContext)
    parameters: tuple[ParameterSpec, ...] = Field(default=(), max_length=128)
    allometric_exponents: FrozenFloatMap = Field(default_factory=lambda: MappingProxyType({}))
    iiv: tuple[IIVSpec, ...] = Field(default=(), max_length=128)
    omega_matrix: tuple[tuple[float, ...], ...] | None = Field(default=None, max_length=64)
    sigma_matrix: tuple[tuple[float, ...], ...] | None = Field(default=None, max_length=64)
    correlations: FrozenFloatMap = Field(default_factory=lambda: MappingProxyType({}))
    residual_error: tuple[ResidualErrorSpec, ...] = Field(default=(), max_length=32)
    predictions: tuple[PredictionPoint, ...] = Field(default=(), max_length=2000)
    justifications: tuple[JustificationRecord, ...] = Field(default=(), max_length=128)

    @model_validator(mode="before")
    @classmethod
    def _coerce_collections_without_mutating_caller(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        copied = dict(data)
        for k in ("allometric_exponents", "correlations"):
            if k in copied and isinstance(copied[k], Mapping):
                if len(copied[k]) > 128:
                    raise ValueError(f"{k} exceeds maximum length of 128.")
                copied[k] = dict(copied[k])
        return copied
