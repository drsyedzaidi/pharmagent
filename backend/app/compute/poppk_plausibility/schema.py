"""Immutable typed schemas for PopPK physiological plausibility screening.

All input, catalog, finding, and report models are frozen (`frozen=True`,
`extra="forbid"`) so evaluation is side-effect-free and never mutates caller
inputs.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SeverityLevel = Literal["INFO", "WARN", "HIGH", "CRITICAL"]
FindingStatus = Literal["PASS", "FLAGGED", "UNRESOLVED", "NOT_APPLICABLE"]
OverrideStatus = Literal["none", "justification_required", "accepted_with_justification", "rejected"]
ScreeningStatus = Literal[
    "SCREENING_PASS",
    "REQUIRES_JUSTIFICATION",
    "REVIEW_ACCEPTED_WITH_JUSTIFICATION",
    "CRITICAL_IMPLAUSIBILITY",
    "INCOMPLETE_CONTEXT",
]
PopulationKind = Literal[
    "adult",
    "pediatric",
    "neonatal",
    "pregnancy",
    "renal_impairment",
    "hepatic_impairment",
    "elderly",
    "unknown",
]
RouteKind = Literal["iv", "oral", "sc", "im", "extravascular", "unknown"]
ModalityKind = Literal["small_molecule", "monoclonal_antibody", "peptide_biologic", "other", "unknown"]
ScaleKind = Literal["systemic", "apparent", "auto"]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RuleApproval(_FrozenModel):
    status: Literal["pending_owner_review", "approved"]
    by: str | None = None
    date: str | None = None


class RuleDefinition(_FrozenModel):
    rule_id: str = Field(pattern=r"^PL-\d{3}$")
    slug: str = Field(min_length=1)
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    title: str = Field(min_length=1)
    category: str = Field(min_length=1)
    applies_to_modalities: tuple[str, ...]
    default_severity: SeverityLevel
    override_allowed: bool
    thresholds: dict[str, float | int | bool | str] = Field(default_factory=dict)
    scientific_rationale: str = Field(min_length=1)
    recommendation: str = Field(min_length=1)
    reference_basis: str = Field(min_length=1)
    source_kind: Literal[
        "physiological_reference",
        "mathematical_identity",
        "allometric_theory",
        "modeling_heuristic",
    ]
    approval: RuleApproval


class ReferencePhysiology(_FrozenModel):
    adult_reference_weight_kg: float
    adult_hepatic_blood_flow_l_h: float
    adult_cardiac_output_l_h: float
    adult_gfr_l_h: float
    adult_renal_plasma_flow_l_h: float
    plasma_volume_l_kg: float
    blood_volume_l_kg: float
    ecf_volume_l_kg: float
    tbw_volume_l_kg: float
    min_central_volume_l_kg: float
    max_central_volume_l_kg: float
    max_vss_volume_l_kg: float


class RuleCatalog(_FrozenModel):
    catalog_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    reference_physiology: ReferencePhysiology
    rules: tuple[RuleDefinition, ...]

    def by_id(self, rule_id: str) -> RuleDefinition:
        for rule in self.rules:
            if rule.rule_id == rule_id:
                return rule
        raise KeyError(f"Unknown plausibility rule_id: {rule_id}")


@functools.lru_cache(maxsize=1)
def load_rule_catalog() -> RuleCatalog:
    """Load and validate the static rule catalog JSON."""
    catalog_path = Path(__file__).with_name("catalog.json")
    raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    return RuleCatalog.model_validate(raw)


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
    fraction_unbound_plasma: float | None = None
    blood_to_plasma_ratio: float | None = None
    renal_secretion_known: bool = False
    has_tmdd_or_nonlinear_cl: bool = False


class ParameterSpec(_FrozenModel):
    name: str = Field(min_length=1)
    value: float
    unit: str | None = None
    scale: ScaleKind = "auto"
    fixed: bool = False
    lower_bound: float | None = None
    upper_bound: float | None = None


class NormalizedParameter(_FrozenModel):
    name: str
    canonical_name: str
    param_class: str
    observed_value: float
    observed_unit: str | None
    normalized_value: float | None
    normalized_unit: str | None
    weight_normalized: bool = False
    resolved_scale: Literal["systemic", "apparent", "not_applicable"] = "not_applicable"
    unit_status: Literal["valid", "missing", "incompatible", "unknown"] = "valid"
    unit_message: str = ""


class IIVSpec(_FrozenModel):
    parameter: str = Field(min_length=1)
    omega_var: float | None = None
    cv_pct: float | None = None
    distribution: Literal["lognormal", "normal", "logit"] = "lognormal"
    fixed: bool = False


class ResidualErrorSpec(_FrozenModel):
    kind: Literal["proportional", "additive", "combined", "exponential"] = "proportional"
    sigma_var: float | None = None
    sd_or_cv: float | None = None
    unit: str | None = None


class PredictionPoint(_FrozenModel):
    time: float
    pred: float | None = None
    ipred: float | None = None
    unit: str = "mg/L"
    subject_id: str | None = None


class JustificationRecord(_FrozenModel):
    rule_id: str = Field(pattern=r"^PL-\d{3}$")
    parameter: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    scientific_basis_or_citation: str = Field(min_length=1)
    reviewer: str = Field(min_length=1)
    timestamp: str = Field(min_length=1)

    @field_validator("rationale", "scientific_basis_or_citation", "reviewer", "timestamp")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Justification fields must not be blank.")
        return stripped


class PopPKModelInput(_FrozenModel):
    model_id: str = Field(default="unnamed_model", min_length=1)
    context: PopPKContext = Field(default_factory=PopPKContext)
    parameters: tuple[ParameterSpec, ...] = ()
    allometric_exponents: dict[str, float] = Field(default_factory=dict)
    iiv: tuple[IIVSpec, ...] = ()
    omega_matrix: tuple[tuple[float, ...], ...] | None = None
    sigma_matrix: tuple[tuple[float, ...], ...] | None = None
    correlations: dict[str, float] = Field(default_factory=dict)
    residual_error: tuple[ResidualErrorSpec, ...] = ()
    predictions: tuple[PredictionPoint, ...] = ()
    justifications: tuple[JustificationRecord, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _coerce_collections_without_mutating_caller(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        copied = dict(data)
        for key in ("allometric_exponents", "correlations"):
            if key in copied and isinstance(copied[key], dict):
                copied[key] = dict(copied[key])
        return copied


class PlausibilityFinding(_FrozenModel):
    rule_id: str
    rule_version: str
    category: str
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
