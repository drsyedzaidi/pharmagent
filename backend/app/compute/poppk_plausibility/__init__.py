"""PopPK Physiological Plausibility Checker package.

Provides deterministic, context-aware physiological and mathematical screening
of population pharmacokinetic (PopPK) model parameters with non-destructive
human justification governance.
"""
from __future__ import annotations

from app.compute.poppk_plausibility.benchmark import (
    BenchmarkCase,
    BenchmarkCaseResult,
    BenchmarkSummary,
    build_benchmark_cases,
    run_plausibility_benchmark,
)
from app.compute.poppk_plausibility.context import (
    ResolvedPhysiologyContext,
    resolve_physiology_context,
)
from app.compute.poppk_plausibility.evaluator import (
    apply_justifications,
    evaluate_poppk_plausibility,
)
from app.compute.poppk_plausibility.report import (
    MANDATORY_DISCLAIMER,
    PlausibilityReport,
    build_plausibility_report,
)
from app.compute.poppk_plausibility.rules import evaluate_all_rules
from app.compute.poppk_plausibility.schema import (
    ClearanceMatrixKind,
    CompartmentalStructureKind,
    EliminationPathwayKind,
    FindingStatus,
    FormulationReleaseKind,
    IIVSpec,
    JustificationRecord,
    ModalityKind,
    NormalizedParameter,
    OverrideStatus,
    ParameterSpec,
    PlausibilityFinding,
    PopPKContext,
    PopPKModelInput,
    PopulationKind,
    PredictionPoint,
    ReferencePhysiology,
    ResidualErrorSpec,
    RouteKind,
    RuleApprovalStatus,
    RuleCatalog,
    RuleDefinition,
    RuleType,
    ScaleKind,
    ScreeningStatus,
    SeverityLevel,
    load_rule_catalog,
)
from app.compute.poppk_plausibility.units import (
    UnitCompatibilityError,
    classify_parameter,
    normalize_parameter_spec,
    normalize_parameter_unit,
)

__all__ = [
    "BenchmarkCase",
    "BenchmarkCaseResult",
    "BenchmarkSummary",
    "ClearanceMatrixKind",
    "CompartmentalStructureKind",
    "EliminationPathwayKind",
    "FindingStatus",
    "FormulationReleaseKind",
    "IIVSpec",
    "JustificationRecord",
    "MANDATORY_DISCLAIMER",
    "ModalityKind",
    "NormalizedParameter",
    "OverrideStatus",
    "ParameterSpec",
    "PlausibilityFinding",
    "PlausibilityReport",
    "PopPKContext",
    "PopPKModelInput",
    "PopulationKind",
    "PredictionPoint",
    "ReferencePhysiology",
    "ResidualErrorSpec",
    "ResolvedPhysiologyContext",
    "RouteKind",
    "RuleApprovalStatus",
    "RuleCatalog",
    "RuleDefinition",
    "RuleType",
    "ScaleKind",
    "ScreeningStatus",
    "SeverityLevel",
    "UnitCompatibilityError",
    "apply_justifications",
    "build_benchmark_cases",
    "build_plausibility_report",
    "classify_parameter",
    "evaluate_all_rules",
    "evaluate_poppk_plausibility",
    "load_rule_catalog",
    "normalize_parameter_spec",
    "normalize_parameter_unit",
    "resolve_physiology_context",
    "run_plausibility_benchmark",
]
