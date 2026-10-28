"""Workflow-stage domain contracts (Phase A of the workflow reorganization).

Additive and side-effect free: versioned, immutable, content-hashed records plus a read-only projection of legacy
``PharmState``. Nothing here executes science, persists, signs or routes; the existing orchestrator and audit system
remain authoritative.
"""
from .base import SCHEMA_VERSION, FrozenMap, Unknown, WorkflowContract
from .contracts import (
    ArtifactBinding,
    ArtifactRef,
    AssumptionSet,
    BlqRepresentation,
    Bounds,
    ColumnSchema,
    DatasetVersion,
    DatasetVersionRef,
    ExecutionRecord,
    Finding,
    ReviewRecord,
    TransformationRecord,
)
from .errors import ContractError, safe_errors
from .events import AuditEntryRef, WorkflowEvent
from .lineage import check_lineage
from .stages import ALLOWED_ADVANCES, GATE_STAGE, ReviewGate, StageTransition, WorkflowStage, is_allowed

__all__ = [
    "ALLOWED_ADVANCES", "ArtifactBinding", "ArtifactRef", "AssumptionSet", "AuditEntryRef", "BlqRepresentation",
    "Bounds", "ColumnSchema", "ContractError", "DatasetVersion", "DatasetVersionRef", "ExecutionRecord", "Finding",
    "FrozenMap", "GATE_STAGE", "ReviewGate", "ReviewRecord", "SCHEMA_VERSION", "StageTransition",
    "TransformationRecord", "Unknown", "WorkflowContract", "WorkflowEvent", "WorkflowStage", "check_lineage",
    "is_allowed", "safe_errors",
]
