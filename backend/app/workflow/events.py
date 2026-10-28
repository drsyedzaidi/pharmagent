"""Domain-event schema for the workflow. A schema only.

Persistence, hash-chaining, signing and anchoring stay with the existing audit system. This record deliberately has no
chain, sequence, signature or seal fields, so it cannot become a second source of truth; a later phase connects events
to the audit chain.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, field_validator

from .base import (
    Frozen,
    FrozenMap,
    Identifier,
    JsonObject,
    Label,
    NonNegInt,
    Sha256Hex,
    WorkflowContract,
    require_unique,
)
from .stages import WorkflowStage

EventType = Literal["dataset_registered", "transformation_recorded", "execution_started", "execution_finished",
                    "finding_raised", "review_recorded", "stage_transitioned", "approval_invalidated"]


class AuditEntryRef(Frozen):
    """An UNVERIFIED pointer to an entry of the existing audit chain that this event was derived from. The audit entry
    stays the compliance evidence; this only says which one. Nothing here checks that the entry exists, that the hash is
    what the audit chain computes, or that the event content matches the entry; a later phase must resolve and verify it
    against the keyed chain."""
    session_id: Label
    index: NonNegInt
    entry_hash: Sha256Hex


class WorkflowEvent(WorkflowContract):
    """A domain-event record (a schema only; see the module docstring). ``claimed_actor_kind`` and ``actor`` are what
    the producer claimed, not evidence. ``subject_id`` and ``subject_hash`` point at another record but this schema does
    not identify the record type, so a hash from the wrong record family is not detectable at this layer."""
    RECORD_TYPE = "WorkflowEvent"
    event_type: EventType
    stage: WorkflowStage | None = None
    claimed_actor_kind: Literal["human", "agent", "system"]
    actor: Label
    subject_id: Identifier
    subject_hash: Sha256Hex
    detail: JsonObject = Field(default_factory=FrozenMap)
    source_audit_refs: Annotated[tuple[AuditEntryRef, ...], Field(max_length=100)] = ()

    @field_validator("source_audit_refs")
    @classmethod
    def _unique_refs(cls, v: tuple[AuditEntryRef, ...]) -> tuple[AuditEntryRef, ...]:
        require_unique(tuple((r.session_id, r.index) for r in v), "audit references")
        return v
