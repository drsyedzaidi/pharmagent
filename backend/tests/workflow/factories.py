"""Builders for workflow-contract tests: fixed timestamps, no clock reads."""
from __future__ import annotations

from datetime import UTC, datetime

from app.workflow.contracts import (
    ArtifactRef,
    AssumptionSet,
    ColumnSchema,
    DatasetVersion,
    DatasetVersionRef,
    ExecutionRecord,
    Finding,
    ReviewRecord,
    TransformationRecord,
)
from app.workflow.stages import ReviewGate, WorkflowStage

T0 = datetime(2026, 10, 26, 12, 0, 0, tzinfo=UTC)
T1 = datetime(2026, 10, 26, 12, 5, 0, tzinfo=UTC)
PRODUCER = "pharmagent.test"


def artifact(data: bytes = b"payload", kind: str = "output", media_type: str = "text/csv", **kw) -> ArtifactRef:
    return ArtifactRef.from_bytes(data, kind=kind, media_type=media_type, producer=PRODUCER,
                                  created_at=kw.pop("created_at", T0), **kw)


def dataset(version: int | None = 1, parents: tuple[str, ...] = (), **kw) -> DatasetVersion:
    """A dataset version. ``version`` is only the display number; identity is the opaque ``id``."""
    base = dict(
        id=f"dsv:ds_1:{version}", dataset_id="ds_1", display_version=version, created_at=T0, producer=PRODUCER,
        parents=parents, source="oral_pk.csv", data_digest="a" * 64, subject_id_column="ID",
        columns=(ColumnSchema(name="ID", role="ID", dtype="int64", unit="1"),
                 ColumnSchema(name="DV", role="DV", dtype="float64", unit="ng/mL")),
        n_subjects=12, n_dose_records=12, n_observation_records=120)
    base.update(kw)
    return DatasetVersion(**base)


def transformation(inputs=None, outputs=None, **kw) -> TransformationRecord:
    v1 = dataset()
    v2 = dataset(version=2, parents=(v1.id,))
    inputs = inputs if inputs is not None else (DatasetVersionRef.to(v1),)
    outputs = outputs if outputs is not None else (DatasetVersionRef.to(v2),)
    base = dict(id="transform:t1", created_at=T0, producer=PRODUCER, parents=tuple(r.version_id for r in inputs),
                inputs=inputs, outputs=outputs, operation="drop_rows", parameters={"where": "TIME<0"}, actor="alice",
                reason="pre-dose rows are not analysable", affected_rows=3, reversible=True)
    base.update(kw)
    return TransformationRecord(**base)


def execution(**kw) -> ExecutionRecord:
    base = dict(id="exec:e1", created_at=T0, producer=PRODUCER, stage=WorkflowStage.NCA, tool="compute_nca",
                engine="pharmagent", engine_version="0.1.0", environment_id="env:abc", config={"route": "oral"},
                seed=7, status="succeeded", started_at=T0, finished_at=T1)
    base.update(kw)
    return ExecutionRecord(**base)


def finding(**kw) -> Finding:
    base = dict(id="finding:f1", created_at=T0, producer=PRODUCER, stage=WorkflowStage.DATA_CURATION,
                severity="warning", code="blq.high_fraction", message="30% of observations are BLQ",
                source="automated")
    base.update(kw)
    return Finding(**base)


def review(art: ArtifactRef | None = None, **kw) -> ReviewRecord:
    art = art or artifact()
    base = dict(id="review:r1", created_at=T0, producer=PRODUCER, gate=ReviewGate.CLEANED_DATASET,
                decision="approved", actor_label="Dr. Alice Example",
                reason="cleaning verified", bound_artifacts=({"artifact_id": art.id, "digest": art.digest},))
    base.update(kw)
    return ReviewRecord(**base)


def assumptions(**kw) -> AssumptionSet:
    base = dict(id="assume:a1", created_at=T0, producer=PRODUCER)
    base.update(kw)
    return AssumptionSet(**base)
