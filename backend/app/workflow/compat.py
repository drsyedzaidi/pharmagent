"""Read-only projection of a legacy ``PharmState`` (or a serialized session row) onto the workflow contracts.

Pure: it reads the state it is given and nothing else. No clock, no disk, no database, no audit chain. Whatever the
legacy state never recorded (units, dose-record counts, executions, reviews, assumptions, per-dataset versions) is
returned as an explicit ``Unknown`` with the reason, never inferred. Stage evidence reports which result fields are
populated; it says nothing about whether a stage was completed, valid or approved.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from pydantic import TypeAdapter, ValidationError

from app.core.pharmstate import PharmState

from .base import ID_PATTERN, Frozen, Label, MigrationInfo, Unknown, WorkflowContract
from .contracts import AssumptionSet, BlqRepresentation, ColumnSchema, DatasetVersion
from .errors import ContractError
from .hashing import canonical_json, sha256_hex
from .stages import WorkflowStage

PRODUCER = "pharmagent.workflow.compat"
_ID_RE = re.compile(ID_PATTERN)
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
S = WorkflowStage

#: Which legacy PharmState result fields evidence output from which stage. Each field appears under one stage only.
STAGE_EVIDENCE_FIELDS: dict[WorkflowStage, tuple[str, ...]] = {
    S.DATA_CURATION: ("dataset_id", "dataset_metadata", "data_quality"),
    S.EDA: ("spaghetti_data", "flexplot_data"),
    S.NCA: ("nca_parameters", "nca_summary", "nca_plot_data", "be_results", "dose_prop_results",
            "compartmental_results"),
    S.MODEL_DEVELOPMENT: ("pk_model_results", "nlme_results", "scm_results", "poppk_results", "covariate_results",
                          "model_results", "engine_comparison_results", "prior_check_results"),
    S.MODEL_EVALUATION: ("vpc_results", "diagnostics_results", "forest_results", "review_results", "bootstrap_results",
                         "sir_results", "profile_results"),
    S.SIMULATION: ("simulation_results", "dose_sweep_results", "simest_results", "clinsim_results",
                   "exposure_forest_results", "special_pop_results", "individual_exposures", "pediatric_results",
                   "protocol_sim"),
    S.REPORTING: ("report_path", "report_sections", "memo_path", "memo_results", "regulatory_report_path"),
}


class StageEvidence(Frozen):
    stage: WorkflowStage
    fields: tuple[str, ...]


class LegacyProjection(WorkflowContract):
    session_id: Label | Unknown
    dataset: DatasetVersion | Unknown
    environment_id: Label | Unknown
    run_provenance_status: Label | Unknown
    stage_evidence: tuple[StageEvidence, ...]
    executions: Unknown
    reviews: Unknown
    assumptions: AssumptionSet


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, dict, str, set)):
        return len(value) > 0
    return True


_LABEL = TypeAdapter(Label)
MAX_STATE_JSON_CHARS = 10_000_000


def _label_or_unknown(value: Any, reason: str) -> str | Unknown:
    """The value as a Label if it is one; otherwise an explicit Unknown, so one odd legacy value cannot fail the row."""
    try:
        return _LABEL.validate_python(value) if value else _unavailable(reason)
    except ValidationError:
        return _unavailable("legacy value is not representable as a label")


def _legacy_migration(what: str) -> MigrationInfo:
    return MigrationInfo(from_schema_version=0, note=f"projected from legacy PharmState; {what}")


def _unavailable(reason: str) -> Unknown:
    return Unknown(status="unavailable", reason=reason)


def _role_column(roles: Mapping[str, Any], role: str) -> str | Unknown:
    return next((col for col, r in roles.items() if r == role), _unavailable(f"no column has the {role} role"))


def _blq(state: PharmState, roles: Mapping[str, Any]) -> BlqRepresentation | Unknown:
    basis = (state.data_quality or {}).get("blq_basis")
    if basis == "CENS":
        col = _role_column(roles, "CENS")
        if isinstance(col, Unknown):
            return _unavailable("legacy state records a CENS basis but no censoring column")
        return BlqRepresentation(basis="cens_flag", censoring_column=col)
    if basis == "DV<=0":
        return BlqRepresentation(basis="dv_nonpositive")
    return _unavailable("legacy state records no BLQ basis")


def _count(value: Any, what: str) -> int | Unknown:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return _unavailable(f"legacy state does not record {what}")


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA_RE.fullmatch(value))


def _project_dataset(state: PharmState, as_of: datetime) -> DatasetVersion | Unknown:
    if not state.dataset_id:
        return _unavailable("no dataset is loaded in this legacy state")
    meta = state.dataset_metadata or {}
    roles = meta.get("detected_roles") or {}
    digest = meta.get("dataset_sha256")
    path = state.dataset_path
    source = re.split(r"[\\/]", path)[-1] if path else ""
    try:
        return DatasetVersion(
            id=f"dataset:{state.dataset_id}:legacy:{digest[:32] if _valid_digest(digest) else 'nodigest'}",
            dataset_id=state.dataset_id, created_at=as_of,
            producer=PRODUCER, migration=_legacy_migration("dataset versions were not tracked"),
            source=source or _unavailable("legacy state records no dataset path"),
            data_digest=digest if _valid_digest(digest)
            else _unavailable("legacy state records no valid dataset SHA-256"),
            subject_id_column=_role_column(roles, "ID"),
            columns=tuple(ColumnSchema(name=c["name"], role=c.get("role"), dtype=c.get("dtype"),
                                       unit=_unavailable("legacy metadata records no units"))
                          for c in meta.get("columns") or []),
            n_subjects=_count(meta.get("n_subjects"), "the subject count"),
            n_dose_records=_unavailable("legacy metadata does not count dose records"),
            n_observation_records=_count((state.data_quality or {}).get("n_observations"), "the observation count"),
            blq=_blq(state, roles))
    except (ValidationError, KeyError, TypeError, AttributeError) as exc:
        where = (".".join(str(p) for p in exc.errors()[0]["loc"]) if isinstance(exc, ValidationError)
                 else type(exc).__name__)
        return _unavailable(f"legacy dataset metadata is not representable ({where})")


def _environment_id(state: PharmState) -> str | Unknown:
    if not state.run_provenance:
        return _unavailable("legacy state records no run provenance")
    return "env:" + sha256_hex(canonical_json([list(pair) for pair in state.run_provenance]))[:32]


def _projection_id(session_id: str) -> str:
    """Three disjoint namespaces: no session id, a usable id, and a hash of an unusable one."""
    if not session_id:
        return "legacy:none"
    candidate = f"legacy:id:{session_id}"
    if _ID_RE.fullmatch(candidate):
        return candidate
    return "legacy:h:" + sha256_hex(session_id.encode("utf-8", "surrogatepass"))[:32]


def project_pharmstate(state: PharmState, *, as_of: datetime) -> LegacyProjection:
    """Project ``state`` as of ``as_of`` (supplied by the caller: the projection never reads the clock)."""
    pid = _projection_id(state.session_id)
    evidence = tuple(
        StageEvidence(stage=stage, fields=tuple(f for f in fields if _present(getattr(state, f))))
        for stage, fields in STAGE_EVIDENCE_FIELDS.items()
        if any(_present(getattr(state, f)) for f in fields))
    return LegacyProjection(
        id=pid, created_at=as_of, producer=PRODUCER,
        migration=_legacy_migration("this state predates workflow contracts"),
        session_id=_label_or_unknown(state.session_id, "legacy state has no session id"),
        dataset=_project_dataset(state, as_of), environment_id=_environment_id(state),
        run_provenance_status=_label_or_unknown(state.run_provenance_status,
                                                "legacy state records no provenance status"),
        stage_evidence=evidence,
        executions=_unavailable("tool executions are recorded in the audit chain, not in PharmState"),
        reviews=_unavailable("human reviews are recorded in the audit chain and the session's pending review, "
                             "not in PharmState"),
        assumptions=AssumptionSet(id="legacy-assumptions:" + sha256_hex(pid.encode("utf-8", "surrogatepass"))[:32],
                                  created_at=as_of,
                                  producer=PRODUCER,
                                  migration=_legacy_migration("assumptions were not recorded as a set")))


def project_session_row(row: Mapping[str, Any], *, as_of: datetime) -> LegacyProjection:
    """Project a serialized session row as ``SessionStore`` holds it: ``state`` (mapping) or ``state_json`` (text)."""
    if not isinstance(row, Mapping):
        raise ContractError("session row must be a mapping")
    raw = row.get("state")
    if raw is None and row.get("state_json"):
        text = row["state_json"]
        if not isinstance(text, str | bytes):
            raise ContractError("session state_json must be text")
        if len(text) > MAX_STATE_JSON_CHARS:
            raise ContractError("session state_json is too large to read")
        try:
            raw = json.loads(text)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ContractError(f"session state_json is not valid JSON: {type(exc).__name__}") from exc
    if raw is None:
        raise ContractError("session row has no state")
    try:
        state = PharmState.model_validate(raw)
    except (ValidationError, TypeError, RecursionError) as exc:
        raise ContractError(f"session state is not a readable PharmState: {type(exc).__name__}") from exc
    return project_pharmstate(state, as_of=as_of)
