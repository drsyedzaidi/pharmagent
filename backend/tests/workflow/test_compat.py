"""The legacy PharmState projection: read-only, side-effect free, never inventing what the old state never held."""
import builtins
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.pharmstate import PharmState
from app.workflow.base import Unknown
from app.workflow.compat import STAGE_EVIDENCE_FIELDS, LegacyProjection, project_pharmstate, project_session_row
from app.workflow.errors import ContractError
from app.workflow.stages import WorkflowStage
from tests.workflow.factories import T0

FIX = Path(__file__).parent / "fixtures"


def load(name: str) -> dict:
    return json.loads((FIX / name).read_text())


def state_of(name: str) -> PharmState:
    return PharmState.model_validate(load(name)["state"])


def evidence(p: LegacyProjection) -> dict:
    return {e.stage: set(e.fields) for e in p.stage_evidence}


def test_full_session_projects_what_the_state_actually_holds():
    st = state_of("legacy_session_full.json")
    p = project_pharmstate(st, as_of=T0)
    d = p.dataset
    assert d.dataset_id == "ds_fixture" and d.display_version is None and d.source == "oral_pk.csv"
    assert d.data_digest == st.dataset_metadata["dataset_sha256"] and d.n_subjects == 12
    assert {c.name for c in d.columns} >= {"ID", "TIME", "DV"} and d.subject_id_column == "ID"
    assert all(isinstance(c.unit, Unknown) and c.unit.status == "unavailable" for c in d.columns)
    assert p.environment_id.startswith("env:") and p.run_provenance_status == st.run_provenance_status
    assert evidence(p)[WorkflowStage.NCA] >= {"nca_parameters", "nca_summary"}
    assert WorkflowStage.DATA_CURATION in evidence(p)


def test_pre_provenance_session_marks_every_missing_item_explicitly():
    p = project_pharmstate(state_of("legacy_session_pre_provenance.json"), as_of=T0)
    assert isinstance(p.dataset.data_digest, Unknown) and p.dataset.data_digest.status == "unavailable"
    assert isinstance(p.environment_id, Unknown) and isinstance(p.run_provenance_status, Unknown)
    assert WorkflowStage.NCA not in evidence(p)
    assert isinstance(p.dataset.blq, Unknown) and isinstance(p.dataset.n_observation_records, Unknown)


def test_empty_session_has_no_dataset_and_no_stage_evidence():
    p = project_pharmstate(state_of("legacy_session_empty.json"), as_of=T0)
    assert isinstance(p.dataset, Unknown) and p.stage_evidence == ()


def test_executions_reviews_and_assumptions_are_never_reconstructed_from_state():
    p = project_pharmstate(state_of("legacy_session_full.json"), as_of=T0)
    assert isinstance(p.executions, Unknown) and "audit" in p.executions.reason
    assert isinstance(p.reviews, Unknown)
    for name in ("route", "blq_policy", "exclusion_policy", "bounds", "initial_estimates"):
        assert getattr(p.assumptions, name).status == "unknown"


def test_blq_representation_follows_the_recorded_basis_only():
    st = state_of("legacy_session_full.json")
    dq = {**(st.data_quality or {}), "blq_basis": "CENS"}
    meta = {**st.dataset_metadata, "detected_roles": {**st.dataset_metadata["detected_roles"], "CENS": "CENS"}}
    p = project_pharmstate(st.model_copy(update={"data_quality": dq, "dataset_metadata": meta}), as_of=T0)
    assert p.dataset.blq.basis == "cens_flag" and p.dataset.blq.censoring_column == "CENS"
    p2 = project_pharmstate(st.model_copy(update={"data_quality": {**dq, "blq_basis": "DV<=0"}}), as_of=T0)
    assert p2.dataset.blq.basis == "dv_nonpositive"
    p3 = project_pharmstate(st.model_copy(update={"data_quality": {**dq, "blq_basis": None}}), as_of=T0)
    assert isinstance(p3.dataset.blq, Unknown)


def test_projection_does_not_mutate_the_state_or_touch_the_disk(monkeypatch):
    st = state_of("legacy_session_full.json")
    before = st.model_dump_json()

    def forbidden(*a, **k):
        raise AssertionError("projection must not read files")
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "stat", forbidden)
    project_pharmstate(st, as_of=T0)
    assert st.model_dump_json() == before


def test_projection_is_deterministic_and_round_trips():
    st = state_of("legacy_session_full.json")
    a, b = project_pharmstate(st, as_of=T0), project_pharmstate(st, as_of=T0)
    assert a == b and a.content_hash == b.content_hash and a.canonical_bytes() == b.canonical_bytes()
    assert LegacyProjection.model_validate_json(a.canonical_bytes()) == a
    assert project_pharmstate(st, as_of=T0.replace(minute=1)).content_hash != a.content_hash


def test_projection_declares_itself_as_a_migration_from_unversioned_state():
    p = project_pharmstate(state_of("legacy_session_full.json"), as_of=T0)
    assert p.migration is not None and p.migration.from_schema_version == 0
    assert p.dataset.migration is not None and p.producer == "pharmagent.workflow.compat"


def test_stage_evidence_is_evidence_not_completion_or_approval():
    p = project_pharmstate(state_of("legacy_session_full.json"), as_of=T0)
    for e in p.stage_evidence:
        assert set(type(e).model_fields) == {"stage", "fields"}


def test_every_evidence_field_is_a_real_pharmstate_field():
    real = set(PharmState.model_fields)
    mapped = [f for fields in STAGE_EVIDENCE_FIELDS.values() for f in fields]
    assert not set(mapped) - real
    assert len(mapped) == len(set(mapped)), "a state field may evidence only one stage"


def test_empty_containers_do_not_count_as_evidence():
    st = PharmState(session_id="s1", nca_parameters=[], nca_summary={}, widgets=[])
    assert project_pharmstate(st, as_of=T0).stage_evidence == ()


def test_unrepresentable_legacy_values_become_unknown_not_errors():
    base = state_of("legacy_session_full.json")
    bad_id = base.model_copy(update={"dataset_id": "has space!", "dataset_metadata": {**base.dataset_metadata, "dataset_id": "has space!"}})
    assert isinstance(project_pharmstate(bad_id, as_of=T0).dataset, Unknown)
    cols = [{"name": "bad\x00name", "dtype": "float64", "role": "DV"}]
    bad_col = base.model_copy(update={"dataset_metadata": {**base.dataset_metadata, "columns": cols}})
    p = project_pharmstate(bad_col, as_of=T0)
    assert isinstance(p.dataset, Unknown) and p.dataset.status == "unavailable"
    odd = base.model_copy(update={"session_id": "weird id / with spaces"})
    assert project_pharmstate(odd, as_of=T0).id.startswith("legacy:")


def test_malformed_digest_is_unavailable_not_trusted():
    base = state_of("legacy_session_full.json")
    st = base.model_copy(update={"dataset_metadata": {**base.dataset_metadata, "dataset_sha256": "n/a"}})
    assert project_pharmstate(st, as_of=T0).dataset.data_digest.status == "unavailable"


def test_serialized_session_rows_both_shapes():
    row = load("legacy_session_full.json")
    assert project_session_row(row, as_of=T0) == project_pharmstate(state_of("legacy_session_full.json"), as_of=T0)
    as_json = {**row, "state_json": json.dumps(row["state"])}
    del as_json["state"]
    assert project_session_row(as_json, as_of=T0).dataset.dataset_id == "ds_fixture"
    before = json.dumps(row, sort_keys=True)
    project_session_row(row, as_of=T0)
    assert json.dumps(row, sort_keys=True) == before


def test_serialized_session_rows_that_cannot_be_read_are_refused():
    for bad in ({}, {"state": None}, {"state_json": "{not json"}, {"state": {"nca_parameters": "wrong type"}}):
        with pytest.raises((ContractError, ValidationError)):
            project_session_row(bad, as_of=T0)


def test_naive_as_of_is_rejected():
    with pytest.raises(ValidationError):
        project_pharmstate(PharmState(session_id="s"), as_of=T0.replace(tzinfo=None))


def test_a_cens_basis_without_a_censoring_column_is_unavailable_not_guessed():
    st = state_of("legacy_session_full.json")
    roles = {k: v for k, v in st.dataset_metadata["detected_roles"].items() if v != "CENS"}
    p = project_pharmstate(st.model_copy(update={
        "data_quality": {**(st.data_quality or {}), "blq_basis": "CENS"},
        "dataset_metadata": {**st.dataset_metadata, "detected_roles": roles}}), as_of=T0)
    assert isinstance(p.dataset.blq, Unknown) and "CENS" in p.dataset.blq.reason


def test_scalar_values_count_as_present_evidence():
    from app.workflow.compat import _present
    assert _present(0) and _present(1.5) and _present(False) and _present("x")
    assert not _present(None) and not _present("") and not _present([]) and not _present({})


def test_environment_id_is_derived_from_the_recorded_provenance_only():
    import hashlib
    st = state_of("legacy_session_full.json")
    pairs = [list(p) for p in st.run_provenance]
    expected = "env:" + hashlib.sha256(json.dumps(pairs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:32]
    assert project_pharmstate(st, as_of=T0).environment_id == expected
    other = st.model_copy(update={"run_provenance": (*st.run_provenance, ("extra", "1"))})
    assert project_pharmstate(other, as_of=T0).environment_id != expected


def test_unrepresentable_session_ids_do_not_break_the_projection():
    for sid in ("bad\x00id", "x" * 300):
        p = project_pharmstate(PharmState(session_id=sid), as_of=T0)
        assert isinstance(p.session_id, Unknown) and p.id.startswith("legacy:")


def test_garbage_session_state_raises_the_contract_error():
    for bad in ({"state": "garbage"}, {"state": [1, 2]}, {"state_json": "[]"}):
        with pytest.raises(ContractError):
            project_session_row(bad, as_of=T0)


def test_hostile_serialized_state_is_refused_with_the_contract_error():
    for bad in ({"state_json": "[" * 100_000}, {"state_json": "x" * 10_000_001}, {"state_json": 5}):
        with pytest.raises(ContractError):
            project_session_row(bad, as_of=T0)


def test_odd_legacy_labels_degrade_to_unknown_instead_of_failing_the_row():
    st = PharmState(session_id="s1", run_provenance_status="ver\nified")
    p = project_pharmstate(st, as_of=T0)
    assert isinstance(p.run_provenance_status, Unknown) and p.session_id == "s1"
    leading = PharmState(session_id=" padded ")
    assert isinstance(project_pharmstate(leading, as_of=T0).session_id, Unknown)


def test_state_json_over_the_size_cap_is_refused_even_when_it_is_valid(monkeypatch):
    import app.workflow.compat as compat
    row = load("legacy_session_empty.json")
    text = json.dumps(row["state"])
    assert project_session_row({"state_json": text}, as_of=T0)          # readable under the real cap
    monkeypatch.setattr(compat, "MAX_STATE_JSON_CHARS", 100)
    with pytest.raises(ContractError, match="too large"):
        project_session_row({"state_json": text}, as_of=T0)


def test_a_session_row_that_is_not_a_mapping_is_refused_with_the_contract_error():
    for bad in (None, [], "text", 5):
        with pytest.raises(ContractError):
            project_session_row(bad, as_of=T0)


def test_a_lone_surrogate_in_a_session_id_does_not_crash_the_projection():
    p = project_session_row({"state_json": '{"session_id": "\\ud800x"}'}, as_of=T0)
    assert p.id.startswith("legacy:") and p.verify_integrity()
    q = project_session_row({"state_json": '{"session_id": "\\ud800x"}'}, as_of=T0)
    assert p.id == q.id                                                        # deterministic


def test_a_state_with_no_session_id_gets_a_fixed_placeholder_id_and_an_explicit_unknown():
    p = project_pharmstate(PharmState(), as_of=T0)
    assert p.id == "legacy:none" and isinstance(p.session_id, Unknown)


def test_projection_ids_come_from_disjoint_namespaces():
    import hashlib
    empty = project_pharmstate(PharmState(), as_of=T0).id
    literal = project_pharmstate(PharmState(session_id="none"), as_of=T0).id
    hashed_like = project_pharmstate(PharmState(session_id=hashlib.sha256(b"\x00").hexdigest()[:16]), as_of=T0).id
    odd = project_pharmstate(PharmState(session_id="bad\x00id"), as_of=T0).id
    assert len({empty, literal, hashed_like, odd}) == 4
    assert literal == "legacy:id:none" and odd.startswith("legacy:h:") and empty == "legacy:none"
