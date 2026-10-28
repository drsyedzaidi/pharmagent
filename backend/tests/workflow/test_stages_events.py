import json

import pytest
from pydantic import ValidationError

from app.workflow.events import WorkflowEvent
from app.workflow.stages import ALLOWED_ADVANCES, StageTransition, WorkflowStage, is_allowed
from tests.workflow.factories import PRODUCER, T0, finding

S = WorkflowStage


def transition(frm, to, kind="advance", **kw):
    base = dict(id="transition:t1", created_at=T0, producer=PRODUCER, from_stage=frm, to_stage=to, kind=kind,
                actor="alice", reason="ok")
    base.update(kw)
    return StageTransition.create(**base)


def test_there_are_exactly_the_seven_scientific_stages_in_order():
    assert [s.value for s in S] == ["data_curation", "eda", "nca", "model_development", "model_evaluation",
                                    "simulation", "reporting"]


def test_valid_forward_transitions():
    assert transition(None, S.DATA_CURATION)
    assert transition(S.DATA_CURATION, S.EDA)
    assert transition(S.EDA, S.NCA)
    assert transition(S.EDA, S.MODEL_DEVELOPMENT)            # a population analysis may skip NCA
    assert transition(S.NCA, S.REPORTING)                    # NCA-only workflow
    assert transition(S.MODEL_EVALUATION, S.SIMULATION)
    assert transition(S.SIMULATION, S.REPORTING)


@pytest.mark.parametrize("frm,to", [(None, S.EDA), (S.DATA_CURATION, S.NCA), (S.DATA_CURATION, S.MODEL_DEVELOPMENT),
                                    (S.EDA, S.SIMULATION), (S.MODEL_DEVELOPMENT, S.SIMULATION),
                                    (S.NCA, S.SIMULATION), (S.REPORTING, S.DATA_CURATION)])
def test_invalid_advances_are_refused(frm, to):
    with pytest.raises(ValidationError, match="not an allowed"):
        transition(frm, to)


def test_reporting_is_terminal_for_advances():
    assert ALLOWED_ADVANCES[S.REPORTING] == frozenset()
    assert not is_allowed(S.REPORTING, S.REPORTING, "advance")


def test_rollback_goes_strictly_backwards_and_needs_a_reason():
    assert transition(S.MODEL_EVALUATION, S.MODEL_DEVELOPMENT, "rollback", reason="refit with covariate")
    assert transition(S.REPORTING, S.DATA_CURATION, "rollback", reason="data error found")
    with pytest.raises(ValidationError):
        transition(S.EDA, S.NCA, "rollback", reason="x")
    with pytest.raises(ValidationError):
        transition(S.NCA, S.EDA, "rollback", reason="")
    with pytest.raises(ValidationError):
        transition(None, S.EDA, "rollback", reason="x")


def test_rerun_stays_in_the_same_stage():
    assert transition(S.NCA, S.NCA, "rerun", reason="changed lambda_z points")
    with pytest.raises(ValidationError):
        transition(S.NCA, S.EDA, "rerun", reason="x")
    with pytest.raises(ValidationError):
        transition(None, S.DATA_CURATION, "rerun", reason="x")


def test_transition_needs_an_actor_and_valid_kind():
    with pytest.raises(ValidationError):
        transition(S.EDA, S.NCA, actor="")
    with pytest.raises(ValidationError):
        transition(S.EDA, S.NCA, kind="teleport")


# --- WorkflowEvent: a schema only, never a second audit chain --------------------------------------
def test_event_references_a_record_by_id_and_hash():
    f = finding()
    ev = WorkflowEvent.create(id="event:e1", created_at=T0, producer=PRODUCER, event_type="finding_raised", stage=S.DATA_CURATION,
                       claimed_actor_kind="system", actor="qc", subject_id=f.id, subject_hash=f.content_hash)
    assert ev.subject_hash == f.content_hash and ev.detail == {}


def test_event_has_no_chain_signature_or_persistence_fields():
    fields = set(WorkflowEvent.model_fields)
    assert not fields & {"prev_hash", "previous_hash", "index", "sequence", "signature", "seal", "anchor", "entry_hash"}
    payload = json.loads(WorkflowEvent.create(id="event:e1", created_at=T0, producer=PRODUCER, event_type="stage_transitioned",
                                       claimed_actor_kind="human", actor="alice", subject_id="t", subject_hash="a" * 64).canonical_bytes())
    payload["prev_hash"] = "a" * 64
    with pytest.raises(ValidationError):
        WorkflowEvent.model_validate(payload)


def test_event_validation():
    base = dict(id="event:e1", created_at=T0, producer=PRODUCER, event_type="finding_raised", claimed_actor_kind="system",
                actor="qc", subject_id="finding:f1", subject_hash="a" * 64)
    for bad in ({"event_type": "mystery"}, {"claimed_actor_kind": "robot"}, {"actor": ""}, {"subject_hash": "xyz"},
                {"subject_id": "bad id"}, {"detail": {"x": float("nan")}}):
        with pytest.raises(ValidationError):
            WorkflowEvent.create(**{**base, **bad})
    assert WorkflowEvent.create(**base, detail={"k": [1, "v"]}).detail == {"k": [1, "v"]}


def test_there_is_a_persistence_free_module():
    import importlib
    import pkgutil

    import app.workflow as pkg
    for m in pkgutil.iter_modules(pkg.__path__):
        src = importlib.import_module(f"app.workflow.{m.name}")
        names = " ".join(dir(src))
        assert "sqlite3" not in names and "SessionStore" not in names and "AuditChain" not in names


def test_is_allowed_refuses_anything_it_does_not_recognise():
    assert not is_allowed(None, S.DATA_CURATION, "rollback")
    assert not is_allowed(None, S.DATA_CURATION, "rerun")
    assert not is_allowed(S.NCA, S.EDA, "teleport")


def test_rollback_and_rerun_without_any_reason_are_refused():
    with pytest.raises(ValidationError, match="needs a reason"):
        transition(S.NCA, S.EDA, "rollback", reason=None)
    with pytest.raises(ValidationError, match="needs a reason"):
        transition(S.NCA, S.NCA, "rerun", reason=None)
    assert transition(S.EDA, S.NCA, reason=None).reason is None


# --- events retain references to the legacy audit entries they were derived from -------------------
def _event_with(refs):
    return WorkflowEvent.create(id="event:e9", created_at=T0, producer=PRODUCER, event_type="review_recorded", claimed_actor_kind="human",
                         actor="alice", subject_id="review:r1", subject_hash="b" * 64, source_audit_refs=refs)


def test_event_retains_references_to_source_audit_entries():
    from app.workflow.events import AuditEntryRef
    refs = (AuditEntryRef(session_id="sess_1", index=4, entry_hash="c" * 64),
            AuditEntryRef(session_id="sess_1", index=5, entry_hash="d" * 64))
    ev = _event_with(refs)
    assert ev.source_audit_refs == refs and ev.verify_integrity()
    again = WorkflowEvent.model_validate_json(ev.canonical_bytes())
    assert again.source_audit_refs == refs and again == ev


def test_audit_references_are_part_of_the_hash_and_default_to_none():
    from app.workflow.events import AuditEntryRef
    base = _event_with(())
    assert base.source_audit_refs == ()
    other = _event_with((AuditEntryRef(session_id="sess_1", index=1, entry_hash="c" * 64),))
    assert other.content_hash != base.content_hash


def test_audit_reference_validation():
    from app.workflow.events import AuditEntryRef
    for bad in ({"index": -1}, {"index": True}, {"entry_hash": "xyz"}, {"session_id": ""}):
        with pytest.raises(ValidationError):
            AuditEntryRef(**{"session_id": "s", "index": 0, "entry_hash": "c" * 64, **bad})
    with pytest.raises(ValidationError, match="unique"):
        _event_with((AuditEntryRef(session_id="s", index=1, entry_hash="c" * 64),) * 2)


def test_the_event_schema_defines_no_way_to_read_or_write_the_audit_chain():
    import app.workflow.events as ev
    names = " ".join(dir(ev))
    assert "AuditChain" not in names and "audit_seal" not in names


def test_is_allowed_coerces_its_inputs_so_a_lying_stage_cannot_pass():
    class Liar(str):
        def __eq__(self, other):
            return True

        def __hash__(self):
            return hash("nca")                          # collides with a legal target of EDA

    assert not is_allowed(S.EDA, Liar("simulation"), "advance")
    assert is_allowed("eda", "nca", "advance") and not is_allowed("eda", "teleport", "advance")


def test_an_unrecognised_source_stage_is_never_mistaken_for_the_start():
    assert not is_allowed("bogus", "data_curation", "advance")
    assert not is_allowed(object(), S.EDA, "advance")
    assert is_allowed(None, "data_curation", "advance")
