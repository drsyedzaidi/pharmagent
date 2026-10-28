"""Phase A hardening: loading vs constructing, nested revalidation, UTF-8, hash domains, ids, strictness, labels."""
import hashlib
import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.core.pharmstate import PharmState
from app.workflow.base import FrozenMap
from app.workflow.compat import project_pharmstate
from app.workflow.contracts import (
    ArtifactBinding,
    BlqRepresentation,
    ColumnSchema,
    DatasetVersionRef,
    ReviewRecord,
)
from app.workflow.events import AuditEntryRef, WorkflowEvent
from tests.workflow.factories import (
    T0,
    artifact,
    assumptions,
    dataset,
    execution,
    finding,
    review,
    transformation,
)


def families():
    """One populated instance of every persisted record family."""
    from app.workflow.stages import StageTransition, WorkflowStage
    art = artifact(b"payload")
    ev = WorkflowEvent(id="event:e1", created_at=T0, producer="p", event_type="finding_raised", actor_kind="system",
                       actor="qc", subject_id="finding:f1", subject_hash="a" * 64)
    st = StageTransition(id="transition:t1", created_at=T0, producer="p", from_stage=None,
                         to_stage=WorkflowStage.DATA_CURATION, kind="advance", actor="alice")
    return [art, dataset(), transformation(), assumptions(), execution(), finding(), review(art), st, ev]


FAMILY_IDS = ["ArtifactRef", "DatasetVersion", "TransformationRecord", "AssumptionSet", "ExecutionRecord", "Finding",
              "ReviewRecord", "StageTransition", "WorkflowEvent"]


# --- 1. loading persisted records is not constructing fresh ones ----------------------------------------------------
@pytest.mark.parametrize("rec", families(), ids=FAMILY_IDS)
def test_load_round_trips_every_family_from_json_text_bytes_and_mappings(rec):
    raw = rec.canonical_bytes()
    for source in (raw, raw.decode(), json.loads(raw)):
        loaded = type(rec).load(source)
        assert loaded == rec and loaded.content_hash == rec.content_hash


@pytest.mark.parametrize("rec", families(), ids=FAMILY_IDS)
def test_a_stored_record_with_a_blank_or_missing_hash_is_refused_not_resealed(rec):
    data = json.loads(rec.canonical_bytes())
    for broken in ({**data, "content_hash": ""}, {k: v for k, v in data.items() if k != "content_hash"}):
        with pytest.raises(ValidationError, match="content_hash"):
            type(rec).load(broken)
        with pytest.raises(ValidationError, match="content_hash"):
            type(rec).load(json.dumps(broken))


def test_tampering_with_a_stored_review_and_stripping_its_hash_cannot_launder_an_approval():
    art, other = artifact(b"cleaned"), artifact(b"swapped")
    data = json.loads(review(art).canonical_bytes())
    data["bound_artifacts"] = [{"artifact_id": other.id, "digest": other.digest}]
    data.pop("content_hash")
    with pytest.raises(ValidationError, match="content_hash"):
        ReviewRecord.load(data)
    data["content_hash"] = review(art).content_hash            # a stale hash is refused too
    with pytest.raises(ValidationError, match="content_hash"):
        ReviewRecord.load(data)


def test_fresh_construction_still_seals_automatically():
    assert finding().content_hash and finding().verify_integrity()


def test_load_refuses_unsupported_shapes_and_unknown_fields():
    data = json.loads(finding().canonical_bytes())
    for bad in ({**data, "extra": 1}, {**data, "schema_version": 2}):
        with pytest.raises(ValidationError):
            type(finding()).load(bad)


# --- 2. nested instances are revalidated ------------------------------------------------------------------------------
def test_forged_nested_instances_are_refused_at_construction():
    art = artifact(b"x")
    forged_binding = ArtifactBinding.model_construct(artifact_id="whatever", digest=art.digest)
    with pytest.raises(ValidationError):
        review(art, bound_artifacts=(forged_binding,))
    with pytest.raises(ValidationError):
        dataset(columns=(ColumnSchema.model_construct(name="", role=None, dtype=None, unit="mg"),))
    with pytest.raises(ValidationError):
        dataset(blq=BlqRepresentation.model_construct(basis="cens_flag", censoring_column=None, lloq=None))
    v1 = dataset()
    with pytest.raises(ValidationError):
        transformation(inputs=(DatasetVersionRef.model_construct(version_id="bad id", record_hash="x"),),
                       outputs=(DatasetVersionRef.to(dataset(version=2, parents=(v1.id,))),), parents=("bad id",))
    with pytest.raises(ValidationError):
        WorkflowEvent(id="event:e9", created_at=T0, producer="p", event_type="finding_raised", actor_kind="system",
                      actor="qc", subject_id="finding:f1", subject_hash="a" * 64,
                      source_audit_refs=(AuditEntryRef.model_construct(session_id="s", index=-1, entry_hash="zz"),))


def test_nested_forgeries_do_not_survive_model_copy_either():
    art = artifact(b"x")
    rec = review(art)
    forged = ArtifactBinding.model_construct(artifact_id="whatever", digest=art.digest)
    with pytest.raises(ValidationError):
        rec.model_copy(update={"bound_artifacts": (forged,), "content_hash": ""})


# --- 3. only well-formed UTF-8 text is accepted anywhere in JSON content --------------------------------------------
@pytest.mark.parametrize("bad", [chr(0xD800), "x" + chr(0xDC00) + "y", chr(0xDBFF)])
def test_lone_surrogates_are_rejected_in_json_values_keys_and_nested_content(bad):
    for params in ({"k": bad}, {bad: 1}, {"a": [bad]}, {"a": {"b": [{"c": bad}]}}, {"a": [[1, [bad]]]}):
        with pytest.raises(ValidationError, match="UTF-8|surrogate|unicode"):
            transformation(parameters=params)
        with pytest.raises(ValidationError, match="UTF-8|surrogate|unicode"):
            execution(config=params)


def test_valid_unicode_including_astral_characters_is_still_accepted():
    params = {"naïve": ["µg/L", "\U0001f9ea", " -is-fine-in-json-values"]}
    assert transformation(parameters=params).verify_integrity()


def test_free_text_reasons_and_notes_also_refuse_surrogates():
    from app.workflow.base import MigrationInfo, Unknown
    with pytest.raises(ValidationError, match="UTF-8|surrogate|unicode"):
        Unknown(status="unknown", reason="bad " + chr(0xD800) + " text")
    with pytest.raises(ValidationError, match="UTF-8|surrogate|unicode"):
        MigrationInfo(from_schema_version=0, note="bad " + chr(0xDC00) + " text")


def test_invalid_utf8_bytes_are_refused_when_loading_stored_json():
    raw = finding().canonical_bytes().replace(b"BLQ", b"\xff\xfe")
    with pytest.raises(ValidationError):
        type(finding()).load(raw)
    with pytest.raises(ValidationError):
        type(finding()).model_validate_json(raw)


# --- 4. hashes are domain-separated by record type and schema version -----------------------------------------------
def test_hash_payload_names_its_domain_record_type_and_schema_version():
    f = finding()
    body = json.loads(f.canonical_bytes())
    body.pop("content_hash")
    payload = {"domain": "pharmagent.workflow.record", "record_type": "Finding", "schema_version": 1, "fields": body}
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    assert f.content_hash == expected


def test_two_record_types_with_identical_fields_never_share_a_hash():
    class OtherFinding(type(finding())):
        pass

    base = finding()
    other = OtherFinding(**{k: getattr(base, k) for k in type(base).model_fields if k != "content_hash"})
    assert other.model_dump(exclude={"content_hash"}) == base.model_dump(exclude={"content_hash"})
    assert other.content_hash != base.content_hash


# --- 5. projection ids ------------------------------------------------------------------------------------------------
def _hex_tail(value: str, prefix: str) -> str:
    assert value.startswith(prefix), value
    return value[len(prefix):]


def test_hashed_projection_and_assumption_ids_carry_at_least_128_bits():
    p = project_pharmstate(PharmState(session_id="bad\x00id"), as_of=T0)
    tail = _hex_tail(p.id, "legacy:h:")
    assert len(tail) >= 32 and set(tail) <= set("0123456789abcdef")
    assert len(_hex_tail(p.assumptions.id, "legacy-assumptions:")) >= 32


def test_legacy_dataset_ids_embed_the_data_digest_so_different_data_never_collides():
    meta = {"dataset_id": "ds_1", "n_subjects": 3, "detected_roles": {"ID": "ID"}, "columns": []}
    ids = set()
    for digest in ("a" * 64, "b" * 64, None):
        m = dict(meta) if digest is None else {**meta, "dataset_sha256": digest}
        p = project_pharmstate(PharmState(session_id="s", dataset_id="ds_1", dataset_metadata=m,
                                          dataset_path="x.csv"), as_of=T0)
        ids.add(p.dataset.id)
    assert len(ids) == 3
    assert f"dataset:ds_1:legacy:{'a' * 32}" in ids and "dataset:ds_1:legacy:nodigest" in ids


# --- 6. FrozenMap cannot be re-initialised ----------------------------------------------------------------------------
def test_frozen_map_init_cannot_be_called_again_on_a_live_instance():
    m = FrozenMap({"a": 1})
    with pytest.raises(TypeError, match="already"):
        m.__init__({"zz": 9})
    assert dict(m) == {"a": 1}


def test_a_live_records_mapping_cannot_be_rewritten_through_init():
    e = WorkflowEvent(id="event:e1", created_at=T0, producer="p", event_type="finding_raised", actor_kind="system",
                      actor="qc", subject_id="finding:f1", subject_hash="a" * 64, detail={"k": 1})
    with pytest.raises(TypeError):
        e.detail.__init__({"hijack": True})
    assert e.verify_integrity() and "hijack" not in e.detail


# --- 7. strict schema version and timestamps --------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [1.0, "1", True, None, 1.5])
def test_schema_version_must_be_the_exact_integer(bad):
    data = json.loads(finding().canonical_bytes())
    data["schema_version"] = bad
    with pytest.raises(ValidationError):
        type(finding()).load(data)
    with pytest.raises(ValidationError):
        type(finding())(**{**{k: getattr(finding(), k) for k in type(finding()).model_fields if k != "content_hash"},
                           "schema_version": bad})


@pytest.mark.parametrize("bad", [0, 1_700_000_000, 1_700_000_000.5, True, "2026-10-26T12:00:00",
                                 "2026-10-26 12:00:00", "2026-10-26T12:00:00.1234567Z", "yesterday", b"2026-10-26T12:00:00Z"])
def test_timestamps_must_be_datetimes_or_iso_strings_with_an_offset_and_at_most_microseconds(bad):
    with pytest.raises(ValidationError):
        finding(created_at=bad)
    with pytest.raises(ValidationError):
        execution(started_at=bad, finished_at=None, status="running")


@pytest.mark.parametrize("good", ["2026-10-26T12:00:00Z", "2026-10-26T12:00:00+00:00", "2026-10-26T08:00:00-04:00",
                                  "2026-10-26T12:00:00.123456Z", datetime(2026, 10, 26, 12, tzinfo=UTC)])
def test_well_formed_timestamps_are_still_accepted_and_normalised_to_utc(good):
    f = finding(created_at=good)
    assert f.created_at.utcoffset().total_seconds() == 0 and f.created_at.year == 2026


# --- 8. blank-looking text --------------------------------------------------------------------------------------------
BLANKS = ["ㅤ", "ᅟ", "ᅠ", "ﾠ", "⠀", "឴", "឵", "͏"]


@pytest.mark.parametrize("blank", BLANKS)
def test_blank_looking_characters_are_refused_alone_or_mixed_into_labels(blank):
    for text in (blank, f"a{blank}", f"{blank}a", f"Dr. {blank} Smith"):
        with pytest.raises(ValidationError):
            review(actor_label=text)
        with pytest.raises(ValidationError):
            finding(producer=text)
        with pytest.raises(ValidationError):
            ColumnSchema(name=text)
    with pytest.raises(ValidationError):
        finding(message=f"text {blank} text")


def test_ordinary_non_ascii_names_are_still_accepted():
    for fine in ("José Álvarez", "张伟", "O’Brien", "Müller-Lehmann"):
        assert review(actor_label=fine).actor_label == fine


# --- 9. documentation states what the code does -----------------------------------------------------------------------
DOC = "docs/architecture/workflow-orchestration.md"


def _doc() -> str:
    from pathlib import Path
    return (Path(__file__).resolve().parents[3] / DOC).read_text()


def test_the_doc_lists_every_module_and_the_load_construct_distinction():
    from pathlib import Path
    doc = _doc()
    modules = {p.name for p in (Path(__file__).resolve().parents[2] / "app" / "workflow").glob("*.py") if p.name != "__init__.py"}
    for name in modules:
        assert f"`{name}`" in doc, f"{name} missing from the module table"
    assert "WorkflowContract.load" in doc and "domain-separated" in doc


def test_the_doc_no_longer_overstates_what_re_validation_catches():
    doc = _doc()
    assert "which catches a forged record that was rehashed" not in doc
    assert "does **not** catch a structurally valid forgery" in doc
    assert "not exhaustive" in doc            # the blank-looking character list
