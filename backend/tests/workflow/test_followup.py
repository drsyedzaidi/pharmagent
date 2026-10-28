"""Second hardening round: explicit sealing, stable record types, text/number strictness, error hygiene."""
import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.workflow.base import MigrationInfo, Unknown, WorkflowContract
from app.workflow.contracts import ArtifactRef, Finding, ReviewRecord
from app.workflow.errors import ContractError
from app.workflow.events import AuditEntryRef, WorkflowEvent
from app.workflow.stages import WorkflowStage, is_allowed
from tests.workflow.factories import T0, artifact, assumptions, dataset, execution, finding, review, transformation
from tests.workflow.test_hardening import FAMILY_IDS, families


def fields_of(rec):
    return {k: getattr(rec, k) for k in type(rec).model_fields if k != "content_hash"}


# --- item 3: generic validation requires and verifies a hash; sealing is an explicitly named API --------------------
@pytest.mark.parametrize("rec", families(), ids=FAMILY_IDS)
def test_every_generic_entry_point_refuses_a_blank_or_missing_hash(rec):
    cls, data = type(rec), json.loads(rec.canonical_bytes())
    for broken in ({**data, "content_hash": ""}, {k: v for k, v in data.items() if k != "content_hash"}):
        with pytest.raises(ValidationError, match="content_hash"):
            cls.model_validate(broken)
        with pytest.raises(ValidationError, match="content_hash"):
            cls.model_validate_json(json.dumps(broken))
        with pytest.raises(ValidationError, match="content_hash"):
            cls.load(broken)
        with pytest.raises(ValidationError, match="content_hash"):
            cls(**broken)


def test_the_constructor_error_tells_the_caller_how_to_create_a_record():
    with pytest.raises(ValidationError, match="create"):
        Finding(**fields_of(finding()))


@pytest.mark.parametrize("rec", families(), ids=FAMILY_IDS)
def test_create_seals_and_equals_the_record_loaded_back(rec):
    made = type(rec).create(**fields_of(rec))
    assert made == rec and made.verify_integrity()
    assert type(rec).model_validate(json.loads(made.canonical_bytes())) == made


def test_create_refuses_a_caller_supplied_hash():
    with pytest.raises(TypeError, match="content_hash"):
        Finding.create(**fields_of(finding()), content_hash="a" * 64)


def test_replace_makes_a_new_sealed_record_and_leaves_the_original_alone():
    f = finding()
    g = f.replace(message="a different message")
    assert g.message == "a different message" and g.content_hash != f.content_hash and g.verify_integrity()
    assert f.message != g.message and f.replace() == f
    with pytest.raises(ContractError, match="unknown fields"):
        f.replace(zzz=1)
    with pytest.raises(TypeError, match="content_hash"):
        f.replace(content_hash="a" * 64)


def test_model_copy_can_no_longer_reseal_a_changed_record():
    f = finding()
    assert f.model_copy() == f and f.model_copy(deep=True) == f
    for update in ({"message": "changed"}, {"message": "changed", "content_hash": ""}, {"content_hash": ""}):
        with pytest.raises(ValidationError, match="content_hash"):
            f.model_copy(update=update)


def test_stripping_the_hash_from_a_tampered_review_cannot_launder_it_on_any_entry_point():
    art, other = artifact(b"cleaned"), artifact(b"swapped")
    data = json.loads(review(art).canonical_bytes())
    data["bound_artifacts"] = [{"artifact_id": other.id, "digest": other.digest}]
    data.pop("content_hash")
    for call in (ReviewRecord.model_validate, ReviewRecord.load, lambda d: ReviewRecord.model_validate_json(json.dumps(d)),
                 lambda d: ReviewRecord(**d)):
        with pytest.raises(ValidationError, match="content_hash"):
            call(data)


# --- item 1: the advisory name lint judges NEW records only ------------------------------------------------------------
def test_stored_reviews_stay_readable_after_the_lint_is_tightened_but_new_ones_are_held_to_it():
    import app.workflow.contracts as contracts
    art = artifact(b"x")
    stored = review(art, actor_label="Pat Smith")
    raw, original = stored.canonical_bytes(), contracts.AUTOMATED_NAME_BRANDS
    try:
        contracts.AUTOMATED_NAME_BRANDS = (*original, "smith")
        assert ReviewRecord.load(raw) == stored and ReviewRecord.model_validate_json(raw) == stored
        assert stored.binding_status({art.id: art.digest}) == "current"
        assert stored.model_copy() == stored
        with pytest.raises(ValidationError, match="automated agent"):
            review(art, actor_label="Pat Smith")
        with pytest.raises(ValidationError, match="automated agent"):
            stored.replace(reason="re-reviewed")
    finally:
        contracts.AUTOMATED_NAME_BRANDS = original


# --- item 2: the Unicode rules apply to Unknown.reason and MigrationInfo.note --------------------------------------
@pytest.mark.parametrize("bad", ["a\u202eb", "nul\x00byte", "zero\u200bwidth", "   ", "\x1b[2J", "line\u2028sep", "\u3164"])
def test_unknown_reasons_and_migration_notes_follow_the_text_rules(bad):
    with pytest.raises(ValidationError):
        Unknown(status="unknown", reason=bad)
    with pytest.raises(ValidationError):
        MigrationInfo(from_schema_version=0, note=bad)


def test_ordinary_reasons_and_notes_are_still_accepted():
    assert Unknown(status="unavailable", reason="legacy state records no units\nsee audit chain").reason
    assert MigrationInfo(from_schema_version=0, note="projected from legacy state").note


# --- item 4: strict numbers and booleans -------------------------------------------------------------------------------
def test_migration_version_seed_and_reversible_are_exact_types():
    with pytest.raises(ValidationError):
        MigrationInfo(from_schema_version=True, note="n")
    for seed in (True, "7", 3.0, 7.5):
        with pytest.raises(ValidationError):
            execution(seed=seed)
    for flag in ("no", "true", 1, 0, "False"):
        with pytest.raises(ValidationError):
            transformation(reversible=flag)
    assert execution(seed=7).seed == 7 and transformation(reversible=False).reversible is False


def test_counts_and_sizes_refuse_strings_and_floats_so_stored_bytes_stay_canonical():
    art = artifact(b"abc")
    data = json.loads(art.canonical_bytes())
    for bad in ("3", 3.0, True):
        with pytest.raises(ValidationError):
            ArtifactRef.load({**data, "size_bytes": bad})
    with pytest.raises(ValidationError):
        transformation(affected_rows="3")
    with pytest.raises(ValidationError):
        dataset(n_subjects=12.0)


def test_float_fields_refuse_strings_and_booleans():
    for bad in ("5.0", True):
        with pytest.raises(ValidationError):
            assumptions(initial_estimates={"CL": bad})
    assert assumptions(initial_estimates={"CL": 5}).initial_estimates["CL"] == 5


# --- item 5: is_allowed sanitises `kind` too -----------------------------------------------------------------------
def test_is_allowed_refuses_a_lying_kind():
    class LyingKind(str):
        def __eq__(self, other):
            return True

        __hash__ = str.__hash__

    assert not is_allowed(None, "data_curation", LyingKind("rollback"))
    assert not is_allowed(WorkflowStage.EDA, WorkflowStage.NCA, 5)
    assert is_allowed(None, "data_curation", "advance")


# --- item 6: blank-looking text ----------------------------------------------------------------------------------------
IGNORABLE = [chr(cp) for cp in (0xFE0F, 0x180B, 0xE0100, 0x00AD, 0x2060, 0xFEFF, 0x16FE4, 0xE0001)]
UNASSIGNED_OR_PRIVATE = [chr(cp) for cp in (0xE000, 0x0378, 0xF0000)]


@pytest.mark.parametrize("ch", IGNORABLE + UNASSIGNED_OR_PRIVATE)
def test_default_ignorable_private_and_unassigned_characters_are_refused(ch):
    for text in (ch, f"a{ch}", f"{ch}a"):
        with pytest.raises(ValidationError):
            finding(producer=text)
        with pytest.raises(ValidationError):
            review(actor_label=text)
    with pytest.raises(ValidationError):
        finding(message=f"text {ch} text")


def test_a_label_or_message_of_only_combining_marks_is_refused_but_accented_names_are_fine():
    for only_marks in ("\u0301", "\u0301\u0302"):
        with pytest.raises(ValidationError):
            finding(producer=only_marks)
        with pytest.raises(ValidationError):
            finding(message=only_marks)
    for fine in ("Jos\u00e9", "Jose\u0301", "\u5f20\u4f1f", "O\u2019Brien"):
        assert review(actor_label=fine).actor_label == fine


# --- item 7: explicit, unique, stable record-type constants ------------------------------------------------------------
@pytest.mark.parametrize("rec", families(), ids=FAMILY_IDS)
def test_every_family_declares_its_record_type_explicitly(rec):
    cls = type(rec)
    assert "RECORD_TYPE" in cls.__dict__ and isinstance(cls.RECORD_TYPE, str) and cls.RECORD_TYPE


def test_the_hash_uses_the_declared_constant_not_the_class_name():
    class Local(Finding):
        RECORD_TYPE = "TestRecordTypeConstantA"

    import hashlib
    base = fields_of(finding())
    rec = Local.create(**base)
    body = json.loads(rec.canonical_bytes())
    body.pop("content_hash")
    payload = {"domain": "pharmagent.workflow.record", "record_type": "TestRecordTypeConstantA", "schema_version": 1, "fields": body}
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    assert rec.content_hash == expected and rec.content_hash != finding().content_hash


def test_a_subclass_without_its_own_constant_or_with_a_duplicate_one_cannot_be_defined():
    with pytest.raises(TypeError, match="RECORD_TYPE"):
        class NoConstant(Finding):
            pass

    with pytest.raises(TypeError, match="already"):
        class Duplicate(Finding):
            RECORD_TYPE = "Finding"

    with pytest.raises(TypeError, match="RECORD_TYPE"):
        class Blank(Finding):
            RECORD_TYPE = ""


def test_the_base_class_cannot_be_instantiated_as_a_record():
    with pytest.raises(TypeError):
        WorkflowContract.create(id="x", created_at=T0, producer="p")


# --- item 8: to_utc uses the real datetime method ---------------------------------------------------------------------
def test_a_datetime_subclass_with_a_lying_astimezone_cannot_change_the_instant():
    class Liar(datetime):
        def astimezone(self, tz=None):
            return datetime(2000, 1, 1, tzinfo=UTC)

    f = finding(created_at=Liar(2026, 10, 26, 12, tzinfo=UTC))
    assert f.created_at == datetime(2026, 10, 26, 12, tzinfo=UTC) and type(f.created_at) is datetime


# --- item 9: validation errors do not echo the offending input -----------------------------------------------------------
def test_printed_validation_errors_do_not_echo_phi_like_input():
    secret = "JANE-DOE-MRN-998877"
    attempts = (
        lambda: Unknown(status="unknown", reason=secret + "\x00"),
        lambda: finding(producer=secret + "\u202e"),
        lambda: finding(message=secret + "\x1b"),
        lambda: review(actor_label=secret + "\u3164"),
        lambda: dataset(source=secret + "\x00"),
        lambda: AuditEntryRef(session_id=secret + "\x00", index=0, entry_hash="a" * 64),
    )
    for attempt in attempts:
        with pytest.raises(ValidationError) as caught:
            attempt()
        # the PRINTED and repr'd forms (what reaches logs) hide the input; errors() is programmatic and can include it
        assert secret not in str(caught.value) and secret not in repr(caught.value)
        assert secret not in repr(caught.value.errors(include_input=False))


# --- item 11: naming and honest docstrings ----------------------------------------------------------------------------
def test_event_actor_kind_is_named_as_the_claim_it_is():
    fields = set(WorkflowEvent.model_fields)
    assert "claimed_actor_kind" in fields and "actor_kind" not in fields
    assert "claim" in (WorkflowEvent.__doc__ or "").lower()


def test_approval_and_audit_reference_docstrings_say_structural_only():
    doc = (ReviewRecord.is_current_approval.__doc__ or "").lower()
    assert "structural" in doc and "not" in doc and "authoriz" in doc
    assert "unverified" in (AuditEntryRef.__doc__ or "").lower()


def test_the_event_documents_that_its_subject_reference_carries_no_record_type():
    assert "subject_record_type" not in WorkflowEvent.model_fields
    doc = " ".join((WorkflowEvent.__doc__ or "").lower().split())
    assert "does not identify the record type" in doc


# --- source hygiene: no invisible, bidirectional or otherwise non-ASCII characters in the Phase A code ----------------
def test_phase_a_sources_are_pure_ascii_so_no_invisible_or_bidi_character_can_hide_in_them():
    import glob
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    files = sorted(glob.glob(str(root / "app" / "workflow" / "*.py")) + glob.glob(str(root / "tests" / "workflow" / "*.py")))
    assert len(files) > 10
    offenders = {}
    for name in files:
        text = Path(name).read_text(encoding="utf-8")
        bad = sorted({hex(ord(c)) for c in text if ord(c) > 127})
        if bad:
            offenders[Path(name).name] = bad
    assert not offenders, f"non-ASCII characters found: {offenders}"


def test_the_architecture_doc_has_no_invisible_or_control_characters():
    import unicodedata
    text = _doc_text()
    bad = sorted({hex(ord(c)) for c in text if unicodedata.category(c) in {"Cc", "Cf", "Zl", "Zp"} and c not in "\n\t"})
    assert not bad, f"invisible characters in the doc: {bad}"


def _doc_text() -> str:
    from pathlib import Path
    return (Path(__file__).resolve().parents[3] / "docs" / "architecture" / "workflow-orchestration.md").read_text(encoding="utf-8")
