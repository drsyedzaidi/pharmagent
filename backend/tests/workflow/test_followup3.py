"""Third hardening round: top-level-only sealing, a PHI-safe error boundary, redacted repr, context-aware Unicode."""
import json
import random
import time
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.workflow.base import FrozenMap, Unknown, WorkflowContract
from app.workflow.contracts import (
    ArtifactRef,
    AssumptionSet,
    BlqRepresentation,
    ColumnSchema,
    DatasetVersion,
    ExecutionRecord,
    Finding,
    ReviewRecord,
)
from app.workflow.errors import ContractError, safe_errors
from app.workflow.events import AuditEntryRef, WorkflowEvent
from app.workflow.lineage import check_lineage
from tests.workflow.factories import T0, artifact, assumptions, dataset, finding, review, transformation
from tests.workflow.test_hardening import FAMILY_IDS, families

SECRET = "JANE-DOE-MRN-4455667"


def C(*codepoints):
    return "".join(chr(cp) for cp in codepoints)


def fields_of(rec):
    return {k: getattr(rec, k) for k in type(rec).model_fields if k != "content_hash"}


# --- replace() / create() only ever seal a verified or top-level record --------------------------------------------
def test_replace_refuses_a_record_that_no_longer_matches_its_hash():
    f = finding()
    object.__setattr__(f, "message", "TAMPERED")
    assert not f.verify_integrity()
    with pytest.raises(ContractError, match="integrity"):
        f.replace(severity="error")
    assert finding().replace(severity="error").verify_integrity()


def test_create_seals_only_the_top_level_record_never_a_hashless_nested_one():
    from app.workflow.compat import LegacyProjection
    ds = dataset()
    nested = json.loads(ds.canonical_bytes())
    nested.pop("content_hash")
    base = dict(id="legacy:x", created_at=T0, producer="p", session_id="s", environment_id="env:1",
                run_provenance_status="verified", stage_evidence=(), assumptions=assumptions(),
                executions=Unknown(status="unavailable", reason="x"), reviews=Unknown(status="unavailable", reason="x"))
    with pytest.raises(ValidationError, match="content_hash"):
        LegacyProjection.create(**base, dataset=nested)
    forged = DatasetVersion.model_construct(**{**fields_of(ds), "content_hash": ""})
    with pytest.raises(ValidationError, match="content_hash"):
        LegacyProjection.create(**base, dataset=forged)
    assert LegacyProjection.create(**base, dataset=ds).dataset == ds


def test_a_non_mapping_validation_context_is_not_an_attribute_error():
    data = json.loads(finding().canonical_bytes())
    for ctx in ("x", [1], 5):
        assert Finding.model_validate(data, context=ctx) == finding()
    blank = {**data, "content_hash": ""}
    for ctx in ("x", [1], 5):
        with pytest.raises(ValidationError, match="content_hash"):
            Finding.model_validate(blank, context=ctx)


def test_the_seal_marker_must_name_the_class_being_created():
    from app.workflow.base import SEAL
    blank = {**json.loads(finding().canonical_bytes()), "content_hash": ""}
    for marker in (True, 1, "Finding", object(), ReviewRecord):
        with pytest.raises(ValidationError, match="content_hash"):
            Finding.model_validate(blank, context={SEAL: marker})


# --- error text never echoes input; errors have a safe, scrubbed form ------------------------------------------------
def test_schema_version_errors_do_not_echo_the_offending_value():
    data = json.loads(finding().canonical_bytes())
    with pytest.raises(ValidationError) as caught:
        Finding.load({**data, "schema_version": SECRET})
    assert SECRET not in str(caught.value) and SECRET not in repr(caught.value.errors(include_input=False))


def test_contract_errors_carry_ids_in_detail_not_in_the_message():
    with pytest.raises(ContractError) as unknown:
        finding().replace(**{SECRET: 1})
    assert SECRET not in str(unknown.value)
    child = dataset(id="dsv:" + "child", parents=("dsv:" + SECRET.lower(),))
    with pytest.raises(ContractError) as missing:
        check_lineage([child])
    assert SECRET.lower() not in str(missing.value) and "missing" in str(missing.value)
    assert SECRET.lower() in repr(missing.value.detail)


def test_safe_errors_returns_only_type_scrubbed_location_and_a_fixed_message():
    data = json.loads(finding().canonical_bytes())
    with pytest.raises(ValidationError) as caught:
        Finding.load({**data, SECRET: 1, "message": SECRET + chr(0)})
    cleaned = safe_errors(caught.value)
    assert cleaned and all(set(e) == {"type", "loc", "msg"} for e in cleaned)
    assert SECRET not in json.dumps(cleaned)
    assert ["<key>"] in [list(e["loc"]) for e in cleaned] and ["message"] in [list(e["loc"]) for e in cleaned]


def test_safe_errors_scrubs_dynamic_dict_keys_but_keeps_schema_names_and_indices():
    with pytest.raises(ValidationError) as caught:
        ReviewRecord.create(id="r", created_at=T0, producer="p", gate="final_model", decision="rejected", actor_label="Pat",
                            reason="r", rule_versions={SECRET: chr(0)}, comments=("ok", chr(0)))
    locs = [list(e["loc"]) for e in safe_errors(caught.value)]
    assert ["rule_versions", "<key>"] in locs or ["rule_versions", "<key>", "[key]"] in locs
    assert ["comments", 1] in locs
    assert SECRET not in json.dumps(safe_errors(caught.value))


def test_safe_errors_survives_lone_surrogates_that_break_default_json():
    with pytest.raises(ValidationError) as caught:
        Finding.load({**json.loads(finding().canonical_bytes()), "message": SECRET + chr(0xD800)})
    json.dumps(safe_errors(caught.value))


@pytest.mark.parametrize("rec", families(), ids=FAMILY_IDS)
def test_no_field_of_any_record_family_leaks_input_through_the_safe_error_boundary(rec):
    cls, good = type(rec), json.loads(rec.canonical_bytes())
    scenarios = [lambda name: ("replace", SECRET + chr(0)), lambda name: ("replace", {SECRET: chr(0)}),
                 lambda name: ("replace", [SECRET, {SECRET: chr(0)}]), lambda name: ("extra", SECRET)]
    raised = 0
    for name in good:
        if name == "content_hash":
            continue
        for scenario in scenarios:
            kind, value = scenario(name)
            data = dict(good)
            if kind == "replace":
                data[name] = value
            else:
                data[value] = 1
            try:
                cls.load({**data, "content_hash": good["content_hash"]})
            except ValidationError as exc:
                raised += 1
                assert SECRET not in json.dumps(safe_errors(exc)), (name, kind)
    assert raised > 20


# --- records are not log-safe by default: repr/str show type and a short hash only ------------------------------
def test_repr_and_str_of_every_record_hide_free_text_and_identifiers():
    rec = review(artifact(b"x"), actor_label="Dr. " + SECRET, reason=SECRET, comments=(SECRET,),
                 legacy_metadata={"k": SECRET}, rule_versions={SECRET: "1"})
    for text in (repr(rec), str(rec)):
        assert SECRET not in text and "ReviewRecord" in text and rec.content_hash[:12] in text


@pytest.mark.parametrize("rec", families(), ids=FAMILY_IDS)
def test_repr_of_each_family_names_the_type_and_hash_prefix_only(rec):
    for text in (repr(rec), str(rec)):
        assert type(rec).RECORD_TYPE in text and rec.content_hash[:12] in text
        assert rec.id not in text and len(text) < 120


def test_nested_models_and_frozen_maps_are_redacted_too():
    assert SECRET not in repr(ColumnSchema(name=SECRET)) and SECRET not in repr(Unknown(status="unknown", reason=SECRET))
    assert SECRET not in repr(FrozenMap({"k": SECRET, SECRET: 1})) and SECRET not in repr(AuditEntryRef(
        session_id=SECRET, index=0, entry_hash="a" * 64))
    assert "FrozenMap" in repr(FrozenMap({"k": 1}))


# --- Unicode policy: context-sensitive joiners, create-only Cn/Co -----------------------------------------------------
ZWNJ, ZWJ, VS16 = 0x200C, 0x200D, 0xFE0F
PERSIAN = C(0x645, 0x6CC, ZWNJ, 0x62E, 0x648, 0x627, 0x647, 0x645)
DEVANAGARI = C(0x915, 0x94D, ZWJ, 0x937)
SINHALA = C(0xD9A, 0xDCA, ZWJ, 0xDBB)
HEART = C(0x2764, VS16)
HEALTH_WORKER = C(0x1F468, ZWJ, 0x2695, VS16)
KEYCAP = C(0x31, VS16, 0x20E3)


@pytest.mark.parametrize("name", ["PERSIAN", "DEVANAGARI", "SINHALA", "HEART", "HEALTH_WORKER", "KEYCAP"])
def test_native_script_and_emoji_joiners_are_accepted_where_they_belong(name):
    text = globals()[name]
    assert review(actor_label="Dr. " + text).actor_label == "Dr. " + text
    assert finding(message="note: " + text).message == "note: " + text
    assert dataset(source=text + ".csv").source == text + ".csv"


@pytest.mark.parametrize("bad", [C(0x61, ZWNJ, 0x62), C(ZWNJ, 0x61), C(0x61, ZWNJ), C(0x61, ZWJ, 0x62), C(0x61, ZWJ),
                                 C(ZWJ, 0x61), C(0x61, VS16), C(VS16, 0x61), C(0x61, VS16, VS16), C(0x61, 0x200E, 0x62),
                                 C(0x61, 0x200F, 0x62), C(0x2764, ZWNJ, 0x2764), C(0x61, ZWJ, 0x2764)])
def test_joiners_in_the_wrong_place_and_other_invisible_marks_are_still_refused(bad):
    for make in (lambda t: review(actor_label=t), lambda t: finding(message="x " + t + " y"), lambda t: ColumnSchema(name=t)):
        with pytest.raises(ValidationError):
            make(bad)


def test_a_file_name_that_is_not_representable_degrades_only_the_source_not_the_dataset():
    from app.core.pharmstate import PharmState
    from app.workflow.compat import project_pharmstate
    meta = {"dataset_id": "ds_1", "n_subjects": 3, "detected_roles": {"ID": "ID"}, "columns": [],
            "dataset_sha256": "a" * 64}
    for path in ("/data/" + PERSIAN + ".csv", "/data/" + HEART + ".csv"):
        p = project_pharmstate(PharmState(session_id="s", dataset_id="ds_1", dataset_metadata=meta, dataset_path=path), as_of=T0)
        assert isinstance(p.dataset, DatasetVersion) and p.dataset.source.endswith(".csv")
    bad = project_pharmstate(PharmState(session_id="s", dataset_id="ds_1", dataset_metadata=meta,
                                        dataset_path="/data/bad" + chr(0) + "name.csv"), as_of=T0)
    assert isinstance(bad.dataset, DatasetVersion) and isinstance(bad.dataset.source, Unknown)
    assert bad.dataset.data_digest == "a" * 64 and bad.dataset.n_subjects == 3


def test_private_use_and_unassigned_characters_are_refused_on_create_but_stored_records_stay_loadable():
    co = chr(0xE000)
    stored = ReviewRecord.model_construct(**{**fields_of(review()), "actor_label": "Pat " + co})
    object.__setattr__(stored, "content_hash", stored.compute_hash())
    raw = stored.canonical_bytes()
    assert ReviewRecord.load(raw) == stored and ReviewRecord.model_validate_json(raw) == stored
    assert stored.validated_copy() == stored
    art = artifact(b"x")
    keep = ReviewRecord.model_construct(**{**fields_of(review(art)), "actor_label": "Pat " + co})
    object.__setattr__(keep, "content_hash", keep.compute_hash())
    assert keep.binding_status({art.id: art.digest}) == "current"
    with pytest.raises(ValidationError):
        review(actor_label="Pat " + co)
    with pytest.raises(ValidationError):
        keep.replace(reason="re-reviewed")


def _reference_bad(value, allow_layout):
    import unicodedata
    for c in value:
        if allow_layout and c in "\n\t":
            continue
        if unicodedata.category(c) in {"Cc", "Cf", "Zl", "Zp"}:
            return True
    return False


def test_the_ascii_fast_path_agrees_with_the_character_by_character_rule():
    from app.workflow.base import check_text
    rng = random.Random(7)
    pool = [chr(c) for c in range(0, 128)] + [C(0xE9), C(0x5F20), C(0x2019)]
    for _ in range(3000):
        text = "".join(rng.choice(pool) for _ in range(rng.randint(0, 12)))
        for allow_layout in (False, True):
            try:
                check_text(text, allow_layout=allow_layout, creating=True)
                refused = False
            except ValueError:
                refused = True
            if text.isascii():
                assert refused == _reference_bad(text, allow_layout), repr(text)


def test_checking_long_ascii_text_is_much_faster_than_a_per_character_scan():
    from app.workflow.base import check_text
    text = ("lorem ipsum dolor sit amet " * 40000)[:1_000_000]
    t = time.perf_counter()
    check_text(text, allow_layout=True, creating=True)
    fast = time.perf_counter() - t
    t = time.perf_counter()
    _reference_bad(text, True)
    slow = time.perf_counter() - t
    assert fast * 4 < slow, (fast, slow)


# --- smaller defects ----------------------------------------------------------------------------------------------
def test_to_utc_reads_the_real_datetime_fields_not_subclass_overrides():
    class Lying(datetime):
        year = property(lambda self: 1999)
        hour = property(lambda self: 3)

    assert finding(created_at=Lying(2026, 10, 26, 12, tzinfo=UTC)).created_at == datetime(2026, 10, 26, 12, tzinfo=UTC)


def test_lloq_refuses_strings_and_bytes_like_every_other_float_field():
    for bad in ("5.0", " 1.5 ", "1e3", b"1.5", True):
        with pytest.raises(ValidationError):
            BlqRepresentation(basis="none", lloq=bad)
    assert BlqRepresentation(basis="none", lloq=5).lloq == 5


def test_the_abstract_base_gives_a_clear_error_when_asked_for_a_hash():
    ghost = WorkflowContract.model_construct(id="x")
    with pytest.raises(TypeError, match="RECORD_TYPE"):
        ghost.compute_hash()
    assert ghost.verify_integrity() is False


def test_frozen_map_is_not_equal_to_things_that_are_not_mappings():
    assert FrozenMap({"a": 1}) != 5 and FrozenMap({"a": 1}) != [("a", 1)] and FrozenMap({}) != None  # noqa: E711


def test_a_missing_dataset_path_makes_the_source_unknown_without_dropping_the_dataset():
    from app.core.pharmstate import PharmState
    from app.workflow.compat import project_pharmstate
    meta = {"dataset_id": "ds_1", "n_subjects": 3, "detected_roles": {"ID": "ID"}, "columns": [], "dataset_sha256": "b" * 64}
    for path in (None, "", "/data/"):
        p = project_pharmstate(PharmState(session_id="s", dataset_id="ds_1", dataset_metadata=meta, dataset_path=path), as_of=T0)
        assert isinstance(p.dataset, DatasetVersion) and isinstance(p.dataset.source, Unknown)
        assert "no dataset path" in p.dataset.source.reason


def test_frozen_map_equality_with_a_deep_foreign_mapping_is_false_not_an_error():
    deep = {}
    for _ in range(100):
        deep = {"k": deep}
    assert (FrozenMap({"a": 1}) == deep) is False


def test_record_type_constants_must_be_plain_ascii_identifiers():
    for bad in ("Has Space", "caf" + chr(0xE9), "1Starts", "a-b", ""):
        with pytest.raises(TypeError, match="RECORD_TYPE"):
            type("Bad", (Finding,), {"RECORD_TYPE": bad, "__module__": __name__})


def test_record_type_is_a_declared_literal_not_derived_from_the_class_name():
    expected = {"ArtifactRef", "DatasetVersion", "TransformationRecord", "AssumptionSet", "ExecutionRecord", "Finding",
                "ReviewRecord", "StageTransition", "WorkflowEvent", "LegacyProjection"}
    from app.workflow.compat import LegacyProjection
    from app.workflow.stages import StageTransition
    classes = (ArtifactRef, DatasetVersion, type(transformation()), AssumptionSet, ExecutionRecord, Finding, ReviewRecord,
               StageTransition, WorkflowEvent, LegacyProjection)
    assert {c.RECORD_TYPE for c in classes} == expected


def test_a_trailing_zero_microsecond_timestamp_hashes_to_its_pinned_vector():
    f = Finding.create(id="finding:us", created_at=datetime(2026, 10, 26, 12, 0, 0, 120000, tzinfo=UTC), producer="pharmagent.test",
                       stage="data_curation", severity="info", code="c.us", message="m", source="automated")
    assert '"created_at":"2026-10-26T12:00:00.120000Z"' in f.canonical_bytes().decode()
    assert f.content_hash == "2ca845c221804276d2219951c7425b81f5f0be38c6219e4112cefd22f50285b7"


# --- documentation states the boundaries -----------------------------------------------------------------------------
def _doc():
    from pathlib import Path
    return " ".join((Path(__file__).resolve().parents[3] / "docs" / "architecture" / "workflow-orchestration.md")
                    .read_text(encoding="utf-8").split())


def test_the_doc_makes_the_phi_error_boundary_a_phase_b_blocking_requirement():
    doc = _doc()
    assert "safe_errors" in doc and "blocking" in doc.lower() and "include_input=False" in doc
    assert "loc" in doc and ".json()" in doc and "repr" in doc


def test_the_doc_states_the_unicode_and_hash_stability_limits():
    doc = _doc()
    for phrase in ("ZWNJ", "VS16", "create-only", "U+16FE4", "CRLF", "trailing-zero", "pydantic"):
        assert phrase in doc, phrase
    assert "validating never seals" not in doc.lower() or "only the top-level" in doc


def test_the_doc_describes_hand_computed_vectors_accurately():
    doc = _doc()
    assert "ten of them were computed by hand" not in doc
    assert "except the legacy projection" in doc


def test_junk_seal_markers_never_crash_and_never_lint_or_seal_a_valid_stored_record():
    from app.workflow.base import SEAL
    stored = json.loads(review().canonical_bytes())
    for marker in (True, 1, "Finding", object(), None, [ReviewRecord]):
        assert ReviewRecord.model_validate(stored, context={SEAL: marker}) == review()


def test_the_abstract_base_cannot_create_a_record():
    with pytest.raises(TypeError, match="RECORD_TYPE"):
        WorkflowContract.create(id="x", created_at=T0, producer="p")


def test_safe_errors_asks_pydantic_to_leave_out_input_url_and_context():
    calls = []

    class Recording:
        def errors(self, **kwargs):
            calls.append(kwargs)
            return [{"type": "value_error", "loc": ("message",), "msg": "Value error, fixed text",
                     "input": SECRET, "url": "https://example.invalid/" + SECRET, "ctx": {"error": SECRET}}]

    cleaned = safe_errors(Recording())
    assert calls == [{"include_input": False, "include_url": False, "include_context": False}]
    assert SECRET not in json.dumps(cleaned) and cleaned == [{"type": "value_error", "loc": ("message",), "msg": "Value error, fixed text"}]
