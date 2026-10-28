"""Fourth hardening round: string-only keys, a leak-free error chain, complete repr redaction, honest warnings, a
creation-vs-load text policy, a bounded public ``load()``, and no self-nested contracts.

Non-ASCII characters are always built with ``chr()``: the hygiene test forbids them in source."""
import json
import logging
import traceback
import warnings

import pytest
from pydantic import TypeAdapter, ValidationError

from app.workflow.base import FrozenMap, Unknown, WorkflowContract
from app.workflow.contracts import ColumnSchema, ExecutionRecord, Finding, ReviewRecord
from app.workflow.errors import ContractError, safe_errors
from tests.workflow.factories import T0, artifact, dataset, execution, finding, review, transformation

SECRET = "PHISECRET"
ZWNJ, ZWJ, VS16, KEYCAP = chr(0x200C), chr(0x200D), chr(0xFE0F), chr(0x20E3)
VIRAMA = chr(0x94D)                               # DEVANAGARI SIGN VIRAMA
KA, SSA = chr(0x915), chr(0x937)
HEART, SUN, GRIN = chr(0x2764), chr(0x2600), chr(0x1F600)
WOMAN, MAN, GIRL, BOY = chr(0x1F469), chr(0x1F468), chr(0x1F467), chr(0x1F466)
BEH, KHA, YEH, ALEF, HAMZA, HEH, FATHA = (chr(c) for c in (0x628, 0x62E, 0x6CC, 0x627, 0x621, 0x647, 0x64E))
MEEM = chr(0x645)
PRIVATE_USE = chr(0xE000)


def C(*codepoints):
    return "".join(chr(cp) for cp in codepoints)


def tamper_finding():
    f = finding()
    object.__setattr__(f, "stage", SECRET)
    return f


def forge(record, **changes):
    """A stored record with arbitrary content and a CORRECT hash (what an older version or a legacy writer produced)."""
    cls = type(record)
    fields = {k: getattr(record, k) for k in cls.model_fields if k != "content_hash"}
    fields.update(changes)
    raw = cls.model_construct(**fields, content_hash="")
    object.__setattr__(raw, "content_hash", raw.compute_hash())
    return cls.load(raw.model_dump_json())


def check(text, *, creating, layout=False):
    from app.workflow.base import check_text
    try:
        check_text(text, allow_layout=layout, creating=creating)
        return True
    except ValueError:
        return False


def no_secret(obj):
    return SECRET not in repr(obj) and SECRET not in json.dumps(obj, default=str)


# --- 1. mapping keys must be strings, and the error says nothing about them --------------------------------------
def test_integer_mapping_keys_are_refused_by_every_mapping_field_with_a_static_location():
    cases = [
        (lambda k: review(rule_versions={k: "v"}), ("rule_versions",)),
        (lambda k: execution(config={k: 1}), ("config",)),
        (lambda k: execution(config={"outer": {k: 1}}), ("config",)),
        (lambda k: execution(config={"outer": [{k: 1}]}), ("config",)),
        (lambda k: transformation(parameters={k: 1}), ("parameters",)),
    ]
    for build, expected in cases:
        with pytest.raises(ValidationError) as caught:
            build(8675309)
        locs = [e["loc"] for e in safe_errors(caught.value)]
        assert locs == [expected], locs
        assert no_secret(safe_errors(caught.value)) and "8675309" not in json.dumps(safe_errors(caught.value))


def test_integer_keys_in_estimates_and_bounds_leave_no_trace():
    from app.workflow.contracts import AssumptionSet
    for field in ("initial_estimates", "bounds"):
        with pytest.raises(ValidationError) as caught:
            AssumptionSet.create(id="a1", created_at=T0, producer="p", **{field: {8675309: 1.0}})
        assert "8675309" not in json.dumps(safe_errors(caught.value)), field


def test_a_frozen_map_is_simply_unequal_to_a_foreign_mapping_with_non_string_keys():
    assert FrozenMap({"a": 1}) != {1: "a"} and FrozenMap({}) != {1: 2}


# --- 2. safe_errors keeps list indices only where a sequence field makes them genuine ----------------------------
def test_safe_errors_keeps_a_genuine_list_index_and_drops_dynamic_integer_keys():
    with pytest.raises(ValidationError) as caught:
        dataset(columns=(ColumnSchema(name="ID", role="ID"), {"name": 5}))
    assert [e["loc"][:2] for e in safe_errors(caught.value)][0] == ("columns", 1)

    adapter = TypeAdapter(dict[int, str])
    with pytest.raises(ValidationError) as caught:
        adapter.validate_python({8675309: 5})
    cleaned = safe_errors(caught.value)
    assert "8675309" not in json.dumps(cleaned) and cleaned[0]["loc"] == ("<key>",)


def test_an_integer_key_inside_a_sequence_element_is_not_mistaken_for_an_index():
    with pytest.raises(ValidationError) as caught:
        review(bound_artifacts=({8675309: "x"},))
    cleaned = safe_errors(caught.value)
    assert "8675309" not in json.dumps(cleaned)
    assert cleaned and all(e["loc"][:2] == ("bound_artifacts", 0) for e in cleaned)


def test_a_mapping_key_that_happens_to_equal_a_schema_name_is_still_redacted():
    with pytest.raises(ValidationError) as caught:
        review(rule_versions={"id": chr(0)})
    assert [e["loc"] for e in safe_errors(caught.value)] == [("rule_versions", "<key>")]
    with pytest.raises(ValidationError) as caught:
        execution(config={"stage": chr(0)})
    assert [e["loc"] for e in safe_errors(caught.value)] == [("config",)]


def test_safe_errors_redacts_an_integer_after_a_mapping_field_even_for_foreign_shapes():
    from pydantic import BaseModel

    class Holder(BaseModel):
        config: dict[int, str]

    with pytest.raises(ValidationError) as caught:
        Holder(config={8675309: 5})
    assert "8675309" not in json.dumps(safe_errors(caught.value))


def test_safe_errors_fails_safely_for_anything_that_is_not_a_validation_error():
    for exc in (ValueError(SECRET), KeyError(SECRET), None, "text " + SECRET, object()):
        cleaned = safe_errors(exc)
        assert cleaned == [{"type": "unsupported_exception", "loc": (),
                            "msg": "error details are not available for this kind of exception"}], cleaned
    contract = safe_errors(ContractError("integrity check failed", detail=SECRET))
    assert contract == [{"type": "contract_error", "loc": (), "msg": "integrity check failed"}]


def test_safe_errors_does_not_echo_a_discriminated_union_tag():
    from typing import Annotated, Literal

    from pydantic import BaseModel, Field

    class One(BaseModel):
        kind: Literal["one"]

    class Two(BaseModel):
        kind: Literal["two"]

    adapter = TypeAdapter(Annotated[One | Two, Field(discriminator="kind")])
    with pytest.raises(ValidationError) as caught:
        adapter.validate_python({"kind": SECRET})
    assert SECRET in str(caught.value)                   # pydantic itself echoes it; safe_errors must not
    assert no_secret(safe_errors(caught.value))


# --- 3. ContractError keeps no raw exception, by attribute or by chain --------------------------------------------
def assert_clean_error(exc, *, text=None):
    assert exc.__cause__ is None and exc.__context__ is None
    assert not any(isinstance(a, BaseException) for a in exc.args)
    assert not any(isinstance(v, BaseException) for v in vars(exc).values())
    rendered = "".join(traceback.format_exception(exc)) + repr(exc) + str(exc) + json.dumps(exc.detail, default=str)
    assert SECRET not in rendered, rendered


def test_a_tamper_error_carries_no_raw_validation_error_and_no_phi():
    rec = review()
    object.__setattr__(rec, "rule_versions", FrozenMap({SECRET + "rk": "v" + chr(0) + "bad"}))
    for call in (rec.validated_copy, lambda: rec.replace(reason="x")):
        with pytest.raises(ContractError) as caught:
            call()
        assert_clean_error(caught.value)
        assert isinstance(caught.value.detail, list) and set(caught.value.detail[0]) == {"type", "loc", "msg"}


def test_a_tamper_error_does_not_reach_the_log_with_phi(caplog):
    rec = review()
    object.__setattr__(rec, "rule_versions", FrozenMap({SECRET + "rk": "v" + chr(0) + "bad"}))
    with caplog.at_level(logging.ERROR):
        try:
            rec.validated_copy()
        except ContractError:
            logging.getLogger("t").exception("integrity failure")
    assert SECRET not in caplog.text


def test_canonical_json_failures_carry_no_chain():
    from app.workflow.hashing import canonical_json
    with pytest.raises(ContractError) as caught:
        canonical_json({"a": float("nan")})
    assert caught.value.__cause__ is None and caught.value.__context__ is None


def test_projection_errors_carry_no_chain_and_no_state_content():
    from app.workflow.compat import project_session_row
    bad_rows = [
        {"state": {"dataset_metadata": SECRET}},
        {"state": {"nca_parameters": SECRET}},
        {"state": {"session_id": {SECRET: 1}}},
        {"state_json": json.dumps([SECRET])},
        {"state_json": '{"a": "' + SECRET + '" "b"'},
    ]
    for row in bad_rows:
        with pytest.raises(ContractError) as caught:
            project_session_row(row, as_of=T0)
        assert_clean_error(caught.value)


# --- 4. every representation protocol is redacted ------------------------------------------------------------------
def test_no_representation_protocol_shows_field_values():
    from pydantic import BaseModel
    col = ColumnSchema(name=SECRET + "ID")
    ds = dataset(source=SECRET + "_file.csv")
    for obj in (col, ds, finding(message=SECRET), Unknown(status="unknown", reason=SECRET)):
        assert list(obj.__repr_args__()) == []
        assert list(obj.__rich_repr__()) == []
        assert SECRET not in repr(obj) and SECRET not in str(obj)
        assert SECRET not in "".join(map(str, obj.__pretty__(lambda x: x))) if hasattr(obj, "__pretty__") else True
        assert SECRET not in BaseModel.__repr_str__(obj, ", ")
    import pprint
    assert SECRET not in pprint.pformat([col, ds])


def test_rich_pretty_printing_is_redacted():
    rich_pretty = pytest.importorskip("rich.pretty")
    ds = dataset(source=SECRET + "_file.csv")
    for obj in (ColumnSchema(name=SECRET + "ID"), ds, FrozenMap({SECRET: 1}), review(reason=SECRET)):
        assert SECRET not in rich_pretty.pretty_repr(obj)


# --- 5. the tamper path fixes the cause of the warning rather than hiding it ---------------------------------------
def test_tamper_detection_emits_no_warning_and_no_phi():
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        f = tamper_finding()
        assert f.verify_integrity() is False
        with pytest.raises(ContractError) as caught:
            f.validated_copy()
        assert_clean_error(caught.value)
        with pytest.raises(ContractError):
            f.canonical_bytes()
        with pytest.raises(ContractError):
            f.replace(severity="error")
    assert [str(w.message) for w in recorded] == []


def test_unserialisable_content_is_not_intact_but_does_not_raise_from_verify_integrity():
    f = finding()
    object.__setattr__(f, "stage", 5.5)
    assert f.verify_integrity() is False


# --- 6. compat helpers: an unrepresentable value degrades one field, and the context cannot seal -------------------
def test_a_private_use_or_unassigned_legacy_value_degrades_to_unknown_instead_of_failing_the_row():
    from app.core.pharmstate import PharmState
    from app.workflow.compat import project_pharmstate
    for bad in (PRIVATE_USE, chr(0x378)):
        state = PharmState(session_id="s" + bad, run_provenance_status="ok" + bad)
        p = project_pharmstate(state, as_of=T0)
        assert isinstance(p.session_id, Unknown) and isinstance(p.run_provenance_status, Unknown)
        assert p.session_id.reason == "legacy value is not representable as a label"
    ok = project_pharmstate(PharmState(session_id="s", run_provenance_status="ok"), as_of=T0)
    assert ok.session_id == "s" and ok.run_provenance_status == "ok"


def test_compat_never_passes_a_sealing_context():
    import app.workflow.compat as compat
    assert not hasattr(compat, "SEAL") and not hasattr(compat, "_LABEL_CHECK")


# --- 7. text policy: stricter joiner/VS16 rules at creation, load semantics for stored values -----------------------
CREATE_REFUSED = {
    "digit plus VS16 without a keycap": "DV1" + VS16,
    "hash plus VS16 without a keycap": "#" + VS16,
    "degree sign plus VS16": C(0xB0) + VS16,
    "copyright plus VS16": C(0xA9) + VS16,
    "double exclamation plus VS16 (outside the supported emoji blocks)": C(0x203C) + VS16,
    "left-right arrow plus VS16 (outside the supported emoji blocks)": C(0x2194) + VS16,
    "VS16 alone": VS16,
    "VS16 twice": HEART + VS16 + VS16,
    "keycap base plus VS16 plus the wrong follower": "5" + VS16 + "x",
    "ZWNJ after a right-joining letter": ALEF + ZWNJ + BEH,
    "ZWNJ before a non-joining letter": BEH + ZWNJ + HAMZA,
    "ZWNJ after a combining mark": BEH + FATHA + ZWNJ + BEH,
    "ZWNJ before a combining mark": BEH + ZWNJ + FATHA + BEH,
    "ZWNJ between Latin letters": "a" + ZWNJ + "b",
    "ZWNJ at the start": ZWNJ + BEH,
    "ZWNJ at the end": BEH + ZWNJ,
    "ZWJ after a Latin letter and a virama": "x" + VIRAMA + ZWJ,
    "ZWJ chain after a virama": KA + (VIRAMA + ZWJ) * 3,
    "ZWNJ chain after a virama": KA + VIRAMA + ZWNJ + VIRAMA + ZWNJ,
    "ZWJ after a lone virama": VIRAMA + ZWJ,
    "ZWJ between a keycap base and an emoji": "5" + ZWJ + GRIN,
    "ZWJ between two non-emoji symbols": C(0xB0) + ZWJ + C(0xA9),
    "ZWJ after a skin tone but before a non-emoji": WOMAN + chr(0x1F3FD) + ZWJ + "a",
    "eight emoji joined": GRIN + (ZWJ + GRIN) * 8,
    "six stacked combining marks": "a" + chr(0x301) * 6,
}
CREATE_ACCEPTED = {
    "plain": "Site 5",
    "keycap sequence": "5" + VS16 + KEYCAP,
    "hash keycap": "#" + VS16 + KEYCAP,
    "red heart": HEART + VS16,
    "sun": SUN + VS16,
    "grinning face with VS16": GRIN + VS16,
    "Persian miyam": MEEM + YEH + ZWNJ + KHA + chr(0x648) + ALEF + HEH + MEEM,
    "Persian plural suffix": C(0x6A9, 0x62A, 0x627, 0x628) + ZWNJ + HEH + ALEF,
    "Persian possessive": C(0x62E, 0x627, 0x646, 0x647) + ZWNJ + ALEF + MEEM,
    "Devanagari conjunct": KA + VIRAMA + ZWJ + SSA,
    "Devanagari half form": KA + VIRAMA + ZWNJ + SSA,
    "Devanagari with nukta": KA + chr(0x93C) + VIRAMA + ZWNJ + SSA,
    "Sinhala": C(0xD9A, 0xDCA, 0x200D, 0xDBB),
    "Malayalam chillu": C(0xD28, 0xD4D, 0x200D),
    "family": MAN + ZWJ + WOMAN + ZWJ + GIRL + ZWJ + BOY,
    "health worker with skin tone": WOMAN + chr(0x1F3FD) + ZWJ + chr(0x2695) + VS16,
    "rainbow flag": chr(0x1F3F3) + VS16 + ZWJ + chr(0x1F308),
    "seven ZWJ links": GRIN + (ZWJ + GRIN) * 7,
    "five stacked combining marks": "a" + chr(0x301) * 5,
}


@pytest.mark.parametrize("name", sorted(CREATE_REFUSED))
def test_creation_refuses_the_unsupported_joiner_and_mark_sequences(name):
    assert not check(CREATE_REFUSED[name], creating=True), name


@pytest.mark.parametrize("name", sorted(CREATE_ACCEPTED))
def test_creation_accepts_the_documented_supported_subset(name):
    assert check(CREATE_ACCEPTED[name], creating=True), name


@pytest.mark.parametrize("name", sorted(CREATE_REFUSED))
def test_stored_text_is_checked_under_load_semantics_so_context_rules_never_orphan_a_record(name):
    text = CREATE_REFUSED[name]
    assert check(text, creating=False), name


def test_load_semantics_still_refuse_the_characters_that_are_never_acceptable():
    for bad in (chr(0x200B), chr(0x202E), chr(0xFE00), chr(0x3164), chr(0x2800), chr(0x7), chr(0x2028)):
        assert not check("a" + bad + "b", creating=False)


def test_the_run_caps_are_pinned():
    from app.workflow import base
    assert base.MAX_MARK_RUN == 5 and base.MAX_ZWJ_LINKS == 7


def test_a_dataset_cannot_be_created_with_two_visually_identical_column_names():
    twin = ColumnSchema(name="DV1" + VS16)                       # constructing a nested value is load semantics
    assert twin.name == "DV1" + VS16
    with pytest.raises(ValidationError):
        dataset(columns=(ColumnSchema(name="DV1"), twin), subject_id_column="DV1")


# --- 8. replace() applies the creation policy to changed values only -----------------------------------------------
def test_replace_does_not_re_reject_unchanged_stored_text():
    legacy = forge(finding(), producer="p" + PRIVATE_USE, message="DV1" + VS16 + " is odd")
    assert legacy.producer.endswith(PRIVATE_USE)
    updated = legacy.replace(severity="error")
    assert updated.producer == legacy.producer and updated.message == legacy.message and updated.severity == "error"
    assert updated.verify_integrity() and updated.content_hash != legacy.content_hash


def test_replace_still_applies_the_creation_policy_to_every_changed_value():
    legacy = forge(finding(), producer="p" + PRIVATE_USE)
    for changes in ({"message": "x" + PRIVATE_USE}, {"message": "DV1" + VS16}, {"producer": "q" + PRIVATE_USE}):
        with pytest.raises(ValidationError):
            legacy.replace(**changes)


def test_replace_applies_the_policy_inside_a_changed_nested_value():
    rec = review()
    with pytest.raises(ValidationError):
        rec.replace(comments=("fine", "bad " + PRIVATE_USE))
    assert rec.replace(comments=("fine",)).comments == ("fine",)


def test_replace_keeps_the_review_name_lint_for_new_text():
    rec = review()
    with pytest.raises(ValidationError):
        rec.replace(actor_label="Claude Assistant")


def test_load_accepts_stored_text_that_creation_would_refuse_and_create_does_not():
    legacy = forge(finding(), message="DV1" + VS16)
    assert Finding.load(legacy.model_dump_json()) == legacy
    with pytest.raises(ValidationError):
        finding(message="DV1" + VS16)


# --- 9. JSON content is text-checked recursively --------------------------------------------------------------------
JSON_FIELDS = [
    (lambda v: execution(config=v), "config"),
    (lambda v: transformation(parameters=v), "parameters"),
    (lambda v: review(legacy_metadata=v), "legacy_metadata"),
]


@pytest.mark.parametrize("bad", [chr(0x202E), chr(0), chr(0x1B), chr(0x200B), chr(0x2028), PRIVATE_USE, chr(0x3164)])
def test_json_keys_and_string_values_are_text_checked_at_every_depth(bad):
    shapes = [{"k": "a" + bad}, {"a" + bad: 1}, {"k": {"n": "a" + bad}}, {"k": ["x", {"n": ["a" + bad]}]},
              {"k": {"a" + bad: [1]}}]
    for build, _ in JSON_FIELDS:
        for shape in shapes:
            with pytest.raises(ValidationError):
                build(shape)


def test_json_content_may_still_hold_layout_whitespace_and_empty_strings_and_ordinary_text():
    value = {"where": "TIME<0\n\tAND DV>0", "empty": "", "unit": chr(0xB5) + "g/mL", "n": [1, 2.5, None, True]}
    assert execution(config=value).config["empty"] == ""
    assert execution(config={"k": ["x", {"deep": "ok"}]}).verify_integrity()


def test_json_keys_do_not_allow_layout_whitespace():
    with pytest.raises(ValidationError):
        execution(config={"a\nb": 1})


def test_json_text_loaded_from_storage_is_not_re_rejected_for_creation_only_rules():
    rec = forge(execution(), config={"note": "a" + PRIVATE_USE})
    assert ExecutionRecord.load(rec.model_dump_json()) == rec
    assert rec.replace(seed=8).config["note"] == "a" + PRIVATE_USE


# --- 10. contracts that nest their own class are refused ------------------------------------------------------------
class SelfNest(WorkflowContract):
    RECORD_TYPE = "SelfNestT4"
    child: "SelfNest | None" = None


class MutualA(WorkflowContract):
    RECORD_TYPE = "MutualAT4"
    b: "MutualB | None" = None


class MutualB(WorkflowContract):
    RECORD_TYPE = "MutualBT4"
    a: "MutualA | None" = None


MutualA.model_rebuild()
MutualB.model_rebuild()


def test_a_contract_cannot_seal_a_nested_child_of_its_own_class():
    child = {"id": "s2", "created_at": T0, "producer": "p"}
    with pytest.raises(ContractError, match="nest"):
        SelfNest.create(id="s1", created_at=T0, producer="p", child=child)
    assert SelfNest.create(id="s1", created_at=T0, producer="p").child is None


def test_a_contract_cannot_launder_its_own_class_through_another_record():
    inner = MutualA.create(id="a2", created_at=T0, producer="p")
    legit_b = MutualB.create(id="b1", created_at=T0, producer="p", a=inner)
    stripped = json.loads(legit_b.model_dump_json())
    stripped["a"].pop("content_hash")
    with pytest.raises(ContractError, match="nest"):
        MutualA.create(id="a1", created_at=T0, producer="p", b=stripped)
    with pytest.raises(ContractError, match="nest"):
        MutualA.create(id="a1", created_at=T0, producer="p", b=legit_b)


def test_a_different_class_nested_inside_a_record_is_still_allowed():
    assert MutualB.create(id="b9", created_at=T0, producer="p", a=MutualA.create(id="a9", created_at=T0, producer="p"))
    from app.workflow.compat import LegacyProjection
    assert LegacyProjection.model_fields["dataset"] is not None


# --- 11. load() is a bounded, strict public boundary ---------------------------------------------------------------
def test_load_refuses_an_oversized_document_before_parsing_it():
    from app.workflow.base import MAX_RECORD_BYTES
    text = finding().model_dump_json()
    ok = text + " " * (MAX_RECORD_BYTES - len(text))
    assert Finding.load(ok) == finding() and Finding.load(ok.encode()) == finding()
    for too_big in (ok + " ", (ok + " ").encode(), bytearray((ok + " ").encode())):
        with pytest.raises(ContractError, match="too large"):
            Finding.load(too_big)


def test_load_refuses_duplicate_keys_wherever_they_appear():
    good = json.loads(finding().model_dump_json())
    text = finding().model_dump_json()
    dup_top = text[:-1] + ',"message":"DECOY"}'
    dup_hash = text[:-1] + ',"content_hash":"' + "0" * 64 + '"}'
    nested = ReviewRecord.load(review().model_dump_json())
    nested_text = nested.model_dump_json().replace('"artifact_id"', '"artifact_id":"zzz","artifact_id"', 1)
    for doc, cls in ((dup_top, Finding), (dup_hash, Finding), (nested_text, ReviewRecord)):
        with pytest.raises(ContractError, match="duplicate") as caught:
            cls.load(doc)
        assert_clean_error(caught.value)
        with pytest.raises(ContractError, match="duplicate"):
            cls.load(doc.encode())
    assert Finding.load(json.dumps(good)) == finding()


def test_load_duplicate_key_error_does_not_echo_the_key():
    doc = finding().model_dump_json()[:-1] + ',"' + SECRET + '":1,"' + SECRET + '":2}'
    with pytest.raises(ContractError) as caught:
        Finding.load(doc)
    assert_clean_error(caught.value)


def test_load_turns_absurd_nesting_into_a_contract_error_not_a_crash():
    doc = '{"a":' + "[" * 100000 + "]" * 100000 + "}"
    with pytest.raises((ContractError, ValidationError)):
        Finding.load(doc)


def test_load_keeps_reporting_malformed_json_as_a_validation_error_and_accepts_mappings():
    with pytest.raises(ValidationError) as caught:
        Finding.load('{"id": "x",')
    assert no_secret(safe_errors(caught.value))
    assert Finding.load(json.loads(finding().model_dump_json())) == finding()
    assert artifact().verify_integrity() and Finding.load(finding().model_dump_json()).verify_integrity()


def test_load_accepts_a_bytearray_and_utf8_text():
    f = finding(message="caf" + chr(0xE9) + " " + chr(0x5F20))
    assert Finding.load(bytearray(f.model_dump_json().encode("utf-8"))) == f


# --- 12. documentation states the supported subset precisely ---------------------------------------------------------
def test_the_doc_describes_the_supported_subset_and_makes_no_contextj_claim():
    from pathlib import Path
    doc = (Path(__file__).resolve().parents[3] / "docs" / "architecture" / "workflow-orchestration.md").read_text()
    low = doc.lower()
    assert "not idna contextj" in low
    assert "contextj-style" not in low
    for needle in ("keycap", "pinned", "virama", "load semantics", "creation policy", "duplicate", "string keys",
                   "maximum record size"):
        assert needle in low, needle


# --- 13. boundaries and tables pinned (found by mutation testing) -------------------------------------------------
DUAL = [0x626, 0x628, 0x649, 0x64A, 0x67E, 0x686, 0x6A9, 0x6AF, 0x6CC, *range(0x62A, 0x62F), *range(0x633, 0x63B),
        *range(0x641, 0x648)]
RIGHT = [0x627, 0x629, 0x648, 0x698, *range(0x622, 0x626), *range(0x62F, 0x633)]
NEITHER = [0x621, 0x640, 0x63B, 0x63F, 0x6D5, 0x62F + 0x100]


def test_every_pinned_persian_letter_behaves_as_documented_around_a_zwnj():
    for left in DUAL:
        for right in DUAL + RIGHT:
            assert check(C(left, 0x200C, right), creating=True), (hex(left), hex(right))
    for left in RIGHT + NEITHER:
        assert not check(C(left, 0x200C, 0x628), creating=True), hex(left)
    for right in NEITHER:
        assert not check(C(0x628, 0x200C, right), creating=True), hex(right)
    from app.workflow import base
    assert base._DUAL_JOINING == frozenset(DUAL) and base._RIGHT_JOINING == frozenset(RIGHT)
    import unicodedata
    for cp in DUAL + RIGHT:
        assert unicodedata.category(chr(cp)).startswith("L") and "ARABIC" in unicodedata.name(chr(cp))


def test_every_skin_tone_modifier_joins_an_emoji_sequence():
    for cp in range(0x1F3FB, 0x1F400):
        assert check(WOMAN + chr(cp) + ZWJ + chr(0x2695) + VS16, creating=True), hex(cp)


def test_the_emoji_block_edges_are_exact():
    edges = {0x25FF: False, 0x2600: True, 0x27BF: True, 0x27C0: False, 0x1F000: True, 0x1FAFF: False}
    for cp, accepted in edges.items():
        import unicodedata
        if unicodedata.category(chr(cp)) != "So":
            continue
        assert check(chr(cp) + VS16, creating=True) == accepted, hex(cp)
    assert check(chr(0x1F9C0) + VS16, creating=True)


def test_a_virama_needs_a_letter_of_its_own_script_directly_before_it_or_a_nukta():
    zero = chr(0x966)                                              # DEVANAGARI DIGIT ZERO: same script, not a letter
    for text in (zero + VIRAMA + ZWJ + KA, KA + "x" + VIRAMA + ZWJ + KA, KA + VIRAMA + VIRAMA + ZWJ + KA,
                 "x" + VIRAMA + ZWNJ + KA, chr(0x995) + VIRAMA + ZWJ + KA, KA + chr(0x93C) * 6 + VIRAMA + ZWJ + KA):
        assert not check(text, creating=True), [hex(ord(c)) for c in text]
    assert check(KA + chr(0x93C) * 5 + VIRAMA + ZWJ + KA, creating=True) is False   # nukta x5 + virama = 6 marks
    assert check(KA + chr(0x93C) * 4 + VIRAMA + ZWJ + KA, creating=True)


def test_run_caps_reset_at_the_right_places():
    assert check("a" + chr(0x301) * 3 + "\n" + chr(0x301) * 3, creating=True, layout=True)
    assert check("a" + chr(0x301) * 3 + "b" + chr(0x301) * 3, creating=True)
    assert check(KA + VIRAMA + ZWJ + chr(0x301) * 5, creating=True)
    assert check(GRIN + (ZWJ + GRIN) * 4 + "a" + GRIN + (ZWJ + GRIN) * 4, creating=True)
    assert check(GRIN + (ZWJ + GRIN) * 4 + "\n" + GRIN + (ZWJ + GRIN) * 4, creating=True, layout=True)
    assert check(GRIN + (ZWJ + GRIN) * 7 + chr(0x4E2D) + GRIN + (ZWJ + GRIN) * 7, creating=True)
    assert not check(GRIN + (ZWJ + GRIN) * 4 + VS16 + (ZWJ + GRIN) * 4, creating=True)


def test_an_unassigned_code_point_in_an_emoji_block_is_not_an_emoji():
    import unicodedata
    unassigned = next(cp for cp in range(0x1FAFF, 0x1F000, -1) if unicodedata.category(chr(cp)) == "Cn")
    assert not check(chr(unassigned) + VS16, creating=True) and not check(GRIN + ZWJ + chr(unassigned), creating=True)


def test_the_creation_policy_marker_must_be_exactly_true_in_a_mapping_context():
    from pydantic import TypeAdapter

    from app.workflow.base import CREATE_POLICY, Label
    adapter = TypeAdapter(Label)
    text = "a" + PRIVATE_USE
    for context in (None, "creating", [CREATE_POLICY], {CREATE_POLICY: False}, {CREATE_POLICY: 1}, {}, {"other": True}):
        assert adapter.validate_python(text, context=context) == text, context
    with pytest.raises(ValidationError):
        adapter.validate_python(text, context={CREATE_POLICY: True})


def test_replace_validates_each_changed_field_and_sealing_stays_with_the_top_level():
    legacy = forge(finding(), producer="p" + PRIVATE_USE)
    assert legacy.replace(code="other.code").producer == legacy.producer
    with pytest.raises(ValidationError):
        legacy.replace(code="bad" + PRIVATE_USE)
    with pytest.raises(TypeError, match="seals the record itself"):
        legacy.replace(content_hash="0" * 64)
    with pytest.raises(TypeError, match="seals the record itself"):
        Finding.create(**{**json.loads(finding().canonical_bytes()), "content_hash": "0" * 64})


def test_replace_also_refuses_a_nested_record_of_its_own_class():
    base = SelfNest.create(id="s1", created_at=T0, producer="p")
    with pytest.raises(ContractError, match="nest"):
        base.replace(child=SelfNest.create(id="s2", created_at=T0, producer="p"))


class TupleNest(WorkflowContract):
    RECORD_TYPE = "TupleNestT4"
    kids: "tuple[TupleNest, ...]" = ()


def test_nesting_is_refused_inside_sequences_and_mappings_too():
    kid = TupleNest.create(id="k1", created_at=T0, producer="p")
    with pytest.raises(ContractError, match="nest"):
        TupleNest.create(id="t1", created_at=T0, producer="p", kids=(kid,))
    from app.workflow.base import Frozen, _refuse_self_nesting
    holder = TupleNest.model_construct(id="t2", created_at=T0, producer="p", kids=())
    object.__setattr__(holder, "kids", (FrozenMap({"deep": (kid,)}),))
    with pytest.raises(ContractError, match="nest"):
        _refuse_self_nesting(holder)
    assert issubclass(TupleNest, Frozen)


def test_oversize_and_duplicate_checks_have_exact_boundaries():
    from app.workflow.base import MAX_RECORD_BYTES
    assert MAX_RECORD_BYTES == 4 * 1024 * 1024
    text = finding().model_dump_json()
    with pytest.raises(ContractError, match="too large"):
        Finding.load(text + " " * (MAX_RECORD_BYTES - len(text) + 1))
    assert Finding.load(text + " " * (MAX_RECORD_BYTES - len(text))) == finding()
    unique_but_similar = text[:-1] + ',"message2":"x"}'
    with pytest.raises(ValidationError):
        Finding.load(unique_but_similar)                              # unknown field, but NOT reported as a duplicate


def test_deeply_nested_json_is_a_contract_error_with_a_fixed_message():
    doc = "[" * 200000 + "]" * 200000
    with pytest.raises(ContractError, match="nested too deeply") as caught:
        Finding.load(doc)
    assert_clean_error(caught.value)


def test_direct_helpers_chain_nothing():
    import datetime as dt

    from app.workflow.base import _utf8, to_utc
    with pytest.raises(ValueError, match="UTF-8") as caught:
        _utf8(chr(0xD800))
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    far = dt.datetime(9999, 12, 31, 23, 59, tzinfo=dt.timezone(dt.timedelta(hours=-23)))
    with pytest.raises(ValueError, match="representable") as caught:
        to_utc(far)
    assert caught.value.__cause__ is None and caught.value.__context__ is None


def test_a_hostile_current_mapping_gives_a_clean_contract_error():
    from collections.abc import Mapping

    class Hostile(Mapping):
        def __iter__(self):
            raise RuntimeError(SECRET)

        def __len__(self):
            return 1

        def __getitem__(self, key):
            raise KeyError(key)

    with pytest.raises(ContractError) as caught:
        review().binding_status(Hostile())
    assert_clean_error(caught.value)


def test_safe_errors_caps_long_messages_and_handles_name_collisions_between_models():
    from typing import Literal

    from pydantic import BaseModel

    many = tuple(f"option-{i:03d}" for i in range(60))
    with pytest.raises(ValidationError) as caught:
        TypeAdapter(Literal[many]).validate_python("nope")
    assert len(caught.value.errors()[0]["msg"]) > 200
    assert all(len(e["msg"]) <= 200 for e in safe_errors(caught.value))

    from collections.abc import Mapping as AbcMapping

    from app.workflow.base import Frozen

    class OneKind(Frozen):
        collide: tuple[int, ...] = ()
        viewed: AbcMapping[str, int] = {}

    class OtherKind(Frozen):
        collide: dict[str, int] = {}

    with pytest.raises(ValidationError) as caught:
        OneKind(collide=(1, "x"))
    assert safe_errors(caught.value)[0]["loc"][1] == "<key>"           # 'collide' is also a mapping field elsewhere
    with pytest.raises(ValidationError) as caught:
        OneKind(viewed={"id": "x"})
    assert safe_errors(caught.value)[0]["loc"] == ("viewed", "<key>")      # 'id' is a schema name but here a key
    assert issubclass(OtherKind, BaseModel)


def test_safe_errors_trusts_only_one_index_level_and_never_a_bool():
    from app.workflow.errors import _clean_loc
    names, sequences = frozenset({"columns", "name"}), frozenset({"columns"})
    assert _clean_loc(("columns", 3, "name"), names, sequences, frozenset()) == ("columns", 3, "name")
    assert _clean_loc(("columns", 3, 4), names, sequences, frozenset()) == ("columns", 3, "<key>")
    assert _clean_loc((3,), names, sequences, frozenset()) == ("<key>",)
    assert _clean_loc(("columns", True), names, sequences, frozenset()) == ("columns", "<key>")
    assert _clean_loc(("nope", "name"), names, sequences, frozenset()) == ("<key>", "name")


# --- 14. more mutation-driven pins ------------------------------------------------------------------------------------
def test_vs16_needs_a_symbol_other_inside_the_documented_blocks():
    import unicodedata
    for cp in (0x2768, 0x2776, 0x1F100, 0x2122, 0x23F0, 0x25B6, 0x2B50, 0x1FB00):
        assert not check(chr(cp) + VS16, creating=True), hex(cp)
        if unicodedata.category(chr(cp)) == "So":
            assert not check(GRIN + ZWJ + chr(cp), creating=True), hex(cp)      # ZWJ needs a supported pictograph too


def test_the_redaction_layers_do_not_depend_on_each_other(monkeypatch):
    from app.workflow.base import Frozen
    monkeypatch.setattr(Frozen, "__repr_args__", lambda self: iter([("name", SECRET)]))
    col = ColumnSchema(name="ID")
    assert repr(col) == "ColumnSchema(<redacted>)" and str(col) == "ColumnSchema(<redacted>)"
    assert list(col.__rich_repr__()) == [] and list(col.__pretty__(lambda x: x)) == [repr(col)]
    rec = finding()
    assert repr(rec) == f"Finding(hash={rec.content_hash[:12]})"


def test_pretty_printing_yields_exactly_the_redacted_form():
    for obj in (ColumnSchema(name=SECRET), Unknown(status="unknown", reason=SECRET), finding(message=SECRET)):
        assert list(obj.__pretty__(lambda x: x)) == [repr(obj)]


def test_union_and_annotated_and_list_field_shapes_all_keep_their_genuine_indices():
    from typing import Annotated, Optional

    from pydantic import Field

    from app.workflow.base import Frozen

    class Shapes(Frozen):
        old_union: Optional[tuple[int, ...]] = None            # noqa: UP045  (typing.Union on purpose)
        new_union: tuple[int, ...] | None = None
        wrapped: Optional[Annotated[tuple[int, ...], Field(max_length=5)]] = None  # noqa: UP045
        plain_list: list[int] = []

    for field in ("old_union", "new_union", "wrapped", "plain_list"):
        with pytest.raises(ValidationError) as caught:
            Shapes(**{field: (1, "x")})
        assert safe_errors(caught.value)[0]["loc"][:2] == (field, 1), field


def test_a_legacy_file_name_with_a_private_use_character_degrades_only_the_source():
    from app.core.pharmstate import PharmState
    from app.workflow.compat import project_pharmstate
    meta = {"dataset_id": "ds_1", "n_subjects": 3, "detected_roles": {"ID": "ID"}, "columns": [],
            "dataset_sha256": "a" * 64}
    state = PharmState(session_id="s", dataset_id="ds_1", dataset_metadata=meta,
                       dataset_path="/data/a" + PRIVATE_USE + "b.csv")
    p = project_pharmstate(state, as_of=T0)
    assert not isinstance(p.dataset, Unknown) and isinstance(p.dataset.source, Unknown)
    assert p.dataset.source.reason == "the data file name is not representable as a label"


def test_empty_legacy_labels_say_so_rather_than_claiming_they_are_unrepresentable():
    from app.core.pharmstate import PharmState
    from app.workflow.compat import project_pharmstate
    p = project_pharmstate(PharmState(session_id="", run_provenance_status=""), as_of=T0)
    assert p.session_id.reason == "legacy state has no session id"
    assert p.run_provenance_status.reason == "legacy state records no provenance status"
