"""Fifth hardening round: replace() uses its own validated values and hides inputs, advisory lint judges only new text,
the nesting walk is complete, load() is exact-typed and byte-bounded, labels are NFC with ASCII spaces only.

Non-ASCII characters are always built with ``chr()``: the hygiene test forbids them in source."""
import json
import traceback
import unicodedata
from collections.abc import Mapping

import pytest
from pydantic import ValidationError

from app.workflow.base import FrozenMap
from app.workflow.contracts import ColumnSchema, ExecutionRecord, Finding, ReviewRecord
from app.workflow.errors import ContractError, safe_errors
from tests.workflow.factories import T0, dataset, execution, finding, review
from tests.workflow.test_followup4 import (
    GRIN,
    PRIVATE_USE,
    SECRET,
    VS16,
    ZWJ,
    TupleNest,
    assert_clean_error,
    check,
    forge,
)

NUL = chr(0)
E_ACUTE = chr(0xE9)
E_DECOMPOSED = "e" + chr(0x301)
NBSP = chr(0xA0)


def rendered(exc):
    return "".join(traceback.format_exception(exc)) + repr(exc) + str(exc)


# --- 1. replace() uses the value it validated, never re-reading the caller's object --------------------------------
def test_replace_keeps_every_element_of_a_one_shot_iterator():
    assert execution().replace(parents=(p for p in ["parentA", "parentB"])).parents == ("parentA", "parentB")
    two = (ColumnSchema(name="ID", role="ID"), ColumnSchema(name="DV", role="DV"))
    assert len(dataset().replace(columns=(c for c in two)).columns) == 2


class Shifty(Mapping):
    """A mapping whose values change after the first read of each: clean first, tainted afterwards."""

    def __init__(self):
        self.reads = 0

    def __iter__(self):
        return iter(["k"])

    def __len__(self):
        return 1

    def __getitem__(self, key):
        self.reads += 1
        return "clean" if self.reads == 1 else "tainted" + PRIVATE_USE


def test_replace_cannot_be_driven_by_a_mapping_that_changes_between_reads():
    shifty = Shifty()
    result = execution().replace(config=shifty)
    assert result.config["k"] == "clean" and PRIVATE_USE not in json.dumps(result.config.to_plain())
    assert shifty.reads == 1                                   # read once: thaw takes one snapshot of the mapping


# --- 2. replace() hides the input and keeps only the static field name ---------------------------------------------
def test_a_replace_failure_never_shows_the_offending_value():
    for field, value in (("actor_label", SECRET + NUL), ("reason", SECRET + NUL), ("producer", SECRET + NUL),
                         ("comments", (SECRET + NUL,))):
        with pytest.raises(ValidationError) as caught:
            review().replace(**{field: value})
        assert SECRET not in rendered(caught.value), field
        assert [e["loc"][0] for e in safe_errors(caught.value)] == [field] * len(safe_errors(caught.value)), field
        assert SECRET not in json.dumps(safe_errors(caught.value))


def test_a_replace_failure_hides_json_input_like_create_does():
    with pytest.raises(ValidationError) as via_replace:
        execution().replace(config={"a": SECRET + NUL})
    with pytest.raises(ValidationError) as via_create:
        execution(config={"a": SECRET + NUL})
    assert SECRET not in rendered(via_replace.value) and SECRET not in rendered(via_create.value)
    assert [e["loc"] for e in safe_errors(via_replace.value)] == [e["loc"] for e in safe_errors(via_create.value)]
    with pytest.raises(ValidationError) as keyed:
        review().replace(rule_versions={SECRET: NUL})
    with pytest.raises(ValidationError) as keyed_create:
        review(rule_versions={SECRET: NUL})
    assert (SECRET in str(keyed.value)) == (SECRET in str(keyed_create.value))      # same exposure as create()
    assert SECRET not in json.dumps(safe_errors(keyed.value))


def test_unknown_field_names_are_never_echoed():
    rec = finding()
    for call in (lambda: rec.replace(**{SECRET: 1}), lambda: rec.model_copy(update={SECRET: 1})):
        with pytest.raises(ContractError, match="unknown fields") as caught:
            call()
        assert_clean_error(caught.value)
        assert caught.value.detail == {"code": "unknown_field", "count": 1}
        assert SECRET not in rendered(caught.value)
    with pytest.raises(ContractError) as caught:
        rec.replace(**{SECRET: 1, SECRET + "2": 2})
    assert caught.value.detail == {"code": "unknown_field", "count": 2}


# --- 3. the advisory name lint judges only new or changed labels -----------------------------------------------------
def test_replacing_the_reason_does_not_re_lint_an_unchanged_stored_label():
    stored = forge(review(), actor_label="Claude Assistant")
    assert stored.actor_label == "Claude Assistant"
    updated = stored.replace(reason="re-reviewed")
    assert updated.actor_label == "Claude Assistant" and updated.reason == "re-reviewed"
    assert updated.verify_integrity()


def test_a_new_or_changed_label_is_still_linted():
    with pytest.raises(ValidationError, match="automated"):
        review(actor_label="Claude Assistant")
    with pytest.raises(ValidationError, match="automated"):
        review().replace(actor_label="Claude Assistant")
    with pytest.raises(ValidationError, match="automated"):
        forge(review(), actor_label="Claude Assistant").replace(actor_label="Copilot Bot")
    assert review().replace(actor_label="Dr. Bob Smith").actor_label == "Dr. Bob Smith"


def test_stored_labels_are_never_linted_on_load_or_validation():
    stored = forge(review(), actor_label="Claude Assistant")
    assert ReviewRecord.load(stored.model_dump_json()) == stored and stored.validated_copy() == stored


# --- 4. the self-nesting walk is complete and cycle-safe ----------------------------------------------------------------
def holder_with(kids):
    kid = TupleNest.create(id="k1", created_at=T0, producer="p")
    holder = TupleNest.model_construct(id="t2", created_at=T0, producer="p", kids=())
    object.__setattr__(holder, "kids", kids(kid))
    return holder


@pytest.mark.parametrize("wrap", [
    lambda k: [k], lambda k: (k,), lambda k: {k}, lambda k: frozenset({k}), lambda k: ((k,),), lambda k: [[k]],
    lambda k: {"x": k}, lambda k: FrozenMap({"x": [k]}), lambda k: {"x": {"y": (k,)}},
], ids=["list", "tuple", "set", "frozenset", "tuple-of-tuples", "list-of-lists", "dict", "frozenmap-list", "deep-dict"])
def test_a_same_class_record_is_found_inside_any_container(wrap):
    from app.workflow.base import _refuse_self_nesting
    with pytest.raises(ContractError, match="nest"):
        _refuse_self_nesting(holder_with(wrap))


def test_a_same_class_record_is_found_inside_an_arbitrary_pydantic_model():
    from pydantic import BaseModel

    from app.workflow.base import _refuse_self_nesting

    class Box(BaseModel):
        inner: object = None

    with pytest.raises(ContractError, match="nest"):
        _refuse_self_nesting(holder_with(lambda k: Box(inner=Box(inner=[k]))))


def test_the_nesting_walk_terminates_on_cycles_and_shared_objects():
    from app.workflow.base import _refuse_self_nesting
    loop: list = []
    loop.append(loop)
    shared = (1, 2)
    for kids in (loop, [shared, shared, {"a": shared}], [loop, loop]):
        _refuse_self_nesting(holder_with(lambda k, kids=kids: kids))                     # no error, no hang
    mapping: dict = {}
    mapping["self"] = mapping
    _refuse_self_nesting(holder_with(lambda k: mapping))


# --- 5. the adapter cache is bounded -------------------------------------------------------------------------------------
def test_the_per_field_validator_cache_is_bounded():
    from app.workflow.base import _field_model
    info = _field_model.cache_info()
    assert isinstance(info.maxsize, int) and 0 < info.maxsize <= 1024


# --- 6. load() takes exact str, bytes or bytearray only; the bound is in bytes -------------------------------------
def test_load_refuses_subclasses_of_str_and_bytes():
    good = finding().model_dump_json()

    class Bytes(bytes):
        pass

    class Text(str):
        pass

    class Liar(bytes):
        def __len__(self):
            return 5

        def decode(self, *args, **kwargs):
            return good

    for odd in (Bytes(good.encode()), Text(good), Liar(b"x" * (5 * 1024 * 1024))):
        with pytest.raises(ContractError, match="exact") as caught:
            Finding.load(odd)
        assert_clean_error(caught.value)
    assert Finding.load(good) == Finding.load(good.encode()) == Finding.load(bytearray(good.encode()))


def test_load_reads_a_mutable_buffer_once():
    good = bytearray(finding().model_dump_json().encode())
    assert Finding.load(good) == finding() and good == bytearray(finding().model_dump_json().encode())


def test_the_load_bound_is_in_utf8_bytes_for_text_too():
    from app.workflow.base import MAX_RECORD_BYTES
    wide = chr(0x4E2D) * (MAX_RECORD_BYTES // 3 + 10)             # fewer characters than the bound, more bytes
    assert len(wide) < MAX_RECORD_BYTES < len(wide.encode())
    with pytest.raises(ContractError, match="too large"):
        Finding.load(wide)
    with pytest.raises(ContractError, match="too large"):
        Finding.load(wide.encode())
    text = finding().model_dump_json()
    assert Finding.load(text + " " * (MAX_RECORD_BYTES - len(text))) == finding()


def test_a_lone_surrogate_in_a_text_document_is_a_clean_failure():
    with pytest.raises((ValidationError, ContractError)) as caught:
        Finding.load('{"id": "' + chr(0xD800) + '"}')
    assert chr(0xD800) not in str(caught.value) or isinstance(caught.value, ValidationError)


# --- 7. excessive nesting in load() is one deterministic ContractError --------------------------------------------------
@pytest.mark.parametrize("doc", ["[" * 100000 + "]" * 100000, '{"a":' * 100000 + "1" + "}" * 100000,
                                 "[" * 73 + "]" * 73, '{"a":' * 73 + "1" + "}" * 73, "[" * 1000000],
                         ids=["list-100k", "dict-100k", "list-73", "dict-73", "list-1m-unclosed"])
def test_excessive_json_nesting_is_always_the_same_contract_error(doc):
    with pytest.raises(ContractError, match="nested too deeply") as caught:
        Finding.load(doc)
    assert_clean_error(caught.value)
    with pytest.raises(ContractError, match="nested too deeply"):
        Finding.load(doc.encode())


def test_excessive_nesting_in_a_python_mapping_is_the_same_contract_error():
    deep: dict = {}
    cursor = deep
    for _ in range(200):
        cursor["a"] = {}
        cursor = cursor["a"]
    cycle: dict = {}
    cycle["a"] = cycle
    inner: list = []
    listed: dict = {"x": inner}
    for _ in range(100):
        inner.append([])
        inner = inner[0]
    for hostile in (deep, cycle, listed):
        with pytest.raises(ContractError, match="nested too deeply"):
            Finding.load(hostile)


def test_the_deepest_legal_record_still_loads():
    config: dict = {}
    cursor = config
    for _ in range(63):
        cursor["a"] = {}
        cursor = cursor["a"]
    rec = execution(config=config)
    assert ExecutionRecord.load(rec.model_dump_json()) == rec and ExecutionRecord.load(json.loads(rec.model_dump_json())) == rec


# --- 8. labels are NFC with ASCII spaces only, at creation and at load -----------------------------------------------
def label_makers():
    return [lambda t: finding(producer=t), lambda t: review(actor_label=t), lambda t: ColumnSchema(name=t),
            lambda t: dataset(source=t), lambda t: review(rule_versions={t: "v"}), lambda t: review(rule_versions={"k": t})]


def test_labels_must_already_be_in_nfc_and_are_never_rewritten():
    assert unicodedata.normalize("NFC", E_DECOMPOSED) == E_ACUTE and E_DECOMPOSED != E_ACUTE
    for make in label_makers():
        assert make("caf" + E_ACUTE) is not None
        with pytest.raises(ValidationError, match="NFC"):
            make("caf" + E_DECOMPOSED)


def test_only_the_ascii_space_is_allowed_inside_a_label():
    spaces = [chr(cp) for cp in range(0x110000) if unicodedata.category(chr(cp)) == "Zs"]
    assert chr(0x20) in spaces and NBSP in spaces and chr(0x3000) in spaces and len(spaces) >= 17
    for make in label_makers():
        assert make("a b") is not None
        for space in spaces:
            if space == " ":
                continue
            with pytest.raises(ValidationError):
                make("a" + space + "b")


def test_canonical_form_checks_also_apply_when_loading_a_stored_record():
    for field, bad in (("producer", "caf" + E_DECOMPOSED), ("producer", "a" + NBSP + "b")):
        with pytest.raises(ValidationError):
            forge(finding(), **{field: bad})
    stored = json.loads(finding().model_dump_json())
    stored["producer"] = "a" + NBSP + "b"
    with pytest.raises(ValidationError):
        Finding.load(stored)


def test_free_text_messages_and_json_values_keep_their_wider_rules():
    assert finding(message="a" + NBSP + "b").message == "a" + NBSP + "b"
    assert finding(message="caf" + E_DECOMPOSED).message == "caf" + E_DECOMPOSED
    assert execution(config={"note": "caf" + E_DECOMPOSED + NBSP}).config["note"] == "caf" + E_DECOMPOSED + NBSP


def test_json_keys_are_nfc_and_ascii_space_only_values_are_not():
    assert execution(config={"caf" + E_ACUTE: 1, "a b": 2}).config["a b"] == 2
    for bad_key in ("caf" + E_DECOMPOSED, "a" + NBSP + "b"):
        with pytest.raises(ValidationError):
            execution(config={bad_key: 1})
        with pytest.raises(ValidationError):
            execution(config={"outer": [{bad_key: 1}]})


def test_contextual_zwnj_in_a_label_is_still_accepted():
    persian = chr(0x645) + chr(0x6CC) + chr(0x200C) + chr(0x62E) + chr(0x648) + chr(0x627) + chr(0x647) + chr(0x645)
    assert finding(producer=persian).producer == persian and unicodedata.is_normalized("NFC", persian)


# --- 9. emoji ZWJ links are counted per sequence --------------------------------------------------------------------------
def test_adjacent_emoji_sequences_are_counted_separately():
    family = GRIN + (ZWJ + GRIN) * 3
    assert check(family * 2, creating=True) and check(family * 3, creating=True) and check(family * 6, creating=True)
    assert check(family + VS16 + family, creating=True)
    assert check(GRIN + (ZWJ + GRIN) * 7, creating=True) and not check(GRIN + (ZWJ + GRIN) * 8, creating=True)
    assert not check(family * 2 + ZWJ + GRIN + (ZWJ + GRIN) * 4, creating=True)       # one sequence of 3+1+... links


def test_vs16_and_skin_tones_continue_a_sequence_without_resetting_it():
    tone = chr(0x1F3FD)
    chain = GRIN + VS16 + ZWJ + GRIN + tone + ZWJ + GRIN + VS16 + ZWJ + GRIN
    assert check(chain, creating=True)
    assert not check(GRIN + (VS16 + ZWJ + GRIN) * 8, creating=True)
    assert not check(GRIN + (tone + ZWJ + GRIN) * 8, creating=True)


# --- 10. the unsupported Bengali spelling is pinned and documented ----------------------------------------------------------
def test_the_bengali_ra_zwj_virama_spelling_is_unsupported_at_creation_and_accepted_on_load():
    text = chr(0x9B0) + chr(0x200D) + chr(0x9CD) + chr(0x9AF)
    assert not check(text, creating=True) and check(text, creating=False)
    with pytest.raises(ValidationError):
        finding(message=text)
    assert forge(finding(), message=text).message == text


# --- 11. the pinned Persian sets are verified by Unicode name, independently of the code-point lists --------------------------
DUAL_NAMES = ("YEH WITH HAMZA ABOVE", "BEH", "ALEF MAKSURA", "YEH", "PEH", "TCHEH", "KEHEH", "GAF", "FARSI YEH", "TEH",
              "THEH", "JEEM", "HAH", "KHAH", "SEEN", "SHEEN", "SAD", "DAD", "TAH", "ZAH", "AIN", "GHAIN", "FEH", "QAF",
              "KAF", "LAM", "MEEM", "NOON", "HEH")
RIGHT_NAMES = ("ALEF WITH MADDA ABOVE", "ALEF WITH HAMZA ABOVE", "WAW WITH HAMZA ABOVE", "ALEF WITH HAMZA BELOW", "ALEF",
               "TEH MARBUTA", "DAL", "THAL", "REH", "ZAIN", "WAW", "JEH")


def test_the_pinned_joining_sets_are_exactly_the_documented_letters():
    from app.workflow import base
    dual = {ord(unicodedata.lookup("ARABIC LETTER " + n)) for n in DUAL_NAMES}
    right = {ord(unicodedata.lookup("ARABIC LETTER " + n)) for n in RIGHT_NAMES}
    assert base._DUAL_JOINING == frozenset(dual) and base._RIGHT_JOINING == frozenset(right)
    assert not dual & right and len(dual) == len(DUAL_NAMES) and len(right) == len(RIGHT_NAMES)


# --- 12. the documentation states the new rules and the honest limits -----------------------------------------------------------
def test_the_doc_states_the_canonical_forms_and_the_honest_limits():
    from pathlib import Path
    doc = " ".join((Path(__file__).resolve().parents[3] / "docs" / "architecture" / "workflow-orchestration.md")
                   .read_text(encoding="utf-8").split())
    low = doc.lower()
    for needle in ("nfc", "no-break space", "structural compatibility", "bengali", "residual", "exact `str`",
                   "utf-8 bytes", "u+2028", "does not protect against every visual lookalike", "unchanged stored label"):
        assert needle in low, needle
    assert "never appears in `loc` at all" not in doc


def test_the_nesting_limit_is_exactly_72_containers():
    from app.workflow.base import MAX_LOAD_DEPTH
    assert MAX_LOAD_DEPTH == 72
    at_limit = '{"a":' * 72 + "1" + "}" * 72
    over = '{"a":' * 73 + "1" + "}" * 73
    with pytest.raises(ValidationError):                       # allowed by the depth check, then rejected as a record
        Finding.load(at_limit)
    with pytest.raises(ContractError, match="nested too deeply"):
        Finding.load(over)
    mapping_at_limit: dict = {}
    cursor = mapping_at_limit
    for _ in range(71):
        cursor["a"] = {}
        cursor = cursor["a"]
    with pytest.raises(ValidationError):
        Finding.load(mapping_at_limit)
    cursor["a"] = {}
    with pytest.raises(ContractError, match="nested too deeply"):
        Finding.load(mapping_at_limit)


def test_replace_validates_unchanged_values_only_under_load_semantics_and_changed_ones_under_creation():
    legacy = forge(finding(), producer="p" + PRIVATE_USE, message="DV1" + VS16)
    again = legacy.replace(severity="error")
    assert again.producer == legacy.producer and again.message == legacy.message
    with pytest.raises(ValidationError):
        legacy.replace(message="DV1" + VS16)                       # the same text, but now CHANGED: judged afresh
