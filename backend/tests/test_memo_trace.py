"""The memo trace check: every numeric token in memo prose must be in the value table.

Semantics under test (see app/compute/memo.py):
  * a trace tag is exactly ``[#<digits>]`` or ``[#?]``; tags are removed before scanning
  * literals in the table (verbatim state strings such as a QC check text) are masked
  * a numeric token is a standalone number: it must not be glued to a word, digit or
    underscore on its left, so identifiers (``ds_978f9ca8``, ``AUC0``, ``theta1``) are not numbers
  * a token is traced iff it is NUMERICALLY EQUAL (Decimal) to a table value's rendered text:
    ``8.650`` matches ``8.65``; ``8.6`` does not match ``8.65``; sign matters
"""
import pytest

from app.compute.memo import (
    TAG_RE,
    AuditTrace,
    MemoTraceError,
    TracedValue,
    extract_numbers,
    memo_bad_trace_tags,
    memo_untraced_numbers,
)
from app.core.audit import AuditChain, hash_payload
from app.tools.builtins import default_registry


def _tv(text, key="k", idx=1, kind="number"):
    return TracedValue(key=key, text=text, audit_index=idx, kind=kind)


# ── tokenisation ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("text,expected", [
    ("n = 12.", ["12"]),                                  # sentence period is not a decimal point
    ("GM 8.65 (gCV 16.98%)", ["8.65", "16.98"]),
    ("effect -0.12 and +5", ["-0.12", "+5"]),
    ("range 0.31–1.27", ["0.31", "1.27"]),           # en dash is a separator, not a sign
    ("range 1.2-3.4", ["1.2", "3.4"]),                    # hyphen between numbers is a separator
    ("range -0.5–-0.1", ["-0.5", "-0.1"]),
    ("p = 1.97e-09 and 3E+2", ["1.97e-09", "3E+2"]),
    ("share .5 of 10mg", [".5", "10"]),                   # unit glued AFTER a number still counts
    ("value −0.4", ["-0.4"]),                        # unicode minus is normalised
    ("1,234", ["1", "234"]),                              # no thousands grouping is ever rendered
    ("R² and t½ and CL/F", []),                 # superscripts / vulgar fractions are not digits
    ("dataset ds_978f9ca8 theta1 AUC0 CMT1 Q2W", []),     # identifiers are not numbers
    ("see 6675992c7d", ["6675992"]),                      # a bare hex run IS scanned: do not print digests
    ("", []),
])
def test_extract_numbers(text, expected):
    assert extract_numbers(text) == expected


def test_trace_tags_are_stripped_but_only_the_exact_form():
    assert extract_numbers("GM 8.65 [#5] and 2 [#?]") == ["8.65", "2"]
    # malformed look-alikes are NOT tags, so a number cannot hide inside one
    assert extract_numbers("x [# 12] y [#12.5] z [#12") == ["12", "12.5", "12"]
    assert TAG_RE.fullmatch("[#0]") and TAG_RE.fullmatch("[#?]")
    assert not TAG_RE.fullmatch("[#-1]")


# ── membership ─────────────────────────────────────────────────────────────────
def test_all_traced_returns_empty():
    table = [_tv("12"), _tv("8.65"), _tv("-0.12"), _tv("1.97e-09")]
    assert memo_untraced_numbers("12 subjects, GM 8.65 [#2], -0.12, p 1.97e-09", table) == []


def test_invented_number_is_returned_in_order_and_unique():
    out = memo_untraced_numbers("GM 17.3 then 8.65 then 17.3 and 99", [_tv("8.65")])
    assert out == ["17.3", "99"]


def test_match_is_numeric_equality_not_rounding_and_sign_sensitive():
    table = [_tv("8.65")]
    assert memo_untraced_numbers("8.650", table) == []          # trailing zeros are the same value
    assert memo_untraced_numbers("8.6", table) == ["8.6"]       # rounding is NOT tolerated
    assert memo_untraced_numbers("8.7", table) == ["8.7"]
    assert memo_untraced_numbers("-8.65", table) == ["-8.65"]   # sign matters
    assert memo_untraced_numbers("8", table) == ["8"]


def test_low_precision_tokens_do_not_match_by_rounding():
    # the whole reason membership is exact: a bare "1" must not be excused by 0.7 or 1.2
    assert memo_untraced_numbers("1", [_tv("0.7"), _tv("1.2")]) == ["1"]


def test_table_may_hold_strings_ints_floats_and_traced_values():
    assert memo_untraced_numbers("12 and 8.65 and 3.5 and 7", [12, 8.65, "3.5", _tv("7")]) == []


def test_bool_and_nan_never_count_as_table_values():
    assert memo_untraced_numbers("1 and 0", [True, False, float("nan")]) == ["1", "0"]


def test_empty_table_leaves_every_number_untraced():
    assert memo_untraced_numbers("a 1 b 2.5", []) == ["1", "2.5"]


def test_literals_are_masked_so_state_strings_with_digits_pass():
    qc = "[WARN] AUC %extrap <= 20%: 1 subject(s) > 20.0%"
    memo = f"QC [#7]\n{qc}\nsubjects 12"
    table = [_tv(qc, kind="literal"), _tv("12")]
    assert memo_untraced_numbers(memo, table) == []
    # ...but the mask removes only that exact string: a stray number beside it is still caught
    assert memo_untraced_numbers(memo + " and 101", table) == ["101"]


def test_non_numeric_plain_strings_in_the_table_act_as_literals():
    assert memo_untraced_numbers("study ABC-101 ok", ["ABC-101"]) == []


def test_exponent_and_decimal_normalisation():
    assert memo_untraced_numbers("0.00000197", [_tv("1.97e-06")]) == []
    assert memo_untraced_numbers("1.9700E-06", [_tv("1.97e-06")]) == []


# ── tag validity ───────────────────────────────────────────────────────────────
def test_bad_trace_tags_flag_indices_beyond_the_audit_chain():
    text = "a [#0] b [#9] c [#10] d [#?] e [#12]"
    assert memo_bad_trace_tags(text, n_audit_entries=10) == ["[#10]", "[#12]"]
    assert memo_bad_trace_tags(text, n_audit_entries=13) == []
    assert memo_bad_trace_tags("no tags 1 2 3", n_audit_entries=0) == []


def test_memo_trace_error_is_a_value_error():
    assert issubclass(MemoTraceError, ValueError)


# ── audit lookup ───────────────────────────────────────────────────────────────
def _trail():
    chain = AuditChain()
    for agent, tool, out in [("system", "session", {}), ("data_manager", "load_dataset", {"a": 1}),
                             ("nca", "compute_nca", {"x": 1}), ("nca", "compute_nca", {"x": 2}),
                             ("supervisor", "compute_nca", {"x": 9}), ("statistician", "indirect_comparison", {"i": 1}),
                             ("statistician", "indirect_comparison", {"i": 2})]:
        chain.append(agent=agent, tool=tool, action="a", inputs={}, outputs=out, timestamp="t")
    return chain.to_list()


def test_latest_picks_the_highest_matching_entry_and_ignores_supervisor_and_system():
    trace = AuditTrace(_trail())
    assert trace.latest(("compute_nca",)) == 3            # entry 4 is a supervisor proposal: skipped
    assert trace.latest(("load_dataset",)) == 1
    assert trace.latest(("session",)) is None             # system entries are never a result source
    assert trace.latest(("run_qc",)) is None
    assert AuditTrace(()).latest(("compute_nca",)) is None
    assert trace.n_entries == 7


def test_by_output_matches_the_exact_audited_payload():
    trace = AuditTrace(_trail())
    assert trace.by_output("indirect_comparison", {"i": 1}) == 5
    assert trace.by_output("indirect_comparison", {"i": 2}) == 6
    assert trace.by_output("indirect_comparison", {"i": 3}) is None
    assert trace.by_output("compute_nca", {"x": 9}) is None   # only the supervisor entry has it
    assert hash_payload({"i": 1}) == _trail()[5]["outputs_hash"]


def test_every_source_tool_is_a_registered_tool():
    from app.compute.memo import SOURCE_TOOLS
    names = set(default_registry().names())
    for field, tools in SOURCE_TOOLS.items():
        assert set(tools) <= names, field


# ── hardening from review: masking is exact and token-bounded ───────────────────
def test_punctuation_only_literals_do_not_mask_anything():
    # a literal of "." must not split 3.5 into 3 and 5; "-" must not turn -3 into 3
    assert memo_untraced_numbers("3.5", [_tv(".", kind="literal"), "3", "5"]) == ["3.5"]
    assert memo_untraced_numbers("-3", [_tv("-", kind="literal"), "3"]) == ["-3"]


def test_a_numeric_only_literal_is_a_number_not_a_mask():
    assert memo_untraced_numbers("1", [_tv("1", kind="literal")]) == []
    assert memo_untraced_numbers("1 and 2", [_tv("1", kind="literal")]) == ["2"]
    assert memo_untraced_numbers("12", [_tv("1", kind="literal")]) == ["12"]     # not a substring mask


def test_literal_masking_is_token_bounded():
    # the one-letter literal "e" must not eat the exponent marker of 1.97e-09
    assert memo_untraced_numbers("p 1.97e-09", [_tv("e", kind="literal"), "1.97e-09"]) == []
    # a literal glued to extra digits is not that literal: the stray digits are scanned
    assert memo_untraced_numbers("STUDY-1012", [_tv("STUDY-101", kind="literal")]) == ["1012"]
    assert memo_untraced_numbers("STUDY-101.", [_tv("STUDY-101", kind="literal")]) == []


def test_tags_are_replaced_by_a_space_so_neighbours_cannot_fuse():
    assert extract_numbers("foo[#1]77") == ["77"]
    assert memo_untraced_numbers("foo[#1]77", [_tv("5")]) == ["77"]


def test_tag_digits_must_be_ascii():
    assert extract_numbers("x [#\uff11\uff12] y") == ["\uff11\uff12"]


@pytest.mark.parametrize("minus", ["\u2212", "\ufe63", "\uff0d"])
def test_minus_look_alikes_are_read_as_a_sign(minus):
    assert extract_numbers(f"effect {minus}3") == ["-3"]
    assert memo_untraced_numbers(f"effect {minus}3", ["3"]) == ["-3"]


def test_malformed_numeric_runs_are_one_untraceable_token():
    assert extract_numbers("version 1.2.3 here") == ["1.2.3"]
    assert memo_untraced_numbers("1.2.3", [1.2, 3]) == ["1.2.3"]
    assert extract_numbers("n = 12.") == ["12"]                  # a sentence period is still not a decimal point
