"""r5b: characters a reader may take as part of, or as a reordering of, a number fail closed.

Bidi overrides / embeddings earlier on the line, private-use / noncharacter / replacement / unassigned
/ lone-surrogate characters next to a number, ASCII control characters inside it, and mixed numbers
('12 1/2') must never let a fragment of the printed number pass. Every failing page has an honest twin
that still passes.
"""
from __future__ import annotations

import pytest

from app.evidence import models as M
from app.evidence.numgrammar import tokens
from tests.evidence_r5b_support import check

RLO, LRO, RLE, LRE, PDF = "\u202e", "\u202d", "\u202b", "\u202a", "\u202c"
RLI, LRI, FSI, PDI = "\u2067", "\u2066", "\u2068", "\u2069"


def _fails(value, snippet, page, **kw):
    res = check(value, snippet, page, **kw)
    assert res.status == "fail", (page, res)


def _passes(value, snippet, page, **kw):
    res = check(value, snippet, page, **kw)
    assert res.status == "pass" and res.number == pytest.approx(float(value)), (page, res)


# ------------------------------------------------------------------ bidi ---

@pytest.mark.parametrize("opener", [RLO, LRO, RLE, LRE, RLI, LRI, FSI])
@pytest.mark.parametrize("closer", ["", PDF, PDI])
@pytest.mark.parametrize("cell", [False, True])
def test_an_open_bidi_run_earlier_on_the_line_makes_the_number_unresolved(opener, closer, cell):
    closes = {RLO: PDF, LRO: PDF, RLE: PDF, LRE: PDF}.get(opener, PDI)
    page = f"{opener}AUC 12.5 mg/L" + (closes if closer else "")
    _fails("12.5", "12.5", page, cell=cell)


def test_the_override_from_the_report_does_not_verify():
    _fails("12.5", "12.5", f"{RLO}AUC 12.5 mg/L{PDF}")


def test_a_bidi_run_on_an_earlier_line_is_harmless_but_one_on_the_same_line_is_not():
    """Was test_a_bidi_run_closed_before_the_number_or_on_an_earlier_line_is_harmless: a run closed before the number
    on the SAME line passed. A control is outside the alphabet and the line tier is an allow-list (no bidi state
    machine): any one on the number's line makes it unresolved_context; only a control on ANOTHER line is harmless.
    U+2029 is outside the alphabet, not a line break."""
    _passes("12.5", "12.5", f"{RLO}abc\nAUC 12.5 mg/L")
    for page in (f"{RLO}abc{PDF} AUC 12.5 mg/L", f"{RLI}abc{PDI} AUC 12.5 mg/L", f"{RLO}abc\u2029AUC 12.5 mg/L"):
        res = check("12.5", "12.5", page)
        assert (res.status, res.reason) == ("fail", M.REASON_UNRESOLVED_CONTEXT), (page, res)


def test_an_isolate_stays_open_when_only_an_embedding_closer_follows():
    _fails("12.5", "12.5", f"{RLI}abc{PDF} AUC 12.5 mg/L")
    _fails("12.5", "12.5", f"{RLO}abc{PDI} AUC 12.5 mg/L")


def test_a_number_before_the_override_on_the_same_line_is_unresolved():
    """Was test_a_number_before_the_override_is_read_normally (a pass): the control is on the number's line."""
    res = check("12.5", "12.5", f"AUC 12.5 mg/L {RLO}abc{PDF}")
    assert (res.status, res.reason) == ("fail", M.REASON_UNRESOLVED_CONTEXT), res


def test_bidi_controls_taint_the_whole_line_in_tokens():
    """Was [1.0, None, 3.0, 4.0, None] (positional bidi state): both lines hold a control, so every token is None."""
    toks = tokens(f"1 {RLO} 2 {PDF} 3\n4 {LRO} 5", ".")
    assert [t.value for t in toks] == [None, None, None, None, None]


# ---------------------------------------------- unreadable characters ---

UNREADABLE = ["\ue000", "\ufffd", "\ufffc", "\uffff", "\ufdd0", "\U000f0000", "\u0378", "\ud800", "\udfff",
              "\uf02d", "\uf8ff", "\U0010ffff", "\x80", "\x9f"]


@pytest.mark.parametrize("ch", UNREADABLE)
@pytest.mark.parametrize("template", ["AUC {c}12 mg/L", "AUC 12{c} mg/L", "AUC {c} 12 mg/L", "AUC 12 {c} mg/L"])
def test_an_unreadable_character_next_to_the_digits_leaves_them_unresolved(ch, template):
    _fails("12", "12", template.format(c=ch))


@pytest.mark.parametrize("ch", ["\ue000", "\ufffd", "\uffff", "\ud800"])
def test_unreadable_character_next_to_a_decimal_or_in_a_cell(ch):
    _fails("1.5", "1.5", f"AUC 1.5{ch} mg/L")
    _fails("12", "12", f"12{ch}", cell=True)
    _fails("12", "12", f"{ch}12", cell=True)


def test_every_private_use_code_point_is_glue():
    for cp in range(0xE000, 0xF900, 37):
        _fails("12", "12", f"AUC 12{chr(cp)} mg/L")
        _fails("12", "12", f"AUC {chr(cp)}12 mg/L")


def test_unreadable_character_two_words_away_still_taints_the_line():
    """Was test_unreadable_character_two_words_away_does_not_matter (both passed): an OUTSIDE character anywhere on
    the number's line makes it unresolved_context; the same character on another line does not."""
    for page in ("AUC 12 mg/L \ue000", "\ufffd AUC 12 mg/L"):
        res = check("12", "12", page)
        assert (res.status, res.reason) == ("fail", M.REASON_UNRESOLVED_CONTEXT), (page, res)
    _passes("12", "12", "\ufffd\nAUC 12 mg/L")


# ------------------------------------------------------ control chars ---

@pytest.mark.parametrize("ctl", ["\x00", "\x01", "\x07", "\x08", "\x0e", "\x1b", "\x1c", "\x1f", "\x7f"])
def test_an_ascii_control_inside_the_digits_leaves_them_unresolved(ctl):
    _fails("1", "1", f"AUC 1{ctl}2 mg/L")
    _fails("12", "12", f"AUC 12{ctl}3 mg/L")
    _fails("1", "1", f"1{ctl}2", cell=True)


@pytest.mark.parametrize("ctl", ["\x00", "\x1f", "\x7f"])
def test_an_ascii_control_before_a_number_may_be_its_sign(ctl):
    _fails("12", "12", f"AUC {ctl}12 mg/L")


def test_tabs_and_line_breaks_are_still_ordinary_gaps():
    _passes("12", "12", "AUC\t12\nmg/L")
    toks = tokens("AUC 1\x7f2 mg/L", ".")
    assert len(toks) == 1 and toks[0].value is None
    assert toks[0].start == 4 and toks[0].end == 7          # offsets map back to the raw text


# ------------------------------------------------------- mixed numbers ---

MIXED = ["12 \u00bd", "12\u2009\u00bd", "12\u00a0\u00bd", "12\u200a\u00bd", "12 \u00bc", "12 \u2153",
         "12 1/2", "12 1\u20442", "12 3/4", "12 \u00be", "12\u202f1/2"]


@pytest.mark.parametrize("mixed", MIXED)
def test_a_whole_number_followed_by_a_fraction_is_unresolved(mixed):
    _fails("12", "12", f"AUC {mixed} mg/L")
    _fails("12", "12", mixed, cell=True)


def test_a_fraction_before_the_number_is_unresolved_and_plain_neighbours_pass():
    _fails("12", "12", "AUC \u00bd 12 mg/L")
    _passes("12", "12", "AUC 12 mg/L and 1/2 of subjects")
    _passes("12", "12", "AUC 12, half 1/2")
