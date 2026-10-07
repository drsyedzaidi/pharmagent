"""r5, rewritten for the r6 numeric grammar: invisible characters and combining marks around a number fail closed.

Before r6 an allow-list-less reader folded invisible characters out (Cf, Default_Ignorable) and kept a positional
bidi state, so a ZWJ around a multiplier "read like the clean text". Now a character that is not in the allowed
alphabet inside a numeric expression, in its word, one word away or elsewhere on its line makes the expression
unresolved, whatever the character is: a number never reads through an invisible one.

Two layers:
1. Targeted regressions: a ZWJ / variation selector / combining mark around the multiplier of
   '1.2 x 10 3' (raised exponent) must not turn the token into a plain 1.2; invisible characters, bidi controls
   and marks leave a token unresolved (unresolved_context); clean text still reads, locates and round-trips.
2. A DIFFERENTIAL FUZZ: for ~30 numeric expressions, every sampled character (all relevant Unicode
   categories + the full variation-selector ranges) is inserted at every gap kind (before the
   number, inside it, around the multiplier, around the exponent, after it) of the page, of the
   snippet, and of both. The verify() outcome must equal the clean outcome (status and value) or be
   a fail: never a pass with a different value, never a pass where the clean text fails. (The exhaustive tier,
   tests/test_evidence_r6_exhaustive.py, sweeps every code point at every boundary of the tokenizer and a
   stratified sample of verify(); this file keeps the r5 expression set and its gap kinds.)
"""
from __future__ import annotations

import time
import unicodedata

import pytest

from app.evidence import models as M
from app.evidence.extract_rules import value_number
from app.evidence.normalize import normalise
from app.evidence.numgrammar import cls, parse_claim, tokens
from app.evidence.value_checks import censored_flags, scaled_flags
from app.evidence.verify import verify
from tests.evidence_timing import limit

ZWJ, ZWNJ, ZWSP, WJ, SHY = "‍", "‌", "​", "⁠", "­"
VS1, VS17 = "️", "\U000e0100"
STRIKE, ENCLOSE = "̸", "⃠"
RLM, RLE, LRI, ALM = "‏", "‫", "⁦", "؜"


def _doc(text: str, style: str = ".", tables: list | None = None) -> dict:
    return {"sha256": "x" * 64, "decimal_style": style,
            "pages": [{"n": 1, "text": text, "small_text": "", "columns": None, "tables": tables or []}]}


def _prop(value: str, snippet: str, style: str = ".", kind_: str = "text", cell: dict | None = None) -> dict:
    loc = M.Locator(page=1, kind=kind_, snippet=snippet, cell=cell, decimal_style=style).to_json()
    return {"source_id": "S1", "field": "Cmax", "group_label": "", "value": value,
            "value_num": value_number(value, style), "unit": None, "locator": loc, "extractor": "llm"}


# ------------------------------------------------------------ targeted ---

RAISED = ["Cmax 1.2 × 103 pg", "Cmax 1.2 x 10 3 pg"]


@pytest.mark.parametrize("raised", RAISED)
@pytest.mark.parametrize("mark", [ZWJ, ZWNJ, ZWSP, WJ, SHY, VS1, VS17, "᠋", "ㅤ", "\U000e0041",
                                  STRIKE, ENCLOSE, RLM, RLE, LRI, ALM, "￹"])
def test_mark_around_multiplier_never_turns_raised_exponent_into_its_mantissa(raised, mark):
    for i in range(len(raised) + 1):
        page = raised[:i] + mark + raised[i:]
        res = verify(_prop("1.2", "Cmax 1.2"), _doc(page))
        assert res.status == "fail", (hex(ord(mark)), i, res)


def test_zero_width_joiner_makes_every_token_unresolved_context():
    """Was test_zero_width_joiner_is_read_like_the_clean_text (the ZWJ read like the clean text, value None with the
    mantissa 1.2): the clean text is unresolved_shape (a raised exponent), the ZWJ text is unresolved_context."""
    for clean_text in ("Cmax 1.2 x 10 3 pg", "Cmax 1.2 × 103 pg"):
        folded = clean_text.replace(" 10", ZWJ + "10", 1)
        a, b = tokens(folded, ".")[0], tokens(clean_text, ".")[0]
        assert (a.value, a.reason) == (None, "unresolved_context")
        assert (b.value, b.reason, b.parts[0]) == (None, "unresolved_shape", 1.2)
        assert folded[a.start:a.end].startswith("1.2")


def test_zero_width_joiner_in_a_readable_exponent_fails_and_the_clean_exponent_passes():
    """Was test_zero_width_joiner_in_a_readable_exponent_still_passes_with_clean_value ('1.2 ×<ZWJ>10^3' passed
    with 1200.0): a ZWJ is outside the alphabet, so the power is unresolved_context."""
    clean_page = "Cmax 1.2 × 10^3 pg"
    res = verify(_prop("1.2 x 10^3", "Cmax 1.2 × 10^3"), _doc(clean_page))
    assert res.status == "pass" and res.number == pytest.approx(1200.0)
    page = "Cmax 1.2 ×" + ZWJ + "10^3 pg"
    res = verify(_prop("1.2 x 10^3", "Cmax 1.2 ×" + ZWJ + "10^3"), _doc(page))
    assert (res.status, res.reason) == ("fail", M.REASON_UNRESOLVED_CONTEXT), res
    assert "U+200D ZERO WIDTH JOINER" in res.detail


@pytest.mark.parametrize("mark", [STRIKE, ENCLOSE, "́", "⃝", "ः"])
@pytest.mark.parametrize("text, mark_at", [("12.5", 0), ("12.5", 2), ("12.5", 3), ("-12.5", 0),
                                          ("1.2 × 10^3", 4), ("1.2 x 10^3", 4), ("1.2 x 10^3", 9),
                                          ("1.2e-3", 3), ("1.2e-3", 5)])
def test_mark_on_a_digit_point_sign_or_multiplier_makes_the_token_unresolved(mark, text, mark_at):
    marked = text[:mark_at + 1] + mark + text[mark_at + 1:]
    toks = tokens(marked, ".")
    assert toks and all(t.value is None for t in toks), (marked, toks)
    assert parse_claim(marked) is None


def test_mark_on_the_multiplier_makes_citing_the_mantissa_unresolved_context():
    """Was test_mark_keeps_the_mantissa_so_citing_it_is_a_partial_number (reason partial_number): the 1.2 is a part of
    an expression that touches a combining mark, and the verifier now names that cause (unresolved_context, with the
    code point in the reviewer sentence) instead of calling it a partial number."""
    page = "Cmax 1.2 ×" + STRIKE + " 10^3 pg"
    res = verify(_prop("1.2", "Cmax 1.2"), _doc(page))
    assert (res.status, res.reason) == ("fail", M.REASON_UNRESOLVED_CONTEXT)
    assert "U+0338 COMBINING LONG SOLIDUS OVERLAY" in res.detail
    res = verify(_prop("1.2 x 10^3", "Cmax 1.2 ×" + STRIKE + " 10^3"), _doc(page))
    assert (res.status, res.reason) == ("fail", M.REASON_UNRESOLVED_CONTEXT)


@pytest.mark.parametrize("bidi", [RLM, "‎", RLE, "‭", LRI, "⁩", ALM])
@pytest.mark.parametrize("text", ["Cmax 12.5 pg", "Cmax 1.2e-3 pg", "Cmax 1.2 x 10^3 pg"])
def test_bidi_control_inside_or_next_to_a_number_leaves_it_unresolved(bidi, text):
    a = text.index("1")
    b = text.index(" pg")
    for i in range(a, b + 1):
        toks = tokens(text[:i] + bidi + text[i:], ".")
        assert toks and all(t.value is None for t in toks), (hex(ord(bidi)), i, toks)


def test_bidi_control_on_the_numbers_line_makes_it_unresolved_even_far_away():
    """Was test_bidi_control_far_from_a_number_does_not_touch_it ([12.5]): the control is two words away on the
    number's line, and an outside character anywhere on the line makes its numbers unresolved (the line tier: the
    allow-list needs no bidi state machine). On another line (not the adjacent word) it is harmless."""
    assert [t.value for t in tokens(RLM + "Cmax " + "12.5 pg", ".")] == [None]
    assert [t.value for t in tokens(RLM + "Cmax\nAUC 12.5 pg", ".")] == [12.5]      # a control on another line


@pytest.mark.parametrize("text, i", [("Cmax 12.5 pg", 5), ("Cmax -12.5 pg", 5), ("Cmax 1.2e-3 pg", 5)])
def test_invisible_character_directly_before_a_number_leaves_it_unresolved(text, i):
    assert tokens(text[:i] + SHY + text[i:], ".")[0].value is None


@pytest.mark.parametrize("text", ["Cmax 1.2e-3 pg", "Cmax 1.2 x 10^-3 pg", "Cmax 1.2e3 pg"])
def test_invisible_character_before_an_exponent_sign_or_digits_is_unresolved(text):
    i = text.index("e") + 1 if "e" in text.split()[1] else text.index("^") + 1
    folded = text[:i] + SHY + text[i:]
    assert tokens(folded, ".")[0].value is None


def test_invisible_character_between_digits_is_unresolved():
    assert tokens("Cmax 12" + ZWSP + "145 pg", ".")[0].value is None


def test_offsets_are_native_and_an_invisible_character_is_never_dropped():
    """Was test_offsets_map_back_to_the_raw_text (a ZWJ inside the power was folded out and the token read 1500):
    offsets index the raw string and nothing is cleaned, so the same text now gives one UNRESOLVED expression that
    spans the ZWJ; the locator keeps the ZWJ too (it is no longer dropped by normalise)."""
    raw = ZWJ + "Cmax " + "1.5 \u00d7" + ZWJ + "10^3 pg" + ZWJ
    toks = tokens(raw, ".")
    assert len(toks) == 1 and toks[0].value is None and toks[0].reason == "unresolved_context"
    t = toks[0]
    assert raw[t.start:t.end].startswith("1.5") and ZWJ in raw[t.start:t.end]
    norm, idx = normalise(raw)
    assert ZWJ in norm and len(norm) == len(idx)
    clean = "Cmax 1.5 \u00d710^3 pg"
    assert [(clean[x.start:x.end], x.value) for x in tokens(clean, ".")] == [("1.5 \u00d710^3", pytest.approx(1500.0))]


def test_clean_locator_round_trips_and_an_invisible_character_after_the_literal_fails():
    """Was test_passing_record_locator_round_trips_with_invisible_characters_around_the_literal (ZWJ and ZWSP around
    the literal passed): clean text still round-trips; with the invisible characters the cited number is
    unresolved_context (they sit on its line, one of them glued after it)."""
    clean_page = "Cmax was 1.5 ng/mL"
    res = verify(_prop("1.5", "Cmax was 1.5"), _doc(clean_page))
    assert res.status == "pass" and res.number == 1.5
    a, b = res.locator["token"]
    assert clean_page[a:b] == "1.5" and clean_page[res.span[0]:res.span[1]] == "Cmax was 1.5"
    page = "Cmax" + ZWJ + " was 1.5" + ZWSP + " ng/mL"
    res = verify(_prop("1.5", "Cmax was 1.5"), _doc(page))
    assert res.status == "fail" and res.reason in (M.REASON_SNIPPET_NOT_ON_PAGE, M.REASON_UNRESOLVED_CONTEXT)
    res = verify(_prop("1.5", "Cmax" + ZWJ + " was 1.5"), _doc(page))
    assert (res.status, res.reason) == ("fail", M.REASON_UNRESOLVED_CONTEXT), res


def test_censored_qualifier_is_flagged_when_clean_and_the_record_fails_when_marked():
    """Was test_censored_qualifier_survives_invisible_characters_and_marks (pass + censored flag for every spelling:
    a vacuous loop that skipped any failing one). Clean spellings pass and are flagged; a spelling with an invisible
    character or mark between the qualifier and the number fails unresolved_context."""
    for before in ("<", "< ", "BLQ "):
        page = f"conc {before}0.5 ng/mL"
        res = verify(_prop("0.5", f"conc {before}0.5"), _doc(page))
        assert res.status == "pass", before
        p = {"field": "Cmax", "locator": res.locator, "verify_status": "pass"}
        assert censored_flags(p, _doc(page)["pages"][0]), repr(before)
    for before in ("<" + ZWJ, "<" + ZWSP + " ", "≤" + RLM + " ", "<" + STRIKE, "BL" + ZWJ + "Q "):
        page = f"conc {before}0.5 ng/mL"
        res = verify(_prop("0.5", f"conc {before}0.5"), _doc(page))
        assert (res.status, res.reason) == ("fail", M.REASON_UNRESOLVED_CONTEXT), (before, res)


def test_scaled_header_survives_invisible_characters_and_marks():
    for header in ("AUC ×10^3", "AUC ×" + ZWJ + "10^3", "AUC x" + SHY + "10^3", "AUC ×" + STRIKE + "10^3",
                   "AUC ×" + RLM + "10³"):
        page = {"tables": [{"rows": [["Group", header], ["A", "12"]]}]}
        p = {"field": "AUC", "unit": None,
             "locator": {"kind": "table", "cell": {"table": 0, "row": 1, "col": 1}}}
        assert scaled_flags(p, page), repr(header)


# -------------------------------------------------------- differential ---

def _every(first: int, last: int, stride: int = 1) -> list[str]:
    return [chr(c) for c in range(first, last + 1, stride)]


def _sample() -> list[str]:
    chars = _every(0x200B, 0x200F) + _every(0x2060, 0x2064) + _every(0x206A, 0x206F)
    chars += [SHY, "͏", "﻿", "ᅟ", "ᅠ", "ㅤ", "ﾠ", "឴", "឵"]
    chars += _every(0x180B, 0x180F) + [chr(0xE0001)] + _every(0xE0020, 0xE007F, 7)
    chars += ["؜"] + _every(0x202A, 0x202E) + _every(0x2066, 0x2069) + _every(0xFFF9, 0xFFFB)
    chars += _every(0x0600, 0x0605) + ["۝", "܏", "\U000110bd"]
    chars += _every(0x0300, 0x036F, 3) + _every(0x20D0, 0x20F0, 2) + ["⃝", "⃢", "҃", "ְ"]
    chars += ["ः", "ा", "ா", "\U0001d165"]
    chars += [chr(c) for c in range(0x110000) if unicodedata.category(chr(c)) == "Cf"][::6]
    seen: dict[str, None] = dict.fromkeys(chars)
    return list(seen)


SAMPLE = _sample()
VARIATION_SELECTORS = _every(0xFE00, 0xFE0F) + _every(0xE0100, 0xE01EF)

# name, page, snippet, value, decimal style, (multiplier, exponent) offsets in the page when scientific
EXPRESSIONS = [
    ("plain", "Cmax was 12.5 ng/mL", "Cmax was 12.5", "12.5", "."),
    ("decimal comma", "Cmax var 12,5 ng/mL", "Cmax var 12,5", "12,5", ","),
    ("thousands", "AUC reached 1,234 ng.h/mL", "AUC reached 1,234", "1,234", "."),
    ("negative", "change was -20 percent", "change was -20", "-20", "."),
    ("percent", "response in 45% of subjects", "response in 45%", "45", "."),
    ("range low", "range 1.95-2.74 L/h", "range 1.95", "1.95", "."),
    ("range high", "range 1.95-2.74 L/h", "1.95-2.74", "2.74", "."),
    ("plus minus", "CL 5.3 ± 0.4 L/h", "CL 5.3 ± 0.4", "5.3", "."),
    ("plus minus sd", "CL 5.3 ± 0.4 L/h", "5.3 ± 0.4", "0.4", "."),
    ("e-notation", "k was 1.2e-3 per h", "k was 1.2e-3", "1.2e-3", "."),
    ("e-notation mantissa", "k was 1.2e-3 per h", "k was 1.2", "1.2", "."),
    ("caret", "Cmax 1.2 x 10^3 cells", "Cmax 1.2 x 10^3", "1.2 x 10^3", "."),
    ("caret times", "Cmax 1.2 × 10^3 cells", "Cmax 1.2 × 10^3", "1.2 × 10^3", "."),
    ("caret negative", "Km 4.5 × 10^-3 M", "Km 4.5 × 10^-3", "4.5 × 10^-3", "."),
    ("caret mantissa", "Cmax 1.2 x 10^3 cells", "Cmax 1.2", "1.2", "."),
    ("caret ten", "Cmax 1.2 x 10^3 cells", "x 10^3 cells", "10", "."),
    ("caret exponent", "Cmax 1.2 x 10^3 cells", "10^3 cells", "3", "."),
    ("superscript mantissa", "Cmax 1.2 × 10³ cells", "Cmax 1.2", "1.2", "."),
    ("raised", "Cmax 1.2 × 103 cells", "Cmax 1.2", "1.2", "."),
    ("raised spaced", "Cmax 1.2 x 10 3 cells", "Cmax 1.2", "1.2", "."),
    ("raised digits", "Cmax 1.2 × 103 cells", "× 103 cells", "103", "."),
    ("raised exponent", "Cmax 1.2 x 10 3 cells", "10 3 cells", "3", "."),
    ("raised negative", "Cmax 1.2 x 10 -3 cells", "Cmax 1.2", "1.2", "."),
    ("censored", "conc <0.5 ng/mL", "conc <0.5", "0.5", "."),
    ("unit after", "Dose 150 mg daily", "Dose 150 mg", "150", "."),
    ("p value", "p = 0.032 overall", "p = 0.032", "0.032", "."),
    ("dot thousands", "AUC 1.250.000 units", "AUC 1.250.000", "1.250.000", ","),
    ("middot", "Lancet 12·4 units", "Lancet 12·4", "12.4", "."),
    ("truncated", "Clearance was 12.4 L/h", "2.4 L/h", "2.4", "."),
    ("fragment", "AUC 1,234 ng.h/mL", "234 ng.h/mL", "234", "."),
]


def _gap_kinds(page: str, snippet: str, value: str) -> dict[str, list[int]]:
    """Page offsets of each gap kind (insertion happens BEFORE page[g])."""
    off = page.index(snippet)
    a = off + len(snippet.rstrip("%"))
    lit = max((m for m in (value, value.replace(",", "."), value.replace(".", "·")) if m in page),
              key=len, default=None)
    start = page.index(lit) if lit else a - 1
    end = start + (len(lit) if lit else 1)
    gaps: dict[str, set[int]] = {"before": {max(0, start - 2), start - 1, start}, "after": {end, end + 1, end + 2},
                                 "inside": set(), "multiplier": set(), "exponent": set()}
    for g in range(start + 1, end):
        ch = page[g - 1:g + 1]
        if any(c in ch for c in "x×*") or (g < len(page) and page[g] in " ^" and "10" in page[g:g + 5]):
            gaps["multiplier"].add(g)
        elif "10" in page[start:g] or "e" in page[start:g] or "^" in page[start:g]:
            gaps["exponent"].add(g)
        else:
            gaps["inside"].add(g)
    full = page.find("10", start)
    if full >= 0:
        gaps["multiplier"].update({full - 1, full})
        gaps["exponent"].update({full + 2, full + 3})
    gaps["multiplier"] |= {g for g in range(start, min(len(page), start + 8)) if page[g - 1:g] in "x×*"}
    return {k: {g for g in v if 0 <= g <= len(page)} for k, v in gaps.items()}


def _outcome(res: M.VerifyResult) -> tuple[str, float | None]:
    return res.status, None if res.number is None else round(res.number, 12)


def _check(clean_res: M.VerifyResult, res: M.VerifyResult, ctx: tuple) -> None:
    if res.status == "fail":
        return
    assert clean_res.status == "pass" and _outcome(res) == _outcome(clean_res), (ctx, clean_res, res)
    assert res.literal is not None and res.literal == (clean_res.literal or ""), ctx


def _modes(page: str, snippet: str, g: int, ch: str) -> list[tuple[str, str, str]]:
    off = page.index(snippet)
    ins_page = page[:g] + ch + page[g:]
    out = [("page", ins_page, snippet)]
    if 0 <= g - off <= len(snippet):
        ins_snip = snippet[:g - off] + ch + snippet[g - off:]
        out += [("snippet", page, ins_snip), ("both", ins_page, ins_snip)]
    return out


def _run(name, page, snippet, value, style, chars, per_kind=None):
    clean_res = verify(_prop(value, snippet, style), _doc(page, style))
    kinds = _gap_kinds(page, snippet, value)
    count = 0
    for gap_kind, gaps in kinds.items():
        ordered = sorted(gaps)
        for g in ordered if per_kind is None else ordered[:per_kind]:
            for ch in chars:
                for mode, p, s in _modes(page, snippet, g, ch):
                    res = verify(_prop(value, s, style), _doc(p, style))
                    _check(clean_res, res, (name, gap_kind, mode, g, hex(ord(ch))))
                    count += 1
    return count


def test_sample_covers_every_relevant_category():
    assert len(SAMPLE) >= 80 and len(VARIATION_SELECTORS) == 256
    cats = {unicodedata.category(c) for c in SAMPLE}
    assert {"Cf", "Mn", "Me", "Mc", "Lo"} <= cats
    # whatever its category, every sampled character is simply not in the allowed alphabet
    assert all(cls(c) == "outside" for c in SAMPLE + VARIATION_SELECTORS)
    for block in ((0x0300, 0x036F), (0x20D0, 0x20FF), (0x180B, 0x180F), (0xE0000, 0xE007F), (0x202A, 0x202E),
                  (0x2066, 0x2069), (0xFFF9, 0xFFFB), (0x3164, 0x3164), (0x200B, 0x200F), (0x2060, 0x2064)):
        assert any(block[0] <= ord(c) <= block[1] for c in SAMPLE), block


def test_every_gap_kind_exists_for_the_scientific_expressions():
    for name, page, snippet, value, style in EXPRESSIONS:
        if name in ("caret", "raised", "raised spaced", "caret negative"):
            kinds = _gap_kinds(page, snippet, value)
            assert all(kinds[k] for k in kinds), (name, kinds)


@pytest.mark.parametrize("name, page, snippet, value, style", EXPRESSIONS, ids=[e[0] for e in EXPRESSIONS])
def test_differential_sample_characters_at_every_gap(name, page, snippet, value, style):
    assert _run(name, page, snippet, value, style, SAMPLE) > 0


def test_differential_full_variation_selector_ranges():
    started = time.monotonic()
    total = 0
    for name, page, snippet, value, style in EXPRESSIONS:
        total += _run(name, page, snippet, value, style, VARIATION_SELECTORS, per_kind=2)
    assert total > 10_000
    # 60 s strictly; limit() scales it only while a line tracer is installed (the coverage gate), so the
    # work done (total > 10_000 checks) stays the regression guard and the wall clock stays strict otherwise
    assert time.monotonic() - started < limit(60)


TABLE_CELLS = ["12.5", "1.2 × 103", "1.2 x 10^3", "<0.5"]


@pytest.mark.parametrize("cell_text", TABLE_CELLS)
def test_differential_in_a_table_cell(cell_text):
    cell = {"table": 0, "row": 1, "col": 1, "header": "Cmax", "row_label": "A"}
    value = "1.2" if "10" in cell_text and "^" not in cell_text else cell_text.lstrip("<")
    if "^" in cell_text:
        value = cell_text

    def run(ctext, snippet):
        tables = [{"index": 0, "strategy": "lines", "rows": [["Group", "Cmax"], ["A", ctext]]}]
        return verify(_prop(value, snippet, ".", "table", cell), _doc(f"Group Cmax\nA {ctext}", tables=tables))

    clean_res = run(cell_text, cell_text)
    for g in range(len(cell_text) + 1):
        for ch in SAMPLE[::3] + VARIATION_SELECTORS[::16]:
            ins = cell_text[:g] + ch + cell_text[g:]
            _check(clean_res, run(ins, cell_text), (cell_text, "cell", g, hex(ord(ch))))
            _check(clean_res, run(ins, ins), (cell_text, "both", g, hex(ord(ch))))
