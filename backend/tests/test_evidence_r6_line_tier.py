"""r6 numeric grammar: the three tiers of the one-separator rule (word, window, line) and the reviewer fragments.

Word    a character outside INSIDE+CONTEXT in the expression's own word  -> the whole word is one unresolved expression
Window  such a character one word before or after (a run of gap characters of any length is ONE separator, a line
        break included), or in the right operand behind a multiplier word -> every token of the word is unresolved
Line    an OUTSIDE character (or a ``(cid:N)`` marker) anywhere on the token's line -> every token of that line

r7 (owner decision 5): there is no LINE list. One allow-list (INSIDE, CONTEXT, GAP and the equivalents) decides all three
tiers; a character of no list taints its word, its window and its line (tests/test_evidence_r7_line_derived.py).
"""
from __future__ import annotations

import pytest

from app.evidence.numgrammar import tokens
from tests.evidence_timing import limit

ZWSP, ZWJ, SHY, RLO, PDF, RLI, PDI, LRM, PUA, BRAILLE = ("\u200b", "\u200d", "\u00ad", "\u202e", "\u202c", "\u2067",
                                                      "\u2069", "\u200e", "\ue000", "\u2800")


def V(text: str, style: str = ".", mode: str = "page") -> list[float | None]:
    return [t.value for t in tokens(text, style, mode=mode)]


def reasons(text: str, mode: str = "page") -> list[str]:
    return [t.reason for t in tokens(text, mode=mode)]


# ------------------------------------------------------------------ the word tier ---
@pytest.mark.parametrize("text", [
    f"12.{PUA}5", f"1{ZWSP}2", f"12{SHY}5", f"{ZWSP}12", f"12{ZWSP}", f"<{BRAILLE}12", f"1.2 x{ZWJ}10^3 mg",
    f"12{LRM}", f"1{chr(0x0301)}2", "12\u202f\u200b5", "12\u066b5", "\u0661\u0662 7", "12\u00be",
    "12\u2028", "12\x0b3", "12\x0c3", "12\x855", "12\u3000 7", "12\ufffd", "1\u20602",
])
def test_an_outside_character_in_the_word_makes_the_word_one_unresolved_context_expression(text):
    toks = tokens(text)
    assert toks and all(t.value is None for t in toks), (text, toks)
    assert any(t.reason == "unresolved_context" for t in toks)


def test_the_word_tier_names_the_character_and_where_it_sits():
    (t,) = tokens(f"12.{PUA}5")
    assert (t.reason, t.detail, (t.start, t.end)) == ("unresolved_context", "U+E000 inside the number", (0, 5))
    assert tokens(f"12{ZWSP}")[0].detail == "U+200B inside the number"
    assert tokens("12\u2028")[0].detail == "U+2028 inside the number"


def test_a_number_made_unresolved_by_its_window_keeps_its_value_as_a_part():
    (t,) = tokens("AUC\u2122_ 12.5")                     # r7: the registered sign is CONTEXT, the low line is not
    assert (t.value, t.parts, t.reason) == (None, (12.5,), "unresolved_context")
    (u,) = tokens("12.\ue0005")
    assert (u.value, u.parts) == (None, ())


# ---------------------------------------------------------------- the window tier ---
@pytest.mark.parametrize("text, where", [
    ("Adalat_ 30 mg", "one word before"), ("AUC0–‘ 1234", "one word before"),
    ("-20 ′C", "one word after"), ("$18 years", "inside the number"), ("£150 mg", "inside the number"),
    ("t½ 11.4 h", "one word before"), ("12 ½", "one word after"), ("• 150 mg", "one word before"),
    ("Mü\u0161ller 2019", "one word before"), ("140 \u212b²", "inside the number"),
    ("dose — 12", "one word before"), ("CL 12 —", "one word after"), ("ng\u22c5h/mL 1234", "one word before"),
])
def test_a_character_outside_the_context_one_word_away_makes_the_number_unresolved(text, where):
    toks = [t for t in tokens(text) if t.parts or t.value is not None or t.reason]
    hit = next(t for t in toks if t.detail.endswith(where))
    assert hit.value is None and hit.reason == "unresolved_context"


def test_the_context_characters_one_word_away_are_fine():
    for ok in ("Cmax 12 (ng/mL)", "AUC [12.5] μg/L", "AUC (12) μg/L", "n = 12; p = 0.03", "12/kg", "ΔOFV 3.84", "ω = 0.40",
               "t1/2 = 8 h", "37°C", "see (12)", "x 12", "dose of 12 mg."):
        assert [v for v in V(ok) if v is None] == [], ok


def test_the_window_is_two_words_wide_and_no_wider():
    # r7: an unlisted character on the SAME line taints it (derived rule); the window is measured across a line break
    assert V("•\nok 150 mg") == [150.0] and V("150 mg\nok •") == [150.0]
    assert V("•\n150 mg") == [None] and V("150\n• mg") == [None]
    assert V("• ok 150 mg") == [None] and V("150 mg ok •") == [None]


def test_a_run_of_gap_characters_of_any_length_and_mix_is_one_separator():
    for pad in (" ", "  ", "\t", "\n", "\r\n", " \n\t \r" * 50, " " * 5000, "\n" * 2000):
        toks = tokens("AUC <" + pad + "12.5 mg/L")
        assert [(t.value, t.qualified) for t in toks] == [(12.5, True)], repr(pad[:5])
    assert V("•" + " " * 5000 + "12") == [None] and V("12" + "\n" * 5000 + "•") == [None]


def test_the_window_crosses_a_line_break_because_the_break_is_a_gap():
    assert V("ok •\n12 mg") == [None] and V("12\n• ok") == [None] and V("12 mg\n• ok") == [12.0]
    assert V("ok •\nok 12 mg") == [12.0]


def test_the_right_operand_behind_a_multiplier_word_belongs_to_the_expression():
    assert V(f"1.2 × {ZWSP}10^3") == [None]            # the operand word joins the left token: one expression
    assert V("1.2 x • 10^3") == [None, None]
    assert V("1.2 x\n• 10^3") == [None, None] and V("1.2 x 10^3\n•") == [None]
    assert V("1.2 x 10^3 •") == [None]
    assert V("2 x 10 •mg") == [None] and V("2 x 100 mg") == [2.0, 100.0]
    assert all(v is None for v in V(f"2 x 10 {ZWSP}mg"))


# --------------------------------------------------------------------- the line tier ---
@pytest.mark.parametrize("text", [
    f"{RLO}abc{PDF} AUC 12.5 mg/L", f"{RLI}abc{PDI} AUC 12.5 mg/L", f"{RLO}abc\u2029 AUC 12.5 mg/L",
    f"AUC 12.5 mg/L {RLO}abc{PDF}", f"AUC 12 mg/L {PUA}", f"{PUA} AUC 12 mg/L", f"{RLI}AUC was 12.5",
    f"AUC 1 2 3 4 5 6 mg/L {ZWSP}", f"{RLI}abc{PDI} AUC 12.5 mg/L", "\u2067AUC was 12.5 mg/L",
    "AUC was 12.5 \u2060 mg/L", f"{RLI}abc{LRM} AUC 12.5", "AUC 12.5 mg/L\ufffd", "12 ok ok ok ok ok ok ok \u2028",
])
def test_an_outside_character_anywhere_on_the_line_makes_every_number_of_the_line_unresolved(text):
    toks = tokens(text)
    assert toks and all(t.value is None for t in toks), (text, toks)
    assert any("elsewhere on the same line" in t.detail for t in toks) or any(t.reason for t in toks)


def test_other_lines_keep_their_values_when_they_are_not_adjacent():
    assert V(f"{RLO}abc\nAUC 12.5 mg/L") == [12.5]
    assert V(f"{RLO}abc\nok\nAUC 12.5 mg/L") == [12.5]
    assert V(f"AUC 12.5 mg/L\nok\n{RLO}abc") == [12.5]
    assert V(f"{PUA}\nok\n12 mg") == [12.0] and V(f"12 mg\nok\n{PUA}") == [12.0]
    assert V(f"{PUA}\n12 mg") == [None]                    # the first word of the next line is one word away
    assert V("1 2\n3 4\n\ufffd 5\n6 7") == [1.0, 2.0, 3.0, None, None, 6.0, 7.0]


def test_a_character_of_no_list_taints_the_whole_line_and_a_folded_context_character_taints_nothing():
    # r7 replaces the LINE-class test: the derived rule taints the line for every character of no list
    assert V("• ok ok 12 mg ok ok") == [None] and V("12 mg ok ok †") == [12.0]
    assert V("Smith† reported 12.4 mg and 5 ng") == [12.4, 5.0]
    assert V("partner_ 33 574") == [None, None] and V("partner§ 33 574") == [33.0, 574.0]


def test_a_cell_a_snippet_and_a_value_are_one_line_whatever_line_breaks_they_hold():
    for mode in ("cell", "claim"):
        assert V(f"12\nok\n{PUA}", mode=mode) == [None]
        assert V(f"{PUA}\nok\n12", mode=mode) == [None]
    assert V(f"12\nok\n{PUA}") == [12.0]


def test_the_cid_marker_is_one_outside_character():
    for text in ("31.1(cid:1)0.7", "10 (cid:1)M", "(cid:3)85%", "7.29 ×10(cid:2879)(cid:2871)", "37(cid:2)C",
                 "AUC 12 (cid:1) mg/L", "(cid:1) ok ok 12 mg", "≥(cid:3)85%"):
        toks = tokens(text)
        assert toks and all(t.value is None for t in toks), (text, toks)
    t = tokens("12 (cid:7) mg")
    assert [x.detail for x in t] == ["(cid:7) one word after"]
    assert V("(cid: 12) and (cid:x 12") == [12.0, 12.0] and V("12 (cid:x) 5") == [12.0, 5.0]
    assert V("\n(cid:1)\nok\n12 mg") == [12.0] and V("(cid:1)") == []


def test_the_detail_is_a_fixed_fragment_never_exception_text():
    assert tokens(f"{ZWSP} 12 {ZWSP}")[0].detail == "U+200B one word before"
    assert tokens(f"ok ok {PUA}\n12 mg ok")[0].detail == "U+E000 one word before"
    assert tokens(f"12 ok ok {RLO}")[0].detail == "U+202E elsewhere on the same line"
    for t in tokens(f"x {ZWSP} 1 \u2028 {PUA}"):
        assert t.detail.startswith(("U+", "(cid:")) and t.detail.count("\n") == 0


# ---------------------------------------------------------------- bidi, marks, hostile input ---
@pytest.mark.parametrize("hostile", ["\u202a", "\u202b", "\u202c", "\u202d", "\u202e", "\u2066", "\u2067", "\u2068",
                                     "\u2069", "\u061c", "\u200e", "\u200f", "\u200b", "\u200c", "\u200d", "\u2060",
                                     "\ufeff", "\u00ad", "\u034f", "\u180e", "\ufe0f", "\ufe00", "\U000e0001",
                                     "\U000e0041", "\u3164", "\uffa0", "\u115f", "\u0600", "\u06dd", "\u0300",
                                     "\u20d2", "\u0338", "\u0e47", "\ufffc", "\ufffd", "\x00", "\x1f", "\x7f", "\x85",
                                     "\U000f0000", "\ue000", "\u0378", "\U0010ffff", "\U0001d7ce", "\u2028"])
def test_every_invisible_or_unreadable_character_makes_a_touching_number_unresolved(hostile):
    for text in (f"{hostile}12.5", f"12.5{hostile}", f"1{hostile}2.5", f"AUC {hostile} 12.5", f"12.5 {hostile} mg",
                 f"AUC was 12.5 mg {hostile}"):
        toks = tokens(text)
        assert toks and all(t.value is None for t in toks), (repr(text), toks)


def test_text_without_an_ascii_digit_holds_no_number_at_all():
    assert tokens("\uff11\uff12 \u0661\u0662 \u2460 \u00be \u2155 ² ⁴") == []


def test_a_clean_twin_of_every_hostile_row_still_reads():
    assert V("AUC was 12.5 mg") == [12.5] and V("AUC 12.5 mg ok") == [12.5]


def test_the_marks_between_a_qualifier_and_its_number_never_hide_the_qualifier():
    for mark in (ZWSP, ZWJ, SHY, LRM, "\u0338", "\ufeff"):
        for text in (f"<{mark}0.5", f"< {mark} 0.5", f"≤{mark}1.0", f"AUC <{mark} 12.4", f"12{mark}-15"):
            assert all(v is None for v in V(text)), repr(text)


# ------------------------------------------------------------------ linear time ---
def _time(text: str) -> float:
    import time
    t0 = time.perf_counter()
    tokens(text)
    return time.perf_counter() - t0


@pytest.mark.parametrize("name, text", [
    ("1 MB of digits", "1234567890" * 100_000),
    ("1 MB of digits and combining marks", "1\u0301" * 500_000),
    ("1 MB of digits and zero-width spaces", "1" + ZWSP + "2" + ZWSP * 3 + "3" * 3 + "\u200d" * 7 + "45" * 600_000 + "6"),
    ("1 MB of digit groups", "1." * 500_000),
    ("1 MB of commas between digits", "1,2," * 250_000),
    ("1 MB of outside characters then a number", "\ufffd" * 1_000_000 + "12"),
    ("1 MB of gap then a number", " " * 1_000_000 + "12"),
    ("1 MB of letters", "mg " * 333_333 + "12"),
])
def test_one_megabyte_of_hostile_text_tokenises_in_under_a_second(name, text):
    assert len(text) >= 900_000, name
    assert _time(text) < limit(1.0), name


def test_a_dense_megabyte_page_tokenises_in_about_a_second_and_scales_linearly():
    unit = "Cmax was 12.5 ng/mL (CV 23%) after 150 mg, n = 12; AUC 3,412 (812) h. "
    small, big = unit * 7_000, unit * 14_000                       # 0.5 MB and 1 MB
    t_small, t_big = _time(small), _time(big)
    assert t_big < limit(2.0), t_big
    assert t_big < 3.0 * t_small + 0.05, (t_small, t_big)         # linear: doubling the input about doubles the time


def test_the_bidi_marks_and_padding_attacks_are_linear():
    base = "1 2 3 4 5 " * 6_000
    clean = _time(base)
    marks = _time(("1 " + ZWSP) * 60_000)
    assert marks < max(limit(1.5), 8 * clean) and tokens(("1 " + ZWSP) * 60_000)[-1].value is None
    padded = "AUC <" + " " * 5_000 + "12.5 mg/L"
    assert _time(padded) < max(limit(0.5), 8 * _time("AUC <12.5 mg/L") + 0.05)
    assert _time(("12 " + ZWSP + "\n") * 40_000) < limit(2.0)
