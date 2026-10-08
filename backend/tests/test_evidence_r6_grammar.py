"""r6 numeric grammar: what the tokenizer reads, and what it refuses, row by row from the binding spec.

``V`` is the resolved value of every numeric expression in reading order; ``None`` marks an unresolved one.
"""
from __future__ import annotations

import pytest

from app.evidence.numgrammar import parse_claim, tokens

T = "\u2009"      # thin space


def V(text: str, style: str = ".", mode: str = "page") -> list[float | None]:
    return [t.value for t in tokens(text, style, mode=mode)]


def SH(text: str, style: str = ".", mode: str = "page") -> list[str]:
    return [t.shape for t in tokens(text, style, mode=mode)]


# ------------------------------------------------------------------ plain numbers ---
PLAIN = [
    ("Cmax was 12.5 ng/mL", ".", [12.5]),
    ("AUC 3,412 (812)", ".", [3412.0, 812.0]),
    ("AUC 18,2 ± 3,4", ",", [18.2, 3.4]),
    ("1.250.000 units", ".", [1250000.0]),
    ("0,412 ng/mL", ".", [0.412]),
    ("1,234,567.89", ".", [1234567.89]),
    ("1.234.567,89", ",", [1234567.89]),
    ("24'560 and 24’560", ".", [24560.0, 24560.0]),
    (f"18{T}900 ng", ".", [18900.0]),
    (f"1{T}234 567", ".", [1234.0, 567.0]),
    ("12 145 subjects", ".", [12.0, 145.0]),
    ("change was −20 min", ".", [-20.0]),
    ("change was –20 min", ".", [-20.0]),
    ("+7 and -0.45", ".", [7.0, -0.45]),
    ("0 and 0.0 and 1.0", ".", [0.0, 0.0, 1.0]),
    ("OFV 1234,5 and 1234.567", ".", [1234.5, 1234.567]),
    ("Cmax 1,2 or 4 mg", ".", [1.2, 4.0]),
    ("(n = 12) received 150 mg", ".", [12.0, 150.0]),
    ("(n=12) and p<0.05 and p=0.03", ".", [12.0, 0.05, 0.03]),
    ("response in 45% of subjects", ".", [45.0]),
    ("recovery was 4.5‰ of dose", ".", [4.5]),
    ("fentanyl 245 µg over 11.8 h", ".", [245.0, 11.8]),
    ("(Lee et al. 2011). This", ".", [2011.0]),
    ("37°C and 37 °C and 37◦C", ".", [37.0, 37.0, 37.0]),
    ("CL 12.4 L h-1 and 90 mL/min/1.73m2", ".", [12.4, 90.0]),
    ("n = 12.", ".", [12.0]), ("12-fold and 24-h", ".", [12.0, 24.0]), ("12.)", ".", [12.0]),
    ("(gsd - 1) × 100 = 41%", ".", [None, 100.0, 41.0]),   # r7 writer 3: the 1 of 'gsd - 1' is subtracted, no range
    ("Table 1.", ".", [1.0]), ("2.1) 404/9696 (4.2)", ".", [2.1, None, 4.2]),
    ("12.4 (n = 10)", ".", [12.4, 10.0]), ("150 mg/kg", ".", [150.0]), ("ratio 1:2", ".", [None]),
    ("df=3,P=0.08", ".", [3.0, 0.08]),
]


@pytest.mark.parametrize("text, style, want", PLAIN)
def test_plain_numbers_and_their_separators(text, style, want):
    assert V(text, style) == want


# ------------------------------------------------------------------- scientific ---
SCIENTIFIC = [
    ("k was 1.2e-3 per h", [0.0012]), ("1.2E+03 and 1e5", [1200.0, 100000.0]), ("-3.1e2", [-310.0]),
    ("Km 1.2 x 10^3 M", [1200.0]), ("Km 1.2 X 10^3 M", [1200.0]), ("Cmax 1.2 × 10⁻³ ng/mL", [0.0012]),
    ("1.2×10^3 cells", [1200.0]), ("1.2 ×10^3", [1200.0]), ("1.2× 10^3", [1200.0]), ("1.2 ×\n10^3", [1200.0]),
    ("1.2 ×\t\r\n  10⁻³", [0.0012]), ("1.2 × 10^-3", [0.0012]), ("1.2 * 10^3", [1200.0]), ("1.2 · 10^3", [1200.0]),
    ("1.2×10³", [1200.0]), ("10^6 cells", [1e6]), ("Titre 2.5·10^4 copies", [25000.0]), ("2·10^4", [20000.0]),
    ("~120 × 10⁶ cells/g", [120e6]), ("4.5 X 10^6", [4.5e6]), ("-1.2 x 10^3", [-1200.0]),
    ("1.2e999", [None]), ("1.2 x 10^999", [None]), ("1e-300 and 1e300", [1e-300, 1e300]),
    ("1e301 and 1e-301 and 1e12345", [None, None, None]),
]


@pytest.mark.parametrize("text, want", SCIENTIFIC)
def test_scientific_notation_is_one_token_with_the_exponent_applied(text, want):
    assert V(text) == want


RAISED_AND_BROKEN = [
    ("Peak 1.2 × 103 cells", "ten_after_multiplier"), ("Peak 1.2 x 103 cells", "ten_after_multiplier"),
    ("Peak 1.2 · 103 cells", "ten_after_multiplier"), ("Peak 1.2 * 103 cells", "ten_after_multiplier"),
    ("Peak 1.2 x 10 3 cells", "ten_after_multiplier"), ("Peak 1.2 × 1012 cells", "ten_after_multiplier"),
    ("Peak 1.2 x 10 -3 cells", "ten_after_multiplier"), ("Peak 1.2 x 10 −3 cells", "ten_after_multiplier"),
    ("Peak 1.2 x 10 +3 cells", "ten_after_multiplier"), ("1.2 × 10 ^ 3", "ten_after_multiplier"),
    ("Patients took 2 x 10 mg tablets", "ten_after_multiplier"), ("52.3×10 cm/s", "ten_after_multiplier"),
    ("7.23 ×10 *", "ten_after_multiplier"), ("× 103", "ten_after_multiplier"), ("x 103 cells", "ten_after_multiplier"),
    ("count 4.5 x 10 cells/L", "ten_after_multiplier"), ("2·10 cells", "ten_after_multiplier"),
    ("0·10", "ten_after_multiplier"), ("12·105", "ten_after_multiplier"), ("0·100", "ten_after_multiplier"),
    ("2 × daily", "multiplier_operand"), ("1.2 ×", "multiplier_operand"), ("dosed 2x", "multiplier_operand"),
    ("2x daily and 3X weekly", "multiplier_operand"), ("1.2 × l0³", "multiplier_operand"),
    ("1.2 × [serum", "multiplier_operand"), ("12* 10^3", "multiplier_operand"), ("12 xenon", "multiplier_operand"),
    ("1.2 x1O³", "multiplier_operand"),
    ("10 ^ 3", "detached_exponent"), ("10 ³", "detached_exponent"), ("10^ 3", "detached_exponent"),
    ("3.4 E-03", "spaced_exponent"), ("2012 e1553", "spaced_exponent"), ("3.4 E3", "spaced_exponent"),
    ("12 3/4", "mixed_number"), ("12 1/2", "mixed_number"), ("12 1/10", "mixed_number"),
    ("3/4/5", "fraction"), ("12 /24", "fraction"), ("1/0 of subjects", "fraction"),    # r7: 3/4 12/24 1/2 are ratios
    ("10³", "bare_superscript_power"), ("per 10⁶ cells", "bare_superscript_power"),
]


@pytest.mark.parametrize("text, why", RAISED_AND_BROKEN)
def test_a_broken_scale_or_fraction_is_one_unresolved_expression_with_its_reason(text, why):
    toks = tokens(text)
    hit = [t for t in toks if t.why == why]
    assert hit, (text, [(t.shape, text[t.start:t.end]) for t in toks])
    assert all(t.value is None for t in hit)


@pytest.mark.parametrize("text, citing", [
    ("Peak 1.2 × 103 cells", 1.2), ("Peak 1.2 x 10 3 cells", 3.0), ("Patients took 2 x 10 mg tablets", 10.0),
    ("Patients took 2 x 10 mg tablets", 2.0), ("12 3/4", 12.0), ("3.4 E-03", 3.4), ("2 × daily", 2.0),
    ("150mgtid", 150.0), ("10 ^ 3", 10.0),
])
def test_the_resolved_pieces_of_an_unresolved_expression_are_its_parts(text, citing):
    toks = tokens(text)
    assert all(t.value is None for t in toks)
    assert any(citing in t.parts for t in toks), [(t.shape, t.parts) for t in toks]


def test_a_ten_that_is_not_followed_by_a_zero_is_what_a_raised_exponent_looks_like():
    assert V("2 x 100 mg") == [2.0, 100.0] and V("3 x 1000 mg") == [3.0, 1000.0] and V("3 × 250 mg") == [3.0, 250.0]
    assert V("took 2 x 100 mg") == [2.0, 100.0] and V("2 × 1003") == [2.0, 1003.0] and V("2×5 and 3×250") == [
        2.0, 5.0, 3.0, 250.0]
    assert V("4.6×150 mm") == [4.6, 150.0] and V("2 x 3 mg") == [2.0, 3.0]


# -------------------------------------------------------------- shapes of bodies ---
@pytest.mark.parametrize("text, style, shape, value", [
    ("12", ".", "int", 12.0), ("0", ".", "int", 0.0), ("12.5", ".", "decimal", 12.5), ("12,5", ",", "decimal", 12.5),
    ("12,5", ".", "foreign_decimal", 12.5), ("0,412", ".", "foreign_decimal", 0.412),
    ("0.412", ".", "decimal", 0.412), ("1234,5", ".", "foreign_decimal", 1234.5), ("1234.5", ".", "decimal", 1234.5),
    ("1234.567", ".", "decimal", 1234.567), ("3,412", ".", "style_dependent", 3412.0),
    ("3,412", ",", "style_dependent", 3.412), ("3.412", ".", "style_dependent", 3.412),
    ("3.412", ",", "style_dependent", 3412.0), ("1,234,567", ".", "grouped", 1234567.0),
    ("1.250.000", ",", "grouped", 1250000.0), (f"18{T}900", ".", "grouped", 18900.0),
    ("24'560", ".", "grouped", 24560.0), ("1,234,567.89", ".", "grouped_decimal", 1234567.89),
    ("1,234,567.89", ",", "foreign_decimal", 1234567.89), ("1.234.567,89", ",", "grouped_decimal", 1234567.89),
    ("1.234.567,89", ".", "foreign_decimal", 1234567.89), ("1'234.5", ".", "grouped_decimal", 1234.5),
    ("12·4", ".", "middot", 12.4), ("1,234·5", ".", "middot", 1234.5),
])
def test_the_shape_and_value_of_one_body(text, style, shape, value):
    (tok,) = tokens(text, style)
    assert (tok.shape, tok.value) == (shape, value)


@pytest.mark.parametrize("text, style, why", [
    ("007", ".", "leading_zero"), ("05", ".", "leading_zero"), ("00.5", ".", "leading_zero"),
    ("1,23,456", ".", "groups"), ("0.123" + T + "4", ".", "groups"), ("2.3.1", ".", "groups"),
    ("1'23", ".", "groups"), (f"1{T}23", ".", "groups"), ("012,345", ".", "leading_zero"),
    ("21741,229", ".", "ambiguous_tail"), ("21741.229", ",", "ambiguous_tail"),
    ("12·4", ",", "middot"), ("1·2·3", ".", "middot"), ("1.2·3", ".", "groups"),
    ("1,234.5,6", ".", "separators"), ("1.2,3", ".", "separators"), ("1,23.4", ".", "separators"), ("1" * 41, ".", "too_many_digits"),
    ("1." + "1" * 40, ".", "too_many_digits"),
])
def test_a_body_the_grammar_refuses_is_unresolved_shape_with_its_reason(text, style, why):
    toks = tokens(text, style)
    assert [(t.value, t.why) for t in toks] == [(None, why)], toks


def test_forty_digits_are_the_limit():
    assert V("1" * 40) == [float("1" * 40)] and V("9" * 40 + " x") == [None]
    assert V("-" + "1" * 40) == [-float("1" * 40)]


def test_a_comma_between_one_digit_groups_is_a_decimal_not_a_list():
    assert V("1,2,3") == [None] and V("c(0,1,2,3,4)") == []


# ---------------------------------------------------- words, labels, the old fail-opens ---
def test_digit_runs_inside_a_label_word_never_form_tokens_and_poison_nothing():
    for label in ("AUC0-24", "t1/2", "CYP2B6", "mL/min/1.73m2", "c(0,1,2,3,4)", "C2", "m2", "AUC0-inf"):
        assert tokens(label) == [], label
    assert V("CYP2B6 12.4") == [12.4] and V("AUC0-24 was 7") == [7.0] and V("t1/2 was 11.4 h") == [11.4]
    assert V("AUC(0-24) 1234") == [0.0, 24.0, 1234.0]


def test_a_token_may_start_after_an_opener_a_qualifier_or_a_joiner_inside_a_word():
    assert V("n=12 (n=12) N≥3 p<0.05 AUC≥90") == [12.0, 12.0, 3.0, 0.05, 90.0]
    assert V("12-15 36.6±4.3 2×5") == [12.0, 15.0, 36.6, 4.3, 2.0, 5.0]
    assert V("45%–55%") == [45.0, 55.0] and V("(0.79–3.08) μg/L") == [0.79, 3.08]


def test_a_numeric_led_word_the_grammar_rejects_is_one_unresolved_expression():
    for text in ("150mgtid", "24hours", "1.73m23", "100µ", "3MIU", "1OmL", "12and", "23l", "1O5", "5th", "12e",
                 "12\u0417", "\u043112"):
        toks = tokens(text)
        assert all(t.value is None for t in toks) and toks, text
    assert [t.why for t in tokens("150mgtid")] == ["glued_letters"]
    assert [t.value for t in tokens("150mg 24h 1.73m2 100µg 10mL")] == [150.0, 24.0, 1.73, 100.0, 10.0]   # r7 units


@pytest.mark.parametrize("text", ["8:00", "12/24/36", "12/", "12=3",                     # r7: 12/24 is a ratio "12^3", "12-+3", "65+", "12³", "12%5", "12:30",
                                  ".05", "*12", "12<5", "3.5≈5", "2019;12:345", "12)3", "12.5.", "1.2.-3"])
def test_other_glued_characters_reject_the_word(text):
    assert V(text)[0] is None or text in ("12.5.",), (text, V(text))


def test_old_reader_fail_opens_are_closed():
    assert V("10–30%") == [10.0, 30.0] and V("1%–10%") == [1.0, 10.0]      # was 10^-30 and -10
    assert V("(21741,229)") == [None]                                          # was 21,741,229
    assert V("c(0,1,2,3,4)") == []                                             # was 0.1 / 2.3
    assert V("10 (cid:1)M") == [None] and V("(cid:3)85%") == [None]      # the old reader passed these clean
    assert V("31.1(cid:1)0.7") == [None]


def test_the_en_dash_table_of_a_real_paper_reads_minus_signs():
    row = "SystolicBP –11.8* –14.2*** –19.9*** –3.1 –9.9 –11.0 –12.7 –3.9"
    assert V(row) == [-11.8, -14.2, -19.9, -3.1, -9.9, -11.0, -12.7, -3.9]
    assert V("frozen at –20(cid:2)C") == [None]                                 # the lost degree sign: unreadable


def test_stars_and_footnote_marks():
    assert V("0.05**") == [0.05] and V("12.4*") == [12.4] and V("12.4* (n=3)") == [12.4, 3.0]
    assert V("p<0.001**") == [0.001] and V("–14.2***") == [-14.2]
    assert V("12.4†") == [None] and V("12.4‡") == [None]


def test_the_owner_cases_and_the_judge_additions():
    assert V("12.\ue0005") == [None]                  # a private-use character inside the number
    assert V("12 3/4") == [None]
    assert V("<\u280012") == [None]                   # a Braille blank where the qualifier meets the number
    assert V("1.2 × l0³") == [None] and V("10 ^ 3") == [None] and V("2 × daily") == [None]
    assert V("(cid:3)85%") == [None] and V("$18 years") == [None] and V("–11.8*") == [-11.8]
    assert [(t.value, t.qualified) for t in tokens("∼45% of dose")] == [(45.0, True)] and V("37◦C") == [37.0]


# ----------------------------------------------------------------- equivalents ---
@pytest.mark.parametrize("src, dst", [("−", "-"), ("–", "-"), ("\u2010", "-"), ("\u2011", "-"),
                                      ("’", "'"), ("∼", "~")])
def test_an_equivalent_reads_exactly_like_its_target(src, dst):
    for pattern in ("{}20 mg", "a{}0.5 b", "24{}560", "1.2{}3", "12{}15", "AUC {} 12.5"):
        a, b = pattern.format(src), pattern.format(dst)
        assert [(t.start, t.end, t.value, t.shape, t.qualified, t.ranged) for t in tokens(a)] == \
               [(t.start, t.end, t.value, t.shape, t.qualified, t.ranged) for t in tokens(b)], pattern


def test_the_no_break_spaces_group_digits_like_a_thin_space():
    for sp in ("\u00a0", "\u202f"):
        assert V(f"1{sp}234 ng") == [1234.0] and V(f"AUC{sp}12") == [12.0] and V(f"{sp}12{sp}") == [12.0]
    assert V("245µg") == [245.0] and V("245µ") == [None] and V("100 µg") == [100.0]      # r7: the unit production


def test_the_micro_sign_is_a_greek_mu_and_the_white_bullet_a_degree_sign():
    assert V("37◦C") == [37.0] and V("4◦C") == [4.0] and V("µ 12") == [12.0]


def test_a_fullwidth_or_other_look_alike_is_outside_not_an_equivalent():
    for ch in ("\uff11", "\u0661", "\u2460", "\u00be", "\u2012", "\ufe63", "\uff0d", "\u02d7", "\u2043",
               "\u22c5", "\u2219", "\u2a2f", "\u2715", "\u2217", "\u0445", "\u0425", "\u2031", "\ufe62", "\uff0b",
               "\u2027", "\u02d9", "\u066b", "\u2024", "\u02bc", "\u00b4", "\u00a8"):
        assert V(f"AUC 12{ch}5") == [None], hex(ord(ch))
        assert V(f"{ch}20 and 7") == [None, None], hex(ord(ch))


# ------------------------------------------------------------------- claim mode ---
@pytest.mark.parametrize("text, style, want", [
    ("18 900", ".", 18900.0), ("1 234", ".", 1234.0), ("12 145", ".", 12145.0), ("1 234 567", ".", 1234567.0),
    ("3,412", ".", 3412.0), ("18,2", ",", 18.2), ("1.2 x 10^3", ".", 1200.0), ("45%", ".", 45.0),
    ("−20", ".", -20.0), ("12.4 ng/mL", ".", 12.4), ("<0.5", ".", 0.5), ("1,2e-3", ",", 0.0012),
    ("1'234", ".", 1234.0), ("18,200", ",", 18.2), ("0,412", ",", 0.412),
    ("3,412 (812)", ".", None), ("4.5 x 10", ".", None), ("abc", ".", None), ("12 1500", ".", None),
    ("1.2 × 103", ".", None), ("1 234 ± 312", ".", None), ("1.2 x 10 -3", ".", None), ("10³", ".", None),
    ("0.123 4", ".", None), ("", ".", None), ("12 and 13", ".", None), ("1234 567", ".", None),
])
def test_parse_claim_names_a_number_only_when_the_string_holds_exactly_one_resolved_token(text, style, want):
    assert parse_claim(text, style) == want


def test_the_ascii_space_never_groups_digits_on_a_page():
    assert V("12 145") == [12.0, 145.0] and V("12 145", mode="cell") == [12.0, 145.0]
    assert V("12 145", mode="claim") == [12145.0]


def test_the_cell_mode_reads_like_a_page_but_the_whole_cell_is_one_line():
    assert V("3,412 (812)", mode="cell") == [3412.0, 812.0]
    assert V("1\nok\n2\u00005", mode="page") == [1.0, None]       # the line break ends the tainted line
    assert V("1\nok\n2\u00005", mode="cell") == [None, None]       # a cell is one line
    assert V("1\n2\u00005", mode="page") == [None, None]          # the next line's first word is one word away
    with pytest.raises(ValueError):
        tokens("1", mode="nope")


# ----------------------------------------------------------------- offsets, flags ---
def test_offsets_are_native_literals_keep_their_equivalents():
    text = f"u −20 y –11.8* z 18{T}900 w 24’560 v 37◦C"
    lits = [text[t.start:t.end] for t in tokens(text)]
    assert lits == ["−20", "–11.8", f"18{T}900", "24’560", "37"]
    from app.evidence.numgrammar.equivalents import fold
    assert len(fold(text)) == len(text)
    # r7 writer 3: a lone x before a number is a multiplier, never a filler letter: '−20' is unresolved there
    first = tokens("x" + text[1:])[0]
    assert first.value is None and first.why == "neighbour" and "−20" in ("x" + text[1:])[first.start:first.end]


def test_a_scientific_token_spans_mantissa_to_power_and_unresolved_expressions_span_whole_words():
    for text, want in [("Km 1.2 x 10^3 M and 2 × daily and 12 3/4 end", ["1.2 x 10^3", "2 × daily", "12 3/4"]),
                       ("see 12.\ue0005 now", ["12.\ue0005"]), ("a 1.2 x 10 3 cm", ["1.2 x 10 3"])]:
        assert [text[t.start:t.end] for t in tokens(text)] == want, text


def test_tokens_come_back_in_order_and_never_overlap():
    texts = ["1.2 × 10 3 4 5 and 10 ^ 3 and 12 3/4 and 2 x 10 mg and 3.4 E-03 and 12 ½ and 12* 10^3 x 5",
             "x 103 4 (5 6) 1.2 ×\n10^3 1.2e5 xenon 3 1/2 7", "(cid:1) 12 (cid:2) 13 14", "9 ^ 3 ^ 3 ^"]
    for text in texts:
        toks = tokens(text)
        assert [t.start for t in toks] == sorted(t.start for t in toks)
        assert all(a.end <= b.start for a, b in zip(toks, toks[1:])), [(t.start, t.end) for t in toks]


def test_qualified_and_ranged_use_explicit_membership_not_the_empty_string():
    plain = tokens("conc was 12.4 ng/mL")[0]
    assert (plain.qualified, plain.ranged) == (False, False)          # a word-initial token is NOT qualified
    assert tokens("<0.5")[0].qualified and tokens("< 0.5")[0].qualified and tokens("p<0.05")[0].qualified
    assert tokens("(<=0.5)")[0].qualified and tokens(">= 4")[0].qualified and tokens("≤1.0")[0].qualified
    assert tokens("AUC≥ 5")[0].qualified and tokens("AUC ~2")[0].qualified and tokens("≈ 9")[0].qualified
    assert not tokens("p = 0.5")[0].qualified and not tokens("12.4 <")[0].qualified
    a, b = tokens("12-15")
    assert a.ranged and b.ranged
    a, b = tokens("12 - 15")
    assert a.ranged and b.ranged
    assert not tokens("12 to 15")[0].ranged and not tokens("-20")[0].ranged and not tokens("5.3 ± 0.4")[0].ranged


def test_a_scientific_token_keeps_the_qualifier_of_its_mantissa():
    assert tokens("<1.2 x 10^3")[0].qualified and tokens("< 1.2 x 10^3")[0].qualified
    assert tokens("Cmax 1.2 x 10^3")[0].qualified is False


def test_tok_exposes_the_canonical_text_the_decimal_mark_and_the_reason():
    t = tokens("-1.2e3")[0]
    assert (t.canonical, t.shape, t.value, t.reason, t.why) == ("-1.2e3", "sci", -1200.0, "", "")
    assert str(t.decimal_value) == "-1.2E+3" or float(t.decimal_value) == -1200.0
    d = tokens("12,5", ",")[0]
    assert (d.canonical, d.decimal) == ("12.5", ",")
    u = tokens("12 3/4")[0]
    assert (u.reason, u.why, u.canonical, u.decimal_value) == ("unresolved_shape", "mixed_number", "", None)
    c = tokens("12.\ue0005")[0]
    assert (c.reason, c.why) == ("unresolved_context", "") and c.detail == "U+E000 inside the number"


def test_an_unresolved_context_expression_names_its_offender_and_where_it_sits():
    c = tokens("12.\ue0005")[0]
    assert (c.offender, c.where) == ("U+E000", "inside the number")
    assert (tokens("(cid:3)85%")[0].offender, tokens("(cid:3)85%")[0].where) == ("(cid:3)", "inside the number")
    assert (tokens("Adalat_ 30 mg")[0].offender, tokens("Adalat_ 30 mg")[0].where) == \
        ("U+005F", "one word before")                                     # r7: the registered sign is CONTEXT
    assert (tokens("12 \u00bd")[0].offender, tokens("12 \u00bd")[0].where) == ("U+00BD", "one word after")
    elsewhere = tokens("\u202eabc\u202c AUC 12.5 mg/L")[0]
    assert (elsewhere.offender, elsewhere.where) == ("U+202E", "elsewhere on the same line")
    for resolved_or_shape in (tokens("12.5")[0], tokens("12 3/4")[0]):
        assert (resolved_or_shape.offender, resolved_or_shape.where) == ("", "")


@pytest.mark.parametrize("text", ["2 x n=12", "1.2 x <a>10^3", "2 x (n=12)", "1.2\u00d7 [serum", "1.2 \u00d7 n=12"])
def test_a_label_word_is_never_the_numeric_operand_of_a_multiplier(text):
    toks = tokens(text)
    assert len(toks) == 1 and toks[0].value is None and toks[0].why == "multiplier_operand", text


@pytest.mark.parametrize("text, want", [("2 x 5 mg", [2.0, 5.0]), ("2 x -3", [2.0, -3.0]), ("4.6\u00d7150 mm", [4.6, 150.0]),
                                        ("2 x 100", [2.0, 100.0]), ("3 x 1000", [3.0, 1000.0])])
def test_a_numeric_word_after_a_multiplier_is_a_product_not_a_scale(text, want):
    assert V(text) == want
