"""r6 round 2, finding 1: the snippet locator has no second normalisation table.

``normalise`` used to apply NFKC per character (with a code-point cap for U+FDFA and a test against the numeric
alphabet), Unicode ``\\s`` for whitespace, ``str.isalpha`` / ``isspace`` for de-hyphenation and ``str.casefold`` for every
character. It is now ``fold()`` (the approved equivalents, the reader's own function), seven explicit ligature
spellings, the lower case of the letters the alphabet lists, the GAP set plus the thin space for whitespace and the
alphabet's letters for de-hyphenation. Nothing else is equated, dropped or expanded, so two texts locate each other
only when the reader would read them the same way (apart from case, spacing and the seven ligatures).
"""
from __future__ import annotations

import pytest

from app.evidence import letter_tables as LT
from app.evidence import models as M
from app.evidence import normalize as N
from app.evidence.normalize import dehyphenate_with_map, normalise, strip_ws_with_map
from app.evidence.numgrammar import alphabet as A
from app.evidence.numgrammar import equivalents as EQ
from tests import evidence_r6_pins as P
from tests import evidence_r6_round2_spec as S
from tests.evidence_r6_round1_support import run
from tests.evidence_r6_sampling import non_outside_cps, stratified_outside

SAMPLE = sorted(set(non_outside_cps()) | set(stratified_outside(2048)) | set(S.LIGATURE_PIN) | set(S.CASE_PIN)
                | {0x2007, 0x2003, 0x3000, 0xFDFA, 0xFF21, 0x1D436, 0x212A, 0x017F, 0x2126, 0x212B, 0x3398})


# ------------------------------------------------------------------ the tables ---

def test_the_ligature_table_is_pinned_and_expands_to_ascii_letters_only():
    assert {ord(k): v for k, v in LT.LIGATURES.items()} == S.LIGATURE_PIN
    assert all(v.isascii() and v.isalpha() and v == v.lower() for v in LT.LIGATURES.values())
    assert not set(LT.LIGATURES) & (set(EQ.EQUIV) | A.INSIDE | A.CONTEXT | A.GAP)     # outside the reader


def test_the_case_table_is_pinned_one_to_one_and_complete_for_the_alphabets_letters():
    assert {ord(k): ord(v) for k, v in LT.CASE_PAIRS.items()} == S.CASE_PIN
    letters = A.ASCII_LETTERS | A.GREEK | A.ACCENTED          # r7: kappa is Greek (CONTEXT), chi is outside
    assert letters == A.LETTERS == S.LETTERS_PIN and len(letters) == 98 and "\u03c7" not in letters
    assert LT.UPPER_LETTERS == frozenset(LT.CASE_PAIRS) and len(LT.CASE_PAIRS) == 34
    assert (len(LT.LIGATURES), len(LT.HEADER_SPELLINGS)) == (7, 1) and LT.UPPER_LETTERS <= A.LETTERS
    for ch in letters:                                        # the explicit pairs agree with Unicode on every letter
        assert LT.casefold_letters(ch) == ch.casefold() and len(ch.casefold()) == 1, ch
    assert LT.casefold_letters("ΔΩΣÅ𝐶\u212a\u017f") == "δωσå𝐶\u212a\u017f"     # an unlisted character stays as it is


# ------------------------------------------------------------- one character at a time ---

def test_a_character_between_two_letters_becomes_exactly_what_the_specification_says():
    bad = []
    for cp in SAMPLE:
        norm, m = normalise("a" + chr(cp) + "b")
        want = "a" + S.spec_locator(chr(cp)) + "b"
        if norm != want or not (len(m) == len(norm) and m[0] == 0 and m[-1] == 2 and set(m[1:-1]) == {1}):
            bad.append((hex(cp), norm, want, m))
    assert bad[:10] == []


def test_a_compatibility_form_is_never_the_character_it_looks_like():
    """NFKC used to turn each of these into plain ASCII letters (or into text that was longer than the character), so a
    snippet typed with the plain spelling located a page that printed the other one."""
    for ch in ("𝐶", "𝑡", "Ｃ", "ｍ", "™", "№", "Ⅳ", "ℓ", "㎎", "℃", "\u212a", "\u212b", "\u2126", "\u017f", "ﷺ", "¨", "´", "ª",
               "²", "½", "⁄"):
        assert normalise(f"x{ch}y")[0] == f"x{ch}y", ch
    assert normalise("𝐶max")[0] != normalise("Cmax")[0]
    assert normalise("ﷺ" * 50)[0] == "ﷺ" * 50


def test_the_equivalents_are_the_readers_own_fold():
    for src, dst in EQ.EQUIV.items():
        a, b = normalise(f"a{src}b")[0], normalise(f"a{dst}b")[0]
        assert a == b, (hex(ord(src)), a, b)
    assert normalise("−20")[0] == normalise("-20")[0] == normalise("–20")[0] == normalise("‑20")[0]
    assert normalise("1 234")[0] == normalise("1 234")[0] == normalise("1 234")[0] == "1 234"


def test_a_ligature_is_spelled_with_its_letters_and_the_index_map_follows():
    norm, m = normalise("ﬁne ﬂow ﬃ")
    assert norm == "fine flow ffi" and m == [0, 0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 8, 8]
    assert normalise("ﬆ")[0] == normalise("ﬅ")[0] == "st"


def test_only_the_listed_capitals_are_folded_to_lower_case():
    assert normalise("CMAX ΔOFV ÅÖ")[0] == "cmax δofv åö"
    assert normalise("\u212a")[0] == "\u212a" and normalise("\u017f")[0] == "\u017f"   # Kelvin sign, long s: not listed letters
    assert normalise("É")[0] == "É"                                        # E-acute is not listed (only its lower case)


# ----------------------------------------------------- whitespace and de-hyphenation ---

def test_whitespace_collapses_over_the_gap_set_and_the_thin_space_only():
    assert strip_ws_with_map("  a \t b\n\n c ")[0] == "a b c"
    for other in (" ", " ", "　", "\u000b", "\u0085", " "):
        assert strip_ws_with_map(f"a{other}b")[0] == f"a{other}b", hex(ord(other))     # not a space of the alphabet


def test_a_hyphen_at_a_line_break_joins_only_the_letters_of_the_alphabet():
    assert dehyphenate_with_map("clear-\nance")[0] == "clearance"
    assert dehyphenate_with_map("Müller-\nWeiss")[0] == "MüllerWeiss"
    assert dehyphenate_with_map("α-\nβ")[0] == "αβ" and dehyphenate_with_map("κ-\nλ")[0] == "κλ"
    for text in ("0-\n24", "прибор-\nно", "a - \nb", "x-\n٣", "-\nb", "a-\n", "κ-\nχ"):     # r7: chi is outside
        assert dehyphenate_with_map(text)[0] == text, text


def test_the_wrapped_word_still_locates_and_the_map_points_into_the_raw_text():
    raw = "ﬁrst clear-\nance  was\n2.31 L/h"
    norm, m = normalise(raw)
    assert norm == "first clearance was 2.31 l/h"
    i = norm.index("clearance")
    assert raw[m[i]:m[i + 8] + 1] == "clear-\nance"


# ------------------------------------------------------ end to end: a text-field record ---

@pytest.mark.parametrize("page, snippet", [("\ufb01nal dose 100 mg", "Final dose 100 mg"), ("\ufb02ow 100 mg", "flow 100 mg"),
                                           ("dose\u00a0100 mg", "dose 100 mg"), ("Dose  100 mg", "dose 100 mg"),
                                           ("Dose \u22125 mg", "Dose -5 mg"), ("Clearance was 100\nmg", "clearance was 100 mg"),
                                           ("Dose 100 mg\u200b", "Dose 100 mg")])
def test_a_snippet_still_locates_through_the_equivalents_ligatures_case_and_spacing(page, snippet):
    got = run(page, snippet, snippet, field="Cell dose")
    assert (got.status, got.reason) == ("pass", M.REASON_OK), (page, snippet, got)


@pytest.mark.parametrize("page, snippet", [("Dose \U0001d436100 mg", "Dose C100 mg"), ("Dose \uff23100 mg", "Dose C100 mg"),
                                           ("Dose 100 \u338e", "Dose 100 mg"), ("Dose\u2003100 mg", "Dose 100 mg"),
                                           ("Dose\u3000100 mg", "Dose 100 mg"), ("Dose 100 mg\u2122", "Dose 100 mgTM")])
def test_a_page_that_differs_by_a_compatibility_form_or_an_unlisted_space_does_not_hold_the_snippet(page, snippet):
    got = run(page, snippet, snippet, field="Cell dose")
    assert (got.status, got.reason) == ("fail", M.REASON_SNIPPET_NOT_ON_PAGE), (page, snippet, got)


@pytest.mark.parametrize("snippet", ["Dose \ufb01 100 mg", "Dose \uff23100 mg", "Dose\u2003100 mg", "Dose 100 mg\u200b",
                                     "Dose \U0001d436100 mg", "Dose 100 \u338e"])
def test_a_snippet_with_a_character_outside_the_alphabet_is_refused_whatever_the_page_holds(snippet):
    for page in (snippet, "Dose C100 mg", "Dose 100 mg"):
        got = run(page, snippet, snippet, field="Cell dose")
        assert (got.status, got.reason) == ("fail", M.REASON_UNRESOLVED_CONTEXT), (page, snippet, got)


def test_the_locator_module_has_no_second_table_and_no_unicode_property():
    import ast
    from pathlib import Path
    tree = ast.parse(Path(N.__file__).read_text("utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree)
                                                                          if isinstance(n, ast.Attribute)}
    assert not names & {"unicodedata", "nfkc_char", "nfkc_with_map", "MAX_NFKC_EXPANSION", "isalpha", "isspace",
                        "isupper", "islower", "isdigit", "normalize"}
    # r7 round 4: every compiled pattern of the module (they are built from the lists now, none is a literal), was:
    # the literal patterns of the module's re.compile calls
    import re
    patterns = [v.pattern for v in vars(N).values() if isinstance(v, re.Pattern)]
    assert patterns and not any("\\s" in p or "\\w" in p or "\\b" in p for p in patterns)


def test_expected_class_of_the_ligatures_and_case_pairs_is_still_outside_or_a_listed_letter():
    for cp in S.LIGATURE_PIN:
        assert P.expected_class(cp) == "outside"        # a ligature is no character of a number: the reader taints it
