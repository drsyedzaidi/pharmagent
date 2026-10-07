"""r6 round 2, findings 2 and 3: units, header keys and the ink pairing fold through the readers own table, and the unit
token reader holds no character the alphabet puts outside.

Findings 2 (unit_norm, extract_rules.norm_key, parse_ink_match used NFKC and ``str.is*`` outside the reader's one
equivalence table) and 3 (``record_checks`` accepted U+2031, which the alphabet classifies OUTSIDE). A unit that holds a
character outside INSIDE, CONTEXT and GAP is unresolved: it equals no unit (not even another spelling of itself with a
different case), is recognised as none and raises the unverified-unit warning. A header key keeps every character it
cannot fold, so a header with an invisible or look-alike character matches no field.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from app.evidence import parse_ink_match as PIM
from app.evidence import record_checks as RC
from app.evidence import unit_norm as UN
from app.evidence.extract_rules import norm_key
from app.evidence.numgrammar import alphabet as A
from app.evidence.numgrammar.equivalents import EQUIV, fold
from tests import evidence_r6_round2_spec as S
from tests.evidence_r6_sampling import non_outside_cps, stratified_outside

EVIDENCE = Path(__file__).resolve().parents[1] / "app" / "evidence"
SAMPLE = sorted(set(non_outside_cps()) | set(stratified_outside(2048)) | set(S.LIGATURE_PIN) | set(S.CASE_PIN)
                | {0x2007, 0x2003, 0x3000, 0xFDFA, 0xFF21, 0x1D436, 0x212A, 0x017F, 0x2126, 0x212B, 0x3398, 0x338E})


def _imports_unicodedata(module: str) -> bool:
    tree = ast.parse((EVIDENCE / module).read_text("utf-8"))
    return any((isinstance(n, ast.Import) and any(a.name == "unicodedata" for a in n.names))
               or (isinstance(n, ast.ImportFrom) and n.module == "unicodedata") for n in ast.walk(tree))


@pytest.mark.parametrize("module", ["unit_norm.py", "extract_rules.py", "parse_ink_match.py", "normalize.py"])
def test_the_three_modules_of_the_finding_and_the_locator_import_no_unicodedata(module):
    assert not _imports_unicodedata(module)


# ----------------------------------------------------------------------------- units ---

def test_a_unit_is_compared_through_the_equivalents_case_and_the_synonyms():
    assert UN.normalise_unit("µg/mL") == UN.normalise_unit("μg/ml") == UN.normalise_unit("ug/ML") == "ug/ml"
    assert UN.normalise_unit("mcg") == "ug" and UN.normalise_unit("Hours") == "h" and UN.normalise_unit("litres") == "l"
    assert UN.normalise_unit("mg·kg⁻¹") == UN.normalise_unit("mg kg-1") == UN.normalise_unit("mg·kg⁻¹")
    assert UN.normalise_unit("m²") == UN.normalise_unit("m2") and UN.normalise_unit("mg kg") == "mgkg"
    assert UN.normalise_unit("−1") == UN.normalise_unit("-1") and UN.normalise_unit(None) == ""
    assert UN.normalise_unit("shrs") == "shrs" and UN.normalise_unit("secs") == "secs"   # a synonym is a whole word only
    assert UN.canonical_unit("mcg/cc") == "ug/ml" and UN.same_unit("µg", "mcg") and not UN.same_unit("mg", "mg/kg")


@pytest.mark.parametrize("unit", ["㎎", "ｍｇ", "mг", "mg​", "mg­", "ﬁ", "m g", "mg̀", "𝐦g",
                                  "mg½", "mg_h", "mg", "Åmg"])
def test_a_unit_with_a_character_outside_the_alphabet_is_unresolved_and_equals_no_unit(unit):
    assert not UN.unit_resolved(unit)
    assert UN.normalise_unit(unit) != UN.normalise_unit("mg") and not UN.same_unit(unit, "mg")
    assert UN.canonical_unit(unit) != "mg" and RC.recognised_unit(unit) is None
    assert UN.prefix_conflict(unit, "mg") is None and UN.prefix_conflict("mg", unit) is None
    if unit.swapcase() != unit:
        assert UN.normalise_unit(unit) != UN.normalise_unit(unit.swapcase())      # never a case-insensitive match


def test_an_unresolved_unit_keeps_its_own_characters_so_two_different_ones_never_agree():
    a, b = "mg​", "mg‌"
    assert UN.normalise_unit(a) != UN.normalise_unit(b)
    assert UN.normalise_unit(a) == UN.normalise_unit(a) and UN.normalise_unit(a).endswith(a)


def test_every_character_is_judged_by_the_alphabet_in_a_unit():
    bad = []
    for cp in SAMPLE:
        if 0xD800 <= cp <= 0xDFFF:
            continue
        c, unit = chr(cp), "mg" + chr(cp)
        resolved = S.spec_unit_resolved(c)
        if UN.unit_resolved(unit) != resolved:
            bad.append((hex(cp), "resolved" if resolved else "unresolved"))
        elif not resolved and (UN.same_unit(unit, "mg") or UN.canonical_unit(unit) == "mg"
                               or RC.recognised_unit(unit) is not None):
            bad.append((hex(cp), "a unit with this character equals or is read as mg"))
    assert bad[:10] == []


def test_the_case_of_a_mega_or_milli_prefix_is_judged_on_the_alphabets_letters_only():
    assert UN.prefix_conflict("MIU", "mIU") == "mismatch" and UN.prefix_conflict("Mg", "mg") == "mismatch"
    assert UN.prefix_conflict("mg/mL", "MG/ML") == "ambiguous" and UN.prefix_conflict("mg", "mg") is None
    assert UN.prefix_conflict("MΩ", "mΩ") == "mismatch" and UN.prefix_conflict("mg", "ng") is None
    assert UN.prefix_conflict("mg", "Mg​") is None                              # unresolved: no case judgment


def test_unit_norm_holds_no_category_test_and_no_shorthand_class():
    tree = ast.parse((EVIDENCE / "unit_norm.py").read_text("utf-8"))
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not attrs & {"isalpha", "isupper", "islower", "isspace", "isdigit", "normalize", "category"}
    patterns = [c.args[0].value for c in ast.walk(tree) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                and c.func.attr == "compile" and c.args and isinstance(c.args[0], ast.Constant)]
    assert not any("\\s" in p or "\\b" in p or "\\w" in p for p in patterns)


# ----------------------------------------------------------------------- header keys ---

def test_a_header_key_folds_the_equivalents_the_infinity_sign_and_case_and_drops_the_separators():
    assert norm_key("AUC0–∞") == norm_key("AUC0-inf") == norm_key("AUC 0 - INF") == "auc0inf"
    assert norm_key("µg") == norm_key("μg") == "μg" and norm_key("CL/F (L/h)") == "cl/f(l/h)"
    assert norm_key("Cmax (ng/mL)") == norm_key("Cmax (ng/mL)") and norm_key(None) == ""
    assert norm_key("Cmax ") == "cmax" and norm_key("t_max") == "tmax" and norm_key("t.max, h") == "tmaxh"
    assert norm_key("−Cmax") == norm_key("-Cmax") == "cmax"                     # a minus sign is a hyphen: dropped


@pytest.mark.parametrize("header", ["C​max", "Ｃmax", "С" + "max", "𝐶max", "Cmax ", "C­max", "Cmax‎",
                                    "Cmаx", "ﬁ"])
def test_a_header_with_a_character_the_key_cannot_fold_matches_no_field(header):
    assert norm_key(header) != "cmax"


def test_a_header_key_keeps_every_character_it_cannot_fold():
    bad = []
    for cp in SAMPLE:
        if 0xD800 <= cp <= 0xDFFF:
            continue
        c = chr(cp)
        key = norm_key("C" + c + "max")
        dropped = c in S.KEY_STRIP_PIN
        if (key == "cmax") != dropped:
            bad.append((hex(cp), key, "dropped" if dropped else "kept"))
    assert bad[:10] == []


def test_the_key_table_is_pinned():
    from app.evidence.numgrammar import equivalents as EQ
    assert {ord(k): v for k, v in EQ.HEADER_SPELLINGS.items()} == S.HEADER_SPELLING_PIN


# --------------------------------------------------------------------- the ink pairing ---

class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _page():
    return _Obj(page_obj=_Obj(mediabox=(0, 0, 612, 792), rotate=0))


def _pair(text_layer: str, glyph: str):
    """One pdfplumber char at (100, 200) whose pdfium glyph (visible, same origin) decodes as *glyph*."""
    chars = [{"text": text_layer, "matrix": (1, 0, 0, 1, 100, 200)}]
    return PIM._transfer(_page(), chars, [glyph], np.array([[100.0, 200.0]]), np.array([True]), np.array([False]))


def test_a_glyph_is_paired_with_its_character_by_the_equivalents_table_and_nothing_else():
    hidden, flags, _, _ = _pair("−", "-")                   # an approved equivalent: the same glyph
    assert hidden == set() and flags == {}
    hidden, flags, _, _ = _pair("2", "2")
    assert hidden == set() and flags == {}


@pytest.mark.parametrize("text_layer, glyph", [("²", "2"), ("2", "²"), ("２", "2"), ("2", "２"),
                                               ("₂", "2"), ("¹", "1"), ("⁰", "0")])
def test_a_superscript_or_fullwidth_digit_is_never_the_digit_nfkc_made_of_it(text_layer, glyph):
    """NFKC paired '²' with '2', so a text layer that says '²' over a glyph that draws '2' (or the reverse) went
    unflagged. Now the two texts differ and the number characters in doubt are flagged."""
    hidden, flags, _, _ = _pair(text_layer, glyph)
    assert hidden == set()
    assert set(flags) == {PIM.MISMATCH} and set(text_layer + glyph) & flags[PIM.MISMATCH]


def test_a_ligature_glyph_is_a_visible_letter_pair_and_flags_nothing():
    for text_layer, glyph in (("ﬁ", "fi"), ("fi", "ﬁ")):
        hidden, flags, _, _ = _pair(text_layer, glyph)
        assert hidden == set() and flags == {}, (text_layer, glyph)


def test_the_pairing_reads_texts_through_fold_only():
    src = (EVIDENCE / "parse_ink_match.py").read_text("utf-8")
    assert "unicodedata" not in src and "_norm" not in src
    assert fold("−") == "-" and "−" in EQUIV


# ------------------------------------------------------------------ the unit token reader ---

def test_the_unit_token_reader_holds_only_the_alphabets_percent_signs():
    assert RC._UNIT_TOKEN_RE.match("‱") is None and RC._UNIT_TOKEN_RE.match(" ‱") is None
    for sign in A.PERCENTS:
        assert RC._UNIT_TOKEN_RE.match(sign).group(1) == sign and sign in RC._BASES
    assert "‱" not in RC._BASES and "‱" not in RC._UNIT_TOKEN_RE.pattern
    assert A.cls("‱") == "outside"
    assert RC.recognised_unit("‱") is None and RC.recognised_unit("‰") == "‰"
    assert RC.recognised_unit("%") == "%"
