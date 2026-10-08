"""r7 round 2: every character class of the reader comes from the four permitted lists (findings of the round-2 audit).

* no ``str.isascii`` anywhere in app/evidence: the scale reading strips only what the reader cannot read (OUTSIDE_RE);
* every role set of the reader is defined once in alphabet.py, as a subset of INSIDE, CONTEXT and GAP; no other module of
  the reader package types a character set (frozenset / set / ``in "..."`` / a tuple of characters);
* the flag word lists (scale words) live with the value checks, not in the reader;
* the tables around the reader (ligatures, header spellings, case pairs) live outside the reader package;
* no-argument ``split`` / ``strip`` (Unicode whitespace: an OUTSIDE space vanishes) in the modules whose text reaches the
  reader, a unit or a glyph check: GAP-based helpers instead (a header unit with an OUTSIDE space stays unresolved);
* a fail-closed glyph verdict marks every number character, not only the digits;
* the screening tokens are built from the alphabet: a non-ASCII digit is never a letter.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.evidence import gaps
from app.evidence.extract_rules import split_unit
from app.evidence.numgrammar import alphabet as A
from app.evidence.numgrammar import scale_shapes
from app.evidence.screen_rules import stem_phrase
from app.evidence.unit_norm import unit_resolved

EVIDENCE = Path(__file__).resolve().parents[1] / "app" / "evidence"
READER = EVIDENCE / "numgrammar"
LIST_FILES = {"alphabet.py", "equivalents.py"}
# modules whose text reaches tokens(), a unit or a glyph check (no Unicode-whitespace split / strip there)
TEXT_PATHS = ("parse_tables.py", "parse_guard.py", "parse_fonts.py", "parse_ink.py", "parse_marks.py", "parse_ink_match.py",
              "extract_rules.py", "record_checks.py", "value_checks.py", "unit_norm.py", "verify.py", "normalize.py")


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text("utf-8"))


# ------------------------------------------------------------------------------------------ isascii ---
def test_no_module_of_the_evidence_pipeline_calls_isascii():
    found = [(p.name, n.lineno) for p in EVIDENCE.rglob("*.py") for n in ast.walk(_tree(p))
             if isinstance(n, ast.Attribute) and n.attr == "isascii"]
    assert found == []


def test_the_scale_reading_strips_only_what_the_reader_cannot_read():
    assert scale_shapes("AUC ×10³ (ng·h/mL)") and scale_shapes("x1000") and scale_shapes("(thousands)")
    assert scale_shapes("x1​000") and scale_shapes("× 1,000") and scale_shapes("x10³")
    assert not scale_shapes("Cmax (μg/L)") and not scale_shapes("AUC0-∞ (ng·h/mL)")
    assert not scale_shapes("Temperature (°C)") and not scale_shapes("BSA (m²)")


# ------------------------------------------------------------------------------------- role sets ---
ROLE_SETS = ("SIGNS", "SEPARATORS", "EXP_MARKERS", "MULTIPLIERS", "SUP_DIGITS", "SUP_SIGNS", "PERCENTS", "QUALIFIERS",
             "LETTERS", "ASCII_LETTERS", "OPENERS_BR", "CLOSERS_BR", "STARTERS", "HYPHEN", "OPERATORS", "RANGE_WORDS",
             "NUMBER_CLOSERS", "JOINERS", "TRAILING", "SUP_CHARS", "SIGN_CHARS", "DECIMAL_MARKS", "GROUP_ONLY",
             "RATIO_MARKS", "EQUALS", "MULT_LETTERS", "LIST_MARKS")


@pytest.mark.parametrize("name", ROLE_SETS)
def test_every_role_set_of_the_reader_is_a_subset_of_the_permitted_lists(name):
    role = getattr(A, name)
    assert role and frozenset(role) <= A.INSIDE | A.CONTEXT | A.GAP, (name, sorted(set(role) - A.INSIDE - A.CONTEXT))


def test_the_derived_role_sets_follow_from_the_lists():
    assert A.ASCII_LETTERS == A.CONTEXT_LETTERS | A.EXP_MARKERS | A.MULT_LETTERS and len(A.ASCII_LETTERS) == 52
    assert A.GROUP_ONLY == A.SEPARATORS - A.DECIMAL_MARKS - {"·"} and A.DECIMAL_MARKS == frozenset(".,")
    assert A.RATIO_MARKS == frozenset("/:") and A.EQUALS == frozenset("=")
    assert ":" not in A.STARTERS          # round 2: a colon never starts a value (reference pages, DOIs, times: measured)


def _char_set_literals(path: Path) -> list[tuple[int, str]]:
    """frozenset/set/maketrans over a string constant, ``x in "chars"`` and ``x in ("a", "b")`` (character sets)."""
    out = []
    for n in ast.walk(_tree(path)):
        if isinstance(n, ast.Call):
            fn = n.func
            name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else "")
            chars = [c for a in n.args for c in ([a] if isinstance(a, ast.Constant) else getattr(a, "elts", []))
                     if isinstance(c, ast.Constant) and isinstance(c.value, str)
                     and (c is a or len(c.value) == 1)] if name in ("frozenset", "set") else []
            if chars or (name == "maketrans" and any(isinstance(c, ast.Constant) and isinstance(c.value, str)
                                                     for a in n.args for c in ast.walk(a))):
                out.append((n.lineno, name))
            if isinstance(fn, ast.Attribute) and fn.attr in ("strip", "rstrip", "lstrip") and n.args and isinstance(
                    n.args[0], ast.Constant):
                out.append((n.lineno, fn.attr))
        if isinstance(n, ast.Compare) and any(isinstance(op, (ast.In, ast.NotIn)) for op in n.ops):
            for comp in n.comparators:
                if isinstance(comp, ast.Constant) and isinstance(comp.value, str) and len(comp.value) > 1:
                    out.append((n.lineno, "in-literal"))
                if isinstance(comp, (ast.Tuple, ast.Set, ast.List)) and any(
                        isinstance(e, ast.Constant) and isinstance(e.value, str) and len(e.value) == 1 for e in comp.elts):
                    out.append((n.lineno, "in-tuple"))
    return out


@pytest.mark.parametrize("path", sorted(p for p in READER.glob("*.py") if p.name not in LIST_FILES), ids=lambda p: p.name)
def test_no_module_of_the_reader_package_types_a_character_set_of_its_own(path):
    assert _char_set_literals(path) == [], path.name


def test_the_guard_catches_a_hand_typed_set(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("X = frozenset('.,')\ndef f(c):\n    return c in '/:' or c in ('n', 'N') or c.rstrip('.,')\n")
    assert len(_char_set_literals(bad)) == 4


def test_the_reader_package_holds_no_flag_word_list_and_no_table_around_the_reader():
    from app.evidence.numgrammar import api, equivalents
    assert not hasattr(api, "_SCALE_WORDS") and not hasattr(api, "MULT_CHARS")
    assert not any(hasattr(equivalents, n) for n in ("LIGATURES", "HEADER_SPELLINGS", "CASE_PAIRS", "casefold_letters"))
    from app.evidence import letter_tables as LT
    assert A.LETTERS >= frozenset(LT.CASE_PAIRS) and len(LT.CASE_PAIRS) == 34 and LT.UPPER_LETTERS == frozenset(LT.CASE_PAIRS)


# ------------------------------------------------------------------------- Unicode-whitespace helpers ---
@pytest.mark.parametrize("name", TEXT_PATHS)
def test_no_unicode_whitespace_split_or_strip_where_text_reaches_the_reader(name):
    calls = [n.lineno for n in ast.walk(_tree(EVIDENCE / name)) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr in ("split", "strip", "rstrip", "lstrip")
             and not n.args and not n.keywords]
    assert calls == [], (name, calls)


def test_the_gap_helpers_split_and_strip_only_the_gap_set():
    assert gaps.collapse(" a \t b\n") == "a b" and gaps.strip(" 　a　 ") == "　a　"
    assert gaps.split("a b c") == ["a b", "c"] and gaps.blank(" \t") and not gaps.blank("　")
    assert gaps.blank(" ") and gaps.blank(" ") and gaps.blank("") and not gaps.blank("​")


def test_a_header_unit_with_an_outside_space_stays_unresolved():
    label, unit = split_unit("Dose (mg /kg)")
    assert unit is not None and not unit_resolved(unit), (label, unit)
    assert split_unit("Dose  (mg /kg)") == ("Dose", "mg /kg") and split_unit("Cmax (ng/mL)") == ("Cmax", "ng/mL")


def test_a_table_cell_keeps_an_outside_space_and_its_thin_space_group():
    from app.evidence.parse_tables import _clean_cell
    assert _clean_cell(" 12　.5 ") == "12　.5" and _clean_cell("18 900") == "18 900"
    assert _clean_cell("a \n b") == "a b"


# --------------------------------------------------------------------------- glyph verdict, screening ---
def test_a_fail_closed_glyph_verdict_marks_every_number_character():
    for name in ("parse.py", "parse_fonts.py"):
        src = (EVIDENCE / name).read_text("utf-8")
        assert "0123456789" not in src and "NUMBER_CHARS" in src, name


def test_a_non_ascii_digit_is_never_a_screening_letter():
    assert stem_phrase("in vitro") == ("in", "vitro")
    assert stem_phrase("CYP2D6 ٣٤") == ("cyp", "2", "d", "6")             # Arabic-indic digits are no token
    assert stem_phrase("Rats") == ("rat",)
