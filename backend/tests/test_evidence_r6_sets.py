"""r6 numeric grammar, layer A: the literal sets, the equivalence table and ``cls`` for EVERY code point (r7: four
classes; the LINE list is deleted and the line rule is derived from the four permitted lists).

The sets are pinned twice (alphabet.py and tests/evidence_r6_pins.py, typed independently) and printed, so any
addition shows up as a diff in both places. Nothing in the reader may use a Unicode category: an AST test pins that
the package imports no ``unicodedata``, uses no ``\\d \\w \\s \\b`` class and no ``str.is*`` test, and that
``float()`` is only ever called on a canonical ASCII string the reader built.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from app.evidence import numgrammar as NG
from app.evidence.numgrammar import alphabet as A
from app.evidence.numgrammar import equivalents as EQ
from tests import evidence_r6_pins as P

PACKAGE = Path(NG.__file__).parent
ALL_CPS = range(0x110000)


def _cps(chars) -> tuple[int, ...]:
    return tuple(sorted(ord(c) for c in chars))


def _fmt(cps) -> str:
    return " ".join(f"{c:04X}" for c in sorted(cps))


# ---------------------------------------------------------------- the sets ---

def test_the_sets_are_pinned_entry_by_entry_against_the_independent_copy():
    assert _cps(A.INSIDE) == tuple(sorted(P.INSIDE_SET))
    assert _cps(A.GAP) == tuple(sorted(P.GAP_SET))
    assert _cps(A.CONTEXT) == tuple(sorted(P.CONTEXT_SET))
    assert {ord(k): ord(v) for k, v in EQ.EQUIV.items()} == P.EQUIV_PIN


def test_the_class_sizes_are_pinned():
    assert (len(A.INSIDE), len(A.GAP), len(A.CONTEXT), len(EQ.EQUIV)) == (
        P.COUNTS["inside"], P.COUNTS["gap"], P.COUNTS["context"], P.COUNTS["equiv"]) == (45, 4, 115, 10)
    assert len(P.INSIDE_CPS) == len(P.INSIDE_SET) and len(P.CONTEXT_CPS) == len(P.CONTEXT_SET)
    assert len(P.GAP_CPS) == len(P.GAP_SET) and set(P.COUNTS) == {"inside", "gap", "context", "equiv"}


def test_a_character_is_in_exactly_one_class_and_an_equivalent_source_is_in_none():
    classes = [A.INSIDE, A.GAP, A.CONTEXT]
    for i, a in enumerate(classes):
        for b in classes[i + 1:]:
            assert not (a & b), sorted(a & b)
    for src, dst in EQ.EQUIV.items():
        assert not any(src in c for c in classes), f"{src!r} is an equivalent source and also listed"
        assert any(dst in c for c in classes), f"target {dst!r} of {src!r} must be an allowed character"
    assert all(len(c) == 1 for c in A.INSIDE | A.GAP | A.CONTEXT | set(EQ.EQUIV) | set(EQ.EQUIV.values()))


def test_the_context_set_holds_the_48_ascii_letters_that_are_not_inside():
    assert len(A.CONTEXT_LETTERS) == 48 and not (A.CONTEXT_LETTERS & set("eExX"))
    assert A.ASCII_LETTERS == frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    assert len(A.GREEK) == 17 and "\u03ba" in A.GREEK and "\u03c7" not in A.GREEK and A.cls("\u03c7") == "outside"
    assert len(A.ACCENTED) == 29 and len(A.CONTEXT_MARKS) == 12 and A.LETTERS == A.ASCII_LETTERS | A.GREEK | A.ACCENTED


def test_print_the_exact_contents_of_every_class(capsys):
    """The visible record of the alphabet: ``pytest -s tests/test_evidence_r6_sets.py`` prints it."""
    table = {"inside": A.INSIDE, "gap": A.GAP, "context": A.CONTEXT}
    with capsys.disabled():
        for name, chars in table.items():
            print(f"\n{name} ({len(chars)}): {_fmt(_cps(chars))}")
        print("\nequivalents (10): " + " ".join(f"{ord(k):04X}->{ord(v):04X}" for k, v in EQ.EQUIV.items()))
    assert capsys.readouterr().out == ""


def test_the_equivalence_table_is_one_to_one_and_total():
    assert all(len(k) == 1 and len(v) == 1 for k, v in EQ.EQUIV.items())
    text = "".join(EQ.EQUIV) + "a1"
    assert len(EQ.fold(text)) == len(text)
    assert EQ.fold("−–\u2010\u2011") == "----"
    assert EQ.fold("24’60") == "24'60" and EQ.fold("µ") == "μ"
    assert EQ.fold("∼") == "~" and EQ.fold("◦") == "°"


def test_the_thin_space_sources_are_the_ones_the_table_maps_to_a_thin_space():
    from app.evidence import parse_guard as G
    assert EQ.THIN_SOURCES == frozenset({"\u2009", "\u202f", "\u00a0"}) == G.THIN_SPACES
    assert EQ.THIN_SOURCES == {EQ.THIN} | {k for k, v in EQ.EQUIV.items() if v == EQ.THIN}
    for other in ("\u2007", "\u2008", "\u200a", "\u2003", "\u3000"):
        assert other not in EQ.THIN_SOURCES and A.cls(other) == "outside"


def test_a_thin_space_groups_digits_only_between_two_digits():
    t = "\u2009"
    assert EQ.fold(f"1{t}2") == f"1{t}2"
    assert EQ.fold(f"a{t}2 1{t}") == "a 2 1 " and EQ.fold(f"{t}1") == " 1"
    assert EQ.fold("1\u202f234") == f"1{t}234" and EQ.fold("1\u00a0234") == f"1{t}234"
    assert EQ.fold("x\u202fy") == "x y"


def test_claim_mode_reads_a_thousands_position_ascii_space_as_a_thin_space():
    t = "\u2009"
    assert EQ.fold("18 900", claim=True) == f"18{t}900"
    assert EQ.fold("1 234 567", claim=True) == f"1{t}234{t}567"
    for stays in ("12 1500", "0.123 4", "1234 567", "12  145", "a 123"):
        assert EQ.fold(stays, claim=True) == stays, stays
    assert EQ.fold("18 900", claim=False) == "18 900"


# ------------------------------------------------------------ layer A: all code points ---

def test_cls_agrees_with_the_independent_copy_on_every_code_point():
    wrong = []
    counts = {"inside": 0, "gap": 0, "context": 0, "outside": 0}
    for cp in ALL_CPS:
        want = P.expected_class(cp)
        counts[want] += 1
        if A.cls(chr(cp)) != want:
            wrong.append((f"{cp:04X}", A.cls(chr(cp)), want))
    assert wrong == []
    # an equivalent classifies as its target: 8 of the 10 become inside (- x4, ' x1, thin space x2, ~ x1), 2 context
    assert counts["inside"] == 45 + 8 and counts["context"] == 115 + 2 and counts["gap"] == 4
    assert sum(counts.values()) == 0x110000


def test_every_code_point_not_listed_is_outside():
    listed = set(P.INSIDE_SET | P.GAP_SET | P.CONTEXT_SET | set(P.EQUIV_PIN))
    sample = [cp for cp in ALL_CPS if cp not in listed]
    assert len(sample) == 0x110000 - len(listed)
    assert all(A.cls(chr(cp)) == "outside" for cp in sample)


def test_the_allowlist_regexes_agree_with_cls_on_every_code_point():
    text = "".join(map(chr, ALL_CPS))
    # one allow-list for every tier (word, window, line): the derived rule of r7
    outside = {m.start() for m in A.OUTSIDE_RE.finditer(text)}
    assert outside == {cp for cp in ALL_CPS if P.expected_class(cp) == "outside"}
    inside_only = {m.start() for m in re.finditer(A.char_class(A.INSIDE, negate=False), text)}
    assert inside_only == {cp for cp in ALL_CPS if chr(cp) in A.INSIDE}


@pytest.mark.parametrize("cp", [0x00, 0x0B, 0x0C, 0x1C, 0x7F, 0x85, 0xA0 + 0x100, 0xAD, 0x200B, 0x200D, 0x2028, 0x2029,
                                0x202E, 0x2060, 0xFE0F, 0xFEFF, 0xFFFC, 0xFFFD, 0xE000, 0xF0000, 0x10FFFF, 0x2800,
                                0x3000, 0x0300, 0x20D0, 0x1D7CE, 0xFF11, 0x0661, 0x2460, 0x00BE, 0xFF0E, 0xD800])
def test_the_characters_the_old_reader_special_cased_are_all_outside(cp):
    assert A.cls(chr(cp)) == "outside"


def test_cls_refuses_anything_but_one_character():
    for bad in ("", "ab"):
        with pytest.raises(ValueError):
            A.cls(bad)


def test_the_cid_marker_is_a_string_rule():
    assert A.CID_MARKER.fullmatch("(cid:3)") and A.CID_MARKER.fullmatch("(cid:2879)")
    for s in ("(cid:)", "(cid:x)", "cid:3", "(cid: 3)", "(Cid:3)"):
        assert A.CID_MARKER.fullmatch(s) is None


# ----------------------------------------------------------- implementation constraints ---

def _modules() -> list[tuple[Path, ast.Module]]:
    return [(p, ast.parse(p.read_text("utf-8"))) for p in sorted(PACKAGE.glob("*.py"))]


def test_the_package_imports_no_unicodedata_and_uses_no_category_test():
    banned_attrs = {"isdigit", "isalpha", "isspace", "isalnum", "isnumeric", "isdecimal", "isidentifier",
                    "isprintable", "category", "normalize", "isupper", "islower", "casefold", "lower", "upper"}
    for path, tree in _modules():
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(a.name != "unicodedata" for a in node.names), path
            if isinstance(node, ast.ImportFrom):
                assert node.module != "unicodedata", path
            if isinstance(node, ast.Attribute) and node.attr in banned_attrs:
                # the reviewer-facing label of a scale word may casefold ('Thousands'); nothing else may
                assert node.attr == "casefold" and path.name == "api.py", (path.name, node.lineno, node.attr)


def _docstring_nodes(tree: ast.Module) -> set[int]:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                out.add(id(first.value))
    return out


def test_no_regex_uses_a_shorthand_class():
    pattern = re.compile(r"\\[dDwWsSbB]")
    for path, tree in _modules():
        docs = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
                assert pattern.search(node.value) is None, (path.name, node.lineno, node.value[:40])


def test_float_is_called_only_on_a_canonical_string_and_int_only_on_digits_the_reader_matched():
    for path, tree in _modules():
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            for call in [c for c in ast.walk(fn) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)]:
                if call.func.id == "float":
                    assert (path.name, fn.name) == ("body.py", "to_float"), (path.name, fn.name, call.lineno)
                if call.func.id == "int":
                    assert (path.name, fn.name) == ("body.py", "exp_int"), (path.name, fn.name, call.lineno)


def test_every_file_stays_under_the_line_limit():
    for path in PACKAGE.glob("*.py"):
        assert len(path.read_text("utf-8").splitlines()) <= 400, path.name


def test_every_regex_class_is_built_from_the_literal_sets_or_spells_ascii_digits():
    """Regexes in the package hold only explicit classes: [0-9], literal characters, or classes built from a set."""
    for path, tree in _modules():
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "compile"
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "re" and node.args
                    and isinstance(node.args[0], ast.Constant)):
                src = node.args[0].value
                assert "\\p{" not in src and "[[:" not in src and "(?i" not in src, (path.name, src)
