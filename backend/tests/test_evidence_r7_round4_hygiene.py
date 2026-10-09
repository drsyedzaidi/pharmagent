"""r7 round 4: hygiene findings of the round-4 audit (5, 26, 27, 28, 29), each a regression.

* EVERY module of app/evidence is guarded: no hand-typed character set or one-character terminal and no terminal typed
  into a regex fragment, except the named exemptions (``tests/evidence_r7_hygiene_scan.py``: the list modules, template
  syntax, prose patterns), each with its reason; an exemption that no longer matches fails;
* the unit, citation and chi-square regexes are built from the role sets, and 'per' is defined once (alphabet.PER);
* the decimal style of a field and the unit token of a record read the role sets (a times sign joins a unit like a
  middle dot: 'ng×h/mL');
* the reader package reads no word list and no letter table: the scale label check, the loose word match and the dose
  rule's candidates live outside it.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.evidence.numgrammar import alphabet as A
from tests.evidence_r6_round1_support import run
from tests.evidence_r7_hygiene_scan import LIST_MODULES, NAMED_LITERALS, NAMED_REGEX, hits, regex_terminals

EVIDENCE = Path(__file__).resolve().parents[1] / "app" / "evidence"
MODULES = sorted(EVIDENCE.rglob("*.py"))


def _rel(path: Path) -> str:
    return str(path.relative_to(EVIDENCE))


@pytest.mark.parametrize("path", MODULES, ids=_rel)
def test_every_module_types_no_character_set_or_terminal_but_its_named_exemptions(path):
    rel = _rel(path)
    if rel in LIST_MODULES:
        return
    assert [h for h in hits(path) if (rel, h[2]) not in NAMED_LITERALS] == [], rel


@pytest.mark.parametrize("path", MODULES, ids=_rel)
def test_every_regex_fragment_types_no_terminal_but_its_named_exemptions(path):
    rel = _rel(path)
    assert [r for r in regex_terminals(path) if (rel, r[1]) not in NAMED_REGEX] == [], rel


def test_every_named_exemption_still_matches_a_use():
    used_lit = {(_rel(p), h[2]) for p in MODULES for h in hits(p)}
    used_rx = {(_rel(p), r[1]) for p in MODULES for r in regex_terminals(p)}
    assert sorted(set(NAMED_LITERALS) - used_lit) == [] and sorted(set(NAMED_REGEX) - used_rx) == []
    assert all((EVIDENCE / m).exists() for m in LIST_MODULES)


def test_the_regex_guard_catches_a_typed_terminal(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("import re\nfrom x import char_class\nA = char_class(S) + '(?:/|per)'\nB = re.compile('\\\\[' + A)\n"
                   "C = re.compile('[23](?![0-9])|' + A)\nD = re.compile('[0-9]+(?:' + A + ')')\n")
    assert regex_terminals(bad) == [(3, "/per"), (4, "\\["), (5, "[]")]


def test_the_strip_guard_reads_the_characters_a_call_strips(tmp_path):
    ok = tmp_path / "ok.py"
    ok.write_text("from app.evidence import gaps\ndef f(rec, w, s):\n    return gaps.strip(rec['value']) + "
                  "w.rstrip(''.join(s)) + gaps.strip(w + ' ' + s)\n")
    assert hits(ok) == []
    bad = tmp_path / "bad.py"
    bad.write_text("def f(w):\n    return w.rstrip('.,;:') + w.lstrip('(' + ')')\n")
    assert [k for _, k, _ in hits(bad)] == ["rstrip", "lstrip", "lstrip"]


def test_the_grammar_terminals_are_defined_once_in_the_alphabet():
    assert (A.PER, A.DF, A.SPACE, A.DASH) == ("per", "df", " ", "-")
    assert (A.ROUND_OPEN, A.ROUND_CLOSE, A.SQUARE_CLOSE) == ("(", ")", "]") and A.PLAIN_POWERS == frozenset("23")
    for role in ("ROUND_OPEN", "ROUND_CLOSE", "SQUARE_CLOSE", "SPACE", "DASH", "PLAIN_POWERS"):
        assert frozenset(getattr(A, role)) <= A.INSIDE | A.CONTEXT | A.GAP, role
    literal_per = [str(p.relative_to(EVIDENCE)) for p in MODULES for n in ast.walk(ast.parse(p.read_text("utf-8")))
                   if isinstance(n, ast.Constant) and n.value == "per"]
    assert literal_per == ["numgrammar/alphabet.py"], literal_per


def test_the_decimal_style_of_a_field_reads_the_decimal_marks():
    from app.evidence.extract_rules import decimal_style, value_number
    assert decimal_style({"decimal": ","}, {"decimal_style": "."}) == ","
    assert decimal_style({"decimal": ";"}, {"decimal_style": ","}) == ","
    assert decimal_style(None, {}) == "." and value_number("3,5", ",") == 3.5 and value_number("3.5", ",") == 3.5


def test_a_times_sign_joins_a_printed_unit_like_a_middle_dot():
    from app.evidence.record_checks import recognised_unit
    for unit in ("ng·h/mL", "ng×h/mL"):
        r = run(f"AUC was 12.5 {unit} overall", f"AUC was 12.5 {unit}", "12.5", field="AUC", unit=unit)
        assert r.status == "pass" and "unit_unverified" not in r.flags and "unit_mismatch" not in r.flags, (unit, r)
    assert recognised_unit("mg/kg.") == "mg/kg" and recognised_unit("h-1") == "h-1"


def test_the_reader_package_reads_no_word_list_and_no_letter_table():
    for path in sorted((EVIDENCE / "numgrammar").glob("*.py")):
        tree = ast.parse(path.read_text("utf-8"))
        imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        assert not imported & {"app.evidence.word_lists", "app.evidence.letter_tables"}, path.name
    from app.evidence import dose_rule, scale_label, word_match
    assert dose_rule.find_dose("a dose of 150 mg daily") == [(10, 13, 150.0, "mg")]
    assert scale_label.scale_shapes("AUC (thousands)") and word_match.loose_in("bel0w", frozenset({"below"}))


# (module) -> why a hand-typed code-point range there classifies no character of a number (r7 round 4, finding 29)
NAMED_RANGES = {"parse_guard.py": "the private-use placeholder pool of the thin-space protection: the first code point no "
                                  "glyph of the page already is, never a classifier"}


def test_a_hand_typed_code_point_range_is_a_named_exemption():
    found = {_rel(p) for p in MODULES for n in ast.walk(ast.parse(p.read_text("utf-8")))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "range"
             and any(isinstance(a, ast.Constant) and isinstance(a.value, int) and a.value >= 0x80 for a in n.args)}
    assert found == set(NAMED_RANGES), found
