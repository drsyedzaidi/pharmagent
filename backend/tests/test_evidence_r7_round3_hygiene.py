"""r7 round 3: hygiene findings of the round-3 audit (each PoC is a regression).

* the model's value and unit keep an OUTSIDE character at their ends (no Unicode-whitespace strip in extract_llm);
* unit_norm, value_checks and verify_messages take their character sets from alphabet.py (no hand-typed set);
* no one-character terminal is typed as a literal in the reader package (a Compare against a one-character string,
  ``startswith`` / ``endswith`` of one): the terminals come from alphabet.py, the chi-square terminals too;
* every role set of the reader is pinned as a subset of the permitted lists (the single-character ones included);
* the dead shape detector ``words.exp_like`` is gone;
* layout, column and QC modules classify whitespace with the GAP set only (no isspace / no-argument strip);
* a digit class in a regex outside the grammar module is an explicit, named exemption (the caption patterns use none).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from app.evidence import gaps
from app.evidence import models as M
from app.evidence.numgrammar import alphabet as A
from app.evidence.numgrammar import unreadable
from app.evidence.unit_norm import unit_resolved
from tests.evidence_r6_round1_support import doc
from tests.evidence_r7_hygiene_scan import NAMED_LITERALS, hits

EVIDENCE = Path(__file__).resolve().parents[1] / "app" / "evidence"
READER = EVIDENCE / "numgrammar"
LIST_FILES = {"alphabet.py", "equivalents.py"}
GUARDED = sorted(p for p in READER.glob("*.py") if p.name not in LIST_FILES) + [
    EVIDENCE / n for n in ("unit_norm.py", "value_checks.py", "verify_messages.py", "value_bounds.py",
                           "value_scale.py", "value_context.py", "verify_table.py", "verify_columns.py",
                           # r7 round 4: the extraction rules and the record checks read the decimal style and units,
                           # and the modules that left the reader package keep its guard
                           "extract_rules.py", "record_checks.py", "scale_label.py", "word_match.py", "dose_rule.py",
                           "value_ranges.py")]
TEXT_PATHS = ("extract_llm.py", "parse_layout.py", "verify_columns.py", "qc_flags.py", "value_bounds.py",
              "value_scale.py", "value_context.py", "verify_table.py")


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text("utf-8"))


# ------------------------------------------------------------------------------------ extract_llm ---
def test_the_model_value_and_unit_keep_an_outside_character_at_their_ends():
    from app.evidence.extract_llm import _to_proposal
    page = doc("Cmax 12.5 ng/mL")
    rec = {"field": "Cmax", "page": 1, "snippet": "Cmax 12.5", "kind": "text", "value": "12.5　", "unit": "mg　"}
    p = _to_proposal(rec, {"cmax": {"name": "Cmax", "type": "number"}}, page, "S1")
    assert p["value"] == "12.5　" and p["unit"] == "mg　", p
    assert unreadable(p["value"]) == (4, "U+3000") and not unit_resolved(p["unit"])
    from app.evidence.verify import verify
    res = verify(p, page)
    assert (res.status, res.reason) == ("fail", M.REASON_UNRESOLVED_CONTEXT), res
    rec_gap = {**rec, "value": " 12.5\n", "unit": " mg "}
    p2 = _to_proposal(rec_gap, {"cmax": {"name": "Cmax", "type": "number"}}, page, "S1")
    assert (p2["value"], p2["unit"]) == ("12.5", "mg")          # the GAP set is still stripped


@pytest.mark.parametrize("name", TEXT_PATHS)
def test_no_unicode_whitespace_classification_in_layout_column_qc_and_llm_modules(name):
    calls = [n.lineno for n in ast.walk(_tree(EVIDENCE / name)) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and (n.func.attr == "isspace" or (
                 n.func.attr in ("split", "strip", "rstrip", "lstrip") and not n.args and not n.keywords))]
    assert calls == [], (name, calls)


def test_layout_word_mapping_skips_only_the_gap_set():
    from app.evidence.parse_layout import map_words, word_box
    words = [{"text": "12.5"}, {"text": "mg"}]
    assert map_words("12.5 mg", words) == [(0, 4), (5, 7)]
    assert map_words("12.5　mg", words) is None                     # an OUTSIDE space is not spacing
    chars = [{"text": "1"}, {"text": " "}]
    assert word_box({"chars": chars}, {id(chars[0]): (0.0, 0.0, 1.0, 1.0)}) is None
    plain = [{"text": "1"}, {"text": " "}]
    assert word_box({"chars": plain}, {id(plain[0]): (0.0, 0.0, 1.0, 1.0)}) == (0.0, 0.0, 1.0, 1.0)


def test_a_column_line_of_outside_spaces_is_kept_and_same_value_strips_only_gaps():
    from app.evidence.qc_flags import same_value
    from app.evidence.verify_columns import ColumnLines
    lines = ColumnLines(["a\n　\nb", "c"])
    assert any("　" in x for x in lines.left)
    assert same_value({"value": " 12 "}, {"value": "12"}) and not same_value({"value": "12　"}, {"value": "12"})


# --------------------------------------------------------------------------- hand-typed character sets ---
def _hand_typed(path: Path) -> list[tuple[int, str]]:
    """frozenset/set over characters, ``in "xy"``, one-character tuples, a Compare against a ONE-character string and
    ``startswith`` / ``endswith`` of one (round 3: single terminals come from alphabet.py too). r7 round 4: the scanner
    lives in ``tests/evidence_r7_hygiene_scan.py`` and judges a strip call by the characters it strips (``gaps.strip``
    and a dict key inside the argument are no character set); a module may still name an exemption there."""
    return [(line, kind) for line, kind, _ in hits(path)]


@pytest.mark.parametrize("path", GUARDED, ids=lambda p: p.name)
def test_no_hand_typed_character_or_single_terminal_in_the_reader_and_its_flag_and_unit_modules(path):
    # r7 round 4: a literal the round-4 hygiene names (NAMED_LITERALS, with its reason) is no hit; none of the modules
    # guarded in round 3 holds one, so for them this is the round-3 assertion unchanged
    rel = str(path.relative_to(EVIDENCE))
    assert [h for h in hits(path) if (rel, h[2]) not in NAMED_LITERALS] == [], path.name


def test_the_guard_catches_a_single_character_terminal(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("def f(c, w, s):\n    return c == '^' or '/' in w or w.endswith('-') or c != '0' or c in ('a', 'b') "
                   "or w.rstrip(s + ')') or s - {\"'\"}\n")
    assert len(_hand_typed(bad)) == 7


def test_the_scale_check_reads_the_unit_headers_of_the_extraction_rules():
    src = "".join((EVIDENCE / n).read_text("utf-8") for n in ("value_checks.py", "value_scale.py"))
    assert '("unit", "units")' not in src and "UNIT_HEADERS" in src


SINGLE_ROLES = ("STAR", "DEGREE", "SUP_MINUS", "NAME_LETTER", "CARET", "PLUS_MINUS", "MIDDOT", "POINT", "COMMA",
                "ZERO", "SQUARE_OPEN", "SUP_TWO")
ROLE_SETS = ("SLASH", "PRODUCT_DOTS", "N_LETTERS", "REF_SEPARATORS", "APOSTROPHE", "THIN_GROUP")


@pytest.mark.parametrize("name", SINGLE_ROLES + ROLE_SETS)
def test_every_single_character_role_and_the_remaining_role_sets_are_members_of_the_lists(name):
    role = getattr(A, name)
    assert role and frozenset(role) <= A.INSIDE | A.CONTEXT | A.GAP, (name, role)


def test_the_chi_square_terminals_are_grammar_data_of_the_alphabet_module():
    from app.evidence.numgrammar import productions
    assert A.CHI == "χ" and A.SUP_TWO == "²" and productions.CHI is A.CHI
    assert A.cls(A.CHI) == "outside"                        # chi stays outside: only the form masks it
    src = (READER / "productions.py").read_text("utf-8")
    assert "\\u03c7" not in src and "\\u00b2" not in src and "χ" not in src


def test_unit_norm_derives_its_separator_case_prefix_and_micro_sets():
    from app.evidence import unit_norm as U
    assert U._CASE_PREFIXES == frozenset("Mm") and U._MICRO == "μ"
    assert U._STRIP == A.GAP | A.THIN_GROUP | A.DECIMAL_MARKS - A.LIST_MARKS | (A.MULTIPLIERS - A.MULT_LETTERS)
    assert U.NOT_CONTINUATIONS == frozenset({"x", "fold", "%", "‰"})


def test_the_dead_exponent_detector_is_gone():
    from app.evidence.numgrammar import words
    assert not hasattr(words, "exp_like")


# ------------------------------------------------------------------------------ digit classes outside ---
DIGIT_CLASS_EXEMPT = {
    "ingest.py": "file-name sanitising of an upload (never page text)",
    "qc_flags.py": "the identifier shape of a model flag code (never page text)",
    "parse_budget.py": "hex escapes inside the raw PDF bytes (never page text)",
    # r7 round 4: screen_rules.py builds its keyword tokens from DIGITS (char_class): no exemption any more
    "config_tools.py": "the identifier shape of a template field name (never page text)",
}


def _digit_class_modules() -> set[str]:
    found = set()
    for path in EVIDENCE.glob("*.py"):
        for n in ast.walk(_tree(path)):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) \
                    and n.func.value.id == "re" and any(
                        isinstance(c, ast.Constant) and isinstance(c.value, (str, bytes))
                        and re.search(b"0-9", c.value if isinstance(c.value, bytes) else c.value.encode()) is not None
                        for a in n.args for c in ast.walk(a)):
                found.add(path.name)
    return found


def test_a_digit_class_outside_the_grammar_module_is_a_named_exemption():
    assert _digit_class_modules() == set(DIGIT_CLASS_EXEMPT)


def test_the_caption_patterns_are_built_from_the_lists():
    from app.evidence.parse_tables import CAPTION_RE, STOP_RE
    assert CAPTION_RE.match("Table 2") and CAPTION_RE.match("Table\t12") and STOP_RE.match("Fig. 3")
    assert CAPTION_RE.match("Table 2") is None and STOP_RE.match("Figure　3") is None
    assert CAPTION_RE.match("Table ٢") is None
    assert gaps.strip(" x ") == "x"


def test_the_style_sweep_checks_a_derived_context_set_that_covers_every_former_line_member():
    from tests import evidence_r6_layer_b as B
    from tests import evidence_r6_pins as P
    assert not hasattr(P, "FOLDED_FROM_LINE")
    assert {cp for cp in P.CONTEXT_SET if cp >= 0x80} <= P.GLUE_SWEPT          # the 40 non-ASCII former LINE members
    assert {ord("&"), ord("@")} <= P.GLUE_SWEPT and not {ord("("), ord("["), ord("a")} & P.GLUE_SWEPT
    assert B.sweep_style(sorted(P.GLUE_SWEPT)) == []
