"""r6 scope tests: (1) no category, property or hand-typed look-alike list anywhere in app/evidence except the named
exceptions below; (2) layer B, every code point at the boundaries of the paths around the tokenizer.

(1) The reader's AST guard used to cover six modules; the parse, glyph, label, style and unit modules hid the same kinds
of lists (round-1 audit). It now walks EVERY module of app/evidence. A use is allowed only when it is named here with
its reason, and an entry that no longer matches any use fails the test, so the list cannot rot.
"""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import pytest

from tests import evidence_r6_layer_b as B
from tests import evidence_r6_oracle as O
from tests import evidence_r6_pins as P

EVIDENCE = Path(__file__).resolve().parents[1] / "app" / "evidence"
STR_METHODS = {"isdigit", "isalpha", "isnumeric", "isdecimal", "isalnum", "isspace", "isupper", "islower", "isidentifier",
               "isprintable"}
SHORTHAND = re.compile(r"\\[dDwWsSbB]")
LIST_FILES = {"numgrammar/alphabet.py", "numgrammar/equivalents.py"}          # the allowed alphabet and the one table

# (module, kind) -> why it is allowed: none of them reads a number out of text, and none can make a number readable
ALLOWED = {
    ("extract_rules.py", "unicodedata"): "norm_key: the comparison key of a template header or alias (not a value)",
    ("extract_rules.py", "regex-shorthand"): "header-key stripping, a bracketed unit and the dose-context prose pattern",
    ("extract_rules.py", "nonascii-list"): "the micro sign as the first character of a trailing unit word",
    ("normalize.py", "unicodedata"): "nfkc_char: the snippet locator's NFKC, never into or out of the numeric alphabet",
    ("normalize.py", "str-is"): "the locator's de-hyphenation and whitespace collapse (matching only)",
    ("normalize.py", "regex-shorthand"): "the locator's whitespace-run pattern",
    ("numgrammar/api.py", "nonascii-list"): "the unit WORDS of the keyless dose rule (mg, μg, mg/kg ...)",
    ("parse_ink_match.py", "unicodedata"): "NFKC pairs a ligature glyph with its letters; identity is judged by fold()",
    ("parse_layout.py", "str-is"): "maps word texts onto page-text offsets (whitespace skipping)",
    ("parse_tables.py", "regex-shorthand"): "the 'Table N' caption pattern",
    ("prompts.py", "regex-shorthand"): "parsing of model output",
    ("proposal_check.py", "regex-shorthand"): "the instruction-text detector (prompt injection), prose only",
    ("qc_flags.py", "unicodedata"): "group_key: the comparison form of a group label (format characters dropped)",
    ("qc_flags.py", "regex-shorthand"): "group_key whitespace collapse",
    ("screen_rules.py", "regex-shorthand"): "keyword tokenisation of the screening criteria",
    ("unit_norm.py", "unicodedata"): "the unit vocabulary: compatibility spellings of a unit (㎎, ⁻¹) fold to its symbol",
    ("unit_norm.py", "str-is"): "unit prefix case (MIU vs mIU)",
    ("unit_norm.py", "regex-shorthand"): "unit synonym and separator patterns",
    ("verify_messages.py", "unicodedata"): "names a code point for a human; never classifies one",
}


def _uses(tree: ast.Module, rel: str) -> set[str]:
    kinds: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name == "unicodedata" for a in node.names):
            kinds.add("unicodedata")
        if isinstance(node, ast.ImportFrom) and node.module == "unicodedata":
            kinds.add("unicodedata")
        if isinstance(node, ast.Attribute):
            if node.attr in STR_METHODS:
                kinds.add("str-is")
            if isinstance(node.value, ast.Name) and node.value.id == "unicodedata":
                kinds.add("unicodedata")
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "re" and node.args and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str) and SHORTHAND.search(node.args[0].value)):
            kinds.add("regex-shorthand")
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else "")
            if name in ("frozenset", "set", "maketrans") and rel not in LIST_FILES and any(
                    isinstance(c, ast.Constant) and isinstance(c.value, str) and not c.value.isascii()
                    for a in node.args for c in ast.walk(a)):
                kinds.add("nonascii-list")
    return kinds


def _all_uses() -> dict[str, set[str]]:
    out = {}
    for path in sorted(EVIDENCE.rglob("*.py")):
        rel = str(path.relative_to(EVIDENCE))
        kinds = _uses(ast.parse(path.read_text("utf-8")), rel)
        if kinds:
            out[rel] = kinds
    return out


def test_no_module_of_the_evidence_pipeline_holds_a_category_property_or_look_alike_list_that_is_not_named():
    uses = _all_uses()
    unnamed = {(rel, kind) for rel, kinds in uses.items() for kind in kinds if (rel, kind) not in ALLOWED}
    assert unnamed == set(), f"name them in ALLOWED with a reason, or remove them: {sorted(unnamed)}"


def test_every_named_exception_still_matches_a_use():
    uses = _all_uses()
    stale = sorted((rel, kind) for (rel, kind) in ALLOWED if kind not in uses.get(rel, set()))
    assert stale == [], f"remove the exceptions that no longer apply: {stale}"


def test_the_reader_package_has_no_unicodedata_and_the_only_hand_typed_lists_are_in_its_two_list_files():
    for path in sorted((EVIDENCE / "numgrammar").glob("*.py")):
        rel = f"numgrammar/{path.name}"
        kinds = _uses(ast.parse(path.read_text("utf-8")), rel)
        assert kinds <= ({"nonascii-list"} if rel == "numgrammar/api.py" else set()), (rel, kinds)


def test_every_pinned_exception_names_a_module_that_exists():
    assert all((EVIDENCE / rel).is_file() for rel, _ in ALLOWED)


# ------------------------------------------------------------------------------ layer B ---
# the characters of the round-1 audit: placeholder, spaces, expanders, superscripts, look-alike multipliers
AUDITED = (0xE000, 0x2007, 0x2008, 0x200A, 0x200B, 0xFDFA, 0xFDFB, 0xFF10, 0xFF11, 0x2070, 0x00B9, 0x2080, 0x2715, 0x2A2F,
           0x22C5, 0x2217, 0x0445, 0x03C7, 0x00AD, 0x0661, 0x2212, 0x2012, 0x2014, 0x223C, 0x00BD, 0x2460)


@pytest.fixture(scope="module")
def sample() -> list[int]:
    return sorted(set(O.non_outside_cps()) | set(O.stratified_outside(2048)) | set(AUDITED))


@pytest.mark.parametrize("name", sorted(B.SWEEPS))
def test_layer_b_sample_has_no_violation(name, sample):
    assert B.SWEEPS[name](sample) == []


def test_layer_b_sample_is_representative(sample):
    assert len(sample) >= 1500
    assert {P.expected_class(cp) for cp in sample} == {"inside", "gap", "context", "line", "outside"}
    assert set(AUDITED) <= set(sample)


def test_layer_b_catches_the_placeholder_the_audit_found(monkeypatch):
    from app.evidence import parse_guard as G
    monkeypatch.setattr(G, "_free_mark", lambda page: "")
    assert any("U+E000" in m for m in B.sweep_thin([0xE000, 0x41, 0x2007]))


def test_layer_b_catches_a_widened_thin_space_set(monkeypatch):
    from app.evidence import parse_guard as G
    monkeypatch.setattr(G, "THIN_SPACES", G.THIN_SPACES | {" "})
    assert any("U+2007" in m for m in B.sweep_thin([0x2007]))


def test_layer_b_catches_the_deleted_multiplier(monkeypatch):
    from app.evidence.numgrammar import api
    clean = api._scale_view
    monkeypatch.setattr(api, "scale_shapes", lambda label: clean(api.fold("".join(c for c in label if c.isascii()))))
    monkeypatch.setattr(B, "scale_shapes", api.scale_shapes)
    assert B.sweep_scale([0x2715])


def test_layer_b_catches_nfkc_equating_a_superscript_digit(monkeypatch):
    import unicodedata

    from app.evidence import normalize as N
    monkeypatch.setattr(N, "nfkc_char", lambda ch: unicodedata.normalize("NFKC", ch))
    assert any("U+2070" in m or "U+00B9" in m for m in B.sweep_text([0x2070, 0x00B9]))


def test_layer_b_catches_a_decoy_digit_that_votes(monkeypatch):
    from app.evidence import normalize as N
    monkeypatch.setattr(N, "decimal_evidence", lambda text: (len(text.split("\n")), 0))
    monkeypatch.setattr(B, "decimal_evidence", N.decimal_evidence)
    assert B.sweep_style([0x0663])


@pytest.mark.exhaustive
@pytest.mark.skipif(os.environ.get("R6_EXHAUSTIVE") != "1", reason="set R6_EXHAUSTIVE=1 for the 1.1 M code-point run")
def test_layer_b_every_code_point_on_a_process_pool():
    workers = int(os.environ.get("R6_WORKERS", max(1, min(12, (os.cpu_count() or 2) - 2))))
    result = B.full_run(workers)
    assert result["failures"] == [] and result["errors"] == 0, result
    assert set(result["code_points"].values()) == {0x110000 - 2048}          # every scalar value, surrogates excluded
    assert result["dirty"] == [] or os.environ.get("R6_ALLOW_DIRTY") == "1", (
        "the run is evidence of a dirty tree, not of the commit: commit first or set R6_ALLOW_DIRTY=1", result["dirty"])
