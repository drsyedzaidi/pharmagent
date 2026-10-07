"""r6 round 2, finding 4: no dead code or stale legacy constant is left after the rewrite.

* the test-only helpers (``verify.find_number_in_snippet``, ``numgrammar.find_n`` / ``token_at`` / ``words_around``) live in
  ``tests/evidence_r6_helpers.py``; ``is_digit_run`` is internal to the package;
* every name a package re-exports has a caller in ``app/`` (outside the package, or inside it);
* no module constant of ``app/evidence`` is referenced nowhere (``models.py`` holds the documented vocabularies that the
  frontend mirrors and that the tests pin: those are exempt);
* ``REASON_NON_ASCII_DIGITS`` is gone from the code: a stored row keeps its ``verify_reason`` string, which the store and
  the API pass through and which the frontend still labels.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.evidence import config_tools, regions, relayout
from app.evidence import models as M
from app.evidence import numgrammar as NG
from app.evidence import verify as V
from app.evidence.numgrammar import api
from app.evidence.qc_flags import scripted_flags
from app.evidence.store import EvidenceStore
from tests import evidence_r6_helpers as H

BACKEND = Path(__file__).resolve().parents[1]
APP = BACKEND / "app"
EVIDENCE = APP / "evidence"


def _trees() -> dict[Path, ast.Module]:
    return {p: ast.parse(p.read_text("utf-8")) for p in sorted(APP.rglob("*.py"))}


def _used_names(tree: ast.Module) -> set[str]:
    """The names a module USES: loads and attribute accesses (an import or a ``__all__`` entry is no use)."""
    return ({n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
            | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)})


# ----------------------------------------------------------------- the helpers moved to the tests ---

@pytest.mark.parametrize("name", ["find_n", "token_at", "words_around"])
def test_the_test_only_helpers_are_not_part_of_the_reader(name):
    assert not hasattr(NG, name) and not hasattr(api, name) and name not in NG.__all__ and name not in api.__all__
    assert callable(getattr(H, name))


def test_the_test_only_verifier_helper_is_not_part_of_the_verifier():
    assert not hasattr(V, "find_number_in_snippet") and "find_number_in_snippet" not in getattr(V, "__all__", ())
    assert H.find_number_in_snippet("3412", "Cmax 3,412 (812)", ".", False) == "3,412"


def test_the_digit_run_helper_is_internal_to_the_package():
    assert "is_digit_run" not in NG.__all__ and "is_digit_run" not in api.__all__ and not hasattr(NG, "is_digit_run")
    assert api.is_digit_run("123", 1, 3) and not api.is_digit_run("1234", 1, 3)


def test_the_helpers_in_the_tests_read_like_the_old_ones():
    assert H.find_n("The study (n=12) and N = 40") == [(13, 15, 12), (25, 27, 40)]
    assert H.token_at("AUC 12.5 mg", 4, 8).value == 12.5 and H.token_at("AUC 12.5 mg", 0, 3) is None
    assert H.words_around("AUC <   12.5 mg/L", 9, 13) == ("<", "mg/L")


def test_every_name_the_reader_package_exports_has_a_caller_in_the_application():
    """A name re-exported by ``numgrammar`` must be used by an ``app/`` module: outside the package, or (for a helper
    the others build on) inside it. A name only the tests call is a test helper, not part of the reader."""
    used: set[str] = set()
    for path, tree in _trees().items():
        if path.name == "__init__.py" and path.parent.name == "numgrammar":
            continue
        used |= _used_names(tree)
    dead = sorted(n for n in NG.__all__ if n not in used)
    assert dead == [], f"re-exported with no caller in app/: {dead}"
    assert set(api.__all__) <= set(NG.__all__) | {"digit_group"}, sorted(set(api.__all__) - set(NG.__all__))


def test_no_module_constant_of_the_evidence_package_is_referenced_nowhere():
    trees = _trees()
    used: set[str] = set()
    for tree in trees.values():
        used |= _used_names(tree)
    dead = []
    for path, tree in trees.items():
        if EVIDENCE not in path.parents or path.name == "models.py":
            continue
        for node in tree.body:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) \
                else []
            for t in targets:
                names = [t] if isinstance(t, ast.Name) else [e for e in getattr(t, "elts", []) if isinstance(e, ast.Name)]
                dead += [f"{path.relative_to(APP)}:{node.lineno} {n.id}" for n in names
                         if n.id.isupper() and n.id not in used]
    assert dead == []


@pytest.mark.parametrize("module, name", [(config_tools, "LAYOUT_KEYS"), (regions, "REASONS"), (relayout, "STATUSES"),
                                          (relayout, "OUTCOMES")])
def test_the_constants_the_audit_found_unused_are_gone(module, name):
    assert not hasattr(module, name)


# ----------------------------------------------------------------------- the legacy reason ---

def test_the_legacy_reason_constant_is_gone_and_the_tuple_holds_what_the_verifier_emits():
    assert not hasattr(M, "REASON_NON_ASCII_DIGITS") and "non_ascii_digits" not in M.REASON_CODES
    assert len(M.REASON_CODES) == len(set(M.REASON_CODES)) == 15
    assert M.REASON_UNRESOLVED_CONTEXT in M.REASON_CODES and M.REASON_UNRESOLVED_SHAPE in M.REASON_CODES


def test_a_row_stored_by_the_old_reader_keeps_its_reason_through_the_store_and_the_flags():
    from app.evidence import store as evstore
    st: EvidenceStore = evstore.configure(":memory:")
    src = st.add_source(session_id="s1", sha256="a" * 64, filename="paper.pdf", pdf_path="/x/a.pdf",
                        parse_path="/x/a.pages.json", pages=1, title="t", metadata={}, text_chars=10,
                        parse_status="parsed", decimal_style=".", uploaded_by="tester")
    loc = M.Locator(page=1, kind="text", snippet="Cmax was ١٢ ng/mL").to_json()
    legacy = {"field": "Cmax", "group_label": "", "value": "12", "value_num": None, "unit": "ng/mL", "locator": loc,
              "extractor": "llm", "verify_status": "fail", "verify_reason": "non_ascii_digits", "flags": []}
    (rec,) = st.insert_records("s1", src, "t" * 64, [legacy])
    assert st.get_record(rec.id).verify_reason == "non_ascii_digits"
    assert [r.verify_reason for r in st.list_records("s1")] == ["non_ascii_digits"]
    flags = scripted_flags([M.as_mapping(st.get_record(rec.id))], {"fields": [{"name": "Cmax", "type": "number"}]}, [])[0]
    assert any(f["code"] == M.FLAG_VERIFY_FAIL and "(non_ascii_digits)" in f["message"] for f in flags)
