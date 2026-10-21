"""The conventions catalog (slice 1, read-only): typed schema, validation, test-coverage check, generated rule text, and
an exportable report. Nothing here changes runtime behaviour; the catalog is not wired into any tool yet."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.conventions import (
    Catalog,
    Convention,
    check_test_coverage,
    load_catalog,
    render_report_json,
    render_report_markdown,
    rule_text,
)
from app.conventions.__main__ import main as cli_main

BACKEND = Path(__file__).resolve().parents[1]


def _conv(**over):
    base = {
        "id": "nca.example", "title": "Example", "domain": "nca", "applies_to": ["compute_nca"],
        "statement": "Cut-off is {cutoff}.", "parameters": {"cutoff": 3.84}, "version": "1.0.0",
        "source": {"kind": "owner_decision", "reference": "owner 2026-10-21"},
        "approval": {"status": "pending_owner_review"},
        "tests": ["tests/test_conventions_catalog.py::test_the_shipped_catalog_loads_and_validates"],
    }
    return {**base, **over}


# ── schema ─────────────────────────────────────────────────────────────────────────────────────
def test_a_valid_convention_parses_and_renders_its_parameters_into_the_text():
    c = Convention(**_conv())
    assert rule_text(c) == "Cut-off is 3.84."


@pytest.mark.parametrize("bad", [
    {"id": "Bad Id"}, {"id": "nca"}, {"version": "1.0"}, {"tests": []}, {"domain": "astrology"},
    {"statement": "Cut-off is {missing}."},                          # placeholder with no parameter
    {"parameters": {"cutoff": 3.84, "unused": 1}},                    # parameter the text never uses
    {"source": {"kind": "hearsay", "reference": "x"}}, {"source": {"kind": "owner_decision", "reference": ""}},
    {"approval": {"status": "approved"}},                             # approved needs who and when
    {"extra_field": 1},
])
def test_invalid_conventions_are_rejected(bad):
    with pytest.raises(ValidationError):
        Convention(**_conv(**bad))


def test_an_approved_convention_needs_an_approver_and_a_date():
    ok = Convention(**_conv(approval={"status": "approved", "by": "owner", "date": "2026-10-21"}))
    assert ok.approval.status == "approved"


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValidationError):
        Catalog(catalog_version="1.0.0", conventions=[_conv(), _conv()])


# ── the shipped catalog ────────────────────────────────────────────────────────────────────────
def test_the_shipped_catalog_loads_and_validates():
    cat = load_catalog()
    assert cat.catalog_version and len(cat.conventions) >= 12


def test_the_shipped_catalog_has_the_initial_nca_be_and_profile_conventions():
    ids = {c.id for c in load_catalog().conventions}
    assert {"nca.linear_up_log_down", "nca.auc_last_ends_at_tlast", "nca.blq_leading_middle_trailing", "nca.measured_zero",
            "nca.blq_flag_codes", "nca.ss_final_interval_scope", "nca.ss_endpoint_closure",
            "nca.ss_incomplete_interval_fail_closed", "nca.lambda_z_window", "nca.pre_dose_records",
            "be.average_tost", "profile.ofv_cutoff"} <= ids


def test_every_referenced_test_exists():
    assert check_test_coverage(load_catalog(), BACKEND) == []


def test_a_reference_to_a_missing_test_or_file_is_reported():
    cat = Catalog(catalog_version="1.0.0", conventions=[
        _conv(id="x.a", tests=["tests/test_conventions_catalog.py::test_no_such_test"]),
        _conv(id="x.b", tests=["tests/no_such_file.py::test_x"]),
        _conv(id="x.c", tests=["not-a-node-id"]),
    ])
    problems = {(p.convention_id): p.reason for p in check_test_coverage(cat, BACKEND)}
    assert set(problems) == {"x.a", "x.b", "x.c"}


def test_the_catalog_numbers_match_the_code_they_describe():
    """A catalog number that disagrees with the code is a documentation bug; pin the ones that can be read from code."""
    from app.compute.dosing import extract_ss_intervals_checked
    from app.compute.profile import _chi2_1df

    by_id = {c.id: c for c in load_catalog().conventions}
    assert by_id["profile.ofv_cutoff"].parameters["cutoff"] == round(_chi2_1df(0.95), 2)
    sig = inspect.signature(extract_ss_intervals_checked)
    assert by_id["nca.ss_endpoint_closure"].parameters["snap_fraction"] == sig.parameters["tol_frac"].default
    from app.compute.dosing import PRE_DOSE_TOL
    assert by_id["nca.ss_endpoint_closure"].parameters["pre_dose_tol"] == PRE_DOSE_TOL
    from app.compute.bioequivalence import be_one_parameter
    be = inspect.signature(be_one_parameter).parameters
    bp = by_id["be.average_tost"].parameters
    assert (bp["alpha"], bp["lower"], bp["upper"]) == (be["alpha"].default, be["lower"].default, be["upper"].default)
    assert bp["ci_percent"] == round(100 * (1 - be["alpha"].default))
    from app.compute.nca import _MIN_LAMBDA_Z
    assert by_id["nca.lambda_z_window"].parameters["min_lambda_z"] == _MIN_LAMBDA_Z


def test_the_lambda_z_entry_pins_the_threshold_and_the_tie_rule_with_dedicated_tests():
    c = {c.id: c for c in load_catalog().conventions}["nca.lambda_z_window"]
    names = " ".join(c.tests)
    assert "test_nca_lambda_z.py::test_equal_fits_prefer_the_longest_window" in names
    assert "test_nca_lambda_z.py::test_a_decline_below_the_threshold_gives_no_lambda_z" in names
    assert "materially" not in c.statement and "{min_lambda_z}" in c.statement
    assert not any("not pinned" in g.lower() for g in c.known_gaps)


def test_every_shipped_convention_is_pending_owner_review_until_the_owner_approves():
    assert {c.approval.status for c in load_catalog().conventions} == {"pending_owner_review"}


# ── generated text and the report ──────────────────────────────────────────────────────────────
def test_rule_text_is_generated_not_stored():
    raw = json.loads((BACKEND / "app" / "conventions" / "catalog.json").read_text())
    assert all("rule_text" not in c for c in raw["conventions"])


def test_the_markdown_report_lists_every_convention_with_version_approval_tests_and_gaps():
    cat = load_catalog()
    md = render_report_markdown(cat, check_test_coverage(cat, BACKEND))
    for c in cat.conventions:
        assert c.id in md and rule_text(c) in md
    assert "pending_owner_review" in md and "Known gaps" in md and "Coverage problems: 0" in md


def test_the_json_report_round_trips_and_carries_generated_text():
    cat = load_catalog()
    rep = json.loads(render_report_json(cat, check_test_coverage(cat, BACKEND)))
    assert rep["catalog_version"] == cat.catalog_version and rep["coverage_problems"] == []
    assert {c["id"]: c["rule_text"] for c in rep["conventions"]} == {c.id: rule_text(c) for c in cat.conventions}


# ── CLI ────────────────────────────────────────────────────────────────────────────────────────
def test_cli_check_passes_on_the_shipped_catalog(capsys):
    assert cli_main(["--check"]) == 0


def test_cli_writes_a_markdown_report(tmp_path):
    out = tmp_path / "r.md"
    assert cli_main(["report", "--format", "md", "--output", str(out)]) == 0
    assert "nca.blq_leading_middle_trailing" in out.read_text()


def test_cli_check_fails_on_a_broken_catalog(tmp_path):
    bad = json.loads((BACKEND / "app" / "conventions" / "catalog.json").read_text())
    bad["conventions"][0]["tests"] = ["tests/no_such_file.py::test_x"]
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    assert cli_main(["--check", "--catalog", str(p)]) == 1


def test_the_package_changes_no_runtime_module():
    """Slice 1 is read-only: nothing under app/ outside app/conventions imports it."""
    offenders = [str(p) for p in (BACKEND / "app").rglob("*.py")
                 if "conventions" not in p.parts and "app.conventions" in p.read_text()]
    assert offenders == []


# ── owner review of PR #16: approval semantics and wording corrections ─────────────────────────
@pytest.mark.parametrize("extra", [{"by": "owner"}, {"date": "2026-10-21"}, {"by": "owner", "date": "2026-10-21"}])
def test_a_pending_entry_rejects_an_approver_or_a_date(extra):
    with pytest.raises(ValidationError):
        Convention(**_conv(approval={"status": "pending_owner_review", **extra}))


@pytest.mark.parametrize("bad", ["not-a-date", "2026-13-40", "2026-02-30", "20261021", "2026-10-21T00:00", "21/10/2026", ""])
def test_an_approval_date_must_be_an_iso_calendar_date(bad):
    with pytest.raises(ValidationError):
        Convention(**_conv(approval={"status": "approved", "by": "owner", "date": bad}))


def test_the_reports_say_that_approval_is_not_regulatory_validation():
    cat = load_catalog()
    md = render_report_markdown(cat, [])
    assert "not regulatory validation" in md
    assert "not regulatory validation" in json.loads(render_report_json(cat, []))["approval_meaning"]


def test_nothing_is_called_m1_without_an_authoritative_source():
    for c in load_catalog().conventions:
        assert "M1" not in c.title and "M1" not in c.statement, c.id
    assert "nca.blq_m1" not in {c.id for c in load_catalog().conventions}


def test_the_endpoint_closure_wording_is_consistent():
    c = {c.id: c for c in load_catalog().conventions}["nca.ss_endpoint_closure"]
    low = c.statement.lower()
    assert "at least one" in low and "measured" in low and "at most one" in low
    assert "distinct" in low and "periodicity" in low and "needs its own sample" not in low
    assert any("integration rule" in g.lower() for g in c.known_gaps)


def test_the_be_entry_discloses_dropped_rows_and_the_unchecked_guidance():
    c = {c.id: c for c in load_catalog().conventions}["be.average_tost"]
    gaps = " ".join(c.known_gaps).lower()
    assert "not counted" in gaps and "guidance" in gaps
    assert "implementation behaviour" in c.statement.lower() and "silent" not in c.statement.lower()


def test_the_cutoff_entry_states_the_objective_scale_and_its_open_source():
    c = {c.id: c for c in load_catalog().conventions}["profile.ofv_cutoff"]
    assert "-2 log-likelihood" in c.statement
    assert any("source" in g.lower() for g in c.known_gaps)


def test_the_corrected_entries_were_version_bumped():
    by_id = {c.id: c for c in load_catalog().conventions}
    for cid in ("nca.blq_leading_middle_trailing", "nca.ss_final_interval_scope", "nca.ss_endpoint_closure",
                "nca.lambda_z_window", "be.average_tost", "profile.ofv_cutoff"):
        assert tuple(map(int, by_id[cid].version.split("."))) >= (1, 1, 0), cid
    assert tuple(map(int, load_catalog().catalog_version.split("."))) >= (1, 1, 0)


# ── coverage checker hardening (independent review of baf9b8e) ─────────────────────────────────
def _tmp_backend(tmp_path, files: dict[str, bytes | str]) -> Path:
    root = tmp_path / "backend"
    (root / "tests").mkdir(parents=True)
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        (p.write_bytes if isinstance(body, bytes) else p.write_text)(body)
    return root


def _problems(root, *refs):
    cat = Catalog(catalog_version="1.0.0", conventions=[_conv(id=f"x.r{i}", tests=[r]) for i, r in enumerate(refs)])
    return {p.ref: p.reason for p in check_test_coverage(cat, root)}


GOOD = "def test_ok():\n    assert True\n"


def test_a_top_level_test_function_is_accepted(tmp_path):
    root = _tmp_backend(tmp_path, {"tests/test_a.py": GOOD})
    assert _problems(root, "tests/test_a.py::test_ok") == {}


@pytest.mark.parametrize("ref", [
    "tests/../app/test_x.py::test_ok",              # traversal to a real file with a real test function
    "tests/sub/../../app/test_x.py::test_ok",
    "/abs/tests/test_x.py::test_ok",                # absolute
    "tests\\test_a.py::test_ok",                    # backslash
    "app/test_x.py::test_ok",                       # not under tests/
])
def test_references_outside_the_tests_root_are_rejected(tmp_path, ref):
    root = _tmp_backend(tmp_path, {"tests/test_a.py": GOOD, "tests/sub/test_s.py": GOOD, "app/test_x.py": GOOD})
    reason = _problems(root, ref).get(ref, "")
    assert reason and "does not exist" not in reason and "not found" not in reason


def test_a_symlink_escaping_the_tests_root_is_rejected(tmp_path):
    root = _tmp_backend(tmp_path, {"outside/test_evil.py": GOOD})
    (root / "tests" / "test_link.py").symlink_to(root / "outside" / "test_evil.py")
    assert "outside the tests root" in _problems(root, "tests/test_link.py::test_ok")["tests/test_link.py::test_ok"]


def test_malformed_python_is_reported_not_raised(tmp_path):
    root = _tmp_backend(tmp_path, {"tests/test_bad.py": "def test_ok(:\n"})
    assert "parse" in _problems(root, "tests/test_bad.py::test_ok")["tests/test_bad.py::test_ok"]


def test_a_non_utf8_file_is_reported_not_raised(tmp_path):
    root = _tmp_backend(tmp_path, {"tests/test_bin.py": b"def test_ok():\n    x = '\xff\xfe'\n"})
    assert "read" in _problems(root, "tests/test_bin.py::test_ok")["tests/test_bin.py::test_ok"]


@pytest.mark.parametrize("ref,body", [
    ("tests/test_n.py::test_inner", "def test_outer():\n    def test_inner():\n        pass\n"),   # nested
    ("tests/test_h.py::_helper", "def _helper():\n    pass\n"),                                # not a test name
    ("tests/test_h.py::helper", "def helper():\n    pass\n"),
    ("tests/test_c.py::test_m", "class TestX:\n    def test_m(self):\n        pass\n"),            # method: not a module-level node id
    ("tests/helpers.py::test_ok", GOOD),                                                        # file pytest does not collect
])
def test_only_top_level_collectable_test_functions_count(tmp_path, ref, body):
    root = _tmp_backend(tmp_path, {ref.split("::")[0]: body})
    assert ref in _problems(root, ref)


def test_the_cli_reports_an_unreadable_test_instead_of_crashing(tmp_path, capsys):
    raw = json.loads((BACKEND / "app" / "conventions" / "catalog.json").read_text())
    raw["conventions"][0]["tests"] = ["tests/../tests/test_conventions_catalog.py::test_the_shipped_catalog_loads_and_validates"]
    p = tmp_path / "c.json"
    p.write_text(json.dumps(raw))
    assert cli_main(["--check", "--catalog", str(p)]) == 1


def test_the_report_says_coverage_is_citation_not_semantic_proof():
    cat = load_catalog()
    md = render_report_markdown(cat, [])
    assert "does not show that a test exercises the rule" in md
    assert "does not show that a test exercises the rule" in json.loads(render_report_json(cat, []))["coverage_meaning"]


def test_the_endpoint_closure_states_the_separate_start_window():
    c = {c.id: c for c in load_catalog().conventions}["nca.ss_endpoint_closure"]
    assert "{pre_dose_tol}" in c.statement and "before the dose" in c.statement


def test_every_entry_cites_at_least_two_behavioural_tests_except_where_a_gap_says_why():
    for c in load_catalog().conventions:
        assert len(c.tests) >= 2 or c.known_gaps, c.id
