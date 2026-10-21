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
    assert {"nca.linear_up_log_down", "nca.auc_last_ends_at_tlast", "nca.blq_m1", "nca.measured_zero",
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
    from app.compute.bioequivalence import be_one_parameter
    be = inspect.signature(be_one_parameter).parameters
    bp = by_id["be.average_tost"].parameters
    assert (bp["alpha"], bp["lower"], bp["upper"]) == (be["alpha"].default, be["lower"].default, be["upper"].default)
    assert bp["ci_percent"] == round(100 * (1 - be["alpha"].default))


def test_known_gaps_are_recorded_for_the_lambda_z_tie_rule():
    c = {c.id: c for c in load_catalog().conventions}["nca.lambda_z_window"]
    assert any("tie" in g.lower() for g in c.known_gaps)


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
    assert "nca.blq_m1" in out.read_text()


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
