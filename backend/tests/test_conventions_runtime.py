"""Slice 2 (part 1): the conventions catalog used at runtime without changing any number or existing text.

NCA results gain a `conventions` block naming the catalog version and each applied convention (id, version, approval,
generated rule text). The catalog is validated at application startup and bundled into the desktop build. Run
provenance is guarded code and is NOT touched here."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import settings
from app.conventions import load_catalog, rule_text
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore

BACKEND = Path(__file__).resolve().parents[1]

# These expected sets are written by hand from reading the code paths. They pin which conventions each path names;
# they cannot show that a list is semantically complete. That judgement belongs to owner review.
SD_FLAG = {"nca.linear_up_log_down", "nca.auc_last_ends_at_tlast", "nca.blq_leading_middle_trailing",
           "nca.measured_zero", "nca.blq_flag_codes", "nca.pre_dose_records", "nca.lambda_z_window"}
SD_NOFLAG = {"nca.linear_up_log_down", "nca.auc_last_ends_at_tlast", "nca.pre_dose_records", "nca.lambda_z_window"}
SS_NOFLAG = {"nca.linear_up_log_down", "nca.lambda_z_window", "nca.ss_final_interval_scope",
             "nca.ss_endpoint_closure", "nca.ss_incomplete_interval_fail_closed"}
SS_FLAG = SS_NOFLAG | {"nca.blq_leading_middle_trailing", "nca.measured_zero", "nca.blq_flag_codes"}


# ── the runtime helper ─────────────────────────────────────────────────────────────────────────
def test_applied_conventions_carry_version_approval_and_generated_text():
    from app.conventions.runtime import applied_conventions

    out = applied_conventions(["nca.linear_up_log_down", "be.average_tost"])
    by_id = {c.id: c for c in load_catalog().conventions}
    assert out["catalog_version"] == load_catalog().catalog_version
    got = {e["id"]: e for e in out["conventions"]}
    assert set(got) == {"nca.linear_up_log_down", "be.average_tost"}
    for cid, e in got.items():
        c = by_id[cid]
        assert e["version"] == c.version and e["rule_text"] == rule_text(c)
        assert e["approval"] == c.approval.status and e["approved_by"] == c.approval.by
    assert got["be.average_tost"]["approval"] == "pending_owner_review" and got["be.average_tost"]["approved_by"] is None
    assert "not regulatory validation" in out["approval_meaning"]


def test_an_unknown_convention_id_is_a_programming_error():
    from app.conventions.runtime import applied_conventions

    with pytest.raises(KeyError):
        applied_conventions(["nca.no_such_convention"])


def test_startup_validation_fails_fast_on_a_broken_catalog(tmp_path):
    from app.conventions.runtime import validate_catalog_at_startup

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"catalog_version": "1.0", "conventions": []}))
    with pytest.raises(ValueError):
        validate_catalog_at_startup(bad)
    assert validate_catalog_at_startup() == load_catalog().catalog_version


def test_the_application_validates_the_catalog_at_startup():
    assert "validate_catalog_at_startup()" in (BACKEND / "app" / "main.py").read_text()


def test_the_desktop_build_bundles_the_catalog():
    spec = (BACKEND.parent / "desktop" / "PharmAgent.spec").read_text()
    assert '"conventions" / "catalog.json"' in spec and '"app/conventions"' in spec


# ── NCA results name the conventions they applied ──────────────────────────────────────────────
@pytest.fixture
def nca(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)

    def _run(lines: list[str]) -> dict:
        path = tmp_path / "d.csv"
        path.write_text("\n".join(lines) + "\n")
        orch = Orchestrator(llm=MockLLM(), store=SessionStore(":memory:"))
        sid = orch.create_session().id
        orch.run_tool(sid, "load_dataset", "data_manager", {"path": str(path)})
        return orch.run_tool(sid, "compute_nca", "nca", {})["result"]["nca_summary"]

    return _run


SD = [(0, 0.0), (1, 5.0), (2, 8.0), (4, 4.0), (8, 2.0), (12, 1.0), (24, 0.25)]
SS = [(0.0, 2.0), (1.0, 6.0), (2.0, 8.0), (4.0, 5.0), (8.0, 3.0), (12.0, 2.0)]


def _sd(flag: bool) -> list[str]:
    head = "ID,TIME,DV,AMT,EVID" + (",BLQ" if flag else "")
    rows = ["A,0,,100,1" + (",0" if flag else "")]
    rows += [f"A,{t},{c},0,0" + (",0" if flag else "") for t, c in SD]
    return [head, *rows]


def _ss(flag: bool) -> list[str]:
    head = "ID,TIME,DV,AMT,EVID,II,ADDL" + (",BLQ" if flag else "")
    rows = ["S,0,,100,1,12,3" + (",0" if flag else "")]
    rows += [f"S,{36 + t},{c},0,0,," + (",0" if flag else "") for t, c in SS]
    return [head, *rows]


@pytest.mark.parametrize("lines,expected", [
    (_sd(True), SD_FLAG), (_sd(False), SD_NOFLAG), (_ss(True), SS_FLAG), (_ss(False), SS_NOFLAG),
], ids=["single-dose-flag", "single-dose-no-flag", "steady-state-flag", "steady-state-no-flag"])
def test_nca_results_name_the_conventions_applied_on_each_path(nca, lines, expected):
    block = nca(lines)["conventions"]
    assert {e["id"] for e in block["conventions"]} == expected
    assert block["catalog_version"] == load_catalog().catalog_version
    by_id = {c.id: c for c in load_catalog().conventions}
    assert all(e["rule_text"] == rule_text(by_id[e["id"]]) for e in block["conventions"])


def test_the_existing_blq_rule_text_is_unchanged(nca):
    summary = nca(_sd(True))
    assert summary["blq"]["rule"].startswith("M1:")          # text the UI already shows; replaced in a later slice


def test_the_result_stays_json_serialisable(nca):
    json.dumps(nca(_ss(True)))
