"""Attack round 3 regressions on approval and export:

- the conflict gate and the duplicate flag compare the UNIT too (2.31 L/h vs 2.31 mL/min);
- no piece of a ';'- or ','-separated text cell starts a spreadsheet formula (group label, model
  flag code, PDF title), and model flag codes are plain identifiers;
- approval and export re-check the source file's hash (a changed PDF is never approved or shipped);
- the wide file's '<field>_unit' list lines up with '<field>' and '<field>_literal';
- a re-export under another template keeps its own wide file, and the result describes that file.
"""
from __future__ import annotations

import csv
import io
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import settings
from app.core.pharmstate import PharmState
from app.core.provenance import file_sha256
from app.evidence import export_tools, extract_tools, llm_access
from app.evidence import store as evstore
from app.evidence.qc_flags import merge_llm_flags
from app.tools.base import ToolContext
from tests import evidence_fixtures as fx
from tests.test_evidence_extract import SID, SchemaLLM, _extract, _include, _records, _template


@pytest.fixture(scope="module")
def papers(tmp_path_factory) -> dict[str, Path]:
    return fx.build_all(tmp_path_factory.mktemp("papers"))


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    st = evstore.configure(":memory:")
    llm_access.set_llm(None)
    yield SimpleNamespace(store=st, ctx=ToolContext(data_dir=str(tmp_path), actor="alice"), sid=SID, dir=tmp_path)
    llm_access.set_llm(None)


def _decide(env, record_id, **extra):
    args = {"record_id": record_id, "decision": "approve", "reason": "checked", "confirm": True, **extra}
    return extract_tools.decide_records(PharmState(session_id=env.sid), replace(env.ctx, actor="alice"), args)


def _export(env):
    return export_tools.export_evidence_dataset(PharmState(session_id=env.sid), env.ctx, {}).result


def _propose(env, records, flags=None):
    llm_access.set_llm(SchemaLLM({"records": records}, flags))
    _extract(env)
    llm_access.set_llm(None)


def _pdf(env, title: str, *paras: str, name: str = "paper_q") -> dict:
    d = fx._Doc(title, "ascii")
    d.line(title, 13)
    for p in paras:
        d.para(p)
    return {name: d.save(env.dir / f"{name}.pdf")}


CL = {"field": "cl", "page": 1, "kind": "text", "group": ""}


# -------------------------------------------------------- unit-aware conflicts ---

def test_same_number_in_another_unit_is_a_conflict(env):
    papers = _pdf(env, "Clearance of drug Q in adults",
                  "Results. The apparent clearance (CL/F) was 2.31 L/h in the main analysis.",
                  "Sensitivity analysis. Clearance was 2.31 mL/min after excluding outliers.")
    src = _include(env, papers, names=("paper_q",))["paper_q"]
    _template(env, {"fields": [{"name": "cl", "type": "number"}]})
    _propose(env, [{**CL, "value": "2.31", "unit": "L/h", "snippet": "(CL/F) was 2.31 L/h in the main analysis"},
                   {**CL, "value": "2.31", "unit": "mL/min", "snippet": "Clearance was 2.31 mL/min after excluding"}])
    recs = sorted(_records(env, src, "cl"), key=lambda r: r.unit)
    assert [r.verify_status for r in recs] == ["pass", "pass"]
    assert all("duplicate" in {f["code"] for f in r.flags} for r in recs), [r.flags for r in recs]
    _decide(env, recs[0].id)
    with pytest.raises(ValueError, match="already holds"):
        _decide(env, recs[1].id)
    assert _decide(env, recs[1].id, allow_conflict=True).result["record"]["status"] == "approved"


# --------------------------------------------------------------- formula guard ---

PAYLOAD = "=cmd|' /C calc'!A0"


def _formula_cells(line: str) -> list[str]:
    cells = []
    for delim in (";", ","):
        cells += [c for c in next(csv.reader(io.StringIO(line), delimiter=delim)) if c.lstrip().startswith("=")]
    return cells


def test_no_semicolon_piece_of_an_exported_cell_is_a_formula(env, papers):
    _include(env, papers, names=("paper_d",))
    _template(env, {"fields": [{"name": "cl", "type": "number"}]})
    _propose(env, [{**CL, "value": "2.31", "unit": "L/h", "group": "Arm A;" + PAYLOAD + ";", "snippet": "was 2.31 L/h"}],
             flags={"flags": [{"code": "magnitude", "severity": "warn", "message": "check", "field": "cl"},
                              {"code": "=1+1", "severity": "warn", "message": "check", "field": "cl"}]})
    rec = env.store.list_records(SID)[0]
    assert all(f["code"] != "=1+1" for f in rec.flags), rec.flags
    _decide(env, rec.id)
    res = _export(env)
    for path in (res["long_path"], res["wide_path"]):
        for line in Path(path).read_text("utf-8").splitlines():
            assert not _formula_cells(line), (path, line)


def test_pdf_title_cannot_carry_a_formula_into_the_export(env):
    papers = _pdf(env, "Population PK of drug W;=1+1;",
                  "Results. The mean apparent clearance (CL/F) was 2.31 L/h (n = 10).")
    _include(env, papers, names=("paper_q",))
    _template(env, {"fields": [{"name": "n", "type": "integer", "aliases": ["N"]}]})
    _extract(env)
    for r in env.store.list_records(SID):
        if r.verify_status == "pass":
            _decide(env, r.id)
    lines = Path(_export(env)["long_path"]).read_text("utf-8").splitlines()
    assert any("Population PK" in line for line in lines)
    assert not any(_formula_cells(line) for line in lines)


def test_model_flag_codes_must_be_identifiers():
    merged = merge_llm_flags([], [{"code": c, "severity": "warn", "message": "m", "field": "cl"}
                                  for c in ("=1+1", "ok_code", "a;b", "Units")])
    assert [f["code"] for f in merged] == ["ok_code"]


# ------------------------------------------------------- changed source file ---

def test_changed_source_is_never_approved_or_exported(env, papers):
    src = _include(env, papers, names=("paper_d", "paper_a"))["paper_d"]
    _template(env, {"fields": [{"name": "cl", "type": "number"}]})
    _propose(env, [{**CL, "value": "2.31", "unit": "L/h", "snippet": "was 2.31 L/h"}])
    rec = [r for r in _records(env, src, "cl") if r.value == "2.31"][0]
    assert rec.verify_status == "pass"
    original = Path(src.pdf_path).read_bytes()
    Path(src.pdf_path).write_bytes(papers["paper_a"].read_bytes())
    with pytest.raises(ValueError, match="source"):
        _decide(env, rec.id)
    assert env.store.get_record(rec.id).status == "proposed"
    Path(src.pdf_path).write_bytes(original)
    _decide(env, rec.id)
    Path(src.pdf_path).write_bytes(papers["paper_a"].read_bytes())
    with pytest.raises(ValueError, match="source"):
        _export(env)


def test_export_leaves_out_rows_of_a_changed_source(env):
    text = "Results. The apparent clearance (CL/F) was 2.31 L/h in the main analysis."
    papers = {**_pdf(env, "Clearance of drug Q", text), **_pdf(env, "Clearance of drug R", text, name="paper_r")}
    srcs = _include(env, papers, names=("paper_q", "paper_r"))
    _template(env, {"fields": [{"name": "cl", "type": "number"}]})
    _propose(env, [{**CL, "value": "2.31", "unit": "L/h", "snippet": "was 2.31 L/h"}])
    recs = env.store.list_records(SID)
    assert {r.source_id for r in recs if r.verify_status == "pass"} == {s.id for s in srcs.values()}
    for r in recs:
        _decide(env, r.id)
    Path(srcs["paper_q"].pdf_path).write_bytes(b"%PDF-1.4 changed")
    res = _export(env)
    assert [x["source_id"] for x in res["left_out"]] == [srcs["paper_q"].id]
    assert "changed" in res["left_out"][0]["reason"]
    assert srcs["paper_q"].sha256 not in Path(res["long_path"]).read_text("utf-8")
    assert srcs["paper_r"].sha256 in Path(res["long_path"]).read_text("utf-8")


# ------------------------------------------------------------------ wide file ---

def test_wide_units_line_up_with_values(env):
    papers = _pdf(env, "Clearance of drug Q in adults",
                  "Results. The apparent clearance (CL/F) was 3.10 L/h in the main analysis.",
                  "Second analysis. Clearance was 2.31 L/h in women.",
                  "Sensitivity analysis. Clearance was 0.50 mL/min after excluding outliers.")
    _include(env, papers, names=("paper_q",))
    _template(env, {"fields": [{"name": "cl", "type": "number"}]})
    _propose(env, [{**CL, "value": "3.10", "unit": "L/h", "snippet": "was 3.10 L/h in the main"},
                   {**CL, "value": "2.31", "unit": "L/h", "snippet": "was 2.31 L/h in women"},
                   {**CL, "value": "0.50", "unit": "mL/min", "snippet": "was 0.50 mL/min after"}])
    for r in env.store.list_records(SID):
        assert r.verify_status == "pass", r
        _decide(env, r.id, allow_conflict=True)
    res = _export(env)
    long_pairs = {(row["value_num"], row["unit"]) for row in csv.DictReader(open(res["long_path"], encoding="utf-8"))}
    wide = next(csv.DictReader(open(res["wide_path"], encoding="utf-8")))
    values, units = wide["cl"].split(" | "), wide["cl_unit"].split(" | ")
    assert len(units) == len(values) == 3
    assert set(zip(values, units)) == long_pairs


def test_reexport_under_a_new_template_keeps_its_own_wide_file(env, papers):
    _include(env, papers, names=("paper_d",))
    _template(env, {"fields": [{"name": "cl", "type": "number"}]})
    _propose(env, [{**CL, "value": "2.31", "unit": "L/h", "snippet": "was 2.31 L/h"}])
    _decide(env, env.store.list_records(SID)[0].id)
    first = _export(env)
    t2 = _template(env, {"fields": [{"name": "cl", "type": "number"}, {"name": "vd", "type": "number"}]})
    time.sleep(1.1)
    second = _export(env)
    header = Path(second["wide_path"]).read_text("utf-8").splitlines()[0].split(",")
    assert header == second["wide_columns"] and "vd" in header
    assert file_sha256(second["wide_path"]) == second["wide_sha256"] == second["dataset"]["wide_sha256"]
    assert second["template_hash"] == second["dataset"]["template_hash"] == t2
    assert Path(first["wide_path"]).is_file() and first["wide_path"] != second["wide_path"]
    stored = {d.id: d for d in env.store.list_datasets(SID)}
    assert stored[second["dataset"]["id"]].wide_sha256 == second["wide_sha256"]
