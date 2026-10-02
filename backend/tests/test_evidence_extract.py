"""Extraction tool: deterministic table/text rules (keyless), LLM proposals verified literally
against the parsed page, scripted QC flags, verify_records and decide_records. Tools are called
directly with a ToolContext on a tmp data_dir and a ':memory:' ledger; no HTTP.
"""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import settings
from app.core.pharmstate import PharmState, apply_writes
from app.evidence import extract_rules, extract_tools, ingest, llm_access, screen_tools
from app.evidence import models as M
from app.evidence import store as evstore
from app.evidence.config_tools import validate_criteria, validate_template
from app.evidence.normalize import normalise
from app.evidence.prompts import EVIDENCE_SYSTEM, EXTRACT_SCHEMA, FLAG_SCHEMA
from app.tools.base import ToolContext
from tests import evidence_fixtures as fx
from tests.test_evidence_llm import FakeLLM
from tests.test_evidence_screen import CRITERIA

INCLUDE_ALL = [{"id": "C2", "text": "Reports Cmax, AUC or clearance", "kind": "inclusion", "required": True,
                "keywords": ["Cmax", "AUC", "clearance"]}]
TEMPLATE = {"fields": [
    {"name": "cmax", "type": "number", "unit": "ng/mL", "required": True, "min": 0, "max": 100000,
     "aliases": ["Cmax", "Cmax,ss"]},
    {"name": "auc", "type": "number", "unit": "µg h/mL", "aliases": ["AUC0-inf", "AUC"]},
    {"name": "half_life", "type": "number", "unit": "h", "min": 0, "max": 200, "aliases": ["t1/2", "half-life"]},
    {"name": "n", "type": "integer", "aliases": ["N"]},
    {"name": "dose", "type": "number", "unit": "mg", "aliases": ["Dose"]},
    {"name": "clint", "type": "number", "unit": "µL/min/10^6 cells", "aliases": ["CLint"]},
    {"name": "fu", "type": "number", "aliases": ["Fraction unbound"]},
    {"name": "cl", "type": "number", "unit": "L/h", "aliases": ["CL/F", "CL"]},
    {"name": "cl_lower", "type": "number", "unit": "L/h"},
    {"name": "cl_upper", "type": "number", "unit": "L/h"},
    {"name": "cmax_se", "type": "number", "unit": "ng/mL"},
    {"name": "v", "type": "number", "unit": "L", "aliases": ["V/F"]},
    {"name": "species", "type": "text"},
]}
SID = "sess_extract"
ALL = ("paper_a", "paper_b", "paper_c", "paper_d")


class SchemaLLM(FakeLLM):
    """Keyed-looking client answering by schema: records for EXTRACT_SCHEMA, flags for FLAG_SCHEMA."""

    def __init__(self, records: dict | None, flags: dict | None = None) -> None:
        super().__init__(records)
        self.flags = flags

    def extract_json(self, system, user, schema, *, max_tokens=512):
        self.calls.append({"system": system, "user": user, "schema": schema})
        return self.flags if schema is FLAG_SCHEMA else self.answer


@pytest.fixture(scope="module")
def papers(tmp_path_factory) -> dict[str, Path]:
    return fx.build_all(tmp_path_factory.mktemp("papers"))


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    st = evstore.configure(":memory:")
    llm_access.set_llm(None)
    yield SimpleNamespace(store=st, ctx=ToolContext(data_dir=str(tmp_path)), sid=SID, dir=tmp_path)
    llm_access.set_llm(None)


def _ingest(env, path: Path, name: str, sid: str = SID):
    return ingest.ingest_pdf(path.read_bytes(), name, sid, "alice", data_dir=env.dir, store=env.store)


def _include(env, papers, names=ALL, criteria=INCLUDE_ALL, sid: str = SID) -> dict:
    """Ingest, screen with *criteria* and approve every verdict (include or not)."""
    sources = {n: _ingest(env, papers[n], f"{n}.pdf", sid) for n in names}
    env.store.save_config(sid, "criteria", validate_criteria(criteria), "alice")
    screen_tools.screen_publications(PharmState(session_id=sid), env.ctx, {})
    for s in env.store.list_screens(sid):
        env.store.decide_screen(s.id, "approved", "alice", "agree")
    return sources


def _template(env, template=TEMPLATE, sid: str = SID) -> str:
    return env.store.save_config(sid, "template", validate_template(copy.deepcopy(template)), "alice")


def _extract(env, args: dict | None = None, state: PharmState | None = None):
    state = state or PharmState(session_id=env.sid)
    res = extract_tools.extract_evidence(state, env.ctx, args or {})
    return res, apply_writes(state, "evidence", res.writes)


def _records(env, source=None, field=None, **kw) -> list:
    rows = env.store.list_records(env.sid, **kw)
    return [r for r in rows if (source is None or r.source_id == source.id) and (field is None or r.field == field)]


def _one(env, source, field, group=None):
    rows = [r for r in _records(env, source, field) if group is None or r.group_label == group]
    assert len(rows) == 1, [(r.field, r.group_label, r.value, r.extractor) for r in rows]
    return rows[0]


def _codes(rec) -> set[str]:
    return {f["code"] for f in rec.flags}


def _page_text(src, n: int, column: int | None = None) -> str:
    page = ingest.load_pages(src)["pages"][n - 1]
    return page["columns"][column] if column is not None else page["text"]


def _assert_roundtrip(src, rec) -> None:
    loc = rec.locator
    assert rec.verify_status == "pass" and rec.verify_reason == "", (rec.field, rec.verify_reason)
    assert loc["span"] is not None, (rec.field, loc)
    s, e = loc["span"]
    raw = _page_text(src, loc["page"], loc.get("column"))
    assert normalise(raw[s:e])[0] == normalise(loc["snippet"])[0]
    if loc["literal"] is not None:
        assert normalise(loc["literal"])[0] in normalise(raw[s:e])[0]


def _llm_records(*recs: dict) -> dict:
    return {"records": [{"kind": "text", "unit": None, "group": "", **r} for r in recs]}


# -------------------------------------------------------------- preconditions ---

def test_extract_requires_template_then_included_sources(env, papers):
    _ingest(env, papers["paper_a"], "a.pdf")
    with pytest.raises(ValueError, match="template"):
        _extract(env)
    _template(env)
    with pytest.raises(ValueError, match="include"):
        _extract(env)


def test_only_approved_include_verdicts_are_extracted(env, papers):
    sources = _include(env, papers, criteria=CRITERIA)          # paper_c -> exclude (approved exclude)
    _template(env)
    res, _ = _extract(env)
    done = {s["source_id"] for s in res.result["sources"]}
    assert sources["paper_c"].id not in done and sources["paper_a"].id in done
    assert any(s["source_id"] == sources["paper_c"].id and "include" in s["reason"] for s in res.result["skipped"])
    assert _records(env, sources["paper_c"]) == []


def test_source_ids_filter_and_unknown_id(env, papers):
    sources = _include(env, papers)
    _template(env)
    res, _ = _extract(env, {"source_ids": [sources["paper_a"].id]})
    assert {s["source_id"] for s in res.result["sources"]} == {sources["paper_a"].id}
    with pytest.raises(ValueError, match="unknown"):
        _extract(env, {"source_ids": ["src_nope"]})


# ---------------------------------------------------------- deterministic rules ---

def test_paper_a_ruled_table_records(env, papers):
    sources = _include(env, papers, names=("paper_a",))
    _template(env)
    res, state = _extract(env)
    src = sources["paper_a"]
    cmax = _one(env, src, "cmax")
    assert cmax.value == "3,412" and cmax.value_num == 3412 and cmax.unit == "ng/mL"
    assert cmax.group_label == "150 mg" and cmax.extractor == "table" and cmax.status == "proposed"
    loc = cmax.locator
    assert loc["kind"] == "table" and loc["page"] == 1 and loc["snippet"] == "3,412 (812)"
    assert loc["cell"]["header"] == "Cmax (ng/mL)" and loc["cell"]["row_label"] == "150 mg"
    assert loc["literal"] == "3,412" and loc["decimal_style"] == "."
    assert M.FLAG_COMPOSITE_CELL in _codes(cmax)
    auc = _one(env, src, "auc", "150 mg")
    assert auc.value_num == 24560 and auc.unit == "ng h/mL" and M.FLAG_UNIT_MISMATCH in _codes(auc)
    hl = _one(env, src, "half_life")
    assert hl.value == "11.4" and hl.unit == "h" and M.FLAG_UNIT_MISMATCH not in _codes(hl)
    n_tab = _one(env, src, "n", "150 mg")
    assert n_tab.value_num == 12 and n_tab.locator["cell"]["header"] == "n"
    dose = _one(env, src, "dose", "150 mg")
    assert dose.value == "150" and dose.unit == "mg" and dose.locator["snippet"] == "150 mg"
    assert res.result["n_new"] == len(_records(env)) and state.extraction_results["n_records"] == res.result["n_new"]


@pytest.mark.skipif(not fx.UNICODE_FONT, reason="no Unicode TTF found; ASCII variant built")
def test_paper_b_unicode_cells_and_unit_column(env, papers):
    sources = _include(env, papers, names=("paper_b",))
    _template(env)
    _extract(env)
    src = sources["paper_b"]
    cmax = _one(env, src, "cmax")
    assert cmax.value == "1 234" and cmax.value_num == 1234 and cmax.unit == "ng/mL"
    assert cmax.locator["snippet"] == "1 234 ± 312" and cmax.locator["literal"] == "1 234"
    assert cmax.group_label == "" and M.FLAG_COMPOSITE_CELL in _codes(cmax)
    auc = _one(env, src, "auc")
    assert auc.value_num == 18900 and auc.unit == "ng h/mL"
    n = _one(env, src, "n")
    assert n.value_num == 24 and n.locator["kind"] == "text" and "n\n= 24" in n.locator["snippet"]
    assert "Delta tmax" in next(s["unmatched_headers"] for s in [_last_source(env, src)])


def _last_source(env, src) -> dict:
    res, _ = _extract(env, {"source_ids": [src.id]})
    return next(s for s in res.result["sources"] if s["source_id"] == src.id)


def test_paper_c_comma_decimal_layout_table(env, papers):
    sources = _include(env, papers, names=("paper_c",))
    _template(env)
    _extract(env)
    src = sources["paper_c"]
    rat = _one(env, src, "clint", "Rat")
    assert rat.value == "18,2" and rat.value_num == 18.2 and rat.locator["decimal_style"] == ","
    assert rat.unit == "µL/min/10^6 cells" and rat.locator["page"] == 2
    assert rat.locator["cell"]["strategy"] == "layout" and M.FLAG_TABLE_LOW_CONFIDENCE in _codes(rat)
    mouse = _one(env, src, "clint", "Mouse")
    assert mouse.value_num == 22.7
    fu = _one(env, src, "fu", "Rat")
    assert fu.value == "0,12" and fu.value_num == 0.12
    assert all(r.verify_status == "pass" for r in _records(env, src))


def test_text_rules_find_n_and_dose(env, papers):
    sources = _include(env, papers, names=("paper_a", "paper_d"))
    _template(env)
    _extract(env)
    a, d = sources["paper_a"], sources["paper_d"]
    n_txt = _one(env, a, "n", "")
    assert n_txt.value_num == 12 and n_txt.locator["kind"] == "text" and "n = 12" in n_txt.locator["snippet"]
    dose = _one(env, a, "dose", "")
    assert dose.value == "150" and dose.value_num == 150 and dose.unit == "mg"
    assert "150 mg" in dose.locator["snippet"] and dose.locator["literal"] == "150"
    n_d = _one(env, d, "n")
    assert n_d.value_num == 10 and _records(env, d, "dose") == []


def test_every_rule_record_verifies_with_raw_span_round_trip(env, papers):
    sources = _include(env, papers)
    _template(env)
    _extract(env)
    rows = _records(env)
    assert len(rows) >= 12
    by_id = {s.id: s for s in sources.values()}
    for rec in rows:
        _assert_roundtrip(by_id[rec.source_id], rec)
        assert rec.verify_reason in M.REASON_CODES
        assert all(f["code"] in M.FLAG_CODES and f["origin"] == "script" for f in rec.flags)


def test_unmatched_headers_are_listed(env, papers):
    sources = _include(env, papers, names=("paper_a",))
    fields = [f for f in TEMPLATE["fields"] if f["name"] != "half_life"]
    _template(env, {"fields": fields})
    res, _ = _extract(env)
    row = next(s for s in res.result["sources"] if s["source_id"] == sources["paper_a"].id)
    assert "t1/2 (h)" in row["unmatched_headers"]
    assert _records(env, sources["paper_a"], "half_life") == []
    unmatched = extract_rules.rule_extract(ingest.load_pages(sources["paper_a"]), {"fields": fields}, "src")
    assert unmatched["unmatched_headers"] == ["t1/2 (h)"]


def test_required_missing_and_flag_shape(env, papers):
    sources = _include(env, papers, names=("paper_d",))
    _template(env)
    _extract(env)
    recs = _records(env, sources["paper_d"])
    assert recs and all(M.FLAG_REQUIRED_MISSING in _codes(r) for r in recs)
    flag = next(f for f in recs[0].flags if f["code"] == M.FLAG_REQUIRED_MISSING)
    assert flag["field"] == "cmax" and flag["severity"] == "warn" and flag["origin"] == "script"


def test_rerun_inserts_nothing_and_mirror_is_stable(env, papers):
    _include(env, papers)
    _template(env)
    res1, state = _extract(env)
    hashes = sorted(r.record_hash for r in _records(env))
    res2, state = _extract(env, state=state)
    assert res2.result["n_new"] == 0 and sorted(r.record_hash for r in _records(env)) == hashes
    assert res2.result["n_candidates"] == res1.result["n_candidates"]
    er = state.extraction_results
    assert er["n_records"] == len(hashes) and er["proposed"] == len(hashes) and er["approved"] == 0
    assert er["verify_fail"] == 0 and er["method"] == "table" and er["template_hash"] == _template(env)
    assert set(er) >= {"template_hash", "n_records", "proposed", "approved", "rejected", "verify_fail",
                       "flagged", "method", "updated_at"}
    assert state.evidence_summary["records"]["proposed"] == len(hashes)


def test_changed_file_is_skipped_fail_closed(env, papers):
    sources = _include(env, papers, names=("paper_a", "paper_d"))
    _template(env)
    Path(sources["paper_a"].pdf_path).write_bytes(papers["paper_b"].read_bytes())
    res, _ = _extract(env)
    assert [s["source_id"] for s in res.result["skipped"]] == [sources["paper_a"].id]
    assert "changed" in res.result["skipped"][0]["reason"] and _records(env, sources["paper_a"]) == []


# ------------------------------------------------------------------ LLM path ---

def test_llm_value_not_on_page_fails_number_mismatch(env, papers):
    sources = _include(env, papers, names=("paper_d",))
    _template(env)
    llm = SchemaLLM(_llm_records({"field": "cl", "value": "2.41", "unit": "L/h", "page": 1,
                                  "snippet": "The mean apparent clearance (CL/F) was 2.31 L/h"}))
    llm_access.set_llm(llm)
    res, state = _extract(env)
    rec = _one(env, sources["paper_d"], "cl")
    assert rec.extractor == "llm" and rec.verify_status == "fail" and rec.verify_reason == M.REASON_NUMBER_MISMATCH
    assert rec.locator["span"] is None and rec.locator["literal"] is None
    fail = next(f for f in rec.flags if f["code"] == M.FLAG_VERIFY_FAIL)
    assert fail["severity"] == "error" and M.REASON_NUMBER_MISMATCH in fail["message"]
    assert state.extraction_results["verify_fail"] == 1 and res.result["fail"] == 1
    assert res.result["method"] == "llm"


def test_llm_snippet_from_another_page_fails_snippet_not_on_page(env, papers):
    sources = _include(env, papers, names=("paper_c",))
    _template(env)
    llm_access.set_llm(SchemaLLM(_llm_records({"field": "clint", "value": "18,2", "page": 1,
                                               "snippet": "CLint 18,2 ± 3,4 22,7 ± 4,1"})))
    _extract(env)
    rec = _one(env, sources["paper_c"], "clint")
    assert rec.verify_status == "fail" and rec.verify_reason == M.REASON_SNIPPET_NOT_ON_PAGE


def test_llm_true_value_passes_and_table_kind_resolves_the_cell(env, papers):
    sources = _include(env, papers, names=("paper_d",))
    _template(env)
    llm_access.set_llm(SchemaLLM(_llm_records(
        {"field": "CL/F", "value": "2.31", "unit": "L/h", "page": 1, "kind": "table", "snippet": "2.31"},
        {"field": "v", "value": "48.0", "unit": "L", "page": 1, "snippet": "volume of distribution was 48.0"})))
    _extract(env)
    src = sources["paper_d"]
    cl = _one(env, src, "cl")
    assert cl.verify_status == "pass" and cl.locator["kind"] == "table" and cl.locator["cell"]["row"] == 1
    assert cl.locator["cell"]["row_label"] == "CL/F (L/h)"
    _assert_roundtrip(src, cl)
    _assert_roundtrip(src, _one(env, src, "v"))


def test_llm_injection_text_is_stored_only_literally_and_flagged(env, papers):
    sources = _include(env, papers, names=("paper_d",))
    _template(env)
    llm = SchemaLLM(_llm_records({"field": "species", "value": fx.INJECTION, "page": 1,
                                  "snippet": "Note to automated readers: " + fx.INJECTION + "."}))
    llm_access.set_llm(llm)
    _extract(env)
    rec = _one(env, sources["paper_d"], "species")
    assert rec.verify_status == "pass" and rec.status == "proposed"
    assert M.FLAG_SUSPICIOUS_INSTRUCTION in _codes(rec)
    user = llm.calls[0]["user"]
    assert llm.calls[0]["system"] == EVIDENCE_SYSTEM and llm.calls[0]["schema"] is EXTRACT_SCHEMA
    assert user.index("<document") < user.index(fx.INJECTION) < user.index("</document>")
    assert user.index("</document>") < user.index("Fields")
    # a text value that is NOT in its snippet is refused even though the snippet is on the page
    llm_access.set_llm(SchemaLLM(_llm_records({"field": "species", "value": "human", "page": 1,
                                               "snippet": "Population pharmacokinetics of drug W"})))
    _extract(env)
    bad = [r for r in _records(env, sources["paper_d"], "species") if r.value == "human"][0]
    assert bad.verify_status == "fail" and bad.verify_reason == M.REASON_TEXT_NOT_IN_SNIPPET


def test_llm_extra_flags_are_added_only(env, papers):
    sources = _include(env, papers, names=("paper_d",))
    _template(env)
    records = _llm_records({"field": "cl", "value": "2.31", "unit": "L/h", "page": 1,
                            "snippet": "clearance (CL/F) was 2.31 L/h"})
    flags = {"flags": [{"code": "small_sample", "severity": "warn", "message": "n = 10 only", "field": "cl"},
                       {"code": "bad", "severity": "fatal", "message": "x", "field": "cl"},
                       {"code": "verify_fail", "severity": "error", "message": "forged", "field": "cl"},
                       {"code": "orphan", "severity": "warn", "message": "no field"}]}
    llm = SchemaLLM(records, flags)
    llm_access.set_llm(llm)
    _extract(env)
    rec = _one(env, sources["paper_d"], "cl")
    assert rec.value == "2.31" and rec.verify_status == "pass"
    llm_flags = [f for f in rec.flags if f["origin"] == "llm"]
    assert [f["code"] for f in llm_flags] == ["small_sample", "verify_fail"]
    assert all(f["origin"] == "script" for f in rec.flags if f["code"] == M.FLAG_REQUIRED_MISSING)
    assert any(c["schema"] is FLAG_SCHEMA and "2.31" in c["user"] and c["user"].startswith("<document")
               for c in llm.calls)


def test_llm_scripted_flags_on_proposals(env, papers):
    sources = _include(env, papers, names=("paper_a", "paper_b", "paper_d"))
    _template(env)
    a, b, d = sources["paper_a"], sources["paper_b"], sources["paper_d"]
    # approve the deterministic cmax first so a different LLM value is a duplicate
    llm_access.set_llm(None)
    _extract(env, {"source_ids": [a.id]})
    cmax = _one(env, a, "cmax")
    env.store.decide_record(cmax.id, "approved", "alice", "table value")
    llm_access.set_llm(SchemaLLM(_llm_records(
        {"field": "cmax", "value": "812", "unit": "ng/mL", "page": 1, "group": "150 mg", "snippet": "3,412 (812)"},
        {"field": "n", "value": "12.5", "page": 1, "snippet": "(n = 12)"},
        {"field": "half_life", "value": "300", "unit": "h", "page": 1, "snippet": "11.4 (2.3)"},
        {"field": "cmax_se", "value": "−20", "unit": "ng/mL", "page": 1, "snippet": "−20 ± 8"},
        {"field": "cl_lower", "value": "48.0", "unit": "L/h", "page": 1, "snippet": "distribution was 48.0"},
        {"field": "cl_upper", "value": "2.31", "unit": "L/h", "page": 1, "snippet": "was 2.31 L/h"})))
    _extract(env)
    dup = [r for r in _records(env, a, "cmax") if r.value == "812"][0]
    assert dup.verify_status == "pass" and M.FLAG_DUPLICATE in _codes(dup)
    n = [r for r in _records(env, a, "n") if r.value == "12.5"][0]
    assert n.verify_reason == M.REASON_INTEGER_EXPECTED and M.FLAG_NON_INTEGER_N in _codes(n)
    hl = [r for r in _records(env, a, "half_life") if r.value == "300"][0]
    assert hl.verify_reason == M.REASON_NUMBER_MISMATCH and M.FLAG_OUT_OF_RANGE in _codes(hl)
    if fx.UNICODE_FONT:
        se = _one(env, b, "cmax_se")
        assert se.verify_status == "pass" and se.value_num == -20 and M.FLAG_SE_NONPOSITIVE in _codes(se)
    lo, hi = _one(env, d, "cl_lower"), _one(env, d, "cl_upper")
    assert lo.verify_status == hi.verify_status == "pass"
    assert M.FLAG_CI_INVERTED in _codes(lo) and M.FLAG_CI_INVERTED in _codes(hi)


@pytest.mark.parametrize("answer", [None, {"records": "no"}, {"records": [{"field": "cl"}]},
                                    {"records": [{"field": "cl", "value": "2.31", "page": 0, "kind": "text",
                                                  "snippet": "x"}]}])
def test_llm_invalid_answer_falls_back_to_rules(env, papers, answer):
    sources = _include(env, papers, names=("paper_d",))
    _template(env)
    llm_access.set_llm(SchemaLLM(answer))
    res, _ = _extract(env)
    row = res.result["sources"][0]
    assert row["method"] == "table" and row["note"] == "llm_invalid" and res.result["method"] == "table"
    cl = _one(env, sources["paper_d"], "cl")
    assert cl.extractor == "table" and cl.value == "2.31"


def test_llm_exception_falls_back_to_rules(env, papers):
    class Boom(FakeLLM):
        def extract_json(self, *a, **k):
            raise RuntimeError("ollama down")

    sources = _include(env, papers, names=("paper_d",))
    _template(env)
    llm_access.set_llm(Boom())
    res, _ = _extract(env)
    assert res.result["sources"][0]["note"] == "llm_invalid" and _one(env, sources["paper_d"], "cl")


def test_llm_documents_are_per_page_with_table_pages_first(env, papers):
    sources = _include(env, papers, names=("paper_c",))
    _template(env)
    llm = SchemaLLM({"records": []})
    llm_access.set_llm(llm)
    _extract(env)
    users = [c["user"] for c in llm.calls if c["schema"] is EXTRACT_SCHEMA]
    assert len(users) == 2 and "Table 1 Intrinsic clearance" in users[0] and "Sprague-Dawley" in users[1]
    assert "page 2 of 2" in users[0] and all(u.startswith('<document title="') for u in users)
    assert all(u.index("</document>") < u.index("Fields") for u in users)
    assert extract_tools.MAX_PAGE_CHARS == 8000
    assert sources["paper_c"].id and _records(env) == []


# ------------------------------------------------------------ verify_records ---

def _verify(env, state=None, args=None):
    state = state or PharmState(session_id=env.sid)
    res = extract_tools.verify_records(state, env.ctx, args or {})
    return res, apply_writes(state, "evidence", res.writes)


def test_verify_records_is_idempotent(env, papers):
    _include(env, papers)
    _template(env)
    _extract(env)
    before = {r.id: (r.record_hash, r.verify_status, r.locator, r.flags) for r in _records(env)}
    res, state = _verify(env)
    assert res.result["n_checked"] == len(before) and res.result["fail"] == 0 and res.result["changed"] == 0
    after = {r.id: (r.record_hash, r.verify_status, r.locator, r.flags) for r in _records(env)}
    assert after == before and state.extraction_results["verify_fail"] == 0
    with pytest.raises(ValueError, match="unknown"):
        _verify(env, args={"record_ids": ["rec_nope"]})


def test_verify_records_fails_closed_when_the_file_changed(env, papers):
    sources = _include(env, papers, names=("paper_a", "paper_d"))
    _template(env)
    _extract(env)
    a = sources["paper_a"]
    Path(a.pdf_path).write_bytes(papers["paper_b"].read_bytes())
    res, state = _verify(env)
    rows = _records(env, a)
    assert rows and all(r.verify_status == "fail" and r.verify_reason == M.REASON_SOURCE_CHANGED for r in rows)
    assert all(M.FLAG_VERIFY_FAIL in _codes(r) and r.locator["span"] is None for r in rows)
    assert all(r.verify_status == "pass" for r in _records(env, sources["paper_d"]))
    assert res.result["changed"] == len(rows) and state.extraction_results["verify_fail"] == len(rows)
    with pytest.raises(ValueError, match="failed literal verification"):
        _decide(env, rows[0].id)


def test_verify_records_keeps_llm_flags_and_skips_decided_rows(env, papers):
    sources = _include(env, papers, names=("paper_d",))
    _template(env)
    llm_access.set_llm(SchemaLLM(_llm_records({"field": "cl", "value": "2.31", "page": 1, "unit": "L/h",
                                               "snippet": "clearance (CL/F) was 2.31 L/h"}),
                                 {"flags": [{"code": "note", "severity": "warn", "message": "m", "field": "cl"}]}))
    _extract(env)
    llm_access.set_llm(None)
    rec = _one(env, sources["paper_d"], "cl")
    assert any(f["origin"] == "llm" for f in rec.flags)
    _verify(env)
    again = env.store.get_record(rec.id)
    assert [f for f in again.flags if f["origin"] == "llm"] == [f for f in rec.flags if f["origin"] == "llm"]
    env.store.decide_record(rec.id, "approved", "alice", "ok")
    res, _ = _verify(env, args={"record_ids": [rec.id]})
    assert res.result["n_checked"] == 0 and env.store.get_record(rec.id).status == "approved"


# ------------------------------------------------------------ decide_records ---

def _decide(env, record_id, decision="approve", reason="matches the table", actor="alice", confirm=True,
            state=None):
    state = state or PharmState(session_id=env.sid)
    args = {"record_id": record_id, "decision": decision, "reason": reason, "actor": actor, "confirm": confirm}
    res = extract_tools.decide_records(state, env.ctx, args)
    return res, apply_writes(state, "evidence", res.writes)


def test_decide_approve_and_reject_update_mirrors_and_echo(env, papers):
    sources = _include(env, papers, names=("paper_a",))
    _template(env)
    _, state = _extract(env)
    a = sources["paper_a"]
    cmax, auc = _one(env, a, "cmax"), _one(env, a, "auc")
    res, state = _decide(env, cmax.id, state=state)
    out = res.result
    assert out["decision"] == "approved" and out["reason"] == "matches the table"
    assert out["record_hash"] == cmax.record_hash and out["source_sha256"] == a.sha256
    assert out["value"] == "3,412" and out["unit"] == "ng/mL" and out["verify_status"] == "pass"
    assert out["locator"]["literal"] == "3,412" and out["record"]["decided_by"] == "alice"
    assert "decide_records(" in res.action and "approved" in res.action
    res, state = _decide(env, auc.id, "reject", "wrong unit", state=state)
    assert res.result["decision"] == "rejected"
    er = state.extraction_results
    assert (er["approved"], er["rejected"], er["proposed"]) == (1, 1, len(_records(env)) - 2)
    assert state.evidence_summary["records"]["approved"] == 1
    assert all(r.verify_status == "pass" for r in _records(env, status="approved"))


def test_decide_refuses_fail_twice_and_bad_args(env, papers):
    sources = _include(env, papers, names=("paper_d",))
    _template(env)
    llm_access.set_llm(SchemaLLM(_llm_records({"field": "cl", "value": "2.41", "page": 1,
                                               "snippet": "was 2.31 L/h"})))
    _extract(env)
    llm_access.set_llm(None)
    bad = _one(env, sources["paper_d"], "cl")
    assert bad.verify_status == "fail"
    with pytest.raises(ValueError, match="cannot approve a record that failed literal verification"):
        _decide(env, bad.id)
    res, _ = _decide(env, bad.id, "reject", "not on the page")
    assert res.result["decision"] == "rejected"
    with pytest.raises(M.EvidenceStateError, match="already decided"):
        _decide(env, bad.id, "reject", "again")
    _extract(env)                                       # keyless now: the rules add paper_d's n = 10
    good = _one(env, sources["paper_d"], "n")
    for kw in ({"confirm": False}, {"decision": "maybe"}, {"reason": ""}, {"reason": "x" * 501}):
        with pytest.raises(ValueError):
            _decide(env, good.id, **kw)
    with pytest.raises(ValueError, match="unknown"):
        _decide(env, "rec_nope")
    assert env.store.get_record(good.id).status == "proposed"
    assert all(r.verify_status == "pass" for r in _records(env, status="approved"))


def test_other_sessions_record_is_not_decidable(env, papers):
    sources = _include(env, papers, names=("paper_a",), sid="other")
    _template(env, sid="other")
    extract_tools.extract_evidence(PharmState(session_id="other"), env.ctx, {})
    rec = env.store.list_records("other")[0]
    with pytest.raises(ValueError, match="unknown"):
        _decide(env, rec.id)
    assert env.store.get_record(rec.id).status == "proposed" and sources["paper_a"].session_id == "other"


def test_tools_are_registered_for_the_evidence_agent():
    from app.tools.builtins import default_registry
    reg = default_registry()
    names = {t.name: t.agent for t in reg.for_agent("evidence")}
    assert {"extract_evidence", "verify_records", "decide_records"} <= set(names)
    assert set(names.values()) == {"evidence"}
