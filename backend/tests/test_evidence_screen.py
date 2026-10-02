"""Screening tool: keyword path (keyless), LLM path with quote verification, prompt-injection
proof, decide_screening. Tools are called directly with a ToolContext on a tmp data_dir and a
':memory:' ledger; no HTTP.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import settings
from app.core.pharmstate import PharmState, apply_writes
from app.evidence import ingest, llm_access, screen_rules, screen_tools
from app.evidence import store as evstore
from app.evidence.config_tools import validate_criteria
from app.evidence.models import EvidenceStateError
from app.evidence.prompts import EVIDENCE_SYSTEM
from app.tools.base import ToolContext
from tests import evidence_fixtures as fx
from tests.test_evidence_llm import FakeLLM

CRITERIA = [
    {"id": "C1", "text": "Human adult pharmacokinetic study", "kind": "inclusion", "required": True,
     "keywords": ["human", "adult", "patients", "volunteers"]},
    {"id": "C2", "text": "Reports Cmax, AUC or clearance", "kind": "inclusion", "required": True,
     "keywords": ["Cmax", "AUC", "clearance"]},
    {"id": "E1", "text": "In-vitro or animal study", "kind": "exclusion", "required": False,
     "keywords": ["in vitro", "rat", "mouse", "animal", "hepatocytes"]},
]
SID = "sess_screen"


@pytest.fixture(scope="module")
def papers(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("papers")
    paths = fx.build_all(d)
    paths["paper_d_clean"] = fx.build_pdf(d / "clean", "paper_d", injection=False)
    return paths


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    st = evstore.configure(":memory:")
    llm_access.set_llm(None)
    yield SimpleNamespace(store=st, ctx=ToolContext(data_dir=str(tmp_path)), sid=SID, dir=tmp_path)
    llm_access.set_llm(None)


def _ingest(env, path: Path, name: str, sid: str = SID):
    return ingest.ingest_pdf(path.read_bytes(), name, sid, "alice", data_dir=env.dir, store=env.store)


def _ingest_all(env, papers, names=("paper_a", "paper_b", "paper_c", "paper_d")) -> dict:
    return {n: _ingest(env, papers[n], f"{n}.pdf") for n in names}


def _save_criteria(env, criteria=CRITERIA, sid: str = SID) -> str:
    return env.store.save_config(sid, "criteria", validate_criteria(criteria), "alice")


def _screen(env, args: dict | None = None, state: PharmState | None = None):
    state = state or PharmState(session_id=env.sid)
    res = screen_tools.screen_publications(state, env.ctx, args or {})
    return res, apply_writes(state, "evidence", res.writes)


def _by_source(res, sources: dict) -> dict:
    rows = {s["source_id"]: s for s in res.result["screens"]}
    return {name: rows[src.id] for name, src in sources.items()}


def _page_text(src, n: int = 1) -> str:
    return ingest.load_pages(src)["pages"][n - 1]["text"]


# ------------------------------------------------------------- preconditions ---

def test_screen_requires_saved_criteria(env, papers):
    _ingest(env, papers["paper_a"], "a.pdf")
    with pytest.raises(ValueError, match="criteria"):
        _screen(env)


def test_screen_requires_parsed_sources(env, papers):
    _save_criteria(env)
    with pytest.raises(ValueError, match="publication"):
        _screen(env)


def test_unknown_source_id_is_an_error(env, papers):
    _save_criteria(env)
    _ingest(env, papers["paper_a"], "a.pdf")
    with pytest.raises(ValueError, match="unknown"):
        _screen(env, {"source_ids": ["src_nope"]})


def test_other_sessions_source_is_unknown(env, papers):
    _save_criteria(env)
    other = _ingest(env, papers["paper_a"], "a.pdf", sid="sess_other")
    _ingest(env, papers["paper_b"], "b.pdf")
    with pytest.raises(ValueError, match="unknown"):
        _screen(env, {"source_ids": [other.id]})


# ------------------------------------------------------------ keyword path ---

def test_keyword_verdicts_over_the_fixture_papers(env, papers):
    h = _save_criteria(env)
    sources = _ingest_all(env, papers)
    res, state = _screen(env)
    rows = _by_source(res, sources)
    for name in ("paper_a", "paper_b", "paper_d"):
        assert rows[name]["verdict"] == "include", (name, rows[name]["reasoning"])
        assert rows[name]["score"] >= 0.75
    assert rows["paper_c"]["verdict"] == "exclude"
    assert "E1" in rows["paper_c"]["reasoning"] and "page 1" in rows["paper_c"]["reasoning"]
    for row in rows.values():
        assert row["status"] == "proposed" and row["method"] == "keyword"
        assert row["criteria_hash"] == h and row["id"].startswith("scr_")
        assert {c["criterion_id"] for c in row["criteria"]} == {"C1", "C2", "E1"}
    # explanation names matched keywords and their pages
    a = rows["paper_a"]["reasoning"].lower()
    assert "human" in a and "cmax" in a and "page 1" in a
    # mirrors
    sr = state.screening_results
    assert sr["criteria_hash"] == h and sr["n_screened"] == 4 and sr["method"] == "keyword"
    assert sr["proposed"] == 4 and sr["approved"] == 0 and sr["rejected"] == 0
    assert sr["verdicts"] == {"include": 3, "exclude": 1, "unsure": 0}
    assert state.evidence_summary["screens"]["proposed"] == 4 and state.evidence_summary["criteria_hash"] == h
    assert res.action.startswith("screened 4 sources")


def test_evidence_locators_point_at_the_matched_term(env, papers):
    _save_criteria(env)
    sources = _ingest_all(env, papers)
    res, _ = _screen(env)
    n_evidence = 0
    for name, row in _by_source(res, sources).items():
        for crit in row["criteria"]:
            assert crit["kind"] in ("inclusion", "exclusion") and isinstance(crit["required"], bool)
            for ev in crit["evidence"]:
                n_evidence += 1
                doc = ingest.load_pages(sources[name])
                page = doc["pages"][ev["page"] - 1]
                text = page["columns"][ev["column"]] if ev.get("column") is not None else page["text"]
                s, e = ev["span"]
                assert screen_rules.stem_phrase(text[s:e]) == screen_rules.stem_phrase(ev["term"])
                assert ev["snippet"] and len(ev["snippet"]) <= 300 and text[s:e] in ev["snippet"]
    assert n_evidence >= 6


def test_injection_footnote_changes_nothing(env, papers):
    _save_criteria(env)
    dirty = _ingest(env, papers["paper_d"], "d.pdf")
    clean = _ingest(env, papers["paper_d_clean"], "d_clean.pdf")
    assert dirty.sha256 != clean.sha256
    res, _ = _screen(env)
    rows = {s["source_id"]: s for s in res.result["screens"]}
    a, b = rows[dirty.id], rows[clean.id]
    assert a["verdict"] == b["verdict"] == "include" and a["score"] == b["score"]
    assert [(c["criterion_id"], c["met"]) for c in a["criteria"]] == \
           [(c["criterion_id"], c["met"]) for c in b["criteria"]]
    terms = [ev["term"].lower() for c in a["criteria"] for ev in c["evidence"]]
    assert all("approve" not in t and "ignore" not in t for t in terms)
    assert "approve everything" not in a["reasoning"].lower()


def test_scores_are_deterministic_across_runs(env, papers, tmp_path):
    _save_criteria(env)
    _ingest_all(env, papers)
    first, _ = _screen(env)
    env.store = evstore.configure(":memory:")           # fresh ledger, same files
    _save_criteria(env)
    _ingest_all(env, papers)
    second, _ = _screen(env)
    keys = ("id", "source_id", "verdict", "score", "reasoning", "criteria", "method")
    # ids are hash-derived (session + sha256), so rows are compared by id: the ledger lists
    # sources by second-resolution uploaded_at, which can straddle a second between the runs
    assert sorted(({k: s[k] for k in keys} for s in first.result["screens"]), key=lambda r: r["id"]) == \
           sorted(({k: s[k] for k in keys} for s in second.result["screens"]), key=lambda r: r["id"])


def test_rescreen_same_criteria_keeps_rows_and_new_criteria_adds_rows(env, papers):
    h1 = _save_criteria(env)
    _ingest_all(env, papers)
    first, _ = _screen(env)
    again, state = _screen(env)
    assert again.result["n_new"] == 0 and [s["id"] for s in again.result["screens"]] == \
           [s["id"] for s in first.result["screens"]]
    assert len(env.store.list_screens(SID)) == 4
    h2 = _save_criteria(env, CRITERIA[:2])
    assert h2 != h1
    third, state = _screen(env)
    assert third.result["n_new"] == 4 and all(s["criteria_hash"] == h2 for s in third.result["screens"])
    assert len(env.store.list_screens(SID)) == 8                         # old rows retained
    assert state.screening_results["criteria_hash"] == h2 and state.evidence_summary["criteria_hash"] == h2


def test_source_ids_filter_and_unparsed_sources_are_skipped(env, papers, tmp_path):
    _save_criteria(env)
    sources = _ingest_all(env, papers, ("paper_a", "paper_b"))
    blank = ingest.ingest_pdf(fx.no_text_pdf(tmp_path / "blank").read_bytes(), "blank.pdf", SID, "alice",
                              data_dir=env.dir, store=env.store)
    res, _ = _screen(env, {"source_ids": [sources["paper_a"].id, blank.id]})
    assert [s["source_id"] for s in res.result["screens"]] == [sources["paper_a"].id]
    assert res.result["skipped"] == [{"source_id": blank.id, "reason": "parse_status no_text"}]


def test_changed_file_on_disk_is_skipped_fail_closed(env, papers):
    _save_criteria(env)
    src = _ingest(env, papers["paper_a"], "a.pdf")
    Path(src.pdf_path).write_bytes(papers["paper_b"].read_bytes())
    with pytest.raises(ValueError, match="publication"):
        _screen(env)                                     # nothing screenable: every source skipped
    env.store = evstore.configure(":memory:")
    _save_criteria(env)
    _ingest(env, papers["paper_b"], "b.pdf")
    src2 = env.store.add_source(session_id=SID, sha256=src.sha256, filename="a.pdf", pdf_path=src.pdf_path,
                                parse_path=src.parse_path, pages=1, title=None, metadata={}, text_chars=10,
                                parse_status="parsed", decimal_style=".", licence_note=None,
                                uploaded_by="alice")
    res, _ = _screen(env)
    assert res.result["skipped"] == [{"source_id": src2.id, "reason": "source file changed"}]
    assert len(res.result["screens"]) == 1


# ---------------------------------------------------------- keyword rules ---

def test_keyword_rules_derive_terms_from_text_when_no_keywords():
    crit = {"id": "X", "text": "Reports the terminal half-lives in rats", "kind": "inclusion",
            "required": True, "keywords": []}
    terms = screen_rules.criterion_terms(crit)
    assert "reports" not in terms and "the" not in terms and "in" not in terms
    assert {"terminal", "half-lives", "rats"} <= set(terms)


def test_keyword_rules_stem_and_phrase_matching():
    assert screen_rules.stem_phrase("Rats") == screen_rules.stem_phrase("rat")
    assert screen_rules.stem_phrase("rate") != screen_rules.stem_phrase("rat")
    assert screen_rules.stem_phrase("in\nvitro") == screen_rules.stem_phrase("In Vitro")
    assert screen_rules.stem_phrase("Studies") == screen_rules.stem_phrase("study")
    pages = [{"n": 1, "text": "Mouse hepatocytes were\nstudied in vitro.", "columns": None}]
    out = screen_rules.keyword_screen(CRITERIA, pages)
    e1 = next(c for c in out["criteria"] if c["criterion_id"] == "E1")
    assert e1["met"] is True and {ev["term"] for ev in e1["evidence"]} == {"in vitro", "mouse", "hepatocytes"}
    assert out["verdict"] == "exclude" and out["score"] == 0.0


def test_keyword_rules_unsure_when_a_required_criterion_is_unmet():
    pages = [{"n": 1, "text": "Adult patients received the drug.", "columns": None}]
    out = screen_rules.keyword_screen(CRITERIA, pages)
    assert out["verdict"] == "unsure" and 0 < out["score"] < 0.75
    assert "C2" in out["reasoning"] and "not met" in out["reasoning"]
    nothing = screen_rules.keyword_screen(CRITERIA, [{"n": 1, "text": "Lorem ipsum.", "columns": None}])
    assert nothing["verdict"] == "exclude" and nothing["score"] == 0.0


# ---------------------------------------------------------------- LLM path ---

def _llm_answer(**overrides) -> dict:
    base = {"verdict": "include", "score": 0.9, "reasoning": "healthy human PK study reporting Cmax",
            "criteria": [
                {"criterion_id": "C1", "met": True, "quote": "healthy human volunteers", "page": 1},
                {"criterion_id": "C2", "met": True, "quote": "Cmax was 3,412 ng/mL", "page": 1},
                {"criterion_id": "E1", "met": False, "quote": None, "page": None}]}
    return {**base, **overrides}


def test_llm_include_is_still_a_proposal_and_the_prompt_wraps_the_document(env, papers):
    _save_criteria(env)
    src = _ingest(env, papers["paper_a"], "a.pdf")
    llm = FakeLLM(_llm_answer())
    llm_access.set_llm(llm)
    res, state = _screen(env)
    row = res.result["screens"][0]
    assert row["status"] == "proposed" and row["method"] == "llm" and row["verdict"] == "include"
    assert row["score"] == 0.9 and state.screening_results["method"] == "llm"
    c1 = next(c for c in row["criteria"] if c["criterion_id"] == "C1")
    assert c1["met"] is True and c1["evidence"][0]["page"] == 1
    s, e = c1["evidence"][0]["span"]
    assert _page_text(src)[s:e] == "healthy human volunteers"
    call = llm.calls[0]
    assert call["system"] == EVIDENCE_SYSTEM
    assert call["user"].startswith("<document ") and "</document>" in call["user"]
    assert "Twelve healthy adult volunteers" in call["user"]
    assert call["user"].index("</document>") < call["user"].index("C1")      # task after the block
    assert "Pharmacokinetics of drug X" in call["user"]


def test_llm_quote_not_on_page_fails_the_criterion_and_downgrades_to_unsure(env, papers):
    _save_criteria(env)
    _ingest(env, papers["paper_a"], "a.pdf")
    bad = _llm_answer()
    bad["criteria"][0] = {"criterion_id": "C1", "met": True, "quote": "healthy martian volunteers", "page": 1}
    llm_access.set_llm(FakeLLM(bad))
    res, _ = _screen(env)
    row = res.result["screens"][0]
    c1 = next(c for c in row["criteria"] if c["criterion_id"] == "C1")
    assert c1["met"] is False and c1["note"] == "quote not found" and c1["evidence"] == []
    assert row["verdict"] == "unsure" and row["method"] == "llm"
    assert "quote not found" in row["reasoning"]


def test_llm_quote_not_found_on_an_optional_criterion_keeps_the_verdict(env, papers):
    _save_criteria(env)
    _ingest(env, papers["paper_a"], "a.pdf")
    ans = _llm_answer()
    ans["criteria"][2] = {"criterion_id": "E1", "met": True, "quote": "rat hepatocytes", "page": 1}
    llm_access.set_llm(FakeLLM(ans))
    row = _screen(env)[0].result["screens"][0]
    e1 = next(c for c in row["criteria"] if c["criterion_id"] == "E1")
    assert e1["met"] is False and e1["note"] == "quote not found" and row["verdict"] == "include"


def test_llm_page_out_of_range_counts_as_quote_not_found(env, papers):
    _save_criteria(env)
    _ingest(env, papers["paper_a"], "a.pdf")
    ans = _llm_answer()
    ans["criteria"][0] = {"criterion_id": "C1", "met": True, "quote": "healthy human volunteers", "page": 7}
    llm_access.set_llm(FakeLLM(ans))
    row = _screen(env)[0].result["screens"][0]
    assert row["verdict"] == "unsure" and row["criteria"][0]["note"] == "quote not found"


def test_llm_met_claim_without_a_quote_is_unverified(env, papers):
    _save_criteria(env)
    _ingest(env, papers["paper_a"], "a.pdf")
    ans = _llm_answer()
    ans["criteria"][0] = {"criterion_id": "C1", "met": True}
    llm_access.set_llm(FakeLLM(ans))
    row = _screen(env)[0].result["screens"][0]
    assert row["criteria"][0]["met"] is None and row["criteria"][0]["note"] == "no quote"
    assert row["verdict"] == "unsure"


def test_llm_missing_or_unknown_criteria_are_normalised(env, papers):
    _save_criteria(env)
    _ingest(env, papers["paper_a"], "a.pdf")
    ans = _llm_answer(criteria=[{"criterion_id": "ZZ", "met": True, "quote": "Cmax", "page": 1},
                               {"criterion_id": "C2", "met": True, "quote": "Cmax was 3,412 ng/mL", "page": 1}])
    llm_access.set_llm(FakeLLM(ans))
    row = _screen(env)[0].result["screens"][0]
    ids = [c["criterion_id"] for c in row["criteria"]]
    assert ids == ["C1", "C2", "E1"]                                     # config order, ZZ dropped
    assert row["criteria"][0]["met"] is None and row["criteria"][0]["note"] == "not assessed"
    assert row["verdict"] == "unsure"                                     # C1 required, unassessed


@pytest.mark.parametrize("answer", [None, {"verdict": "approved", "score": 1, "reasoning": "", "criteria": []},
                                    {"verdict": "include"}, "include"])
def test_llm_invalid_answer_falls_back_to_keywords(env, papers, answer):
    _save_criteria(env)
    _ingest(env, papers["paper_a"], "a.pdf")
    llm_access.set_llm(FakeLLM(answer))
    res, state = _screen(env)
    row = res.result["screens"][0]
    assert row["method"] == "keyword" and row["verdict"] == "include"
    assert all(c["note"] == "llm_invalid" for c in row["criteria"])
    assert "llm_invalid" in row["reasoning"] and state.screening_results["method"] == "keyword"


def test_llm_exception_falls_back_to_keywords(env, papers):
    _save_criteria(env)
    _ingest(env, papers["paper_a"], "a.pdf")

    class Boom(FakeLLM):
        def extract_json(self, *a, **k):
            raise RuntimeError("ollama down")
    llm_access.set_llm(Boom())
    row = _screen(env)[0].result["screens"][0]
    assert row["method"] == "keyword" and row["criteria"][0]["note"] == "llm_invalid"


def test_llm_document_is_title_plus_first_two_pages_capped(env, papers):
    _save_criteria(env)
    src = _ingest(env, papers["paper_c"], "c.pdf")
    llm = FakeLLM(None)
    llm_access.set_llm(llm)
    _screen(env)
    user = llm.calls[0]["user"]
    assert user.startswith(f'<document title="{src.title}"') and "Discussion. Intrinsic clearance" in user
    assert screen_tools.MAX_DOC_CHARS == 6000


# ----------------------------------------------------------------- decide ---

def _decide(env, state, screen_id, decision="approve", reason="looks right", actor="alice", confirm=True):
    args = {"screen_id": screen_id, "decision": decision, "reason": reason, "confirm": confirm}
    res = screen_tools.decide_screening(state, replace(env.ctx, actor=actor), args)   # as the registry does
    return res, apply_writes(state, "evidence", res.writes)


def test_decide_approve_and_reject_update_mirrors(env, papers):
    _save_criteria(env)
    sources = _ingest_all(env, papers)
    res, state = _screen(env)
    rows = _by_source(res, sources)
    out, state = _decide(env, state, rows["paper_a"]["id"])
    assert out.result["screen"]["status"] == "approved" and out.result["screen"]["decided_by"] == "alice"
    assert out.result["screen"]["decision_reason"] == "looks right"
    assert out.action == f"decide_screening({rows['paper_a']['id']}, include, approved)"
    assert out.result["screen"]["decided_verdict"] == "include"
    out, state = _decide(env, state, rows["paper_c"]["id"], decision="reject", reason="in vitro")
    assert out.result["screen"]["status"] == "rejected"
    sr = state.screening_results
    assert (sr["proposed"], sr["approved"], sr["rejected"]) == (2, 1, 1)
    db = {s.id: s for s in env.store.list_screens(SID)}
    assert db[rows["paper_a"]["id"]].decided_by == "alice" and db[rows["paper_a"]["id"]].decided_at


def test_decide_twice_is_a_state_error(env, papers):
    _save_criteria(env)
    _ingest_all(env, papers, ("paper_a",))
    res, state = _screen(env)
    sid = res.result["screens"][0]["id"]
    _, state = _decide(env, state, sid)
    with pytest.raises(EvidenceStateError, match="already decided"):
        _decide(env, state, sid, decision="reject")


@pytest.mark.parametrize("bad", [
    {"reason": ""}, {"reason": None}, {"reason": "x" * 501}, {"decision": "maybe"}, {"decision": "approved"},
    {"confirm": False}, {"screen_id": "scr_nope"}, {"screen_id": ""},
])
def test_decide_validation(env, papers, bad):
    _save_criteria(env)
    _ingest_all(env, papers, ("paper_a",))
    res, state = _screen(env)
    args = {"screen_id": res.result["screens"][0]["id"], "decision": "approve", "reason": "ok",
            "actor": "alice", "confirm": True, **bad}
    with pytest.raises(ValueError):
        screen_tools.decide_screening(state, env.ctx, args)
    assert env.store.list_screens(SID)[0].status == "proposed"


def test_decide_refuses_another_sessions_screen(env, papers):
    _save_criteria(env)
    _save_criteria(env, sid="sess_other")
    _ingest(env, papers["paper_a"], "a.pdf", sid="sess_other")
    other_state = PharmState(session_id="sess_other")
    res = screen_tools.screen_publications(other_state, env.ctx, {})
    with pytest.raises(ValueError, match="unknown"):
        screen_tools.decide_screening(PharmState(session_id=SID), env.ctx,
                                      {"screen_id": res.result["screens"][0]["id"], "decision": "approve",
                                       "reason": "ok", "confirm": True})
    assert env.store.list_screens("sess_other")[0].status == "proposed"


def test_tools_are_registered_for_the_evidence_agent():
    from app.tools.builtins import default_registry
    reg = default_registry()
    names = {t.name for t in reg.for_agent("evidence")}
    assert "screen_publications" in names and "decide_screening" not in names     # human-only
    assert reg.get("decide_screening").human_only and reg.get("decide_screening").agent == "evidence"
    assert reg.get("screen_publications").expensive is False
