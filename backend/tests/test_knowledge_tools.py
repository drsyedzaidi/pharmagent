"""mikebrain M1: the ``consult_mikebrain`` reviewer tool and the advisory-only
``adversarial_review(consult_mikebrain=True)`` integration.

Keyless and offline: MockLLM abstains (``no_evidence``) so the deterministic path
runs; a stub LLM injects crafted JSON to exercise grounding; the embedder is
``NoEmbedder`` so no Ollama is contacted.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.core.audit import AuditChain, hash_payload
from app.core.llm import MockLLM
from app.core.pharmstate import AGENT_WRITE_FIELDS, PharmState, PharmStateError, apply_writes
from app.knowledge import manifest as mf
from app.knowledge.extract import extract_text
from app.knowledge.index import NoEmbedder
from app.tools import knowledge_tools as kt
from app.tools.base import CrossAgentToolError, ToolContext
from app.tools.builtins import default_registry

FIXTURES = Path(__file__).parent / "fixtures" / "knowledge"
NOTES = sorted(FIXTURES.glob("*.txt"))
VERBATIM = "Covariate screening on shrunken EBEs is therefore uninformative"
QUESTION = "Is covariate screening on EBEs valid when eta shrinkage is 45 percent?"


def _state(**over) -> PharmState:
    base = dict(session_id="s1", nlme_results={"status": "ok", "converged": True, "condition_number": 12.0,
                                               "shrinkage_pct": {"V": 45.0}, "theta_rse_pct": {"CL": 9.0}})
    return PharmState(**{**base, **over})


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "knowledge"
    for p in NOTES:
        mf.add_document(root, mf.DocumentRecord(str(p), mf.file_sha256(p), p.stem, 2026, "paper", "t",
                                                datetime.now(UTC).isoformat()), extract_text(p))
    monkeypatch.setattr(kt, "default_embedder", lambda: NoEmbedder())
    return tmp_path


@pytest.fixture
def empty_dir(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(kt, "default_embedder", lambda: NoEmbedder())
    return tmp_path


class _StubLLM:
    model = "stub-model"

    def __init__(self, reply: str | Exception) -> None:
        self.reply, self.calls = reply, []

    def complete(self, system, user, *, max_tokens=1024):
        self.calls.append({"system": system, "user": user})
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _chunk_id_for(root: Path, needle: str) -> str:
    from app.knowledge.index import CorpusIndex
    for c in CorpusIndex.load(root).chunks:
        if needle in " ".join(c.text.split()):
            return c.chunk_id
    raise AssertionError("fixture sentence not found")


def _execute(reg, name, state, data_dir, args, audit=None, agent=None):
    audit = audit if audit is not None else AuditChain()
    return reg.execute(name, state=state, ctx=ToolContext(data_dir=str(data_dir)), args=args,
                       audit=audit, timestamp="2026-10-23T00:00:00Z", actor="tester", caller_agent=agent), audit


# ── registration / ownership ────────────────────────────────────────────────────

def test_tool_registered_for_reviewer_and_not_expensive():
    t = default_registry().get("consult_mikebrain")
    assert t.agent == "reviewer" and t.expensive is False and t.proposable is False
    assert t.input_schema["required"] == ["question"]
    assert "knowledge_results" in AGENT_WRITE_FIELDS["reviewer"]
    with pytest.raises(PharmStateError):
        apply_writes(PharmState(), "modeler", {"knowledge_results": []})


def test_other_agents_cannot_execute_it(data_dir: Path):
    with pytest.raises(CrossAgentToolError):
        _execute(default_registry(), "consult_mikebrain", _state(), data_dir, {"question": QUESTION}, agent="modeler")


# ── keyless path (MockLLM abstains) ─────────────────────────────────────────────

def test_mock_llm_complete_abstains():
    out = json.loads(MockLLM().complete("sys", "user"))
    assert out["verdict"] == "no_evidence" and out["citations"] == []


def test_openai_compat_complete_posts_plain_chat_with_long_timeout(monkeypatch):
    """The local critic call: no tools, temperature 0, and the longer COMPLETE_TIMEOUT_S
    (a cold 31B model exceeded the 120 s routing timeout in a real run)."""
    from app.core import llm as llm_mod
    from app.core.llm import OpenAICompatLLM
    calls = []

    class _Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"role": "assistant", "content": '{"verdict": "no_evidence"}'}}]}

    def post(url, json=None, headers=None, timeout=None):  # noqa: A002 - mirrors requests.post
        calls.append({"url": url, "json": json, "timeout": timeout})
        return _Resp()
    monkeypatch.setattr(llm_mod.requests, "post", post)
    llm = OpenAICompatLLM(base_url="http://127.0.0.1:11434/v1", model="gemma4:31b", api_key=None,
                          reasoning_effort="none")
    out = llm.complete("SYS", "USER", max_tokens=77)
    assert out == '{"verdict": "no_evidence"}'
    c = calls[0]
    assert c["url"].endswith("/chat/completions") and c["timeout"] == OpenAICompatLLM.COMPLETE_TIMEOUT_S > 120
    assert c["json"]["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]
    assert c["json"]["max_tokens"] == 77 and c["json"]["temperature"] == 0
    assert "tools" not in c["json"] and "reasoning_effort" not in c["json"]


def test_consult_writes_knowledge_results_and_audit_entry(data_dir: Path):
    (st2, res), audit = _execute(default_registry(), "consult_mikebrain", _state(), data_dir,
                                 {"question": QUESTION, "context": {"model": "1cmt", "nested": {"x": 1}}})
    rec = res.result
    assert rec["status"] == "ok" and rec["verdict"] == "no_evidence" and rec["grounding"] == "abstained"
    assert rec["embedding"] == "unavailable" and rec["provider"] == "mock" and rec["model"] == "mock"
    assert rec["corpus_version"] == mf.load_manifest(data_dir / "knowledge").corpus_version
    assert len(rec["chunk_hashes"]) == len(rec["retrieved"]) == 6
    assert all(len(h) == 64 for h in rec["chunk_hashes"])
    assert all("text" not in r for r in rec["retrieved"])          # no passages in state/audit
    assert isinstance(rec["elapsed_s"], float) and rec["elapsed_s"] >= 0
    # privacy-safe summary: aggregates + scalar context only, never raw rows
    assert rec["state_summary"]["nlme_diagnostics"]["max_shrinkage_pct"] == 45.0
    assert rec["state_summary"]["context"] == {"model": "1cmt"}
    # state
    assert st2.knowledge_results == [rec] and st2.last_agent is None
    # audit entry carries the result (corpus_version, chunk hashes, model, elapsed)
    e = audit.entries[-1]
    assert e.tool == "consult_mikebrain" and e.agent == "reviewer" and e.actor == "tester"
    assert e.outputs_hash == hash_payload(rec) and audit.verify()
    assert "advisory only" in res.summary


def test_knowledge_results_is_bounded_newest_last(data_dir: Path):
    reg = default_registry()
    state = _state(knowledge_results=[{"question": f"old {i}"} for i in range(kt.KNOWLEDGE_MAX)])
    (st2, _), _ = _execute(reg, "consult_mikebrain", state, data_dir, {"question": QUESTION})
    assert len(st2.knowledge_results) == kt.KNOWLEDGE_MAX
    assert st2.knowledge_results[-1]["question"] == QUESTION and st2.knowledge_results[0]["question"] == "old 1"


def test_missing_question_is_rejected(data_dir: Path):
    with pytest.raises(ValueError):
        _execute(default_registry(), "consult_mikebrain", _state(), data_dir, {})
    with pytest.raises(ValueError):
        _execute(default_registry(), "consult_mikebrain", _state(), data_dir, {"question": "q", "context": "no"})


def test_no_corpus_is_reported_not_raised(empty_dir: Path):
    (st2, res), audit = _execute(default_registry(), "consult_mikebrain", _state(), empty_dir, {"question": QUESTION})
    rec = res.result
    assert rec["status"] == "no_corpus" and rec["verdict"] == "no_evidence" and rec["retrieved"] == []
    assert rec["corpus_version"] == mf.Manifest().corpus_version
    assert st2.knowledge_results == [rec] and audit.entries[-1].tool == "consult_mikebrain"


# ── grounding through the tool (stub LLM) ───────────────────────────────────────

def test_supported_verdict_survives_only_with_verbatim_quote(data_dir: Path, monkeypatch):
    root = data_dir / "knowledge"
    cid = _chunk_id_for(root, VERBATIM)
    stub = _StubLLM(json.dumps({"verdict": "supported", "rationale": "the note says so", "citations": [
        {"chunk_id": cid, "quote": "Covariate screening on shrunken EBEs is therefore\nuninformative"},
    ]}))
    monkeypatch.setattr(kt, "get_llm", lambda: stub)
    (st2, res), _ = _execute(default_registry(), "consult_mikebrain", _state(), data_dir, {"question": QUESTION})
    rec = res.result
    assert rec["verdict"] == "supported" and rec["grounding"] == "ok" and rec["model"] == "stub-model"
    assert rec["citations"][0]["chunk_id"] == cid and cid in [r["chunk_id"] for r in rec["retrieved"]]
    assert "Answer ONLY from the excerpts" in stub.calls[0]["system"]
    assert f"[{cid}]" in stub.calls[0]["user"] and "Excerpts:" in stub.calls[0]["user"]


def test_paraphrase_and_foreign_chunk_are_dropped_and_verdict_downgraded(data_dir: Path, monkeypatch):
    cid = _chunk_id_for(data_dir / "knowledge", VERBATIM)
    stub = _StubLLM(json.dumps({"verdict": "contradicted", "rationale": "r", "citations": [
        {"chunk_id": cid, "quote": "covariate screening on shrunken EBEs is pointless and should be avoided"},
        {"chunk_id": "ffffffffffff:0", "quote": VERBATIM},
    ]}))
    monkeypatch.setattr(kt, "get_llm", lambda: stub)
    (_, res), _ = _execute(default_registry(), "consult_mikebrain", _state(), data_dir, {"question": QUESTION})
    rec = res.result
    assert rec["verdict"] == "no_evidence" and rec["grounding"] == "failed" and rec["raw_verdict"] == "contradicted"
    assert rec["citations"] == [] and {d["reason"] for d in rec["dropped_citations"]} == {
        "quote is not a verbatim substring of the cited chunk", "chunk_id not among retrieved chunks"}


def test_unparseable_reply_is_no_evidence(data_dir: Path, monkeypatch):
    monkeypatch.setattr(kt, "get_llm", lambda: _StubLLM("The literature clearly supports this."))
    (_, res), _ = _execute(default_registry(), "consult_mikebrain", _state(), data_dir, {"question": QUESTION})
    assert res.result["verdict"] == "no_evidence" and res.result["grounding"] == "unparseable"


def test_llm_failure_is_recorded_not_raised(data_dir: Path, monkeypatch):
    monkeypatch.setattr(kt, "get_llm", lambda: _StubLLM(RuntimeError("model 'gemma4:31b' not found")))
    (st2, res), audit = _execute(default_registry(), "consult_mikebrain", _state(), data_dir, {"question": QUESTION})
    rec = res.result
    assert rec["status"] == "llm_error" and "not found" in rec["error"] and rec["verdict"] == "no_evidence"
    assert rec["chunk_hashes"] and st2.knowledge_results == [rec] and audit.verify()


def test_state_summary_never_contains_raw_rows():
    state = _state(nca_parameters=[{"subject": 1, "Cmax": 4.5, "AUC_last": 30.0, "lambda_z": 0.1,
                                    "pct_AUC_extrap": 3.0}],
                   review_results={"status": "FINDINGS BLOCK GOAL", "goal_met": False,
                                   "counts": {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 0, "LOW": 0},
                                   "findings": [{"id": "nlme-converge", "severity": "HIGH", "resolved": False}]},
                   pending_tool={"tool": "run_nlme", "args": {"model": "1cmt"}})
    s = kt.compact_state_summary(state, {"a": 1, "rows": [1, 2, 3], "b": True, "c": 1.5})
    assert set(s) == {"nca_diagnostics", "nlme_diagnostics", "review_diagnostics", "pending_tool", "context"}
    assert s["nca_diagnostics"] == {"n_subjects": 1, "missing_lambda_z": 0, "high_extrap_gt20pct": 0}
    assert s["review_diagnostics"]["blocking_ids"] == ["nlme-converge"] and s["pending_tool"] == "run_nlme"
    assert s["context"] == {"a": 1, "b": True, "c": 1.5}
    assert "Cmax" not in json.dumps(s) and "4.5" not in json.dumps(s)
    q = kt.build_query("why?", s)
    assert q.startswith("why?") and "max_shrinkage_pct 45.0" in q and len(q) <= kt.MAX_QUERY_CHARS


# ── adversarial_review integration: advisory only ───────────────────────────────

def _failing_state() -> PharmState:
    return _state(nlme_results={"status": "ok", "converged": False, "condition_number": 5000.0,
                                "shrinkage_pct": {"V": 45.0}, "theta_rse_pct": {"CL": 9.0}})


def test_review_with_mikebrain_leaves_counts_findings_and_goal_unchanged(data_dir: Path):
    reg = default_registry()
    (plain_state, plain), _ = _execute(reg, "adversarial_review", _failing_state(), data_dir, {})
    (adv_state, adv), audit = _execute(reg, "adversarial_review", _failing_state(), data_dir,
                                       {"consult_mikebrain": True})
    assert "advisories" not in plain.result and "advisory_status" not in plain.result
    for key in ("counts", "goal_met", "status", "findings", "n_findings", "checked", "nothing_checked",
                "unverifiable", "goal"):
        assert adv.result[key] == plain.result[key], key
    assert adv.result["goal_met"] is False and adv.result["counts"]["HIGH"] == 2
    assert adv.result["advisory_status"] == "ok"
    advisories = adv.result["advisories"]
    assert 1 <= len(advisories) <= kt.MAX_ADVISORY_QUERIES == len(advisories)
    for a in advisories:
        assert a["kind"] == "literature_advisory" and "severity" not in a
        assert a["verdict"] == "no_evidence" and a["grounding"] == "abstained"
        assert a["corpus_version"] and a["model"] == "mock" and a["chunk_hashes"]
        assert a["finding_id"] in {f["id"] for f in plain.result["findings"]}
    assert adv_state.review_results["advisories"] == advisories
    assert adv_state.knowledge_results is None                  # advisories live under review_results only
    assert "advisory only, not counted" in adv.summary and audit.verify()


def test_review_flag_is_strict_opt_in(data_dir: Path):
    reg = default_registry()
    for flag in ("false", "yes", 1, None, False):
        (_, res), _ = _execute(reg, "adversarial_review", _failing_state(), data_dir, {"consult_mikebrain": flag})
        assert "advisories" not in res.result, repr(flag)
    (_, res), _ = _execute(reg, "adversarial_review", _failing_state(), data_dir, {"consult_mikebrain": "true"})
    assert res.result["advisory_status"] == "ok"


def test_review_flag_without_corpus_or_findings(data_dir: Path):
    reg = default_registry()
    no_corpus = data_dir / "elsewhere"
    no_corpus.mkdir()
    (_, res), _ = _execute(reg, "adversarial_review", _failing_state(), no_corpus, {"consult_mikebrain": True})
    assert res.result["advisories"] == [] and res.result["advisory_status"] == "no_corpus"
    assert res.result["goal_met"] is False
    clean = PharmState(session_id="s", nlme_results={"status": "ok", "converged": True, "condition_number": 3.0})
    (_, res), _ = _execute(reg, "adversarial_review", clean, data_dir, {"consult_mikebrain": True})
    assert res.result["advisories"] == [] and res.result["advisory_status"] == "no_findings"
    assert res.result["goal_met"] is True


def test_review_survives_an_advisory_crash(data_dir: Path, monkeypatch):
    import app.tools.knowledge_tools as mod

    def boom(*a, **k):
        raise OSError("disk full while caching embeddings")
    monkeypatch.setattr(mod, "review_advisories", boom)
    reg = default_registry()
    (plain_state, plain), _ = _execute(reg, "adversarial_review", _failing_state(), data_dir, {})
    (_, res), audit = _execute(reg, "adversarial_review", _failing_state(), data_dir, {"consult_mikebrain": True})
    assert res.result["advisories"] == [] and res.result["advisory_status"].startswith("error: ")
    assert res.result["counts"] == plain.result["counts"] and res.result["goal_met"] is False
    assert audit.verify()


def test_review_advisories_use_one_llm_and_grounding(data_dir: Path, monkeypatch):
    cid = _chunk_id_for(data_dir / "knowledge", VERBATIM)
    stub = _StubLLM(json.dumps({"verdict": "supported", "rationale": "r",
                                "citations": [{"chunk_id": cid, "quote": VERBATIM}]}))
    monkeypatch.setattr(kt, "get_llm", lambda: stub)
    (_, res), _ = _execute(default_registry(), "adversarial_review", _failing_state(), data_dir,
                           {"consult_mikebrain": True})
    adv = res.result["advisories"]
    assert len(stub.calls) == len(adv) == kt.MAX_ADVISORY_QUERIES
    # the quoted chunk is only retrieved for the shrinkage finding; elsewhere it is a foreign id → downgraded
    by_id = {a["finding_id"]: a for a in adv}
    assert by_id["nlme-shrink-V"]["verdict"] == "supported" and by_id["nlme-shrink-V"]["grounding"] == "ok"
    assert all(a["grounding"] in ("ok", "failed") for a in adv)
    assert res.result["counts"]["HIGH"] == 2 and res.result["goal_met"] is False
