"""mikebrain M3: the literature critic at the planner's human gate — advisory only.

Keyless and offline. No population fit runs: ``poppk_goal`` with the mock planner stops at the
``run_nlme`` gate, which is where the critic is consulted. Properties pinned here:

  * off by default, and then it leaves NO trace (same audit chain, same state, no LLM call);
  * on, it adds exactly one audited ``reviewer`` entry before the planner's ``propose_tool`` entry and a
    ``critic`` block on the pending proposal, and changes nothing else about the proposal;
  * an objection ("contradicted") cannot block, alter or reroute anything: approve, reject and the job
    hand-off behave exactly as without it, and ``critic`` never reaches the tool's arguments;
  * every failure mode (no corpus, bad corpus, LLM down, unexpected exception) still parks the proposal.
"""
from __future__ import annotations

import itertools
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.config import settings
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.pharmstate import PharmState
from app.core.store import SessionStore
from app.knowledge import gate_critic
from app.knowledge import manifest as mf
from app.knowledge.extract import extract_text
from app.knowledge.index import NoEmbedder
from app.tools import knowledge_tools as kt

SAMPLE = str(Path(__file__).parent.parent / "sample_data" / "oral_pk.csv")
FIXTURES = Path(__file__).parent / "fixtures" / "knowledge"
NOTES = sorted(FIXTURES.glob("*.txt"))
CHAT = f"load {SAMPLE} and then run NCA and then bootstrap the parameter uncertainty and then write the report"


# ── harness ─────────────────────────────────────────────────────────────────────

def _orch() -> Orchestrator:
    return Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                        store=SessionStore(":memory:"))


def _add_corpus(data_dir: Path) -> None:
    root = data_dir / "knowledge"
    for p in NOTES:
        mf.add_document(root, mf.DocumentRecord(str(p), mf.file_sha256(p), p.stem, 2026, "paper", "t",
                                                datetime.now(UTC).isoformat()), extract_text(p))


class _Critic:
    """A scripted mikebrain LLM. ``mode`` picks the verdict; citations are quoted verbatim from the
    first excerpt the prompt actually contained, so grounding is real, not mocked."""
    model = "stub-critic"

    def __init__(self, mode: str = "contradicted") -> None:
        self.mode, self.prompts = mode, []

    def complete(self, system, user, *, max_tokens=1024):
        self.prompts.append(user)
        if self.mode == "error":
            raise RuntimeError("model 'gemma4:31b' timed out")
        if self.mode == "no_evidence":
            return json.dumps({"verdict": "no_evidence", "rationale": "nothing relevant", "citations": []})
        m = re.search(r"\[([0-9a-f]{12}:\d+)\] \([^)]*\)\n(.+)", user)
        cid, quote = m.group(1), " ".join(m.group(2).split())[:60]
        return json.dumps({"verdict": self.mode, "rationale": "per the cited note",
                           "citations": [{"chunk_id": cid, "quote": quote}]})


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    """Data dir isolated from backend/data; the critic LLM is the keyless mock unless a test swaps it."""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(kt, "default_embedder", lambda: NoEmbedder())
    monkeypatch.setattr(kt, "get_llm", MockLLM)
    return tmp_path


def _enable(monkeypatch, critic: _Critic | None = None) -> None:
    monkeypatch.setattr(settings, "mikebrain_gate_critic", True)
    if critic is not None:
        monkeypatch.setattr(kt, "get_llm", lambda: critic)


def _gate(orch: Orchestrator | None = None):
    orch = orch or _orch()
    sid = orch.create_session().id
    return orch, sid, orch.start_workflow(sid, "poppk_goal", {"path": SAMPLE})


def _norm(obj) -> str:
    """JSON text with the random dataset and session ids neutralised, so two separate runs are comparable."""
    text = json.dumps(obj, sort_keys=True, default=str)
    return re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|ds_[0-9a-f]{6,}|sess_[0-9a-f]{6,}", "ID", text)


def _audit(orch, sid) -> list[tuple[str, str, str]]:
    return [(e.agent, e.tool, e.action) for e in orch.get_session(sid).audit.entries]


def _core(pending: dict) -> dict:
    """The proposal minus the advisory and its clock reading — what the human is asked to approve. (An enabled
    run consumes clock ticks for the consultation, so ``proposed_at`` is later by design.)"""
    return {k: v for k, v in pending.items() if k not in ("critic", "proposed_at")}


# ── off by default ──────────────────────────────────────────────────────────────

def test_the_feature_is_off_by_default():
    assert settings.mikebrain_gate_critic is False and gate_critic.enabled() is False


def test_off_leaves_no_trace_even_with_a_corpus(env, monkeypatch):
    base_orch, base_sid, base = _gate()
    _add_corpus(env)
    spy = _Critic()
    monkeypatch.setattr(kt, "get_llm", lambda: spy)
    orch, sid, out = _gate()
    assert out["status"] == "awaiting_review" and "critic" not in out["state"]["pending_tool"]
    assert spy.prompts == [] and out["state"]["knowledge_results"] is None
    # Same chain entry for entry. The counter clock makes timestamps part of the proof: not one extra tick was
    # consumed. (Hashes embed the session id, so they cannot be compared across two sessions.)
    shape = lambda o, s: _norm([(e.index, e.timestamp, e.agent, e.tool, e.action, e.actor, e.reason)    # noqa: E731
                                for e in o.get_session(s).audit.entries])
    assert shape(orch, sid) == shape(base_orch, base_sid)
    assert _norm(out["state"]) == _norm(base["state"]) and _norm(out["executed"]) == _norm(base["executed"])


# ── on: the happy path and every stance ─────────────────────────────────────────

def test_enabled_without_a_corpus_makes_no_llm_call_and_says_so(env, monkeypatch):
    spy = _Critic()
    _enable(monkeypatch, spy)
    orch, sid, out = _gate()
    critic = out["state"]["pending_tool"]["critic"]
    assert critic["stance"] == "not_consulted" and critic["status"] == "no_corpus" and critic["advisory"] is True
    assert critic["note"] == gate_critic.ADVISORY_NOTE and critic["source"] == "mikebrain"
    assert spy.prompts == [] and not [a for a in _audit(orch, sid) if a[:2] == ("reviewer", "consult_mikebrain")]
    assert out["audit_ok"] and out["status"] == "awaiting_review"


def test_enabled_with_the_keyless_mock_reports_no_evidence_and_audits_before_the_proposal(env, monkeypatch):
    _add_corpus(env)
    _enable(monkeypatch)
    orch, sid, out = _gate()
    pending = out["state"]["pending_tool"]
    critic = pending["critic"]
    assert critic["stance"] == "no_evidence" and critic["status"] == "ok" and critic["grounding"] == "abstained"
    assert critic["provider"] == "mock" and critic["citations"] == [] and len(critic["chunk_hashes"]) >= 1
    assert out["proposal"] == pending                                   # the same block reaches the API payload
    audit = _audit(orch, sid)
    consults = [i for i, a in enumerate(audit) if a[:2] == ("reviewer", "consult_mikebrain")]
    i_gate = audit.index(("planner", "run_nlme", "policy_gated"))
    i_propose = audit.index(("planner", "run_nlme", "propose_tool"))
    assert len(consults) == 1 and i_gate < consults[0] < i_propose == len(audit) - 1
    propose = orch.get_session(sid).audit.entries[i_propose]
    from app.core.audit import hash_payload  # the advisory is part of what was audited
    assert propose.outputs_hash == hash_payload({
        "status": "pending_approval", "agent": "modeler", "rationale": pending["rationale"],
        "gate_reason": pending["gate_reason"], "critic": critic})
    assert len(out["state"]["knowledge_results"]) == 1 and out["audit_ok"]


@pytest.mark.parametrize("mode, stance", [("contradicted", "objection"), ("supported", "corroboration"),
                                          ("no_evidence", "no_evidence")])
def test_each_verdict_maps_to_an_explicit_stance_with_real_citations(env, monkeypatch, mode, stance):
    _add_corpus(env)
    _enable(monkeypatch, _Critic(mode))
    _, _, out = _gate()
    critic = out["state"]["pending_tool"]["critic"]
    assert critic["stance"] == stance and critic["meaning"] == gate_critic.STANCES[stance]
    assert critic["grounding"] == ("abstained" if mode == "no_evidence" else "ok")
    assert bool(critic["citations"]) is (mode != "no_evidence")
    for c in critic["citations"]:
        assert len(c["quote"]) >= 20 and c["chunk_id"]


def test_stance_requires_a_grounded_citation():
    base = {"status": "ok", "citations": [{"chunk_id": "a", "quote": "x" * 30}]}
    assert gate_critic.stance_of({**base, "verdict": "contradicted"}) == "objection"
    assert gate_critic.stance_of({**base, "verdict": "supported"}) == "corroboration"
    assert gate_critic.stance_of({**base, "verdict": "no_evidence"}) == "no_evidence"
    assert gate_critic.stance_of({"status": "ok", "verdict": "contradicted", "citations": []}) == "no_evidence"
    assert gate_critic.stance_of({"status": "ok", "verdict": "supported"}) == "no_evidence"
    for status in ("llm_error", "corpus_error", "no_corpus", None):
        assert gate_critic.stance_of({**base, "verdict": "contradicted", "status": status}) == "unavailable"


def test_an_ungrounded_objection_is_downgraded_by_the_existing_validator(env, monkeypatch):
    _add_corpus(env)

    class Paraphraser(_Critic):
        def complete(self, system, user, *, max_tokens=1024):
            cid = re.search(r"\[([0-9a-f]{12}:\d+)\]", user).group(1)
            return json.dumps({"verdict": "contradicted", "rationale": "r", "citations": [
                {"chunk_id": cid, "quote": "this sentence does not appear in the excerpt at all"}]})
    _enable(monkeypatch, Paraphraser())
    _, _, out = _gate()
    critic = out["state"]["pending_tool"]["critic"]
    assert critic["stance"] == "no_evidence" and critic["grounding"] == "failed" and critic["citations"] == []


# ── advisory only: nothing about the proposal or the human's options changes ─────

def test_an_objection_changes_nothing_about_the_proposal_or_the_policy_decision(env, monkeypatch):
    base_orch, base_sid, base = _gate()
    _add_corpus(env)
    _enable(monkeypatch, _Critic("contradicted"))
    orch, sid, out = _gate()
    assert out["state"]["pending_tool"]["critic"]["stance"] == "objection"
    assert _core(out["state"]["pending_tool"]) == _core(base["state"]["pending_tool"])
    assert out["status"] == base["status"] == "awaiting_review"
    assert _norm(out["review"]) == _norm(base["review"]) and _norm(out["executed"]) == _norm(base["executed"])
    assert out["state"]["planner_history"] == base["state"]["planner_history"]
    assert out["state"]["current_step"] == base["state"]["current_step"]
    planner = lambda o, s: [(e.action, e.tool, e.inputs_hash, e.outputs_hash)    # noqa: E731
                            for e in o.get_session(s).audit.entries
                            if e.agent == "planner" and e.action != "propose_tool"]
    assert planner(orch, sid) == planner(base_orch, base_sid)             # plan + policy entries identical


def test_reject_after_an_objection_is_signed_exactly_as_without_it(env, monkeypatch):
    base_orch, base_sid, _ = _gate()
    base_out = base_orch.resume_workflow(base_sid, approve=False, actor="alice", reason="wrong structural model")
    _add_corpus(env)
    _enable(monkeypatch, _Critic("contradicted"))
    orch, sid, _ = _gate()
    out = orch.resume_workflow(sid, approve=False, actor="alice", reason="wrong structural model")
    assert out["status"] == base_out["status"] == "rejected" and out["state"]["pending_tool"] is None
    tail = lambda o, s: [(e.agent, e.tool, e.action, e.actor, e.reason)       # noqa: E731
                         for e in o.get_session(s).audit.entries[-2:]]
    assert tail(orch, sid) == tail(base_orch, base_sid)
    assert out["audit_ok"]


def test_approval_after_a_corroboration_still_defers_expensive_work_to_the_job_queue(env, monkeypatch):
    base_orch, base_sid, _ = _gate()
    base_out = base_orch.resume_workflow(base_sid, approve=True, actor="alice", reason="ok")
    _add_corpus(env)
    _enable(monkeypatch, _Critic("supported"))
    orch, sid, _ = _gate()
    n = len(orch.get_session(sid).audit.entries)
    out = orch.resume_workflow(sid, approve=True, actor="alice", reason="ok")
    assert out["status"] == base_out["status"] == "awaiting_job" and out["next_tool"] == "run_nlme"
    assert out["state"]["pending_tool"]["critic"]["stance"] == "corroboration"   # kept with the kept proposal
    assert out["state"]["nlme_results"] is None and len(orch.get_session(sid).audit.entries) == n


def test_the_critic_never_reaches_the_tool_the_human_approves(env, monkeypatch):
    _add_corpus(env)
    _enable(monkeypatch, _Critic("contradicted"))
    orch, sid, out = _gate()
    assert "critic" in out["state"]["pending_tool"]
    call = orch.take_pending_tool(sid, actor="alice", reason="approved despite the objection")
    assert set(call) == {"tool", "agent", "args"} and call["tool"] == "run_nlme"
    assert "critic" not in call["args"] and call["args"].get("confirm") is True
    approve = next(e for e in orch.get_session(sid).audit.entries if e.action == "approve_tool")
    assert approve.actor == "alice" and approve.reason == "approved despite the objection"
    assert orch.get_session(sid).audit.verify()


def test_chat_proposals_are_never_critiqued(env, monkeypatch):
    _add_corpus(env)
    spy = _Critic("contradicted")
    _enable(monkeypatch, spy)
    orch = _orch()
    sid = orch.create_session(owner="alice").id
    out = orch.chat(sid, CHAT, actor="alice")
    assert out["pending_tool"]["tool"] == "run_bootstrap" and "critic" not in out["pending_tool"]
    assert spy.prompts == []


def test_only_the_planner_gate_calls_the_critic():
    app_dir = Path(__file__).parent.parent / "app"
    users = {str(p.relative_to(app_dir)) for p in app_dir.rglob("*.py")
             if "critique_proposal(" in p.read_text(encoding="utf-8")}
    assert users == {"core/orchestrator.py", "knowledge/gate_critic.py"}
    orch_src = (app_dir / "core" / "orchestrator.py").read_text(encoding="utf-8")
    assert orch_src.count("gate_critic.critique_proposal(") == 1


# ── failure containment ─────────────────────────────────────────────────────────

def test_an_llm_failure_is_reported_and_the_proposal_is_still_parked(env, monkeypatch):
    _add_corpus(env)
    _enable(monkeypatch, _Critic("error"))
    orch, sid, out = _gate()
    critic = out["state"]["pending_tool"]["critic"]
    assert critic["stance"] == "unavailable" and critic["status"] == "llm_error" and critic["citations"] == []
    assert out["status"] == "awaiting_review" and out["state"]["pending_tool"]["tool"] == "run_nlme"
    assert out["audit_ok"]


def test_a_corrupt_corpus_is_reported_and_the_proposal_is_still_parked(env, monkeypatch):
    _add_corpus(env)
    mf.manifest_path(env / "knowledge").write_text("{broken", encoding="utf-8")
    _enable(monkeypatch, _Critic())
    _, _, out = _gate()
    critic = out["state"]["pending_tool"]["critic"]
    assert critic["stance"] == "unavailable" and critic["status"] == "corpus_error"
    assert str(env) not in json.dumps(critic) and out["status"] == "awaiting_review"


def test_an_unexpected_exception_cannot_stop_a_proposal_or_leak_into_it(env, monkeypatch):
    _add_corpus(env)
    _enable(monkeypatch, _Critic())
    orch = _orch()
    real = orch.registry.execute

    def boom(name, **kw):
        if name == "consult_mikebrain":
            raise RuntimeError("kaboom at /Users/someone/secret/path")
        return real(name, **kw)
    monkeypatch.setattr(orch.registry, "execute", boom)
    sid = orch.create_session().id
    out = orch.start_workflow(sid, "poppk_goal", {"path": SAMPLE})
    critic = out["state"]["pending_tool"]["critic"]
    assert critic["stance"] == "unavailable" and critic["status"] == "error"
    assert "secret" not in json.dumps(out["state"]["pending_tool"]) and out["state"]["knowledge_results"] is None
    assert out["status"] == "awaiting_review" and out["audit_ok"]


# ── the question put to the literature ──────────────────────────────────────────

def test_the_question_names_the_choice_and_hides_subject_identifiers():
    q = gate_critic.gate_question(
        "run_scm", {"covariates": ["WT", "CRCL"], "forward_alpha": 0.05},
        "Shrinkage on V is 45 percent; subject 1771 and nca-clf-sign-1771 look odd", "run_scm changes the covariate model",
        "Converged population PK model for subject cohort A")
    assert "run_scm" in q and '"forward_alpha": 0.05' in q and "covariate model" in q and "45 percent" in q
    assert "1771" not in q and "a subject" in q and "a per-subject finding" in q


def test_the_question_is_size_capped_and_tolerates_odd_arguments():
    q = gate_critic.gate_question("run_nlme", {"blob": "x" * 5000}, "r" * 5000, "g" * 5000, "y" * 5000)
    assert len(q) < 2400 and q.count("x") <= gate_critic.MAX_ARGS_CHARS
    for field in "ryg":
        assert q.count(field * (gate_critic.MAX_TEXT_CHARS + 1)) == 0       # each free-text field capped at 400
    assert "run_nlme" in gate_critic.gate_question("run_nlme", {"s": {1, 2}}, "", "", "")   # {1, 2} is not JSON


def test_the_critic_sees_aggregates_only_never_rows(env, monkeypatch):
    _add_corpus(env)
    spy = _Critic("no_evidence")
    _enable(monkeypatch, spy)
    _gate()
    prompt = spy.prompts[0]
    assert "Analysis state (aggregates only)" in prompt and "run_nlme" in prompt
    assert not {"TIME", "AMT", "DV"} & set(re.findall(r"\b[A-Z]{2,4}\b", prompt.split("Excerpts:")[0]))
    summary = json.loads(re.search(r"aggregates only\): (\{.*\})\n", prompt).group(1))
    assert set(summary) <= set(kt.SUMMARY_KEYS) | {"context"}
    assert summary["context"] == {"proposed_tool": "run_nlme", "gated_by": "planner policy"}


def test_redact_subjects_keeps_numbers_but_not_identities():
    assert kt.redact_subjects("Subject S-07 had CL 12.5; see recompute-cmax-S07") == \
        "a subject had CL 12.5; see a per-subject finding"
    assert kt.redact("Subject S-07 had CL 12.5") == "a subject had CL N"             # the M1 behaviour is unchanged


def test_state_unchanged_when_off_even_if_called_directly():
    state = PharmState(session_id="s")
    out_state, critic = gate_critic.critique_proposal(
        None, state, None, None, tool_name="t", args={}, rationale="", gate_reason="", goal="", clock=None)  # type: ignore[arg-type]
    assert out_state is state and critic is None
