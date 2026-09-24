"""Regression tests for the branch's security/integrity review findings.

Companion to test_identity_audit.py (which covers finding F1 — the versioned,
action-binding audit hash). Here:

  F2  a long-running fit (NLME/SCM/engine comparison) must NOT execute
      synchronously from a chat turn — it is routed to its bounded job endpoint.
  F4  CSV / DDE formula injection: cells starting with = + - @ (or tab / CR) are
      neutralized in every CSV export path (exporters and CDISC).
  F5  a persisted session whose dataset file no longer matches its recorded
      sha256 must fail closed on rehydration — the dataset is not loaded and the
      tamper is flagged, never silently trusted.
"""
from __future__ import annotations

import itertools
from pathlib import Path

from app.core.exporters import sanitize_cell

SAMPLE = str(Path(__file__).parent.parent / "sample_data" / "oral_pk.csv")


# ── F2 — chat must not run expensive fits synchronously ───────────────────────


class _PickTool:
    """A stub LLM that selects `tool_name` once, then stops."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        self._served = False

    def classify(self, message, options, descriptions):   # pragma: no cover - unused
        return options[0] if options else "modeler"

    def select_tool(self, agent, message, tools, state_summary):
        if self._served:
            return None
        self._served = True
        return {"name": self.tool_name, "input": {}}


def _modeler_turn(tool_name: str):
    from app.agents.base import Agent
    from app.core.audit import AuditChain
    from app.core.pharmstate import PharmState
    from app.tools.base import ToolContext
    from app.tools.builtins import default_registry

    agent = Agent(name="modeler", system_prompt="")
    return agent.run_turn(
        state=PharmState(), message="fit it", llm=_PickTool(tool_name),
        registry=default_registry(), ctx=ToolContext(), audit=AuditChain(),
        clock=lambda: "t0", actor="tester")


def test_expensive_tools_are_flagged():
    """The three long-running fits carry expensive=True; a cheap tool does not."""
    from app.tools.builtins import default_registry
    reg = default_registry()
    expensive = {n for n in reg.names() if reg.get(n).expensive}
    assert {"run_nlme", "run_scm", "run_engine_comparison"} <= expensive
    assert not reg.get("fit_pk_model").expensive


def test_registry_is_the_enforcement_point():
    """Enforcement lives at registry.execute (the single choke point), so it cannot
    be bypassed by naming a tool the calling agent does not own. Admission-
    controlled callers opt in with allow_expensive=True."""
    import pytest

    from app.core.audit import AuditChain
    from app.core.pharmstate import PharmState
    from app.tools.base import ExpensiveToolError, ToolContext
    from app.tools.builtins import default_registry
    reg = default_registry()
    kw = dict(state=PharmState(), ctx=ToolContext(), args={},
              audit=AuditChain(), timestamp="t0")
    with pytest.raises(ExpensiveToolError):
        reg.execute("run_nlme", **kw)                       # default: refused
    # the opt-in path is reachable (it fails later, on the missing dataset, not here)
    try:
        reg.execute("run_nlme", **kw, allow_expensive=True)
    except ExpensiveToolError:                              # pragma: no cover
        raise AssertionError("allow_expensive=True must not be refused") from None
    except Exception:
        pass                                                # any other failure is fine


def test_guard_holds_for_a_tool_outside_the_agents_own_list():
    """The pre-fix guard keyed off the agent's tool list, so an agent naming a tool
    it does not own (here: nca naming run_nlme) skipped admission control."""
    from app.agents.base import Agent
    from app.core.audit import AuditChain
    from app.core.pharmstate import PharmState
    from app.tools.base import ToolContext
    from app.tools.builtins import default_registry
    reg = default_registry()
    assert "run_nlme" not in {t.name for t in reg.for_agent("nca")}
    res = Agent(name="nca", system_prompt="").run_turn(
        state=PharmState(), message="go", llm=_PickTool("run_nlme"), registry=reg,
        ctx=ToolContext(), audit=AuditChain(), clock=lambda: "t0", actor="x")
    assert any(c.get("skipped") == "expensive" for c in res.tool_calls)


def test_workflow_leg_with_a_fit_is_not_run_inline():
    """Human approval is a scientific decision, not queue admission. A leg that
    reaches a long fit must stop rather than run it on the caller's thread."""
    from app.core.llm import MockLLM
    from app.core.orchestrator import Orchestrator
    from app.core.store import SessionStore
    from app.workflows import get_workflow
    orch = Orchestrator(llm=MockLLM(),
                        clock=lambda c=itertools.count(): f"t{next(c)}",
                        store=SessionStore(":memory:"))
    # poppk_modeling reaches run_engine_comparison BEFORE its gate
    assert orch.workflow_needs_job(get_workflow("poppk_modeling"), 0) is True
    # nca_full has no expensive step at all
    assert orch.workflow_needs_job(get_workflow("nca_full"), 0) is False
    # poppk_full's fits sit AFTER the gate, so reaching that gate needs no job
    assert orch.workflow_needs_job(get_workflow("poppk_full"), 0) is False
    out = orch.start_workflow(orch.create_session(owner="a").id,
                              "poppk_modeling", {"path": SAMPLE})
    assert out["status"] == "awaiting_job"
    assert out["next_tool"] == "run_engine_comparison"
    assert out["state"]["engine_comparison_results"] is None


def test_workflow_endpoint_hands_an_expensive_leg_to_the_job_queue():
    import itertools as _it

    from fastapi.testclient import TestClient

    import app.main as main
    from app.core.llm import MockLLM
    from app.core.orchestrator import Orchestrator
    from app.core.store import SessionStore
    main.orch = Orchestrator(llm=MockLLM(),
                             clock=lambda c=_it.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    client = TestClient(main.app)
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(f"/api/sessions/{sid}/workflow",
                    json={"name": "poppk_modeling", "params": {"path": SAMPLE}}).json()
    assert r.get("kind") == "workflow_start" and "job_id" in r
    # a workflow with no expensive leg still runs inline
    sid2 = client.post("/api/sessions").json()["id"]
    r2 = client.post(f"/api/sessions/{sid2}/workflow",
                     json={"name": "nca_full", "params": {"path": SAMPLE}}).json()
    assert r2["status"] == "awaiting_review" and "job_id" not in r2


def test_skill_replay_refuses_expensive_steps():
    """Replaying a captured skill must not re-run NLME/SCM inline on the request
    thread: the step is recorded as an error, and no fit result is written."""
    from app.core.llm import MockLLM
    from app.core.orchestrator import Orchestrator
    from app.core.skills import Skill
    from app.core.store import SessionStore
    orch = Orchestrator(llm=MockLLM(),
                        clock=lambda c=itertools.count(): f"t{next(c)}",
                        store=SessionStore(":memory:"))
    orch.skills.save(Skill(name="replay-probe", description="probe", goal="probe",
                           steps=[{"agent": "modeler", "tool": "run_nlme", "args": {}}],
                           source_session=None, owner="alice", created_at="t0",
                           version=1))
    out = orch.run_skill("replay-probe", dataset_path=SAMPLE, owner="alice")
    step = [e for e in out["executed"] if e["tool"] == "run_nlme"]
    assert step and step[0]["status"] == "error"
    assert "long-running fit" in step[0]["error"]
    assert out["state"]["nlme_results"] is None


def test_chat_refuses_expensive_nlme():
    res = _modeler_turn("run_nlme")
    assert any(c.get("skipped") == "expensive" for c in res.tool_calls)
    assert res.state.nlme_results is None            # nothing was computed
    assert any("dedicated control" in m for m in res.messages)


def test_chat_refuses_expensive_scm_and_engine():
    for name, field in (("run_scm", "scm_results"),
                        ("run_engine_comparison", "engine_comparison_results")):
        res = _modeler_turn(name)
        assert any(c.get("skipped") == "expensive" for c in res.tool_calls), name
        assert getattr(res.state, field) is None, name


# ── F4 — CSV / DDE formula-injection neutralization ───────────────────────────


def test_sanitize_cell_neutralizes_formula_leads():
    for payload in ("=cmd|'/C calc'!A0", "+1+1", "-2+3", "@SUM(A1)",
                    "\t=1", "\r=1", "  =evil()"):
        assert sanitize_cell(payload) == "'" + payload, payload


def test_sanitize_cell_passes_through_safe_values():
    for safe in ("subject-01", "1.23", "AUC_inf", "", "study 5"):
        assert sanitize_cell(safe) == safe
    assert sanitize_cell(42) == 42                   # non-strings untouched
    assert sanitize_cell(None) is None


def test_nca_csv_export_neutralizes_injection():
    from app.core.exporters import export_csv
    from app.core.pharmstate import PharmState
    state = PharmState(nca_parameters=[{"subject": "=DDE()", "Cmax": 9.0}])
    csv_text = export_csv(state, "nca")
    assert "'=DDE()" in csv_text                      # neutralized
    # no data row begins a raw formula
    assert not any(line.startswith("=DDE()") for line in csv_text.splitlines())


def test_cdisc_csv_export_neutralizes_injection():
    from app.core import cdisc
    out = cdisc._csv_bytes([{"USUBJID": "=cmd()", "AVAL": 1.0}],
                           ["USUBJID", "AVAL"]).decode()
    assert "'=cmd()" in out


# ── F5 — persisted dataset digest must be verified on rehydration ─────────────


def _orch(store):
    from app.core.llm import MockLLM
    from app.core.orchestrator import Orchestrator
    return Orchestrator(llm=MockLLM(),
                        clock=lambda c=itertools.count(): f"t{next(c)}",
                        store=store)


def _load_session(store, path: str) -> str:
    """Create a session, import `path`, persist; return the session id."""
    orch = _orch(store)
    sid = orch.create_session(owner="alice").id
    orch.start_workflow(sid, "nca_full", {"path": path})   # runs load_dataset first
    return sid


def test_tamper_detection_is_audited():
    """Part 11: a detection that leaves no trace is not a control. The mismatch must
    append a signed audit entry — once, on the transition, not on every restart."""
    from app.config import settings
    from app.core.store import SessionStore
    ds_file = Path(settings.data_dir) / "tamper_audit_probe_f5.csv"
    ds_file.write_bytes(Path(SAMPLE).read_bytes())
    try:
        store = SessionStore(":memory:")
        sid = _load_session(store, str(ds_file))
        ds_file.write_text(ds_file.read_text() + "\n999,999,999\n")
        sess = _orch(store).sessions[sid]
        hits = [e for e in sess.audit.entries if e.tool == "dataset_integrity"]
        assert len(hits) == 1
        assert "sha256_mismatch" in hits[0].action
        assert sess.audit.verify() is True          # the new entry chains correctly
    finally:
        ds_file.unlink(missing_ok=True)


def test_deleted_dataset_is_not_reported_as_tampering():
    """file_sha256 returns 'n/a' for a missing file; reporting that as
    'sha256_mismatch' would cry tamper every time a file is simply moved away."""
    from app.config import settings
    from app.core.store import SessionStore
    ds_file = Path(settings.data_dir) / "gone_probe_f5.csv"
    ds_file.write_bytes(Path(SAMPLE).read_bytes())
    store = SessionStore(":memory:")
    sid = _load_session(store, str(ds_file))
    ds_file.unlink()                                # file simply disappears
    sess = _orch(store).sessions[sid]
    md = sess.state.dataset_metadata or {}
    assert md.get("dataset_integrity") == "file_missing"
    assert sess.state.dataset_id not in sess.ctx.dataset_store   # still fails closed


def test_integrity_detection_is_persisted_not_just_in_memory():
    """A detection that lives only in the process is not a record: the next
    restart would re-read the original row and re-detect it forever."""
    from app.config import settings
    from app.core.store import SessionStore
    ds_file = Path(settings.data_dir) / "persist_probe_f5.csv"
    ds_file.write_bytes(Path(SAMPLE).read_bytes())
    try:
        store = SessionStore(":memory:")
        sid = _load_session(store, str(ds_file))
        ds_file.write_text(ds_file.read_text() + "\n999,999,999\n")
        _orch(store)                       # first restart: detects + must persist
        row = store.load(sid) if hasattr(store, "load") else None
        if row is not None:                # the verdict is durable in the DB
            assert (row["state"].get("dataset_metadata") or {}).get(
                "dataset_integrity") == "sha256_mismatch"
        # second restart re-reads the PERSISTED row: still flagged, and the
        # audit entry is not duplicated (the transition already happened).
        again = _orch(store).sessions[sid]
        assert (again.state.dataset_metadata or {}).get(
            "dataset_integrity") == "sha256_mismatch"
        hits = [e for e in again.audit.entries if e.tool == "dataset_integrity"]
        assert len(hits) == 1, f"expected one durable detection, got {len(hits)}"
    finally:
        ds_file.unlink(missing_ok=True)


def test_review_is_unverifiable_without_raw_data():
    """The reviewer's whole job is recomputing reported numbers from raw data.
    With results present but no dataset, the verdict must be UNVERIFIABLE — not
    an approval that launders a data-integrity failure into a clean bill."""
    import itertools as _it

    from app.core.llm import MockLLM
    from app.core.orchestrator import Orchestrator
    from app.core.store import SessionStore
    from app.tools.review_tools import adversarial_review
    orch = Orchestrator(llm=MockLLM(), clock=lambda c=_it.count(): f"t{next(c)}",
                        store=SessionStore(":memory:"))
    sid = orch.create_session(owner="a").id
    orch.start_workflow(sid, "nca_full", {"path": SAMPLE})
    sess = orch.get_session(sid)

    with_data = adversarial_review(sess.state, sess.ctx, {})
    assert with_data.result["goal_met"] is True
    assert with_data.result["status"] == "GOAL MET"

    class EmptyCtx:                        # the F5 fail-closed situation
        dataset_store: dict = {}
        data_dir = "data"
    without = adversarial_review(sess.state, EmptyCtx(), {})
    assert without.result["goal_met"] is False
    assert without.result["status"] == "UNVERIFIABLE"
    assert without.result["checked"]["nca_recompute"] is False
    assert "NOT independently verified" in without.summary


def test_integrity_flag_clears_when_the_digest_matches_again():
    """A sticky flag would mark a session tampered forever — and _persist would write
    that stale verdict back into stored provenance."""
    from app.config import settings
    from app.core.store import SessionStore
    ds_file = Path(settings.data_dir) / "restore_probe_f5.csv"
    original = Path(SAMPLE).read_bytes()
    ds_file.write_bytes(original)
    try:
        store = SessionStore(":memory:")
        sid = _load_session(store, str(ds_file))
        ds_file.write_bytes(original + b"\n999,999,999\n")        # tamper
        flagged = _orch(store).sessions[sid]
        assert (flagged.state.dataset_metadata or {}).get("dataset_integrity") \
            == "sha256_mismatch"
        ds_file.write_bytes(original)                              # restore
        healed = _orch(store).sessions[sid]
        assert "dataset_integrity" not in (healed.state.dataset_metadata or {})
        assert healed.state.dataset_id in healed.ctx.dataset_store  # loads again
    finally:
        ds_file.unlink(missing_ok=True)


def test_cdisc_export_refuses_rather_than_shipping_an_empty_adpc():
    """A package with a fully populated ADPP and a silently empty ADPC reads as a
    complete submission dataset — refuse instead."""
    import itertools as _it

    from fastapi.testclient import TestClient

    import app.main as main
    from app.core.llm import MockLLM
    from app.core.orchestrator import Orchestrator
    from app.core.store import SessionStore
    main.orch = Orchestrator(llm=MockLLM(),
                             clock=lambda c=_it.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    client = TestClient(main.app)
    sid = client.post("/api/sessions").json()["id"]
    client.post(f"/api/sessions/{sid}/workflow",
                json={"name": "nca_full", "params": {"path": SAMPLE}})
    assert client.get(f"/api/sessions/{sid}/cdisc").status_code == 200   # normal path
    main.orch.get_session(sid).ctx.dataset_store.clear()                 # source gone
    r = client.get(f"/api/sessions/{sid}/cdisc")
    assert r.status_code == 409
    assert "ADPC would be empty" in r.json()["error"]["message"]


def test_cdisc_export_refuses_when_roles_are_unmapped():
    """`df is not None` is not enough: build_adpc also yields zero rows when the
    ID/TIME/DV roles are unmapped, which shipped an empty ADPC at HTTP 200."""
    import itertools as _it

    from fastapi.testclient import TestClient

    import app.main as main
    from app.core.llm import MockLLM
    from app.core.orchestrator import Orchestrator
    from app.core.store import SessionStore
    main.orch = Orchestrator(llm=MockLLM(),
                             clock=lambda c=_it.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    client = TestClient(main.app)
    sid = client.post("/api/sessions").json()["id"]
    client.post(f"/api/sessions/{sid}/workflow",
                json={"name": "nca_full", "params": {"path": SAMPLE}})
    sess = main.orch.get_session(sid)
    md = dict(sess.state.dataset_metadata or {})
    md["detected_roles"] = {k: v for k, v in md.get("detected_roles", {}).items()
                            if v != "DV"}                       # drop the DV mapping
    sess.state = sess.state.model_copy(update={"dataset_metadata": md})
    r = client.get(f"/api/sessions/{sid}/cdisc")
    assert r.status_code == 409
    assert "unmapped role" in r.json()["error"]["message"]


def test_intact_dataset_rehydrates_and_loads():
    from app.core.store import SessionStore
    store = SessionStore(":memory:")
    sid = _load_session(store, SAMPLE)
    orch2 = _orch(store)                                    # fresh instance rehydrates
    sess = orch2.sessions[sid]
    ds_id = sess.state.dataset_id
    assert ds_id and ds_id in sess.ctx.dataset_store       # df loaded
    assert (sess.state.dataset_metadata or {}).get("dataset_integrity") != "sha256_mismatch"


def test_tampered_dataset_fails_closed_on_rehydration():
    # The importable copy must live under an allowed data root (paths are confined),
    # so drop a uniquely-named copy into data/ and remove it afterward.
    from app.config import settings
    from app.core.store import SessionStore
    ds_file = Path(settings.data_dir) / "tamper_probe_f5.csv"
    ds_file.write_bytes(Path(SAMPLE).read_bytes())         # copy we can mutate
    try:
        store = SessionStore(":memory:")
        sid = _load_session(store, str(ds_file))
        ds_file.write_text(ds_file.read_text() + "\n999,999,999\n")   # tamper post-import
        orch2 = _orch(store)                               # rehydrate after tamper
        sess = orch2.sessions[sid]
        md = sess.state.dataset_metadata or {}
        assert md.get("dataset_integrity") == "sha256_mismatch"        # flagged
        assert sess.state.dataset_id not in sess.ctx.dataset_store     # NOT loaded
    finally:
        ds_file.unlink(missing_ok=True)
