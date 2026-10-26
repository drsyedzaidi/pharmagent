"""Test helpers that satisfy the deliverable QC gates through the REAL tools.

Reports, the Module 2.7.2 document and the briefing memo refuse to build unless NCA
QC is current, every flagged lambda_z fit is accepted by a human analyst, and any
NLME fit has a current PopPK estimation QC verdict. Tests that exercise those
deliverables call these helpers instead of weakening the gate.
"""
from __future__ import annotations

from app.core.audit import AuditChain, hash_payload
from app.core.pharmstate import PharmState
from app.tools.base import ToolContext
from app.tools.builtins import default_registry
from app.tools.qc_tools import is_nca_qc_current

_OPEN = {"awaiting_analyst_decision", "revision_requested"}
_NOTE = "test fixture: analyst accepts flagged terminal fits"


def pass_qc_gates_via_orch(orch, sid: str, actor: str | None = None) -> None:
    """Run run_qc if stale, then record the analyst's accept the way the review endpoint does."""
    state = orch.get_session(sid).state
    if state.nca_parameters and not is_nca_qc_current(state):
        orch.run_tool(sid, "run_qc", "qc", {}, actor=actor)
    if (orch.get_session(sid).state.lambda_z_review or {}).get("status") in _OPEN:
        orch.run_tool(sid, "review_lambda_z_fits", "qc", {"decision": "accept", "notes": _NOTE},
                      actor=actor, allow_human_only=True)


def accept_via_api(client, sid: str, headers=None) -> None:
    """The analyst accepts every flagged lambda_z fit through the HTTP review control."""
    r = client.post(f"/api/sessions/{sid}/lambda-z-review", headers=headers,
                    json={"decision": "accept", "notes": _NOTE})
    assert r.status_code in (200, 400), r.text   # 400 = nothing flagged to review


def qc_gated_state(state: PharmState) -> PharmState:
    """State-level equivalent for direct tool calls: real run_qc + human accept.

    PopPK QC is stamped as reviewed for the fixture fit (digest-bound, as in
    memo_fixtures) because these fixtures are rendering inputs, not healthy fits.
    """
    reg, audit, ctx = default_registry(), AuditChain(), ToolContext()
    if state.nca_parameters:
        state, _ = reg.execute("run_qc", state=state, ctx=ctx, args={}, audit=audit, timestamp="t")
        if (state.lambda_z_review or {}).get("status") in _OPEN:
            state, _ = reg.execute("review_lambda_z_fits", state=state, ctx=ctx,
                                   args={"decision": "accept", "notes": _NOTE}, audit=audit,
                                   timestamp="t", allow_human_only=True)
    nl = state.nlme_results or {}
    if nl.get("status") == "ok":
        state = state.model_copy(update={"poppk_qc_verdict": "PASS",
                                         "poppk_qc_nlme_digest": hash_payload(nl)})
    return state
