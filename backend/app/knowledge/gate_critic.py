"""mikebrain at the planner's human gate (M3) — advisory only.

When the goal planner parks a GATED proposal for a human (``Orchestrator._propose_goal_step``), this
module asks mikebrain whether the owner's literature supports, contradicts or says nothing about the
choice, and returns a compact ``critic`` block that rides on ``state.pending_tool`` for the human to read.

What it is not: a gate. It runs AFTER the deterministic policy has already decided "gated"; it cannot
change that decision, the proposal's tool or arguments, or what the human may do, and any failure
degrades to ``stance: "unavailable"`` while the proposal is parked exactly as before. The consultation
itself goes through ``registry.execute("consult_mikebrain")``, the same audited choke point as every
other tool, so it is an audit entry of agent ``reviewer`` that precedes the planner's ``propose_tool``
entry, and the block is repeated in that entry's outputs so the advisory the human saw is bound to the
proposal.

Off unless ``settings.mikebrain_gate_critic``; with it on but no corpus, no LLM call is made and the
block says ``not_consulted``. See docs/mikebrain_planner_gate.md.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from app.config import settings
from app.core.audit import AuditChain
from app.core.pharmstate import PharmState
from app.knowledge.manifest import ManifestError, load_manifest
from app.tools.base import ToolContext, ToolRegistry
from app.tools.knowledge_tools import knowledge_root, redact_subjects

log = logging.getLogger("pharmagent")

SOURCE = "mikebrain"
TOOL = "consult_mikebrain"
MAX_ARGS_CHARS = 400
MAX_TEXT_CHARS = 400
MAX_RATIONALE_CHARS = 600
#: what the human is told about the block, verbatim, every time
ADVISORY_NOTE = ("Advisory only: literature evidence from the owner's private corpus. It does not change the "
                 "policy decision, the gate, or the options of the human who decides.")
#: stance -> meaning for the reader at the gate
STANCES: dict[str, str] = {
    "objection": "the literature contradicts this choice (verbatim citations attached)",
    "corroboration": "the literature supports this choice (verbatim citations attached)",
    "no_evidence": "the corpus was consulted and says nothing decisive either way",
    "not_consulted": "mikebrain was not consulted (no corpus)",
    "unavailable": "mikebrain could not answer (see status); nothing was decided by its absence",
}


def enabled() -> bool:
    return bool(settings.mikebrain_gate_critic)


def gate_question(tool_name: str, args: dict[str, Any], rationale: str, gate_reason: str, goal: str) -> str:
    """The literature question for one gated proposal. Planner text is stripped of subject identifiers; the
    arguments are configuration (model key, method, alpha), never data, and are size-capped."""
    try:
        shown = json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        shown = "{}"
    return (
        f"A planner working toward the goal \"{redact_subjects(goal)[:MAX_TEXT_CHARS]}\" proposes to run "
        f"{tool_name} with arguments {shown[:MAX_ARGS_CHARS]}. The policy gated it because: "
        f"{redact_subjects(gate_reason)[:MAX_TEXT_CHARS]}. The planner's stated reason: "
        f"\"{redact_subjects(rationale)[:MAX_TEXT_CHARS]}\". Does the pharmacometrics literature support, "
        f"contradict, or not address this choice for the current analysis?")


def stance_of(record: dict[str, Any]) -> str:
    """objection / corroboration only for a grounded verdict: ``consult`` already downgrades a verdict
    without a valid verbatim citation to no_evidence, and this refuses to trust that without checking."""
    if record.get("status") != "ok":
        return "unavailable"
    cited = bool(record.get("citations"))
    if record.get("verdict") == "contradicted" and cited:
        return "objection"
    if record.get("verdict") == "supported" and cited:
        return "corroboration"
    return "no_evidence"


def _block(stance: str, **fields: Any) -> dict[str, Any]:
    return {"source": SOURCE, "advisory": True, "stance": stance, "meaning": STANCES[stance],
            "note": ADVISORY_NOTE, **fields}


def critic_block(record: dict[str, Any]) -> dict[str, Any]:
    """The compact view of a ``consult_mikebrain`` record that rides on the pending proposal."""
    return _block(
        stance_of(record), status=record.get("status"), verdict=record.get("verdict"),
        grounding=record.get("grounding"), rationale=str(record.get("rationale") or "")[:MAX_RATIONALE_CHARS],
        citations=list(record.get("citations") or []), corpus_version=record.get("corpus_version"),
        chunk_hashes=list(record.get("chunk_hashes") or []), provider=record.get("provider"),
        model=record.get("model"), elapsed_s=record.get("elapsed_s"))


def critique_proposal(registry: ToolRegistry, state: PharmState, ctx: ToolContext, audit: AuditChain, *,
                      tool_name: str, args: dict[str, Any], rationale: str, gate_reason: str, goal: str,
                      clock: Callable[[], str], actor: str = "") -> tuple[PharmState, dict[str, Any] | None]:
    """``(state, critic)`` for one gated planner proposal.

    ``critic`` is None while the feature is off (nothing happens: no clock tick, no audit entry, no LLM
    call, ``state`` returned as given). Otherwise the state returned is ``state`` plus the consultation's
    ``knowledge_results`` record, and ``critic`` is the block for ``pending_tool``. Never raises: any
    failure is logged and reported as ``unavailable`` with ``state`` unchanged.
    """
    if not enabled():
        return state, None
    try:
        root = knowledge_root(ctx)
        if load_manifest(root).is_empty:
            return state, _block("not_consulted", status="no_corpus")
        new_state, res = registry.execute(
            TOOL, state=state, ctx=ctx, audit=audit, timestamp=clock(), actor=actor,
            args={"question": gate_question(tool_name, args, rationale, gate_reason, goal),
                  "context": {"proposed_tool": tool_name, "gated_by": "planner policy"}})
        return new_state, critic_block(res.result)
    except ManifestError as exc:
        log.warning("mikebrain_gate_critic_corpus_error", extra={"error": str(exc)[:300]})
        return state, _block("unavailable", status="corpus_error")
    except Exception as exc:  # noqa: BLE001 - an advisory must never stop a proposal reaching the human
        log.warning("mikebrain_gate_critic_failed", extra={"error": f"{type(exc).__name__}: {str(exc)[:300]}"})
        return state, _block("unavailable", status="error")
