"""Orchestrator — sessions, chat routing, and workflow execution.

Holds per-session state (PharmState), the server-side dataset store, and the
audit chain. Drives both the free-chat path (Supervisor routes → agent runs a
tool-use loop) and the deterministic workflow path (steps from a template,
pausing at review gates).
"""
from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.agents.definitions import AGENTS
from app.agents.supervisor import Supervisor
from app.config import settings
from app.core.audit import AuditChain
from app.core.llm import LLM, get_llm
from app.core.pharmstate import PharmState, apply_writes
from app.core.provenance import collect_provenance
from app.core.skills import Skill, SkillStore, distill_steps
from app.core.store import SessionStore
from app.tools.base import ToolContext, ToolRegistry
from app.tools.builtins import default_registry
from app.tools.data_tools import _read as _read_dataset
from app.workflows import get_workflow


def _iso_clock() -> str:
    return datetime.now(UTC).isoformat()


class AccessError(Exception):
    """Raised when a caller requests a session they do not own."""


@dataclass
class Session:
    id: str
    state: PharmState
    ctx: ToolContext
    audit: AuditChain = field(default_factory=AuditChain)
    history: list[dict[str, Any]] = field(default_factory=list)
    pending_review: dict[str, Any] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    owner: str | None = None
    created_at: str = ""
    # Replayable analysis command log (agent, tool, args) — the capture source for
    # skills. Independent of the audit chain, whose inputs are hashed.
    commands: list[dict[str, Any]] = field(default_factory=list)


class Orchestrator:
    def __init__(self, llm: LLM | None = None, registry: ToolRegistry | None = None,
                 clock: Callable[[], str] | None = None,
                 store: SessionStore | None = None,
                 skills: SkillStore | None = None) -> None:
        self.llm = llm or get_llm()
        self.registry = registry or default_registry()
        self.supervisor = Supervisor(self.llm)
        self.clock = clock or _iso_clock
        self.store = store if store is not None else SessionStore(str(settings.db_path))
        # Skills live in their own table; an injected in-memory test store gets an
        # in-memory skill store too so tests never touch the real DB file.
        self.skills = skills if skills is not None else SkillStore(
            ":memory:" if store is not None else str(settings.db_path))
        self.sessions: dict[str, Session] = {}
        # Per-session re-entrant locks serialize mutations (and let readers take a
        # consistent snapshot) when background jobs run a tool off the request
        # thread. See app/core/jobs.py.
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()
        self._load_persisted()

    def session_lock(self, sid: str) -> threading.RLock:
        """Return (creating if needed) the per-session mutation lock."""
        with self._locks_guard:
            lock = self._locks.get(sid)
            if lock is None:
                lock = threading.RLock()
                self._locks[sid] = lock
            return lock

    # -- persistence -------------------------------------------------------
    def _load_persisted(self) -> None:
        for row in self.store.load_all():
            ctx = ToolContext(data_dir=str(settings.data_dir))
            if row.get("dataset_path") and row.get("dataset_id"):
                try:  # re-hydrate the dataset from disk (path is confined)
                    ctx.dataset_store[row["dataset_id"]] = _read_dataset(row["dataset_path"])
                except Exception:
                    pass  # dataset file gone — session still usable for review/audit
            self.sessions[row["id"]] = Session(
                id=row["id"], state=PharmState.model_validate(row["state"]),
                ctx=ctx, audit=AuditChain.from_list(row["audit"]),
                history=row["history"], pending_review=row["pending"],
                params=row["params"], owner=row["owner"], created_at=row["created_at"],
                commands=row.get("commands") or [])

    def _persist(self, sess: Session) -> None:
        self.store.save(
            id=sess.id, owner=sess.owner, created_at=sess.created_at,
            updated_at=self.clock(), state=sess.state.model_dump(),
            audit=sess.audit.to_list(), history=sess.history,
            pending=sess.pending_review, params=sess.params,
            dataset_id=sess.state.dataset_id, dataset_path=sess.state.dataset_path,
            commands=sess.commands)

    @staticmethod
    def _log_command(sess: Session, agent: str, tool: str, args: dict[str, Any]) -> None:
        """Record a replayable analysis command (skill capture source)."""
        sess.commands.append({"agent": agent, "tool": tool, "args": dict(args or {})})

    # -- sessions ----------------------------------------------------------
    def create_session(self, owner: str | None = None) -> Session:
        sid = f"sess_{uuid.uuid4().hex[:8]}"
        sess = Session(id=sid, state=PharmState(session_id=sid),
                       ctx=ToolContext(data_dir=str(settings.data_dir)),
                       owner=owner, created_at=self.clock())
        # Genesis provenance entry: stamp the software/platform fingerprint into
        # the audit chain so every result traces to the run that produced it.
        sess.audit.append(agent="system", tool="session", action="session_created",
                          inputs=collect_provenance(), outputs={"session_id": sid},
                          timestamp=self.clock(), actor=owner or "anonymous")
        self.sessions[sid] = sess
        self._persist(sess)
        return sess

    def get_session(self, sid: str, owner: str | None = None) -> Session:
        if sid not in self.sessions:
            raise KeyError(f"unknown session: {sid}")
        sess = self.sessions[sid]
        if owner is not None and sess.owner is not None and sess.owner != owner:
            raise AccessError(f"session {sid} belongs to another user")
        return sess

    # -- free chat ---------------------------------------------------------
    def chat(self, sid: str, message: str, actor: str | None = None) -> dict[str, Any]:
        # Mutates state, the hash-linked audit chain, history, and persistence —
        # take the per-session lock like every other mutating path (RLock, so the
        # nested get_session is fine), or concurrent chats can corrupt state/audit.
        with self.session_lock(sid):
            sess = self.get_session(sid)
            agent_name, method = self.supervisor.route(message)
            sess.state = apply_writes(sess.state, "supervisor", {"last_agent": agent_name})
            agent = AGENTS[agent_name]
            res = agent.run_turn(
                state=sess.state, message=message, llm=self.llm,
                registry=self.registry, ctx=sess.ctx, audit=sess.audit, clock=self.clock,
                actor=actor or "",
            )
            sess.state = res.state
            payload = {
                "agent": agent_name, "routed_by": method,
                "messages": res.messages, "tool_calls": res.tool_calls,
                "state": sess.state.model_dump(),
            }
            sess.history.append({"role": "user", "content": message})
            sess.history.append({"role": "assistant", **payload})
            self._persist(sess)
            return payload

    # -- direct PK model fit (UI model picker bypasses NL routing) ---------
    def run_pk_model(self, sid: str, *, model_key: str | None = None,
                     compare: bool = False, models: list[str] | None = None,
                     actor: str | None = None) -> dict[str, Any]:
        sess = self.get_session(sid)
        sess.state = apply_writes(sess.state, "supervisor", {"last_agent": "modeler"})
        args: dict[str, Any] = {}
        if model_key:
            args["model_key"] = model_key
        if compare:
            args["compare"] = True
        if models:
            args["models"] = models
        sess.state, res = self.registry.execute(
            "fit_pk_model", state=sess.state, ctx=sess.ctx,
            args=args, audit=sess.audit, timestamp=self.clock(), actor=actor or "",
        )
        self._log_command(sess, "modeler", "fit_pk_model", args)
        self._persist(sess)
        return {"agent": "modeler", "tool": "fit_pk_model", "summary": res.summary,
                "state": sess.state.model_dump(), "result": res.result,
                "audit_ok": sess.audit.verify()}

    def simulate_pk(self, sid: str, params: dict[str, Any],
                    actor: str | None = None) -> dict[str, Any]:
        sess = self.get_session(sid)
        sess.state = apply_writes(sess.state, "supervisor", {"last_agent": "simulator"})
        sess.state, res = self.registry.execute(
            "simulate_pk_profile", state=sess.state, ctx=sess.ctx,
            args=params or {}, audit=sess.audit, timestamp=self.clock(), actor=actor or "",
        )
        self._log_command(sess, "simulator", "simulate_pk_profile", params or {})
        self._persist(sess)
        return {"agent": "simulator", "tool": "simulate_pk_profile", "summary": res.summary,
                "state": sess.state.model_dump(), "result": res.result,
                "audit_ok": sess.audit.verify()}

    def run_tool(self, sid: str, tool: str, agent: str, args: dict[str, Any],
                 actor: str | None = None) -> dict[str, Any]:
        """Execute a single tool directly (UI-driven diagnostics) with audit + writes.

        Held under the per-session lock so a tool run on a background job thread
        cannot interleave with another mutation or an inconsistent read.
        """
        with self.session_lock(sid):
            sess = self.get_session(sid)
            sess.state = apply_writes(sess.state, "supervisor", {"last_agent": agent})
            sess.state, res = self.registry.execute(
                tool, state=sess.state, ctx=sess.ctx, args=args or {},
                audit=sess.audit, timestamp=self.clock(), actor=actor or "",
            )
            self._log_command(sess, agent, tool, args or {})
            self._persist(sess)
            return {"agent": agent, "tool": tool, "summary": res.summary,
                    "state": sess.state.model_dump(), "result": res.result,
                    "audit_ok": sess.audit.verify()}

    def review_loop(self, sid: str, *, goal: str | None = None, max_iter: int = 3,
                    actor: str | None = None) -> dict[str, Any]:
        """Run the adversarial reviewer in a loop until the checkable goal is met
        (or no further progress / max_iter). Each pass is an audited tool run.

        The compute is deterministic, so without a remediation that changes inputs a
        second pass yields the same findings — the loop therefore stops as soon as the
        goal is met OR a pass produces no fewer findings than the previous one (no
        progress). Scientific findings escalate to the human of record rather than
        being auto-resolved; the loop reports the converged finding set and verdict.
        """
        passes: list[dict[str, Any]] = []
        prev_n: int | None = None
        last: dict[str, Any] = {}
        for _ in range(max(1, max_iter)):
            out = self.run_tool(sid, "adversarial_review", "reviewer",
                                {"goal": goal} if goal else {}, actor=actor)
            res = out.get("result") or {}
            last = out
            passes.append({"n_findings": res.get("n_findings"),
                           "counts": res.get("counts"), "goal_met": res.get("goal_met")})
            n = res.get("n_findings")
            if res.get("goal_met") or (prev_n is not None and n is not None and n >= prev_n):
                break
            prev_n = n
        result = last.get("result") or {}
        return {
            "goal": result.get("goal", goal or ""),
            "goal_met": result.get("goal_met", False),
            "iterations": len(passes),
            "passes": passes,
            "findings": result.get("findings", []),
            "counts": result.get("counts", {}),
            "state": last.get("state"),
            "audit_ok": last.get("audit_ok"),
        }

    # -- skills (captured, replayable workflows) ---------------------------
    def capture_skill(self, sid: str, name: str, *, description: str = "",
                      goal: str = "", actor: str | None = None) -> dict[str, Any]:
        """Distill a session's command log into a named, replayable skill."""
        with self.session_lock(sid):
            sess = self.get_session(sid)
            steps = distill_steps(sess.commands)
            if not steps:
                raise ValueError("nothing to capture: run an analysis on this session first")
            skill = Skill(name=name, description=description, goal=goal, steps=steps,
                          source_session=sid, owner=sess.owner,
                          created_at=self.clock(), version=1)
            self.skills.save(skill)
            sess.audit.append(agent="reviewer", tool="capture_skill",
                              action=f"captured skill '{name}' ({len(steps)} steps)",
                              inputs={"name": name, "n_steps": len(steps)},
                              outputs={"steps": [s["tool"] for s in steps]},
                              timestamp=self.clock(), actor=actor or "anonymous")
            self._persist(sess)
            saved = self.skills.get(name)
            return {"skill": saved.to_dict() if saved else skill.to_dict()}

    def run_skill(self, name: str, *, dataset_path: str, owner: str | None = None,
                  actor: str | None = None) -> dict[str, Any]:
        """Replay a captured skill on a new dataset in a fresh session."""
        skill = self.skills.get(name)
        if skill is None:
            raise KeyError(f"unknown skill: {name}")
        sess = self.create_session(owner=owner)
        executed: list[dict[str, Any]] = []
        # 1. load the new dataset (replay always supplies its own data)
        sess.state, res = self.registry.execute(
            "load_dataset", state=sess.state, ctx=sess.ctx,
            args={"path": dataset_path}, audit=sess.audit,
            timestamp=self.clock(), actor=actor or "")
        self._log_command(sess, "data_manager", "load_dataset", {"path": dataset_path})
        executed.append({"tool": "load_dataset", "status": "ok", "summary": res.summary})
        # 2. replay the captured analysis steps in order
        for step in skill.steps:
            try:
                sess.state, res = self.registry.execute(
                    step["tool"], state=sess.state, ctx=sess.ctx,
                    args=dict(step.get("args") or {}), audit=sess.audit,
                    timestamp=self.clock(), actor=actor or "")
                self._log_command(sess, step.get("agent", ""), step["tool"],
                                  step.get("args") or {})
                executed.append({"tool": step["tool"], "status": "ok",
                                 "summary": res.summary})
            except Exception as exc:  # a step failed — record and continue the trail
                executed.append({"tool": step["tool"], "status": "error",
                                 "error": str(exc)})
        self._persist(sess)
        return {"skill": name, "session_id": sess.id, "executed": executed,
                "state": sess.state.model_dump(), "audit_ok": sess.audit.verify()}

    def set_roles(self, sid: str, overrides: dict[str, str],
                  actor: str | None = None, reason: str = "") -> dict[str, Any]:
        """Apply user column-role overrides to the dataset metadata so every
        downstream tool uses the corrected mapping. Recorded in the audit chain."""
        with self.session_lock(sid):
            sess = self.get_session(sid)
            meta = dict(sess.state.dataset_metadata or {})
            roles = dict(meta.get("detected_roles") or {})
            for col, role in (overrides or {}).items():
                if role:
                    roles[col] = role
                else:
                    roles.pop(col, None)
            meta["detected_roles"] = roles
            sess.state = apply_writes(sess.state, "data_manager", {"dataset_metadata": meta})
            sess.audit.append(agent="data_manager", tool="set_roles",
                              action="column role override", inputs=overrides,
                              outputs=roles, timestamp=self.clock(),
                              actor=actor or "anonymous", reason=reason)
            self._persist(sess)
            return {"detected_roles": roles, "state": sess.state.model_dump()}

    # -- workflows ---------------------------------------------------------
    def start_workflow(self, sid: str, name: str, params: dict[str, Any] | None = None,
                       actor: str | None = None) -> dict[str, Any]:
        # Held under the per-session lock (re-entrant): the whole advance — audit
        # appends + state RMW, including the long engine step — is serialized
        # against concurrent run_tool / background jobs on the same session.
        with self.session_lock(sid):
            sess = self.get_session(sid)
            wf = get_workflow(name)
            sess.state = apply_writes(sess.state, "supervisor",
                                      {"workflow_name": name, "current_step": 0})
            sess.params = params or {}
            return self._advance(sess, wf, params or {}, actor=actor or "")

    def resume_workflow(self, sid: str, approve: bool = True,
                        actor: str | None = None, reason: str = "") -> dict[str, Any]:
        with self.session_lock(sid):
            sess = self.get_session(sid)
            if not sess.pending_review:
                raise ValueError("no pending review to resume")
            wf = get_workflow(sess.state.workflow_name)
            params = sess.params
            review = dict(sess.pending_review)
            # Signed audit entry for the human review decision (Part 11 e-signature):
            # the approval/rejection itself is recorded, with actor + reason.
            sess.audit.append(
                agent="qc", tool="human_review",
                action=("approved" if approve else "rejected")
                + f" after step {review.get('after_step')}",
                inputs={"approve": approve, "review": review},
                outputs={"workflow": sess.state.workflow_name},
                timestamp=self.clock(), actor=actor or "anonymous", reason=reason)
            if not approve:
                sess.pending_review = None
                self._persist(sess)
                return {"status": "rejected", "at_step": sess.state.current_step,
                        "state": sess.state.model_dump(), "audit_ok": sess.audit.verify()}
            sess.pending_review = None
            return self._advance(sess, wf, params, actor=actor or "")

    def _advance(self, sess: Session, wf: dict[str, Any], params: dict[str, Any],
                 actor: str = "") -> dict[str, Any]:
        steps = wf["steps"]
        executed: list[dict[str, Any]] = []
        i = sess.state.current_step
        while i < len(steps):
            step = steps[i]
            args = dict(step.get("args", {}))
            if step["tool"] == "load_dataset" and "path" not in args and params.get("path"):
                args["path"] = params["path"]
            sess.state, res = self.registry.execute(
                step["tool"], state=sess.state, ctx=sess.ctx,
                args=args, audit=sess.audit, timestamp=self.clock(), actor=actor,
            )
            self._log_command(sess, step.get("agent", ""), step["tool"], args)
            executed.append({"step": i, "label": step.get("label", step["tool"]),
                             "tool": step["tool"], "summary": res.summary})
            i += 1
            sess.state = apply_writes(sess.state, "supervisor", {"current_step": i})
            if step.get("gate"):
                sess.pending_review = {"after_step": i - 1, "label": step.get("label")}
                self._persist(sess)
                return {"status": "awaiting_review", "executed": executed,
                        "review": sess.pending_review, "state": sess.state.model_dump(),
                        "audit_ok": sess.audit.verify()}
        self._persist(sess)
        return {"status": "complete", "executed": executed,
                "state": sess.state.model_dump(), "audit_ok": sess.audit.verify(),
                "audit": sess.audit.to_list()}
