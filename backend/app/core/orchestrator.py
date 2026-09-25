"""Orchestrator — sessions, chat routing, and workflow execution.

Holds per-session state (PharmState), the server-side dataset store, and the
audit chain. Drives both the free-chat path (Supervisor routes → agent runs a
tool-use loop) and the deterministic workflow path (steps from a template,
pausing at review gates).
"""
from __future__ import annotations

import logging
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
from app.core.audit_seal import (
    NO_SEAL,
    AuditIntegrityError,
    AuditSecurity,
    audit_security_from_settings,
)
from app.core.llm import LLM, get_llm
from app.core.pharmstate import PharmState, apply_writes
from app.core.provenance import collect_provenance
from app.core.skills import Skill, SkillStore, distill_steps
from app.core.store import SecureWriteConflict, SessionStore
from app.tools.base import ExpensiveToolError, ToolContext, ToolRegistry
from app.tools.builtins import default_registry
from app.tools.data_tools import _read as _read_dataset
from app.workflows import get_workflow

log = logging.getLogger("pharmagent")
_AUDIT_SECURITY_FROM_SETTINGS = object()


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
                 skills: SkillStore | None = None,
                 audit_security: AuditSecurity | None | object =
                 _AUDIT_SECURITY_FROM_SETTINGS) -> None:
        self.llm = llm or get_llm()
        self.registry = registry or default_registry()
        self.supervisor = Supervisor(self.llm)
        self.clock = clock or _iso_clock
        self.store = store if store is not None else SessionStore(str(settings.db_path))
        # Skills live in their own table; an injected in-memory test store gets an
        # in-memory skill store too so tests never touch the real DB file.
        self.skills = skills if skills is not None else SkillStore(
            ":memory:" if store is not None else str(settings.db_path))
        if audit_security is _AUDIT_SECURITY_FROM_SETTINGS:
            # An injected store denotes a hermetic test/dev orchestrator and keeps
            # the historical hash-only behaviour unless security is also injected.
            self.audit_security = (
                None if store is not None else audit_security_from_settings(settings)
            )
        else:
            self.audit_security = audit_security
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
        from app.core.provenance import file_sha256  # local: keep import graph light
        rows = self.store.load_all()
        persisted_ids = {row["id"] for row in rows}
        if self.audit_security is not None:
            orphaned = set(self.audit_security.anchor_heads()) - persisted_ids
            if orphaned:
                raise AuditIntegrityError(
                    "external audit anchor exists for a missing database session"
                )
        for row in rows:
            if self.audit_security is not None:
                self.audit_security.assert_current(
                    row["id"], row["audit"], row["audit_seal"],
                    owner=row["owner"], created_at=row["created_at"], recover=True
                )
            ctx = ToolContext(data_dir=str(settings.data_dir))
            state = PharmState.model_validate(row["state"])
            path, dsid = row.get("dataset_path"), row.get("dataset_id")
            audit = AuditChain.from_list(row["audit"])
            integrity_changed = False
            if path and dsid:
                recorded = (state.dataset_metadata or {}).get("dataset_sha256")
                try:  # re-hydrate the dataset from disk (path is confined)
                    # file_sha256 returns "n/a" for a missing/unreadable file, which
                    # must NOT be reported as tampering — distinguish the two.
                    actual = file_sha256(path) if recorded else None
                    if recorded and actual == "n/a":
                        verdict = "file_missing"
                    elif recorded and actual != recorded:
                        verdict = "sha256_mismatch"
                    else:
                        verdict = None
                    if verdict:
                        # Fail closed: the file no longer matches the recorded
                        # provenance hash (or is gone). Do NOT load it — analyses must
                        # not run on unverifiable data — and flag it for an audited
                        # re-import instead of silently trusting it.
                        state = state.model_copy(update={"dataset_metadata":
                            {**(state.dataset_metadata or {}), "dataset_integrity": verdict}})
                        # Part 11: a detection that leaves no trace is not a control.
                        # Record it once — only on the transition into the failed
                        # state — so a restart loop cannot pad the chain. The
                        # transition is durable (persisted below); otherwise every
                        # restart would re-detect and the finding would never survive.
                        if (row["state"].get("dataset_metadata") or {}).get(
                                "dataset_integrity") != verdict:
                            audit.append(
                                agent="system", tool="dataset_integrity",
                                action=f"integrity_check_failed({verdict})",
                                inputs={"dataset_id": dsid, "recorded_sha256": recorded},
                                outputs={"observed_sha256": actual, "loaded": False},
                                timestamp=self.clock(), actor="system",
                                reason="dataset failed its recorded sha256 on rehydration")
                            log.warning("dataset integrity %s for session %s (%s)",
                                        verdict, row["id"], dsid)
                            integrity_changed = True
                    else:
                        ctx.dataset_store[dsid] = _read_dataset(path)
                        if (state.dataset_metadata or {}).get("dataset_integrity"):
                            # The digest matches again (file restored, or re-imported),
                            # so clear the flag — a sticky failure would otherwise mark
                            # the session tampered forever, and _persist would write
                            # that stale verdict back into stored provenance.
                            md = dict(state.dataset_metadata or {})
                            md.pop("dataset_integrity", None)
                            state = state.model_copy(update={"dataset_metadata": md})
                            integrity_changed = True
                except Exception:
                    pass  # unreadable/corrupt file — session still usable for review/audit
            sess = Session(
                id=row["id"], state=state,
                ctx=ctx, audit=audit,
                history=row["history"], pending_review=row["pending"],
                params=row["params"], owner=row["owner"], created_at=row["created_at"],
                commands=row.get("commands") or [])
            self.sessions[row["id"]] = sess
            if integrity_changed:
                # Write the verdict AND its audit entry back, or the detection lives
                # only in this process: the next restart would re-read the original
                # row, re-detect, and the finding would never become part of the
                # record. Persisting also makes the transition test above correct.
                self._persist(sess)

    def _persist(self, sess: Session) -> None:
        values = {
            "id": sess.id,
            "owner": sess.owner,
            "created_at": sess.created_at,
            "updated_at": self.clock(),
            "state": sess.state.model_dump(),
            "audit": sess.audit.to_list(),
            "history": sess.history,
            "pending": sess.pending_review,
            "params": sess.params,
            "dataset_id": sess.state.dataset_id,
            "dataset_path": sess.state.dataset_path,
            "commands": sess.commands,
        }
        if self.audit_security is None:
            self.store.save(**values)
            return

        row = self.store.get(sess.id)
        previous = None
        expected_audit_json = None
        expected_seal_json = None
        if row is not None:
            previous = self.audit_security.assert_current(
                sess.id, row["audit"], row["audit_seal"],
                owner=row["owner"], created_at=row["created_at"], recover=True
            )
            expected_audit_json = row["_audit_json_raw"]
            expected_seal_json = row["_audit_seal_json_raw"]
            current = row["audit"]
            proposed = values["audit"]
            if len(proposed) < len(current) or proposed[:len(current)] != current:
                raise AuditIntegrityError(
                    "in-memory audit does not extend the sealed persisted audit"
                )
        elif self.audit_security.anchor_head(sess.id) is not None:
            raise AuditIntegrityError(
                "cannot recreate a session whose external anchor already exists"
            )

        proposed_entries = values["audit"]
        if (
            previous is not None
            and proposed_entries == row["audit"]
            and previous.key_id == self.audit_security.keyring.active_key_id
        ):
            seal = previous
        else:
            seal = self.audit_security.create_seal(
                session_id=sess.id,
                owner=sess.owner,
                created_at=sess.created_at,
                entries=proposed_entries,
                previous=previous,
                baseline="fresh",
            )
        try:
            self.store.save_secured(
                **values,
                audit_seal=seal.to_dict(),
                expected_audit_json=expected_audit_json,
                expected_audit_seal_json=expected_seal_json,
            )
        except SecureWriteConflict as exc:
            raise AuditIntegrityError(
                "persisted audit changed during secured save"
            ) from exc
        if previous is None or seal.seal_mac != previous.seal_mac:
            expected_mac = previous.seal_mac if previous is not None else NO_SEAL
            # DB-first is intentional. If this raises, the request must not report
            # success; startup can recover only the exact signed one-generation
            # successor linked to the current external anchor.
            self.audit_security.advance(expected_mac, seal)

    def audit_status(self, sid: str) -> dict:
        """Return an honest structural/MAC/anchor status for the persisted row."""
        sess = self.get_session(sid)
        if self.audit_security is None:
            chain = sess.audit.verify_status()
            return {
                **chain,
                "chain_ok": chain["ok"],
                "mode": "hash_only",
                "mac_ok": False,
                "anchor_ok": False,
                "verified": False,
                "baseline": "unsealed",
                "generation": None,
                "key_id": None,
                "trusted_since_index": None,
            }
        row = self.store.get(sid)
        if row is None:
            return {
                "ok": False,
                "chain_ok": False,
                "mode": "enforced",
                "mac_ok": False,
                "anchor_ok": False,
                "verified": False,
                "baseline": None,
                "generation": None,
                "key_id": None,
                "trusted_since_index": None,
                "legacy_entries": 0,
                "unverifiable_entries": 1,
                "n_entries": 0,
                "degraded": False,
            }
        return self.audit_security.inspect(
            sid, row["audit"], row["audit_seal"],
            owner=row["owner"], created_at=row["created_at"]
        )

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
            pending = None
            if res.proposal is not None:
                # A proposable expensive tool the agent selected: record it for a
                # human decision (tamper-evidently) — nothing was computed.
                ts = self.clock()
                previous = sess.state.pending_tool
                if previous:
                    # An undecided proposal is being replaced: leave a trace, so the
                    # chain shows it was abandoned rather than silently vanishing.
                    sess.audit.append(
                        agent="supervisor", tool=previous["tool"], action="supersede_tool",
                        inputs=previous.get("args") or {},
                        outputs={"status": "superseded", "by": res.proposal["tool"]},
                        timestamp=ts, actor=actor or "")
                pending = {**res.proposal, "proposed_at": ts, "proposed_by": actor or "",
                           "message": message}
                sess.state = apply_writes(sess.state, "supervisor", {"pending_tool": pending})
                sess.audit.append(
                    agent="supervisor", tool=pending["tool"], action="propose_tool",
                    inputs=pending["args"],
                    outputs={"status": "pending_approval", "agent": pending["agent"]},
                    timestamp=ts, actor=actor or "")
            payload = {
                "agent": agent_name, "routed_by": method,
                "messages": res.messages, "tool_calls": res.tool_calls,
                "pending_tool": pending,
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
        # Serialize with every other mutation of this session (like run_tool):
        # this path mutates state, audit, commands and persists, so concurrent
        # requests without the lock can lose updates or persist a torn snapshot.
        with self.session_lock(sid):
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
        with self.session_lock(sid):
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
                # This is the admission-controlled entry: the NLME/SCM/engine
                # endpoints reach it via JobManager.submit, which bounds them.
                allow_expensive=True,
            )
            self._log_command(sess, agent, tool, args or {})
            self._persist(sess)
            return {"agent": agent, "tool": tool, "summary": res.summary,
                    "state": sess.state.model_dump(), "result": res.result,
                    "audit_ok": sess.audit.verify()}

    # -- chat-proposed expensive tools: the human decides ------------------
    def take_pending_tool(self, sid: str, *, actor: str | None = None,
                          reason: str = "") -> dict[str, Any]:
        """Approve ``state.pending_tool``: audit the decision, clear it, and return
        the call to submit ({tool, agent, args}) with ``confirm=True`` — the
        human approval IS the confirm the tool requires.

        The proposal is cleared HERE, before any job is submitted, so a repeated
        approval (double click, retry) finds nothing and cannot double-submit.
        Raises KeyError when nothing is pending.
        """
        with self.session_lock(sid):
            sess = self.get_session(sid)
            pending = sess.state.pending_tool
            if not pending:
                raise KeyError("no pending tool proposal")
            args = {**(pending.get("args") or {}), "confirm": True}
            sess.audit.append(
                agent="supervisor", tool=pending["tool"], action="approve_tool",
                inputs=args, outputs={"status": "approved", "agent": pending["agent"]},
                timestamp=self.clock(), actor=actor or "", reason=reason)
            sess.state = apply_writes(sess.state, "supervisor", {"pending_tool": None})
            self._persist(sess)
            return {"tool": pending["tool"], "agent": pending["agent"], "args": args}

    def reject_pending_tool(self, sid: str, *, actor: str | None = None,
                            reason: str = "") -> dict[str, Any]:
        """Reject ``state.pending_tool``: audited, cleared, nothing computed.
        Raises KeyError when nothing is pending."""
        with self.session_lock(sid):
            sess = self.get_session(sid)
            pending = sess.state.pending_tool
            if not pending:
                raise KeyError("no pending tool proposal")
            sess.audit.append(
                agent="supervisor", tool=pending["tool"], action="reject_tool",
                inputs=pending.get("args") or {}, outputs={"status": "rejected"},
                timestamp=self.clock(), actor=actor or "", reason=reason)
            sess.state = apply_writes(sess.state, "supervisor", {"pending_tool": None})
            self._persist(sess)
            return {"status": "rejected", "tool": pending["tool"],
                    "state": sess.state.model_dump(), "audit_ok": sess.audit.verify()}

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
            # Carry the verdict and its caveats through: "not goal_met" alone cannot
            # distinguish real blocking findings from a review that had no raw data
            # to verify against (UNVERIFIABLE) or nothing to inspect (INCOMPLETE).
            "status": result.get("status", ""),
            "unverifiable": result.get("unverifiable", False),
            "unverifiable_reason": result.get("unverifiable_reason", ""),
            "checked": result.get("checked", {}),
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
        # create_session already registered this id, so a concurrent reader can reach
        # it mid-replay: hold the same per-session lock every other mutator uses.
        with self.session_lock(sess.id):
            # 1. load the new dataset (replay always supplies its own data)
            sess.state, res = self.registry.execute(
                "load_dataset", state=sess.state, ctx=sess.ctx,
                args={"path": dataset_path}, audit=sess.audit,
                timestamp=self.clock(), actor=actor or "")
            self._log_command(sess, "data_manager", "load_dataset", {"path": dataset_path})
            executed.append({"tool": "load_dataset", "status": "ok", "summary": res.summary})
            # 2. replay the captured analysis steps in order. allow_expensive is NOT
            #    passed: a replay runs synchronously here, so a captured NLME/SCM step
            #    is refused (recorded as an error below) rather than re-running a
            #    multi-minute fit outside the job queue.
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
                       actor: str | None = None,
                       allow_expensive: bool = False) -> dict[str, Any]:
        # Held under the per-session lock (re-entrant): the whole advance — audit
        # appends + state RMW, including the long engine step — is serialized
        # against concurrent run_tool / background jobs on the same session.
        with self.session_lock(sid):
            sess = self.get_session(sid)
            wf = get_workflow(name)
            sess.state = apply_writes(sess.state, "supervisor",
                                      {"workflow_name": name, "current_step": 0})
            sess.params = params or {}
            return self._advance(sess, wf, params or {}, actor=actor or "",
                                 allow_expensive=allow_expensive)

    def resume_workflow(self, sid: str, approve: bool = True,
                        actor: str | None = None, reason: str = "",
                        allow_expensive: bool = False) -> dict[str, Any]:
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
            return self._advance(sess, wf, params, actor=actor or "",
                                 allow_expensive=allow_expensive)

    def workflow_needs_job(self, wf: dict[str, Any], from_step: int = 0) -> bool:
        """True if advancing from ``from_step`` would reach a long-running fit.

        Only the steps that would actually run in this leg are considered — a
        template whose expensive steps sit after a gate does not need a job to
        reach that gate. Callers use this to hand the leg to the JobManager
        instead of running it on the request thread.
        """
        for step in wf["steps"][from_step:]:
            try:
                if self.registry.get(step["tool"]).expensive:
                    return True
            except KeyError:
                pass
            if step.get("gate"):
                break        # this leg stops here; later steps are the next leg's
        return False

    def _advance(self, sess: Session, wf: dict[str, Any], params: dict[str, Any],
                 actor: str = "", allow_expensive: bool = False) -> dict[str, Any]:
        steps = wf["steps"]
        executed: list[dict[str, Any]] = []
        i = sess.state.current_step
        while i < len(steps):
            step = steps[i]
            args = dict(step.get("args", {}))
            if step["tool"] == "load_dataset" and "path" not in args and params.get("path"):
                args["path"] = params["path"]
            try:
                sess.state, res = self.registry.execute(
                    step["tool"], state=sess.state, ctx=sess.ctx,
                    args=args, audit=sess.audit, timestamp=self.clock(), actor=actor,
                    # Being human-initiated (or human-approved at a gate) is NOT queue
                    # admission control: a gate says "this analysis is scientifically
                    # right to run", not "there is capacity to run it now". Only the
                    # JobManager path sets this, so a long fit cannot occupy a request
                    # thread or slip past the concurrency caps.
                    allow_expensive=allow_expensive,
                )
            except ExpensiveToolError as exc:
                # Stop cleanly at the expensive step WITHOUT advancing current_step,
                # so the same leg resumes unchanged once it is submitted as a job.
                self._persist(sess)
                return {"status": "awaiting_job", "executed": executed,
                        "next_step": i, "next_tool": step["tool"],
                        "message": f"{exc} Re-issue this workflow leg as a background job.",
                        "state": sess.state.model_dump(),
                        "audit_ok": sess.audit.verify()}
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
