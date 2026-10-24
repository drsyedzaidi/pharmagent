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

from app.agents.base import Agent
from app.agents.definitions import AGENTS
from app.agents.supervisor import Supervisor
from app.config import settings
from app.core import policy
from app.core.audit import AuditChain
from app.core.audit_seal import (
    NO_SEAL,
    AuditIntegrityError,
    AuditSecurity,
    audit_security_from_settings,
)
from app.core.llm import LLM, get_llm
from app.core.pharmstate import PharmState, apply_writes
from app.core.provenance import (
    ProvenanceQuarantined,
    collect_provenance,
    genesis_outputs,
    verify_run_provenance,
)
from app.core.skills import Skill, SkillStore, distill_steps
from app.core.store import SecureWriteConflict, SessionStore
from app.tools.base import ExpensiveToolError, ToolContext, ToolRegistry
from app.tools.builtins import default_registry
from app.tools.data_tools import _read as _read_dataset
from app.workflows import get_workflow

log = logging.getLogger("pharmagent")
_AUDIT_SECURITY_FROM_SETTINGS = object()
# The planner reads the same compact, privacy-safe diagnostics view the chat agents
# do (Agent._state_summary); it owns no tools and never executes anything itself.
_PLANNER_VIEW = Agent(name="planner", system_prompt="")


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
    # Keyed-seal baseline of this session ("fresh" = sealed since creation, "operator_attested_legacy" = enrolled
    # later); None in hash-only mode. Read from the seal at creation/restore; drives run_provenance_assurance.
    seal_baseline: str | None = None
    # In-memory incremental chain-verification state (never persisted): see provenance.chain_status.
    chain_cache: dict[str, Any] = field(default_factory=dict)


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

    def set_llm(self, llm: LLM) -> None:
        """Swap the routing/tool-selection model at runtime (UI provider switch).
        Tools, state and audit are untouched — the LLM only decides what to call."""
        self.llm = llm
        self.supervisor = Supervisor(llm)

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
            seal_baseline = None
            if self.audit_security is not None:
                seal_baseline = self.audit_security.inspect(
                    row["id"], row["audit"], row["audit_seal"], owner=row["owner"], created_at=row["created_at"]
                ).get("baseline")
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
                commands=row.get("commands") or [], seal_baseline=seal_baseline)
            self._refresh_provenance_status(sess)
            self.sessions[row["id"]] = sess
            if integrity_changed:
                # Write the verdict AND its audit entry back, or the detection lives
                # only in this process: the next restart would re-read the original
                # row, re-detect, and the finding would never become part of the
                # record. Persisting also makes the transition test above correct.
                self._persist(sess, system_record=True)

    def _persist(self, sess: Session, *, system_record: bool = False) -> None:
        if not system_record:
            self._ensure_writable(sess)   # quarantined sessions are read-only (system integrity records excepted)
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
        prov_status = sess.state.run_provenance_status
        prov_assurance = sess.state.run_provenance_assurance
        if self.audit_security is None:
            chain = sess.audit.verify_status()
            return {
                "run_provenance_status": prov_status, "run_provenance_assurance": prov_assurance,
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
                "run_provenance_status": prov_status, "run_provenance_assurance": prov_assurance,
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
        return {**self.audit_security.inspect(
            sid, row["audit"], row["audit_seal"],
            owner=row["owner"], created_at=row["created_at"]
        ), "run_provenance_status": prov_status, "run_provenance_assurance": prov_assurance}

    @staticmethod
    def _log_command(sess: Session, agent: str, tool: str, args: dict[str, Any]) -> None:
        """Record a replayable analysis command (skill capture source)."""
        sess.commands.append({"agent": agent, "tool": tool, "args": dict(args or {})})

    # -- sessions ----------------------------------------------------------
    def create_session(self, owner: str | None = None) -> Session:
        sid = f"sess_{uuid.uuid4().hex[:8]}"
        prov = collect_provenance()     # recorded once, at creation: kept in state and bound into the genesis entry
        sess = Session(id=sid, state=PharmState(session_id=sid, run_provenance=tuple(prov.items())),
                       ctx=ToolContext(data_dir=str(settings.data_dir)),
                       owner=owner, created_at=self.clock(),
                       seal_baseline="fresh" if self.audit_security is not None else None)
        # Genesis provenance entry: stamp the software/platform fingerprint into
        # the audit chain so every result traces to the run that produced it.
        sess.audit.append(agent="system", tool="session", action="session_created",
                          inputs=prov, outputs=genesis_outputs(sid),
                          timestamp=self.clock(), actor=owner or "anonymous")
        self._refresh_provenance_status(sess)
        self.sessions[sid] = sess
        self._persist(sess)
        return sess

    def get_session(self, sid: str, owner: str | None = None) -> Session:
        if sid not in self.sessions:
            raise KeyError(f"unknown session: {sid}")
        sess = self.sessions[sid]
        if owner is not None and sess.owner is not None and sess.owner != owner:
            raise AccessError(f"session {sid} belongs to another user")
        with self.session_lock(sid):      # the refresh writes sess.state: serialise with mutating paths (RLock)
            self._refresh_provenance_status(sess)
        return sess

    def _refresh_provenance_status(self, sess: Session) -> None:
        """Recompute run_provenance_status from the whole audit chain. Never trusts a stored value."""
        status = verify_run_provenance(sess.state.run_provenance, sess.audit, sess.id, cache=sess.chain_cache)
        if self.audit_security is None:
            assurance = "hash_only"
        else:           # what THIS session's seal covers, not merely that keyed mode is on
            assurance = {"fresh": "keyed_seal",
                         "operator_attested_legacy": "keyed_seal_attested_legacy"}.get(sess.seal_baseline or "",
                                                                                        "keyed_seal_unconfirmed")
        if (sess.state.run_provenance_status, sess.state.run_provenance_assurance) != (status, assurance):
            sess.state = sess.state.model_copy(update={"run_provenance_status": status,
                                                       "run_provenance_assurance": assurance})

    @staticmethod
    def _ensure_writable(sess: Session) -> None:
        if sess.state.run_provenance_status == "mismatch":
            raise ProvenanceQuarantined(
                f"session {sess.id} is quarantined: its run provenance does not match its audit record "
                "(run_provenance_status=mismatch). Audit inspection and raw state remain available; tools, state "
                "writes and trusted exports are refused.")

    # -- free chat ---------------------------------------------------------
    def chat(self, sid: str, message: str, actor: str | None = None) -> dict[str, Any]:
        # Mutates state, the hash-linked audit chain, history, and persistence —
        # take the per-session lock like every other mutating path (RLock, so the
        # nested get_session is fine), or concurrent chats can corrupt state/audit.
        with self.session_lock(sid):
            sess = self.get_session(sid)
            self._ensure_writable(sess)
            primary_agent, primary_method = self.supervisor.route(message)
            plan = self.supervisor.route_plan(message)
            all_messages: list[str] = []
            all_tool_calls: list[dict[str, Any]] = []
            agents_executed: list[str] = []
            observations: list[dict[str, Any]] = []
            pending = None

            for agent_name, _ in plan:
                sess.state = apply_writes(sess.state, "supervisor", {"last_agent": agent_name})
                agent = AGENTS[agent_name]
                res = agent.run_turn(
                    state=sess.state, message=message, llm=self.llm,
                    registry=self.registry, ctx=sess.ctx, audit=sess.audit, clock=self.clock,
                    actor=actor or "", prior_observations=observations,
                )
                sess.state = res.state
                agents_executed.append(agent_name)
                for tc in res.tool_calls:
                    if "summary" in tc:
                        self._log_command(sess, tc.get("agent", agent_name), tc["tool"], tc.get("args") or {})
                        observations.append({"agent": agent_name, "tool": tc["tool"],
                                             "status": "ok", "summary": tc["summary"]})
                # For multi-agent handoffs, only append idle_hint if no tool ran across the whole turn
                ran_or_errored = bool(res.tool_calls) or res.proposal is not None
                if ran_or_errored or len(plan) == 1:
                    all_messages.extend(res.messages)
                all_tool_calls.extend(res.tool_calls)
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
                    break
                if any("error" in tc or "skipped" in tc for tc in res.tool_calls):
                    break

            if not all_messages:
                from app.agents.base import idle_hint
                all_messages.append(idle_hint(primary_agent, sess.state))

            payload = {
                "agent": primary_agent, "routed_by": primary_method,
                "agents_executed": agents_executed,
                "messages": all_messages, "tool_calls": all_tool_calls,
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
            self._ensure_writable(sess)
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
            self._ensure_writable(sess)
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
            self._ensure_writable(sess)
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
            self._ensure_writable(sess)
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
            self._ensure_writable(sess)
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
                    actor: str | None = None,
                    auto_remediate: bool = False) -> dict[str, Any]:
        """Run the adversarial reviewer in a loop until the checkable goal is met
        (or no further progress / max_iter). Each pass is an audited tool run.

        Scientific findings (e.g. non-convergence, implausible CL/F, tampered
        untraced numbers) ALWAYS escalate to the pharmacometrician of record.
        When ``auto_remediate=True``, deterministic non-scientific blockers
        (such as ``INCOMPLETE`` when a dataset is loaded but NCA has not yet
        run, or ``memo-stale`` when ``build_briefing_memo`` purely needs a
        rebuild after an updated analysis) are remediated via audited tool
        execution between review passes.
        """
        passes: list[dict[str, Any]] = []
        remediations: list[dict[str, Any]] = []
        prev_n: int | None = None
        last: dict[str, Any] = {}
        for _ in range(max(1, max_iter)):
            out = self.run_tool(sid, "adversarial_review", "reviewer",
                                {"goal": goal} if goal else {}, actor=actor)
            res = out.get("result") or {}
            last = out
            passes.append({"n_findings": res.get("n_findings"),
                           "counts": res.get("counts"), "goal_met": res.get("goal_met"),
                           "status": res.get("status")})
            n = res.get("n_findings")
            if res.get("goal_met"):
                break
            remediated_this_pass = False
            if auto_remediate:
                sess = self.get_session(sid)
                self._ensure_writable(sess)
                finding_ids = {f.get("id") for f in (res.get("findings") or [])}
                # 1. INCOMPLETE with a loaded dataset: run deterministic NCA
                if res.get("status") == "INCOMPLETE" and sess.state.dataset_id:
                    self.run_tool(sid, "compute_nca", "nca", {}, actor=actor)
                    remediations.append({"tool": "compute_nca", "reason": "status=INCOMPLETE"})
                    remediated_this_pass = True
                # 2. Purely stale briefing memo (no manually injected untraced numbers):
                #    rebuild the memo deterministically from the current state + audit trail.
                elif "memo-stale" in finding_ids and "memo-untraced-numbers" not in finding_ids:
                    self.run_tool(sid, "build_briefing_memo", "report", {}, actor=actor)
                    remediations.append({"tool": "build_briefing_memo", "reason": "finding=memo-stale"})
                    remediated_this_pass = True
            if not remediated_this_pass and (prev_n is not None and n is not None and n >= prev_n):
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
            "remediations": remediations,
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
            self._ensure_writable(sess)
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
            self._ensure_writable(sess)
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
            self._ensure_writable(sess)
            wf = get_workflow(name)
            sess.state = apply_writes(sess.state, "supervisor",
                                      {"workflow_name": name, "current_step": 0,
                                       "planner_history": None})
            sess.params = params or {}
            return self._advance(sess, wf, params or {}, actor=actor or "",
                                 allow_expensive=allow_expensive)

    def resume_workflow(self, sid: str, approve: bool = True,
                        actor: str | None = None, reason: str = "",
                        allow_expensive: bool = False) -> dict[str, Any]:
        with self.session_lock(sid):
            sess = self.get_session(sid)
            self._ensure_writable(sess)
            if not sess.pending_review:
                raise ValueError("no pending review to resume")
            wf = get_workflow(sess.state.workflow_name)
            params = sess.params
            review = dict(sess.pending_review)
            if wf.get("kind") == "goal" and approve:
                blocked = self._goal_resume_blocker(sess, allow_expensive=allow_expensive)
                if blocked is not None:
                    return blocked
            # Signed audit entry for the human review decision (Part 11 e-signature):
            # the approval/rejection itself is recorded, with actor + reason.
            sess.audit.append(
                agent="qc", tool="human_review",
                action=("approved" if approve else "rejected")
                + f" after step {review.get('after_step')}",
                inputs={"approve": approve, "review": review},
                outputs={"workflow": sess.state.workflow_name},
                timestamp=self.clock(), actor=actor or "anonymous", reason=reason)
            if wf.get("kind") == "goal":
                return self._resume_goal(sess, wf, params, approve=approve, actor=actor or "",
                                         reason=reason, allow_expensive=allow_expensive)
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

        A goal template has no step list. Its policy gates every long-running
        fit, so a goal leg never EXECUTES one before a human approves it: the
        start leg (from_step 0) always runs inline. A resume leg begins by
        executing the approved proposal, which may be that fit, so it is handed
        to the queue whenever the template's allow-list can reach one.
        """
        if wf.get("kind") == "goal":
            if from_step == 0:
                return False
            for name in wf.get("allowed_tools", ()):
                try:
                    if self.registry.get(name).expensive:
                        return True
                except KeyError:
                    pass
            return False
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
        if wf.get("kind") == "goal":
            return self._advance_goal(sess, wf, params, actor=actor,
                                      allow_expensive=allow_expensive)
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
                             "tool": step["tool"], "agent": step.get("agent", ""),
                             "summary": res.summary})
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

    # -- goal-driven leg: planner -> policy -> registry ---------------------
    def _goal_payload(self, sess: Session, status: str, executed: list[dict[str, Any]],
                      **extra: Any) -> dict[str, Any]:
        self._persist(sess)
        return {"status": status, "executed": executed, **extra,
                "state": sess.state.model_dump(), "audit_ok": sess.audit.verify()}

    def _record_plan(self, sess: Session, record: dict[str, Any]) -> None:
        history = list(sess.state.planner_history or [])
        sess.state = apply_writes(sess.state, "supervisor",
                                  {"planner_history": [*history, record]})

    def _advance_goal(self, sess: Session, wf: dict[str, Any], params: dict[str, Any], *,
                      actor: str = "", allow_expensive: bool = False,
                      approved: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run one leg of a ``"kind": "goal"`` template.

        Each iteration: the LLM PROPOSES a tool (``llm.plan``), the deterministic
        policy admits / gates / rejects it, and an admitted autonomous step runs
        through ``registry.execute`` with the same audit wrapping and the same
        ``executed[]`` record a step template produces. A gated proposal is parked
        in ``state.pending_tool`` (``proposed_by="planner"``) and the leg returns
        ``awaiting_review``; ``approved`` is the call ``take_pending_tool`` handed
        back when the human said yes, and it runs first on the resumed leg.
        ``allow_expensive`` keeps its meaning: only the JobManager path sets it,
        so a long fit the policy gated and a human approved still cannot run on a
        request thread. Every proposal and decision is an audit entry
        (agent="planner"); the readable history lives in ``state.planner_history``.
        """
        goal = wf["goal"]
        roster = {name: self.registry.get(name) for name in wf["allowed_tools"]}
        budget = policy.budget_from(wf)
        executed: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = []
        if approved is None and sess.state.planner_history is None:
            # start_workflow cleared the history: this is a fresh run of the goal.
            sess.state = apply_writes(sess.state, "supervisor", {"planner_history": []})
            sess.audit.append(
                agent="planner", tool="plan", action="goal_start",
                inputs={"workflow": wf["name"], "goal": goal, "allowed_tools": sorted(roster)},
                outputs={"max_steps": budget.max_steps, "max_gated": budget.max_gated},
                timestamp=self.clock(), actor=actor)
        if approved is not None:
            halted = self._run_goal_step(
                sess, approved["tool"], approved["args"], actor=actor,
                allow_expensive=allow_expensive, executed=executed, observations=observations,
                label=f"planner (approved): {approved['tool']}")
            if halted is not None:
                return halted
        while True:
            history = list(sess.state.planner_history or [])
            exhausted = policy.budget_exhausted(history, budget)
            if exhausted:
                sess.audit.append(agent="planner", tool="plan", action="stop",
                                  inputs={"workflow": wf["name"]}, outputs={"stop_reason": exhausted},
                                  timestamp=self.clock(), actor=actor)
                return self._goal_payload(sess, "stopped", executed, stop_reason=exhausted)
            summary = _PLANNER_VIEW._state_summary(sess.state, observations)
            summary["dataset_configured"] = bool(sess.state.dataset_id or params.get("path"))
            try:
                plan = self.llm.plan(goal, summary, observations, history,
                                     list(roster.values())) or {}
            except Exception as exc:
                # A hosted planner failed (network, 5xx, timeout). Everything executed so
                # far is already persisted; record the failure and end the leg cleanly.
                sess.audit.append(agent="planner", tool="plan", action="plan_error",
                                  inputs={"workflow": wf["name"]}, outputs={"error": str(exc)},
                                  timestamp=self.clock(), actor=actor)
                return self._goal_payload(sess, "stopped", executed,
                                          stop_reason=f"planner failed: {exc}")
            tool_name = plan.get("tool")
            raw_args = plan.get("args")
            args = dict(raw_args) if isinstance(raw_args, dict) else {}
            # The goal runs on the session's dataset: the planner may not point the
            # analysis at another file or another loaded dataset without a gate.
            args.pop("dataset_id", None)
            if tool_name == "load_dataset" and params.get("path"):
                args["path"] = params["path"]
            rationale = str(plan.get("rationale") or "")
            step = len(history)
            repaired = bool(plan.get("repaired") or (history and not history[-1].get("allow")))
            repair_reason = str(
                plan.get("repair_reason")
                or (history[-1].get("reason") if history and not history[-1].get("allow") else "")
                or ""
            )
            if plan.get("repaired"):
                sess.audit.append(
                    agent="planner", tool=tool_name or "plan", action="plan_repair",
                    inputs={"workflow": wf["name"], "step": step, "repair_reason": repair_reason},
                    outputs={"repaired_tool": tool_name, "repaired_args": args},
                    timestamp=self.clock(), actor=actor)
            sess.audit.append(
                agent="planner", tool=tool_name or "plan", action="plan",
                inputs={"workflow": wf["name"], "step": step, "args": args},
                outputs={"tool": tool_name, "rationale": rationale,
                         "stop_reason": plan.get("stop_reason")},
                timestamp=self.clock(), actor=actor)
            if not tool_name:
                reason = str(plan.get("stop_reason") or "planner selected no tool")
                if (
                    (plan.get("malformed_tool") or plan.get("premature_stop"))
                    and not plan.get("repair_attempted")
                    and policy.consecutive_rejections(history) < 1
                ):
                    rejected_name = str(plan.get("malformed_tool") or "plan")
                    sess.audit.append(
                        agent="planner", tool=rejected_name, action="policy_rejected",
                        inputs=args,
                        outputs={"allow": False, "mode": policy.MODE_REJECTED,
                                 "reason": reason, "stop": False},
                        timestamp=self.clock(), actor=actor)
                    self._record_plan(sess, {"step": step, "tool": rejected_name, "args": args,
                                             "rationale": rationale, "allow": False,
                                             "mode": policy.MODE_REJECTED, "reason": reason})
                    observations.append({"agent": "planner", "tool": rejected_name,
                                         "status": "rejected", "summary": reason})
                    continue
                # A reasoned stop ("goal met", template exhausted) completes the goal; a
                # silent one (no text, only unknown tool names) is reported as stopped.
                status = "complete" if rationale else "stopped"
                return self._goal_payload(sess, status, executed, stop_reason=reason,
                                          audit=sess.audit.to_list())
            decision = policy.decide(tool_name, args, sess.state, roster=roster,
                                     history=history, budget=budget)
            sess.audit.append(
                agent="planner", tool=tool_name, action=f"policy_{decision.mode}",
                inputs=args,
                outputs={"allow": decision.allow, "mode": decision.mode,
                         "reason": decision.reason, "stop": decision.stop},
                timestamp=self.clock(), actor=actor)
            plan_rec: dict[str, Any] = {
                "step": step, "tool": tool_name, "args": args,
                "rationale": rationale, "allow": decision.allow,
                "mode": decision.mode, "reason": decision.reason,
            }
            if repaired:
                plan_rec["repaired"] = True
                if repair_reason:
                    plan_rec["repair_reason"] = repair_reason
            self._record_plan(sess, plan_rec)
            if not decision.allow:
                observations.append({"agent": "planner", "tool": tool_name,
                                     "status": "rejected", "summary": decision.reason})
                if decision.stop:
                    return self._goal_payload(sess, "stopped", executed,
                                              stop_reason=decision.reason)
                continue
            if decision.mode == policy.MODE_GATED:
                return self._propose_goal_step(sess, wf, tool_name, args, rationale,
                                               decision.reason, executed=executed, actor=actor)
            halted = self._run_goal_step(
                sess, tool_name, args, actor=actor, allow_expensive=allow_expensive,
                executed=executed, observations=observations, label=f"planner: {tool_name}")
            if halted is not None:
                return halted

    def _run_goal_step(self, sess: Session, tool_name: str, args: dict[str, Any], *,
                       actor: str, allow_expensive: bool, executed: list[dict[str, Any]],
                       observations: list[dict[str, Any]], label: str) -> dict[str, Any] | None:
        """Execute one admitted (or human-approved) step through the registry.
        None on success; otherwise the payload that ends the leg."""
        tool = self.registry.get(tool_name)
        i = sess.state.current_step
        try:
            sess.state, res = self.registry.execute(
                tool_name, state=sess.state, ctx=sess.ctx, args=args, audit=sess.audit,
                timestamp=self.clock(), actor=actor, allow_expensive=allow_expensive)
        except ExpensiveToolError as exc:
            # Same contract as a step leg: nothing advanced; re-issue as a job.
            return self._goal_payload(
                sess, "awaiting_job", executed, next_step=i, next_tool=tool_name,
                message=f"{exc} Re-issue this workflow leg as a background job.")
        except Exception as exc:
            # A deterministic tool refused (e.g. prerequisites it checks itself). The
            # leg stops rather than letting an automated loop retry around an error.
            sess.audit.append(agent="planner", tool=tool_name, action="tool_error",
                              inputs=args, outputs={"error": str(exc)},
                              timestamp=self.clock(), actor=actor)
            return self._goal_payload(sess, "stopped", executed,
                                      stop_reason=f"{tool_name} failed: {exc}")
        self._log_command(sess, tool.agent, tool_name, args)
        executed.append({"step": i, "label": label, "tool": tool_name,
                         "agent": tool.agent, "summary": res.summary})
        observations.append({"agent": tool.agent, "tool": tool_name,
                             "status": "ok", "summary": res.summary})
        sess.state = apply_writes(sess.state, "supervisor", {"current_step": i + 1})
        # Persist per step: a fit that took tens of minutes must survive a planner
        # call failing on the very next iteration (the proposal is already consumed).
        self._persist(sess)
        return None

    def _propose_goal_step(self, sess: Session, wf: dict[str, Any], tool_name: str,
                           args: dict[str, Any], rationale: str, gate_reason: str, *,
                           executed: list[dict[str, Any]], actor: str) -> dict[str, Any]:
        """Park a gated proposal exactly where a chat-proposed expensive tool goes
        (``state.pending_tool``), plus a pending review so the existing resume
        endpoint carries the human's signed decision. Nothing is computed."""
        tool = self.registry.get(tool_name)
        ts = self.clock()
        previous = sess.state.pending_tool
        if previous:
            sess.audit.append(
                agent="planner", tool=previous["tool"], action="supersede_tool",
                inputs=previous.get("args") or {},
                outputs={"status": "superseded", "by": tool_name}, timestamp=ts, actor=actor)
        pending = {"tool": tool_name, "agent": tool.agent, "args": args, "proposed_at": ts,
                   "proposed_by": "planner", "message": wf["goal"], "rationale": rationale,
                   "workflow": wf["name"], "gate_reason": gate_reason}
        sess.state = apply_writes(sess.state, "supervisor", {"pending_tool": pending})
        sess.audit.append(
            agent="planner", tool=tool_name, action="propose_tool", inputs=args,
            outputs={"status": "pending_approval", "agent": tool.agent,
                     "rationale": rationale, "gate_reason": gate_reason},
            timestamp=ts, actor=actor)
        sess.pending_review = {"after_step": sess.state.current_step - 1,
                               "label": f"Planner proposes {tool_name}",
                               "proposal": {"tool": tool_name, "agent": tool.agent, "args": args}}
        return self._goal_payload(sess, "awaiting_review", executed,
                                  review=sess.pending_review, proposal=pending)

    @staticmethod
    def _planner_pending(sess: Session) -> dict[str, Any] | None:
        """The pending proposal iff the planner of THIS goal workflow made it. A chat
        proposal (``proposed_by`` = the actor id, no ``workflow``) is never one, so
        the goal resume cannot be made to run a call the policy never classified."""
        pending = sess.state.pending_tool
        if (pending and pending.get("proposed_by") == "planner"
                and pending.get("workflow") == sess.state.workflow_name):
            return pending
        return None

    def _goal_resume_blocker(self, sess: Session, *,
                             allow_expensive: bool) -> dict[str, Any] | None:
        """Checked BEFORE an approval is signed: an approved long-running fit goes to
        the job queue without consuming the proposal or the review, so the same
        resume is re-issued as a job (the ExpensiveToolError contract, applied
        before anything is spent)."""
        pending = self._planner_pending(sess)
        if pending is None:
            return None
        if not allow_expensive and self.registry.get(pending["tool"]).expensive:
            return {"status": "awaiting_job", "executed": [],
                    "next_step": sess.state.current_step, "next_tool": pending["tool"],
                    "message": (f"'{pending['tool']}' is a long-running fit — it runs only as a "
                                "bounded background job. Re-issue this approval as a background job."),
                    "state": sess.state.model_dump(), "audit_ok": sess.audit.verify()}
        return None

    def _resume_goal(self, sess: Session, wf: dict[str, Any], params: dict[str, Any], *,
                     approve: bool, actor: str, reason: str,
                     allow_expensive: bool) -> dict[str, Any]:
        """The human decided on a planner proposal (the signed human_review entry is
        already in the chain). Rejection clears it through reject_pending_tool and
        ends the workflow; approval goes through take_pending_tool — whose
        ``confirm=True`` IS the approval — then the goal loop resumes with that call.
        A proposal already decided through /chat/pending_tool leaves nothing to take:
        a rejection then just closes the review, an approval resumes planning from
        the state as it now is (the policy gates any further fit again)."""
        sess.pending_review = None
        pending = self._planner_pending(sess)
        if not approve:
            if pending is not None:
                self.reject_pending_tool(sess.id, actor=actor, reason=reason)
            else:
                self._persist(sess)
            return {"status": "rejected", "at_step": sess.state.current_step,
                    "state": sess.state.model_dump(), "audit_ok": sess.audit.verify()}
        call = (self.take_pending_tool(sess.id, actor=actor, reason=reason)
                if pending is not None else None)
        return self._advance_goal(sess, wf, params, actor=actor,
                                  allow_expensive=allow_expensive, approved=call)
