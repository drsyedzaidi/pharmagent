"""Skills — captured, replayable analysis workflows.

A *skill* is a named, version-controlled sequence of tool calls distilled from a
session that already worked ("skills are captured, not authored"). Replaying a
skill on a new dataset reproduces the whole validated pipeline — NCA, model fit,
adversarial review, report — without re-deriving the steps.

Capture source is the session command log (orchestrator records every analysis
tool call with its args). This is independent of the audit chain, whose inputs
are hashed for tamper-evidence and so cannot be replayed from.

Storage is a small SQLite table in the same DB file as sessions.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

# Tools that should NOT become skill steps: dataset I/O (replay supplies a fresh
# dataset) and pure-visualization/profiling noise. Everything else — the analysis
# decisions — is the reusable substance.
_SKIP_TOOLS = {"load_dataset", "profile_pk_dataset", "validate_cdisc",
               "generate_spaghetti_plot"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    name           TEXT PRIMARY KEY,
    description    TEXT,
    goal           TEXT,
    steps_json     TEXT NOT NULL,
    source_session TEXT,
    owner          TEXT,
    created_at     TEXT,
    version        INTEGER NOT NULL DEFAULT 1
);
"""


def distill_steps(command_log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn a raw session command log into ordered, replayable skill steps.

    Drops dataset I/O and profiling; collapses consecutive duplicate (tool,args)
    calls (e.g. the same analysis re-run while iterating) to the last occurrence.
    """
    steps: list[dict[str, Any]] = []
    for cmd in command_log:
        tool = cmd.get("tool")
        if not tool or tool in _SKIP_TOOLS:
            continue
        step = {"agent": cmd.get("agent", ""), "tool": tool,
                "args": dict(cmd.get("args") or {})}
        # collapse an immediately-repeated identical step (iterating on the same tool)
        if steps and steps[-1]["tool"] == tool:
            steps[-1] = step
        else:
            steps.append(step)
    return steps


class Skill:
    """In-memory view of a stored skill."""

    def __init__(self, *, name: str, description: str, goal: str,
                 steps: list[dict[str, Any]], source_session: str | None,
                 owner: str | None, created_at: str, version: int) -> None:
        self.name = name
        self.description = description
        self.goal = goal
        self.steps = steps
        self.source_session = source_session
        self.owner = owner
        self.created_at = created_at
        self.version = version

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "goal": self.goal,
                "steps": self.steps, "source_session": self.source_session,
                "owner": self.owner, "created_at": self.created_at,
                "version": self.version}

    def to_markdown(self) -> str:
        """Human-readable, diffable SKILL.md form."""
        lines = [f"# Skill: {self.name}", "", self.description or "_(no description)_", ""]
        if self.goal:
            lines += [f"**Goal:** {self.goal}", ""]
        lines += [f"**Version:** {self.version}  ·  **Captured from:** "
                  f"{self.source_session or 'n/a'}  ·  **Created:** {self.created_at}", "",
                  "## Steps", ""]
        for i, s in enumerate(self.steps, 1):
            arg = json.dumps(s.get("args") or {}, sort_keys=True)
            arg = "" if arg == "{}" else f" `{arg}`"
            lines.append(f"{i}. **{s['tool']}** ({s.get('agent', '')}){arg}")
        return "\n".join(lines) + "\n"


class SkillStore:
    def __init__(self, db_path: str | Path) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def save(self, skill: Skill) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO skills
                   (name, description, goal, steps_json, source_session, owner,
                    created_at, version)
                   VALUES (?,?,?,?,?,?,?,?)
                   ON CONFLICT(name) DO UPDATE SET
                     description=excluded.description, goal=excluded.goal,
                     steps_json=excluded.steps_json, source_session=excluded.source_session,
                     version=skills.version+1""",
                (skill.name, skill.description, skill.goal,
                 json.dumps(skill.steps, default=str), skill.source_session,
                 skill.owner, skill.created_at, skill.version))
            self._conn.commit()

    def get(self, name: str) -> Skill | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM skills WHERE name=?", (name,)).fetchone()
        return self._row(r) if r else None

    def list(self, owner: str | None = None) -> list[Skill]:
        with self._lock:
            if owner is None:
                rows = self._conn.execute(
                    "SELECT * FROM skills ORDER BY created_at").fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM skills WHERE owner IS NULL OR owner=? ORDER BY created_at",
                    (owner,)).fetchall()
        return [self._row(r) for r in rows]

    def delete(self, name: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM skills WHERE name=?", (name,))
            self._conn.commit()
            return cur.rowcount > 0

    @staticmethod
    def _row(r: sqlite3.Row) -> Skill:
        return Skill(
            name=r["name"], description=r["description"] or "", goal=r["goal"] or "",
            steps=json.loads(r["steps_json"]), source_session=r["source_session"],
            owner=r["owner"], created_at=r["created_at"], version=r["version"])
