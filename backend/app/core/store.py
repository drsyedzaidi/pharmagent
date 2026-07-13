"""SQLite persistence for sessions and their audit trails.

Sessions, PharmState, the SHA-256 audit chain, workflow position, and pending
review are serialized to a single SQLite file so nothing is lost on restart.
Raw datasets are NOT stored in the DB — only the dataset path — and are
re-read from disk on load (the path is confined to the allowed data roots).

A module-level connection with a lock is sufficient for the single-worker
deployment; swap for a connection pool if the app is scaled out.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id           TEXT PRIMARY KEY,
    owner        TEXT,
    created_at   TEXT,
    updated_at   TEXT,
    state_json   TEXT NOT NULL,
    audit_json   TEXT NOT NULL,
    history_json TEXT NOT NULL,
    pending_json TEXT,
    params_json  TEXT,
    dataset_id   TEXT,
    dataset_path TEXT,
    commands_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_sessions_owner ON sessions(owner);
"""


class SessionStore:
    def __init__(self, db_path: str | Path) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            # Migration-safe column add for DBs created before the command log.
            try:
                self._conn.execute("ALTER TABLE sessions ADD COLUMN commands_json TEXT")
            except sqlite3.OperationalError:
                pass  # column already exists
            self._conn.commit()

    def save(self, *, id: str, owner: str | None, created_at: str, updated_at: str,
             state: dict[str, Any], audit: list[dict], history: list[dict],
             pending: dict | None, params: dict, dataset_id: str | None,
             dataset_path: str | None, commands: list[dict] | None = None) -> None:
        row = (id, owner, created_at, updated_at,
               json.dumps(state, default=str), json.dumps(audit, default=str),
               json.dumps(history, default=str),
               json.dumps(pending, default=str) if pending is not None else None,
               json.dumps(params, default=str), dataset_id, dataset_path,
               json.dumps(commands or [], default=str))
        with self._lock:
            self._conn.execute(
                """INSERT INTO sessions
                   (id, owner, created_at, updated_at, state_json, audit_json,
                    history_json, pending_json, params_json, dataset_id, dataset_path,
                    commands_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     owner=excluded.owner, updated_at=excluded.updated_at,
                     state_json=excluded.state_json, audit_json=excluded.audit_json,
                     history_json=excluded.history_json, pending_json=excluded.pending_json,
                     params_json=excluded.params_json, dataset_id=excluded.dataset_id,
                     dataset_path=excluded.dataset_path, commands_json=excluded.commands_json""",
                row)
            self._conn.commit()

    def load_all(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM sessions").fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get(self, sid: str) -> dict[str, Any] | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        return self._row_to_dict(r) if r else None

    @staticmethod
    def _row_to_dict(r: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": r["id"], "owner": r["owner"],
            "created_at": r["created_at"], "updated_at": r["updated_at"],
            "state": json.loads(r["state_json"]),
            "audit": json.loads(r["audit_json"]),
            "history": json.loads(r["history_json"]),
            "pending": json.loads(r["pending_json"]) if r["pending_json"] else None,
            "params": json.loads(r["params_json"]) if r["params_json"] else {},
            "dataset_id": r["dataset_id"], "dataset_path": r["dataset_path"],
            "commands": json.loads(r["commands_json"]) if r["commands_json"] else [],
        }
