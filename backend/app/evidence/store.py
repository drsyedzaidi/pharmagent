"""EvidenceStore: the SQLite ledger for sources, criteria/template configs, screens, records, datasets.

Shares the SessionStore file (settings.db_path). Ids are hash-derived so re-runs
are idempotent. Approved records are frozen by SQL triggers; decide() is the
only status-change path and reverify() only touches proposed rows. No DELETE API.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import fields
from pathlib import Path
from typing import Any

from app.core.store import _json as canonical_json
from app.evidence.models import (
    DECISIONS,
    Dataset,
    EvidenceStateError,
    Record,
    Screen,
    Source,
    body_hash,
    derived_verdict,
    now_iso,
    sha256_text,
)
from app.evidence.models import (
    record_hash as _record_hash,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence_source (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, sha256 TEXT NOT NULL, filename TEXT NOT NULL,
    pdf_path TEXT NOT NULL, parse_path TEXT NOT NULL, pages INTEGER NOT NULL, title TEXT,
    metadata_json TEXT, text_chars INTEGER NOT NULL,
    parse_status TEXT NOT NULL CHECK(parse_status IN ('parsed','no_text','failed')),
    decimal_style TEXT NOT NULL DEFAULT '.', licence_note TEXT, uploaded_by TEXT NOT NULL,
    uploaded_at TEXT NOT NULL, UNIQUE(session_id, sha256));
CREATE INDEX IF NOT EXISTS ix_evsrc_session ON evidence_source(session_id);
CREATE TABLE IF NOT EXISTS evidence_config (
    session_id TEXT NOT NULL, kind TEXT NOT NULL CHECK(kind IN ('criteria','template')),
    body_json TEXT NOT NULL, body_hash TEXT NOT NULL, saved_by TEXT NOT NULL, saved_at TEXT NOT NULL,
    PRIMARY KEY(session_id, kind));
CREATE TABLE IF NOT EXISTS evidence_screen (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
    source_id TEXT NOT NULL REFERENCES evidence_source(id), criteria_hash TEXT NOT NULL,
    verdict TEXT NOT NULL CHECK(verdict IN ('include','exclude','unsure')), score REAL NOT NULL,
    reasoning TEXT NOT NULL, criteria_json TEXT NOT NULL,
    method TEXT NOT NULL CHECK(method IN ('llm','keyword')),
    status TEXT NOT NULL DEFAULT 'proposed' CHECK(status IN ('proposed','approved','rejected')),
    decided_by TEXT, decided_at TEXT, decision_reason TEXT, created_at TEXT NOT NULL,
    decided_verdict TEXT CHECK(decided_verdict IN ('include','exclude')),
    UNIQUE(source_id, criteria_hash));
CREATE INDEX IF NOT EXISTS ix_evscr_session ON evidence_screen(session_id, status);
CREATE TABLE IF NOT EXISTS evidence_record (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
    source_id TEXT NOT NULL REFERENCES evidence_source(id), template_hash TEXT NOT NULL,
    field TEXT NOT NULL, group_label TEXT NOT NULL DEFAULT '', value TEXT NOT NULL, value_num REAL,
    unit TEXT, locator_json TEXT NOT NULL,
    extractor TEXT NOT NULL CHECK(extractor IN ('llm','table','manual')),
    verify_status TEXT NOT NULL CHECK(verify_status IN ('pass','fail')),
    verify_reason TEXT NOT NULL DEFAULT '', flags_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'proposed' CHECK(status IN ('proposed','approved','rejected')),
    decided_by TEXT, decided_at TEXT, decision_reason TEXT, record_hash TEXT NOT NULL,
    created_at TEXT NOT NULL, UNIQUE(session_id, record_hash));
CREATE INDEX IF NOT EXISTS ix_evrec_session ON evidence_record(session_id, status);
CREATE INDEX IF NOT EXISTS ix_evrec_key ON evidence_record(source_id, field, group_label);
CREATE TRIGGER IF NOT EXISTS trg_evrec_frozen BEFORE UPDATE ON evidence_record
    WHEN OLD.status='approved' BEGIN SELECT RAISE(ABORT,'approved record is immutable'); END;
CREATE TRIGGER IF NOT EXISTS trg_evrec_nodelete BEFORE DELETE ON evidence_record
    WHEN OLD.status='approved' BEGIN SELECT RAISE(ABORT,'approved record is immutable'); END;
CREATE TABLE IF NOT EXISTS evidence_dataset (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, template_hash TEXT NOT NULL,
    long_path TEXT NOT NULL, wide_path TEXT NOT NULL, long_sha256 TEXT NOT NULL,
    wide_sha256 TEXT NOT NULL, record_hashes_json TEXT NOT NULL, n_records INTEGER NOT NULL,
    n_sources INTEGER NOT NULL, exported_by TEXT NOT NULL, exported_at TEXT NOT NULL);
"""

# column -> dataclass attribute for JSON-encoded columns
_JSON_COLS = {"metadata_json": "metadata", "criteria_json": "criteria", "locator_json": "locator",
              "flags_json": "flags", "record_hashes_json": "record_hashes"}


def _load(cls, row: sqlite3.Row):
    data: dict[str, Any] = {}
    for key in row.keys():
        if key in _JSON_COLS:
            data[_JSON_COLS[key]] = json.loads(row[key]) if row[key] is not None else None
        else:
            data[key] = row[key]
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


def source_id(session_id: str, sha256: str) -> str:
    return "src_" + sha256_text(f"{session_id}:{sha256}")[:12]


def screen_id(src_id: str, criteria_hash: str) -> str:
    return "scr_" + sha256_text(f"{src_id}:{criteria_hash}")[:12]


def record_id(session_id: str, rhash: str) -> str:
    return "rec_" + sha256_text(f"{session_id}:{rhash}")[:16]


def _check_decision(decision: str) -> None:
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}")


class EvidenceStore:
    def __init__(self, db_path: str | Path) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            cols = {r[1] for r in self._conn.execute("PRAGMA table_info(evidence_screen)")}
            if "decided_verdict" not in cols:          # ledgers created before the reviewer's inclusion
                self._conn.execute("ALTER TABLE evidence_screen ADD COLUMN decided_verdict TEXT "
                                   "CHECK(decided_verdict IN ('include','exclude'))")
            self._conn.commit()

    # ------------------------------------------------------------ sources --
    def add_source(self, *, session_id: str, sha256: str, filename: str, pdf_path: str, parse_path: str,
                   pages: int, title: str | None, metadata: dict[str, str], text_chars: int,
                   parse_status: str, decimal_style: str = ".", licence_note: str | None = None,
                   uploaded_by: str) -> Source:
        sid = source_id(session_id, sha256)
        with self._lock:
            self._conn.execute(
                """INSERT INTO evidence_source (id, session_id, sha256, filename, pdf_path,
                   parse_path, pages, title, metadata_json, text_chars, parse_status, decimal_style,
                   licence_note, uploaded_by, uploaded_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(session_id, sha256) DO NOTHING""",
                (sid, session_id, sha256, filename, str(pdf_path), str(parse_path), pages, title,
                 canonical_json(metadata or {}), text_chars, parse_status, decimal_style, licence_note,
                 uploaded_by, now_iso()))
            self._conn.commit()
        return self.get_source(sid)

    def get_source(self, sid: str) -> Source | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM evidence_source WHERE id=?", (sid,)).fetchone()
        return _load(Source, r) if r else None

    def list_sources(self, session_id: str) -> list[Source]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM evidence_source WHERE session_id=? ORDER BY uploaded_at, id",
                                      (session_id,)).fetchall()
        return [_load(Source, r) for r in rows]

    # ------------------------------------------------------------- config --
    def save_config(self, session_id: str, kind: str, body: Any, saved_by: str) -> str:
        h = body_hash(body)
        with self._lock:
            self._conn.execute(
                """INSERT INTO evidence_config (session_id, kind, body_json, body_hash, saved_by, saved_at)
                   VALUES (?,?,?,?,?,?) ON CONFLICT(session_id, kind) DO UPDATE SET
                   body_json=excluded.body_json, body_hash=excluded.body_hash,
                   saved_by=excluded.saved_by, saved_at=excluded.saved_at""",
                (session_id, kind, canonical_json(body), h, saved_by, now_iso()))
            self._conn.commit()
        return h

    def get_config(self, session_id: str, kind: str) -> dict[str, Any] | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM evidence_config WHERE session_id=? AND kind=?",
                                   (session_id, kind)).fetchone()
        if not r:
            return None
        return {"kind": kind, "body": json.loads(r["body_json"]), "body_hash": r["body_hash"],
                "saved_by": r["saved_by"], "saved_at": r["saved_at"]}

    # ------------------------------------------------------------ screens --
    def upsert_screens(self, session_id: str, screens: list[dict[str, Any]]) -> list[Screen]:
        """Insert-or-skip; returns the stored row for every input (existing rows win)."""
        ids = []
        with self._lock:
            for s in screens:
                sid = screen_id(s["source_id"], s["criteria_hash"])
                ids.append(sid)
                self._conn.execute(
                    """INSERT INTO evidence_screen (id, session_id, source_id, criteria_hash,
                       verdict, score, reasoning, criteria_json, method, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT(source_id, criteria_hash) DO NOTHING""",
                    (sid, session_id, s["source_id"], s["criteria_hash"], s["verdict"], float(s["score"]),
                     s["reasoning"], canonical_json(s.get("criteria") or []), s["method"], now_iso()))
            self._conn.commit()
        return [self._screen(i) for i in ids]

    def _screen(self, sid: str) -> Screen:
        with self._lock:
            r = self._conn.execute("SELECT * FROM evidence_screen WHERE id=?", (sid,)).fetchone()
        if not r:
            raise EvidenceStateError(f"screen {sid} not found")
        return _load(Screen, r)

    def list_screens(self, session_id: str, status: str | None = None,
                     source_id: str | None = None) -> list[Screen]:
        sql, args = "SELECT * FROM evidence_screen WHERE session_id=?", [session_id]
        if status:
            sql += " AND status=?"; args.append(status)
        if source_id:
            sql += " AND source_id=?"; args.append(source_id)
        with self._lock:
            rows = self._conn.execute(sql + " ORDER BY created_at, id", args).fetchall()
        return [_load(Screen, r) for r in rows]

    def decide_screen(self, sid: str, decision: str, decided_by: str, reason: str,
                      decided_verdict: str | None = None) -> Screen:
        """*decided_verdict* is the reviewer's inclusion; omitted, approve / reject of the proposal implies it."""
        _check_decision(decision)
        if decided_verdict is None:
            decided_verdict = derived_verdict(self._screen(sid).verdict, decision)
        with self._lock:
            cur = self._conn.execute(
                "UPDATE evidence_screen SET status=?, decided_by=?, decided_at=?, decision_reason=?, "
                "decided_verdict=? WHERE id=? AND status='proposed'",
                (decision, decided_by, now_iso(), reason, decided_verdict, sid))
            self._conn.commit()
        if cur.rowcount == 0:
            raise EvidenceStateError(f"screen {sid} already decided or unknown")
        return self._screen(sid)

    # ------------------------------------------------------------ records --
    def insert_records(self, session_id: str, source: Source, template_hash: str,
                       proposals: list[dict[str, Any]]) -> list[Record]:
        """Insert proposed records; a proposal whose claim identity already exists is skipped."""
        inserted: list[str] = []
        with self._lock:
            for p in proposals:
                group = p.get("group_label") or ""
                rhash = _record_hash(source_sha256=source.sha256, template_hash=template_hash,
                                     field=p["field"], group=group, value=str(p["value"]),
                                     unit=p.get("unit"), locator=p["locator"], extractor=p["extractor"])
                rid = record_id(session_id, rhash)
                cur = self._conn.execute(
                    """INSERT INTO evidence_record (id, session_id, source_id, template_hash, field,
                       group_label, value, value_num, unit, locator_json, extractor, verify_status,
                       verify_reason, flags_json, record_hash, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(session_id, record_hash) DO NOTHING""",
                    (rid, session_id, source.id, template_hash, p["field"], group, str(p["value"]),
                     p.get("value_num"), p.get("unit"), canonical_json(p["locator"]), p["extractor"],
                     p.get("verify_status", "fail"), p.get("verify_reason", ""),
                     canonical_json(p.get("flags") or []), rhash, now_iso()))
                if cur.rowcount:
                    inserted.append(rid)
            self._conn.commit()
        return [self._record(i) for i in inserted]

    def _record(self, rid: str) -> Record:
        with self._lock:
            r = self._conn.execute("SELECT * FROM evidence_record WHERE id=?", (rid,)).fetchone()
        if not r:
            raise EvidenceStateError(f"record {rid} not found")
        return _load(Record, r)

    def get_record(self, rid: str) -> Record | None:
        try:
            return self._record(rid)
        except EvidenceStateError:
            return None

    def list_records(self, session_id: str, status: str | None = None, source_id: str | None = None,
                     field: str | None = None) -> list[Record]:
        sql, args = "SELECT * FROM evidence_record WHERE session_id=?", [session_id]
        for col, val in (("status", status), ("source_id", source_id), ("field", field)):
            if val:
                sql += f" AND {col}=?"; args.append(val)
        with self._lock:
            rows = self._conn.execute(sql + " ORDER BY created_at, id", args).fetchall()
        return [_load(Record, r) for r in rows]

    def decide_record(self, rid: str, decision: str, decided_by: str, reason: str) -> Record:
        """The only status UPDATE path; a second decision raises EvidenceStateError."""
        _check_decision(decision)
        with self._lock:
            cur = self._conn.execute(
                "UPDATE evidence_record SET status=?, decided_by=?, decided_at=?, decision_reason=? "
                "WHERE id=? AND status='proposed'", (decision, decided_by, now_iso(), reason, rid))
            self._conn.commit()
        if cur.rowcount == 0:
            raise EvidenceStateError(f"record {rid} already decided or unknown")
        return self._record(rid)

    def reverify(self, rid: str, verify_status: str, verify_reason: str, locator: dict[str, Any],
                 flags: list[dict[str, Any]]) -> bool:
        """Refresh verifier output on a proposed record only. Returns False when not proposed."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE evidence_record SET verify_status=?, verify_reason=?, locator_json=?, flags_json=? "
                "WHERE id=? AND status='proposed'",
                (verify_status, verify_reason, canonical_json(locator), canonical_json(flags), rid))
            self._conn.commit()
        return cur.rowcount > 0

    def approved_record_hashes(self, session_id: str) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT record_hash FROM evidence_record WHERE session_id=? AND status='approved' "
                "ORDER BY record_hash", (session_id,)).fetchall()
        return [r[0] for r in rows]

    # ----------------------------------------------------------- datasets --
    def add_dataset(self, session_id: str, template_hash: str, long_path: str, wide_path: str,
                    long_sha256: str, wide_sha256: str, record_hashes: list[str], *, n_records: int,
                    n_sources: int, exported_by: str) -> Dataset:
        did = "ds_" + long_sha256[:12]
        with self._lock:
            self._conn.execute(
                """INSERT INTO evidence_dataset (id, session_id, template_hash, long_path, wide_path,
                   long_sha256, wide_sha256, record_hashes_json, n_records, n_sources, exported_by, exported_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO NOTHING""",
                (did, session_id, template_hash, str(long_path), str(wide_path), long_sha256, wide_sha256,
                 canonical_json(sorted(record_hashes)), n_records, n_sources, exported_by, now_iso()))
            self._conn.commit()
            r = self._conn.execute("SELECT * FROM evidence_dataset WHERE id=?", (did,)).fetchone()
        return _load(Dataset, r)

    def list_datasets(self, session_id: str) -> list[Dataset]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM evidence_dataset WHERE session_id=? ORDER BY exported_at, id",
                                      (session_id,)).fetchall()
        return [_load(Dataset, r) for r in rows]

    # ------------------------------------------------------------- counts --
    def counts(self, session_id: str) -> dict[str, Any]:
        def by_status(table: str) -> dict[str, int]:
            out = {"proposed": 0, "approved": 0, "rejected": 0}
            for r in self._conn.execute(f"SELECT status, COUNT(*) FROM {table} WHERE session_id=? GROUP BY status",
                                        (session_id,)):
                out[r[0]] = r[1]
            return out

        def n(table: str) -> int:
            return self._conn.execute(f"SELECT COUNT(*) FROM {table} WHERE session_id=?", (session_id,)).fetchone()[0]

        with self._lock:
            return {"sources": n("evidence_source"), "screens": by_status("evidence_screen"),
                    "records": by_status("evidence_record"), "datasets": n("evidence_dataset")}


_store: EvidenceStore | None = None
_store_lock = threading.Lock()


def configure(db_path: str | Path) -> EvidenceStore:
    """Point the singleton at a database (tests use ':memory:')."""
    global _store
    with _store_lock:
        _store = EvidenceStore(db_path)
        return _store


def get_store() -> EvidenceStore:
    """Lazy singleton; the path is read from settings on first use."""
    global _store
    with _store_lock:
        if _store is None:
            from app.config import settings
            _store = EvidenceStore(settings.db_path)
        return _store
