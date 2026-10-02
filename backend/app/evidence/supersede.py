"""Run history for proposed records: which proposal of a slot (source, canonical field, normalised
group) is current. After an extraction run, a live proposal of a slot that the run proposed again
but under a different claim is flagged 'superseded' (warn); the run's own claims are current. The
flag records history the verifier cannot recompute, so verify_records carries it over unchanged, and
a later run that proposes the same claim again clears it. Pure functions; nothing is mutated.
"""
from __future__ import annotations

from typing import Any

from app.evidence import models as M
from app.evidence.qc_flags import slot_key
from app.evidence.store import record_id

Slot = tuple[str, str, str]
MAX_LISTED = 3


def claim_ids(session_id: str, source_sha256: str, template_hash: str, proposals: list[dict[str, Any]]) -> set[str]:
    """Record ids the store gives these proposals (claim identity, as insert_records computes it)."""
    return {record_id(session_id, M.record_hash(
        source_sha256=source_sha256, template_hash=template_hash, field=p["field"], group=p.get("group_label") or "",
        value=str(p["value"]), unit=p.get("unit"), locator=p["locator"], extractor=p["extractor"]))
        for p in proposals}


def carried(rec: Any) -> list[dict[str, Any]]:
    """The superseded flags a record already holds (kept across re-verification)."""
    return [dict(f) for f in M.as_mapping(rec).get("flags") or [] if f.get("code") == M.FLAG_SUPERSEDED]


def _flag(rec: dict[str, Any], current: list[dict[str, Any]]) -> dict[str, Any]:
    listed = ", ".join(f"{c.get('value')!r} ({c.get('id')})" for c in current[:MAX_LISTED])
    return {"code": M.FLAG_SUPERSEDED, "severity": "warn", "field": rec.get("field") or "",
            "group": rec.get("group_label") or "", "origin": "script",
            "message": f"{rec.get('field')}: a later extraction run proposed {listed} for this source/group; "
                       "this proposal is from an earlier run"}


def run_marks(rows: list[Any], run_ids: set[str], template: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """record id -> its superseded flags after a run whose claim ids are *run_ids*: none for the run's
    own claims, a fresh flag for another proposal in a slot the run touched, the old flags elsewhere."""
    recs = [M.as_mapping(r) for r in rows]
    current: dict[Slot, list[dict[str, Any]]] = {}
    for r in recs:
        if r["id"] in run_ids:
            current.setdefault(slot_key(r, template), []).append(r)
    out: dict[str, list[dict[str, Any]]] = {}
    for r in recs:
        mine = current.get(slot_key(r, template))
        if r["id"] in run_ids:
            out[r["id"]] = []
        elif mine:
            out[r["id"]] = [_flag(r, mine)]
        else:
            out[r["id"]] = carried(r)
    return out
