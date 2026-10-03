"""Decisions that predate a template change. Records are keyed on the template hash, so re-saving the template
(even adding one field) makes the next extraction propose every claim again as a NEW row. A proposal that
repeats a claim the reviewer already decided under an EARLIER template (same source, canonical field,
normalised group, value and unit) gets a ``decided_earlier`` warning that names that decision, so a rejected
value does not come back looking fresh and a repeat of an approved one is not approved blindly. A flag only:
nothing is decided, hidden or copied over. Pure; derived again on every verification."""
from __future__ import annotations

from typing import Any

from app.evidence import models as M
from app.evidence.qc_flags import same_value, slot_key

MAX_REASON_SHOWN = 160


def _decision_flag(rec: dict[str, Any], earlier: dict[str, Any]) -> dict[str, Any]:
    verb = earlier["status"]
    reason = str(earlier.get("decision_reason") or "").strip()
    shown = reason if len(reason) <= MAX_REASON_SHOWN else reason[:MAX_REASON_SHOWN] + "..."
    tail = (" Do not approve it again unless the new page shows something different." if verb == "rejected"
            else " This repeats it; approve it again only if the new template needs the row.")
    return {"code": M.FLAG_DECIDED_EARLIER, "severity": "warn", "field": rec.get("field") or "",
            "group": rec.get("group_label") or "", "origin": "script",
            "message": f"{rec.get('field')}: this exact value was {verb} earlier under another template "
                       f"({earlier.get('decided_by') or 'a reviewer'}: {shown or 'no reason recorded'}).{tail}"}


def decided_records(store: Any, sid: str, approved: list[Any]) -> list[Any]:
    """Approved (*approved*, already read) and rejected records of the session, any template."""
    return [*approved, *store.list_records(sid, status="rejected")]


def earlier_flags(proposals: list[dict[str, Any]], decided: list[Any], template: dict[str, Any],
                  template_hash: str) -> list[list[dict[str, Any]]]:
    """One flag list per proposal (same order): its own template hash is ``p['template_hash']`` or
    *template_hash*; only approved / rejected records of ANOTHER hash count; the latest decision wins."""
    rows = [M.as_mapping(r) for r in decided or []]
    rows = [r for r in rows if r.get("status") in M.DECISIONS]
    rows.sort(key=lambda r: str(r.get("decided_at") or ""))
    out: list[list[dict[str, Any]]] = []
    for p in proposals:
        th = p.get("template_hash") or template_hash
        key = slot_key(p, template)
        mine = [r for r in rows if r.get("template_hash") != th and slot_key(r, template) == key and same_value(r, p)]
        out.append([_decision_flag(p, mine[-1])] if mine else [])
    return out
