"""Which publications the reviewer currently includes: a source is included only by a DECIDED screen
whose criteria hash is the current criteria's and whose reviewer inclusion is 'include'.

There is no fallback to a screen made under other criteria: after the criteria change, a source is
not included until it is screened and decided again (screens of one second could not even be
ordered reliably). Extraction reads only included sources, a record of a source that is not
included cannot be approved, and the export leaves out approved records of such sources.
"""
from __future__ import annotations

from typing import Any

from app.evidence.models import Screen, Source, effective_verdict
from app.evidence.store import EvidenceStore


def current_screens(store: EvidenceStore, sid: str) -> dict[str, Screen]:
    """source_id -> its screen under the current criteria (none when no criteria are saved)."""
    crit = store.get_config(sid, "criteria")
    if not crit:
        return {}
    return {s.source_id: s for s in store.list_screens(sid) if s.criteria_hash == crit["body_hash"]}


def exclusion_reason(screens: dict[str, Screen], source_id: str) -> str | None:
    """None when *source_id* is included under the current criteria, else why it is not."""
    scr = screens.get(source_id)
    if scr is None:
        return "not screened under the current criteria"
    verdict = effective_verdict(scr)
    if verdict != "include":
        return f"screen {scr.id}: {verdict or 'no reviewer decision'}"
    return None


def included_sources(store: EvidenceStore, sid: str, sources: list[Source]
                     ) -> tuple[list[Source], list[dict[str, Any]]]:
    """(parsed sources included under the current criteria, [{source_id, reason}] for the rest)."""
    screens = current_screens(store, sid)
    included, skipped = [], []
    for src in sources:
        why = exclusion_reason(screens, src.id)
        if why is not None:
            skipped.append({"source_id": src.id, "reason": f"not included by the reviewer ({why})"})
        elif src.parse_status != "parsed":
            skipped.append({"source_id": src.id, "reason": f"parse_status {src.parse_status}"})
        else:
            included.append(src)
    return included, skipped
