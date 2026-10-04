"""Evidence agent tool: ``rebuild_source_layout`` (derive the page-position sidecar of an older source).

Deterministic and derived-data only: it re-parses the stored PDF in the sandbox and writes the layout
sidecar when, and only when, every page text equals the stored parse (relayout.py). It never touches
pages.json, the ledger or a record, so it is not human-only; each call writes one audit entry carrying the
source id, the outcome, the sidecar sha256 and a digest of the per-page text hashes (the entry's outputs hash
binds the full per-page list). It acts on one source of the
calling session (a source id of another session is refused).
"""
from __future__ import annotations

from typing import Any

from app.core.pharmstate import PharmState
from app.evidence import relayout
from app.evidence.store import get_store
from app.tools.base import Tool, ToolContext, ToolResult

AGENT = "evidence"
SUMMARY = {
    "rebuilt": "Rebuilt the page positions of {name}: page text unchanged, records untouched.",
    "unchanged": "{name} already has usable page positions; nothing was parsed or written.",
    "text_changed": "{name} was NOT rebuilt: its page text parses differently now. Compare every value with the PDF.",
    "unreadable": "{name} was NOT rebuilt: the PDF could not be parsed again. Compare every value with the PDF.",
}


def rebuild_source_layout(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    source_id = args.get("source_id")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("'source_id' is required")
    src = get_store().get_source(source_id)
    if src is None or src.session_id != state.session_id:
        raise ValueError(f"unknown evidence source: {source_id}")
    out = (relayout.take_handover(state.session_id, src.id)
           or relayout.rebuild_layout(src, state.session_id, ctx.data_dir))
    return ToolResult(
        summary=SUMMARY[out["outcome"]].format(name=src.filename),
        result=out,
        action=(f"rebuild_source_layout({src.id}, outcome={out['outcome']}, "
                f"sidecar_sha256={out['sidecar_sha256'] or 'none'}, pages={out['n_pages']}, "
                f"page_text_sha256={out['pages_sha256'] or 'none'})"))


TOOLS = [
    Tool(name="rebuild_source_layout", agent=AGENT,
         description=("Rebuild the page-position sidecar of one older source (parsed before positions were "
                      "stored) so its records can be outlined on the page image. Re-parses the stored PDF in "
                      "the sandbox and writes ONLY the sidecar, and only when every page text equals the stored "
                      "parse; otherwise nothing is written and the source is marked text_changed or "
                      "unreadable. Records and the parse are never changed. Idempotent."),
         input_schema={"type": "object", "properties": {"source_id": {"type": "string"}},
                       "required": ["source_id"]},
         run=rebuild_source_layout),
]
