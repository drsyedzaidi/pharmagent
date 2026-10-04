"""Layout rebuild endpoints (mounted on the evidence router, same prefix and auth):

``POST .../sources/{source_id}/relayout`` rebuilds one source's page-position sidecar now (the parse runs
on its own worker limiter and admission limits, outside the session lock); ``POST .../relayout`` rebuilds
every source of the session that lacks one, at most MAX_BULK per call, one at a time through the parse queue, as a
JobManager job (poll /jobs/{id}); the response says how many were picked and how many remain.
Each rebuild is one audited ``rebuild_source_layout`` tool call. A source of another session is 404.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends

from app.evidence import relayout, relayout_limits
from app.evidence.config_tools import source_row
from app.evidence.routes import AGENT, http_error, owned_source
from app.evidence.store import get_store

JOB_KIND = "evidence_relayout"
TOOL = "rebuild_source_layout"


def _rebuild(orch: Any, sid: str, source_id: str, actor: str, data_dir: str) -> dict[str, Any]:
    """The rebuild runs HERE, outside the session lock (it can take minutes); the audited tool call then
    takes the lock only to write its entry, and reports the outcome handed over to it."""
    src = get_store().get_source(source_id)
    if src is None or src.session_id != sid:
        raise ValueError(f"unknown evidence source: {source_id}")
    outcome = relayout.outside_outcome(src, data_dir) or relayout.rebuild_layout(src, sid)   # confined first
    relayout.hand_over(sid, source_id, outcome)
    try:
        return orch.run_tool(sid, TOOL, AGENT, {"source_id": source_id}, actor=actor)
    finally:
        relayout.take_handover(sid, source_id)           # nothing left behind if the tool never ran


def _one(orch: Any, sid: str, source_id: str, actor: str, data_dir: str) -> dict[str, Any]:
    try:
        out = _rebuild(orch, sid, source_id, actor, data_dir)
    except Exception as exc:          # mapped, never swallowed (a busy parser is a 503 that records nothing)
        raise http_error(exc) from exc
    src = get_store().get_source(source_id)
    return {"source": source_row(src) if src else None, **out["result"], "summary": out["summary"]}


def _bulk(orch: Any, sid: str, source_ids: list[str], actor: str, data_dir: str) -> dict[str, Any]:
    """One audited call per source, sequentially; a failure is that source's entry, never the job's."""
    results: list[dict[str, Any]] = []
    for source_id in source_ids:
        try:
            out = _rebuild(orch, sid, source_id, actor, data_dir)
            results.append({"source_id": source_id, **out["result"]})
        except Exception as exc:
            results.append({"source_id": source_id, "outcome": "error", "reason": http_error(exc).detail})
    counts: dict[str, int] = {}
    for r in results:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    return {"results": results, "counts": counts}


def add_layout_routes(router: APIRouter, owned_session: Callable[..., Any], actor_id: Callable[..., str],
                      get_orch: Callable[[], Any], get_jobs: Callable[[], Any] | None) -> None:

    @router.post("/sources/{source_id}/relayout")
    async def relayout_source(sid: str, source_id: str, sess=Depends(owned_session),
                              actor: str = Depends(actor_id)) -> dict:
        src = owned_source(sid, source_id)
        async with relayout_limits.relayout_slot(sid):
            return await relayout_limits.run_blocking(_one, get_orch(), sid, src.id, actor, str(sess.ctx.data_dir))

    @router.post("/relayout")
    def relayout_all(sid: str, sess=Depends(owned_session), actor: str = Depends(actor_id)) -> dict:
        pending = relayout.pending_sources(get_store().list_sources(sid))
        picked = [s.id for s in pending[:relayout.MAX_BULK]]
        base = {"selected": len(picked), "remaining": len(pending) - len(picked), "cap": relayout.MAX_BULK,
                "source_ids": picked, "kind": JOB_KIND, "tool": TOOL}
        if not picked:
            return {**base, "job_id": None, "status": "done"}
        orch, data_dir = get_orch(), str(sess.ctx.data_dir)
        if get_jobs is None:                                  # no JobManager: inline
            return {**base, "job_id": None, "status": "done", **_bulk(orch, sid, picked, actor, data_dir)}
        job_id = get_jobs().submit(session_id=sid, kind=JOB_KIND,       # JobRejected -> main's 429 handler
                                   fn=lambda: _bulk(orch, sid, picked, actor, data_dir))
        return {**base, "job_id": job_id, "status": "running"}
