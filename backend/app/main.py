"""FastAPI application — PharmAgent backend."""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from app.config import settings
from app.core import cdisc, exporters
from app.core.jobs import JobManager, JobRejected
from app.core.logging_config import configure_logging
from app.core.orchestrator import AccessError, Orchestrator
from app.workflows import WORKFLOWS

configure_logging()
log = logging.getLogger("pharmagent")

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
ALLOWED_UPLOAD_SUFFIXES = {".csv"}

app = FastAPI(title=f"{settings.app_name} API", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=settings.cors_origins,
    allow_methods=["*"], allow_headers=["*"],
)


def _error_body(code: str, message: str, request_id: str) -> dict:
    return {"success": False, "error": {"code": code, "message": message,
                                        "request_id": request_id}}


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Assign/propagate a request id, time the request, and emit a structured
    access log line. The id is returned in X-Request-ID and embedded in errors."""
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.request_id = rid
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:  # unhandled — logged here, converted to an envelope below
        dt = (time.perf_counter() - t0) * 1000
        log.exception("request_failed", extra={"request_id": rid,
                      "method": request.method, "path": request.url.path,
                      "duration_ms": round(dt, 1)})
        return JSONResponse(status_code=500, headers={"X-Request-ID": rid},
                            content=_error_body("internal_error",
                                                "internal server error", rid))
    dt = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-ID"] = rid
    log.info("request", extra={"request_id": rid, "method": request.method,
             "path": request.url.path, "status": response.status_code,
             "duration_ms": round(dt, 1)})
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    rid = getattr(request.state, "request_id", "")
    return JSONResponse(status_code=exc.status_code, headers={"X-Request-ID": rid},
                        content=_error_body(f"http_{exc.status_code}", str(exc.detail), rid))


@app.exception_handler(JobRejected)
async def job_rejected_handler(request: Request, exc: JobRejected) -> JSONResponse:
    rid = getattr(request.state, "request_id", "")
    return JSONResponse(status_code=429,
                        headers={"X-Request-ID": rid, "Retry-After": "5"},
                        content=_error_body("job_rejected", str(exc), rid))


orch = Orchestrator()
# Background-job runner for long tools (NLME, SCM). Module-level singleton; the
# job callables look up the current `orch` global at call time (tests reassign it).
jobs = JobManager(clock=orch.clock)


# ── auth ─────────────────────────────────────────────────────────────────────
def current_owner(authorization: str | None = Header(default=None)) -> str | None:
    """Bearer-token gate. Open (returns None) when no api_token is configured.

    Returns a NON-SECRET, stable principal derived from the token — never the
    token itself. The returned value is persisted as the session ``owner`` and
    written into the audit chain as the actor, so it must not carry the raw
    credential (a DB backup or audit export would otherwise disclose the token)."""
    import hashlib
    if not settings.api_token:
        return None
    token = None
    if authorization and authorization[:7].lower() == "bearer ":
        token = authorization[7:].strip()
    if token != settings.api_token:
        raise HTTPException(401, "missing or invalid bearer token")
    return "token:" + hashlib.sha256(token.encode()).hexdigest()[:16]


def owned_session(sid: str, owner: str | None = Depends(current_owner)):
    """Resolve a session, enforcing ownership when auth is enabled."""
    try:
        return orch.get_session(sid, owner=owner)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except AccessError as e:
        raise HTTPException(403, str(e))


def actor_id(owner: str | None = Depends(current_owner)) -> str:
    """The authenticated identity recorded on audit entries; 'anonymous' when
    auth is open. Ready for real per-user identities when auth is added."""
    return owner or "anonymous"


# ── request bodies ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str


class WorkflowStartRequest(BaseModel):
    workflow: str
    params: dict | None = None


class WorkflowRequest(BaseModel):
    name: str
    params: dict | None = None


class ResumeRequest(BaseModel):
    approve: bool = True
    reason: str = ""          # reason-for-change / approval note (audited)
    # Approving a gate runs every remaining step in one call. For a template
    # whose remainder is a population fit (poppk_full: NLME -> SCM -> VPC) that
    # is minutes of compute, so the client can ask for a job id to poll instead
    # of holding the connection open. Default False keeps existing callers
    # byte-identical.
    background: bool = False


class RolesRequest(BaseModel):
    overrides: dict[str, str]
    reason: str = ""


class PkModelRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_key: str | None = None
    compare: bool = False
    models: list[str] | None = None


class SimulateRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_key: str | None = None
    dose: float = 100.0
    tau: float = 24.0
    n_doses: int = 1
    tmax: float | None = None
    wt: float = 70.0
    rate: float = 0.0
    params: dict | None = None


class DoseSweepRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_key: str | None = None
    doses: list[float] | None = None
    dose: float = 100.0
    tau: float = 24.0
    n_doses: int = 1
    tmax: float | None = None


class RefitLzRequest(BaseModel):
    subject: str
    selected_times: list[float]
    selected_concs: list[float]


class FlexplotRequest(BaseModel):
    y: str                          # outcome (required)
    x: str | None = None            # optional predictor
    color_by: str | None = None
    panel_by: str | None = None
    fit: str = "loess"              # loess | linear | none
    geom: str = "points"           # points | line | smooth | density
    center: str = "median_iqr"     # median_iqr | mean_se | mean_sd
    ghost: bool = False
    log_y: bool = False
    jitter: float = Field(0.2, ge=0.0, le=1.0)
    n_bins: int = Field(10, ge=1, le=200)
    ci: float = Field(0.95, gt=0.0, lt=1.0)


# ── public endpoints ─────────────────────────────────────────────────────────
@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "app": settings.app_name, "org": settings.org_name,
            "llm": "mock" if settings.llm_is_mock else settings.model,
            "auth": "required" if settings.api_token else "open"}


@app.get("/api/workflows")
def list_workflows() -> dict:
    return {"workflows": [w["name"] for w in WORKFLOWS.values()]}


@app.get("/api/pk_models")
def pk_models() -> dict:
    from app.compute.pk_models import list_models
    return {"models": list_models()}


# ── sessions ─────────────────────────────────────────────────────────────────
@app.post("/api/sessions")
def create_session(owner: str | None = Depends(current_owner)) -> dict:
    sess = orch.create_session(owner=owner)
    return {"id": sess.id, "created_at": sess.created_at}


@app.post("/api/sessions/{sid}/upload")
async def upload_for_session(sid: str, file: UploadFile = File(...),
                             sess=Depends(owned_session)) -> dict:
    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(400, f"unsupported file type: {suffix or '(none)'}; CSV only")
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(file.filename or "upload").stem
    dest = data_dir / f"{stem}_{uuid.uuid4().hex[:6]}.csv"
    size = 0
    with dest.open("wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"file exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")
            f.write(chunk)
    import pandas as pd

    from app.core.schema_extractor import extract_schema
    try:
        df = pd.read_csv(dest)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, f"could not parse CSV: {e}")
    meta = extract_schema(df, dataset_id=dest.stem)
    meta["dataset_path"] = str(dest)
    return {"dataset_id": dest.stem, "metadata": meta}


@app.post("/api/sessions/{sid}/chat")
def chat(sid: str, req: ChatRequest, sess=Depends(owned_session),
         actor: str = Depends(actor_id)) -> dict:
    return orch.chat(sid, req.message, actor=actor)


@app.post("/api/sessions/{sid}/roles")
def set_roles(sid: str, req: RolesRequest, sess=Depends(owned_session),
              actor: str = Depends(actor_id)) -> dict:
    return orch.set_roles(sid, req.overrides, actor=actor, reason=req.reason)


@app.post("/api/sessions/{sid}/workflow/start")
def start_workflow_v2(sid: str, req: WorkflowStartRequest, sess=Depends(owned_session),
                      actor: str = Depends(actor_id)) -> dict:
    try:
        return orch.start_workflow(sid, req.workflow, req.params, actor=actor)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.post("/api/sessions/{sid}/workflow")
def start_workflow(sid: str, req: WorkflowRequest, sess=Depends(owned_session),
                   actor: str = Depends(actor_id)) -> dict:
    try:
        return orch.start_workflow(sid, req.name, req.params, actor=actor)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.post("/api/sessions/{sid}/workflow/resume")
def resume_workflow(sid: str, req: ResumeRequest, sess=Depends(owned_session),
                    actor: str = Depends(actor_id)) -> dict:
    # A rejection does no compute — it records the signed decision and returns,
    # so it always runs inline. Only an approval can start a long leg.
    if req.background and req.approve:
        if not orch.get_session(sid).pending_review:
            raise HTTPException(400, "no pending review to resume")
        job_id = jobs.submit(
            session_id=sid, kind="workflow_resume",
            fn=lambda: orch.resume_workflow(sid, True, actor=actor, reason=req.reason))
        return {"job_id": job_id, "status": "running", "kind": "workflow_resume"}
    try:
        return orch.resume_workflow(sid, req.approve, actor=actor, reason=req.reason)
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/sessions/{sid}/nca/refit_lz")
def refit_lz(sid: str, req: RefitLzRequest, sess=Depends(owned_session)) -> dict:
    from app.compute.nca import refit_lambda_z_manual
    if len(req.selected_times) != len(req.selected_concs):
        raise HTTPException(400, "selected_times and selected_concs must be the same length")
    try:
        return refit_lambda_z_manual(req.selected_times, req.selected_concs)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/sessions/{sid}/pkmodel")
def run_pk_model(sid: str, req: PkModelRequest, sess=Depends(owned_session),
                 actor: str = Depends(actor_id)) -> dict:
    try:
        return orch.run_pk_model(sid, model_key=req.model_key,
                                 compare=req.compare, models=req.models, actor=actor)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/sessions/{sid}/simulate")
def simulate_pk(sid: str, req: SimulateRequest, sess=Depends(owned_session),
                actor: str = Depends(actor_id)) -> dict:
    try:
        return orch.simulate_pk(sid, req.model_dump(), actor=actor)
    except ValueError as e:
        raise HTTPException(400, str(e))


class VpcRequest(BaseModel):
    """Optional VPC options; all-default reproduces the plain pooled VPC."""
    stratify_by: str | None = None
    dose_normalize: bool = False
    x_by: str = "time"
    exposure_check: bool = False
    blq_check: bool = False


@app.post("/api/sessions/{sid}/vpc")
def run_vpc(sid: str, req: VpcRequest | None = None, sess=Depends(owned_session),
            actor: str = Depends(actor_id)) -> dict:
    args = req.model_dump() if req is not None else {}
    try:
        return orch.run_tool(sid, "run_vpc", "modeler", args, actor=actor)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/sessions/{sid}/diagnostics")
def run_diagnostics(sid: str, sess=Depends(owned_session),
                    actor: str = Depends(actor_id)) -> dict:
    try:
        return orch.run_tool(sid, "run_diagnostics", "modeler", {}, actor=actor)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/sessions/{sid}/forest")
def run_covariate_forest(sid: str, sess=Depends(owned_session),
                         actor: str = Depends(actor_id)) -> dict:
    try:
        return orch.run_tool(sid, "run_covariate_forest", "modeler", {}, actor=actor)
    except ValueError as e:
        raise HTTPException(400, str(e))


class NlmeRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    method: str = "focei"
    model_key: str | None = None
    iiv_params: list[str] | None = None
    error_model: str = "proportional"


@app.post("/api/sessions/{sid}/nlme")
def run_nlme(sid: str, req: NlmeRequest, sess=Depends(owned_session),
             actor: str = Depends(actor_id)) -> dict:
    """Submit the (slow) population fit as a background job; poll /jobs/{id}."""
    body = req.model_dump()
    job_id = jobs.submit(session_id=sid, kind="nlme",
                         fn=lambda: orch.run_tool(sid, "run_nlme", "modeler", body, actor=actor))
    return {"job_id": job_id, "status": "running", "kind": "nlme"}


class SimestRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    confirm: bool = False
    design: dict = Field(default_factory=dict)
    n_rep: int = 5
    params: list[str] | None = None
    ci_target_pct: float | None = None
    method: str = "focei"


@app.post("/api/sessions/{sid}/simest")
def run_simest(sid: str, req: SimestRequest, sess=Depends(owned_session),
               actor: str = Depends(actor_id)) -> dict:
    """Submit the simulation-estimation precision check as a background job;
    poll /jobs/{id}. `agent="simulator"` (never "modeler") -- this tool is not
    LLM-reachable from chat; see app.tools.simest_tools for why that matters.
    Runs several real NLME fits (minutes to tens of minutes) -- requires
    `confirm=true` in the request body."""
    body = req.model_dump()
    job_id = jobs.submit(session_id=sid, kind="simest",
                         fn=lambda: orch.run_tool(sid, "run_simest", "simulator", body, actor=actor))
    return {"job_id": job_id, "status": "running", "kind": "simest"}


class EngineComparisonRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    candidates: list[dict] | None = None
    engines: list[str] | None = None
    model_key: str | None = None


@app.post("/api/sessions/{sid}/engine_comparison")
def engine_comparison(sid: str, req: EngineComparisonRequest, sess=Depends(owned_session),
                      actor: str = Depends(actor_id)) -> dict:
    """Submit the (slow) cross-engine model comparison as a background job.

    Fits the candidate(s) across the requested engines (pharmagent FOCE-I/SAEM and,
    if installed, nlmixr2) and ranks them by engine-agnostic prediction accuracy.
    Poll /jobs/{id} for the result; the winner is written to
    ``state.engine_comparison_results``.
    """
    body = req.model_dump()
    job_id = jobs.submit(session_id=sid, kind="engine_comparison",
                         fn=lambda: orch.run_tool(sid, "run_engine_comparison", "modeler",
                                                  body, actor=actor))
    return {"job_id": job_id, "status": "running", "kind": "engine_comparison"}


class ScmRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_key: str | None = None
    iiv_params: list[str] | None = None
    error_model: str = "proportional"
    candidates: list[dict] | None = None
    forward_p: float = 0.05
    backward_p: float = 0.01


@app.post("/api/sessions/{sid}/scm")
def run_scm(sid: str, req: ScmRequest, sess=Depends(owned_session),
            actor: str = Depends(actor_id)) -> dict:
    """Submit the (slow) stepwise covariate search as a background job."""
    body = req.model_dump()
    job_id = jobs.submit(session_id=sid, kind="scm",
                         fn=lambda: orch.run_tool(sid, "run_scm", "modeler", body, actor=actor))
    return {"job_id": job_id, "status": "running", "kind": "scm"}


class ForecastRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    dose: float = 100.0
    tau: float = 24.0
    measured: list[dict] | None = None
    wt: float = 70.0
    cov: dict | None = None
    target: float | None = None
    target_metric: str = "cmin"
    tmax: float | None = None


@app.post("/api/sessions/{sid}/forecast")
def run_forecast(sid: str, req: ForecastRequest, sess=Depends(owned_session),
                 actor: str = Depends(actor_id)) -> dict:
    """MAP/TDM forecast from the fitted NLME model (fast — runs synchronously)."""
    try:
        return orch.run_tool(sid, "forecast_map", "modeler", req.model_dump(), actor=actor)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/sessions/{sid}/jobs/{job_id}")
def get_job(sid: str, job_id: str, sess=Depends(owned_session)) -> dict:
    """Poll a background job. status is 'running' | 'done' | 'error'; when done,
    ``result`` is the tool's normal response (summary + state + audit_ok)."""
    job = jobs.get(job_id)
    if not job or job.get("session_id") != sid:
        raise HTTPException(404, "job not found")
    return job


@app.post("/api/sessions/{sid}/dosesweep")
def run_dose_sweep(sid: str, req: DoseSweepRequest, sess=Depends(owned_session),
                   actor: str = Depends(actor_id)) -> dict:
    try:
        return orch.run_tool(sid, "run_dose_sweep", "simulator", req.model_dump(), actor=actor)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/sessions/{sid}/flexplot")
def run_flexplot(sid: str, req: FlexplotRequest, sess=Depends(owned_session),
                 actor: str = Depends(actor_id)) -> dict:
    try:
        return orch.run_tool(sid, "generate_flexplot", "data_manager", req.model_dump(), actor=actor)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/sessions/{sid}/variables")
def list_variables(sid: str, sess=Depends(owned_session)) -> dict:
    """Typed variable list for the flexplot picker (metadata only, no raw rows)."""
    from app.compute.flexplot import plottable_variables
    meta = sess.state.dataset_metadata
    if not meta:
        raise HTTPException(400, "no dataset loaded")
    return {"variables": plottable_variables(meta),
            "detected_roles": meta.get("detected_roles", {})}


@app.get("/api/sessions/{sid}/audit")
def get_audit(sid: str, sess=Depends(owned_session)) -> dict:
    # Take the session lock so a concurrent background job can't be mid-append.
    with orch.session_lock(sid):
        entries = sess.audit.to_list()
        return {"entries": entries, "verified": sess.audit.verify(), "count": len(entries)}


@app.get("/api/sessions/{sid}/state")
def get_state(sid: str, sess=Depends(owned_session)) -> dict:
    with orch.session_lock(sid):
        return sess.state.model_dump()


@app.get("/api/sessions/{sid}/exports")
def list_exports(sid: str, sess=Depends(owned_session)) -> dict:
    return {"available": exporters.available(sess.state),
            "control": exporters.control_available(sess.state)}


@app.get("/api/sessions/{sid}/export/{kind}")
def export_csv(sid: str, kind: str, sess=Depends(owned_session)) -> Response:
    try:
        body = exporters.export_csv(sess.state, kind)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not body:
        raise HTTPException(404, f"no {kind} results to export")
    return Response(content=body, media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{kind}_{sid}.csv"'})


@app.get("/api/sessions/{sid}/export/control/{kind}")
def export_control(sid: str, kind: str, sess=Depends(owned_session)) -> Response:
    """NONMEM (.ctl) or mrgsolve (.cpp) control stream seeded from the NLME fit."""
    try:
        text, ext = exporters.export_control(sess.state, kind)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return Response(content=text, media_type="text/plain",
                    headers={"Content-Disposition": f'attachment; filename="{kind}_{sid}.{ext}"'})


@app.get("/api/sessions/{sid}/cdisc")
def export_cdisc(sid: str, sess=Depends(owned_session)) -> Response:
    """Download a CDISC ADaM-aligned package (ADPC + ADPP + define.xml) as a zip."""
    if not sess.state.nca_parameters:
        raise HTTPException(404, "run NCA first — no parameters to export as ADaM")
    roles = (sess.state.dataset_metadata or {}).get("detected_roles", {})
    df = sess.ctx.dataset_store.get(sess.state.dataset_id)
    body = cdisc.build_package(sess.state, df, roles)
    return Response(content=body, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="cdisc_adam_{sid}.zip"'})


class CaptureSkillRequest(BaseModel):
    name: str
    description: str = ""
    goal: str = ""


class RunSkillRequest(BaseModel):
    dataset_path: str


@app.post("/api/sessions/{sid}/capture-skill")
def capture_skill(sid: str, req: CaptureSkillRequest, sess=Depends(owned_session),
                  actor: str = Depends(actor_id)) -> dict:
    """Distill this session's analysis sequence into a named, replayable skill."""
    try:
        return orch.capture_skill(sid, req.name, description=req.description,
                                  goal=req.goal, actor=actor)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/skills")
def list_skills(owner: str | None = Depends(current_owner)) -> dict:
    return {"skills": [s.to_dict() for s in orch.skills.list(owner=owner)]}


@app.get("/api/skills/{name}")
def get_skill(name: str, owner: str | None = Depends(current_owner)) -> dict:
    skill = orch.skills.get(name)
    if skill is None:
        raise HTTPException(404, f"unknown skill: {name}")
    return skill.to_dict()


@app.get("/api/skills/{name}/markdown")
def get_skill_markdown(name: str, owner: str | None = Depends(current_owner)) -> Response:
    skill = orch.skills.get(name)
    if skill is None:
        raise HTTPException(404, f"unknown skill: {name}")
    return Response(content=skill.to_markdown(), media_type="text/markdown",
                    headers={"Content-Disposition": f'attachment; filename="{name}.SKILL.md"'})


@app.delete("/api/skills/{name}")
def delete_skill(name: str, owner: str | None = Depends(current_owner)) -> dict:
    if not orch.skills.delete(name):
        raise HTTPException(404, f"unknown skill: {name}")
    return {"deleted": name}


@app.post("/api/skills/{name}/run")
def run_skill(name: str, req: RunSkillRequest,
              owner: str | None = Depends(current_owner),
              actor: str = Depends(actor_id)) -> dict:
    """Replay a captured skill on a new dataset in a fresh session."""
    try:
        return orch.run_skill(name, dataset_path=req.dataset_path, owner=owner, actor=actor)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))


class ReviewRequest(BaseModel):
    goal: str | None = None
    max_iter: int = 3


@app.post("/api/sessions/{sid}/review")
def adversarial_review(sid: str, req: ReviewRequest, sess=Depends(owned_session),
                       actor: str = Depends(actor_id)) -> dict:
    """Run the adversarial reviewer loop: independently recompute + challenge the
    current results, emit severity-ranked findings, loop until the goal is met."""
    try:
        return orch.review_loop(sid, goal=req.goal, max_iter=req.max_iter, actor=actor)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/sessions/{sid}/report")
def generate_report(sid: str, sess=Depends(owned_session),
                    actor: str = Depends(actor_id)) -> dict:
    """(Re)generate the comprehensive DOCX report from all current results."""
    try:
        return orch.run_tool(sid, "generate_report", "report", {}, actor=actor)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))


@app.get("/api/sessions/{sid}/report/{filename}")
def download_report(sid: str, filename: str, sess=Depends(owned_session)) -> FileResponse:
    report_path = sess.state.report_path
    if not report_path or Path(report_path).name != filename:
        raise HTTPException(404, "report not found")
    p = Path(report_path)
    if not p.exists():
        raise HTTPException(404, "report file missing")
    return FileResponse(str(p), filename=filename,
                        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")


class Report272Request(BaseModel):
    drug_name: str = ""
    sponsor: str = ""
    study_id: str = ""
    route: str = "oral"
    indication: str = ""
    pop_description: str = "healthy adult volunteers"
    dose_range: str = ""
    matrix: str = "plasma"
    assay_lloq: str = ""


@app.post("/api/sessions/{sid}/report/272")
def generate_report_272(sid: str, req: Report272Request,
                        sess=Depends(owned_session),
                        actor: str = Depends(actor_id)) -> dict:
    """Generate an ICH M4E Module 2.7.2 CTD-structured DOCX from current results."""
    try:
        return orch.run_tool(sid, "generate_272", "regulatory",
                             req.model_dump(exclude_none=True), actor=actor)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))


@app.get("/api/sessions/{sid}/report/272/{filename}")
def download_report_272(sid: str, filename: str,
                        sess=Depends(owned_session)) -> FileResponse:
    report_path = sess.state.regulatory_report_path
    if not report_path or Path(report_path).name != filename:
        raise HTTPException(404, "Module 2.7.2 report not found or filename mismatch")
    p = Path(report_path)
    if not p.exists():
        raise HTTPException(404, "report file missing")
    return FileResponse(str(p), filename=filename,
                        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")


# ── static frontend (single-process desktop app) ───────────────────────────────
# When the React frontend has been built (`npm run build` -> frontend/dist), serve
# it from the same origin so the whole app runs on one port (no separate Vite dev
# server). The SPA calls same-origin /api/*, which the routes above already own —
# this mount is registered LAST, so /api never falls through to static files. In
# dev (no dist dir) this is skipped entirely and the Vite proxy handles /api.
def _frontend_dist() -> Path | None:
    env = settings.__dict__.get("frontend_dist") or None
    import os
    cand = os.environ.get("PHARMAGENT_FRONTEND_DIST")
    candidates = [Path(cand)] if cand else []
    candidates.append(Path(__file__).resolve().parents[2] / "frontend" / "dist")
    if env:
        candidates.insert(0, Path(env))
    for c in candidates:
        if (c / "index.html").is_file():
            return c
    return None


_DIST = _frontend_dist()
if _DIST is not None:
    from fastapi.responses import FileResponse as _FR
    from fastapi.staticfiles import StaticFiles

    _INDEX = _DIST / "index.html"

    @app.get("/", include_in_schema=False)
    def _spa_root() -> _FR:
        return _FR(str(_INDEX), media_type="text/html")

    # Hashed asset bundles live under /assets; mount them directly.
    app.mount("/assets", StaticFiles(directory=str(_DIST / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def _spa_fallback(full_path: str) -> _FR:
        """Serve a real static file if it exists, else index.html for client-side
        routing. /api/* is matched by the routes above before reaching here."""
        target = (_DIST / full_path).resolve()
        # confine to the dist dir via a true directory-boundary check (a string
        # prefix would also match a sibling like ``dist-evil``) and require a real file
        if target.is_relative_to(_DIST.resolve()) and target.is_file():
            return _FR(str(target))
        return _FR(str(_INDEX), media_type="text/html")

    log.info("serving_frontend", extra={"dist": str(_DIST)})
