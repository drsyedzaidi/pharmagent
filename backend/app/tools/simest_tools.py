"""Simulation-estimation tool wiring.

SAFETY: ``run_simest`` runs several real NLME fits (real minutes to tens of
minutes), and the project's guardrail is "never submit a real NLME/SCM fit
from an automated loop". It is reachable from the agent (chat) path as a
PROPOSAL only:

* ``agent="simulator"`` is a routable agent, so a design-check request reaches
  it and its LLM turn may select this tool and compose ``design``.
* ``expensive=True``: ``ToolRegistry.execute`` refuses it on the synchronous
  chat path (same choke point as ``run_nlme``), so the turn can never run it.
* ``proposable=True``: instead of being skipped, the refused call becomes
  ``state.pending_tool`` (audited ``propose_tool``). Nothing is computed.
* The pharmacometrician approves or rejects it
  (``POST /api/sessions/{sid}/chat/pending_tool``). Approval is the human
  ``confirm`` this tool requires (an LLM-supplied ``confirm`` is dropped); the
  approved call is submitted to the JobManager and run via
  ``Orchestrator.run_tool`` -- the same admission-controlled path as the
  ``/simest`` endpoint. Skill replay and workflow templates still refuse it.

``confirm=True`` is required UNCONDITIONALLY (not threshold-gated on a
subject/replicate count -- a threshold is porous: a routine design well under
any reasonable count still costs real minutes).
"""
from __future__ import annotations

from typing import Any

from app.compute.simest import CITATION_UNVERIFIED
from app.compute.simest import run_simest as compute_run_simest
from app.core.pharmstate import PharmState
from app.tools.base import Tool, ToolContext, ToolResult

# Statuses reached before any replicate ran (fast validation failures) --
# these must NOT overwrite a previously successful `simest_results` (an
# admission-control-style rejection destroying a completed run is exactly the
# defect this module's tests guard against).
_PRE_RUN_STATUSES = {"confirm_required", "no_nlme", "covariates_unsupported", "no_params",
                     "invalid_design"}


def run_simest(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    if not args.get("confirm"):
        status = {"status": "confirm_required",
                  "message": ("Simulation-estimation runs several real NLME fits "
                              "(minutes to tens of minutes total, and holds this "
                              "session's lock for the duration). Pass confirm=true "
                              "to proceed.")}
        return ToolResult(summary="Simulation-estimation requires confirm=true.",
                          action="run_simest(confirm_required)", writes={}, result=status)

    nl = state.nlme_results if (state.nlme_results or {}).get("status") == "ok" else None
    if nl is None:
        status = {"status": "no_nlme",
                  "message": "Fit a converged run_nlme model (covariate-free) first."}
        return ToolResult(summary="Simulation-estimation skipped: no converged NLME fit.",
                          action="run_simest(no_nlme)", writes={}, result=status)

    model_key = nl.get("model_key")
    design = args.get("design") or {}
    n_rep = int(args.get("n_rep", 5))
    params = tuple(args["params"]) if args.get("params") else None
    ci_target_pct = args.get("ci_target_pct")
    method = args.get("method", "focei")
    if method not in ("focei", "saem"):
        method = "focei"

    from app.compute.nlme import population_fit  # lazy: heavy + optional dependency

    def fit_fn(subjects: list[dict], seed: int) -> dict:
        res = population_fit(model_key, subjects, method=method,
                             iiv_params=list(nl.get("iiv_params") or []),
                             error_model=nl.get("error_model", "proportional"),
                             compute_uncertainty=True, seed=seed)
        return {"status": "ok", **res}

    try:
        out = compute_run_simest(model_key, design, nl, fit_fn=fit_fn, n_rep=n_rep,
                                 params=params, ci_target_pct=ci_target_pct)
    except ValueError as e:
        status = {"status": "invalid_design", "message": str(e)}
        return ToolResult(summary=f"Simulation-estimation skipped: {e}",
                          action="run_simest(invalid_design)", writes={}, result=status)

    if out["status"] in _PRE_RUN_STATUSES:
        return ToolResult(
            summary=f"Simulation-estimation skipped: {out.get('message', out['status'])}",
            action=f"run_simest({out['status']})", writes={}, result=out)

    crit = out.get("criterion", {})
    summary = (f"Simulation-estimation on {model_key}: {out['n_rep_completed']}/"
              f"{out['n_rep_planned']} replicate(s) completed, {out['n_point_evaluable']} "
              f"point-evaluable, {out['n_ci_evaluable']} CI-evaluable ({out['ci_validity']}). "
              f"Strict pass rate {crit.get('pct_within_60_140_strict')}%. {CITATION_UNVERIFIED}")
    return ToolResult(
        summary=summary,
        action=f"run_simest({model_key})",
        writes={"simest_results": out},
        result={"status": out["status"], "n_rep_completed": out["n_rep_completed"],
                "n_point_evaluable": out["n_point_evaluable"], "n_ci_evaluable": out["n_ci_evaluable"],
                "criterion": crit, "ci_validity": out["ci_validity"]})


TOOLS = [
    Tool("run_simest",
         "Simulation-estimation precision check for a proposed single-arm PK "
         "sampling design: simulate replicate trials from a converged run_nlme "
         "fit (COVARIATE-FREE models only) and re-fit each to check whether the "
         "95% CI of the structural-parameter estimate falls within 60-140% of "
         "its own point estimate (course-lecture criterion, citation unverified "
         "-- see the returned `citation` field). REAL cost: several real NLME "
         "fits, several minutes to tens of minutes total -- `confirm=true` is "
         "REQUIRED. `design` = {n_subjects, obs_t, dose|dose_per_kg, n_doses, "
         "tau, wt_mean, wt_cv_pct, lloq}. Capped at 10 replicates; this is a "
         "bounded precision CHECK, not a publication-grade study.",
         "simulator",
         {"type": "object",
          "properties": {
              "confirm": {"type": "boolean"},
              "design": {"type": "object"},
              "n_rep": {"type": "integer"},
              "params": {"type": "array", "items": {"type": "string"}},
              "ci_target_pct": {"type": "number"},
              "method": {"type": "string", "enum": ["focei", "saem"]},
          },
          "required": ["confirm", "design"]},
         run_simest, expensive=True, proposable=True),
]
