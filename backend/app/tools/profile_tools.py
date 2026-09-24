"""Log-likelihood profiling tool wiring.

SAFETY: ``agent="simulator"``, the deliberately unroutable agent shared with
``run_simest`` / ``run_bootstrap`` / ``run_sir`` — absent from
``app.agents.definitions.AGENTS``/``DESCRIPTIONS`` and
``app.agents.supervisor.KEYWORDS``, so no chat turn can route to it.

Cost: each profile point is a constrained re-optimization whose every objective
evaluation is a full population Laplace pass. Measured at ~120 s for ONE
parameter on a 16-subject model; a realistic cohort is far more, and the cost
scales with the number of parameters profiled. ``confirm=true`` is required and
``params`` should normally name the one or two parameters actually in question
rather than defaulting to all of them.
"""
from __future__ import annotations

from typing import Any

from app.compute.profile import run_profile as compute_run_profile
from app.core.pharmstate import PharmState
from app.tools.base import Tool, ToolContext, ToolResult

_PRE_RUN_STATUSES = {"confirm_required", "no_fit", "no_dataset",
                     "no_parameters", "profile_unavailable"}


def run_profile(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    if not args.get("confirm"):
        status = {"status": "confirm_required",
                  "message": ("Profiling re-optimizes the model at each of tens of "
                              "fixed parameter values (minutes per parameter, holding "
                              "the session lock). Pass confirm=true, and prefer naming "
                              "`params` rather than profiling everything.")}
        return ToolResult(summary="Profiling requires confirm=true.",
                          action="run_profile(confirm_required)",
                          writes={}, result=status)

    nl = state.nlme_results if (state.nlme_results or {}).get("status") == "ok" else None
    if nl is None:
        status = {"status": "no_fit",
                  "message": "Fit a converged run_nlme model first — profiling "
                             "measures how much the objective worsens away from "
                             "its estimates."}
        return ToolResult(summary="Profiling skipped: no converged NLME fit.",
                          action="run_profile(no_fit)", writes={}, result=status)

    df = ctx.dataset_store.get(state.dataset_id)
    if df is None:
        status = {"status": "no_dataset", "message": "Load a dataset first."}
        return ToolResult(summary="Profiling skipped: no dataset.",
                          action="run_profile(no_dataset)", writes={}, result=status)

    from app.compute.nlme import profile_ofv_factory
    from app.tools.pkmodel_tools import _build_subjects, _roles

    model_key = nl.get("model_key")
    subjects, _multi, _has_pd = _build_subjects(df, _roles(df, state), with_blq=True)
    fac = profile_ofv_factory(model_key, subjects, nl)
    if fac is None or not fac.get("estimates"):
        status = {"status": "profile_unavailable",
                  "message": ("Could not build the constrained-reoptimization "
                              "closure for this fit.")}
        return ToolResult(summary="Profiling skipped: closure unavailable.",
                          action="run_profile(profile_unavailable)",
                          writes={}, result=status)

    params = tuple(args["params"]) if args.get("params") else None
    if params:
        unknown = [p for p in params if p not in fac["estimates"]]
        if unknown:
            status = {"status": "no_parameters",
                      "message": (f"cannot profile {unknown}; available structural "
                                  f"parameters are {sorted(fac['estimates'])}.")}
            return ToolResult(summary=f"Profiling skipped: unknown parameters {unknown}.",
                              action="run_profile(no_parameters)",
                              writes={}, result=status)

    try:
        out = compute_run_profile(
            profile_ofv_fn=fac["profile_ofv_fn"], estimates=fac["estimates"],
            ofv_hat=fac["ofv_hat"], initial_step=fac["initial_step"],
            params=params, ci_level=float(args.get("ci_level", 0.95)))
    except ValueError as e:
        status = {"status": "invalid_request", "message": str(e)}
        return ToolResult(summary=f"Profiling skipped: {e}",
                          action="run_profile(invalid_request)",
                          writes={}, result=status)

    if out["status"] in _PRE_RUN_STATUSES:
        return ToolResult(summary=f"Profiling skipped: {out.get('message', out['status'])}",
                          action=f"run_profile({out['status']})",
                          writes={}, result=out)

    d = out["diagnostics"]
    lead = ""
    if d["fit_not_at_optimum"]:
        # The most important thing profiling can tell you, and the one finding
        # the other three precision methods structurally cannot produce: they
        # all ASSUME the reported optimum is one.
        lead = (" NOT AT AN OPTIMUM: a constrained fit beat the reported objective "
                f"for {', '.join(b['parameter'] for b in d['fit_not_at_optimum'])} — "
                "re-fit before using any of these intervals.")
    return ToolResult(
        summary=(f"Profiled {out['n_parameters']} parameter(s) on {model_key} "
                 f"in {out['n_evaluations']} constrained re-optimizations "
                 f"(dOFV cut-off {out['dofv_cutoff']})." + lead),
        action=f"run_profile({model_key})",
        writes={"profile_results": out},
        result={"status": out["status"], "dofv_cutoff": out["dofv_cutoff"],
                "n_evaluations": out["n_evaluations"],
                "parameters": [{k: p[k] for k in
                                ("parameter", "estimate", "profile_lo", "profile_hi",
                                 "asymmetry_ratio")} for p in out["parameters"]],
                "diagnostics": d, "notes": out["notes"]})


TOOLS = [
    Tool("run_profile",
         "Log-likelihood profiling: fixes a parameter at a range of values, "
         "re-optimizes all the others at each, and reads the confidence limits "
         "off where the objective worsens by the chi-square cut-off (3.84 at "
         "95%). The fourth parameter-precision method named in the FDA PopPK "
         "guidance. Its limits are found independently on each side, so the "
         "interval is ASYMMETRIC whenever the likelihood is — which a "
         "standard-error interval cannot express. It also independently checks "
         "whether the fit was at an optimum at all, which bootstrap, SIR and "
         "standard errors all simply assume. It gives interval BOUNDS ONLY — no "
         "distribution to simulate from, one parameter at a time. EXPENSIVE "
         "(minutes per parameter), so `confirm=true` is required and `params` "
         "should name the parameters actually in question.",
         "simulator",
         {"type": "object",
          "properties": {
              "confirm": {"type": "boolean"},
              "params": {"type": "array", "items": {"type": "string"}},
              "ci_level": {"type": "number", "minimum": 0.5, "maximum": 0.999},
          },
          "required": ["confirm"]},
         run_profile),
]
