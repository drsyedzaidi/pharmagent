"""Non-parametric bootstrap tool wiring.

SAFETY: registered under ``agent="simulator"``, the same deliberately
unroutable agent as ``run_simest``. ``simulator`` is absent from
``app.agents.definitions.AGENTS``/``DESCRIPTIONS`` and
``app.agents.supervisor.KEYWORDS``, so ``Supervisor.route`` cannot select it
and ``Agent.run_turn`` never puts this tool in the list an LLM sees. A bootstrap
runs hundreds of real NLME fits -- easily hours -- and the project guardrail is
"never submit a real NLME/SCM fit from an automated loop". An LLM-reachable
registration would violate that on the first chat turn asking about parameter
precision.

``confirm=True`` is required unconditionally. A threshold on subject or
replicate count would be porous: 200 replicates of a *small* model is still
hundreds of fits.
"""
from __future__ import annotations

from typing import Any

from app.compute.bootstrap import run_bootstrap as compute_run_bootstrap
from app.core.pharmstate import PharmState
from app.tools.base import Tool, ToolContext, ToolResult

# Statuses reached before any replicate ran. These must NOT overwrite a
# previously successful `bootstrap_results`: an admission-style rejection
# destroying a completed multi-hour run is the defect the simest tests guard.
_PRE_RUN_STATUSES = {"confirm_required", "no_fit", "insufficient_subjects",
                     "no_parameters", "no_dataset"}


def run_bootstrap(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    if not args.get("confirm"):
        status = {"status": "confirm_required",
                  "message": ("A bootstrap re-fits the model on hundreds of resampled "
                              "datasets (hours, holding the session lock). Pass "
                              "confirm=true to run it.")}
        return ToolResult(summary="Bootstrap requires confirm=true.",
                          action="run_bootstrap(confirm_required)",
                          writes={}, result=status)

    nl = state.nlme_results if (state.nlme_results or {}).get("status") == "ok" else None
    if nl is None:
        status = {"status": "no_fit",
                  "message": "Fit a converged run_nlme model first — the bootstrap "
                             "resamples around it and is compared against its "
                             "asymptotic standard errors."}
        return ToolResult(summary="Bootstrap skipped: no converged NLME fit.",
                          action="run_bootstrap(no_fit)", writes={}, result=status)

    df = ctx.dataset_store.get(state.dataset_id)
    if df is None:
        status = {"status": "no_dataset", "message": "Load a dataset first."}
        return ToolResult(summary="Bootstrap skipped: no dataset.",
                          action="run_bootstrap(no_dataset)", writes={}, result=status)

    # Local import: pkmodel_tools imports this module's siblings, and the NLME
    # engine is heavy — keep both off the module-import path.
    from app.compute.nlme import population_fit
    from app.tools.pkmodel_tools import _build_subjects, _roles

    model_key = nl.get("model_key")
    subjects, _multi, _has_pd = _build_subjects(df, _roles(df, state), with_blq=True)

    # Stratify on a per-subject covariate when asked. Without it a resample can
    # under-represent a study arm or dose group; see the compute docstring.
    strat_col = args.get("stratify_by")
    strata = None
    if strat_col:
        strata = [str((s.get("cov") or {}).get(strat_col, "?")) for s in subjects]

    n_boot = int(args.get("n_boot", 200))
    params = tuple(args["params"]) if args.get("params") else None
    method = args.get("method", "focei")
    if method not in ("focei", "saem", "focei_saem", "auto"):
        method = "focei"

    def fit_fn(rep_subjects: list[dict], seed: int) -> dict:
        # compute_uncertainty=False: the bootstrap IS the uncertainty estimate,
        # so a per-replicate Hessian would be pure cost for an unused number.
        res = population_fit(model_key, rep_subjects, method=method,
                             iiv_params=list(nl.get("iiv_params") or []),
                             error_model=nl.get("error_model", "proportional"),
                             covariate_model=_cov_spec(nl),
                             compute_uncertainty=False, seed=seed)
        return {"status": "ok", **res}

    try:
        out = compute_run_bootstrap(model_key, subjects, nl, fit_fn=fit_fn,
                                    n_boot=n_boot, params=params, strata=strata)
    except ValueError as e:
        status = {"status": "invalid_request", "message": str(e)}
        return ToolResult(summary=f"Bootstrap skipped: {e}",
                          action="run_bootstrap(invalid_request)",
                          writes={}, result=status)

    if out["status"] in _PRE_RUN_STATUSES:
        return ToolResult(
            summary=f"Bootstrap skipped: {out.get('message', out['status'])}",
            action=f"run_bootstrap({out['status']})", writes={}, result=out)

    if out["status"] != "ok":
        return ToolResult(
            summary=f"Bootstrap incomplete: {out.get('message', out['status'])}",
            action=f"run_bootstrap({out['status']})",
            writes={"bootstrap_results": out}, result=out)

    widest = max((c for c in out["comparison"]
                  if c.get("width_ratio_boot_over_asymptotic") is not None),
                 key=lambda c: c["width_ratio_boot_over_asymptotic"], default=None)
    tail = ""
    if widest is not None:
        r = widest["width_ratio_boot_over_asymptotic"]
        tail = (f" Widest divergence: {widest['parameter']} bootstrap CI is "
                f"{r:.2f}x the asymptotic width"
                + (" (asymptotic SEs look optimistic)." if r > 1.25 else "."))
    return ToolResult(
        summary=(f"Bootstrap on {model_key}: {out['n_ok']}/{out['n_completed']} "
                 f"replicates usable ({100 * out['success_rate']:.0f}%), "
                 f"{len(out['parameters'])} parameters."
                 + (" Stratified." if out["stratified"] else "") + tail),
        action=f"run_bootstrap({model_key})",
        writes={"bootstrap_results": out},
        result={"status": out["status"], "n_ok": out["n_ok"],
                "n_completed": out["n_completed"],
                "success_rate": out["success_rate"],
                "stratified": out["stratified"],
                "parameters": out["parameters"],
                "comparison": out["comparison"],
                "notes": out["notes"]})


def _cov_spec(nl: dict[str, Any]) -> list[dict] | None:
    """Re-state the fitted covariate model so each replicate estimates the SAME
    structure. Omitting it would bootstrap a *different* (covariate-free) model
    and the intervals would not describe the fit being reported."""
    effects = nl.get("covariate_effects") or []
    spec = [{"param": e["param"], "covariate": e["covariate"],
             "kind": e.get("kind", "power"),
             **({"center": e["center"]} if e.get("center") is not None else {})}
            for e in effects if e.get("param") and e.get("covariate")]
    return spec or None


TOOLS = [
    Tool("run_bootstrap",
         "Non-parametric bootstrap of a converged population fit: resamples "
         "SUBJECTS with replacement, re-fits each replicate, and reports "
         "empirical percentile confidence intervals alongside the model's "
         "asymptotic ones. The FDA PopPK guidance names bootstrap as one of "
         "several parameter-precision methods and states no single method "
         "suffices; the asymptotic SE assumes multivariate-normal uncertainty, "
         "and comparing the two intervals is a direct check on that assumption. "
         "VERY EXPENSIVE — hundreds of real fits, typically hours — so "
         "`confirm=true` is required. Optionally stratify by a covariate so "
         "each study arm or dose group stays represented.",
         "simulator",
         {"type": "object",
          "properties": {
              "confirm": {"type": "boolean"},
              "n_boot": {"type": "integer", "minimum": 1, "maximum": 1000},
              "params": {"type": "array", "items": {"type": "string"}},
              "stratify_by": {"type": "string"},
              "method": {"type": "string",
                         "enum": ["focei", "saem", "focei_saem", "auto"]},
          },
          "required": ["confirm"]},
         run_bootstrap),
]
