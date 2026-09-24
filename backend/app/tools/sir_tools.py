"""Sampling Importance Resampling tool wiring.

SAFETY: registered under ``agent="simulator"``, the deliberately unroutable
agent used by ``run_simest`` and ``run_bootstrap`` — absent from
``app.agents.definitions.AGENTS``/``DESCRIPTIONS`` and
``app.agents.supervisor.KEYWORDS``, so ``Supervisor.route`` cannot select it
and ``Agent.run_turn`` never shows this tool to an LLM.

SIR is much cheaper than a bootstrap — M objective evaluations and NO refits —
but M is thousands of full population Laplace passes plus one numeric Hessian
to build the proposal, so it is still minutes to tens of minutes. ``confirm``
is therefore required, consistent with the other two.
"""
from __future__ import annotations

from typing import Any

from app.compute.sir import run_sir as compute_run_sir
from app.core.pharmstate import PharmState
from app.tools.base import Tool, ToolContext, ToolResult

# Reached before any sampling ran; must not overwrite a completed run.
_PRE_RUN_STATUSES = {"confirm_required", "no_fit", "no_dataset",
                     "no_parameters", "proposal_unavailable"}


def run_sir(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    if not args.get("confirm"):
        status = {"status": "confirm_required",
                  "message": ("SIR evaluates the objective at thousands of sampled "
                              "parameter vectors and builds a proposal from a numeric "
                              "Hessian (minutes to tens of minutes, holding the session "
                              "lock). Pass confirm=true to run it.")}
        return ToolResult(summary="SIR requires confirm=true.",
                          action="run_sir(confirm_required)", writes={}, result=status)

    nl = state.nlme_results if (state.nlme_results or {}).get("status") == "ok" else None
    if nl is None:
        status = {"status": "no_fit",
                  "message": "Fit a converged run_nlme model first — SIR samples "
                             "around its estimates and uses its covariance as the "
                             "proposal distribution."}
        return ToolResult(summary="SIR skipped: no converged NLME fit.",
                          action="run_sir(no_fit)", writes={}, result=status)

    df = ctx.dataset_store.get(state.dataset_id)
    if df is None:
        status = {"status": "no_dataset", "message": "Load a dataset first."}
        return ToolResult(summary="SIR skipped: no dataset.",
                          action="run_sir(no_dataset)", writes={}, result=status)

    from app.compute.nlme import sir_inputs  # lazy: heavy engine
    from app.tools.pkmodel_tools import _build_subjects, _roles

    model_key = nl.get("model_key")
    subjects, _multi, _has_pd = _build_subjects(df, _roles(df, state), with_blq=True)
    inp = sir_inputs(model_key, subjects, nl)
    if inp is None:
        status = {"status": "proposal_unavailable",
                  "message": ("Could not build a usable proposal distribution: the "
                              "information matrix at these estimates is not usable. "
                              "A bootstrap makes no such requirement and is the "
                              "fallback here.")}
        return ToolResult(summary="SIR skipped: no usable proposal.",
                          action="run_sir(proposal_unavailable)",
                          writes={}, result=status)

    # Carry the proposal's provenance into the result. A near-singular
    # information matrix is regularized to build the proposal, which makes it
    # LOOK better conditioned than the fit that produced it — the reader has to
    # be told, or SIR's intervals read as more trustworthy than they are.
    note = None
    if inp.get("near_singular"):
        note = ("PROPOSAL CAVEAT: the information matrix at these estimates is "
                "near-singular"
                + (f" (fit condition number {inp['fit_condition_number']:.3g})"
                   if inp.get("fit_condition_number") else "")
                + ". It was regularized to build the proposal, so the proposal "
                  "looks better conditioned than the fit does. SIR can correct a "
                  "proposal that is too wide, but cannot recover a direction the "
                  "data do not identify — treat these intervals as a lower bound "
                  "on the true uncertainty and prefer the bootstrap here.")

    try:
        out = compute_run_sir(
            ofv_fn=inp["ofv_fn"], x_hat=inp["x_hat"], cov=inp["cov"],
            ofv_hat=inp["ofv_hat"], decode_fn=inp["decode_fn"],
            n_resample=int(args.get("n_resample", 1000)),
            n_samples=args.get("n_samples"),
            inflation=float(args.get("inflation", 1.0)),
            n_estimated_params=inp["n_par"],
            proposal_note=note)
    except ValueError as e:
        status = {"status": "invalid_request", "message": str(e)}
        return ToolResult(summary=f"SIR skipped: {e}",
                          action="run_sir(invalid_request)", writes={}, result=status)

    if out["status"] in _PRE_RUN_STATUSES:
        return ToolResult(summary=f"SIR skipped: {out.get('message', out['status'])}",
                          action=f"run_sir({out['status']})", writes={}, result=out)
    if out["status"] != "ok":
        return ToolResult(
            summary=f"SIR incomplete: {out.get('message', out['status'])}",
            action=f"run_sir({out['status']})",
            writes={"sir_results": out}, result=out)

    d = out["diagnostics"]
    return ToolResult(
        summary=(f"SIR on {model_key}: {out['n_samples']} samples -> "
                 f"{out['n_resample']} resampled (M/m {out['m_over_m_ratio']}), "
                 f"ESS {d['effective_sample_size']:.0f}, resampled dOFV mean "
                 f"{d['dofv_mean_resampled']} vs df reference {d['df_reference']}."
                 + (" " + note.split(".")[0] + "." if note else "")),
        action=f"run_sir({model_key})",
        writes={"sir_results": out},
        result={"status": out["status"], "n_samples": out["n_samples"],
                "n_resample": out["n_resample"],
                "m_over_m_ratio": out["m_over_m_ratio"],
                "inflation": out["inflation"],
                "parameters": out["parameters"],
                "diagnostics": d, "notes": out["notes"]})


TOOLS = [
    Tool("run_sir",
         "Sampling importance resampling (Dosne et al. 2016) for parameter "
         "uncertainty: samples parameter vectors from the fit's covariance "
         "matrix, weights each by how much better the DATA like it than the "
         "proposal predicted, and resamples to give confidence intervals that "
         "need no normality assumption. Named by the FDA PopPK guidance "
         "alongside bootstrap and asymptotic standard errors. Unlike a "
         "bootstrap it runs NO refits — only objective evaluations — so it is "
         "far cheaper, and unlike asymptotic standard errors it can express "
         "ASYMMETRIC intervals. Still minutes to tens of minutes, so "
         "`confirm=true` is required. Use `inflation` > 1 when the proposal "
         "looks too narrow (the diagnostics say so).",
         "simulator",
         {"type": "object",
          "properties": {
              "confirm": {"type": "boolean"},
              "n_resample": {"type": "integer", "minimum": 1, "maximum": 5000},
              "n_samples": {"type": "integer", "minimum": 1, "maximum": 20000},
              "inflation": {"type": "number", "minimum": 0.1, "maximum": 10.0},
          },
          "required": ["confirm"]},
         run_sir),
]
