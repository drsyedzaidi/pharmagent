"""Governed protocol simulation workflow tools.

Provides:
  - ``load_approved_fit`` (modeler): verifies converged NLME fit + approved review
  - ``propose_simulation_plan`` (simulator): builds structured plan object for gate review
  - ``simulate_trial_pta`` (simulator): multi-replicate band x regimen PTA + MC SE
  - ``simulate_special_populations`` (simulator): plan-scoped special-pop / pediatric sim
  - ``propagate_parameter_uncertainty`` (simulator, expensive+proposable): outer-loop
    parameter uncertainty (bootstrap / SIR / profile) propagated into PTA intervals
"""
from __future__ import annotations

import time
from typing import Any

import pandas as pd

from app.compute.protocol_sim import (
    TOOL_VERSION,
    UNCERTAINTY_METHODS,
    build_simulation_plan,
    canonical_hash,
    extract_approved_fit,
    propagate_uncertainty_to_pta,
    run_plan_special_populations,
    run_trial_pta,
)
from app.core.pharmstate import PharmState, ProtocolSimDict
from app.tools.base import Tool, ToolContext, ToolResult


def _dataset_covariate_ranges(state: PharmState, ctx: ToolContext) -> dict[str, list[float]]:
    """Extract observed [min, max] ranges from the loaded dataset for WT, AGE, CRCL, PMA."""
    ranges: dict[str, list[float]] = {}
    if not state.dataset_id or state.dataset_id not in ctx.dataset_store:
        return ranges
    df = ctx.dataset_store[state.dataset_id]
    col_map = {c.upper(): c for c in df.columns}

    for candidates, target_key in (
        (("WT", "WEIGHT", "BW", "BWT", "BODYWEIGHT"), "wt_kg"),
        (("AGE", "AGE_YR", "AGEY"), "age_yr"),
        (("CRCL", "EGFR", "CLCR", "RF", "GFR"), "crcl_ml_min"),
        (("PMA", "PMA_WEEKS", "PMAW"), "pma_weeks"),
    ):
        for cand in candidates:
            if cand in col_map:
                vals = pd.to_numeric(df[col_map[cand]], errors="coerce").dropna()
                if len(vals) > 0:
                    ranges[target_key] = [round(float(vals.min()), 2), round(float(vals.max()), 2)]
                break
    return ranges


def _available_uncertainty_methods(state: PharmState) -> list[str]:
    """Engines whose REAL result is stored in state; the plan defaults to the best."""
    stored = (("sir", state.sir_results), ("bootstrap", state.bootstrap_results),
              ("profile", state.profile_results))
    return [m for m, res in stored if isinstance(res, dict) and res.get("status") == "ok"]


def load_approved_fit(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Verify converged NLME fit and approved adversarial review, and initialize
    ``state.protocol_sim`` with ``source_fit`` and ``provenance``."""
    t0 = time.perf_counter()
    ds_ranges = _dataset_covariate_ranges(state, ctx)
    source_fit, provenance = extract_approved_fit(
        state.nlme_results,
        state.review_results,
        dataset_covariate_ranges=ds_ranges,
    )
    elapsed = round(time.perf_counter() - t0, 6)
    provenance["wall_clock"]["load_approved_fit"] = elapsed
    ps = ProtocolSimDict({
        **(state.protocol_sim or {}),
        "source_fit": source_fit,
        "provenance": provenance,
    })
    return ToolResult(
        summary=(
            f"Loaded approved fit {source_fit['fit_id']} ({source_fit['label']}, "
            f"review hash {provenance['review_hash']})."
        ),
        action=f"load_approved_fit({source_fit['fit_id']})",
        writes={"protocol_sim": ps},
        result={"status": "ok", "fit_id": source_fit["fit_id"], "review_hash": provenance["review_hash"]},
    )


def propose_simulation_plan(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Emit a structured simulation plan object into ``state.protocol_sim.plan``."""
    t0 = time.perf_counter()
    ps = dict(state.protocol_sim or {})
    source_fit = ps.get("source_fit")
    provenance = dict(ps.get("provenance") or {})
    if not source_fit:
        ds_ranges = _dataset_covariate_ranges(state, ctx)
        source_fit, provenance = extract_approved_fit(
            state.nlme_results,
            state.review_results,
            dataset_covariate_ranges=ds_ranges,
        )
        ps["source_fit"] = source_fit

    plan = build_simulation_plan(source_fit, args,
                                 available_methods=_available_uncertainty_methods(state))
    plan_hash = canonical_hash(plan)
    elapsed = round(time.perf_counter() - t0, 6)

    tool_versions = dict(provenance.get("tool_versions") or {})
    tool_versions["propose_simulation_plan"] = TOOL_VERSION
    wall_clock = dict(provenance.get("wall_clock") or {})
    wall_clock["propose_simulation_plan"] = elapsed

    provenance.update({
        "plan_hash": plan_hash,
        "seed": int(plan["seed"]),
        "tool_versions": tool_versions,
        "wall_clock": wall_clock,
    })
    ps["plan"] = plan
    ps["provenance"] = provenance
    wrapped = ProtocolSimDict(ps)
    return ToolResult(
        summary=(
            f"Proposed simulation plan ({len(plan['bands'])} band(s), "
            f"{len(plan['regimens'])} regimen(s), {plan['n_subjects']} subjects x "
            f"{plan['replicates']} replicates, seed={plan['seed']}, "
            f"uncertainty={plan['uncertainty_method']}, plan_hash={plan_hash})."
        ),
        action=f"propose_simulation_plan({plan_hash})",
        writes={"protocol_sim": wrapped},
        result={"status": "ok", "plan": plan, "plan_hash": plan_hash},
    )


def simulate_trial_pta(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Run trial PTA simulation across the approved plan's bands, regimens, and replicates."""
    t0 = time.perf_counter()
    ps = dict(state.protocol_sim or {})
    source_fit = ps.get("source_fit")
    plan = ps.get("plan")
    if not source_fit or not plan:
        raise ValueError(
            "simulate_trial_pta requires protocol_sim.source_fit and protocol_sim.plan. "
            "Run load_approved_fit and propose_simulation_plan first."
        )

    pta_out = run_trial_pta(source_fit, plan)
    elapsed = round(time.perf_counter() - t0, 6)

    provenance = dict(ps.get("provenance") or {})
    plan_hash = canonical_hash(plan)
    tool_versions = dict(provenance.get("tool_versions") or {})
    tool_versions["simulate_trial_pta"] = TOOL_VERSION
    wall_clock = dict(provenance.get("wall_clock") or {})
    wall_clock["simulate_trial_pta"] = elapsed
    provenance.update({
        "plan_hash": plan_hash,
        "seed": int(plan.get("seed", 42)),
        "tool_versions": tool_versions,
        "wall_clock": wall_clock,
    })

    ps["pta"] = pta_out
    ps["recommendations"] = pta_out.get("recommendations", [])
    ps["provenance"] = provenance
    wrapped = ProtocolSimDict(ps)
    return ToolResult(
        summary=(
            f"Simulated trial PTA over {len(pta_out['rows'])} band x regimen cells "
            f"({pta_out['replicates']} replicates of {pta_out['n_subjects']} subjects, "
            f"seed={pta_out['seed']})."
        ),
        action=f"simulate_trial_pta(seed={pta_out['seed']})",
        writes={"protocol_sim": wrapped},
        result={
            "status": "ok",
            "seed": pta_out["seed"],
            "n_rows": len(pta_out["rows"]),
            "recommendations": pta_out["recommendations"],
        },
    )


def simulate_special_populations(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Run special-population and pediatric simulations strictly for the bands in ``protocol_sim.plan``."""
    t0 = time.perf_counter()
    ps = dict(state.protocol_sim or {})
    source_fit = ps.get("source_fit")
    plan = ps.get("plan")
    if not source_fit or not plan:
        raise ValueError(
            "simulate_special_populations requires protocol_sim.source_fit and protocol_sim.plan."
        )

    sp_out = run_plan_special_populations(source_fit, plan)
    elapsed = round(time.perf_counter() - t0, 6)

    provenance = dict(ps.get("provenance") or {})
    tool_versions = dict(provenance.get("tool_versions") or {})
    tool_versions["simulate_special_populations"] = TOOL_VERSION
    wall_clock = dict(provenance.get("wall_clock") or {})
    wall_clock["simulate_special_populations"] = elapsed
    provenance.update({
        "plan_hash": canonical_hash(plan),
        "seed": int(plan.get("seed", 42)),
        "tool_versions": tool_versions,
        "wall_clock": wall_clock,
    })

    ps["special_populations"] = sp_out
    ps["provenance"] = provenance
    wrapped = ProtocolSimDict(ps)
    return ToolResult(
        summary=(
            f"Simulated special populations for plan bands {sp_out['bands_simulated']} "
            f"(seed={sp_out['seed']})."
        ),
        action=f"simulate_special_populations(seed={sp_out['seed']})",
        writes={"protocol_sim": wrapped},
        result={
            "status": "ok",
            "bands_simulated": sp_out["bands_simulated"],
            "seed": sp_out["seed"],
        },
    )


def propagate_parameter_uncertainty(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Propagate parameter uncertainty as an outer loop into PTA.

    Uses the bootstrap / SIR / profile result stored in state when the plan names
    one; otherwise an asymptotic log-normal approximation that is labelled as such
    (and that the adversarial review flags when the plan asked for an engine).
    """
    t0 = time.perf_counter()
    ps = dict(state.protocol_sim or {})
    source_fit = ps.get("source_fit")
    plan = dict(ps.get("plan") or {})
    if not source_fit or not plan:
        raise ValueError(
            "propagate_parameter_uncertainty requires protocol_sim.source_fit and protocol_sim.plan."
        )

    if args.get("method"):
        plan["uncertainty_method"] = str(args["method"]).lower()
    if args.get("seed") is not None:
        plan["seed"] = int(args["seed"])

    n_draws = int(args.get("n_draws", 20))
    unc_out, updated_pta = propagate_uncertainty_to_pta(
        source_fit=source_fit,
        plan=plan,
        pta_result=ps.get("pta"),
        n_draws=n_draws,
        existing_bootstrap=state.bootstrap_results,
        existing_sir=state.sir_results,
        existing_profile=state.profile_results,
    )
    elapsed = round(time.perf_counter() - t0, 6)

    provenance = dict(ps.get("provenance") or {})
    tool_versions = dict(provenance.get("tool_versions") or {})
    tool_versions["propagate_parameter_uncertainty"] = TOOL_VERSION
    wall_clock = dict(provenance.get("wall_clock") or {})
    wall_clock["propagate_parameter_uncertainty"] = elapsed
    provenance.update({
        "plan_hash": canonical_hash(plan),
        "seed": int(plan.get("seed", 42)),
        "tool_versions": tool_versions,
        "wall_clock": wall_clock,
    })

    ps["plan"] = plan
    ps["pta"] = updated_pta
    ps["recommendations"] = updated_pta.get("recommendations", [])
    ps["uncertainty"] = unc_out
    ps["provenance"] = provenance
    wrapped = ProtocolSimDict(ps)
    requested = unc_out.get("requested_method")
    if unc_out["status"] == "unverifiable":
        summary = (
            f"Parameter uncertainty NOT propagated: {unc_out['reason']}. Run run_sir, "
            "run_bootstrap or run_profile on the source fit, then re-run this step."
        )
        action = "propagate_parameter_uncertainty(unverifiable)"
    else:
        if not unc_out.get("approximation"):
            how = f"from the stored {unc_out['method']} result"
        elif requested == unc_out["method"]:
            how = "asymptotic log-normal approximation from theta RSE%, as the approved plan requested"
        else:
            how = (f"asymptotic log-normal approximation from theta RSE%; the approved plan "
                   f"requested {requested} but no {requested} result is stored in state")
        summary = (
            f"Propagated parameter uncertainty ({unc_out['method']}: {how}) over "
            f"{unc_out['n_draws']} outer-loop draws into PTA intervals (seed={unc_out['seed']})."
        )
        action = f"propagate_parameter_uncertainty({unc_out['method']})"
    return ToolResult(
        summary=summary,
        action=action,
        writes={"protocol_sim": wrapped},
        result={
            "status": unc_out["status"],
            "method": unc_out["method"],
            "requested_method": requested,
            "approximation": bool(unc_out.get("approximation")),
            "reason": unc_out.get("reason"),
            "n_draws": unc_out["n_draws"],
            "seed": unc_out["seed"],
        },
    )


TOOLS = [
    Tool(
        "load_approved_fit",
        "Verify that a converged NLME population fit (nlme_results.status == 'ok') "
        "and an approved adversarial review (review_results.goal_met == True) are "
        "present in state, and load the approved fit and its provenance into "
        "protocol_sim.source_fit.",
        "modeler",
        {"type": "object", "properties": {}, "required": []},
        load_approved_fit,
    ),
    Tool(
        "propose_simulation_plan",
        "Propose a structured protocol simulation plan (bands, regimens, metrics, "
        "target_window with provenance, n_subjects, replicates, seed, and "
        "uncertainty_method) into protocol_sim.plan for human review.",
        "simulator",
        {
            "type": "object",
            "properties": {
                "bands": {"type": "array", "items": {"type": "object"}},
                "regimens": {"type": "array", "items": {"type": "object"}},
                "metrics": {"type": "array", "items": {"type": "string"}},
                "target_window": {"type": "object"},
                "n_subjects": {"type": "integer", "minimum": 1, "maximum": 100000},
                "replicates": {"type": "integer", "minimum": 1, "maximum": 1000},
                "seed": {"type": "integer"},
                "uncertainty_method": {"type": "string", "enum": list(UNCERTAINTY_METHODS)},
                "target_pta": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
            },
            "required": [],
        },
        propose_simulation_plan,
    ),
    Tool(
        "simulate_trial_pta",
        "Simulate virtual patient cohorts from the approved fit's IIV and the "
        "plan's covariate bands across replicates, returning PTA per band per "
        "regimen with Monte Carlo standard error (mc_se) and peak/trough limit checks.",
        "simulator",
        {"type": "object", "properties": {}, "required": []},
        simulate_trial_pta,
    ),
    Tool(
        "simulate_special_populations",
        "Run special-population and pediatric exposure simulations strictly for "
        "the covariate bands named in protocol_sim.plan using the plan's seed.",
        "simulator",
        {"type": "object", "properties": {}, "required": []},
        simulate_special_populations,
    ),
    Tool(
        "propagate_parameter_uncertainty",
        "Propagate parameter uncertainty as an outer loop into PTA per band per "
        "regimen so PTA carries a 95% parameter-uncertainty interval. Uses the "
        "bootstrap, SIR or likelihood-profile result stored in state when the "
        "approved plan names one; otherwise an asymptotic log-normal approximation "
        "from theta RSE%, labelled as such.",
        "simulator",
        {
            "type": "object",
            "properties": {
                "confirm": {"type": "boolean"},
                "method": {"type": "string", "enum": list(UNCERTAINTY_METHODS)},
                "n_draws": {"type": "integer", "minimum": 1, "maximum": 20000},
                "seed": {"type": "integer"},
            },
            "required": [],
        },
        propagate_parameter_uncertainty,
        expensive=True,
        proposable=True,
    ),
]
