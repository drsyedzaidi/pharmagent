"""Modeler tool: cross-engine model comparison.

Fits candidate PK models across multiple estimation engines and selects the best
by ENGINE-AGNOSTIC prediction accuracy (prediction RMSE + VPC coverage). Native
OFV/AIC/BIC are reported per engine but never compared across engines. Wraps
``app.engines`` and follows the same dataset/roles/ToolResult conventions as
``run_nlme``.
"""
from __future__ import annotations

from typing import Any

from app.engines import (
    CandidateSpec,
    MockEngineAdapter,
    Nlmixr2Adapter,
    PharmAgentAdapter,
    run_matrix_subjects,
    select_winner,
)

from .base import Tool, ToolContext, ToolResult

_DEFAULT_ENGINES = ["pharmagent_focei", "nlmixr2"]

_ADAPTER_FACTORY = {
    "pharmagent": lambda: PharmAgentAdapter(),
    "pharmagent_focei": lambda: PharmAgentAdapter(method="focei"),
    "pharmagent_saem": lambda: PharmAgentAdapter(method="saem"),
    "nlmixr2": lambda: Nlmixr2Adapter(),
    "nlmixr2_focei": lambda: Nlmixr2Adapter(),
    "mock": lambda: MockEngineAdapter(),
}


def _resolve_candidates(args: dict[str, Any], state: Any) -> list[CandidateSpec]:
    raw = args.get("candidates")
    if not raw and args.get("model_key"):
        raw = [{"model_key": args["model_key"]}]
    if not raw:  # fall back to the last fitted structural model
        mk = None
        if state.nlme_results and state.nlme_results.get("model_key"):
            mk = state.nlme_results["model_key"]
        elif state.pk_model_results and state.pk_model_results.get("status") == "ok":
            pm = state.pk_model_results
            if pm.get("mode") == "compare":  # compare payload carries best_model, not model_key
                mk = pm.get("best_model") or (pm.get("best") or {}).get("model_key")
            else:  # fit payload splats a top-level model_key
                mk = pm.get("model_key")
        raw = [{"model_key": mk}] if mk else []
    specs = []
    for c in raw:
        if not c.get("model_key"):
            continue
        specs.append(CandidateSpec(
            model_key=c["model_key"], iiv_params=c.get("iiv_params"),
            error_model=c.get("error_model", "proportional"),
            method=c.get("method", "focei")))
    return specs


def _resolve_adapters(names: list[str] | None) -> list:
    adapters = []
    for n in (names or _DEFAULT_ENGINES):
        factory = _ADAPTER_FACTORY.get(str(n).lower())
        if factory:
            adapters.append(factory())
    return adapters or [PharmAgentAdapter(method="focei")]


def _skip(summary: str, tag: str, status: dict[str, Any]) -> ToolResult:
    return ToolResult(summary=summary, action=f"run_engine_comparison({tag})",
                      writes={"engine_comparison_results": status}, result=status)


def run_engine_comparison(state, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Fit candidates across engines; pick the winner on engine-agnostic metrics."""
    from app.compute.pk_models import REGISTRY
    from app.tools.pkmodel_tools import _build_subjects, _roles  # lazy: avoid import cycle

    candidates = _resolve_candidates(args, state)
    if not candidates:
        return _skip("Engine comparison skipped: no candidate model.", "no_model",
                     {"status": "no_model",
                      "message": "Provide candidates (model_key) or fit a model first."})

    unknown = sorted({c.model_key for c in candidates if c.model_key not in REGISTRY})
    if unknown:
        return _skip(f"Engine comparison skipped: unknown model(s) {unknown}.", "unknown_model",
                     {"status": "unknown_model", "message": f"unknown model_key(s): {unknown}"})

    if not state.dataset_id or state.dataset_id not in ctx.dataset_store:
        return _skip("Engine comparison skipped: no dataset loaded.", "no_dataset",
                     {"status": "no_dataset", "message": "Load a dataset first."})

    df = ctx.dataset_store[state.dataset_id]
    subjects, _multi, _has_pd = _build_subjects(df, _roles(df, state), with_blq=True)
    if len(subjects) < 2:
        return _skip("Engine comparison skipped: too few subjects.", "insufficient",
                     {"status": "insufficient", "message": "Need >=2 subjects for a population fit."})

    adapters = _resolve_adapters(args.get("engines"))
    if not any(a.available() for a in adapters):
        return _skip("Engine comparison skipped: no estimation engine available.", "no_engine",
                     {"status": "no_engine",
                      "message": "No estimation engine available (install R/nlmixr2 for external engines)."})

    matrix = run_matrix_subjects(subjects, candidates, adapters)
    sel = select_winner(matrix["results"])
    winner = sel["winner"]

    payload = {
        "status": "ok",
        "winner": winner.to_audit_dict() if winner else None,
        "prediction_ranking": [r.to_audit_dict() for r in sel["prediction_ranking"]],
        "within_engine_likelihood": sel["within_engine_likelihood"],
        "results": [r.to_audit_dict() for r in matrix["results"]],
        "selection_metric": sel["selection_metric"],
        "note": sel["note"],
        "n_engines": matrix["n_engines"],
        "n_available": matrix["n_available"],
        "n_candidates": matrix["n_candidates"],
    }
    wname = f"{winner.engine} / {winner.model_name}" if winner else "none"
    return ToolResult(
        summary=(f"Compared {matrix['n_candidates']} model(s) across "
                 f"{matrix['n_available']} engine(s); winner {wname} by prediction "
                 f"accuracy (OFV/AIC/BIC not compared across engines)."),
        action=f"run_engine_comparison({matrix['n_available']} engines, "
               f"{matrix['n_candidates']} candidates)",
        writes={"engine_comparison_results": payload},
        result={"status": "ok", "winner": payload["winner"],
                "prediction_ranking": payload["prediction_ranking"][:5],
                "within_engine_likelihood": sel["within_engine_likelihood"],
                "selection_metric": sel["selection_metric"], "note": sel["note"]})


TOOLS = [
    Tool(
        name="run_engine_comparison",
        description=(
            "Fit one or more candidate PK models across multiple estimation engines "
            "(pharmagent FOCE-I/SAEM and, if installed, nlmixr2) and select the best "
            "model+engine by ENGINE-AGNOSTIC prediction accuracy — prediction RMSE and "
            "VPC coverage computed on a common footing. Native OFV/AIC/BIC are reported "
            "per engine but are NEVER compared across engines (different algorithms' "
            "likelihoods are not comparable). Requires a loaded dataset; unavailable "
            "engines are skipped gracefully."),
        agent="modeler",
        input_schema={
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "description": "Candidate models to fit. Defaults to the last fitted model.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "model_key": {"type": "string"},
                            "iiv_params": {"type": "array", "items": {"type": "string"}},
                            "error_model": {"type": "string",
                                            "enum": ["proportional", "additive", "combined"]},
                            "method": {"type": "string", "enum": ["focei", "saem"]},
                        },
                        "required": ["model_key"],
                    },
                },
                "engines": {
                    "type": "array", "items": {"type": "string"},
                    "description": ("Engine names: pharmagent_focei, pharmagent_saem, "
                                    "nlmixr2. Defaults to [pharmagent_focei, nlmixr2]. "
                                    "Unavailable engines are skipped."),
                },
            },
        },
        run=run_engine_comparison,
    ),
]
