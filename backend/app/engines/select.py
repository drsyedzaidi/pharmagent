"""Cross-engine selection — the scientific crux.

The winner is chosen ONLY on engine-agnostic prediction metrics. The ranking key
never reads ``ofv``/``aic``/``bic``, and the likelihood summary is bucketed *by
engine* so a cross-engine likelihood comparison is not even representable. That
invariant is what the test-suite locks down.

Ranking (ascending = better):
  1. pred_rmse            (log-scale individual-prediction error)  ↑ lower better
  2. vpc_coverage90       tie-break, higher better
  3. pred_r2              tie-break, higher better
  4. |pred_bias|          tie-break, lower better
Non-``ok`` / non-converged results are excluded; missing metrics sort last.
"""
from __future__ import annotations

from typing import Any

from .base import EngineResult

SELECTION_METRIC = (
    "pred_rmse asc; ties -> vpc_coverage90 desc, pred_r2 desc, |pred_bias| asc"
)
_NOTE = ("OFV/AIC/BIC are within-engine only and are excluded from cross-engine "
         "ranking (different algorithms' likelihoods are not comparable).")


def _rank_key(r: EngineResult) -> tuple:
    return (
        r.pred_rmse is None,
        r.pred_rmse if r.pred_rmse is not None else float("inf"),
        -(r.vpc_coverage90 or 0.0),
        -(r.pred_r2 or 0.0),
        abs(r.pred_bias) if r.pred_bias is not None else float("inf"),
    )


def select_winner(results: list[EngineResult]) -> dict[str, Any]:
    """Rank converged results by prediction accuracy and pick a cross-engine winner.

    The returned dict also carries a SEPARATE, per-engine likelihood table that is
    never consulted for the winner.
    """
    scored = [r for r in results if r.status == "ok" and r.converged]
    ranking = sorted(scored, key=_rank_key)
    winner = ranking[0] if ranking else None

    # Within-engine-only likelihood table (bucketed by engine; AIC-ascending).
    within: dict[str, list[dict[str, Any]]] = {}
    for r in scored:
        within.setdefault(r.engine, []).append(
            {"model": r.model_name, "ofv": r.ofv, "aic": r.aic, "bic": r.bic})
    for eng in within:
        within[eng].sort(key=lambda x: (x["aic"] is None, x["aic"] if x["aic"] is not None else 0.0))

    return {
        "winner": winner,
        "prediction_ranking": ranking,
        "within_engine_likelihood": within,
        "selection_metric": SELECTION_METRIC,
        "note": _NOTE,
        "n_scored": len(scored),
        "n_skipped": len(results) - len(scored),
    }
