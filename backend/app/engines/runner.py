"""Cross-engine runner — fan candidates × engines, tolerate absent/failed engines.

Errors are data, not exceptions (the app's convention): a missing engine becomes
an ``absent`` row and a crashing fit becomes a ``failed`` row, so one bad engine
never kills the matrix. The core works on a ``subjects`` list (no pandas); a thin
``run_matrix`` wrapper builds that list from a dataframe via the app's canonical
converter so no second dataset parser is introduced.
"""
from __future__ import annotations

from typing import Any

from .base import CandidateSpec, EngineAdapter, EngineResult


def _absent(engine_name: str) -> EngineResult:
    return EngineResult(engine=engine_name, status="absent",
                        message="engine unavailable (binary/license/runtime missing)")


def _failed(engine_name: str, model_key: str, msg: str) -> EngineResult:
    return EngineResult(engine=engine_name, model_name=model_key,
                        status="failed", message=msg)


def run_matrix_subjects(subjects: list[dict], candidates: list[CandidateSpec],
                        adapters: list[EngineAdapter], *,
                        seed: int = 20250614) -> dict[str, Any]:
    """Fit every candidate on every available engine over a prepared subjects list."""
    results: list[EngineResult] = []
    for eng in adapters:
        if not eng.available():
            results.append(_absent(eng.name))
            continue
        for spec in candidates:
            try:
                results.append(eng.fit(spec, subjects, seed=seed))
            except Exception as exc:  # never let one fit kill the matrix
                results.append(_failed(eng.name, spec.model_key, str(exc)))
    return {
        "results": results,
        "n_engines": len(adapters),
        "n_available": sum(1 for a in adapters if a.available()),
        "n_candidates": len(candidates),
    }


def run_matrix(df, roles: dict[str, str], candidates: list[CandidateSpec],
               adapters: list[EngineAdapter], *, seed: int = 20250614) -> dict[str, Any]:
    """Dataframe entry point: build subjects with the app's canonical converter,
    then delegate to :func:`run_matrix_subjects`."""
    from app.tools.pkmodel_tools import _build_subjects  # lazy: pandas/app layer

    subjects, _multi, _has_pd = _build_subjects(df, roles, with_blq=True)
    return run_matrix_subjects(subjects, candidates, adapters, seed=seed)
