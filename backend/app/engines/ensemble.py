"""Ensemble (consensus) engine — combine several engines' fits into one.

Motivated by Käser et al. (Mol. Pharmaceutics 2026), who found that ensembles of
two or three prediction methods generally beat any single method for first-in-
human PK. Here the consensus is formed at the *parameter* level: the geometric
mean of each contributing engine's population parameters (PK parameters are
positive and roughly lognormal, so the geometric mean is the natural centre),
then scored through the same ``score_from_population`` as every other engine, so
the consensus competes for the winner on identical footing.

This does NOT assert that the ensemble is more accurate — that is data-dependent.
It provides the mechanism and lets the engine-agnostic ranking decide.
"""
from __future__ import annotations

import math

from .base import EngineResult
from .scoring import score_from_population


def _geomean(vals: list[float]) -> float | None:
    pos = [v for v in vals if v is not None and v > 0]
    if not pos:
        return None
    return math.exp(sum(math.log(v) for v in pos) / len(pos))


def _mean(vals: list[float]) -> float | None:
    xs = [v for v in vals if v is not None]
    return sum(xs) / len(xs) if xs else None


def _mean_coefficient(coefs: list):
    """Average a covariate-effect coefficient across engines. Continuous effects
    carry a scalar (an exponent or slope, possibly negative → arithmetic mean);
    a categorical effect carries a ``{level: coef}`` dict → averaged per level."""
    dicts = [c for c in coefs if isinstance(c, dict)]
    if dicts:
        keys = set().union(*(d.keys() for d in dicts))
        return {k: _mean([d[k] for d in dicts if k in d]) for k in keys}
    return _mean([c for c in coefs if isinstance(c, (int, float))])


def _aggregate_cov_effects(usable: list[EngineResult]) -> list[dict]:
    """Consensus covariate effects. Members share the candidate's covariate
    structure, so effects are aligned by (param, covariate) and their coefficients
    averaged; structural fields are taken from the first member."""
    members = [r for r in usable if r.covariate_effects]
    if not members:
        return []
    out: list[dict] = []
    for eff in members[0].covariate_effects:
        key = (eff.get("param"), eff.get("covariate"))
        coefs = [e.get("coefficient") for m in members for e in m.covariate_effects
                 if (e.get("param"), e.get("covariate")) == key]
        merged = {k: v for k, v in eff.items() if k not in ("rse_pct", "description")}
        merged["coefficient"] = _mean_coefficient([c for c in coefs if c is not None])
        out.append(merged)
    return out


def build_ensemble(results: list[EngineResult], model_key: str, subjects: list[dict],
                   *, name: str = "ensemble", min_engines: int = 2) -> EngineResult | None:
    """Consensus EngineResult from the converged, usable fits in ``results``.

    Returns None when fewer than ``min_engines`` engines produced a usable fit —
    a consensus of one is not a consensus.
    """
    usable = [r for r in results
              if r.status == "ok" and r.converged and r.params and r.engine != name]
    if len(usable) < min_engines:
        return None

    # Geometric-mean each structural parameter across the engines that report it.
    param_keys = set().union(*(r.params.keys() for r in usable))
    params: dict[str, float] = {}
    for k in param_keys:
        gm = _geomean([r.params[k] for r in usable if k in r.params])
        if gm is not None:
            params[k] = gm

    omega_keys = set().union(*(r.omega_cv_pct.keys() for r in usable))
    omega_cv = {k: v for k in omega_keys
                if (v := _mean([r.omega_cv_pct[k] for r in usable if k in r.omega_cv_pct])) is not None}
    sigma_prop = _mean([(r.sigma or {}).get("prop") for r in usable])
    sigma_add = _mean([(r.sigma or {}).get("add") for r in usable])
    # Use the most common iiv/error spec among contributors (they share the candidate).
    iiv = usable[0].iiv_params or list(omega_cv)
    error_model = usable[0].error_model or "proportional"
    cov_effects = _aggregate_cov_effects(usable)  # covariate-model candidates

    sc = score_from_population(
        model_key, subjects, theta=params, omega_cv_pct=omega_cv,
        sigma_prop=sigma_prop, sigma_add=sigma_add, iiv_params=iiv,
        error_model=error_model, covariate_effects=cov_effects or None,
    )
    return EngineResult(
        engine=name, engine_version="ensemble/geomean",
        model_name=model_key, converged=True, runtime_s=0.0,
        params=params, omega_cv_pct=omega_cv,
        sigma={"prop": sigma_prop, "add": sigma_add}, iiv_params=iiv,
        error_model=error_model, covariate_effects=cov_effects, n_subjects=len(subjects),
        raw={"members": [r.engine for r in usable], "agg": "geomean"}, **sc,
    )
