"""Mock engine adapter — keyless, deterministic stand-in for an external engine.

It does NOT re-implement estimation. It runs the real native fit once
(deterministic given the seed), then multiplies every parameter by ``exp(bias)``
to emulate an engine that lands at slightly different estimates. ``bias=0`` is
the *oracle* (params == native truth → best prediction scores); ``bias>0`` is a
plausibly-worse engine. This lets the whole runner/selection stack be tested with
no Monolix, no license, and no R.
"""
from __future__ import annotations

import math

from .base import CandidateSpec, EngineResult
from .scoring import score_from_population


class MockEngineAdapter:
    def __init__(self, name: str = "mock", bias: float = 0.0) -> None:
        self.name = name
        self._bias = bias

    def available(self) -> bool:
        return True

    def fit(self, spec: CandidateSpec, subjects: list[dict], *,
            seed: int = 20250614) -> EngineResult:
        from app.compute.nlme import population_fit  # lazy: heavy

        r = population_fit(
            spec.model_key, subjects, method="focei",
            iiv_params=spec.iiv_params, error_model=spec.error_model,
            covariate_model=spec.covariate_model, seed=seed,
        )
        f = math.exp(self._bias)
        theta = {k: v * f for k, v in r["theta"].items()}
        sigma = r.get("sigma", {}) or {}
        sc = score_from_population(
            spec.model_key, subjects, theta=theta,
            omega_cv_pct=r.get("omega_cv_pct", {}),
            sigma_prop=sigma.get("prop"), sigma_add=sigma.get("add"),
            iiv_params=r.get("iiv_params", []), error_model=r.get("error_model", "proportional"),
        )
        # OFV is deliberately left on this engine's own (incomparable) scale.
        return EngineResult(
            engine=self.name, engine_version="mock/0", model_name=spec.model_key,
            converged=True, runtime_s=0.0,
            ofv=(r.get("ofv") + self._bias * 100.0) if r.get("ofv") is not None else None,
            params=theta, omega_cv_pct=r.get("omega_cv_pct", {}), sigma=sigma,
            iiv_params=r.get("iiv_params", []), error_model=r.get("error_model", ""),
            n_subjects=int(r.get("n_subjects", 0)), n_obs=int(r.get("n_obs", 0)),
            raw=r, **sc,
        )
