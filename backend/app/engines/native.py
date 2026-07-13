"""Native PharmAgent adapter — wraps the real FOCE-I / SAEM fitter.

FOCE-I and SAEM are surfaced as two *distinct* engines (``pharmagent_focei`` /
``pharmagent_saem``): exactly the case where comparing native OFV is invalid, so
they are ranked by prediction scoring like any other engine pair.
"""
from __future__ import annotations

import time

from .base import NATIVE_VERSION, CandidateSpec, EngineResult, aic_bic
from .scoring import score_from_population


class PharmAgentAdapter:
    """Native in-process engine. ``method`` (focei|saem), when given, overrides
    the candidate's method so FOCE-I and SAEM can be registered as two distinct
    engines — precisely the pair whose native OFV must NOT be compared."""

    def __init__(self, method: str | None = None) -> None:
        self._method = method
        self.name = f"pharmagent_{method.lower()}" if method else "pharmagent"

    def available(self) -> bool:
        return True  # in-process; no external binary/license

    def fit(self, spec: CandidateSpec, subjects: list[dict], *,
            seed: int = 20250614) -> EngineResult:
        from app.compute.nlme import population_fit  # lazy: heavy

        method = (self._method or spec.method).lower()
        engine = f"pharmagent_{method}"
        t0 = time.perf_counter()
        try:
            r = population_fit(
                spec.model_key, subjects, method=method,
                iiv_params=spec.iiv_params, error_model=spec.error_model,
                covariate_model=spec.covariate_model, seed=seed,
            )
        except (KeyError, ValueError) as exc:
            return EngineResult(
                engine=engine, engine_version=NATIVE_VERSION,
                model_name=spec.model_key, status="failed", message=str(exc),
            )
        dt = time.perf_counter() - t0

        sigma = r.get("sigma", {}) or {}
        sc = score_from_population(
            spec.model_key, subjects, theta=r["theta"],
            omega_cv_pct=r.get("omega_cv_pct", {}),
            sigma_prop=sigma.get("prop"), sigma_add=sigma.get("add"),
            iiv_params=r.get("iiv_params", []), error_model=r.get("error_model", "proportional"),
            covariate_effects=r.get("covariate_effects"),
        )
        aic, bic = aic_bic(r)
        return EngineResult(
            engine=engine, engine_version=NATIVE_VERSION,
            model_name=spec.model_key, converged=bool(r.get("converged")),
            runtime_s=dt, ofv=r.get("ofv"), aic=aic, bic=bic,
            params=r.get("theta", {}), rse_pct=r.get("theta_rse_pct", {}),
            omega_cv_pct=r.get("omega_cv_pct", {}), sigma=sigma,
            iiv_params=r.get("iiv_params", []), error_model=r.get("error_model", ""),
            covariate_effects=r.get("covariate_effects", []),
            condition_number=r.get("condition_number"),
            shrinkage_pct=r.get("shrinkage_pct", {}),
            n_subjects=int(r.get("n_subjects", 0)), n_obs=int(r.get("n_obs", 0)),
            raw=r, **sc,
        )
