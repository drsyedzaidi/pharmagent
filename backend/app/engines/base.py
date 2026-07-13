"""Cross-engine orchestration — normalized result + adapter contract.

The scientific crux lives here in one rule: an estimation engine's likelihood
(`ofv`, and the `aic`/`bic` derived from it) is on that engine's *own*
algorithm's scale and is **within-engine-only** — never valid to compare across
engines. Cross-engine ranking uses the engine-agnostic prediction metrics
(`pred_rmse`, `vpc_coverage90`, ...) that ``scoring.score_predictions`` computes
by running every engine's estimates through the *same* simulator on the *same*
data. ``EngineResult`` keeps the two kinds of number in clearly separate fields.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

# Adapter-reported version stamp for the native engine (OQ-1: no version key
# exists in the fit result; we stamp it here).
NATIVE_VERSION = "pharmagent-nlme/0.1"


@dataclass(frozen=True)
class CandidateSpec:
    """A candidate model to fit — mirrors ``population_fit`` kwargs exactly."""
    model_key: str
    iiv_params: list[str] | None = None
    error_model: str = "proportional"
    covariate_model: list[dict] | None = None
    method: str = "focei"           # consumed only by the native pharmagent adapter

    @property
    def label(self) -> str:
        return self.model_key


@dataclass(frozen=True)
class EngineResult:
    """One engine's fit of one candidate, normalized across all engines."""
    engine: str                                   # "pharmagent_focei" | "mock" | "nlmixr2" ...
    engine_version: str = ""
    model_name: str = ""
    converged: bool = False
    runtime_s: float | None = None

    # --- WITHIN-ENGINE-ONLY likelihood metrics (never cross-engine ranked) ---
    ofv: float | None = None
    aic: float | None = None
    bic: float | None = None

    # --- parameter estimates ---
    params: dict[str, float] = field(default_factory=dict)
    rse_pct: dict[str, float | None] = field(default_factory=dict)
    omega_cv_pct: dict[str, float] = field(default_factory=dict)
    sigma: dict[str, float | None] = field(default_factory=dict)
    iiv_params: list[str] = field(default_factory=list)
    error_model: str = ""
    covariate_effects: list[dict] = field(default_factory=list)

    # --- ENGINE-AGNOSTIC goodness (the cross-engine ranking basis) ---
    pred_rmse: float | None = None                # log-scale RMSE of ind. predictions
    pred_bias: float | None = None                # mean(log ipred - log obs)
    pred_r2: float | None = None                  # log-scale R^2
    vpc_coverage90: float | None = None           # frac. pcVPC bins with obs median in sim-median 90% CI
    n_map_fallback: int = 0                        # subjects whose EBE fell back to typical (diagnostic)

    # --- diagnostics passthrough ---
    condition_number: float | None = None
    shrinkage_pct: dict[str, float] = field(default_factory=dict)
    n_subjects: int = 0
    n_obs: int = 0

    # --- graceful-degrade markers ---
    status: str = "ok"                            # "ok" | "failed" | "absent"
    message: str = ""
    raw: dict[str, Any] = field(default_factory=dict)  # verbatim native result (winner export)

    def to_audit_dict(self) -> dict[str, Any]:
        """JSON-safe, ``raw``-stripped view for the SHA-256 audit chain (OQ-6)."""
        d = asdict(self)
        d.pop("raw", None)
        return d


@runtime_checkable
class EngineAdapter(Protocol):
    name: str

    def available(self) -> bool:
        """False when the engine's binary / license / runtime is missing —
        the runner then records an ``absent`` row instead of calling ``fit``."""
        ...

    def fit(self, spec: CandidateSpec, subjects: list[dict], *,
            seed: int = 20250614) -> EngineResult: ...


# ── AIC / BIC derivation (no native key exists — verified in nlme.py) ───────
def _n_coefficients(effect: dict) -> int:
    """Coefficient count for one covariate effect.

    For a categorical effect the native result's ``levels`` field is ALREADY the
    non-reference set (K-1 levels for a K-level covariate — see nlme.py, which
    sets ``n_coef = len(levels)``), so it contributes ``len(levels)``
    coefficients directly. Continuous effects contribute one.
    """
    if effect.get("kind") == "categorical":
        levels = effect.get("levels") or []
        return max(len(levels), 1)
    return 1


def k_from_result(r: dict) -> int:
    """Estimated-parameter count k from a native NLME result dict."""
    n_cov = sum(_n_coefficients(e) for e in r.get("covariate_effects", []))
    sigma = r.get("sigma", {}) or {}
    return (len(r.get("theta", {})) + n_cov + len(r.get("omega_cv_pct", {}))
            + (sigma.get("prop") is not None) + (sigma.get("add") is not None))


def aic_bic(r: dict) -> tuple[float | None, float | None]:
    """Within-engine AIC/BIC from OFV and k. Returns (None, None) if no OFV."""
    ofv = r.get("ofv")
    if ofv is None or not math.isfinite(ofv):
        return None, None
    k = k_from_result(r)
    n = int(r.get("n_obs", 0))
    aic = ofv + 2 * k
    bic = ofv + k * math.log(n) if n > 0 else None
    return aic, bic
