"""Shared plumbing for the exposure-response compute modules.

Everything here is pure and dependency-light (numpy / scipy only). The modules
that build on it -- ``er_models`` (logistic), ``er_survival`` (Kaplan-Meier,
Cox), ``er_bootstrap`` and ``optimus`` -- share one refusal vocabulary so a
caller (and the audit trail) sees the same structured statuses everywhere:

    invalid_input | insufficient_events | zero_variance_exposure |
    zero_variance_covariate | separation | monotone_likelihood |
    non_convergence | singular_information

A refusal is data, never an exception and never a number: statistics computed
on data that cannot support them (a separated logistic fit, three events) are
worse than no answer in a regulated analysis.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

# Two-sided normal quantile for the default 95% interval.
Z95 = 1.959963984540054
SIG_DIGITS = 12                 # >> any stated validation tolerance (1e-5)
MIN_EVENTS_HARD = 5             # below this (either class) a fit is refused
EVENTS_PER_VARIABLE = 10        # guideline below which Wald intervals mislead
MAX_N_BOOT = 1000
DEFAULT_SEED = 20250614         # project-wide default (see compute/bootstrap.py)


class ErRefusal(Exception):
    """Raised inside the compute cores; converted to a structured dict at the edge."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def refusal(status: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"status": status, "message": message, **extra}


def sig(x: Any, digits: int = SIG_DIGITS) -> float | None:
    """Round to ``digits`` significant figures; non-finite -> None (JSON-safe)."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    if v == 0.0:
        return 0.0
    return float(f"{v:.{digits}g}")


def sig_list(xs: Sequence[Any]) -> list[float | None]:
    return [sig(v) for v in xs]


def sig_matrix(m: np.ndarray) -> list[list[float | None]]:
    return [[sig(v) for v in row] for row in np.asarray(m, dtype=float)]


def to_float_array(values: Any, name: str) -> np.ndarray:
    """1-D float array; ``None`` becomes NaN; booleans become 0/1."""
    try:
        arr = np.array([float("nan") if v is None else float(v) for v in values], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ErRefusal("invalid_input", f"{name} must be a list of numbers ({exc})") from exc
    if arr.ndim != 1:
        raise ErRefusal("invalid_input", f"{name} must be one-dimensional")
    return arr


def z_crit(ci_level: float) -> float:
    from scipy.stats import norm
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1); got {ci_level}")
    return float(norm.ppf(1.0 - (1.0 - ci_level) / 2.0))


def wald_row(est: float, se: float, z: float) -> dict[str, float | None]:
    """Wald statistics for one coefficient (z, two-sided p, CI on the coefficient scale)."""
    from scipy.stats import norm
    stat = est / se if se > 0 else float("nan")
    return {"estimate": sig(est), "se": sig(se), "z": sig(stat),
            "p": sig(2.0 * norm.sf(abs(stat))) if math.isfinite(stat) else None,
            "lo": sig(est - z * se), "hi": sig(est + z * se)}


def sd_of(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1)) if x.size > 1 else 0.0


def is_constant(x: np.ndarray) -> bool:
    """Zero variance with a relative tolerance (floating-point-identical values)."""
    if x.size == 0:
        return True
    return bool(float(np.ptp(x)) <= 1e-12 * float(np.max(np.abs(x))))
