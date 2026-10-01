"""Bucher adjusted indirect comparison (closed-form, deterministic).

Given two direct comparisons that share a common comparator B -- A vs B and
C vs B -- the adjusted indirect comparison of A vs C is (Bucher et al. 1997,
J Clin Epidemiol 50:683-91):

    d_AC = d_AB - d_CB            var(d_AC) = se_AB^2 + se_CB^2

computed on the *analysis scale*: the natural log for ratio measures (HR, OR,
RR; exponentiated back for reporting) and the identity for difference measures
(MD, RD, SMD). Each input is an effect estimate with a standard error on the
analysis scale OR a confidence interval, which is converted with

    se = (g(U) - g(L)) / (2 z),   g = ln (ratio) or identity (difference)

Validity rests on transitivity / similarity of the two trial sets (comparable
populations, doses, outcome definitions and effect modifiers); no data can
check that, so it is returned as an assumption alongside every result.
"""
from __future__ import annotations

import math
from typing import Any

from scipy.stats import norm

from app.compute.textguard import reject_numerals

RATIO_SCALES = ("HR", "OR", "RR")
DIFFERENCE_SCALES = ("MD", "RD", "SMD")
DEFAULT_CI_LEVEL = 0.95
METHOD = "bucher_adjusted_indirect_comparison"
ASSUMPTION = (
    "Valid only under transitivity: the two sets of trials must be similar in "
    "population, dose, outcome definition and effect modifiers, and the "
    "comparator B must be the same. Heterogeneity within each direct comparison "
    "is not modelled here.")
_MAX_LABEL = 80


def normalize_scale(scale: Any) -> tuple[str, str]:
    """('HR', 'ratio') from a case-insensitive scale name; ValueError otherwise."""
    key = scale.strip().upper() if isinstance(scale, str) else ""
    if key in RATIO_SCALES:
        return key, "ratio"
    if key in DIFFERENCE_SCALES:
        return key, "difference"
    raise ValueError(
        f"unknown scale {scale!r}: use one of {', '.join(RATIO_SCALES + DIFFERENCE_SCALES)}")


def _level(value: Any, what: str = "ci_level") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{what} must be a number strictly between 0 and 1, got {value!r}")
    return float(value)


def _z(ci_level: float) -> float:
    z = float(norm.ppf(0.5 + ci_level / 2.0))
    if not (math.isfinite(z) and z > 0.0):
        raise ValueError(f"ci_level {ci_level!r} is too close to 0 or 1 to give a usable interval")
    return z


def _finite(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{what} must be finite, got {value!r}")
    return float(value)


def _to_analysis(value: float, kind: str, what: str) -> float:
    if kind == "ratio":
        if value <= 0.0:
            raise ValueError(f"{what} must be positive on a ratio scale, got {value!r}")
        return math.log(value)
    return value


def _from_analysis(value: float, kind: str) -> float:
    if kind != "ratio":
        return value
    try:
        return math.exp(value)
    except OverflowError:
        raise ValueError("result is out of numeric range on the ratio scale: "
                         "the inputs are too extreme for a meaningful comparison") from None


def se_from_ci(lower: float, upper: float, *, ci_level: float = DEFAULT_CI_LEVEL,
               scale: str = "HR") -> float:
    """SE on the analysis scale from a CI: (g(U) - g(L)) / (2 z)."""
    _, kind = normalize_scale(scale)
    lo, hi = _finite(lower, "ci_lower"), _finite(upper, "ci_upper")
    if not lo < hi:
        raise ValueError(f"ci_lower must be below ci_upper, got {lo!r} and {hi!r}")
    glo, ghi = _to_analysis(lo, kind, "ci_lower"), _to_analysis(hi, kind, "ci_upper")
    return (ghi - glo) / (2.0 * _z(_level(ci_level)))


def ci_from_se(estimate: float, se: float, *, ci_level: float = DEFAULT_CI_LEVEL,
               scale: str = "HR") -> tuple[float, float]:
    """CI (natural scale) from an estimate and an analysis-scale SE."""
    _, kind = normalize_scale(scale)
    est = _to_analysis(_finite(estimate, "estimate"), kind, "estimate")
    s = _finite(se, "se")
    if s <= 0.0:
        raise ValueError(f"se must be > 0, got {s!r}")
    half = _z(_level(ci_level)) * s
    return _from_analysis(est - half, kind), _from_analysis(est + half, kind)


def _direct(name: str, spec: Any, kind: str, scale: str) -> dict[str, Any]:
    """Validate one direct comparison and reduce it to (estimate, SE) on the analysis scale."""
    if not isinstance(spec, dict):
        raise ValueError(f"{name} must be an object with estimate and se or a CI")
    if "estimate" not in spec:
        raise ValueError(f"{name}: estimate is required")
    est = _finite(spec["estimate"], f"{name}.estimate")
    g_est = _to_analysis(est, kind, f"{name}.estimate")
    has_se = spec.get("se") is not None
    lo_in, hi_in = spec.get("ci_lower"), spec.get("ci_upper")
    has_ci = lo_in is not None or hi_in is not None
    if has_se and has_ci:
        raise ValueError(f"{name}: give se or a CI, not both")
    if not has_se and not has_ci:
        raise ValueError(f"{name}: give se or a CI (ci_lower and ci_upper)")
    out: dict[str, Any] = {
        "label": str(spec.get("label", "") or "")[:_MAX_LABEL],
        "estimate": est, "analysis_estimate": g_est}
    if has_se:
        se = _finite(spec["se"], f"{name}.se")
        if se <= 0.0:
            raise ValueError(f"{name}: se must be > 0, got {se!r}")
        out.update(se=se, se_source="supplied")
        return out
    if lo_in is None or hi_in is None:
        raise ValueError(f"{name}: give both ci_lower and ci_upper")
    lo, hi = _finite(lo_in, f"{name}.ci_lower"), _finite(hi_in, f"{name}.ci_upper")
    if not lo < hi:
        raise ValueError(f"{name}: ci_lower must be below ci_upper, got {lo!r} and {hi!r}")
    if not lo <= est <= hi:
        raise ValueError(f"{name}: estimate must lie inside its CI ({lo!r} to {hi!r})")
    supplied = spec.get("ci_level") is not None
    level = _level(spec["ci_level"] if supplied else DEFAULT_CI_LEVEL, f"{name}.ci_level")
    se = se_from_ci(lo, hi, ci_level=level, scale=scale)
    if not (math.isfinite(se) and se > 0.0):
        raise ValueError(f"{name}: se must be > 0 and finite; the CI is too narrow or too wide to give one")
    out.update(se=se, se_source="from_ci", ci_lower=lo, ci_upper=hi, ci_level=level,
               ci_level_source="supplied" if supplied else "default")
    return out


def bucher_indirect(ab: Any, cb: Any, *, scale: str, ci_level: float = DEFAULT_CI_LEVEL,
                    treatments: tuple[str, str, str] = ("A", "B", "C")) -> dict[str, Any]:
    """Indirect A-vs-C effect through comparator B from A-vs-B and C-vs-B.

    ``ab`` / ``cb``: {"estimate": x, "se": s} or {"estimate": x, "ci_lower": l,
    "ci_upper": u[, "ci_level": 0.95]}, optional "label". Raises ValueError with
    a specific message on any invalid input. Pure: inputs are not mutated.
    """
    scale_name, kind = normalize_scale(scale)
    level = _level(ci_level)
    a, b, c = (reject_numerals(str(t).strip()[:_MAX_LABEL], "treatment label") or d
               for t, d in zip(treatments, "ABC"))
    d_ab, d_cb = _direct("ab", ab, kind, scale_name), _direct("cb", cb, kind, scale_name)

    g = d_ab["analysis_estimate"] - d_cb["analysis_estimate"]
    se = math.hypot(d_ab["se"], d_cb["se"])         # variances add; hypot cannot underflow/overflow early
    half = _z(level) * se
    z_stat = g / se
    result = {
        "status": "ok",
        "method": METHOD,
        "scale": scale_name,
        "scale_type": kind,
        "analysis_scale": "log" if kind == "ratio" else "identity",
        "contrast": f"{a} vs {c} via {b}",
        "treatments": {"A": a, "B": b, "C": c},
        "estimate": _from_analysis(g, kind),
        "analysis_estimate": g,
        "se": se,
        "ci_level": level,
        "ci_lower": _from_analysis(g - half, kind),
        "ci_upper": _from_analysis(g + half, kind),
        "z": z_stat,
        "p_value": float(2.0 * norm.sf(abs(z_stat))),
        "inputs": {"ab": d_ab, "cb": d_cb},
        "assumptions": ASSUMPTION,
    }
    numbers = [result[k] for k in ("estimate", "analysis_estimate", "se", "ci_lower", "ci_upper", "z",
                                   "p_value")]
    if not all(math.isfinite(v) for v in numbers):
        raise ValueError("result is not finite: the inputs are outside a numerically meaningful range")
    return result
