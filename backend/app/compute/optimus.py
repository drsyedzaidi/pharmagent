"""Optimus-style dose selection from efficacy and toxicity exposure-response models.

FDA's Project Optimus asks sponsors to choose a dose by weighing benefit against
harm across several candidate doses, not to default to the maximum tolerated
dose. This module is the arithmetic of that comparison and nothing more:

    U(d) = P_eff(d) - w * P_tox(d)

* ``P_eff`` / ``P_tox`` come from two fitted LOGISTIC E-R models (see
  ``er_models``) evaluated at the exposure each dose produces;
* ``w`` is the clinical-utility weight. It is a CLINICAL JUDGEMENT, so it is
  never defaulted silently: the caller must declare it (``utility_weight``) or
  explicitly acknowledge the default with ``utility="default_w=1"``; either way
  it is echoed in the output with its provenance;
* an optional toxicity cap ``P_tox(d) <= cap`` excludes doses before the argmax;
* the **selected dose** is the feasible dose with the highest utility at the
  point estimates (ties go to the LOWER dose -- the conservative choice);
* **uncertainty** is propagated by re-evaluating every dose under parameter
  draws -- the E-R bootstrap replicates when the fit carries them, otherwise
  seeded draws from the asymptotic multivariate normal (the same idea as
  ``clinsim.sample_theta_draws``). The result reports, per dose, the probability
  it is the optimum, the probability it is feasible, and percentile bands on
  P_eff, P_tox and utility. The RNG seed is an input and is echoed.

Stated limits (also returned in ``notes``): exposure at each dose is the TYPICAL
exposure, so P(d) is the response probability at that exposure rather than the
population-average rate over between-subject exposure variability; efficacy and
toxicity parameter uncertainty is propagated independently; and the table is only
as valid as the E-R models are inside the exposure range they were fitted on
(``exposure_extrapolated`` / ``dose_extrapolated`` flag the rest).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.special import expit

from app.compute.er_common import DEFAULT_SEED, ErRefusal, refusal, sig

_MAX_DOSES = 100
_MIN_DRAWS, _MAX_DRAWS = 100, 20000
FORMULA = "U(d) = P_eff(d) - w * P_tox(d)"
DEFAULT_ACK = "default_w=1"
_NOTES = [
    "Exposure at each dose is the TYPICAL exposure: P(d) is the response probability at that exposure, "
    "not the population-average rate over between-subject exposure variability.",
    "Efficacy and toxicity parameter uncertainty are propagated independently.",
    "The utility weight w is a clinical judgement declared by the user; the selection is only as good as "
    "that declaration. The pharmacometrician of record decides.",
]


def resolve_utility(utility_weight: Any, utility: Any) -> dict[str, Any]:
    """The declared utility, or ``ErRefusal``. Never invents a weight."""
    if utility_weight is not None:
        try:
            w = float(utility_weight)
        except (TypeError, ValueError) as exc:
            raise ErRefusal("invalid_input", "utility_weight must be a number") from exc
        if not math.isfinite(w) or w < 0:
            raise ErRefusal("invalid_input", "utility_weight must be a finite number >= 0")
        return {"formula": FORMULA, "w": w, "w_source": "declared", "declared": f"w={w:g}"}
    if utility == DEFAULT_ACK:
        return {"formula": FORMULA, "w": 1.0, "w_source": "default_acknowledged", "declared": DEFAULT_ACK}
    if utility not in (None, ""):
        raise ErRefusal("invalid_input", f"utility must be exactly '{DEFAULT_ACK}' (or pass utility_weight); "
                                         f"got {utility!r}")
    raise ErRefusal("utility_not_declared",
                    "The clinical-utility weight w in U(d) = P_eff(d) - w * P_tox(d) is a clinical judgement "
                    "and is not defaulted silently. Declare it with utility_weight (>= 0), or acknowledge the "
                    f"default explicitly with utility='{DEFAULT_ACK}'.")


def _grid(doses: Any) -> np.ndarray:
    try:
        d = np.array([float(v) for v in doses], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ErRefusal("invalid_input", "doses must be a list of numbers") from exc
    if d.size < 2:
        raise ErRefusal("invalid_input", "need at least two candidate doses to select between")
    if d.size > _MAX_DOSES:
        raise ErRefusal("invalid_input", f"at most {_MAX_DOSES} candidate doses")
    if not np.all(np.isfinite(d)) or np.any(d <= 0):
        raise ErRefusal("invalid_input", "doses must be finite and > 0")
    if np.unique(d).size != d.size:
        raise ErRefusal("invalid_input", "duplicate doses in the grid")
    return np.sort(d)


def _need(m: dict[str, Any], key: str, positive: bool = True) -> float:
    try:
        v = float(m[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ErRefusal("invalid_input", f"exposure_mapping needs a numeric '{key}'") from exc
    if not math.isfinite(v) or (positive and v <= 0):
        raise ErRefusal("invalid_input", f"exposure_mapping '{key}' must be finite" + (" and > 0" if positive else ""))
    return v


def exposures_for(doses: np.ndarray, mapping: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any], tuple | None]:
    """Typical exposure at each dose, the mapping echo, and the dose range the
    mapping is supported on (``None`` when unconstrained)."""
    kind = (mapping or {}).get("type")
    if kind in ("linear", "power"):
        d0, e0 = _need(mapping, "reference_dose"), _need(mapping, "reference_exposure")
        b = 1.0 if kind == "linear" else _need(mapping, "exponent", positive=False)
        return e0 * (doses / d0) ** b, {"type": kind, "reference_dose": d0, "reference_exposure": e0,
                                        "exponent": b}, None
    if kind == "dose_proportionality":
        a, b = _need(mapping, "intercept", positive=False), _need(mapping, "slope", positive=False)
        levels = [float(v) for v in mapping.get("dose_levels") or []]
        echo = {"type": kind, "intercept": a, "slope": b, "parameter": mapping.get("parameter"),
                "dose_levels": levels}
        return np.exp(a + b * np.log(doses)), echo, ((min(levels), max(levels)) if levels else None)
    if kind == "by_dose":
        table = {float(k): float(v) for k, v in (mapping.get("values") or {}).items()}
        missing = [f"{d:g}" for d in doses if not any(math.isclose(d, k) for k in table)]
        if missing:
            raise ErRefusal("invalid_input",
                            f"by_dose exposure_mapping has no exposure for dose(s) {', '.join(missing)}")
        vals = np.array([next(v for k, v in table.items() if math.isclose(d, k)) for d in doses])
        if np.any(vals <= 0) or not np.all(np.isfinite(vals)):
            raise ErRefusal("invalid_input", "by_dose exposures must be finite and > 0")
        return vals, {"type": kind, "values": {f"{d:g}": float(v) for d, v in zip(doses, vals)}}, None
    raise ErRefusal("invalid_input", "exposure_mapping.type must be one of linear, power, "
                                     "dose_proportionality, by_dose")


def _fit_params(fit: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(fit, dict) or fit.get("status") != "ok" or fit.get("model") != "logistic":
        raise ErRefusal("invalid_input", f"{name} must be a successful logistic exposure-response fit "
                                         "(binary endpoints only; a Cox hazard ratio has no dose-level probability)")
    beta = np.array([fit["coef"]["intercept"]["estimate"], fit["coef"]["slope"]["estimate"]], dtype=float)
    return beta, np.array(fit["cov"], dtype=float)


def _draws(fit: dict[str, Any], beta: np.ndarray, cov: np.ndarray, n: int | None,
           rng: np.random.Generator) -> tuple[np.ndarray, str]:
    boot = fit.get("bootstrap") or {}
    if boot.get("status") == "ok" and boot.get("draws"):
        d = np.array(boot["draws"], dtype=float)
        return (d[:n] if n else d), "bootstrap"
    return rng.multivariate_normal(beta, cov, size=n or _MIN_DRAWS, method="eigh"), "asymptotic_mvn"


def _band(arr: np.ndarray, ci_level: float) -> tuple[np.ndarray, np.ndarray]:
    a = (1.0 - ci_level) / 2.0
    return np.quantile(arr, a, axis=0), np.quantile(arr, 1.0 - a, axis=0)


def _mappings(mapping: dict[str, Any], eff: dict[str, Any], tox: dict[str, Any]) -> tuple[dict, dict]:
    if "efficacy" in mapping and "toxicity" in mapping:
        return mapping["efficacy"], mapping["toxicity"]
    if eff.get("exposure_label") != tox.get("exposure_label"):
        raise ErRefusal("exposure_metric_mismatch",
                        f"The efficacy fit uses exposure '{eff.get('exposure_label')}' but the toxicity fit uses "
                        f"'{tox.get('exposure_label')}'; one exposure mapping cannot serve both. Pass "
                        "exposure_mapping={'efficacy': {...}, 'toxicity': {...}}.")
    return mapping, mapping


def select_dose(*, doses: Any, exposure_mapping: dict[str, Any], efficacy_fit: dict[str, Any],
                toxicity_fit: dict[str, Any], utility_weight: Any = None, utility: Any = None,
                toxicity_cap: float | None = None, n_draws: int = 2000, seed: int = DEFAULT_SEED,
                ci_level: float = 0.95) -> dict[str, Any]:
    """Select a dose by utility; always returns a JSON-safe dict with ``status``."""
    try:
        return _select(doses, exposure_mapping, efficacy_fit, toxicity_fit, utility_weight, utility,
                       toxicity_cap, n_draws, seed, ci_level)
    except ErRefusal as exc:
        return refusal(exc.status, exc.message)


def _select(doses, exposure_mapping, efficacy_fit, toxicity_fit, utility_weight, utility, toxicity_cap,
            n_draws, seed, ci_level) -> dict[str, Any]:
    util = resolve_utility(utility_weight, utility)
    w = util["w"]
    grid = _grid(doses)
    if toxicity_cap is not None and not (isinstance(toxicity_cap, (int, float)) and 0.0 < float(toxicity_cap) <= 1.0):
        raise ErRefusal("invalid_input", "toxicity_cap must be a probability in (0, 1]")
    if not _MIN_DRAWS <= int(n_draws) <= _MAX_DRAWS:
        raise ErRefusal("invalid_input", f"n_draws must be between {_MIN_DRAWS} and {_MAX_DRAWS}")
    if not 0.0 < ci_level < 1.0:
        raise ErRefusal("invalid_input", "ci_level must be in (0, 1)")
    be, ce = _fit_params(efficacy_fit, "efficacy_fit")
    bt, ct = _fit_params(toxicity_fit, "toxicity_fit")
    map_e, map_t = _mappings(exposure_mapping or {}, efficacy_fit, toxicity_fit)
    exp_e, echo_e, rng_e = exposures_for(grid, map_e)
    exp_t, echo_t, rng_t = exposures_for(grid, map_t)

    p_eff = expit(be[0] + be[1] * exp_e)
    p_tox = expit(bt[0] + bt[1] * exp_t)
    utility_pt = p_eff - w * p_tox
    feasible = np.ones(grid.size, dtype=bool) if toxicity_cap is None else p_tox <= float(toxicity_cap)

    rng = np.random.default_rng(seed)
    d_e, src_e = _draws(efficacy_fit, be, ce, int(n_draws), rng)
    d_t, src_t = _draws(toxicity_fit, bt, ct, int(n_draws), rng)
    k = min(len(d_e), len(d_t))
    d_e, d_t = d_e[:k], d_t[:k]
    pe_d = expit(d_e[:, [0]] + d_e[:, [1]] * exp_e[None, :])
    pt_d = expit(d_t[:, [0]] + d_t[:, [1]] * exp_t[None, :])
    u_d = pe_d - w * pt_d
    f_d = np.ones_like(u_d, dtype=bool) if toxicity_cap is None else pt_d <= float(toxicity_cap)
    masked = np.where(f_d, u_d, -np.inf)
    best = np.argmax(masked, axis=1)
    any_ok = f_d.any(axis=1)
    prob_opt = np.array([np.mean((best == j) & any_ok) for j in range(grid.size)])
    prob_none = float(np.mean(~any_ok))
    (pe_lo, pe_hi), (pt_lo, pt_hi), (u_lo, u_hi) = (_band(a, ci_level) for a in (pe_d, pt_d, u_d))

    warnings: list[str] = []
    dose_extra = np.zeros(grid.size, dtype=bool)
    for rng_d in (rng_e, rng_t):
        if rng_d is not None:
            dose_extra |= (grid < rng_d[0]) | (grid > rng_d[1])
    exp_extra = np.zeros(grid.size, dtype=bool)
    for ex, fit in ((exp_e, efficacy_fit), (exp_t, toxicity_fit)):
        lo, hi = fit["exposure_range"]
        exp_extra |= (ex < lo) | (ex > hi)
    if dose_extra.any():
        warnings.append("Dose(s) " + ", ".join(f"{d:g}" for d in grid[dose_extra]) + " lie outside the dose "
                        "levels studied for the exposure-per-dose mapping: the exposure is an extrapolation.")
    if exp_extra.any():
        warnings.append("Dose(s) " + ", ".join(f"{d:g}" for d in grid[exp_extra]) + " map to exposures outside "
                        "the observed exposure range of an E-R model: its probability there is an extrapolation.")

    rows = [{
        "dose": sig(grid[j]), "exposure": sig(exp_e[j]), "exposure_tox": sig(exp_t[j]),
        "p_eff": sig(p_eff[j]), "p_eff_lo": sig(pe_lo[j]), "p_eff_hi": sig(pe_hi[j]),
        "p_tox": sig(p_tox[j]), "p_tox_lo": sig(pt_lo[j]), "p_tox_hi": sig(pt_hi[j]),
        "utility": sig(utility_pt[j]), "utility_lo": sig(u_lo[j]), "utility_hi": sig(u_hi[j]),
        "feasible": bool(feasible[j]), "prob_feasible": sig(float(f_d[:, j].mean())),
        "prob_optimal": sig(prob_opt[j]),
        "dose_extrapolated": bool(dose_extra[j]), "exposure_extrapolated": bool(exp_extra[j]),
    } for j in range(grid.size)]
    out: dict[str, Any] = {
        "status": "ok", "method": "Optimus-style utility selection from logistic E-R models",
        "utility": util, "toxicity_cap": toxicity_cap,
        "exposure_mapping": {"efficacy": echo_e, "toxicity": echo_t},
        "exposure_metric": {"efficacy": efficacy_fit.get("exposure_label"),
                            "toxicity": toxicity_fit.get("exposure_label")},
        "doses": rows, "prob_none_feasible": sig(prob_none),
        "uncertainty": {"source": {"efficacy": src_e, "toxicity": src_t}, "n_draws": int(k),
                        "seed": seed, "ci_level": ci_level},
        "warnings": warnings, "notes": list(_NOTES),
    }
    if not feasible.any():
        out["status"] = "no_feasible_dose"
        out["message"] = (f"No candidate dose satisfies P_tox <= {toxicity_cap:g} at the point estimates; "
                          "no dose is selected. Review the cap, the grid, or the toxicity model.")
        return out
    j = int(np.argmax(np.where(feasible, utility_pt, -np.inf)))      # first max -> lowest dose on ties
    edge = "min" if j == 0 else "max" if j == grid.size - 1 else None
    out["selected"] = {"dose": sig(grid[j]), "utility": sig(utility_pt[j]), "p_eff": sig(p_eff[j]),
                       "p_tox": sig(p_tox[j]), "prob_optimal": sig(prob_opt[j]), "at_grid_edge": edge}
    mode = int(np.argmax(prob_opt))
    out["most_probable_optimal_dose"] = sig(grid[mode])
    if edge:
        out["warnings"].append(
            f"The selected dose ({grid[j]:g}) sits at the {'lowest' if edge == 'min' else 'highest'} edge of the "
            "dose grid: the true optimum may lie beyond it and the E-R models would be extrapolated. "
            "Widen the grid before relying on this selection.")
    if mode != j:
        out["warnings"].append(
            f"The most probable optimal dose under parameter uncertainty ({grid[mode]:g}) differs from the "
            f"point-estimate selection ({grid[j]:g}).")
    return out

