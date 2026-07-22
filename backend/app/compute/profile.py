"""Log-likelihood profiling for parameter confidence intervals.

The fourth and last of the parameter-precision methods named in the FDA's 2022
Population Pharmacokinetics guidance, alongside asymptotic standard errors,
bootstrap and sampling importance resampling.

The idea is direct: fix one parameter at a value away from its estimate,
RE-OPTIMIZE every other parameter, and see how much the objective worsens. The
confidence limits are the values where that worsening reaches the chi-square
cut-off for one degree of freedom -- 3.84 at 95%. Formally, for parameter p,

    dOFV(v) = min OFV(x)  subject to  x_p = v      minus   OFV(x_hat)

and the 95% limits are the two v where ``dOFV(v) = 3.84``.

What it buys over the asymptotic standard error is that the two limits are
found INDEPENDENTLY, so the interval is asymmetric whenever the likelihood is.
The Wald interval cannot express that: it is ``estimate +/- 1.96*SE`` by
construction. For an exactly quadratic objective the two agree exactly -- which
is the analytic check this module is tested against -- and they diverge exactly
to the extent the likelihood is non-quadratic, which is the information the
method exists to provide.

What it does NOT buy, and this is Dosne et al.'s stated drawback: profiling
gives the BOUNDS of an interval and nothing else. There is no joint uncertainty
distribution, no correlation structure, and nothing to simulate from. Bootstrap
and SIR both return a full set of parameter vectors; this returns two numbers
per parameter. It is also univariate -- profiling two parameters says nothing
about their joint region.

Cost sits between the other methods: each grid point is a constrained
re-optimization rather than a full refit on resampled data, and the search is
adaptive (bracket outward, then bisect), so a parameter typically costs on the
order of ten to twenty optimizations rather than the hundreds a bootstrap needs.

``profile_ofv_fn`` is injected, so this module runs no estimator and is testable
against objectives whose profile is known in closed form.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np

_ROUND_DP = 6
_MAX_SECONDS = 60.0 * 60.0
_MAX_EVALS_PER_SIDE = 40          # bracketing + bisection budget, per direction
_BISECT_TOL_REL = 1e-3            # stop when the bracket is this tight
_MAX_STEP_DOUBLINGS = 60          # backstop only; the real limit is _MAX_REACH
# How far the outward search may travel, as a multiple of |estimate|. Bounding
# the search by DISTANCE rather than by a doubling count is what keeps a tiny
# initial step (a near-zero asymptotic SE, or none at all) from producing a
# FALSE "unbounded" verdict -- the search would simply have stopped short of a
# crossing that exists. Extra doublings are one cheap evaluation each.
_MAX_REACH = 1.0e3
# A profile that dips BELOW the reported optimum means the fit was not at one.
_NEGATIVE_DOFV_TOL = -1e-6


def _round(x: float | None, dp: int = _ROUND_DP) -> float | None:
    if x is None or not isinstance(x, (int, float, np.floating)):
        return None
    xf = float(x)
    return round(xf, dp) if math.isfinite(xf) else None


def _chi2_1df(level: float) -> float:
    """Chi-square(1) quantile -- the dOFV cut-off for a `level` interval."""
    try:
        from scipy.stats import chi2
        return float(chi2.ppf(level, 1))
    except Exception:                                  # pragma: no cover
        return 3.841458820694124 if abs(level - 0.95) < 1e-9 else float("nan")


def _profile_one_side(evaluate: Callable[[float], float], start: float,
                      direction: int, step0: float, cut: float,
                      budget: dict[str, int]) -> dict[str, Any]:
    """Find where dOFV crosses ``cut`` in one direction from ``start``.

    Two phases, because each evaluation is a constrained re-optimization and
    a fixed grid would spend most of them in uninformative places:

    1. **Bracket** -- step outward, doubling, until dOFV exceeds the cut-off.
    2. **Bisect** -- narrow the bracketing interval to the crossing.

    Returns the limit and the points visited, or an explicit reason it failed
    to find one. A limit is never extrapolated beyond the region actually
    evaluated: if the objective never reaches the cut-off the interval is
    reported as unbounded on that side, which is a real finding about the
    data, not a failure to be papered over.
    """
    pts: list[tuple[float, float]] = []
    lo_v, lo_d = start, 0.0                            # known: dOFV(start) == 0
    hi_v: float | None = None
    hi_d: float | None = None
    step = abs(step0)
    negative_at: float | None = None

    reach = max(abs(start), 1.0) * _MAX_REACH
    for _ in range(_MAX_STEP_DOUBLINGS):
        if budget["n"] <= 0:
            return {"limit": None, "reason": "evaluation budget exhausted",
                    "points": pts, "negative_dofv_at": negative_at}
        if step > reach:
            break
        v = start + direction * step
        d = evaluate(v)
        budget["n"] -= 1
        if math.isfinite(d):
            pts.append((v, d))
            if d < _NEGATIVE_DOFV_TOL and negative_at is None:
                negative_at = v
            if d >= cut:
                hi_v, hi_d = v, d
                break
            lo_v, lo_d = v, d
        step *= 2.0

    if hi_v is None:
        return {"limit": None,
                "reason": ("objective never reached the cut-off out to "
                           f"{reach:.3g} from the estimate; the interval is "
                           "unbounded on this side"),
                "points": pts, "negative_dofv_at": negative_at}

    # Bisect. dOFV is monotone in |v - estimate| for a well-behaved likelihood;
    # where it is not, the bracket still contains A crossing and the
    # non-monotonicity is reported separately rather than silently averaged.
    while budget["n"] > 0 and abs(hi_v - lo_v) > _BISECT_TOL_REL * max(abs(start), 1e-8):
        mid = 0.5 * (lo_v + hi_v)
        d = evaluate(mid)
        budget["n"] -= 1
        if not math.isfinite(d):
            break
        pts.append((mid, d))
        if d < _NEGATIVE_DOFV_TOL and negative_at is None:
            negative_at = mid
        if d >= cut:
            hi_v, hi_d = mid, d
        else:
            lo_v, lo_d = mid, d

    # Linear interpolation between the final bracketing pair is accurate here
    # because dOFV is locally quadratic in the parameter, hence near-linear
    # over a bracket this tight.
    if hi_d is not None and hi_d > lo_d:
        frac = (cut - lo_d) / (hi_d - lo_d)
        limit = lo_v + frac * (hi_v - lo_v)
    else:
        limit = hi_v
    return {"limit": float(limit), "reason": None, "points": pts,
            "negative_dofv_at": negative_at}


def run_profile(*, profile_ofv_fn: Callable[[str, float], float],
                estimates: dict[str, float],
                ofv_hat: float,
                params: tuple[str, ...] | None = None,
                initial_step: dict[str, float] | None = None,
                ci_level: float = 0.95,
                max_seconds: float = _MAX_SECONDS,
                progress: Callable[[dict], None] | None = None
                ) -> dict[str, Any]:
    """Profile-likelihood confidence intervals.

    Parameters
    ----------
    profile_ofv_fn:
        ``(parameter_name, fixed_value) -> OFV`` with that parameter held and
        every other re-optimized. This is the entire cost and the only place an
        estimator is touched; this module never runs one itself.
    estimates:
        The point estimates, ``{name: value}``. Profiling is centred on these.
    ofv_hat:
        Objective at those estimates. dOFV is measured against it.
    params:
        Which to profile. Defaults to all of ``estimates``. Each costs roughly
        10-20 constrained optimizations, so profiling everything is rarely what
        you want.
    initial_step:
        Per-parameter starting step for the outward search, normally the
        asymptotic SE. A sensible step matters only for speed: too small just
        costs extra doublings. Defaults to 20% of |estimate|.
    ci_level:
        Interval mass. The dOFV cut-off is the chi-square(1) quantile at this
        level -- 3.84 at 95%.

    Returns
    -------
    JSON-safe dict with ``parameters`` (profile limits, their asymmetry, and
    the evaluated points), ``diagnostics``, and ``notes``.
    """
    if not estimates:
        return {"status": "no_parameters", "message": "no estimates to profile."}
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1); got {ci_level}")
    if not math.isfinite(float(ofv_hat)):
        return {"status": "no_fit",
                "message": "objective at the estimates is not finite."}

    names = tuple(params) if params else tuple(estimates.keys())
    missing = [p for p in names if p not in estimates]
    if missing:
        raise ValueError(f"cannot profile parameters absent from estimates: {missing}")

    cut = _chi2_1df(ci_level)
    if not math.isfinite(cut):
        raise ValueError(f"no chi-square cut-off available for ci_level={ci_level}")

    t0 = time.time()
    out: list[dict[str, Any]] = []
    n_evals = 0
    non_monotone: list[str] = []
    better_than_optimum: list[dict[str, Any]] = []

    for name in names:
        if time.time() - t0 > min(float(max_seconds), _MAX_SECONDS):
            break
        est = float(estimates[name])
        step0 = float((initial_step or {}).get(name) or 0.0)
        if not (math.isfinite(step0) and step0 > 0.0):
            step0 = max(abs(est) * 0.20, 1e-6)

        seen: list[tuple[float, float]] = []

        def evaluate(v: float, _n=name, _seen=seen) -> float:
            val = profile_ofv_fn(_n, v)
            d = float(val) - float(ofv_hat) if val is not None and math.isfinite(
                float(val)) else math.nan
            return d

        budget = {"n": _MAX_EVALS_PER_SIDE}
        lower = _profile_one_side(evaluate, est, -1, step0, cut, budget)
        n_evals += _MAX_EVALS_PER_SIDE - budget["n"]
        budget = {"n": _MAX_EVALS_PER_SIDE}
        upper = _profile_one_side(evaluate, est, +1, step0, cut, budget)
        n_evals += _MAX_EVALS_PER_SIDE - budget["n"]

        pts = sorted(lower["points"] + upper["points"], key=lambda t: t[0])
        lo, hi = lower["limit"], upper["limit"]

        # A dOFV below zero means the constrained fit BEAT the reported optimum:
        # the original fit was not converged. That is a finding about the fit,
        # not about the interval, and it invalidates the profile built on it.
        neg = lower["negative_dofv_at"] or upper["negative_dofv_at"]
        if neg is not None:
            better_than_optimum.append({"parameter": name, "at_value": _round(neg)})

        # Monotonicity: away from the estimate dOFV should rise. A dip means a
        # second optimum in that direction and the reported limit is only the
        # FIRST crossing.
        for side in (
            [(v, d) for v, d in pts if v < est][::-1],
            [(v, d) for v, d in pts if v > est],
        ):
            prev = -math.inf
            for _v, d in side:
                if d < prev - 1e-6:
                    non_monotone.append(name)
                    break
                prev = max(prev, d)
            if name in non_monotone:
                break

        asym = None
        if lo is not None and hi is not None and hi > lo:
            up, dn = hi - est, est - lo
            if dn > 0:
                asym = _round(up / dn)

        out.append({
            "parameter": name,
            "estimate": _round(est),
            "profile_lo": _round(lo),
            "profile_hi": _round(hi),
            "lower_reason": lower["reason"],
            "upper_reason": upper["reason"],
            "asymmetry_ratio": asym,
            "n_evaluations": len(pts),
            "profile": [{"value": _round(v), "dofv": _round(d)} for v, d in pts],
        })
        if progress is not None:
            progress({"done": len(out), "total": len(names), "evaluations": n_evals})

    notes: list[str] = []
    if better_than_optimum:
        notes.append(
            "A constrained fit found a BETTER objective than the reported "
            f"optimum for {', '.join(b['parameter'] for b in better_than_optimum)}: "
            "the original fit was not at a minimum, so these intervals are "
            "measured from the wrong reference and should not be used. Re-fit "
            "first (method='auto' searches multiple starts).")
    if non_monotone:
        notes.append(
            f"The profile is not monotone for {', '.join(sorted(set(non_monotone)))}: "
            "the objective dips again away from the estimate, which indicates a "
            "second optimum. The reported limit is the FIRST crossing, not "
            "necessarily the edge of the supported region.")
    unbounded = [p["parameter"] for p in out
                 if p["profile_lo"] is None or p["profile_hi"] is None]
    if unbounded:
        notes.append(
            f"No crossing found on at least one side for {', '.join(unbounded)}: "
            "the data do not bound that parameter at this level. Reported as "
            "None rather than extrapolated beyond the evaluated region.")
    notes.append(
        "Profiling gives interval BOUNDS only -- no joint distribution, no "
        "correlations, nothing to simulate from, and one parameter at a time "
        "(Dosne et al. 2016). Use bootstrap or SIR when a full uncertainty "
        "distribution is needed.")

    return {
        "status": "ok",
        "method": "log-likelihood profiling",
        "ci_level": ci_level,
        "dofv_cutoff": _round(cut),
        "n_parameters": len(out),
        "n_evaluations": n_evals,
        "seconds": _round(time.time() - t0, 1),
        "parameters": out,
        "diagnostics": {
            "fit_not_at_optimum": better_than_optimum,
            "non_monotone_parameters": sorted(set(non_monotone)),
            "unbounded_parameters": unbounded,
        },
        "notes": notes,
    }
