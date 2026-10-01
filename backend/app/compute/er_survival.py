"""Time-to-event exposure-response: Kaplan-Meier, log-rank and Cox proportional hazards.

* **Kaplan-Meier** with Greenwood variance. ``se_log`` is the Greenwood
  standard error of log S (what R's ``survfit`` reports as ``std.err``) and
  ``se`` is the standard error of S itself, ``S * se_log``. Confidence bands:
  ``log`` (R's default), ``plain`` and ``log-log``. The median follows R's
  convention: the first time S(t) <= 0.5, or the midpoint of the flat step when
  S(t) is exactly 0.5; its limits are where the lower and upper bands first
  reach 0.5 (``None`` when never reached).
* **Log-rank** test across groups (e.g. exposure quartiles).
* **Cox proportional hazards** by the partial likelihood, Newton-Raphson on the
  analytic score and information, **Efron** ties by default (R's ``coxph``
  default) or **Breslow**. Hazard ratios are reported per unit and per SD of
  each covariate with Wald intervals. No left truncation or time-varying
  covariates: one row per subject.

Validated against R ``survival`` 3.8.3 (tests/test_er_survival.py).

Refusals (structured, never a number): ``invalid_input``, ``insufficient_events``
(fewer than ``MIN_EVENTS_HARD``), ``zero_variance_covariate``,
``monotone_likelihood`` (a coefficient diverges because a covariate orders the
event times perfectly), ``singular_information`` (collinear covariates),
``non_convergence``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import chi2, norm

from app.compute.er_common import (
    EVENTS_PER_VARIABLE,
    MIN_EVENTS_HARD,
    ErRefusal,
    is_constant,
    refusal,
    sd_of,
    sig,
    sig_list,
    sig_matrix,
    to_float_array,
    z_crit,
)

_TOL = 1e-8                      # "exactly 0.5" tolerance for the median step rule
_CONF_TYPES = ("log", "plain", "log-log")
_TIES = ("efron", "breslow")
_MAX_ITER = 60
_STEP_TOL = 1e-10
_LL_TOL = 1e-12                 # relative slack for rounding noise in the monotonicity check
_MONOTONE_BETA_Z = 12.0          # |coef| per SD beyond this (HR ~ 1.6e5): diverging


def _clean_tte(time: Any, event: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = to_float_array(time, "time")
    e = to_float_array(event, "event")
    if t.size != e.size:
        raise ErRefusal("invalid_input", f"time has {t.size} values but event has {e.size}")
    keep = np.isfinite(t) & np.isfinite(e)
    if t.size == 0 or not keep.any():
        raise ErRefusal("invalid_input", "no complete (time, event) pairs")
    if np.any(t[keep] < 0):
        raise ErRefusal("invalid_input", "time must be non-negative")
    if not np.all((e[keep] == 0.0) | (e[keep] == 1.0)):
        raise ErRefusal("invalid_input", "event must be binary (1 = event, 0 = censored)")
    return t, e, keep


# ── Kaplan-Meier ─────────────────────────────────────────────────────────────

def _first_time_at_or_below(times: np.ndarray, curve: list[float | None], level: float) -> float | None:
    for tm, v in zip(times, curve):
        if v is not None and v <= level + _TOL:
            return float(tm)
    return None


def _median(times: np.ndarray, surv: np.ndarray) -> float | None:
    """R's survfit median: first S(t) <= 0.5; midpoint of the step when S == 0.5."""
    below = np.nonzero(surv <= 0.5 + _TOL)[0]
    if below.size == 0:
        return None
    i = int(below[0])
    if abs(surv[i] - 0.5) > _TOL:
        return float(times[i])
    later = np.nonzero(surv[i + 1:] < 0.5 - _TOL)[0]
    return float((times[i] + times[i + 1 + later[0]]) / 2.0) if later.size else float(times[i])


def _bands(surv: np.ndarray, se_log: np.ndarray, zc: float, conf_type: str
           ) -> tuple[list[float | None], list[float | None]]:
    lo: list[float | None] = []
    hi: list[float | None] = []
    for s, sl in zip(surv, se_log):
        if s <= 0.0 or not math.isfinite(sl):
            lo.append(None); hi.append(None)                    # Greenwood undefined at S = 0
        elif conf_type == "log":
            lo.append(float(s * math.exp(-zc * sl))); hi.append(float(min(1.0, s * math.exp(zc * sl))))
        elif conf_type == "plain":
            lo.append(float(max(0.0, s - zc * s * sl))); hi.append(float(min(1.0, s + zc * s * sl)))
        elif s >= 1.0:                                          # log-log needs 0 < S < 1
            lo.append(None); hi.append(None)
        else:
            ll, spread = math.log(-math.log(s)), zc * sl / abs(math.log(s))
            lo.append(float(math.exp(-math.exp(ll + spread)))); hi.append(float(math.exp(-math.exp(ll - spread))))
    return lo, hi


def kaplan_meier(time: Any, event: Any, *, ci_level: float = 0.95,
                 conf_type: str = "log") -> dict[str, Any]:
    """Kaplan-Meier estimate with Greenwood variance, one row per distinct time."""
    if conf_type not in _CONF_TYPES:
        return refusal("invalid_input", f"conf_type must be one of {_CONF_TYPES}")
    try:
        zc = z_crit(ci_level)
        t_all, e_all, keep = _clean_tte(time, event)
    except ErRefusal as exc:
        return refusal(exc.status, exc.message)
    except ValueError as exc:
        return refusal("invalid_input", str(exc))
    t, e = t_all[keep], e_all[keep]
    times, inv = np.unique(t, return_inverse=True)
    total = np.bincount(inv, minlength=times.size).astype(float)
    d = np.bincount(inv, weights=e, minlength=times.size)
    c = total - d
    n_risk = t.size - np.concatenate([[0.0], np.cumsum(total)[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        surv = np.cumprod(1.0 - d / n_risk)
        green = np.cumsum(np.where(d > 0, d / (n_risk * (n_risk - d)), 0.0))
    se_log = np.sqrt(green)                    # inf once n - d == 0 (S has reached 0)
    lo, hi = _bands(surv, se_log, zc, conf_type)
    rows = [{"time": sig(tm), "n_risk": int(nr), "n_event": int(dd), "n_censor": int(cc),
             "surv": sig(s), "se_log": sig(sl), "se": sig(s * sl) if math.isfinite(sl) else None,
             "lo": sig(a) if a is not None else None, "hi": sig(b) if b is not None else None}
            for tm, nr, dd, cc, s, sl, a, b in zip(times, n_risk, d, c, surv, se_log, lo, hi)]
    med = _median(times, surv)
    return {
        "status": "ok", "method": "Kaplan-Meier (Greenwood variance)", "conf_type": conf_type,
        "ci_level": ci_level, "n": int(t.size), "n_events": int(d.sum()), "n_censored": int(c.sum()),
        "n_dropped_non_finite": int((~keep).sum()),
        "table": rows,
        "median": {"estimate": sig(med) if med is not None else None,
                   "lo": _opt(_first_time_at_or_below(times, lo, 0.5)),
                   "hi": _opt(_first_time_at_or_below(times, hi, 0.5))},
        "warnings": ([] if d.sum() else ["no events: the median and survival curve carry no information"]),
    }


def _opt(v: float | None) -> float | None:
    return sig(v) if v is not None else None


def logrank_test(time: Any, event: Any, group: Any) -> dict[str, Any]:
    """Log-rank (Mantel-Haenszel) test across >= 2 groups; matches R ``survdiff``."""
    try:
        t_all, e_all, keep = _clean_tte(time, event)
    except ErRefusal as exc:
        return refusal(exc.status, exc.message)
    labels = np.array([str(g) for g in group])
    if labels.size != t_all.size:
        return refusal("invalid_input", f"group has {labels.size} values but time has {t_all.size}")
    t, e, g = t_all[keep], e_all[keep], labels[keep]
    names = sorted(set(g.tolist()))
    k = len(names)
    if k < 2:
        return refusal("invalid_input", "log-rank needs at least two groups")
    obs, exp = np.zeros(k), np.zeros(k)
    var = np.zeros((k, k))
    for u in np.unique(t[e == 1]):
        at_risk = t >= u
        n_g = np.array([(at_risk & (g == nm)).sum() for nm in names], dtype=float)
        n, dd = n_g.sum(), float(((t == u) & (e == 1)).sum())
        obs += np.array([((t == u) & (e == 1) & (g == nm)).sum() for nm in names], dtype=float)
        exp += dd * n_g / n
        if n > 1:
            var += dd * (n - dd) / (n - 1.0) * (np.diag(n_g) / n - np.outer(n_g, n_g) / n ** 2)
    diff = (obs - exp)[:-1]
    try:
        stat = float(diff @ np.linalg.solve(var[:-1, :-1], diff))
    except np.linalg.LinAlgError:
        return refusal("singular_information", "log-rank variance matrix is singular (a group has no events).")
    return {"status": "ok", "df": k - 1, "chisq": sig(stat), "p": sig(chi2.sf(stat, k - 1)),
            "groups": [{"group": nm, "n": int((g == nm).sum()), "observed": sig(o), "expected": sig(x)}
                       for nm, o, x in zip(names, obs, exp)]}


# ── Cox proportional hazards ─────────────────────────────────────────────────

@dataclass(frozen=True)
class CoxMLE:
    beta: np.ndarray            # original covariate units
    cov: np.ndarray
    loglik0: float
    loglik: float
    n_iter: int
    n_tied_event_times: int


class _CoxData:
    """Time-sorted covariates plus the index structure of the tied event times.

    Every sum over a risk set is a reverse cumulative sum read at the first
    observation of that time; every sum over the tied events is a ``reduceat``
    over the time group. Efron's correction then needs one (group, fraction)
    pair per event (fraction l/d for the l-th of d tied events), which makes the
    whole likelihood a handful of vectorised array operations -- a bootstrap
    refits it hundreds of times, so a Python loop over event times is too slow.
    Breslow is the same with every fraction set to zero.
    """

    def __init__(self, time: np.ndarray, event: np.ndarray, xs: np.ndarray, ties: str) -> None:
        order = np.argsort(time, kind="stable")
        t, self.ev, self.xs = time[order], event[order], xs[order]
        _, self.starts = np.unique(t, return_index=True)
        d_group = np.add.reduceat(self.ev, self.starts)
        event_groups = np.nonzero(d_group > 0)[0]
        d_ev = d_group[event_groups].astype(int)
        self.pair_group = np.repeat(event_groups, d_ev)
        within = np.arange(self.pair_group.size) - np.repeat(np.cumsum(d_ev) - d_ev, d_ev)
        self.frac = within / np.repeat(d_ev, d_ev) if ties == "efron" else np.zeros(self.pair_group.size)
        self.n_tied = int((d_ev > 1).sum())
        self.xx = self.xs[:, :, None] * self.xs[:, None, :]

    def terms(self, beta: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """(log partial likelihood, score, information) at ``beta``."""
        eta = self.xs @ beta
        shift = float(eta.max())                 # cancels exactly in the likelihood; avoids overflow
        w = np.exp(eta - shift)
        s, g, f = self.starts, self.pair_group, self.frac
        r0 = np.cumsum(w[::-1])[::-1][s]
        r1 = np.cumsum((w[:, None] * self.xs)[::-1], axis=0)[::-1][s]
        r2 = np.cumsum((w[:, None, None] * self.xx)[::-1], axis=0)[::-1][s]
        ew = w * self.ev
        d0 = np.add.reduceat(ew, s)
        d1 = np.add.reduceat(ew[:, None] * self.xs, s, axis=0)
        d2 = np.add.reduceat(ew[:, None, None] * self.xx, s, axis=0)
        phi = r0[g] - f * d0[g]
        m1 = (r1[g] - f[:, None] * d1[g]) / phi[:, None]
        loglik = float(np.sum((eta - shift) * self.ev) - np.sum(np.log(phi)))
        score = (self.xs * self.ev[:, None]).sum(axis=0) - m1.sum(axis=0)
        info = (np.einsum("m,mpq->pq", 1.0 / phi, r2[g] - f[:, None, None] * d2[g])
                - np.einsum("mp,mq->pq", m1, m1))
        return loglik, score, info


def cox_mle(time: np.ndarray, event: np.ndarray, x: np.ndarray, ties: str) -> CoxMLE:
    """Partial-likelihood MLE on validated finite data. Raises ``ErRefusal``."""
    n_events = int(event.sum())
    if n_events < MIN_EVENTS_HARD:
        raise ErRefusal("insufficient_events",
                        f"Only {n_events} events (minimum {MIN_EVENTS_HARD}); a Cox model is not estimable "
                        "from so few.")
    if any(is_constant(x[:, j]) for j in range(x.shape[1])):
        raise ErRefusal("zero_variance_covariate", "A covariate is constant across subjects; its effect "
                                                   "cannot be estimated.")
    sd = x.std(axis=0, ddof=1)
    data = _CoxData(time, event, (x - x.mean(axis=0)) / sd, ties)     # standardised: conditioning only
    beta = np.zeros(x.shape[1])
    ll0, score, info = data.terms(beta)
    ll = ll0
    converged, n_iter = False, 0
    for n_iter in range(1, _MAX_ITER + 1):
        try:
            step = np.linalg.solve(info, score)
        except np.linalg.LinAlgError as exc:
            raise ErRefusal("singular_information",
                            "The information matrix is singular (collinear covariates).") from exc
        scale = 1.0
        for _ in range(40):                                   # step halving: loglik must not fall
            cand = beta + scale * step
            cand_ll, cand_score, cand_info = data.terms(cand)
            if np.isfinite(cand_ll) and cand_ll >= ll - _LL_TOL * (1.0 + abs(ll)):
                break
            scale *= 0.5
        else:
            break
        beta, ll, score, info = cand, cand_ll, cand_score, cand_info
        if np.max(np.abs(beta)) > _MONOTONE_BETA_Z:           # diverging: stop before the information underflows
            raise ErRefusal("monotone_likelihood",
                            "A covariate orders the event times (almost) perfectly, so its coefficient "
                            "diverges and the partial likelihood has no finite maximum. No coefficients "
                            "are reported.")
        if not (np.all(np.isfinite(score)) and np.all(np.isfinite(info))):
            break
        if np.max(np.abs(scale * step)) < _STEP_TOL:
            converged = True
            break
    if not converged:
        raise ErRefusal("non_convergence", f"Newton iterations did not converge in {_MAX_ITER} steps.")
    try:
        cov_z = np.linalg.inv(info)
    except np.linalg.LinAlgError as exc:
        raise ErRefusal("singular_information", "The information matrix is singular.") from exc
    back = 1.0 / sd
    return CoxMLE(beta=beta * back, cov=cov_z * np.outer(back, back), loglik0=ll0, loglik=ll,
                  n_iter=n_iter, n_tied_event_times=data.n_tied)


def cox_ph(time: Any, event: Any, covariates: dict[str, Any], *, ties: str = "efron",
           ci_level: float = 0.95) -> dict[str, Any]:
    """Cox PH on one or more covariates (first = the exposure by convention)."""
    if ties not in _TIES:
        return refusal("invalid_input", f"ties must be one of {_TIES} (Efron is R's default)")
    if not covariates:
        return refusal("invalid_input", "at least one covariate is required")
    try:
        zc = z_crit(ci_level)
        t_all, e_all, keep = _clean_tte(time, event)
        cols = {name: to_float_array(v, name) for name, v in covariates.items()}
        for name, arr in cols.items():
            if arr.size != t_all.size:
                raise ErRefusal("invalid_input", f"covariate {name} has {arr.size} values but time has {t_all.size}")
            keep = keep & np.isfinite(arr)
        names = list(cols)
        x = np.column_stack([cols[nm][keep] for nm in names])
        t, e = t_all[keep], e_all[keep]
        fit = cox_mle(t, e, x, ties)
    except ErRefusal as exc:
        return refusal(exc.status, exc.message)
    except ValueError as exc:
        return refusal("invalid_input", str(exc))

    se = np.sqrt(np.diag(fit.cov))
    n_events = int(e.sum())
    rows = []
    for j, nm in enumerate(names):
        b, s, sd = float(fit.beta[j]), float(se[j]), sd_of(x[:, j])
        z = b / s
        rows.append({
            "name": nm, "coef": sig(b), "se": sig(s), "z": sig(z), "p": sig(2.0 * norm.sf(abs(z))),
            "hr": sig(math.exp(b)), "hr_lo": sig(math.exp(b - zc * s)), "hr_hi": sig(math.exp(b + zc * s)),
            "sd": sig(sd), "hr_per_sd": sig(math.exp(b * sd)),
            "hr_per_sd_lo": sig(math.exp((b - zc * s) * sd)), "hr_per_sd_hi": sig(math.exp((b + zc * s) * sd)),
        })
    lr = 2.0 * (fit.loglik - fit.loglik0)
    warnings: list[str] = []
    if n_events / len(names) < EVENTS_PER_VARIABLE:
        warnings.append(
            f"Only {n_events / len(names):.1f} events per variable ({n_events} events, {len(names)} "
            f"covariate(s); guideline >= {EVENTS_PER_VARIABLE}): coefficients are imprecise and Wald "
            "intervals may be unreliable.")
    return {
        "status": "ok", "model": "cox", "ties": ties, "ci_level": ci_level,
        "n": int(t.size), "n_events": n_events, "n_dropped_non_finite": int((~keep).sum()),
        "n_tied_event_times": fit.n_tied_event_times,
        "covariates": rows, "cov": sig_matrix(fit.cov),
        "loglik_null": sig(fit.loglik0), "loglik": sig(fit.loglik),
        "lr_chisq": sig(lr), "lr_df": len(names), "lr_p": sig(chi2.sf(lr, len(names))),
        "converged": True, "n_iter": fit.n_iter,
        "exposure_range": [sig(float(x[:, 0].min())), sig(float(x[:, 0].max()))],
        "coef_vector": sig_list(fit.beta),
        "warnings": warnings,
    }
