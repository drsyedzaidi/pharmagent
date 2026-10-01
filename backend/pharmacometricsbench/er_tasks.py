"""Exposure-response tasks (category ``er``): logistic OR, KM median, Cox HR, dose selection.

An opt-in pack, kept out of the frozen v0 set (``build_taskset(include_er=True)``
or ``build_er_taskset()``). As everywhere in the bench, ground truth is whatever
the validated compute function returns on the generated data, so an agent that
calls the tool scores 1.0 and one that free-hands the statistic does not. The
oracle bodies here are the same functions the generators use, so they cannot drift.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit

from app.compute.er_models import fit_logistic_er
from app.compute.er_survival import cox_ph, kaplan_meier
from app.compute.optimus import select_dose

from .spec import Target, Task

ER_BASE_SEED = 6000
KINDS = ("logistic_or", "km_median", "cox_hr", "optimal_dose")


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _fit_dict(intercept: float, slope: float) -> dict[str, Any]:
    """A logistic fit with declared coefficients and no sampling uncertainty."""
    return {"status": "ok", "model": "logistic", "ci_level": 0.95, "exposure_label": "AUC",
            "exposure_range": [0.0, 1e12],
            "coef": {"intercept": {"estimate": intercept}, "slope": {"estimate": slope}},
            "cov": [[0.0, 0.0], [0.0, 0.0]]}


# ── oracle (validated compute) ──────────────────────────────────────────────
def er_oracle(d: dict[str, Any]) -> dict[str, Any]:
    kind = d["analysis"]
    if kind == "logistic_or":
        r = fit_logistic_er(d["exposure"], d["response"])
        return {"or_per_unit": r["or_per_unit"]["estimate"], "or_per_sd": r["or_per_sd"]["estimate"],
                "p_slope": r["coef"]["slope"]["p"]}
    if kind == "km_median":
        r = kaplan_meier(d["time"], d["event"])
        before = [row["surv"] for row in r["table"] if row["time"] <= d["t_star"]]
        return {"km_median": r["median"]["estimate"], "km_surv_at_t": before[-1] if before else 1.0}
    if kind == "cox_hr":
        c = cox_ph(d["time"], d["event"], {"exposure": d["exposure"]})["covariates"][0]
        return {"hr_per_unit": c["hr"], "hr_per_sd": c["hr_per_sd"], "p_exposure": c["p"]}
    if kind == "optimal_dose":
        out = select_dose(
            doses=d["doses"], exposure_mapping=d["exposure_mapping"],
            efficacy_fit=_fit_dict(d["efficacy"]["intercept"], d["efficacy"]["slope"]),
            toxicity_fit=_fit_dict(d["toxicity"]["intercept"], d["toxicity"]["slope"]),
            utility_weight=d["utility_weight"], toxicity_cap=d.get("toxicity_cap"), n_draws=100, seed=1)
        s = out["selected"]
        return {"selected_dose": s["dose"], "utility_at_selected": s["utility"],
                "p_eff_at_selected": s["p_eff"], "p_tox_at_selected": s["p_tox"]}
    return {}


# ── naive (free-handed) ─────────────────────────────────────────────────────
def er_naive(d: dict[str, Any]) -> dict[str, Any]:
    kind = d["analysis"]
    if kind == "logistic_or":                       # median-split 2x2 odds ratio, not the regression slope
        x, y = np.array(d["exposure"], float), np.array(d["response"], float)
        hi = x > np.median(x)
        a, b = y[hi].sum() + 0.5, (hi.sum() - y[hi].sum()) + 0.5
        c, e = y[~hi].sum() + 0.5, ((~hi).sum() - y[~hi].sum()) + 0.5
        or_split = (a * e) / (b * c)
        gap = x[hi].mean() - x[~hi].mean()
        return {"or_per_unit": float(or_split ** (1.0 / gap)), "or_per_sd": float(or_split), "p_slope": 0.05}
    if kind == "km_median":                         # ignores censoring
        t = np.array(d["time"], float)
        return {"km_median": float(np.median(t)), "km_surv_at_t": float((t > d["t_star"]).mean())}
    if kind == "cox_hr":                            # crude event-rate ratio, high vs low half
        t, e = np.array(d["time"], float), np.array(d["event"], float)
        x = np.array(d["exposure"], float)
        hi = x > np.median(x)
        ratio = (e[hi].sum() / t[hi].sum()) / (e[~hi].sum() / t[~hi].sum())
        gap = x[hi].mean() - x[~hi].mean()
        return {"hr_per_unit": float(ratio ** (1.0 / gap)),
                "hr_per_sd": float(ratio ** (np.std(x, ddof=1) / gap)), "p_exposure": 0.05}
    if kind == "optimal_dose":                      # maximise efficacy, ignore toxicity: the highest dose
        top = max(d["doses"])
        m = d["exposure_mapping"]
        e = m["reference_exposure"] * (top / m["reference_dose"])
        pe = float(expit(d["efficacy"]["intercept"] + d["efficacy"]["slope"] * e))
        pt = float(expit(d["toxicity"]["intercept"] + d["toxicity"]["slope"] * e))
        return {"selected_dose": top, "utility_at_selected": pe, "p_eff_at_selected": pe,
                "p_tox_at_selected": pt}
    return {}


# ── generators ──────────────────────────────────────────────────────────────
def _task(i: int, kind: str, prompt: str, dataset: dict[str, Any], targets: list[Target], oracle: str,
          meta: dict[str, Any]) -> Task:
    return Task(task_id=f"er-{i:03d}", category="er", prompt=prompt,
                dataset={"analysis": kind, **dataset}, targets=targets, oracle=oracle, meta=meta)


def _logistic(i: int, seed: int) -> Task:
    for attempt in range(50):
        r = _rng(seed + 1000 * attempt)
        n = 60
        x = np.round(np.exp(r.normal(np.log(100.0), 0.4, n)), 2)
        a, b = -2.6, float(r.uniform(0.015, 0.03))
        y = (r.random(n) < expit(a + b * x)).astype(int)
        d = {"exposure": x.tolist(), "response": y.tolist()}
        if fit_logistic_er(d["exposure"], d["response"])["status"] == "ok":
            break
    truth = er_oracle({"analysis": "logistic_or", **d})
    return _task(
        i, "logistic_or",
        "Subject-level data: exposure (AUC, mg*h/L) and a binary adverse-event flag. Fit the logistic "
        "exposure-response model logit P(AE) = b0 + b1*AUC by maximum likelihood. Report the odds ratio per "
        "1 mg*h/L of AUC, the odds ratio per 1 SD of AUC (sample SD, n-1 denominator) and the Wald p-value "
        "of the slope.",
        {**d, "exposure_unit": "mg*h/L"},
        [Target("or_per_unit", truth["or_per_unit"], {"type": "abs", "abs": 0.001}),
         Target("or_per_sd", truth["or_per_sd"], {"type": "rel", "rel": 0.02}),
         Target("p_slope", truth["p_slope"], {"type": "rel", "rel": 0.05})],
        "app.compute.er_models.fit_logistic_er", {"true_slope": b})


def _km(i: int, seed: int) -> Task:
    for attempt in range(50):
        r = _rng(seed + 1000 * attempt)
        n = 40
        t_ev, t_ce = r.exponential(12.0, n), r.uniform(4.0, 40.0, n)
        t = np.maximum(np.round(np.minimum(t_ev, t_ce), 1), 0.1)
        e = (t_ev <= t_ce).astype(int)
        res = kaplan_meier(t.tolist(), e.tolist())
        if res["median"]["estimate"] is not None and res["n_events"] >= 15:
            break
    d = {"time": t.tolist(), "event": e.tolist(), "t_star": round(float(res["median"]["estimate"]) / 2.0, 1)}
    truth = er_oracle({"analysis": "km_median", **d})
    return _task(
        i, "km_median",
        "Time-to-event data (time in months, event = 1 for the event, 0 for censored). Compute the "
        "Kaplan-Meier estimate and report the median survival time (the first time S(t) <= 0.5; midpoint of "
        f"the step if S is exactly 0.5) and the survival probability S(t) at t = {d['t_star']} months.",
        d,
        [Target("km_median", truth["km_median"], {"type": "rel", "rel": 0.01}),
         Target("km_surv_at_t", truth["km_surv_at_t"], {"type": "rel", "rel": 0.02})],
        "app.compute.er_survival.kaplan_meier", {})


def _cox(i: int, seed: int) -> Task:
    for attempt in range(50):
        r = _rng(seed + 1000 * attempt)
        n = 80
        x = np.round(np.exp(r.normal(np.log(100.0), 0.35, n)), 1)
        beta = float(r.uniform(0.008, 0.02))
        t_ev = r.exponential(1.0 / (0.04 * np.exp(beta * (x - 100.0))))
        t_ce = r.uniform(8.0, 50.0, n)
        t = np.maximum(np.round(np.minimum(t_ev, t_ce), 1), 0.1)
        e = (t_ev <= t_ce).astype(int)
        d = {"time": t.tolist(), "event": e.tolist(), "exposure": x.tolist()}
        if cox_ph(d["time"], d["event"], {"exposure": d["exposure"]})["status"] == "ok":
            break
    truth = er_oracle({"analysis": "cox_hr", **d})
    return _task(
        i, "cox_hr",
        "Time-to-event data with a per-subject exposure (AUC). Fit a Cox proportional-hazards model with "
        "exposure as the single covariate (Efron ties). Report the hazard ratio per 1 unit of AUC, the "
        "hazard ratio per 1 SD of AUC (sample SD, n-1) and the Wald p-value.",
        d,
        [Target("hr_per_unit", truth["hr_per_unit"], {"type": "abs", "abs": 0.001}),
         Target("hr_per_sd", truth["hr_per_sd"], {"type": "rel", "rel": 0.02}),
         Target("p_exposure", truth["p_exposure"], {"type": "rel", "rel": 0.05})],
        "app.compute.er_survival.cox_ph", {"true_beta": beta})


def _dose(i: int, seed: int) -> Task:
    doses = [10.0, 25.0, 50.0, 100.0, 150.0, 200.0]
    for attempt in range(200):
        r = _rng(seed + 1000 * attempt)
        d = {"doses": doses,
             "exposure_mapping": {"type": "linear", "reference_dose": 100.0,
                                  "reference_exposure": float(np.round(r.uniform(100.0, 160.0), 1))},
             "efficacy": {"intercept": float(np.round(r.uniform(-2.5, -1.5), 2)),
                          "slope": float(np.round(r.uniform(0.02, 0.035), 3))},
             "toxicity": {"intercept": float(np.round(r.uniform(-5.0, -4.0), 2)),
                          "slope": float(np.round(r.uniform(0.018, 0.028), 3))},
             "utility_weight": float(np.round(r.uniform(1.0, 2.5), 1)),
             "toxicity_cap": float(np.round(r.uniform(0.25, 0.4), 2))}
        truth = er_oracle({"analysis": "optimal_dose", **d})
        if min(doses) < truth["selected_dose"] < max(doses):
            break
    return _task(
        i, "optimal_dose",
        "Dose selection (Project Optimus style). Efficacy and toxicity follow logistic exposure-response "
        "models logit P = intercept + slope * exposure (exposure in mg*h/L). Exposure scales linearly with "
        "dose: exposure = reference_exposure * dose / reference_dose. The declared clinical utility is "
        "U(d) = P_eff(d) - w * P_tox(d) with the given weight w; only doses with P_tox <= toxicity_cap are "
        "eligible. Report the dose on the grid that maximises U, and P_eff, P_tox and U at that dose.",
        d,
        [Target("selected_dose", truth["selected_dose"], {"type": "exact"}),
         Target("utility_at_selected", truth["utility_at_selected"], {"type": "rel", "rel": 0.01}),
         Target("p_eff_at_selected", truth["p_eff_at_selected"], {"type": "rel", "rel": 0.02}),
         Target("p_tox_at_selected", truth["p_tox_at_selected"], {"type": "rel", "rel": 0.02})],
        "app.compute.optimus.select_dose", {})


_BUILDERS = {"logistic_or": _logistic, "km_median": _km, "cox_hr": _cox, "optimal_dose": _dose}


def build_er_taskset() -> list[Task]:
    """One reproducible task per analysis (logistic OR, KM median, Cox HR, selected dose)."""
    return [_BUILDERS[k](i, ER_BASE_SEED + i) for i, k in enumerate(KINDS)]


__all__ = ["er_oracle", "er_naive", "build_er_taskset", "KINDS"]
