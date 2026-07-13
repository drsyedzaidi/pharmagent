"""Dose sweep — multi-level exposure metrics over a dosing regimen.

Pure, deterministic compute module for the PharmAgent platform. Re-uses the
shared ODE simulator (``app.compute.pk_simulate.simulate_timecourse``) to
forward-simulate a model across several dose levels and reports steady-state
style exposure metrics over the last dosing interval.

For each dose level the simulator is run on a dense grid and the following
exposure metrics are computed over the LAST dosing interval
``[t_last, t_last + tau]`` with ``t_last = (n_doses - 1) * tau``:

  - ``cmax``    : maximum predicted central concentration over the whole profile
  - ``auc_tau`` : trapezoidal AUC of cp over the last interval (numpy.trapezoid)
  - ``cavg``    : ``auc_tau / tau`` (interval-average concentration)
  - ``ctrough`` : cp at the grid time closest to ``t_last + tau``

These metrics behave analytically: linear PK is dose-proportional (auc_tau
scales with dose), while saturable (Michaelis-Menten) elimination yields
more-than-dose-proportional exposure (dose-normalized auc_tau rises with dose).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.compute.pk_models import get_model
from app.compute.pk_simulate import simulate_timecourse


def _interval_metrics(
    times: list[float], cp: list[float], *, t_last: float, tau: float, tmax: float
) -> dict[str, float]:
    """Compute cmax / auc_tau / cavg / ctrough over the last dosing interval.

    ``cmax`` is taken over the whole profile; the AUC and trough are restricted
    to the last interval ``[t_last, min(t_last + tau, tmax)]``.
    """
    t_arr = np.asarray(times, dtype=float)
    cp_arr = np.asarray(cp, dtype=float)

    cmax = float(np.max(cp_arr))

    t_end = min(t_last + tau, tmax)
    in_interval = (t_arr >= t_last) & (t_arr <= t_end)
    t_win = t_arr[in_interval]
    cp_win = cp_arr[in_interval]

    if t_win.size >= 2:
        auc_tau = float(np.trapezoid(cp_win, t_win))
    else:
        auc_tau = 0.0

    cavg = auc_tau / tau if tau > 0 else 0.0

    # cp at the grid time closest to the end of the last interval
    if t_arr.size:
        idx = int(np.argmin(np.abs(t_arr - (t_last + tau))))
        ctrough = float(cp_arr[idx])
    else:
        ctrough = float(cp_arr[-1]) if cp_arr.size else 0.0

    return {
        "cmax": round(cmax, 6),
        "auc_tau": round(auc_tau, 6),
        "cavg": round(cavg, 6),
        "ctrough": round(ctrough, 6),
    }


def dose_sweep(
    model_key: str,
    params: dict,
    doses: list[float],
    *,
    tau: float,
    n_doses: int,
    tmax: float,
    wt: float = 70.0,
    n_points: int = 160,
) -> dict:
    """Simulate ``model_key`` across several dose levels and report exposure.

    For each dose ``d`` in ``doses`` the model is forward-simulated with
    ``simulate_timecourse`` and exposure metrics over the last dosing interval
    are collected. Profiles are returned in the same order as ``doses``.

    Returns a dict with keys ``model_key``, ``label``, ``tau``, ``n_doses``,
    ``tmax`` and ``profiles``; each profile carries ``dose``, ``times``, ``cp``
    (and ``eff`` for PK/PD models) plus the rounded ``cmax`` / ``auc_tau`` /
    ``cavg`` / ``ctrough`` metrics.
    """
    model = get_model(model_key)
    t_last = (n_doses - 1) * tau

    profiles: list[dict[str, Any]] = []
    for d in doses:
        sim = simulate_timecourse(
            model,
            params,
            dose=d,
            tau=tau,
            n_doses=n_doses,
            tmax=tmax,
            n_points=n_points,
            wt=wt,
        )
        times = sim["times"]
        cp = sim["cp"]
        metrics = _interval_metrics(
            times, cp, t_last=t_last, tau=tau, tmax=tmax
        )

        profile: dict[str, Any] = {"dose": d, "times": times, "cp": cp}
        if model.has_pd and "eff" in sim:
            profile["eff"] = sim["eff"]
        profile.update(
            cmax=metrics["cmax"],
            auc_tau=metrics["auc_tau"],
            cavg=metrics["cavg"],
            ctrough=metrics["ctrough"],
        )
        profiles.append(profile)

    return {
        "model_key": model_key,
        "label": model.label,
        "tau": tau,
        "n_doses": n_doses,
        "tmax": tmax,
        "profiles": profiles,
    }
