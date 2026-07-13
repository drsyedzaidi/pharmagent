"""Generic ODE simulator for the PK / PK-PD model library.

Integrates any ``PKModel`` over a dosing schedule with scipy ``solve_ivp``,
handling bolus doses, zero-order infusions (rate), oral absorption lag, and
allometric weight scaling. Returns predicted central concentration (and PD
effect for PK/PD models) at the requested observation times.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from app.compute.pk_models import PKModel

_ATOL = 1e-9
_RTOL = 1e-7


def scale_params(model: PKModel, params: dict[str, float], wt: float) -> dict[str, float]:
    """Apply allometric WT scaling (centered at 70 kg) to flow/volume params."""
    out = dict(params)
    f = wt / 70.0
    for name, expo in model.allometric.items():
        if name in out and expo:
            out[name] = out[name] * (f ** expo)
    return out


def _initial(model: PKModel, p: dict[str, float]) -> np.ndarray:
    if model.init_state is not None:
        return np.asarray(model.init_state(p), dtype=float)
    return np.zeros(model.n_cmt, dtype=float)


def simulate(model: PKModel, params: dict[str, float],
             doses: list[dict[str, Any]], obs_times,
             *, wt: float = 70.0) -> dict[str, np.ndarray]:
    """Simulate the model.

    ``doses``: list of {"time", "amt", optional "cmt", optional "rate"}.
    Returns {"cp": ndarray} and, for PK/PD models, also {"eff": ndarray},
    aligned to ``obs_times``.
    """
    p = scale_params(model, params, wt)
    lag = float(p.get(model.lag_param, 0.0)) if model.lag_param else 0.0

    boluses: list[tuple[float, int, float]] = []
    infusions: list[tuple[float, float, int, float]] = []
    for d in doses:
        t = float(d["time"]) + lag
        cmt = int(d.get("cmt", model.dose_cmt))
        amt = float(d["amt"])
        rate = float(d.get("rate") or 0.0)
        if rate > 0:
            infusions.append((t, t + amt / rate, cmt, rate))
        else:
            boluses.append((t, cmt, amt))

    obs = np.asarray(obs_times, dtype=float)
    # ordered breakpoints: doses, infusion edges, observations
    bps = sorted(set([0.0]
                     + [b[0] for b in boluses]
                     + [e for inf in infusions for e in inf[:2]]
                     + obs.tolist()))

    # Fast path: LINEAR model with only bolus doses -> propagate by matrix
    # exponential (exact). expm(A*dt) is cached per unique dt within this call.
    use_linear = model.amat is not None and not infusions
    A = model.amat(p) if use_linear else None
    _expm_cache: dict[float, np.ndarray] = {}

    def step_linear(y, dt):
        key = round(dt, 9)
        E = _expm_cache.get(key)
        if E is None:
            E = expm(A * dt)
            _expm_cache[key] = E
        return E @ y

    def rhs(t, y):
        dy = np.asarray(model.rhs(t, y, p), dtype=float)
        for (s, e, cmt, rate) in infusions:
            if s <= t < e:
                dy[cmt] += rate
        return dy

    y = _initial(model, p)
    recorded: list[tuple[float, np.ndarray]] = []
    cur = bps[0]
    for (t, cmt, amt) in boluses:
        if abs(t - cur) < 1e-12:
            y[cmt] += amt
    recorded.append((cur, y.copy()))

    for nt in bps[1:]:
        if nt > cur + 1e-12:
            if use_linear:
                y = step_linear(y, nt - cur)
            else:
                sol = solve_ivp(rhs, (cur, nt), y, t_eval=[nt],
                                rtol=_RTOL, atol=_ATOL, method="LSODA")
                y = sol.y[:, -1].copy()
            cur = nt
        for (t, cmt, amt) in boluses:
            if abs(t - nt) < 1e-9:
                y[cmt] += amt
        recorded.append((nt, y.copy()))

    rec_t = np.array([r[0] for r in recorded])
    rec_y = [r[1] for r in recorded]

    def state_at(tq: float) -> np.ndarray:
        i = int(np.argmin(np.abs(rec_t - tq)))
        return rec_y[i]

    cp = np.array([float(model.cp(state_at(t), p)) for t in obs])
    out: dict[str, np.ndarray] = {"cp": cp}
    if model.eff is not None:
        out["eff"] = np.array([float(model.eff(state_at(t), p)) for t in obs])
    return out


def simulate_timecourse(model: PKModel, params: dict[str, float], *,
                        dose: float, tau: float, n_doses: int, tmax: float,
                        n_points: int = 160, wt: float = 70.0,
                        rate: float = 0.0) -> dict[str, Any]:
    """Forward-simulate a dosing regimen on a dense time grid (for plotting).

    Returns {"times", "cp", optionally "eff"} as plain lists.
    """
    n_doses = max(int(n_doses), 1)
    doses = [{"time": k * tau, "amt": dose, "rate": rate} for k in range(n_doses)
             if k * tau <= tmax]
    dose_times = [d["time"] for d in doses]
    grid = sorted(set(np.linspace(0.0, tmax, n_points).tolist() + dose_times))
    sim = simulate(model, params, doses, grid, wt=wt)
    out: dict[str, Any] = {"times": [round(t, 4) for t in grid],
                           "cp": [round(float(v), 6) for v in sim["cp"]]}
    if "eff" in sim:
        out["eff"] = [round(float(v), 6) for v in sim["eff"]]
    return out
