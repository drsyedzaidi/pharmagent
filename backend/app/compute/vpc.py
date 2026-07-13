"""Goodness-of-fit (observed vs predicted) and visual predictive check (VPC).

Pure, deterministic compute on top of the structural PK / PK-PD simulator in
``app.compute.pk_simulate``. No file I/O, no network, no global mutable state.

Two public functions:

    obs_vs_pred
        Pair every positive observed concentration with its individual
        prediction (IPRED, simulated with the subject's own estimated
        parameters) and its population prediction (PRED, simulated with the
        population typical parameters). Goodness-of-fit is reported on the log
        scale (concentrations are strictly positive): ``r2_log_ipred`` is the
        coefficient of determination of log-observed vs log-IPRED, and
        ``rmse_log_ipred`` is the root-mean-square error on the log scale.

    vpc_band
        Simulate ``n_sim`` virtual subjects whose structural parameters are
        drawn from independent log-normal between-subject distributions
        (``p_i = typical * exp(eta)``, ``eta ~ N(0, sd)``), then return the per
        time-point 5th / 50th / 95th percentiles of the predicted concentration.
        A fixed ``numpy.random.default_rng(seed)`` makes the band deterministic.

All returned floats are rounded to 6 decimal places; non-finite predictions are
dropped pairwise so degenerate simulations never poison the summary statistics.
"""
from __future__ import annotations

import numpy as np

from app.compute.pk_models import get_model
from app.compute.pk_simulate import simulate

# Percentiles reported by the VPC band.
_VPC_PERCENTILES = (5.0, 50.0, 95.0)
# Decimal places for all reported floats.
_ROUND_DP = 6


def _cv_pct_to_sd(cv_pct: float | None) -> float:
    """Convert a between-subject CV% to a log-normal SD.

    For a log-normal random effect ``exp(eta)`` with ``eta ~ N(0, sd)``, the
    coefficient of variation on the natural scale is ``sqrt(exp(sd^2) - 1)``.
    Inverting gives ``sd = sqrt(ln(1 + (cv/100)^2))``. A missing or
    non-positive CV maps to ``sd = 0`` (no variability on that parameter).
    """
    if cv_pct is None:
        return 0.0
    cv = float(cv_pct)
    if cv <= 0.0:
        return 0.0
    return float(np.sqrt(np.log1p((cv / 100.0) ** 2)))


def obs_vs_pred(model_key: str, subjects: list[dict],
                individual_params_by_subject: dict, typical_params: dict,
                *, wt_default: float = 70.0) -> dict:
    """Pair observed concentrations with individual and population predictions.

    Parameters
    ----------
    model_key:
        Registry key of the structural model (e.g. ``"oral_1cmt"``).
    subjects:
        ``[{"subject", "doses": [{time, amt}], "obs_t", "obs_c", "wt"}]``.
    individual_params_by_subject:
        ``{subject_id: {param: value}}`` for the converged subjects only.
    typical_params:
        ``{param: value}`` population typical parameters (e.g. geometric means).
    wt_default:
        Weight used when a subject has no ``wt`` entry.

    Returns
    -------
    dict with keys ``"observed"``, ``"ipred"``, ``"pred"`` (parallel lists of
    paired values, rounded to 6 dp) and ``"gof"`` with ``"r2_log_ipred"``,
    ``"rmse_log_ipred"`` (float or None) and ``"n"`` (int).
    """
    model = get_model(model_key)

    observed: list[float] = []
    ipred: list[float] = []
    pred: list[float] = []

    for subject in subjects:
        sid = subject.get("subject")
        indiv = individual_params_by_subject.get(sid)
        if indiv is None:
            continue  # only subjects with individual params contribute

        obs_t = np.asarray(subject.get("obs_t", []), dtype=float)
        obs_c = np.asarray(subject.get("obs_c", []), dtype=float)
        if obs_t.size == 0 or obs_c.size == 0 or obs_t.size != obs_c.size:
            continue

        doses = list(subject.get("doses", []))
        wt = float(subject.get("wt", wt_default))

        sim_ipred = simulate(model, dict(indiv), doses, obs_t, wt=wt)["cp"]
        sim_pred = simulate(model, dict(typical_params), doses, obs_t, wt=wt)["cp"]

        for obs_val, ip_val, pr_val in zip(obs_c, sim_ipred, sim_pred):
            obs_f = float(obs_val)
            if not np.isfinite(obs_f) or obs_f <= 0.0:
                continue  # only positive observations are paired (log scale)
            ip_f = float(ip_val)
            pr_f = float(pr_val)
            if not np.isfinite(ip_f) or not np.isfinite(pr_f):
                continue  # drop non-finite predictions pairwise
            observed.append(obs_f)
            ipred.append(ip_f)
            pred.append(pr_f)

    gof = _gof_log(observed, ipred)

    return {
        "observed": [round(v, _ROUND_DP) for v in observed],
        "ipred": [round(v, _ROUND_DP) for v in ipred],
        "pred": [round(v, _ROUND_DP) for v in pred],
        "gof": gof,
    }


def _gof_log(observed: list[float], ipred: list[float]) -> dict:
    """Log-scale R^2 and RMSE of observed vs individual prediction.

    Both inputs are already filtered to strictly positive, finite pairs. R^2 is
    ``1 - SS_res/SS_tot`` of (log obs vs log ipred); RMSE is
    ``sqrt(mean((log ipred - log obs)^2))``. Returns None for the metrics when
    there are too few points or the total sum of squares is degenerate (zero).
    """
    n = len(observed)
    if n == 0:
        return {"r2_log_ipred": None, "rmse_log_ipred": None, "n": 0}

    log_obs = np.log(np.asarray(observed, dtype=float))
    log_ip = np.log(np.asarray(ipred, dtype=float))

    residuals = log_ip - log_obs
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((log_obs - np.mean(log_obs)) ** 2))
    # R^2 undefined when observations carry no spread (single point / constant).
    r2: float | None = None if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot

    return {
        "r2_log_ipred": None if r2 is None else round(r2, _ROUND_DP),
        "rmse_log_ipred": round(rmse, _ROUND_DP),
        "n": n,
    }


def vpc_band(model_key: str, typical_params: dict, iiv_cv_by_param: dict,
             doses: list[dict], *, tmax: float, n_grid: int = 80,
             n_sim: int = 500, seed: int = 20250614,
             wt: float = 70.0) -> dict:
    """Visual predictive check band of predicted concentration over time.

    Simulate ``n_sim`` virtual subjects on a common grid
    ``t = linspace(0, tmax, n_grid)``. Each structural parameter ``p`` is drawn
    log-normally: ``p_i = typical_params[p] * exp(eta)`` with
    ``eta ~ N(0, sd_p)`` and ``sd_p`` derived from ``iiv_cv_by_param[p]`` via
    :func:`_cv_pct_to_sd`. Determinism is guaranteed by seeding
    ``numpy.random.default_rng(seed)``.

    Returns
    -------
    dict with keys ``"times"``, ``"p05"``, ``"p50"``, ``"p95"`` (parallel lists
    of length ``n_grid``, rounded to 6 dp). The percentiles are taken across the
    virtual subjects at each time point.
    """
    model = get_model(model_key)
    grid = np.linspace(0.0, float(tmax), int(n_grid))

    rng = np.random.default_rng(seed)
    param_names = model.params
    sds = {p: _cv_pct_to_sd(iiv_cv_by_param.get(p)) for p in param_names}

    cp_matrix = np.empty((int(n_sim), grid.size), dtype=float)
    for i in range(int(n_sim)):
        params_i: dict[str, float] = {}
        for p in param_names:
            base = float(typical_params[p])
            sd = sds[p]
            eta = rng.normal(0.0, sd) if sd > 0.0 else 0.0
            params_i[p] = base * float(np.exp(eta))
        cp_matrix[i, :] = simulate(model, params_i, list(doses), grid, wt=wt)["cp"]

    p05, p50, p95 = np.percentile(cp_matrix, _VPC_PERCENTILES, axis=0)

    return {
        "times": [round(float(t), _ROUND_DP) for t in grid],
        "p05": [round(float(v), _ROUND_DP) for v in p05],
        "p50": [round(float(v), _ROUND_DP) for v in p50],
        "p95": [round(float(v), _ROUND_DP) for v in p95],
    }


def pcvpc(model_key: str, subjects: list[dict], typical_params: dict,
          iiv_cv_by_param: dict, *, sigma_prop: float = 0.0,
          sigma_add: float = 0.0, n_bins: int = 8, n_sim: int = 200,
          seed: int = 20250614, wt_default: float = 70.0) -> dict:
    """Prediction-corrected VPC (Bergstrand et al. 2011).

    Each observation is prediction-corrected by the bin-median population
    prediction — ``pcY = y * median_bin(PRED) / PRED`` — which removes the
    influence of differing doses/covariates/designs across subjects so a single
    overlaid band is interpretable. Observations are binned by quantiles of
    observation time. ``n_sim`` replicates of the whole dataset are simulated
    under the full model (IIV draws + residual error) and prediction-corrected
    the same way; per bin we report the observed 5/50/95 percentiles, the
    simulated 5/50/95 (median across replicates), and the 90% CI of the
    simulated median (the shaded "median band" used to judge fit).

    Returns ``{"status", "n_bins", "n_sim", "bins": [{t, n, obs_p05/50/95,
    sim_p05/50/95, sim_med_lo, sim_med_hi}]}``.
    """
    model = get_model(model_key)
    param_names = model.params
    sds = {p: _cv_pct_to_sd(iiv_cv_by_param.get(p)) for p in param_names}

    # ── observed table: PRED (population, eta=0) at every observation ──
    rows: list[list[float]] = []          # [t, y, pred, subject_index]
    subj_rec: list[dict] = []             # per-subject design + global obs indices
    for si, s in enumerate(subjects):
        obs_t = np.asarray(s.get("obs_t", []), dtype=float)
        obs_c = np.asarray(s.get("obs_c", []), dtype=float)
        if obs_t.size == 0 or obs_t.size != obs_c.size:
            continue
        doses = list(s.get("doses", []))
        wt = float(s.get("wt", wt_default))
        pred = np.asarray(simulate(model, dict(typical_params), doses, obs_t, wt=wt)["cp"],
                          dtype=float)
        start = len(rows)
        subj_rec.append({"doses": doses, "wt": wt, "obs_t": obs_t,
                         "idx": list(range(start, start + obs_t.size))})
        for k in range(obs_t.size):
            rows.append([float(obs_t[k]), float(obs_c[k]), float(pred[k]), float(si)])

    if not rows:
        return {"status": "empty", "n_bins": 0, "n_sim": int(n_sim), "bins": []}

    arr = np.array(rows, dtype=float)
    t_all, y_all, pred_all = arr[:, 0], arr[:, 1], arr[:, 2]
    valid = np.isfinite(y_all) & (y_all > 0) & np.isfinite(pred_all) & (pred_all > 0)
    if not np.any(valid):
        return {"status": "empty", "n_bins": 0, "n_sim": int(n_sim), "bins": []}

    # ── time-quantile bins ──
    tv = t_all[valid]
    edges = np.unique(np.quantile(tv, np.linspace(0.0, 1.0, int(n_bins) + 1)))
    nb = max(len(edges) - 1, 1)
    bin_idx = np.clip(np.digitize(t_all, edges[1:-1]), 0, nb - 1)

    bin_pred = np.ones(nb, dtype=float)
    bin_t = np.zeros(nb, dtype=float)
    bin_n = np.zeros(nb, dtype=int)
    for b in range(nb):
        m = valid & (bin_idx == b)
        if np.any(m):
            bin_pred[b] = float(np.median(pred_all[m]))
            bin_t[b] = float(np.median(t_all[m]))
            bin_n[b] = int(np.sum(m))

    corr = bin_pred[bin_idx] / np.where(pred_all > 0, pred_all, np.nan)   # pc factor
    pcy = y_all * corr

    obs_pct = np.full((nb, 3), np.nan)
    for b in range(nb):
        m = valid & (bin_idx == b)
        if np.any(m):
            obs_pct[b] = np.percentile(pcy[m], _VPC_PERCENTILES)

    # ── n_sim replicates of the whole dataset under the full data model ──
    rng = np.random.default_rng(seed)
    rep_p = np.full((int(n_sim), nb, 3), np.nan)
    for r in range(int(n_sim)):
        sim_vals = np.full(len(rows), np.nan)
        for s in subj_rec:
            params_i = {p: float(typical_params[p]) *
                        float(np.exp(rng.normal(0.0, sds[p]) if sds[p] > 0 else 0.0))
                        for p in param_names}
            cp = np.asarray(simulate(model, params_i, s["doses"], s["obs_t"],
                                     wt=s["wt"])["cp"], dtype=float)
            var = sigma_add ** 2 + (sigma_prop * cp) ** 2
            ysim = cp + np.sqrt(np.maximum(var, 0.0)) * rng.normal(0.0, 1.0, cp.size)
            for k, gidx in enumerate(s["idx"]):
                sim_vals[gidx] = ysim[k]
        pcsim = sim_vals * corr
        for b in range(nb):
            m = valid & (bin_idx == b) & np.isfinite(pcsim)
            if np.any(m):
                rep_p[r, b] = np.percentile(pcsim[m], _VPC_PERCENTILES)

    sim_p05 = np.nanmedian(rep_p[:, :, 0], axis=0)
    sim_p50 = np.nanmedian(rep_p[:, :, 1], axis=0)
    sim_p95 = np.nanmedian(rep_p[:, :, 2], axis=0)
    med_lo = np.nanpercentile(rep_p[:, :, 1], 5.0, axis=0)
    med_hi = np.nanpercentile(rep_p[:, :, 1], 95.0, axis=0)

    def _r(x: float) -> float | None:
        return None if not np.isfinite(x) else round(float(x), _ROUND_DP)

    bins = [{
        "t": _r(bin_t[b]), "n": int(bin_n[b]),
        "obs_p05": _r(obs_pct[b, 0]), "obs_p50": _r(obs_pct[b, 1]), "obs_p95": _r(obs_pct[b, 2]),
        "sim_p05": _r(sim_p05[b]), "sim_p50": _r(sim_p50[b]), "sim_p95": _r(sim_p95[b]),
        "sim_med_lo": _r(med_lo[b]), "sim_med_hi": _r(med_hi[b]),
    } for b in range(nb)]
    return {"status": "ok", "n_bins": nb, "n_sim": int(n_sim), "bins": bins}
