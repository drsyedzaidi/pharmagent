"""Model goodness-of-fit residual diagnostics for the PharmAgent PK platform.

Pure, deterministic compute built on top of the structural PK / PK-PD simulator
in ``app.compute.pk_simulate``. No file I/O, no network, no agent imports, no
global mutable state. Concentrations are strictly positive, so weighted
residuals are formed on the log scale (proportional-error parameterisation).

Two public functions:

    fit_residuals
        Individual weighted residuals (IWRES) of observed vs individual
        prediction. For every subject that has individual (post-hoc) parameter
        estimates, the model is simulated at the subject's observation times
        with the subject's own parameters (IPRED) and with the population
        typical parameters (PRED). With a proportional residual-error model the
        log-scale individual weighted residual is

            iwres = log(obs) - log(ipred),

        and the standardised IWRES rescales those residuals by their population
        mean and standard deviation. Only strictly positive, finite
        observation/prediction pairs contribute.

    npde
        Simulation-based *normalized prediction discrepancies* (npd; Comets
        et al. 2008, CMPB 90:154-166, Eqs. 4 and 7). For each subject a
        predictive distribution of concentrations is built by drawing ``n_sim``
        virtual parameter sets from independent log-normal between-subject
        distributions (``p_i = typical * exp(eta)``, ``eta ~ N(0, sd_p)``),
        simulating each at the subject's own regimen and observation times, and
        adding a residual-error draw so each simulated concentration is a full
        predictive observation

            y_sim = f(theta_i) + eps,
            eps ~ N(0, sigma_add^2 + (sigma_prop * f)^2).

        The empirical (mid-rank) cumulative probability of each observation
        within its simulated column is mapped through the standard-normal
        quantile function to give the discrepancy.

        The residual-error term is REQUIRED, not optional: the reference
        distribution is the distribution of an *observation*, not of the latent
        individual curve. Omitting it narrows the cloud, pushes real
        observations into the tails, and spuriously inflates ``|npd|`` for a
        correctly specified model (empirically, sd ~ 1.26 and ~12% beyond
        +/-1.96 instead of ~1.0 and ~5%). This mirrors ``app.compute.vpc.pcvpc``.

        NOTE: this is the *uncorrelated* discrepancy (npd), NOT the fully
        decorrelated NPDE of Brendel et al. (2006): the within-subject
        decorrelation step (whitening each subject's residual vector by the
        empirical mean and covariance of its simulated observations) is
        intentionally omitted. Nguyen et al. 2017 (CPT:PSP 6:87-109, Table 1
        footnote e) note npd is often preferred for graphical diagnostics
        because decorrelation can create or mask trends. npd equals NPDE exactly
        when each subject contributes a single observation; with multiple
        observations per subject they remain valid but still-correlated
        discrepancies.

All returned floats are rounded to 6 decimal places. Non-finite pairs are
dropped pairwise so degenerate simulations never poison the summary statistics.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from app.compute.dosing import time_after_dose
from app.compute.pk_models import get_model
from app.compute.pk_simulate import simulate

# Decimal places for all reported floats.
_ROUND_DP = 6
# Outlier threshold for the |NPDE| > 1.96 fraction (central 95% of N(0,1)).
_NPDE_OUTLIER = 1.96


def _keep_mask(subject: dict, n_obs: int) -> np.ndarray:
    """Boolean keep-mask over a subject's observations, dropping BLQ (censored)
    rows so a below-quantitation value (which carries the LLOQ in DV) is never
    scored as a quantified observation. Subjects without an ``obs_blq`` flag are
    unaffected (all-True), so every existing caller keeps its behaviour.
    """
    blq = subject.get("obs_blq")
    if not blq:
        return np.ones(n_obs, dtype=bool)
    m = ~np.asarray(blq, dtype=bool)
    if m.size != n_obs:  # length mismatch => be conservative, keep all
        return np.ones(n_obs, dtype=bool)
    return m


def _cv_pct_to_sd(cv_pct: float | None) -> float:
    """Convert a between-subject CV% to a log-normal SD.

    Mirrors ``app.compute.vpc._cv_pct_to_sd``. For a log-normal random effect
    ``exp(eta)`` with ``eta ~ N(0, sd)`` the coefficient of variation on the
    natural scale is ``sqrt(exp(sd^2) - 1)``; inverting gives
    ``sd = sqrt(ln(1 + (cv/100)^2))``. A missing or non-positive CV maps to
    ``sd = 0`` (no variability on that parameter).
    """
    if cv_pct is None:
        return 0.0
    cv = float(cv_pct)
    if cv <= 0.0:
        return 0.0
    return float(np.sqrt(np.log1p((cv / 100.0) ** 2)))


def fit_residuals(model_key: str, subjects: list[dict],
                  individual_params_by_subject: dict, typical_params: dict,
                  *, wt_default: float = 70.0) -> dict:
    """Log-scale individual weighted residuals (IWRES).

    Parameters
    ----------
    model_key:
        Registry key of the structural model (e.g. ``"oral_1cmt"``).
    subjects:
        ``[{"subject", "doses": [{time, amt}], "obs_t", "obs_c", "wt"}]``.
    individual_params_by_subject:
        ``{subject_id: {param: value}}`` for the converged subjects only.
        Subjects without an entry are skipped.
    typical_params:
        ``{param: value}`` population typical parameters (for PRED).
    wt_default:
        Weight used when a subject has no ``wt`` entry.

    Returns
    -------
    dict with keys ``"time"``, ``"obs"``, ``"ipred"``, ``"pred"``, ``"iwres"``,
    ``"iwres_std"``, ``"tad"`` (parallel lists of paired values, rounded to 6
    dp; ``tad`` entries are ``None`` for an observation before any dose) and
    ``"summary"`` = ``{"n", "iwres_mean", "iwres_sd", "n_tad_null"}``.
    ``iwres_mean`` and ``iwres_sd`` are ``None`` when there are no
    contributing pairs.
    """
    model = get_model(model_key)

    times: list[float] = []
    obs: list[float] = []
    ipred: list[float] = []
    pred: list[float] = []
    iwres: list[float] = []
    tad: list[float | None] = []

    for subject in subjects:
        sid = subject.get("subject")
        indiv = individual_params_by_subject.get(sid)
        if indiv is None:
            continue  # only subjects with individual params contribute

        obs_t = np.asarray(subject.get("obs_t", []), dtype=float)
        obs_c = np.asarray(subject.get("obs_c", []), dtype=float)
        if obs_t.size == 0 or obs_c.size == 0 or obs_t.size != obs_c.size:
            continue

        keep = _keep_mask(subject, obs_t.size)  # drop BLQ (censored) rows
        obs_t = obs_t[keep]
        obs_c = obs_c[keep]
        if obs_t.size == 0:
            continue

        doses = list(subject.get("doses", []))
        wt = float(subject.get("wt", wt_default))

        sim_ipred = simulate(model, dict(indiv), doses, obs_t, wt=wt)["cp"]
        sim_pred = simulate(model, dict(typical_params), doses, obs_t, wt=wt)["cp"]
        tad_all = time_after_dose(obs_t, doses)

        for t_val, obs_val, ip_val, pr_val, tad_val in zip(
                obs_t, obs_c, sim_ipred, sim_pred, tad_all):
            obs_f = float(obs_val)
            ip_f = float(ip_val)
            pr_f = float(pr_val)
            if not np.isfinite(obs_f) or obs_f <= 0.0:
                continue  # only positive observations (log scale)
            if not np.isfinite(ip_f) or ip_f <= 0.0:
                continue  # IPRED must be positive to form log residual
            if not np.isfinite(pr_f):
                continue  # drop non-finite population prediction pairwise
            times.append(float(t_val))
            obs.append(obs_f)
            ipred.append(ip_f)
            pred.append(pr_f)
            iwres.append(np.log(obs_f) - np.log(ip_f))
            tad.append(tad_val)

    n = len(iwres)
    if n == 0:
        return {
            "time": [], "obs": [], "ipred": [], "pred": [],
            "iwres": [], "iwres_std": [], "tad": [],
            "summary": {"n": 0, "iwres_mean": None, "iwres_sd": None, "n_tad_null": 0},
        }

    iwres_arr = np.asarray(iwres, dtype=float)
    mean = float(np.mean(iwres_arr))
    sd = float(np.std(iwres_arr))  # population sd
    if sd == 0.0:
        iwres_std = np.zeros_like(iwres_arr)  # guard sd == 0 -> zeros
    else:
        iwres_std = (iwres_arr - mean) / sd

    return {
        "time": [round(v, _ROUND_DP) for v in times],
        "obs": [round(v, _ROUND_DP) for v in obs],
        "ipred": [round(v, _ROUND_DP) for v in ipred],
        "pred": [round(v, _ROUND_DP) for v in pred],
        "iwres": [round(float(v), _ROUND_DP) for v in iwres_arr],
        "iwres_std": [round(float(v), _ROUND_DP) for v in iwres_std],
        "tad": [None if v is None else round(v, _ROUND_DP) for v in tad],
        "summary": {
            "n": n,
            "iwres_mean": round(mean, _ROUND_DP),
            "iwres_sd": round(sd, _ROUND_DP),
            "n_tad_null": sum(1 for v in tad if v is None),
        },
    }


def npde(model_key: str, subjects: list[dict], typical_params: dict,
         iiv_cv_by_param: dict, *, sigma_prop: float, sigma_add: float,
         n_sim: int = 500, seed: int = 20250614,
         wt_default: float = 70.0) -> dict:
    """Simulation-based normalized prediction discrepancies (npd; see module docs).

    Parameters
    ----------
    model_key:
        Registry key of the structural model (e.g. ``"oral_1cmt"``).
    subjects:
        ``[{"subject", "doses": [{time, amt}], "obs_t", "obs_c", "wt"}]``.
    typical_params:
        ``{param: value}`` population typical parameters (geometric means).
    iiv_cv_by_param:
        ``{param: cv_percent}`` between-subject CV% per structural parameter.
        Missing / non-positive entries imply no variability on that parameter.
    sigma_prop:
        Proportional residual-error coefficient (natural scale). Contributes
        ``(sigma_prop * f)`` to the residual SD of each simulated concentration.
        REQUIRED — omitting the residual term biases ``|npd|`` upward (see the
        module docstring); pass ``0.0`` only for a residual-error-free model.
    sigma_add:
        Additive residual-error SD (natural scale, concentration units).
    n_sim:
        Number of virtual subjects drawn per real subject.
    seed:
        Seed for ``numpy.random.default_rng`` -> bit-for-bit reproducibility.
    wt_default:
        Weight used when a subject has no ``wt`` entry.

    Returns
    -------
    dict with keys ``"metric"`` (``"npd"``), ``"time"``, ``"pred"``
    (per-observation simulated median), ``"npde"``, ``"tad"`` (parallel lists,
    rounded to 6 dp; ``tad`` entries are ``None`` for an observation before
    any dose) and ``"summary"`` = ``{"n", "mean", "sd", "pct_outside_1_96",
    "n_tad_null", "sigma_prop", "sigma_add"}``. The moment floats are ``None``
    when there are no contributing observations.
    """
    model = get_model(model_key)
    rng = np.random.default_rng(seed)
    n_sim = int(n_sim)
    sigma_prop = float(sigma_prop)
    sigma_add = float(sigma_add)

    param_names = model.params
    sds = {p: _cv_pct_to_sd(iiv_cv_by_param.get(p)) for p in param_names}

    # Clip bound keeps norm.ppf finite when an observation is below/above the
    # whole simulated cloud: F in [1/(2 n_sim), 1 - 1/(2 n_sim)].
    clip_lo = 1.0 / (2.0 * n_sim)
    clip_hi = 1.0 - clip_lo

    times: list[float] = []
    pred: list[float] = []
    pde: list[float] = []
    tad: list[float | None] = []

    for subject in subjects:
        obs_t = np.asarray(subject.get("obs_t", []), dtype=float)
        obs_c = np.asarray(subject.get("obs_c", []), dtype=float)
        if obs_t.size == 0 or obs_c.size == 0 or obs_t.size != obs_c.size:
            continue

        keep = _keep_mask(subject, obs_t.size)  # drop BLQ (censored) rows
        obs_t = obs_t[keep]
        obs_c = obs_c[keep]
        if obs_t.size == 0:
            continue

        doses = list(subject.get("doses", []))
        wt = float(subject.get("wt", wt_default))
        tad_all = time_after_dose(obs_t, doses)

        # Predictive matrix: n_sim virtual subjects x n_obs. Each row is the
        # FULL predictive draw y_sim = f(theta_i) + eps (between-subject
        # parameter draw plus a residual-error draw), mirroring
        # app.compute.vpc.pcvpc. The structural rows are filled first, then the
        # residual noise is added vectorized over the whole matrix.
        sim = np.empty((n_sim, obs_t.size), dtype=float)
        for i in range(n_sim):
            params_i: dict[str, float] = {}
            for p in param_names:
                base = float(typical_params[p])
                sd = sds[p]
                eta = rng.normal(0.0, sd) if sd > 0.0 else 0.0
                params_i[p] = base * float(np.exp(eta))
            sim[i, :] = simulate(model, params_i, list(doses), obs_t, wt=wt)["cp"]

        # Residual error: SD = sqrt(sigma_add^2 + (sigma_prop * f)^2) per cell.
        resid_var = sigma_add ** 2 + (sigma_prop * sim) ** 2
        sim = sim + np.sqrt(np.maximum(resid_var, 0.0)) * rng.standard_normal(sim.shape)

        col_median = np.median(sim, axis=0)

        for j in range(obs_t.size):
            obs_val = float(obs_c[j])
            if not np.isfinite(obs_val) or obs_val <= 0.0:
                continue  # only positive observations
            col = sim[:, j]
            finite = col[np.isfinite(col)]
            if finite.size == 0:
                continue  # degenerate simulated column
            less = float(np.count_nonzero(finite < obs_val))
            equal = float(np.count_nonzero(finite == obs_val))
            f = (less + 0.5 * equal) / n_sim
            f = min(max(f, clip_lo), clip_hi)
            times.append(float(obs_t[j]))
            pred.append(float(col_median[j]))
            pde.append(float(norm.ppf(f)))
            tad.append(tad_all[j])

    n = len(pde)
    if n == 0:
        return {
            "metric": "npd",
            "time": [], "pred": [], "npde": [], "tad": [],
            "summary": {"n": 0, "mean": None, "sd": None,
                        "pct_outside_1_96": None, "n_tad_null": 0,
                        "sigma_prop": round(sigma_prop, _ROUND_DP),
                        "sigma_add": round(sigma_add, _ROUND_DP)},
        }

    pde_arr = np.asarray(pde, dtype=float)
    mean = float(np.mean(pde_arr))
    sd = float(np.std(pde_arr))
    pct_outside = 100.0 * float(np.count_nonzero(np.abs(pde_arr) > _NPDE_OUTLIER)) / n

    return {
        "metric": "npd",
        "time": [round(v, _ROUND_DP) for v in times],
        "pred": [round(v, _ROUND_DP) for v in pred],
        "npde": [round(float(v), _ROUND_DP) for v in pde_arr],
        "tad": [None if v is None else round(v, _ROUND_DP) for v in tad],
        "summary": {
            "n": n,
            "mean": round(mean, _ROUND_DP),
            "sd": round(sd, _ROUND_DP),
            "pct_outside_1_96": round(pct_outside, _ROUND_DP),
            "n_tad_null": sum(1 for v in tad if v is None),
            "sigma_prop": round(sigma_prop, _ROUND_DP),
            "sigma_add": round(sigma_add, _ROUND_DP),
        },
    }
