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

from app.compute.dosing import time_after_dose
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
          seed: int = 20250614, wt_default: float = 70.0,
          correction: str = "pred", x_by: str = "time") -> dict:
    """Prediction-corrected VPC (Bergstrand et al. 2011).

    Each observation is corrected by a per-observation factor that is applied
    identically to the observed and the simulated concentrations, so a single
    overlaid band is interpretable across differing doses/covariates/designs.
    ``correction`` selects the factor:

    * ``"pred"`` (default): prediction correction,
      ``pcY = y * median_bin(PRED) / PRED`` (Bergstrand et al. 2011).
    * ``"dose"``: dose normalization, ``pcY = y * dose_ref / dose_i`` with
      ``dose_ref`` the modal first-dose amount. This is the dose-normalized VPC
      the FDA-cited course recommends for pooling across dose groups; pooling
      raw concentrations instead lets the lowest dose set the lower band and the
      highest set the upper, which is misleading.
    * ``"none"``: no correction (raw pooled) — mainly to demonstrate that
      artifact.

    ``x_by`` selects the binning axis: ``"time"`` (observation time, default) or
    ``"tad"`` (time after dose, via :func:`app.compute.dosing.time_after_dose`).
    TAD is what makes a multiple-dose profile interpretable, since absolute time
    spans many dose intervals.

    Observations are binned by quantiles of the chosen axis. ``n_sim`` replicates
    of the whole dataset are simulated under the full model (IIV draws + residual
    error) and corrected the same way; per bin we report the observed 5/50/95
    percentiles, the simulated 5/50/95 (median across replicates), and the 90% CI
    of the simulated median (the shaded "median band" used to judge fit).

    With ``correction="pred"`` and ``x_by="time"`` this is byte-for-byte the
    original prediction-corrected VPC.

    Returns ``{"status", "n_bins", "n_sim", "correction", "x_by",
    "bins": [{t, n, obs_p05/50/95, sim_p05/50/95, sim_med_lo, sim_med_hi}]}``.
    """
    correction = correction.lower()
    if correction not in ("pred", "dose", "none"):
        raise ValueError(f"correction must be 'pred'|'dose'|'none'; got {correction!r}")
    if x_by.lower() not in ("time", "tad"):
        raise ValueError(f"x_by must be 'time'|'tad'; got {x_by!r}")
    x_by = x_by.lower()
    model = get_model(model_key)
    param_names = model.params
    sds = {p: _cv_pct_to_sd(iiv_cv_by_param.get(p)) for p in param_names}

    # ── observed table: PRED (population, eta=0) at every observation ──
    rows: list[list[float]] = []          # [t, y, pred, subject_index, tad, dose]
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
        # TAD per observation (None before the first dose -> NaN, excluded from
        # a TAD-binned VPC). Dose = first-dose amount, for dose normalization.
        tad = time_after_dose(list(obs_t), doses)
        dose_amt = float(doses[0]["amt"]) if doses else float("nan")
        start = len(rows)
        subj_rec.append({"doses": doses, "wt": wt, "obs_t": obs_t,
                         "idx": list(range(start, start + obs_t.size))})
        for k in range(obs_t.size):
            tv = tad[k]
            rows.append([float(obs_t[k]), float(obs_c[k]), float(pred[k]), float(si),
                         float(tv) if tv is not None else float("nan"), dose_amt])

    if not rows:
        return {"status": "empty", "n_bins": 0, "n_sim": int(n_sim),
                "correction": correction, "x_by": x_by, "bins": []}

    arr = np.array(rows, dtype=float)
    t_all, y_all, pred_all = arr[:, 0], arr[:, 1], arr[:, 2]
    tad_all, dose_all = arr[:, 4], arr[:, 5]
    valid = np.isfinite(y_all) & (y_all > 0) & np.isfinite(pred_all) & (pred_all > 0)
    # Binning axis. Default "time" leaves `valid` and the edges exactly as before.
    x_all = t_all if x_by == "time" else tad_all
    if x_by == "tad":
        valid = valid & np.isfinite(tad_all)
    if not np.any(valid):
        return {"status": "empty", "n_bins": 0, "n_sim": int(n_sim),
                "correction": correction, "x_by": x_by, "bins": []}

    # ── quantile bins on the chosen axis ──
    xv = x_all[valid]
    edges = np.unique(np.quantile(xv, np.linspace(0.0, 1.0, int(n_bins) + 1)))
    nb = max(len(edges) - 1, 1)
    bin_idx = np.clip(np.digitize(x_all, edges[1:-1]), 0, nb - 1)

    bin_pred = np.ones(nb, dtype=float)
    bin_t = np.zeros(nb, dtype=float)
    bin_n = np.zeros(nb, dtype=int)
    for b in range(nb):
        m = valid & (bin_idx == b)
        if np.any(m):
            bin_pred[b] = float(np.median(pred_all[m]))
            bin_t[b] = float(np.median(x_all[m]))
            bin_n[b] = int(np.sum(m))

    if correction == "pred":
        corr = bin_pred[bin_idx] / np.where(pred_all > 0, pred_all, np.nan)
    elif correction == "dose":
        # Normalize every observation to a common reference dose (the modal
        # first-dose amount), so all dose groups share one comparable scale.
        finite_dose = dose_all[valid & np.isfinite(dose_all) & (dose_all > 0)]
        dose_ref = float(np.median(finite_dose)) if finite_dose.size else 1.0
        corr = dose_ref / np.where(dose_all > 0, dose_all, np.nan)
    else:                                                    # "none"
        corr = np.ones_like(y_all)
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
    return {"status": "ok", "n_bins": nb, "n_sim": int(n_sim),
            "correction": correction, "x_by": x_by, "bins": bins}


# Covariates with at least this many distinct numeric values are treated as
# continuous and split into quartiles; fewer, and each value is its own stratum.
# Matches app.compute.flexplot's continuous/categorical threshold.
_MIN_CONTINUOUS_LEVELS = 5


def _partition_by_stratum(subjects: list[dict], stratify_by: str | None
                          ) -> tuple[dict[str, list[dict]], str]:
    """Group subjects into strata by ``stratify_by``.

    Returns ``(partition, kind)`` where ``partition`` maps a stratum label to
    its subjects and ``kind`` is one of ``"all" | "dose" | "categorical" |
    "quartile" | "missing"``:

    * ``None`` -> a single ``"All"`` stratum (used for a pooled dose-normalized
      VPC).
    * ``"DOSE"`` -> grouped by first-dose amount (``doses[0]["amt"]``), NOT a
      covariate column, because that is the dose the band is about.
    * a categorical covariate (few distinct values) -> one stratum per level.
    * a continuous covariate (>= ``_MIN_CONTINUOUS_LEVELS`` distinct numeric
      values) -> quartile strata Q1..Q4, mirroring the course lab's
      ``findInterval(WT, quantile(...))``.

    Subjects missing the covariate are dropped from the partition (they cannot
    be placed); the caller reports how many.
    """
    if not stratify_by:
        return {"All": list(subjects)}, "all"

    key = stratify_by.strip()
    if key.upper() == "DOSE":
        parts: dict[str, list[dict]] = {}
        for s in subjects:
            doses = s.get("doses") or []
            amt = float(doses[0]["amt"]) if doses else None
            parts.setdefault("missing" if amt is None else f"{amt:g}", []).append(s)
        return parts, "dose"

    present = [(s, (s.get("cov") or {}).get(key)) for s in subjects]
    present = [(s, v) for s, v in present if v is not None]
    if not present:
        return {}, "missing"

    nums = [v for _s, v in present if isinstance(v, (int, float))]
    distinct = {round(float(v), 6) for v in nums}
    is_continuous = len(nums) == len(present) and len(distinct) >= _MIN_CONTINUOUS_LEVELS

    parts = {}
    if is_continuous:
        arr = np.array([float(v) for _s, v in present], dtype=float)
        q = np.quantile(arr, [0.25, 0.5, 0.75])
        for s, v in present:
            b = int(np.digitize(float(v), q))          # 0..3
            parts.setdefault(f"Q{b + 1}", []).append(s)
        return parts, "quartile"

    for s, v in present:
        lbl = f"{v:g}" if isinstance(v, (int, float)) else str(v)
        parts.setdefault(lbl, []).append(s)
    return parts, "categorical"


def stratified_vpc(model_key: str, subjects: list[dict], typical_params: dict,
                   iiv_cv_by_param: dict, *, stratify_by: str | None,
                   correction: str = "pred", x_by: str = "time",
                   sigma_prop: float = 0.0, sigma_add: float = 0.0,
                   n_bins: int = 8, n_sim: int = 200, seed: int = 20250614,
                   wt_default: float = 70.0, min_subjects: int = 5) -> dict:
    """Prediction-corrected VPC computed within each stratum.

    A single pooled VPC across dose groups is misleading — the lowest dose sets
    the lower band and the highest the upper (FDA-cited course, Week 11). The
    two remedies are stratification (this function with ``stratify_by``) and
    dose normalization (``correction="dose"``, which also pools correctly with
    ``stratify_by=None``). Both are computed by the same :func:`pcvpc` engine,
    once per stratum, so the total simulation cost is about one pooled VPC.

    Returns ``{"status", "stratify_by", "kind", "correction", "x_by",
    "strata": [{"label", "n", "bins": [...]}], "skipped": [...]}``. Strata with
    fewer than ``min_subjects`` subjects are reported in ``skipped`` rather than
    producing an under-powered band.
    """
    partition, kind = _partition_by_stratum(subjects, stratify_by)
    if kind == "missing":
        return {"status": "missing_covariate", "stratify_by": stratify_by,
                "message": f"no subject carries covariate {stratify_by!r}."}

    strata: list[dict] = []
    skipped: list[dict] = []
    # Numeric-aware ordering so 25 mg precedes 100 mg and Q1 precedes Q10.
    for label in sorted(partition, key=lambda x: (0, float(x)) if _isnum(x) else (1, x)):
        grp = partition[label]
        if len(grp) < min_subjects:
            skipped.append({"label": label, "n": len(grp),
                            "reason": f"fewer than {min_subjects} subjects"})
            continue
        pc = pcvpc(model_key, grp, typical_params, iiv_cv_by_param,
                   sigma_prop=sigma_prop, sigma_add=sigma_add, n_bins=n_bins,
                   n_sim=n_sim, seed=seed, wt_default=wt_default,
                   correction=correction, x_by=x_by)
        strata.append({"label": label, "n": len(grp), "bins": pc.get("bins", [])})

    return {"status": "ok" if strata else "no_strata",
            "stratify_by": stratify_by, "kind": kind,
            "correction": correction, "x_by": x_by,
            "strata": strata, "skipped": skipped}


def _isnum(s: str) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def _sorted_labels(labels) -> list:
    """Numeric-aware label order (25 before 100, Q1 before Q10)."""
    return sorted(labels, key=lambda x: (0, float(x)) if _isnum(x) else (1, str(x)))


def _trapz_partial_auc(t: np.ndarray, c: np.ndarray) -> float:
    """Linear-trapezoidal partial AUC over the given points (AUC(0-tlast)).

    Matches the course lab's ``mrgmisc::auc_partial(time, conc)`` (a simple
    linear trapezoid, NOT the NCA linear-up/log-down rule). Because the same
    rule is applied to the observed AND the simulated concentrations, any
    trapezoidal bias cancels in the observed-vs-simulated comparison.
    Concentrations are floored at 0 first (additive residual error can drive a
    simulated value negative, which is not a physical exposure contribution).
    """
    t = np.asarray(t, dtype=float)
    c = np.maximum(np.asarray(c, dtype=float), 0.0)
    if t.size < 2:
        return 0.0
    order = np.argsort(t, kind="stable")
    ts, cs = t[order], c[order]
    # Linear trapezoid; explicit sum avoids the numpy 1.x trapz / 2.x trapezoid split.
    return float(np.sum((cs[:-1] + cs[1:]) * 0.5 * np.diff(ts)))


def _exposure_window_mask(obs_t: np.ndarray, doses: list[dict]) -> np.ndarray:
    """Points included in a subject's exposure summary.

    Single dose -> the whole observed curve (AUC(0-tlast), Cmax). Multiple dose
    -> only the LAST dosing interval (points at/after the last dose time), i.e.
    steady-state AUC_tau / Cmax,ss — mirroring the course lab, which windows the
    MAD/renal exposure checks to the final interval (``TIME > 132``) while the
    single-dose SAD check uses the full profile.
    """
    if len(doses) > 1:
        last = max(float(d["time"]) for d in doses)
        return obs_t >= last
    return np.ones(obs_t.shape, dtype=bool)


# Simulated-mean interval reported by the exposure predictive check.
_EXP_CI = (2.5, 97.5)


def exposure_predictive_check(model_key: str, subjects: list[dict], typical_params: dict,
                              iiv_cv_by_param: dict, *, group_by: str | None = "DOSE",
                              sigma_prop: float = 0.0, sigma_add: float = 0.0,
                              n_sim: int = 200, seed: int = 20250614,
                              wt_default: float = 70.0, n_hist_bins: int = 24,
                              min_subjects: int = 3) -> dict:
    """Exposure predictive check (IU PopPK Week 11, ``model-vpc.R``).

    Summarise each subject's exposure over its window (:func:`_exposure_window_mask`):
    partial AUC (linear trapezoid) and Cmax = max(observed). The check statistic
    is the GROUP MEAN exposure (grouped by dose or a covariate). ``n_sim``
    replicates of the whole dataset are simulated under the full model (IIV draws
    + residual error); each yields a simulated group mean, and the observed group
    mean should fall inside the simulated 2.5-97.5% interval. This checks that the
    model reproduces exposure central tendency per group — the quantity that
    drives exposure-based dosing — which a concentration-time VPC does not test
    directly.

    Returns ``{"status", "group_by", "kind", "n_sim", "ci",
    "groups": [{"label", "n", "auc": M, "cmax": M}], "skipped", "multiple_dose"}``
    where each metric ``M`` is ``{observed, sim_median, sim_lo, sim_hi, within,
    hist: {edges, counts}}``.
    """
    model = get_model(model_key)
    param_names = model.params
    sds = {p: _cv_pct_to_sd(iiv_cv_by_param.get(p)) for p in param_names}

    partition, kind = _partition_by_stratum(subjects, group_by)
    if kind == "missing":
        return {"status": "missing_covariate", "group_by": group_by,
                "message": f"no subject carries covariate {group_by!r}."}

    # Per-subject observed exposure + design, kept per group. A subject needs at
    # least two points inside its window for a trapezoidal AUC.
    groups: dict[str, list[dict]] = {}
    multiple_dose = False
    for label in partition:
        recs = []
        for s in partition[label]:
            obs_t = np.asarray(s.get("obs_t", []), dtype=float)
            obs_c = np.asarray(s.get("obs_c", []), dtype=float)
            if obs_t.size < 2 or obs_t.size != obs_c.size:
                continue
            doses = list(s.get("doses", []))
            multiple_dose = multiple_dose or len(doses) > 1
            mask = _exposure_window_mask(obs_t, doses)
            if int(np.sum(mask)) < 2:
                continue
            recs.append({
                "doses": doses, "wt": float(s.get("wt", wt_default)),
                "obs_t": obs_t, "mask": mask,
                "auc_obs": _trapz_partial_auc(obs_t[mask], obs_c[mask]),
                "cmax_obs": float(np.max(np.maximum(obs_c[mask], 0.0))),
            })
        if recs:
            groups[label] = recs

    if not groups:
        return {"status": "empty", "group_by": group_by, "kind": kind, "groups": []}

    labels = [lb for lb in _sorted_labels(groups) if len(groups[lb]) >= min_subjects]
    skipped = [{"label": lb, "n": len(groups[lb]),
                "reason": f"fewer than {min_subjects} subjects"}
               for lb in _sorted_labels(groups) if len(groups[lb]) < min_subjects]
    if not labels:
        return {"status": "no_groups", "group_by": group_by, "kind": kind,
                "groups": [], "skipped": skipped, "multiple_dose": multiple_dose}

    # Simulate n_sim replicates; accumulate each group's mean AUC/Cmax per replicate.
    rng = np.random.default_rng(seed)
    sim_auc = {lb: np.empty(int(n_sim)) for lb in labels}
    sim_cmax = {lb: np.empty(int(n_sim)) for lb in labels}
    for r in range(int(n_sim)):
        for lb in labels:
            recs = groups[lb]
            a = np.empty(len(recs))
            cm = np.empty(len(recs))
            for j, rec in enumerate(recs):
                params_i = {p: float(typical_params[p]) *
                            float(np.exp(rng.normal(0.0, sds[p]) if sds[p] > 0 else 0.0))
                            for p in param_names}
                cp = np.asarray(simulate(model, params_i, rec["doses"], rec["obs_t"],
                                         wt=rec["wt"])["cp"], dtype=float)
                var = sigma_add ** 2 + (sigma_prop * cp) ** 2
                ysim = cp + np.sqrt(np.maximum(var, 0.0)) * rng.normal(0.0, 1.0, cp.size)
                m = rec["mask"]
                a[j] = _trapz_partial_auc(rec["obs_t"][m], ysim[m])
                cm[j] = float(np.max(np.maximum(ysim[m], 0.0)))
            sim_auc[lb][r] = float(np.mean(a))
            sim_cmax[lb][r] = float(np.mean(cm))

    def _summ(arr: np.ndarray, obs_val: float) -> dict:
        lo, hi = (float(v) for v in np.percentile(arr, _EXP_CI))
        counts, edges = np.histogram(arr, bins=int(n_hist_bins))
        return {"observed": round(obs_val, _ROUND_DP),
                "sim_median": round(float(np.median(arr)), _ROUND_DP),
                "sim_lo": round(lo, _ROUND_DP), "sim_hi": round(hi, _ROUND_DP),
                "within": bool(lo <= obs_val <= hi),
                "hist": {"edges": [round(float(e), _ROUND_DP) for e in edges],
                         "counts": [int(c) for c in counts]}}

    out = []
    for lb in labels:
        recs = groups[lb]
        out.append({
            "label": lb, "n": len(recs),
            "auc": _summ(sim_auc[lb], float(np.mean([r["auc_obs"] for r in recs]))),
            "cmax": _summ(sim_cmax[lb], float(np.mean([r["cmax_obs"] for r in recs]))),
        })
    return {"status": "ok", "group_by": group_by, "kind": kind, "n_sim": int(n_sim),
            "ci": list(_EXP_CI), "groups": out, "skipped": skipped,
            "multiple_dose": multiple_dose}


# Simulated fraction-BLQ band reported by the BLQ-incidence predictive check.
_BLQ_BAND = (5.0, 95.0)


def blq_predictive_check(model_key: str, subjects: list[dict], typical_params: dict,
                         iiv_cv_by_param: dict, *, lloq: float | None,
                         sigma_prop: float = 0.0, sigma_add: float = 0.0,
                         n_bins: int = 8, n_sim: int = 200, seed: int = 20250614,
                         wt_default: float = 70.0, x_by: str = "time") -> dict:
    """Fraction-below-LLOQ predictive check (Bergstrand & Karlsson 2009).

    Checks whether the model reproduces the fraction of observations that fall
    below the LLOQ over time — the categorical companion to the concentration
    VPC when a dataset carries censored (BLQ) data. Per bin the observed
    fraction is the share of observations flagged censored (``obs_blq``); the
    simulated fraction is the share of simulated concentrations below ``lloq``,
    reported as the simulated median and 5-95% band across ``n_sim`` replicates.
    Simulated values are NOT floored before the comparison — an additive-error
    draw below the LLOQ (including a negative one) is a genuine BLQ event.

    Requires a known ``lloq`` and per-subject ``obs_blq`` flags (build subjects
    with ``with_blq=True``). Returns ``{"status": "no_lloq"}`` without a LLOQ and
    ``{"status": "no_blq"}`` when no observation is censored (nothing to check).
    """
    if x_by.lower() not in ("time", "tad"):
        raise ValueError(f"x_by must be 'time'|'tad'; got {x_by!r}")
    x_by = x_by.lower()
    if lloq is None or not np.isfinite(lloq) or lloq <= 0:
        return {"status": "no_lloq",
                "message": ("BLQ-incidence VPC needs a positive LLOQ (a LLOQ column, or "
                            "BLQ rows carrying the LLOQ in DV).")}

    model = get_model(model_key)
    param_names = model.params
    sds = {p: _cv_pct_to_sd(iiv_cv_by_param.get(p)) for p in param_names}

    rows: list[list[float]] = []          # [x, is_blq]
    subj_rec: list[dict] = []
    n_blq_total = 0
    for s in subjects:
        obs_t = np.asarray(s.get("obs_t", []), dtype=float)
        if obs_t.size == 0:
            continue
        blq_list = s.get("obs_blq")
        blq = (np.asarray(blq_list, dtype=bool) if blq_list is not None
               else np.zeros(obs_t.shape, dtype=bool))
        if blq.size != obs_t.size:
            continue
        doses = list(s.get("doses", []))
        wt = float(s.get("wt", wt_default))
        if x_by == "tad":
            tad = time_after_dose(list(obs_t), doses)
            xv = np.array([t if t is not None else np.nan for t in tad], dtype=float)
        else:
            xv = obs_t
        start = len(rows)
        subj_rec.append({"doses": doses, "wt": wt, "obs_t": obs_t,
                         "idx": list(range(start, start + obs_t.size))})
        for k in range(obs_t.size):
            rows.append([float(xv[k]), 1.0 if bool(blq[k]) else 0.0])
            n_blq_total += int(bool(blq[k]))

    if not rows:
        return {"status": "empty", "n_bins": 0, "n_sim": int(n_sim),
                "lloq": round(float(lloq), _ROUND_DP), "x_by": x_by, "bins": []}
    if n_blq_total == 0:
        return {"status": "no_blq", "lloq": round(float(lloq), _ROUND_DP),
                "message": "no observation is below the LLOQ; nothing to check."}

    arr = np.array(rows, dtype=float)
    x_all, blq_all = arr[:, 0], arr[:, 1] > 0.5
    valid = np.isfinite(x_all)
    if not np.any(valid):
        return {"status": "empty", "n_bins": 0, "n_sim": int(n_sim),
                "lloq": round(float(lloq), _ROUND_DP), "x_by": x_by, "bins": []}

    xv = x_all[valid]
    edges = np.unique(np.quantile(xv, np.linspace(0.0, 1.0, int(n_bins) + 1)))
    nb = max(len(edges) - 1, 1)
    bin_idx = np.clip(np.digitize(x_all, edges[1:-1]), 0, nb - 1)

    bin_x = np.zeros(nb)
    bin_n = np.zeros(nb, dtype=int)
    obs_frac = np.full(nb, np.nan)
    for b in range(nb):
        m = valid & (bin_idx == b)
        if np.any(m):
            bin_x[b] = float(np.median(x_all[m]))
            bin_n[b] = int(np.sum(m))
            obs_frac[b] = float(np.mean(blq_all[m]))

    # n_sim replicates: per bin, the simulated fraction below the LLOQ.
    rng = np.random.default_rng(seed)
    rep_frac = np.full((int(n_sim), nb), np.nan)
    for r in range(int(n_sim)):
        sim_below = np.zeros(len(rows), dtype=bool)
        for s in subj_rec:
            params_i = {p: float(typical_params[p]) *
                        float(np.exp(rng.normal(0.0, sds[p]) if sds[p] > 0 else 0.0))
                        for p in param_names}
            cp = np.asarray(simulate(model, params_i, s["doses"], s["obs_t"],
                                     wt=s["wt"])["cp"], dtype=float)
            var = sigma_add ** 2 + (sigma_prop * cp) ** 2
            ysim = cp + np.sqrt(np.maximum(var, 0.0)) * rng.normal(0.0, 1.0, cp.size)
            for k, gidx in enumerate(s["idx"]):
                sim_below[gidx] = ysim[k] < lloq
        for b in range(nb):
            m = valid & (bin_idx == b)
            if np.any(m):
                rep_frac[r, b] = float(np.mean(sim_below[m]))

    sim_med = np.nanmedian(rep_frac, axis=0)
    sim_lo = np.nanpercentile(rep_frac, _BLQ_BAND[0], axis=0)
    sim_hi = np.nanpercentile(rep_frac, _BLQ_BAND[1], axis=0)

    def _r(x: float) -> float | None:
        return None if not np.isfinite(x) else round(float(x), _ROUND_DP)

    bins = [{"x": _r(bin_x[b]), "n": int(bin_n[b]), "obs_frac": _r(obs_frac[b]),
             "sim_med": _r(sim_med[b]), "sim_lo": _r(sim_lo[b]), "sim_hi": _r(sim_hi[b])}
            for b in range(nb)]
    return {"status": "ok", "n_bins": nb, "n_sim": int(n_sim),
            "lloq": round(float(lloq), _ROUND_DP), "x_by": x_by,
            "n_blq": int(n_blq_total), "bins": bins}
