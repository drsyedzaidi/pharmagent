"""Clinical trial simulation and probability of target attainment (PTA).

Week-12 "Simulation" of the IU PopPK course turns a fitted population model into
a dosing recommendation: sample a virtual population (covariates resampled from
the analysis dataset, between-subject variability drawn from Omega), simulate a
dosing regimen for every subject across a grid of dose levels, and report the
fraction of subjects whose exposure meets a clinical target — the probability of
target attainment. The recommended dose is the smallest dose reaching the target
fraction for an efficacy criterion (metric ABOVE a threshold), or the largest
dose still meeting it for a safety criterion (metric BELOW a threshold).

Pure, deterministic-given-seed compute on top of the shared simulator
(:func:`app.compute.pk_simulate.simulate_timecourse`) and the exposure-metric
extractor (:func:`app.compute.dose_sweep._interval_metrics`). No file I/O, no
network, no global mutable state.

Design notes matching the course lab (Exercise 2, ``zero_re(sigma)``):

  * Target attainment is judged on each subject's INDIVIDUAL exposure (IPRED) —
    residual/assay error is NOT added, since PTA concerns the patient's true
    exposure, not a noisy measurement.
  * Covariates are resampled as whole rows from the fitted subjects, preserving
    the real covariate correlation structure, rather than sampling marginals.
  * Per-subject typical values apply the fitted covariate model first, then IIV
    (``p_i = theta * cov_factor * exp(eta)``); WT allometry is applied by the
    simulator, so a fitted WT covariate is intentionally never added (it would
    double-count the built-in allometric scaling).
"""
from __future__ import annotations

import numpy as np

from app.compute.dose_sweep import _interval_metrics
from app.compute.pk_models import get_model
from app.compute.pk_simulate import simulate_timecourse

_METRICS = ("cmax", "auc_tau", "cavg", "ctrough")
_REPORT_PCTL = (5.0, 25.0, 50.0, 75.0, 95.0)


def _cv_pct_to_omega2(cv_pct: float | None) -> float:
    """Lognormal %CV -> variance omega2 = ln(1 + (cv/100)^2)."""
    if cv_pct is None:
        return 0.0
    cv = float(cv_pct)
    return float(np.log1p((cv / 100.0) ** 2)) if cv > 0 else 0.0


def _covariate_applier(covariate_effects: list[dict] | None):
    """Return f(theta, cov) -> covariate-adjusted typical values.

    Reuses the fitted covariate model (``nlme._CovEffect``) so the multiplicative
    factors are identical to what the estimator applied — no reimplementation to
    drift from. Returns the identity when there is no covariate model.
    """
    if not covariate_effects:
        return lambda theta, cov: theta
    from app.compute.nlme import _cov_effects_from_records  # lazy: heavy (scipy)

    effects, coefs = _cov_effects_from_records(covariate_effects)

    def apply(theta: dict[str, float], cov: dict) -> dict[str, float]:
        p = dict(theta)
        i = 0
        for eff in effects:
            c = coefs[i:i + eff.n_coef]
            if eff.param in p:
                p[eff.param] = p[eff.param] * eff.factor(c, (cov or {}).get(eff.covariate))
            i += eff.n_coef
        return p

    return apply


def _sample_population(cov_rows: list[dict] | None, wt_rows: list[float] | None,
                       n_subjects: int, rng: np.random.Generator,
                       wt_default: float) -> list[dict]:
    """Resample ``n_subjects`` virtual patients from the observed covariate/WT
    rows (whole-row bootstrap, preserving covariate correlations). Without a
    source, every subject gets the default weight and no covariates."""
    rows = cov_rows or []
    weights = wt_rows or []
    if not rows and not weights:
        return [{"wt": wt_default, "cov": {}} for _ in range(n_subjects)]
    n_src = max(len(rows), len(weights))
    idx = rng.integers(0, n_src, size=n_subjects)
    pop = []
    for j in idx:
        cov = dict(rows[j]) if j < len(rows) else {}
        wt = float(weights[j]) if j < len(weights) else float(cov.get("WT", wt_default))
        pop.append({"wt": wt, "cov": cov})
    return pop


def sample_theta_draws(theta: dict, theta_rse_pct: dict | None, n_draws: int,
                       seed: int = 20250614) -> list[dict]:
    """``n_draws`` lognormal parameter sets ``theta[p] * exp(N(0, rse_p/100))``
    for structural params carrying a positive RSE% (others held fixed). Feeds the
    parameter-uncertainty PTA band / sensitivity analysis. Empty when no RSE is
    available (the caller then reports a point estimate only)."""
    rse = {p: float(v) for p, v in (theta_rse_pct or {}).items()
           if v is not None and float(v) > 0 and p in theta}
    if not rse:
        return []
    rng = np.random.default_rng(seed)
    return [{p: float(theta[p]) * float(np.exp(rng.normal(0.0, rse[p] / 100.0)))
             for p in rse} for _ in range(int(max(1, n_draws)))]


# KDIGO-style renal-function categories by eGFR (mL/min/1.73m^2). The course lab
# uses findInterval(EGFR, c(15,30,60,90,Inf)); here G4 (<30) and G5 (<15) are
# folded into a single "Severe" bucket, giving 4 categories: <30 Severe,
# [30,60) Moderate, [60,90) Mild, >=90 Normal (right-open, via np.digitize).
_RENAL_EDGES = (30.0, 60.0, 90.0)
_RENAL_LABELS = ("Severe", "Moderate", "Mild", "Normal")
_RENAL_KEYS = {"RF", "EGFR", "RENAL", "CRCL", "CLCR", "GFR", "EGFR_CKD"}
_MIN_CONTINUOUS_LEVELS = 5

# Pediatric age categories (years), matching the course lab's
# findInterval(AGE, c(2,6,12,18)) -> 2-<6 / 6-<12 / 12-<18. Restricted to >=2 y,
# where clearance maturation is essentially complete so weight allometry alone
# extrapolates (the model carries no maturation function).
_PED_AGE_EDGES = (6.0, 12.0)
_PED_AGE_LABELS = ("2 to <6 y", "6 to <12 y", "12 to <18 y")
_PED_AGE_RANGE = (2.0, 18.0)
# Age-group-specific weight bins (kg) from the lab's NHANES weight distributions.
_PED_WT_RANGE = {"2 to <6 y": (12.0, 25.0), "6 to <12 y": (20.0, 60.0),
                 "12 to <18 y": (40.0, 100.0)}
_PED_WT_EDGES = {"2 to <6 y": (18.0,), "6 to <12 y": (40.0,), "12 to <18 y": (70.0,)}
_PED_WT_LABELS = {"2 to <6 y": ("12 to <18 kg", "18 to <25 kg"),
                  "6 to <12 y": ("20 to <40 kg", "40 to <60 kg"),
                  "12 to <18 y": ("40 to <70 kg", "70 to <100 kg")}


def _scale_wt(params: dict, wt: float, wt_exponents: dict | None) -> tuple[dict, float]:
    """Apply a MODEL-ESTIMATED allometric exponent instead of the simulator's
    built-in fixed 0.75/1.0.

    When ``wt_exponents`` (e.g. ``{"CL": 0.663, "VC": 1.087}``) is given, scale
    each param by ``(wt/70)^exp`` here and return ``sim_wt = 70`` so the
    simulator's built-in fixed allometry becomes a no-op (factor 1) — the effect
    is applied exactly once. Otherwise return the params unchanged and
    ``sim_wt = wt`` (built-in fixed allometry, the default byte-for-byte path).
    Weight allometry and IIV are both multiplicative exponentials, so applying
    this before or after the eta multiply gives the same result. The caller must
    NOT also carry a WT covariate in ``covariate_effects`` (that would double).
    """
    if not wt_exponents:
        return params, wt
    f = float(wt) / 70.0
    scaled = dict(params)
    for p, expo in wt_exponents.items():
        if p in scaled and expo:
            scaled[p] = scaled[p] * (f ** float(expo))
    return scaled, 70.0


def _num(v) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _renal_label(egfr: float) -> str:
    """eGFR -> Severe/Moderate/Mild/Normal (KDIGO-ish, per the course lab)."""
    return _RENAL_LABELS[int(np.digitize(egfr, _RENAL_EDGES))]


def _sorted_labels(labels) -> list:
    """Numeric-aware label order (25 before 100, Q1 before Q10)."""
    def _isnum(s) -> bool:
        try:
            float(s)
            return True
        except (TypeError, ValueError):
            return False
    return sorted(labels, key=lambda x: (0, float(x)) if _isnum(x) else (1, str(x)))


def _stratify_source(cov_rows: list[dict], wt_rows: list[float], stratify_by: str
                     ) -> tuple[dict[str, list[int]], str]:
    """Group source-row indices into strata by ``stratify_by``.

    Returns ``(partition, kind)`` with ``kind`` in ``renal | categorical |
    quartile | missing``. A renal covariate (eGFR/CrCl/RF) is binned into the
    KDIGO categories; a categorical covariate splits per level; a continuous
    covariate splits into quartiles Q1..Q4.
    """
    key = stratify_by.strip()
    upper = key.upper()
    present = [(i, (cov_rows[i] or {}).get(key)) for i in range(len(cov_rows))]
    present = [(i, v) for i, v in present if v is not None]
    if not present:
        return {}, "missing"

    if upper in _RENAL_KEYS:
        nums = [(i, _num(v)) for i, v in present]
        nums = [(i, v) for i, v in nums if v is not None]
        if not nums:
            return {}, "missing"
        parts: dict[str, list[int]] = {}
        for i, v in nums:
            parts.setdefault(_renal_label(v), []).append(i)
        return parts, "renal"

    vals = [v for _i, v in present]
    numeric = [_num(v) for v in vals]
    is_continuous = (all(n is not None for n in numeric)
                     and len({round(float(n), 6) for n in numeric}) >= _MIN_CONTINUOUS_LEVELS)
    parts = {}
    if is_continuous:
        arr = np.array([float(n) for n in numeric], dtype=float)
        q = np.quantile(arr, [0.25, 0.5, 0.75])
        for (i, _v), n in zip(present, numeric):
            parts.setdefault(f"Q{int(np.digitize(float(n), q)) + 1}", []).append(i)
        return parts, "quartile"
    for i, v in present:
        parts.setdefault(str(v), []).append(i)
    return parts, "categorical"


def _sample_indices(idx_pool: list[int], n: int, rng: np.random.Generator) -> list[int]:
    """Resample n indices from the pool with replacement (empty pool -> empty)."""
    if not idx_pool:
        return []
    return [idx_pool[j] for j in rng.integers(0, len(idx_pool), size=int(n))]


def special_population_simulation(
    model_key: str, *, theta: dict, omega_cv_pct: dict, iiv_params: list[str],
    cov_rows: list[dict], wt_rows: list[float], stratify_by: str, doses: list[float],
    tau: float, n_doses: int, covariate_effects: list[dict] | None = None,
    metrics: tuple[str, ...] = ("auc_tau", "cmax"), reference_stratum: str = "Normal",
    reference_dose: float | None = None, n_per_stratum: int = 600,
    seed: int = 20250614, wt_default: float = 70.0, n_points: int = 160,
    max_per_stratum: int = 2000, max_doses: int = 12,
) -> dict:
    """Special-population exposure simulation (Week-13 ``renal-simulations.R``).

    Stratify the virtual-population source by a categorized covariate (renal
    function from eGFR, or quartiles of a continuous covariate), sample
    ``n_per_stratum`` subjects per stratum, simulate a steady-state regimen at
    every dose, and report the exposure distribution (``metrics`` over the last
    interval — AUC_tau ≈ AUCss, cmax ≈ Cmax,ss). The ``reference_stratum`` at
    ``reference_dose`` gives the 5-95% comparison band; each stratum-dose median
    is flagged within/above/below it, and each stratum gets the dose whose median
    exposure lands back inside the reference band — the special-population dose
    adjustment.

    Returns ``{status, model_key, label, stratify_by, kind, metrics,
    reference_stratum, reference_dose, tau, n_doses, n_per_stratum,
    reference_band: {metric: {lo, hi, median}}, strata: [{label, n, doses:
    [{dose, metric: {p05,p25,p50,p75,p95, within_ref}}], recommended_dose,
    note}], skipped}``.
    """
    bad = [m for m in metrics if m not in _METRICS]
    if bad:
        raise ValueError(f"metrics must be in {_METRICS}; got {bad}")
    dose_grid = sorted({float(d) for d in doses if float(d) > 0})
    if not dose_grid:
        return {"status": "no_doses", "message": "no positive dose levels supplied."}
    if len(dose_grid) > max_doses:
        return {"status": "too_many_doses",
                "message": f"dose grid capped at {max_doses}; got {len(dose_grid)}."}
    if not cov_rows:
        return {"status": "no_covariates",
                "message": "special-population simulation needs the dataset covariates."}

    partition, kind = _stratify_source(cov_rows, wt_rows, stratify_by)
    if kind == "missing":
        return {"status": "missing_covariate", "stratify_by": stratify_by,
                "message": f"no subject carries covariate {stratify_by!r}."}

    n_per = int(max(1, min(n_per_stratum, max_per_stratum)))
    tau = float(tau)
    n_doses = int(max(1, n_doses))
    tmax = tau * n_doses
    t_last = (n_doses - 1) * tau
    ref_dose = float(reference_dose) if reference_dose is not None else dose_grid[len(dose_grid) // 2]

    model = get_model(model_key)
    typical = {**model.defaults, **{k: float(v) for k, v in theta.items()}}
    apply_cov = _covariate_applier(covariate_effects)
    sds = {p: float(np.sqrt(_cv_pct_to_omega2(omega_cv_pct.get(p)))) for p in iiv_params}
    rng = np.random.default_rng(seed)

    def _exposure(dose: float, si: int, eta: dict) -> dict:
        cov = cov_rows[si] or {}
        wt = float(wt_rows[si]) if si < len(wt_rows) else float(cov.get("WT", wt_default))
        theta_i = apply_cov(typical, cov)
        params_i = {k: float(theta_i[k]) for k in theta_i}
        for p in iiv_params:
            if p in params_i:
                params_i[p] = params_i[p] * float(np.exp(eta[p]))
        sim = simulate_timecourse(model, params_i, dose=dose, tau=tau, n_doses=n_doses,
                                  tmax=tmax, n_points=n_points, wt=wt)
        return _interval_metrics(sim["times"], sim["cp"], t_last=t_last, tau=tau, tmax=tmax)

    def _dist(pool: list[int], dose: float) -> dict:
        """Sampled exposure metrics for one (stratum, dose)."""
        picks = _sample_indices(pool, n_per, rng)
        vals = {m: np.empty(len(picks)) for m in metrics}
        for k, si in enumerate(picks):
            eta = {p: float(rng.normal(0.0, sds[p])) if sds[p] > 0 else 0.0 for p in iiv_params}
            m = _exposure(dose, si, eta)
            for met in metrics:
                vals[met][k] = m[met]
        return {met: vals[met][np.isfinite(vals[met])] for met in metrics}

    labels = _sorted_renal(partition) if kind == "renal" else _sorted_labels(partition)
    skipped = [{"label": lb, "n": len(partition[lb])} for lb in labels if not partition[lb]]
    labels = [lb for lb in labels if partition[lb]]
    if not labels:
        return {"status": "no_strata", "stratify_by": stratify_by, "kind": kind,
                "strata": [], "skipped": skipped}

    # Reference band: the reference stratum at the reference dose (fall back to the
    # first stratum if the named reference is absent, e.g. no "Normal" subjects).
    ref_label = reference_stratum if reference_stratum in partition and partition[reference_stratum] else labels[0]
    ref_dist = _dist(partition[ref_label], ref_dose)
    reference_band = {}
    for met in metrics:
        a = ref_dist[met]
        if a.size:
            lo, med, hi = np.percentile(a, [5.0, 50.0, 95.0])
            reference_band[met] = {"lo": _r(lo), "hi": _r(hi), "median": _r(med)}
        else:
            reference_band[met] = {"lo": None, "hi": None, "median": None}

    out_strata = []
    for lb in labels:
        pool = partition[lb]
        dose_rows = []
        for d in dose_grid:
            dist = _dist(pool, d)
            row = {"dose": round(d, 6)}
            for met in metrics:
                a = dist[met]
                if a.size:
                    p05, p25, p50, p75, p95 = np.percentile(a, _REPORT_PCTL)
                    band = reference_band[met]
                    within = (band["lo"] is not None
                              and band["lo"] <= float(p50) <= band["hi"])
                    row[met] = {"p05": _r(p05), "p25": _r(p25), "p50": _r(p50),
                                "p75": _r(p75), "p95": _r(p95), "within_ref": bool(within)}
                else:
                    row[met] = {"p05": None, "p25": None, "p50": None, "p75": None,
                                "p95": None, "within_ref": False}
            dose_rows.append(row)
        rec_dose, note = _recommend_special(dose_rows, reference_band, metrics[0], lb, ref_label)
        out_strata.append({"label": lb, "n": len(pool), "doses": dose_rows,
                           "recommended_dose": rec_dose, "note": note})

    return {
        "status": "ok", "model_key": model_key, "label": model.label,
        "stratify_by": stratify_by, "kind": kind, "metrics": list(metrics),
        "reference_stratum": ref_label, "reference_dose": round(ref_dose, 6),
        "tau": tau, "n_doses": n_doses, "n_per_stratum": n_per,
        "reference_band": reference_band, "strata": out_strata, "skipped": skipped,
    }


def reference_population(n: int = 4000, seed: int = 20250614
                         ) -> tuple[list[dict], list[float]]:
    """A representative adult covariate distribution spanning renal-function
    categories — a stand-in reference population when the analysis dataset lacks
    renal-impaired subjects (Week-13 supplements sparse severe-RI numbers from an
    external source). AGE ~ U(18,85), SEX 50/50, WT ~ N(80,18) kg, serum
    creatinine lognormal, and eGFR from the MDRD equation (mL/min/1.73m^2):
    ``175 * SCr^-1.154 * AGE^-0.203 * 0.742^(female)``.

    NOTE: SYNTHETIC / representative, not literal NHANES (the course lab pulls
    NHANES over the network via ``nhanesA``; PharmAgent bundles no external data
    and does no runtime fetch). Returns ``(cov_rows, wt_rows)`` matching the
    :func:`special_population_simulation` input contract.
    """
    rng = np.random.default_rng(seed)
    n = int(max(1, n))
    age = rng.uniform(18.0, 85.0, n)
    sex = rng.integers(0, 2, n)                       # 0 male, 1 female
    wt = np.clip(rng.normal(80.0, 18.0, n), 40.0, 160.0)
    scr = np.clip(rng.lognormal(np.log(0.9), 0.5, n), 0.4, 8.0)   # mg/dL
    egfr = 175.0 * scr ** -1.154 * age ** -0.203 * np.where(sex == 1, 0.742, 1.0)
    egfr = np.clip(egfr, 5.0, 150.0)
    # SEX as float (not int): the dataset covariate pipeline casts numerics to
    # float, so a fitted categorical covariate stores its levels as str(1.0)="1.0".
    # Emitting int here would str()-match as "1" and silently drop the effect
    # (categorical matching in nlme._CovEffect.factor is by string equality).
    cov_rows = [{"AGE": round(float(age[i]), 1), "SEX": float(sex[i]),
                 "WT": round(float(wt[i]), 1), "SCR": round(float(scr[i]), 2),
                 "EGFR": round(float(egfr[i]), 1)} for i in range(n)]
    return cov_rows, [round(float(w), 1) for w in wt]


def individual_exposures(
    model_key: str, *, theta: dict, subjects: list[dict], etas: dict,
    dose: float, tau: float, n_doses: int, covariate_effects: list[dict] | None = None,
    iiv_params: list[str] | None = None, metrics: tuple[str, ...] = ("auc_tau", "cmax"),
    wt_default: float = 70.0, n_points: int = 160, group_key: str | None = None,
    wt_exponents: dict | None = None,
) -> dict:
    """Per-subject steady-state exposure from EBEs (Week-13 ``individual-exposures.R``).

    For each fitted subject, individual parameters are the covariate-adjusted
    typical values times ``exp(eta_i)`` (the subject's stored empirical-Bayes
    estimate); a steady-state regimen is simulated and AUCss (``auc_tau``) and
    Cmax,ss (``cmax``) over the last interval are reported. This is the reference
    exposure table the special-population simulation compares against.

    ``etas``: ``{subject_id: {param: eta}}``; a subject without a stored eta uses
    eta = 0. ``group_key`` (e.g. a renal-function column) adds a per-group summary.
    Returns ``{status, model_key, label, dose, tau, n_doses, metrics,
    subjects: [{subject, group?, auc_ss, cmax_ss, ...}], groups?}``.
    """
    bad = [m for m in metrics if m not in _METRICS]
    if bad:
        raise ValueError(f"metrics must be in {_METRICS}; got {bad}")
    if not subjects:
        return {"status": "empty", "message": "no fitted subjects."}
    model = get_model(model_key)
    typical = {**model.defaults, **{k: float(v) for k, v in theta.items()}}
    apply_cov = _covariate_applier(covariate_effects)
    iiv_params = iiv_params or []
    tmax = float(tau) * int(max(1, n_doses))
    t_last = (int(max(1, n_doses)) - 1) * float(tau)

    recs = []
    for s in subjects:
        sid = s.get("subject")
        cov = s.get("cov") or {}
        wt = float(s.get("wt", wt_default))
        eta = etas.get(sid) or etas.get(str(sid)) or {}
        theta_i = apply_cov(typical, cov)
        params_i = {k: float(theta_i[k]) for k in theta_i}
        for p in iiv_params:
            if p in params_i:
                params_i[p] = params_i[p] * float(np.exp(float(eta.get(p, 0.0))))
        params_i, sim_wt = _scale_wt(params_i, wt, wt_exponents)
        sim = simulate_timecourse(model, params_i, dose=float(dose), tau=float(tau),
                                  n_doses=int(max(1, n_doses)), tmax=tmax,
                                  n_points=n_points, wt=sim_wt)
        m = _interval_metrics(sim["times"], sim["cp"], t_last=t_last, tau=float(tau), tmax=tmax)
        rec = {"subject": str(sid), "auc_ss": _r(m["auc_tau"]), "cmax_ss": _r(m["cmax"])}
        if group_key is not None:
            gv = cov.get(group_key)
            # Bin a renal covariate into KDIGO categories so the group summary is
            # meaningful (raw eGFR would otherwise make one group per subject).
            if group_key.upper() in _RENAL_KEYS and _num(gv) is not None:
                rec["group"] = _renal_label(float(gv))
            else:
                rec["group"] = gv
        recs.append(rec)

    out = {"status": "ok", "model_key": model_key, "label": model.label,
           "dose": round(float(dose), 6), "tau": float(tau), "n_doses": int(max(1, n_doses)),
           "metrics": list(metrics), "subjects": recs}
    if group_key is not None:
        groups: dict[str, list[dict]] = {}
        for r in recs:
            groups.setdefault(str(r.get("group")), []).append(r)
        summary = []
        for g, rs in groups.items():
            gsum = {"group": g, "n": len(rs)}
            for met, fld in (("auc_ss", "auc_ss"), ("cmax_ss", "cmax_ss")):
                arr = np.array([r[fld] for r in rs if r[fld] is not None], dtype=float)
                if arr.size:
                    p05, p50, p95 = np.percentile(arr, [5.0, 50.0, 95.0])
                    gsum[met] = {"p05": _r(p05), "median": _r(p50), "p95": _r(p95)}
            summary.append(gsum)
        out["groups"] = summary
    return out


def _sorted_renal(partition) -> list:
    """Severe → Normal order for renal strata (others appended)."""
    order = {lab: i for i, lab in enumerate(_RENAL_LABELS)}
    return sorted(partition, key=lambda x: (order.get(x, 99), x))


def _recommend_special(dose_rows: list[dict], reference_band: dict, metric: str,
                       label: str, ref_label: str) -> tuple[float | None, str]:
    """The dose whose median ``metric`` lands inside the reference band — the
    special-population dose that normalizes exposure to the reference group."""
    band = reference_band.get(metric) or {}
    if band.get("lo") is None:
        return None, "no reference band available."
    if label == ref_label:
        return None, "reference stratum."
    within = [r for r in dose_rows if r[metric].get("within_ref")]
    if within:
        best = min(within, key=lambda r: abs((r[metric]["p50"] or 0) - (band["median"] or 0)))
        return best["dose"], (f"dose {best['dose']:g} brings median {metric} into the "
                              f"{ref_label} reference range.")
    # None land inside — say which side the exposure sits on at every dose.
    meds = [r[metric]["p50"] for r in dose_rows if r[metric]["p50"] is not None]
    if meds and all(m > band["hi"] for m in meds):
        return None, f"exposure above the {ref_label} range at all doses — consider a lower dose."
    if meds and all(m < band["lo"] for m in meds):
        return None, f"exposure below the {ref_label} range at all doses — consider a higher dose."
    return None, f"no simulated dose matches the {ref_label} reference range."


def _pediatric_strata(cov_rows: list[dict], age_key: str, wt_key: str
                      ) -> tuple[dict[str, list[int]], dict[str, tuple[str, str]], list[str]]:
    """Cross the pediatric age bins (2-<6 / 6-<12 / 12-<18 y) with each age
    group's weight bins into ordered composite strata (``age · weight``). Subjects
    outside 2-<18 y or outside their age group's weight range are dropped.
    Returns ``(partition, meta, ordered_labels)`` where meta[label]=(age, weight)."""
    parts: dict[str, list[int]] = {}
    meta: dict[str, tuple[str, str]] = {}
    for i, row in enumerate(cov_rows):
        r = row or {}
        a = _num(r.get(age_key))
        w = _num(r.get(wt_key))
        if a is None or w is None:
            continue
        if not (_PED_AGE_RANGE[0] <= a < _PED_AGE_RANGE[1]):
            continue
        age_label = _PED_AGE_LABELS[int(np.digitize(a, _PED_AGE_EDGES))]
        lo, hi = _PED_WT_RANGE[age_label]
        if not (lo <= w < hi):
            continue
        wt_label = _PED_WT_LABELS[age_label][int(np.digitize(w, _PED_WT_EDGES[age_label]))]
        label = f"{age_label} · {wt_label}"
        parts.setdefault(label, []).append(i)
        meta[label] = (age_label, wt_label)
    order = [f"{al} · {wl}" for al in _PED_AGE_LABELS for wl in _PED_WT_LABELS[al]]
    return parts, meta, [lb for lb in order if parts.get(lb)]


def _recommend_pediatric(dose_rows: list[dict], metric: str, band: dict) -> tuple[float | None, str]:
    """The dose that maximizes the % of subjects within the adult reference range
    (ties -> lowest dose) — the pediatric exposure-matching dose. Returns None when
    no dose puts ANY subject inside the band (exposure off the adult range across
    the whole grid), matching the PTA / renal recommenders."""
    cand = [(r["dose"], r[metric].get("pct_within_ref")) for r in dose_rows]
    cand = [(d, p) for d, p in cand if p is not None]
    if not cand:
        return None, "no adult reference band — cannot match exposure."
    best_p = max(p for _d, p in cand)
    if best_p <= 0.0:
        meds = [r[metric]["p50"] for r in dose_rows if r[metric].get("p50") is not None]
        hi, lo = (band or {}).get("hi"), (band or {}).get("lo")
        if meds and hi is not None and all(m > hi for m in meds):
            return None, "exposure above the adult range at every dose — no matching dose in the grid."
        if meds and lo is not None and all(m < lo for m in meds):
            return None, "exposure below the adult range at every dose — no matching dose in the grid."
        return None, "no dose puts any subject inside the adult range."
    best_d = min(d for d, p in cand if p == best_p)
    return best_d, (f"dose {best_d:g} maximizes overlap with the adult range "
                    f"({best_p:.0f}% of subjects within).")


def pediatric_reference_population(n: int = 6000, seed: int = 20250614
                                   ) -> tuple[list[dict], list[float]]:
    """A representative pediatric covariate distribution (ages 2-<18 y) with a
    realistic weight-for-age relationship — a stand-in when the analysis dataset
    has no pediatric subjects (Week-14 draws these from NHANES). AGE ~ U(2,18);
    median weight follows a smooth growth curve
    ``exp(2.247 + 0.1431*AGE - 0.001913*AGE^2)`` kg fit to CDC 50th-percentile
    weight-for-age (~12.5 kg at 2 y, ~20.8 at 6, ~40 at 12, ~62 at 17 — the
    adolescent acceleration a log-linear fit misses) with lognormal between-subject
    spread (~19% CV), clipped to 8-120 kg. EGFR fixed at 90 mL/min/1.73m^2 (normal
    renal function — the conservative pediatric assumption, matching the lab).

    NOTE: SYNTHETIC / representative, not literal NHANES (no runtime fetch). The
    lab's per-age-group weight bins still drop out-of-range subjects, so the
    lightest young tail of each band is under-represented by construction (matching
    the lab). Returns ``(cov_rows, wt_rows)`` for :func:`pediatric_simulation`.
    """
    rng = np.random.default_rng(seed)
    n = int(max(1, n))
    # Clip the rounded age to [2, 17.9] so rounding never yields exactly 18.0
    # (which would fall outside the pediatric range and be dropped downstream).
    age = np.clip(np.round(rng.uniform(_PED_AGE_RANGE[0], _PED_AGE_RANGE[1], n), 1),
                  _PED_AGE_RANGE[0], 17.9)
    median_wt = np.exp(2.247 + 0.1431 * age - 0.001913 * age ** 2)
    wt = np.clip(median_wt * np.exp(rng.normal(0.0, 0.185, n)), 8.0, 120.0)
    cov_rows = [{"AGE": round(float(age[i]), 1), "WT": round(float(wt[i]), 1),
                 "EGFR": 90.0} for i in range(n)]
    return cov_rows, [round(float(w), 1) for w in wt]


def pediatric_simulation(
    model_key: str, *, theta: dict, omega_cv_pct: dict, iiv_params: list[str],
    cov_rows: list[dict], wt_rows: list[float], reference_exposures: dict,
    doses: list[float], tau: float, n_doses: int,
    covariate_effects: list[dict] | None = None,
    metrics: tuple[str, ...] = ("auc_tau", "cmax"), age_key: str = "AGE",
    wt_key: str = "WT", n_per_stratum: int = 1000, seed: int = 20250614,
    wt_default: float = 70.0, n_points: int = 160, wt_exponents: dict | None = None,
    max_per_stratum: int = 2000, max_doses: int = 12,
) -> dict:
    """Pediatric dose-finding by exposure matching (Week-13/14 ``pediatric-simulations.R``).

    Stratify a pediatric virtual population by age x weight, simulate steady-state
    exposure at each dose, and compare each stratum's distribution to an EXTERNAL
    adult reference band (``reference_exposures`` = adult AUCss/Cmax,ss at the
    label dose, e.g. from :func:`individual_exposures` on the normal-renal adults).
    Unlike the renal special-population sim, the reference is a different
    population (adults), not an in-population stratum. Reports the % of each
    stratum's subjects whose exposure lands inside the adult 5-95% range per dose,
    and recommends the dose that maximizes that overlap.

    ``wt_exponents`` (e.g. ``{"CL": 0.663, "VC": 1.087}``) uses model-estimated
    allometric exponents instead of the built-in fixed 0.75/1.0 — the pediatric
    extrapolation hinges on the estimated weight-clearance relationship.

    Returns ``{status, model_key, label, kind:"pediatric", metrics,
    reference_metric, reference_band:{metric:{lo,hi,median,n}}, allometry, tau,
    n_doses, n_per_stratum, strata:[{label, age_label, wt_label, n, doses:[{dose,
    metric:{p05..p95, within_ref, pct_within_ref}}], recommended_dose, note}],
    skipped}``.
    """
    bad = [m for m in metrics if m not in _METRICS]
    if bad:
        raise ValueError(f"metrics must be in {_METRICS}; got {bad}")
    dose_grid = sorted({float(d) for d in doses if float(d) > 0})
    if not dose_grid:
        return {"status": "no_doses", "message": "no positive dose levels supplied."}
    if len(dose_grid) > max_doses:
        return {"status": "too_many_doses",
                "message": f"dose grid capped at {max_doses}; got {len(dose_grid)}."}
    if not cov_rows:
        return {"status": "no_covariates", "message": "pediatric simulation needs covariates."}

    # Reference band = adult exposure percentiles (external, not simulated here).
    reference_band = {}
    for m in metrics:
        a = np.asarray([v for v in (reference_exposures or {}).get(m, []) if v is not None],
                       dtype=float)
        a = a[np.isfinite(a)]
        if a.size:
            lo, med, hi = np.percentile(a, [5.0, 50.0, 95.0])
            reference_band[m] = {"lo": _r(lo), "hi": _r(hi), "median": _r(med), "n": int(a.size)}
        else:
            reference_band[m] = {"lo": None, "hi": None, "median": None, "n": 0}
    if all(b["lo"] is None for b in reference_band.values()):
        return {"status": "no_reference",
                "message": "no adult reference exposures supplied for the comparison band."}

    parts, meta, labels = _pediatric_strata(cov_rows, age_key, wt_key)
    if not labels:
        return {"status": "no_strata",
                "message": "no subjects in the pediatric age/weight ranges (2-<18 y).",
                "strata": [], "skipped": []}

    n_per = int(max(1, min(n_per_stratum, max_per_stratum)))
    tau = float(tau)
    n_doses = int(max(1, n_doses))
    tmax = tau * n_doses
    t_last = (n_doses - 1) * tau
    model = get_model(model_key)
    typical = {**model.defaults, **{k: float(v) for k, v in theta.items()}}
    apply_cov = _covariate_applier(covariate_effects)
    sds = {p: float(np.sqrt(_cv_pct_to_omega2(omega_cv_pct.get(p)))) for p in iiv_params}
    rng = np.random.default_rng(seed)

    def _exposure(dose: float, si: int, eta: dict) -> dict:
        cov = cov_rows[si] or {}
        wt = float(wt_rows[si]) if si < len(wt_rows) else float(cov.get(wt_key, wt_default))
        theta_i = apply_cov(typical, cov)
        params_i = {k: float(theta_i[k]) for k in theta_i}
        for p in iiv_params:
            if p in params_i:
                params_i[p] = params_i[p] * float(np.exp(eta[p]))
        params_i, sim_wt = _scale_wt(params_i, wt, wt_exponents)
        sim = simulate_timecourse(model, params_i, dose=dose, tau=tau, n_doses=n_doses,
                                  tmax=tmax, n_points=n_points, wt=sim_wt)
        return _interval_metrics(sim["times"], sim["cp"], t_last=t_last, tau=tau, tmax=tmax)

    def _dist(pool: list[int], dose: float) -> dict:
        picks = _sample_indices(pool, n_per, rng)
        vals = {m: np.empty(len(picks)) for m in metrics}
        for k, si in enumerate(picks):
            eta = {p: float(rng.normal(0.0, sds[p])) if sds[p] > 0 else 0.0 for p in iiv_params}
            mm = _exposure(dose, si, eta)
            for m in metrics:
                vals[m][k] = mm[m]
        return {m: vals[m][np.isfinite(vals[m])] for m in metrics}

    out_strata = []
    for lb in labels:
        pool = parts[lb]
        age_label, wt_label = meta[lb]
        dose_rows = []
        for d in dose_grid:
            dist = _dist(pool, d)
            row = {"dose": round(d, 6)}
            for m in metrics:
                a = dist[m]
                band = reference_band[m]
                if a.size:
                    p05, p25, p50, p75, p95 = np.percentile(a, _REPORT_PCTL)
                    if band["lo"] is not None:
                        pct = 100.0 * float(np.mean((a >= band["lo"]) & (a <= band["hi"])))
                        within = bool(band["lo"] <= float(p50) <= band["hi"])
                    else:
                        pct, within = None, False
                    row[m] = {"p05": _r(p05), "p25": _r(p25), "p50": _r(p50), "p75": _r(p75),
                              "p95": _r(p95), "within_ref": within,
                              "pct_within_ref": _r(pct) if pct is not None else None}
                else:
                    row[m] = {"p05": None, "p25": None, "p50": None, "p75": None, "p95": None,
                              "within_ref": False, "pct_within_ref": None}
            dose_rows.append(row)
        rec, note = _recommend_pediatric(dose_rows, metrics[0], reference_band.get(metrics[0], {}))
        out_strata.append({"label": lb, "age_label": age_label, "wt_label": wt_label,
                           "n": len(pool), "doses": dose_rows, "recommended_dose": rec, "note": note})

    return {
        "status": "ok", "model_key": model_key, "label": model.label, "kind": "pediatric",
        "metrics": list(metrics), "reference_metric": metrics[0], "reference_band": reference_band,
        "allometry": "estimated" if wt_exponents else "fixed", "tau": tau, "n_doses": n_doses,
        "n_per_stratum": n_per, "strata": out_strata, "skipped": [],
    }


def clinical_trial_simulation(
    model_key: str, *, theta: dict, omega_cv_pct: dict, iiv_params: list[str],
    doses: list[float], tau: float, n_doses: int,
    metric: str = "ctrough", threshold: float | None = None,
    direction: str = "above", target_fraction: float = 0.9,
    cov_rows: list[dict] | None = None, wt_rows: list[float] | None = None,
    covariate_effects: list[dict] | None = None, n_subjects: int = 500,
    seed: int = 20250614, wt_default: float = 70.0, n_points: int = 160,
    max_subjects: int = 2000, max_doses: int = 24,
    param_draws: list[dict] | None = None,
) -> dict:
    """Probability of target attainment (PTA) across a dose grid.

    Simulate ``n_subjects`` virtual patients (covariates resampled from
    ``cov_rows``/``wt_rows``, IIV drawn from ``omega_cv_pct``) at every dose in
    ``doses`` for an ``n_doses``-dose regimen at interval ``tau``. For each dose,
    the PTA is the fraction of subjects whose ``metric`` (``cmax``/``auc_tau``/
    ``cavg``/``ctrough`` over the last interval) is on the target side of
    ``threshold`` (``direction`` = ``"above"`` for efficacy, ``"below"`` for
    safety). The recommended dose is the smallest (above) / largest (below) dose
    reaching ``target_fraction``.

    Returns ``{status, model_key, label, metric, threshold, direction,
    target_fraction, tau, n_doses, n_subjects, with_covariates, with_iiv,
    doses: [{dose, pta, n, metric_p05..p95}], recommended_dose,
    recommendation_note}``. ``threshold=None`` skips PTA and returns the exposure
    distribution per dose only.
    """
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {_METRICS}; got {metric!r}")
    if direction not in ("above", "below"):
        raise ValueError(f"direction must be 'above'|'below'; got {direction!r}")
    dose_grid = sorted({float(d) for d in doses if float(d) > 0})
    if not dose_grid:
        return {"status": "no_doses", "message": "no positive dose levels supplied."}
    if len(dose_grid) > max_doses:
        return {"status": "too_many_doses",
                "message": f"dose grid capped at {max_doses}; got {len(dose_grid)}."}
    n_subjects = int(max(1, min(n_subjects, max_subjects)))
    tau = float(tau)
    n_doses = int(max(1, n_doses))
    tmax = tau * n_doses

    model = get_model(model_key)
    typical = {**model.defaults, **{k: float(v) for k, v in theta.items()}}
    apply_cov = _covariate_applier(covariate_effects)
    sds = {p: float(np.sqrt(_cv_pct_to_omega2(omega_cv_pct.get(p))))
           for p in iiv_params}
    with_iiv = any(v > 0 for v in sds.values())

    rng = np.random.default_rng(seed)
    pop = _sample_population(cov_rows, wt_rows, n_subjects, rng, wt_default)
    # Pre-draw IIV etas (subjects x params) so results are seed-deterministic and
    # independent of the dose-grid iteration order.
    etas = {p: rng.normal(0.0, sds[p], size=n_subjects) if sds[p] > 0
            else np.zeros(n_subjects) for p in iiv_params}
    with_covariates = bool(covariate_effects) and bool(cov_rows)

    def _run_pop(theta_set: dict) -> dict:
        """Per-dose metric array for the FIXED (population, etas) — only theta
        changes. Common random numbers isolate the parameter-set's effect."""
        out: dict[float, np.ndarray] = {}
        for d in dose_grid:
            vals = np.empty(n_subjects)
            for si, subj in enumerate(pop):
                theta_i = apply_cov(theta_set, subj["cov"])
                params_i = {k: float(theta_i[k]) for k in theta_i}
                for p in iiv_params:
                    if p in params_i:
                        params_i[p] = params_i[p] * float(np.exp(etas[p][si]))
                sim = simulate_timecourse(model, params_i, dose=d, tau=tau,
                                          n_doses=n_doses, tmax=tmax, n_points=n_points,
                                          wt=subj["wt"])
                m = _interval_metrics(sim["times"], sim["cp"],
                                      t_last=(n_doses - 1) * tau, tau=tau, tmax=tmax)
                vals[si] = m[metric]
            out[d] = vals
        return out

    def _pta(vals: np.ndarray) -> float | None:
        finite = vals[np.isfinite(vals)]
        if threshold is None or not finite.size:
            return None
        hit = finite > float(threshold) if direction == "above" else finite < float(threshold)
        return float(np.mean(hit))

    point = _run_pop(typical)
    dose_rows: list[dict] = []
    for d in dose_grid:
        vals = point[d]
        finite = vals[np.isfinite(vals)]
        pcts = (np.percentile(finite, _REPORT_PCTL) if finite.size
                else [float("nan")] * len(_REPORT_PCTL))
        p = _pta(vals)
        dose_rows.append({
            "dose": round(d, 6), "n": int(finite.size),
            "metric_p05": _r(pcts[0]), "metric_p25": _r(pcts[1]),
            "metric_median": _r(pcts[2]), "metric_p75": _r(pcts[3]),
            "metric_p95": _r(pcts[4]),
            "pta": (None if p is None else round(p, 6)),
        })

    # Parameter-uncertainty PTA band + sensitivity (Week-12 Ex 3/4). Each draw is
    # a parameter set (theta ± its RSE, from the tool); the population is reused
    # (common random numbers) so the band reflects PARAMETER uncertainty.
    sensitivity = None
    if param_draws and threshold is not None:
        draw_ptas: dict[float, list[float]] = {d: [] for d in dose_grid}
        records: list[dict] = []
        for pdraw in param_draws:
            theta_d = {**typical, **{k: float(v) for k, v in pdraw.items()}}
            arrays_d = _run_pop(theta_d)
            pta_list: list[float | None] = []       # aligned to dose_grid order
            for d in dose_grid:
                pd = _pta(arrays_d[d])
                pta_list.append(None if pd is None else round(pd, 6))
                if pd is not None:
                    draw_ptas[d].append(pd)
            records.append({"theta": {k: round(float(v), 6) for k, v in pdraw.items()},
                            "pta": pta_list})
        for row in dose_rows:
            arr = draw_ptas.get(row["dose"]) or []
            if arr:
                lo, hi = np.percentile(arr, [2.5, 97.5])
                row["pta_lo"] = round(float(lo), 6)
                row["pta_hi"] = round(float(hi), 6)
        sensitivity = {"n_draws": len(param_draws),
                       "params": list(param_draws[0].keys()), "records": records}

    rec_dose, rec_note = _recommend(dose_rows, threshold, direction, target_fraction)
    return {
        "status": "ok", "model_key": model_key, "label": model.label,
        "metric": metric, "threshold": (None if threshold is None else round(float(threshold), 6)),
        "direction": direction, "target_fraction": round(float(target_fraction), 6),
        "tau": tau, "n_doses": n_doses, "n_subjects": n_subjects,
        "with_covariates": with_covariates, "with_iiv": with_iiv,
        "n_param_draws": len(param_draws or []), "sensitivity": sensitivity,
        "doses": dose_rows, "recommended_dose": rec_dose, "recommendation_note": rec_note,
    }


def _recommend(dose_rows: list[dict], threshold: float | None, direction: str,
               target_fraction: float) -> tuple[float | None, str]:
    """Smallest dose (efficacy/above) or largest dose (safety/below) whose PTA
    reaches ``target_fraction``."""
    if threshold is None:
        return None, "no threshold supplied — exposure distribution only, no PTA."
    meeting = [r for r in dose_rows if r.get("pta") is not None and r["pta"] >= target_fraction]
    if not meeting:
        return None, (f"no dose reaches {target_fraction:.0%} target attainment "
                      f"({direction} {threshold:g}).")
    if direction == "above":                 # efficacy: lowest sufficient dose
        best = min(meeting, key=lambda r: r["dose"])
    else:                                    # safety: highest still-safe dose
        best = max(meeting, key=lambda r: r["dose"])
    return best["dose"], (f"dose {best['dose']:g} attains {best['pta']:.1%} "
                          f"({direction} {threshold:g}), meeting the {target_fraction:.0%} target.")


def _r(x: float) -> float | None:
    return None if not np.isfinite(x) else round(float(x), 6)


def _coef_ses(covariate_effects: list[dict] | None) -> np.ndarray:
    """Per-coefficient standard errors from the stored ``rse_pct``
    (SE = rse_pct/100 * |coef|), flattened in the same order as
    ``nlme._cov_effects_from_records``. A missing/None RSE gives SE 0 (the
    coefficient is treated as known — its exposure ratio carries no width)."""
    ses: list[float] = []
    for r in covariate_effects or []:
        if r.get("kind", "power") == "categorical":
            levels = r.get("levels") or []
            cf = r.get("coefficient") or {}
            rse = r.get("rse_pct") or {}
            for lv in levels:
                c = float(cf.get(lv, 0.0))
                rp = rse.get(lv) if isinstance(rse, dict) else rse
                ses.append(abs(c) * float(rp) / 100.0 if rp else 0.0)
        else:
            c = float(r.get("coefficient") or 0.0)
            rp = r.get("rse_pct")
            ses.append(abs(c) * float(rp) / 100.0 if rp else 0.0)
    return np.asarray(ses, dtype=float)


def exposure_covariate_forest(
    model_key: str, *, theta: dict, covariate_effects: list[dict],
    scenarios: list[dict], reference_cov: dict, dose: float, tau: float,
    n_doses: int, ref_wt: float = 70.0, n_draws: int = 500, seed: int = 20250614,
    n_points: int = 160, band: tuple[float, float] = (0.8, 1.25),
) -> dict:
    """Simulated exposure covariate forest (Week-12 ``forest-plots.R``).

    For a reference patient and each covariate scenario level, forward-simulate a
    steady-state regimen (NO between-subject variability — ``zero_re``) and report
    the RELATIVE exposure (AUC over the last interval, and Cmax) versus the
    reference, with a 95% interval from covariate-coefficient uncertainty (drawn
    from the fitted coefficient ± its RSE). The shaded ``band`` (default
    0.8-1.25) marks the range usually judged not clinically meaningful.

    ``scenarios``: ``[{covariate, is_weight, levels: [{label, value}]}]``. A
    ``is_weight`` scenario varies the simulator's weight (allometric scaling); any
    other varies that covariate in the fitted covariate model. ``reference_cov``
    holds the reference covariate values (weight scenarios use ``ref_wt``).

    Returns ``{status, model_key, label, dose, tau, n_doses, n_draws, band,
    reference: {auc, cmax}, rows: [{covariate, label, value, is_weight,
    rel_auc: {median, lo, hi}, rel_cmax: {median, lo, hi}}]}``.
    """
    if not covariate_effects and not any(s.get("is_weight") for s in scenarios):
        return {"status": "no_covariate_model",
                "message": "no fitted covariate effects to build an exposure forest."}
    model = get_model(model_key)
    typical = {**model.defaults, **{k: float(v) for k, v in theta.items()}}

    from app.compute.nlme import _cov_effects_from_records  # lazy: heavy (scipy)
    effects, coefs0 = _cov_effects_from_records(covariate_effects)
    ses = _coef_ses(covariate_effects)
    if ses.size != coefs0.size:                      # defensive: shape mismatch
        ses = np.zeros_like(coefs0)

    tmax = tau * n_doses
    t_last = (n_doses - 1) * tau

    def _apply(coefs: np.ndarray, cov: dict) -> dict:
        p = dict(typical)
        i = 0
        for eff in effects:
            c = coefs[i:i + eff.n_coef]
            if eff.param in p:
                p[eff.param] = p[eff.param] * eff.factor(c, (cov or {}).get(eff.covariate))
            i += eff.n_coef
        return p

    def _exposure(coefs: np.ndarray, cov: dict, wt: float) -> tuple[float, float]:
        params = _apply(coefs, cov) if effects else dict(typical)
        sim = simulate_timecourse(model, params, dose=dose, tau=tau, n_doses=n_doses,
                                  tmax=tmax, n_points=n_points, wt=wt)
        m = _interval_metrics(sim["times"], sim["cp"], t_last=t_last, tau=tau, tmax=tmax)
        return m["auc_tau"], m["cmax"]

    rng = np.random.default_rng(seed)
    draws = (coefs0 + rng.standard_normal((int(n_draws), coefs0.size)) * ses
             if coefs0.size else np.zeros((int(n_draws), 0)))

    # Reference exposure per draw (shared across scenarios).
    ref_auc = np.empty(int(n_draws))
    ref_cmax = np.empty(int(n_draws))
    for k in range(int(n_draws)):
        ref_auc[k], ref_cmax[k] = _exposure(draws[k], reference_cov, ref_wt)

    def _summ(rel: np.ndarray) -> dict:
        f = rel[np.isfinite(rel)]
        if f.size == 0:
            return {"median": None, "lo": None, "hi": None}
        lo, hi = np.percentile(f, [2.5, 97.5])
        return {"median": _r(float(np.median(f))), "lo": _r(float(lo)), "hi": _r(float(hi))}

    rows = []
    for sc in scenarios:
        is_wt = bool(sc.get("is_weight"))
        for lv in sc.get("levels", []):
            ra = np.empty(int(n_draws))
            rc = np.empty(int(n_draws))
            for k in range(int(n_draws)):
                if is_wt:
                    a, cm = _exposure(draws[k], reference_cov, float(lv["value"]))
                else:
                    cov = {**reference_cov, sc["covariate"]: lv["value"]}
                    a, cm = _exposure(draws[k], cov, ref_wt)
                ra[k] = a / ref_auc[k] if ref_auc[k] else np.nan
                rc[k] = cm / ref_cmax[k] if ref_cmax[k] else np.nan
            rows.append({
                "covariate": sc["covariate"], "label": lv.get("label", str(lv["value"])),
                "value": lv["value"], "is_weight": is_wt,
                "rel_auc": _summ(ra), "rel_cmax": _summ(rc),
            })
    # Point-estimate reference exposure (coefs0 = fitted coefficients).
    ref0_auc, ref0_cmax = _exposure(coefs0, reference_cov, ref_wt)
    return {
        "status": "ok", "model_key": model_key, "label": model.label,
        "dose": round(float(dose), 6), "tau": float(tau), "n_doses": int(n_doses),
        "n_draws": int(n_draws), "band": [float(band[0]), float(band[1])],
        "reference": {"auc": _r(ref0_auc), "cmax": _r(ref0_cmax), "wt": float(ref_wt)},
        "rows": rows,
    }
