"""Statistical-analysis advisor: parametric vs non-parametric recommendations.

Deterministic rules over per-subject exposure metrics (Cmax, AUC, Tmax, ...):

* PK exposures are conventionally log-normal, so the default is a log
  transformation with geometric means and CV%. Shapiro-Wilk on the raw and
  log scale, plus skewness, decides whether that convention holds.
* Tmax is sampled on a discrete grid, so it is ALWAYS analysed
  non-parametrically (median, range, Wilcoxon / Mann-Whitney, Hodges-Lehmann).
* The test family follows the design: one group (descriptive + CI), two
  paired groups (crossover), two independent groups (parallel), or >2 groups.
* Levene's test on the log scale flags unequal variances (-> Welch variants).
* With small groups the normality tests are under-powered: the log-scale
  parametric convention is kept and a non-parametric sensitivity analysis
  is recommended instead of switching family on a weak signal.

Everything here is a recommendation for the pharmacometrician of record; no
hypothesis test result is reported as a conclusion about the drug.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats

ALPHA = 0.05
MIN_N_NORMALITY = 3       # scipy.stats.shapiro needs >= 3 values
SMALL_GROUP_N = 8         # below this, Shapiro-Wilk has little power
STRONG_SKEW = 1.0         # |skew| above this on the log scale is a red flag
OUTLIER_IQR_K = 1.5
RANK_ONLY_METRICS = ("Tmax",)                       # discrete sampling grid
LOG_SCALE_METRICS = ("Cmax", "AUC_last", "AUC_inf", "AUC_tau", "CL_F", "Vz_F",
                     "t_half", "AUC", "Cmin", "Cavg")


def _clean(values: list[float | None]) -> np.ndarray:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    return arr[np.isfinite(arr)]


def _r(x: float | None, dp: int = 4) -> float | None:
    if x is None or not math.isfinite(x):
        return None
    return round(float(x), dp)


def _shapiro_p(arr: np.ndarray) -> float | None:
    if len(arr) < MIN_N_NORMALITY or np.ptp(arr) == 0:
        return None
    return float(stats.shapiro(arr).pvalue)


def _n_outliers(arr: np.ndarray) -> int:
    if len(arr) < 4:
        return 0
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - OUTLIER_IQR_K * iqr, q3 + OUTLIER_IQR_K * iqr
    return int(((arr < lo) | (arr > hi)).sum())


def distribution_profile(values: list[float | None]) -> dict[str, Any]:
    """Descriptive + normality diagnostics on the raw and log scale."""
    arr = _clean(values)
    n = int(len(arr))
    out: dict[str, Any] = {"n": n}
    if n == 0:
        return out
    pos = arr[arr > 0]
    log_arr = np.log(pos) if len(pos) else np.array([])
    out.update({
        "mean": _r(arr.mean()), "sd": _r(arr.std(ddof=1)) if n > 1 else None,
        "median": _r(float(np.median(arr))),
        "min": _r(arr.min()), "max": _r(arr.max()),
        "n_nonpositive": int(n - len(pos)),
        "geometric_mean": _r(math.exp(log_arr.mean())) if len(log_arr) else None,
        "cv_pct": (_r(100.0 * math.sqrt(math.exp(log_arr.var(ddof=1)) - 1.0), 2)
                   if len(log_arr) > 1 else None),
        "skew_raw": _r(float(stats.skew(arr, bias=False))) if n > 2 else None,
        "skew_log": _r(float(stats.skew(log_arr, bias=False))) if len(log_arr) > 2 else None,
        "shapiro_raw_p": _r(_shapiro_p(arr)),
        "shapiro_log_p": _r(_shapiro_p(log_arr)) if len(log_arr) else None,
        "n_outliers_log": _n_outliers(log_arr) if len(log_arr) else 0,
    })
    return out


def levene_log_p(groups: dict[str, list[float | None]]) -> float | None:
    """Levene's test for equal variances across groups on the log scale."""
    arrays = []
    for vals in groups.values():
        arr = _clean(vals)
        arr = arr[arr > 0]
        if len(arr) >= MIN_N_NORMALITY:
            arrays.append(np.log(arr))
    if len(arrays) < 2:
        return None
    return float(stats.levene(*arrays, center="median").pvalue)


def choose_scale_and_family(metric: str, prof: dict[str, Any]) -> dict[str, Any]:
    """Decide log/raw/rank scale and parametric vs non-parametric family."""
    n = prof.get("n", 0)
    if metric in RANK_ONLY_METRICS:
        return {"scale": "rank", "family": "non-parametric",
                "rationale": (f"{metric} is observed on a discrete sampling grid, so it is "
                              "analysed by ranks (median, range) regardless of sample size.")}
    if n < MIN_N_NORMALITY:
        return {"scale": "log", "family": "descriptive",
                "rationale": f"Only {n} value(s): report descriptively; no inferential test."}
    p_log, p_raw = prof.get("shapiro_log_p"), prof.get("shapiro_raw_p")
    skew_log = prof.get("skew_log")
    nonpos = prof.get("n_nonpositive", 0)
    if nonpos:
        return {"scale": "rank", "family": "non-parametric",
                "rationale": (f"{nonpos} non-positive value(s) prevent a log transform; use "
                              "rank-based methods or address the BLQ handling first.")}
    log_ok = p_log is None or p_log >= ALPHA
    strong_skew = skew_log is not None and abs(skew_log) > STRONG_SKEW
    if n < SMALL_GROUP_N:
        return {"scale": "log", "family": "parametric",
                "rationale": (f"n={n} is too small for Shapiro-Wilk to have power; keep the "
                              "conventional log-scale parametric analysis and add a "
                              "non-parametric sensitivity analysis.")}
    if log_ok and not strong_skew:
        raw_note = ("" if p_raw is None or p_raw >= ALPHA
                    else f" Raw scale is non-normal (Shapiro p={p_raw:.3g}), confirming the "
                         "log transform.")
        return {"scale": "log", "family": "parametric",
                "rationale": (f"Log-scale values are consistent with normality (Shapiro "
                              f"p={p_log if p_log is not None else 'n/a'})." + raw_note)}
    return {"scale": "rank", "family": "non-parametric",
            "rationale": (f"Log-scale values still depart from normality (Shapiro p="
                          f"{p_log:.3g}, skew={skew_log}); use rank-based methods as the "
                          "primary analysis and log-scale parametric as a sensitivity check.")}


def recommend_tests(*, family: str, n_groups: int, paired: bool,
                    equal_variance: bool | None) -> dict[str, str | None]:
    """Map (family, design) to a primary test and an alternative-family test."""
    if n_groups <= 1:
        return {"primary": ("geometric mean with 95% CI (t-interval on log scale)"
                            if family == "parametric" else "median with 95% CI (bootstrap or sign-based)"),
                "sensitivity": None}
    welch = equal_variance is False
    if n_groups == 2 and paired:
        par = "paired t-test on log scale (crossover: linear mixed model with sequence, period, treatment)"
        npar = "Wilcoxon signed-rank test with Hodges-Lehmann estimate"
    elif n_groups == 2:
        par = ("Welch t-test on log scale" if welch else "two-sample t-test on log scale")
        npar = "Mann-Whitney U test with Hodges-Lehmann estimate"
    elif paired:
        par = "repeated-measures ANOVA / linear mixed model on log scale"
        npar = "Friedman test with post-hoc Wilcoxon signed-rank (adjusted)"
    else:
        par = ("Welch ANOVA on log scale with Games-Howell post-hoc" if welch
               else "one-way ANOVA on log scale with Tukey HSD post-hoc")
        npar = "Kruskal-Wallis test with Dunn post-hoc (adjusted)"
    if family == "parametric":
        return {"primary": par, "sensitivity": npar}
    if family == "non-parametric":
        return {"primary": npar, "sensitivity": par}
    return {"primary": "descriptive summary only", "sensitivity": None}


def _covariate_advice(cov: dict[str, Any], prof_by_metric: dict[str, dict[str, Any]]) -> dict[str, Any]:
    kind = cov["kind"]
    if kind == "categorical":
        return {"name": cov["name"], "kind": kind, "n_levels": cov.get("n_levels"),
                "recommendation": ("compare exposure across levels with the group test above "
                                   "(log-scale ANOVA / Kruskal-Wallis); Fisher's exact test for "
                                   "categorical outcomes")}
    parametric = all(p.get("family") == "parametric" for p in prof_by_metric.values()) \
        if prof_by_metric else True
    return {"name": cov["name"], "kind": kind,
            "recommendation": (("Pearson correlation of log-exposure vs covariate (linear regression "
                                "on log scale)") if parametric
                               else "Spearman rank correlation of exposure vs covariate")
            + "; formal covariate effects belong in the population PK model (SCM)."}


def advise(*, exposures: dict[str, dict[str, list[float | None]]], design: dict[str, Any],
           covariates: list[dict[str, Any]] | None = None,
           context: str | None = None) -> dict[str, Any]:
    """Build the full recommendation.

    ``exposures`` maps metric -> group label -> values (one per subject/period).
    ``design`` carries n_subjects, group_var, groups, n_per_group, paired.
    ``context`` is an optional study type hint ('bioequivalence', 'dose_levels').
    """
    n_groups = max(len(design.get("groups") or []), 1)
    paired = bool(design.get("paired"))
    metrics: dict[str, Any] = {}
    for metric, by_group in exposures.items():
        pooled = [v for vals in by_group.values() for v in vals]
        prof = distribution_profile(pooled)
        decision = choose_scale_and_family(metric, prof)
        if (context == "bioequivalence" and metric not in RANK_ONLY_METRICS
                and decision["family"] == "non-parametric" and not prof.get("n_nonpositive")):
            # Regulators (FDA/EMA) accept only the log-scale ANOVA for Cmax/AUC in ABE;
            # the rank-based result becomes the sensitivity analysis, not the primary.
            decision = {"scale": "log", "family": "parametric",
                        "rationale": decision["rationale"] + " Overridden for bioequivalence: the "
                        "regulatory primary analysis is the log-scale ANOVA; keep the "
                        "rank-based test as a sensitivity analysis."}
        lev = levene_log_p(by_group) if n_groups > 1 and metric not in RANK_ONLY_METRICS else None
        equal_var = None if lev is None else lev >= ALPHA
        tests = recommend_tests(family=decision["family"], n_groups=n_groups, paired=paired,
                                equal_variance=equal_var)
        metrics[metric] = {**prof, **decision, "levene_log_p": _r(lev),
                           "equal_variance": equal_var,
                           "primary_test": tests["primary"],
                           "sensitivity_test": tests["sensitivity"],
                           "summary_statistic": ("median (min-max)" if decision["scale"] == "rank"
                                                 else "geometric mean (CV%)")}
    recs = _general_recommendations(metrics, design, context)
    cov_out = [_covariate_advice(c, metrics) for c in (covariates or [])]
    return {"status": "ok", "design": {**design, "n_groups": n_groups},
            "metrics": metrics, "covariates": cov_out,
            "recommendations": recs, "caveats": _caveats(metrics, design)}


def _general_recommendations(metrics: dict[str, Any], design: dict[str, Any],
                             context: str | None) -> list[dict[str, str]]:
    recs: list[dict[str, str]] = []
    n_groups = max(len(design.get("groups") or []), 1)
    label = design.get("design_label", "single-group")
    recs.append({"topic": "design",
                 "recommendation": f"Treat the data as a {label} design with {n_groups} group(s).",
                 "rationale": "Test choice follows the design (paired vs independent) before the distribution."})
    families = {m: v["family"] for m, v in metrics.items()}
    if any(f == "parametric" for f in families.values()):
        recs.append({"topic": "transformation",
                     "recommendation": "Log-transform Cmax and AUC; report geometric means and CV%.",
                     "rationale": ("PK exposures are multiplicative (log-normal) by convention and by "
                                   "the diagnostics here.")})
    if any(f == "non-parametric" for m, f in families.items() if m not in RANK_ONLY_METRICS):
        recs.append({"topic": "family",
                     "recommendation": "Use rank-based tests as the primary analysis for the flagged metrics.",
                     "rationale": ("Normality fails even after log transformation, or non-positive "
                                   "values block the transform.")})
    if "Tmax" in metrics:
        recs.append({"topic": "Tmax",
                     "recommendation": "Summarise Tmax as median (range); compare with Wilcoxon / Mann-Whitney.",
                     "rationale": "Tmax is discrete (sampling grid) and never treated parametrically."})
    if context == "bioequivalence":
        recs.append({"topic": "bioequivalence",
                     "recommendation": ("Regulatory ABE: ANOVA on log-transformed Cmax and AUC with sequence, "
                                        "period, treatment; 90% CI of the GMR within 80-125%."),
                     "rationale": ("FDA/EMA require the log-scale parametric ANOVA; non-parametric "
                                   "methods are accepted only for Tmax.")})
    if context == "dose_levels" and n_groups > 1:
        recs.append({"topic": "dose proportionality",
                     "recommendation": "Assess dose proportionality with the power model (log exposure vs log dose).",
                     "rationale": "Groups are dose levels; a slope with CI is more informative than pairwise tests."})
    if any(v.get("equal_variance") is False for v in metrics.values()):
        recs.append({"topic": "variance",
                     "recommendation": "Use Welch-type (unequal-variance) parametric tests.",
                     "rationale": "Levene's test on the log scale rejects equal variances across groups."})
    return recs


def _caveats(metrics: dict[str, Any], design: dict[str, Any]) -> list[str]:
    out: list[str] = []
    n_per = design.get("n_per_group") or {}
    small = [g for g, n in n_per.items() if n < SMALL_GROUP_N]
    if small:
        out.append(f"Groups with n<{SMALL_GROUP_N} ({', '.join(map(str, small))}): normality tests "
                   "are under-powered; the parametric choice rests on convention.")
    for m, v in metrics.items():
        if v.get("n_outliers_log"):
            out.append(f"{m}: {v['n_outliers_log']} log-scale outlier(s) by the 1.5xIQR rule; "
                       "inspect before finalising the analysis.")
        if v.get("n_nonpositive"):
            out.append(f"{m}: {v['n_nonpositive']} non-positive value(s); resolve BLQ handling.")
    out.append("These are recommendations for the pharmacometrician of record, not conclusions.")
    return out
