"""Subject-cohort construction and covariate evaluation points from a PK/PD DataFrame.

Pure domain module (`app.compute`): depends only on `numpy`, `pandas`, and
`app.compute.dosing`. Contains zero imports from `app.core`, `app.tools`, or
`app.agents` so `app.engines` and `app.tools` can both consume it without
upward or circular dependencies.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.compute.dosing import dose_events

_WT_NAMES = {"wt", "weight", "bw", "bwt", "bodyweight"}
_MIN_N_FOR_COV_PERCENTILE = 5  # below this, a percentile is not trustworthy -> center-only row


def _num_or_none(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_subjects(df: pd.DataFrame, roles: dict[str, str], *,
                   with_blq: bool = False) -> tuple[list[dict], bool, bool]:
    """Build per-subject dosing, observation, BLQ, PD, weight, and covariate records."""
    id_col = next((c for c, r in roles.items() if r == "ID"), None)
    time_col = next((c for c, r in roles.items() if r == "TIME"), None)
    dv_col = next((c for c, r in roles.items() if r == "DV"), None)
    amt_col = next((c for c, r in roles.items() if r == "AMT"), None)
    ii_col = next((c for c, r in roles.items() if r == "II"), None)
    addl_col = next((c for c, r in roles.items() if r == "ADDL"), None)
    dvid_col = next((c for c, r in roles.items() if r == "DVID"), None)
    pd_col = next((c for c, r in roles.items() if r == "PD"), None)
    # BLQ (M3) only when requested AND a CENS column exists, and only on the
    # single-endpoint path — keeps every other caller's behaviour unchanged.
    cens_col = next((c for c, r in roles.items() if r == "CENS"), None)
    do_blq = bool(with_blq and cens_col and not dvid_col)
    # Infusion rate and dosing compartment, when the dataset carries them: a
    # 2 h infusion fitted as a bolus, or an SC dose silently redirected to the
    # central compartment, biases every parameter downstream.
    rate_col = next((c for c, r in roles.items() if r == "RATE"), None)
    cmt_col = next((c for c, r in roles.items() if r == "CMT"), None)
    if not (id_col and time_col and dv_col and amt_col):
        raise ValueError("dataset needs ID/TIME/DV/AMT roles for PK model fitting")

    used = {c for c, r in roles.items()
            if r in {"ID", "TIME", "DV", "AMT", "EVID", "MDV", "CMT", "II", "ADDL",
                     "DVID", "PD", "CENS", "RATE"}}
    wt_col = next((c for c in df.columns if c not in used and c.strip().lower() in _WT_NAMES), None)
    # Candidate covariate columns: anything not a PK structural role (includes WT,
    # AGE, SEX, CRCL, ...). Baseline (first non-null) value per subject is taken.
    cov_cols = [c for c in df.columns if c not in used and c != id_col]

    dft = df.copy()
    dft[time_col] = pd.to_numeric(dft[time_col], errors="coerce")
    dft[dv_col] = pd.to_numeric(dft[dv_col], errors="coerce")
    if pd_col:
        dft[pd_col] = pd.to_numeric(dft[pd_col], errors="coerce")
    if dvid_col:
        dft[dvid_col] = pd.to_numeric(dft[dvid_col], errors="coerce")

    has_pd = bool(pd_col) or bool(dvid_col)
    subjects: list[dict] = []
    multi = False
    for sid, g in dft.groupby(id_col):
        rows = g.to_dict("records")
        doses = dose_events(rows, time_col=time_col, amt_col=amt_col,
                            ii_col=ii_col, addl_col=addl_col,
                            rate_col=rate_col, cmt_col=cmt_col)
        if len(doses) > 1:
            multi = True
        # PK observations: positive concentrations at t>0. Oral first-order
        # absorption models give C(0)=0, so a measurable pre-/at-dose (t<=0)
        # sample is structurally unfittable and would distort the fit (inflating
        # residual error / collapsing IIV) — exclude it, matching the
        # compartmental fitter.
        if dvid_col:
            pk = g[g[dvid_col] == 1].dropna(subset=[time_col, dv_col])
            pk = pk[(pk[dv_col] > 0) & (pk[time_col] > 0)]
            pdo = g[g[dvid_col] == 2].dropna(subset=[time_col, dv_col])
        elif do_blq:
            pk = g.dropna(subset=[time_col, dv_col])
            cens = pd.to_numeric(pk[cens_col], errors="coerce").fillna(0) == 1
            pk = pk[(pk[time_col] > 0) & ((pk[dv_col] > 0) | cens)]
            pdo = g.iloc[0:0]
        else:
            pk = g.dropna(subset=[time_col, dv_col])
            pk = pk[(pk[dv_col] > 0) & (pk[time_col] > 0)]
            pdo = g.dropna(subset=[time_col, pd_col]) if pd_col else g.iloc[0:0]
        if pk.empty or not doses:
            continue
        wt = 70.0
        if wt_col is not None:
            w = pd.to_numeric(g[wt_col], errors="coerce").dropna()
            if len(w):
                wt = float(w.iloc[0])
        cov: dict[str, Any] = {}
        for c in cov_cols:
            s = g[c].dropna()
            if not len(s):
                continue
            v0 = s.iloc[0]
            num = pd.to_numeric(pd.Series([v0]), errors="coerce").iloc[0]
            cov[c] = float(num) if pd.notna(num) else str(v0)
        subj = {"subject": sid, "doses": doses,
                "obs_t": pk[time_col].to_numpy(float),
                "obs_c": pk[dv_col].to_numpy(float), "wt": wt, "cov": cov}
        if do_blq:
            blq = (pd.to_numeric(pk[cens_col], errors="coerce").fillna(0) == 1).to_numpy(bool)
            subj["obs_blq"] = blq.tolist()
            blq_dv = pk[dv_col].to_numpy(float)[blq]   # NONMEM: BLQ rows carry LLOQ in DV
            subj["lloq"] = float(np.median(blq_dv)) if blq_dv.size else None
        if has_pd and len(pdo):
            pd_dv = dv_col if dvid_col else pd_col
            subj["pd_t"] = pdo[time_col].to_numpy(float)
            subj["pd_e"] = pdo[pd_dv].to_numpy(float)
        subjects.append(subj)
    return subjects, multi, has_pd


def dataset_lloq(df: pd.DataFrame, roles: dict[str, str]) -> float | None:
    """LLOQ from an explicit LLOQ-role column (median positive value), else None."""
    lloq_col = next((c for c, r in roles.items() if r == "LLOQ"), None)
    if not lloq_col or lloq_col not in df.columns:
        return None
    vals = pd.to_numeric(df[lloq_col], errors="coerce").dropna()
    vals = vals[vals > 0]
    return float(vals.median()) if len(vals) else None


def cov_eval_points(subjects: list[dict], effects: list[dict], percentiles: list[float]
                    ) -> tuple[dict[str, list], dict[str, str], dict[str, dict]]:
    """Evaluation points + reference-level recovery for `covariate_forest`,
    from the CURRENT dataset's subjects (never from the fit result, which
    carries no raw covariate distribution).

    Categorical reference level mirrors `app.compute.nlme._build_cov_effects`
    exactly (`max(uniq, key=vals.count)`, over `str(subject.cov[covariate])`
    for subjects with a non-null value) — the fitted result never stores
    which level was chosen as reference, so this is the only way to recover
    a meaningful row label. Continuous percentiles are skipped (falling back
    to a center-only row inside `covariate_forest`) below a minimum sample
    size, since a percentile from a handful of subjects is not trustworthy.
    """
    cov_values: dict[str, list] = {}
    ref_levels: dict[str, str] = {}
    cov_stats: dict[str, dict] = {}
    for eff in effects:
        cov = eff["covariate"]
        if cov in cov_values:
            continue
        if eff["kind"] == "categorical":
            vals = [str(s.get("cov", {}).get(cov)) for s in subjects
                    if s.get("cov", {}).get(cov) is not None]
            if not vals:
                continue
            uniq = sorted(set(vals))
            ref_levels[cov] = max(uniq, key=vals.count)
            cov_values[cov] = uniq
            cov_stats[cov] = {"n_cov": len(vals), "levels": uniq}
        else:
            nums = [n for n in (_num_or_none(s.get("cov", {}).get(cov)) for s in subjects)
                    if n is not None]
            if len(nums) < _MIN_N_FOR_COV_PERCENTILE:
                continue
            lo, hi = np.percentile(nums, percentiles)
            cov_values[cov] = [float(lo), float(hi)]
            cov_stats[cov] = {"n_cov": len(nums), "cov_min": float(min(nums)), "cov_max": float(max(nums))}
    return cov_values, ref_levels, cov_stats
