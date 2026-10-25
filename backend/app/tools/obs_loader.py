"""Shared observation loader for single-dose profile analyses (NCA, bioequivalence, compartmental, CDISC export).

One place decides which rows are concentrations: EVID==0 / AMT==0 rows only, below-LOQ rows handled by rule M1, records
before the dose removed and times expressed since dose. ``compute_nca``, ``run_bioequivalence`` and
``fit_compartmental_models`` all call ``load_observations`` so a censored record is never integrated or fitted as a
real concentration in one tool and excluded in another."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.compute.dosing import parse_blq_flag


def _blq_flags(values: pd.Series, col: str) -> pd.Series:
    """The below-LOQ flag column as booleans (``parse_blq_flag``). Unrecognised values stop the analysis: an unknown
    code must not quietly mean "not BLQ" while the result says M1 was applied."""
    bad: set[str] = set()

    def parse(v: Any) -> bool:
        try:
            return parse_blq_flag(v)
        except ValueError:
            bad.add(str(v))
            return False

    flags = values.map(parse).astype(bool)
    if bad:
        raise ValueError(f"below-LOQ flag column {col!r} has unrecognised value(s) "
                         f"{', '.join(repr(b) for b in sorted(bad))}: use 1/0, Y/N, YES/NO or TRUE/FALSE "
                         "(blank = not flagged)")
    return flags


def _from_dose_time(obs: pd.DataFrame, id_col: str, time_col: str, dv_col: str, cens_col: str | None,
                    dose_times: dict[Any, float]) -> pd.DataFrame:
    """Records before the dose add no area. Per subject (dose time = its first dose record, else TIME 0), rows before
    the dose are removed and remaining times are expressed as time since that dose (t - t_dose). When no usable record
    (BLQ-flagged, or a finite DV) is AT the dose time, the pre-dose record(s) closest to it supply C = 0 there if they
    are BLQ-flagged or a measured 0 (no drug before the dose); a positive pre-dose value is not carried to the dose
    time."""
    flagged = (obs[cens_col].astype(bool) if cens_col is not None
               else pd.Series(False, index=obs.index))
    usable = flagged | np.isfinite(obs[dv_col].to_numpy(dtype=float))
    parts = []
    for sid, g in obs.groupby(id_col, sort=False):
        td = dose_times.get(sid, 0.0)
        before = g[time_col] < td
        post = g[~before].copy()
        pre = g[before & usable.loc[g.index]]
        if len(pre) and not (usable.loc[post.index] & (post[time_col] == td)).any():
            latest = pre[pre[time_col] == pre[time_col].max()]
            if (flagged.loc[latest.index] | (latest[dv_col] == 0)).all():
                post = pd.concat([latest.iloc[[0]].assign(**{time_col: td, dv_col: 0.0}), post])
        if td != 0.0:
            post[time_col] = post[time_col] - td
        parts.append(post)
    return pd.concat(parts) if parts else obs


def _apply_m1(obs: pd.DataFrame, id_col: str, time_col: str, dv_col: str, cens_col: str) -> pd.DataFrame:
    """Rule M1 for observations flagged below-LOQ (``parse_blq_flag``), per subject in time order: BLQ rows BEFORE the
    first quantifiable value are a 0 baseline (their value is set to 0, also when it was blank); BLQ rows after it, in
    the middle or trailing, are excluded. Quantifiable means unflagged with a finite numeric TIME and a finite,
    POSITIVE DV: a row without a time or a DV carries no usable measurement, and an unflagged 0 is a measured value
    that is kept as an observed zero but is not a quantifiable concentration, so neither ends the baseline. A subject
    with nothing quantifiable keeps all its rows as zeros (as before)."""
    flag = obs[cens_col].map(parse_blq_flag).astype(bool)
    work = obs.assign(_blq=flag.to_numpy(), _t=pd.to_numeric(obs[time_col], errors="coerce"))
    work = work.sort_values([id_col, "_t"], kind="stable")
    parts = []
    for _, g in work.groupby(id_col, sort=False):
        blq = g["_blq"].to_numpy()
        dv = pd.to_numeric(g[dv_col], errors="coerce").to_numpy(dtype=float)
        measured = np.isfinite(dv) & np.isfinite(g["_t"].to_numpy(dtype=float))
        quantified = np.flatnonzero(~blq & measured & (dv > 0))
        first_q = int(quantified[0]) if quantified.size else len(g)
        leading = blq & (np.arange(len(g)) < first_q)
        g = g.copy()
        g.loc[g.index[leading], dv_col] = 0.0
        parts.append(g[((~blq) & measured) | leading])
    return pd.concat(parts).drop(columns=["_blq", "_t"]) if parts else obs


PRE_DOSE_RULE = ("Records before the dose (the subject's first dose record, else TIME 0) add no area: with no sample "
                 "at the dose time, a BLQ-flagged or zero pre-dose record closest to it sets C = 0 there; other "
                 "pre-dose records are not used. Times are reported relative to that dose time.")


@dataclass
class LoadedObservations:
    """Observation rows ready for analysis, plus what was dosed and what was censored."""
    obs: pd.DataFrame                       # M1-applied, time since dose, finite TIME/DV only
    dose_by_subject: dict[Any, float]       # keyed by profile key (subject id, or tuple when ``group_cols`` given)
    dose_times: dict[Any, float]
    n_blq: int
    cens_col: str | None
    dose_rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    key_col: str = "_pkey"


def _first_col(roles: dict[str, str], role: str) -> str | None:
    return next((c for c, r in roles.items() if r == role), None)


def load_observations(df: pd.DataFrame, roles: dict[str, str], *,
                      group_cols: list[str] | None = None) -> LoadedObservations:
    """EVID==0 / MDV==0 observation rows with the M1 below-LOQ rule applied (see ``_apply_m1``).

    Below-LOQ comes from the CENS role column (flag codes per ``parse_blq_flag``); a record flagged there never
    contributes its DV (NONMEM style rows carry the LLOQ in DV). Without a CENS column, an LLOQ role column flags
    DV < LLOQ. With neither, concentrations are used as given. ``group_cols`` makes a profile (subject, treatment):
    the dose and its time are then taken per profile, and the profile key is column ``_pkey`` (a tuple)."""
    df = df.copy()
    id_col, time_col, dv_col = _first_col(roles, "ID"), _first_col(roles, "TIME"), _first_col(roles, "DV")
    if not (id_col and time_col and dv_col):
        raise ValueError("dataset missing required ID/TIME/DV roles")
    amt_col, evid_col = _first_col(roles, "AMT"), _first_col(roles, "EVID")
    mdv_col, cens_col, lloq_col = _first_col(roles, "MDV"), _first_col(roles, "CENS"), _first_col(roles, "LLOQ")

    df[time_col] = pd.to_numeric(df[time_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df[dv_col] = pd.to_numeric(df[dv_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if amt_col:
        df[amt_col] = pd.to_numeric(df[amt_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if cens_col is not None:
        df[cens_col] = _blq_flags(df[cens_col], cens_col)
    elif lloq_col is not None:
        lloq = pd.to_numeric(df[lloq_col], errors="coerce")
        cens_col = "_cens_from_lloq"
        df[cens_col] = (df[dv_col] < lloq).fillna(False).astype(bool)

    key_col = "_pkey"
    cols = group_cols or [id_col]
    df[key_col] = df[cols[0]] if len(cols) == 1 else list(zip(*(df[c] for c in cols)))

    if evid_col:
        ev = pd.to_numeric(df[evid_col], errors="coerce").fillna(0)
        dose_rows, obs = df[ev == 1], df[ev == 0]
    elif amt_col:
        dose_rows = df[df[amt_col].fillna(0) > 0]
        obs = df[df[amt_col].fillna(0) == 0]
    else:
        dose_rows, obs = df.iloc[0:0], df
    if mdv_col:
        obs = obs[pd.to_numeric(obs[mdv_col], errors="coerce").fillna(0) == 0]

    dose_by_subject: dict[Any, float] = {}
    if amt_col and len(dose_rows):
        for key, g in dose_rows.groupby(key_col, sort=False):
            amts = g[amt_col].dropna()
            if len(amts):
                dose_by_subject[key] = float(amts.iloc[0])
    timed = dose_rows[np.isfinite(dose_rows[time_col].to_numpy(dtype=float))]
    dose_times = {key: float(t) for key, t in timed.groupby(key_col, sort=False)[time_col].min().items()}

    if cens_col is not None:
        n_blq = int(obs[cens_col].sum())
    else:
        n_blq = int((obs[dv_col] <= 0).sum())

    obs = _from_dose_time(obs, key_col, time_col, dv_col, cens_col, dose_times)
    if cens_col is not None:
        obs = _apply_m1(obs, key_col, time_col, dv_col, cens_col)
    obs = obs.dropna(subset=[time_col, dv_col])
    return LoadedObservations(obs=obs, dose_by_subject=dose_by_subject, dose_times=dose_times, n_blq=n_blq,
                              cens_col=cens_col, dose_rows=dose_rows, key_col=key_col)
