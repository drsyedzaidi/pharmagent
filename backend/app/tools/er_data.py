"""Per-subject analysis tables for the exposure-response tools.

Exposure-response is a SUBJECT-level analysis: one exposure, one response (or
time and event), one row per subject. The loaded dataset is often a long
concentration-time file, so this helper

* collapses to one row per subject, REQUIRING every requested column to be
  constant within a subject (a value that varies across a subject's rows is an
  error naming the column, never silently the first/last/mean);
* optionally takes exposure from the per-subject NCA results in state when the
  dataset has no exposure column, joining on the subject id as a string;
* refuses non-numeric columns instead of coercing them to NaN and losing the
  subject;
* fingerprints the analysis arrays (sha256) so a later bootstrap can prove it
  is resampling the very data the stored fit was computed from.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.core.pharmstate import PharmState
from app.tools.nca_tools import _roles

ENDPOINTS = ("binary", "time_to_event")


@dataclass(frozen=True)
class Analysis:
    endpoint: str
    subjects: list[str]
    exposure: list[float | None]
    response: list[float | None] | None = None
    time: list[float | None] | None = None
    event: list[float | None] | None = None
    covariates: dict[str, list[float | None]] = field(default_factory=dict)
    strata: list[str] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        blob = json.dumps({"s": self.subjects, "x": self.exposure, "y": self.response, "t": self.time,
                           "e": self.event, "c": self.covariates}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()


def _numeric(series: pd.Series, name: str) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    bad = out.isna() & series.notna()
    if bad.any():
        sample = ", ".join(repr(v) for v in series[bad].unique()[:3])
        raise ValueError(f"column {name!r} has non-numeric values ({sample}); encode as numbers "
                         "(binary response/event as 0/1)")
    return out


def _col(df: pd.DataFrame, args: dict[str, Any], key: str, *, required: bool = True) -> str | None:
    name = args.get(key)
    if not name:
        if required:
            raise ValueError(f"{key} is required")
        return None
    if name not in df.columns:
        raise ValueError(f"column {name!r} ({key}) not found in the dataset; columns: "
                         f"{', '.join(map(str, df.columns[:30]))}")
    return str(name)


def _subject_column(df: pd.DataFrame, state: PharmState, args: dict[str, Any]) -> str | None:
    if args.get("subject_column"):
        return _col(df, args, "subject_column")
    return next((c for c, r in _roles(df, state).items() if r == "ID"), None)


def _per_subject(df: pd.DataFrame, id_col: str | None, cols: list[str]) -> pd.DataFrame:
    """One row per subject; ``cols`` must be constant within each subject."""
    if id_col is None:
        out = df[cols].copy()
        out.index = [str(i) for i in range(len(out))]
        return out
    ids = df[id_col].astype(str)
    g = df[cols].groupby(ids, sort=False)
    varies = [c for c in cols if (g[c].nunique(dropna=True) > 1).any()]
    if varies:
        raise ValueError(f"column(s) {varies} vary within a subject; exposure-response needs ONE value per "
                         "subject (use exposure_source='nca' for exposure, and subject-level response / "
                         "time / event columns)")
    return g.first()


def _nca_exposure(state: PharmState, metric: str) -> dict[str, float]:
    rows = state.nca_parameters or []
    if not rows:
        raise ValueError("no NCA results in state: run NCA first, or give exposure_column")
    if not any(metric in r for r in rows):
        have = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float))})
        raise ValueError(f"exposure_metric {metric!r} not in the NCA results; available: {', '.join(have)}")
    return {str(r["subject"]): r.get(metric) for r in rows if r.get(metric) is not None}


def _clean(vals: pd.Series) -> list[float | None]:
    return [None if pd.isna(v) else float(v) for v in vals]


def build_analysis(df: pd.DataFrame, state: PharmState, args: dict[str, Any]) -> Analysis:
    """The per-subject arrays for one E-R analysis described by ``args``."""
    endpoint = args.get("endpoint")
    if endpoint not in ENDPOINTS:
        raise ValueError(f"endpoint must be one of {ENDPOINTS}")
    id_col = _subject_column(df, state, args)
    resp = _col(df, args, "response_column") if endpoint == "binary" else None
    tcol = _col(df, args, "time_column") if endpoint == "time_to_event" else None
    ecol = _col(df, args, "event_column") if endpoint == "time_to_event" else None
    covs = [c for c in (args.get("covariate_columns") or []) if c]
    for c in covs:
        _col(df, {"c": c}, "c")
    if args.get("exposure_column") in covs:
        raise ValueError("covariate_columns must not include the exposure column")
    strat = _col(df, args, "stratify_by", required=False)
    x_col = args.get("exposure_column")
    source = args.get("exposure_source") or ("dataset" if x_col and x_col in df.columns else "nca")
    if source not in ("dataset", "nca"):
        raise ValueError("exposure_source must be 'dataset' or 'nca'")
    if source == "dataset":
        x_col = _col(df, args, "exposure_column")
    metric = args.get("exposure_metric")
    if source == "nca" and not metric:
        raise ValueError("exposure_metric (an NCA parameter such as AUC_inf, Cmax, AUC_tau) is required "
                         "when exposure comes from the NCA results")

    wanted = [c for c in (resp, tcol, ecol, *covs, strat, x_col if source == "dataset" else None) if c]
    table = _per_subject(df, id_col, list(dict.fromkeys(wanted)))
    subjects = [str(s) for s in table.index]
    n_missing = 0
    if source == "dataset":
        exposure = _clean(_numeric(table[x_col], x_col))
        label = args.get("exposure_label") or x_col
    else:
        nca = _nca_exposure(state, metric)
        exposure = [nca.get(s) for s in subjects]
        n_missing = sum(v is None for v in exposure)
        label = args.get("exposure_label") or metric
    if label in covs:
        raise ValueError(f"covariate {label!r} has the same name as the exposure; rename one of them")
    meta = {"exposure_source": source, "exposure_label": label, "subject_column": id_col,
            "n_subjects": len(subjects), "n_missing_exposure": n_missing,
            "one_row_per_subject_assumed": id_col is None}
    return Analysis(
        endpoint=endpoint, subjects=subjects, exposure=exposure,
        response=_clean(_numeric(table[resp], resp)) if resp else None,
        time=_clean(_numeric(table[tcol], tcol)) if tcol else None,
        event=_clean(_numeric(table[ecol], ecol)) if ecol else None,
        covariates={c: _clean(_numeric(table[c], c)) for c in covs},
        strata=[str(v) for v in table[strat]] if strat else None, meta=meta)


def spec_from_args(args: dict[str, Any], label: str) -> dict[str, Any]:
    """The re-runnable description of an analysis (stored on the fit so a
    bootstrap can rebuild exactly the same arrays)."""
    keys = ("endpoint", "exposure_column", "exposure_source", "exposure_metric", "exposure_label",
            "response_column", "time_column", "event_column", "covariate_columns", "subject_column")
    return {"label": label, **{k: args[k] for k in keys if args.get(k) not in (None, "", [])}}
