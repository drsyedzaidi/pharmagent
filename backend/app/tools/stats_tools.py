"""Statistician Agent tools: recommend how to analyse the loaded PK data.

Reads the server-side dataset (never exposed to the LLM) and, when present,
the per-subject NCA parameters, then hands derived exposures to
``app.compute.stats_advice`` for the parametric / non-parametric decision.
Only summary statistics and recommendations are written to state.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.compute.stats_advice import advise
from app.core.pharmstate import PharmState
from app.core.schema_extractor import detect_roles
from app.tools.base import Tool, ToolContext, ToolResult

_GROUP_HINTS = ("trt", "treat", "arm", "group", "form", "period", "seq", "cohort")
_MAX_GROUP_LEVELS = 6
_NCA_METRICS = ("Cmax", "AUC_last", "AUC_inf", "Tmax", "t_half")


def _roles(state: PharmState, df: pd.DataFrame) -> dict[str, str]:
    meta = state.dataset_metadata or {}
    return meta.get("detected_roles") or detect_roles(list(df.columns))


def _col(roles: dict[str, str], role: str) -> str | None:
    return next((c for c, r in roles.items() if r == role), None)


def _per_subject_constant(df: pd.DataFrame, id_col: str, col: str) -> bool:
    return bool((df.groupby(id_col)[col].nunique(dropna=True) <= 1).all())


def _detect_group_col(df: pd.DataFrame, roles: dict[str, str], id_col: str) -> str | None:
    """A low-cardinality, non-role column; TRT/ARM-like names win."""
    candidates = [c for c in df.columns if c not in roles
                  and df[c].nunique(dropna=True) <= _MAX_GROUP_LEVELS
                  and df[c].nunique(dropna=True) >= 2]
    hinted = [c for c in candidates if any(h in c.lower() for h in _GROUP_HINTS)]
    return (hinted or candidates or [None])[0]


def _detect_covariates(df: pd.DataFrame, roles: dict[str, str], id_col: str,
                       group_col: str | None) -> list[dict[str, Any]]:
    out = []
    for c in df.columns:
        if c in roles or c == group_col or not _per_subject_constant(df, id_col, c):
            continue
        nun = int(df[c].nunique(dropna=True))
        if nun < 2:
            continue
        numeric = pd.api.types.is_numeric_dtype(df[c])
        kind = "continuous" if numeric and nun > _MAX_GROUP_LEVELS else "categorical"
        out.append({"name": c, "kind": kind, "n_levels": nun if kind == "categorical" else None})
    return out


def _observed_exposures(df: pd.DataFrame, id_col: str, time_col: str, dv_col: str,
                        group_col: str | None) -> tuple[dict[str, dict[str, list]], dict[str, set]]:
    """Observed Cmax / Tmax / AUClast (linear trapezoid) per (subject, group)."""
    obs = df.copy()
    obs[time_col] = pd.to_numeric(obs[time_col], errors="coerce")
    obs[dv_col] = pd.to_numeric(obs[dv_col], errors="coerce")
    obs = obs.dropna(subset=[time_col, dv_col])
    keys = [id_col] + ([group_col] if group_col else [])
    metrics: dict[str, dict[str, list]] = {"Cmax": {}, "Tmax": {}, "AUC_last": {}}
    subjects_by_group: dict[str, set] = {}
    for key, g in obs.groupby(keys):
        sid = key[0] if isinstance(key, tuple) else key
        grp = str(key[1]) if group_col else "all"
        g = g.sort_values(time_col)
        t, c = g[time_col].to_numpy(float), g[dv_col].to_numpy(float)
        if len(t) < 2 or not (c > 0).any():
            continue
        i = int(np.argmax(c))
        metrics["Cmax"].setdefault(grp, []).append(float(c[i]))
        metrics["Tmax"].setdefault(grp, []).append(float(t[i]))
        metrics["AUC_last"].setdefault(grp, []).append(float(np.trapezoid(c, t)))
        subjects_by_group.setdefault(grp, set()).add(sid)
    return metrics, subjects_by_group


def _nca_exposures(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, list]], dict[str, set]]:
    """Per-subject NCA metrics grouped by dose level (or a single group)."""
    doses = {r.get("dose") for r in rows}
    by_dose = len(doses) > 1
    metrics: dict[str, dict[str, list]] = {m: {} for m in _NCA_METRICS}
    subjects_by_group: dict[str, set] = {}
    for r in rows:
        grp = str(r.get("dose")) if by_dose else "all"
        subjects_by_group.setdefault(grp, set()).add(r.get("subject"))
        for m in _NCA_METRICS:
            if r.get(m) is not None:
                metrics[m].setdefault(grp, []).append(float(r[m]))
    return {m: g for m, g in metrics.items() if g}, subjects_by_group


def recommend_statistics(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    dsid = args.get("dataset_id") or state.dataset_id
    if not dsid or dsid not in ctx.dataset_store:
        raise ValueError("no dataset loaded — load a dataset before asking for statistical advice")
    df = ctx.dataset_store[dsid]
    roles = _roles(state, df)
    id_col, time_col, dv_col = _col(roles, "ID"), _col(roles, "TIME"), _col(roles, "DV")
    if not (id_col and time_col and dv_col):
        raise ValueError("dataset needs ID, TIME and DV columns for statistical advice")

    group_col = _detect_group_col(df, roles, id_col)
    covariates = _detect_covariates(df, roles, id_col, group_col)
    if group_col and _per_subject_constant(df, id_col, group_col):
        paired = False
    elif group_col:
        paired = True           # the same subject appears under several levels: crossover
    else:
        paired = False

    context = None
    if state.nca_parameters and not group_col:
        exposures, subj = _nca_exposures(state.nca_parameters)
        source, group_var = "nca", ("dose" if len(subj) > 1 else None)
        context = "dose_levels" if group_var else None
    else:
        exposures, subj = _observed_exposures(df, id_col, time_col, dv_col, group_col)
        source, group_var = "observed", group_col
        levels = {str(v).upper() for v in subj}
        if group_col and paired and levels & {"R", "T", "REF", "TEST", "REFERENCE"}:
            context = "bioequivalence"
    if not exposures:
        raise ValueError("no positive concentration data to derive exposures from")

    groups = sorted(subj)
    design = {
        "n_subjects": int(len(set().union(*subj.values()))),
        "group_var": group_var, "groups": groups,
        "n_per_group": {g: len(s) for g, s in subj.items()},
        "paired": paired,
        "design_label": ("crossover" if paired else "parallel-group" if len(groups) > 1
                         else "single-group"),
        "source": source,
    }
    res = advise(exposures=exposures, design=design, covariates=covariates, context=context)
    fam = {m: v["family"] for m, v in res["metrics"].items()}
    n_par = sum(f == "parametric" for f in fam.values())
    n_npar = sum(f == "non-parametric" for f in fam.values())
    return ToolResult(
        summary=(f"Statistical advice ({design['design_label']}, {len(groups)} group(s), "
                 f"{design['n_subjects']} subjects, {source} exposures): "
                 f"{n_par} metric(s) parametric on log scale, {n_npar} non-parametric. "
                 f"{res['recommendations'][1]['recommendation'] if len(res['recommendations']) > 1 else ''}"
                 ).strip(),
        action=f"recommend_statistics({dsid})",
        writes={"stats_advice": res},
        result=res,
    )


TOOLS = [
    Tool("recommend_statistics",
         "Recommend how to analyse the loaded PK data: log transformation, parametric vs "
         "non-parametric family per exposure metric (Shapiro-Wilk, skewness, Levene), the "
         "test matching the design (paired/crossover vs parallel, 2 vs >2 groups), Tmax "
         "handling, covariate correlation methods, and regulatory conventions.",
         "statistician",
         {"type": "object",
          "properties": {"dataset_id": {"type": "string"}}, "required": []},
         recommend_statistics),
]
