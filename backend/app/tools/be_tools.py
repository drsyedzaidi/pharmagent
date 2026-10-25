"""Bioequivalence Agent tools: test/reference GMR + 90% CI from NCA exposures.

Computes per-(subject, treatment) NCA exposures, then the average-bioequivalence
ratio and confidence interval for Cmax and AUC against the 80-125% limits.
Requires a treatment/formulation column distinguishing test from reference.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from app.compute.bioequivalence import PAIRED_LABEL, assess_bioequivalence
from app.compute.nca import INF_DERIVED, Profile, lambda_z_exclusion_note, nca_subject
from app.core.pharmstate import PharmState
from app.core.schema_extractor import detect_roles
from app.tools.base import Tool, ToolContext, ToolResult, require_dataset
from app.tools.obs_loader import load_observations

_TRT_NAMES = {"trt", "treat", "treatment", "form", "formulation", "drug", "period_trt"}
_REF_TOKENS = {"r", "ref", "reference", "rld"}
_BE_PARAMS = ("Cmax", "AUC_last", "AUC_inf")


def _roles(df: pd.DataFrame, state: PharmState) -> dict[str, str]:
    if state.dataset_metadata and state.dataset_metadata.get("detected_roles"):
        return state.dataset_metadata["detected_roles"]
    return detect_roles(list(df.columns))


def _find_treatment_col(df: pd.DataFrame, roles: dict[str, str]) -> str | None:
    used = {c for c, r in roles.items() if r in {"ID", "TIME", "DV", "AMT", "EVID", "MDV", "CMT"}}
    for c in df.columns:
        if c in used:
            continue
        if c.strip().lower() in _TRT_NAMES:
            return c
    return None


def _exposures_by_treatment(
    obs: pd.DataFrame, *, id_col: str, time_col: str, dv_col: str, trt_col: str,
    include_unacceptable_lambda_z: bool = False, excluded: dict[Any, list[Any]] | None = None,
) -> dict[Any, dict[Any, dict[str, float]]]:
    """Return {treatment: {subject: {Cmax, AUC_last, AUC_inf}}}. A profile whose terminal fit is unacceptable has
    AUC_inf left out (None) unless ``include_unacceptable_lambda_z``; its subject id is added to ``excluded[trt]``."""
    out: dict[Any, dict[Any, dict[str, float]]] = {}
    for (sid, trt), g in obs.groupby([id_col, trt_col]):
        g = g.dropna(subset=[time_col, dv_col]).sort_values(time_col)
        if len(g) < 3:
            continue
        prof = Profile(subject=sid, time=g[time_col].to_numpy(float),
                       conc=g[dv_col].to_numpy(float), dose=float("nan"))
        p = nca_subject(prof)
        vals = {k: p.get(k) for k in _BE_PARAMS}
        if p.get("lambda_z_acceptable") is False and not include_unacceptable_lambda_z:
            vals.update({k: None for k in _BE_PARAMS if k in INF_DERIVED})
            if excluded is not None:
                excluded.setdefault(trt, []).append(sid)
        out.setdefault(trt, {})[sid] = vals
    return out


def _col(df: pd.DataFrame, names: set[str]) -> str | None:
    return next((c for c in df.columns if str(c).strip().lower() in names), None)


def _design_columns(df: pd.DataFrame, obs: pd.DataFrame, id_col: str, trt_col: str, test_lv: Any, ref_lv: Any,
                    common: list[Any]) -> dict[str, dict[Any, Any]] | None:
    """Per-subject sequence and (test period, reference period) from PERIOD / SEQUENCE columns, or None when the
    columns are missing or are not a two-period, multi-sequence layout. Subjects without a usable value are simply
    absent from the returned maps (the caller reports them as excluded)."""
    per_col, seq_col = _col(df, {"period", "per", "prd"}), _col(df, {"sequence", "seq", "sequ"})
    if per_col is None or seq_col is None:
        return None
    o = obs[obs[id_col].isin(common)]
    first_period = o.dropna(subset=[per_col]).groupby([id_col, trt_col])[per_col].first()
    sequence = o.dropna(subset=[seq_col]).groupby(id_col)[seq_col].first()
    periods: dict[Any, tuple[Any, Any]] = {}
    for sid in common:
        if (sid, test_lv) in first_period.index and (sid, ref_lv) in first_period.index:
            tp, rp = first_period[(sid, test_lv)], first_period[(sid, ref_lv)]
            if tp != rp:
                periods[sid] = (tp, rp)
    levels = {p for pr in periods.values() for p in pr}
    if len(levels) != 2 or len({sequence.get(s) for s in periods if s in sequence.index}) < 2:
        return None
    return {"periods": {s: pr for s, pr in periods.items() if s in sequence.index},
            "sequence": {s: sequence[s] for s in periods if s in sequence.index}}


def _pick_levels(levels: list[Any], ref_arg: Any, test_arg: Any) -> tuple[Any, Any] | None:
    if ref_arg in levels and test_arg in levels and ref_arg != test_arg:
        return test_arg, ref_arg
    if len(levels) != 2:
        return None
    ref = next((lv for lv in levels if str(lv).strip().lower() in _REF_TOKENS), None)
    if ref is None:
        ordered = sorted(levels, key=lambda x: str(x))
        return ordered[0], ordered[1]  # (test, reference)
    test = next(lv for lv in levels if lv != ref)
    return test, ref


def run_bioequivalence(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    dsid, df = require_dataset(ctx, state, args)
    df = df.copy()
    roles = _roles(df, state)
    id_col = next((c for c, r in roles.items() if r == "ID"), None)
    time_col = next((c for c, r in roles.items() if r == "TIME"), None)
    dv_col = next((c for c, r in roles.items() if r == "DV"), None)
    trt_col = args.get("treatment_col") or _find_treatment_col(df, roles)

    if not (id_col and time_col and dv_col):
        raise ValueError("dataset missing required ID/TIME/DV roles for BE")

    if not trt_col:
        status = {"status": "no_treatment_column",
                  "message": ("No test/reference treatment column found. Add a column "
                              "named TRT/TREATMENT/FORMULATION with two levels (e.g. T, R).")}
        return ToolResult(
            summary="Bioequivalence skipped: no treatment column (need test vs reference).",
            action="run_bioequivalence(no_treatment_column)",
            writes={"be_results": status}, result=status)

    loaded = load_observations(df, roles, group_cols=[id_col, trt_col])
    obs = loaded.obs

    levels = list(pd.unique(obs[trt_col].dropna()))
    picked = _pick_levels(levels, args.get("reference"), args.get("test"))
    if picked is None:
        status = {"status": "ambiguous_treatments",
                  "message": f"Need exactly two treatment levels; found {levels}."}
        return ToolResult(
            summary=f"Bioequivalence skipped: treatment levels {levels} not a clean test/reference pair.",
            action="run_bioequivalence(ambiguous_treatments)",
            writes={"be_results": status}, result=status)
    test_lv, ref_lv = picked

    include = bool(args.get("include_unacceptable_lambda_z", False))
    excluded: dict[Any, list[Any]] = {}
    exposures = _exposures_by_treatment(
        obs, id_col=id_col, time_col=time_col, dv_col=dv_col, trt_col=trt_col,
        include_unacceptable_lambda_z=include, excluded=excluded)
    test_subj = exposures.get(test_lv, {})
    ref_subj = exposures.get(ref_lv, {})

    common = sorted(set(test_subj) & set(ref_subj), key=lambda x: str(x))
    # A subject seen under only one treatment is a dropout / incomplete crossover. The design is never changed for
    # it: with any overlap the study is a crossover analysed on its completers, and the excluded ids are reported.
    warnings: list[str] = []
    crossover = None
    if common:
        excluded_ids = sorted((set(test_subj) | set(ref_subj)) - set(common), key=str)
        design_cols = _design_columns(df, obs, id_col, trt_col, test_lv, ref_lv, common)
        if design_cols is not None:
            keep = [sid for sid in common if sid in design_cols["periods"]]
            excluded_ids += [sid for sid in common if sid not in keep]
            common = keep
            crossover = {"sequences": [design_cols["sequence"][sid] for sid in common],
                         "test_periods": [design_cols["periods"][sid][0] for sid in common],
                         "ref_periods": [design_cols["periods"][sid][1] for sid in common]}
        else:
            warnings.append("No usable PERIOD and SEQUENCE columns: crossover analysed as a paired t on log "
                            "differences (df = n - 1) without period or sequence adjustment. Add PERIOD and SEQUENCE "
                            "columns for the standard 2x2 crossover model.")
        test_by_param = {p: [test_subj[s][p] for s in common] for p in _BE_PARAMS}
        ref_by_param = {p: [ref_subj[s][p] for s in common] for p in _BE_PARAMS}
        res = assess_bioequivalence(test_by_param, ref_by_param, paired=True, crossover=crossover)
        if crossover is None:
            res["design"] = PAIRED_LABEL
    else:
        excluded_ids = []
        test_by_param = {p: [v[p] for v in test_subj.values()] for p in _BE_PARAMS}
        ref_by_param = {p: [v[p] for v in ref_subj.values()] for p in _BE_PARAMS}
        res = assess_bioequivalence(test_by_param, ref_by_param, paired=False)
    res.update({"status": "ok", "test_level": str(test_lv), "reference_level": str(ref_lv),
                "n_test": len(test_subj), "n_reference": len(ref_subj), "n_completers": len(common),
                "excluded_subjects": {"n": len(excluded_ids), "subject_ids": excluded_ids,
                                      "reason": "not observed under both treatments, or period/sequence missing "
                                                "(dropout); analysed on completers, design unchanged"},
                "warnings": warnings})
    ex_note = lambda_z_exclusion_note([s for v in excluded.values() for s in v], include)
    ex_note["subjects"] = {"T" if lv == test_lv else "R": excluded.get(lv, []) for lv in (test_lv, ref_lv)}
    res["lambda_z_excluded"] = ex_note

    verdict = "BIOEQUIVALENT" if res.get("bioequivalent") else "NOT bioequivalent"
    return ToolResult(
        summary=(f"Bioequivalence ({res['design']}): {test_lv} vs {ref_lv} — {verdict} "
                 f"(Cmax/AUC GMR + 90% CI vs 80-125%)."),
        action=f"run_bioequivalence({dsid}, {test_lv} vs {ref_lv})",
        writes={"be_results": res}, result=res)


TOOLS = [
    Tool("run_bioequivalence",
         "Assess average bioequivalence: test/reference geometric mean ratio and "
         "90% confidence interval for Cmax and AUC against the 80-125% limits. "
         "Requires a treatment/formulation column.",
         "be",
         {"type": "object",
          "properties": {"dataset_id": {"type": "string"},
                         "treatment_col": {"type": "string"},
                         "test": {"type": "string"},
                         "reference": {"type": "string"},
                         "include_unacceptable_lambda_z": {
                             "type": "boolean",
                             "description": "Use AUC_inf of profiles whose lambda_z fit fails the acceptance "
                                            "criteria. Default false (excluded and reported)."}},
          "required": []},
         run_bioequivalence),
]
