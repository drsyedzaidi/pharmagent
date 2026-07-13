"""Bioequivalence Agent tools: test/reference GMR + 90% CI from NCA exposures.

Computes per-(subject, treatment) NCA exposures, then the average-bioequivalence
ratio and confidence interval for Cmax and AUC against the 80-125% limits.
Requires a treatment/formulation column distinguishing test from reference.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from app.compute.bioequivalence import assess_bioequivalence
from app.compute.nca import Profile, nca_subject
from app.core.pharmstate import PharmState
from app.core.schema_extractor import detect_roles
from app.tools.base import Tool, ToolContext, ToolResult

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
) -> dict[Any, dict[Any, dict[str, float]]]:
    """Return {treatment: {subject: {Cmax, AUC_last, AUC_inf}}}."""
    out: dict[Any, dict[Any, dict[str, float]]] = {}
    for (trt, sid), g in obs.groupby([trt_col, id_col]):
        g = g.dropna(subset=[time_col, dv_col]).sort_values(time_col)
        if len(g) < 3:
            continue
        prof = Profile(subject=sid, time=g[time_col].to_numpy(float),
                       conc=g[dv_col].to_numpy(float), dose=float("nan"))
        p = nca_subject(prof)
        out.setdefault(trt, {})[sid] = {k: p.get(k) for k in _BE_PARAMS}
    return out


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
    dsid = args.get("dataset_id") or state.dataset_id
    df = ctx.dataset_store[dsid].copy()
    roles = _roles(df, state)
    id_col = next((c for c, r in roles.items() if r == "ID"), None)
    time_col = next((c for c, r in roles.items() if r == "TIME"), None)
    dv_col = next((c for c, r in roles.items() if r == "DV"), None)
    amt_col = next((c for c, r in roles.items() if r == "AMT"), None)
    evid_col = next((c for c, r in roles.items() if r == "EVID"), None)
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

    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[dv_col] = pd.to_numeric(df[dv_col], errors="coerce")
    if evid_col:
        ev = pd.to_numeric(df[evid_col], errors="coerce").fillna(0)
        obs = df[ev == 0]
    elif amt_col:
        amt = pd.to_numeric(df[amt_col], errors="coerce").fillna(0)
        obs = df[amt == 0]
    else:
        obs = df

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

    exposures = _exposures_by_treatment(
        obs, id_col=id_col, time_col=time_col, dv_col=dv_col, trt_col=trt_col)
    test_subj = exposures.get(test_lv, {})
    ref_subj = exposures.get(ref_lv, {})

    common = sorted(set(test_subj) & set(ref_subj), key=lambda x: str(x))
    paired = len(common) >= 2 and len(common) == len(test_subj) == len(ref_subj)

    if paired:
        test_by_param = {p: [test_subj[s][p] for s in common] for p in _BE_PARAMS}
        ref_by_param = {p: [ref_subj[s][p] for s in common] for p in _BE_PARAMS}
    else:
        test_by_param = {p: [v[p] for v in test_subj.values()] for p in _BE_PARAMS}
        ref_by_param = {p: [v[p] for v in ref_subj.values()] for p in _BE_PARAMS}

    res = assess_bioequivalence(test_by_param, ref_by_param, paired=paired)
    res.update({"status": "ok", "test_level": str(test_lv), "reference_level": str(ref_lv),
                "n_test": len(test_subj), "n_reference": len(ref_subj)})

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
                         "reference": {"type": "string"}},
          "required": []},
         run_bioequivalence),
]
