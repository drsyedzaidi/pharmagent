"""Bioequivalence Agent tools: test/reference GMR + 90% CI from NCA exposures.

Computes per-(subject, treatment) NCA exposures, then average bioequivalence for Cmax and AUC
under the safety contract in app.compute.be_contract: the design, regulatory profile and test
and reference levels are explicit, and every excluded value is listed.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from app.compute.be_contract import PROFILES, BEContractError, assess_average_be
from app.compute.nca import Profile, nca_subject
from app.core.pharmstate import PharmState
from app.core.schema_extractor import detect_roles
from app.tools.base import Tool, ToolContext, ToolResult, require_dataset

_TRT_NAMES = {"trt", "treat", "treatment", "form", "formulation", "drug", "period_trt"}
_BE_PARAMS = ("Cmax", "AUC_last", "AUC_inf")
_REQUIRED = ("design", "profile", "test", "reference")


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


def _py(v: Any) -> Any:
    """numpy scalars -> plain Python, so ledger entries serialise and compare as the data shows them."""
    return v.item() if hasattr(v, "item") else v


def _exposures_by_treatment(
    obs: pd.DataFrame, *, id_col: str, time_col: str, dv_col: str, trt_col: str,
) -> tuple[list[dict], list[dict]]:
    """Per-(subject, treatment) NCA exposures as contract records, plus ledger entries for profiles too short to
    analyse (fewer than 3 samples): those subjects are left out of every parameter for that treatment."""
    records: list[dict] = []
    ledger: list[dict] = []
    for (trt, sid), g in obs.groupby([trt_col, id_col]):
        g = g.dropna(subset=[time_col, dv_col]).sort_values(time_col)
        if len(g) < 3:
            ledger.append({"subject": _py(sid), "treatment": _py(trt), "parameter": "*",
                           "reason": "fewer_than_3_samples", "original_value": int(len(g)), "prespecified": False})
            continue
        prof = Profile(subject=sid, time=g[time_col].to_numpy(float),
                       conc=g[dv_col].to_numpy(float), dose=float("nan"))
        p = nca_subject(prof)
        records.append({"subject": _py(sid), "treatment": _py(trt), "params": {k: p.get(k) for k in _BE_PARAMS}})
    return records, ledger


def _row_ledger(obs: pd.DataFrame, raw_dv: pd.Series, *, id_col: str, time_col: str, dv_col: str,
                trt_col: str) -> tuple[pd.DataFrame, list[dict]]:
    """Drop observation rows that cannot be concentrations, listing each one with its original value."""
    ledger: list[dict] = []
    keep = []
    for idx, row in obs.iterrows():
        t, dv, raw = row[time_col], row[dv_col], raw_dv.loc[idx]
        reason = None
        if pd.isna(t):
            reason = "missing_time"
        elif pd.isna(dv):
            reason = "missing_dv" if pd.isna(raw) else "non_numeric_dv"
        elif not math.isfinite(float(dv)):
            reason = "non_finite_dv"
        if reason is None:
            keep.append(idx)
            continue
        ledger.append({"subject": _py(row[id_col]), "treatment": _py(row[trt_col]), "parameter": "*",
                       "reason": reason, "original_value": _py(raw), "time": None if pd.isna(t) else float(t),
                       "prespecified": False})
    return obs.loc[keep], ledger


def _refusal(status: str, message: str, **extra: Any) -> ToolResult:
    payload = {"status": status, "message": message, **extra}
    return ToolResult(summary=f"Bioequivalence not run: {message}", action=f"run_bioequivalence({status})",
                      writes={"be_results": payload}, result=payload)


def _match_level(levels: list[Any], wanted: Any) -> Any:
    """The data's own level equal to the requested one, compared as text (a CSV may read 1 where the user typed
    "1"). Unknown levels are returned unchanged so the contract names them in its refusal; text matching two
    different data levels is refused."""
    hits = [lv for lv in levels if str(lv) == str(wanted)]
    if len(hits) > 1:
        raise BEContractError(f"treatment level {wanted!r} is ambiguous: it matches {hits!r}")
    return hits[0] if hits else wanted


def run_bioequivalence(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Average BE under the safety contract (app.compute.be_contract): design, profile, test and reference are
    required, nothing is guessed, and every excluded value is listed."""
    dsid, df = require_dataset(ctx, state, args)
    missing = [k for k in _REQUIRED if args.get(k) in (None, "")]
    if args.get("design") == "crossover" and not args.get("period_col"):
        missing.append("period_col")       # without periods, replicate or duplicated rows cannot be told apart
    if missing:
        return _refusal("missing_required_args",
                        f"state {', '.join(missing)} explicitly (design: crossover or parallel; profile: "
                        f"{', '.join(sorted(PROFILES))}; test and reference: treatment levels)",
                        missing=missing)
    if args.get("group_col"):
        return _refusal("unsupported_design", "multi-group studies need a group-aware model, which is not "
                        "implemented; analyse each group separately only if the protocol allows it")
    df = df.copy()
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
        return _refusal("no_treatment_column", "no test/reference treatment column found; add a column named "
                        "TRT/TREATMENT/FORMULATION or pass treatment_col")

    raw_dv = df[dv_col].copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[dv_col] = pd.to_numeric(df[dv_col], errors="coerce")
    if evid_col:
        obs = df[pd.to_numeric(df[evid_col], errors="coerce").fillna(0) == 0]
    elif amt_col:
        obs = df[pd.to_numeric(df[amt_col], errors="coerce").fillna(0) == 0]
    else:
        obs = df

    if obs[id_col].isna().any() or obs[trt_col].isna().any():
        return _refusal("invalid_arguments", "observation rows without a subject id or treatment cannot be "
                        "assigned; fix the data")
    if obs.duplicated([id_col, trt_col, time_col]).any():
        return _refusal("unsupported_design", "duplicate sampling rows (same subject, treatment and time): "
                        "replicate or duplicated data is not pooled")
    period_col = args.get("period_col")
    if period_col:
        if period_col not in obs.columns:
            return _refusal("invalid_arguments", f"period_col {period_col!r} is not a column")
        if (obs.groupby([id_col, trt_col])[period_col].nunique() > 1).any():
            return _refusal("unsupported_design", "a subject received the same treatment in more than one period: "
                            "replicate designs are not supported")
        if args["design"] == "crossover":
            per = obs.groupby([id_col, trt_col])[period_col].first().reset_index()
            if per.duplicated([id_col, period_col]).any():
                return _refusal("invalid_arguments", "a subject has test and reference in the same period, which "
                                "is not a crossover")

    obs, dropped = _row_ledger(obs, raw_dv, id_col=id_col, time_col=time_col, dv_col=dv_col, trt_col=trt_col)
    levels = list(pd.unique(obs[trt_col]))
    try:
        test_lv, ref_lv = _match_level(levels, args["test"]), _match_level(levels, args["reference"])
        records, short = _exposures_by_treatment(obs, id_col=id_col, time_col=time_col, dv_col=dv_col,
                                                 trt_col=trt_col)
        res = assess_average_be(
            records, design=args["design"], profile=args["profile"], test=_py(test_lv), reference=_py(ref_lv),
            prespecified_exclusions=args.get("prespecified_exclusions"),
            limits=args.get("limits"), limits_justification=args.get("limits_justification"))
    except BEContractError as exc:
        status = "unsupported_design" if "replicate" in str(exc) else "invalid_arguments"
        return _refusal(status, str(exc))
    res = {**res, "status": "ok", "exclusions": dropped + short + res["exclusions"],
           "test_level": str(test_lv), "reference_level": str(ref_lv)}
    verdict = "BIOEQUIVALENT" if res["bioequivalent"] else "NOT bioequivalent"
    return ToolResult(
        summary=(f"Bioequivalence ({res['design']}, profile {res['profile']}): {test_lv} vs {ref_lv} - {verdict}; "
                 f"{len(res['exclusions'])} excluded value(s) listed; model is interim (no period/sequence terms)."),
        action=f"run_bioequivalence({dsid}, {res['profile']}, {res['design']}, {test_lv} vs {ref_lv})",
        writes={"be_results": res}, result=res)


TOOLS = [
    Tool("run_bioequivalence",
         "Assess average bioequivalence (test/reference geometric mean ratio and 90% CI for Cmax and AUC). "
         "REQUIRES design ('crossover' or 'parallel'), profile ('fda_2026', 'ich_m13a' or 'ema_2010'), and the "
         "test and reference treatment levels; nothing is inferred. Optional: treatment_col, period_col, "
         "prespecified_exclusions [{subject, reason}], limits [lower, upper] with limits_justification.",
         "be",
         {"type": "object",
          "properties": {"dataset_id": {"type": "string"},
                         "design": {"type": "string", "enum": ["crossover", "parallel"]},
                         "profile": {"type": "string", "enum": ["fda_2026", "ich_m13a", "ema_2010"]},
                         "treatment_col": {"type": "string"},
                         "period_col": {"type": "string"},
                         "group_col": {"type": "string"},
                         "test": {"type": "string"},
                         "reference": {"type": "string"},
                         "prespecified_exclusions": {"type": "array", "items": {"type": "object"}},
                         "limits": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                         "limits_justification": {"type": "string"}},
          "required": []},
         run_bioequivalence),
]
