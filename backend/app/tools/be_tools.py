"""Bioequivalence Agent tools: test/reference GMR + 90% CI from NCA exposures.

Computes per-(subject, treatment) NCA exposures, then average bioequivalence for Cmax and AUC
under the safety contract in app.compute.be_contract: the design, regulatory profile and test
and reference levels are explicit, and every excluded value is listed.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from app.compute.be_contract import (
    PROFILES,
    BEContractError,
    BEUnsupportedDesign,
    assess_average_be,
    ordered_periods,
    period_label,
)
from app.compute.be_qc import natural_key
from app.compute.nca import Profile, nca_subject
from app.core.pharmstate import PharmState
from app.core.schema_extractor import detect_roles
from app.tools.base import Tool, ToolContext, ToolResult, require_dataset

_TRT_NAMES = {"trt", "treat", "treatment", "form", "formulation", "drug", "period_trt"}
_BE_PARAMS = ("Cmax", "AUC_last", "AUC_inf")
_REQUIRED = ("design", "profile", "test", "reference")
_NO_CONCLUSION = "No bioequivalence conclusion was calculated."


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
    obs: pd.DataFrame, *, id_col: str, time_col: str, dv_col: str, trt_col: str, period_col: str | None = None,
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
        rec = {"subject": _py(sid), "treatment": _py(trt), "params": {k: p.get(k) for k in _BE_PARAMS}}
        rec.update(_qc_inputs(g, time_col, dv_col))
        if period_col:                       # one value per (subject, treatment): _check_periods refused anything else
            rec["period"] = _py(g[period_col].iloc[0])
        records.append(rec)
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
                       "reason": reason, "original_value": None if pd.isna(raw) else _py(raw),  # strict JSON
                       "time": None if pd.isna(t) else float(t),
                       "prespecified": False})
    return obs.loc[keep], ledger


def _check_periods(obs: pd.DataFrame, period_col: str, *, id_col: str, trt_col: str, roles: set[str],
                   crossover: bool, compared: tuple[Any, Any]) -> ToolResult | None:
    """The period column must be its own column, set on every observation row, give each subject one period per
    treatment, and (crossover) put test and reference in different periods in at least two sequences. Only the
    compared treatments are examined: any other arm is ignored by the analysis."""
    if period_col not in obs.columns:
        return _refusal("invalid_arguments", f"period_col {period_col!r} is not a column")
    if period_col in roles:
        return _refusal("invalid_arguments", f"period_col {period_col!r} is the subject, treatment, time or "
                        "concentration column")
    if obs[period_col].isna().any():
        return _refusal("invalid_arguments", "observation rows without a period cannot be assigned; a missing "
                        "period could hide a repeated treatment")
    obs = obs[obs[trt_col].astype(str).isin({str(c) for c in compared})]
    try:                                    # the same label rules (and ordering) as the contract applies
        labels = {period_label(_py(v)) for v in obs[period_col].unique()}
        order = ordered_periods(labels)
        rank = {lab: i for i, lab in enumerate(order)}
    except BEContractError as exc:
        return _refusal("invalid_arguments", f"period column {period_col!r}: {exc}")
    if (obs.groupby([id_col, trt_col])[period_col].nunique() > 1).any():
        return _refusal("unsupported_design", "a subject has more than one period label for the same treatment: a "
                        "replicate design (not supported) or inconsistent period labels")
    if not crossover:
        return None
    per = obs.groupby([id_col, trt_col], as_index=False)[period_col].first()
    if per.duplicated([id_col, period_col]).any():
        return _refusal("invalid_arguments", "a subject has test and reference in the same period, which is not a "
                        "crossover")
    if len(order) > 2:
        return _refusal("unsupported_design", f"only a two-period crossover is supported; found {len(order)} periods")
    per = per.assign(_rank=per[period_col].map(lambda v: rank[period_label(_py(v))]))
    sequences = {tuple(g.sort_values("_rank")[trt_col].map(str)) for _, g in per.groupby(id_col) if len(g) > 1}
    if len(sequences) < 2:
        return _refusal("invalid_arguments", "every subject received the treatments in the same order: one "
                        "sequence is not a randomised crossover, and period and formulation cannot be separated")
    return None


def _missing_args(df: pd.DataFrame, state: PharmState, args: dict[str, Any], missing: list[str]) -> ToolResult:
    """Refuse without defaults, but say exactly what to supply: every missing field, the accepted values (the
    treatment levels are read from the data when a treatment column is found) and a minimal valid call."""
    trt_col = args.get("treatment_col") or _find_treatment_col(df, _roles(df, state))
    levels = (sorted(map(str, pd.unique(df[trt_col].dropna()))) if trt_col and trt_col in df.columns else [])
    accepted = {"design": ["crossover", "parallel"], "profile": sorted(PROFILES),
                "test": levels or ["<a treatment level>"], "reference": levels or ["<a treatment level>"],
                "period_col": "a column naming each subject's period (required for crossover)"}
    example = {"design": "crossover", "profile": "fda_2026",
               "test": levels[1] if len(levels) > 1 else "T", "reference": levels[0] if levels else "R",
               "period_col": "PERIOD"}
    return _refusal("missing_required_args", f"missing required argument(s): {', '.join(missing)}; nothing is "
                    "inferred", missing=missing, accepted_values=accepted, example=example)


def _qc_inputs(g: pd.DataFrame, time_col: str, dv_col: str) -> dict[str, Any]:
    """The pre-dose concentration (the highest sample at or before time 0, or None when there is none) and whether
    Cmax falls on the first post-dose sample, for one subject and treatment (rows sorted by time)."""
    pre = g[g[time_col] <= 0]
    post = g[g[time_col] > 0]
    # every sample at or before time 0 is a pre-dose sample and the highest decides: any one above 5 % of Cmax
    # signals carry-over; more than one is reported
    out: dict[str, Any] = {"predose": float(pre[dv_col].max()) if len(pre) else None, "predose_samples": int(len(pre))}
    if len(pre):
        out["predose_time"] = float(pre.loc[pre[dv_col].idxmax(), time_col])        # the sample the rule used
    if len(post):
        peak_time = post.loc[post[dv_col].idxmax(), time_col]
        out["cmax_first_point"] = bool(peak_time == post[time_col].min())
    return out


def _merge_pre_contract_entries(res: dict[str, Any], entries: list[dict[str, Any]]) -> None:
    """Rows and records removed before the contract saw the data (non-numeric or missing values, profiles with fewer
    than 3 samples) belong in the disposition table too: a note for a dropped row, a reason (and a row, when the
    subject vanished entirely) for a dropped record."""
    rows = {r["subject"]: r for r in res["dispositions"]}
    names = list(res["parameters"])
    for e in entries:
        subject = e["subject"]
        row = rows.get(subject)
        if row is None:
            row = {"subject": subject, "sequence": None, "arm": None, "periods": {}, "qc_flags": [],
                   "disposition": "excluded", "parameters": {n: {"status": "excluded", "reasons": []} for n in names}}
            rows[subject] = row
        if e["reason"] == "fewer_than_3_samples":
            for info in row["parameters"].values():
                if info["status"] == "excluded" and e["reason"] not in info["reasons"]:
                    info["reasons"].append(e["reason"])
        t = e.get("time")
        row.setdefault("notes", []).append(f"{e['reason']}" + (f" at t={t}" if t is not None else "")
                                           + f" ({e['treatment']})")
    res["dispositions"] = sorted(rows.values(), key=lambda r: natural_key(r["subject"]))


def _refusal(status: str, message: str, **extra: Any) -> ToolResult:
    payload = {"status": status, "message": f"{message}. {_NO_CONCLUSION}", **extra}
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
        return _missing_args(df, state, args, missing)
    if args.get("group_col"):
        return _refusal("unsupported_design", "multi-group studies need a group-aware model, which is not "
                        "implemented; analyse each group separately only if the protocol allows it")
    df = df.copy().reset_index(drop=True)      # a unique index: ledger rows are looked up by label
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
    timed = obs[obs[time_col].notna()]          # rows without a time are ledgered below, not called duplicates
    if timed.duplicated([id_col, trt_col, time_col]).any():
        return _refusal("unsupported_design", "duplicate sampling rows (same subject, treatment and time): "
                        "replicate or duplicated data is not pooled")
    present = sorted({str(x) for x in obs[trt_col].dropna().unique()})
    for name in ("test", "reference"):
        if str(args[name]) not in present:
            return _refusal("invalid_arguments", f"{name} level {args[name]!r} is not in the data (levels: {present})")
    period_col = args.get("period_col")
    if period_col:
        refusal = _check_periods(obs, period_col, id_col=id_col, trt_col=trt_col, roles={
            id_col, trt_col, time_col, dv_col}, crossover=args["design"] == "crossover",
            compared=(args["test"], args["reference"]))
        if refusal:
            return refusal

    obs, dropped = _row_ledger(obs, raw_dv, id_col=id_col, time_col=time_col, dv_col=dv_col, trt_col=trt_col)
    levels = list(pd.unique(obs[trt_col]))
    try:
        test_lv, ref_lv = _match_level(levels, args["test"]), _match_level(levels, args["reference"])
        records, short = _exposures_by_treatment(obs, id_col=id_col, time_col=time_col, dv_col=dv_col,
                                                 trt_col=trt_col, period_col=period_col)
        res = assess_average_be(
            records, design=args["design"], profile=args["profile"], test=_py(test_lv), reference=_py(ref_lv),
            prespecified_exclusions=args.get("prespecified_exclusions"),
            limits=args.get("limits"), limits_justification=args.get("limits_justification"),
            variance=args.get("variance"), decision_parameters=args.get("decision_parameters"),
            decision_justification=args.get("decision_justification"),
            predose_policy=args.get("predose_policy") or "profile_rule",
            predose_justification=args.get("predose_justification"))
    except BEContractError as exc:
        return _refusal("unsupported_design" if isinstance(exc, BEUnsupportedDesign) else "invalid_arguments", str(exc))
    res = {**res, "status": "ok", "exclusions": dropped + short + res["exclusions"],
           "test_level": str(test_lv), "reference_level": str(ref_lv)}
    _merge_pre_contract_entries(res, dropped + short)
    verdict = "BIOEQUIVALENT" if res["bioequivalent"] else "NOT bioequivalent"
    return ToolResult(
        summary=(f"Bioequivalence ({res['design']}, profile {res['profile']}): {test_lv} vs {ref_lv} - {verdict}; "
                 f"{len(res['exclusions'])} excluded value(s) listed, {res['qc_summary']['excluded']} excluded by QC; "
                 "interim model, validation pending."),
        action=f"run_bioequivalence({dsid}, {res['profile']}, {res['design']}, {test_lv} vs {ref_lv})",
        writes={"be_results": res}, result=res)


TOOLS = [
    Tool("run_bioequivalence",
         "Assess average bioequivalence (test/reference geometric mean ratio and 90% CI for Cmax and AUC). "
         "REQUIRES design ('crossover' or 'parallel'), profile ('fda_2026', 'ich_m13a' or 'ema_2010'), and the "
         "test and reference treatment levels; nothing is inferred. A crossover is analysed with the 2x2 "
         "fixed-effects model (sequence, subject within sequence, period, formulation) and needs period_col. "
         "Optional: treatment_col, variance ('pooled' or 'welch', parallel only), "
         "prespecified_exclusions [{subject, reason}], limits [lower, upper] with limits_justification. Cmax and "
         "AUC_last decide (AUC_inf is supportive unless named in decision_parameters with a justification); a "
         "pre-dose above 5% of Cmax drops that subject-period (predose_policy 'flag_only' needs a justification).",
         "be",
         {"type": "object",
          "properties": {"dataset_id": {"type": "string"},
                         "design": {"type": "string", "enum": ["crossover", "parallel"]},
                         "profile": {"type": "string", "enum": ["fda_2026", "ich_m13a", "ema_2010"]},
                         "treatment_col": {"type": "string"},
                         "period_col": {"type": "string"},
                         "group_col": {"type": "string"},
                         "variance": {"type": "string", "enum": ["pooled", "welch"]},
                         "decision_parameters": {"type": "array", "items": {"type": "string"}},
                         "decision_justification": {"type": "string"},
                         "predose_policy": {"type": "string", "enum": ["profile_rule", "flag_only"]},
                         "predose_justification": {"type": "string"},
                         "test": {"type": "string"},
                         "reference": {"type": "string"},
                         "prespecified_exclusions": {"type": "array", "items": {"type": "object"}},
                         "limits": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                         "limits_justification": {"type": "string"}},
          "required": []},
         run_bioequivalence),
]
