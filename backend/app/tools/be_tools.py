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
from app.compute.nca import INF_DERIVED, Profile, lambda_z_exclusion_note, nca_subject
from app.core.obs_loader import load_observations
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


def _col(df: pd.DataFrame, names: set[str]) -> str | None:
    return next((c for c in df.columns if str(c).strip().lower() in names), None)


def _py(v: Any) -> Any:
    """numpy scalars -> plain Python, so ledger entries serialise and compare as the data shows them."""
    return v.item() if hasattr(v, "item") else v


def _exposures_by_treatment(
    obs: pd.DataFrame, *, id_col: str, time_col: str, dv_col: str, trt_col: str,
    include_unacceptable_lambda_z: bool = False,
) -> tuple[list[dict], list[dict], list[tuple[Any, Any]]]:
    """Per-(subject, treatment) NCA exposures as contract records, ledger entries for profiles too short to
    analyse (fewer than 3 samples: left out of every parameter for that treatment) and the (subject, treatment)
    pairs whose AUC_inf was withheld because the terminal fit is unacceptable (unless
    ``include_unacceptable_lambda_z``)."""
    records: list[dict] = []
    ledger: list[dict] = []
    withheld: list[tuple[Any, Any]] = []
    for (trt, sid), g in obs.groupby([trt_col, id_col]):
        g = g.dropna(subset=[time_col, dv_col]).sort_values(time_col)
        if len(g) < 3:
            ledger.append({"subject": _py(sid), "treatment": _py(trt), "parameter": "*",
                           "reason": "fewer_than_3_samples", "original_value": int(len(g)), "prespecified": False})
            continue
        prof = Profile(subject=sid, time=g[time_col].to_numpy(float),
                       conc=g[dv_col].to_numpy(float), dose=float("nan"))
        p = nca_subject(prof)
        vals = {k: p.get(k) for k in _BE_PARAMS}
        if p.get("lambda_z_acceptable") is False and not include_unacceptable_lambda_z:
            vals.update({k: None for k in _BE_PARAMS if k in INF_DERIVED})
            withheld.append((_py(sid), _py(trt)))
        records.append({"subject": _py(sid), "treatment": _py(trt), "params": vals})
    return records, ledger, withheld


_SEQ_NAMES = {"sequence", "seq", "sequ"}


def _crossover_layout(obs: pd.DataFrame, *, id_col: str, trt_col: str, period_col: str, seq_col: str | None,
                      test_lv: Any, ref_lv: Any) -> tuple[dict[Any, tuple[Any, Any, Any]] | None, str | None]:
    """{subject: (sequence, test period, reference period)} for the 2x2 crossover model, or (None, why) when the
    sequence column is missing or the layout is not two periods in at least two sequences."""
    if seq_col is None or seq_col not in obs.columns:
        return None, "no SEQUENCE column (pass sequence_col)"
    first_period = obs.dropna(subset=[period_col]).groupby([id_col, trt_col])[period_col].first()
    sequence = obs.dropna(subset=[seq_col]).groupby(id_col)[seq_col].first()
    layout: dict[Any, tuple[Any, Any, Any]] = {}
    for sid in sequence.index:
        if (sid, test_lv) in first_period.index and (sid, ref_lv) in first_period.index:
            tp, rp = first_period[(sid, test_lv)], first_period[(sid, ref_lv)]
            if tp != rp:
                layout[_py(sid)] = (_py(sequence[sid]), _py(tp), _py(rp))
    if len({p for v in layout.values() for p in v[1:]}) != 2 or len({v[0] for v in layout.values()}) < 2:
        return None, "the periods are not two levels in at least two sequences"
    return layout, None


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
                   crossover: bool) -> ToolResult | None:
    """The period column must be its own column, set on every observation row, give each subject one period per
    treatment, and (crossover) put test and reference in different periods in at least two sequences."""
    if period_col not in obs.columns:
        return _refusal("invalid_arguments", f"period_col {period_col!r} is not a column")
    if period_col in roles:
        return _refusal("invalid_arguments", f"period_col {period_col!r} is the subject, treatment, time or "
                        "concentration column")
    if obs[period_col].isna().any():
        return _refusal("invalid_arguments", "observation rows without a period cannot be assigned; a missing "
                        "period could hide a repeated treatment")
    if (obs.groupby([id_col, trt_col])[period_col].nunique() > 1).any():
        return _refusal("unsupported_design", "a subject received the same treatment in more than one period: "
                        "replicate designs are not supported")
    if not crossover:
        return None
    per = obs.groupby([id_col, trt_col], as_index=False)[period_col].first()
    if per.duplicated([id_col, period_col]).any():
        return _refusal("invalid_arguments", "a subject has test and reference in the same period, which is not a "
                        "crossover")
    sequences = {tuple(g.sort_values(period_col)[trt_col].map(str)) for _, g in per.groupby(id_col) if len(g) > 1}
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
    trt_col = args.get("treatment_col") or _find_treatment_col(df, roles)
    if not (id_col and time_col and dv_col):
        raise ValueError("dataset missing required ID/TIME/DV roles for BE")
    if not trt_col:
        return _refusal("no_treatment_column", "no test/reference treatment column found; add a column named "
                        "TRT/TREATMENT/FORMULATION or pass treatment_col")

    raw_dv = df[dv_col].copy()
    # One loader for every profile analysis (EVID/MDV rows removed, below-LOQ rule M1, records before the dose
    # dropped); ``raw_obs`` is the observation rows before M1 so the ledger still lists every unusable row.
    loaded = load_observations(df, roles, group_cols=[id_col, trt_col])
    raw = loaded.raw_obs
    if raw[id_col].isna().any() or raw[trt_col].isna().any():
        return _refusal("invalid_arguments", "observation rows without a subject id or treatment cannot be "
                        "assigned; fix the data")
    timed = raw[raw[time_col].notna()]          # rows without a time are ledgered below, not called duplicates
    if timed.duplicated([id_col, trt_col, time_col]).any():
        return _refusal("unsupported_design", "duplicate sampling rows (same subject, treatment and time): "
                        "replicate or duplicated data is not pooled")
    period_col = args.get("period_col")
    if period_col:
        refusal = _check_periods(raw, period_col, id_col=id_col, trt_col=trt_col, roles={
            id_col, trt_col, time_col, dv_col}, crossover=args["design"] == "crossover")
        if refusal:
            return refusal

    # the loader turns +-inf into missing; the ledger classifies the DV as it was supplied
    shown = raw.assign(**{dv_col: pd.to_numeric(raw_dv.loc[raw.index], errors="coerce")})
    _, dropped = _row_ledger(shown, raw_dv, id_col=id_col, time_col=time_col, dv_col=dv_col, trt_col=trt_col)
    obs = loaded.obs
    levels = list(pd.unique(obs[trt_col]))
    try:
        test_lv, ref_lv = _match_level(levels, args["test"]), _match_level(levels, args["reference"])
        include_lz = bool(args.get("include_unacceptable_lambda_z", False))
        records, short, withheld = _exposures_by_treatment(
            obs, id_col=id_col, time_col=time_col, dv_col=dv_col, trt_col=trt_col,
            include_unacceptable_lambda_z=include_lz)
        layout, layout_note = None, None
        if args["design"] == "crossover":
            seq_col = args.get("sequence_col") or _col(df, _SEQ_NAMES)
            layout, layout_note = _crossover_layout(obs, id_col=id_col, trt_col=trt_col, period_col=period_col,
                                                    seq_col=seq_col, test_lv=test_lv, ref_lv=ref_lv)
        res = assess_average_be(
            records, design=args["design"], profile=args["profile"], test=_py(test_lv), reference=_py(ref_lv),
            prespecified_exclusions=args.get("prespecified_exclusions"),
            limits=args.get("limits"), limits_justification=args.get("limits_justification"), layout=layout)
    except BEContractError as exc:
        status = "unsupported_design" if "replicate" in str(exc) else "invalid_arguments"
        return _refusal(status, str(exc))
    lz_pairs = set(withheld)
    ledger = [{**e, "reason": "lambda_z_unacceptable"}
              if e["parameter"] in INF_DERIVED and (e["subject"], e["treatment"]) in lz_pairs
              and e["reason"] == "missing_value" else e for e in res["exclusions"]]
    exclusions = dropped + short + ledger
    out_ids = sorted({e["subject"] for e in exclusions
                      if e["reason"] in {"fewer_than_3_samples", "no_evaluable_counterpart", "no_period_or_sequence"}},
                     key=str)
    lz_note = lambda_z_exclusion_note(sorted({s for s, _ in withheld}, key=str), include_lz)
    lz_note["subjects"] = {"T": [s for s, t in withheld if t == _py(test_lv)],
                           "R": [s for s, t in withheld if t == _py(ref_lv)]}
    warnings = ([f"Crossover analysed as a paired t on log differences without period or sequence terms: "
                 f"{layout_note}."] if args["design"] == "crossover" and layout is None else [])
    res = {**res, "status": "ok", "exclusions": exclusions,
           "test_level": str(test_lv), "reference_level": str(ref_lv),
           "excluded_subjects": {"n": len(out_ids), "subject_ids": out_ids,
                                 "reason": "not evaluable under both treatments, or period/sequence missing; "
                                           "the design is never changed for them"},
           "n_completers": len({r["subject"] for r in records if r["treatment"] == _py(test_lv)}
                               & {r["subject"] for r in records if r["treatment"] == _py(ref_lv)}
                               - set(out_ids)),
           "lambda_z_excluded": lz_note, "warnings": warnings,
           "n_blq_records": loaded.n_blq,
           "blq_rule": "below-LOQ records handled by rule M1 (leading BLQ = 0, later BLQ excluded)"}
    verdict = "BIOEQUIVALENT" if res["bioequivalent"] else "NOT bioequivalent"
    return ToolResult(
        summary=(f"Bioequivalence ({res['design']}, profile {res['profile']}): {test_lv} vs {ref_lv} - {verdict}; "
                 f"{len(res['exclusions'])} excluded value(s) listed; model: {res['model'].split(':')[0]}."),
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
                         "sequence_col": {"type": "string"},
                         "include_unacceptable_lambda_z": {
                             "type": "boolean",
                             "description": "Use AUC_inf of profiles whose lambda_z fit fails the acceptance "
                                            "criteria. Default false (withheld, ledgered and reported)."},
                         "group_col": {"type": "string"},
                         "test": {"type": "string"},
                         "reference": {"type": "string"},
                         "prespecified_exclusions": {"type": "array", "items": {"type": "object"}},
                         "limits": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                         "limits_justification": {"type": "string"}},
          "required": []},
         run_bioequivalence),
]
