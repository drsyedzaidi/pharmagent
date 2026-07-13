"""Data Manager tools: load, profile, validate, visualize datasets."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import settings
from app.core.pharmstate import PharmState
from app.core.schema_extractor import detect_roles, extract_schema
from app.tools.base import Tool, ToolContext, ToolResult


def _safe_path(path: str) -> Path:
    """Resolve and confine a dataset path to the allowed data roots.

    Prevents arbitrary server-side file reads (path traversal / absolute paths
    outside the configured data and sample directories).
    """
    p = Path(path).resolve()
    roots = settings.allowed_data_dirs
    if not any(p == r or r in p.parents for r in roots):
        raise ValueError(
            f"path not permitted: dataset files must live under {', '.join(str(r) for r in roots)}")
    if not p.exists():
        raise ValueError(f"file not found: {p}")
    return p


def _read(path: str) -> pd.DataFrame:
    p = _safe_path(path)
    suf = p.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(p)
    if suf in (".xpt",):
        return pd.read_sas(p, format="xport")
    if suf == ".sas7bdat":
        return pd.read_sas(p)
    raise ValueError(f"unsupported file type: {suf}")


def _roles(df: pd.DataFrame, meta: dict | None) -> dict[str, str]:
    if meta and meta.get("detected_roles"):
        return meta["detected_roles"]
    return detect_roles(list(df.columns))


def load_dataset(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    from app.core.provenance import file_sha256  # local: keep core deps light
    path = args["path"]
    df = _read(path)
    dataset_id = args.get("dataset_id") or f"ds_{uuid.uuid4().hex[:8]}"
    ctx.dataset_store[dataset_id] = df  # raw df stays server-side
    meta = extract_schema(df, dataset_id=dataset_id)
    meta["dataset_sha256"] = file_sha256(path)  # data-integrity fingerprint (ALCOA+)
    return ToolResult(
        summary=f"Loaded {dataset_id}: {meta['n_records']} records, "
                f"{meta['n_subjects']} subjects, {meta['n_columns']} columns.",
        action=f"load_dataset({path})",
        writes={"dataset_id": dataset_id, "dataset_path": path, "dataset_metadata": meta},
        result=meta,  # metadata-only — safe for the LLM
    )


def profile_pk_dataset(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    dsid = args.get("dataset_id") or state.dataset_id
    df = ctx.dataset_store[dsid]
    meta = state.dataset_metadata or extract_schema(df, dataset_id=dsid)
    roles = _roles(df, meta)
    id_col = next((c for c, r in roles.items() if r == "ID"), None)
    dv_col = next((c for c, r in roles.items() if r == "DV"), None)

    dv = pd.to_numeric(df[dv_col], errors="coerce") if dv_col else pd.Series(dtype=float)
    n_obs = int(dv.notna().sum()) if dv_col else 0
    blq_pct = round(100.0 * float((dv <= 0).sum()) / max(n_obs, 1), 2) if dv_col else None
    missing_pct = round(100.0 * float(df.isna().sum().sum()) / (df.shape[0] * df.shape[1]), 2)

    sparse = []
    if id_col and dv_col:
        counts = df.groupby(id_col)[dv_col].apply(lambda s: pd.to_numeric(s, errors="coerce").notna().sum())
        sparse = [str(k) for k, v in counts.items() if v < 3]

    quality = {
        "n_observations": n_obs,
        "blq_pct": blq_pct,
        "total_missing_pct": missing_pct,
        "sparse_subjects": sparse,
        "n_sparse_subjects": len(sparse),
        "quality_flags": (
            ([f"{len(sparse)} subjects with <3 observations"] if sparse else [])
            + ([f"BLQ {blq_pct}%"] if (blq_pct or 0) > 0 else [])
        ),
    }
    return ToolResult(
        summary=f"Profiled {dsid}: {n_obs} obs, {missing_pct}% missing, "
                f"{len(sparse)} sparse subject(s).",
        action=f"profile_pk_dataset({dsid})",
        writes={"data_quality": quality},
        result=quality,
    )


def validate_cdisc(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    dsid = args.get("dataset_id") or state.dataset_id
    df = ctx.dataset_store[dsid]
    roles = _roles(df, state.dataset_metadata)
    have = set(roles.values())
    checks = []

    def add(name, ok, detail):
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    add("ID column present", "ID" in have, "subject identifier")
    add("TIME column present", "TIME" in have, "time variable")
    add("DV column present", "DV" in have, "dependent variable / concentration")
    # monotonic time per subject
    id_col = next((c for c, r in roles.items() if r == "ID"), None)
    time_col = next((c for c, r in roles.items() if r == "TIME"), None)
    mono = True
    if id_col and time_col:
        for _, g in df.groupby(id_col):
            t = pd.to_numeric(g[time_col], errors="coerce").dropna().tolist()
            if any(b < a for a, b in zip(t, t[1:])):
                mono = False
                break
    add("Time monotonic within subject", mono, "no out-of-order timepoints")

    verdict = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
    return ToolResult(
        summary=f"CDISC validation: {verdict} ({sum(c['status']=='PASS' for c in checks)}/{len(checks)} checks).",
        action=f"validate_cdisc({dsid})",
        result={"checks": checks, "verdict": verdict},
    )


def generate_spaghetti_plot(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    dsid = args.get("dataset_id") or state.dataset_id
    df = ctx.dataset_store[dsid]
    roles = _roles(df, state.dataset_metadata)
    id_col = next((c for c, r in roles.items() if r == "ID"), None)
    time_col = next((c for c, r in roles.items() if r == "TIME"), None)
    dv_col = next((c for c, r in roles.items() if r == "DV"), None)
    amt_col = next((c for c, r in roles.items() if r == "AMT"), None)
    evid_col = next((c for c, r in roles.items() if r == "EVID"), None)
    log_scale = bool(args.get("log_scale", True))

    # Use only observation rows (exclude dosing rows)
    if evid_col:
        ev = pd.to_numeric(df[evid_col], errors="coerce").fillna(0)
        obs_df = df[ev == 0]
    elif amt_col:
        obs_df = df[df[amt_col].fillna(0) == 0]
    else:
        obs_df = df

    series = []
    n_blq = 0
    if id_col and time_col and dv_col:
        for sid, g in obs_df.groupby(id_col):
            g = g.sort_values(time_col)
            t_all = pd.to_numeric(g[time_col], errors="coerce").tolist()
            c_all = pd.to_numeric(g[dv_col], errors="coerce").tolist()
            obs_t, obs_c, blq_t = [], [], []
            for t, c in zip(t_all, c_all):
                if c is None or (c != c):  # NaN
                    continue
                if c > 0:
                    obs_t.append(round(t, 4) if t == t else t)
                    obs_c.append(round(c, 6))
                else:
                    blq_t.append(round(t, 4) if t == t else t)
                    n_blq += 1
            series.append({"id": str(sid), "x": obs_t, "y": obs_c, "blq_x": blq_t})

    spaghetti_data = {
        "series": series,
        "log_scale": log_scale,
        "n_subjects": len(series),
        "blq_excluded": n_blq,
        "x_label": time_col or "Time",
        "y_label": dv_col or "Concentration",
    }
    return ToolResult(
        summary=f"Concentration-time plot ready: {len(series)} subjects, {n_blq} BLQ excluded.",
        action=f"generate_spaghetti_plot({dsid})",
        writes={"spaghetti_data": spaghetti_data},
        result={"n_subjects": len(series), "n_blq": n_blq},
    )


def _schema(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


TOOLS = [
    Tool("load_dataset", "Load a PK dataset (CSV/XPT/SAS) and extract a metadata-only schema.",
         "data_manager",
         _schema({"path": {"type": "string", "description": "file path"},
                  "dataset_id": {"type": "string"}}, ["path"]),
         load_dataset),
    Tool("profile_pk_dataset", "Compute data-quality metrics (BLQ%, missing, sparse subjects).",
         "data_manager",
         _schema({"dataset_id": {"type": "string"}}, []), profile_pk_dataset),
    Tool("validate_cdisc", "Validate basic CDISC/NONMEM structure and time ordering.",
         "data_manager", _schema({"dataset_id": {"type": "string"}}, []), validate_cdisc),
    Tool("generate_spaghetti_plot", "Build an interactive concentration-time spaghetti plot.",
         "data_manager", _schema({"dataset_id": {"type": "string"}}, []), generate_spaghetti_plot),
]
