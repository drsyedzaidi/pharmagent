"""SchemaExtractor — the privacy boundary.

Before any dataset is described to the LLM, it is reduced to metadata only:
column names, dtypes, subject/record counts, dose levels, time range, and
aggregate statistics. Individual rows, identifiers, and raw concentration
values are NEVER placed in `dataset_metadata` and therefore never reach the
model.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

# Exact (lowercased) names that pin a role unambiguously.
ROLE_EXACT: dict[str, str] = {
    "id": "ID", "usubjid": "ID", "subject": "ID", "subjid": "ID", "subjectid": "ID",
    "evid": "EVID", "mdv": "MDV", "cmt": "CMT", "compartment": "CMT",
    "dvid": "DVID", "ytype": "DVID",
    "cens": "CENS", "blq": "CENS", "lloq": "LLOQ",
    "route": "ROUTE", "rte": "ROUTE",
    "pd": "PD", "effect": "PD", "resp": "PD", "response": "PD", "biomarker": "PD",
    "dv": "DV", "conc": "DV", "concentration": "DV", "cobs": "DV", "y": "DV",
    "amt": "AMT", "amount": "AMT", "dose": "AMT",
    "addl": "ADDL", "ii": "II", "tau": "II",
    "tad": "TAD", "tafd": "TAD",
    "time": "TIME", "atime": "TIME", "ntime": "TIME",
}

# Ordered substring rules (first match wins). Order matters: II must beat AMT
# so "interdose interval" is not read as a dose; TAD must beat TIME; PD (effect)
# must beat DV/AMT so an effect column isn't read as concentration.
_SUBSTRING_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("ADDL", ("addl", "additional")),
    ("II",   ("interdose", "interval", "tau")),
    ("DVID", ("dvid", "ytype")),
    ("CENS", ("cens", "blq")),
    ("ROUTE", ("route",)),
    ("PD",   ("effect", "response", "biomarker", "pdresp")),
    ("TAD",  ("tad", "tafd", "timeold", "timeafter", "time after", "time_after")),
    ("TIME", ("time",)),
    ("DV",   ("conc", "cobs")),
    ("AMT",  ("dose", "amt", "amount")),
    ("CMT",  ("compartment",)),
]


def detect_roles(columns: list[str]) -> dict[str, str]:
    """Map each column to a PK role. Exact names win; otherwise ordered
    substring rules apply (II before AMT, TAD before TIME)."""
    roles: dict[str, str] = {}
    for col in columns:
        low = col.strip().lower()
        if low in ROLE_EXACT:
            roles[col] = ROLE_EXACT[low]
            continue
        for role, pats in _SUBSTRING_RULES:
            if any(p in low for p in pats):
                roles[col] = role
                break
    return roles


def _num_summary(s: pd.Series) -> dict[str, Any]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {"n": 0}
    return {
        "n": int(s.size),
        "n_unique": int(s.nunique()),
        "min": _safe(float(s.min())),
        "max": _safe(float(s.max())),
        "mean": _safe(float(s.mean())),
        "median": _safe(float(s.median())),
        "missing": int(s.isna().sum()),
    }


def _safe(x: float) -> float | None:
    return None if (x is None or math.isnan(x) or math.isinf(x)) else round(x, 6)


def extract_schema(df: pd.DataFrame, *, dataset_id: str) -> dict[str, Any]:
    """Produce a metadata-only summary safe to send to the LLM."""
    roles = detect_roles(list(df.columns))
    id_col = next((c for c, r in roles.items() if r == "ID"), None)
    time_col = next((c for c, r in roles.items() if r == "TIME"), None)
    amt_col = next((c for c, r in roles.items() if r == "AMT"), None)

    columns_meta = []
    for col in df.columns:
        columns_meta.append(
            {
                "name": col,
                "dtype": str(df[col].dtype),
                "role": roles.get(col, "UNKNOWN"),
                "n_missing": int(df[col].isna().sum()),
                "summary": _num_summary(df[col]) if pd.api.types.is_numeric_dtype(
                    pd.to_numeric(df[col], errors="coerce")
                ) else {"n_unique": int(df[col].nunique())},
            }
        )

    n_subjects = int(df[id_col].nunique()) if id_col else None
    dose_levels = (
        sorted(pd.to_numeric(df[amt_col], errors="coerce").dropna().unique().tolist())
        if amt_col else None
    )
    time_range = None
    if time_col:
        t = pd.to_numeric(df[time_col], errors="coerce").dropna()
        if not t.empty:
            time_range = [_safe(float(t.min())), _safe(float(t.max()))]

    return {
        "dataset_id": dataset_id,
        "n_records": int(len(df)),
        "n_subjects": n_subjects,
        "n_columns": int(df.shape[1]),
        "detected_roles": roles,
        "dose_levels": dose_levels,
        "time_range": time_range,
        "columns": columns_meta,
        # explicit assurance flag for downstream consumers/auditors
        "privacy": "metadata_only; no individual rows or raw concentrations included",
    }
