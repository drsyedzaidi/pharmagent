"""CDISC ADaM-style export: ADPC (concentrations), ADPP (NCA parameters), define.xml.

Produces a downloadable ADaM-aligned package. This is CDISC-*aligned* (BDS
structure, standard PARAMCDs, a Define-XML 2.0 document) — not a fully
controlled-terminology, SDTM-traceable submission package, which additionally
requires CT codelists, SDTM source datasets, and sponsor-specific metadata.
"""
from __future__ import annotations

import csv
import io
import zipfile
from typing import Any
from xml.sax.saxutils import escape

import pandas as pd

from app.core.pharmstate import PharmState

# NCA result key -> (PARAMCD, PARAM) using common CDISC PK parameter terms.
_ADPP_PARAMS: dict[str, tuple[str, str]] = {
    "Cmax": ("CMAX", "Maximum Concentration"),
    "Tmax": ("TMAX", "Time of CMAX"),
    "Cmin": ("CMIN", "Minimum Concentration (ss)"),
    "Cavg": ("CAVG", "Average Concentration (ss)"),
    "AUC_last": ("AUCLST", "AUC from Time 0 to Last Nonzero Conc"),
    "AUC_inf": ("AUCIFO", "AUC Infinity Obs"),
    "AUC_tau": ("AUCTAU", "AUC over Dosing Interval"),
    "t_half": ("LAMZHL", "Half-Life Lambda z"),
    "lambda_z": ("LAMZ", "Lambda z"),
    "CL_F": ("CLFO", "Total CL Obs by F"),
    "Vz_F": ("VZFO", "Vz Obs by F"),
    "Vss": ("VSSO", "Vss Obs"),
    "MRT": ("MRTLST", "MRT"),
    "pct_AUC_extrap": ("AUCPEO", "AUC %Extrapolation Obs"),
    "fluctuation_pct": ("FLUCP", "Fluctuation % (ss)"),
    "accumulation_ratio": ("ARAUC", "Accumulation Ratio"),
}


def _csv_bytes(rows: list[dict[str, Any]], columns: list[str]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


def build_adpp(state: PharmState) -> tuple[list[dict], list[str]]:
    """ADPP — PK parameters in BDS long format (one row per subject x PARAMCD)."""
    params = state.nca_parameters or []
    study = (state.dataset_metadata or {}).get("dataset_id", "STUDY")
    ss = bool((state.nca_summary or {}).get("steady_state"))
    rows: list[dict] = []
    for p in params:
        subj = p.get("subject")
        for key, (pcd, plbl) in _ADPP_PARAMS.items():
            val = p.get(key)
            if val is None:
                continue
            rows.append({
                "STUDYID": study, "USUBJID": f"{study}-{subj}", "SUBJID": subj,
                "PARAMCD": pcd, "PARAM": plbl, "AVAL": val,
                "DOSEA": p.get("dose"), "PPSTRESU": "", "PARCAT1": "NCA",
                "AVISIT": "Steady State" if ss else "Single Dose",
            })
    cols = ["STUDYID", "USUBJID", "SUBJID", "PARCAT1", "PARAMCD", "PARAM",
            "AVAL", "DOSEA", "AVISIT"]
    return rows, cols


def build_adpc(df: pd.DataFrame | None, roles: dict[str, str],
               state: PharmState) -> tuple[list[dict], list[str]]:
    """ADPC — plasma concentrations in BDS long format (one row per sample)."""
    cols = ["STUDYID", "USUBJID", "SUBJID", "PARAMCD", "PARAM", "AVAL", "ATPTN", "AVISIT"]
    if df is None:
        return [], cols
    study = (state.dataset_metadata or {}).get("dataset_id", "STUDY")
    id_col = next((c for c, r in roles.items() if r == "ID"), None)
    time_col = next((c for c, r in roles.items() if r == "TIME"), None)
    dv_col = next((c for c, r in roles.items() if r == "DV"), None)
    if not (id_col and time_col and dv_col):
        return [], cols
    t = pd.to_numeric(df[time_col], errors="coerce")
    dv = pd.to_numeric(df[dv_col], errors="coerce")
    rows: list[dict] = []
    for i in range(len(df)):
        if pd.isna(dv.iloc[i]):
            continue
        subj = df[id_col].iloc[i]
        rows.append({
            "STUDYID": study, "USUBJID": f"{study}-{subj}", "SUBJID": subj,
            "PARAMCD": "CONC", "PARAM": "Plasma Drug Concentration",
            "AVAL": float(dv.iloc[i]),
            "ATPTN": None if pd.isna(t.iloc[i]) else float(t.iloc[i]),
            "AVISIT": "PK Profile",
        })
    return rows, cols


_DEFINE_HEAD = """<?xml version="1.0" encoding="UTF-8"?>
<ODM xmlns="http://www.cdisc.org/ns/odm/v1.3"
     xmlns:def="http://www.cdisc.org/ns/def/v2.0"
     FileType="Snapshot" ODMVersion="1.3.2" SourceSystem="PharmAgent">
  <Study OID="{study}">
    <GlobalVariables>
      <StudyName>{study}</StudyName>
      <StudyDescription>PharmAgent NCA analysis datasets (ADaM-aligned)</StudyDescription>
      <ProtocolName>{study}</ProtocolName>
    </GlobalVariables>
    <MetaDataVersion OID="MDV.1" Name="ADaM Define" def:DefineVersion="2.0.0">
"""
_DEFINE_TAIL = "    </MetaDataVersion>\n  </Study>\n</ODM>\n"


def build_define_xml(datasets: dict[str, list[str]], study: str) -> bytes:
    """Minimal Define-XML 2.0 describing the exported ADaM datasets/variables."""
    parts = [_DEFINE_HEAD.format(study=escape(str(study)))]
    for name, cols in datasets.items():
        parts.append(f'      <ItemGroupDef OID="IG.{name}" Name="{name}" '
                     f'Repeating="Yes" Purpose="Analysis" def:Structure="BDS" '
                     f'def:Class="BASIC DATA STRUCTURE">\n')
        parts.append(f"        <Description><TranslatedText xml:lang=\"en\">"
                     f"{escape(name)} analysis dataset</TranslatedText></Description>\n")
        for i, c in enumerate(cols, 1):
            parts.append(f'        <ItemRef ItemOID="IT.{name}.{c}" OrderNumber="{i}" '
                         f'Mandatory="No"/>\n')
        parts.append("      </ItemGroupDef>\n")
    seen: set[str] = set()
    for name, cols in datasets.items():
        for c in cols:
            oid = f"IT.{name}.{c}"
            if oid in seen:
                continue
            seen.add(oid)
            dtype = "float" if c in {"AVAL", "ATPTN", "DOSEA"} else "text"
            parts.append(f'      <ItemDef OID="{oid}" Name="{c}" DataType="{dtype}">\n'
                         f"        <Description><TranslatedText xml:lang=\"en\">"
                         f"{escape(c)}</TranslatedText></Description>\n      </ItemDef>\n")
    parts.append(_DEFINE_TAIL)
    return "".join(parts).encode("utf-8")


def build_package(state: PharmState, df: pd.DataFrame | None,
                  roles: dict[str, str]) -> bytes:
    """Zip containing ADPC.csv, ADPP.csv, and define.xml."""
    study = (state.dataset_metadata or {}).get("dataset_id", "STUDY")
    adpp_rows, adpp_cols = build_adpp(state)
    adpc_rows, adpc_cols = build_adpc(df, roles, state)
    define = build_define_xml({"ADPC": adpc_cols, "ADPP": adpp_cols}, study)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("ADPP.csv", _csv_bytes(adpp_rows, adpp_cols))
        z.writestr("ADPC.csv", _csv_bytes(adpc_rows, adpc_cols))
        z.writestr("define.xml", define)
        z.writestr("README.txt",
                   b"CDISC ADaM-aligned export from PharmAgent.\n"
                   b"ADPP.csv  - NCA parameters (BDS, standard PARAMCDs)\n"
                   b"ADPC.csv  - plasma concentrations (BDS)\n"
                   b"define.xml - Define-XML 2.0 metadata\n"
                   b"Note: ADaM-aligned, not a fully CT-coded/SDTM-traceable submission package.\n")
    return buf.getvalue()
