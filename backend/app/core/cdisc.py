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

from app.core.obs_loader import load_observations
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


def _units(state: PharmState) -> dict[str, str]:
    """Concentration/time/dose units from ``dataset_metadata["units"]`` ({"conc", "time", "dose"}). The session
    state does not capture units at upload, so unless the caller put them there every unit is blank rather than
    guessed (a wrong unit in a regulatory dataset is worse than an empty one)."""
    u = (state.dataset_metadata or {}).get("units")
    u = u if isinstance(u, dict) else {}
    return {k: str(u.get(k) or "").strip() for k in ("conc", "time", "dose")}


def _pp_unit(key: str, u: dict[str, str]) -> str:
    """PPSTRESU for an NCA result key from the conc/time/dose units; blank when a needed unit is unknown."""
    c, t, d = u["conc"], u["time"], u["dose"]
    auc = f"{t}*{c}" if c and t else ""
    table = {
        "Cmax": c, "Cmin": c, "Cavg": c, "Tmax": t, "t_half": t, "MRT": t, "AUC_last": auc, "AUC_inf": auc,
        "AUC_tau": auc, "lambda_z": f"1/{t}" if t else "", "pct_AUC_extrap": "%", "fluctuation_pct": "%",
        "CL_F": f"{d}/({auc})" if d and auc else "", "Vz_F": f"{d}/({c})" if d and c else "",
    }
    table["Vss"] = table["Vz_F"]
    return table.get(key, "")


def _csv_bytes(rows: list[dict[str, Any]], columns: list[str]) -> bytes:
    from app.core.exporters import sanitize_cell  # shared formula-injection guard
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    w.writeheader()
    w.writerows({k: sanitize_cell(v) for k, v in r.items()} for r in rows)
    return buf.getvalue().encode("utf-8")


def build_adpp(state: PharmState) -> tuple[list[dict], list[str]]:
    """ADPP — PK parameters in BDS long format (one row per subject x PARAMCD)."""
    params = state.nca_parameters or []
    study = (state.dataset_metadata or {}).get("dataset_id", "STUDY")
    ss = bool((state.nca_summary or {}).get("steady_state"))
    units = _units(state)
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
                "DOSEA": p.get("dose"), "PPSTRESU": _pp_unit(key, units), "PARCAT1": "NCA",
                "AVISIT": "Steady State" if ss else "Single Dose",
            })
    cols = ["STUDYID", "USUBJID", "SUBJID", "PARCAT1", "PARAMCD", "PARAM",
            "AVAL", "PPSTRESU", "DOSEA", "AVISIT"]
    return rows, cols


def build_adpc(df: pd.DataFrame | None, roles: dict[str, str],
               state: PharmState) -> tuple[list[dict], list[str]]:
    """ADPC — plasma concentrations in BDS long format (one row per sample).

    Observation rows only (EVID/AMT/MDV, via the shared ``load_observations``), never dosing records. A
    below-LOQ record (CENS role column, or DV < an LLOQ column) is kept but flagged ``BLQFL='Y'`` with AVAL blank and
    the LLOQ in ``LLOQ``: a censored DV is not a measured concentration. ``ANL01FL`` is 'Y' for the records the NCA
    uses under rule M1 (quantifiable values and the leading-BLQ zero baseline), 'N' for the rest."""
    cols = ["STUDYID", "USUBJID", "SUBJID", "PARAMCD", "PARAM", "AVAL", "AVALU", "BLQFL", "LLOQ", "ANL01FL",
            "ATPTN", "AVISIT"]
    if df is None:
        return [], cols
    study = (state.dataset_metadata or {}).get("dataset_id", "STUDY")
    id_col = next((c for c, r in roles.items() if r == "ID"), None)
    time_col = next((c for c, r in roles.items() if r == "TIME"), None)
    dv_col = next((c for c, r in roles.items() if r == "DV"), None)
    if not (id_col and time_col and dv_col):
        return [], cols
    lloq_col = next((c for c, r in roles.items() if r == "LLOQ"), None)
    loaded = load_observations(df, roles)
    cens_col, analysed = loaded.cens_col, set(loaded.obs.index)
    conc_unit = _units(state)["conc"]
    rows: list[dict] = []
    for idx, rec in loaded.raw_obs.iterrows():
        blq = bool(cens_col is not None and rec[cens_col])
        dv = rec[dv_col]
        if pd.isna(dv) and not blq:
            continue
        if lloq_col is not None and pd.notna(pd.to_numeric(rec[lloq_col], errors="coerce")):
            lloq = float(pd.to_numeric(rec[lloq_col]))
        else:
            lloq = float(dv) if blq and pd.notna(dv) else None
        t = rec[time_col]
        subj = rec[id_col]
        rows.append({
            "STUDYID": study, "USUBJID": f"{study}-{subj}", "SUBJID": subj,
            "PARAMCD": "CONC", "PARAM": "Plasma Drug Concentration",
            "AVAL": None if blq else float(dv), "AVALU": conc_unit,
            "BLQFL": "Y" if blq else "", "LLOQ": lloq, "ANL01FL": "Y" if idx in analysed else "N",
            "ATPTN": None if pd.isna(t) else float(t),
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
            dtype = "float" if c in {"AVAL", "ATPTN", "DOSEA", "LLOQ"} else "text"
            parts.append(f'      <ItemDef OID="{oid}" Name="{c}" DataType="{dtype}">\n'
                         f"        <Description><TranslatedText xml:lang=\"en\">"
                         f"{escape(c)}</TranslatedText></Description>\n      </ItemDef>\n")
    parts.append(_DEFINE_TAIL)
    return "".join(parts).encode("utf-8")


# Fixed member timestamp: writestr(name, ...) stamps wall-clock time (2 s DOS resolution), so two
# builds of one session could differ and miss the sha256 recorded in the export's provenance manifest.
_ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)


def _zip_member(name: str) -> zipfile.ZipInfo:
    """The ZipInfo writestr(name, ...) would create in a ZIP_DEFLATED archive, minus the wall clock."""
    info = zipfile.ZipInfo(name, date_time=_ZIP_DATE_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16  # ?rw-------, as writestr sets for a named member
    return info


def build_package(state: PharmState, df: pd.DataFrame | None,
                  roles: dict[str, str]) -> bytes:
    """Zip containing ADPC.csv, ADPP.csv, and define.xml. Byte-reproducible for a given input."""
    study = (state.dataset_metadata or {}).get("dataset_id", "STUDY")
    adpp_rows, adpp_cols = build_adpp(state)
    adpc_rows, adpc_cols = build_adpc(df, roles, state)
    define = build_define_xml({"ADPC": adpc_cols, "ADPP": adpp_cols}, study)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(_zip_member("ADPP.csv"), _csv_bytes(adpp_rows, adpp_cols))
        z.writestr(_zip_member("ADPC.csv"), _csv_bytes(adpc_rows, adpc_cols))
        z.writestr(_zip_member("define.xml"), define)
        z.writestr(_zip_member("README.txt"),
                   b"CDISC ADaM-aligned export from PharmAgent.\n"
                   b"ADPP.csv  - NCA parameters (BDS, standard PARAMCDs)\n"
                   b"ADPC.csv  - plasma concentrations (BDS)\n"
                   b"define.xml - Define-XML 2.0 metadata\n"
                   b"Note: ADaM-aligned, not a fully CT-coded/SDTM-traceable submission package.\n")
    return buf.getvalue()
