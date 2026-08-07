"""CDISC ADaM -> NONMEM-style analysis dataset.

PharmAgent's compute layer expects one flat table with NONMEM roles
(ID/TIME/DV/AMT/EVID/CMT/MDV). Clinical deliverables arrive as ADaM: dosing in
ADEX, concentrations in ADPC, subject-level covariates in ADSL, keyed by
USUBJID. This module performs that join.

Every derivation an analyst would otherwise do by hand is explicit here:

  * relative time — hours from each subject's FIRST dose, computed from the
    ISO timestamps (``EXSTDTC``/``PCDTC``) rather than trusting a planned-time
    column, so actual deviations and multiple-dose schedules come out right;
  * route -> compartment — SC/IM/PO dose into the depot (CMT 1), IV into the
    central compartment (CMT 2);
  * infusions — a dose whose EXENDTC is after its EXSTDTC becomes a zero-order
    RATE = AMT / duration, not a bolus;
  * BLQ — the M3 convention (DV=0, MDV=1, BLQ=1) so a likelihood-based method
    can use the censoring rather than dropping or imputing the record.

Anything the datasets do not state (LLOQ being the usual one) is INFERRED and
reported in ``issues`` rather than silently assumed — a missing analysis fact
should surface where a reviewer sees it, not decide a result quietly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

#: Routes that deposit drug in a depot compartment requiring absorption.
_EXTRAVASCULAR = ("SC", "SUBCUTANEOUS", "IM", "INTRAMUSCULAR", "PO", "ORAL")

#: ADSL columns promoted to model covariates, mapped to conventional names.
#: Deliberately curated: every extra column in the emitted frame is treated as
#: a covariate downstream, so dumping all of ADSL would create dozens of
#: spurious candidates and make covariate screening meaningless.
_COVARIATE_MAP = {
    "WEIGHTBL": "WT", "HEIGHTBL": "HT", "AGE": "AGE", "BMIBL": "BMI",
    "EGFRBL": "EGFR", "CREATBL": "CREAT", "ALBBL": "ALB", "CRPBL": "CRP",
    "ALTBL": "ALT", "ASTBL": "AST",
}

_TIME_DIVISOR = {"hour": 1.0, "day": 24.0}


@dataclass
class AdamResult:
    """The analysis dataset plus everything needed to defend how it was built."""
    df: pd.DataFrame
    id_map: pd.DataFrame                      # ID <-> USUBJID, for traceability
    meta: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


def _dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", format="mixed")


def _is_extravascular(route: Any) -> bool:
    return str(route).strip().upper() in _EXTRAVASCULAR


def _cmt_for_route(route: Any, *, central: int) -> int:
    """NONMEM (1-based) dosing compartment for a route.

    ``central`` is where the central compartment sits in THIS dataset's
    numbering, which depends on whether a depot exists: a dataset containing
    any extravascular dose needs an absorption model, so depot=1 and
    central=2; an IV-only dataset is fitted with a model that has no depot, so
    central=1. Hardcoding central=2 would dose the PERIPHERAL compartment of
    ``iv_2cmt`` — in range, so nothing would raise, and every parameter would
    be quietly wrong.
    """
    return 1 if _is_extravascular(route) else central


def build_analysis_dataset(
    adsl: pd.DataFrame,
    adex: pd.DataFrame,
    adpc: pd.DataFrame,
    *,
    analyte: str | None = None,
    time_unit: str = "hour",
    lloq: float | None = None,
    pk_flag: str | None = "PKFL",
    keep_usubjid: bool = False,
) -> AdamResult:
    """Join ADSL/ADEX/ADPC into one NONMEM-style table.

    ``analyte`` selects one PCTESTCD when the file holds several (required in
    that case — silently pooling two analytes into one DV column would be a
    data error, not a convenience). ``time_unit`` controls whether TIME is in
    hours or days; days makes estimates directly comparable to per-day
    literature parameters. ``lloq`` is inferred from the smallest reported
    concentration when not supplied.
    """
    if time_unit not in _TIME_DIVISOR:
        raise ValueError(f"time_unit must be one of {sorted(_TIME_DIVISOR)}")

    issues: list[str] = []
    adsl, adex, adpc = adsl.copy(), adex.copy(), adpc.copy()

    # -- analyte selection ---------------------------------------------------
    analytes = sorted(adpc["PCTESTCD"].dropna().unique()) if "PCTESTCD" in adpc else []
    if analyte:
        if analytes and analyte not in analytes:
            raise ValueError(f"analyte {analyte!r} not in ADPC; available: {analytes}")
        adpc = adpc[adpc["PCTESTCD"] == analyte]
    elif len(analytes) > 1:
        raise ValueError(
            f"ADPC holds {len(analytes)} analytes {analytes}; pass analyte= to choose one")
    elif analytes:
        analyte = analytes[0]

    # -- subjects: need BOTH a dose and a PK record to be modelable ----------
    if pk_flag and pk_flag in adsl.columns:
        eligible = set(adsl.loc[adsl[pk_flag].astype(str).str.upper() == "Y", "USUBJID"])
    else:
        eligible = set(adsl["USUBJID"])
        if pk_flag:
            issues.append(f"{pk_flag} not in ADSL — no analysis-set flag applied")

    dosed = adex.loc[pd.to_numeric(adex["EXDOSE"], errors="coerce") > 0]
    subjects = sorted(eligible & set(dosed["USUBJID"]) & set(adpc["USUBJID"]))
    if not subjects:
        raise ValueError("no subject has an analysis-set flag, a dose, and a PK record")

    dropped = len(eligible) - len(subjects)
    if dropped > 0:
        issues.append(f"{dropped} flagged subject(s) excluded: no dose and/or no "
                      f"{analyte or 'PK'} record (placebo arms are expected here)")

    id_map = pd.DataFrame({"USUBJID": subjects})
    id_map["ID"] = range(1, len(subjects) + 1)       # sequential 1..n
    ids = dict(zip(id_map["USUBJID"], id_map["ID"], strict=True))

    dosed = dosed[dosed["USUBJID"].isin(subjects)].copy()
    adpc = adpc[adpc["USUBJID"].isin(subjects)].copy()

    # -- time origin: each subject's first dose ------------------------------
    dosed["_start"] = _dt(dosed["EXSTDTC"])
    dosed["_end"] = _dt(dosed["EXENDTC"]) if "EXENDTC" in dosed else pd.NaT
    if dosed["_start"].isna().any():
        raise ValueError("ADEX has dose records with an unparseable EXSTDTC")
    t0 = dosed.groupby("USUBJID")["_start"].min()

    div = _TIME_DIVISOR[time_unit]

    def _rel(when: pd.Series, who: pd.Series) -> pd.Series:
        return (when - who.map(t0)).dt.total_seconds() / 3600.0 / div

    # -- dose records --------------------------------------------------------
    # Dataset CMT numbering is model-specific (as in NONMEM). A depot exists
    # only if some dose is extravascular; that shifts where central sits.
    any_ev = bool(dosed["EXROUTE"].map(_is_extravascular).any()) if "EXROUTE" in dosed else True
    central_cmt = 2 if any_ev else 1

    dur_h = (dosed["_end"] - dosed["_start"]).dt.total_seconds() / 3600.0
    dur_h = dur_h.where(dur_h > 0)                    # 0/NaN duration => bolus
    amt = pd.to_numeric(dosed["EXDOSE"], errors="coerce")
    doses = pd.DataFrame({
        "ID": dosed["USUBJID"].map(ids),
        "USUBJID": dosed["USUBJID"],
        "TIME": _rel(dosed["_start"], dosed["USUBJID"]),
        "AMT": amt,
        # RATE is per TIME unit, so it must follow the same scaling as TIME.
        "RATE": (amt / (dur_h / div)).fillna(0.0),
        "DV": 0.0, "EVID": 1, "MDV": 1, "BLQ": 0,
        "CMT": (dosed["EXROUTE"].map(lambda r: _cmt_for_route(r, central=central_cmt))
                if "EXROUTE" in dosed else 1),
    })
    n_inf = int(dur_h.notna().sum())
    if n_inf:
        issues.append(f"{n_inf} dose record(s) treated as zero-order infusions "
                      f"(median {dur_h.median():.2f} h) — RATE set, not bolus")

    # -- observations --------------------------------------------------------
    conc_col = next((c for c in ("PCSTRESN", "AVAL") if c in adpc.columns), None)
    if conc_col is None:
        raise ValueError("ADPC has neither PCSTRESN nor AVAL")
    conc = pd.to_numeric(adpc[conc_col], errors="coerce")

    # BLQ: an explicit character result wins; otherwise a missing numeric with a
    # present original value (i.e. reported but not quantifiable).
    orig = adpc["PCORRES"].astype(str).str.upper().str.strip() if "PCORRES" in adpc else ""
    is_blq = conc.isna() & (orig.str.contains("BLQ|BLOQ|<", regex=True, na=False)
                            if len(orig) else conc.isna())

    if lloq is None and conc.notna().any():
        lloq = float(conc[conc > 0].min())
        issues.append(f"LLOQ not stated in ADPC — inferred as {lloq:g} "
                      f"(smallest quantified value); confirm against the bioanalytical "
                      f"report before relying on the M3 fit")

    obs = pd.DataFrame({
        "ID": adpc["USUBJID"].map(ids),
        "USUBJID": adpc["USUBJID"],
        "TIME": _rel(_dt(adpc["PCDTC"]), adpc["USUBJID"]),
        "AMT": 0.0, "RATE": 0.0,
        "DV": conc.fillna(0.0),                       # M3: censored DV recorded as 0
        "EVID": 0,
        "MDV": is_blq.astype(int),                    # BLQ is not a plain observation
        "BLQ": is_blq.astype(int),
        "CMT": 2,
    })
    unusable = obs["TIME"].isna().sum()
    if unusable:
        obs = obs[obs["TIME"].notna()]
        issues.append(f"{unusable} PK record(s) dropped: PCDTC missing/unparseable")
    if int(is_blq.sum()):
        issues.append(f"{int(is_blq.sum())} BLQ record(s) coded M3 (DV=0, MDV=1, BLQ=1)")

    # -- assemble ------------------------------------------------------------
    df = pd.concat([doses, obs], ignore_index=True)

    covs = {src: dst for src, dst in _COVARIATE_MAP.items() if src in adsl.columns}
    keep = ["USUBJID", *covs]
    sl = adsl[adsl["USUBJID"].isin(subjects)][keep].drop_duplicates("USUBJID")
    sl = sl.rename(columns=covs)
    if "SEX" in adsl.columns:                          # numeric encoding for covariates
        sl = sl.merge(adsl[["USUBJID", "SEX"]].drop_duplicates("USUBJID"), on="USUBJID")
        sl["SEXN"] = (sl.pop("SEX").astype(str).str.upper().str[0] == "M").astype(int)
    if "POPULATION" in adsl.columns:
        pop = adsl[["USUBJID", "POPULATION"]].drop_duplicates("USUBJID")
        sl = sl.merge(pop, on="USUBJID")
        sl["DISEASE"] = (sl.pop("POPULATION").astype(str).str.lower() != "healthy").astype(int)
    df = df.merge(sl, on="USUBJID", how="left")

    # dose before observation at an identical timestamp, else the dose is missed
    df = df.sort_values(["ID", "TIME", "EVID"], ascending=[True, True, False])
    df = df.reset_index(drop=True)
    if not keep_usubjid:
        df = df.drop(columns=["USUBJID"])

    missing_wt = "WT" not in df.columns
    if missing_wt:
        issues.append("no baseline weight in ADSL — allometric scaling unavailable")

    meta = {
        "analyte": analyte, "time_unit": time_unit, "lloq": lloq,
        "n_subjects": len(subjects), "n_doses": int(len(doses)),
        "n_observations": int((df["EVID"] == 0).sum()),
        "n_blq": int(df["BLQ"].sum()),
        "studies": sorted(adsl.loc[adsl["USUBJID"].isin(subjects), "STUDYID"]
                          .dropna().unique()) if "STUDYID" in adsl else [],
        "covariates": sorted(c for c in df.columns
                             if c not in ("ID", "TIME", "AMT", "RATE", "DV",
                                          "EVID", "MDV", "BLQ", "CMT")),
        "routes": sorted(dosed["EXROUTE"].dropna().unique()) if "EXROUTE" in dosed else [],
        # CMT numbering is model-specific: fit this dataset with a model whose
        # compartment order matches, or the dose enters the wrong compartment.
        "dose_cmt_convention": ("1=depot, 2=central (extravascular doses present)"
                                if any_ev else "1=central (IV only, no depot)"),
    }
    return AdamResult(df=df, id_map=id_map, meta=meta, issues=issues)
