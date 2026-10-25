"""PMxbench adapter (ISoP AI/ML SIG, ``tools/pmxbench/scenario_00``).

Two pure functions and a thin CLI:

* ``prepare_dataset(df)`` turns the scenario's ``data.csv`` into a frame the
  normal PharmAgent popPK tools read without any role override, and records
  every transformation in a report.
* ``build_submission(state)`` fills ``submission.template.yaml`` from a
  PharmState after the owner has run the analysis. An item the analysis did not
  answer is ``None`` (YAML ``null``), never a guess.

Nothing here fits a model, and nothing reads the benchmark's answer key.

CLI::

    python -m app.benchmarks.pmxbench prepare --data data.csv --out prepared.csv
    python -m app.benchmarks.pmxbench submit --state state.json --out submission.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from app.core.pharmstate import PharmState
from app.core.provenance import QUARANTINED_STATUS, ProvenanceQuarantined
from app.core.schema_extractor import detect_roles

# sap.md §5 and §8: assay LLOQ 0.1 mg/L; BLQ = 1 rows carry DV = 0 as a flag.
SCENARIO_00_LLOQ = 0.1
# sap.md §4 and Appendix A: single 1-hour infusion, TIME in days.
INFUSION_DURATION_DAYS = 1.0 / 24.0

REQUIRED_COLUMNS = ("ID", "TIME", "DV", "AMT", "BLQ", "ROWID", "WT", "ALB", "CRCL")
COVARIATES = ("WT", "ALB", "CRCL")
ANALYSIS_COLUMNS = ("ID", "TIME", "DV", "AMT", "RATE", "CENS", *COVARIATES)
# What detect_roles must assign to the prepared frame. Columns without a role
# are covariates to the PharmAgent subject builder.
EXPECTED_ROLES = {"ID": "ID", "TIME": "TIME", "DV": "DV", "AMT": "AMT",
                  "RATE": "RATE", "CENS": "CENS"}
# Source columns left out of the analysis frame. Any column without a
# structural role is screened as a covariate, so these would pollute SCM.
DROPPED_COLUMNS = {
    "DAY": "nominal day; duplicates TIME and would be screened as a covariate",
    "DOSE": "assigned dose level; detect_roles maps it to a second AMT column, and it is not a "
            "pre-specified covariate (sap.md §7)",
    "BLQ": "replaced by CENS",
    "ROWID": "record key, not a covariate; prepared rows keep the source order, so "
             "report['rowid_by_row'][i] is the ROWID of prepared row i",
}

# Final-model parameter -> submission name, per supported structural model.
_IV_LINEAR: dict[str, tuple[int, dict[str, str]]] = {
    "iv_1cmt": (1, {"CL": "CL", "V": "V"}),
    "iv_2cmt": (2, {"CL": "CL", "VC": "V", "Q": "Q", "VP": "V2"}),
    "iv_3cmt": (3, {"CL": "CL", "VC": "V", "Q2": "Q", "VP2": "V2", "Q3": "Q3", "VP3": "V3"}),
}
_ERROR_TERMS = {"proportional": ("prop",), "additive": ("add",), "combined": ("prop", "add")}
ANSWER_KEYS = ("structural_ncmt", "disposition_params", "error_model", "cov_effects",
               "outlier_records")
RUN_INFO_KEYS = ("harness", "model", "harness_version", "run_utc")


# ─────────────────────────────── prepare ────────────────────────────────────

def _numeric(s: pd.Series, name: str) -> pd.Series:
    cleaned = s if pd.api.types.is_numeric_dtype(s) else s.replace(".", np.nan)
    try:
        return pd.to_numeric(cleaned, errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"column {name} is not numeric: {exc}") from exc


def _rows(mask: pd.Series, rowid: pd.Series) -> list[str]:
    return rowid[mask].astype(str).tolist()


def _validate(df: pd.DataFrame, lloq: float, infusion_duration: float) -> dict[str, pd.Series]:
    """Fail closed on anything the SAP's data specification does not allow."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    clash = [c for c in ("RATE", "CENS") if c in df.columns]
    if clash:
        raise ValueError(f"refusing to overwrite existing columns: {clash}")
    if not (lloq > 0 and infusion_duration > 0):
        raise ValueError("lloq and infusion_duration must be positive")
    col = {c: _numeric(df[c], c) for c in ("TIME", "DV", "AMT", "BLQ", *COVARIATES)}
    rowid = df["ROWID"]
    if rowid.isna().any() or rowid.astype(str).duplicated().any():
        raise ValueError("ROWID must be present and unique on every record")
    if df["ID"].isna().any() or col["TIME"].isna().any() or col["AMT"].isna().any():
        raise ValueError("ID, TIME and AMT must be present on every record")
    if (col["AMT"] < 0).any():
        raise ValueError(f"negative AMT on ROWID {_rows(col['AMT'] < 0, rowid)}")
    blq, dv, dosing = col["BLQ"], col["DV"], col["AMT"] > 0
    checks = {
        "BLQ is not 0/1": ~blq.isin([0.0, 1.0]),
        "BLQ = 1 on a dosing record": (blq == 1) & dosing,
        "observation without DV": ~dosing & dv.isna(),
        "BLQ = 1 with DV != 0": (blq == 1) & (dv != 0),
        "uncensored DV <= 0": ~dosing & (blq == 0) & (dv <= 0),
    }
    for what, bad in checks.items():
        if bad.any():
            raise ValueError(f"{what} on ROWID {_rows(bad, rowid)[:10]}")
    for c in COVARIATES:
        if col[c].isna().any():
            raise ValueError(f"covariate {c} missing on ROWID {_rows(col[c].isna(), rowid)[:10]}")
    _check_subjects(df["ID"], dosing, col["TIME"])
    return col


def _check_subjects(ids: pd.Series, dosing: pd.Series, time: pd.Series) -> None:
    """sap.md Appendix A: one dosing record per subject at TIME 0, then observations.
    PharmAgent's subject builder uses only observations at TIME > 0 and would
    silently skip a subject without a dose or without one of those."""
    per = pd.DataFrame({"ID": ids, "dose": dosing, "dose_t0": dosing & (time == 0),
                        "obs": ~dosing & (time > 0)}).groupby("ID").sum()
    bad = per.index[(per["dose"] != 1) | (per["dose_t0"] != 1) | (per["obs"] < 1)].tolist()
    if bad:
        raise ValueError("each subject needs exactly one dosing record at TIME 0 and at least one post-dose "
                         f"observation; not so for ID {bad[:10]}")


def _covariate_report(ids: pd.Series, col: dict[str, pd.Series]) -> dict[str, Any]:
    frame = pd.DataFrame({"ID": ids, **{c: col[c] for c in COVARIATES}})
    base = frame.groupby("ID")[list(COVARIATES)].first()
    varying = {c: int((frame.groupby("ID")[c].nunique() > 1).sum()) for c in COVARIATES}
    corr = base.corr()
    pairs = {f"{a}~{b}": round(float(corr.loc[a, b]), 3) + 0.0  # +0.0 drops a "-0.0"
             for i, a in enumerate(COVARIATES) for b in COVARIATES[i + 1:]}
    summary = {c: {k: round(float(v), 4) for k, v in base[c].describe().items()} for c in COVARIATES}
    return {"baseline_summary": summary, "baseline_pearson_r": pairs,
            "n_subjects_time_varying": varying,
            "note": "PharmAgent takes each subject's first (baseline) covariate value."}


def _dropped(df: pd.DataFrame) -> list[dict[str, str]]:
    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS and c not in DROPPED_COLUMNS]
    return ([{"column": c, "reason": r} for c, r in DROPPED_COLUMNS.items() if c in df.columns]
            + [{"column": c, "reason": "not in the scenario_00 data specification (sap.md Appendix A)"}
               for c in extra])


def _transformations(df: pd.DataFrame, n_dosing: int, n_cens: int, lloq: float,
                     infusion_duration: float) -> list[dict[str, Any]]:
    dropped = [d["column"] for d in _dropped(df)]
    return [
        {"step": "derive_rate", "column": "RATE", "rows": n_dosing,
         "detail": f"RATE = AMT / {infusion_duration:.6g} on AMT > 0 rows (1-hour infusion, "
                   "TIME in days); 0 on observation rows"},
        {"step": "map_cens", "column": "CENS", "rows": n_cens,
         "detail": "CENS = BLQ (1 = below the LLOQ); the M3 censoring flag"},
        {"step": "censored_dv_to_lloq", "column": "DV", "rows": n_cens,
         "detail": f"DV 0 -> {lloq} on CENS = 1 rows. The source DV = 0 is a censoring flag, "
                   "not a concentration; PharmAgent's M3 reads the limit from DV on censored "
                   "rows, so leaving 0 would set the limit to 0."},
        {"step": "drop_columns", "column": ",".join(dropped), "rows": int(len(df)),
         "detail": "see dropped_columns"},
    ]


def _prepare_report(df: pd.DataFrame, col: dict[str, pd.Series], lloq: float,
                    infusion_duration: float) -> dict[str, Any]:
    dosing, cens = col["AMT"] > 0, col["BLQ"] == 1
    rowid = df["ROWID"].astype(str)
    below = ~dosing & ~cens & (col["DV"] < lloq)
    return {
        "adapter": "app.benchmarks.pmxbench", "scenario": "scenario_00",
        "n_records": int(len(df)), "n_subjects": int(df["ID"].nunique()),
        "n_dosing_records": int(dosing.sum()), "n_observation_records": int((~dosing).sum()),
        "n_censored": int(cens.sum()),
        "n_censored_predose": int((cens & (col["TIME"] <= 0)).sum()),
        "n_censored_postdose": int((cens & (col["TIME"] > 0)).sum()),
        "lloq": lloq, "lloq_source": "sap.md §5, §8" if lloq == SCENARIO_00_LLOQ else "caller-supplied",
        "infusion_duration": infusion_duration, "time_unit": "day",
        "units": {"TIME": "day", "DV": "mg/L", "AMT": "mg", "RATE": "mg/day",
                  "CL, Q": "L/day", "volumes": "L"},
        "roles": {**EXPECTED_ROLES, **dict.fromkeys(COVARIATES, "covariate"),
                  "WT": "covariate; also the allometric weight column (70 kg reference)"},
        "transformations": _transformations(df, int(dosing.sum()), int(cens.sum()), lloq,
                                            infusion_duration),
        "dropped_columns": _dropped(df),
        "rowid_by_row": rowid.tolist(),
        "checks": {
            "validated": ["BLQ is 0/1", "no BLQ on dosing records", "every observation has a DV",
                          "BLQ records carry DV = 0", "no uncensored DV <= 0", "unique ROWID",
                          "covariates present", "one dosing record per subject at TIME 0"],
            "quantified_below_lloq": {
                "n": int(below.sum()), "rowids": _rows(below, rowid),
                "note": "BLQ = 0 but DV < LLOQ; kept as quantified observations. "
                        "A data-handling note, not an outlier verdict."},
        },
        "covariates": _covariate_report(df["ID"], col),
        "notes": [
            "Predose records (TIME <= 0) stay in the file; PharmAgent's subject builder keeps only "
            "observations at TIME > 0, so the M3 likelihood sees the post-dose censored records only.",
            "M3 consumers (run_nlme, run_scm, run_diagnostics, the run_vpc BLQ check) keep the post-dose "
            "censored records as censored. The others (two-stage fit_pk_model, the run_vpc overlay, "
            "run_prior_check, the forest/exposure helpers) exclude them (M1) and report n_blq_dropped; "
            "none scores them at the LLOQ that DV carries (docs/pmxbench.md, G5).",
            "No record is excluded or flagged as an outlier here: PharmAgent has no record-level "
            "outlier screen (docs/pmxbench.md).",
        ],
    }


def prepare_dataset(df: pd.DataFrame, *, lloq: float = SCENARIO_00_LLOQ,
                    infusion_duration: float = INFUSION_DURATION_DAYS
                    ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Map scenario_00 ``data.csv`` onto PharmAgent's popPK column roles.

    Returns a NEW frame with exactly ``ANALYSIS_COLUMNS`` (same rows, same order
    as the input) and a report of every transformation. The input is not
    modified. Raises ``ValueError`` on data the SAP does not allow.

    The censored-DV step deviates from "leave DV = 0 on BLQ rows": PharmAgent's
    M3 term takes the limit from DV on censored rows (the NONMEM convention, see
    ``_build_subjects``), so DV = 0 there would make the limit 0 and give every
    censored record a constant, wrong likelihood term. The flag lives in CENS.
    """
    col = _validate(df, lloq, infusion_duration)
    dosing, cens = col["AMT"] > 0, col["BLQ"] == 1
    out = pd.DataFrame({
        "ID": df["ID"].to_numpy(), "TIME": col["TIME"].to_numpy(),
        "DV": col["DV"].where(~cens, lloq).to_numpy(), "AMT": col["AMT"].to_numpy(),
        "RATE": (col["AMT"] / infusion_duration).where(dosing, 0.0).to_numpy(),
        "CENS": cens.astype(int).to_numpy(),
        **{c: col[c].to_numpy() for c in COVARIATES},
    })
    roles = detect_roles(list(out.columns))
    if roles != EXPECTED_ROLES:
        raise RuntimeError(f"prepared columns no longer auto-detect as expected: {roles}")
    return out, _prepare_report(df, col, lloq, infusion_duration)


# ─────────────────────────────── submission ─────────────────────────────────

_SCM_M3_NOTE = ("The final model is the SCM final fit; run_scm applies M3, like run_nlme, so censored "
                "records enter its likelihood as censored.")
_SCM_NOT_M3_NOTE = ("The final model is the SCM final fit, which reports no censored records (n_blq 0), so "
                    "it was not fitted with M3 (e.g. an SCM run from before M3 support): censored records, "
                    "whose DV carries the LLOQ, may have entered its likelihood as observations at the LLOQ.")
_SOFTWARE_KEYS = ("python", "numpy", "scipy", "pandas")
_LEFT_NULL = "structural and parameter items left null."


def _ok(result: dict[str, Any] | None) -> dict[str, Any] | None:
    return result if (result or {}).get("status") == "ok" else None


def _dataset_problem(state: PharmState) -> str | None:
    """The answers are only meaningful on the prepare_dataset output: RATE (infusion),
    CENS (M3) and no record keys in the covariate pool."""
    meta = state.dataset_metadata or {}
    names = {c.get("name") for c in meta.get("columns") or []}
    if names == set(ANALYSIS_COLUMNS) and (meta.get("detected_roles") or {}) == EXPECTED_ROLES:
        return None
    return ("The session dataset is not the prepare_dataset output (columns or roles differ): doses may "
            "be bolus, M3 may be off and record keys may be screened as covariates; " + _LEFT_NULL)


def _final_model(state: PharmState) -> tuple[dict | None, str | None, list[str]]:
    """The final fit: the SCM final fit when SCM selected covariates, else the stored
    NLME fit. When both exist they must describe the same model (structure and
    residual error); otherwise which fit is final is ambiguous and nothing is reported."""
    nl, scm = _ok(state.nlme_results), _ok(state.scm_results)
    scm_final = (scm or {}).get("final") or {}
    if nl and scm and ((scm.get("model_key"), scm_final.get("error_model"))
                       != (nl.get("model_key"), nl.get("error_model"))):
        return None, None, ["SCM and the stored NLME fit describe different models (structure or residual "
                            "error); rerun SCM on the final model. " + _LEFT_NULL.capitalize()]
    if scm and scm.get("selected"):
        m3_note = _SCM_M3_NOTE if (scm_final or {}).get("n_blq") else _SCM_NOT_M3_NOTE
        final, source, notes = scm_final, "run_scm final model", [m3_note]
    else:
        final, source, notes = nl, "run_nlme", []
    if not final:
        return None, None, ["No population (NLME) fit in the session; " + _LEFT_NULL]
    if not final.get("converged") or final.get("degenerate_params"):
        return None, None, [*notes, f"The final fit ({source}) did not converge; " + _LEFT_NULL]
    return final, source, notes


def _disposition(final: dict, names: dict[str, str]) -> tuple[dict[str, float] | None, list[str]]:
    theta = final.get("theta") or {}
    if any(theta.get(p) is None for p in names):
        return None, ["Final fit lacks a structural parameter: disposition_params left null."]
    return {sub: float(theta[p]) for p, sub in names.items()}, []


def _error_terms(final: dict) -> tuple[dict[str, float] | None, list[str]]:
    terms = _ERROR_TERMS.get(final.get("error_model"))
    sigma = final.get("sigma") or {}
    if terms is None or any(sigma.get(t) is None for t in terms):
        return None, [f"Residual-error model {final.get('error_model')!r} not mapped: error_model left null."]
    return {t: float(sigma[t]) for t in terms}, []


def _cov_effects(state: PharmState, final: dict,
                 names: dict[str, str]) -> tuple[dict[str, dict[str, Any]] | None, list[str]]:
    scm = _ok(state.scm_results)
    if scm is None:
        return None, ["SCM was not run: cov_effects left null (not {})."]
    if scm.get("note") or not (scm.get("final") or {}).get("converged"):
        why = scm.get("note") or "SCM final fit did not converge"
        return None, [f"SCM did not complete a covariate search ({why}): cov_effects left null."]
    selected = scm.get("selected") or []
    if not selected:
        return {}, ["cov_effects {}: no tested candidate was retained. WT was never a candidate "
                    "(fixed allometry), so {} says nothing about a WT effect beyond the fixed exponents."]
    records = {(r.get("param"), r.get("covariate")): r for r in final.get("covariate_effects") or []}
    out: dict[str, dict[str, Any]] = {}
    for eff in selected:
        rec = records.get((eff.get("param"), eff.get("covariate")))
        sub = names.get(eff.get("param"))
        if rec is None or sub is None or rec.get("coefficient") is None:
            return None, [f"SCM-selected effect {eff} has no estimate in the final fit: cov_effects left null."]
        out = {**out, sub: {**out.get(sub, {}), eff["covariate"]: rec["coefficient"]}}
    return out, []


def _software(state: PharmState) -> dict[str, str] | None:
    """Software versions, only from a run provenance record verified against the audit chain."""
    prov = dict(state.run_provenance or ())
    if state.run_provenance_status != "verified" or not prov:
        return None
    return {"PharmAgent": f"{prov.get('app_version', 'n/a')} (git {prov.get('git_sha', 'n/a')})",
            **{k: prov[k] for k in _SOFTWARE_KEYS if k in prov}}


def _analysis_steps(state: PharmState, final: dict | None, source: str | None) -> list[str]:
    steps: list[str] = []
    meta = state.dataset_metadata or {}
    if meta:
        steps.append(f"Dataset {meta.get('dataset_id')}: {meta.get('n_records')} records, "
                     f"{meta.get('n_subjects')} subjects; roles {meta.get('detected_roles')}.")
    pk = _ok(state.pk_model_results)
    if pk and pk.get("mode") == "compare":
        order = " < ".join(r.get("model_key", "?") for r in pk.get("ranking") or [])
        steps.append(f"Structural comparison (two-stage fits, total AIC, best first): {order}.")
    if final is not None:
        steps.append(f"Final model ({source}): {final.get('label')} [{final.get('model_key')}], "
                     f"{final.get('method')}, OFV {final.get('ofv')}, IIV on {final.get('iiv_params')}, "
                     f"{final.get('error_model')} residual error, {final.get('n_blq', 0)} censored "
                     f"records under M3; {final.get('iterations')} iterations, condition number "
                     f"{final.get('condition_number')}, covariance: {final.get('cov_note')}.")
        steps.append("Body weight: allometric exponents fixed at 0.75 (CL, Q) and 1.0 (volumes), "
                     "centred at 70 kg; not estimated, so not reported under cov_effects.")
    scm = _ok(state.scm_results)
    if scm:
        sel = ", ".join(f"{e.get('param')}~{e.get('covariate')}" for e in scm.get("selected") or []) or "none"
        out = "; ".join(f"{d.get('covariate')} on {d.get('param')} ({d.get('reason')})"
                        for d in scm.get("screened_out") or []) or "none"
        steps.append(f"SCM: forward p {scm.get('forward_p')}, backward p {scm.get('backward_p')}, "
                     f"{scm.get('n_candidates')} candidate(s), selected {sel}; not tested: {out}.")
    return steps


def _answers(state: PharmState) -> tuple[dict[str, Any], dict | None, str | None, list[str]]:
    answers: dict[str, Any] = dict.fromkeys(ANSWER_KEYS)
    problem = _dataset_problem(state)
    if problem:
        return answers, None, None, [problem]
    final, source, notes = _final_model(state)
    if final is None:
        return answers, None, None, notes
    mapping = _IV_LINEAR.get(final.get("model_key"))
    if mapping is None:
        return answers, final, source, [*notes, f"Final model {final.get('model_key')!r} is outside the IV "
                                        "linear mapping (scenario_00 is a 1-hour IV infusion); " + _LEFT_NULL]
    ncmt, names = mapping
    (disp, n1), (err, n2), (cov, n3) = (_disposition(final, names), _error_terms(final),
                                        _cov_effects(state, final, names))
    answers = {**answers, "structural_ncmt": ncmt, "disposition_params": disp,
               "error_model": err, "cov_effects": cov}
    return answers, final, source, [*notes, *n1, *n2, *n3]


def build_submission(state: PharmState, *,
                     run_info: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Fill submission.template.yaml from ``state``; ``None`` = not answered.

    ``run_info`` carries what PharmState does not record (harness, model,
    harness_version, run_utc); unknown keys raise ``ValueError``. A quarantined
    session (run provenance mismatch) raises ``ProvenanceQuarantined``.
    """
    if state.run_provenance_status == QUARANTINED_STATUS:
        raise ProvenanceQuarantined("session is quarantined (run_provenance_status=mismatch); "
                                    "no submission is built from it")
    info = dict(run_info or {})
    unknown = sorted(set(info) - set(RUN_INFO_KEYS))
    if unknown:
        raise ValueError(f"unknown run_info keys: {unknown}")
    answers, final, source, notes = _answers(state)
    # outlier_records stays None: PharmAgent has no record-level outlier screen.
    notes = [*notes, "outlier_records: not addressed (no record-level outlier screen in PharmAgent)."]
    provenance = {"tool": "PharmAgent", "harness": info.get("harness"), "model": info.get("model"),
                  "harness_version": info.get("harness_version"), "software": _software(state),
                  "run_utc": info.get("run_utc"),
                  "analysis_steps": _analysis_steps(state, final, source) + notes}
    return {"provenance": provenance, "answers": answers}


# ─────────────────────────────────── CLI ────────────────────────────────────

_YAML_HEADER = ("# submission.yaml for PMB-MAB-01, written by PharmAgent (app.benchmarks.pmxbench).\n"
                "# null = not addressed by the analysis (sap.md §9: report only what the analysis supports).\n")


def _refuse_overwrite(paths: list[Path], force: bool) -> None:
    taken = [str(p) for p in paths if p.exists()]
    if taken and not force:
        raise ValueError(f"refusing to overwrite {taken}; pass --force")


def _cmd_prepare(ns: argparse.Namespace) -> str:
    from app.core.provenance import file_sha256
    from app.tools.data_tools import NONMEM_NA_VALUES

    src, out = Path(ns.data), Path(ns.out)
    rep = Path(ns.report) if ns.report else out.with_suffix(".report.json")
    _refuse_overwrite([out, rep], ns.force)
    prepared, report = prepare_dataset(pd.read_csv(src, na_values=NONMEM_NA_VALUES), lloq=ns.lloq)
    report = {**report, "source_file": src.name, "source_sha256": file_sha256(src)}
    prepared.to_csv(out, index=False, na_rep=".")
    rep.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return (f"wrote {out} ({report['n_records']} records, {report['n_censored']} censored) "
            f"and {rep}")


def _cmd_submit(ns: argparse.Namespace) -> str:
    out = Path(ns.out)
    _refuse_overwrite([out], ns.force)
    state = PharmState.model_validate(json.loads(Path(ns.state).read_text(encoding="utf-8")))
    run_info = {k: getattr(ns, k) for k in RUN_INFO_KEYS if getattr(ns, k) is not None}
    sub = build_submission(state, run_info=run_info)
    out.write_text(_YAML_HEADER + yaml.safe_dump(sub, sort_keys=False, allow_unicode=True),
                   encoding="utf-8")
    unanswered = [k for k, v in sub["answers"].items() if v is None]
    return f"wrote {out}; unanswered (null): {unanswered or 'none'}"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m app.benchmarks.pmxbench", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    prep = sub.add_parser("prepare", help="scenario data.csv -> PharmAgent analysis CSV + report JSON")
    prep.add_argument("--data", required=True)
    prep.add_argument("--out", required=True)
    prep.add_argument("--report", help="default: <out>.report.json")
    prep.add_argument("--lloq", type=float, default=SCENARIO_00_LLOQ)
    prep.add_argument("--force", action="store_true")
    sm = sub.add_parser("submit", help="PharmState JSON (GET /api/sessions/{id}/state) -> submission.yaml")
    sm.add_argument("--state", required=True)
    sm.add_argument("--out", required=True)
    for k in RUN_INFO_KEYS:
        sm.add_argument(f"--{k.replace('_', '-')}", dest=k)
    sm.add_argument("--force", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    ns = _parser().parse_args(argv)
    try:
        msg = _cmd_prepare(ns) if ns.cmd == "prepare" else _cmd_submit(ns)
    except (ValueError, OSError, RuntimeError, ProvenanceQuarantined) as exc:
        print(f"pmxbench {ns.cmd}: {exc}", file=sys.stderr)
        return 2
    print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
