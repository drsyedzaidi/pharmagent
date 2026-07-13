"""CSV exporters for analysis results held in PharmState.

Pure functions that turn a result dict into a CSV string — used by the download
endpoints so any tabular analysis (NCA, BE, dose-proportionality, PK-model fit,
dose sweep, VPC) can be exported, not just the NCA DOCX report.
"""
from __future__ import annotations

import csv
import io
from typing import Any

from app.compute.nmexport import build_mrgsolve, build_nonmem
from app.core.pharmstate import PharmState


def _csv(rows: list[dict[str, Any]], columns: list[str]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()


def _nca(state: PharmState) -> str:
    params = state.nca_parameters or []
    if not params:
        return ""
    cols = list({k for p in params for k in p.keys() if not isinstance(p.get(k), (list, dict))})
    preferred = ["subject", "dose", "route", "Cmax", "Tmax", "AUC_last", "AUC_inf",
                 "AUC_tau", "Cmin", "Cavg", "t_half", "CL_F", "Vz_F", "Vss", "MRT",
                 "fluctuation_pct", "accumulation_ratio", "pct_AUC_extrap",
                 "lambda_z", "lambda_z_r2_adj"]
    ordered = [c for c in preferred if c in cols] + [c for c in cols if c not in preferred]
    return _csv(params, ordered)


def _be(state: PharmState) -> str:
    r = state.be_results or {}
    params = r.get("parameters") or {}
    rows = [{"parameter": k, **v} for k, v in params.items()]
    return _csv(rows, ["parameter", "gmr_pct", "ci_lower_pct", "ci_upper_pct",
                       "within_limits", "cv_intra_pct"])


def _dose_prop(state: PharmState) -> str:
    r = state.dose_prop_results or {}
    params = r.get("parameters") or {}
    rows = [{"parameter": k, **v} for k, v in params.items()]
    return _csv(rows, ["parameter", "slope", "slope_ci_lower", "slope_ci_upper",
                       "r_squared", "dose_ratio", "proportional"])


def _pk_model(state: PharmState) -> str:
    r = state.pk_model_results or {}
    fits = r.get("individual_fits")
    if not fits and r.get("best"):
        fits = r["best"].get("individual_fits")
    fits = fits or []
    rows = []
    for f in fits:
        row = {"subject": f.get("subject"), "converged": f.get("converged"),
               "aic": f.get("aic"), "r_squared": f.get("r_squared")}
        row.update(f.get("params") or {})
        rse = f.get("rse_pct") or {}
        row.update({f"{k}_RSE%": v for k, v in rse.items()})
        rows.append(row)
    cols = list({k for r0 in rows for k in r0})
    head = [c for c in ["subject", "converged", "aic", "r_squared"] if c in cols]
    return _csv(rows, head + [c for c in cols if c not in head])


def _dose_sweep(state: PharmState) -> str:
    r = state.dose_sweep_results or {}
    rows = [{"dose": p["dose"], "cmax": p["cmax"], "auc_tau": p["auc_tau"],
             "cavg": p["cavg"], "ctrough": p["ctrough"]} for p in (r.get("profiles") or [])]
    return _csv(rows, ["dose", "cmax", "auc_tau", "cavg", "ctrough"])


def _vpc(state: PharmState) -> str:
    r = state.vpc_results or {}
    ovp = r.get("obs_vs_pred") or {}
    o, ip, pr = ovp.get("observed", []), ovp.get("ipred", []), ovp.get("pred", [])
    rows = [{"observed": o[i], "ipred": ip[i], "pred": pr[i]} for i in range(len(o))]
    return _csv(rows, ["observed", "ipred", "pred"])


def _nlme(state: PharmState) -> str:
    r = state.nlme_results or {}
    if r.get("status") != "ok":
        return ""
    theta = r.get("theta") or {}
    omega = r.get("omega_cv_pct") or {}
    rse = r.get("theta_rse_pct") or {}
    shr = r.get("shrinkage_pct") or {}
    rows = [{"parameter": p, "typical": v, "rse_pct": rse.get(p),
             "iiv_cv_pct": omega.get(p), "shrinkage_pct": shr.get(p)} for p, v in theta.items()]
    sig = r.get("sigma") or {}
    rows.append({"parameter": "sigma_prop", "typical": sig.get("prop")})
    rows.append({"parameter": "sigma_add", "typical": sig.get("add")})
    rows.append({"parameter": "OFV", "typical": r.get("ofv")})
    return _csv(rows, ["parameter", "typical", "rse_pct", "iiv_cv_pct", "shrinkage_pct"])


_EXPORTERS = {
    "nca": _nca, "be": _be, "dose_prop": _dose_prop, "pk_model": _pk_model,
    "dose_sweep": _dose_sweep, "vpc": _vpc, "nlme": _nlme,
}


def available(state: PharmState) -> list[str]:
    """Which export kinds currently have data."""
    out = []
    if state.nca_parameters:
        out.append("nca")
    if (state.be_results or {}).get("status") == "ok":
        out.append("be")
    if (state.dose_prop_results or {}).get("status") == "ok":
        out.append("dose_prop")
    if (state.pk_model_results or {}).get("status") == "ok":
        out.append("pk_model")
    if (state.nlme_results or {}).get("status") == "ok":
        out.append("nlme")
    if (state.dose_sweep_results or {}).get("status") == "ok":
        out.append("dose_sweep")
    if (state.vpc_results or {}).get("status") == "ok":
        out.append("vpc")
    return out


def export_csv(state: PharmState, kind: str) -> str:
    if kind not in _EXPORTERS:
        raise ValueError(f"unknown export kind: {kind}")
    return _EXPORTERS[kind](state)


# ── control-stream exports (NONMEM / mrgsolve), seeded from the NLME fit ──────
_CONTROL = {"nonmem": (build_nonmem, "ctl"), "mrgsolve": (build_mrgsolve, "cpp")}


def control_available(state: PharmState) -> list[str]:
    """Which control-stream exports are possible (need a fitted NLME model whose
    structural form has a closed-form mapping)."""
    nl = state.nlme_results or {}
    if nl.get("status") != "ok":
        return []
    return [kind for kind, (fn, _ext) in _CONTROL.items() if fn(nl) is not None]


def export_control(state: PharmState, kind: str) -> tuple[str, str]:
    """Return (text, file_extension) for a NONMEM/mrgsolve control stream."""
    if kind not in _CONTROL:
        raise ValueError(f"unknown control export: {kind}")
    nl = state.nlme_results or {}
    if nl.get("status") != "ok":
        raise ValueError("run a population (NLME) fit first")
    fn, ext = _CONTROL[kind]
    text = fn(nl)
    if not text:
        raise ValueError(f"{kind} export unavailable for this structural model "
                         "(no closed-form mapping)")
    return text, ext
