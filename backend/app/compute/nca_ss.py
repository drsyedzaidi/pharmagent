"""Steady-state non-compartmental analysis (multiple-dose).

Operates on one dosing-interval profile per subject (time-after-dose over
[0, tau]). Reuses the validated linear-up/log-down AUC and best-fit lambda_z
from ``nca`` but reports interval (tau) exposures instead of single-dose
0->inf exposures:

    AUC_tau, Cmax_ss, Cmin_ss, Cavg_ss, CL_F = Dose/AUC_tau, fluctuation %,
    swing %, accumulation ratio, lambda_z, t_half, Vz_F.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from app.compute.nca import _auc_intervals, _best_lambda_z, summarize_by_dose


def nca_ss_subject(subject: Any, tad, conc, dose: float, tau: float) -> dict[str, Any]:
    """Steady-state NCA parameters for one subject's dosing interval."""
    order = np.argsort(np.asarray(tad, dtype=float))
    t = np.asarray(tad, dtype=float)[order]
    c = np.asarray(conc, dtype=float)[order]

    cmax = float(np.max(c))
    tmax = float(t[int(np.argmax(c))])
    cmin = float(np.min(c))
    ctrough = float(c[0])
    auc_tau, _ = _auc_intervals(t, c)
    cavg = auc_tau / tau if tau > 0 else float("nan")
    cl_f = dose / auc_tau if auc_tau > 0 else float("nan")
    fluctuation = 100.0 * (cmax - cmin) / cavg if cavg and cavg > 0 else None
    swing = 100.0 * (cmax - cmin) / cmin if cmin > 0 else None

    out: dict[str, Any] = {
        "subject": subject,
        "dose": dose,
        "tau": tau,
        "steady_state": True,
        "Cmax": round(cmax, 6),
        "Tmax": round(tmax, 6),
        "Cmin": round(cmin, 6),
        "Ctrough": round(ctrough, 6),
        "AUC_tau": round(auc_tau, 6),
        "AUC_last": round(auc_tau, 6),   # interval AUC (alias for table reuse)
        "AUC_inf": None,                 # not defined for steady-state interval
        "Cavg": round(cavg, 6) if math.isfinite(cavg) else None,
        "CL_F": round(cl_f, 6) if math.isfinite(cl_f) else None,
        "fluctuation_pct": round(fluctuation, 4) if fluctuation is not None else None,
        "swing_pct": round(swing, 4) if swing is not None else None,
        "pct_AUC_extrap": None,
    }

    lz = _best_lambda_z(t, c, tmax)
    if lz and math.isfinite(cl_f):
        lam = lz["lambda_z"]
        vz_f = cl_f / lam
        rac = 1.0 / (1.0 - math.exp(-lam * tau)) if tau > 0 else None
        out.update({
            "lambda_z": round(lam, 6),
            "t_half": round(math.log(2) / lam, 6),
            "Vz_F": round(vz_f, 6),
            "accumulation_ratio": round(rac, 4) if rac is not None else None,
            "lambda_z_r2_adj": lz["lambda_z_r2_adj"],
            "lambda_z_n_points": lz["lambda_z_n_points"],
        })
    else:
        out.update({k: None for k in (
            "lambda_z", "t_half", "Vz_F", "accumulation_ratio",
            "lambda_z_r2_adj", "lambda_z_n_points")})
    return out


def run_nca_ss(profiles: dict[Any, Any]) -> dict[str, Any]:
    """Top-level: steady-state NCA over per-subject interval profiles.

    ``profiles`` maps subject -> ``app.compute.dosing.SSProfile``.
    Returns the same shape as ``run_nca`` plus ``steady_state`` flags.
    """
    params = [
        nca_ss_subject(p.subject, p.tad, p.conc, float(p.dose), float(p.tau))
        for p in profiles.values()
    ]
    params.sort(key=lambda r: str(r["subject"]))
    summary = summarize_by_dose(params)
    summary["steady_state"] = True
    return {"nca_parameters": params, "nca_summary": summary, "steady_state": True}
