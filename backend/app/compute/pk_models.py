"""Structural PK / PK-PD model library (ported from the PopPK Workbench).

Each model is an ODE system defined as pure Python. Conventions match the
mrgsolve registry they were ported from:
  - Allometric scaling: CL / Q / Vmax ^0.75, volumes ^1.0, centered at 70 kg.
  - Compartment 0 is the dosing compartment (CENT for IV, DEPOT/TR1 for oral).
  - CP (central concentration) = central amount / central volume.
  - PK/PD models add a response state or an algebraic effect on top of a
    1-compartment oral PK base.

Parameters passed to ``rhs``/``cp``/``eff`` are the REALIZED (already
WT-scaled) structural values, e.g. {"CL": ..., "V": ..., "KA": ...}.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PKModel:
    key: str
    label: str
    group: str                       # IV linear | Oral | Nonlinear | PK/PD
    is_iv: bool
    has_pd: bool
    params: tuple[str, ...]          # fittable structural params (realized names)
    defaults: dict[str, float]
    n_cmt: int
    dose_cmt: int                    # 0-based index where dose enters
    allometric: dict[str, float]     # param -> WT exponent
    rhs: Callable[[float, Any, dict], list[float]]
    cp: Callable[[Any, dict], float]
    init_state: Callable[[dict], list[float]] | None = None
    eff: Callable[[Any, dict], float] | None = None
    lag_param: str | None = None     # name of a lag-time param, if any
    # Constant system matrix A (dy/dt = A y) for LINEAR models -> the simulator
    # propagates via matrix exponential (exact, fast). None => use the ODE solver.
    amat: Callable[[dict], np.ndarray] | None = None


# Standard allometric exponent for a named parameter.
def _allo(*names: str) -> dict[str, float]:
    """0.75 on flows (CL/Q/Vmax), 1.0 on volumes, 0 elsewhere."""
    out: dict[str, float] = {}
    for n in names:
        if n in ("CL", "Q", "Q2", "Q3", "VMAX"):
            out[n] = 0.75
        elif n in ("V", "VC", "VP", "VP2", "VP3"):
            out[n] = 1.0
    return out


# ─────────────────────────── PK models ──────────────────────────────────────

def _iv1(t, y, p):
    CENT = y[0]
    return [-(p["CL"] / p["V"]) * CENT]

def _iv2(t, y, p):
    CENT, PER = y
    CL, VC, Q, VP = p["CL"], p["VC"], p["Q"], p["VP"]
    return [-(CL / VC) * CENT - (Q / VC) * CENT + (Q / VP) * PER,
            (Q / VC) * CENT - (Q / VP) * PER]

def _iv3(t, y, p):
    CENT, P2, P3 = y
    CL, VC, Q2, VP2, Q3, VP3 = p["CL"], p["VC"], p["Q2"], p["VP2"], p["Q3"], p["VP3"]
    return [-(CL / VC) * CENT - (Q2 / VC) * CENT + (Q2 / VP2) * P2 - (Q3 / VC) * CENT + (Q3 / VP3) * P3,
            (Q2 / VC) * CENT - (Q2 / VP2) * P2,
            (Q3 / VC) * CENT - (Q3 / VP3) * P3]

def _oral1(t, y, p):
    DEPOT, CENT = y
    return [-p["KA"] * DEPOT,
            p["KA"] * DEPOT - (p["CL"] / p["V"]) * CENT]

def _oral2(t, y, p):
    DEPOT, CENT, PER = y
    KA, CL, VC, Q, VP = p["KA"], p["CL"], p["VC"], p["Q"], p["VP"]
    return [-KA * DEPOT,
            KA * DEPOT - (CL / VC) * CENT - (Q / VC) * CENT + (Q / VP) * PER,
            (Q / VC) * CENT - (Q / VP) * PER]

def _oral1_transit(t, y, p):
    TR1, TR2, TR3, CENT = y
    KTR = 4.0 / p["MTT"]
    return [-KTR * TR1,
            KTR * (TR1 - TR2),
            KTR * (TR2 - TR3),
            KTR * TR3 - (p["CL"] / p["V"]) * CENT]

def _iv1_mm(t, y, p):
    CENT = y[0]
    C = CENT / p["V"]
    return [-p["VMAX"] * C / (p["KM"] + C)]

def _oral1_mm(t, y, p):
    DEPOT, CENT = y
    C = CENT / p["V"]
    return [-p["KA"] * DEPOT,
            p["KA"] * DEPOT - p["VMAX"] * C / (p["KM"] + C)]

def _iv1_mixed(t, y, p):
    CENT = y[0]
    C = CENT / p["V"]
    return [-(p["CL"] / p["V"]) * CENT - p["VMAX"] * C / (p["KM"] + C)]


# ── linear system matrices (dy/dt = A y) for matrix-exponential propagation ──
def _A_iv1(p):
    return np.array([[-(p["CL"] / p["V"])]], dtype=float)
def _A_iv2(p):
    CL, VC, Q, VP = p["CL"], p["VC"], p["Q"], p["VP"]
    return np.array([[-(CL + Q) / VC, Q / VP], [Q / VC, -Q / VP]], dtype=float)
def _A_iv3(p):
    CL, VC, Q2, VP2, Q3, VP3 = p["CL"], p["VC"], p["Q2"], p["VP2"], p["Q3"], p["VP3"]
    return np.array([[-(CL + Q2 + Q3) / VC, Q2 / VP2, Q3 / VP3],
                     [Q2 / VC, -Q2 / VP2, 0.0],
                     [Q3 / VC, 0.0, -Q3 / VP3]], dtype=float)
def _A_oral1(p):
    return np.array([[-p["KA"], 0.0], [p["KA"], -(p["CL"] / p["V"])]], dtype=float)
def _A_oral2(p):
    KA, CL, VC, Q, VP = p["KA"], p["CL"], p["VC"], p["Q"], p["VP"]
    return np.array([[-KA, 0.0, 0.0],
                     [KA, -(CL + Q) / VC, Q / VP],
                     [0.0, Q / VC, -Q / VP]], dtype=float)
def _A_transit(p):
    KTR = 4.0 / p["MTT"]
    return np.array([[-KTR, 0, 0, 0], [KTR, -KTR, 0, 0],
                     [0, KTR, -KTR, 0], [0, 0, KTR, -(p["CL"] / p["V"])]], dtype=float)
def _A_effect(p):  # [DEPOT, CENT, CE]; d/dt CE = (KE0/V)*CENT - KE0*CE
    KA, CL, V, KE0 = p["KA"], p["CL"], p["V"], p["KE0"]
    return np.array([[-KA, 0.0, 0.0],
                     [KA, -CL / V, 0.0],
                     [0.0, KE0 / V, -KE0]], dtype=float)


def _cp_v(y, p):    return y[0] / p["V"]
def _cp_vc(y, p):   return y[0] / p["VC"]
def _cp_v_2(y, p):  return y[1] / p["V"]    # central is index 1 (oral)
def _cp_vc_2(y, p): return y[1] / p["VC"]
def _cp_v_3(y, p):  return y[3] / p["V"]    # transit: central index 3


# ─────────────────────────── PK/PD models ───────────────────────────────────
# All PK/PD models use a 1-cmt oral PK base: state = [DEPOT, CENT, (PD...)].

def _pkpd_pk(t, depot, cent, p):
    """Shared depot/central derivatives for the PK base."""
    return -p["KA"] * depot, p["KA"] * depot - (p["CL"] / p["V"]) * cent


def _pd_direct_rhs(t, y, p):
    DEPOT, CENT = y[0], y[1]
    dd, dc = _pkpd_pk(t, DEPOT, CENT, p)
    return [dd, dc]

def _eff_linear(y, p):   return p["E0"] + p["SLOPE"] * (y[1] / p["V"])
def _eff_emax(y, p):
    cp = y[1] / p["V"]
    return p["E0"] + p["EMAX"] * cp / (p["EC50"] + cp)
def _eff_sigmoid(y, p):
    cp = y[1] / p["V"]
    h = p["HILL"]
    return p["E0"] + p["EMAX"] * cp**h / (p["EC50"]**h + cp**h)

def _effect_cmt_rhs(t, y, p):
    DEPOT, CENT, CE = y
    dd, dc = _pkpd_pk(t, DEPOT, CENT, p)
    cp = CENT / p["V"]
    return [dd, dc, p["KE0"] * (cp - CE)]
def _eff_effect_emax(y, p):
    ce = y[2]
    return p["E0"] + p["EMAX"] * ce / (p["EC50"] + ce)

def _idr_init(p):       return [0.0, 0.0, p["KIN"] / p["KOUT"]]
def _idr1_rhs(t, y, p):  # inhibit kin
    DEPOT, CENT, RESP = y
    dd, dc = _pkpd_pk(t, DEPOT, CENT, p)
    cp = CENT / p["V"]
    return [dd, dc, p["KIN"] * (1 - p["IMAX"] * cp / (p["IC50"] + cp)) - p["KOUT"] * RESP]
def _idr2_rhs(t, y, p):  # inhibit kout
    DEPOT, CENT, RESP = y
    dd, dc = _pkpd_pk(t, DEPOT, CENT, p)
    cp = CENT / p["V"]
    return [dd, dc, p["KIN"] - p["KOUT"] * (1 - p["IMAX"] * cp / (p["IC50"] + cp)) * RESP]
def _idr3_rhs(t, y, p):  # stimulate kin
    DEPOT, CENT, RESP = y
    dd, dc = _pkpd_pk(t, DEPOT, CENT, p)
    cp = CENT / p["V"]
    return [dd, dc, p["KIN"] * (1 + p["EMAX"] * cp / (p["EC50"] + cp)) - p["KOUT"] * RESP]
def _idr4_rhs(t, y, p):  # stimulate kout
    DEPOT, CENT, RESP = y
    dd, dc = _pkpd_pk(t, DEPOT, CENT, p)
    cp = CENT / p["V"]
    return [dd, dc, p["KIN"] - p["KOUT"] * (1 + p["EMAX"] * cp / (p["EC50"] + cp)) * RESP]
def _eff_resp(y, p):    return y[2]
def _cp_pkpd(y, p):     return y[1] / p["V"]


REGISTRY: dict[str, PKModel] = {
    "iv_1cmt": PKModel("iv_1cmt", "1-cmt IV (linear)", "IV linear", True, False,
        ("CL", "V"), {"CL": 5, "V": 50}, 1, 0, _allo("CL", "V"), _iv1, _cp_v, amat=_A_iv1),
    "iv_2cmt": PKModel("iv_2cmt", "2-cmt IV (linear)", "IV linear", True, False,
        ("CL", "VC", "Q", "VP"), {"CL": 5, "VC": 30, "Q": 10, "VP": 50}, 2, 0,
        _allo("CL", "VC", "Q", "VP"), _iv2, _cp_vc, amat=_A_iv2),
    "iv_3cmt": PKModel("iv_3cmt", "3-cmt IV (linear)", "IV linear", True, False,
        ("CL", "VC", "Q2", "VP2", "Q3", "VP3"),
        {"CL": 5, "VC": 30, "Q2": 10, "VP2": 50, "Q3": 2, "VP3": 100}, 3, 0,
        _allo("CL", "VC", "Q2", "VP2", "Q3", "VP3"), _iv3, _cp_vc, amat=_A_iv3),
    "oral_1cmt": PKModel("oral_1cmt", "1-cmt oral (linear)", "Oral", False, False,
        ("CL", "V", "KA"), {"CL": 5, "V": 50, "KA": 1}, 2, 0, _allo("CL", "V"),
        _oral1, _cp_v_2, amat=_A_oral1),
    "oral_1cmt_lag": PKModel("oral_1cmt_lag", "1-cmt oral + lag", "Oral", False, False,
        ("CL", "V", "KA", "ALAG"), {"CL": 5, "V": 50, "KA": 1, "ALAG": 0.3}, 2, 0,
        _allo("CL", "V"), _oral1, _cp_v_2, lag_param="ALAG", amat=_A_oral1),
    "oral_2cmt": PKModel("oral_2cmt", "2-cmt oral (linear)", "Oral", False, False,
        ("CL", "VC", "Q", "VP", "KA"), {"CL": 5, "VC": 50, "Q": 10, "VP": 100, "KA": 1},
        3, 0, _allo("CL", "VC", "Q", "VP"), _oral2, _cp_vc_2, amat=_A_oral2),
    "oral_1cmt_transit": PKModel("oral_1cmt_transit", "1-cmt oral transit abs.", "Oral",
        False, False, ("CL", "V", "MTT"), {"CL": 5, "V": 50, "MTT": 1}, 4, 0,
        _allo("CL", "V"), _oral1_transit, _cp_v_3, amat=_A_transit),
    "iv_1cmt_mm": PKModel("iv_1cmt_mm", "1-cmt IV Michaelis-Menten", "Nonlinear", True,
        False, ("VMAX", "KM", "V"), {"VMAX": 100, "KM": 5, "V": 50}, 1, 0,
        _allo("VMAX", "V"), _iv1_mm, _cp_v),
    "oral_1cmt_mm": PKModel("oral_1cmt_mm", "1-cmt oral Michaelis-Menten", "Nonlinear",
        False, False, ("VMAX", "KM", "V", "KA"), {"VMAX": 100, "KM": 5, "V": 50, "KA": 1},
        2, 0, _allo("VMAX", "V"), _oral1_mm, _cp_v_2),
    "iv_1cmt_mixed": PKModel("iv_1cmt_mixed", "1-cmt IV mixed (lin + MM)", "Nonlinear",
        True, False, ("CL", "VMAX", "KM", "V"), {"CL": 5, "VMAX": 100, "KM": 5, "V": 50},
        1, 0, _allo("CL", "VMAX", "V"), _iv1_mixed, _cp_v),

    # ── PK/PD (1-cmt oral PK base) ──
    "pkpd_direct_linear": PKModel("pkpd_direct_linear", "Direct linear PD", "PK/PD",
        False, True, ("CL", "V", "KA", "E0", "SLOPE"),
        {"CL": 5, "V": 50, "KA": 1, "E0": 10, "SLOPE": 5}, 2, 0, _allo("CL", "V"),
        _pd_direct_rhs, _cp_pkpd, eff=_eff_linear, amat=_A_oral1),
    "pkpd_direct_emax": PKModel("pkpd_direct_emax", "Direct Emax PD", "PK/PD",
        False, True, ("CL", "V", "KA", "E0", "EMAX", "EC50"),
        {"CL": 5, "V": 50, "KA": 1, "E0": 10, "EMAX": 100, "EC50": 2}, 2, 0,
        _allo("CL", "V"), _pd_direct_rhs, _cp_pkpd, eff=_eff_emax, amat=_A_oral1),
    "pkpd_direct_sigmoid": PKModel("pkpd_direct_sigmoid", "Direct sigmoid Emax PD",
        "PK/PD", False, True, ("CL", "V", "KA", "E0", "EMAX", "EC50", "HILL"),
        {"CL": 5, "V": 50, "KA": 1, "E0": 10, "EMAX": 100, "EC50": 2, "HILL": 2}, 2, 0,
        _allo("CL", "V"), _pd_direct_rhs, _cp_pkpd, eff=_eff_sigmoid, amat=_A_oral1),
    "pkpd_effect_emax": PKModel("pkpd_effect_emax", "Effect-cmt Emax (hysteresis)",
        "PK/PD", False, True, ("CL", "V", "KA", "E0", "EMAX", "EC50", "KE0"),
        {"CL": 5, "V": 50, "KA": 1, "E0": 10, "EMAX": 100, "EC50": 2, "KE0": 0.3}, 3, 0,
        _allo("CL", "V"), _effect_cmt_rhs, _cp_pkpd, eff=_eff_effect_emax, amat=_A_effect),
    "pkpd_idr1_inhib_kin": PKModel("pkpd_idr1_inhib_kin", "IDR I — inhibit kin", "PK/PD",
        False, True, ("CL", "V", "KA", "KIN", "KOUT", "IMAX", "IC50"),
        {"CL": 5, "V": 50, "KA": 1, "KIN": 10, "KOUT": 1, "IMAX": 0.9, "IC50": 2}, 3, 0,
        _allo("CL", "V"), _idr1_rhs, _cp_pkpd, init_state=_idr_init, eff=_eff_resp),
    "pkpd_idr2_inhib_kout": PKModel("pkpd_idr2_inhib_kout", "IDR II — inhibit kout",
        "PK/PD", False, True, ("CL", "V", "KA", "KIN", "KOUT", "IMAX", "IC50"),
        {"CL": 5, "V": 50, "KA": 1, "KIN": 10, "KOUT": 1, "IMAX": 0.9, "IC50": 2}, 3, 0,
        _allo("CL", "V"), _idr2_rhs, _cp_pkpd, init_state=_idr_init, eff=_eff_resp),
    "pkpd_idr3_stim_kin": PKModel("pkpd_idr3_stim_kin", "IDR III — stimulate kin",
        "PK/PD", False, True, ("CL", "V", "KA", "KIN", "KOUT", "EMAX", "EC50"),
        {"CL": 5, "V": 50, "KA": 1, "KIN": 10, "KOUT": 1, "EMAX": 4, "EC50": 2}, 3, 0,
        _allo("CL", "V"), _idr3_rhs, _cp_pkpd, init_state=_idr_init, eff=_eff_resp),
    "pkpd_idr4_stim_kout": PKModel("pkpd_idr4_stim_kout", "IDR IV — stimulate kout",
        "PK/PD", False, True, ("CL", "V", "KA", "KIN", "KOUT", "EMAX", "EC50"),
        {"CL": 5, "V": 50, "KA": 1, "KIN": 10, "KOUT": 1, "EMAX": 4, "EC50": 2}, 3, 0,
        _allo("CL", "V"), _idr4_rhs, _cp_pkpd, init_state=_idr_init, eff=_eff_resp),
}

PK_KEYS = [k for k, m in REGISTRY.items() if not m.has_pd]
PKPD_KEYS = [k for k, m in REGISTRY.items() if m.has_pd]


def list_models() -> list[dict[str, Any]]:
    """Registry metadata for the UI (grouped model picker)."""
    return [{"key": m.key, "label": m.label, "group": m.group, "is_iv": m.is_iv,
             "has_pd": m.has_pd, "params": list(m.params)} for m in REGISTRY.values()]


def get_model(key: str) -> PKModel:
    if key not in REGISTRY:
        raise KeyError(f"unknown PK model: {key}")
    return REGISTRY[key]
