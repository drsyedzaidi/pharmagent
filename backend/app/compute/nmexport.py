"""Export a fitted population model as a NONMEM control stream or an mrgsolve
model, seeded with the estimated THETA / OMEGA / SIGMA and covariate effects.

This closes the round-trip: a model fitted in PharmAgent can be continued or
verified in the user's own estimator (NONMEM) or simulator (mrgsolve).

Supported structural models are the closed-form library compartmental models
(1/2/3-compartment IV and oral, with optional absorption lag). NONMEM uses the
matching ADVAN/TRANS subroutines; mrgsolve uses an explicit $ODE block. Models
without a closed-form mapping (transit absorption, Michaelis-Menten, PK/PD)
return a note rather than a wrong control stream.

Covariate effects (power / linear / exponential) are written into $PK / $MAIN;
categorical effects are flagged for manual IF-block coding (kept honest rather
than guessed).
"""
from __future__ import annotations

import math
from typing import Any


def _cv_to_omega2(cv_pct: float) -> float:
    """Lognormal %CV -> variance (NONMEM/mrgsolve OMEGA element)."""
    return math.log(1.0 + (float(cv_pct) / 100.0) ** 2)


# our-param -> (NONMEM param symbol); ADVAN/TRANS + scaling per model.
_NONMEM: dict[str, dict[str, Any]] = {
    "iv_1cmt": {"advan": 1, "trans": 2, "depot": False, "s_cmt": 1, "s_sym": "V",
                "map": [("CL", "CL"), ("V", "V")]},
    "oral_1cmt": {"advan": 2, "trans": 2, "depot": True, "s_cmt": 2, "s_sym": "V",
                  "map": [("CL", "CL"), ("V", "V"), ("KA", "KA")]},
    "oral_1cmt_lag": {"advan": 2, "trans": 2, "depot": True, "s_cmt": 2, "s_sym": "V",
                      "map": [("CL", "CL"), ("V", "V"), ("KA", "KA")],
                      "alag": ("ALAG", 1)},
    "iv_2cmt": {"advan": 3, "trans": 4, "depot": False, "s_cmt": 1, "s_sym": "V1",
                "map": [("CL", "CL"), ("VC", "V1"), ("Q", "Q"), ("VP", "V2")]},
    "oral_2cmt": {"advan": 4, "trans": 4, "depot": True, "s_cmt": 2, "s_sym": "V2",
                  "map": [("CL", "CL"), ("VC", "V2"), ("Q", "Q"), ("VP", "V3"), ("KA", "KA")]},
    "iv_3cmt": {"advan": 11, "trans": 4, "depot": False, "s_cmt": 1, "s_sym": "V1",
                "map": [("CL", "CL"), ("VC", "V1"), ("Q2", "Q2"), ("VP2", "V2"),
                        ("Q3", "Q3"), ("VP3", "V3")]},
}

# mrgsolve ODE structure: number of disposition compartments + depot flag.
_MRG: dict[str, dict[str, Any]] = {
    "iv_1cmt": {"n": 1, "depot": False, "map": [("CL", "CL"), ("V", "V")]},
    "oral_1cmt": {"n": 1, "depot": True, "map": [("CL", "CL"), ("V", "V"), ("KA", "KA")]},
    "oral_1cmt_lag": {"n": 1, "depot": True, "lag": "ALAG",
                      "map": [("CL", "CL"), ("V", "V"), ("KA", "KA"), ("ALAG", "ALAG")]},
    "iv_2cmt": {"n": 2, "depot": False, "map": [("CL", "CL"), ("VC", "VC"), ("Q", "Q"), ("VP", "VP")]},
    "oral_2cmt": {"n": 2, "depot": True, "map": [("CL", "CL"), ("VC", "VC"), ("Q", "Q"), ("VP", "VP"), ("KA", "KA")]},
    "iv_3cmt": {"n": 3, "depot": False, "map": [("CL", "CL"), ("VC", "VC"), ("Q2", "Q2"),
                                                ("VP2", "VP2"), ("Q3", "Q3"), ("VP3", "VP3")]},
}


def _g(x: float) -> str:
    return f"{float(x):.6g}"


def _cov_omega_sigma(nl: dict) -> tuple[list[str], float, float, str]:
    """Shared bits: iiv list, sigmas, error model."""
    iiv = list(nl.get("iiv_params") or [])
    sig = nl.get("sigma") or {}
    return iiv, float(sig.get("prop") or 0.0), float(sig.get("add") or 0.0), \
        (nl.get("error_model") or "proportional").lower()


def _omega_export_order(nl: dict, iiv: list[str]) -> tuple[list[str], list[str]]:
    """Eta order for export, with any correlated block made CONTIGUOUS.

    Both target formats describe a correlated Omega as a square block over
    *consecutive* etas -- NONMEM's ``$OMEGA BLOCK(n)``, mrgsolve's ``@block``
    -- so a block on non-adjacent parameters (say CL and KA of [CL, V, KA])
    cannot be written in the fitted order. The exporter generates ``$PK`` /
    ``$MAIN`` itself and derives every eta index from this list, so re-ordering
    is purely internal and keeps the two consistent.

    Returns ``(eta_order, block_names)``; ``block_names`` is empty when there is
    no usable block, in which case the caller writes a plain diagonal.
    """
    blk = [p for p in (nl.get("omega_block") or []) if p in iiv]
    if len(blk) < 2 or not nl.get("omega_matrix"):
        return list(iiv), []
    return blk + [p for p in iiv if p not in blk], blk


def _omega_var(nl: dict, p: str, omega_cv: dict) -> float:
    """Variance of one eta for export.

    Prefers the fitted ``omega_matrix`` when it exists so that, on a block fit,
    the diagonal entries written outside the block come from the same source as
    the ones inside it -- rather than mixing exact covariances with variances
    re-derived from a rounded %CV.
    """
    om = nl.get("omega_matrix")
    order = list(nl.get("iiv_params") or [])
    if om and p in order:
        return float(om[order.index(p)][order.index(p)])
    return _cv_to_omega2(omega_cv.get(p, 30.0))


def _omega_lower_rows(nl: dict, blk: list[str]) -> list[list[float]]:
    """Lower-triangular rows of the fitted block covariance, in ``blk`` order.

    Indices come from ``iiv_params`` because that is the order
    ``omega_matrix`` was written in; reading it positionally against a
    re-ordered eta list would transpose the covariances onto the wrong pair.
    """
    om = nl.get("omega_matrix") or []
    order = list(nl.get("iiv_params") or [])
    idx = {p: order.index(p) for p in blk}
    return [[float(om[idx[p]][idx[q]]) for q in blk[:i + 1]]
            for i, p in enumerate(blk)]


def _nm_cov_multiplier(ce: dict, theta_idx: int) -> str | None:
    """NONMEM $PK multiplier string for a continuous covariate effect, or None
    (categorical -> handled by the caller as a manual-coding note)."""
    cov, kind = ce["covariate"], ce.get("kind", "power")
    center = ce.get("center")
    if kind == "power":
        return f"({cov}/{_g(center)})**THETA({theta_idx})"
    if kind == "linear":
        return f"(1 + THETA({theta_idx})*({cov} - {_g(center)}))"
    if kind == "exponential":
        return f"EXP(THETA({theta_idx})*({cov} - {_g(center)}))"
    return None


def build_nonmem(nl: dict) -> str | None:
    """NONMEM control stream seeded from the fitted result, or None if the
    structural model has no closed-form ADVAN mapping."""
    spec = _NONMEM.get(nl.get("model_key"))
    if not spec:
        return None
    theta = nl.get("theta") or {}
    iiv, sprop, sadd, emodel = _cov_omega_sigma(nl)
    omega_cv = nl.get("omega_cv_pct") or {}
    cov_effects = nl.get("covariate_effects") or []
    # Re-order BEFORE numbering: every eta index below derives from `iiv`, so a
    # block made contiguous here stays consistent with $PK automatically.
    iiv, blk = _omega_export_order(nl, iiv)
    eta_no = {p: k + 1 for k, p in enumerate(iiv)}

    # THETA indices: structural params, then continuous covariate coefficients.
    theta_lines: list[str] = []
    idx = 0
    theta_idx: dict[str, int] = {}
    for our, sym in spec["map"]:
        idx += 1
        theta_idx[our] = idx
        theta_lines.append(f"  (0, {_g(theta.get(our, 1.0))})   ; {idx}. TV{sym}")
    if spec.get("alag"):
        our_alag, _cmt = spec["alag"]
        idx += 1
        theta_idx[our_alag] = idx
        theta_lines.append(f"  (0, {_g(theta.get(our_alag, 0.1))})   ; {idx}. ALAG")

    cov_mult: dict[str, list[str]] = {}
    cov_notes: list[str] = []
    cov_cols: list[str] = []
    for ce in cov_effects:
        if ce["covariate"] not in cov_cols:
            cov_cols.append(ce["covariate"])
        if ce.get("kind") == "categorical":
            cov_notes.append(f"; NOTE: covariate {ce['covariate']} on {ce['param']} is "
                             f"categorical — add IF(...) THETA blocks manually.")
            continue
        idx += 1
        mult = _nm_cov_multiplier(ce, idx)
        coef = ce.get("coefficient", 0.0)
        theta_lines.append(f"  {_g(coef)}   ; {idx}. {ce['param']}~{ce['covariate']} ({ce['kind']})")
        cov_mult.setdefault(ce["param"], []).append(mult)

    # $PK block
    pk: list[str] = list(cov_notes)
    for our, sym in spec["map"]:
        pk.append(f"  TV{sym} = THETA({theta_idx[our]})")
        for mult in cov_mult.get(our, []):
            pk.append(f"  TV{sym} = TV{sym} * {mult}")
        if our in eta_no:
            pk.append(f"  {sym} = TV{sym} * EXP(ETA({eta_no[our]}))")
        else:
            pk.append(f"  {sym} = TV{sym}")
    if spec.get("alag"):
        our_alag, cmt = spec["alag"]
        pk.append(f"  ALAG{cmt} = THETA({theta_idx[our_alag]})")
    pk.append(f"  S{spec['s_cmt']} = {spec['s_sym']}")

    # $ERROR + $SIGMA
    if emodel == "proportional":
        err = ["  IPRED = F", "  Y = IPRED*(1 + EPS(1))"]
        sigma = [f"  {_g(sprop ** 2)}   ; proportional variance"]
    elif emodel == "additive":
        err = ["  IPRED = F", "  Y = IPRED + EPS(1)"]
        sigma = [f"  {_g(sadd ** 2)}   ; additive variance"]
    else:  # combined
        err = ["  IPRED = F", "  Y = IPRED*(1 + EPS(1)) + EPS(2)"]
        sigma = [f"  {_g(sprop ** 2)}   ; proportional variance",
                 f"  {_g(sadd ** 2)}   ; additive variance"]

    if blk:
        omega = [f"$OMEGA BLOCK({len(blk)})"]
        for i, row in enumerate(_omega_lower_rows(nl, blk)):
            note = (f"; IIV {blk[i]} (variance)" if i == 0
                    else "; cov(" + ", ".join(blk[:i]) + f") , IIV {blk[i]}")
            omega.append("  " + " ".join(_g(v) for v in row) + f"   {note}")
        rest = [p for p in iiv if p not in blk]
        if rest:
            omega.append("$OMEGA")
            omega += [f"  {_g(_omega_var(nl, p, omega_cv))}"
                      f"   ; IIV {p} (variance)" for p in rest]
    else:
        omega = [f"  {_g(_cv_to_omega2(omega_cv.get(p, 30.0)))}"
                 f"   ; IIV {p} (variance)" for p in iiv]

    inp = "ID TIME DV AMT EVID MDV" + ("".join(f" {c}" for c in cov_cols))
    label = nl.get("label", nl.get("model_key", "model"))
    out = [
        f"$PROBLEM PharmAgent export — {label}",
        f"$INPUT {inp}",
        "$DATA data.csv IGNORE=@",
        f"$SUBROUTINE ADVAN{spec['advan']} TRANS{spec['trans']}",
        "$PK", *pk,
        "$ERROR", *err,
        "$THETA", *theta_lines,
        # The block form emits its own $OMEGA BLOCK(n) / $OMEGA headers.
        *(omega if blk else ["$OMEGA", *(omega or ["  0.09   ; (no IIV estimated)"])]),
        "$SIGMA", *sigma,
        "$ESTIMATION METHOD=1 INTERACTION MAXEVAL=9999 PRINT=5",
        "$COVARIANCE",
    ]
    return "\n".join(out) + "\n"


def build_mrgsolve(nl: dict) -> str | None:
    """mrgsolve model (explicit $ODE) seeded from the fitted result, or None for
    structural models without a closed-form compartmental mapping."""
    spec = _MRG.get(nl.get("model_key"))
    if not spec:
        return None
    theta = nl.get("theta") or {}
    iiv, sprop, sadd, emodel = _cov_omega_sigma(nl)
    omega_cv = nl.get("omega_cv_pct") or {}
    iiv, blk = _omega_export_order(nl, iiv)   # block first; see NONMEM note
    cov_effects = nl.get("covariate_effects") or []
    n, depot = spec["n"], spec.get("depot", False)

    # $PARAM: TV<param> + covariate columns (with neutral defaults).
    param_kv = [f"TV{our}={_g(theta.get(our, 1.0))}" for our, _ in spec["map"]]
    cov_cols: list[str] = []
    for ce in cov_effects:
        if ce["covariate"] not in cov_cols and ce.get("kind") != "categorical":
            cov_cols.append(ce["covariate"])
    param_kv += [f"{c}={_g(_center_default(cov_effects, c))}" for c in cov_cols]

    eta_lab = [f"E{p}" for p in iiv]

    # $MAIN: realize individual parameters (+ covariate effects + IIV).
    main: list[str] = []
    cov_by_param: dict[str, list[str]] = {}
    for ce in cov_effects:
        if ce.get("kind") == "categorical":
            continue
        cov_by_param.setdefault(ce["param"], []).append(_mrg_cov_multiplier(ce))
    for our, _ in spec["map"]:
        expr = f"TV{our}"
        for mult in cov_by_param.get(our, []):
            expr += f" * {mult}"
        if our in iiv:
            expr += f" * exp(E{our})"
        main.append(f"  double {our} = {expr};")

    # $ODE: linear disposition (+ first-order depot).
    ode: list[str] = []
    if depot:
        ode.append("  dxdt_DEPOT = -KA*DEPOT;")
    cl_v = "(CL/V)" if "V" in dict(spec["map"]) else "(CL/VC)"
    central_in = "KA*DEPOT" if depot else "0"
    if n == 1:
        ode.append(f"  dxdt_CENT = {central_in} - {cl_v}*CENT;")
        conc_v = "V"
    elif n == 2:
        ode.append(f"  dxdt_CENT = {central_in} - (CL/VC)*CENT - (Q/VC)*CENT + (Q/VP)*PERIPH;")
        ode.append("  dxdt_PERIPH = (Q/VC)*CENT - (Q/VP)*PERIPH;")
        conc_v = "VC"
    else:  # n == 3
        ode.append(f"  dxdt_CENT = {central_in} - (CL/VC)*CENT - (Q2/VC)*CENT + (Q2/VP2)*PERIPH2"
                   " - (Q3/VC)*CENT + (Q3/VP3)*PERIPH3;")
        ode.append("  dxdt_PERIPH2 = (Q2/VC)*CENT - (Q2/VP2)*PERIPH2;")
        ode.append("  dxdt_PERIPH3 = (Q3/VC)*CENT - (Q3/VP3)*PERIPH3;")
        conc_v = "VC"

    cmts = (["DEPOT"] if depot else []) + ["CENT"] + \
           (["PERIPH"] if n == 2 else ["PERIPH2", "PERIPH3"] if n == 3 else [])
    if spec.get("lag"):
        # Lag belongs on the DOSING compartment. The only lag model is oral
        # (oral_1cmt_lag), where the dose enters DEPOT, so set ALAG_DEPOT;
        # ALAG_CENT would attach the lag to a compartment that receives no dose.
        lag_cmt = "DEPOT" if depot else "CENT"
        main.append(f"  ALAG_{lag_cmt} = ALAG;  // absorption lag on the dosing compartment")

    if emodel == "proportional":
        err = ["  double IPRED = CENT/" + conc_v + ";",
               "  double Y = IPRED*(1 + EPS(1));"]
        sigma = f"$SIGMA {_g(sprop ** 2)}  // proportional"
    elif emodel == "additive":
        err = ["  double IPRED = CENT/" + conc_v + ";",
               "  double Y = IPRED + EPS(1);"]
        sigma = f"$SIGMA {_g(sadd ** 2)}  // additive"
    else:
        err = ["  double IPRED = CENT/" + conc_v + ";",
               "  double Y = IPRED*(1 + EPS(1)) + EPS(2);"]
        sigma = f"$SIGMA {_g(sprop ** 2)} {_g(sadd ** 2)}  // prop, add"

    if blk:
        # mrgsolve @block takes the lower triangle, row-major.
        tri = " ".join(_g(v) for row in _omega_lower_rows(nl, blk) for v in row)
        omega_lines = ["$OMEGA @block @labels " + " ".join(f"E{p}" for p in blk),
                       tri]
        rest = [p for p in iiv if p not in blk]
        if rest:
            omega_lines += [
                "$OMEGA @labels " + " ".join(f"E{p}" for p in rest),
                " ".join(_g(_omega_var(nl, p, omega_cv)) for p in rest)]
        omega_block_str = "\n".join(omega_lines)
    else:
        omega_vals = " ".join(
            _g(_cv_to_omega2(omega_cv.get(p, 30.0))) for p in iiv)
        omega_block_str = ("$OMEGA @labels " + " ".join(eta_lab) + "\n"
                           + (omega_vals or "0.09"))
    label = nl.get("label", nl.get("model_key", "model"))
    out = [
        f"// PharmAgent export — {label}",
        "$PARAM " + ", ".join(param_kv),
        "$CMT " + " ".join(cmts),
        omega_block_str,
        sigma,
        "$MAIN", *main,
        "$ODE", *ode,
        "$TABLE", *err, "$CAPTURE IPRED Y",
    ]
    return "\n".join(out) + "\n"


def _center_default(cov_effects: list[dict], cov: str) -> float:
    for ce in cov_effects:
        if ce["covariate"] == cov and ce.get("center") is not None:
            return float(ce["center"])
    return 1.0


def _mrg_cov_multiplier(ce: dict) -> str:
    cov, kind, center = ce["covariate"], ce.get("kind", "power"), ce.get("center")
    coef = _g(ce.get("coefficient", 0.0))
    if kind == "power":
        return f"pow({cov}/{_g(center)}, {coef})"
    if kind == "linear":
        return f"(1 + {coef}*({cov} - {_g(center)}))"
    return f"exp({coef}*({cov} - {_g(center)}))"
