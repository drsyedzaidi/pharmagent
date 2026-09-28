"""Clinical-pharmacology calculators (PharmKit's toolkit inside PharmAgent).

Each tool is a deterministic closed form from ``app.compute.clinpharm``; none
needs a dataset. Inputs are validated at the boundary (a missing or non-numeric
required input is a clear ValueError naming the field, never a crash), every
result records the formula used, and the last result plus a bounded history
are written to state so the chat/UI can show them and the report can cite
them. Cheap and side-effect free, so every tool is chat-reachable.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from app.compute import clinpharm as cp
from app.core.pharmstate import PharmState
from app.tools.base import Tool, ToolContext, ToolResult

AGENT = "clinpharm"
HISTORY_MAX = 20


def _num(args: dict[str, Any], name: str, *, required: bool = True,
         default: float | None = None, positive: bool = False) -> float | None:
    v = args.get(name)
    if v is None or v == "":
        if required:
            raise ValueError(f"missing required input: {name}")
        return default
    try:
        x = float(v)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must be a number, got {v!r}") from e
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite")
    if positive and x <= 0:
        raise ValueError(f"{name} must be > 0")
    return x


def _sex(args: dict[str, Any]) -> str:
    s = str(args.get("sex") or "").strip().lower()
    if s in ("m", "male"):
        return "male"
    if s in ("f", "female"):
        return "female"
    raise ValueError("sex must be 'male' or 'female'")


def _r(x: Any, d: int = 4) -> Any:
    return round(x, d) if isinstance(x, float) else x


def _emit(state: PharmState, tool: str, label: str, inputs: dict[str, Any],
          outputs: dict[str, Any], formula: str, note: str | None = None,
          summary: str = "") -> ToolResult:
    rec = {"status": "ok", "tool": tool, "label": label,
           "inputs": {k: _r(v) for k, v in inputs.items()},
           "outputs": {k: _r(v) for k, v in outputs.items()},
           "formula": formula, "note": note}
    history = [*(state.clinpharm_history or []), rec][-HISTORY_MAX:]
    return ToolResult(summary=summary, action=f"{tool}",
                      writes={"clinpharm_results": rec, "clinpharm_history": history},
                      result=rec)


# ── tools ───────────────────────────────────────────────────────────────────
def calc_half_life(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """t½ ↔ ke, or ke from two terminal points; always reports both."""
    if args.get("c1") is not None or args.get("c2") is not None:
        c1, t1 = _num(args, "c1", positive=True), _num(args, "t1")
        c2, t2 = _num(args, "c2", positive=True), _num(args, "t2")
        ke = cp.ke_from_two_points(c1, t1, c2, t2)
        inputs, formula = {"c1": c1, "t1": t1, "c2": c2, "t2": t2}, "ke = (ln C1 − ln C2)/(t2 − t1); t½ = ln2/ke"
    elif args.get("half_life") is not None:
        hl = _num(args, "half_life", positive=True)
        ke, inputs, formula = cp.ke_from_half_life(hl), {"half_life": hl}, "ke = ln2 / t½"
    elif args.get("ke") is not None:
        ke = _num(args, "ke", positive=True)
        inputs, formula = {"ke": ke}, "t½ = ln2 / ke"
    else:
        raise ValueError("give half_life, or ke, or two points (c1,t1,c2,t2)")
    hl = cp.half_life_from_ke(ke)
    out = {"ke": ke, "half_life": hl, "time_to_90pct_ss": cp.time_to_fraction_of_steady_state(ke, 0.9),
           "time_to_97pct_ss": cp.time_to_fraction_of_steady_state(ke, 0.97)}
    return _emit(state, "calc_half_life", "Half-life & rate constant", inputs, out, formula,
                 summary=(f"ke = {ke:.4g} per time unit, t½ = {hl:.4g}; "
                          f"90% of steady state after {out['time_to_90pct_ss']:.3g}."))


def calc_accumulation(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Rac and time/doses to a fraction of steady state for dosing every tau."""
    tau = _num(args, "tau", positive=True)
    if args.get("ke") is not None:
        ke = _num(args, "ke", positive=True)
    else:
        ke = cp.ke_from_half_life(_num(args, "half_life", positive=True))
    frac = _num(args, "fraction", required=False, default=0.9)
    rac = cp.accumulation_ratio(ke, tau)
    t_frac = cp.time_to_fraction_of_steady_state(ke, frac)
    out = {"accumulation_ratio": rac, "time_to_fraction_ss": t_frac,
           "doses_to_fraction_ss": math.ceil(t_frac / tau), "fraction": frac,
           "half_life": cp.half_life_from_ke(ke)}
    return _emit(state, "calc_accumulation", "Accumulation & steady state",
                 {"ke": ke, "tau": tau, "fraction": frac}, out,
                 "Rac = 1/(1 − e^(−ke·τ)); t_f = −ln(1 − f)/ke",
                 summary=(f"Rac = {rac:.3g} for τ = {tau:g}; {frac:.0%} of steady state after "
                          f"{t_frac:.3g} ({out['doses_to_fraction_ss']} doses)."))


def calc_dose_regimen(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Loading dose (C_target·V/F) and maintenance dose (C_avg,ss·CL·τ/F)."""
    f = _num(args, "bioavailability", required=False, default=1.0, positive=True)
    out: dict[str, Any] = {}
    inputs: dict[str, Any] = {"bioavailability": f}
    if args.get("target_conc") is not None or args.get("volume") is not None:
        ct, v = _num(args, "target_conc", positive=True), _num(args, "volume", positive=True)
        out["loading_dose"] = cp.loading_dose(ct, v, f)
        inputs.update(target_conc=ct, volume=v)
    if args.get("cavg_ss") is not None or args.get("clearance") is not None:
        cavg, cl = _num(args, "cavg_ss", positive=True), _num(args, "clearance", positive=True)
        tau = _num(args, "tau", positive=True)
        out["maintenance_dose"] = cp.maintenance_dose(cavg, cl, tau, f)
        out["dose_rate"] = out["maintenance_dose"] / tau
        inputs.update(cavg_ss=cavg, clearance=cl, tau=tau)
    if not out:
        raise ValueError("give target_conc + volume (loading) and/or cavg_ss + clearance + tau (maintenance)")
    parts = []
    if "loading_dose" in out:
        parts.append(f"loading dose {out['loading_dose']:.4g}")
    if "maintenance_dose" in out:
        parts.append(f"maintenance dose {out['maintenance_dose']:.4g} per τ")
    return _emit(state, "calc_dose_regimen", "Loading & maintenance dose", inputs, out,
                 "LD = C_target·V/F; MD = C_avg,ss·CL·τ/F", summary="; ".join(parts) + ".")


def calc_renal_function(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Cockcroft-Gault CrCl, CKD-EPI 2021 eGFR, optional clearance-proportional dose adjustment."""
    age = _num(args, "age", positive=True)
    scr = _num(args, "scr_mg_dl", positive=True)
    sex = _sex(args)
    wt = _num(args, "weight_kg", required=False, positive=True)
    out: dict[str, Any] = {"egfr_ckd_epi_2021": cp.ckd_epi_2021(age, scr, sex)}
    inputs: dict[str, Any] = {"age": age, "scr_mg_dl": scr, "sex": sex}
    if wt is not None:
        out["crcl_cockcroft_gault"] = cp.cockcroft_gault(age, wt, scr, sex)
        inputs["weight_kg"] = wt
    if args.get("normal_dose") is not None:
        dose = _num(args, "normal_dose", positive=True)
        ref = _num(args, "reference_crcl", required=False, default=120.0, positive=True)
        fe = _num(args, "fraction_renal", required=False, default=1.0)
        crcl = out.get("crcl_cockcroft_gault", out["egfr_ckd_epi_2021"])
        out["adjusted_dose"] = cp.renal_dose_adjustment(dose, crcl, ref, fe)
        out["adjustment_basis"] = "Cockcroft-Gault CrCl" if wt is not None else "CKD-EPI eGFR"
        inputs.update(normal_dose=dose, reference_crcl=ref, fraction_renal=fe)
    txt = f"eGFR (CKD-EPI 2021) {out['egfr_ckd_epi_2021']:.3g} mL/min/1.73m²"
    if "crcl_cockcroft_gault" in out:
        txt += f"; CrCl (Cockcroft-Gault) {out['crcl_cockcroft_gault']:.3g} mL/min"
    if "adjusted_dose" in out:
        txt += f"; adjusted dose {out['adjusted_dose']:.4g}"
    return _emit(state, "calc_renal_function", "Renal function & dose", inputs, out,
                 "CG: (140−age)·wt/(72·SCr)·(0.85 F); CKD-EPI 2021 race-free; dose·(1 − fe·(1 − min(CrCl/ref,1)))",
                 note="Estimates only — not a substitute for clinical judgement or product labelling.",
                 summary=txt + ".")


def calc_allometric(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Y2 = Y1·(BW2/BW1)^exponent; default 0.75 (CL) or 1.0 (V) by parameter kind."""
    value = _num(args, "value")
    from_bw, to_bw = _num(args, "from_bw", positive=True), _num(args, "to_bw", positive=True)
    kind = str(args.get("kind") or "CL").upper()
    exp_default = 1.0 if kind.startswith("V") else 0.75
    exponent = _num(args, "exponent", required=False, default=exp_default)
    scaled = cp.allometric_scale(value, from_bw, to_bw, exponent)
    return _emit(state, "calc_allometric", "Allometric scaling",
                 {"value": value, "from_bw": from_bw, "to_bw": to_bw, "exponent": exponent, "kind": kind},
                 {"scaled_value": scaled, "ratio": scaled / value if value else None},
                 "Y2 = Y1·(BW2/BW1)^exponent",
                 summary=(f"{kind} {value:.4g} at {from_bw:g} kg → {scaled:.4g} at {to_bw:g} kg "
                          f"(exponent {exponent:g})."))


def convert_concentration(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """mg/L ↔ µmol/L via molar mass."""
    mw = _num(args, "molar_mass", positive=True)
    if args.get("mg_per_l") is not None:
        x = _num(args, "mg_per_l")
        out, inputs = {"umol_per_l": cp.mass_to_molar(x, mw)}, {"mg_per_l": x, "molar_mass": mw}
        s = f"{x:g} mg/L = {out['umol_per_l']:.4g} µmol/L (MW {mw:g})."
    elif args.get("umol_per_l") is not None:
        x = _num(args, "umol_per_l")
        out, inputs = {"mg_per_l": cp.molar_to_mass(x, mw)}, {"umol_per_l": x, "molar_mass": mw}
        s = f"{x:g} µmol/L = {out['mg_per_l']:.4g} mg/L (MW {mw:g})."
    else:
        raise ValueError("give mg_per_l or umol_per_l together with molar_mass")
    return _emit(state, "convert_concentration", "Concentration converter", inputs, out,
                 "µM = mg/L / MW · 1000", summary=s)


def calc_be_sample_size(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """2×2 crossover average-BE (TOST) sample size, normal approximation."""
    cv = _num(args, "cv_intra", positive=True)
    if cv > 1.5:
        cv = cv / 100.0      # a percentage was given
    gmr = _num(args, "gmr", required=False, default=0.95, positive=True)
    power = _num(args, "power", required=False, default=0.8)
    alpha = _num(args, "alpha", required=False, default=0.05)
    lower = _num(args, "lower", required=False, default=0.8)
    upper = _num(args, "upper", required=False, default=1.25)
    r = cp.be_sample_size(cv, gmr=gmr, power=power, alpha=alpha, lower=lower, upper=upper)
    return _emit(state, "calc_be_sample_size", "BE sample size (2×2 crossover)",
                 {"cv_intra": cv, "gmr": gmr, "power": power, "alpha": alpha, "lower": lower, "upper": upper},
                 {"total": r["total"], "per_sequence": r["per_sequence"], "sigma_w": r["sigma_w"]},
                 "TOST normal approximation (Chow & Liu)", note=r["note"],
                 summary=(f"N = {r['total']} subjects ({r['per_sequence']} per sequence) for "
                          f"CV {cv:.0%}, GMR {gmr:g}, power {power:.0%}."))


def quick_one_compartment(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Analytic one-compartment profile (IV bolus / infusion / oral) — no dataset needed."""
    route = str(args.get("route") or "oral").lower()
    dose, v = _num(args, "dose", positive=True), _num(args, "volume", positive=True)
    if args.get("ke") is not None:
        ke = _num(args, "ke", positive=True)
    elif args.get("half_life") is not None:
        ke = cp.ke_from_half_life(_num(args, "half_life", positive=True))
    else:
        ke = _num(args, "clearance", positive=True) / v
    t_end = _num(args, "t_end", required=False, default=None, positive=True) or 5 * cp.half_life_from_ke(ke)
    inputs: dict[str, Any] = {"route": route, "dose": dose, "volume": v, "ke": ke, "t_end": t_end}
    fn: Callable[[float], float]
    if route in ("iv", "bolus", "iv_bolus"):
        fn = lambda t: cp.iv_bolus_conc(dose, v, ke, t)  # noqa: E731
        out: dict[str, Any] = {"c0": dose / v, "auc_inf": dose / (v * ke)}
    elif route in ("infusion", "iv_infusion"):
        t_inf = _num(args, "t_inf", positive=True)
        inputs["t_inf"] = t_inf
        fn = lambda t: cp.iv_infusion_conc(dose, t_inf, v, ke, t)  # noqa: E731
        out = {"c_end_of_infusion": fn(t_inf), "auc_inf": dose / (v * ke)}
    else:
        ka = _num(args, "ka", positive=True)
        f = _num(args, "bioavailability", required=False, default=1.0, positive=True)
        inputs.update(ka=ka, bioavailability=f)
        fn = lambda t: cp.oral_one_comp_conc(dose, v, ka, ke, t, f)  # noqa: E731
        tmax = cp.oral_tmax(ka, ke)
        out = {"tmax": tmax, "cmax": fn(tmax), "auc_inf": f * dose / (v * ke)}
    out["half_life"] = cp.half_life_from_ke(ke)
    out["profile"] = [{"t": _r(p["t"]), "c": _r(p["c"], 6)} for p in cp.sample_profile(fn, t_end, 120)]
    kind = ("iv" if route in ("iv", "bolus", "iv_bolus")
            else "infusion" if route in ("infusion", "iv_infusion") else "oral")
    formulas = {"oral": "C = F·D·ka/(V(ka−ke))·(e^(−ke t) − e^(−ka t))",
                "iv": "C = D/V·e^(−ke t)", "infusion": "R0/(V ke)·(1 − e^(−ke t)), then decay"}
    return _emit(state, "quick_one_compartment", "One-compartment simulator", inputs,
                 out, formulas[kind],
                 summary=(f"{route} 1-cmt: t½ {out['half_life']:.3g}, AUC∞ {out['auc_inf']:.4g}"
                          + (f", Cmax {out['cmax']:.4g} at tmax {out['tmax']:.3g}" if "cmax" in out else "") + "."))


_NUM = {"type": "number"}
TOOLS = [
    Tool("calc_half_life",
         "Half-life ↔ elimination rate constant, or ke from two terminal-phase points "
         "(c1 at t1, c2 at t2); also time to 90%/97% of steady state.",
         AGENT, {"type": "object", "properties": {"half_life": _NUM, "ke": _NUM, "c1": _NUM, "t1": _NUM,
                                                   "c2": _NUM, "t2": _NUM}, "required": []},
         calc_half_life),
    Tool("calc_accumulation",
         "Accumulation ratio at steady state and time/doses to reach a fraction of steady "
         "state for a dosing interval tau, from ke or half-life.",
         AGENT, {"type": "object", "properties": {"tau": _NUM, "ke": _NUM, "half_life": _NUM,
                                                   "fraction": {"type": "number", "description": "default 0.9"}},
                 "required": ["tau"]},
         calc_accumulation),
    Tool("calc_dose_regimen",
         "Loading dose from target concentration and volume (LD = C·V/F) and/or maintenance "
         "dose from target average steady-state concentration, clearance and tau (MD = Css·CL·τ/F).",
         AGENT, {"type": "object", "properties": {"target_conc": _NUM, "volume": _NUM, "cavg_ss": _NUM,
                                                   "clearance": _NUM, "tau": _NUM, "bioavailability": _NUM},
                 "required": []},
         calc_dose_regimen),
    Tool("calc_renal_function",
         "Cockcroft-Gault creatinine clearance (needs weight) and CKD-EPI 2021 eGFR from age, "
         "serum creatinine (mg/dL) and sex; optional clearance-proportional dose adjustment "
         "(normal_dose, reference_crcl default 120, fraction_renal default 1).",
         AGENT, {"type": "object", "properties": {"age": _NUM, "scr_mg_dl": _NUM,
                                                   "sex": {"type": "string", "enum": ["male", "female"]},
                                                   "weight_kg": _NUM, "normal_dose": _NUM,
                                                   "reference_crcl": _NUM, "fraction_renal": _NUM},
                 "required": ["age", "scr_mg_dl", "sex"]},
         calc_renal_function),
    Tool("calc_allometric",
         "Allometric scaling of a parameter across body weight: Y2 = Y1·(BW2/BW1)^exponent "
         "(exponent defaults 0.75 for clearance, 1.0 for volume).",
         AGENT, {"type": "object", "properties": {"value": _NUM, "from_bw": _NUM, "to_bw": _NUM,
                                                   "exponent": _NUM,
                                                   "kind": {"type": "string", "enum": ["CL", "V"]}},
                 "required": ["value", "from_bw", "to_bw"]},
         calc_allometric),
    Tool("convert_concentration",
         "Convert a concentration between mg/L and µmol/L given the molar mass (g/mol).",
         AGENT, {"type": "object", "properties": {"mg_per_l": _NUM, "umol_per_l": _NUM, "molar_mass": _NUM},
                 "required": ["molar_mass"]},
         convert_concentration),
    Tool("calc_be_sample_size",
         "Sample size for a 2×2 crossover average-bioequivalence study (TOST, normal "
         "approximation) from intra-subject CV (fraction or %), expected GMR, power, alpha "
         "and the acceptance bounds.",
         AGENT, {"type": "object", "properties": {"cv_intra": _NUM, "gmr": _NUM, "power": _NUM, "alpha": _NUM,
                                                   "lower": _NUM, "upper": _NUM},
                 "required": ["cv_intra"]},
         calc_be_sample_size),
    Tool("quick_one_compartment",
         "Analytic one-compartment concentration-time profile without a dataset: route "
         "oral (needs ka, optional bioavailability), iv (bolus) or infusion (needs t_inf); "
         "dose, volume and ke / half_life / clearance.",
         AGENT, {"type": "object", "properties": {"route": {"type": "string", "enum": ["oral", "iv", "infusion"]},
                                                   "dose": _NUM, "volume": _NUM, "ke": _NUM, "half_life": _NUM,
                                                   "clearance": _NUM, "ka": _NUM, "bioavailability": _NUM,
                                                   "t_inf": _NUM, "t_end": _NUM},
                 "required": ["dose", "volume"]},
         quick_one_compartment),
]
