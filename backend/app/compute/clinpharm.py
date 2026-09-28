"""Clinical-pharmacology closed forms (ported from PharmKit's ``pk.ts``).

Pure, unit-agnostic functions: pass a consistent unit system (dose mg,
concentration mg/L, volume L, time h → CL in L/h, ...). Each routine names the
formula it implements so a result can be checked against a reference. NCA and
full simulation are NOT here — PharmAgent already has ``compute.nca`` and the
model library; these are the quick single-formula calculators.
"""
from __future__ import annotations

import math
from typing import Literal

from scipy.stats import norm

LN2 = math.log(2.0)
Sex = Literal["male", "female"]


# ── elementary single-/repeat-dose relationships ────────────────────────────
def ke_from_half_life(half_life: float) -> float:
    """ke = ln2 / t½."""
    if half_life <= 0:
        raise ValueError("half-life must be > 0")
    return LN2 / half_life


def half_life_from_ke(ke: float) -> float:
    """t½ = ln2 / ke."""
    if ke <= 0:
        raise ValueError("ke must be > 0")
    return LN2 / ke


def ke_from_two_points(c1: float, t1: float, c2: float, t2: float) -> float:
    """Terminal-phase ke from two timed concentrations: (ln C1 − ln C2)/(t2 − t1)."""
    if t2 <= t1:
        raise ValueError("t2 must be greater than t1")
    if c1 <= 0 or c2 <= 0:
        raise ValueError("concentrations must be > 0")
    return (math.log(c1) - math.log(c2)) / (t2 - t1)


def accumulation_ratio(ke: float, tau: float) -> float:
    """Rac = 1 / (1 − e^(−ke·τ))."""
    if ke <= 0 or tau <= 0:
        raise ValueError("ke and tau must be > 0")
    return 1.0 / (1.0 - math.exp(-ke * tau))


def time_to_fraction_of_steady_state(ke: float, frac: float) -> float:
    """t = −ln(1 − f) / ke for a fraction f of steady state."""
    if not 0 < frac < 1:
        raise ValueError("fraction must be in (0, 1)")
    if ke <= 0:
        raise ValueError("ke must be > 0")
    return -math.log(1.0 - frac) / ke


def volume_from_dose_c0(dose: float, c0: float) -> float:
    """V = D / C0 (IV bolus, back-extrapolated C0)."""
    if c0 <= 0:
        raise ValueError("C0 must be > 0")
    return dose / c0


def clearance_from_auc(dose: float, auc_inf: float, bioavailability: float = 1.0) -> float:
    """CL = F·D / AUC∞."""
    if auc_inf <= 0:
        raise ValueError("AUC must be > 0")
    return bioavailability * dose / auc_inf


def loading_dose(target_conc: float, volume: float, bioavailability: float = 1.0) -> float:
    """LD = C_target · V / F."""
    if bioavailability <= 0:
        raise ValueError("bioavailability must be > 0")
    return target_conc * volume / bioavailability


def maintenance_dose(cavg_ss: float, clearance: float, tau: float,
                     bioavailability: float = 1.0) -> float:
    """MD = C_avg,ss · CL · τ / F."""
    if bioavailability <= 0:
        raise ValueError("bioavailability must be > 0")
    return cavg_ss * clearance * tau / bioavailability


# ── one-compartment analytic profiles ───────────────────────────────────────
def iv_bolus_conc(dose: float, volume: float, ke: float, t: float) -> float:
    """C(t) = (D/V)·e^(−ke·t)."""
    return dose / volume * math.exp(-ke * t)


def iv_infusion_conc(dose: float, t_inf: float, volume: float, ke: float, t: float) -> float:
    """Constant-rate infusion of length t_inf (R0 = D/t_inf): rise to R0/(V·ke),
    then mono-exponential decline from C_end."""
    r0 = dose / t_inf
    plateau = r0 / (volume * ke)
    if t <= t_inf:
        return plateau * (1.0 - math.exp(-ke * t))
    c_end = plateau * (1.0 - math.exp(-ke * t_inf))
    return c_end * math.exp(-ke * (t - t_inf))


def oral_one_comp_conc(dose: float, volume: float, ka: float, ke: float, t: float,
                       bioavailability: float = 1.0) -> float:
    """C(t) = F·D·ka / (V·(ka − ke)) · (e^(−ke·t) − e^(−ka·t)); ka→ke limit handled."""
    fd = bioavailability * dose
    if abs(ka - ke) < 1e-9:
        return fd / volume * ka * t * math.exp(-ke * t)
    return fd * ka / (volume * (ka - ke)) * (math.exp(-ke * t) - math.exp(-ka * t))


def oral_tmax(ka: float, ke: float) -> float:
    """tmax = ln(ka/ke) / (ka − ke)."""
    if abs(ka - ke) < 1e-9:
        return 1.0 / ke
    return math.log(ka / ke) / (ka - ke)


def sample_profile(fn, t_end: float, n: int = 120) -> list[dict[str, float]]:
    """n+1 evenly spaced (t, c) points on [0, t_end]."""
    return [{"t": (t := t_end * i / n), "c": fn(t)} for i in range(n + 1)]


# ── renal function ──────────────────────────────────────────────────────────
def cockcroft_gault(age_years: float, weight_kg: float, scr_mg_dl: float, sex: Sex) -> float:
    """CrCl (mL/min) = (140 − age)·weight / (72·SCr) · (0.85 if female)."""
    if scr_mg_dl <= 0:
        raise ValueError("serum creatinine must be > 0")
    base = (140.0 - age_years) * weight_kg / (72.0 * scr_mg_dl)
    return base * 0.85 if sex == "female" else base


def ckd_epi_2021(age_years: float, scr_mg_dl: float, sex: Sex) -> float:
    """CKD-EPI 2021 creatinine (race-free), eGFR mL/min/1.73 m²:
    142·min(SCr/κ,1)^α·max(SCr/κ,1)^−1.200·0.9938^age·(1.012 if female);
    κ = 0.7 F / 0.9 M, α = −0.241 F / −0.302 M."""
    if scr_mg_dl <= 0:
        raise ValueError("serum creatinine must be > 0")
    kappa = 0.7 if sex == "female" else 0.9
    alpha = -0.241 if sex == "female" else -0.302
    ratio = scr_mg_dl / kappa
    return (142.0 * min(ratio, 1.0) ** alpha * max(ratio, 1.0) ** -1.2
            * 0.9938 ** age_years * (1.012 if sex == "female" else 1.0))


def renal_dose_adjustment(normal_dose: float, patient_crcl: float, reference_crcl: float = 120.0,
                          fraction_renal: float = 1.0) -> float:
    """Clearance-proportional adjustment: dose · (1 − fe·(1 − min(CrCl/ref, 1)))."""
    if reference_crcl <= 0:
        raise ValueError("reference CrCl must be > 0")
    if not 0 <= fraction_renal <= 1:
        raise ValueError("fraction renal must be in [0, 1]")
    ratio = min(patient_crcl / reference_crcl, 1.0)
    return normal_dose * (1.0 - fraction_renal * (1.0 - ratio))


# ── unit conversion / allometry ─────────────────────────────────────────────
def mass_to_molar(mg_per_l: float, molar_mass_g_mol: float) -> float:
    """µmol/L = mg/L / MW · 1000."""
    if molar_mass_g_mol <= 0:
        raise ValueError("molar mass must be > 0")
    return mg_per_l / molar_mass_g_mol * 1000.0


def molar_to_mass(umol_per_l: float, molar_mass_g_mol: float) -> float:
    """mg/L = µmol/L · MW / 1000."""
    if molar_mass_g_mol <= 0:
        raise ValueError("molar mass must be > 0")
    return umol_per_l * molar_mass_g_mol / 1000.0


def allometric_scale(value: float, from_bw: float, to_bw: float, exponent: float) -> float:
    """Y2 = Y1 · (BW2/BW1)^exponent (CL ≈ 0.75, V ≈ 1.0)."""
    if from_bw <= 0 or to_bw <= 0:
        raise ValueError("body weights must be > 0")
    return value * (to_bw / from_bw) ** exponent


# ── bioequivalence sample size (2×2 crossover, average BE, TOST) ────────────
def be_sample_size(cv_intra: float, gmr: float = 0.95, power: float = 0.8, alpha: float = 0.05,
                   lower: float = 0.8, upper: float = 1.25) -> dict:
    """Normal-approximation sample size (Chow & Liu / Julious) for a 2×2
    crossover TOST. ``total`` is rounded up to an even number. The exact
    noncentral-t / simulation design typically lands within a couple of subjects."""
    if cv_intra <= 0:
        raise ValueError("intra-subject CV must be > 0")
    if not 0 < power < 1 or not 0 < alpha < 1:
        raise ValueError("power and alpha must be in (0, 1)")
    if not lower < gmr < upper:
        raise ValueError("GMR must lie inside the acceptance bounds")
    sigma_w = math.sqrt(math.log(1.0 + cv_intra * cv_intra))
    delta = math.log(gmr)
    theta1, theta2 = math.log(lower), math.log(upper)
    z_alpha = float(norm.ppf(1.0 - alpha))
    if abs(delta) < 1e-6:
        z_beta = float(norm.ppf(1.0 - (1.0 - power) / 2.0))
        n_per_seq = (z_alpha + z_beta) ** 2 * sigma_w ** 2 / theta2 ** 2
    else:
        margin = theta2 - delta if delta > 0 else delta - theta1
        z_beta = float(norm.ppf(power))
        n_per_seq = (z_alpha + z_beta) ** 2 * sigma_w ** 2 / margin ** 2
    per_sequence = max(2, math.ceil(n_per_seq))
    total = per_sequence * 2
    return {
        "per_sequence": per_sequence,
        "total": total if total % 2 == 0 else total + 1,
        "sigma_w": sigma_w,
        "note": ("Normal approximation (Chow & Liu). Confirm with an exact noncentral-t / "
                 "simulation design for a regulatory submission."),
    }
