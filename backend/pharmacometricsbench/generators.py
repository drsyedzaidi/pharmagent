"""Seeded task generators. Ground truth is whatever the validated compute tool
returns on the generated data — so a tool-calling agent scores 1.0 by
construction, and the task set is fully reproducible from a seed.
"""
from __future__ import annotations

import numpy as np

from app.compute.bioequivalence import be_one_parameter
from app.compute.compartmental import conc_1cmt_oral, fit_one_subject
from app.compute.dose_proportionality import power_model
from app.compute.nca import Profile, nca_subject
from app.compute.pk_models import get_model
from app.compute.pk_simulate import simulate_timecourse

from .spec import Target, Task

# Sampling times shared by profile-based categories (hours).
_TIMES = np.array([0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 18, 24], dtype=float)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# ── NCA ────────────────────────────────────────────────────────────────────
def gen_nca(i: int, seed: int) -> Task:
    r = _rng(seed)
    dose = float(r.choice([50, 100, 200, 400]))
    ka = float(r.uniform(0.6, 2.0))
    CL = float(r.uniform(3.0, 12.0))
    V = float(r.uniform(20.0, 60.0))
    conc = conc_1cmt_oral(_TIMES, dose, ka, CL, V)
    truth = nca_subject(Profile(f"S{i:02d}", _TIMES, conc, dose))
    return Task(
        task_id=f"nca-{i:03d}",
        category="nca",
        prompt=(
            f"A single oral dose of {int(dose)} mg was given. Using the plasma "
            "concentration-time data (time in h, concentration in mg/L), report "
            "Cmax (mg/L), AUC0-inf (mg*h/L) and terminal half-life t1/2 (h). "
            "Use linear-up/log-down trapezoidal AUC with tail extrapolation."
        ),
        dataset={
            "time": _TIMES.tolist(),
            "conc": [round(float(c), 6) for c in conc],
            "dose": dose,
            "route": "oral",
        },
        targets=[
            Target("Cmax", truth["Cmax"], {"type": "rel", "rel": 0.05}, "mg/L"),
            Target("AUC_inf", truth["AUC_inf"], {"type": "rel", "rel": 0.05}, "mg*h/L"),
            Target("t_half", truth["t_half"], {"type": "rel", "rel": 0.10}, "h"),
        ],
        oracle="app.compute.nca.nca_subject",
        meta={"true_params": {"ka": ka, "CL": CL, "V": V, "dose": dose}},
    )


# ── Bioequivalence (paired crossover) ──────────────────────────────────────
def gen_be(i: int, seed: int) -> Task:
    r = _rng(seed)
    n = int(r.choice([12, 18, 24]))
    ref = np.exp(r.normal(np.log(1000.0), 0.25, size=n))       # reference exposures
    # true log GMR chosen to land near the 80-125% decision boundary sometimes
    log_gmr = float(r.uniform(np.log(0.85), np.log(1.15)))
    test = ref * np.exp(log_gmr + r.normal(0.0, 0.12, size=n))  # within-subject noise
    truth = be_one_parameter(test.tolist(), ref.tolist(), paired=True)
    return Task(
        task_id=f"be-{i:03d}",
        category="be",
        prompt=(
            "Crossover bioequivalence study, paired Test vs Reference Cmax "
            "exposures for the same subjects. Report the geometric mean ratio "
            "(GMR, %), the lower and upper 90% confidence-interval bounds (%), "
            "and whether the product is bioequivalent (within 80-125%)."
        ),
        dataset={
            "test": [round(float(v), 4) for v in test],
            "ref": [round(float(v), 4) for v in ref],
            "design": "crossover", "parameter": "Cmax", "alpha": 0.10,
        },
        targets=[
            Target("gmr_pct", truth["gmr_pct"], {"type": "rel", "rel": 0.02}, "%"),
            Target("ci_lower_pct", truth["ci_lower_pct"], {"type": "rel", "rel": 0.03}, "%"),
            Target("ci_upper_pct", truth["ci_upper_pct"], {"type": "rel", "rel": 0.03}, "%"),
            Target("within_limits", truth["within_limits"], {"type": "exact"}),
        ],
        oracle="app.compute.bioequivalence.be_one_parameter",
        meta={"true_log_gmr": log_gmr, "n": n},
    )


# ── Dose proportionality (power model) ─────────────────────────────────────
def gen_dp(i: int, seed: int) -> Task:
    r = _rng(seed)
    doses = np.array([25, 50, 100, 200, 400], dtype=float)
    reps = int(r.choice([1, 2]))
    doses = np.repeat(doses, reps)
    slope = float(r.uniform(0.80, 1.20))       # 1.0 == perfectly proportional
    A = float(r.uniform(5.0, 15.0))
    values = A * doses ** slope * np.exp(r.normal(0.0, 0.08, size=doses.size))
    truth = power_model(doses.tolist(), values.tolist())
    return Task(
        task_id=f"dp-{i:03d}",
        category="dp",
        prompt=(
            "Dose-proportionality assessment across escalating doses. Fit the "
            "power model ln(AUC) = a + b*ln(dose) and report the slope b and "
            "whether exposure is dose-proportional (Smith criterion, alpha=0.10)."
        ),
        dataset={
            "doses": [float(d) for d in doses],
            "values": [round(float(v), 4) for v in values],
            "parameter": "AUC_inf", "alpha": 0.10,
        },
        targets=[
            Target("slope", truth["slope"], {"type": "rel", "rel": 0.05}),
            Target("proportional", truth["proportional"], {"type": "exact"}),
        ],
        oracle="app.compute.dose_proportionality.power_model",
        meta={"true_slope": slope},
    )


# ── One-compartment structural PK (parameter recovery) ─────────────────────
def gen_compartmental(i: int, seed: int) -> Task:
    r = _rng(seed)
    dose = float(r.choice([100, 200, 400]))
    ka = float(r.uniform(0.8, 1.8))
    CL = float(r.uniform(4.0, 10.0))
    V = float(r.uniform(25.0, 55.0))
    clean = conc_1cmt_oral(_TIMES, dose, ka, CL, V)
    noisy = clean * np.exp(r.normal(0.0, 0.05, size=clean.size))   # 5% proportional
    truth = fit_one_subject(_TIMES, noisy, dose, model="1cmt")
    p = truth["params"]
    return Task(
        task_id=f"cmt-{i:03d}",
        category="compartmental",
        prompt=(
            "Fit a one-compartment oral model with first-order absorption to "
            f"this concentration-time profile ({int(dose)} mg dose) on the "
            "proportional-error (log) scale. Report clearance CL (L/h), central "
            "volume V (L) and absorption rate ka (1/h)."
        ),
        dataset={
            "time": _TIMES.tolist(),
            "conc": [round(float(c), 6) for c in noisy],
            "dose": dose, "model": "1cmt_oral",
        },
        targets=[
            Target("CL", p["CL"], {"type": "rel", "rel": 0.10}, "L/h"),
            Target("V", p["V"], {"type": "rel", "rel": 0.10}, "L"),
            Target("ka", p["ka"], {"type": "rel", "rel": 0.15}, "1/h"),
        ],
        oracle="app.compute.compartmental.fit_one_subject",
        meta={"true_params": {"ka": ka, "CL": CL, "V": V, "dose": dose},
              "converged": truth["converged"]},
    )


# ── Steady-state exposure prediction (forward simulation) ──────────────────
def ss_exposure(model_key: str, params: dict[str, float],
                dose: float, tau: float, n_doses: int,
                n_points: int = 400) -> dict[str, float]:
    """Steady-state Cmax and AUC over the last dosing interval, by forward
    simulation. Shared by the generator (ground truth) and the oracle agent so
    they cannot drift."""
    model = get_model(model_key)
    tmax = tau * n_doses
    tc = simulate_timecourse(model, params, dose=dose, tau=tau, n_doses=n_doses,
                             tmax=tmax, n_points=n_points)
    t = np.asarray(tc["times"], dtype=float)
    cp = np.asarray(tc["cp"], dtype=float)
    last = t >= (n_doses - 1) * tau - 1e-9      # final dosing interval = steady state
    tw, cw = t[last], cp[last]
    return {"Cmax_ss": float(cw.max()),
            "AUC_tau": float(np.trapezoid(cw, tw))}


def gen_exposure(i: int, seed: int) -> Task:
    r = _rng(seed)
    dose = float(r.choice([50, 100, 200]))
    tau = float(r.choice([12, 24]))
    n_doses = int(r.choice([5, 7]))
    params = {"CL": float(r.uniform(3.0, 10.0)),
              "V": float(r.uniform(25.0, 55.0)),
              "KA": float(r.uniform(0.8, 1.8))}
    truth = ss_exposure("oral_1cmt", params, dose, tau, n_doses)
    return Task(
        task_id=f"exp-{i:03d}",
        category="exposure",
        prompt=(
            "A one-compartment oral model (first-order absorption) has the "
            "clearance CL (L/h), central volume V (L) and absorption rate ka (1/h) "
            f"given below. For a regimen of {int(dose)} mg every {int(tau)} h for "
            f"{n_doses} doses, predict by forward simulation the steady-state peak "
            "concentration Cmax_ss (mg/L) and the AUC over one dosing interval at "
            "steady state, AUC_tau (mg*h/L). Accuracy is judged within 2-fold."
        ),
        dataset={
            "model": "oral_1cmt",
            "params": {k: round(v, 6) for k, v in params.items()},
            "dose": dose, "tau": tau, "n_doses": n_doses, "route": "oral",
        },
        targets=[
            Target("Cmax_ss", truth["Cmax_ss"], {"type": "twofold"}, "mg/L"),
            Target("AUC_tau", truth["AUC_tau"], {"type": "twofold"}, "mg*h/L"),
        ],
        oracle="app.compute.pk_simulate.simulate_timecourse",
        meta={"regimen": {"dose": dose, "tau": tau, "n_doses": n_doses}},
    )


_GENERATORS = {
    "nca": gen_nca,
    "be": gen_be,
    "dp": gen_dp,
    "compartmental": gen_compartmental,
    "exposure": gen_exposure,
}
# Base seeds keep categories independent yet reproducible.
_BASE_SEED = {"nca": 1000, "be": 2000, "dp": 3000, "compartmental": 4000,
              "exposure": 5000}


def build_taskset(per_category: int = 6) -> list[Task]:
    """Generate a reproducible task set: ``per_category`` tasks per category."""
    tasks: list[Task] = []
    for cat, gen in _GENERATORS.items():
        for i in range(per_category):
            tasks.append(gen(i, _BASE_SEED[cat] + i))
    return tasks
