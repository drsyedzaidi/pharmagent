# Result: PharmAgent PK/PD simulator vs mrgsolve, run 2026-10-16

Run by Claude on the user's explicit instruction ("run it"). Software: mrgsolve 1.6.1, R version 4.5.1 (2025-06-13).
Inputs: `pd/scenarios.csv`, `pd/doses.csv`, `pd/obs.csv` (combined SHA-256 `76d80e3bbbe2dd612b34ecc3b46a8fa31cf1840fb6375beeeb2044cff49ad341`). mrgsolve output committed as
`reference/pd/mrgsolve_sim.csv` (SHA-256 `9d96462e23d4f7b8e5ec5525ab4782a18007f93d7b8420688d4a9eec3423b916`). The nine PK/PD models in `run_mrgsolve.R` ran on their first execution
with no change.

## Result
**All 36 scenarios agree within 1e-5, for the concentration and for the response** (9 models x single, five doses at 24 h,
100 kg, high dose).

| | Normalised max difference |
|---|---|
| Concentration, direct and effect-compartment models | 1.8e-11 to 6.3e-11 |
| Concentration, indirect-response and lifespan models | 2.3e-07 to 3.2e-07 (worst 3.2e-07, lifespan 100 kg) |
| Response, direct and effect-compartment models | 1.0e-11 to 1.7e-10 |
| Response, indirect-response and lifespan models | 5.2e-08 to 1.3e-06 (worst 1.3e-06, IDR IV high dose) |

Worst Cmax/peak difference 1.8e-07 (concentration) and 9.2e-08 (response); worst AUC 7.2e-07 and 7.3e-07.
The pattern is the one stated before the run: models PharmAgent integrates with the exact matrix exponential (direct,
effect compartment) agree to ~1e-10, and models it integrates numerically (indirect response, lifespan; LSODA rtol 1e-7)
agree to ~1e-7 to ~1e-6, which is its own solver tolerance. No gap appeared in the response alone, so the response initial
states (baseline KIN/KOUT, transit amounts at 1) match.

## What this does and does not show
- PharmAgent's PK/PD simulator reproduces mrgsolve on these 36 scenarios. As with the PK set, the mrgsolve models were
  written by the same author from the registry equations; the independent evidence is the closed-form and near-exact
  integration tests in `tests/test_crosscheck_mrgsolve.py`, which also pass.
- Not covered: population simulation (IIV, residual error), steady-state dosing flags, covariates beyond weight, PK/PD with
  infusions or multiple routes, parameter values other than the registry defaults.
