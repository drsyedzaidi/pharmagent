# mrgsolve cross-check of PharmAgent's PK simulator

Question: for the same model, parameters, doses and time grid, does PharmAgent's ODE simulator
(`app/compute/pk_simulate.simulate`, scipy) give the concentrations mrgsolve gives?

There are two scenario sets: the **PK set** (38 scenarios, first result below) and the **PK/PD set** in `pd/` (36
scenarios over the nine PK/PD models; added after the first run, see "PK/PD set"). 

**Scope and limits of the PK set.** Ten structural PK models (1-, 2-, 3-compartment IV; 1-compartment oral with and without lag;
2-compartment oral; three-transit-compartment absorption; Michaelis-Menten IV and oral; mixed linear+MM IV) in 38
scenarios: single dose, five doses every 12 h, a 100 kg subject, IV infusion, and a saturating dose. Deterministic
individual simulation only: no IIV, no residual error, no steady-state dosing flag, no covariates beyond weight. A pass
shows agreement on these 38 scenarios; the PK/PD models are the second set and the population simulators are not covered.

## Two independent checks
1. **Closed form (no R).** `tests/test_crosscheck_mrgsolve.py` compares PharmAgent with exact solutions: exponential,
   superposition for multiple doses, the Bateman function, the lagged Bateman function, the bi-exponential, the
   infusion solution, the allometric weight rule, and the Lambert-W solution of Michaelis-Menten elimination. Run it any time.
2. **mrgsolve (R, you run it).** The ten models are written in `run_mrgsolve.R` FROM THE EQUATIONS in the registry entry
   named in each comment, not generated from it; the allometric scaling rule is implemented separately in R from the
   stated convention. So a gap can come from the simulator, from the registry's scaling, or from my transcription:
   read it, do not assume.

## Commands (run from `backend/`)
```bash
D=scripts/crosscheck_mrgsolve

python3 $D/make_scenarios.py                         # tables are committed; this reproduces them byte for byte
python3 $D/run_pharmagent_sim.py $D $D/out           # PharmAgent (scipy). No R.

# YOU run this one. It is never run automatically. It compiles ten small mrgsolve models.
Rscript $D/run_mrgsolve.R $D $D/out

python3 $D/compare_sims.py $D/out/pharmagent_sim.csv $D/out/mrgsolve_sim.csv     # exit 0 / 1 / 2
```
`run_mrgsolve.R` was written without being executed (by design: no R is run on your behalf). If it errors, the usual
suspects are the mrgsolve version's handling of `ALAG_DEPOT`, `obsonly`, or record order at a shared dose/observation time.
`out/mrgsolve_versions.txt` records what ran.

## Conventions side by side
| | PharmAgent | mrgsolve script |
|---|---|---|
| Integration | exact matrix exponential for linear bolus cases; LSODA (rtol 1e-7, atol 1e-9) otherwise | LSODA, rtol = atol = 1e-10 |
| Observation at a dose time | state AFTER the dose | dose records sorted before observations |
| Infusion | zero-order for amt/rate hours | `rate` column, duration amt/rate |
| Lag | every dose shifted by ALAG | `ALAG_DEPOT = ALAG` |
| Weight | `p * (wt/70)^exponent` from the registry | same rule, exponent chosen by parameter name |
| Transit absorption | 3 compartments, ktr = 4/MTT | same |

Metric per scenario (`compare_sims.py`): max|a-b| / max|b|, relative Cmax difference, relative AUC difference;
default tolerance 1e-5.

## What to look for (stated before any mrgsolve run; hypotheses, not results)
1. Linear bolus models should agree to roughly 1e-8 or better (exact exponential vs a tightly-toleranced ODE solver).
2. Infusion and nonlinear models are limited by PharmAgent's own solver tolerance (about 1e-6 against the closed form);
   expect agreement near 1e-6, inside the 1e-5 tolerance.
3. A mismatch confined to the single point at a dose time points at the shared-time record convention, not the model.
4. A mismatch in every `:weight` scenario but no other points at the scaling rule; a mismatch only for the lag model at
   early times points at how the lag is applied.

## Result of the first run
See `RESULTS-2026-10-16.md`: all 38 scenarios agree with mrgsolve 1.6.1 within 1e-5 (worst 1.4e-6); hypotheses 1 and 2
held, 3 and 4 were not triggered. The mrgsolve output is committed in `reference/`, and a test regenerates PharmAgent's side
and compares it with that reference, so the check runs in CI without R. Re-run mrgsolve only when the scenarios or the R
models change (`Rscript ...`, then copy `out/mrgsolve_sim.csv` into `reference/` and update the hash in the results file).

## PK/PD set (`pd/`): nine models, 36 scenarios
All nine registry PK/PD models share a 1-compartment oral PK base and add a response: three **direct** models (linear,
Emax, sigmoid), an **effect compartment** (hysteresis), four **indirect response** models (inhibit kin, inhibit kout,
stimulate kin, stimulate kout, starting at the baseline KIN/KOUT) and a five-compartment **transit/lifespan** chain (cell count =
BASE x the mean transit amount). Scenarios per model: single dose (100, 96 h), five doses every 24 h (168 h), a 100 kg subject, and
a high dose (1000, 96 h) that drives the response toward its maximum. Both the concentration and the response (`eff`)
are compared.

Checks that need no R (already in `tests/test_crosscheck_mrgsolve.py`, all passing): the direct models equal their function
of the exact concentration; the effect compartment equals its exact convolution solution; the four indirect-response models and the
lifespan chain equal an independent integration at rtol 1e-12 written from the documented equations. The mrgsolve arm:

```bash
D=scripts/crosscheck_mrgsolve
python3 $D/run_pharmagent_sim.py $D/pd $D/out_pd
# YOU run this one:
Rscript $D/run_mrgsolve.R $D/pd $D/out_pd
python3 $D/compare_sims.py $D/out_pd/pharmagent_sim.csv $D/out_pd/mrgsolve_sim.csv
```
Hypotheses stated before any mrgsolve PK/PD run: (1) the direct and effect-compartment models agree to about the PK
level (1e-9 to 1e-6); (2) the indirect-response and lifespan models agree to about 1e-6 (PharmAgent's solver tolerance);
(3) a gap that appears only in the response and not in the concentration would point at the initial state of the response
(`RESP_0`, `T1_0..T5_0`) or at the response function, not at the PK.

### Result of the PK/PD run
See `RESULTS-PD-2026-10-16.md`: all 36 scenarios agree with mrgsolve 1.6.1 within 1e-5 for concentration and response (worst
1.3e-6); the three hypotheses held. The mrgsolve output is in `reference/pd/` and is regression-tested without R.
