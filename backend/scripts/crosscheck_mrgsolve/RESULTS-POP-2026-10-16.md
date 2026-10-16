# Result: PharmAgent population simulator vs mrgsolve on SHARED draws, run 2026-10-16

Run by Claude on the user's explicit instruction ("run the population arm R script"). Software: mrgsolve 1.6.1, R 4.5.1.
Inputs (`pop/design.csv`, `doses.csv`, `obs.csv`, `draws.csv`, concatenated) SHA-256 `593965d0bbf1d8a9cf5b365b6f8223a4c5c4a7d09abcd25f3f4219eb0e064844`.
mrgsolve outputs committed under `reference/pop/`: `mrgsolve_pop.csv` `5553e7aed733d9d499d0bdf236466ebec219e09547764434c7985d1522f9fe10`, `mrgsolve_eta.csv` `af59465c2b71d79ca4e32e7ee4e0cb9e404855131a38cfbcedf4ff5ebf4eb0c6`.
The R script ran on its first execution with no change. The first run of `compare_pop.py` crashed on a bug in the comparer
itself (`main` passed the four files to `compare()` in the wrong order; only `compare()` had been tested): fixed, with a test
of `main`, before any number below was produced.

## Result: PASS, all inside the tolerances fixed before the run
1200 observations, 300 etas. eta max abs 1.1e-16 (tol 1e-12); ipred max rel 7.2e-07 and dv max rel 7.2e-07 (tol 1e-5); BLQ flips 0.

| scenario | obs | etas | BLQ flagged | max eta abs | max ipred rel | max dv rel |
|---|---|---|---|---|---|---|
| pop_iv1 (combined error) | 240 | 60 | 0 | 2.8e-17 | 4.4e-09 | 6.8e-09 |
| pop_oral1 (proportional) | 240 | 90 | 0 | 1.1e-16 | 3.6e-10 | 3.6e-10 |
| pop_iv2_infusion_100kg (additive) | 240 | 60 | 0 | 1.1e-16 | 6.7e-07 | 6.7e-07 |
| pop_mm (Michaelis-Menten) | 180 | 60 | 0 | 5.6e-17 | 7.2e-07 | 7.2e-07 |
| pop_lag_blq (lag, proportional, LLOQ 0.3) | 300 | 30 | 45 | 5.6e-17 | 4.6e-10 | 4.6e-10 |

The ~1e-6 scenarios are the ones PharmAgent integrates numerically (infusion, nonlinear elimination), the same pattern as the
deterministic cross-check. The BLQ scenario exercised 45 flags with no flip.

## What this does and does not show
- Given the SAME standard-normal draws, PharmAgent's `simulate_replicate_subject` and an independent re-implementation of the
  population model in R/mrgsolve give the same etas, individual predictions, simulated observations and BLQ flags: eta = sqrt(omega2)*z,
  theta*exp(eta), weight scaling after the eta, the structural solution, the residual model, the BLQ rule.
- The R re-implementation was written by the same author from PharmAgent's formulas, so this checks arithmetic and transcription, not
  that those formulas are the right population model. It does not compare random-number generators (they cannot agree); that is
  Layer B, PharmAgent's own RNG against exact analytic statistics, which runs in CI.
- Not covered: correlated (block) omega, covariate models, estimation, and one design per error model (30 subjects each).
