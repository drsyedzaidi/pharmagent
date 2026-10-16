# Population cross-check result, 2026-10-16

mrgsolve 1.6.1, R 4.5.1 (reference/pop/mrgsolve_versions.txt). 5 scenarios x 30 subjects, 1200 observations, 300 etas,
shared draws (pop/draws.csv). Tolerances were stated before the run (README).

| quantity | max deviation | tolerance |
|---|---|---|
| eta | 1.1e-16 abs | 1e-12 |
| ipred | 7.2e-7 rel (pop_mm, subject 19, t=8) | 1e-5 |
| dv | 7.2e-7 rel (same point) | 1e-5 |
| BLQ flags | 0 flips | exact |

PASS. The population layer (eta, weight scaling after the eta, residual model, BLQ flag) agrees with an independent
implementation; the worst gap is the solver tolerance on the Michaelis-Menten model, as in the structural check.
Not covered: see README "Not covered". One comparer bug (argument order in main) was found and fixed on this run.
