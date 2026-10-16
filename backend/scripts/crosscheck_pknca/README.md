# PKNCA cross-check of PharmAgent's NCA

Question: on the same table, does PharmAgent's `compute_nca` give the numbers an independent engine (PKNCA) gives,
and where it does not, is the difference a convention or a defect?

**Scope and limits.** Six simulated subjects (pharmaverse `adppk`, see `tests/fixtures/pharmaverse/NOTICE.md`), first
dosing interval only. A pass shows agreement on this table; it does not validate PharmAgent's NCA in general, and
the data are simulated, not clinical. Nothing here runs a model fit.

## The shared table
`nca_input.csv` (committed, built by `make_nca_input.py`): first dose only, nominal time `NFRLT`, observations
`<= 24 h`, blank source values dropped (missing, not zero), the source's pre-dose zero kept and flagged `BLQ = 1`.
The source codes below-LOQ inconsistently (pre-dose `0`, later blanks); the table records that choice, it does not hide it.

## Commands (run from `backend/`)
```bash
# 1  PharmAgent's own NCA tool on the table (deterministic; no model, no network)
python3 scripts/crosscheck_pknca/run_pharmagent.py scripts/crosscheck_pknca/nca_input.csv scripts/crosscheck_pknca/out

# 2  textbook NumPy re-implementation (pre-flight; NOT independent of the method)
python3 scripts/crosscheck_pknca/reference_numpy.py scripts/crosscheck_pknca/nca_input.csv scripts/crosscheck_pknca/out

# 3  PKNCA, the independent reference.  YOU run this; it is never run automatically.
Rscript scripts/crosscheck_pknca/run_pknca.R scripts/crosscheck_pknca/nca_input.csv scripts/crosscheck_pknca/out

# 4  compare (exit 0 all within tolerance, 1 a difference, 2 a reference not produced yet)
python3 scripts/crosscheck_pknca/compare.py scripts/crosscheck_pknca/out/pharmagent.csv \
  scripts/crosscheck_pknca/out/numpy_reference.csv scripts/crosscheck_pknca/out/pknca_default.csv \
  scripts/crosscheck_pknca/out/pknca_matched.csv
```
`run_pknca.R` was written without being executed (by design, no R is run on your behalf). If it errors, the first
suspects are PKNCA parameter names for your installed version; `out/pknca_versions.txt` records what was used.
Tolerances are at the top of `compare.py`.

## Conventions side by side
| | PharmAgent (from `app/compute/nca.py`) | `pknca_default` | `pknca_matched` |
|---|---|---|---|
| AUC rule | linear-up / log-down | linear-up / log-down | same |
| AUClast | integrates all observation rows given | up to Tlast | same |
| BLQ | rows are used as given (zeros included) | first keep, middle drop, last keep | all kept |
| lambda_z | windows of >= 3 points after Tmax, strict max adjusted R^2 | tolerance 1e-4 favours more points | factor 1e-12 (PKNCA requires (0,1), so this stands for 0), >= 3 points, Tmax excluded |
| route | extravascular | extravascular | same |

## What to look for (stated before the run; hypotheses, not results)
1. Cmax, Tmax, AUClast should agree to rounding in both configurations.
2. `pknca_default` may pick a different terminal window than PharmAgent (PharmAgent's strict maximum of adjusted R^2
   tends to choose the shortest window); if so lambda_z, half-life and AUCinf differ and `lambda.z.n.points` shows why.
   `pknca_matched` should then agree. A remaining gap there would be a real finding.
3. Code reading, not yet tested on data: PharmAgent's AUClast integrates every row, so trailing zeros (BLQ coded as 0)
   after the last measurable value would add area beyond Tlast, and AUCinf would add the extrapolation on top. This
   table has no trailing zeros inside 0-24 h, so it cannot show it; a table that does is the next test.

`numpy_reference` already agrees with PharmAgent on all 48 values (worst relative difference 2.5e-06, rounding), and
`tests/test_crosscheck_pknca.py` pins PharmAgent to the analytic mono-exponential solution. Those say PharmAgent
implements its own stated method; only the PKNCA arms say whether that method matches an independent engine.

## Result of the first run
See `RESULTS-2026-10-16.md`: agreement on all 48 values in both PKNCA configurations. Hypothesis 2 was not borne out
(this table's terminal phase is unambiguous), hypothesis 3 remains untested.

## Stress table
`make_stress_table.py` builds 11 synthetic subjects with known parameters (noisy terminal phases, trailing/middle/leading
BLQ, no terminal phase, all-BLQ, a bump) in two BLQ codings. See `STRESS-FINDINGS-2026-10-16.md`: it found that PharmAgent
integrates AUC_last through BLQ rows coded as zero. The PKNCA arm of the stress run is not done yet.
