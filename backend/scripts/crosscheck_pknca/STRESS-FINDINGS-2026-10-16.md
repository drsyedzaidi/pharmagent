# Stress table: what PharmAgent's NCA does with noisy terminal phases and BLQ (2026-10-16)

Data: `stress_zero.csv` / `stress_dropped.csv` (same 11 synthetic subjects, known generating parameters in
`stress_truth.json`, two BLQ codings). Engines run so far: PharmAgent `compute_nca` and the textbook NumPy reference.
**PKNCA has NOT been run on this table**; the commands are at the end. Truth = analytic AUC(0-inf) = Dose/(V*ke), available
for the noise-free subjects only.

## Findings (PharmAgent, reproduced; fixes NOT made)
**F1. A BLQ value coded as 0 is used as a measurement, and AUC_last runs past PharmAgent's own Tlast.** Confirmed.
With BLQ rows present as `DV = 0, BLQ = 1`, AUClast and AUCinf are larger than the standard definition (AUClast ends at
the last measurable concentration). The excess is exactly the triangle from the reported Tlast to the next zero row
(S03 0.43958, S08 0.86297, S01 0.87826, equal to five decimals). Versus the analytic truth: AUCinf +6.2% (S08), +1.4% (S01,
S06), +1.2% (S03) for PharmAgent against +1.1%, -0.4%, 0.0%, -1.0% for the textbook. With the BLQ rows absent the two
engines agree to rounding on every area (all 10 subjects). Cause by code reading: `compute_nca` detects the BLQ flag
(role CENS) but only counts it; `_auc_intervals` integrates every observation row. The tool's own result text says
"M1: BLQ records excluded from lambda_z and AUC", which this contradicts. No test pins BLQ behaviour in the NCA tests.
Pinned: strict xfail `test_auclast_stops_at_tlast_when_blq_is_coded_as_zero` and a characterisation of the mechanism.

**F2. A middle BLQ coded 0 is a real zero.** Both PharmAgent and the textbook give AUCinf about 12% low (S05, forced
BLQ at 16 h: -11.7% and -13.9% against truth 62.5), and about -0.6% once the row is dropped. This is a convention, not a bug
in either; PKNCA's default drops middle BLQ, so the PKNCA run will show which convention each engine follows.

**F3. A fully-BLQ subject is reported as a number or vanishes.** Zero-coded: AUC_last = 0.0 (a value for nothing
measurable). Rows absent: the subject is missing from the result with no message. Characterised, not judged.

**F4. Terminal-window ties.** PharmAgent compares adjusted R^2 after rounding to 6 decimals, so near-exact exponentials
tie and the longest window wins (6 points on S01 where a strict float comparison picks 3). lambda_z agrees within 3.2e-4
either way. Whether this matches PKNCA (whose default tolerance also prefers more points) is for the PKNCA run.

No finding on: Cmax, Tmax (identical in all 11), a profile with no terminal phase (S07 and S08 give no lambda_z in either
engine), adjusted R^2 (within 5e-7).

## Limits
Synthetic data with a single model structure; the "textbook" reference follows the same conventions as PharmAgent and is
not an independent engine; the truth check covers six noise-free subjects only.

## To get the PKNCA arm (you run it; nothing runs R automatically), from `backend/`
```bash
for v in zero dropped; do
  python3 scripts/crosscheck_pknca/run_pharmagent.py scripts/crosscheck_pknca/stress_$v.csv scripts/crosscheck_pknca/out_stress_$v
  python3 scripts/crosscheck_pknca/reference_numpy.py scripts/crosscheck_pknca/stress_$v.csv scripts/crosscheck_pknca/out_stress_$v
  Rscript scripts/crosscheck_pknca/run_pknca.R scripts/crosscheck_pknca/stress_$v.csv scripts/crosscheck_pknca/out_stress_$v
  python3 scripts/crosscheck_pknca/compare.py scripts/crosscheck_pknca/out_stress_$v/pharmagent.csv \
    scripts/crosscheck_pknca/out_stress_$v/pknca_default.csv scripts/crosscheck_pknca/out_stress_$v/pknca_matched.csv
  python3 scripts/crosscheck_pknca/truth_check.py scripts/crosscheck_pknca/stress_truth.json \
    scripts/crosscheck_pknca/out_stress_$v/pharmagent.csv scripts/crosscheck_pknca/out_stress_$v/pknca_default.csv \
    scripts/crosscheck_pknca/out_stress_$v/pknca_matched.csv
done
```
