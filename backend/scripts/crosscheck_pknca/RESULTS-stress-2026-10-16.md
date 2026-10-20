# Stress table vs PKNCA, after the F1 fix (2026-10-16)

PKNCA 0.12.1 / R 4.5.1, run on the user's explicit instruction. PKNCA outputs are committed in `reference_results/`
with input hashes. Fix under test: branch `fix/nca-blq-auclast` (AUC_last ends at Tlast; BLQ-flagged rows follow rule M1:
leading = 0 baseline, middle and trailing excluded).

## The fix, against an independent engine
`stress_zero.csv` (BLQ coded `DV = 0, BLQ = 1`), PharmAgent vs **PKNCA defaults** (BLQ first keep / middle drop / last keep):
- **AUClast: 11/11 subjects agree, worst relative difference 9e-09.** This includes S05 (middle BLQ), where PKNCA also
  drops the value. Against PKNCA's `matched` configuration (middle zero kept) S05 is the only AUClast that differs
  (60.88 vs 52.42), which is the convention difference, not an error.
- Cmax 11/11, Tmax identical.
- Before the fix the same comparison against the textbook reference showed AUClast/AUCinf too high by 1.2-6.2% (versus the
  analytic AUC(0-inf)); after it every noise-free subject is within 3% of the truth (tests pin this).

## What still differs, and why (not changed)
- **Terminal window (F4).** PKNCA's default tolerance (adjusted-R^2 factor 1e-4) prefers more points than PharmAgent's rule
  (compares adjusted R^2 rounded to 6 decimals, longest window on ties):

  | subject | points PharmAgent / PKNCA default / PKNCA strict | lambda_z PharmAgent | PKNCA default | PKNCA strict | generating ke |
  |---|---|---|---|---|---|
  | S01_clean | 6 / 8 / 4 | 0.099996 | 0.099783 | 0.100000 | 0.10 |
  | S03_trailing_blq | 4 / 6 / 3 | 0.24996 | 0.24878 | 0.24999 | 0.25 |
  | S05_middle_blq | 6 / 8 / 4 | 0.079998 | 0.079869 | 0.080000 | 0.08 |
  | S06_leading_blq | 6 / 7 / 4 | 0.099968 | 0.099751 | 0.100000 | 0.10 |
  | S07_tmax_last | 3 / 5 / 3 | 0.029999 | 0.029827 | 0.029999 | 0.03 |

  lambda_z agrees within 0.6% everywhere; on noise-free data PharmAgent is closer to the generating ke than PKNCA's default.
  AUCinf differs by at most 7e-4 relative. A convention, defensible either way; it is stated in each subject's output
  (`lambda_z_n_points`, adjusted R^2).
- **No sample at the dose time (F5, new).** In `stress_dropped.csv` (the t = 0 BLQ row is absent) PKNCA returns `NA` for the
  AUCs ("Requesting an AUC range starting (0) before the first measurement ... is not allowed"). PharmAgent integrates from its
  first sample and does not impute C(0) = 0, so the dose-to-first-sample area is omitted (0.4% of AUC for S01; more when the
  first sample is later or higher). Both are conventions; the common extravascular convention is to impute C(0) = 0. Not changed.
- **All-BLQ subject (F3).** PKNCA gives AUClast = 0 and AUCinf = 0; PharmAgent gives AUClast = 0 and no AUCinf. Not changed.

## Tooling note
`reference_numpy.py` follows the zero-as-data convention and so disagrees with the fixed PharmAgent on S05 by design; the
tests use PKNCA's default output for that case.
