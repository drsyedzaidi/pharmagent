# PKNCA reference results (committed so tests can compare offline; R is never run by the tests)

Produced by `run_pknca.R` (PKNCA 0.12.1, R 4.5.1, see each folder's `pknca_versions.txt`) on 2026-10-16, run by Claude on the
user's explicit instruction ("run the run" / "then run PKNCA"). Re-create with the commands in `../README.md`.

| folder | input table | input SHA-256 |
|---|---|---|
| `pharmaverse/` | `../nca_input.csv` | `df1a05b380344c013f06705b50dd4c05f6ed5297dbfdd63e9ba376f4127da4a3` |
| `stress_zero/` | `../stress_zero.csv` | `65bc99f7befa88dd2269321524f9b684af3a7e04b71dc75d7b51375e4d63818d` |
| `stress_dropped/` | `../stress_dropped.csv` | `8e2eb7e98173f2345cea6be65711f716525adc5a52f2e186f8f920e2e244140d` |

`pknca_default` = PKNCA's own options. `pknca_matched` = linear-up/log-down, all concentrations kept (zeros as zeros),
strict-maximum adjusted R^2 (factor 1e-12), >= 3 points, Tmax excluded. In `stress_dropped/` PKNCA returns `NA` for the AUCs
whenever there is no sample at the interval start (time 0): "Requesting an AUC range starting (0) before the first
measurement is not allowed". Do not edit these files by hand; regenerate them.
