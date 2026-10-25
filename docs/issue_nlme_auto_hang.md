# Pre-Existing Timeout / Hang Issue: `tests/test_nlme.py` and `tests/test_omega_block_saem.py`

## 1. Summary Across Both Branches

`git diff --stat origin/main -- backend/app/compute/nlme.py backend/app/compute/pk_simulate.py backend/tests/test_nlme.py backend/tests/test_omega_block_saem.py` is empty (`0` lines changed on `feat/protocol-simulation-workflow`). Both files exceed the 600-second per-file timeout (`rc=137`, `TIMEOUT`) at the exact same test index on both `feat/protocol-simulation-workflow` (`2ffc892065226367aabbeeb6db6704425f4e221e`) and `origin/main` (`d2547a05b87cfd559a031c214a0cd46d83dafb84`):

| Test File | `feat/protocol-simulation-workflow` (`2ffc892`) | `origin/main` (`d2547a0`) | Classification |
| :--- | :--- | :--- | :--- |
| `tests/test_nlme.py` | `rc=137` (`TIMEOUT` at 600 s after 27/52 tests) | `rc=137` (`TIMEOUT` at 600 s after 27/52 tests) | `pre_existing` |
| `tests/test_omega_block_saem.py` | `rc=137` (`TIMEOUT` at 600 s after 14/31 tests) | `rc=137` (`TIMEOUT` at 600 s after 14/31 tests) | `pre_existing` |

---

## 2. Entry 1: `tests/test_nlme.py`

### 2.1 Isolation Run (`-x -v -o faulthandler_timeout=120` under 600 s wrapper, `evidence/isolate_test_nlme.log`)

- **Last test printed before 600 s timeout (`rc=137`):**
  - Test #27 (`tests/test_nlme.py::test_scm_selects_true_covariate_and_rejects_noise`) triggered `Timeout (0:02:00)!` at 120 s before completing (`PASSED [ 51%]`), and Test #28 (`tests/test_nlme.py::test_scm_parallel_matches_serial_selection`) was executing when the 600 s file wrapper killed the process (`rc=137`).
  - When run past the 600 s file wrapper (`evidence/pytest_full_stuck_20261023.log`), Test #50 (`tests/test_nlme.py::test_auto_is_deterministic`) stalls for >300 s (`Timeout (0:05:00)!`).

```text
tests/test_nlme.py::test_no_covariate_model_yields_empty_effects PASSED  [ 46%]
tests/test_nlme.py::test_covariate_recovers_known_exponent PASSED        [ 48%]
tests/test_nlme.py::test_covariate_coefficient_rse_reported PASSED       [ 50%]
tests/test_nlme.py::test_scm_selects_true_covariate_and_rejects_noise Timeout (0:02:00)!
Thread 0x0000000200518e40 (most recent call first):
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent/backend/.venv/lib/python3.13/site-packages/scipy/linalg/_matfuncs.py", line 386 in expm
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent-protocol-simulation/backend/app/compute/pk_simulate.py", line 116 in step_linear
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent-protocol-simulation/backend/app/compute/pk_simulate.py", line 119 in simulate
  File PASSED [ 51%]rc=137
```

### 2.2 Faulthandler Dump on `tests/test_nlme.py::test_auto_is_deterministic` (`evidence/pytest_full_stuck_20261023.log`)

```text
Thread 0x0000000205021e40 (most recent call first):
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent-protocol-simulation/backend/app/compute/pk_simulate.py", line 79 in step_linear
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent-protocol-simulation/backend/app/compute/pk_simulate.py", line 101 in simulate
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent-protocol-simulation/backend/app/compute/nlme.py", line 804 in _predict
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent-protocol-simulation/backend/app/compute/nlme.py", line 984 in predict
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent-protocol-simulation/backend/app/compute/nlme.py", line 1009 in _ind_obj
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent-protocol-simulation/backend/app/compute/nlme.py", line 1124 in half_obj
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent-protocol-simulation/backend/app/compute/nlme.py", line 1059 in _numeric_hessian
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent-protocol-simulation/backend/app/compute/nlme.py", line 1127 in _laplace_subject
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent-protocol-simulation/backend/app/compute/nlme.py", line 1167 in _population_ofv
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent-protocol-simulation/backend/app/compute/nlme.py", line 1674 in outer_obj
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent/backend/.venv/lib/python3.13/site-packages/scipy/optimize/_optimize.py", line 560 in function_wrapper
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent/backend/.venv/lib/python3.13/site-packages/scipy/optimize/_optimize.py", line 3236 in myfunc
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent/backend/.venv/lib/python3.13/site-packages/scipy/optimize/_optimize.py", line 3038 in bracket
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent/backend/.venv/lib/python3.13/site-packages/scipy/optimize/_optimize.py", line 2467 in get_bracket_info
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent/backend/.venv/lib/python3.13/site-packages/scipy/optimize/_optimize.py", line 2500 in optimize
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent/backend/.venv/lib/python3.13/site-packages/scipy/optimize/_optimize.py", line 2735 in _minimize_scalar_brent
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent/backend/.venv/lib/python3.13/site-packages/scipy/optimize/_optimize.py", line 3136 in _recover_from_bracket_error
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent/backend/.venv/lib/python3.13/site-packages/scipy/optimize/_optimize.py", line 3243 in _linesearch_powell
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent/backend/.venv/lib/python3.13/site-packages/scipy/optimize/_optimize.py", line 3567 in _minimize_powell
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent/backend/.venv/lib/python3.13/site-packages/scipy/optimize/_minimize.py", line 775 in minimize
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent-protocol-simulation/backend/app/compute/nlme.py", line 1690 in focei_fit
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent-protocol-simulation/backend/app/compute/nlme.py", line 2271 in _auto_fit
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent-protocol-simulation/backend/app/compute/nlme.py", line 2218 in population_fit
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent-protocol-simulation/backend/tests/test_nlme.py", line 862 in test_auto_is_deterministic
```

### 2.3 Call Path
`tests/test_nlme.py::test_auto_is_deterministic` (line 862) $\to$ `nlme.population_fit` (line 2218) $\to$ `nlme._auto_fit` (line 2271) $\to$ `nlme.focei_fit` (line 1690) $\to$ `scipy.optimize._minimize_powell` / `_linesearch_powell` / `_recover_from_bracket_error` $\to$ `nlme.outer_obj` (line 1674) $\to$ `nlme._population_ofv` (line 1167) $\to$ `nlme._laplace_subject` (line 1127) $\to$ `nlme._numeric_hessian` (line 1059) $\to$ `nlme.half_obj` (line 1124) $\to$ `nlme._ind_obj` (line 1009) $\to$ `nlme.predict` (line 984) $\to$ `nlme._predict` (line 804) $\to$ `pk_simulate.simulate` (line 101) $\to$ `pk_simulate.step_linear` (line 79).

---

## 3. Entry 2: `tests/test_omega_block_saem.py`

### 3.1 Isolation Run (`-x -v -o faulthandler_timeout=120` under 600 s wrapper, `evidence/isolate_test_omega_block_saem.log`)

- **Hanging test ID:** `tests/test_omega_block_saem.py::test_every_method_honours_omega_block[auto]` (Test #15 of 31; tests 1–14 pass, then `[auto]` hits `Timeout (0:02:00)!` at 120 s and is killed by the 600 s wrapper with `rc=137`).

```text
tests/test_omega_block_saem.py::test_every_method_honours_omega_block[saem] PASSED [ 38%]
tests/test_omega_block_saem.py::test_every_method_honours_omega_block[focei] PASSED [ 41%]
tests/test_omega_block_saem.py::test_every_method_honours_omega_block[focei_saem] PASSED [ 45%]
tests/test_omega_block_saem.py::test_every_method_honours_omega_block[auto] Timeout (0:02:00)!
Thread 0x0000000202282e40 (most recent call first):
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent-protocol-simulation/backend/app/compute/pk_simulate.py", line 101 in simulate
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent-protocol-simulation/backend/app/compute/nlme.py", line 804 in _predict
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent-protocol-simulation/backend/app/compute/nlme.py", line 984 in predict
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent-protocol-simulation/backend/app/compute/nlme.py", line 1009 in _ind_obj
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent-protocol-simulation/backend/app/compute/nlme.py", line 1082 in obj
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent/backend/.venv/lib/python3.13/site-packages/scipy/optimize/_optimize.py", line 560 in function_wrapper
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent/backend/.venv/lib/python3.13/site-packages/scipy/optimize/_optimize.py", line 883 in _minimize_neldermead
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent/backend/.venv/lib/python3.13/site-packages/scipy/optimize/_minimize.py", line 772 in minimize
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent-protocol-simulation/backend/app/compute/nlme.py", line 1087 in _conditional_mode
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent-protocol-simulation/backend/app/compute/nlme.py", line 1111 in _laplace_subject
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent-protocol-simulation/backend/app/compute/nlme.py", line 1167 in _population_ofv
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent-protocol-simulation/backend/app/compute/nlme.py", line 1674 in outer_obj
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent/backend/.venv/lib/python3.13/site-packages/scipy/optimize/_optimize.py", line 560 in function_wrapper
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent/backend/.venv/lib/python3.13/site-packages/scipy/optimize/_optimize.py", line 3236 in myfunc
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent/backend/.venv/lib/python3.13/site-packages/scipy/optimize/_optimize.py", line 2572 in optimize
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent/backend/.venv/lib/python3.13/site-packages/scipy/optimize/_optimize.py", line 2735 in _minimize_scalar_brent
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent/backend/.venv/lib/python3.13/site-packages/scipy/optimize/_optimize.py", line 3136 in _recover_from_bracket_error
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent/backend/.venv/lib/python3.13/site-packages/scipy/optimize/_optimize.py", line 3243 in _linesearch_powell
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent/backend/.venv/lib/python3.13/site-packages/scipy/optimize/_optimize.py", line 3567 in _minimize_powell
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent/backend/.venv/lib/python3.13/site-packages/scipy/optimize/_minimize.py", line 775 in minimize
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent-protocol-simulation/backend/app/compute/nlme.py", line 1690 in focei_fit
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent-protocol-simulation/backend/app/compute/nlme.py", line 2369 in _focei_saem_fit
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent-protocol-simulation/backend/app/compute/nlme.py", line 2297 in _auto_fit
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent-protocol-simulation/backend/app/compute/nlme.py", line 2218 in population_fit
  File "/Users/ZaidiSahab/Desktop/Bupropion/pharmagent-protocol-simulation/backend/tests/test_omega_block_saem.py", line 206 in test_every_method_honours_omega_block
...
rc=137
```

### 3.2 Call Path
`tests/test_omega_block_saem.py::test_every_method_honours_omega_block[auto]` (line 206) $\to$ `nlme.population_fit` (line 2218) $\to$ `nlme._auto_fit` (line 2297) $\to$ `nlme._focei_saem_fit` (line 2369) $\to$ `nlme.focei_fit` (line 1690) $\to$ `scipy.optimize._minimize_powell` / `_linesearch_powell` / `_recover_from_bracket_error` $\to$ `nlme.outer_obj` (line 1674) $\to$ `nlme._population_ofv` (line 1167) $\to$ `nlme._laplace_subject` (line 1111) $\to$ `nlme._conditional_mode` (line 1087) $\to$ `scipy.optimize._minimize_neldermead` $\to$ `nlme.obj` (line 1082) $\to$ `nlme._ind_obj` (line 1009) $\to$ `nlme.predict` (line 984) $\to$ `nlme._predict` (line 804) $\to$ `pk_simulate.simulate` (line 101).
