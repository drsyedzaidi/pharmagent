# PMxbench scenario_00 in PharmAgent

[PMxbench](https://github.com/AIML-SIG/Agentic-workflows/tree/main/tools/pmxbench)
(ISoP AI/ML SIG, MIT licence) is a synthetic population PK study with a known
answer. Scenario_00 (PMB-MAB-01) is a monoclonal antibody: 120 subjects, single
1-hour IV infusion at 100/300/600 mg, TIME in days, assay LLOQ 0.1 mg/L, and
candidate covariates WT, ALB and CRCL.

This page covers how to run scenario_00 end to end in PharmAgent, how to score
the result, and where PharmAgent cannot yet answer what the benchmark scores.

The adapter (`backend/app/benchmarks/pmxbench.py`) only reshapes data and
fills in the submission file. It runs no fit, and its tests run no fit. Every
number in a submission comes from a run the owner starts.

## Rules

- **No answer key.** The repository holds only the three public payload files
  (`data.csv`, `sap.md`, `submission.template.yaml`), pinned to upstream commit
  `e7de2630c5906d74e31681baa53bbd32ddb2cf33` in
  `backend/tests/fixtures/pmxbench/scenario_00/`. The upstream answer key and
  data generator are never fetched, copied, read into code or referenced. A test
  (`test_no_answer_key_file_anywhere_in_the_repository`) fails if a file named
  `truth.*`, `generate.R` or `score.py` appears anywhere in the repository.
  Scoring happens outside it. Besides the payload, two public files were read:
  `tools/pmxbench/LICENSE` (vendored as the fixture `LICENSE`, for the MIT
  notice) and `tools/pmxbench/README.md` (read only, for the scoring command).
- **Blind run.** Scenario_00's answer key is public upstream in the maintainers'
  template folder. To keep the run blind, do not open it before you submit.
- **Unanswered means null.** `sap.md` §9 says to leave an item unanswered rather
  than guess. `build_submission` writes `null` for every item the analysis did
  not answer, and explains why in `provenance.analysis_steps`.

## 1. Prepare the dataset

```bash
cd backend
python -m app.benchmarks.pmxbench prepare \
  --data tests/fixtures/pmxbench/scenario_00/data.csv \
  --out data/pmxbench_s00.csv
```

This writes `data/pmxbench_s00.csv` and `data/pmxbench_s00.report.json`. The
`data/` directory is the default `PHARMAGENT_DATA_DIR`, one of the roots
`load_dataset` may read. Existing files are not overwritten without `--force`.

| Step | What happens | Why |
|---|---|---|
| `derive_rate` | `RATE = AMT / (1/24)` on the 120 dosing rows (2400, 7200 or 14400 mg/day), 0 elsewhere | 1-hour infusion with TIME in days; without RATE the dose is a bolus |
| `map_cens` | `CENS = BLQ` (156 rows: 120 predose, 36 post-dose) | `CENS` is the M3 flag the subject builder reads |
| `censored_dv_to_lloq` | `DV` 0 → 0.1 on `CENS = 1` rows | see below |
| `drop_columns` | `DAY`, `DOSE`, `BLQ`, `ROWID` left out | every column without a structural role is screened as a covariate: `ROWID` and `DAY` would enter SCM, and `DOSE` is auto-detected as a second `AMT` column |

The prepared rows keep the source order, so `report["rowid_by_row"][i]` is the
`ROWID` of prepared row `i`. The prepared columns
(`ID TIME DV AMT RATE CENS WT ALB CRCL`) auto-detect to the right roles, so no
role override is needed. The function fails closed (`ValueError`) on records
the SAP's data specification does not allow: an uncensored `DV <= 0`, `BLQ = 1`
with a non-zero DV or on a dosing row, a missing DV on an observation,
duplicate `ROWID`, missing covariates, a subject without exactly one dosing
record at TIME 0 or without observations, or existing `RATE`/`CENS` columns.
Any column outside the data specification is dropped and listed in
`dropped_columns`.

**Why censored DV becomes the LLOQ.** The SAP describes `DV = 0` on a BLQ
record as a censoring flag, not a concentration. In PharmAgent the flag is
`CENS`. The M3 likelihood term takes its limit from `DV` on censored rows (the
NONMEM convention; `_build_subjects` sets `lloq` to the median DV of the
censored rows). If those rows kept `DV = 0`, the limit would be 0 and each
censored record would add the same constant, wrong term to the OFV.
`test_raw_file_would_break_m3_and_pollute_the_covariate_pool` shows this on the
unprepared file.

**The trade-off.** Only the M3 consumers read `CENS`: `run_nlme`,
`run_diagnostics` and the `run_vpc` BLQ check. Every other caller of the
subject builder keeps rows with `DV > 0`. That includes the two-stage
`fit_pk_model` (whose AIC ranks the structures), `run_scm` and the `run_vpc`
overlay. Those tools score the 36 post-dose censored records as observations
at 0.1 mg/L (LLOQ substitution).
`test_non_m3_consumers_score_censored_records_at_the_lloq` pins this. Leaving
`DV = 0` would make those tools drop the records instead, but would break M3 in
the population fit itself, so the adapter chooses the LLOQ. The proper fix is
G5.

The report also records:

- Baseline covariate summaries and Pearson correlations. In scenario_00, WT~ALB
  is r = 0.40, WT~CRCL is r = 0.00 and ALB~CRCL is r = −0.07.
- Two quantified observations below the LLOQ (`BLQ = 0`, DV < 0.1; ROWIDs 224
  and 363). This is a data-handling note, not an outlier verdict.
- That predose records stay in the file but are not used: the subject builder
  keeps only observations at `TIME > 0`, so M3 sees the 36 post-dose censored
  records.

## 2. Run the analysis (owner)

Start the server as usual. Do not use `--reload`. If `PHARMAGENT_API_TOKEN` is
set, add `-H "Authorization: Bearer $TOKEN"` to each call. Population fits run
as background jobs: poll `GET /api/sessions/$SID/jobs/<job_id>` until each one
finishes.

```bash
API=http://127.0.0.1:8000/api
SID=$(curl -s -X POST $API/sessions | jq -r .id)
curl -s -X POST $API/sessions/$SID/workflow/start -H 'content-type: application/json' \
  -d "{\"workflow\": \"poppk_full\", \"params\": {\"path\": \"$PWD/data/pmxbench_s00.csv\"}}"
```

The `poppk_full` template loads, profiles and validates the data, draws the
spaghetti plot, compares structural models, and then stops at its first human
gate.

**Re-run the structural comparison with IV models (needed).** The template's
comparison step compares the *oral* candidate set (gap G1). At the gate,
compare the IV models instead. This replaces `pk_model_results`:

```bash
curl -s -X POST $API/sessions/$SID/pkmodel -H 'content-type: application/json' \
  -d '{"compare": true, "models": ["iv_1cmt", "iv_2cmt", "iv_3cmt"]}'
```

Next, choose one of two options:

- **A. Continue the template:** `POST /workflow/resume {"approve": true, "reason": "..."}`.
  The rest of the template runs `run_nlme` and `run_scm` with their default
  arguments. For `iv_2cmt` that means IIV on CL and VC, a proportional error
  model, and automatic SCM candidates on CL and VC (G3).
- **B. Recommended:** reject the gate with a reason
  (`{"approve": false, "reason": "IV infusion: modelling steps run with explicit arguments"}`;
  the rejection is signed into the audit chain), then run the modelling steps
  with explicit arguments:

```bash
# Structural / residual-error candidates. Each call REPLACES nlme_results, so
# record each OFV, make the final model's call the LAST /nlme call, and run SCM
# with that same model_key and error_model (the adapter refuses a mismatch).
curl -s -X POST $API/sessions/$SID/nlme -H 'content-type: application/json' \
  -d '{"model_key": "iv_2cmt", "iiv_params": ["CL", "VC"], "error_model": "proportional", "method": "focei"}'
# Covariates (SCM on the same model and IIV)
curl -s -X POST $API/sessions/$SID/scm -H 'content-type: application/json' \
  -d '{"model_key": "iv_2cmt", "iiv_params": ["CL", "VC"], "error_model": "proportional"}'
# Diagnostics and review. CWRES/npd need the two-stage fit (pk_model_results)
# to be the SAME model as the final NLME fit, and the VPC band is simulated from
# the two-stage fit (G10), so fit the final structure two-stage first.
# (This replaces the stored comparison: save the earlier /pkmodel compare response
# if you want the ranking; analysis_steps only describes a stored comparison.)
curl -s -X POST $API/sessions/$SID/pkmodel -H 'content-type: application/json' \
  -d '{"model_key": "iv_2cmt"}'
curl -s -X POST $API/sessions/$SID/diagnostics
curl -s -X POST $API/sessions/$SID/vpc -H 'content-type: application/json' -d '{"blq_check": true}'
curl -s -X POST $API/sessions/$SID/forest
curl -s -X POST $API/sessions/$SID/review -H 'content-type: application/json' -d '{}'
# Export the state for the submission
curl -s $API/sessions/$SID/state > state.json
```

The model key, IIV and error model above are examples. Choosing them is the
analyst's job. `method` may be `focei_saem` or `auto` for start-sensitive fits;
`auto` is slow. The fit's `converged` flag is permissive: it is set when the
optimizer reports success *or* the OFV improved on its starting value. A
FOCE-I local minimum can therefore pass as converged. Judge convergence from
the iterations, condition number and covariance note that `analysis_steps`
records, and from a second start (`focei_saem`).

## 3. Write submission.yaml

```bash
python -m app.benchmarks.pmxbench submit --state state.json --out submission.yaml \
  --harness none --run-utc 2026-10-24T12:00:00Z    # --model / --harness-version if an LLM drove the run
```

| Key | Source in PharmState | null when |
|---|---|---|
| `structural_ncmt` | final model key: `iv_1cmt` → 1, `iv_2cmt` → 2, `iv_3cmt` → 3 | no converged final fit; final model outside the IV linear family; session dataset is not the `prepare` output; SCM and the stored NLME fit describe different models |
| `disposition_params` | final `theta`, renamed CL→CL, VC→V, Q→Q, VP→V2. These are typical values at the 70 kg allometric reference. The 3-cmt names (Q2→Q, VP2→V2, Q3, VP3→V3) are a convention the scorer may not share | as above |
| `error_model` | final `sigma`: proportional → `prop`, additive → `add`, combined → both | unmapped error model |
| `cov_effects` | SCM-selected effects only, keyed by the standard parameter name; value = the final fit's coefficient (power exponent for continuous covariates). `{}` when SCM searched and retained nothing; WT is never a candidate (fixed allometry), so `{}` says nothing about a WT effect | as above; SCM not run; SCM stopped early (`note`: no usable candidates, base model not converged) or its final fit did not converge |
| `outlier_records` | none | always: PharmAgent has no record-level outlier screen (G7) |
| `provenance` | `tool: PharmAgent`; `software` from the session's run provenance, only when its status is `verified` (a quarantined session is refused outright); harness, model, harness_version and run_utc from CLI flags (PharmState does not record them, G9); `analysis_steps` describes the final model, SCM and every null | |

**Which fit is the final model?** The adapter uses the same single-provenance
rule as the covariate tools. If SCM selected covariates, the final model is the
SCM final fit; otherwise it is the stored NLME fit. When both exist, they must
have the same model key and residual-error model. Otherwise which fit is final
is ambiguous, and every structural and parameter item is null. An SCM final
fit is flagged in `analysis_steps` because `run_scm` does not apply M3 (G5).
An M3 refit of the SCM-selected model is not possible over HTTP today:
`/nlme` takes no `covariate_model`, and `run_nlme` is not proposable from chat.

## 4. Score

- **Official:** paste `submission.yaml` into a PMxbench submission issue upstream.
  The maintainers score it and post the result to the leaderboard.
- **Local:** clone the upstream repository **outside** this one. In the
  maintainers' template folder, run the scorer as the public
  `tools/pmxbench/README.md` describes (`python3 score.py <submission.yaml>`).
  Never copy that folder, its answer key or its scorer into this repository; the
  answer-key test fails if they land here.

## 5. Capability gaps

Each scored item, the PharmAgent tool that answers it, and its status. These
were checked against the code on 2026-10-23 without running a fit.

| Item | PharmAgent tool | Status | Evidence / caveat |
|---|---|---|---|
| `structural_ncmt` | `fit_pk_model` (compare, two-stage AIC) → `run_nlme` | **partial** | `iv_1cmt`, `iv_2cmt` and `iv_3cmt` exist, and `dose_events` turns RATE into a zero-order infusion. But `poppk_full`'s comparison uses `_DEFAULT_ORAL`, and `_DEFAULT_IV` is defined but unused, so IV data is ranked against oral models unless `models` is passed (G1). Selection is by two-stage AIC, and that fit scores censored records at the LLOQ (G5). There is no NLME-level ΔOFV comparison, and `nlme_results` keeps only the last fit (G2) |
| `disposition_params` at 70 kg | `run_nlme` theta (or SCM final) | **supported** | Allometry is centred at 70 kg (`pk_simulate.scale_params`), so theta is the 70 kg typical value. Units are L/day and L. Default IIV on `iv_2cmt` is CL + VC: `V` in `iiv_params`, and in the default, means the model's central volume (G3, fixed) |
| `error_model` | `run_nlme` sigma | **partial** | Proportional, additive and combined are all supported. The template default is proportional, and there is no automated error-model comparison (G2) |
| `cov_effects` | `run_scm` | **partial** | Forward p 0.05 / backward p 0.01, estimated coefficients, a post-selection caveat and `screened_out` reporting all work. But: WT exponents are fixed at 0.75/1.0 and cannot be estimated (G4). On `iv_2cmt` the automatic candidates target CL and VC, and a covariate effect on a parameter the model lacks is rejected before any fit (G3, fixed). SCM has no M3: censored records enter at the LLOQ (G5). There is no decoy-covariate analysis (G6) |
| `outlier_records` | none | **missing** | No record-level screen. `profile_pk_dataset` reports sparse subjects and BLQ %; `validate_cdisc` checks the columns and monotonic time; `stats_advice` counts IQR outliers on exposures, not records (G7) |
| 1- and 2-cmt IV infusion | `iv_1cmt`, `iv_2cmt` + RATE | **supported** | An infusion disables the matrix-exponential fast path (`dosing.dose_events`), so fits integrate ODEs and run slower |
| M3 with CENS | `run_nlme`, `run_diagnostics`, `run_vpc` (BLQ incidence) | **supported** with the adapter | Needs the LLOQ in DV on censored rows (step 1). Gaps: the two-stage fit, SCM and the VPC overlay score censored records at the LLOQ (G5). `run_diagnostics` reports npd as `blq_unsupported` when BLQ records exist. `profile_pk_dataset`'s BLQ % counts DV ≤ 0, so it shows 0 % on prepared data (G8) |
| TIME in days | all estimation tools | **supported** | Estimation is unit-agnostic and RATE is per day. Only CL and V/VC get data-derived starting values; Q and VP start at the model defaults (10, 50) whatever the time unit. `focei_saem` / `auto` exist for start-sensitive fits |
| 70 kg reference reporting | allometric scaling | **supported** (fixed exponents) | The SAP asks for supported relationships to be estimated rather than fixed; an estimated WT exponent is missing (G4) |
| Correlated decoy covariate | `run_scm` collinearity screen | **partial** | The screen drops a candidate correlated at \|r\| > 0.3 with an already-kept candidate on the same parameter, by name order. WT (fixed allometry) is never in the screen: scenario_00's baseline WT~ALB r = 0.40 is above the threshold but never compared. ALB is tested conditional on fixed-exponent WT, with no competing-covariate comparison (G6) |
| VPC / diagnostics (not scored) | `run_vpc`, `run_diagnostics` | **partial** | The VPC band and pcVPC are simulated from the two-stage fit's typical values and IIV, with only sigma taken from NLME. CWRES/npd need the two-stage and NLME model keys to match, or they report `needs_nlme`; npd reports `blq_unsupported` whenever BLQ records exist (G10) |
| Provenance | `build_submission` | **partial** | Software versions come from run provenance. The LLM provider/model and harness are not recorded in PharmState, so they come from CLI flags (G9) |

### Proposals

None of these are implemented in this PR. In particular, the outlier screen
(G7) is a proposal only.

**G1 — Route-aware structural candidates.** When `compare` is called without
`models`, `fit_pk_model` should pick its candidate set from the data: IV
(`_DEFAULT_IV`) when every dose has `RATE > 0`, or a ROUTE/CMT column marks IV;
oral otherwise. It should record which set and why in `pk_model_results`.
Separately, `start_workflow` `params` could be allowed to override a step's
`args` (here `models`), validated against the tool schema, so templates stay
fixed but the analyst can steer them. Test: an IV-infusion fixture through
`poppk_full` ranks only IV models.

**G2 — NLME-level model comparison.** Structure and residual error should be
chosen on the population likelihood, not on two-stage AIC. A job-backed,
gated `compare_nlme` tool would fit a list of
(model_key, iiv_params, error_model) candidates and keep every fit's OFV,
condition number and convergence in a table (not only the last fit). It would
report ΔOFV and AIC/BIC and leave the choice at a human gate. It is expensive
(one NLME fit per candidate), so it must go through the existing
expensive-tool admission and never run from an automated loop.

**G3 — Central-volume alias for IIV and SCM defaults (fixed).** `V` in
`iiv_params`, and in the default `["CL", "V"]`, now means the model's central
volume (`V`, or `VC` on 2- and 3-cmt models). Duplicates collapse and order is
kept. `run_scm` and `scm` resolve IIV with the same rule as the fits, so the
automatic candidates on `iv_2cmt` name `VC`. A covariate effect on a parameter
the model does not have now raises before any fit (`focei_fit`, `saem_fit`,
`scm`). It used to be a coefficient that `_apply_cov` ignored: ΔOFV 0, but
still counted in `n_candidates`.

**G4 — Estimated allometry.** Add `allometry: "fixed" | "estimated" | "off"` to
`run_nlme` and `run_scm`. `estimated` replaces the fixed exponents with power
covariate coefficients on WT centred at 70 kg, so theta stays the 70 kg
typical value. WT then enters SCM as an ordinary, testable candidate, the
collinearity screen can see it, and `build_submission` can report the
estimated WT exponents under `cov_effects`. Keep `fixed` as the default, so
existing results are bitwise-unchanged.

**G5 — Honour CENS in every consumer.** Only `run_nlme`, `run_diagnostics`
and the `run_vpc` BLQ check build subjects with `with_blq=True`. Every other
caller (`fit_pk_model`'s two-stage fits, `run_scm`, the `run_vpc` overlay, the
forest/exposure/simulation helpers and `run_prior_check`) keeps `DV > 0` rows.
Under the NONMEM convention (DV carries the LLOQ), those callers score censored
records as observations at the LLOQ; under `DV = 0` they drop them. The
choice is silent either way. Fix:
- SCM: pass `with_blq=True`, so the FOCE-I search and final fit use M3, and
  their OFVs are comparable with the base fit. A test should assert that
  `scm_results["final"]["n_blq"]` equals the base fit's.
- Estimators without M3: drop `CENS = 1` rows explicitly (M1) and record the
  count; never substitute the LLOQ.
- Add `covariate_model` to `/nlme` (`NlmeRequest`), so an SCM-selected model
  can be refitted under M3 over HTTP.

**G6 — Decoy (correlated-covariate) handling.** Include the allometric WT in the
collinearity report, even while it is fixed. When two correlated covariates
are each significant alone, report their univariate ΔOFVs, the joint model and
the ΔOFV of each conditional on the other, then flag the pair for a human
decision instead of letting name order pick. The prior mechanistic
information in the SAP (a mAb is not renally cleared) belongs in that review,
not in an automatic rule.

**G7 — Record-level outlier screen (proposal only).** Add a data-quality tool
that writes `data_quality["outlier_records"]`: ROWID-level flags, each with the
method, statistic and threshold. Two complementary screens:

1. Before modelling: within-subject profile checks. After the end of infusion,
   a single-dose linear IV profile should decline monotonically, so a large
   log-concentration rise relative to the neighbouring timepoints is flagged.
2. After the base fit: conditional residuals (`|CWRES|` above a pre-specified
   threshold) from the stored NLME fit.

Flags are never excluded automatically. Each exclusion is a signed human
decision, with a sensitivity fit with and without the flagged records. Only
then would `build_submission` fill `outlier_records`.

**G8 — Profile BLQ from CENS.** When a `CENS` column exists, `profile_pk_dataset`
should report BLQ % from it rather than from `DV <= 0`, so prepared (NONMEM
convention) datasets do not show 0 % BLQ.

**G9 — LLM/harness provenance.** Record the LLM provider and model id in
effect when a workflow starts, and the identity of every chat-proposed tool
call, in the session (audit-bound). `build_submission` could then fill
`provenance.model` from the session instead of a CLI flag.

**G10 — VPC from the population fit.** `run_vpc` takes its typical values and
IIV from the two-stage fit in `pk_model_results`, using only sigma from
`nlme_results`. After an NLME fit, the VPC should simulate from the NLME theta,
Omega, sigma and covariate model by default (the same single-provenance rule
as `run_diagnostics`), and report which fit it used. A test should pin that the
band changes when only `nlme_results` changes.
