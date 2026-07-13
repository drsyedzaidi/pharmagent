# Sourcing Plan — First-in-Human PBPK/ML PK Prediction Benchmark

> ## ⚠ Verified against the live PK-DB API (2026-07-12) — read this first
>
> A hands-on loader build (`pharmacometricsbench/pkdb/loader.py`) reverse-engineered
> the PK-DB REST API end to end. The finding **refines source #1 below**:
>
> - The API is two-step: `GET /api/v1/filter/?studies__substance=<drug>&download=true`
>   returns a **ZIP of CSVs**. The working entry filters are `studies__substance`
>   and `interventions__substance` (by name).
> - **Anonymously, the drug-level PK answer keys are NOT exposed.** `outputs.csv`
>   comes back **empty** and every `outputs__*` filter zeroes the result set. The
>   `clearance` rows that *do* appear in `individuals.csv`/`groups.csv` are
>   **creatinine clearance** (`substance == 'creatinine'`) — a renal covariate, not
>   the drug's clearance. Verified across caffeine, paracetamol, omeprazole,
>   codeine, midazolam, theophylline.
> - What anonymous access **does** give (real + cited): a **dosing + study
>   catalogue** — substance, dose, route, form per study, with reference sid/name
>   and a **licence** flag. Licence matters: caffeine is **92 closed / 13 open**
>   studies, so only a small open subset is redistributable for a benchmark.
> - The drug PK parameters exist in PK-DB and are openly licensed (per its README),
>   but reaching them needs an **authenticated account token** (free PK-DB account),
>   the web UI, or a Zenodo snapshot — not the anonymous REST path.
>
> **Consequence:** the PK-DB loader ships now and harvests the real dosing/study
> catalogue; the answer-key path (`harvest_pk_parameters` → `fih_tasks`) is built
> and unit-tested but **activates only once `PKDB_API_TOKEN` is set**. Until then,
> the honest large-N answer-key route is Drugs@FDA (source #2). No PK values are
> fabricated to fill the gap. Live evidence: `papers/PKDB_loader_run.txt`.

**Goal:** per-compound rows of `predict-side inputs` → `observed human oral AUCinf + Cmax`, graded within 2-fold, every value traceable to a real citation. This plan tells you which sources to pull, how to join them, exactly what to extract, and where the landmines are.

**Critical framing up front:** the target paper (Käser et al., *Mol. Pharm.* 2026, doi:10.1021/acs.molpharmaceut.6c00429) is **not a data source** — it is Roche-internal (all 6 authors Roche; the 40 molecules are Roche pipeline compounds; per-compound observed AUC/Cmax are *not* published, only aggregate %-within-2-fold). Use it as the **method template and the accuracy bar to beat**, never as an answer key. Do not assume its 40-compound set or any of its numbers.

---

## 1. Ranked shortlist (best 2–4 sources)

### #1 — PK-DB (Grzegorzewski et al., *NAR* 2021) — **answer-key backbone (machine-readable)**
- **Pointer:** https://pk-db.com · REST API https://pk-db.com/api/v1/swagger/ · code https://github.com/matthiaskoenig/pkdb · DOI 10.1093/nar/gkaa990
- **Why #1:** the only *open, programmatic, NCA-computed* source of **observed human oral AUCinf, Cmax, tmax, t½, CL, dose, route** with each value traced to a source publication. This is the honest ground truth the benchmark grades against.
- **Access / license:** open; software LGPL-3.0; per-record values trace to original study DOIs. **BUT** (verified 2026-07-12, see banner) the PK-parameter (AUC/Cmax/CL) values are **auth-gated** — the anonymous REST download returns only the dosing/study catalogue with an empty `outputs.csv`. Budget a free PK-DB account + token to reach the answer keys. Dosing/study data (incl. per-study licence flag) is fully anonymous.
- **Gap:** narrow breadth — ~10 well-curated substances in the cited release (acetaminophen, caffeine, codeine, diazepam, glucose, midazolam, morphine, oxazepam, simvastatin, torasemide). Great as a validated seed spine, not large-N. **Further narrowed by licence**: only the open-licence subset per drug is redistributable (e.g. caffeine 13 of 105 studies).

### #2 — Drugs@FDA Clinical Pharmacology & Biopharmaceutics Reviews — **answer-key scale-out (authoritative FiH)**
- **Pointer:** https://www.accessdata.fda.gov/scripts/cder/daf/ → per-NDA "Clinical Pharmacology Biopharmaceutics Review(s)" PDF
- **Why #2:** the single most authoritative source of *actual FiH/SAD oral* AUC(0-inf), Cmax, Tmax, t½, CL/F, V/F by dose level — the real first-in-human numbers, not steady-state label summaries. Public domain, freely citable by NDA number + review date. Many reviews also carry sponsor fu, blood:plasma, in-vitro CLint, BCS class in the same PDF.
- **Access / license:** open, US-Gov public domain (no copyright) — the cleanest license in the whole set.
- **Gap:** semi-structured PDFs → manual/NLP extraction per compound. Budget the curation time; this is the labor cost of scaling past PK-DB's ~10.

### #3 — OSP PBPK Model Library — **fully-provisioned per-compound template rows**
- **Pointer:** https://github.com/Open-Systems-Pharmacology/OSP-PBPK-Model-Library · reports at open-systems-pharmacology.org/OSP-PBPK-Model-Library/ (v12.x, ~47+ qualified small molecules)
- **Why #3:** the only openly-licensed library where each compound ships **BOTH the drug input set** (logP, fu, blood:plasma, MW, pKa, CLint/enzyme kinetics, permeability, solubility in the `.pksim5` file) **AND the observed human data** it was qualified against (oral + IV profiles in the evaluation report). Closest thing to one ready-made benchmark row. Per-compound repos (Midazolam-Model, Rifampicin-Model, …) give a per-drug clinical-reference DOI trail.
- **Access / license:** open; GPLv2 model code; reports freely downloadable.
- **Gaps:** (a) models are *optimized/qualified fits*, not blind FiH predictions — so use OSP for the **observed** column and the **inputs**, but do NOT let OSP's own predictions leak into your prediction side. (b) AUC/Cmax must be extracted from digitized profiles + dose, not read from a clean table; parsing `.pksim5` needed.

### #4 — Input-layer feedstock (pick per compound): TDC ADMET + AZ ChEMBL deposit + Biogen ADME + Lombardo 2018
- **TDC** (https://tdcommons.ai, `pip install pytdc`, MIT/CC): ML-ready SMILES→value for fu (PPBR), CLint (hepatocyte + microsome), Caco-2 permeability, solubility. Best standardized predictor-side feedstock.
- **AZ in-vitro DMPK** (ChEMBL deposit CHEMBL3301361, DOI 10.6019/CHEMBL3301361): fu, hepatocyte **and** microsome CLint, solubility, logD, pKa on publicly-disclosed compounds.
- **Biogen ADME** (Fang 2023, Polaris `adme-fang-v1`, DOI 10.1021/acs.jcim.3c00160): cleanest modern prospectively-collected HLM CLint, MDR1-MDCK permeability, solubility, fu.
- **Lombardo 2018** (DOI 10.1124/dmd.118.082966, PMID 30115648): the fu + human **CL/Vss** truth values (IV-derived) + physchem panel — feeds the disposition leg and lets you sanity-check CL/F back-outs.
- **ChEMBL** (CC BY-SA 3.0): computed MW, logP, PSA, pKa for *any* compound by SMILES to fill missing physchem.

> All #4 sources are **prediction-side only** (no human AUC/Cmax). They must be **joined by structure** onto the #1–#3 answer keys.

---

## 2. Recipe — minimal viable seed set (~10–20 small molecules)

**Design principle:** pick compounds that appear in **multiple** confirmed sources so each row is fully provisioned without guessing.

### Step A — choose the seed compounds by intersection
Start from the overlap of **PK-DB ∩ OSP library ∩ Obach/Lombardo**. Candidates that recur across confirmed sources and have clean oral human PK:

`midazolam, verapamil, sildenafil, caffeine, acetaminophen, diazepam, theophylline, propranolol, alprazolam, simvastatin, omeprazole, fluconazole, felodipine, digoxin, torasemide, codeine`

That is ~16 — a valid MVP. Every one has an oral human answer key in PK-DB and/or an OSP evaluation report, and physchem/fu in Lombardo + ChEMBL.

### Step B — assign the answer key (observed oral AUCinf, Cmax, dose)
1. **First choice:** PK-DB REST API — pull the oral study group with reported AUCinf + Cmax + dose. Record the underlying study DOI PK-DB cites.
2. **If not in PK-DB:** OSP per-compound evaluation report — take the observed (not predicted) oral profile, read/derive AUCinf + Cmax at the stated dose, record the clinical reference DOI from the model README.
3. **For breadth beyond that / true FiH numbers:** the Drugs@FDA ClinPharm review — extract the SAD oral AUC(0-inf)/Cmax table, cite NDA # + review date.

### Step C — assign the prediction-side inputs
Resolve every seed compound to a canonical **InChIKey** (name → SMILES → InChIKey via PubChem or ChEMBL) and join inputs on InChIKey:
- **fu (plasma):** Lombardo 2018 → fallback TDC PPBR → fallback AZ deposit.
- **In-vitro CLint (hepatocyte and/or microsome):** AZ ChEMBL deposit or Biogen ADME or TDC clearance tasks. Record the assay system (hep vs HLM) — they are not interchangeable.
- **Permeability:** TDC Caco-2 or Biogen MDR1-MDCK.
- **Solubility:** TDC / Biogen / AZ.
- **Physchem (MW, logP, PSA, pKa, blood:plasma):** ChEMBL computed + Lombardo measured logD/fup/BPR (measured only for 331 of 1352 — flag when absent).

### Step D — the join key (do this deterministically)
```
name  --(PubChem/ChEMBL)-->  canonical SMILES  -->  InChIKey (14-char skeleton)
join ALL sources on InChIKey; never on drug name string (salt/stereo/synonym drift)
```
Keep salt vs free-base explicit — dose is stated as salt on labels but PK is free-base; note the form so the dose→AUC math is right.

### Step E — write one row per compound, one provenance cell per value
Every numeric cell gets a companion `*_source` cell (DOI / NDA# / API record id). No source → leave the value blank and set a `missing_fields` flag (see §5). Do not interpolate.

---

## 3. Exact fields to extract per drug

| Field | Unit | Primary source | Notes |
|---|---|---|---|
| `compound_name` | — | any | plus synonyms |
| `inchikey` / `smiles` | — | PubChem/ChEMBL | **join key** |
| `salt_form` | — | label/FDA | affects dose math |
| **Answer key (observed human, oral):** | | | |
| `dose` | mg | PK-DB / FDA / OSP | record fasted/fed + formulation (IR vs modified-release) |
| `AUCinf` | µg·h/mL (or ng·h/mL — normalize) | PK-DB / FDA / OSP report | AUC(0-∞); note extrapolated % if given |
| `Cmax` | ng/mL | PK-DB / FDA / OSP report | |
| `tmax`, `t_half`, `CL_F`, `V_F` | h, h, L/h, L | same | supporting, for QC |
| `route` = oral | — | — | **hard filter — exclude IV-only rows** |
| **Disposition truth (for the derive-path / QC):** | | | |
| `CL` (IV), `Vss` | L/h, L | Lombardo 2018 / Obach 2008 | to reconstruct/cross-check exposure |
| `F` (oral bioavailability) | fraction | Varma 2010 / Hosea-Obach 2009 | needed if deriving AUC = F·Dose/CL |
| **Prediction-side inputs:** | | | |
| `fu_plasma` | fraction | Lombardo / TDC PPBR / AZ | |
| `CLint_hep` and/or `CLint_HLM` | µL/min/10⁶ cells; µL/min/mg | AZ / Biogen / TDC | **label the system** |
| `fu_inc` | fraction | Oxford baae063 (2024) | incubation binding to scale CLint |
| `blood_to_plasma` | ratio | Lombardo(331) / OSP `.pksim5` | |
| `permeability` | 10⁻⁶ cm/s | TDC Caco-2 / Biogen MDR1-MDCK | |
| `solubility` | µg/mL or log(mol/L) | TDC / Biogen / AZ | |
| `MW, logP, logD, pKa, PSA` | — | ChEMBL / Lombardo | |
| **Provenance (mandatory, per value):** | | | |
| `<field>_source` | DOI / NDA# / API id | — | one per numeric cell |

---

## 4. Licensing / attribution obligations & proprietary pitfalls

**Safe to redistribute with attribution:**
- **Drugs@FDA reviews / openFDA** — US-Gov **public domain**, no copyright. Cite NDA # + review date. Cleanest.
- **PK-DB** — open, FAIR; software LGPL-3.0. Cite the NAR paper + each underlying study DOI.
- **OSP library** — GPLv2 model code, reports openly downloadable. Cite the compound repo + its clinical references.
- **TDC** — MIT code / CC-style data; cite TDC + the *original* source of each dataset (many are Lombardo/Obach re-derivations — attribute the origin, not just TDC).
- **ChEMBL** — CC BY-SA 3.0 (share-alike: derivative DB inherits BY-SA).
- **AZ deposit** CHEMBL3301361, **Biogen/Polaris**, **Oxford baae063** — open/CC; verify the exact card before shipping.

**Attribution mechanics:** because values chain through aggregators, always cite the **primary** source, not just the redistributor (e.g. a fu that flows Lombardo→TDC must credit Lombardo 2018).

**Verify-before-use (license text not fully pinned):** Lombardo/Obach/Wood/Varma are journal-supplement terms (ASPET/ACS "unknown" reuse) — factual PK numbers are reused field-wide with citation, but do **not** re-host the supplementary Excel verbatim as your own dataset without checking ASPET/ACS terms; store your extracted, re-keyed values with citations instead.

**Proprietary landmines — hard avoid:**
- **Käser/Roche 2026** — do NOT lift its compound list, structures, or any per-drug number. Aggregate stats only, as a benchmark bar.
- **OrBiTo** (Ahmad 2020, EJPB) — perfect *design template* (paired inputs → observed oral AUC/Cmax, 2-fold grading) but the underlying compound DB is **consortium/industry-restricted**, many APIs proprietary. Use the methodology, not the data.
- **DrugBank** — free academic only; **commercial use gated**. Use as a cross-check to confirm PK-DB/FDA values, not as a primary feed.
- **OSP predictions** — its qualified-fit predicted values must not leak into your prediction column (that would be train/test contamination). Take only OSP's *observed* data and *inputs*.

---

## 5. NEVER-FABRICATE checklist

- [ ] **Every numeric cell has a `*_source`** (DOI / NDA#+date / API record id / repo path). No source → cell stays empty.
- [ ] **Missing field → flag, don't guess.** Add compound to a `missing_fields` list naming the absent field (e.g. `sildenafil: CLint_hep missing`). A partly-populated row is fine; an invented value is not.
- [ ] **No drug name, DOI, or PMID written from memory** — resolve each against the live source (PubChem/ChEMBL/PK-DB API/FDA). If a DOI can't be re-resolved, mark the row unverified.
- [ ] **Route hard-filter:** only oral rows enter the answer key. IV CL/Vss stays in the disposition/QC columns, never mislabeled as oral AUCinf.
- [ ] **Units normalized and recorded** before any 2-fold grading; keep the raw unit + the converted value both.
- [ ] **Assay provenance kept:** hepatocyte vs microsome CLint, Caco-2 vs MDR1-MDCK permeability — never merge as one column.
- [ ] **Derived values labeled as derived.** If AUC comes from `F·Dose/CL` rather than a measured oral study, tag it `derived` with the F, Dose, CL sources — do not present it as directly observed.
- [ ] **Salt/free-base form recorded** so dose→exposure math is auditable.
- [ ] **Aggregator → primary citation** on every value that passed through TDC/PKSmart/DrugBank.
- [ ] **No number sourced from Käser/Roche or OrBiTo restricted data.**

---

## Honest gap summary
- **The core tension:** the biggest, cleanest public tables (Lombardo/Obach/PKSmart, 670–1352 cpds) are **IV-only** — they give CL/Vss/fu, **not** the oral AUCinf/Cmax the benchmark grades. The oral answer key only exists in PK-DB (open but ~10 cpds) and Drugs@FDA (thousands but unstructured PDFs). **There is no open, large-N, ready-made oral-AUCinf/Cmax table** — expect real curation labor to scale past ~16.
- **MVP is achievable now:** PK-DB + OSP give ~15–20 fully-provisioned oral rows with open licenses today. That is your seed. Scale-out = FDA-review NLP extraction, one NDA at a time.
- **Input completeness is uneven:** fu is broadly available; measured CLint/permeability/solubility co-occurring on the *same* compound as an oral answer key is the scarce join — accept blanks-with-flags rather than imputing.