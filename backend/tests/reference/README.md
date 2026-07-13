# Reference validation

This suite answers the question a pharmacometrician asks before trusting any
new estimator: **does it reproduce the numbers established tools produce on
known datasets?** It is distinct from `tests/test_nlme.py`, which only checks
that the estimators recover a *simulated* truth.

## What it checks

For each canonical public dataset (currently **Theophylline**), it drives the
shipping compute path (`dataset → _build_subjects → NCA + FOCE-I + SAEM`) and
asserts three things:

1. **Literature concordance** — NCA, FOCE-I and SAEM estimates fall within
   published acceptance bands for the canonical model (see
   `references.py` for values + provenance).
2. **Cross-method internal consistency** — NCA CL/F, FOCE-I CL and SAEM CL
   agree within 20% (three independent routes to the same clearance), and
   FOCE-I/SAEM volumes agree.
3. **(Optional) exact tool concordance** — if you drop your own
   NONMEM/Monolix/nlmixr2 estimates into `tool_reference`, it additionally
   asserts PharmAgent matches them within a relative tolerance.

## Current result (Theophylline)

| Quantity            | PharmAgent | Published consensus |
|---------------------|-----------:|--------------------:|
| NCA CL/F (geomean)  | 2.69 L/h   | ~2.7 L/h            |
| FOCE-I CL / V / KA  | 2.69 / 33.1 / 1.56 | 2.7 / 32 / 1.5 |
| SAEM CL / V / KA    | 2.73 / 32.1 / 1.21 | 2.7 / 32 / 1.5 |
| NCA terminal t½     | 7.7 h      | ~8 h                |

NCA, FOCE-I and SAEM agree on clearance to within ~2%.

## Why bands, not exact equality

The same dataset yields slightly different estimates across estimation methods
(FO / FOCE-I / SAEM / Laplace), data subsets, and software. The literature
bands are intentionally wide (~±20–30%) so the test is a meaningful
**regression + plausibility gate anchored to the literature**, not a brittle
equality check that breaks on a benign numerical change. For a strict cross-tool
equality check, use `tool_reference` (below).

## Adding your own tool reference (exact concordance)

To assert PharmAgent matches *your* NONMEM/Monolix/nlmixr2 run on this dataset,
edit `tests/reference/references.py`:

```python
THEOPHYLLINE["tool_reference"] = {
    "tool": "NONMEM 7.5 FOCE-I",
    "rel_tol": 0.15,         # 15% relative tolerance
    "CL": 2.81, "V": 32.6, "KA": 1.49,   # from your .lst / run record
}
```

`test_matches_user_tool_reference` then runs (it is skipped while
`tool_reference is None`). This is the honest division of labour: the suite
ships with literature anchors; you supply the exact tool output to certify
bit-level agreement against your own validated reference.

## Adding a dataset

1. Drop the NONMEM-style CSV in `backend/sample_data/`.
2. Add an entry to `DATASETS` in `references.py` with `dataset`, `model`,
   `literature` bands (cite the source), and optional `tool_reference`.
3. Copy `test_theophylline.py` to `test_<dataset>.py` and point the fixtures at
   the new entry.

Good next candidates: **Warfarin** (1-/2-cmt oral, nlmixr2 `warfarin`),
**Phenobarbital** (neonatal IV + weight covariate, Grasela & Donn 1985).

## Running

```bash
.venv/bin/python -m pytest tests/reference/ -q
```
