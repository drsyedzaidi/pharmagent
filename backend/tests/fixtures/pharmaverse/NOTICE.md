# pharmaverse PK test fixtures: notice and provenance

These files are **subsets of public test datasets** from the pharmaverse project, redistributed under the
**Apache License 2.0** (copies of each package's licence file sit next to this notice).

| File here | Source package | Source file | Pinned commit |
|---|---|---|---|
| `pc_subset.csv` | pharmaverse/pharmaversesdtm 1.5.0.9003 | `inst/extdata/pc.csv` | `7ec53ea88dfe1c168f9efbef59233c4426813b89` |
| `adppk_subset.csv` | pharmaverse/pharmaverseadam 1.4.0 | `inst/extdata/adppk.csv` | `c7d3881173b01504286cf6784622420401ddaa41` |

Copyright holders named by both packages: Cytel Inc., F. Hoffmann-La Roche AG and GlaxoSmithKline LLC (copyright and
funder), with the package authors listed in each package's `DESCRIPTION`. The packages' own documentation describes
`pc` as "a SDTM PC dataset simulated by Antonio Rodriguez Contesti".

**Changes made (Apache-2.0 section 4(b)):** rows were selected by subject. Every kept row is byte-identical to the source
row; no value was edited and no column was dropped. The selection rule and the SHA-256 of the sources and of the
subsets are in `MANIFEST.json`, and `build_subset.py` rebuilds both files from the pinned sources.

**What this data is, and is not.** It is simulated test data in CDISC-style layouts, useful to exercise ingestion, BLQ
handling and NONMEM-style input. It is **not** clinical data and **not** a PK answer key. In particular the companion
`pp` (NCA parameters) dataset in `pharmaversesdtm` does not reproduce the Cmax implied by this `pc`
(checked for Cmax only, 2026-10-16: no subject matched), so do not treat `pp` as the expected result for `pc`.
Use an independent engine (e.g. PKNCA) for a reference.

Provided "AS IS", without warranties or conditions of any kind, as stated in the licence.
