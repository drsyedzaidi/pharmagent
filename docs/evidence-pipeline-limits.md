# Evidence pipeline: limits of phase 1 and later phases

Companion to [evidence-pipeline.md](evidence-pipeline.md).

## Limits of phase 1

- **No OCR.** A scanned PDF parses to `no_text` and is skipped. Glyph shapes are not read either: a
  custom embedded outline under a correct glyph name and cmap passes (`glyph_check` covers the rest).
- **The ink check is a threshold, not a proof.** A glyph that is mostly covered but still shows ink on
  ≥ 5 % of its box counts as visible, and glyphs overlapping inside one text object (kerning) are not
  separated. A number glyph cut short by < 3 px (or < 15 %) on a side, or covered inside its box by
  ink of the background colour, is not flagged (`glyph_ink_mismatch` needs OCR to go further).
- **Parse slots are server-wide, scheduled per session id.** Two parses run at once for all sessions, one per session at most, queued fairly with a 20 s wait; fairness holds per session id only (one client sharing one anonymous owner can open many sessions: a multi-user deployment needs auth-level quotas, and the 4-worker upload limiter still queues uploads first in first out before they reach the slot queue). A dense document can hold
  one until the 120 s parse budget (pages over 25 000 glyphs skip the visibility passes). Rendering roughly doubles parse time (a 20-page paper: ~8 s, ~0.45 GB peak in the child);
  pages past the 60 s render budget are `unverified` and flagged, not trusted.
- **No figure digitising.** Values that only appear in plots cannot be extracted.
- **Two-column detection is a heuristic** (an x-gap band crossed by few lines). Unusual layouts can merge
  columns. Lines that span the gap stay in `text` but in neither column.
- **Unruled tables have low confidence.** The `layout` fallback depends on a `Table N` caption and on
  word gaps, and every record from it is flagged `table_low_confidence`.
- **Header matching is exact after normalisation** (case, separators, ∞/µ folding): add `Cmax,ss` as
  its own alias. Unmatched headers are listed. Keyless rules read only a cell's leading number
  (`3,412 (812)` -> `3,412` + `composite_cell`).
- **Ligatures.** A dropped ligature glyph ("ﬁnal" -> "nal") fails closed (`snippet_not_on_page`,
  "quote not found"). **LLM screening quality is unmeasured**: quotes are proven, verdicts are not.
- **Criteria changes.** A source is included only by a decision under the current criteria hash; after
  an edit, re-screen and decide before extracting, approving or exporting its values.
- **One template per session.** Re-saving it changes the hash. Records under the old hash stay readable
  (the UI marks them "earlier template"), but the wide CSV uses the current field list.
- **Limits.** 25 MB and 300 pages per PDF. Password-protected PDFs are refused; owner-password
  (print-locked) PDFs are accepted.

## Later phases

- **Phase 2 — figure digitiser.** Plot regions with axes calibrated from tick labels; digitised points are
  proposals whose locator is a figure bbox + calibration, each confirmed by a reviewer against an overlay.
- **Phase 3 — merge and MBMA.** Join approved long CSVs by record_hash, harmonise units and dispersion
  as deterministic transformations of approved records, and carry locators into model-input provenance.
