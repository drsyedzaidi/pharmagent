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
- **The page image is a human control, not a proof.** It helps only if the reviewer compares the value with
  the picture (automation bias is likely: `compare_urgent` records say so loudly, and nothing blocks
  Approve on the server; the human-review gate is untouched). It is the one defence for the "custom outline
  under a correct glyph name" limit above, and it shows what pdfium draws, which can differ from Acrobat or
  Preview (font substitution, optional-content layers, form scripts).
- **Outlines are word-level.** The box wraps the whole word, so adjacent punctuation or a neighbouring digit
  in the same word is inside it; a one-digit forgery inside a long word needs the reviewer to read pixels.
  Sub-word boxes would need per-character x boundaries (3-5x the sidecar size). 6 pt footnote digits are
  marginal at 144 dpi (zoom 2 = 216 dpi helps); a page that cannot be rendered at 72 dpi within 6 MP is
  refused (422), and a PNG over 2 MiB after two scale steps likewise.
- **Layout boxes are only as good as the ink check.** A page whose ink check is `unverified` has no
  boxes (the whole page is shown, urgent); boxes come from pdfium glyph boxes, never pdfplumber geometry.
  Sources parsed before evidence-parse 7 have no layout sidecar and are not re-parsed on read: their
  records get the whole-page image (`layout_unavailable`) until the PDF is parsed again (a cache miss
  after a parser bump re-parses it on the next ingest of a NEW session; a re-upload to the same session
  dedupes and does not). A maintenance rebuild that writes the sidecar only when every page text hash equals the stored one is not built.
- **Pre-existing, not fixed here:** pdfplumber's cropbox is wrong for `/Rotate 180` and `270` combined with
  a CropBox (a probe page reports `hidden_chars` 20 and drops visible lines as off-crop, text read
  mirrored). It fails closed (`hidden_text`) but loses text; the layout does not inherit it (pdfium
  geometry), but such a page's words are missing from the sidecar as they are missing from the text.
- **Render resources.** Worst case 2 parse + 2 render children (~2.75 GiB of RSS caps); low-RAM desktop
  builds should set `EVIDENCE_RENDER_SLOTS=1`. Each uncached render spawns an interpreter (~0.2 s);
  rapid record flipping relies on the cache. The render child gets a minimal environment, but RLIMIT_AS is
  not enforced on macOS, there is no network sandbox, and parse / validation children still inherit the full
  environment (a memory-corruption exploit in pdfium could read it): scrub those too when hardening further.
  A frozen desktop build starts children by multiprocessing spawn and cannot scrub the environment.
- **Disk.** `evidence/render/` holds crops of copyrighted pages (<= 256 MiB, LRU) beside the stored PDFs.
- **Pinned renderer.** `pypdfium2` and `pillow` are pinned in `requirements.txt` (and bundled by the desktop
  spec): rendered pixels differ between versions, and `RENDER_VERSION` carries both versions so a change
  invalidates cached images.
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
- **Error headers come from a deny-list.** The global handler forwards the headers of any raised
  `HTTPException` except a fixed unsafe set (`app/response_headers.py`); a route that needs a header in that
  set (a cookie, a CORS header) cannot set it through an exception and must build its own response.

## Later phases

- **Phase 2 — figure digitiser.** Plot regions with axes calibrated from tick labels; digitised points are
  proposals whose locator is a figure bbox + calibration, each confirmed by a reviewer against an overlay.
- **Phase 3 — merge and MBMA.** Join approved long CSVs by record_hash, harmonise units and dispersion
  as deterministic transformations of approved records, and carry locators into model-input provenance.
