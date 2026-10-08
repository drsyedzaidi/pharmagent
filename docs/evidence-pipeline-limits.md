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
  the picture (automation bias is likely: `compare_urgent` records say so loudly, the Records tab holds Approve
  until the reviewer ticks that they compared the value, and records the tick in the decision reason; that is a
  UI control, so nothing blocks Approve on the server or through the chat and API, and the human-review gate is
  untouched; a tick is a statement, not proof that anyone looked). It is the one defence for the "custom outline
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
  records get the whole-page image (`layout_unavailable`, `layout_status` `missing`) until the sidecar is
  rebuilt (`rebuild_source_layout`, the Relayout endpoints) or the PDF is parsed again in a new session (a
  re-upload to the same session dedupes and does not).
- **A rebuild is all or nothing, and can refuse.** It writes the sidecar only when every stored page field
  equals a fresh parse and every page text hash matches. If the page text parses differently now (a changed
  parser dependency, a doctored cache) the source is `text_changed`: nothing is written, its records keep the
  whole-page image and `compare_urgent`, and the reviewer extracts again on a new parse (a new session) and
  compares each value with the PDF. A PDF that cannot be parsed again (encrypted, corrupt, killed at a limit) is
  `unreadable`. Every rebuild costs a full parse in the sandbox (up to 150 s, one slot) and bulk calls stop at
  10 sources; a busy parser is a 503 that records nothing. The rebuild proves the page text is unchanged, not
  that the boxes are right: they carry the same word-level and ink-check limits as a fresh parse. The status
  marker of a refused rebuild is per PDF and parser version, so it applies to every session holding the PDF, and
  it is remembered until the stored parse cache changes: a PDF killed once at a time or memory limit stays
  `unreadable` until it is parsed again in a new session. Rebuilds are rate-limited per session and server-wide
  (`relayout_limits.py`).
- **Re-saving the template re-proposes every claim.** Records are keyed on the template hash. A new proposal that
  repeats a claim decided under an earlier template carries a `decided_earlier` flag naming that decision
  (`earlier_decisions.py`); nothing is carried over or decided for the reviewer.
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
- **Stored paths are confined, in the routes.** The image, page-text, PDF-download and layout-rebuild routes (and the
  rebuild tool) resolve the ledger's `pdf_path` and `parse_path` (symlinks followed) and refuse with 409 unless they
  lie inside `data_dir/evidence`, before anything is hashed, read, parsed or served (`path_guard.py`). The other
  agent tools (extract, screen, export, approve) still read the stored paths as the ledger gives them, behind the
  sha256 check, because they carry no data directory; the ledger is internal, so this is defence in depth.
- **Every 429 / 503 carries `Retry-After`** (uploads included, `busy.py`): the 429 gives the seconds until the oldest
  request leaves the window, the saturation 503 a fixed 5 s.
- **The coverage gate runs under a line tracer.** `scripts/evidence_coverage.py` needs every `app/evidence` module at
  80 % or more of its executable lines (`--min`, default 80; the r6 merge gate runs it with `--min 85`; `--data FILE` merges
  chunked runs). Wall-clock assertions use
  `tests/evidence_timing.limit()`, strict normally and scaled x4 only while a tracer is installed.
- **Error headers come from a deny-list.** The global handler forwards the headers of any raised
  `HTTPException` whose name is an RFC 9110 token and whose value is latin-1 without control characters (anything
  else is dropped, never a 500), except a fixed unsafe set (`app/response_headers.py`); a route that needs a header in that
  set (a cookie, a CORS header) cannot set it through an exception and must build its own response.

## What the number reader leaves unresolved by design (r6)

The reader (`numgrammar/`, see "Number reader" in [evidence-pipeline.md](evidence-pipeline.md)) reads a number only when
it matches a small explicit grammar made of an explicit allowed alphabet; everything else is unresolved (value None,
verification fails, the record is flagged, the reviewer compares with the page image). These are losses by design, not
defects; each one is listed so a reviewer is not surprised, and none can pass as another number:

- **Mixed numbers and ratios.** `12 3/4`, `12 1/2`, `12 ½` (`½` is outside: it taints the line) are unresolved. Since r7 a
  ratio of two integers (`3/4`, `140/90 mmHg`, `12 / 15`) is ONE `ratio` token (value n/d, parts n and d): it equals only a
  claim that prints the same ratio, and citing a part alone (`140` of `140/90`) fails `partial_number`, so a blood pressure
  is never read as two exact values. Every other slash or colon shape stays unresolved: `0.1/0.2/0.4 mg/kg`, `5/12/2020`,
  `12 /15`, `12/ 15`, `1 : 2`, `8:00`, a division in running text (`17 / 0.7`), a ratio after a number (`4 12/15`, `2.1)
  404/9696`: a whole part may stand before it) and a keyless proposal of a ratio carries the quotient as its `value_num`.
- **Split decimals and bracketed scales.** `12 .5` and `12 ,5` (a text layer that split a number at its point; both are
  unresolved), `5.2 (x10^3 cells/mL)` and `5.2 [× 10^3]` (the scale hides in a bracket), a reference pair `[1,2]` (never a
  decimal). Since r7 an unmistakable bracketed citation (`efflux [46].`, `studies[47–49]`) is a non-value expression: a record
  citing its number fails, and its neighbours read as if it were a word; a bracket after a number (`89 [46]`, `9[4–9]`) or
  alone in a cell keeps its old reading, and a label glued to a bracket group (`AUC[12–24]`) fails closed as a citation. `(4,6)` stays a decimal: in decimal-comma papers a parenthesised decimal is ordinary, so a reference pair in
  parentheses on a comma page is read as 4.6 (a known gap, not closable without losing every parenthesised SD).
- **Raised exponents without an explicit marker.** `1.2 × 103` and `1.2 x 10 3` (an exponent set at full size but raised: the
  text layer cannot say 10^3 from 103), a bare `× 10` (`7.23 ×10 *`), `2 x 10 mg`, `n = 12 x 10`, the Lancet `0·10`, `12·105`;
  a raised ten without a multiplier (`10³`, `per 10⁶ cells`), a detached exponent (`10 ^ 3`, `10 ³`), a spaced exponent
  (`3.4 E-03`), a star run before a power (`12* 10^3`). A multiplier without a number (`2 × daily`, `dosed 2x`, `2×` meaning
  "twice", `1.2 ×` at a line end, `1.2 × l0³`) is unresolved. The legitimate case fails closed with the illegitimate one.
- **Non-ASCII digits.** Arabic-Indic, fullwidth, circled, Roman, vulgar fractions and every other non-ASCII digit are outside
  the alphabet; a snippet or a proposed value that holds one (a text field's too) fails verification
  (`unresolved_context`). The superscript digits are readable only after a multiplier (`1.2 × 10⁻³`).
- **Look-alike symbols.** Dashes (`‒ ― ﹘ ﹣ － ˗ ➖ ⁃ ⸺`), pluses, dots (`‧ ˙ ٫ ․`), multipliers (`⋅ ∙ ⨯ ✕ ✖ ∗ х Х χ`), comparisons
  (`≦ ≧ ⩽ ⩾ ≲ ≳ ≠`), `÷`, `‱`, fullwidth and small-form signs, Cyrillic and math letters are not in the alphabet and not in the
  equivalents table (ten entries: `− – ‐ ‑ ’` U+202F U+00A0 `µ ∼ ◦`). A number next to one is unresolved: a look-alike multiplier
  would be fail-open, so there is no folding. The em dash (`—`) is outside since r7 (it was a LINE character): every number of
  its line is unresolved, so `Azithromycin — 0.66–1.8y`, `AUCR ≈ 2.3 — a moderate interaction` and a flattened table row
  with `—` for "not available" lose their numbers (166 value-like records of the 60 papers).
- **Bullets, arrows, code characters and symbols on a number's line (r7: the derived line rule).** There is no LINE list:
  a character that is not INSIDE, CONTEXT, GAP or an equivalent's source taints its word, the words one separator away and
  every number of its line. So `• 150 mg`, `cm‖ 33.3`, `t½ 11.4 h`, `$18`, `£150 mg`, an arrow (`ω = 0.40 → CV 41%`), a
  `‘ — … ′` and `_ | ? " # ! { }` anywhere on the line (R code, markdown tables, file names) leave every number of the line
  unresolved. The 42 LINE characters folded into CONTEXT (`κ`, 29 accented letters, `& @ “ ” © ® ™ ∞ § ¶ † ‡`) are allowed
  beside a number (`Adalat® 30 mg`, `partner§ 33 574`, `AUC0–∞ 1234`) but are rejected glued to its digits (`312†`).
  Letters beyond these (`É Š ž ō`, `δ φ ψ ν`) taint the same way until admitted by the corpus rule. `χ` is outside except
  in the chi-square form `χ²(df) = n` (r7, decision 8); the corpus lines `ΔOFV > 6.63 (χ² df=1` are not in the form and lose
  every number of the line. Rotated tables read backwards (`]18[muneuQ&evreH`, digits reversed) are a parse defect the
  reader cannot see: lines that only held a folded mark now read (14 non-value-like records in the 60 papers).
- **Units glued to digits.** `150mg`, `24h`, `1.73m2`, `3MIU`, `10mL`, `100µg`: a letter glued to the digits is rejected
  whatever the letter (a list of allowed unit letters would be a fourth list and would re-admit `10mL` read as `1OmL`). The
  honest cost is mostly text layers that lost every space; per-document counts are in `corpus_after.tsv`.
- **Digits inside label words and lost-space text layers.** `CYP3A4*20`, `AUC0-24`, `t1/2`, `mL/min/1.73m2`, DOIs and
  reference shapes (`2019;13(2):93–103`, `10.1080/14622416`), and any number glued to a word by a text layer without spaces
  (`window(25–100ng/mL)isindicated`) form no token. Leading zeros (`007`, `05`), digit groups that are not groups of three
  (`1,23,456`, `0.123 4`), `A.BBB` with the wrong decimal style (`21741,229`) and more than 40 digits are unresolved.
- **Lines with a character outside the alphabet.** A pdfminer `(cid:N)` marker (a glyph the text layer lost: a lost `≥ ± × ° µ −`),
  a soft hyphen, a Symbol-font glyph read as another character (`¥` for `×`, `£` for `≤`, `Þ ð þ ¼ ª`) or any OUTSIDE character
  anywhere on a line makes every number on that line unresolved: the printed meaning is lost in the text layer. About 0.2% of
  the corpus numbers.
- **Not recovered.** A snippet that omits an invisible character the page holds fails `snippet_not_on_page` (the locator no
  longer drops zero-width characters). A `·` between digits is a decimal point only on a point-decimal page (`12·4`); in a
  document with 3 or more point decimals it also raises `decimal_style_ambiguous` (more likely a product).
- **Stored ledgers.** Rows stored by the old reader keep their `verify_reason` (including `non_ascii_digits`) and their
  token offsets; approved rows are immutable. `verify_records` re-verifies proposed rows under the grammar: expect more
  failures (glued units, `2 x 10 mg`, lines with `(cid:N)`, `140/90`).
- **Measured cost (end-to-end corpus run, 2026-10-07).** Every number the old reader resolved in 448 parse caches became a
  record (a text record on the line holding it, a table record on the cell): 183,274 candidates, 181,225 passed the old
  `verify()`. Under the grammar 164,036 still pass with identical values (0 disagreements) and 17,189 (9.5 %) no longer do:
  135 of 137,153 (0.10 %) in the fixtures, attack PDFs and lecture notes (the 200-digit runs, `_`, look-alike glyphs they
  plant), and 17,054 of 44,072 (38.7 %) in 60 pharmacometrics papers, almost all of them reference lists, DOIs, page ranges,
  dates, ratios, digits inside label words and `(cid:N)` markers, none of which a record cites. The records a reviewer
  could cite (a unit, `±` or qualifier beside the number, or a table cell) lose 1,588 of 44,072 (3.6 % of all records,
  19 % of the value-like ones): a LINE or OUTSIDE character beside the number (`_ — " $ → # §`, R code and lecture notes)
  411, digits glued into words by text layers without spaces 401, glued units 373, ratio and reference shapes
  (`140/90`, `2019;13(2):93`) 179, `(cid:N)` markers 160, the rest (`2×`, leading zeros, old sign bugs) 64. This is above the 1.5 %
  the owner set for ordinary numbers; the alphabet was not tuned to reach it. Every case is listed with its cause in
  `corpus_after.tsv` of the r6 work directory. A number whose own word, neighbouring words and line hold only allowed characters,
  and that is none of the shapes above, resolves exactly as before; one beside an ASCII LINE character (`_ " $ # & ? @ |`: R code,
  file names, markdown tables) does not, by the owner's design.
- **Round-1 corpus delta (same 448 caches, 2026-10-07).** Resolved numbers 165,325 -> 165,209 (-116, 0.07 %, no value
  changed): reference pairs `[1,2]` (52), ratios and dates with a slash or colon word (55), `12 .5` splits (4), `ΔOFV > 6.63
  (χ²` (2), the rest `10.1016/S0024-` shaped references. End-to-end records: 163,918 of the 181,225 that passed the old
  reader still pass (HEAD of round 0: 164,036), 17,307 do not (9.55 %), 0 value disagreements among those that pass both;
  value-like records of the 60 papers that no longer pass 1,606 of 44,072 (3.64 %, was 3.60 %). The table-cell neighbour rule
  costs 22 cells of 900 with a number, all in the attack fixtures (`1.2 ⋅ 10^3`, `12․4`, `12З` next to a value): none in a
  paper. Newly flagged only (the number still resolves): 359 range ends (`... -\n[16]`, page ranges), 26 qualified values
  (`TVCL <- 5` in R code: `<-` holds a `<`).
- **Speed.** Page text reads at about 0.6 times the cost of the old reader; a hostile 1 MB page of separate one-digit
  tokens (`1 ` repeated) costs about 2 s, realistic dense pages at most about 1 s. Round 1 removed the quadratic paths the
  audit found: a 120-row table page (720 records) 6.6 s -> 0.17 s (the page is normalised and tokenised once per request, the
  header rows of a table once), a 100 000-number dose page 14 s -> 0.7 s, two columns with 2 000 occurrences of the snippet
  238 s -> 0.07 s, the style evidence of a 128 KB run of digit groups 40 s -> linear, one 200 000-glyph page of `ﷺ` 3 s and
  470 MB -> bounded (round 1: NFKC capped at four characters of expansion; round 2: the locator has no NFKC, so a glyph
  never expands at all).
- **What round 1 leaves, by design.** (a) The group-label key (`qc_flags.group_key`) keeps NFKC and drops format characters (a
  Unicode category): it is a duplicate detector that only ever MERGES labels (`Healthy` and `Healthy<U+200B>` are one group), so
  a finer key would let an inserted zero-width character dodge the `duplicate` flag; the message that names a code point for a
  human (`verify_messages.py`) uses `unicodedata.name`. Both are named with their reason in `tests/test_evidence_r6_scope.py`,
  which fails for any other use in `app/evidence`. (Round 2 removed the other exceptions: the snippet locator, the unit
  comparison, the template header key and the ink pairing fold through `fold()` and explicit tables, with no NFKC and no
  `str.is*`; see "Round-2 changes" below.) (b) Space characters other than the three the
  equivalents table maps to a thin space are word breaks (pdfplumber's own rule), not OUTSIDE characters: `1<U+2007>234` reads
  as `1` and `234`, each flagged `composite_cell`. (c) Parse caches written before round 1 keep their text (the parser version
  is unchanged: the output differs only for a PDF with a private-use character or U+2007 / U+2008 / U+200A in a number); parse
  the PDF again in a new session. (d) `(4,6)` in parentheses is read as a decimal. (e) A legitimate negative list written
  `−11.8 −14.2` in running text flags both values as range ends (`censored_value`): the shape `12 -15` is the documented range.
  (f) `<-` (R assignment) qualifies the value after it.

- **Round-2 changes (the corpus's parse caches, 449 distinct files today where the bullets above counted 448, and 605 PDFs;
  2026-10-07).** (1) The snippet locator folds through `fold()` and
  explicit tables only (no NFKC, no Unicode `\s`, no `str.is*`, no `str.casefold`; see "Locator"). Census of its inputs, 7,924,403
  characters of page text, columns and cells: the old NFKC step rewrote 837 non-ASCII characters, all of them mathematical italic
  letters of equations (the `𝐶 𝐿 𝑡 𝜃` family) and `¨ ´ ª ™`; U+FB00..U+FB06 occur 0 times, no Unicode space beyond the four GAP
  characters and the thin-space family occurs, and 1 of 2,728 hyphen line breaks joined a letter the alphabet does not list
  (`ð`). Every one of those characters is OUTSIDE or LINE, so a number on their line was already unresolved; what changes is that
  a text field cited on such a line no longer locates through a plain-letter snippet (`snippet_not_on_page`). (2) The ink pairing
  of the parse reads both glyph texts through `fold()`: a text layer that says `²` over a glyph that draws `2` (or the reverse)
  is now `glyph_text_mismatch` instead of paired. Re-parsing the 605 PDFs (602 parse; 1,218 pages) gives the same text hash,
  hidden characters, ink check and glyph check on every page as before. (3) Units and header keys: a unit with a character
  outside INSIDE, CONTEXT and GAP is unresolved (it was NFKC-folded: `㎎` was `mg`); only 2 bracketed-unit cells of the corpus hold
  one (`C₂ > C₁`, not a unit). Of 88 header or row-label cells with a non-ASCII character, `µ` (13) is unchanged and the en
  dash (25) and non-breaking hyphen (1) are now the hyphen they equal in `fold()` (`AUC0–∞` matches the alias `AUC0-inf`; the
  NFKC key kept the dash and matched only an alias that wrote it). (4) End to end, the 181,225 records the old reader passed:
  163,918 still pass, 17,307 do not, 0 value disagreements, and the table of every record that no longer passes is byte-identical
  to round 1 (the corpus snippets are page lines, so the locator change cannot move them). (5) Dead code removed: the
  test-only `verify.find_number_in_snippet` and `numgrammar.find_n` / `token_at` / `words_around` (now in
  `tests/evidence_r6_helpers.py`), `is_digit_run` as an export, the unused constants `config_tools.LAYOUT_KEYS`,
  `regions.REASONS`, `relayout.STATUSES` / `OUTCOMES` and `models.REASON_NON_ASCII_DIGITS` (a stored row keeps the string; see
  "Verifier reason codes"). (6) The unit reader's percent class comes from `alphabet.PERCENTS`: U+2031 (`‱`) and its unit entry
  are gone. (7) The exhaustive tier is gated (see "The exhaustive tier is gated" in evidence-pipeline.md): the full runs of
  this round covered 291 (expression, boundary) pairs x 1,112,064 code points = 323,610,624 variants and, in layer B, seven
  sweeps (scale, style, thin, text, locator, unit, key) x 1,112,064 code points, with 0 errors; the record is
  `backend/tests/evidence_r6_attestation.json`. (8) Coverage (`scripts/evidence_coverage.py`, four chunks merged, 85 % per module):
  every module at or above 87.2 %, 96.5 % overall.

- **r7 owner structure rules (2026-10-07; `pharmagent-evidence-work/r7/corpus_progress.tsv`).** U, the r6 metric (value-like
  records of the 60-paper design corpus that the old reader passed and that no longer pass, over all 44,072 of its records),
  after each rule: r6 HEAD 3.64 % (1,606); decision 5, the LINE list deleted and the line rule derived, 42 of its characters
  folded into CONTEXT and 35 outside, 4.35 % (1,915: the em dash 166, `_` 102, `→` 53 now taint their lines); decision 6,
  the ratio production, 4.34 % (1,912; `12/83(14.5%)` reads its percentage); decision 7, citations as non-value tokens,
  4.40 % (1,941: 1,065 records that cite a reference number now fail, 29 of them counted value-like by the census, all of
  them reference numbers); decision 8, the chi-square form, 4.40 % (no corpus line holds the form); decision 9, the range
  flag of `−11.8 −14.2` pinned by tests, 4.40 %. Value disagreements among records that pass both readers: 0 at every
  step. These structure rules do not reach the 1.5 % target; the residual causes are glued units (361), label digits and
  lost spaces (330), the em dash (167), `(cid:N)` markers (160), parts of ratios and other resolved neighbours (138), `_`
  (102), glued symbols (98) and `→` (53), the first two being the other writers' decisions 2 and 3.

## Later phases

- **Phase 2 — figure digitiser.** Plot regions with axes calibrated from tick labels; digitised points are
  proposals whose locator is a figure bbox + calibration, each confirmed by a reviewer against an overlay.
- **Phase 3 — merge and MBMA.** Join approved long CSVs by record_hash, harmonise units and dispersion
  as deterministic transformations of approved records, and carry locators into model-input provenance.
