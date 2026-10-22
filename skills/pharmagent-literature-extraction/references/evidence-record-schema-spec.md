# Evidence Record Schema Specification (Draft / Disabled)

Companion reference to `pharmagent.literature_extraction` (`v0.1.0`, `runtime_available: false`).

## Canonical Evidence Record Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `record_id` | `str` (`SHA-256`) | Yes | Content-addressed digest over source hash, locator, parameter, value, unit, and statistic type. |
| `source_sha256` | `str` (`SHA-256`) | Yes | Exact byte digest of the ingested source PDF/document. |
| `document_id` | `str` | Yes | Persistent identifier (DOI, NDA/BLA review ID, EPAR ID, or internal CSR ID). |
| `license` | `str` | Yes | License or usage classification (`CC-BY-4.0`, `US-Gov-Public-Domain`, `Sponsor-Confidential`). |
| `publication_status` | `enum` | Yes | `active`, `corrected_by_erratum`, `retracted`, or `superseded_review`. |
| `source_locator` | `object` | Yes | `page_number`, `section_heading`, `table_id`/`figure_id`, `row_header`, `column_header`, `cell_coordinates`. |
| `extraction_kind` | `enum` | Yes | `verbatim` (must pass literal-span check) or `derived` (must cite parent `record_id`s and tool). |
| `verbatim_quote` | `str` | Yes | Exact text span or cell string containing the number and unit. |
| `literal_match_verified` | `bool` | Yes | Deterministic check that `raw_value` appears literally in `verbatim_quote`. |
| `ocr_confidence` | `float` | Yes | Parser/OCR confidence in `[0.0, 1.0]`; values `< 0.98` require manual transcription confirmation. |
| `parameter_name` | `str` | Yes | Parameter or exposure metric (`Cmax`, `AUC_inf`, `CL/F`, `V/F`, `t_half`, `KA`, etc.). |
| `raw_value` | `str` | Yes | Exact string representation of the number before float parsing. |
| `raw_unit` | `str` | Yes | Exact unit string as printed in the source table/text. |
| `statistic_type` | `enum` | Yes | `geometric_mean`, `arithmetic_mean`, `median`, `typical_value_theta`, `individual_value`. |
| `dispersion_type` | `enum` | Yes | `sd`, `se`, `cv_pct`, `geometric_cv_pct`, `ci_90`, `ci_95`, `iqr`, `range`, `rse_pct`, `none`. |
| `dispersion_value` | `str \| null` | Yes | Verbatim dispersion value or interval string. |
| `n_subjects` | `int \| null` | Yes | Sample size $N$ associated with the reported summary statistic. |
| `footnote_texts` | `tuple[str, ...]` | Yes | Verbatim table footnotes attached to the row, column, or cell. |
| `conflict_group_id` | `str \| null` | No | Identifier linking competing records when sources disagree. |
| `verification_status` | `enum` | Yes | `proposed_unverified` (default), `verified`, `rejected`, or `tainted_by_retraction`. |
| `verified_by` | `str \| null` | No | Authenticated human reviewer identity (required when `verification_status == "verified"`). |
| `verified_at` | `str \| null` | No | ISO-8601 timestamp of human verification. |
