/** Evidence-pipeline API contracts. Every interface mirrors a key set pinned in
 *  backend/tests/test_evidence_api_shapes.py (SOURCE_KEYS, SCREEN_ROW_KEYS, ...),
 *  so a renamed backend field fails there before these types drift.
 *  Text from a document (titles, snippets, metadata) is untrusted data: render it
 *  as React text nodes only, never as HTML. */
import type { PharmState } from '../types';

// ── vocabularies (backend app/evidence/models.py) ────────────────────────────
export type ParseStatus = 'parsed' | 'no_text' | 'failed';
/** Whether a source has the layout sidecar the page image needs (backend relayout.layout_info). */
export type LayoutStatus = 'ready' | 'missing' | 'text_changed' | 'unreadable';
export type RelayoutOutcome = 'rebuilt' | 'unchanged' | 'text_changed' | 'unreadable';
export type Verdict = 'include' | 'exclude' | 'unsure';
export type DecisionStatus = 'proposed' | 'approved' | 'rejected';
/** What a reviewer sends for a record; the stored status becomes 'approved' | 'rejected'. */
export type Decision = 'approve' | 'reject';
/** What a reviewer decides for a screened publication (stored as decided_verdict; the status is
 *  'approved' when it agrees with the proposed verdict, 'rejected' when it overrides it). */
export type Inclusion = 'include' | 'exclude';
export type VerifyStatus = 'pass' | 'fail';
export type ScreenMethod = 'llm' | 'keyword';
export type Extractor = 'llm' | 'table' | 'manual';
export type DecimalStyle = '.' | ',';
export type CriterionKind = 'inclusion' | 'exclusion';
export type FieldType = 'number' | 'integer' | 'text';
export type DecimalSetting = 'auto' | '.' | ',';
export type FailReason =
  | 'empty_snippet' | 'snippet_too_long' | 'page_out_of_range' | 'non_ascii_digits' | 'no_number'
  | 'number_mismatch' | 'integer_expected' | 'snippet_not_on_page' | 'table_cell_mismatch'
  | 'source_changed' | 'text_not_in_snippet' | 'partial_number' | 'crosses_columns';
export type FlagCode =
  | 'required_missing' | 'out_of_range' | 'non_integer_n' | 'duplicate' | 'unit_mismatch' | 'unit_missing'
  | 'se_nonpositive' | 'ci_inverted' | 'composite_cell' | 'table_low_confidence' | 'verify_fail'
  | 'non_numeric' | 'suspicious_instruction_text' | 'unit_source_mismatch' | 'unit_unverified' | 'hidden_text'
  | 'superseded' | 'glyph_text_mismatch' | 'glyph_unverified' | 'overlapping_glyphs' | 'censored_value'
  | 'scaled_header' | 'decimal_style_ambiguous' | 'estimate_outside_ci' | 'glyph_ink_mismatch' | 'decided_earlier';

/** Result-card markers the evidence tools push into the transcript (S9 renders them). */
export type EvidenceMarker = '__EVIDENCE_SCREEN__' | '__EVIDENCE_RECORDS__' | '__EVIDENCE_DATASET__';

export type StatusCounts = Record<DecisionStatus, number>;
export type Span = [number, number];

// ── sources ──────────────────────────────────────────────────────────────────
export interface SourceMetadata {
  title: string | null; author: string | null; subject: string | null;
  creator: string | null; producer: string | null; creation_date: string | null;
}

export interface EvidenceSource {
  id: string;
  session_id: string;
  sha256: string;
  filename: string;
  pages: number;
  title: string | null;
  metadata: SourceMetadata;
  text_chars: number;
  parse_status: ParseStatus;
  decimal_style: DecimalStyle;
  licence_note: string | null;
  uploaded_by: string;
  uploaded_at: string;
  layout_status: LayoutStatus;
  /** Why the layout is not ready (server text); null when ready. */
  layout_reason: string | null;
}

/** POST .../sources/{id}/relayout: one audited rebuild. */
export interface RelayoutResult {
  source_id: string; outcome: RelayoutOutcome; reason: string | null;
  layout_status: LayoutStatus; layout_reason: string | null; summary: string;
}
/** One source's entry in a bulk rebuild; `error` is a failure of that source only. */
export interface RelayoutEntry {
  source_id: string; outcome: RelayoutOutcome | 'error'; reason?: string | null; layout_status?: LayoutStatus;
}
export interface RelayoutBulk { results: RelayoutEntry[]; counts: Record<string, number> }
/** POST .../relayout: a job handle (poll it) or, with nothing to do or no job manager, the result inline. */
export type RelayoutAllResponse = {
  selected: number; remaining: number; cap: number; source_ids: string[]; job_id: string | null;
  status: 'running' | 'done';
} & Partial<RelayoutBulk>;

export interface UploadResponse { source: EvidenceSource; duplicate: boolean; summary: string }

// ── configs ──────────────────────────────────────────────────────────────────
export interface Criterion {
  id: string; text: string; kind: CriterionKind; required: boolean; keywords: string[];
}

export interface TemplateField {
  name: string;
  type: FieldType;
  unit: string | null;
  description: string;
  required: boolean;
  min: number | null;
  max: number | null;
  aliases: string[];
  decimal: DecimalSetting;
}

export interface TemplateBody { fields: TemplateField[] }

interface ConfigBase { body_hash: string; saved_by: string; saved_at: string }
export type CriteriaConfig = ConfigBase & { kind: 'criteria'; body: Criterion[] };
export type TemplateConfig = ConfigBase & { kind: 'template'; body: TemplateBody };
export type ConfigResponse<C> = C & { summary: string };

// ── screening ────────────────────────────────────────────────────────────────
export interface ScreenEvidence {
  page: number; snippet: string; span: Span | null; column: number | null;
  /** Keyword path only: the configured term that matched. */
  term?: string;
}

export interface ScreenCriterionResult {
  criterion_id: string;
  kind: CriterionKind;
  required: boolean;
  met: boolean | null;
  evidence?: ScreenEvidence[];
  note?: string;
}

/** Undecided rows carry no decision fields; decided rows always do. */
type Undecided = { status: 'proposed'; decided_by: null; decided_at: null; decision_reason: null };
type Decided = { status: 'approved' | 'rejected'; decided_by: string; decided_at: string; decision_reason: string };
export type DecisionFields = Undecided | Decided;

export type ScreenRow = {
  id: string;
  session_id: string;
  source_id: string;
  criteria_hash: string;
  verdict: Verdict;
  score: number;
  reasoning: string;
  criteria: ScreenCriterionResult[];
  method: ScreenMethod;
  created_at: string;
  /** The reviewer's inclusion; null while proposed (and on rows decided before it was stored). */
  decided_verdict: Inclusion | null;
} & DecisionFields;

export interface ScreeningResults {
  criteria_hash: string;
  n_screened: number;
  verdicts: Record<Verdict, number>;
  proposed: number; approved: number; rejected: number;
  /** Effective inclusion after the reviewer's decisions. */
  included: number; excluded: number; undecided: number;
  method: ScreenMethod | 'mixed';
  updated_at: string;
}

// ── records ──────────────────────────────────────────────────────────────────
export interface TableCell {
  table: number; row: number; col: number; header: string; row_label: string;
  strategy?: 'lines' | 'layout';
}

export interface Locator {
  page: number;
  kind: 'text' | 'table';
  snippet: string;
  /** Raw offsets into the page text (or into columns[column] when column is set). */
  span: Span | null;
  literal: string | null;
  cell: TableCell | null;
  column: number | null;
  decimal_style: DecimalStyle;
  /** Verifier-set raw offsets of the literal in the text it matched (the cell for a table locator). */
  token?: Span | null;
}

export interface QcFlag {
  code: FlagCode | string;
  severity: 'warn' | 'error';
  message: string;
  field: string;
  group: string;
  origin: 'script' | 'llm';
  reason?: string;
}

type Verified = { verify_status: 'pass'; verify_reason: '' };
type Unverified = { verify_status: 'fail'; verify_reason: FailReason };
export type VerifyFields = Verified | Unverified;

export type RecordRow = {
  id: string;
  session_id: string;
  source_id: string;
  template_hash: string;
  field: string;
  group_label: string;
  value: string;
  value_num: number | null;
  unit: string | null;
  locator: Locator;
  extractor: Extractor;
  flags: QcFlag[];
  record_hash: string;
  created_at: string;
} & VerifyFields & DecisionFields;

export interface ExtractionResults {
  template_hash: string;
  n_records: number;
  proposed: number; approved: number; rejected: number;
  verify_fail: number; flagged: number;
  method: Extractor | 'mixed';
  /** Sources whose model pass failed in the last extraction (keyless rules were used instead). */
  llm_fallback: { source_id: string; error: string }[];
  updated_at: string;
}

// ── datasets + snapshot ──────────────────────────────────────────────────────
export interface DatasetRow {
  id: string;
  session_id: string;
  template_hash: string;
  long_path: string;
  wide_path: string;
  long_sha256: string;
  wide_sha256: string;
  record_hashes: string[];
  n_records: number;
  n_sources: number;
  exported_by: string;
  exported_at: string;
}

export interface EvidenceCounts { sources: number; screens: StatusCounts; records: StatusCounts; datasets: number }

/** PharmState.evidence_summary — counts and config hashes only; the DB is the ledger. */
export interface EvidenceSummary {
  sources: number; parsed: number; no_text: number; failed: number;
  screens: StatusCounts; records: StatusCounts; datasets: number;
  criteria_hash: string | null; template_hash: string | null; wide_path: string | null;
  updated_at: string;
}

export interface EvidenceSnapshot {
  session_id: string;
  sources: EvidenceSource[];
  criteria: CriteriaConfig | null;
  template: TemplateConfig | null;
  screens: ScreenRow[];
  records: RecordRow[];
  datasets: DatasetRow[];
  counts: EvidenceCounts;
  summary: EvidenceSummary | null;
}

export interface ParsedTable {
  index: number;
  strategy: 'lines' | 'layout';
  bbox: number[] | null;
  caption: string | null;
  rows: string[][];
}

export interface PageText {
  source_id: string;
  page: number;
  pages: number;
  text: string;
  small_text: string;
  columns: [string, string] | null;
  tables: ParsedTable[];
  decimal_style: DecimalStyle;
}

// ── tool envelopes ───────────────────────────────────────────────────────────
export interface ToolEnvelope<R> {
  agent: string; tool: string; summary: string; state: PharmState; result: R; audit_ok: boolean;
}
/** POST /screen and /extract with a real LLM configured: poll the job. */
export interface JobAccepted { job_id: string; status: 'running'; kind: string; tool: string }
export type ProposeResponse<R> = ToolEnvelope<R> | JobAccepted;

export interface SkippedSource { source_id: string; reason: string }

export interface ScreenResult {
  screens: ScreenRow[]; skipped: SkippedSource[]; criteria_hash: string;
  n_new: number; n_screened: number; method: ScreenMethod | 'mixed';
}

export interface ExtractSourceRow {
  source_id: string; method: Extractor; note: string;
  n_candidates: number; pass: number; fail: number; flagged: number; n_new: number;
  unmatched_headers: string[]; dropped: number;
  /** Why the model pass was abandoned for this source ('' when it was not). */
  llm_error: string;
}

export interface ExtractResult {
  records: RecordRow[]; sources: ExtractSourceRow[]; skipped: SkippedSource[];
  template_hash: string; method: Extractor | 'mixed';
  n_candidates: number; pass: number; fail: number; flagged: number; n_new: number;
}

export interface VerifyRecordsResult { n_checked: number; pass: number; fail: number; changed: number; records: RecordRow[] }

export interface ExportResult {
  dataset: DatasetRow; long_path: string; wide_path: string; long_sha256: string; wide_sha256: string;
  record_hashes: string[]; n_records: number; n_sources: number; template_hash: string;
  long_columns: string[]; wide_columns: string[];
}

export interface ScreenDecision { screen_id: string; decision: Inclusion; reason: string }
/** allow_conflict: approve although an approved record holds a different value for the same slot. */
export interface RecordDecision { record_id: string; decision: Decision; reason: string; allow_conflict?: boolean }

export interface ScreenDecideResponse {
  decided: ScreenRow[]; summary: string; state: PharmState; screening_results: ScreeningResults;
}

export interface RecordDecided {
  record: RecordRow; record_hash: string; source_sha256: string | null; locator: Locator;
  value: string; unit: string | null; verify_status: VerifyStatus; decision: 'approved' | 'rejected'; reason: string;
}

export interface RecordDecideResponse {
  decided: RecordDecided[]; summary: string; state: PharmState; extraction_results: ExtractionResults;
}

/** Snapshot query filters (records list only; counts are never filtered). */
export interface RecordFilter { status?: DecisionStatus; verify?: VerifyStatus; flagged?: boolean }

// ── workspace plumbing (React-free) ─────────────────────────────────────────
/** Runs one mutation with a pending label; errors become the workspace error line.
 *  `onTick` reports elapsed seconds while a job is polled. Resolves true on success. */
export type Run = (label: string, task: (onTick: (elapsedSec: number) => void) => Promise<void>) => Promise<boolean>;
/** Hands a tool envelope (state + summary) to the app; marker selects the result card. */
export type OnTool = (env: ToolEnvelope<unknown>, marker?: EvidenceMarker) => void;
/** Hands a new PharmState + transcript summary to the app (decide endpoints return no envelope). */
export type OnState = (state: PharmState, summary: string, marker?: EvidenceMarker) => void;
/** Workspace tabs, in pipeline order. */
export type EvidenceTab = 'library' | 'criteria' | 'screening' | 'template' | 'records' | 'dataset';
/** A request to show one tab (from a chat card); `n` makes repeated requests distinct. */
export interface EvidenceFocus { tab: EvidenceTab; n: number }

// ── page image (GET .../records/{id}/image/meta) ───────────────────────────
/** A rectangle in PNG pixels (origin top-left). */
export interface ImageRect { x: number; y: number; w: number; h: number }
/** value: the cited word(s); cell: the table cell; line: only the row; page: not located. */
export type ImagePrecision = 'value' | 'cell' | 'line' | 'page';
/** Server-written notice; `text` is shown as-is next to an icon and the word for its severity. */
export interface ImageNotice { code: string; severity: 'warn' | 'error'; text: string }
export interface PageImageMeta {
  record_id: string; source_id: string; page: number; pages: number; zoom: 1 | 2; dpi: number;
  resolved: boolean; precision: ImagePrecision; reason: string | null;
  width: number; height: number; highlight: ImageRect[]; snippet_box: ImageRect | null;
  crop_pt: number[]; image_sha256: string; png_bytes: number; pdf_sha256: string; layout_version: number | null;
  layout_status: LayoutStatus; layout_rebuild_available: boolean;
  extracted: { value: string; unit: string | null; literal: string | null; kind: string };
  page_checks: { ink_check: string | null; hidden_chars: number | null; glyph_check: string[] };
  flags: string[]; compare_urgent: boolean; notices: ImageNotice[];
}
/** Fit: zoom 1 scaled to the width; actual: zoom 1 at natural pixels; larger: zoom 2 at natural pixels. */
export type ZoomMode = 'fit' | 'actual' | 'larger';
