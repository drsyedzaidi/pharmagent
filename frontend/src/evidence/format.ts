/** Pure display + draft helpers for the Evidence workspace (no React). */
import type { AuditEntry, PharmState } from '../types';
import type {
  Criterion, CriterionKind, DecimalSetting, EvidenceMarker, EvidenceSnapshot, FailReason, FieldType, ParseStatus,
  TemplateField, VerifyFields,
} from './types';

/** Client-side mirror of the backend upload cap (ingest.MAX_BYTES). */
export const MAX_PDF_BYTES = 25 * 1024 * 1024;
const FIELD_NAME_RE = /^[A-Za-z_][A-Za-z0-9_]{0,63}$/;

/** Accessible name of a record row: field, group (when set) and publication, so rows with the same
 *  field from different papers are distinguishable to screen readers ("n in paper_b.pdf"). */
export const recordLabel = (r: { field: string; group_label: string }, source: string): string =>
  `${r.field}${r.group_label ? ` · ${r.group_label}` : ''} in ${source}`;

/** First `n` hex characters of a sha256 (mono id chips). */
export const shaPrefix = (h: string | null | undefined, n = 12): string => (h ? h.slice(0, n) : '');

/** The server's error message: main.py wraps errors as {error: {message}}, FastAPI as {detail}. */
export function errorMessage(e: unknown): string {
  const raw = e instanceof Error ? e.message : String(e);
  try {
    const body = JSON.parse(raw) as { error?: { message?: unknown }; detail?: unknown };
    const msg = body.error?.message ?? body.detail;
    if (typeof msg === 'string' && msg) return msg;
  } catch { /* not JSON: the raw text is the message */ }
  return raw || 'request failed';
}

export const PARSE_LABEL: Record<ParseStatus, string> = { parsed: 'parsed', no_text: 'no text', failed: 'failed' };

export function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

/** Latest audit entry of save_evidence_config for one config kind (the action names it). */
export function latestConfigEntry(audit: AuditEntry[], kind: 'criteria' | 'template'): AuditEntry | null {
  for (let i = audit.length - 1; i >= 0; i--) {
    const a = audit[i];
    if (a.tool === 'save_evidence_config' && a.action.startsWith(`save_evidence_config(${kind}`)) return a;
  }
  return null;
}

/** Comma/semicolon separated list -> trimmed, de-duplicated items. */
export function splitList(s: string): string[] {
  return [...new Set(s.split(/[,;]/).map(x => x.trim()).filter(Boolean))];
}

// ── pipeline prerequisites ──────────────────────────────────────────────────
export const parsedCount = (snap: EvidenceSnapshot): number =>
  snap.sources.filter(s => s.parse_status === 'parsed').length;

/** Sources with an approved 'include' verdict: extraction reads only these. */
export const includedCount = (snap: EvidenceSnapshot): number =>
  new Set(snap.screens.filter(s => s.status === 'approved' && s.verdict === 'include').map(s => s.source_id)).size;

// ── criteria draft ──────────────────────────────────────────────────────────
export interface CriterionDraft { id: string; text: string; kind: CriterionKind; required: boolean; keywords: string }

export const toCriterionDraft = (c: Criterion): CriterionDraft => ({ ...c, keywords: c.keywords.join(', ') });

export function blankCriterion(existing: CriterionDraft[]): CriterionDraft {
  const ids = new Set(existing.map(c => c.id));
  let n = existing.length + 1;
  while (ids.has(`c${n}`)) n += 1;
  return { id: `c${n}`, text: '', kind: 'inclusion', required: true, keywords: '' };
}

export const fromCriterionDraft = (d: CriterionDraft): Criterion => ({
  id: d.id.trim(), text: d.text.trim(), kind: d.kind, required: d.required, keywords: splitList(d.keywords),
});

/** First problem with the draft list (mirrors config_tools validation), or ''. */
export function criteriaProblem(rows: CriterionDraft[]): string {
  if (rows.length === 0) return 'Add at least one criterion.';
  if (rows.length > 100) return 'At most 100 criteria.';
  const seen = new Set<string>();
  for (const [i, r] of rows.entries()) {
    const id = r.id.trim();
    if (!id) return `Row ${i + 1}: id is empty.`;
    if (id.length > 64) return `Row ${i + 1}: id is longer than 64 characters.`;
    if (seen.has(id)) return `Row ${i + 1}: id "${id}" is used twice.`;
    seen.add(id);
    if (!r.text.trim()) return `Row ${i + 1}: criterion text is empty.`;
    if (r.text.trim().length > 500) return `Row ${i + 1}: text is longer than 500 characters.`;
  }
  return '';
}

// ── template draft ──────────────────────────────────────────────────────────
export interface FieldDraft {
  name: string; type: FieldType; unit: string; description: string; required: boolean;
  min: string; max: string; aliases: string; decimal: DecimalSetting;
}

export const toFieldDraft = (f: TemplateField): FieldDraft => ({
  ...f, unit: f.unit ?? '', min: f.min === null ? '' : String(f.min), max: f.max === null ? '' : String(f.max),
  aliases: f.aliases.join(', '),
});

export const blankField = (): FieldDraft => ({
  name: '', type: 'number', unit: '', description: '', required: false, min: '', max: '', aliases: '', decimal: 'auto',
});

const num = (s: string): number | null => (s.trim() === '' ? null : Number(s.trim()));

export const fromFieldDraft = (d: FieldDraft): TemplateField => ({
  name: d.name.trim(), type: d.type, unit: d.unit.trim() || null, description: d.description.trim(),
  required: d.required, min: num(d.min), max: num(d.max), aliases: splitList(d.aliases), decimal: d.decimal,
});

export function templateProblem(rows: FieldDraft[]): string {
  if (rows.length === 0) return 'Add at least one field.';
  if (rows.length > 60) return 'At most 60 fields.';
  const seen = new Set<string>();
  for (const [i, r] of rows.entries()) {
    const name = r.name.trim();
    if (!FIELD_NAME_RE.test(name)) return `Row ${i + 1}: name must start with a letter or _ and use letters, digits, _.`;
    if (seen.has(name)) return `Row ${i + 1}: name "${name}" is used twice.`;
    seen.add(name);
    const lo = num(r.min), hi = num(r.max);
    if ((lo !== null && !Number.isFinite(lo)) || (hi !== null && !Number.isFinite(hi))) return `Row ${i + 1}: min/max must be numbers.`;
    if (lo !== null && hi !== null && lo > hi) return `Row ${i + 1}: min is greater than max.`;
  }
  return '';
}

/** Canonical string of a draft list for dirty checks (same shape the server stores). */
export const sameBody = (a: unknown, b: unknown): boolean => JSON.stringify(a) === JSON.stringify(b);

/** Which result card a keyless chat turn routed to the evidence agent produced, judged by
 *  the PharmState mirror it changed (chat responses do not name the tool). An idle-hint
 *  turn changes none, so it pushes no card. */
export function evidenceMarkerFor(prev: PharmState | null, next: PharmState): EvidenceMarker | null {
  if (next.evidence_dataset_path && next.evidence_dataset_path !== prev?.evidence_dataset_path) return '__EVIDENCE_DATASET__';
  if (next.extraction_results && next.extraction_results.updated_at !== prev?.extraction_results?.updated_at) {
    return '__EVIDENCE_RECORDS__';
  }
  if (next.screening_results && next.screening_results.updated_at !== prev?.screening_results?.updated_at) {
    return '__EVIDENCE_SCREEN__';
  }
  return null;
}

/** Plain words for the verifier's reason codes (backend models.REASON_CODES). */
export const REASON_WORDS: Record<FailReason, string> = {
  empty_snippet: 'no snippet cited',
  snippet_too_long: 'snippet over 300 characters',
  page_out_of_range: 'page out of range',
  non_ascii_digits: 'non-ASCII digits',
  no_number: 'value not in snippet',
  number_mismatch: 'number differs from the page',
  integer_expected: 'integer expected',
  snippet_not_on_page: 'snippet not on the cited page',
  table_cell_mismatch: 'table cell does not hold the value',
  source_changed: 'source file changed',
  text_not_in_snippet: 'text not in snippet',
};

export const failWords = (r: VerifyFields): string =>
  r.verify_status === 'pass' ? '' : (REASON_WORDS[r.verify_reason] ?? r.verify_reason);
