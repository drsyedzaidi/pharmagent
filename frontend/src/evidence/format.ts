/** Pure display + draft helpers for the Evidence workspace (no React). */
import type { AuditEntry, PharmState } from '../types';
import type {
  Criterion, CriterionKind, DecimalSetting, EvidenceMarker, EvidenceSnapshot, FailReason, FieldType, Inclusion,
  ParseStatus, ScreenRow, TemplateField, VerifyFields,
} from './types';

/** Client-side mirror of the backend upload cap (ingest.MAX_BYTES). */
export const MAX_PDF_BYTES = 25 * 1024 * 1024;
const FIELD_NAME_RE = /^[A-Za-z_][A-Za-z0-9_]{0,63}$/;

/** Longest value (code points) shown inline in a records row or an accessible name. */
export const VALUE_CLIP = 20;   // fits .ev-col-value (20ch of the mono font) in evidence.css

/** A value cut to `max` code points (never inside a surrogate pair), ending in '…' when cut. */
export function clipValue(value: string, max = VALUE_CLIP): { text: string; clipped: boolean } {
  const cps = Array.from(value);
  if (cps.length <= max) return { text: value, clipped: false };
  return { text: `${cps.slice(0, max - 1).join('')}…`, clipped: true };
}

interface LabelledRecord {
  id: string; field: string; group_label: string; value: string; unit: string | null;
  locator: { page: number }; record_hash: string; source_id: string;
}

/** Accessible name of a record row: field, group, value, page and publication
 *  ("cmax · 150 mg = 3,412 ng/mL, p. 1, paper_a.pdf"). */
export const recordLabel = (r: Omit<LabelledRecord, 'id' | 'record_hash' | 'source_id'>, source: string): string =>
  `${r.field}${r.group_label ? ` · ${r.group_label}` : ''} = ${clipValue(r.value).text}${r.unit ? ` ${r.unit}` : ''}`
  + `, p. ${r.locator.page}, ${source}`;

/** Accessible names for a list of records, unique across it: a name that two records would share
 *  (same field, group, value, page and paper, e.g. a table and a text extraction) gets the record
 *  hash prefix shown in the Source column (10 hex characters). */
export function recordLabels(rows: LabelledRecord[], names: Map<string, string>): Map<string, string> {
  const base = rows.map(r => [r, recordLabel(r, names.get(r.source_id) ?? r.source_id)] as const);
  const count = new Map<string, number>();
  for (const [, l] of base) count.set(l, (count.get(l) ?? 0) + 1);
  return new Map(base.map(([r, l]) => [r.id, (count.get(l) ?? 0) > 1 ? `${l}, record ${shaPrefix(r.record_hash, 10)}` : l]));
}

const SCREEN_TAG_MIN_HEX = 4;

/** Accessible-name stem of every screening row, unique across the whole ledger and stable when the view
 *  filters: "paper_a.pdf, screen scr_1a2b". Several rows can share one file (one verdict per criteria
 *  version), so the stem adds a prefix of the screen id, which the row shows in full in its provenance
 *  line. The prefix is as short as keeps every screen id apart (at least 4 hex digits). `screens` is the
 *  full list, never the filtered one. */
export function screenStems(screens: { id: string; file: string }[]): Map<string, string> {
  const ids = screens.map(s => s.id);
  const lead = Math.max(0, ...ids.map(id => id.indexOf('_') + 1));
  let n = lead + SCREEN_TAG_MIN_HEX;
  while (new Set(ids.map(id => id.slice(0, n))).size < new Set(ids).size) n += 1;
  return new Map(screens.map(s => [s.id, `${s.file}, screen ${s.id.slice(0, n)}`]));
}

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

/** Comma/semicolon separated list -> trimmed, de-duplicated items (criterion keywords). */
export function splitList(s: string): string[] {
  return [...new Set(s.split(/[,;]/).map(x => x.trim()).filter(Boolean))];
}

/** Template aliases are printed labels, which may contain commas ('Cmax,ss', 'AUC0-24,ss'):
 *  only a semicolon or a line break separates them. */
export function splitAliases(s: string): string[] {
  return [...new Set(s.split(/[;\n]/).map(x => x.trim()).filter(Boolean))];
}
export const ALIAS_SEP = '; ';

// ── pipeline prerequisites ──────────────────────────────────────────────────
export const parsedCount = (snap: EvidenceSnapshot): number =>
  snap.sources.filter(s => s.parse_status === 'parsed').length;

/** The reviewer's inclusion for a screening row (mirrors backend models.effective_verdict):
 *  decided_verdict, else what approving / rejecting an include or exclude implied; null while undecided. */
export function inclusionOf(s: ScreenRow): Inclusion | null {
  if (s.status === 'proposed') return null;
  if (s.decided_verdict) return s.decided_verdict;
  if (s.verdict === 'unsure') return null;
  const agrees = s.status === 'approved';
  return s.verdict === 'include' ? (agrees ? 'include' : 'exclude') : (agrees ? 'exclude' : 'include');
}

/** Sources the reviewer included: extraction reads only these. */
export const includedCount = (snap: EvidenceSnapshot): number =>
  new Set(snap.screens.filter(s => inclusionOf(s) === 'include').map(s => s.source_id)).size;

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
  aliases: f.aliases.join(ALIAS_SEP),
});

export const blankField = (): FieldDraft => ({
  name: '', type: 'number', unit: '', description: '', required: false, min: '', max: '', aliases: '', decimal: 'auto',
});

const num = (s: string): number | null => (s.trim() === '' ? null : Number(s.trim()));

export const fromFieldDraft = (d: FieldDraft): TemplateField => ({
  name: d.name.trim(), type: d.type, unit: d.unit.trim() || null, description: d.description.trim(),
  required: d.required, min: num(d.min), max: num(d.max), aliases: splitAliases(d.aliases), decimal: d.decimal,
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
  partial_number: 'only part of a longer number on the page',
  crosses_columns: 'joins text from two page columns',
};

export const failWords = (r: VerifyFields): string =>
  r.verify_status === 'pass' ? '' : (REASON_WORDS[r.verify_reason] ?? r.verify_reason);
