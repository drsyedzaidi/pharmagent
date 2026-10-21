import type { PharmState } from '../types';

export function fmt(v: number | undefined, d = 2) {
  if (v == null || isNaN(v)) return '–';
  return v.toFixed(d);
}

/** %AUC extrapolation above this is flagged in the NCA table, the λz panels
 *  and the concentration-time chart (same limit the QC agent applies). */
export const EXTRAP_LIMIT = 20;

/** Subject id → %AUC extrapolated, for every subject over EXTRAP_LIMIT.
 *  Empty until NCA has run. */
export function flaggedExtrap(st: PharmState | null | undefined): Record<string, number> {
  const out: Record<string, number> = {};
  for (const r of st?.nca_parameters ?? []) {
    if ((r.pct_AUC_extrap ?? 0) > EXTRAP_LIMIT) out[String(r.subject)] = r.pct_AUC_extrap as number;
  }
  return out;
}

/** Natural order for subject ids, so "2" sorts before "10". Returns a new array. */
export function bySubjectId<T>(items: readonly T[], id: (t: T) => unknown): T[] {
  return [...items].sort((a, b) =>
    String(id(a)).localeCompare(String(id(b)), undefined, { numeric: true, sensitivity: 'base' }));
}

/** Rows shown before "Show all N subjects". */
export const NCA_ROWS_COLLAPSED = 4;
export const NCA_ROWS_THRESHOLD = 6;

/** 'h' after a time-based value only when the dataset's time axis is
 *  labelled in hours — units are never invented. */
export function timeInHours(st: PharmState | null | undefined): boolean {
  return /hour|\(h\)|\bh\b/i.test(st?.spaghetti_data?.x_label ?? '');
}

/** Units as stated by the dataset's own column labels (e.g. "Dose (mg)",
 *  "Conc(mg/L)", "Time(h)") — never invented. Derived NCA units (AUC, CL/F,
 *  Vz/F) are stated only when they follow unambiguously from those labels. */
export interface NcaUnits {
  time: string | null; dose: string | null; conc: string | null;
  auc: string | null; clf: string | null; vz: string | null;
}

/** Unit written inside the raw column label, e.g. 'DV (mg/L)' or 'TIME [h]'. */
export function labelUnit(raw: string): string | null {
  const m = /[([]\s*([^)\]]+?)\s*[)\]]/.exec(raw);
  return m && !/^log$/i.test(m[1]) ? m[1] : null;
}

export function roleColumnUnit(st: PharmState | null | undefined, role: string): string | null {
  const roles = st?.dataset_metadata?.detected_roles;
  if (!roles || typeof roles !== 'object') return null;
  for (const [col, r] of Object.entries(roles as Record<string, unknown>)) {
    const u = r === role ? labelUnit(col) : null;
    if (u) return u;
  }
  return null;
}

export function ncaUnits(st: PharmState | null | undefined): NcaUnits {
  const time = timeInHours(st) ? 'h'
    : labelUnit(st?.spaghetti_data?.x_label ?? '') ?? roleColumnUnit(st, 'TIME');
  const conc = labelUnit(st?.spaghetti_data?.y_label ?? '') ?? roleColumnUnit(st, 'DV');
  const dose = roleColumnUnit(st, 'AMT');
  const slash = conc ? conc.indexOf('/') : -1;
  const [cNum, cDen] = conc && slash > 0 ? [conc.slice(0, slash), conc.slice(slash + 1)] : [conc, null];
  const auc = conc && time ? (cDen ? `${cNum}·${time}/${cDen}` : `${conc}·${time}`) : null;
  const volume = dose && cDen && cNum === dose ? cDen : null;
  return { time, dose, conc, auc, clf: volume && time ? `${volume}/${time}` : null, vz: volume };
}

/** " (unit)" for a column header, or nothing when the unit is unknown. */
export function unitSuffix(u: string | null): string {
  return u ? ` (${u})` : '';
}

/** 3 significant figures, no exponent notation (267.84 → "268"). */
export function sig3(v: number): string {
  return Number(v.toPrecision(3)).toString();
}

export function fmtIIV(cv: number | null): { text: string; unstable: boolean } {
  if (cv == null) return { text: '–', unstable: false };
  if (cv > 300) return { text: '≫300% ⚠', unstable: true };
  return { text: fmt(cv, 1) + '%', unstable: false };
}

export const ci = (lo: number | null | undefined, hi: number | null | undefined, d = 3) =>
  lo == null || hi == null ? '–' : `[${fmt(lo, d)}, ${fmt(hi, d)}]`;

export const SPAG_PALETTE = ['#1F66A6','#1D7A5A','#9A5B12','#4A6FA5','#B23A2E','#3B86C9','#16604A','#C77F2A','#5E7388','#2A8F8F'];

/** The API wraps failures as {"error":{"message"}} (or {"detail"}); show the message, not the JSON. */
export function errorText(e: unknown): string {
  const raw = (e as Error).message ?? String(e);
  const start = raw.indexOf('{');
  if (start >= 0) {
    try {
      const j = JSON.parse(raw.slice(start)) as { error?: { message?: string }; detail?: string; message?: string };
      const m = j.error?.message ?? j.detail ?? j.message;
      if (m) return m;
    } catch { /* not JSON */ }
  }
  return raw;
}

/** /api/health "llm" -> badge text: "mock" | "anthropic:<model>" | "openai:<model>@<url>". */
export function describeLlm(label: string): string {
  if (!label || label === 'mock') return 'Keyless mock model';
  const [provider, rest = ''] = label.split(/:(.+)/);
  if (provider === 'anthropic') return `Claude ${rest}`;
  if (provider === 'openai') {
    const [model, url = ''] = rest.split('@');
    if (url.includes('api.openai.com')) return `ChatGPT ${model}`;
    const host = url.includes('127.0.0.1') || url.includes('localhost') ? 'local' : url.replace(/^https?:\/\//, '').split('/')[0];
    return `${model} (${host})`;
  }
  return label;
}

/** "Nice" tick values across [lo, hi] (1/2/5 × 10^k steps), like d3/flexplot. */
export function niceTicks(lo: number, hi: number, count = 5): number[] {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / count, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * mag;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

/** Axis titles were 9 px in the dim text colour — unreadable once the SVG is
 *  scaled into a chat bubble. Titles use the normal text colour, ticks stay dim. */
export const AXIS_TITLE = { fontSize: 11, fontWeight: 600, fill: 'var(--text)' } as const;

/** Sentence for the chart caption naming the subjects whose terminal phase
 *  is flagged in the NCA table below. */
export function flaggedSentence(ids: string[]): string {
  if (ids.length === 0) return '';
  if (ids.length === 1) return ` · subject ${ids[0]} highlighted because its terminal phase is flagged below`;
  return ` · subjects ${ids.join(', ')} highlighted because their terminal phases are flagged below`;
}

export const TICK_TEXT = { fontFamily: 'var(--mono)', fill: 'var(--text-dim)' } as const;

const _VPC_STRUCTURAL_ROLES = new Set(
  ['ID', 'TIME', 'TAD', 'DV', 'AMT', 'EVID', 'MDV', 'CMT', 'II', 'ADDL', 'DVID', 'CENS', 'ROUTE', 'PD']);

/** Covariate columns eligible for VPC stratification: dataset columns without a
 * structural NONMEM role, plus DOSE (which the backend always accepts). The
 * backend's `available` list is authoritative — this is a best-effort menu. */
export function vpcStrataOptions(
  meta: { columns?: { name: string }[]; detected_roles?: Record<string, string> } | null | undefined,
): string[] {
  const cols = meta?.columns ?? [];
  const roles = meta?.detected_roles ?? {};
  const covs = cols.map(c => c.name)
    .filter(n => !_VPC_STRUCTURAL_ROLES.has((roles[n] ?? '').toUpperCase()));
  return Array.from(new Set(['DOSE', ...covs]));
}
