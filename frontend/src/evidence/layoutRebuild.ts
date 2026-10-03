/** Wording and counting for the layout rebuild controls (Library list, page-image notice). Pure, no React. */
import { errorMessage } from './format.ts';
import type { EvidenceSource, LayoutStatus, RelayoutOutcome } from './types';

export const LAYOUT_LABEL: Record<LayoutStatus, string> = {
  ready: 'Page images ready', missing: 'Layout missing', text_changed: 'Text changed', unreadable: 'Unreadable',
};

/** Sources a bulk rebuild would try (the server picks the same ones, at most ten per call). */
export const missingSources = <S extends Pick<EvidenceSource, 'layout_status'>>(sources: readonly S[]): S[] =>
  sources.filter(s => s.layout_status === 'missing');

export const rebuildAllLabel = (n: number): string => `Rebuild all (${n})`;

/** Accessible name of one source's rebuild button: unique even when two PDFs share a file name. */
export const rebuildName = (s: Pick<EvidenceSource, 'id' | 'filename'>): string => `Rebuild layout for ${s.filename} (${s.id})`;

/** What a finished rebuild says in one line. The source's own reason (what to do next) is shown beside it, so a
 *  refusal does not repeat it. */
export function outcomeText(outcome: RelayoutOutcome): string {
  if (outcome === 'rebuilt') return 'Layout rebuilt: page images with the cited place outlined are available again.';
  if (outcome === 'unchanged') return 'Layout was already ready.';
  if (outcome === 'text_changed') return 'Not rebuilt: the page text of this PDF now parses differently.';
  return 'Not rebuilt: the PDF cannot be read again.';
}

const OUTCOME_WORD: Record<string, string> = {
  rebuilt: 'rebuilt', unchanged: 'already ready', text_changed: 'text changed', unreadable: 'unreadable', error: 'failed',
};

/** "2 rebuilt, 1 text changed." plus how many are left when a call stopped at its cap. */
export function bulkSummary(counts: Record<string, number>, remaining: number): string {
  const parts = Object.entries(counts).filter(([, n]) => n > 0).map(([k, n]) => `${n} ${OUTCOME_WORD[k] ?? k}`);
  const base = parts.length > 0 ? `${parts.join(', ')}.` : 'Nothing to rebuild.';
  return remaining > 0 ? `${base} ${remaining} more still to rebuild: run it again.` : base;
}

/** The server's message, plus how long to wait when the parser is busy. */
export function rebuildErrorText(e: unknown): string {
  const msg = errorMessage(e);
  const stop = /[.!?]$/.test(msg) ? msg : `${msg}.`;
  const { status, retryAfter } = e as { status?: number; retryAfter?: number | null };
  if (status === 503 || status === 429) return `${stop} ${retryAfter ? `Try again in ${retryAfter} s.` : 'Try again shortly.'}`;
  return stop;
}

/** Per-source state of a rebuild in the UI: running, finished (with the outcome) or failed. */
export interface RebuildItem { phase: 'running' | 'done' | 'error'; outcome?: RelayoutOutcome; text: string }

/** A bulk entry as a row state. `error` is that source's failure, never the whole job's. */
export function itemFromEntry(e: { outcome: RelayoutOutcome | 'error'; reason?: string | null }): RebuildItem {
  if (e.outcome === 'error') return { phase: 'error', text: `Could not rebuild: ${e.reason || 'no reason given.'}` };
  return { phase: 'done', outcome: e.outcome, text: outcomeText(e.outcome) };
}
