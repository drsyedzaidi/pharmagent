/** The compare gate: Approve stays unavailable until the reviewer says they compared the extracted value
 *  with the page image (or with the PDF itself when no usable picture exists). Pure, no React, no DOM.
 *
 *  This is a UI control, not a server rule: the server, the tools and the review gate are untouched. The
 *  acknowledgement is recorded in the audit trail by composing the decision reason ("Compared with page
 *  image 1a2b3c4d: <reason>"); the server only ever sees a reason string, capped at MAX_REASON.
 *
 *  Fail closed: the only record that needs no tick is one with NO flags at all whose image arrived, was located
 *  and said (explicitly) that it needs no comparison. A flagged record is never calm, whatever its flag. */
import { urgentFlagCodes } from './imageGeometry.ts';
import type { QcFlag } from './types';

/** Server limit on a decision reason (extract_tools.MAX_REASON), counted on the stripped text. */
export const MAX_REASON = 500;
/** Hex characters of the image sha256 carried in the reason. */
export const SHA_TAG = 8;
export const PDF_PREFIX = 'Compared with the PDF itself: ';
export const imagePrefix = (sha256: string): string => `Compared with page image ${sha256.slice(0, SHA_TAG)}: `;

export const LABEL_IMAGE = 'I compared the value with the page image';
export const LABEL_PDF = 'I compared the value with the PDF itself';

/** What the records panel knows about the page image of one record. */
export type ImageFacts =
  /** The cited page is not open: nothing has been checked, whether a comparison is needed is unknown. */
  | { state: 'closed' }
  | { state: 'loading' }
  /** The image failed, or arrived but cannot be drawn: no usable picture. */
  | { state: 'unavailable' }
  | { state: 'ready'; sha256: string; compareUrgent: boolean; located: boolean };

/** The part of the page-image view the gate reads (usePageImage's PageImageView satisfies it). */
export interface ViewLike {
  phase: 'loading' | 'ready' | 'error';
  ready: { meta: { image_sha256: string; compare_urgent: boolean; resolved: boolean; precision: string } } | null;
  failure: { broken?: boolean } | null;
  /** The view's own verdict: a failure left no picture the reviewer can rely on. */
  unavailable?: boolean;
}

/** Image facts of the open record: no view is a closed page; a load in flight is pending; a picture the
 *  reviewer can actually see (including an older zoom after a failed zoom change) is ready; the rest is
 *  unavailable. The meta is untyped JSON at runtime: only an explicit `compare_urgent: false` and an explicit
 *  `resolved: true` count as calm, anything missing or odd is urgent / not located. */
export function factsFromView(view: ViewLike | null): ImageFacts {
  if (view === null) return { state: 'closed' };
  if (view.phase === 'loading') return { state: 'loading' };
  if (view.ready === null || view.failure?.broken === true || view.unavailable === true) return { state: 'unavailable' };
  const { meta } = view.ready;
  return {
    state: 'ready', sha256: meta.image_sha256, compareUrgent: meta.compare_urgent !== false,
    located: meta.resolved === true && meta.precision !== 'page',
  };
}

export interface GateRecord {
  id: string; value: string; source_id: string; flags: readonly QcFlag[]; locator: { page: number; literal: string | null };
}

export type Gate =
  | { mode: 'none'; key: null }
  | { mode: 'pending'; key: null; why: string }
  | { mode: 'image' | 'pdf'; key: string; why: string; label: string; prefix: string };

const KEY_SEP = '\u0000';

/** Which acknowledgement this record needs right now. `none` only for a record with no flags at all, whose
 *  image arrived, was located and carries no urgency; every flag, every unknown (page closed, image failed)
 *  and every unlocated value needs a tick (the picture, or the PDF itself when there is no picture). */
export function gateFor(rec: GateRecord, facts: ImageFacts): Gate {
  if (facts.state === 'loading') return { mode: 'pending', key: null, why: 'The page image is still loading.' };
  const urgent = urgentFlagCodes(rec.flags).length > 0;
  const flagged = rec.flags.length > 0;
  const bound = (mode: 'image' | 'pdf', sha: string) =>
    [rec.id, rec.value, rec.locator.literal ?? '', rec.source_id, rec.locator.page, mode, sha].join(KEY_SEP);
  if (facts.state === 'ready') {
    if (!flagged && !facts.compareUrgent && facts.located) return { mode: 'none', key: null };
    const why = urgent ? 'The text layer may disagree with the page: compare the value with the picture.'
      : flagged ? 'This record carries a QC flag: compare the value with the picture.'
      : !facts.located ? 'The cited value was not located in the picture: find it on the page and compare.'
      : 'This page needs a comparison: compare the value with the picture.';
    return { mode: 'image', key: bound('image', facts.sha256), why, label: LABEL_IMAGE, prefix: imagePrefix(facts.sha256) };
  }
  const why = facts.state === 'unavailable'
    ? 'No page image could be shown for this value: compare it with the PDF itself.'
    : 'Compare it with the PDF itself, or open the cited page and compare it with the picture.';
  return { mode: 'pdf', key: bound('pdf', ''), why, label: LABEL_PDF, prefix: PDF_PREFIX };
}

/** True when the gate asks for nothing, or the stored acknowledgement is the one for exactly this gate. */
export function ticked(gate: Gate, ack: string | null): boolean {
  if (gate.mode === 'none') return true;
  return gate.key !== null && gate.key === ack;
}

/** Longest reviewer reason that still fits once the acknowledgement prefix is added. */
export function maxReasonLength(gate: Gate): number {
  return gate.mode === 'image' || gate.mode === 'pdf' ? MAX_REASON - gate.prefix.length : MAX_REASON;
}

const chars = (n: number) => `${n} character${n === 1 ? '' : 's'}`;

export interface Approval {
  allowed: boolean;
  /** The exact reason to send (prefix + the reviewer's trimmed text); '' while blocked. */
  reason: string;
  /** The same text, for the preview; null while blocked. */
  recorded: string | null;
  /** Why Approve is unavailable, in words; null when allowed. */
  blocked: string | null;
}

const blockedBy = (blocked: string): Approval => ({ allowed: false, reason: '', recorded: null, blocked });
const tickHint = (gate: Gate & { mode: 'image' | 'pdf' }): string => `Tick the box to approve. ${gate.why}`;

/** Decide whether Approve may go ahead for one record and what reason the server will receive. */
export function approval(gate: Gate, isTicked: boolean, reviewerReason: string): Approval {
  if (gate.mode === 'pending') return blockedBy('Approve is unavailable while the page image loads.');
  const text = reviewerReason.trim();
  const lacks: string[] = [];
  if (gate.mode !== 'none' && !isTicked) lacks.push(tickHint(gate));
  if (!text) lacks.push('A reason is required.');
  if (lacks.length > 0) return blockedBy(lacks.join(' '));
  const reason = gate.mode === 'none' ? text : gate.prefix + text;
  if (reason.length > MAX_REASON) {
    return blockedBy(`The recorded text is ${chars(reason.length - MAX_REASON)} over the ${MAX_REASON}-character limit: shorten the reason.`);
  }
  return { allowed: true, reason, recorded: reason, blocked: null };
}

export interface BulkApproval {
  allowed: boolean;
  /** Per record id: the reason to send. Empty while blocked. */
  reasons: Record<string, string>;
  /** True when at least one selected record is covered by the PDF acknowledgement. */
  needsTick: boolean;
  blocked: string | null;
}

/** Bulk approval. A record that needs the picture compared cannot go through it (its tick is bound to the
 *  image on screen); the others share one acknowledgement of a comparison with the PDF itself, composed
 *  into each record's own reason. */
export function bulkApproval(picked: readonly { id: string; gate: Gate }[], isTicked: boolean, reviewerReason: string): BulkApproval {
  const stop = (blocked: string, needsTick = false): BulkApproval => ({ allowed: false, reasons: {}, needsTick, blocked });
  if (picked.length === 0) return stop('Select records first.');
  if (picked.some(p => p.gate.mode === 'image' || p.gate.mode === 'pending')) {
    return stop('A selected record needs its page image compared one by one: open it and approve it alone.');
  }
  const needsTick = picked.some(p => p.gate.mode === 'pdf');
  const text = reviewerReason.trim();
  const lacks: string[] = [];
  if (needsTick && !isTicked) lacks.push(`Tick “${LABEL_PDF.replace('the value', 'every selected value')}” to approve.`);
  if (!text) lacks.push('A reason is required.');
  if (lacks.length > 0) return stop(lacks.join(' '), needsTick);
  const reasons: Record<string, string> = {};
  for (const p of picked) {
    const reason = p.gate.mode === 'pdf' ? p.gate.prefix + text : text;
    if (reason.length > MAX_REASON) {
      return stop(`The recorded text is ${chars(reason.length - MAX_REASON)} over the ${MAX_REASON}-character limit: shorten the reason.`, needsTick);
    }
    reasons[p.id] = reason;
  }
  return { allowed: true, reasons, needsTick, blocked: null };
}
