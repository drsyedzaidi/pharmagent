/** Pure helpers for RecordContext: where a record's snippet sits in its page text. */
import type { Locator, PageText, Span } from './types';

/** `start`: code-point offset of `hit` in the text that was searched (the span the verifier stored, or where the
 *  whitespace-insensitive search found the snippet). */
export interface Highlight { before: string; hit: string; after: string; approx: boolean; start: number }

/** The text a locator's span indexes (backend verify.py): a text locator with a column
 *  set refers to that column's text, every other locator to the page text. */
export function contextText(page: PageText, loc: Locator): { text: string; column: number | null } {
  if (loc.kind === 'text' && loc.column !== null && page.columns && typeof page.columns[loc.column] === 'string') {
    return { text: page.columns[loc.column], column: loc.column };
  }
  return { text: page.text, column: null };
}

const split = (cps: string[], s: number, e: number, approx: boolean): Highlight => ({
  before: cps.slice(0, s).join(''), hit: cps.slice(s, e).join(''), after: cps.slice(e).join(''), approx, start: s,
});

/** Where the verified literal sits inside a marked text: `[before, literal, after]`, or null when it cannot be
 *  placed. `token` is the verifier's [start, end) of the literal in the text `hit` was cut from (Python code points,
 *  like every span) and `start` the code-point offset of `hit` in that text (0 for a table cell). The literal is
 *  marked exactly there and nowhere else: `5` in `Dose 15 mg given to 5 patients` is the second 5, never the one
 *  inside `15`. A token that does not hold the literal marks nothing. Without a token (a record verified before
 *  tokens were stored) the first occurrence of the literal is marked. */
export function markLiteral(hit: string, literal: string | null, token: Span | null | undefined,
  start: number): [string, string, string] | null {
  if (!literal) return null;
  const cps = Array.from(hit);
  if (token) {
    const s = token[0] - start, e = token[1] - start;
    if (!Number.isInteger(s) || !Number.isInteger(e) || s < 0 || s >= e || e > cps.length || cps.slice(s, e).join('') !== literal) return null;
    return [cps.slice(0, s).join(''), literal, cps.slice(e).join('')];
  }
  const at = hit.indexOf(literal);
  return at < 0 ? null : [hit.slice(0, at), literal, hit.slice(at + literal.length)];
}

/** Whitespace-insensitive search of `snippet` in `cps`, as code-point [start, end). */
function looseFind(cps: string[], snippet: string): Span | null {
  const keep: number[] = [];
  const hay: string[] = [];
  cps.forEach((c, i) => { if (!/\s/u.test(c)) { keep.push(i); hay.push(c); } });
  const needle = Array.from(snippet).filter(c => !/\s/u.test(c));
  if (needle.length === 0) return null;
  const hayStr = hay.join('');
  const at = hayStr.indexOf(needle.join(''));
  if (at < 0) return null;
  const k = Array.from(hayStr.slice(0, at)).length;
  return [keep[k], keep[k + needle.length - 1] + 1];
}

/** New scrollTop for `box` that puts the top of `mark` a third of the way down the box. Uses the
 *  two elements' viewport rectangles, so it is right whatever the mark's offsetParent is. */
export function markScrollTop(scrollTop: number, boxTop: number, markTop: number, clientHeight: number): number {
  return Math.max(0, scrollTop + (markTop - boxTop) - clientHeight / 3);
}

/** Split `text` around the cited snippet. The server's span is in Python code points, while
 *  JS strings index UTF-16 units, so slicing goes through Array.from (code points). A null
 *  or out-of-range span falls back to a client-side whitespace-insensitive match, marked
 *  approx so the UI says it is not the verifier's location. Null when nothing matches. */
export function highlight(text: string, span: Span | null, snippet: string): Highlight | null {
  const cps = Array.from(text);
  if (span) {
    const [s, e] = span;
    if (Number.isInteger(s) && Number.isInteger(e) && s >= 0 && s < e && e <= cps.length) return split(cps, s, e, false);
  }
  const loose = looseFind(cps, snippet);
  return loose ? split(cps, loose[0], loose[1], true) : null;
}
