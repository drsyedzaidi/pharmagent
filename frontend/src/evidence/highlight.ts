/** Pure helpers for RecordContext: where a record's snippet sits in its page text. */
import type { Locator, PageText, Span } from './types';

export interface Highlight { before: string; hit: string; after: string; approx: boolean }

/** The text a locator's span indexes (backend verify.py): a text locator with a column
 *  set refers to that column's text, every other locator to the page text. */
export function contextText(page: PageText, loc: Locator): { text: string; column: number | null } {
  if (loc.kind === 'text' && loc.column !== null && page.columns && typeof page.columns[loc.column] === 'string') {
    return { text: page.columns[loc.column], column: loc.column };
  }
  return { text: page.text, column: null };
}

const split = (cps: string[], s: number, e: number, approx: boolean): Highlight => ({
  before: cps.slice(0, s).join(''), hit: cps.slice(s, e).join(''), after: cps.slice(e).join(''), approx,
});

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
