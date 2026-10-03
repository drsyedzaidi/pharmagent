/** Pure helpers for the page image (no React, no DOM): outline geometry, zoom modes, alt text, captions,
 *  announcements and error wording. Kept free of imports beyond types so node can run them directly. */
import { errorMessage } from './format.ts';
import type { ImageNotice, ImageRect, PageImageMeta, QcFlag, ZoomMode } from './types';

/** The most outline rectangles one image carries (the server caps the same number). */
export const MAX_RECTS = 12;
export const ZOOM_MODES: readonly ZoomMode[] = ['fit', 'actual', 'larger'];
export const ZOOM_LABEL: Record<ZoomMode, string> = { fit: 'Fit', actual: 'Actual size', larger: 'Larger' };

export interface RectStyle { left: string; top: string; width: string; height: string }

const pct = (v: number, of: number): string => `${Math.round((v / of) * 100000) / 1000}%`;

/** A PNG-pixel rectangle as percentages of the image, clamped to it, so the outline scales with the picture.
 *  Null for anything that is not a finite, positive-area rectangle inside a real image size. */
export function rectStyle(r: ImageRect, width: number, height: number): RectStyle | null {
  if (![r.x, r.y, r.w, r.h, width, height].every(Number.isFinite) || width <= 0 || height <= 0) return null;
  const x0 = Math.max(0, Math.min(width, r.x)), y0 = Math.max(0, Math.min(height, r.y));
  const x1 = Math.max(0, Math.min(width, r.x + r.w)), y1 = Math.max(0, Math.min(height, r.y + r.h));
  if (x1 <= x0 || y1 <= y0) return null;
  return { left: pct(x0, width), top: pct(y0, height), width: pct(x1 - x0, width), height: pct(y1 - y0, height) };
}

/** The rectangles to outline: none when the value was not located, never more than MAX_RECTS. */
export function visibleRects(meta: PageImageMeta): ImageRect[] {
  return meta.resolved && meta.precision !== 'page' ? meta.highlight.slice(0, MAX_RECTS) : [];
}

export const serverZoom = (mode: ZoomMode): 1 | 2 => (mode === 'larger' ? 2 : 1);

/** The zoom a record opens in when the reviewer has not chosen one: Fit, except for a whole page with no
 *  located value (a source without layout), where Fit draws the table text a few pixels high and the cited
 *  number cannot be read: Actual size. Both use the same server raster, so it is a client-side choice only. */
export function defaultZoomMode(meta: Pick<PageImageMeta, 'resolved' | 'precision'> | null): ZoomMode {
  return meta !== null && (meta.resolved !== true || meta.precision === 'page') ? 'actual' : 'fit';
}

/** The next zoom mode one step up (+1) or down (-1), saturating at either end. */
export function stepMode(mode: ZoomMode, dir: 1 | -1): ZoomMode {
  const i = ZOOM_MODES.indexOf(mode) + dir;
  return ZOOM_MODES[Math.max(0, Math.min(ZOOM_MODES.length - 1, i))];
}

/** Alt text from facts only (page, file, the extracted text, what is outlined); it never says what the
 *  picture shows, because the reviewer is there to read that for themselves. */
export function altText(meta: PageImageMeta, filename: string, extracted: string): string {
  const head = `Page ${meta.page} of ${filename}`;
  const reads = `The extracted text reads “${extracted}”.`;
  if (!meta.resolved || meta.precision === 'page') {
    return `${head}, whole page. The cited value could not be located; find it yourself. ${reads}`;
  }
  const what = meta.precision === 'line' ? 'The cited row is outlined; the exact value was not located.'
    : meta.precision === 'cell' ? 'The outlined box marks the cited table cell.' : 'The outlined box marks the cited region.';
  return `${head}, region around the cited value. ${reads} ${what}`;
}

/** The line under the image. */
export function captionText(meta: PageImageMeta): string {
  const lead = `p. ${meta.page} of ${meta.pages}, rendered by the server at ${meta.dpi} dpi`;
  if (!meta.resolved || meta.precision === 'page') return `${lead}, whole page, location not found`;
  if (meta.precision === 'line') return `${lead}, region around the cited row, row marked, exact value not located`;
  if (meta.precision === 'cell') return `${lead}, region around the cited value, cell outlined`;
  return `${lead}, region around the cited value`;
}

/** The sentence added to the polite live region when an image arrives. */
export function imageAnnouncement(meta: PageImageMeta, decided = false): string {
  const where = !meta.resolved || meta.precision === 'page' ? 'whole page, cited value not located'
    : meta.precision === 'line' ? 'cited row outlined' : 'cited value outlined';
  return `${meta.compare_urgent ? (decided ? 'This page needs a comparison. ' : 'Compare before approving. ') : ''}`
    + `Page image loaded, ${meta.dpi} dpi, ${where}.`;
}

const NAMES: Record<number, string> = {
  0x2212: 'minus sign', 0x2009: 'thin space', 0x202f: 'narrow no-break space', 0x00a0: 'no-break space',
  0x2013: 'en dash', 0x2012: 'figure dash', 0x2010: 'hyphen', 0x2011: 'non-breaking hyphen', 0x2014: 'em dash',
  0x00b5: 'micro sign', 0x03bc: 'Greek mu', 0x00b7: 'middle dot', 0x2219: 'bullet operator', 0x066b: 'Arabic decimal separator',
};

/** Spells out every distinct non-ASCII character of a value ("U+2212 minus sign"); empty for plain ASCII.
 *  Look-alike glyphs are exactly what the reviewer is comparing, so they are named, not just drawn. */
export function nonAsciiNames(text: string): string {
  const seen: string[] = [];
  for (const ch of text) {
    const cp = ch.codePointAt(0) ?? 0;
    if (cp < 0x80) continue;
    const label = `U+${cp.toString(16).toUpperCase().padStart(4, '0')}${NAMES[cp] ? ` ${NAMES[cp]}` : ''}`;
    if (!seen.includes(label)) seen.push(label);
  }
  return seen.join(', ');
}

/** Error wording: the server's own message, plus how long to wait on 429/503 and guidance on a changed file. */
export function imageErrorText(raw: string, status: number | null, retryAfter: number | null): string {
  const msg = errorMessage(raw);
  const stop = /[.!?]$/.test(msg) ? msg : `${msg}.`;
  if (status === 429 || status === 503) {
    return `${stop} ${retryAfter !== null ? `Try again in ${retryAfter} s.` : 'Try again shortly.'}`;
  }
  if (status === 409 && msg.includes('changed on disk')) return `${msg}. The source file changed or is missing; re-check the upload.`;
  return msg;
}

/** The flag codes that mean the text layer may disagree with the page (the server's URGENT_FLAGS). Known from
 *  the record alone, so the reviewer is told even when the image or its meta never arrives. */
export const URGENT_FLAG_CODES: readonly string[] = [
  'glyph_text_mismatch', 'glyph_ink_mismatch', 'glyph_unverified', 'overlapping_glyphs', 'hidden_text',
];

export function urgentFlagCodes(flags: readonly QcFlag[]): string[] {
  return [...new Set(flags.map(f => f.code).filter(c => URGENT_FLAG_CODES.includes(c)))];
}

const DISAGREE_NOTICE: ImageNotice = {
  code: 'text_layer_may_disagree', severity: 'error',
  text: 'Text layer and page may disagree: compare the image with the extracted value before approving.',
};

/** A notice for a record that is already approved or rejected: the same facts, without telling the reviewer
 *  to compare "before approving" a decision that has been made. */
export const settledText = (text: string): string => text.replace(/ before approving(?=[.:]|$)/i, '');

/** The notices to show: an urgent record always carries the disagree notice (from its own flags), the
 *  server's notices follow, and a code is never shown twice. A decided record gets the settled wording. */
export function mergeNotices(urgent: boolean, server: readonly ImageNotice[], decided = false): ImageNotice[] {
  const all = urgent ? [DISAGREE_NOTICE, ...server] : [...server];
  return all.filter((n, i) => all.findIndex(o => o.code === n.code) === i)
    .map(n => (decided ? { ...n, text: settledText(n.text) } : n));
}

/** What the reviewer is told when no page image can be shown. */
export function imageUnavailableText(urgent: boolean, decided = false): string {
  const text = urgent
    ? 'Image unavailable: compare with the PDF itself before approving. The text layer may disagree with the page.'
    : 'Image unavailable: compare with the PDF itself before approving.';
  return decided ? settledText(text) : text;
}

/** The badge in the cited-page heading: an instruction while a decision is open, a statement once it is made. */
export function compareLabel(decided: boolean, urgent: boolean): string {
  if (!decided) return 'Compare before approving';
  return urgent ? 'Text layer and page may disagree' : 'Page needs a comparison with the PDF';
}

export interface Box { x: number; y: number; w: number; h: number }

/** The smallest rectangle holding every rectangle, null for none. */
export function unionRect(rects: readonly ImageRect[]): Box | null {
  if (rects.length === 0) return null;
  const x0 = Math.min(...rects.map(r => r.x)), y0 = Math.min(...rects.map(r => r.y));
  const x1 = Math.max(...rects.map(r => r.x + r.w)), y1 = Math.max(...rects.map(r => r.y + r.h));
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

/** The scroll offset that centres *box* (content pixels) in a *view* over *content*, clamped to what can
 *  scroll. Zeros for anything not finite, so a bad rectangle never throws the viewer somewhere odd. */
export function scrollToCenter(box: Box, view: { w: number; h: number }, content: { w: number; h: number }):
  { left: number; top: number } {
  if (![box.x, box.y, box.w, box.h, view.w, view.h, content.w, content.h].every(Number.isFinite)) return { left: 0, top: 0 };
  const axis = (at: number, size: number, port: number, total: number) =>
    Math.round(Math.max(0, Math.min(total - port, at + size / 2 - port / 2)));
  return { left: axis(box.x, box.w, view.w, content.w), top: axis(box.y, box.h, view.h, content.h) };
}
