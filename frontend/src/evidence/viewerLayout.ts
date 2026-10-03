/** Layout numbers for the page-image viewer (Records source panel). Pure, no React, no DOM. */
import type { LayoutStatus, PageImageMeta, ZoomMode } from './types';

/** The viewer never grows past this (matches `.ev-viewer { max-height }` in pageimage.css). */
export const VIEWER_CAP = 520;
/** Border (1px) and padding (8px) on both sides of the viewer. */
export const VIEWER_CHROME = 18;
/** Height of the placeholder when nothing is known about the image: about a typical crop around the cited value. */
const SKELETON_DEFAULT = 170;

export interface SkeletonStyle {
  minHeight?: string; maxHeight?: string; aspectRatio?: string; innerWidth?: string; innerMaxWidth?: string;
}

/** The placeholder the viewer's place is held with while the image renders. With the meta it has the
 *  viewer's own shape (the image's aspect ratio, at most its natural width, capped at the viewer's height),
 *  so the picture arriving moves nothing. Before the meta, a record that is probably a whole page (a source
 *  without layout, or an unjudged page) holds the full cap; any other holds a typical crop's height. */
export function skeletonStyle(meta: Pick<PageImageMeta, 'width' | 'height'> | null, expectTall: boolean, mode: ZoomMode):
  SkeletonStyle {
  if (meta && meta.width > 0 && meta.height > 0) {
    const shape = { maxHeight: `${VIEWER_CAP}px`, aspectRatio: `${meta.width} / ${meta.height}` };
    // Fit scales the picture to the viewer's width (never past its natural size); the others show natural pixels.
    return mode === 'fit' ? { ...shape, innerWidth: '100%', innerMaxWidth: `${meta.width}px` } : { ...shape, innerWidth: `${meta.width}px` };
  }
  return expectTall ? { minHeight: `${VIEWER_CAP}px`, maxHeight: `${VIEWER_CAP}px` } : { minHeight: `${SKELETON_DEFAULT}px` };
}

/** A whole page is the image when the source has no stored layout, or the page's ink check could not judge it
 *  (the record carries `hidden_text`: no boxes, the whole page, urgent). Other urgent flags (a glyph that may
 *  disagree) still get a crop around the value, so they do not reserve a page. */
export function expectsTallImage(flagCodes: readonly string[], layoutStatus: LayoutStatus | undefined): boolean {
  return flagCodes.includes('hidden_text') || (layoutStatus !== undefined && layoutStatus !== 'ready');
}

const PANE_MARGIN = 24;

/** The scrollTop that brings a box (viewport coordinates) into a scroll pane (viewport coordinates), or null
 *  when it is already fully inside. A box that fits is centred; one taller than the pane aligns its top just
 *  below the pane's top edge. Null for anything not finite, so a bad measurement never moves the page. */
export function outerScrollTop(box: { top: number; height: number }, pane: { top: number; height: number }, scrollTop: number): number | null {
  if (![box.top, box.height, pane.top, pane.height, scrollTop].every(Number.isFinite)) return null;
  if (box.top >= pane.top && box.top + box.height <= pane.top + pane.height) return null;
  const delta = box.height >= pane.height
    ? box.top - (pane.top + PANE_MARGIN)
    : box.top + box.height / 2 - (pane.top + pane.height / 2);
  return Math.max(0, Math.round(scrollTop + delta));
}
