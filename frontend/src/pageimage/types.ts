/** Axis-aligned box in page-normalized coordinates: origin top-left, x right,
 *  y down, every value in [0, 1], x0 < x1 and y0 < y1 (mirrors the backend
 *  `NormBBox`). */
export interface NormBBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

/** Which page of which stored document. `page` is 0-based. */
export interface PageImageLocator {
  sessionId: string;
  documentId: string;
  page: number;
}

export interface RenderOptions {
  /** Pixels per PDF point, 0.5..3 (server default 1.5). */
  scale?: number;
  /** Ask the server for just this region of the page. */
  crop?: NormBBox;
}

export type PageImageState =
  | { status: 'empty' }
  | { status: 'loading' }
  | { status: 'ready'; src: string }
  | { status: 'error'; message: string };
