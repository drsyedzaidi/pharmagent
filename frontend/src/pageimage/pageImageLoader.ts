// Framework-free core of the page-image viewer: URL building, bbox geometry and
// the abortable fetch -> object-URL lifecycle. Kept free of React and of
// ../api so it can be unit-tested directly under node.
import type { NormBBox, PageImageLocator, PageImageState, RenderOptions } from './types.ts';

const API_BASE = '/api';
const SESSION_ID_RE = /^[A-Za-z0-9_-]{1,64}$/;
const DOC_ID_RE = /^[0-9a-f]{32}$/;
export const MIN_SCALE = 0.5;
export const MAX_SCALE = 3;
const MAX_PAGE = 1999;

export function isValidBBox(b: NormBBox | null | undefined): b is NormBBox {
  if (!b) return false;
  const v = [b.x0, b.y0, b.x1, b.y1];
  return v.every(n => Number.isFinite(n) && n >= 0 && n <= 1) && b.x0 < b.x1 && b.y0 < b.y1;
}

/** Request path for one page image, or null when the locator/options are not
 *  something the server would accept (so a bad reference never hits the network). */
export function pageImagePath(loc: PageImageLocator, opts: RenderOptions = {}): string | null {
  if (!SESSION_ID_RE.test(loc.sessionId) || !DOC_ID_RE.test(loc.documentId)) return null;
  if (!Number.isInteger(loc.page) || loc.page < 0 || loc.page > MAX_PAGE) return null;
  const q: string[] = [];
  if (opts.scale !== undefined) {
    if (!Number.isFinite(opts.scale) || opts.scale < MIN_SCALE || opts.scale > MAX_SCALE) return null;
    q.push(`scale=${opts.scale}`);
  }
  if (opts.crop !== undefined) {
    if (!isValidBBox(opts.crop)) return null;
    const c = opts.crop;
    q.push(`x0=${c.x0}`, `y0=${c.y0}`, `x1=${c.x1}`, `y1=${c.y1}`);
  }
  const path = `${API_BASE}/sessions/${loc.sessionId}/documents/${loc.documentId}/pages/${loc.page}.png`;
  return q.length ? `${path}?${q.join('&')}` : path;
}

/** Express a page-normalized `bbox` in the coordinate frame of the image that
 *  was actually fetched. With no `crop` the image is the whole page, so it is
 *  unchanged. Returns null when the box does not overlap the cropped region. */
export function bboxInFrame(bbox: NormBBox | undefined, crop: NormBBox | undefined): NormBBox | null {
  if (!isValidBBox(bbox)) return null;
  if (!crop) return bbox;
  if (!isValidBBox(crop)) return null;
  const x0 = Math.max(bbox.x0, crop.x0);
  const y0 = Math.max(bbox.y0, crop.y0);
  const x1 = Math.min(bbox.x1, crop.x1);
  const y1 = Math.min(bbox.y1, crop.y1);
  if (x0 >= x1 || y0 >= y1) return null;
  const w = crop.x1 - crop.x0;
  const h = crop.y1 - crop.y0;
  return { x0: (x0 - crop.x0) / w, y0: (y0 - crop.y0) / h, x1: (x1 - crop.x0) / w, y1: (y1 - crop.y0) / h };
}

export function bboxCssPercent(b: NormBBox): { left: string; top: string; width: string; height: string } {
  const pct = (n: number) => `${+(n * 100).toFixed(4)}%`;
  return { left: pct(b.x0), top: pct(b.y0), width: pct(b.x1 - b.x0), height: pct(b.y1 - b.y0) };
}

export interface LoadDeps {
  headers?: Record<string, string>;
  fetchImpl?: typeof fetch;
  createObjectURL?: (blob: Blob) => string;
  revokeObjectURL?: (url: string) => void;
}

async function errorMessage(res: Response): Promise<string> {
  try {
    const body = JSON.parse(await res.text());
    const m = body?.error?.message;
    if (typeof m === 'string' && m) return m;
  } catch {
    /* not the JSON error envelope */
  }
  return `Could not load page (HTTP ${res.status})`;
}

/** Start loading a page image. `emit` receives loading -> ready | error. The
 *  returned `cancel()` aborts the request, revokes any object URL it created and
 *  guarantees no further `emit` calls (use it for unmount / param change). */
export function startPageImageLoad(
  path: string,
  deps: LoadDeps,
  emit: (s: PageImageState) => void,
): { cancel: () => void } {
  const doFetch = deps.fetchImpl ?? fetch;
  const create = deps.createObjectURL ?? ((b: Blob) => URL.createObjectURL(b));
  const revoke = deps.revokeObjectURL ?? ((u: string) => URL.revokeObjectURL(u));
  const ctl = new AbortController();
  let cancelled = false;
  let objectUrl: string | null = null;

  const send = (s: PageImageState) => {
    if (!cancelled) emit(s);
  };

  send({ status: 'loading' });
  void (async () => {
    try {
      const res = await doFetch(path, { headers: deps.headers, signal: ctl.signal });
      if (!res.ok) {
        send({ status: 'error', message: await errorMessage(res) });
        return;
      }
      if (!(res.headers.get('content-type') ?? '').toLowerCase().startsWith('image/png')) {
        send({ status: 'error', message: 'Unexpected response from server' });
        return;
      }
      const blob = await res.blob();
      if (cancelled) return;
      objectUrl = create(blob);
      send({ status: 'ready', src: objectUrl });
    } catch (e) {
      if (cancelled || (e instanceof Error && e.name === 'AbortError')) return;
      send({ status: 'error', message: 'Could not load page (network error)' });
    }
  })();

  return {
    cancel() {
      if (cancelled) return;
      cancelled = true;
      ctl.abort();
      if (objectUrl) {
        revoke(objectUrl);
        objectUrl = null;
      }
    },
  };
}
