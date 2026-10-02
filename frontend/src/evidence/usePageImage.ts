import { useCallback, useEffect, useState } from 'react';
import { evidencePageImageBlob, evidencePageImageMeta } from './api';
import { imageErrorText, serverZoom, stepMode } from './imageGeometry';
import { loadPageImage, type LoadedImage } from './pageImageLoad';
import { assertPageImage } from './pngCheck';
import type { PageImageMeta, ZoomMode } from './types';

export type ImagePhase = 'loading' | 'ready' | 'error';
/** *broken*: the image arrived but the browser could not draw it (so it must not be shown at all). */
export interface ImageFailure { message: string; status: number | null; broken?: boolean }
interface Store { ready: (LoadedImage & { recordId: string }) | null; failure: (ImageFailure & { key: string }) | null }

export interface PageImageView {
  phase: ImagePhase;
  /** The image on screen: the current record's, kept while a zoom change loads. Never another record's. */
  ready: LoadedImage | null;
  failure: ImageFailure | null;
  mode: ZoomMode;
  /** The mode the picture on screen is really in: after a failed zoom change it is still the old raster. */
  shownMode: ZoomMode;
  /** True when a failure left no picture the reviewer can use (none loaded, or the browser could not draw it). */
  unavailable: boolean;
  /** True when a failure left an image of another zoom than the one asked for on screen. */
  stale: boolean;
  setMode: (m: ZoomMode) => void;
  step: (dir: 1 | -1) => void;
  retry: () => void;
  /** The <img> could not draw the image: show the failure, not a blank box. */
  markBroken: () => void;
}

/** State machine of one record's page image: aborts and ignores stale responses when the record, zoom or
 *  attempt changes, and owns the object URL (revoked when replaced and on unmount). */
export function usePageImage(sessionId: string, recordId: string): PageImageView {
  const [mode, setMode] = useState<ZoomMode>('fit');
  const [attempt, setAttempt] = useState(0);
  const [store, setStore] = useState<Store>({ ready: null, failure: null });
  const zoom = serverZoom(mode);
  const key = `${sessionId}|${recordId}|${zoom}|${attempt}`;

  useEffect(() => {
    const ac = new AbortController();
    void (async () => {
      try {
        const loaded = await loadPageImage({
          meta: signal => evidencePageImageMeta(sessionId, recordId, zoom, signal),
          blob: async (m: PageImageMeta, signal) => {
            const body = await evidencePageImageBlob(sessionId, recordId, zoom, m.image_sha256, signal);
            await assertPageImage(body);
            return body;
          },
        }, ac.signal);
        if (loaded && !ac.signal.aborted) setStore({ ready: { ...loaded, recordId }, failure: null });
      } catch (e) {
        if (ac.signal.aborted) return;
        const status = (e as { status?: number }).status ?? null;
        const retryAfter = (e as { retryAfter?: number | null }).retryAfter ?? null;
        const raw = e instanceof Error ? e.message : String(e);
        setStore(s => ({ ...s, failure: { key, status, message: imageErrorText(raw, status, retryAfter) } }));
      }
    })();
    return () => ac.abort();
  }, [sessionId, recordId, zoom, attempt, key]);

  const url = store.ready?.url ?? null;
  // Revoke an image URL once it is replaced and when the panel closes; the <img> already moved on by then.
  useEffect(() => () => { if (url) URL.revokeObjectURL(url); }, [url]);

  const own = store.ready && store.ready.recordId === recordId ? store.ready : null;
  const fresh = own && own.meta.zoom === zoom ? own : null;
  const failure = store.failure && store.failure.key === key ? store.failure : null;
  const phase: ImagePhase = failure ? 'error' : fresh ? 'ready' : 'loading';
  const step = useCallback((dir: 1 | -1) => setMode(m => stepMode(m, dir)), []);
  const retry = useCallback(() => setAttempt(a => a + 1), []);
  const markBroken = useCallback(() => setStore(s => ({
    ...s, failure: { key, status: null, broken: true, message: imageErrorText('the page image could not be displayed', null, null) },
  })), [key]);
  const stale = own !== null && failure !== null && own.meta.zoom !== zoom;
  const shownMode: ZoomMode = stale ? (own.meta.zoom === 2 ? 'larger' : 'actual') : mode;
  const unavailable = failure !== null && (own === null || failure.broken === true);
  return { phase, ready: own, failure, mode, shownMode, stale, unavailable, setMode, step, retry, markBroken };
}
