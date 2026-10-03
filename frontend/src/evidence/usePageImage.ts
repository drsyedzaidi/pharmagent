import { useCallback, useEffect, useState } from 'react';
import { evidencePageImageBlob, evidencePageImageMeta } from './api';
import { defaultZoomMode, imageErrorText, serverZoom, stepMode } from './imageGeometry';
import { isFreshImage, loadPageImage, type LoadedImage } from './pageImageLoad';
import { assertPageImage } from './pngCheck';
import type { PageImageMeta, ZoomMode } from './types';

export type ImagePhase = 'loading' | 'ready' | 'error';
/** *broken*: the image arrived but the browser could not draw it (so it must not be shown at all). */
export interface ImageFailure { message: string; status: number | null; broken?: boolean }
interface Store {
  ready: (LoadedImage & { recordId: string; attempt: number }) | null;
  failure: (ImageFailure & { key: string }) | null;
  early: { key: string; meta: PageImageMeta } | null;
}

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
  /** The meta of the image being fetched, known before its PNG arrives: reserves the viewer's height. */
  early: PageImageMeta | null;
  setMode: (m: ZoomMode) => void;
  step: (dir: 1 | -1) => void;
  retry: () => void;
  /** Fetch the meta and the image again (after the source's layout was rebuilt). */
  reload: () => void;
  /** The <img> could not draw the image: show the failure, not a blank box. */
  markBroken: () => void;
}

/** State machine of one record's page image: aborts and ignores stale responses when the record, zoom or
 *  attempt changes, and owns the object URL (revoked when replaced and on unmount). With no record id
 *  (the cited page is closed) it is idle and fetches nothing. The zoom mode belongs to the record: another
 *  record starts in Fit (Actual size for a whole page with no located value) until the reviewer picks one.
 *  A reload (retry, a layout rebuild) makes the picture on screen stale at once: the phase is `loading` until
 *  the new meta and image are in, so nothing trusts the meta from before. */
export function usePageImage(sessionId: string, recordId: string | null): PageImageView {
  const [modeState, setModeState] = useState<{ id: string | null; mode: ZoomMode | null }>({ id: recordId, mode: null });
  const chosen: ZoomMode | null = modeState.id === recordId ? modeState.mode : null;
  const [attempt, setAttempt] = useState(0);
  const [store, setStore] = useState<Store>({ ready: null, failure: null, early: null });
  // Another record (or none: the page closed) starts clean: no image, failure or meta of the last one stays
  // behind, and its object URL is revoked by the effect below once nothing shows it.
  const [seenId, setSeenId] = useState(recordId);
  if (seenId !== recordId) {
    setSeenId(recordId);
    setStore({ ready: null, failure: null, early: null });
  }
  const zoom = serverZoom(chosen ?? 'fit');                 // Fit and Actual size share one raster
  const key = `${sessionId}|${recordId}|${zoom}|${attempt}`;
  const earlyKey = `${sessionId}|${recordId}|${zoom}`;

  useEffect(() => {
    if (recordId === null) return;
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
        }, ac.signal, meta => setStore(s => ({ ...s, early: { key: earlyKey, meta } })));
        if (loaded && !ac.signal.aborted) setStore(s => ({ ...s, ready: { ...loaded, recordId, attempt }, failure: null }));
      } catch (e) {
        if (ac.signal.aborted) return;
        const status = (e as { status?: number }).status ?? null;
        const retryAfter = (e as { retryAfter?: number | null }).retryAfter ?? null;
        const raw = e instanceof Error ? e.message : String(e);
        setStore(s => ({ ...s, failure: { key, status, message: imageErrorText(raw, status, retryAfter) } }));
      }
    })();
    return () => ac.abort();
  }, [sessionId, recordId, zoom, attempt, key, earlyKey]);

  const url = store.ready?.url ?? null;
  // Revoke an image URL once it is replaced and when the panel closes; the <img> already moved on by then.
  useEffect(() => () => { if (url) URL.revokeObjectURL(url); }, [url]);

  const own = store.ready && store.ready.recordId === recordId ? store.ready : null;
  const fresh = own && isFreshImage(own, zoom, attempt) ? own : null;
  const failure = store.failure && store.failure.key === key ? store.failure : null;
  const phase: ImagePhase = failure ? 'error' : fresh ? 'ready' : 'loading';
  const early = !fresh && store.early && store.early.key === earlyKey ? store.early.meta : null;
  const mode: ZoomMode = chosen ?? defaultZoomMode(own?.meta ?? early);
  const setMode = useCallback((m: ZoomMode) => setModeState({ id: recordId, mode: m }), [recordId]);
  const step = useCallback((dir: 1 | -1) => setModeState({ id: recordId, mode: stepMode(mode, dir) }), [recordId, mode]);
  const retry = useCallback(() => setAttempt(a => a + 1), []);
  const markBroken = useCallback(() => setStore(s => ({
    ...s, failure: { key, status: null, broken: true, message: imageErrorText('the page image could not be displayed', null, null) },
  })), [key]);
  const stale = own !== null && failure !== null && own.meta.zoom !== zoom;
  const shownMode: ZoomMode = stale ? (own.meta.zoom === 2 ? 'larger' : 'actual') : mode;
  // A failed zoom change keeps the picture compared so far; a failed reload does not (that picture predates it).
  const unavailable = failure !== null && (own === null || failure.broken === true || own.attempt !== attempt);
  return { phase, ready: own, failure, mode, shownMode, stale, unavailable, early, setMode, step, retry, reload: retry, markBroken };
}
