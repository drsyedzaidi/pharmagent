/** Loads one page image: the meta, then the PNG bound to it by sha, as an object URL. React-free. */
import type { PageImageMeta } from './types';

export interface ImageSource {
  meta: (signal: AbortSignal) => Promise<PageImageMeta>;
  blob: (meta: PageImageMeta, signal: AbortSignal) => Promise<Blob>;
}
export interface LoadedImage { meta: PageImageMeta; url: string }

const isChanged = (e: unknown): boolean => (e as { status?: number } | null)?.status === 409;

/** Meta then image. A 409 (the raster changed between the two calls, e.g. a cache eviction) retries once
 *  through a fresh meta, so the outline and the pixels always belong together. Null when aborted; the
 *  object URL is created only after the last abort check, so an aborted load leaks nothing. The caller owns
 *  the URL and revokes it. */
export async function loadPageImage(src: ImageSource, signal: AbortSignal): Promise<LoadedImage | null> {
  for (let attempt = 0; ; attempt++) {
    try {
      const meta = await src.meta(signal);
      const blob = await src.blob(meta, signal);
      if (signal.aborted) return null;
      return { meta, url: URL.createObjectURL(blob) };
    } catch (e) {
      if (signal.aborted) return null;
      if (!isChanged(e) || attempt >= 1) throw e;
    }
  }
}
