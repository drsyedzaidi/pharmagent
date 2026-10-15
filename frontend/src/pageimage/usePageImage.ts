import { useEffect, useState } from 'react';
import { getToken } from '../api';
import { pageImagePath, startPageImageLoad } from './pageImageLoader.ts';
import type { PageImageLocator, PageImageState, RenderOptions } from './types.ts';

const EMPTY: PageImageState = { status: 'empty' };
const LOADING: PageImageState = { status: 'loading' };
const INVALID: PageImageState = { status: 'error', message: 'Invalid page reference' };

/** Load one page image. `null` locator -> `empty`. The request is aborted and
 *  its object URL revoked on unmount or whenever the page/options change. */
export function usePageImage(loc: PageImageLocator | null, opts?: RenderOptions): PageImageState {
  const path = loc ? pageImagePath(loc, opts) : null;
  // The result remembers which path it belongs to, so a stale result is never
  // shown for a new path and no setState is needed to "reset" on change.
  const [result, setResult] = useState<{ path: string; state: PageImageState } | null>(null);

  useEffect(() => {
    if (!path) return;
    const token = getToken();
    const handle = startPageImageLoad(
      path,
      { headers: token ? { Authorization: `Bearer ${token}` } : undefined },
      state => setResult({ path, state }),
    );
    return handle.cancel;
  }, [path]);

  if (!loc) return EMPTY;
  if (!path) return INVALID;
  return result && result.path === path ? result.state : LOADING;
}
