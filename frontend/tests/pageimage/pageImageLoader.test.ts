// Run with: npm test   (node's built-in runner; no extra dependencies)
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
  bboxCssPercent,
  bboxInFrame,
  isValidBBox,
  pageImagePath,
  startPageImageLoad,
} from '../../src/pageimage/pageImageLoader.ts';
import type { PageImageState } from '../../src/pageimage/types.ts';

const DOC = 'a'.repeat(32);
const LOC = { sessionId: 'sess_1a2b3c4d', documentId: DOC, page: 2 };

function pngResponse(init: { status?: number; type?: string; body?: BodyInit } = {}): Response {
  return new Response(init.body ?? new Uint8Array([137, 80, 78, 71]), {
    status: init.status ?? 200,
    headers: { 'content-type': init.type ?? 'image/png' },
  });
}

/** Resolve once the loader emits a terminal (non-loading) state. */
function run(path: string, deps: Parameters<typeof startPageImageLoad>[1]) {
  const states: PageImageState[] = [];
  let done!: () => void;
  const finished = new Promise<void>(r => (done = r));
  const handle = startPageImageLoad(path, deps, s => {
    states.push(s);
    if (s.status !== 'loading') done();
  });
  return { states, finished, handle };
}

describe('pageImagePath', () => {
  it('builds the page url', () => {
    assert.equal(pageImagePath(LOC), `/api/sessions/sess_1a2b3c4d/documents/${DOC}/pages/2.png`);
  });
  it('adds scale and crop query params', () => {
    const p = pageImagePath(LOC, { scale: 2, crop: { x0: 0.1, y0: 0.2, x1: 0.5, y1: 0.9 } });
    assert.ok(p?.endsWith('/pages/2.png?scale=2&x0=0.1&y0=0.2&x1=0.5&y1=0.9'));
  });
  it('rejects anything the server would not accept (no network call for bad refs)', () => {
    assert.equal(pageImagePath({ ...LOC, documentId: '../../etc/passwd' }), null);
    assert.equal(pageImagePath({ ...LOC, documentId: 'A'.repeat(32) }), null);
    assert.equal(pageImagePath({ ...LOC, sessionId: '../x' }), null);
    assert.equal(pageImagePath({ ...LOC, sessionId: 'a/b' }), null);
    assert.equal(pageImagePath({ ...LOC, page: -1 }), null);
    assert.equal(pageImagePath({ ...LOC, page: 1.5 }), null);
    assert.equal(pageImagePath({ ...LOC, page: 2000 }), null);
    assert.equal(pageImagePath(LOC, { scale: 0.1 }), null);
    assert.equal(pageImagePath(LOC, { scale: 3.5 }), null);
    assert.equal(pageImagePath(LOC, { scale: Number.NaN }), null);
    assert.equal(pageImagePath(LOC, { crop: { x0: 0.6, y0: 0, x1: 0.5, y1: 1 } }), null);
    assert.equal(pageImagePath(LOC, { crop: { x0: 0, y0: 0, x1: 1.2, y1: 1 } }), null);
  });
});

describe('bbox geometry', () => {
  it('validates boxes', () => {
    assert.ok(isValidBBox({ x0: 0, y0: 0, x1: 1, y1: 1 }));
    assert.ok(!isValidBBox({ x0: 0.5, y0: 0, x1: 0.5, y1: 1 }));
    assert.ok(!isValidBBox({ x0: Number.NaN, y0: 0, x1: 1, y1: 1 }));
    assert.ok(!isValidBBox(undefined));
  });
  it('is the identity frame when the whole page was fetched', () => {
    const b = { x0: 0.1, y0: 0.2, x1: 0.3, y1: 0.4 };
    assert.deepEqual(bboxInFrame(b, undefined), b);
  });
  it('re-expresses the box in the crop frame', () => {
    const r = bboxInFrame({ x0: 0.25, y0: 0.5, x1: 0.5, y1: 0.75 }, { x0: 0, y0: 0.5, x1: 0.5, y1: 1 });
    assert.deepEqual(r, { x0: 0.5, y0: 0, x1: 1, y1: 0.5 });
  });
  it('clamps to the crop and returns null when outside it', () => {
    const crop = { x0: 0.5, y0: 0.5, x1: 1, y1: 1 };
    assert.deepEqual(bboxInFrame({ x0: 0.25, y0: 0.25, x1: 0.75, y1: 0.75 }, crop), { x0: 0, y0: 0, x1: 0.5, y1: 0.5 });
    assert.equal(bboxInFrame({ x0: 0, y0: 0, x1: 0.4, y1: 0.4 }, crop), null);
    assert.equal(bboxInFrame({ x0: 0.9, y0: 0.2, x1: 0.1, y1: 0.4 }, undefined), null);
  });
  it('formats css percentages', () => {
    assert.deepEqual(bboxCssPercent({ x0: 0.1, y0: 0.25, x1: 0.6, y1: 0.75 }),
      { left: '10%', top: '25%', width: '50%', height: '50%' });
  });
});

describe('startPageImageLoad', () => {
  it('emits loading then ready with an object url, sending headers and an abort signal', async () => {
    let seen: { path: string; init?: RequestInit } | null = null;
    const { states, finished } = run('/p', {
      headers: { Authorization: 'Bearer t' },
      fetchImpl: async (path, init) => { seen = { path: String(path), init }; return pngResponse(); },
      createObjectURL: () => 'blob:fake-1',
      revokeObjectURL: () => {},
    });
    await finished;
    assert.deepEqual(states.map(s => s.status), ['loading', 'ready']);
    assert.deepEqual(states[1], { status: 'ready', src: 'blob:fake-1' });
    assert.equal(seen!.path, '/p');
    assert.deepEqual(seen!.init?.headers, { Authorization: 'Bearer t' });
    assert.ok(seen!.init?.signal instanceof AbortSignal);
  });

  it('surfaces the server error message from the JSON envelope', async () => {
    const body = JSON.stringify({ success: false, error: { code: 'http_404', message: 'page not found', request_id: 'x' } });
    const { states, finished } = run('/p', {
      fetchImpl: async () => pngResponse({ status: 404, type: 'application/json', body }),
    });
    await finished;
    assert.deepEqual(states.at(-1), { status: 'error', message: 'page not found' });
  });

  it('falls back to a generic message for non-JSON errors', async () => {
    const { states, finished } = run('/p', {
      fetchImpl: async () => pngResponse({ status: 502, type: 'text/html', body: '<html>bad gateway</html>' }),
    });
    await finished;
    assert.deepEqual(states.at(-1), { status: 'error', message: 'Could not load page (HTTP 502)' });
  });

  it('refuses a 200 that is not image/png (never builds an object url from html)', async () => {
    let created = 0;
    const { states, finished } = run('/p', {
      fetchImpl: async () => pngResponse({ type: 'text/html', body: '<script>1</script>' }),
      createObjectURL: () => { created++; return 'blob:x'; },
    });
    await finished;
    assert.equal(states.at(-1)?.status, 'error');
    assert.equal(created, 0);
  });

  it('reports network failure as an error state', async () => {
    const { states, finished } = run('/p', { fetchImpl: async () => { throw new TypeError('fetch failed'); } });
    await finished;
    assert.deepEqual(states.at(-1), { status: 'error', message: 'Could not load page (network error)' });
  });

  it('cancel() aborts the in-flight request and emits nothing further', async () => {
    let signal: AbortSignal | undefined;
    const { states, handle } = run('/p', {
      fetchImpl: (_p, init) => new Promise<Response>((_res, rej) => {
        signal = init?.signal ?? undefined;
        signal?.addEventListener('abort', () => rej(new DOMException('aborted', 'AbortError')));
      }),
    });
    await new Promise(r => setTimeout(r, 5));
    handle.cancel();
    await new Promise(r => setTimeout(r, 10));
    assert.equal(signal?.aborted, true);
    assert.deepEqual(states.map(s => s.status), ['loading']);
  });

  it('cancel() after ready revokes the object url exactly once', async () => {
    const revoked: string[] = [];
    const { finished, handle } = run('/p', {
      fetchImpl: async () => pngResponse(),
      createObjectURL: () => 'blob:fake-2',
      revokeObjectURL: u => revoked.push(u),
    });
    await finished;
    handle.cancel();
    handle.cancel();
    assert.deepEqual(revoked, ['blob:fake-2']);
  });

  it('cancel() before the body arrives creates no object url (no leak)', async () => {
    let created = 0;
    let release!: () => void;
    const gate = new Promise<void>(r => (release = r));
    const { states, handle } = run('/p', {
      fetchImpl: async () => { await gate; return pngResponse(); },
      createObjectURL: () => { created++; return 'blob:never'; },
    });
    handle.cancel();
    release();
    await new Promise(r => setTimeout(r, 20));
    assert.equal(created, 0);
    assert.deepEqual(states.map(s => s.status), ['loading']);
  });
});
