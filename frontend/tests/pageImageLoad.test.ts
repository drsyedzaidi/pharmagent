import assert from 'node:assert/strict';
import { loadPageImage } from '../src/evidence/pageImageLoad.ts';

(globalThis.URL as { createObjectURL: (b: Blob) => string }).createObjectURL = () => 'blob:t';
const meta = { image_sha256: 'x'.repeat(64), width: 10, height: 20 } as never;
const order: string[] = [];
const src = { meta: async () => { order.push('meta'); return meta; }, blob: async () => { order.push('blob'); return new Blob(['png']); } };

await loadPageImage(src, new AbortController().signal, m => order.push(`onMeta:${(m as { height: number }).height}`));
assert.deepEqual(order, ['meta', 'onMeta:20', 'blob'], 'onMeta fires between the meta and the PNG');

// aborted while the meta was in flight: not reported
const ac = new AbortController();
const slow = { meta: async () => { ac.abort(); return meta; }, blob: async () => new Blob(['x']) };
const seen: unknown[] = [];
const r = await loadPageImage(slow, ac.signal, m => seen.push(m));
assert.equal(r, null);
assert.deepEqual(seen, []);

// a 409 retry reports the fresh meta again (the older one is superseded)
const conflict = Object.assign(new Error('changed'), { status: 409 });
let calls = 0;
const flaky = { meta: async () => meta, blob: async () => { if (++calls === 1) throw conflict; return new Blob(['p']); } };
const metas: unknown[] = [];
await loadPageImage(flaky, new AbortController().signal, m => metas.push(m));
assert.equal(metas.length, 2);
console.log('imageload ok');

// A reload makes the picture on screen stale at once (the gate must not trust the old meta while it refetches)
import { isFreshImage } from '../src/evidence/pageImageLoad.ts';
const own = { meta: { zoom: 1 }, attempt: 0 };
assert.equal(isFreshImage(own, 1, 0), true);
assert.equal(isFreshImage(own, 1, 1), false, 'after a reload the old image is stale');
assert.equal(isFreshImage(own, 2, 0), false, 'another zoom is stale');
assert.equal(isFreshImage(null, 1, 0), false);
