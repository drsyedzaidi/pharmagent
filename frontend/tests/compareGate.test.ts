import assert from 'node:assert/strict';
import {
  MAX_REASON, approval, bulkApproval, factsFromView, gateFor, imagePrefix, maxReasonLength, ticked, PDF_PREFIX,
} from '../src/evidence/compareGate.ts';

const SHA = 'ab12cd34' + 'e'.repeat(56);
const rec = (o: object = {}) => ({
  id: 'rec_1', value: '3,412', source_id: 'src_1', flags: [], locator: { page: 2, literal: '3,412' }, ...o,
}) as never;
const ready = (o: object = {}) => ({ state: 'ready', sha256: SHA, compareUrgent: false, located: true, ...o }) as never;

// --- gate modes
assert.equal(gateFor(rec(), ready()).mode, 'none', 'located, calm image, no flags: no tick');
assert.equal(gateFor(rec(), ready({ compareUrgent: true })).mode, 'image');
assert.equal(gateFor(rec(), ready({ located: false })).mode, 'image', 'unlocated value needs the tick');
assert.equal(gateFor(rec({ flags: [{ code: 'glyph_text_mismatch', severity: 'error', message: '' }] }), ready()).mode, 'image',
  'urgent flag needs the tick even if meta says calm');
// A flagged record is never calm: every flag, not only the five glyph / hidden-text codes, needs the tick.
for (const code of ['unit_missing', 'unit_mismatch', 'composite_cell', 'out_of_range', 'table_low_confidence',
  'suspicious_instruction_text', 'decimal_style_ambiguous', 'scaled_header', 'estimate_outside_ci', 'censored_value',
  'duplicate', 'superseded', 'decided_earlier']) {
  const g = gateFor(rec({ flags: [{ code, severity: 'warn', message: '' }] }), ready()) as { mode: string; key: string; why: string };
  assert.equal(g.mode, 'image', `${code} needs the tick`);
  assert.match(g.why, /QC flag/);
  assert.equal(approval(g as never, false, 'checked it').allowed, false, `${code}: Approve stays blocked until ticked`);
  assert.equal(approval(g as never, true, 'checked it').allowed, true);
}
assert.equal(gateFor(rec({ flags: [{ code: 'unit_missing', severity: 'warn', message: '' }] }), { state: 'closed' }).mode, 'pdf');
assert.equal(gateFor(rec(), { state: 'unavailable' }).mode, 'pdf');
assert.equal(gateFor(rec(), { state: 'closed' }).mode, 'pdf', 'unknown image state fails closed to the PDF');
assert.equal(gateFor(rec(), { state: 'loading' }).mode, 'pending');
const g = gateFor(rec(), ready({ compareUrgent: true })) as { label: string; prefix: string; key: string };
assert.equal(g.label, 'I compared the value with the page image');
assert.equal(g.prefix, 'Compared with page image ab12cd34: ');
assert.equal((gateFor(rec(), { state: 'unavailable' }) as { label: string }).label, 'I compared the value with the PDF itself');
assert.equal(imagePrefix(SHA), 'Compared with page image ab12cd34: ');
assert.equal(PDF_PREFIX, 'Compared with the PDF itself: ');

// --- the tick is bound to everything that identifies what was compared
// --- the meta is untyped JSON: only an explicit compare_urgent:false / resolved:true is calm (fail closed)
const rawMeta = (o: object) => ({ image_sha256: SHA, resolved: true, precision: 'cell', ...o });
const factsOf = (meta: object) => factsFromView({ phase: 'ready', ready: { meta: rawMeta(meta) }, failure: null } as never);
assert.equal((factsOf({}) as { compareUrgent: boolean }).compareUrgent, true, 'meta without compare_urgent is urgent');
assert.equal(gateFor(rec(), factsOf({})).mode, 'image', 'so it needs the tick');
assert.equal((factsOf({ compare_urgent: 'no' }) as { compareUrgent: boolean }).compareUrgent, true);
assert.equal((factsOf({ compare_urgent: false }) as { compareUrgent: boolean }).compareUrgent, false);
assert.equal(gateFor(rec(), factsOf({ compare_urgent: false })).mode, 'none');
assert.equal((factsOf({ compare_urgent: false, resolved: undefined }) as { located: boolean }).located, false, 'resolved must be true');
assert.equal(gateFor(rec(), factsOf({ compare_urgent: false, resolved: 1 })).mode, 'image');

// the pending text says it once
assert.equal(approval(gateFor(rec(), { state: 'loading' }), false, 'x').blocked,
  'Approve is unavailable while the page image loads.');

const key = (r: object, f: object) => (gateFor(rec(r), ready({ compareUrgent: true, ...f })) as { key: string }).key;
const base = key({}, {});
assert.notEqual(base, key({ id: 'rec_2' }, {}), 'record');
assert.notEqual(base, key({ value: '3,413' }, {}), 'value');
assert.notEqual(base, key({ locator: { page: 3, literal: '3,412' } }, {}), 'source page');
assert.notEqual(base, key({ locator: { page: 2, literal: '3.412' } }, {}), 'literal');
assert.notEqual(base, key({ source_id: 'src_2' }, {}), 'source');
assert.notEqual(base, key({}, { sha256: 'f'.repeat(64) }), 'loaded image');
assert.notEqual(base, (gateFor(rec(), { state: 'unavailable' }) as { key: string }).key, 'mode');
assert.equal(base, key({}, {}), 'stable');
assert.equal(ticked(gateFor(rec(), ready({ compareUrgent: true })), base), true);
assert.equal(ticked(gateFor(rec(), ready({ compareUrgent: true })), 'x'), false);
assert.equal(ticked(gateFor(rec(), ready({ compareUrgent: true })), null), false);
assert.equal(ticked(gateFor(rec(), { state: 'loading' }), base), false, 'a pending gate is never ticked');
assert.equal(ticked(gateFor(rec(), ready()), base), true, 'no tick needed counts as satisfied');

// --- approval and the composed reason
const needs = gateFor(rec(), ready({ compareUrgent: true }));
let a = approval(needs, false, 'looks right');
assert.equal(a.allowed, false);
assert.match(a.blocked ?? '', /tick/i);
a = approval(needs, true, '  looks right  ');
assert.equal(a.allowed, true);
assert.equal(a.reason, 'Compared with page image ab12cd34: looks right');
assert.equal(a.recorded, a.reason, 'the preview is exactly what is sent');
a = approval(needs, true, '   ');
assert.equal(a.allowed, false);
assert.match(a.blocked ?? '', /reason/i);
a = approval(gateFor(rec(), { state: 'unavailable' }), true, 'checked p.2');
assert.equal(a.reason, 'Compared with the PDF itself: checked p.2');
a = approval(gateFor(rec(), ready()), false, 'plain');
assert.equal(a.allowed, true);
assert.equal(a.reason, 'plain', 'no comparison needed: no prefix');
a = approval(gateFor(rec(), { state: 'loading' }), true, 'x');
assert.equal(a.allowed, false);
assert.match(a.blocked ?? '', /loads|loading/i);

// --- the 500 limit counts the prefix
assert.equal(MAX_REASON, 500);
assert.equal(maxReasonLength(needs), 500 - 'Compared with page image ab12cd34: '.length);
assert.equal(maxReasonLength(gateFor(rec(), ready())), 500);
assert.equal(maxReasonLength(gateFor(rec(), { state: 'closed' })), 500 - PDF_PREFIX.length);
const room = maxReasonLength(needs);
a = approval(needs, true, 'x'.repeat(room));
assert.equal(a.allowed, true);
assert.equal(a.reason.length, 500);
a = approval(needs, true, 'x'.repeat(room + 1));
assert.equal(a.allowed, false);
assert.match(a.blocked ?? '', /1 character/);
a = approval(needs, true, 'x'.repeat(room + 7));
assert.match(a.blocked ?? '', /7 characters/);

// --- bulk: only records that need no image comparison; one tick covers the PDF ones; each reason composed alone
const none = gateFor(rec({ id: 'a' }), ready());
const pdf = gateFor(rec({ id: 'b' }), { state: 'closed' });
const img = gateFor(rec({ id: 'c' }), ready({ compareUrgent: true }));
let b = bulkApproval([{ id: 'a', gate: none }, { id: 'b', gate: pdf }], false, 'batch ok');
assert.equal(b.allowed, false);
assert.match(b.blocked ?? '', /tick/i);
b = bulkApproval([{ id: 'a', gate: none }, { id: 'b', gate: pdf }], true, 'batch ok');
assert.equal(b.allowed, true);
assert.deepEqual(b.reasons, { a: 'batch ok', b: 'Compared with the PDF itself: batch ok' });
b = bulkApproval([{ id: 'a', gate: none }], false, 'batch ok');
assert.equal(b.allowed, true, 'nothing to compare: no tick');
assert.equal(b.needsTick, false);
b = bulkApproval([{ id: 'c', gate: img }], true, 'batch ok');
assert.equal(b.allowed, false, 'a record that needs the picture cannot go through bulk');
b = bulkApproval([{ id: 'a', gate: none }, { id: 'z', gate: gateFor(rec(), { state: 'loading' }) }], true, 'r');
assert.equal(b.allowed, false);
b = bulkApproval([], true, 'r');
assert.equal(b.allowed, false);
b = bulkApproval([{ id: 'b', gate: pdf }], true, 'x'.repeat(500 - PDF_PREFIX.length + 1));
assert.equal(b.allowed, false);
assert.match(b.blocked ?? '', /1 character/);

// --- facts from the image view of the open record
const m = (o: object = {}) => ({ image_sha256: SHA, compare_urgent: false, resolved: true, precision: 'value', ...o });
assert.deepEqual(factsFromView(null), { state: 'closed' });
assert.deepEqual(factsFromView({ phase: 'loading', ready: null, failure: null }), { state: 'loading' });
assert.deepEqual(factsFromView({ phase: 'loading', ready: { meta: m() }, failure: null }), { state: 'loading' }, 'a zoom change in flight');
assert.deepEqual(factsFromView({ phase: 'ready', ready: { meta: m() }, failure: null }),
  { state: 'ready', sha256: SHA, compareUrgent: false, located: true });
assert.equal((factsFromView({ phase: 'ready', ready: { meta: m({ precision: 'page' }) }, failure: null }) as { located: boolean }).located, false);
assert.equal((factsFromView({ phase: 'ready', ready: { meta: m({ resolved: false }) }, failure: null }) as { located: boolean }).located, false);
assert.deepEqual(factsFromView({ phase: 'error', ready: null, failure: { message: 'x' } }), { state: 'unavailable' });
assert.deepEqual(factsFromView({ phase: 'error', ready: { meta: m() }, failure: { broken: true } }), { state: 'unavailable' }, 'a picture the browser cannot draw');
assert.equal(factsFromView({ phase: 'error', ready: { meta: m() }, failure: { message: 'zoom failed' } }).state, 'ready',
  'a failed zoom change still shows the older picture, which is the one compared');
assert.deepEqual(factsFromView({ phase: 'error', ready: { meta: m() }, failure: { message: 'reload failed' }, unavailable: true } as never),
  { state: 'unavailable' }, 'a failed reload leaves nothing to rely on: the picture predates it');
console.log('comparegate ok');
