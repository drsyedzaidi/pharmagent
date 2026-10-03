import assert from 'node:assert/strict';
import { bulkEligible, layoutLeftOut, layoutUsable, leftOutText } from '../src/evidence/bulkEligibility.ts';

const rec = (o: object = {}) => ({ id: 'r1', source_id: 'src_a', status: 'proposed', verify_status: 'pass', flags: [], ...o }) as never;
const src = (id: string, layout_status: string) => [id, { id, layout_status }] as const;
const sources = new Map([src('src_a', 'ready'), src('src_b', 'missing'), src('src_c', 'text_changed'), src('src_d', 'unreadable')]) as never;

assert.equal(bulkEligible(rec(), sources), true, 'verified, unflagged, ready layout');
for (const id of ['src_b', 'src_c', 'src_d', 'src_unknown']) {
  assert.equal(bulkEligible(rec({ source_id: id }), sources), false, `${id}: no usable layout, compare one by one`);
}
assert.equal(bulkEligible(rec({ flags: [{ code: 'unit_missing' }] }), sources), false);
assert.equal(bulkEligible(rec({ verify_status: 'fail' }), sources), false);
assert.equal(bulkEligible(rec({ status: 'approved' }), sources), false);
assert.equal(layoutUsable(undefined), false);
assert.equal(layoutUsable({ layout_status: 'ready' }), true);

const rows = [rec(), rec({ source_id: 'src_b' }), rec({ source_id: 'src_c' }), rec({ source_id: 'src_b', flags: [{ code: 'x' }] }),
  rec({ source_id: 'src_b', status: 'approved' })];
assert.equal(layoutLeftOut(rows, sources), 2, 'only the verified, unflagged, proposed ones are counted');
assert.equal(leftOutText(0), '');
assert.match(leftOutText(1), /^1 verified record is left out because the page layout of its PDF/);
assert.match(leftOutText(2), /^2 verified records are left out because the page layout of their PDF/);
