import assert from 'node:assert/strict';
import {
  LAYOUT_LABEL, itemFromEntry, bulkSummary, missingSources, outcomeText, rebuildErrorText, rebuildName, rebuildAllLabel,
} from '../src/evidence/layoutRebuild.ts';

assert.deepEqual(LAYOUT_LABEL, {
  ready: 'Page images ready', missing: 'Layout missing', text_changed: 'Text changed', unreadable: 'Unreadable',
});
const src = (id: string, layout_status: string, filename = `${id}.pdf`) => ({ id, filename, layout_status }) as never;
assert.deepEqual(missingSources([src('a', 'ready'), src('b', 'missing'), src('c', 'text_changed'), src('d', 'missing')]).map(s => s.id), ['b', 'd']);
assert.equal(rebuildAllLabel(2), 'Rebuild all (2)');

// unique accessible names even when two sources share a filename
const n1 = rebuildName({ id: 'src_aaaa1111', filename: 'paper.pdf' } as never);
const n2 = rebuildName({ id: 'src_bbbb2222', filename: 'paper.pdf' } as never);
assert.notEqual(n1, n2);
assert.ok(n1.startsWith('Rebuild layout for paper.pdf'));

// outcome wording: starts with the state; the source's own reason carries what to do next
assert.match(outcomeText('rebuilt'), /^Layout rebuilt/);
assert.match(outcomeText('unchanged'), /already ready/);
assert.match(outcomeText('text_changed'), /^Not rebuilt/);
assert.match(outcomeText('text_changed'), /parses differently/);
assert.match(outcomeText('unreadable'), /^Not rebuilt/);
assert.match(outcomeText('unreadable'), /cannot be read/);

// bulk summary: counts in words, remaining mentioned, errors included
assert.equal(bulkSummary({ rebuilt: 2, text_changed: 1 }, 0), '2 rebuilt, 1 text changed.');
assert.equal(bulkSummary({ rebuilt: 1, error: 2 }, 3), '1 rebuilt, 2 failed. 3 more still to rebuild: run it again.');
assert.equal(bulkSummary({}, 0), 'Nothing to rebuild.');

// error wording: server message, and how long to wait on a busy parser
const busy = Object.assign(new Error(JSON.stringify({ error: { message: 'parser busy' } })), { status: 503, retryAfter: 7 });
assert.match(rebuildErrorText(busy), /parser busy/);
assert.match(rebuildErrorText(busy), /Try again in 7 s/);
assert.match(rebuildErrorText(new Error('boom')), /boom/);

// one entry of a bulk result becomes the row's state
assert.deepEqual(itemFromEntry({ source_id: 'a', outcome: 'rebuilt' }), { phase: 'done', outcome: 'rebuilt', text: outcomeText('rebuilt') });
assert.deepEqual(itemFromEntry({ source_id: 'a', outcome: 'text_changed', reason: 'why' }),
  { phase: 'done', outcome: 'text_changed', text: outcomeText('text_changed') });
assert.deepEqual(itemFromEntry({ source_id: 'a', outcome: 'error', reason: 'parser busy' }),
  { phase: 'error', text: 'Could not rebuild: parser busy' });
assert.deepEqual(itemFromEntry({ source_id: 'a', outcome: 'error' }), { phase: 'error', text: 'Could not rebuild: no reason given.' });
console.log('rebuild ok');

// The refusal sentence is drawn once: the notice already says it in full
import { showOutcomeNote } from '../src/evidence/layoutRebuild.ts';
const done = (outcome: string) => ({ phase: 'done', outcome, text: 't' }) as never;
assert.equal(showOutcomeNote('text_changed', done('text_changed')), false, 'the notice already explains it');
assert.equal(showOutcomeNote('unreadable', done('unreadable')), false);
assert.equal(showOutcomeNote('missing', done('text_changed')), true, 'status not refreshed yet: keep the result');
assert.equal(showOutcomeNote('ready', done('rebuilt')), true);
assert.equal(showOutcomeNote('missing', { phase: 'running', text: 'x' }), true);
assert.equal(showOutcomeNote('text_changed', { phase: 'error', text: 'x' }), true, 'a failed call is always shown');
assert.equal(showOutcomeNote('missing', undefined), false);
