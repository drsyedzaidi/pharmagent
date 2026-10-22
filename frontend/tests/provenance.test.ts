// Quarantine banner and explicit 423 handling (provenance branch). Pure logic, run with `node --test`.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { describeApiError, provenanceBanner } from '../src/provenance.ts';

test('a mismatched session shows a quarantine error banner', () => {
  const b = provenanceBanner('mismatch', 'hash_only');
  assert.ok(b);
  assert.equal(b.level, 'error');
  assert.match(b.title, /quarantined/i);
  assert.match(b.detail, /read-only/);
  assert.match(b.detail, /audit/i);
});

test('a legacy session shows a legacy_unverified warning banner', () => {
  const b = provenanceBanner('legacy_unverified', 'hash_only');
  assert.ok(b);
  assert.equal(b.level, 'warning');
  assert.match(b.title + b.detail, /legacy_unverified/);
});

test('a verified or unknown session shows no banner', () => {
  assert.equal(provenanceBanner('verified', 'hash_only'), null);
  assert.equal(provenanceBanner(null, null), null);
  assert.equal(provenanceBanner(undefined, undefined), null);
});

test('HTTP 423 becomes an explicit quarantine message', () => {
  const body = JSON.stringify({ success: false, error: { code: 'provenance_mismatch', message: 'x', request_id: 'r1' } });
  const msg = describeApiError(423, body);
  assert.match(msg, /quarantined/i);
  assert.match(msg, /read-only/);
  assert.match(msg, /r1/);
});

test('HTTP 423 with an unreadable body is still explained', () => {
  assert.match(describeApiError(423, 'not json'), /quarantined/i);
});

test('other errors keep the server message, or fall back to the status', () => {
  const body = JSON.stringify({ success: false, error: { code: 'bad', message: 'unsupported file type', request_id: 'r2' } });
  assert.equal(describeApiError(400, body), 'unsupported file type');
  assert.equal(describeApiError(500, 'plain text'), 'plain text');
  assert.equal(describeApiError(404, ''), 'HTTP 404');
});
