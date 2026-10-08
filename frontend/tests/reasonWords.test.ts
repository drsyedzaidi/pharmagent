import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { failDetail, failWords, REASON_WORDS } from '../src/evidence/format.ts';
import type { FailReason, QcFlag } from '../src/evidence/types.ts';

// Every reason code the backend verifier can emit, read from app/evidence/models.py REASON_CODES itself (no
// hand-copied list), so a code added on the server without reviewer text fails here.
const MODELS = readFileSync(new URL('../../backend/app/evidence/models.py', import.meta.url), 'utf8');

/** NAME -> string for every `REASON_X = "x"` and `REASON_A, REASON_B = "a", "b"` line. */
function reasonConstants(src: string): Map<string, string> {
  const out = new Map<string, string>();
  for (const line of src.split('\n')) {
    const m = /^((?:REASON_\w+\s*,\s*)*REASON_\w+)\s*=\s*(".*)$/.exec(line);
    if (!m) continue;
    const names = m[1].split(',').map(s => s.trim());
    const values = [...m[2].replace(/\s+#.*$/, '').matchAll(/"([^"]*)"/g)].map(v => v[1]);
    assert.equal(values.length, names.length, `cannot pair names and strings in: ${line}`);
    names.forEach((n, i) => out.set(n, values[i]));
  }
  return out;
}

function backendReasonCodes(src: string): string[] {
  const tuple = /^REASON_CODES\s*=\s*\(([\s\S]*?)\)/m.exec(src);
  assert.ok(tuple, 'REASON_CODES tuple not found in models.py');
  const consts = reasonConstants(src);
  return tuple[1].split(',').map(s => s.trim()).filter(Boolean).map(name => {
    const code = consts.get(name);
    assert.ok(code !== undefined, `${name} in REASON_CODES has no string assignment`);
    return code;
  });
}

const codes = backendReasonCodes(MODELS);
// the parse really read the tuple (old codes and the r6/r7 numeric-grammar codes are all there)
for (const c of ['', 'empty_snippet', 'crosses_columns', 'unresolved_context', 'unresolved_shape']) {
  assert.ok(codes.includes(c), `backend REASON_CODES parsed without ${JSON.stringify(c)}: ${codes.join(' ')}`);
}

// every failing code has a plain sentence of its own (not the raw code)
const words: Record<string, string> = REASON_WORDS;
for (const code of codes.filter(c => c !== '')) {
  const w = words[code];
  assert.equal(typeof w, 'string', `no reviewer text for backend reason ${code}`);
  assert.ok(w.trim().length > 0 && w !== code && !w.includes('_'), `reviewer text for ${code} is not plain: ${w}`);
}
// ...and the UI names no code the backend never had: only a stored pre-r6 row may still carry non_ascii_digits
const LEGACY = new Set(['non_ascii_digits']);
for (const key of Object.keys(REASON_WORDS)) {
  assert.ok(codes.includes(key) || LEGACY.has(key), `REASON_WORDS names ${key}, which the backend does not emit`);
}

// the numeric-grammar codes point the reviewer at the page image
assert.match(REASON_WORDS.unresolved_context, /unreadable character near the number.*page image/i);
assert.match(REASON_WORDS.unresolved_shape, /form the reader does not accept.*page image/i);

// failDetail: the server-written sentence (it names the offending character and code point) taken from the
// verify_fail flag the backend writes as `<field>: verification failed (<reason>): <detail>`
const detail = "the cited number '12.5' touches U+200B ZERO WIDTH SPACE (right after it), a character outside the "
  + 'allowed number alphabet; the text layer cannot be read as a number here. Compare the value with the page image.';
const flag = (message: string, code: QcFlag['code'] = 'verify_fail'): QcFlag =>
  ({ code, severity: 'error', message, field: 'Cmax', group: '', origin: 'script' });
const failing = (reason: FailReason, flags: QcFlag[]) => ({ verify_status: 'fail' as const, verify_reason: reason, flags });

const ctx = failing('unresolved_context', [flag('Cmax: unit missing', 'unit_missing'),
  flag(`Cmax: verification failed (unresolved_context): ${detail}`)]);
assert.equal(failDetail(ctx), detail);
assert.match(failDetail(ctx), /U\+200B/);
assert.equal(failWords(ctx), REASON_WORDS.unresolved_context);
// a code with no server detail (the backend then writes no ': <detail>' tail)
assert.equal(failDetail(failing('number_mismatch', [flag('Cmax: verification failed (number_mismatch)')])), '');
// the flag must be about this record's own reason; another reason's tail is never shown
assert.equal(failDetail(failing('unresolved_shape', [flag(`Cmax: verification failed (unresolved_context): ${detail}`)])), '');
// a non-verify flag with the same words is never read as the detail
assert.equal(failDetail(failing('unresolved_context',
  [flag(`Cmax: verification failed (unresolved_context): ${detail}`, 'duplicate')])), '');
assert.equal(failDetail(failing('unresolved_context', [])), '');
assert.equal(failDetail({ verify_status: 'pass', verify_reason: '', flags: [flag(`x: verification failed (): y`)] }), '');
