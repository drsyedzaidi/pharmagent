import assert from 'node:assert/strict';
import { highlight, markLiteral } from '../src/evidence/highlight.ts';

// the digits of the cited number are marked at the verifier's token, never at an earlier occurrence
const page = 'Dose 15 mg given to 5 patients';
assert.deepEqual(markLiteral(page, '5', [20, 21], 0), ['Dose 15 mg given to ', '5', ' patients'], 'the 5 at offset 20, not the 5 of 15');
assert.deepEqual(markLiteral('mg given to 5 patients', '5', [20, 21], 8), ['mg given to ', '5', ' patients'], 'start shifts the token into the hit');
// a token that does not hold the literal marks nothing (never a wrong digit)
assert.equal(markLiteral(page, '5', [5, 6], 0), null, 'offset 5 holds the 1 of 15, not the cited 5');
assert.equal(markLiteral(page, '7', [20, 21], 0), null, 'another literal at the token');
assert.equal(markLiteral(page, '5', [20, 99], 0), null, 'token past the end');
assert.equal(markLiteral(page, '5', [-3, 1], 0), null, 'negative start');
assert.equal(markLiteral(page, null, [20, 21], 0), null, 'no literal');
// no token (a record verified before tokens were stored): the first occurrence, as before
assert.deepEqual(markLiteral(page, '5', null, 0), ['Dose 1', '5', ' mg given to 5 patients']);
assert.deepEqual(markLiteral(page, '5', undefined, 0), ['Dose 1', '5', ' mg given to 5 patients']);
assert.equal(markLiteral(page, '9', undefined, 0), null);
// code points, not UTF-16 units: an astral character before the number does not shift the mark
assert.deepEqual(markLiteral('𝒳 = 5 ng', '5', [4, 5], 0), ['𝒳 = ', '5', ' ng']);
// a table cell: the token is cell-relative and the hit is the whole cell
assert.deepEqual(markLiteral('15 (5)', '5', [4, 5], 0), ['15 (', '5', ')']);

// highlight() reports where the hit starts, for an exact span and for the loose search
const exact = highlight(page, [8, 30], 'mg given to 5 patients');
assert.ok(exact && exact.start === 8 && !exact.approx);
assert.deepEqual(markLiteral(exact.hit, '5', [20, 21], exact.start), ['mg given to ', '5', ' patients']);
const loose = highlight(page, null, 'mg  given to 5 patients');
assert.ok(loose && loose.approx && loose.start === 8);
