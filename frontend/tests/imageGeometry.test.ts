import assert from 'node:assert/strict';
import {
  compareLabel, defaultZoomMode, imageAnnouncement, imageUnavailableText, mergeNotices, serverZoom, settledText,
} from '../src/evidence/imageGeometry.ts';
import { pagesText } from '../src/evidence/format.ts';

// The zoom a record opens in: a whole page with no located value opens at Actual size (Fit makes it unreadable)
assert.equal(defaultZoomMode(null), 'fit', 'nothing known yet');
assert.equal(defaultZoomMode({ resolved: true, precision: 'cell' }), 'fit');
assert.equal(defaultZoomMode({ resolved: true, precision: 'value' }), 'fit');
assert.equal(defaultZoomMode({ resolved: true, precision: 'page' }), 'actual');
assert.equal(defaultZoomMode({ resolved: false, precision: 'page' }), 'actual');
assert.equal(defaultZoomMode({ resolved: false, precision: 'cell' } as never), 'actual', 'unresolved is never fit');
assert.equal(serverZoom('fit'), serverZoom('actual'), 'same raster: switching costs no fetch');

// Decided records are not told to compare "before approving"
const disagree = 'Text layer and page may disagree: compare the image with the extracted value before approving.';
assert.equal(settledText(disagree), 'Text layer and page may disagree: compare the image with the extracted value.');
assert.equal(settledText('Nothing to change.'), 'Nothing to change.');
const open = mergeNotices(true, []);
assert.match(open[0].text, /before approving/);
const decided = mergeNotices(true, [{ code: 'x', severity: 'warn', text: 'Compare it before approving.' }], true);
assert.ok(decided.every(n => !/before approving/.test(n.text)), JSON.stringify(decided));
assert.equal(mergeNotices(true, [{ code: 'text_layer_may_disagree', severity: 'error', text: disagree }], true).length, 1, 'no duplicates');
assert.doesNotMatch(imageUnavailableText(true, true), /before approving/);
assert.match(imageUnavailableText(true, false), /before approving/);
assert.equal(compareLabel(false, true), 'Compare before approving');
assert.equal(compareLabel(false, false), 'Compare before approving');
assert.doesNotMatch(compareLabel(true, true), /approving/);
assert.doesNotMatch(compareLabel(true, false), /approving/);
const meta = { compare_urgent: true, dpi: 144, resolved: true, precision: 'cell' } as never;
assert.match(imageAnnouncement(meta), /^Compare before approving\./);
assert.doesNotMatch(imageAnnouncement(meta, true), /approving/);

assert.equal(pagesText(1), '1 page');
assert.equal(pagesText(0), '0 pages');
assert.equal(pagesText(12), '12 pages');
