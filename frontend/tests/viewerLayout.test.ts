import assert from 'node:assert/strict';
import {
  VIEWER_CAP, VIEWER_CHROME, expectsTallImage, outerScrollTop, skeletonStyle,
} from '../src/evidence/viewerLayout.ts';

assert.equal(VIEWER_CAP, 520);
assert.equal(VIEWER_CHROME, 18, '1px border + 8px padding on each side');

// skeleton: as tall as the viewer will be, capped
assert.deepEqual(skeletonStyle(null, true, 'fit'), { maxHeight: '520px', minHeight: '520px' }, 'unknown + tall expected: the cap');
assert.deepEqual(skeletonStyle(null, false, 'fit'), { minHeight: '170px' }, 'unknown + short expected: the old placeholder');
const m = { width: 1224, height: 1584 };
const s = skeletonStyle(m as never, false, 'fit');
assert.equal(s.maxHeight, '520px');
assert.equal(s.aspectRatio, '1224 / 1584');
assert.equal(s.innerMaxWidth, '1224px');
assert.equal(s.innerWidth, '100%', 'fit scales to the viewer');
const a = skeletonStyle(m as never, false, 'actual');
assert.equal(a.innerWidth, '1224px', 'actual size keeps natural pixels');
assert.equal(a.innerMaxWidth, undefined);
// a short crop reserves less than the cap
const c = skeletonStyle({ width: 600, height: 90 } as never, true, 'fit');
assert.equal(c.aspectRatio, '600 / 90');
// degenerate meta falls back to the placeholder
assert.deepEqual(skeletonStyle({ width: 0, height: 0 } as never, false, 'fit'), { minHeight: '170px' });

assert.equal(expectsTallImage(['hidden_text'], 'ready'), true, 'unjudged page: whole page');
assert.equal(expectsTallImage([], 'missing'), true, 'layout missing means a whole page');
assert.equal(expectsTallImage([], 'text_changed'), true);
assert.equal(expectsTallImage([], 'unreadable'), true);
assert.equal(expectsTallImage([], 'ready'), false);
assert.equal(expectsTallImage(['glyph_text_mismatch'], 'ready'), false, 'a glyph flag still gets a crop around the value');
assert.equal(expectsTallImage([], undefined), false);

// outer pane scroll: null when the box is visible, else centre it with a margin
const pane = { top: 100, height: 600 };
assert.equal(outerScrollTop({ top: 300, height: 40 }, pane, 50), null, 'inside: no scroll');
assert.equal(outerScrollTop({ top: 90, height: 40 }, pane, 50), 0, 'above: centre it, never below 0');
assert.equal(outerScrollTop({ top: 90, height: 40 }, pane, 500), 500 + (90 + 20 - (100 + 300)), 'above: centre it');
assert.equal(outerScrollTop({ top: 900, height: 40 }, pane, 0), 0 + (900 + 20 - (100 + 300)), 'below: centre it');
assert.equal(outerScrollTop({ top: 690, height: 40 }, pane, 0), 0 + (690 + 20 - 400), 'partly below the edge: centre it');
assert.equal(outerScrollTop({ top: 110, height: 30 }, pane, 0), null, 'inside the margin band counts as visible only when fully inside');
assert.equal(outerScrollTop({ top: NaN, height: 40 }, pane, 0), null, 'garbage never scrolls');
// a box taller than the pane aligns its top just below the pane top
assert.equal(outerScrollTop({ top: 400, height: 900 }, pane, 10), 10 + (400 - (100 + 24)));
console.log('viewerlayout ok');
