// Component tests for PageImage / PageImageView. TSX is compiled by Vite's own
// SSR loader (already a dev dependency), then rendered with react-dom/server.
// Effects do not run in SSR, so the hook's first paint and the pure view are
// covered here; the hook's abort/revoke behaviour is covered in
// pageImageLoader.test.ts (the loader owns it; the hook only wires it up).
import assert from 'node:assert/strict';
import { after, before, describe, it } from 'node:test';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

let server;
let mod;
const html = el => renderToStaticMarkup(el);

before(async () => {
  server = await createServer({ server: { middlewareMode: true }, appType: 'custom', logLevel: 'silent' });
  mod = await server.ssrLoadModule('/src/pageimage/PageImage.tsx');
});
after(async () => { await server?.close(); });

const READY = { status: 'ready', src: 'blob:abc' };

describe('PageImageView states', () => {
  it('empty', () => {
    const out = html(createElement(mod.PageImageView, { state: { status: 'empty' }, alt: 'a' }));
    assert.match(out, /role="status"/);
    assert.match(out, /No page selected/);
  });
  it('loading', () => {
    const out = html(createElement(mod.PageImageView, { state: { status: 'loading' }, alt: 'a' }));
    assert.match(out, /aria-busy="true"/);
    assert.match(out, /Loading page/);
  });
  it('error is announced and its text is escaped', () => {
    const out = html(createElement(mod.PageImageView, {
      state: { status: 'error', message: '<img src=x onerror=alert(1)>' }, alt: 'a',
    }));
    assert.match(out, /role="alert"/);
    assert.ok(!out.includes('<img src=x'));
    assert.match(out, /&lt;img src=x/);
  });
  it('ready renders the image with alt text and no overlay by default', () => {
    const out = html(createElement(mod.PageImageView, { state: READY, alt: 'Page 3 of the protocol' }));
    assert.match(out, /<img[^>]+src="blob:abc"/);
    assert.match(out, /alt="Page 3 of the protocol"/);
    assert.ok(!out.includes('page-image-bbox'));
  });
});

describe('bbox overlay', () => {
  it('is positioned in percentages of the page', () => {
    const out = html(createElement(mod.PageImageView, {
      state: READY, alt: 'a', bbox: { x0: 0.1, y0: 0.25, x1: 0.6, y1: 0.75 },
    }));
    assert.match(out, /data-testid="page-image-bbox"/);
    assert.match(out, /left:10%/);
    assert.match(out, /top:25%/);
    assert.match(out, /width:50%/);
    assert.match(out, /height:50%/);
  });
  it('maps into the crop frame when a cropped image is shown', () => {
    const out = html(createElement(mod.PageImageView, {
      state: READY, alt: 'a',
      crop: { x0: 0, y0: 0.5, x1: 0.5, y1: 1 },
      bbox: { x0: 0.25, y0: 0.5, x1: 0.5, y1: 0.75 },
    }));
    assert.match(out, /left:50%/);
    assert.match(out, /top:0%/);
    assert.match(out, /width:50%/);
    assert.match(out, /height:50%/);
  });
  it('is omitted when the box lies outside the crop or is invalid', () => {
    const outside = html(createElement(mod.PageImageView, {
      state: READY, alt: 'a', crop: { x0: 0.5, y0: 0.5, x1: 1, y1: 1 }, bbox: { x0: 0, y0: 0, x1: 0.2, y1: 0.2 },
    }));
    assert.ok(!outside.includes('page-image-bbox'));
    const invalid = html(createElement(mod.PageImageView, {
      state: READY, alt: 'a', bbox: { x0: 0.8, y0: 0, x1: 0.2, y1: 1 },
    }));
    assert.ok(!invalid.includes('page-image-bbox'));
  });
});

describe('PageImage (hook first paint)', () => {
  it('shows the empty state for a null locator', () => {
    const out = html(createElement(mod.PageImage, { locator: null }));
    assert.match(out, /No page selected/);
  });
  it('shows loading for a valid locator before any response', () => {
    const out = html(createElement(mod.PageImage, {
      locator: { sessionId: 'sess_1a2b3c4d', documentId: 'b'.repeat(32), page: 0 },
    }));
    assert.match(out, /Loading page/);
  });
  it('shows an error (and makes no request) for an invalid locator', () => {
    const out = html(createElement(mod.PageImage, {
      locator: { sessionId: 'sess_1a2b3c4d', documentId: '../../etc/passwd', page: 0 },
    }));
    assert.match(out, /role="alert"/);
    assert.match(out, /Invalid page reference/);
  });
});
