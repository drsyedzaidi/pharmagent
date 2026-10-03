import { AlertTriangle, OctagonAlert } from 'lucide-react';
import { useEffect, useId, useLayoutEffect, useRef } from 'react';
import { FlagChip } from './badges';
import {
  ZOOM_LABEL, ZOOM_MODES, altText, captionText, imageUnavailableText, mergeNotices, nonAsciiNames, rectStyle,
  scrollToCenter, unionRect, urgentFlagCodes, visibleRects,
} from './imageGeometry';
import { LayoutFallback } from './LayoutFallback';
import type { RelayoutControls } from './useRelayout';
import type { PageImageView } from './usePageImage';
import type { EvidenceSource, ImageNotice, PageImageMeta, RecordRow } from './types';
import { expectsTallImage, outerScrollTop, skeletonStyle } from './viewerLayout';
import './pageimage.css';

/** The extracted value in plain text, large, in the UI face (not monospace, so look-alike glyphs show). It
 *  comes from the text layer, which is exactly what the picture is there to be compared against. */
function ValueCard({ record }: { record: RecordRow }) {
  const shown = record.locator.literal ?? record.value;
  const spelled = nonAsciiNames(shown);
  return (
    <div className="ev-valuecard">
      <span className="ev-label">Extracted value</span>
      <p className="ev-bigvalue" data-testid="extracted-value">{shown}</p>
      {record.unit && <p className="ev-valuemeta">{record.unit}</p>}
      <p className="ev-valuemeta ev-metafield">{record.field}{record.group_label ? ` · ${record.group_label}` : ''}</p>
      {spelled && <p className="ev-valuemeta">Special characters: {spelled}</p>}
      <p className="ev-valuemeta">From the text layer, not the picture.</p>
    </div>
  );
}

/** Notices and flag chips. They come from the record's own flags as well as the image meta, so an urgent
 *  record is told to compare even while the image is loading or when it never arrives. */
function Notices({ notices, record, codes }: { notices: ImageNotice[]; record: RecordRow; codes: string[] }) {
  const flags = record.flags.filter(f => codes.includes(f.code));
  if (notices.length === 0 && flags.length === 0) return null;
  return (
    <div className="ev-notices">
      {notices.map(n => {
        const Icon = n.severity === 'error' ? OctagonAlert : AlertTriangle;
        return (
          <p key={n.code} role="note" className={`ev-notice ${n.severity}`}>
            <Icon size={14} aria-hidden="true" />
            <span><strong>{n.severity === 'error' ? 'Check:' : 'Note:'}</strong> {n.text}</span>
          </p>
        );
      })}
      {flags.length > 0 && <div className="ev-flags">{flags.map((f, i) => <FlagChip key={`${f.code}-${i}`} flag={f} />)}</div>}
    </div>
  );
}

function Picture({ meta, url, mode, alt, onBroken }: {
  meta: PageImageMeta; url: string; mode: PageImageView['mode']; alt: string; onBroken: () => void;
}) {
  const fit = mode === 'fit';
  const dashed = meta.precision === 'line';
  return (
    <div className="ev-imgwrap" style={fit ? { width: '100%', maxWidth: meta.width } : { width: meta.width }}>
      <img src={url} alt={alt} width={meta.width} height={meta.height} className="ev-pageimg" draggable={false} onError={onBroken} />
      {visibleRects(meta).map((r, i) => {
        const s = rectStyle(r, meta.width, meta.height);
        return s ? <span key={i} aria-hidden="true" className={`ev-outline${dashed ? ' dashed' : ''}`} style={s} /> : null;
      })}
    </div>
  );
}

/** The nearest ancestor that really scrolls (the records pane), skipping the viewer itself. */
function scrollParent(el: HTMLElement): HTMLElement | null {
  for (let p = el.parentElement; p; p = p.parentElement) {
    const y = getComputedStyle(p).overflowY;
    if ((y === 'auto' || y === 'scroll') && p.scrollHeight > p.clientHeight) return p;
  }
  return null;
}

/** The cited page as a picture next to the extracted value, with the cited place outlined. The pixels are
 *  untouched; the outline is an aria-hidden overlay. The text-layer view below it keeps working in every state.
 *  Layout: with room (about 540 px) the value card sits beside the picture; narrower, the value shares the
 *  toolbar row, so the picture starts right under the heading. The viewer's height is held from the first
 *  moment (the whole cap for a whole page, the real size once the meta is back), so the picture arriving moves
 *  nothing below it. */
export function PageImage({ view, record, filename, source, rebuild, onRebuilt }: {
  view: PageImageView; record: RecordRow; filename: string;
  /** The source's ledger row: its layout status is known before any image meta arrives. */
  source: EvidenceSource | null;
  rebuild: RelayoutControls;
  /** The layout was (or could not be) rebuilt: the image meta and image are fetched again. */
  onRebuilt: () => void;
}) {
  const { phase, ready, early, failure, shownMode, stale, unavailable, setMode, step, retry, markBroken } = view;
  const shown = record.locator.literal ?? record.value;
  const urgentCodes = urgentFlagCodes(record.flags);
  const urgent = urgentCodes.length > 0;
  const hintId = useId();
  const sectionRef = useRef<HTMLElement>(null);
  const viewerRef = useRef<HTMLDivElement>(null);
  const retryRef = useRef<HTMLButtonElement>(null);
  const retried = useRef(false);
  const meta = ready?.meta ?? null;
  const drawn = ready !== null && !failure?.broken;     // a picture the browser could not draw is never shown
  const layoutOff = source !== null && source.layout_status !== 'ready';

  // The Retry button unmounts when the retry starts: park focus on the section first, then hand it to the
  // viewer (loaded) or back to Retry (failed again), so keyboard focus never falls to <body>.
  const onRetry = () => { retried.current = true; sectionRef.current?.focus(); retry(); };
  useEffect(() => {
    if (!retried.current || phase === 'loading') return;
    retried.current = false;
    const active = document.activeElement;
    if (active !== sectionRef.current && active !== document.body) return;
    (phase === 'ready' ? viewerRef.current : retryRef.current)?.focus();
  }, [phase]);

  // Zoomed in, the picture is wider than the viewer: bring the cited place into view inside the viewer, then
  // bring the viewer's outline into the records pane too (never in Fit, never animated).
  useLayoutEffect(() => {
    const viewer = viewerRef.current, wrap = viewer?.querySelector<HTMLElement>('.ev-imgwrap');
    const box = meta && shownMode !== 'fit' ? unionRect(visibleRects(meta)) : null;
    if (!viewer || !wrap || !meta || !box) return;
    const v = viewer.getBoundingClientRect(), w = wrap.getBoundingClientRect(), k = w.width / meta.width;
    const at = scrollToCenter(
      { x: box.x * k + w.left - v.left + viewer.scrollLeft, y: box.y * k + w.top - v.top + viewer.scrollTop, w: box.w * k, h: box.h * k },
      { w: viewer.clientWidth, h: viewer.clientHeight }, { w: viewer.scrollWidth, h: viewer.scrollHeight });
    viewer.scrollLeft = at.left;
    viewer.scrollTop = at.top;
    const outlines = [...wrap.querySelectorAll<HTMLElement>('.ev-outline')].map(o => o.getBoundingClientRect());
    const pane = scrollParent(viewer);
    if (!pane || outlines.length === 0) return;
    const top = Math.min(...outlines.map(r => r.top)), bottom = Math.max(...outlines.map(r => r.bottom));
    const p = pane.getBoundingClientRect();
    const to = outerScrollTop({ top, height: bottom - top }, { top: p.top, height: p.height }, pane.scrollTop);
    if (to !== null) pane.scrollTo({ top: to, behavior: 'instant' });
  }, [meta, shownMode, drawn]);

  const onKey = (e: React.KeyboardEvent) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === '+' || e.key === '=') { e.preventDefault(); step(1); }
    else if (e.key === '-' || e.key === '_') { e.preventDefault(); step(-1); }
    else if (e.key === '0') { e.preventDefault(); setMode('fit'); }
  };
  const sk = skeletonStyle(early, expectsTallImage(record.flags.map(f => f.code), source?.layout_status), shownMode);
  const serverNotices = layoutOff ? (meta?.notices ?? []).filter(n => n.code !== 'value_not_located') : (meta?.notices ?? []);
  return (
    <div className="ev-compare-host">
    <section className="ev-compare" ref={sectionRef} tabIndex={-1} aria-label="Compare the extracted value with the page image">
      <ValueCard record={record} />
      <div className="ev-imagecol">
        <Notices notices={mergeNotices(urgent, serverNotices)} record={record} codes={meta ? meta.flags : urgentCodes} />
        {source && <LayoutFallback source={source} rebuild={rebuild} onRebuilt={onRebuilt} />}
        <div className="ev-imgtools" role="group" aria-label="Page image zoom">
          {ZOOM_MODES.map(m => (
            <button key={m} type="button" className="btn btn-ghost ev-zoombtn" aria-pressed={shownMode === m} onClick={() => setMode(m)}>
              {ZOOM_LABEL[m]}
            </button>
          ))}
        </div>
        {phase === 'error' && failure && (
          <p className="ev-problem" role="alert">
            <AlertTriangle size={14} aria-hidden="true" />
            <span>Could not load the page image: {/[.!?]$/.test(failure.message) ? failure.message : `${failure.message}.`}{' '}
              {stale && `Still showing the ${meta?.dpi} dpi image.`}{unavailable && imageUnavailableText(urgent)}</span>
            <button type="button" ref={retryRef} className="btn btn-ghost" onClick={onRetry}>Retry</button>
          </p>
        )}
        {phase === 'loading' && !ready && (
          <div className="ev-skeleton" data-testid="image-loading" style={{ minHeight: sk.minHeight, maxHeight: sk.maxHeight }}>
            <div className="ev-skeleton-inner" style={{ aspectRatio: sk.aspectRatio, width: sk.innerWidth, maxWidth: sk.innerMaxWidth }}>
              <p className="ev-empty">Rendering page image…</p>
            </div>
          </div>
        )}
        {ready && drawn && (
          <div className="ev-viewer" ref={viewerRef} tabIndex={0} role="region" aria-label="Page image" aria-describedby={hintId}
            aria-busy={phase === 'loading'} onKeyDown={onKey}>
            <Picture meta={ready.meta} url={ready.url} mode={shownMode} alt={altText(ready.meta, filename, shown)} onBroken={markBroken} />
          </div>
        )}
        {/* The caption and key hint hold their lines while the image renders, so the content below stays put. */}
        {(drawn || (phase === 'loading' && !ready)) && (
          <p className="ev-prov ev-caption" data-testid={ready && phase === 'loading' ? 'image-loading' : undefined}>
            {ready && drawn ? (phase === 'loading' ? 'Rendering page image…' : captionText(ready.meta)) : ''}
          </p>
        )}
        {(drawn || (phase === 'loading' && !ready)) && (
          <p id={hintId} className="ev-prov ev-keyhint">Keys in the picture: + larger, - smaller, 0 fit, arrows scroll.</p>
        )}
      </div>
    </section>
    </div>
  );
}
