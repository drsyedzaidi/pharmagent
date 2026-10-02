import { AlertTriangle, OctagonAlert } from 'lucide-react';
import { FlagChip } from './badges';
import {
  ZOOM_LABEL, ZOOM_MODES, altText, captionText, nonAsciiNames, rectStyle, visibleRects,
} from './imageGeometry';
import type { PageImageView } from './usePageImage';
import type { PageImageMeta, RecordRow } from './types';
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
      <p className="ev-valuemeta">{record.field}{record.group_label ? ` · ${record.group_label}` : ''}</p>
      {spelled && <p className="ev-valuemeta">Special characters: {spelled}</p>}
      <p className="ev-valuemeta">From the text layer, not the picture.</p>
    </div>
  );
}

function Notices({ meta, record }: { meta: PageImageMeta; record: RecordRow }) {
  const flags = record.flags.filter(f => meta.flags.includes(f.code));
  if (meta.notices.length === 0 && flags.length === 0) return null;
  return (
    <div className="ev-notices">
      {meta.notices.map(n => {
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

function Picture({ meta, url, mode, alt }: { meta: PageImageMeta; url: string; mode: PageImageView['mode']; alt: string }) {
  const fit = mode === 'fit';
  const dashed = meta.precision === 'line';
  return (
    <div className="ev-imgwrap" style={fit ? { width: '100%', maxWidth: meta.width } : { width: meta.width }}>
      <img src={url} alt={alt} width={meta.width} height={meta.height} className="ev-pageimg" draggable={false} />
      {visibleRects(meta).map((r, i) => {
        const s = rectStyle(r, meta.width, meta.height);
        return s ? <span key={i} aria-hidden="true" className={`ev-outline${dashed ? ' dashed' : ''}`} style={s} /> : null;
      })}
    </div>
  );
}

/** The cited page as a picture next to the extracted value, with the cited place outlined. The pixels are
 *  untouched; the outline is an aria-hidden overlay. The text-layer view below it keeps working in every state. */
export function PageImage({ view, record, filename }: { view: PageImageView; record: RecordRow; filename: string }) {
  const { phase, ready, failure, mode, setMode, step, retry } = view;
  const shown = record.locator.literal ?? record.value;
  const onKey = (e: React.KeyboardEvent) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === '+' || e.key === '=') { e.preventDefault(); step(1); }
    else if (e.key === '-' || e.key === '_') { e.preventDefault(); step(-1); }
    else if (e.key === '0') { e.preventDefault(); setMode('fit'); }
  };
  return (
    <div className="ev-compare-host">
    <section className="ev-compare" aria-label="Compare the extracted value with the page image">
      <ValueCard record={record} />
      <div className="ev-imagecol">
        {ready && <Notices meta={ready.meta} record={record} />}
        <div className="ev-imgtools" role="group" aria-label="Page image zoom">
          {ZOOM_MODES.map(m => (
            <button key={m} type="button" className="btn btn-ghost ev-zoombtn" aria-pressed={mode === m} onClick={() => setMode(m)}>
              {ZOOM_LABEL[m]}
            </button>
          ))}
        </div>
        {phase === 'error' && failure && (
          <p className="ev-problem" role="alert">
            <AlertTriangle size={14} aria-hidden="true" />
            <span>Could not load the page image: {failure.message}</span>
            <button type="button" className="btn btn-ghost" onClick={retry}>Retry</button>
          </p>
        )}
        {phase === 'loading' && <p className="ev-empty" data-testid="image-loading">Rendering page image…</p>}
        {ready && (
          <div className="ev-viewer" tabIndex={0} role="group" aria-busy={phase === 'loading'} onKeyDown={onKey}
            aria-label="Page image. Plus and minus change zoom; arrow keys scroll.">
            <Picture meta={ready.meta} url={ready.url} mode={mode} alt={altText(ready.meta, filename, shown)} />
          </div>
        )}
        {ready && <p className="ev-prov ev-caption">{captionText(ready.meta)}</p>}
      </div>
    </section>
    </div>
  );
}
