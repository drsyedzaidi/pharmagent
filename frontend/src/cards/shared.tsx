import { useEffect, useRef, useState } from 'react';
import { Maximize2 } from 'lucide-react';

export function ExtrapBadge() {
  return (
    <span className="badge-warn" title="Over the 20% limit" aria-label="over the 20 percent limit">!</span>
  );
}

export function UncertaintyNotes({ notes }: { notes?: string[] }) {
  if (!notes?.length) return null;
  return (
    <ul style={{ fontSize: 11, color: 'var(--text-dim)', margin: '8px 0 0', paddingLeft: 16 }}>
      {notes.map((n, i) => <li key={i}>{n}</li>)}
    </ul>
  );
}

/** Hover a chart → ⤢ opens it full-screen with wheel zoom and drag-to-pan.
 *  The same JSX is rendered again inside the modal, so nothing is rasterised. */
export function Zoomable({ title, style, children }: { title: string; style?: React.CSSProperties; children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="zoomable" style={style}>
      {children}
      <button className="zoom-btn" type="button" aria-label="Open full screen" title={`Open ${title} full screen — wheel to zoom, drag to pan`}
        onClick={() => setOpen(true)}><Maximize2 size={13} aria-hidden="true" /></button>
      {open && <ZoomModal title={title} onClose={() => setOpen(false)}>{children}</ZoomModal>}
    </div>
  );
}

export function ZoomModal({ title, onClose, children }: { title: string; onClose: () => void; children: React.ReactNode }) {
  const [z, setZ] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const drag = useRef<{ x: number; y: number; px: number; py: number } | null>(null);
  const bodyRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(900);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
      if (e.key === '+' || e.key === '=') setZ(v => Math.min(8, v * 1.2));
      if (e.key === '-') setZ(v => Math.max(0.2, v / 1.2));
      if (e.key === '0') { setZ(1); setPan({ x: 0, y: 0 }); }
    };
    window.addEventListener('keydown', onKey);
    const el = bodyRef.current;
    // fit the whole chart: width-limited OR height-limited via the SVG's viewBox aspect
    const fit = () => {
      if (!el) return;
      const vb = el.querySelector('svg')?.getAttribute('viewBox')?.split(/\s+/).map(Number);
      const aspect = vb && vb.length === 4 && vb[3] > 0 ? vb[2] / vb[3] : 2;
      setWidth(Math.max(320, Math.min(el.clientWidth * 0.92, el.clientHeight * 0.92 * aspect)));
    };
    fit();
    window.addEventListener('resize', fit);
    // non-passive so the page does not scroll behind the modal
    const onWheel = (e: WheelEvent) => { e.preventDefault(); setZ(v => Math.min(8, Math.max(0.2, v * (e.deltaY < 0 ? 1.1 : 1 / 1.1)))); };
    el?.addEventListener('wheel', onWheel, { passive: false });
    return () => { window.removeEventListener('keydown', onKey); window.removeEventListener('resize', fit); el?.removeEventListener('wheel', onWheel); };
  }, [onClose]);

  const down = (e: React.MouseEvent) => { drag.current = { x: e.clientX, y: e.clientY, px: pan.x, py: pan.y }; setDragging(true); };
  const move = (e: React.MouseEvent) => {
    if (!drag.current) return;
    setPan({ x: drag.current.px + (e.clientX - drag.current.x), y: drag.current.py + (e.clientY - drag.current.y) });
  };
  const up = () => { drag.current = null; setDragging(false); };

  return (
    <div className="zoom-overlay" onClick={onClose} role="dialog" aria-label={`${title} — enlarged`}>
      <div className="zoom-modal" onClick={e => e.stopPropagation()}>
        <div className="zoom-toolbar">
          <span className="zoom-title">{title}</span>
          <span style={{ color: 'var(--text-dim)' }}>{Math.round(z * 100)}%</span>
          <button className="btn btn-ghost" onClick={() => setZ(v => Math.max(0.2, v / 1.2))} title="Zoom out (−)">−</button>
          <button className="btn btn-ghost" onClick={() => setZ(v => Math.min(8, v * 1.2))} title="Zoom in (+)">+</button>
          <button className="btn btn-ghost" onClick={() => { setZ(1); setPan({ x: 0, y: 0 }); }} title="Reset (0)">reset</button>
          <button className="btn btn-ghost" onClick={onClose} title="Close (Esc)">close</button>
        </div>
        <div ref={bodyRef} className={`zoom-body${dragging ? ' dragging' : ''}`}
          onMouseDown={down} onMouseMove={move} onMouseUp={up} onMouseLeave={up}>
          <div className="zoom-canvas" style={{ width, transform: `translate(-50%, -50%) translate(${pan.x}px, ${pan.y}px) scale(${z})` }}>
            {children}
          </div>
        </div>
        <div style={{ fontSize: 11, color: 'var(--text-dim)', padding: '4px 12px', borderTop: '1px solid var(--border)' }}>
          wheel / + − to zoom · drag to pan · 0 resets · Esc closes
        </div>
      </div>
    </div>
  );
}

/** Two-thumb time window: from / to sliders over the data's x-range plus a
 *  numeric readout and reset. Values are clamped and ordered by the caller. */
export function XRangeSlider({ min, max, value, label, onChange, onReset }: {
  min: number; max: number; value: [number, number]; label: string;
  onChange: (v: [number, number]) => void; onReset: () => void;
}) {
  const span = max - min || 1;
  const step = span / 200;
  const isFull = value[0] <= min + 1e-9 && value[1] >= max - 1e-9;
  const num = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1));
  const slider = { width: 110, verticalAlign: 'middle', accentColor: 'var(--accent)' } as const;
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--text-dim)', flexWrap: 'wrap' }}
      title="Restrict the time axis to a window">
      <span>{label}</span>
      <input type="range" min={min} max={max} step={step} value={value[0]} style={slider} aria-label="window start"
        onChange={e => onChange([Math.min(Number(e.target.value), value[1]), value[1]])} />
      <span style={{ fontFamily: 'var(--mono)', color: 'var(--text)' }}>{num(value[0])}–{num(value[1])}</span>
      <input type="range" min={min} max={max} step={step} value={value[1]} style={slider} aria-label="window end"
        onChange={e => onChange([value[0], Math.max(Number(e.target.value), value[0])])} />
      {!isFull && <button type="button" className="link-btn" onClick={onReset}>Reset time window</button>}
    </span>
  );
}
