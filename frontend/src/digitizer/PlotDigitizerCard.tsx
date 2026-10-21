import React, { useState } from 'react';
import { AlertTriangle, CheckCircle, Plus, RefreshCw, Trash2 } from 'lucide-react';
import type {
  AxisCalibrationSpec,
  AxisScale,
  DigitizedPoint,
  DigitizeResult,
  DigitizerViewState,
  ErrorBarType,
} from './digitizeMath';
import {
  applyAddDigitizerPoint,
  applyPointPixelEdit,
  applyRemoveDigitizerPoint,
  buildResultSourceKey,
  computeSvgOverlayGeometry,
  createDigitizerViewState,
  isPointOutOfBounds,
  togglePointVerified,
  validateAxisSpec,
} from './digitizeMath';

const MONO: React.CSSProperties = {
  fontVariantNumeric: 'tabular-nums',
  fontFamily: 'var(--mono)',
};

function fmt(v: number | null | undefined, digits = 2): string {
  if (v == null || !Number.isFinite(v)) return '–';
  return v.toFixed(digits);
}

export interface PlotDigitizerRecomputePayload {
  x_axis: AxisCalibrationSpec;
  y_axis: AxisCalibrationSpec;
  pixel_sigma: number;
  error_type: ErrorBarType;
  strict_bounds: boolean;
  points: {
    id: string;
    u_px: number;
    v_px: number;
    v_low_px: number | null;
    v_high_px: number | null;
    origin: 'manual' | 'contour_proposed';
    verified: boolean;
  }[];
}

export function PlotDigitizerCard({
  r,
  busy,
  onRecompute,
}: {
  r: DigitizeResult;
  busy?: boolean;
  onRecompute?: (payload: PlotDigitizerRecomputePayload) => Promise<void>;
}) {
  const [viewState, setViewState] = useState<DigitizerViewState>(() =>
    createDigitizerViewState(r),
  );
  const [localBusy, setLocalBusy] = useState(false);
  const [errMsg, setErrMsg] = useState<string | null>(null);

  const incomingKey = buildResultSourceKey(r);
  if (viewState.sourceKey !== incomingKey) {
    setViewState(createDigitizerViewState(r));
  }

  const { points, yScale, pixelSigma, strictBounds, isDirty, derivedPk } = viewState;
  const xAx: AxisCalibrationSpec = r.x_axis;
  const yAx: AxisCalibrationSpec = { ...r.y_axis, scale: yScale };

  async function triggerRecompute(
    nextPts: DigitizedPoint[] = points,
    nextScale: AxisScale = yScale,
    nextSigma: number = pixelSigma,
    nextStrict: boolean = strictBounds,
  ) {
    if (!onRecompute) return;
    const candidateY: AxisCalibrationSpec = { ...r.y_axis, scale: nextScale };
    const xErr = validateAxisSpec(xAx, 'X axis');
    const yErr = validateAxisSpec(candidateY, 'Y axis');
    if (xErr || yErr) {
      setErrMsg(xErr ?? yErr);
      return;
    }
    setErrMsg(null);
    setLocalBusy(true);
    try {
      await onRecompute({
        x_axis: xAx,
        y_axis: candidateY,
        pixel_sigma: nextSigma,
        error_type: r.error_type ?? 'se',
        strict_bounds: nextStrict,
        points: nextPts.map(p => ({
          id: p.id,
          u_px: p.u_px,
          v_px: p.v_px,
          v_low_px: p.v_low_px,
          v_high_px: p.v_high_px,
          origin: p.origin,
          verified: p.verified,
        })),
      });
    } catch (e) {
      setErrMsg((e as Error).message);
    } finally {
      setLocalBusy(false);
    }
  }

  const W = 680;
  const H = 240;
  const PAD_L = 54;
  const PAD_R = 24;
  const PAD_T = 18;
  const PAD_B = 36;
  const geom = computeSvgOverlayGeometry(points, yScale, {
    width: W,
    height: H,
    padL: PAD_L,
    padR: PAD_R,
    padT: PAD_T,
    padB: PAD_B,
  });

  const nVerified = points.filter(p => p.verified).length;
  const nOob = points.filter(p => isPointOutOfBounds(p, xAx, yAx)).length;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {/* Summary KPI bar — hides derived PK values when local edits are uncommitted */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
          gap: 8,
          padding: '8px 12px',
          borderRadius: 6,
          background: 'var(--accent-bg)',
          border: '1px solid var(--border)',
        }}
      >
        <div>
          <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>Digitized Cmax</div>
          <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-h)', ...MONO }}>
            {isDirty || !derivedPk ? (
              <span style={{ fontSize: 12, color: '#9A5B12' }}>Stale — re-calibrate</span>
            ) : (
              `${fmt(derivedPk.cmax, 2)} ${yAx?.unit}`
            )}
          </div>
        </div>
        <div>
          <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>Digitized Tmax</div>
          <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-h)', ...MONO }}>
            {isDirty || !derivedPk ? (
              <span style={{ fontSize: 12, color: '#9A5B12' }}>Stale — re-calibrate</span>
            ) : (
              `${fmt(derivedPk.tmax, 2)} ${xAx?.unit}`
            )}
          </div>
        </div>
        <div>
          <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>Trapezoidal AUClast</div>
          <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-h)', ...MONO }}>
            {isDirty || !derivedPk ? (
              <span style={{ fontSize: 12, color: '#9A5B12' }}>Stale — re-calibrate</span>
            ) : (
              `${fmt(derivedPk.auclast, 1)} ${yAx?.unit}·${xAx?.unit}`
            )}
          </div>
        </div>
        <div>
          <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>Verification &amp; Bounds</div>
          <div
            style={{
              fontSize: 12,
              fontWeight: 600,
              color: nOob > 0 ? '#B23A2E' : nVerified === points.length ? '#1D7A5A' : '#9A5B12',
              ...MONO,
            }}
          >
            {nVerified === points.length ? (
              <CheckCircle size={12} style={{ display: 'inline', marginRight: 4 }} />
            ) : (
              <AlertTriangle size={12} style={{ display: 'inline', marginRight: 4 }} />
            )}
            {nVerified}/{points.length} verified · {nOob} out-of-bounds
            {isDirty ? ' · preview' : ''}
          </div>
        </div>
      </div>

      {errMsg && (
        <div
          role="alert"
          style={{
            padding: '6px 10px',
            borderRadius: 4,
            background: 'rgba(178, 58, 46, 0.1)',
            color: '#B23A2E',
            fontSize: 12,
          }}
        >
          {errMsg}
        </div>
      )}

      {/* Calibration & metadata controls */}
      {onRecompute && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center', fontSize: 12 }}>
          <label className="sim-field">
            Y-axis scale
            <select
              className="model-select"
              value={yScale}
              disabled={busy || localBusy}
              onChange={e => {
                const next = e.target.value as AxisScale;
                setViewState(prev => ({
                  ...prev,
                  yScale: next,
                  isDirty: true,
                  derivedPk: null,
                }));
                void triggerRecompute(points, next, pixelSigma, strictBounds);
              }}
            >
              <option value="linear">Linear</option>
              <option value="log10">Log10 (semi-log)</option>
            </select>
          </label>
          <label className="sim-field">
            Click σ metadata (px)
            <input
              type="number"
              step="0.25"
              min="0.0"
              max="10"
              value={pixelSigma}
              onChange={e =>
                setViewState(prev => ({ ...prev, pixelSigma: Number(e.target.value), isDirty: true, derivedPk: null }))
              }
              style={{ width: 64 }}
            />
          </label>
          <label style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
            <input
              type="checkbox"
              checked={strictBounds}
              onChange={e =>
                setViewState(prev => ({ ...prev, strictBounds: e.target.checked, isDirty: true, derivedPk: null }))
              }
            />
            Strict bounds
          </label>
          <button
            type="button"
            className="chip"
            disabled={busy || localBusy}
            onClick={() => {
              void triggerRecompute();
            }}
          >
            <RefreshCw size={11} style={{ marginRight: 4 }} />
            Re-calibrate (authoritative)
          </button>
          <button
            type="button"
            className="chip"
            disabled={busy || localBusy}
            onClick={() => setViewState(prev => applyAddDigitizerPoint(prev, xAx, yAx))}
          >
            <Plus size={11} style={{ marginRight: 4 }} />
            Add point
          </button>
        </div>
      )}

      {/* Calibrated SVG Curve & Directly Digitized Bounds Overlay */}
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width={W}
        style={{
          maxWidth: '100%',
          height: 'auto',
          display: 'block',
          border: '1px solid var(--border)',
          borderRadius: 6,
          background: 'var(--surface)',
        }}
        role="img"
        aria-label="Calibrated concentration-time digitized curve with directly digitized error-bar bounds"
      >
        {geom.areaPath && <path d={geom.areaPath} fill="var(--accent)" fillOpacity={0.1} />}
        <line x1={PAD_L} x2={W - PAD_R} y1={H - PAD_B} y2={H - PAD_B} stroke="var(--border-strong)" />
        <line x1={PAD_L} x2={PAD_L} y1={PAD_T} y2={H - PAD_B} stroke="var(--border-strong)" />
        {geom.linePath && <path d={geom.linePath} fill="none" stroke="var(--accent)" strokeWidth={2} />}
        {geom.markers.map(m => {
          const color = m.outOfBounds ? '#B23A2E' : m.verified ? 'var(--accent)' : '#9A5B12';
          return (
            <g key={m.id}>
              {m.yLo != null && m.yHi != null && (
                <>
                  <line x1={m.cx} x2={m.cx} y1={m.yHi} y2={m.yLo} stroke={color} strokeWidth={1.6} />
                  <line x1={m.cx - 4} x2={m.cx + 4} y1={m.yHi} y2={m.yHi} stroke={color} strokeWidth={1.6} />
                  <line x1={m.cx - 4} x2={m.cx + 4} y1={m.yLo} y2={m.yLo} stroke={color} strokeWidth={1.6} />
                </>
              )}
              <circle cx={m.cx} cy={m.cy} r={4.2} fill={color} stroke="var(--bg)" strokeWidth={1.2} />
            </g>
          );
        })}
        <text x={(PAD_L + W - PAD_R) / 2} y={H - 8} textAnchor="middle" fontSize={11} fill="var(--text-dim)">
          {xAx?.label} ({xAx?.unit}) · Calibrated [{xAx?.v1} → {xAx?.v2}]
        </text>
        <text
          x={14}
          y={(PAD_T + H - PAD_B) / 2}
          textAnchor="middle"
          fontSize={11}
          fill="var(--text-dim)"
          transform={`rotate(-90 14 ${(PAD_T + H - PAD_B) / 2})`}
        >
          {yAx?.label} ({yAx?.unit}, {yScale})
        </text>
      </svg>

      {/* Editable points table */}
      <div style={{ overflowX: 'auto' }}>
        <table className="data-table" style={{ width: '100%', fontSize: 12 }}>
          <thead>
            <tr>
              <th>Point</th>
              <th>u (px)</th>
              <th>v (px)</th>
              <th>
                {xAx?.label ?? 'X'} ({xAx?.unit})
              </th>
              <th>
                {yAx?.label ?? 'Y'} ({yAx?.unit})
              </th>
              <th>Y low</th>
              <th>Y high</th>
              <th>Origin / Gate</th>
              {onRecompute && <th />}
            </tr>
          </thead>
          <tbody>
            {points.map((p, idx) => {
              const oob = isPointOutOfBounds(p, xAx, yAx);
              return (
                <tr key={p.id}>
                  <td style={MONO}>
                    {p.id}
                    {oob && (
                      <span
                        title="Point lies outside calibrated axis bounds"
                        style={{ color: '#B23A2E', marginLeft: 4, fontWeight: 700 }}
                      >
                        [OOB]
                      </span>
                    )}
                    {p.previewOnly && (
                      <span
                        title="Preview coordinate only — click Re-calibrate for authoritative backend result"
                        style={{ color: '#9A5B12', marginLeft: 4, fontSize: 10 }}
                      >
                        (preview)
                      </span>
                    )}
                  </td>
                  <td>
                    {onRecompute ? (
                      <input
                        type="number"
                        value={p.u_px}
                        onChange={e =>
                          setViewState(prev =>
                            applyPointPixelEdit(prev, idx, 'u_px', Number(e.target.value), xAx, yAx),
                          )
                        }
                        style={{ width: 64, fontSize: 11, ...MONO }}
                      />
                    ) : (
                      <span style={MONO}>{fmt(p.u_px, 1)}</span>
                    )}
                  </td>
                  <td>
                    {onRecompute ? (
                      <input
                        type="number"
                        value={p.v_px}
                        onChange={e =>
                          setViewState(prev =>
                            applyPointPixelEdit(prev, idx, 'v_px', Number(e.target.value), xAx, yAx),
                          )
                        }
                        style={{ width: 64, fontSize: 11, ...MONO }}
                      />
                    ) : (
                      <span style={MONO}>{fmt(p.v_px, 1)}</span>
                    )}
                  </td>
                  <td style={MONO}>{fmt(p.x, 2)}</td>
                  <td style={{ fontWeight: 600, ...MONO }}>{fmt(p.y, 2)}</td>
                  <td style={MONO}>{fmt(p.y_low, 2)}</td>
                  <td style={MONO}>{fmt(p.y_high, 2)}</td>
                  <td>
                    <button
                      type="button"
                      className="chip"
                      style={{
                        padding: '1px 7px',
                        fontSize: 11,
                        color: p.verified ? '#1D7A5A' : '#9A5B12',
                      }}
                      onClick={() =>
                        setViewState(prev => ({
                          ...prev,
                          points: togglePointVerified(prev.points, idx),
                          isDirty: true,
                          derivedPk: null,
                        }))
                      }
                      title="Click to toggle human verification status"
                    >
                      {p.verified ? 'verified' : `${p.origin} · verify`}
                    </button>
                  </td>
                  {onRecompute && (
                    <td>
                      <button
                        type="button"
                        className="chip"
                        style={{ padding: '1px 5px' }}
                        disabled={points.length <= 2}
                        onClick={() => setViewState(prev => applyRemoveDigitizerPoint(prev, idx))}
                        title="Delete point"
                      >
                        <Trash2 size={11} />
                      </button>
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
