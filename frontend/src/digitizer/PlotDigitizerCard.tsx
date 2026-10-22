import React, { useState } from 'react';
import { AlertTriangle, CheckCircle, Plus, RefreshCw, Trash2 } from 'lucide-react';
import type {
  AxisCalibrationSpec,
  AxisScale,
  DigitizedPoint,
  DigitizeResult,
  DigitizerViewState,
  ErrorBarType,
  PlotROISpec,
} from './digitizeMath';
import {
  applyAddDigitizerPoint,
  applyPointPixelEdit,
  applyRemoveDigitizerPoint,
  buildResultSourceKey,
  computeSvgOverlayGeometry,
  createDigitizerViewState,
  evaluatePointBounds,
  getSeriesDisplayMetrics,
  isPointExtrapolated,
  selectDigitizerSeries,
  togglePointVerified,
  validateAxisSpec,
  validatePlotRoi,
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
  plot_roi?: PlotROISpec | null;
  pixel_sigma: number;
  error_type: ErrorBarType;
  strict_bounds: boolean;
  points: {
    id: string;
    series: string;
    u_px: number;
    v_px: number;
    v_low_px: number | null;
    v_high_px: number | null;
    oob_override?: boolean;
    oob_override_reason?: string | null;
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
    setViewState(createDigitizerViewState(r, viewState.selectedSeries));
  }

  const {
    points,
    availableSeries,
    selectedSeries,
    yScale,
    pixelSigma,
    strictBounds,
    isDirty,
  } = viewState;
  const xAx: AxisCalibrationSpec = r.x_axis;
  const yAx: AxisCalibrationSpec = { ...r.y_axis, scale: yScale };
  const plotRoi: PlotROISpec | null = r.plot_roi ?? null;

  const seriesDisplay = getSeriesDisplayMetrics(viewState);
  const activeSeriesPoints = points.filter(
    p => (p.series ?? 'Series 1') === selectedSeries,
  );

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
    const roiErr = validatePlotRoi(plotRoi);
    if (xErr || yErr || roiErr) {
      setErrMsg(xErr ?? yErr ?? roiErr);
      return;
    }
    if (nextStrict && !plotRoi) {
      setErrMsg('strict_bounds=true requires an explicit plot_roi.');
      return;
    }
    setErrMsg(null);
    setLocalBusy(true);
    try {
      await onRecompute({
        x_axis: xAx,
        y_axis: candidateY,
        plot_roi: plotRoi,
        pixel_sigma: nextSigma,
        error_type: r.error_type ?? 'se',
        strict_bounds: nextStrict,
        points: nextPts.map(p => ({
          id: p.id,
          series: p.series ?? 'Series 1',
          u_px: p.u_px,
          v_px: p.v_px,
          v_low_px: p.v_low_px,
          v_high_px: p.v_high_px,
          oob_override: p.oob_override,
          oob_override_reason: p.oob_override_reason,
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
  const geom = computeSvgOverlayGeometry(activeSeriesPoints, yScale, {
    width: W,
    height: H,
    padL: PAD_L,
    padR: PAD_R,
    padT: PAD_T,
    padB: PAD_B,
  });

  const nVerified = activeSeriesPoints.filter(p => p.verified).length;
  const nOob = activeSeriesPoints.filter(
    p => evaluatePointBounds(p, plotRoi).out_of_bounds === true,
  ).length;
  const nExtrap = activeSeriesPoints.filter(p =>
    isPointExtrapolated(p, xAx, yAx),
  ).length;
  const boundsLabel = plotRoi == null ? 'ROI unknown' : `${nOob} OOB`;

  const renderMetricValue = (
    val: number | null | undefined,
    digits: number,
    unitStr: string,
  ) => {
    if (seriesDisplay.displayStatus === 'stale') {
      return <span style={{ fontSize: 12, color: '#9A5B12' }}>Stale — re-calibrate</span>;
    }
    if (seriesDisplay.displayStatus !== 'resolved' || !seriesDisplay.pkMetrics) {
      return (
        <span style={{ fontSize: 11, color: '#B23A2E' }}>
          Withheld ({seriesDisplay.displayStatus})
        </span>
      );
    }
    return `${fmt(val, digits)} ${unitStr}`;
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {/* Series-keyed KPI summary bar */}
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
          <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>Active Series</div>
          <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-h)', ...MONO }}>
            {availableSeries.length > 1 ? (
              <select
                aria-label="Select digitized series"
                className="model-select"
                value={selectedSeries}
                onChange={e =>
                  setViewState(prev => selectDigitizerSeries(prev, e.target.value))
                }
                style={{ fontSize: 12 }}
              >
                {availableSeries.map(sName => (
                  <option key={sName} value={sName}>
                    {sName}
                  </option>
                ))}
              </select>
            ) : (
              selectedSeries
            )}
          </div>
        </div>
        <div>
          <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>
            Digitized Cmax ({selectedSeries})
          </div>
          <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-h)', ...MONO }}>
            {renderMetricValue(seriesDisplay.pkMetrics?.cmax, 2, yAx?.unit ?? '')}
          </div>
        </div>
        <div>
          <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>
            Digitized Tmax ({selectedSeries})
          </div>
          <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-h)', ...MONO }}>
            {renderMetricValue(seriesDisplay.pkMetrics?.tmax, 2, xAx?.unit ?? '')}
          </div>
        </div>
        <div>
          <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>
            Trapezoidal AUClast ({selectedSeries})
          </div>
          <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-h)', ...MONO }}>
            {renderMetricValue(
              seriesDisplay.pkMetrics?.auclast,
              1,
              `${yAx?.unit ?? ''}·${xAx?.unit ?? ''}`,
            )}
          </div>
        </div>
        <div>
          <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>
            Verification &amp; Plot ROI
          </div>
          <div
            style={{
              fontSize: 12,
              fontWeight: 600,
              color:
                nOob > 0
                  ? '#B23A2E'
                  : nVerified === activeSeriesPoints.length
                    ? '#1D7A5A'
                    : '#9A5B12',
              ...MONO,
            }}
          >
            {nVerified === activeSeriesPoints.length && nOob === 0 ? (
              <CheckCircle size={12} style={{ display: 'inline', marginRight: 4 }} />
            ) : (
              <AlertTriangle size={12} style={{ display: 'inline', marginRight: 4 }} />
            )}
            {nVerified}/{activeSeriesPoints.length} verified · {boundsLabel} · {nExtrap} extrap
            {isDirty ? ' · preview' : ''}
          </div>
        </div>
      </div>

      {/* Unresolved / withheld metrics reason banner */}
      {seriesDisplay.displayStatus !== 'resolved' && seriesDisplay.unresolvedReason && (
        <div
          role="status"
          style={{
            padding: '6px 10px',
            borderRadius: 4,
            background: 'rgba(154, 91, 18, 0.12)',
            color: '#9A5B12',
            fontSize: 12,
          }}
        >
          <strong>{seriesDisplay.displayStatus}:</strong> {seriesDisplay.unresolvedReason}
          {seriesDisplay.blockingPointIds.length > 0 && (
            <span style={{ marginLeft: 6, ...MONO }}>
              (blocking: {seriesDisplay.blockingPointIds.join(', ')})
            </span>
          )}
        </div>
      )}

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
                  seriesByKey: null,
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
                setViewState(prev => ({
                  ...prev,
                  pixelSigma: Number(e.target.value),
                  isDirty: true,
                  seriesByKey: null,
                }))
              }
              style={{ width: 64 }}
            />
          </label>
          <label style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
            <input
              type="checkbox"
              checked={strictBounds}
              disabled={!plotRoi}
              title={
                plotRoi
                  ? 'Reject points outside plot_roi on recalibration'
                  : 'Requires an explicit plot_roi'
              }
              onChange={e =>
                setViewState(prev => ({
                  ...prev,
                  strictBounds: e.target.checked,
                  isDirty: true,
                  seriesByKey: null,
                }))
              }
            />
            Strict ROI bounds
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
            onClick={() =>
              setViewState(prev => applyAddDigitizerPoint(prev, xAx, yAx, plotRoi))
            }
          >
            <Plus size={11} style={{ marginRight: 4 }} />
            Add point ({selectedSeries})
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
        aria-label={`Calibrated concentration-time digitized curve for ${selectedSeries}`}
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
          {selectedSeries} · {xAx?.label} ({xAx?.unit}) · Anchors [{xAx?.v1} → {xAx?.v2}]
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
              <th>Series</th>
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
              <th>ROI / Span</th>
              <th>Origin / Gate</th>
              {onRecompute && <th />}
            </tr>
          </thead>
          <tbody>
            {points.map((p, idx) => {
              const bInfo = evaluatePointBounds(p, plotRoi);
              const extrap = isPointExtrapolated(p, xAx, yAx);
              return (
                <tr key={`${p.series ?? 'Series 1'}_${p.id}`}>
                  <td style={MONO}>
                    {p.id}
                    {p.previewOnly && (
                      <span
                        title="Preview coordinate only — click Re-calibrate for authoritative backend result"
                        style={{ color: '#9A5B12', marginLeft: 4, fontSize: 10 }}
                      >
                        (preview)
                      </span>
                    )}
                  </td>
                  <td style={MONO}>{p.series ?? 'Series 1'}</td>
                  <td>
                    {onRecompute ? (
                      <input
                        type="number"
                        value={p.u_px}
                        onChange={e =>
                          setViewState(prev =>
                            applyPointPixelEdit(
                              prev,
                              idx,
                              'u_px',
                              Number(e.target.value),
                              xAx,
                              yAx,
                              plotRoi,
                            ),
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
                            applyPointPixelEdit(
                              prev,
                              idx,
                              'v_px',
                              Number(e.target.value),
                              xAx,
                              yAx,
                              plotRoi,
                            ),
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
                  <td style={MONO}>
                    {bInfo.bounds_status === 'out_of_bounds' ? (
                      <span
                        title="Point lies outside declared plot_roi"
                        style={{ color: '#B23A2E', fontWeight: 700 }}
                      >
                        {p.oob_override ? 'OOB (overridden)' : 'OOB'}
                      </span>
                    ) : bInfo.bounds_status === 'unknown' ? (
                      <span style={{ color: 'var(--text-dim)' }}>unknown</span>
                    ) : (
                      <span style={{ color: '#1D7A5A' }}>in_bounds</span>
                    )}
                    {extrap && (
                      <span
                        title="Outside calibration-anchor span (extrapolated)"
                        style={{ color: '#9A5B12', marginLeft: 4 }}
                      >
                        · extrap
                      </span>
                    )}
                  </td>
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
                          seriesByKey: null,
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
                        onClick={() =>
                          setViewState(prev => applyRemoveDigitizerPoint(prev, idx))
                        }
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
