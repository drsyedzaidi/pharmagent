/**
 * Pure TypeScript geometry, validation, explicit plot-region (ROI) bounds,
 * verified-only series-keyed PK display gates, and preview-only state helpers
 * for PlotDigitizerCard.
 *
 * Tested directly by `frontend/tests/plotDigitizer.test.ts`.
 */

export type AxisScale = 'linear' | 'log10';
export type ErrorBarType = 'se' | 'sd' | 'ci95';
export type BoundsStatus = 'in_bounds' | 'out_of_bounds' | 'unknown';
export type MetricsStatus =
  | 'resolved'
  | 'unresolved_unverified_points'
  | 'unresolved_out_of_bounds'
  | 'unresolved_unverified_and_out_of_bounds'
  | 'unresolved_insufficient_points';

export interface AxisCalibrationSpec {
  p1_px: number;
  p2_px: number;
  v1: number;
  v2: number;
  scale: AxisScale;
  label: string;
  unit: string;
}

export interface PlotROISpec {
  u_min_px: number;
  u_max_px: number;
  v_min_px: number;
  v_max_px: number;
}

export interface DigitizedPoint {
  id: string;
  index?: number;
  series?: string;
  u_px: number;
  v_px: number;
  v_low_px: number | null;
  v_high_px: number | null;
  x: number | null;
  y: number | null;
  y_low: number | null;
  y_high: number | null;
  extrapolated_x?: boolean;
  extrapolated_y?: boolean;
  extrapolated?: boolean;
  bounds_status?: BoundsStatus;
  out_of_bounds_x?: boolean | null;
  out_of_bounds_y?: boolean | null;
  out_of_bounds?: boolean | null;
  oob_override?: boolean;
  oob_override_reason?: string | null;
  origin: 'manual' | 'contour_proposed';
  verified: boolean;
  previewOnly?: boolean;
}

export interface DerivedPkMetrics {
  cmax: number | null;
  tmax: number | null;
  auclast: number | null;
}

export interface SeriesSummary {
  series: string;
  status: string;
  metrics_status: MetricsStatus;
  unresolved_reason: string | null;
  blocking_point_ids: string[];
  unverified_point_ids: string[];
  blocking_oob_point_ids: string[];
  recorded_oob_overrides: { id: string; reason: string }[];
  n_points: number;
  n_verified: number;
  all_verified: boolean;
  n_extrapolated: number;
  n_out_of_bounds: number;
  bounds_status: BoundsStatus;
  x_range: [number, number];
  y_range: [number, number];
  pk_metrics: DerivedPkMetrics;
}

export interface DigitizeResult {
  status: string;
  metrics_status?: MetricsStatus;
  unresolved_reason?: string | null;
  blocking_point_ids?: string[];
  figure_label?: string;
  source_sha256?: string | null;
  export_sha256?: string;
  csv_export?: string;
  pixel_sigma: number;
  anchor_sigma_px?: number;
  error_type: ErrorBarType;
  strict_bounds?: boolean;
  plot_roi?: PlotROISpec | null;
  bounds_status?: BoundsStatus;
  x_axis: AxisCalibrationSpec;
  y_axis: AxisCalibrationSpec;
  n_points: number;
  n_verified: number;
  all_verified: boolean;
  n_extrapolated?: number;
  n_out_of_bounds: number;
  warnings?: string[];
  points: DigitizedPoint[];
  series: Record<string, SeriesSummary>;
  series_pk?: Record<string, DerivedPkMetrics>;
}

export interface DigitizerViewState {
  sourceKey: string;
  points: DigitizedPoint[];
  availableSeries: string[];
  selectedSeries: string;
  seriesByKey: Record<string, SeriesSummary> | null;
  yScale: AxisScale;
  pixelSigma: number;
  strictBounds: boolean;
  isDirty: boolean;
}

export interface SeriesDisplayMetrics {
  selectedSeries: string;
  availableSeries: string[];
  displayStatus: 'stale' | MetricsStatus;
  unresolvedReason: string | null;
  blockingPointIds: string[];
  pkMetrics: DerivedPkMetrics | null;
}

export function validateAxisSpec(ax: AxisCalibrationSpec, name = 'axis'): string | null {
  if (ax.scale !== 'linear' && ax.scale !== 'log10') {
    return `${name} scale must be 'linear' or 'log10'.`;
  }
  if (!Number.isFinite(ax.p1_px) || !Number.isFinite(ax.p2_px)) {
    return `${name} pixel anchors must be finite numbers.`;
  }
  if (Math.abs(ax.p2_px - ax.p1_px) < 1.0) {
    return `${name} pixel anchors must be separated by at least 1 px.`;
  }
  if (!Number.isFinite(ax.v1) || !Number.isFinite(ax.v2)) {
    return `${name} reference values must be finite numbers.`;
  }
  if (Math.abs(ax.v2 - ax.v1) <= 1e-12) {
    return `${name} reference values v1 and v2 must be distinct.`;
  }
  if (ax.scale === 'log10' && (ax.v1 <= 0 || ax.v2 <= 0)) {
    return `${name} log10 scale requires strictly positive reference values.`;
  }
  return null;
}

export function validatePlotRoi(roi: PlotROISpec | null | undefined): string | null {
  if (roi == null) return null;
  if (
    !Number.isFinite(roi.u_min_px) ||
    !Number.isFinite(roi.u_max_px) ||
    !Number.isFinite(roi.v_min_px) ||
    !Number.isFinite(roi.v_max_px)
  ) {
    return 'plot_roi coordinates must be finite numbers.';
  }
  if (roi.u_max_px - roi.u_min_px < 1.0) {
    return 'plot_roi horizontal span must be at least 1 px.';
  }
  if (roi.v_max_px - roi.v_min_px < 1.0) {
    return 'plot_roi vertical span must be at least 1 px.';
  }
  return null;
}

export function pixelToValue(ax: AxisCalibrationSpec, px: number): number {
  const err = validateAxisSpec(ax);
  if (err) throw new Error(err);
  const frac = (px - ax.p1_px) / (ax.p2_px - ax.p1_px);
  if (ax.scale === 'log10') {
    const lv1 = Math.log10(ax.v1);
    const lv2 = Math.log10(ax.v2);
    return Math.pow(10, lv1 + frac * (lv2 - lv1));
  }
  return ax.v1 + frac * (ax.v2 - ax.v1);
}

export function valueToPixel(ax: AxisCalibrationSpec, val: number): number {
  const err = validateAxisSpec(ax);
  if (err) throw new Error(err);
  let frac: number;
  if (ax.scale === 'log10') {
    if (val <= 0) throw new Error('Cannot map non-positive value onto a log10 axis.');
    const lv1 = Math.log10(ax.v1);
    const lv2 = Math.log10(ax.v2);
    frac = (Math.log10(val) - lv1) / (lv2 - lv1);
  } else {
    frac = (val - ax.v1) / (ax.v2 - ax.v1);
  }
  return ax.p1_px + frac * (ax.p2_px - ax.p1_px);
}

export function isPixelInAnchorSpan(ax: AxisCalibrationSpec, px: number, tolPx = 1e-6): boolean {
  const lo = Math.min(ax.p1_px, ax.p2_px) - Math.max(0, tolPx);
  const hi = Math.max(ax.p1_px, ax.p2_px) + Math.max(0, tolPx);
  return px >= lo && px <= hi;
}

export function isPointExtrapolated(
  pt: Pick<DigitizedPoint, 'u_px' | 'v_px' | 'v_low_px' | 'v_high_px'>,
  xAx: AxisCalibrationSpec,
  yAx: AxisCalibrationSpec,
  tolPx = 1e-6,
): boolean {
  if (!isPixelInAnchorSpan(xAx, pt.u_px, tolPx) || !isPixelInAnchorSpan(yAx, pt.v_px, tolPx)) {
    return true;
  }
  if (pt.v_low_px != null && !isPixelInAnchorSpan(yAx, pt.v_low_px, tolPx)) return true;
  if (pt.v_high_px != null && !isPixelInAnchorSpan(yAx, pt.v_high_px, tolPx)) return true;
  return false;
}

export function evaluatePointBounds(
  pt: Pick<DigitizedPoint, 'u_px' | 'v_px' | 'v_low_px' | 'v_high_px'>,
  plotRoi?: PlotROISpec | null,
  tolPx = 1e-6,
): {
  bounds_status: BoundsStatus;
  out_of_bounds: boolean | null;
  out_of_bounds_x: boolean | null;
  out_of_bounds_y: boolean | null;
} {
  if (!plotRoi || validatePlotRoi(plotRoi) !== null) {
    return {
      bounds_status: 'unknown',
      out_of_bounds: null,
      out_of_bounds_x: null,
      out_of_bounds_y: null,
    };
  }
  const tol = Math.max(0, tolPx);
  const inU = (u: number) => u >= plotRoi.u_min_px - tol && u <= plotRoi.u_max_px + tol;
  const inV = (v: number) => v >= plotRoi.v_min_px - tol && v <= plotRoi.v_max_px + tol;

  const oobX = !inU(pt.u_px);
  const oobY =
    !inV(pt.v_px) ||
    (pt.v_low_px != null && !inV(pt.v_low_px)) ||
    (pt.v_high_px != null && !inV(pt.v_high_px));
  const oob = oobX || oobY;
  return {
    bounds_status: oob ? 'out_of_bounds' : 'in_bounds',
    out_of_bounds: oob,
    out_of_bounds_x: oobX,
    out_of_bounds_y: oobY,
  };
}

export function isPointOutOfBounds(
  pt: Pick<DigitizedPoint, 'u_px' | 'v_px' | 'v_low_px' | 'v_high_px'>,
  plotRoi?: PlotROISpec | null,
  tolPx = 1e-6,
): boolean {
  return evaluatePointBounds(pt, plotRoi, tolPx).out_of_bounds === true;
}

export function computePointPreview(
  pt: DigitizedPoint,
  xAx?: AxisCalibrationSpec,
  yAx?: AxisCalibrationSpec,
  plotRoi?: PlotROISpec | null,
): DigitizedPoint {
  const bounds = evaluatePointBounds(pt, plotRoi);
  if (!xAx || !yAx || validateAxisSpec(xAx) !== null || validateAxisSpec(yAx) !== null) {
    return {
      ...pt,
      x: null,
      y: null,
      y_low: null,
      y_high: null,
      ...bounds,
      previewOnly: true,
    };
  }
  const x = pixelToValue(xAx, pt.u_px);
  const y = pixelToValue(yAx, pt.v_px);
  let yLow: number | null = null;
  let yHigh: number | null = null;
  if (pt.v_low_px != null || pt.v_high_px != null) {
    const vLo = pt.v_low_px != null ? pt.v_low_px : pt.v_px;
    const vHi = pt.v_high_px != null ? pt.v_high_px : pt.v_px;
    const valA = pixelToValue(yAx, vLo);
    const valB = pixelToValue(yAx, vHi);
    yLow = Math.min(valA, valB);
    yHigh = Math.max(valA, valB);
  }
  const extrapX = !isPixelInAnchorSpan(xAx, pt.u_px);
  const extrapY =
    !isPixelInAnchorSpan(yAx, pt.v_px) ||
    (pt.v_low_px != null && !isPixelInAnchorSpan(yAx, pt.v_low_px)) ||
    (pt.v_high_px != null && !isPixelInAnchorSpan(yAx, pt.v_high_px));
  return {
    ...pt,
    x,
    y,
    y_low: yLow,
    y_high: yHigh,
    extrapolated_x: extrapX,
    extrapolated_y: extrapY,
    extrapolated: extrapX || extrapY,
    ...bounds,
    previewOnly: true,
  };
}

export function updatePointPixel(
  points: DigitizedPoint[],
  idx: number,
  field: 'u_px' | 'v_px',
  val: number,
  xAx?: AxisCalibrationSpec,
  yAx?: AxisCalibrationSpec,
  plotRoi?: PlotROISpec | null,
): DigitizedPoint[] {
  if (!Number.isFinite(val)) return points;
  return points.map((p, i) => {
    if (i !== idx) return p;
    const edited: DigitizedPoint = {
      ...p,
      [field]: val,
      origin: 'manual',
      verified: true,
    };
    return computePointPreview(edited, xAx, yAx, plotRoi);
  });
}

export function togglePointVerified(points: DigitizedPoint[], idx: number): DigitizedPoint[] {
  return points.map((p, i) => (i === idx ? { ...p, verified: !p.verified } : p));
}

export function addDigitizerPoint(
  points: DigitizedPoint[],
  xAx?: AxisCalibrationSpec,
  yAx?: AxisCalibrationSpec,
  plotRoi?: PlotROISpec | null,
  seriesName?: string,
): DigitizedPoint[] {
  const last = points[points.length - 1];
  const nextId = `pt_${points.length + 1}`;
  let uPx = last ? last.u_px + 25 : xAx ? (xAx.p1_px + xAx.p2_px) / 2 : 150;
  const vPx = last ? last.v_px : yAx ? (yAx.p1_px + yAx.p2_px) / 2 : 250;
  if (xAx) {
    const xLo = Math.min(xAx.p1_px, xAx.p2_px);
    const xHi = Math.max(xAx.p1_px, xAx.p2_px);
    uPx = Math.min(xHi, Math.max(xLo, uPx));
  }
  if (plotRoi) {
    uPx = Math.min(plotRoi.u_max_px, Math.max(plotRoi.u_min_px, uPx));
  }
  const basePt: DigitizedPoint = {
    id: nextId,
    series: seriesName ?? last?.series ?? 'Series 1',
    u_px: uPx,
    v_px: vPx,
    v_low_px: null,
    v_high_px: null,
    x: null,
    y: null,
    y_low: null,
    y_high: null,
    bounds_status: plotRoi ? 'in_bounds' : 'unknown',
    out_of_bounds: plotRoi ? false : null,
    origin: 'manual',
    verified: true,
    previewOnly: true,
  };
  return [...points, computePointPreview(basePt, xAx, yAx, plotRoi)];
}

export function removeDigitizerPoint(points: DigitizedPoint[], idx: number): DigitizedPoint[] {
  if (points.length <= 2) return points;
  return points.filter((_, i) => i !== idx);
}

function compareStrings(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

export function canonicalizePoints(points: DigitizedPoint[]): DigitizedPoint[] {
  return [...points].sort((a, b) => {
    const sa = a.series ?? 'Series 1';
    const sb = b.series ?? 'Series 1';
    const sCmp = compareStrings(sa, sb);
    if (sCmp !== 0) return sCmp;
    const xa = a.x ?? Number.POSITIVE_INFINITY;
    const xb = b.x ?? Number.POSITIVE_INFINITY;
    if (xa !== xb) return xa - xb;
    const idCmp = compareStrings(a.id, b.id);
    if (idCmp !== 0) return idCmp;
    if (a.u_px !== b.u_px) return a.u_px - b.u_px;
    return a.v_px - b.v_px;
  });
}

export function getSortedSeriesNames(r: Pick<DigitizeResult, 'series' | 'points'>): string[] {
  const names = new Set<string>();
  if (r.series) {
    for (const k of Object.keys(r.series)) {
      if (k) names.add(k);
    }
  }
  for (const p of r.points ?? []) {
    names.add(p.series ?? 'Series 1');
  }
  if (names.size === 0) {
    names.add('Series 1');
  }
  return [...names].sort(compareStrings);
}

export function buildResultSourceKey(r: DigitizeResult): string {
  const canonicalPts = canonicalizePoints(r.points ?? []);
  const ptsKey = canonicalPts
    .map(
      p =>
        `${p.series ?? 'Series 1'}:${p.id}:${p.u_px}:${p.v_px}:${p.x ?? 'null'}:${p.y ?? 'null'}:${p.verified}:${p.bounds_status ?? 'unknown'}`,
    )
    .join('|');
  const roiKey = r.plot_roi
    ? `${r.plot_roi.u_min_px},${r.plot_roi.u_max_px},${r.plot_roi.v_min_px},${r.plot_roi.v_max_px}`
    : 'none';
  return [
    r.export_sha256 ?? '',
    r.source_sha256 ?? '',
    r.status ?? '',
    r.y_axis?.scale ?? 'linear',
    r.pixel_sigma ?? 1.5,
    Boolean(r.strict_bounds),
    roiKey,
    ptsKey,
  ].join('::');
}

export function createDigitizerViewState(
  r: DigitizeResult,
  preferredSeries?: string,
): DigitizerViewState {
  const availableSeries = getSortedSeriesNames(r);
  const selectedSeries =
    preferredSeries && availableSeries.includes(preferredSeries)
      ? preferredSeries
      : availableSeries[0];
  const sortedSeriesMap: Record<string, SeriesSummary> = {};
  for (const sName of availableSeries) {
    if (r.series && r.series[sName]) {
      sortedSeriesMap[sName] = r.series[sName];
    }
  }
  return {
    sourceKey: buildResultSourceKey(r),
    points: canonicalizePoints(r.points ?? []).map(p => ({
      ...p,
      series: p.series ?? 'Series 1',
      y_low: p.y_low ?? null,
      y_high: p.y_high ?? null,
      previewOnly: false,
    })),
    availableSeries,
    selectedSeries,
    seriesByKey: sortedSeriesMap,
    yScale: r.y_axis?.scale ?? 'linear',
    pixelSigma: r.pixel_sigma ?? 1.5,
    strictBounds: Boolean(r.strict_bounds),
    isDirty: false,
  };
}

export function syncDigitizerViewState(
  current: DigitizerViewState,
  nextResult: DigitizeResult,
): DigitizerViewState {
  const nextKey = buildResultSourceKey(nextResult);
  if (current.sourceKey === nextKey) {
    return current;
  }
  return createDigitizerViewState(nextResult, current.selectedSeries);
}

export function selectDigitizerSeries(
  state: DigitizerViewState,
  seriesName: string,
): DigitizerViewState {
  if (!state.availableSeries.includes(seriesName)) return state;
  return {
    ...state,
    selectedSeries: seriesName,
  };
}

export function getSeriesDisplayMetrics(state: DigitizerViewState): SeriesDisplayMetrics {
  const { selectedSeries, availableSeries, isDirty, seriesByKey, points } = state;
  if (isDirty || !seriesByKey) {
    return {
      selectedSeries,
      availableSeries,
      displayStatus: 'stale',
      unresolvedReason: 'Local edits pending — click Re-calibrate for authoritative backend metrics.',
      blockingPointIds: [],
      pkMetrics: null,
    };
  }
  const summary = seriesByKey[selectedSeries];
  const seriesPoints = points.filter(p => (p.series ?? 'Series 1') === selectedSeries);
  const localUnverified = seriesPoints.filter(p => !p.verified).map(p => p.id);
  const localBlockingOob = seriesPoints
    .filter(p => p.out_of_bounds === true && !p.oob_override)
    .map(p => p.id);

  if (!summary) {
    return {
      selectedSeries,
      availableSeries,
      displayStatus: 'unresolved_insufficient_points',
      unresolvedReason: `No backend summary available for series '${selectedSeries}'.`,
      blockingPointIds: [],
      pkMetrics: null,
    };
  }

  if (
    summary.metrics_status !== 'resolved' ||
    !summary.all_verified ||
    summary.blocking_point_ids.length > 0 ||
    localUnverified.length > 0 ||
    localBlockingOob.length > 0
  ) {
    const blocking = Array.from(
      new Set([...(summary.blocking_point_ids ?? []), ...localUnverified, ...localBlockingOob]),
    ).sort(compareStrings);
    const status: MetricsStatus =
      summary.metrics_status !== 'resolved'
        ? summary.metrics_status
        : localUnverified.length > 0 && localBlockingOob.length > 0
          ? 'unresolved_unverified_and_out_of_bounds'
          : localUnverified.length > 0
            ? 'unresolved_unverified_points'
            : 'unresolved_out_of_bounds';
    return {
      selectedSeries,
      availableSeries,
      displayStatus: status,
      unresolvedReason:
        summary.unresolved_reason ??
        `Series '${selectedSeries}' metrics withheld until blocking points [${blocking.join(', ')}] are verified/resolved.`,
      blockingPointIds: blocking,
      pkMetrics: null,
    };
  }

  return {
    selectedSeries,
    availableSeries,
    displayStatus: 'resolved',
    unresolvedReason: null,
    blockingPointIds: [],
    pkMetrics: summary.pk_metrics,
  };
}

export function applyPointPixelEdit(
  state: DigitizerViewState,
  idx: number,
  field: 'u_px' | 'v_px',
  val: number,
  xAx: AxisCalibrationSpec,
  yAx: AxisCalibrationSpec,
  plotRoi?: PlotROISpec | null,
): DigitizerViewState {
  if (!Number.isFinite(val)) return state;
  const effectiveY: AxisCalibrationSpec = { ...yAx, scale: state.yScale };
  return {
    ...state,
    points: updatePointPixel(state.points, idx, field, val, xAx, effectiveY, plotRoi),
    isDirty: true,
    seriesByKey: null,
  };
}

export function applyAddDigitizerPoint(
  state: DigitizerViewState,
  xAx: AxisCalibrationSpec,
  yAx: AxisCalibrationSpec,
  plotRoi?: PlotROISpec | null,
): DigitizerViewState {
  const effectiveY: AxisCalibrationSpec = { ...yAx, scale: state.yScale };
  return {
    ...state,
    points: addDigitizerPoint(state.points, xAx, effectiveY, plotRoi, state.selectedSeries),
    isDirty: true,
    seriesByKey: null,
  };
}

export function applyRemoveDigitizerPoint(
  state: DigitizerViewState,
  idx: number,
): DigitizerViewState {
  if (state.points.length <= 2) return state;
  return {
    ...state,
    points: removeDigitizerPoint(state.points, idx),
    isDirty: true,
    seriesByKey: null,
  };
}

export interface SvgOverlayGeometry {
  linePath: string;
  areaPath: string;
  markers: {
    id: string;
    cx: number;
    cy: number;
    yLo: number | null;
    yHi: number | null;
    verified: boolean;
    outOfBounds: boolean;
    extrapolated: boolean;
    previewOnly: boolean;
  }[];
}

export function computeSvgOverlayGeometry(
  points: DigitizedPoint[],
  yScale: AxisScale,
  dims = { width: 680, height: 240, padL: 54, padR: 24, padT: 18, padB: 36 },
): SvgOverlayGeometry {
  const { width: W, height: H, padL, padR, padT, padB } = dims;
  const validPts = points.filter(
    (p): p is DigitizedPoint & { x: number; y: number } =>
      p.x != null && Number.isFinite(p.x) && p.y != null && Number.isFinite(p.y),
  );
  const sortedPts = [...validPts].sort((a, b) => a.x - b.x);
  if (sortedPts.length === 0) {
    return { linePath: '', areaPath: '', markers: [] };
  }

  const xMin = Math.min(0, ...sortedPts.map(p => p.x));
  const xMax = Math.max(1, ...sortedPts.map(p => p.x)) * 1.04;
  const yMax =
    Math.max(
      1,
      ...sortedPts.map(p =>
        p.y_high != null && Number.isFinite(p.y_high) ? Math.max(p.y, p.y_high) : p.y,
      ),
    ) * 1.12;
  const positiveYs = sortedPts
    .flatMap(p => [p.y, p.y_low != null && p.y_low > 0 ? p.y_low : p.y])
    .filter(v => v > 0);
  const yPosMin = positiveYs.length > 0 ? Math.min(...positiveYs) * 0.5 : 0.1;

  const sx = (x: number) => padL + ((x - xMin) / (xMax - xMin || 1)) * (W - padL - padR);
  const sy = (y: number) => {
    if (yScale === 'log10') {
      const clamped = Math.max(yPosMin, y);
      const lMin = Math.log10(Math.max(1e-6, yPosMin));
      const lMax = Math.log10(Math.max(yPosMin * 10, yMax));
      return H - padB - ((Math.log10(clamped) - lMin) / (lMax - lMin || 1)) * (H - padT - padB);
    }
    return H - padB - (y / (yMax || 1)) * (H - padT - padB);
  };

  const linePath = sortedPts
    .map((p, i) => `${i === 0 ? 'M' : 'L'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`)
    .join(' ');
  const baselineY = sy(yScale === 'log10' ? yPosMin : 0);
  const areaPath =
    sortedPts.length >= 2
      ? `${linePath} L${sx(sortedPts[sortedPts.length - 1].x).toFixed(1)},${baselineY.toFixed(1)} L${sx(sortedPts[0].x).toFixed(1)},${baselineY.toFixed(1)} Z`
      : '';

  const markers = sortedPts.map(p => {
    const hasBounds =
      p.y_low != null &&
      Number.isFinite(p.y_low) &&
      p.y_high != null &&
      Number.isFinite(p.y_high);
    return {
      id: p.id,
      cx: sx(p.x),
      cy: sy(p.y),
      yLo: hasBounds ? sy(Math.max(yScale === 'log10' ? yPosMin : 0, p.y_low!)) : null,
      yHi: hasBounds ? sy(p.y_high!) : null,
      verified: p.verified,
      outOfBounds: p.out_of_bounds === true,
      extrapolated: Boolean(p.extrapolated),
      previewOnly: Boolean(p.previewOnly),
    };
  });

  return { linePath, areaPath, markers };
}
