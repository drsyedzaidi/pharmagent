/**
 * Pure TypeScript geometry, validation, and state helpers for PlotDigitizerCard.
 * Tested directly by `frontend/tests/plotDigitizer.test.ts`.
 */

export type AxisScale = 'linear' | 'log10';
export type ErrorBarType = 'se' | 'sd' | 'ci95';

export interface AxisCalibrationSpec {
  p1_px: number;
  p2_px: number;
  v1: number;
  v2: number;
  scale: AxisScale;
  label: string;
  unit: string;
}

export interface DigitizedPoint {
  id: string;
  index?: number;
  series?: string;
  u_px: number;
  v_px: number;
  v_low_px: number | null;
  v_high_px: number | null;
  x: number;
  y: number;
  sigma_x: number;
  sigma_y_cal: number;
  cal_se_y?: number;
  se_bar: number | null;
  bar_se_y?: number | null;
  total_se_y: number;
  ci95_y: [number, number];
  out_of_bounds_x?: boolean;
  out_of_bounds_y?: boolean;
  out_of_bounds?: boolean;
  origin: 'manual' | 'contour_proposed';
  verified: boolean;
}

export interface DerivedPkMetrics {
  cmax: number | null;
  cmax_sigma: number | null;
  tmax: number | null;
  auclast: number | null;
  auclast_sigma: number | null;
  auclast_ci95: [number, number] | null;
}

export interface DigitizeResult {
  status: string;
  figure_label?: string;
  source_sha256?: string | null;
  export_sha256?: string;
  csv_export?: string;
  pixel_sigma: number;
  anchor_sigma_px?: number;
  error_type: ErrorBarType;
  strict_bounds?: boolean;
  x_axis: AxisCalibrationSpec;
  y_axis: AxisCalibrationSpec;
  n_points: number;
  n_verified: number;
  all_verified: boolean;
  n_out_of_bounds: number;
  warnings?: string[];
  points: DigitizedPoint[];
  derived_pk: DerivedPkMetrics;
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

export function isPixelInBounds(ax: AxisCalibrationSpec, px: number, tolPx = 1e-6): boolean {
  const lo = Math.min(ax.p1_px, ax.p2_px) - Math.max(0, tolPx);
  const hi = Math.max(ax.p1_px, ax.p2_px) + Math.max(0, tolPx);
  return px >= lo && px <= hi;
}

export function isPointOutOfBounds(
  pt: Pick<DigitizedPoint, 'u_px' | 'v_px' | 'v_low_px' | 'v_high_px'>,
  xAx: AxisCalibrationSpec,
  yAx: AxisCalibrationSpec,
): boolean {
  if (!isPixelInBounds(xAx, pt.u_px) || !isPixelInBounds(yAx, pt.v_px)) return true;
  if (pt.v_low_px != null && !isPixelInBounds(yAx, pt.v_low_px)) return true;
  if (pt.v_high_px != null && !isPixelInBounds(yAx, pt.v_high_px)) return true;
  return false;
}

export function updatePointPixel(
  points: DigitizedPoint[],
  idx: number,
  field: 'u_px' | 'v_px',
  val: number,
): DigitizedPoint[] {
  if (!Number.isFinite(val)) return points;
  return points.map((p, i) =>
    i === idx ? { ...p, [field]: val, origin: 'manual', verified: true } : p,
  );
}

export function togglePointVerified(points: DigitizedPoint[], idx: number): DigitizedPoint[] {
  return points.map((p, i) => (i === idx ? { ...p, verified: !p.verified } : p));
}

export function addDigitizerPoint(points: DigitizedPoint[]): DigitizedPoint[] {
  const last = points[points.length - 1];
  const nextId = `pt_${points.length + 1}`;
  const nextPt: DigitizedPoint = {
    id: nextId,
    u_px: last ? last.u_px + 25 : 150,
    v_px: last ? last.v_px : 250,
    v_low_px: null,
    v_high_px: null,
    x: last ? last.x + 1 : 0,
    y: last ? last.y : 0,
    sigma_x: 0,
    sigma_y_cal: 0,
    se_bar: null,
    total_se_y: 0,
    ci95_y: [0, 0],
    out_of_bounds: false,
    origin: 'manual',
    verified: true,
  };
  return [...points, nextPt];
}

export function removeDigitizerPoint(points: DigitizedPoint[], idx: number): DigitizedPoint[] {
  if (points.length <= 2) return points;
  return points.filter((_, i) => i !== idx);
}

export interface SvgOverlayGeometry {
  linePath: string;
  areaPath: string;
  markers: {
    id: string;
    cx: number;
    cy: number;
    yLo: number;
    yHi: number;
    verified: boolean;
    outOfBounds: boolean;
  }[];
}

export function computeSvgOverlayGeometry(
  points: DigitizedPoint[],
  yScale: AxisScale,
  dims = { width: 680, height: 240, padL: 54, padR: 24, padT: 18, padB: 36 },
): SvgOverlayGeometry {
  const { width: W, height: H, padL, padR, padT, padB } = dims;
  const sortedPts = [...points].sort((a, b) => a.x - b.x);
  if (sortedPts.length === 0) {
    return { linePath: '', areaPath: '', markers: [] };
  }

  const xMin = Math.min(0, ...sortedPts.map(p => p.x));
  const xMax = Math.max(1, ...sortedPts.map(p => p.x)) * 1.04;
  const yMax = Math.max(1, ...sortedPts.map(p => p.y + p.total_se_y)) * 1.12;
  const positiveYs = sortedPts.map(p => p.y).filter(v => v > 0);
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

  const markers = sortedPts.map(p => ({
    id: p.id,
    cx: sx(p.x),
    cy: sy(p.y),
    yLo: sy(Math.max(yScale === 'log10' ? yPosMin : 0, p.y - p.total_se_y)),
    yHi: sy(p.y + p.total_se_y),
    verified: p.verified,
    outOfBounds: Boolean(p.out_of_bounds),
  }));

  return { linePath, areaPath, markers };
}
