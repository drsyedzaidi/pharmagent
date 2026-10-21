import assert from 'node:assert/strict';
import test from 'node:test';
import {
  addDigitizerPoint,
  computeSvgOverlayGeometry,
  isPointOutOfBounds,
  pixelToValue,
  removeDigitizerPoint,
  togglePointVerified,
  updatePointPixel,
  validateAxisSpec,
  valueToPixel,
  type AxisCalibrationSpec,
  type DigitizedPoint,
} from '../src/digitizer/digitizeMath.ts';

const SAMPLE_X_AXIS: AxisCalibrationSpec = {
  p1_px: 150,
  p2_px: 630,
  v1: 0,
  v2: 24,
  scale: 'linear',
  label: 'Time',
  unit: 'h',
};

const SAMPLE_Y_AXIS_REVERSED: AxisCalibrationSpec = {
  p1_px: 420,
  p2_px: 120,
  v1: 0,
  v2: 100,
  scale: 'linear',
  label: 'Concentration',
  unit: 'ng/mL',
};

const SAMPLE_Y_AXIS_LOG10: AxisCalibrationSpec = {
  p1_px: 500,
  p2_px: 200,
  v1: 1,
  v2: 1000,
  scale: 'log10',
  label: 'Concentration',
  unit: 'ng/mL',
};

const SAMPLE_POINTS: DigitizedPoint[] = [
  {
    id: 'pt_1',
    u_px: 150,
    v_px: 420,
    v_low_px: 420,
    v_high_px: 420,
    x: 0,
    y: 0,
    sigma_x: 0.09,
    sigma_y_cal: 0.6,
    se_bar: 0,
    total_se_y: 0.6,
    ci95_y: [-1.18, 1.18],
    out_of_bounds: false,
    origin: 'contour_proposed',
    verified: false,
  },
  {
    id: 'pt_2',
    u_px: 190,
    v_px: 180,
    v_low_px: 195,
    v_high_px: 165,
    x: 2,
    y: 80,
    sigma_x: 0.08,
    sigma_y_cal: 0.55,
    se_bar: 5.0,
    total_se_y: 5.03,
    ci95_y: [70.14, 89.86],
    out_of_bounds: false,
    origin: 'contour_proposed',
    verified: false,
  },
  {
    id: 'pt_3',
    u_px: 630,
    v_px: 411,
    v_low_px: 414,
    v_high_px: 408,
    x: 24,
    y: 3,
    sigma_x: 0.09,
    sigma_y_cal: 0.6,
    se_bar: 1.0,
    total_se_y: 1.17,
    ci95_y: [0.71, 5.29],
    out_of_bounds: false,
    origin: 'manual',
    verified: true,
  },
];

test('linear, reversed, and log10 pixel/value round-trip conversions', () => {
  assert.equal(pixelToValue(SAMPLE_X_AXIS, 150), 0);
  assert.equal(pixelToValue(SAMPLE_X_AXIS, 390), 12);
  assert.equal(valueToPixel(SAMPLE_X_AXIS, 12), 390);

  // Reversed screen Y axis (p1_px=420 -> 0, p2_px=120 -> 100)
  assert.equal(pixelToValue(SAMPLE_Y_AXIS_REVERSED, 270), 50);
  assert.equal(valueToPixel(SAMPLE_Y_AXIS_REVERSED, 50), 270);

  // Semi-log10 reversed Y axis (p1_px=500 -> 1, p2_px=200 -> 1000)
  assert.ok(Math.abs(pixelToValue(SAMPLE_Y_AXIS_LOG10, 400) - 10) < 1e-12);
  assert.ok(Math.abs(pixelToValue(SAMPLE_Y_AXIS_LOG10, 300) - 100) < 1e-12);
  assert.ok(Math.abs(valueToPixel(SAMPLE_Y_AXIS_LOG10, 100) - 300) < 1e-12);
});

test('validateAxisSpec rejects degenerate and invalid log10 axes', () => {
  assert.equal(validateAxisSpec(SAMPLE_X_AXIS), null);
  assert.match(
    validateAxisSpec({ ...SAMPLE_X_AXIS, p2_px: 150.5 }) ?? '',
    /separated by at least 1 px/,
  );
  assert.match(
    validateAxisSpec({ ...SAMPLE_X_AXIS, v1: 10, v2: 10 }) ?? '',
    /must be distinct/,
  );
  assert.match(
    validateAxisSpec({ ...SAMPLE_Y_AXIS_LOG10, v1: 0 }) ?? '',
    /strictly positive/,
  );
});

test('isPointOutOfBounds detects clicks and error bars outside calibrated span', () => {
  assert.equal(isPointOutOfBounds(SAMPLE_POINTS[1], SAMPLE_X_AXIS, SAMPLE_Y_AXIS_REVERSED), false);
  // Click outside X range (u_px = 700 > 630)
  assert.equal(
    isPointOutOfBounds({ ...SAMPLE_POINTS[1], u_px: 700 }, SAMPLE_X_AXIS, SAMPLE_Y_AXIS_REVERSED),
    true,
  );
  // Error bar extending outside Y range (v_high_px = 100 < 120)
  assert.equal(
    isPointOutOfBounds({ ...SAMPLE_POINTS[1], v_high_px: 100 }, SAMPLE_X_AXIS, SAMPLE_Y_AXIS_REVERSED),
    true,
  );
});

test('point editing, verification toggle, addition, and minimum-2-point deletion guard', () => {
  const edited = updatePointPixel(SAMPLE_POINTS, 0, 'u_px', 165);
  assert.equal(edited[0].u_px, 165);
  assert.equal(edited[0].origin, 'manual');
  assert.equal(edited[0].verified, true);

  const toggled = togglePointVerified(SAMPLE_POINTS, 0);
  assert.equal(toggled[0].verified, true);

  const added = addDigitizerPoint(SAMPLE_POINTS);
  assert.equal(added.length, 4);
  assert.equal(added[3].id, 'pt_4');
  assert.equal(added[3].verified, true);

  const afterOneDelete = removeDigitizerPoint(SAMPLE_POINTS, 0);
  assert.equal(afterOneDelete.length, 2);
  // Cannot delete below 2 points
  const afterSecondDelete = removeDigitizerPoint(afterOneDelete, 0);
  assert.equal(afterSecondDelete.length, 2);
});

test('computeSvgOverlayGeometry builds valid SVG paths and uncertainty whiskers', () => {
  const geomLin = computeSvgOverlayGeometry(SAMPLE_POINTS, 'linear');
  assert.ok(geomLin.linePath.startsWith('M'));
  assert.ok(geomLin.areaPath.endsWith('Z'));
  assert.equal(geomLin.markers.length, 3);
  // Peak at pt_2 (y=80) has higher screen elevation (smaller cy) than baseline pt_1 (y=0)
  assert.ok(geomLin.markers[1].cy < geomLin.markers[0].cy);
  // Upper whisker yHi is above lower whisker yLo on screen (yHi < yLo)
  assert.ok(geomLin.markers[1].yHi < geomLin.markers[1].yLo);

  const geomLog = computeSvgOverlayGeometry(SAMPLE_POINTS.slice(1), 'log10');
  assert.equal(geomLog.markers.length, 2);
  assert.ok(geomLog.markers[0].cy < geomLog.markers[1].cy);
});
