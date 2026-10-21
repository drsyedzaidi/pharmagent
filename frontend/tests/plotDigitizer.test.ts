import assert from 'node:assert/strict';
import test from 'node:test';
import {
  addDigitizerPoint,
  applyAddDigitizerPoint,
  applyPointPixelEdit,
  computeSvgOverlayGeometry,
  createDigitizerViewState,
  isPointOutOfBounds,
  pixelToValue,
  removeDigitizerPoint,
  syncDigitizerViewState,
  togglePointVerified,
  updatePointPixel,
  validateAxisSpec,
  valueToPixel,
  type AxisCalibrationSpec,
  type DigitizedPoint,
  type DigitizeResult,
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
    y_low: 0,
    y_high: 0,
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
    y_low: 75,
    y_high: 85,
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
    y_low: 2,
    y_high: 4,
    out_of_bounds: false,
    origin: 'manual',
    verified: true,
  },
];

const SAMPLE_RESULT: DigitizeResult = {
  status: 'ok',
  export_sha256: 'sha_initial_111',
  pixel_sigma: 1.5,
  error_type: 'se',
  strict_bounds: false,
  x_axis: SAMPLE_X_AXIS,
  y_axis: SAMPLE_Y_AXIS_REVERSED,
  n_points: 3,
  n_verified: 1,
  all_verified: false,
  n_out_of_bounds: 0,
  points: SAMPLE_POINTS,
  derived_pk: {
    cmax: 80,
    tmax: 2,
    auclast: 993,
  },
};

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

test('edit-before-recalibration computes preview coordinates and marks derived PK stale', () => {
  const initial = createDigitizerViewState(SAMPLE_RESULT);
  assert.equal(initial.isDirty, false);
  assert.equal(initial.derivedPk?.cmax, 80);

  // Edit pt_1 u_px from 150 -> 270 (which corresponds to x = 6.0 h on SAMPLE_X_AXIS)
  const afterEdit = applyPointPixelEdit(
    initial,
    0,
    'u_px',
    270,
    SAMPLE_X_AXIS,
    SAMPLE_Y_AXIS_REVERSED,
  );
  assert.equal(afterEdit.isDirty, true);
  assert.equal(afterEdit.derivedPk, null); // Stale derived PK hidden until backend recalibration
  assert.equal(afterEdit.points[0].u_px, 270);
  assert.equal(afterEdit.points[0].x, 6);
  assert.equal(afterEdit.points[0].previewOnly, true);
  assert.equal(afterEdit.points[0].origin, 'manual');
  assert.equal(afterEdit.points[0].verified, true);

  // Adding a point with axes computes preview coordinates (never fake 0s) and marks PK stale
  const afterAdd = applyAddDigitizerPoint(
    initial,
    SAMPLE_X_AXIS,
    SAMPLE_Y_AXIS_REVERSED,
  );
  assert.equal(afterAdd.isDirty, true);
  assert.equal(afterAdd.derivedPk, null);
  assert.equal(afterAdd.points.length, 4);
  const newPt = afterAdd.points[3];
  assert.equal(newPt.id, 'pt_4');
  assert.equal(newPt.previewOnly, true);
  assert.equal(newPt.x, 24);
  assert.equal(newPt.y, 3);

  // Adding a point without axes sets x/y to null rather than fake zeroes
  const addedNoAxes = addDigitizerPoint(SAMPLE_POINTS);
  assert.equal(addedNoAxes[3].x, null);
  assert.equal(addedNoAxes[3].y, null);
  assert.equal(addedNoAxes[3].previewOnly, true);

  // Editing without axes clears x/y to null so stale values are never displayed
  const editedNoAxes = updatePointPixel(SAMPLE_POINTS, 1, 'v_px', 270);
  assert.equal(editedNoAxes[1].x, null);
  assert.equal(editedNoAxes[1].y, null);
  assert.equal(editedNoAxes[1].previewOnly, true);

  const toggled = togglePointVerified(SAMPLE_POINTS, 0);
  assert.equal(toggled[0].verified, true);

  const afterOneDelete = removeDigitizerPoint(SAMPLE_POINTS, 0);
  assert.equal(afterOneDelete.length, 2);
  const afterSecondDelete = removeDigitizerPoint(afterOneDelete, 0);
  assert.equal(afterSecondDelete.length, 2);
});

test('prop-update synchronization replaces local preview state with authoritative backend result', () => {
  const initial = createDigitizerViewState(SAMPLE_RESULT);
  const dirty = applyPointPixelEdit(
    initial,
    1,
    'v_px',
    270, // preview y = 50
    SAMPLE_X_AXIS,
    SAMPLE_Y_AXIS_REVERSED,
  );
  assert.equal(dirty.isDirty, true);
  assert.equal(dirty.points[1].y, 50);
  assert.equal(dirty.points[1].previewOnly, true);
  assert.equal(dirty.derivedPk, null);

  // Same backend prop does not clobber local edits before recalibration completes
  const unchangedSync = syncDigitizerViewState(dirty, SAMPLE_RESULT);
  assert.equal(unchangedSync.isDirty, true);
  assert.equal(unchangedSync.points[1].previewOnly, true);

  // When backend returns updated DigitizeResult after recalibration, state synchronizes
  const nextBackendResult: DigitizeResult = {
    ...SAMPLE_RESULT,
    export_sha256: 'sha_recomputed_222',
    n_verified: 2,
    points: [
      SAMPLE_POINTS[0],
      {
        ...SAMPLE_POINTS[1],
        v_px: 270,
        y: 50,
        y_low: 45,
        y_high: 55,
        origin: 'manual',
        verified: true,
      },
      SAMPLE_POINTS[2],
    ],
    derived_pk: {
      cmax: 50,
      tmax: 2,
      auclast: 618,
    },
  };

  const synced = syncDigitizerViewState(dirty, nextBackendResult);
  assert.equal(synced.isDirty, false);
  assert.equal(synced.points[1].y, 50);
  assert.equal(synced.points[1].previewOnly, false);
  assert.equal(synced.derivedPk?.cmax, 50);
  assert.equal(synced.derivedPk?.auclast, 618);
});

test('computeSvgOverlayGeometry builds valid SVG paths and directly digitized bounds whiskers', () => {
  const geomLin = computeSvgOverlayGeometry(SAMPLE_POINTS, 'linear');
  assert.ok(geomLin.linePath.startsWith('M'));
  assert.ok(geomLin.areaPath.endsWith('Z'));
  assert.equal(geomLin.markers.length, 3);
  // Peak at pt_2 (y=80) has higher screen elevation (smaller cy) than baseline pt_1 (y=0)
  assert.ok(geomLin.markers[1].cy < geomLin.markers[0].cy);
  // Upper whisker yHi is above lower whisker yLo on screen (yHi < yLo)
  assert.ok(geomLin.markers[1].yHi !== null && geomLin.markers[1].yLo !== null);
  assert.ok(geomLin.markers[1].yHi! < geomLin.markers[1].yLo!);

  const geomLog = computeSvgOverlayGeometry(SAMPLE_POINTS.slice(1), 'log10');
  assert.equal(geomLog.markers.length, 2);
  assert.ok(geomLog.markers[0].cy < geomLog.markers[1].cy);
});
