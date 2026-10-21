import assert from 'node:assert/strict';
import test from 'node:test';
import {
  addDigitizerPoint,
  applyAddDigitizerPoint,
  applyPointPixelEdit,
  computeSvgOverlayGeometry,
  createDigitizerViewState,
  evaluatePointBounds,
  getSeriesDisplayMetrics,
  isPointExtrapolated,
  isPointOutOfBounds,
  pixelToValue,
  removeDigitizerPoint,
  selectDigitizerSeries,
  syncDigitizerViewState,
  togglePointVerified,
  updatePointPixel,
  validateAxisSpec,
  validatePlotRoi,
  valueToPixel,
  type AxisCalibrationSpec,
  type DigitizedPoint,
  type DigitizeResult,
  type PlotROISpec,
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

const SAMPLE_PLOT_ROI: PlotROISpec = {
  u_min_px: 140,
  u_max_px: 650,
  v_min_px: 100,
  v_max_px: 430,
};

const SAMPLE_POINTS: DigitizedPoint[] = [
  {
    id: 'pt_1',
    series: 'Oral_100mg',
    u_px: 150,
    v_px: 420,
    v_low_px: 420,
    v_high_px: 420,
    x: 0,
    y: 0,
    y_low: 0,
    y_high: 0,
    extrapolated: false,
    bounds_status: 'in_bounds',
    out_of_bounds: false,
    origin: 'manual',
    verified: true,
  },
  {
    id: 'pt_2',
    series: 'Oral_100mg',
    u_px: 190,
    v_px: 180,
    v_low_px: 195,
    v_high_px: 165,
    x: 2,
    y: 80,
    y_low: 75,
    y_high: 85,
    extrapolated: false,
    bounds_status: 'in_bounds',
    out_of_bounds: false,
    origin: 'manual',
    verified: true,
  },
  {
    id: 'pt_3',
    series: 'Oral_100mg',
    u_px: 630,
    v_px: 411,
    v_low_px: 414,
    v_high_px: 408,
    x: 24,
    y: 3,
    y_low: 2,
    y_high: 4,
    extrapolated: false,
    bounds_status: 'in_bounds',
    out_of_bounds: false,
    origin: 'manual',
    verified: true,
  },
];

const SAMPLE_RESULT: DigitizeResult = {
  status: 'ok',
  metrics_status: 'resolved',
  unresolved_reason: null,
  blocking_point_ids: [],
  export_sha256: 'sha_initial_111',
  pixel_sigma: 1.5,
  error_type: 'se',
  strict_bounds: false,
  plot_roi: SAMPLE_PLOT_ROI,
  bounds_status: 'in_bounds',
  x_axis: SAMPLE_X_AXIS,
  y_axis: SAMPLE_Y_AXIS_REVERSED,
  n_points: 3,
  n_verified: 3,
  all_verified: true,
  n_extrapolated: 0,
  n_out_of_bounds: 0,
  points: SAMPLE_POINTS,
  series: {
    Oral_100mg: {
      series: 'Oral_100mg',
      status: 'ok',
      metrics_status: 'resolved',
      unresolved_reason: null,
      blocking_point_ids: [],
      unverified_point_ids: [],
      blocking_oob_point_ids: [],
      recorded_oob_overrides: [],
      n_points: 3,
      n_verified: 3,
      all_verified: true,
      n_extrapolated: 0,
      n_out_of_bounds: 0,
      bounds_status: 'in_bounds',
      x_range: [0, 24],
      y_range: [0, 80],
      pk_metrics: {
        cmax: 80,
        tmax: 2,
        auclast: 993,
      },
    },
  },
  series_pk: {
    Oral_100mg: {
      cmax: 80,
      tmax: 2,
      auclast: 993,
    },
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

test('validateAxisSpec and validatePlotRoi reject degenerate geometries', () => {
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
  assert.equal(validatePlotRoi(SAMPLE_PLOT_ROI), null);
  assert.match(
    validatePlotRoi({ ...SAMPLE_PLOT_ROI, u_max_px: 100 }) ?? '',
    /horizontal span/,
  );
});

test('calibration anchors (extrapolated) are separated from plot_roi (bounds_status)', () => {
  // Interior calibration ticks: X=[200..450], while plot_roi spans U=[140..650]
  const interiorX: AxisCalibrationSpec = {
    ...SAMPLE_X_AXIS,
    p1_px: 200,
    p2_px: 450,
    v1: 2,
    v2: 12,
  };
  // Point at u_px = 150 is outside anchor span [200..450] (extrapolated), but inside plot_roi [140..650]
  const ptExtrapInRoi = { ...SAMPLE_POINTS[0], u_px: 150, v_px: 300 };
  assert.equal(isPointExtrapolated(ptExtrapInRoi, interiorX, SAMPLE_Y_AXIS_REVERSED), true);
  assert.equal(isPointOutOfBounds(ptExtrapInRoi, SAMPLE_PLOT_ROI), false);
  assert.deepEqual(evaluatePointBounds(ptExtrapInRoi, SAMPLE_PLOT_ROI), {
    bounds_status: 'in_bounds',
    out_of_bounds: false,
    out_of_bounds_x: false,
    out_of_bounds_y: false,
  });

  // When plot_roi is omitted (null/undefined), bounds_status is "unknown" rather than inferred
  assert.equal(isPointOutOfBounds(ptExtrapInRoi, null), false);
  assert.deepEqual(evaluatePointBounds(ptExtrapInRoi, null), {
    bounds_status: 'unknown',
    out_of_bounds: null,
    out_of_bounds_x: null,
    out_of_bounds_y: null,
  });

  // Point outside plot_roi (u_px = 700 > 650) is flagged out_of_bounds
  const ptOutsideRoi = { ...SAMPLE_POINTS[1], u_px: 700 };
  assert.equal(isPointOutOfBounds(ptOutsideRoi, SAMPLE_PLOT_ROI), true);
  assert.equal(evaluatePointBounds(ptOutsideRoi, SAMPLE_PLOT_ROI).bounds_status, 'out_of_bounds');
});

test('unverified or out-of-bounds points withhold series PK metrics until verified', () => {
  const unverifiedResult: DigitizeResult = {
    ...SAMPLE_RESULT,
    status: 'unresolved_unverified_points',
    metrics_status: 'unresolved_unverified_points',
    unresolved_reason: "Series 'Oral_100mg' metrics withheld: unverified points ['pt_spike'].",
    blocking_point_ids: ['pt_spike'],
    all_verified: false,
    n_points: 4,
    n_verified: 3,
    points: [
      ...SAMPLE_POINTS,
      {
        id: 'pt_spike',
        series: 'Oral_100mg',
        u_px: 210,
        v_px: 120,
        v_low_px: null,
        v_high_px: null,
        x: 3,
        y: 100,
        y_low: null,
        y_high: null,
        extrapolated: false,
        bounds_status: 'in_bounds',
        out_of_bounds: false,
        origin: 'contour_proposed',
        verified: false,
      },
    ],
    series: {
      Oral_100mg: {
        ...SAMPLE_RESULT.series.Oral_100mg,
        status: 'unresolved_unverified_points',
        metrics_status: 'unresolved_unverified_points',
        unresolved_reason: "Series 'Oral_100mg' metrics withheld: unverified points ['pt_spike'].",
        blocking_point_ids: ['pt_spike'],
        unverified_point_ids: ['pt_spike'],
        all_verified: false,
        n_points: 4,
        n_verified: 3,
        pk_metrics: { cmax: null, tmax: null, auclast: null },
      },
    },
    series_pk: {
      Oral_100mg: { cmax: null, tmax: null, auclast: null },
    },
  };

  const state = createDigitizerViewState(unverifiedResult);
  const display = getSeriesDisplayMetrics(state);
  assert.equal(display.selectedSeries, 'Oral_100mg');
  assert.equal(display.displayStatus, 'unresolved_unverified_points');
  assert.equal(display.pkMetrics, null);
  assert.deepEqual(display.blockingPointIds, ['pt_spike']);
  assert.match(display.unresolvedReason ?? '', /pt_spike/);
});

test('multi-series keyed state is invariant to series reordering and supports explicit selection', () => {
  const seriesA: DigitizedPoint[] = [
    {
      id: 'a_1',
      series: 'Cohort_A_10mg',
      u_px: 150,
      v_px: 420,
      v_low_px: null,
      v_high_px: null,
      x: 0,
      y: 0,
      y_low: null,
      y_high: null,
      bounds_status: 'in_bounds',
      out_of_bounds: false,
      origin: 'manual',
      verified: true,
    },
    {
      id: 'a_2',
      series: 'Cohort_A_10mg',
      u_px: 230,
      v_px: 300,
      v_low_px: null,
      v_high_px: null,
      x: 4,
      y: 40,
      y_low: null,
      y_high: null,
      bounds_status: 'in_bounds',
      out_of_bounds: false,
      origin: 'manual',
      verified: true,
    },
  ];
  const seriesB: DigitizedPoint[] = [
    {
      id: 'b_1',
      series: 'Cohort_B_100mg',
      u_px: 150,
      v_px: 420,
      v_low_px: null,
      v_high_px: null,
      x: 0,
      y: 0,
      y_low: null,
      y_high: null,
      bounds_status: 'in_bounds',
      out_of_bounds: false,
      origin: 'manual',
      verified: true,
    },
    {
      id: 'b_2',
      series: 'Cohort_B_100mg',
      u_px: 190,
      v_px: 180,
      v_low_px: null,
      v_high_px: null,
      x: 2,
      y: 80,
      y_low: null,
      y_high: null,
      bounds_status: 'in_bounds',
      out_of_bounds: false,
      origin: 'manual',
      verified: true,
    },
  ];

  const summaryA = {
    ...SAMPLE_RESULT.series.Oral_100mg,
    series: 'Cohort_A_10mg',
    n_points: 2,
    n_verified: 2,
    pk_metrics: { cmax: 40, tmax: 4, auclast: 80 },
  };
  const summaryB = {
    ...SAMPLE_RESULT.series.Oral_100mg,
    series: 'Cohort_B_100mg',
    n_points: 2,
    n_verified: 2,
    pk_metrics: { cmax: 80, tmax: 2, auclast: 80 },
  };

  const resAB: DigitizeResult = {
    ...SAMPLE_RESULT,
    points: [...seriesA, ...seriesB],
    series: { Cohort_A_10mg: summaryA, Cohort_B_100mg: summaryB },
    series_pk: { Cohort_A_10mg: summaryA.pk_metrics, Cohort_B_100mg: summaryB.pk_metrics },
  };
  const resBA: DigitizeResult = {
    ...SAMPLE_RESULT,
    points: [...seriesB, ...seriesA],
    series: { Cohort_B_100mg: summaryB, Cohort_A_10mg: summaryA },
    series_pk: { Cohort_B_100mg: summaryB.pk_metrics, Cohort_A_10mg: summaryA.pk_metrics },
  };

  const stateAB = createDigitizerViewState(resAB);
  const stateBA = createDigitizerViewState(resBA);

  assert.deepEqual(stateAB.availableSeries, ['Cohort_A_10mg', 'Cohort_B_100mg']);
  assert.deepEqual(stateAB.availableSeries, stateBA.availableSeries);
  assert.equal(stateAB.selectedSeries, 'Cohort_A_10mg');
  assert.equal(stateBA.selectedSeries, 'Cohort_A_10mg');
  assert.deepEqual(stateAB.points, stateBA.points);
  assert.deepEqual(getSeriesDisplayMetrics(stateAB), getSeriesDisplayMetrics(stateBA));
  assert.equal(getSeriesDisplayMetrics(stateAB).pkMetrics?.cmax, 40);

  // Explicit series selection switches to Cohort_B_100mg
  const selectedB = selectDigitizerSeries(stateAB, 'Cohort_B_100mg');
  assert.equal(selectedB.selectedSeries, 'Cohort_B_100mg');
  assert.equal(getSeriesDisplayMetrics(selectedB).pkMetrics?.cmax, 80);
});

test('edit-before-recalibration computes preview coordinates and marks series PK stale', () => {
  const initial = createDigitizerViewState(SAMPLE_RESULT);
  assert.equal(initial.isDirty, false);
  assert.equal(getSeriesDisplayMetrics(initial).pkMetrics?.cmax, 80);

  // Edit pt_1 u_px from 150 -> 270 (which corresponds to x = 6.0 h on SAMPLE_X_AXIS)
  const afterEdit = applyPointPixelEdit(
    initial,
    0,
    'u_px',
    270,
    SAMPLE_X_AXIS,
    SAMPLE_Y_AXIS_REVERSED,
    SAMPLE_PLOT_ROI,
  );
  assert.equal(afterEdit.isDirty, true);
  assert.equal(getSeriesDisplayMetrics(afterEdit).displayStatus, 'stale');
  assert.equal(getSeriesDisplayMetrics(afterEdit).pkMetrics, null);
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
    SAMPLE_PLOT_ROI,
  );
  assert.equal(afterAdd.isDirty, true);
  assert.equal(getSeriesDisplayMetrics(afterAdd).pkMetrics, null);
  assert.equal(afterAdd.points.length, 4);
  const newPt = afterAdd.points[3];
  assert.equal(newPt.id, 'pt_4');
  assert.equal(newPt.series, 'Oral_100mg');
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
  assert.equal(toggled[0].verified, false);

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
    SAMPLE_PLOT_ROI,
  );
  assert.equal(dirty.isDirty, true);
  assert.equal(dirty.points[1].y, 50);
  assert.equal(dirty.points[1].previewOnly, true);
  assert.equal(getSeriesDisplayMetrics(dirty).pkMetrics, null);

  // Same backend prop does not clobber local edits before recalibration completes
  const unchangedSync = syncDigitizerViewState(dirty, SAMPLE_RESULT);
  assert.equal(unchangedSync.isDirty, true);
  assert.equal(unchangedSync.points[1].previewOnly, true);

  // When backend returns updated DigitizeResult after recalibration, state synchronizes
  const nextBackendResult: DigitizeResult = {
    ...SAMPLE_RESULT,
    export_sha256: 'sha_recomputed_222',
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
    series: {
      Oral_100mg: {
        ...SAMPLE_RESULT.series.Oral_100mg,
        y_range: [0, 50],
        pk_metrics: {
          cmax: 50,
          tmax: 2,
          auclast: 618,
        },
      },
    },
    series_pk: {
      Oral_100mg: {
        cmax: 50,
        tmax: 2,
        auclast: 618,
      },
    },
  };

  const synced = syncDigitizerViewState(dirty, nextBackendResult);
  assert.equal(synced.isDirty, false);
  assert.equal(synced.points[1].y, 50);
  assert.equal(synced.points[1].previewOnly, false);
  assert.equal(getSeriesDisplayMetrics(synced).pkMetrics?.cmax, 50);
  assert.equal(getSeriesDisplayMetrics(synced).pkMetrics?.auclast, 618);
});

test('computeSvgOverlayGeometry builds valid SVG paths and directly digitized bounds whiskers', () => {
  const geomLin = computeSvgOverlayGeometry(SAMPLE_POINTS, 'linear');
  assert.ok(geomLin.linePath.startsWith('M'));
  assert.ok(geomLin.areaPath.endsWith('Z'));
  assert.equal(geomLin.markers.length, 3);
  assert.ok(geomLin.markers[1].cy < geomLin.markers[0].cy);
  assert.ok(geomLin.markers[1].yHi !== null && geomLin.markers[1].yLo !== null);
  assert.ok(geomLin.markers[1].yHi! < geomLin.markers[1].yLo!);

  const geomLog = computeSvgOverlayGeometry(SAMPLE_POINTS.slice(1), 'log10');
  assert.equal(geomLog.markers.length, 2);
  assert.ok(geomLog.markers[0].cy < geomLog.markers[1].cy);
});
