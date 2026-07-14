import type { FlexplotSpec, FlexplotData } from '../types';
import type { ChartOpts } from './FlexplotChart';

type Props = {
  spec: FlexplotSpec;
  onSpecChange: (patch: Partial<FlexplotSpec>) => void;
  opts: ChartOpts;
  onOptsChange: (patch: Partial<ChartOpts>) => void;
  kind: FlexplotData['kind'] | null;
  disabled: boolean;
};

/**
 * Graphic Options row mirroring jamovi's flexplot. Fetch-affecting controls
 * (fit type, center/spread, ghost line) live in `spec`; pure display controls
 * (confidence bands, log Y, transparency) live in `opts` and never re-fetch.
 */
export function GraphicOptions({ spec, onSpecChange, opts, onOptsChange, kind, disabled }: Props) {
  const isScatter = kind === 'scatter';
  const isDot = kind === 'dotplot';
  const isUnivariate = kind === 'histogram' || kind === 'density';

  return (
    <div className="flexplot-options">
      {/* Confidence bands (display-only show/hide of the precomputed band) */}
      <label className="flexplot-check" title="Show the fit's confidence band (scatterplots)">
        <input type="checkbox" checked={opts.showCI} disabled={disabled || !isScatter}
               onChange={e => onOptsChange({ showCI: e.target.checked })} />
        Confidence bands
      </label>

      {/* Ghost line (fetch: the server computes the reference fit). Needs a
          paneled variable — the reference fit is echoed across panels. */}
      <label className="flexplot-check"
             title={spec.panel_by
               ? 'Echo a faint reference fit onto every panel for comparison'
               : 'Pick a Paneled variable to enable the ghost reference line'}>
        <input type="checkbox" checked={spec.ghost} disabled={disabled || !isScatter || !spec.panel_by}
               onChange={e => onSpecChange({ ghost: e.target.checked })} />
        Ghost line
      </label>

      {/* Residualize — reserved for a future multi-predictor release */}
      <label className="flexplot-check flexplot-check-off"
             title="Requires multiple predictors — available in a later release">
        <input type="checkbox" checked={false} disabled />
        Residualize predictor
      </label>

      {/* Univariate view: histogram vs KDE density */}
      {isUnivariate && (
        <label className="flexplot-inline">
          <span>Show</span>
          <select className="model-select flexplot-mini"
                  value={spec.geom === 'density' ? 'density' : 'points'} disabled={disabled}
                  onChange={e => onSpecChange({ geom: e.target.value as FlexplotSpec['geom'] })}>
            <option value="points">Histogram</option>
            <option value="density">Density</option>
          </select>
        </label>
      )}

      {/* Fitted line */}
      <label className="flexplot-inline">
        <span>Fitted line</span>
        <select className="model-select flexplot-mini" value={spec.fit} disabled={disabled || !isScatter}
                onChange={e => onSpecChange({ fit: e.target.value as FlexplotSpec['fit'] })}>
          <option value="loess">Loess</option>
          <option value="linear">Regression</option>
          <option value="none">None</option>
        </select>
      </label>

      {/* Center / spread (dotplots) */}
      <label className="flexplot-inline">
        <span>Center/spread</span>
        <select className="model-select flexplot-mini" value={spec.center} disabled={disabled || !isDot}
                onChange={e => onSpecChange({ center: e.target.value as FlexplotSpec['center'] })}>
          <option value="median_iqr">Median + IQR</option>
          <option value="mean_se">Mean + sterr</option>
          <option value="mean_sd">Mean + SD</option>
        </select>
      </label>

      {/* Log Y (display-only) */}
      <label className="flexplot-check" title="Log-scale the outcome axis">
        <input type="checkbox" checked={opts.logY} disabled={disabled}
               onChange={e => onOptsChange({ logY: e.target.checked })} />
        Log Y
      </label>

      {/* Transparency (display-only). The slider IS transparency: handle
          position and the % readout both track 1 - alpha. */}
      <label className="flexplot-inline" title="Point transparency">
        <span>Transparency</span>
        <input type="range" min={0} max={95} step={5} value={Math.round((1 - opts.alpha) * 100)}
               disabled={disabled}
               onChange={e => onOptsChange({ alpha: 1 - Number(e.target.value) / 100 })} />
        <span className="flexplot-range-val">{Math.round((1 - opts.alpha) * 100)}%</span>
      </label>
    </div>
  );
}
