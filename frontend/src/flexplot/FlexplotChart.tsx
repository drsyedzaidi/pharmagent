import type { FlexplotData, FlexplotCell, FlexplotGhostLine } from '../types';
import { makeLinear, makeLog, linearTicks, decadeTicks, fmtTick, type Scale } from './scales';

// Categorical series palette — same 10 colours as SpaghettiChart (App.tsx),
// indexed by the server-assigned color_index so client and server never disagree.
const PALETTE = ['#1F66A6', '#1D7A5A', '#9A5B12', '#4A6FA5', '#B23A2E',
                 '#3B86C9', '#16604A', '#C77F2A', '#5E7388', '#2A8F8F'];
const color = (i: number) => PALETTE[((i % PALETTE.length) + PALETTE.length) % PALETTE.length];

export interface ChartOpts {
  logY: boolean;
  alpha: number;      // 0..1 point opacity (transparency slider)
  showCI: boolean;    // draw the precomputed confidence band
}

type Props = { data: FlexplotData; opts: ChartOpts };

export function FlexplotChart({ data, opts }: Props) {
  if (data.summary.n === 0) {
    return <div className="flexplot-empty">No data to plot for this selection.</div>;
  }
  const multi = data.panels.length > 1;
  // One large tile for a single panel; small multiples otherwise.
  const tw = multi ? 250 : 560;
  const th = multi ? 190 : 320;

  return (
    <figure className="flexplot-figure">
      {data.legend.length > 1 && <Legend data={data} />}
      <div className="flexplot-grid">
        {data.panels.map(panel => (
          <PanelTile key={panel.id} data={data} panelId={panel.id}
                     strip={panel.strip} w={tw} h={th} opts={opts} />
        ))}
      </div>
      <figcaption className="flexplot-caption">
        {data.kind} · n={data.summary.n}
        {data.summary.n_groups > 1 ? ` · ${data.summary.n_groups} groups` : ''}
        {data.summary.n_panels > 1 ? ` · ${data.summary.n_panels} panels` : ''}
      </figcaption>
    </figure>
  );
}

function Legend({ data }: { data: FlexplotData }) {
  return (
    <div className="flexplot-legend">
      {data.legend.map(e => (
        <span key={e.id} className="flexplot-legend-item">
          <span className="flexplot-swatch" style={{ background: color(e.color_index) }} />
          {e.label}
        </span>
      ))}
    </div>
  );
}

type TileProps = {
  data: FlexplotData; panelId: string; strip: string;
  w: number; h: number; opts: ChartOpts;
};

function PanelTile({ data, panelId, strip, w, h, opts }: TileProps) {
  const ml = 42, mr = 10, mt = strip ? 20 : 8, mb = 30;
  const cells = data.cells.filter(c => c.panel === panelId);
  const isCount = data.kind === 'histogram' || data.kind === 'density';
  const useLog = opts.logY && !isCount && data.kind !== 'histogram';

  // y-domain: shared across panels for the outcome; local [0,max] for count/density.
  let yLo = data.y_range[0], yHi = data.y_range[1];
  if (isCount) {
    yLo = 0;
    yHi = Math.max(1, ...cells.flatMap(c =>
      c.bins ? c.bins.counts : c.density ? c.density.y : [0]));
    yHi *= 1.05;
  }
  const [xLo, xHi] = data.x_range;

  // For a log axis, floor at the smallest strictly-positive outcome value so a
  // predose/BLQ zero (y_range[0] <= 0) doesn't stretch the axis over many empty
  // decades and crush the real data into a sliver.
  const logLo = (() => {
    if (!useLog) return yLo;
    const pos = cells.flatMap(c => [
      ...c.points.y.filter(v => v > 0),
      ...(c.fit ? c.fit.lo.filter(v => v > 0) : []),
    ]);
    return pos.length ? Math.min(...pos) : Math.max(yHi / 1000, 1e-9);
  })();

  const sx: Scale = makeLinear(xLo, xHi, ml, w - mr);
  const sy: Scale = useLog
    ? makeLog(logLo, yHi, h - mb, mt)
    : makeLinear(yLo, yHi, h - mb, mt);

  const yTicks = useLog ? decadeTicks(logLo, yHi) : linearTicks(yLo, yHi, 5);
  const catAxis = data.kind === 'dotplot' && data.x_categories;
  const xTicks = catAxis
    ? data.x_categories!.map((label, i) => ({ v: i, label }))
    : linearTicks(xLo, xHi, 5).map(v => ({ v, label: fmtTick(v) }));

  return (
    <svg viewBox={`0 0 ${w} ${h}`} width={w} style={{ maxWidth: w }} className="flexplot-svg"
         role="img" aria-label={`${data.kind} ${data.y_label}${data.x_label ? ' vs ' + data.x_label : ''}`}>
      {strip && (
        <text x={w / 2} y={13} textAnchor="middle" fontSize="9"
              fill="var(--text-h)" style={{ whiteSpace: 'pre' }}>
          {strip.replace('\n', ' ')}
        </text>
      )}
      {/* axes */}
      <line x1={ml} y1={h - mb} x2={w - mr} y2={h - mb} stroke="var(--border)" />
      <line x1={ml} y1={mt} x2={ml} y2={h - mb} stroke="var(--border)" />
      {yTicks.map((v, k) => {
        const yy = sy(v);
        if (yy < mt - 1 || yy > h - mb + 1) return null;
        return (
          <g key={k}>
            <line x1={ml - 3} y1={yy} x2={ml} y2={yy} stroke="var(--text-dim)" />
            <text x={ml - 5} y={yy + 3} textAnchor="end" fontSize="8" fill="var(--text-dim)">{fmtTick(v)}</text>
          </g>
        );
      })}
      {xTicks.map((t, k) => (
        <g key={k}>
          <line x1={sx(t.v)} y1={h - mb} x2={sx(t.v)} y2={h - mb + 3} stroke="var(--text-dim)" />
          <text x={sx(t.v)} y={h - mb + 11} textAnchor="middle" fontSize="8" fill="var(--text-dim)">{t.label}</text>
        </g>
      ))}
      {/* axis titles */}
      <text x={(ml + w - mr) / 2} y={h - 2} textAnchor="middle" fontSize="9" fill="var(--text-dim)">{data.x_label}</text>
      <text x={11} y={(mt + h - mb) / 2} textAnchor="middle" fontSize="9" fill="var(--text-dim)"
            transform={`rotate(-90 11 ${(mt + h - mb) / 2})`}>{data.y_label}</text>

      {/* ghost lines (faint reference fit echoed on every panel) */}
      {data.ghost_line?.map((g, i) => (
        <GhostLine key={`ghost${i}`} g={g} sx={sx} sy={sy} />
      ))}

      {/* per-cell geometry */}
      {cells.map(cell => (
        <CellGeometry key={cell.group} cell={cell} data={data} sx={sx} sy={sy}
                      opts={opts} yBase={h - mb} />
      ))}
    </svg>
  );
}

function GhostLine({ g, sx, sy }: { g: FlexplotGhostLine; sx: Scale; sy: Scale }) {
  if (g.x.length < 2) return null;
  const pts = g.x.map((x, i) => `${sx(x).toFixed(1)},${sy(g.y[i]).toFixed(1)}`).join(' ');
  return <polyline points={pts} fill="none" stroke={color(g.color_index)}
                   strokeWidth="1" strokeOpacity="0.28" strokeDasharray="3 3" />;
}

type CellProps = {
  cell: FlexplotCell; data: FlexplotData; sx: Scale; sy: Scale;
  opts: ChartOpts; yBase: number;
};

function CellGeometry({ cell, data, sx, sy, opts, yBase }: CellProps) {
  // Colour every cell by its group's palette index (scatter and dotplot alike),
  // so a colour-grouped dotplot keeps its group colours instead of collapsing
  // to one hue.
  const ci = data.legend.find(e => e.id === cell.group)?.color_index ?? 0;
  const c = color(ci);
  // When multiple groups share an x-category, dodge each group's crossbar
  // horizontally so they don't overlap in the same spot.
  const groupIdx = Math.max(0, data.groups.indexOf(cell.group));
  const nGroups = data.groups.length;
  const grouped = nGroups > 1;
  const dodge = grouped ? (groupIdx - (nGroups - 1) / 2) * 10 : 0;

  // histogram bars
  if (cell.bins) {
    const { edges, counts } = cell.bins;
    return (
      <g>
        {counts.map((n, i) => {
          const x0 = sx(edges[i]), x1 = sx(edges[i + 1]);
          const y = sy(n);
          return <rect key={i} x={x0 + 0.5} y={y} width={Math.max(0, x1 - x0 - 1)}
                       height={Math.max(0, yBase - y)} fill="var(--accent)" fillOpacity="0.55" />;
        })}
      </g>
    );
  }

  // density curve + area
  if (cell.density) {
    const { x, y } = cell.density;
    if (x.length < 2) return null;
    const line = x.map((xv, i) => `${sx(xv).toFixed(1)},${sy(y[i]).toFixed(1)}`).join(' ');
    const area = `${sx(x[0]).toFixed(1)},${yBase.toFixed(1)} ${line} ${sx(x[x.length - 1]).toFixed(1)},${yBase.toFixed(1)}`;
    return (
      <g>
        <polygon points={area} fill="var(--accent)" fillOpacity="0.12" />
        <polyline points={line} fill="none" stroke="var(--accent)" strokeWidth="1.6" />
      </g>
    );
  }

  return (
    <g>
      {/* confidence band */}
      {opts.showCI && cell.fit && cell.fit.x.length > 1 && (
        <polygon
          points={
            cell.fit.x.map((x, i) => `${sx(x).toFixed(1)},${sy(cell.fit!.hi[i]).toFixed(1)}`).join(' ') +
            ' ' +
            cell.fit.x.map((x, i) => `${sx(x).toFixed(1)},${sy(cell.fit!.lo[i]).toFixed(1)}`).reverse().join(' ')
          }
          fill={c} fillOpacity="0.12" stroke="none"
        />
      )}
      {/* crossbars (dotplot) — coloured by group, dodged when grouped */}
      {cell.crossbars.map(cb => {
        if (cb.center == null) return null;
        const cx = sx(cb.x_index) + dodge;
        return (
          <g key={cb.group_x}>
            {cb.lo != null && cb.hi != null && (
              <line x1={cx} y1={sy(cb.lo)} x2={cx} y2={sy(cb.hi)} stroke={c} strokeWidth="1.5" />
            )}
            <line x1={cx - 9} y1={sy(cb.center)} x2={cx + 9} y2={sy(cb.center)}
                  stroke={c} strokeWidth="2.5" />
          </g>
        );
      })}
      {/* points */}
      {cell.points.x.map((x, i) => (
        <circle key={i} cx={sx(x)} cy={sy(cell.points.y[i])} r={data.kind === 'dotplot' ? 1.8 : 2}
                fill={c} fillOpacity={opts.alpha} />
      ))}
      {/* fit line */}
      {cell.fit && cell.fit.x.length > 1 && (
        <polyline points={cell.fit.x.map((x, i) => `${sx(x).toFixed(1)},${sy(cell.fit!.y[i]).toFixed(1)}`).join(' ')}
                  fill="none" stroke={c} strokeWidth="2" />
      )}
    </g>
  );
}
