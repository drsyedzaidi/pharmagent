import { useState } from 'react';
import type {
  PediatricMetric,
  PediatricStratum,
  PharmState,
  SpecialPopMetric,
  SpecialPopStratum,
} from '../types';
import { Zoomable } from './shared';
import { fmt } from './helpers';

function DoseSweepCard({ r }: { r: PharmState['dose_sweep_results'] }) {
  if (!r || r.status !== 'ok' || !r.profiles?.length) {
    return <div className="qc-card conditional"><div className="qc-title">Dose sweep — not run</div>
      <div style={{ fontSize: 12 }}>{r?.message}</div></div>;
  }
  const W = 560, H = 230, ml = 48, mr = 16, mt = 12, mb = 30;
  const profs = r.profiles;
  const tmax = Math.max(...profs.flatMap(p => p.times)) || 1;
  const cmax = Math.max(...profs.flatMap(p => p.cp)) * 1.05 || 1;
  const sx = (x: number) => ml + (x / tmax) * (W - ml - mr);
  const sy = (v: number) => H - mb - (v / cmax) * (H - mt - mb);
  const colors = ['#1F66A6', '#1D7A5A', '#9A5B12', '#B23A2E', '#4A6FA5'];
  const path = (p: { times: number[]; cp: number[] }) =>
    p.times.map((x, i) => `${i ? 'L' : 'M'}${sx(x).toFixed(1)} ${sy(p.cp[i]).toFixed(1)}`).join(' ');
  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 4 }}>
        {r.label} · {r.n_doses}× q{r.tau}h · {profs.length} dose levels
      </div>
      <Zoomable title="Dose sweep profiles">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: W }} role="img" aria-label="Dose sweep profiles">
        <line x1={ml} y1={H - mb} x2={W - mr} y2={H - mb} stroke="var(--border)" />
        <line x1={ml} y1={mt} x2={ml} y2={H - mb} stroke="var(--border)" />
        {[0, 0.5, 1].map((f, i) => (
          <text key={i} x={ml - 6} y={sy(cmax * f) + 3} textAnchor="end" fontSize="10" fill="var(--text-dim)">{(cmax * f).toFixed(0)}</text>
        ))}
        {profs.map((p, i) => <path key={i} d={path(p)} fill="none" stroke={colors[i % colors.length]} strokeWidth="1.5" />)}
        <text x={(ml + W - mr) / 2} y={H - 4} textAnchor="middle" fontSize="10" fill="var(--text-dim)">time (h)</text>
        <text x={11} y={(mt + H - mb) / 2} textAnchor="middle" fontSize="10" fill="var(--text-dim)"
          transform={`rotate(-90 11 ${(mt + H - mb) / 2})`}>concentration</text>
      </svg>
      </Zoomable>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', fontSize: 11, margin: '2px 0 6px' }}>
        {profs.map((p, i) => (
          <span key={i} style={{ color: colors[i % colors.length] }}>— {p.dose}</span>
        ))}
      </div>
      <table className="nca-table">
        <thead><tr><th>Dose</th><th>Cmax</th><th>AUC<sub>τ</sub></th><th>Cavg</th><th>Ctrough</th></tr></thead>
        <tbody>
          {profs.map(p => (
            <tr key={p.dose}>
              <td>{fmt(p.dose, 0)}</td><td>{fmt(p.cmax, 1)}</td><td>{fmt(p.auc_tau, 0)}</td>
              <td>{fmt(p.cavg, 1)}</td><td>{fmt(p.ctrough, 1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const CLINSIM_METRICS: { key: string; label: string }[] = [
  { key: 'ctrough', label: 'Ctrough (efficacy)' },
  { key: 'cmax', label: 'Cmax (safety)' },
  { key: 'auc_tau', label: 'AUCτ' },
  { key: 'cavg', label: 'Cavg' },
];

/** Clinical trial simulation → probability of target attainment vs dose, with a
 * dose recommendation. Virtual population sampled from the dataset + fitted IIV. */
function ClinsimCard({ r, onRerun, busy }: {
  r: PharmState['clinsim_results'];
  onRerun?: (o: { doses?: number[]; metric?: string; threshold?: number | null;
    direction?: string; target_fraction?: number; n_subjects?: number;
    param_uncertainty?: boolean }) => void;
  busy?: boolean;
}) {
  const [metric, setMetric] = useState(() => r?.metric ?? 'ctrough');
  const [threshold, setThreshold] = useState(() => (r?.threshold != null ? String(r.threshold) : ''));
  const [direction, setDirection] = useState<string>(() => r?.direction ?? 'above');
  const [targetPct, setTargetPct] = useState(() =>
    String(Math.round((r?.target_fraction ?? 0.9) * 100)));
  const [dosesStr, setDosesStr] = useState(() => (r?.doses ?? []).map(d => d.dose).join(', '));
  const [nSubj, setNSubj] = useState(() => String(r?.n_subjects ?? 500));
  const [paramUnc, setParamUnc] = useState(() => (r?.n_param_draws ?? 0) > 0);
  if (!r || r.status !== 'ok') {
    return <div className="qc-card conditional"><div className="qc-title">Clinical trial simulation — not run</div>
      <div style={{ fontSize: 12 }}>{r?.message}</div></div>;
  }
  const rows = r.doses ?? [];
  const tgt = r.target_fraction ?? 0.9;
  const rec = r.recommended_dose;
  const hasPta = rows.some(d => d.pta != null);
  const W = 560, H = 210, ml = 44, mr = 14, mt = 12, mb = 34;
  const n = rows.length;
  const xAt = (i: number) => ml + (n <= 1 ? 0.5 : i / (n - 1)) * (W - ml - mr);
  // PTA panel (0..1)
  const syP = (v: number) => H - mb - Math.max(0, Math.min(1, v)) * (H - mt - mb);
  const ptaPath = rows.filter(d => d.pta != null)
    .map((d, i) => `${i ? 'L' : 'M'}${xAt(rows.indexOf(d)).toFixed(1)} ${syP(d.pta as number).toFixed(1)}`).join(' ');
  // Parameter-uncertainty PTA band (present only when param draws were run).
  const ptaBandRows = rows.filter(d => d.pta_lo != null && d.pta_hi != null);
  const ptaUp = ptaBandRows.map(d => `${xAt(rows.indexOf(d)).toFixed(1)},${syP(d.pta_hi as number).toFixed(1)}`).join(' ');
  const ptaDn = ptaBandRows.map(d => `${xAt(rows.indexOf(d)).toFixed(1)},${syP(d.pta_lo as number).toFixed(1)}`).reverse().join(' ');
  // Exposure panel domain
  const evals = rows.flatMap(d => [d.metric_p05, d.metric_p95]).filter(v => v != null) as number[];
  const emax = (Math.max(...evals, r.threshold ?? 0) || 1) * 1.05;
  const syE = (v: number) => H - mb - (v / emax) * (H - mt - mb);
  const band = (key: 'metric_p05' | 'metric_p95') => rows.filter(d => d[key] != null);
  const up = band('metric_p95').map(d => `${xAt(rows.indexOf(d)).toFixed(1)},${syE(d.metric_p95 as number).toFixed(1)}`).join(' ');
  const dn = band('metric_p05').map(d => `${xAt(rows.indexOf(d)).toFixed(1)},${syE(d.metric_p05 as number).toFixed(1)}`).reverse().join(' ');
  const medPath = rows.filter(d => d.metric_median != null)
    .map((d, i) => `${i ? 'L' : 'M'}${xAt(rows.indexOf(d)).toFixed(1)} ${syE(d.metric_median as number).toFixed(1)}`).join(' ');
  const fmtDose = (d: number) => d >= 1000 ? `${(d / 1000).toFixed(d % 1000 ? 1 : 0)}k` : `${d}`;
  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 6 }}>
        {r.label} · {r.n_subjects} virtual subjects{r.with_iiv ? ' with IIV' : ' (no IIV)'}
        {r.with_covariates && ' + covariates'} · {r.n_doses}× q{r.tau}h
      </div>
      <div style={{ padding: '8px 12px', marginBottom: 8, borderRadius: 6,
        background: rec != null ? 'rgba(29,122,90,0.12)' : 'rgba(154,91,18,0.12)',
        border: `1px solid ${rec != null ? 'var(--green)' : 'var(--yellow)'}`, fontSize: 13 }}>
        {rec != null
          ? <><b style={{ color: 'var(--green)' }}>Recommended dose: {fmtDose(rec)}</b> — {r.recommendation_note}</>
          : <span style={{ color: 'var(--yellow)' }}>{r.recommendation_note}</span>}
      </div>
      {hasPta && (
        <>
          <div style={{ fontSize: 12, color: 'var(--text-dim)', margin: '2px 0' }}>
            Probability of target attainment ({r.metric} {r.direction} {r.threshold})
            {(r.n_param_draws ?? 0) > 0 && <span> · ▦ {r.n_param_draws}-draw parameter-uncertainty band</span>}
          </div>
          <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W }} role="img"
            aria-label="Probability of target attainment vs dose">
            {[0, 0.25, 0.5, 0.75, 1].map((f, i) => (
              <g key={i}>
                <line x1={ml} y1={syP(f)} x2={W - mr} y2={syP(f)} stroke="var(--border)" strokeOpacity="0.4" />
                <text x={ml - 6} y={syP(f) + 3} textAnchor="end" fontSize="9" fill="var(--text-dim)">{(f * 100).toFixed(0)}%</text>
              </g>
            ))}
            {ptaBandRows.length > 1 && <polygon points={`${ptaUp} ${ptaDn}`} fill="var(--accent)" fillOpacity="0.16" />}
            <line x1={ml} y1={syP(tgt)} x2={W - mr} y2={syP(tgt)} stroke="var(--yellow)" strokeDasharray="4 3" strokeWidth="1.2" />
            <path d={ptaPath} fill="none" stroke="var(--accent)" strokeWidth="1.8" />
            {rows.map((d, i) => d.pta == null ? null : (
              <circle key={i} cx={xAt(i)} cy={syP(d.pta)} r={d.dose === rec ? 4 : 2.6}
                fill={d.dose === rec ? 'var(--green)' : 'var(--accent)'} />
            ))}
            {rows.map((d, i) => (
              <text key={i} x={xAt(i)} y={H - mb + 14} textAnchor="middle" fontSize="9" fill="var(--text-dim)">{fmtDose(d.dose)}</text>
            ))}
            <text x={(ml + W) / 2} y={H - 4} textAnchor="middle" fontSize="10" fill="var(--text-dim)">dose</text>
          </svg>
        </>
      )}
      <div style={{ fontSize: 12, color: 'var(--text-dim)', margin: '6px 0 2px' }}>
        {r.metric} distribution (median + 5–95%)
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W }} role="img"
        aria-label="Exposure metric vs dose">
        {up && dn && <polygon points={`${up} ${dn}`} fill="var(--accent)" fillOpacity="0.16" />}
        {r.threshold != null && (
          <line x1={ml} y1={syE(r.threshold)} x2={W - mr} y2={syE(r.threshold)}
            stroke="var(--yellow)" strokeDasharray="4 3" strokeWidth="1.2" />
        )}
        <path d={medPath} fill="none" stroke="var(--accent)" strokeWidth="1.8" />
        {rows.map((d, i) => d.metric_median == null ? null :
          <circle key={i} cx={xAt(i)} cy={syE(d.metric_median)} r="2.6" fill="var(--accent)" />)}
        <line x1={ml} y1={mt} x2={ml} y2={H - mb} stroke="var(--border)" />
        {rows.map((d, i) => (
          <text key={i} x={xAt(i)} y={H - mb + 14} textAnchor="middle" fontSize="9" fill="var(--text-dim)">{fmtDose(d.dose)}</text>
        ))}
        <text x={(ml + W) / 2} y={H - 4} textAnchor="middle" fontSize="10" fill="var(--text-dim)">dose</text>
      </svg>
      {onRerun && (
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap',
          margin: '10px 0 0', paddingTop: 10, borderTop: '1px solid var(--border)', fontSize: 12 }}>
          <label style={{ color: 'var(--text-dim)' }}>metric{' '}
            <select className="model-select" style={{ maxWidth: 160 }} value={metric} disabled={busy}
              onChange={e => setMetric(e.target.value)}>
              {CLINSIM_METRICS.map(m => <option key={m.key} value={m.key}>{m.label}</option>)}
            </select>
          </label>
          <label style={{ color: 'var(--text-dim)' }}>
            <select className="model-select" style={{ maxWidth: 90 }} value={direction} disabled={busy}
              onChange={e => setDirection(e.target.value)}>
              <option value="above">above</option>
              <option value="below">below</option>
            </select>{' '}
            <input type="number" value={threshold} disabled={busy} placeholder="threshold"
              onChange={e => setThreshold(e.target.value)} style={{ width: 84 }} />
          </label>
          <label style={{ color: 'var(--text-dim)' }}>target{' '}
            <input type="number" value={targetPct} disabled={busy}
              onChange={e => setTargetPct(e.target.value)} style={{ width: 52 }} />%</label>
          <label style={{ color: 'var(--text-dim)' }}>N{' '}
            <input type="number" value={nSubj} disabled={busy}
              onChange={e => setNSubj(e.target.value)} style={{ width: 64 }} /></label>
          <label style={{ color: 'var(--text-dim)', flex: '1 1 140px' }}>doses{' '}
            <input type="text" value={dosesStr} disabled={busy} placeholder="comma-separated"
              onChange={e => setDosesStr(e.target.value)} style={{ width: '65%' }} /></label>
          <label style={{ color: 'var(--text-dim)', display: 'inline-flex', gap: 4, alignItems: 'center' }}
            title="Draw the structural parameters from their RSE (needs an NLME fit) → a PTA confidence band + parameter sensitivity">
            <input type="checkbox" checked={paramUnc} disabled={busy}
              onChange={e => setParamUnc(e.target.checked)} /> param uncertainty
          </label>
          <button className="chip" disabled={busy}
            onClick={() => {
              // An empty / non-positive target% must fall back to the backend
              // default, not send target_fraction:0 (which would trivially
              // green-light every dose since PTA >= 0).
              const tf = Number(targetPct) / 100;
              onRerun({
                doses: dosesStr.split(',').map(s => Number(s.trim())).filter(x => x > 0),
                metric, threshold: threshold === '' ? null : Number(threshold), direction,
                target_fraction: targetPct.trim() === '' || !(tf > 0)
                  ? undefined : Math.min(1, tf),
                n_subjects: Number(nSubj), param_uncertainty: paramUnc,
              });
            }}>{busy ? 'Simulating…' : 'Recompute'}</button>
        </div>
      )}
      <table className="nca-table" style={{ marginTop: 8 }}>
        <thead><tr><th>Dose</th><th>PTA</th><th>{r.metric} median</th><th>5–95%</th><th>n</th></tr></thead>
        <tbody>
          {rows.map((d, i) => (
            <tr key={i} style={d.dose === rec ? { background: 'rgba(29,122,90,0.12)' } : undefined}>
              <td>{fmtDose(d.dose)}</td>
              <td>{d.pta == null ? '–' : `${(d.pta * 100).toFixed(1)}%`}</td>
              <td>{fmt(d.metric_median ?? undefined, 3)}</td>
              <td style={{ color: 'var(--text-dim)' }}>{fmt(d.metric_p05 ?? undefined, 3)}–{fmt(d.metric_p95 ?? undefined, 3)}</td>
              <td style={{ color: 'var(--text-dim)' }}>{d.n}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {r.sensitivity && r.sensitivity.records.length > 0 && (() => {
        const refDose = rec ?? rows[rows.length - 1]?.dose;
        const refIdx = rows.findIndex(d => d.dose === refDose);
        return <ClinsimSensitivityPanel s={r.sensitivity} refDose={refDose} refIdx={refIdx} />;
      })()}
    </div>
  );
}

/** Parameter sensitivity (Week-12 Ex 4): for each structural parameter, a
 * scatter of its uncertainty draw vs the resulting PTA at the reference dose —
 * shows which parameters drive the attainment uncertainty. */
function ClinsimSensitivityPanel({ s, refDose, refIdx }: {
  s: NonNullable<PharmState['clinsim_results']>['sensitivity'];
  refDose?: number; refIdx: number;
}) {
  if (!s || refDose == null || refIdx < 0) return null;
  const W = 210, H = 150, ml = 30, mr = 8, mt = 8, mb = 26;
  const panel = (p: string) => {
    const pts = s.records
      .map(rec => ({ x: rec.theta[p], y: rec.pta[refIdx] }))
      .filter(pt => pt.x != null && pt.y != null) as { x: number; y: number }[];
    if (pts.length < 2) return null;
    const xs = pts.map(pt => pt.x), xmin = Math.min(...xs), xmax = Math.max(...xs);
    const sx = (v: number) => ml + ((v - xmin) / (xmax - xmin || 1)) * (W - ml - mr);
    const sy = (v: number) => H - mb - Math.max(0, Math.min(1, v)) * (H - mt - mb);
    return (
      <div key={p} style={{ width: W, maxWidth: '100%' }}>
        <div style={{ fontSize: 11, color: 'var(--text-dim)', fontWeight: 600 }}>{p}</div>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W }} role="img"
          aria-label={`PTA sensitivity to ${p}`}>
          {[0, 0.5, 1].map((f, i) => (
            <line key={i} x1={ml} y1={sy(f)} x2={W - mr} y2={sy(f)} stroke="var(--border)" strokeOpacity="0.4" />
          ))}
          <text x={ml - 4} y={sy(1) + 3} textAnchor="end" fontSize="8" fill="var(--text-dim)">100%</text>
          <text x={ml - 4} y={sy(0) + 3} textAnchor="end" fontSize="8" fill="var(--text-dim)">0</text>
          {pts.map((pt, i) => <circle key={i} cx={sx(pt.x)} cy={sy(pt.y)} r="1.8" fill="var(--accent)" fillOpacity="0.55" />)}
          <line x1={ml} y1={H - mb} x2={W - mr} y2={H - mb} stroke="var(--border)" />
          <text x={(ml + W) / 2} y={H - 3} textAnchor="middle" fontSize="9" fill="var(--text-dim)">{p} draw</text>
        </svg>
      </div>
    );
  };
  return (
    <div style={{ marginTop: 12 }}>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 4 }}>
        Parameter sensitivity — PTA at dose {refDose} vs each parameter draw ({s.n_draws} draws)
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>{s.params.map(panel)}</div>
    </div>
  );
}

/** Simulated exposure covariate forest: horizontal relative-exposure (AUC or
 * Cmax) rows with a 95% interval, the 0.8–1.25 clinical-relevance band, and the
 * reference at 1.0. */
function ExposureForestCard({ r }: { r: PharmState['exposure_forest_results'] }) {
  const [showMetric, setShowMetric] = useState<'rel_auc' | 'rel_cmax'>('rel_auc');
  if (!r || r.status !== 'ok' || !r.rows?.length) {
    return <div className="qc-card conditional"><div className="qc-title">Exposure forest — not run</div>
      <div style={{ fontSize: 12 }}>{r?.message}</div></div>;
  }
  const rows = r.rows;
  const band = r.band ?? [0.8, 1.25];
  const vals = rows.flatMap(x => [x[showMetric].lo, x[showMetric].hi]).filter(v => v != null) as number[];
  const lo = Math.min(...vals, band[0], 1) * 0.95;
  const hi = Math.max(...vals, band[1], 1) * 1.05;
  const W = 600, rowH = 26, padT = 8, padB = 30, ml = 150, mr = 70;
  const H = padT + rows.length * rowH + padB;
  // log-scale x so ratios are symmetric around 1.
  const lnLo = Math.log(Math.max(lo, 1e-3)), lnHi = Math.log(hi);
  const sx = (v: number) => ml + ((Math.log(Math.max(v, 1e-3)) - lnLo) / (lnHi - lnLo || 1)) * (W - ml - mr);
  const yAt = (i: number) => padT + i * rowH + rowH / 2;
  const ticks = [0.5, 0.8, 1, 1.25, 2, 4].filter(t => t >= lo && t <= hi);
  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 6 }}>
        {r.label} · relative exposure vs reference at dose {r.dose} q{r.tau}h ·
        {' '}{r.n_draws} uncertainty draws
        <span style={{ marginLeft: 10 }}>
          {(['rel_auc', 'rel_cmax'] as const).map(m => (
            <button key={m} type="button" className="chip" aria-pressed={showMetric === m}
              style={{ padding: '1px 8px', marginLeft: 4 }}
              onClick={() => setShowMetric(m)}>{m === 'rel_auc' ? 'AUC' : 'Cmax'}</button>
          ))}
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W }} role="img"
        aria-label="Exposure covariate forest">
        <rect x={sx(band[0])} y={padT} width={Math.max(0, sx(band[1]) - sx(band[0]))}
          height={rows.length * rowH} fill="var(--green)" fillOpacity="0.08" />
        <line x1={sx(1)} y1={padT} x2={sx(1)} y2={padT + rows.length * rowH} stroke="var(--text-dim)" strokeDasharray="3 3" />
        {ticks.map((t, i) => (
          <g key={i}>
            <line x1={sx(t)} y1={padT + rows.length * rowH} x2={sx(t)} y2={padT + rows.length * rowH + 4} stroke="var(--border)" />
            <text x={sx(t)} y={H - 16} textAnchor="middle" fontSize="9" fill="var(--text-dim)">{t}</text>
          </g>
        ))}
        {rows.map((row, i) => {
          const m = row[showMetric];
          if (m.median == null) return null;
          const within = m.lo != null && m.hi != null && m.lo >= band[0] && m.hi <= band[1];
          const col = within ? 'var(--green)' : 'var(--accent)';
          return (
            <g key={i}>
              <text x={ml - 8} y={yAt(i) + 3} textAnchor="end" fontSize="10" fill="var(--text)">
                {row.covariate} = {row.label}</text>
              {m.lo != null && m.hi != null &&
                <line x1={sx(m.lo)} y1={yAt(i)} x2={sx(m.hi)} y2={yAt(i)} stroke={col} strokeWidth="1.4" />}
              <circle cx={sx(m.median)} cy={yAt(i)} r="3.4" fill={col} />
              <text x={W - mr + 6} y={yAt(i) + 3} fontSize="9" fill="var(--text-dim)">
                {m.median?.toFixed(2)} [{m.lo?.toFixed(2)}–{m.hi?.toFixed(2)}]</text>
            </g>
          );
        })}
        <text x={(ml + W - mr) / 2} y={H - 3} textAnchor="middle" fontSize="10" fill="var(--text-dim)">
          {showMetric === 'rel_auc' ? 'relative AUC' : 'relative Cmax'} (fraction of reference)</text>
      </svg>
      <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 2 }}>
        Shaded 0.8–1.25 = commonly judged not clinically meaningful; reference (—) = 1.0.
        Reference AUC {r.reference?.auc}, Cmax {r.reference?.cmax} at WT {r.reference?.wt} kg.
      </div>
    </div>
  );
}

const SP_METRIC_LABEL: Record<string, string> = {
  auc_tau: 'AUCss', cmax: 'Cmax,ss', cavg: 'Cavg,ss', ctrough: 'Ctrough,ss',
};

/** Special-population exposure simulation: per-stratum steady-state exposure
 * (box = IQR, whiskers = 5–95%, median) across a dose grid, overlaid on the
 * reference-stratum band, with a per-stratum dose-adjustment verdict. */
function SpecialPopCard({ r, onRerun, busy }: {
  r: PharmState['special_pop_results'];
  onRerun?: (o: { source?: string; stratify_by?: string | null }) => void;
  busy?: boolean;
}) {
  const [metric, setMetric] = useState(() => (r?.metrics && r.metrics[0]) || 'auc_tau');
  if (!r || r.status !== 'ok' || !r.strata?.length) {
    return <div className="qc-card conditional"><div className="qc-title">Special-population simulation — not run</div>
      <div style={{ fontSize: 12 }}>{r?.message}{r?.available && <> · available: {r.available.join(', ')}</>}</div></div>;
  }
  const strata = r.strata;
  const band = r.reference_band?.[metric];
  const doses = strata[0].doses.map(d => d.dose);
  // Shared y-domain (log) across panels for comparability.
  const all = strata.flatMap(s => s.doses.flatMap(d => {
    const m = d[metric as keyof typeof d] as { p05?: number | null; p95?: number | null } | undefined;
    return [m?.p05, m?.p95];
  })).filter(v => v != null) as number[];
  const lo = Math.max(1e-6, Math.min(...all, band?.lo ?? Infinity) * 0.9);
  const hi = Math.max(...all, band?.hi ?? 0) * 1.1;
  const lnLo = Math.log(lo), lnHi = Math.log(hi);
  const W = 250, H = 170, ml = 40, mr = 8, mt = 8, mb = 30;
  const sy = (v: number) => H - mb - ((Math.log(Math.max(v, 1e-6)) - lnLo) / (lnHi - lnLo || 1)) * (H - mt - mb);
  const n = doses.length;
  const xAt = (i: number) => ml + (n <= 1 ? 0.5 : (i + 0.5) / n) * (W - ml - mr);
  const bw = Math.min(22, (W - ml - mr) / (n * 1.7));
  const panel = (s: SpecialPopStratum) => (
    <div key={s.label} style={{ width: W, maxWidth: '100%' }}>
      <div style={{ fontSize: 11, color: 'var(--text-dim)', fontWeight: 600 }}>
        {s.label} <span>(n = {s.n})</span>
        {s.recommended_dose != null && <span style={{ color: 'var(--green)' }}> · dose {s.recommended_dose}</span>}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W }} role="img"
        aria-label={`Special-population exposure for ${s.label}`}>
        {band?.lo != null && band.hi != null &&
          <rect x={ml} y={sy(band.hi)} width={W - ml - mr} height={Math.max(0, sy(band.lo) - sy(band.hi))}
            fill="var(--text-dim)" fillOpacity="0.12" />}
        {band?.median != null &&
          <line x1={ml} y1={sy(band.median)} x2={W - mr} y2={sy(band.median)} stroke="var(--text-dim)" strokeDasharray="3 3" />}
        {s.doses.map((d, i) => {
          const m = d[metric as keyof typeof d] as SpecialPopMetric | undefined;
          if (!m || m.p50 == null) return null;
          const x = xAt(i), col = m.within_ref ? 'var(--green)' : 'var(--accent)';
          return (
            <g key={i}>
              {m.p05 != null && m.p95 != null &&
                <line x1={x} y1={sy(m.p95)} x2={x} y2={sy(m.p05)} stroke={col} strokeWidth="1" />}
              {m.p25 != null && m.p75 != null &&
                <rect x={x - bw / 2} y={sy(m.p75)} width={bw} height={Math.max(1, sy(m.p25) - sy(m.p75))}
                  fill={col} fillOpacity="0.25" stroke={col} strokeWidth="0.8" />}
              <line x1={x - bw / 2} y1={sy(m.p50)} x2={x + bw / 2} y2={sy(m.p50)} stroke={col} strokeWidth="1.6" />
            </g>
          );
        })}
        <line x1={ml} y1={H - mb} x2={W - mr} y2={H - mb} stroke="var(--border)" />
        {s.doses.map((d, i) => (
          <text key={i} x={xAt(i)} y={H - mb + 12} textAnchor="middle" fontSize="8" fill="var(--text-dim)">{d.dose}</text>
        ))}
        <text x={(ml + W) / 2} y={H - 2} textAnchor="middle" fontSize="9" fill="var(--text-dim)">dose</text>
      </svg>
    </div>
  );
  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 6 }}>
        {r.label} · exposure by <b>{r.stratify_by}</b> vs the <b>{r.reference_stratum}</b> band
        (at dose {r.reference_dose}) · {r.n_per_stratum}/stratum · source: {r.population_source}
        <span style={{ marginLeft: 10 }}>
          {(r.metrics ?? ['auc_tau']).map(mk => (
            <button key={mk} type="button" className="chip" aria-pressed={metric === mk}
              style={{ padding: '1px 8px', marginLeft: 4 }}
              onClick={() => setMetric(mk)}>{SP_METRIC_LABEL[mk] ?? mk}</button>
          ))}
        </span>
      </div>
      {r.covariate_in_model === false &&
        <div style={{ fontSize: 11, color: 'var(--yellow)', marginBottom: 6 }}>
          No fitted {r.stratify_by} effect — strata differ only by allometric weight. Run SCM/NLME with this covariate.
        </div>}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>{strata.map(panel)}</div>
      {onRerun && (
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap', marginTop: 8,
          paddingTop: 8, borderTop: '1px solid var(--border)', fontSize: 12 }}>
          <span style={{ color: 'var(--text-dim)' }}>population:</span>
          {(['dataset', 'reference'] as const).map(src => (
            <button key={src} type="button" className="chip" disabled={busy} aria-pressed={r.population_source === src}
              onClick={() => onRerun({ source: src })}>{src === 'reference' ? 'representative adults' : 'analysis dataset'}</button>
          ))}
          {busy && <span style={{ color: 'var(--text-dim)' }}>simulating…</span>}
        </div>
      )}
      <table className="nca-table" style={{ marginTop: 8 }}>
        <thead><tr><th>{r.stratify_by}</th><th>n</th><th>Adjusted dose</th><th>Verdict</th></tr></thead>
        <tbody>
          {strata.map((s, i) => (
            <tr key={i}>
              <td>{s.label}</td><td style={{ color: 'var(--text-dim)' }}>{s.n}</td>
              <td style={{ color: s.recommended_dose != null ? 'var(--green)' : 'var(--text-dim)' }}>
                {s.recommended_dose ?? '—'}</td>
              <td style={{ fontSize: 11, color: 'var(--text-dim)' }}>{s.note}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 2 }}>
        Shaded band = {r.reference_stratum} 5–95% at dose {r.reference_dose}; box = IQR, whiskers = 5–95%, line = median.
        Green = within the reference range.
      </div>
    </div>
  );
}

/** Per-subject steady-state exposure (AUCss/Cmax,ss) from the fitted EBEs, with a
 * per-group (e.g. renal-function) summary — the reference table for special-pop. */
function IndividualExposuresCard({ r }: { r: PharmState['individual_exposures'] }) {
  if (!r || r.status !== 'ok' || !r.subjects?.length) {
    return <div className="qc-card conditional"><div className="qc-title">Individual exposures — not run</div>
      <div style={{ fontSize: 12 }}>{r?.message}</div></div>;
  }
  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 6 }}>
        {r.label} · steady-state AUCss / Cmax,ss for {r.subjects.length} subjects at {r.dose} q{r.tau}h (EBEs)
      </div>
      {!!r.groups?.length && (
        <table className="nca-table">
          <thead><tr><th>Group</th><th>n</th><th>AUCss median [5–95%]</th><th>Cmax,ss median [5–95%]</th></tr></thead>
          <tbody>
            {r.groups.map((g, i) => (
              <tr key={i}>
                <td>{g.group}</td><td style={{ color: 'var(--text-dim)' }}>{g.n}</td>
                <td>{fmt(g.auc_ss?.median ?? undefined, 2)} <span style={{ color: 'var(--text-dim)' }}>
                  [{fmt(g.auc_ss?.p05 ?? undefined, 2)}–{fmt(g.auc_ss?.p95 ?? undefined, 2)}]</span></td>
                <td>{fmt(g.cmax_ss?.median ?? undefined, 2)} <span style={{ color: 'var(--text-dim)' }}>
                  [{fmt(g.cmax_ss?.p05 ?? undefined, 2)}–{fmt(g.cmax_ss?.p95 ?? undefined, 2)}]</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

/** Pediatric dose-finding: exposure by age×weight stratum vs an adult reference
 * band, with the %-within-adult-range dose-selection curve (Week-14). */
const PED_PALETTE = ['#1F66A6', '#1D7A5A', '#9A5B12', '#B23A2E', '#4A6FA5', '#2A8F8F'];

function PediatricCard({ r, onRerun, busy }: {
  r: PharmState['pediatric_results'];
  onRerun?: (o: { source?: string; wt_exponent_cl?: number | null; wt_exponent_v?: number | null }) => void;
  busy?: boolean;
}) {
  const [metric, setMetric] = useState(() => (r?.metrics && r.metrics[0]) || 'auc_tau');
  const [clExp, setClExp] = useState('');
  const [vExp, setVExp] = useState('');
  if (!r || r.status !== 'ok' || !r.strata?.length) {
    return <div className="qc-card conditional"><div className="qc-title">Pediatric simulation — not run</div>
      <div style={{ fontSize: 12 }}>{r?.message}</div></div>;
  }
  const strata = r.strata;
  const band = r.reference_band?.[metric];
  const doses = strata[0].doses.map(d => d.dose);
  const n = doses.length;

  // --- % within adult range: the dose-selection curve (headline) ---
  const CW = 380, CH = 210, cl = 44, cr = 12, ct = 10, cb = 34;
  const cx = (i: number) => cl + (n <= 1 ? 0.5 : i / (n - 1)) * (CW - cl - cr);
  const cy = (p: number) => CH - cb - (p / 100) * (CH - ct - cb);

  // --- boxplots vs the adult band (shared log-y) ---
  const all = strata.flatMap(s => s.doses.flatMap(d => {
    const m = d[metric as keyof typeof d] as { p05?: number | null; p95?: number | null } | undefined;
    return [m?.p05, m?.p95];
  })).filter(v => v != null) as number[];
  const lo = Math.max(1e-6, Math.min(...all, band?.lo ?? Infinity) * 0.9);
  const hi = Math.max(...all, band?.hi ?? 0) * 1.1;
  const lnLo = Math.log(lo), lnHi = Math.log(hi);
  const W = 250, H = 156, ml = 40, mr = 8, mt = 8, mb = 26;
  const sy = (v: number) => H - mb - ((Math.log(Math.max(v, 1e-6)) - lnLo) / (lnHi - lnLo || 1)) * (H - mt - mb);
  const xAt = (i: number) => ml + (n <= 1 ? 0.5 : (i + 0.5) / n) * (W - ml - mr);
  const bw = Math.min(20, (W - ml - mr) / (n * 1.7));

  const boxPanel = (s: PediatricStratum) => (
    <div key={s.label} style={{ width: W, maxWidth: '100%' }}>
      <div style={{ fontSize: 11, color: 'var(--text-dim)', fontWeight: 600 }}>
        {s.label} <span>(n = {s.n})</span>
        {s.recommended_dose != null && <span style={{ color: 'var(--green)' }}> · dose {s.recommended_dose}</span>}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W }} role="img"
        aria-label={`Pediatric exposure for ${s.label}`}>
        {band?.lo != null && band.hi != null &&
          <rect x={ml} y={sy(band.hi)} width={W - ml - mr} height={Math.max(0, sy(band.lo) - sy(band.hi))}
            fill="var(--text-dim)" fillOpacity="0.12" />}
        {band?.median != null &&
          <line x1={ml} y1={sy(band.median)} x2={W - mr} y2={sy(band.median)} stroke="var(--text-dim)" strokeDasharray="3 3" />}
        {s.doses.map((d, i) => {
          const m = d[metric as keyof typeof d] as PediatricMetric | undefined;
          if (!m || m.p50 == null) return null;
          const x = xAt(i), col = m.within_ref ? 'var(--green)' : 'var(--accent)';
          return (
            <g key={i}>
              {m.p05 != null && m.p95 != null &&
                <line x1={x} y1={sy(m.p95)} x2={x} y2={sy(m.p05)} stroke={col} strokeWidth="1" />}
              {m.p25 != null && m.p75 != null &&
                <rect x={x - bw / 2} y={sy(m.p75)} width={bw} height={Math.max(1, sy(m.p25) - sy(m.p75))}
                  fill={col} fillOpacity="0.25" stroke={col} strokeWidth="0.8" />}
              <line x1={x - bw / 2} y1={sy(m.p50)} x2={x + bw / 2} y2={sy(m.p50)} stroke={col} strokeWidth="1.6" />
            </g>
          );
        })}
        <line x1={ml} y1={H - mb} x2={W - mr} y2={H - mb} stroke="var(--border)" />
        {s.doses.map((d, i) => (
          <text key={i} x={xAt(i)} y={H - mb + 11} textAnchor="middle" fontSize="8" fill="var(--text-dim)">{d.dose}</text>
        ))}
      </svg>
    </div>
  );

  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 6 }}>
        {r.label} · pediatric exposure by <b>age × weight</b> vs the adult range
        ({r.reference_source} at {r.reference_dose}) · {r.allometry} allometry · {r.n_per_stratum}/stratum
        <span style={{ marginLeft: 10 }}>
          {(r.metrics ?? ['auc_tau']).map(mk => (
            <button key={mk} type="button" className="chip" aria-pressed={metric === mk}
              style={{ padding: '1px 8px', marginLeft: 4 }}
              onClick={() => setMetric(mk)}>{SP_METRIC_LABEL[mk] ?? mk}</button>
          ))}
        </span>
      </div>

      <div style={{ fontSize: 11, color: 'var(--text-dim)', fontWeight: 600, marginBottom: 2 }}>
        {SP_METRIC_LABEL[metric] ?? metric} — % of pediatric subjects within the adult range
      </div>
      <svg viewBox={`0 0 ${CW} ${CH}`} style={{ width: '100%', maxWidth: CW }} role="img"
        aria-label="Percent of pediatric subjects within the adult exposure range by dose">
        {[0, 25, 50, 75, 100].map(p => (
          <g key={p}>
            <line x1={cl} y1={cy(p)} x2={CW - cr} y2={cy(p)} stroke="var(--border)" strokeOpacity="0.5" />
            <text x={cl - 5} y={cy(p) + 3} textAnchor="end" fontSize="8" fill="var(--text-dim)">{p}</text>
          </g>
        ))}
        {strata.map((s, si) => {
          const col = PED_PALETTE[si % PED_PALETTE.length];
          const pts = s.doses.map((d, i) => {
            const m = d[metric as keyof typeof d] as PediatricMetric | undefined;
            return m?.pct_within_ref == null ? null : { x: cx(i), y: cy(m.pct_within_ref), rec: d.dose === s.recommended_dose };
          }).filter(Boolean) as { x: number; y: number; rec: boolean }[];
          return (
            <g key={s.label}>
              <polyline points={pts.map(p => `${p.x},${p.y}`).join(' ')} fill="none" stroke={col} strokeWidth="1.6" />
              {pts.map((p, i) => <circle key={i} cx={p.x} cy={p.y} r={p.rec ? 3.4 : 2} fill={col}
                stroke={p.rec ? '#fff' : 'none'} strokeWidth={p.rec ? 1 : 0} />)}
            </g>
          );
        })}
        <line x1={cl} y1={CH - cb} x2={CW - cr} y2={CH - cb} stroke="var(--border)" />
        {doses.map((d, i) => (
          <text key={i} x={cx(i)} y={CH - cb + 12} textAnchor="middle" fontSize="8" fill="var(--text-dim)">{d}</text>
        ))}
        <text x={(cl + CW) / 2} y={CH - 2} textAnchor="middle" fontSize="9" fill="var(--text-dim)">Dose (mg)</text>
      </svg>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '2px 12px', fontSize: 11, marginTop: 2 }}>
        {strata.map((s, si) => (
          <span key={s.label} style={{ color: 'var(--text-dim)' }}>
            <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 2,
              background: PED_PALETTE[si % PED_PALETTE.length], marginRight: 3 }} />{s.label}
          </span>
        ))}
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginTop: 10 }}>{strata.map(boxPanel)}</div>

      {onRerun && (
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', marginTop: 8,
          paddingTop: 8, borderTop: '1px solid var(--border)', fontSize: 12 }}>
          <span style={{ color: 'var(--text-dim)' }}>pediatric covariates:</span>
          {(['reference', 'dataset'] as const).map(src => (
            <button key={src} type="button" className="chip" disabled={busy} aria-pressed={r.population_source === src}
              onClick={() => onRerun({ source: src })}>{src === 'reference' ? 'representative peds' : 'analysis dataset'}</button>
          ))}
          <span style={{ color: 'var(--text-dim)', marginLeft: 6 }}>WT exponent CL</span>
          <input type="number" step="0.01" value={clExp} onChange={e => setClExp(e.target.value)}
            aria-label="Weight exponent on clearance" placeholder="0.75" disabled={busy} style={{ width: 56 }} />
          <span style={{ color: 'var(--text-dim)' }}>V</span>
          <input type="number" step="0.01" value={vExp} onChange={e => setVExp(e.target.value)}
            aria-label="Weight exponent on volume" placeholder="1.0" disabled={busy} style={{ width: 56 }} />
          <button className="chip" disabled={busy}
            onClick={() => onRerun({ wt_exponent_cl: clExp === '' ? null : Number(clExp),
              wt_exponent_v: vExp === '' ? null : Number(vExp) })}>Re-simulate</button>
          {busy && <span style={{ color: 'var(--text-dim)' }}>simulating…</span>}
        </div>
      )}

      <table className="nca-table" style={{ marginTop: 8 }}>
        <thead><tr><th>Age</th><th>Weight</th><th>n</th><th>Matched dose</th><th>Basis</th></tr></thead>
        <tbody>
          {strata.map((s, i) => (
            <tr key={i}>
              <td>{s.age_label}</td><td>{s.wt_label}</td>
              <td style={{ color: 'var(--text-dim)' }}>{s.n}</td>
              <td style={{ color: s.recommended_dose != null ? 'var(--green)' : 'var(--text-dim)' }}>
                {s.recommended_dose ?? '—'}</td>
              <td style={{ fontSize: 11, color: 'var(--text-dim)' }}>{s.note}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 2 }}>
        Shaded band = adult 5–95% at dose {r.reference_dose}; box = IQR, whiskers = 5–95%, line = median.
        Green = median within the adult range. Matched dose maximizes the % of subjects within it.
      </div>
    </div>
  );
}

function SimChart({ sim }: { sim: PharmState['simulation_results'] }) {
  // Hooks must run unconditionally and in the same order every render — keep this
  // above the early return, or a not-run -> ok transition changes hook order and
  // React throws "Rendered more hooks than during the previous render."
  const [logY, setLogY] = useState(false);
  if (!sim || sim.status !== 'ok' || !sim.times || !sim.cp) {
    return <div className="qc-card conditional"><div className="qc-title">Simulation — not run</div>
      <div style={{ fontSize: 12 }}>{sim?.message}</div></div>;
  }
  const W = 580, H = 240, ml = 48, mr = sim.eff ? 48 : 16, mt = 12, mb = 32;
  const t = sim.times, cp = sim.cp, eff = sim.eff;
  const tmax = Math.max(...t) || 1;
  const cpMax = Math.max(...cp) * 1.1 || 1;
  const cpLo = Math.max(Math.min(...cp.filter(v => v > 0), cpMax), cpMax / 1000);  // log floor
  const effArr = eff ?? [];
  const effMax = eff ? Math.max(...effArr) * 1.1 || 1 : 1;
  const effMin = eff ? Math.min(...effArr, 0) : 0;
  const sx = (x: number) => ml + (x / tmax) * (W - ml - mr);
  const syCp = logY
    ? (v: number) => H - mb - ((Math.log10(Math.max(v, cpLo)) - Math.log10(cpLo)) /
        (Math.log10(cpMax) - Math.log10(cpLo) || 1)) * (H - mt - mb)
    : (v: number) => H - mb - (v / cpMax) * (H - mt - mb);
  const syEff = (v: number) => H - mb - ((v - effMin) / (effMax - effMin || 1)) * (H - mt - mb);
  const path = (xs: number[], ys: number[], scale: (v: number) => number) =>
    xs.map((x, i) => `${i ? 'L' : 'M'}${sx(x).toFixed(1)} ${scale(ys[i]).toFixed(1)}`).join(' ');
  const xticks = Array.from({ length: 5 }, (_, i) => (tmax * i) / 4);
  const yticks = logY
    ? Array.from({ length: 4 }, (_, i) => cpLo * (cpMax / cpLo) ** (i / 3))
    : Array.from({ length: 4 }, (_, i) => (cpMax * i) / 3);
  const fmtTick = (v: number) => logY ? Number(v.toPrecision(2)).toString() : v.toFixed(0);

  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 4, display: 'flex',
        alignItems: 'center', gap: 8 }}>
        <span>{sim.label} · {sim.regimen?.n_doses}×{sim.regimen?.dose} q{sim.regimen?.tau}h
          {sim.from_fit ? ' · fitted typical params' : ' · model defaults'} · Cmax≈{fmt(sim.cmax ?? undefined, 1)}</span>
        <button className="chip" style={{ marginLeft: 'auto', padding: '2px 8px', fontSize: 11 }}
          onClick={() => setLogY(v => !v)}>{logY ? 'Linear Y' : 'Log Y'}</button>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: W }} role="img"
        aria-label="Simulated concentration-time profile">
        {/* axes */}
        <line x1={ml} y1={H - mb} x2={W - mr} y2={H - mb} stroke="var(--border)" />
        <line x1={ml} y1={mt} x2={ml} y2={H - mb} stroke="var(--border)" />
        {yticks.map((v, i) => (
          <g key={i}>
            <line x1={ml - 3} y1={syCp(v)} x2={W - mr} y2={syCp(v)} stroke="var(--border-subtle)" />
            <text x={ml - 6} y={syCp(v) + 3} textAnchor="end" fontSize="10" fill="var(--text-dim)">{fmtTick(v)}</text>
          </g>
        ))}
        {xticks.map((v, i) => (
          <text key={i} x={sx(v)} y={H - mb + 14} textAnchor="middle" fontSize="10" fill="var(--text-dim)">{v.toFixed(0)}</text>
        ))}
        <text x={(ml + W - mr) / 2} y={H - 2} textAnchor="middle" fontSize="10" fill="var(--text-dim)">time (h)</text>
        <text x={12} y={(mt + H - mb) / 2} textAnchor="middle" fontSize="10" fill="var(--accent)"
          transform={`rotate(-90 12 ${(mt + H - mb) / 2})`}>concentration</text>
        {/* cp line */}
        <path d={path(t, cp, syCp)} fill="none" stroke="var(--accent)" strokeWidth="1.6" />
        {/* eff line (PK/PD) */}
        {eff && <path d={path(t, effArr, syEff)} fill="none" stroke="var(--green)" strokeWidth="1.6" strokeDasharray="4 3" />}
        {eff && <text x={W - mr + 6} y={(mt + H - mb) / 2} textAnchor="middle" fontSize="10" fill="var(--green)"
          transform={`rotate(90 ${W - mr + 6} ${(mt + H - mb) / 2})`}>effect</text>}
      </svg>
      {eff && (
        <div style={{ fontSize: 11, color: 'var(--text-dim)', display: 'flex', gap: 14 }}>
          <span><span style={{ color: 'var(--accent)' }}>—</span> concentration</span>
          <span><span style={{ color: 'var(--green)' }}>– –</span> effect</span>
        </div>
      )}
    </div>
  );
}

export {
  DoseSweepCard,
  ClinsimCard,
  ClinsimSensitivityPanel,
  ExposureForestCard,
  SpecialPopCard,
  IndividualExposuresCard,
  PediatricCard,
  SimChart,
};
